use crate::driver::{sys, DriverError};
use std::sync::Mutex;

/// Shared lifetime state for one tracked primary-context generation.
///
/// Terminal loss records the first observation for every attached context
/// wrapper. A process must restart before using the device after terminal loss.
#[derive(Debug)]
pub struct PrimaryContextLifetime {
    pub(crate) recorded_id: u64,
    pub(crate) loss: Mutex<Option<PrimaryContextLoss>>,
}

impl PrimaryContextLifetime {
    pub(crate) fn error(&self) -> Option<DriverError> {
        self.loss
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .map(|loss| DriverError(loss.error))
    }

    pub(crate) fn check_primary_active(&self, device: sys::CUdevice) -> Result<(), DriverError> {
        if let Some(error) = self.error() {
            return Err(error);
        }
        let (_, active) = unsafe { crate::driver::result::primary_ctx::get_state(device) }?;
        if !active {
            self.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
        }
        match self.error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(crate) fn record_error(&self, error: DriverError) {
        if error.0 == sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED {
            let mut loss = self
                .loss
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if loss.is_none() {
                *loss = Some(PrimaryContextLoss {
                    recorded_id: self.recorded_id,
                    observed_id: None,
                    error: error.0,
                });
            }
        }
    }
}

/// The first observation that made a tracked primary context unusable.
///
/// The record contains cached identity and status only, so inspecting or
/// formatting it does not query CUDA after loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimaryContextLoss {
    /// The CUDA context identity recorded when tracking began.
    pub recorded_id: u64,
    /// A different observed identity, when the probe obtained one.
    pub observed_id: Option<u64>,
    /// The driver status that caused terminal refusal.
    pub error: sys::CUresult,
}

#[cfg(test)]
mod fathomdb_primary_lifetime {
    use super::*;
    use crate::driver::{result::context_test_seams, DriverError};
    use std::sync::Arc;

    #[test]
    fn terminal_loss_is_shared_and_survives_error_reads_and_oom() {
        let _driver = context_test_seams::Guard::new();
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 17,
            loss: Mutex::new(None),
        });
        let mut first = context_test_seams::context();
        let mut sibling = context_test_seams::context();
        Arc::get_mut(&mut first).unwrap().primary_lifetime = Some(lifetime.clone());
        Arc::get_mut(&mut sibling).unwrap().primary_lifetime = Some(lifetime.clone());
        assert!(!Arc::ptr_eq(&first, &sibling));
        assert!(lifetime.loss.lock().unwrap().is_none());
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        first.record_err::<()>(Err(DriverError(lost)));

        let first_read = first.check_err().map_err(|error| error.0 as u32);
        let second_read = first.check_err().map_err(|error| error.0 as u32);
        sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        let sibling_read = sibling.check_err().map_err(|error| error.0 as u32);
        drop(first);
        drop(sibling);

        assert_eq!(first_read, Err(lost as u32));
        assert_eq!(
            second_read,
            Err(lost as u32),
            "reading an error must not clear terminal loss"
        );
        assert_eq!(
            sibling_read,
            Err(lost as u32),
            "shared terminal loss must take priority over later OOM"
        );
        assert_eq!(
            *lifetime.loss.lock().unwrap(),
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                error: lost
            })
        );
    }

    #[test]
    fn untracked_context_errors_still_clear_on_read() {
        let _driver = context_test_seams::Guard::new();
        let context = context_test_seams::context();
        let error = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        context.record_err::<()>(Err(DriverError(error)));

        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    }

    #[test]
    fn ordinary_tracked_oom_does_not_latch_terminal_loss() {
        let _driver = context_test_seams::Guard::new();
        _driver.set_context_id(23);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 23,
            loss: Mutex::new(None),
        });
        let mut context = context_test_seams::context();
        Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(lifetime.clone());
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        context.record_err::<()>(Err(DriverError(oom)));

        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert!(lifetime.loss.lock().unwrap().is_none());
    }

    #[test]
    fn inactive_primary_is_discovered_before_bind_and_stays_terminal() {
        let driver = context_test_seams::Guard::new();
        driver.set_primary_active(false);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 29,
            loss: Mutex::new(None),
        });
        let mut context = context_test_seams::context();
        Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(lifetime.clone());

        let first = context.bind_to_thread().map_err(|error| error.0 as u32);
        let second = context.bind_to_thread().map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();
        drop(context);

        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        assert_eq!(
            first,
            Err(lost as u32),
            "inactive primary must refuse before binding stale context"
        );
        assert_eq!(second, Err(lost as u32));
        assert_eq!(
            dispatch.primary_queries, 1,
            "cached loss must prevent another primary probe"
        );
        assert_eq!((dispatch.queries, dispatch.binds), (0, 0));
        assert_eq!(
            *lifetime.loss.lock().unwrap(),
            Some(PrimaryContextLoss {
                recorded_id: 29,
                observed_id: None,
                error: lost
            })
        );
    }

    #[test]
    fn active_primary_allows_healthy_binding() {
        let driver = context_test_seams::Guard::new();
        driver.set_context_id(31);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 31,
            loss: Mutex::new(None),
        });
        let mut context = context_test_seams::context();
        Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(lifetime.clone());

        let result = context.bind_to_thread().map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();
        drop(context);

        assert_eq!(result, Ok(()));
        assert_eq!(
            (dispatch.primary_queries, dispatch.queries, dispatch.binds),
            (1, 1, 1)
        );
        assert!(lifetime.loss.lock().unwrap().is_none());
    }

    #[test]
    fn untracked_bind_does_not_probe_primary_state() {
        let driver = context_test_seams::Guard::new();
        driver.set_primary_active(false);
        let context = context_test_seams::context();

        let result = context.bind_to_thread().map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();
        drop(context);

        assert_eq!(result, Ok(()));
        assert_eq!(
            (dispatch.primary_queries, dispatch.queries, dispatch.binds),
            (0, 1, 1)
        );
    }

    #[test]
    fn changed_primary_identity_is_discovered_before_bind() {
        let driver = context_test_seams::Guard::new();
        driver.set_context_id(42);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 17,
            loss: Mutex::new(None),
        });
        let mut context = context_test_seams::context();
        Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(lifetime.clone());

        let first = context.bind_to_thread().map_err(|error| error.0 as u32);
        let second = context.bind_to_thread().map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();
        drop(context);

        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        assert_eq!(first, Err(lost as u32), "replaced context identity must refuse before stale binding");
        assert_eq!(second, Err(lost as u32));
        assert_eq!((dispatch.primary_queries, dispatch.id_queries), (1, 1));
        assert_eq!((dispatch.queries, dispatch.binds), (0, 0));
        assert_eq!(
            *lifetime.loss.lock().unwrap(),
            Some(PrimaryContextLoss { recorded_id: 17, observed_id: Some(42), error: lost })
        );
    }
}
