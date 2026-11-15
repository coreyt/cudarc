use crate::driver::{sys, DriverError};
use std::sync::Mutex;

/// Shared lifetime state for one tracked primary-context generation.
///
/// Terminal loss records the first observation for every attached context
/// wrapper. A process must restart before using the device after terminal loss.
/// Native ownership is held by the shared anchor. Its final cleanup performs
/// the balanced primary release without binding a context, including after loss.
#[derive(Debug)]
pub struct PrimaryContextLifetime {
    pub(crate) recorded_id: u64,
    pub(crate) loss: Mutex<Option<PrimaryContextLoss>>,
    pub(crate) anchor: Option<PrimaryContextAnchor>,
}

#[derive(Debug)]
pub(crate) struct PrimaryContextAnchor {
    pub(crate) device: sys::CUdevice,
    pub(crate) context: sys::CUcontext,
}

// Primary contexts support cross-thread use; the lifetime owns their retain.
unsafe impl Send for PrimaryContextAnchor {}
unsafe impl Sync for PrimaryContextAnchor {}

impl Drop for PrimaryContextAnchor {
    fn drop(&mut self) {
        let context = std::mem::replace(&mut self.context, std::ptr::null_mut());
        if !context.is_null() {
            let _ = unsafe { crate::driver::result::primary_ctx::release(self.device) };
        }
    }
}

impl PrimaryContextLifetime {
    pub(crate) fn error(&self) -> Option<DriverError> {
        self.loss
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .map(|loss| DriverError(loss.error))
    }

    pub(crate) fn check_primary(
        &self,
        device: sys::CUdevice,
        context: sys::CUcontext,
    ) -> Result<(), DriverError> {
        if let Some(error) = self.error() {
            return Err(error);
        }
        let (_, active) = match unsafe { crate::driver::result::primary_ctx::get_state(device) } {
            Ok(state) => state,
            Err(error) => {
                self.record_loss(error, None);
                return Err(self.error().unwrap_or(error));
            }
        };
        if !active {
            self.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
        } else {
            let id = match unsafe { crate::driver::result::ctx::get_id(context) } {
                Ok(id) => id,
                Err(error) => {
                    self.record_loss(error, None);
                    return Err(self.error().unwrap_or(error));
                }
            };
            if id != self.recorded_id {
                self.record_loss(
                    DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED),
                    Some(id),
                );
            }
        }
        match self.error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(crate) fn record_error(&self, error: DriverError) {
        if error.0 == sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED {
            self.record_loss(error, None);
        }
    }

    fn record_loss(&self, error: DriverError, observed_id: Option<u64>) {
        let mut loss = self
            .loss
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if loss.is_none() {
            *loss = Some(PrimaryContextLoss {
                recorded_id: self.recorded_id,
                observed_id,
                error: error.0,
            });
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
            anchor: None,
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
            anchor: None,
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
            anchor: None,
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
            anchor: None,
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
            anchor: None,
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
            "replaced context identity must refuse before stale binding"
        );
        assert_eq!(second, Err(lost as u32));
        assert_eq!((dispatch.primary_queries, dispatch.id_queries), (1, 1));
        assert_eq!((dispatch.queries, dispatch.binds), (0, 0));
        assert_eq!(
            *lifetime.loss.lock().unwrap(),
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: Some(42),
                error: lost
            })
        );
    }

    #[test]
    fn failed_primary_state_probe_latches_its_first_error() {
        let driver = context_test_seams::Guard::new();
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;
        driver.set_primary_error(Some(error));
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 17,
            loss: Mutex::new(None),
            anchor: None,
        });
        let mut context = context_test_seams::context();
        Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(lifetime.clone());

        let first = context.bind_to_thread().map_err(|error| error.0 as u32);
        driver.set_primary_error(None);
        let second = context.bind_to_thread().map_err(|error| error.0 as u32);
        context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        let third = context.bind_to_thread().map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();
        drop(context);

        assert_eq!(first, Err(error as u32));
        assert_eq!(
            second,
            Err(error as u32),
            "a failed liveness query must permanently refuse later binding"
        );
        assert_eq!(
            third,
            Err(error as u32),
            "later OOM must not replace the first failed-probe observation"
        );
        assert_eq!((dispatch.primary_queries, dispatch.id_queries), (1, 0));
        assert_eq!((dispatch.queries, dispatch.binds), (0, 0));
        assert_eq!(
            *lifetime.loss.lock().unwrap(),
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                error
            })
        );
    }

    #[test]
    fn tracked_wrappers_release_only_through_the_final_anchor() {
        for terminal in [false, true] {
            let driver = context_test_seams::Guard::new();
            let retained = unsafe { crate::driver::result::primary_ctx::retain(0) }
                .map_err(|error| error.0 as u32)
                .unwrap();
            assert_eq!(driver.snapshot().retains, 1);
            let loss = terminal.then_some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                error: sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            });
            let lifetime = Arc::new(PrimaryContextLifetime {
                recorded_id: 17,
                loss: Mutex::new(loss),
                anchor: Some(PrimaryContextAnchor {
                    device: 0,
                    context: retained,
                }),
            });
            let mut first = context_test_seams::context();
            let mut sibling = context_test_seams::context();
            assert_eq!(first.cu_ctx as usize, retained as usize);
            assert_eq!(sibling.cu_ctx as usize, retained as usize);
            Arc::get_mut(&mut first).unwrap().primary_lifetime = Some(lifetime.clone());
            Arc::get_mut(&mut sibling).unwrap().primary_lifetime = Some(lifetime.clone());

            drop(first);
            drop(sibling);
            let after_wrappers = driver.snapshot();
            assert_eq!(
                after_wrappers.releases, 0,
                "tracked wrappers borrow the single anchor retain"
            );
            assert_eq!(
                (after_wrappers.primary_queries, after_wrappers.id_queries),
                (0, 0)
            );
            assert_eq!((after_wrappers.queries, after_wrappers.binds), (0, 0));

            drop(lifetime);
            let after_anchor = driver.snapshot();
            assert_eq!((after_anchor.retains, after_anchor.releases), (1, 1));
            assert_eq!(
                (after_anchor.primary_queries, after_anchor.id_queries),
                (0, 0)
            );
            assert_eq!((after_anchor.queries, after_anchor.binds), (0, 0));
        }
    }

    #[test]
    fn untracked_wrapper_keeps_its_independent_primary_retain() {
        let driver = context_test_seams::Guard::new();
        let retained = unsafe { crate::driver::result::primary_ctx::retain(0) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = context_test_seams::context();
        assert_eq!(context.cu_ctx as usize, retained as usize);

        drop(context);

        let dispatch = driver.snapshot();
        assert_eq!((dispatch.retains, dispatch.releases), (1, 1));
        assert_eq!((dispatch.primary_queries, dispatch.id_queries), (0, 0));
        assert_eq!((dispatch.queries, dispatch.binds), (1, 1));
    }

    #[test]
    fn failed_context_id_probe_latches_its_first_error() {
        let driver = context_test_seams::Guard::new();
        driver.set_primary_active(true);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 17,
            loss: Mutex::new(None),
            anchor: None,
        });
        let mut context = context_test_seams::context();
        Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(lifetime.clone());
        assert_eq!(
            context.bind_to_thread().map_err(|error| error.0 as u32),
            Ok(())
        );
        let healthy = driver.snapshot();
        assert_eq!((healthy.primary_queries, healthy.id_queries), (1, 1));
        assert_eq!((healthy.queries, healthy.binds), (1, 1));

        let error = sys::CUresult::CUDA_ERROR_INVALID_CONTEXT;
        driver.set_context_id_error(Some(error));
        let first = context.bind_to_thread().map_err(|error| error.0 as u32);
        let after_error = driver.snapshot();
        driver.set_context_id_error(None);
        let second = context.bind_to_thread().map_err(|error| error.0 as u32);
        context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        let third = context.bind_to_thread().map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();
        drop(context);

        assert_eq!(first, Err(error as u32));
        assert_eq!(
            second,
            Err(error as u32),
            "failed identity query must permanently refuse later binding"
        );
        assert_eq!(
            third,
            Err(error as u32),
            "later OOM must not replace the actual identity-query error"
        );
        assert_eq!(
            (after_error.primary_queries, after_error.id_queries),
            (2, 2)
        );
        assert_eq!((after_error.queries, after_error.binds), (1, 1));
        assert_eq!((dispatch.primary_queries, dispatch.id_queries), (2, 2));
        assert_eq!((dispatch.queries, dispatch.binds), (1, 1));
        assert_eq!(
            *lifetime.loss.lock().unwrap(),
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                error
            })
        );
    }
}
