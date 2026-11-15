use crate::driver::{sys, DriverError};
use std::sync::{Condvar, LockResult, Mutex, MutexGuard};

/// Shared lifetime state for one tracked primary-context generation.
///
/// Terminal loss records the first observation for every attached context
/// wrapper. A process must restart before using the device after terminal loss.
/// Native ownership is held by the shared anchor. Its final cleanup performs
/// the balanced primary release without binding a context, including after loss.
#[derive(Debug)]
pub struct PrimaryContextLifetime {
    pub(crate) recorded_id: u64,
    pub(crate) state: Mutex<PrimaryContextState>,
    pub(crate) ready: Condvar,
    #[cfg(test)]
    pub(crate) probe_waiters: std::sync::atomic::AtomicUsize,
    pub(crate) anchor: Option<PrimaryContextAnchor>,
}

#[derive(Debug, Default)]
pub(crate) struct PrimaryContextState {
    pub(crate) loss: Option<PrimaryContextLoss>,
    pub(crate) active_leases: usize,
    pub(crate) probing: bool,
}

/// Native library domain responsible for an observed cleanup status.
///
/// Domain tags are available independently of optional library features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeLibrary {
    /// NVIDIA cuBLAS.
    Cublas,
    /// NVIDIA cuRAND.
    Curand,
}

/// Authoritative first cause of terminal generation refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimaryContextLossCause {
    /// An actual status observed from a native driver call.
    DriverStatus(sys::CUresult),
    /// An actual native library cleanup status; no driver status was observed.
    NativeLibraryCleanup {
        /// The native library that reported cleanup failure.
        library: NativeLibrary,
        /// The unchanged numeric status reported by that library.
        status: u32,
    },
    /// A successful primary-state query observed that the context was inactive.
    PrimaryInactive,
    /// A successful identity query observed a different context identity.
    IdentityChanged,
    /// Conservative refusal after poison or unwind without a native status.
    SyntheticUnknown,
}

/// Cached projection of terminal loss for driver-returning operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimaryContextRefusal {
    /// An actual native driver status, including actual native UNKNOWN.
    DriverStatus(sys::CUresult),
    /// SYNTHETIC CONTEXT_IS_DESTROYED after an inactive or changed-ID observation.
    SyntheticContextDestroyed,
    /// SYNTHETIC UNKNOWN; inspect the authoritative cause for its origin.
    SyntheticUnknown,
}

impl PrimaryContextRefusal {
    /// Projects refusal into the status required by driver-returning APIs.
    ///
    /// UNKNOWN from `SyntheticUnknown` and CONTEXT_IS_DESTROYED from
    /// `SyntheticContextDestroyed` are SYNTHETIC, never measured driver statuses.
    /// Inspect the refusal variant and cached loss cause to distinguish their
    /// origin. This pure conversion performs no native dispatch.
    pub fn driver_status(self) -> sys::CUresult {
        match self {
            Self::DriverStatus(status) => status,
            Self::SyntheticUnknown => sys::CUresult::CUDA_ERROR_UNKNOWN,
            Self::SyntheticContextDestroyed => sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        }
    }
}

/// Healthy admission to one tracked primary-context generation.
///
/// Keep this lease through native work and its completion or unwind. Dropping
/// it returns admission without probing, binding or synchronizing the context.
#[derive(Debug)]
pub struct PrimaryContextLease {
    lifetime: std::sync::Arc<PrimaryContextLifetime>,
}

#[derive(Debug)]
pub(crate) struct PrimaryContextAnchor {
    pub(crate) device: sys::CUdevice,
    pub(crate) context: sys::CUcontext,
    pub(crate) ordinal: usize,
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

struct PrimaryContextProbe<'a> {
    lifetime: &'a PrimaryContextLifetime,
    completed: bool,
}

impl Drop for PrimaryContextProbe<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let mut state = self.lifetime.lock_state();
            if state.loss.is_none() {
                state.loss = Some(PrimaryContextLoss {
                    recorded_id: self.lifetime.recorded_id,
                    observed_id: None,
                    cause: PrimaryContextLossCause::SyntheticUnknown,
                });
            }
            state.probing = false;
            drop(state);
            self.lifetime.ready.notify_all();
        }
    }
}

impl Drop for PrimaryContextLease {
    fn drop(&mut self) {
        let mut state = self.lifetime.lock_state();
        state.active_leases -= 1;
    }
}

impl PrimaryContextLifetime {
    /// Retains the primary context on `ordinal` and records its identity.
    ///
    /// Keep one shared lifetime for a device, including across context wrappers.
    /// After terminal loss, do not create a replacement lifetime in this process.
    ///
    /// # Errors
    /// Returns NOT_SUPPORTED when tracking is unavailable for the build or
    /// driver, or the driver error from initialization, device lookup, primary
    /// retain or identity lookup. Every successful retain has an owner before
    /// subsequent fallible initialization and is released on failure.
    pub fn new(ordinal: usize) -> Result<std::sync::Arc<Self>, DriverError> {
        if !crate::driver::result::ctx::has_tracking_symbols() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_SUPPORTED));
        }
        let device_ordinal = core::ffi::c_int::try_from(ordinal)
            .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_DEVICE))?;
        crate::driver::result::init()?;
        let device = crate::driver::result::device::get(device_ordinal)?;
        let context = unsafe { crate::driver::result::primary_ctx::retain(device) }?;
        let anchor = PrimaryContextAnchor {
            device,
            context,
            ordinal,
        };
        let recorded_id = unsafe { crate::driver::result::ctx::get_id(anchor.context) }?;
        Ok(std::sync::Arc::new(Self {
            recorded_id,
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(anchor),
        }))
    }

    /// Acquires healthy admission for native work using this generation.
    ///
    /// An idle generation is probed before admission. Nested or concurrent
    /// admitted work shares its healthy epoch. Callers must coordinate any
    /// external reset outside all construction, work, probe and cleanup leases.
    ///
    /// # Errors
    /// Returns the first terminal loss or a probe error, or NOT_SUPPORTED when
    /// admission is unavailable. Terminal refusal performs no native query.
    pub fn admit(self: &std::sync::Arc<Self>) -> Result<PrimaryContextLease, DriverError> {
        let mut state = self.lock_state();
        loop {
            if let Some(loss) = state.loss {
                return Err(DriverError(loss.refusal().driver_status()));
            }
            if state.active_leases > 0 {
                state.active_leases += 1;
                drop(state);
                return Ok(PrimaryContextLease {
                    lifetime: std::sync::Arc::clone(self),
                });
            }
            if state.probing {
                #[cfg(test)]
                self.probe_waiters
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let waited = self.ready.wait(state);
                #[cfg(test)]
                self.probe_waiters
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                state = self.recover_state(waited);
                continue;
            }
            state.probing = true;
            drop(state);
            break;
        }

        let mut probe = PrimaryContextProbe {
            lifetime: self,
            completed: false,
        };
        let result = match self.anchor.as_ref() {
            Some(anchor) => self.check_primary(anchor.device, anchor.context),
            None => Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_SUPPORTED)),
        };
        if let Err(error) = result {
            self.record_loss(error, None);
        }
        let mut state = self.lock_state();
        state.probing = false;
        let result = match state.loss {
            Some(loss) => Err(DriverError(loss.refusal().driver_status())),
            None => {
                state.active_leases += 1;
                Ok(PrimaryContextLease {
                    lifetime: std::sync::Arc::clone(self),
                })
            }
        };
        probe.completed = true;
        drop(state);
        self.ready.notify_all();
        result
    }

    fn lock_state(&self) -> MutexGuard<'_, PrimaryContextState> {
        self.recover_state(self.state.lock())
    }

    fn recover_state<'a>(
        &self,
        locked: LockResult<MutexGuard<'a, PrimaryContextState>>,
    ) -> MutexGuard<'a, PrimaryContextState> {
        match locked {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if state.loss.is_none() {
                    state.loss = Some(PrimaryContextLoss {
                        recorded_id: self.recorded_id,
                        observed_id: None,
                        cause: PrimaryContextLossCause::SyntheticUnknown,
                    });
                }
                state.probing = false;
                self.ready.notify_all();
                state
            }
        }
    }

    /// Returns the context identity recorded when this lifetime was retained.
    ///
    /// This cached observation performs no native dispatch and stays unchanged
    /// after terminal loss. It does not establish current device health.
    pub fn recorded_id(&self) -> u64 {
        self.recorded_id
    }

    /// Returns the first cached terminal refusal, without native dispatch.
    ///
    /// Reading does not clear or replace the record. Ordinary OOM alone has no
    /// loss record; conservative terminal OOM and synthetic UNKNOWN do, so callers
    /// must use this observation rather than classify health by status alone.
    pub fn loss(&self) -> Option<PrimaryContextLoss> {
        self.lock_state().loss
    }

    pub(crate) fn error(&self) -> Option<DriverError> {
        self.lock_state()
            .loss
            .map(|loss| DriverError(loss.refusal().driver_status()))
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
        if let Some(error) = self.error() {
            return Err(error);
        }
        if !active {
            self.record_cause(PrimaryContextLossCause::PrimaryInactive, None);
        } else {
            let id = match unsafe { crate::driver::result::ctx::get_id(context) } {
                Ok(id) => id,
                Err(error) => {
                    self.record_loss(error, None);
                    return Err(self.error().unwrap_or(error));
                }
            };
            if id != self.recorded_id {
                self.record_cause(PrimaryContextLossCause::IdentityChanged, Some(id));
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

    pub(crate) fn record_library_cleanup_loss(&self, library: NativeLibrary, status: u32) {
        self.record_cause(
            PrimaryContextLossCause::NativeLibraryCleanup { library, status },
            None,
        );
    }

    pub(super) fn record_loss(&self, error: DriverError, observed_id: Option<u64>) {
        self.record_cause(PrimaryContextLossCause::DriverStatus(error.0), observed_id);
    }

    fn record_cause(&self, cause: PrimaryContextLossCause, observed_id: Option<u64>) {
        let mut state = self.lock_state();
        if state.loss.is_none() {
            state.loss = Some(PrimaryContextLoss {
                recorded_id: self.recorded_id,
                observed_id,
                cause,
            });
        }
        drop(state);
        self.ready.notify_all();
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
    /// The actual driver/library cause, or an explicit synthetic refusal.
    pub cause: PrimaryContextLossCause,
}

impl PrimaryContextLoss {
    /// Derives a cached refusal without querying or formatting native errors.
    ///
    /// Library and unknown causes project to explicit SYNTHETIC UNKNOWN.
    /// Successful inactive/changed-ID observations project to SYNTHETIC
    /// CONTEXT_IS_DESTROYED. Actual native statuses keep `DriverStatus`.
    pub fn refusal(self) -> PrimaryContextRefusal {
        match self.cause {
            PrimaryContextLossCause::DriverStatus(status) => {
                PrimaryContextRefusal::DriverStatus(status)
            }
            PrimaryContextLossCause::PrimaryInactive | PrimaryContextLossCause::IdentityChanged => {
                PrimaryContextRefusal::SyntheticContextDestroyed
            }
            PrimaryContextLossCause::NativeLibraryCleanup { .. }
            | PrimaryContextLossCause::SyntheticUnknown => PrimaryContextRefusal::SyntheticUnknown,
        }
    }
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
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
        });
        let mut first = context_test_seams::context();
        let mut sibling = context_test_seams::context();
        Arc::get_mut(&mut first).unwrap().primary_lifetime = Some(lifetime.clone());
        Arc::get_mut(&mut sibling).unwrap().primary_lifetime = Some(lifetime.clone());
        assert!(!Arc::ptr_eq(&first, &sibling));
        assert!(lifetime.state.lock().unwrap().loss.is_none());
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
            lifetime.state.lock().unwrap().loss,
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                cause: PrimaryContextLossCause::DriverStatus(lost)
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
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
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
        assert!(lifetime.state.lock().unwrap().loss.is_none());
    }

    #[test]
    fn inactive_primary_is_discovered_before_bind_and_stays_terminal() {
        let driver = context_test_seams::Guard::new();
        driver.set_primary_active(false);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 29,
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
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
            lifetime.state.lock().unwrap().loss,
            Some(PrimaryContextLoss {
                recorded_id: 29,
                observed_id: None,
                cause: PrimaryContextLossCause::PrimaryInactive
            })
        );
    }

    #[test]
    fn active_primary_allows_healthy_binding() {
        let driver = context_test_seams::Guard::new();
        driver.set_context_id(31);
        let lifetime = Arc::new(PrimaryContextLifetime {
            recorded_id: 31,
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
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
        assert!(lifetime.state.lock().unwrap().loss.is_none());
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
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
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
            lifetime.state.lock().unwrap().loss,
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: Some(42),
                cause: PrimaryContextLossCause::IdentityChanged
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
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
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
            lifetime.state.lock().unwrap().loss,
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                cause: PrimaryContextLossCause::DriverStatus(error)
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
                cause: PrimaryContextLossCause::DriverStatus(
                    sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
                ),
            });
            let lifetime = Arc::new(PrimaryContextLifetime {
                recorded_id: 17,
                state: Mutex::new(PrimaryContextState {
                    loss,
                    ..PrimaryContextState::default()
                }),
                ready: Condvar::new(),
                #[cfg(test)]
                probe_waiters: std::sync::atomic::AtomicUsize::new(0),
                anchor: Some(PrimaryContextAnchor {
                    device: 0,
                    context: retained,
                    ordinal: 0,
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
            state: Mutex::new(PrimaryContextState::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            probe_waiters: std::sync::atomic::AtomicUsize::new(0),
            anchor: Some(PrimaryContextAnchor {
                device: 0,
                context: unsafe { crate::driver::result::primary_ctx::retain(0) }
                    .map_err(|error| error.0 as u32)
                    .unwrap(),
                ordinal: 0,
            }),
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
            lifetime.state.lock().unwrap().loss,
            Some(PrimaryContextLoss {
                recorded_id: 17,
                observed_id: None,
                cause: PrimaryContextLossCause::DriverStatus(error)
            })
        );
    }

    #[test]
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    )))]
    fn failed_lifetime_identity_initialization_releases_its_retained_anchor() {
        let driver = context_test_seams::Guard::new();
        let error = sys::CUresult::CUDA_ERROR_INVALID_CONTEXT;
        driver.set_context_id_error(Some(error));

        let outcome = PrimaryContextLifetime::new(0)
            .map(|_| ())
            .map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();

        assert_eq!(
            dispatch.retains, 1,
            "supported lifetime construction must reach one counted primary retain"
        );
        assert_eq!(outcome, Err(error as u32));
        assert_eq!(
            (
                dispatch.symbol_checks,
                dispatch.initializations,
                dispatch.device_queries
            ),
            (1, 1, 1)
        );
        assert_eq!(
            (dispatch.retains, dispatch.releases, dispatch.id_queries),
            (1, 1, 1)
        );
        assert_eq!(dispatch.primary_queries, 0);
        assert_eq!((dispatch.queries, dispatch.binds), (0, 0));
    }

    #[test]
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    )))]
    fn lifetime_construction_owns_one_anchor_until_final_drop() {
        let driver = context_test_seams::Guard::new();
        let outcome = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32);
        assert!(
            outcome.is_ok(),
            "supported tracked lifetime must construct successfully"
        );
        let lifetime = outcome.unwrap();
        let initialized = driver.snapshot();
        assert_eq!(
            (
                initialized.symbol_checks,
                initialized.initializations,
                initialized.device_queries
            ),
            (1, 1, 1)
        );
        assert_eq!(
            (
                initialized.retains,
                initialized.releases,
                initialized.id_queries
            ),
            (1, 0, 1)
        );
        assert_eq!(lifetime.recorded_id, 17);
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        let anchor = lifetime.anchor.as_ref().unwrap();
        assert_eq!(
            (anchor.ordinal, anchor.device, anchor.context as usize),
            (0, 0, 0xC7)
        );

        drop(lifetime);

        let released = driver.snapshot();
        assert_eq!((released.retains, released.releases), (1, 1));
        assert_eq!(
            (released.primary_queries, released.queries, released.binds),
            (0, 0, 0)
        );
    }

    #[test]
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    )))]
    fn missing_tracking_symbols_refuse_before_driver_or_retain() {
        let driver = context_test_seams::Guard::new();
        driver.set_tracking_symbols(false);
        let outcome = PrimaryContextLifetime::new(0)
            .map(|_| ())
            .map_err(|error| error.0 as u32);
        let dispatch = driver.snapshot();

        assert_eq!(outcome, Err(sys::CUresult::CUDA_ERROR_NOT_SUPPORTED as u32));
        assert_eq!(dispatch.symbol_checks, 1);
        assert_eq!((dispatch.initializations, dispatch.device_queries), (0, 0));
        assert_eq!(
            (dispatch.retains, dispatch.releases, dispatch.id_queries),
            (0, 0, 0)
        );
        assert_eq!(
            (dispatch.primary_queries, dispatch.queries, dispatch.binds),
            (0, 0, 0)
        );
    }
}

#[cfg(all(
    test,
    not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))
))]
mod fathomdb_primary_admission {
    use super::*;
    use crate::driver::result::context_test_seams;
    use std::sync::Arc;

    #[test]
    fn healthy_nested_leases_share_one_probe_until_last_drop() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let first = lifetime.admit().map_err(|error| error.0 as u32);
        assert!(
            first.is_ok(),
            "healthy lifetime must provide its first admission lease"
        );
        let first = first.unwrap();
        let nested = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
        let admitted = driver.snapshot();
        assert_eq!((admitted.primary_queries, admitted.id_queries), (1, 2));
        assert_eq!((admitted.queries, admitted.binds), (0, 0));
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 2);

        let worker_lifetime = Arc::clone(&lifetime);
        let worker = std::thread::spawn(move || {
            let driver = context_test_seams::Guard::new();
            let lifetime = worker_lifetime;
            let lease = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
            let dispatch = driver.snapshot();
            drop(lease);
            drop(lifetime);
            dispatch
        });
        let concurrent = worker.join().unwrap();
        assert_eq!((concurrent.primary_queries, concurrent.id_queries), (0, 0));
        assert_eq!((concurrent.queries, concurrent.binds), (0, 0));
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 2);

        drop(first);
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
        drop(nested);
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        let drained = driver.snapshot();
        assert_eq!((drained.primary_queries, drained.id_queries), (1, 2));

        let next = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
        let next_epoch = driver.snapshot();
        assert_eq!((next_epoch.primary_queries, next_epoch.id_queries), (2, 3));
        drop(lifetime);
        assert_eq!(driver.snapshot().releases, 0);
        drop(next);
        let final_drop = driver.snapshot();
        assert_eq!((final_drop.retains, final_drop.releases), (1, 1));
        assert_eq!((final_drop.primary_queries, final_drop.id_queries), (2, 3));
        assert_eq!((final_drop.queries, final_drop.binds), (0, 0));
    }

    #[test]
    fn terminal_loss_refuses_new_leases_during_an_active_epoch() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let lease = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        lifetime.record_error(DriverError(lost));
        let before = driver.snapshot();
        let refused = lifetime.admit().map(|_| ()).map_err(|error| error.0 as u32);
        drop(lease);
        let after_drop = lifetime.admit().map(|_| ()).map_err(|error| error.0 as u32);
        let after = driver.snapshot();

        assert_eq!(refused, Err(lost as u32));
        assert_eq!(after_drop, Err(lost as u32));
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(
            (after.primary_queries, after.id_queries),
            (before.primary_queries, before.id_queries)
        );
        assert_eq!((after.queries, after.binds), (0, 0));
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }

    #[derive(Clone, Copy)]
    enum ProbeMode {
        Healthy,
        QueryError,
        Panic,
        OtherLoss,
    }

    struct ProbePair {
        first: Result<(), u32>,
        second: Result<(), u32>,
        first_native: context_test_seams::State,
        second_native: context_test_seams::State,
        panic_ran: bool,
    }

    struct OwnedProbeThreads {
        lifetime: Arc<PrimaryContextLifetime>,
        probe_release: std::sync::mpsc::Sender<()>,
        first_release: std::sync::mpsc::Sender<()>,
        second_release: std::sync::mpsc::Sender<()>,
        first: Option<std::thread::JoinHandle<context_test_seams::State>>,
        second: Option<std::thread::JoinHandle<context_test_seams::State>>,
        finished: bool,
    }

    impl Drop for OwnedProbeThreads {
        fn drop(&mut self) {
            if !self.finished {
                let mut state = self
                    .lifetime
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if state.loss.is_none() {
                    state.loss = Some(PrimaryContextLoss {
                        recorded_id: self.lifetime.recorded_id,
                        observed_id: None,
                        cause: PrimaryContextLossCause::SyntheticUnknown,
                    });
                }
                state.probing = false;
                drop(state);
                self.lifetime.ready.notify_all();
                let _ = self.probe_release.send(());
                let _ = self.first_release.send(());
                let _ = self.second_release.send(());
                if let Some(first) = self.first.take() {
                    let _ = first.join();
                }
                if let Some(second) = self.second.take() {
                    let _ = second.join();
                }
            }
        }
    }

    fn admission_status(
        result: &std::thread::Result<Result<PrimaryContextLease, DriverError>>,
    ) -> Result<(), u32> {
        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => Err(error.0 as u32),
            Err(_) => Err(sys::CUresult::CUDA_ERROR_UNKNOWN as u32),
        }
    }

    fn run_contended_probe(lifetime: &Arc<PrimaryContextLifetime>, mode: ProbeMode) -> ProbePair {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc::channel;
        use std::time::{Duration, Instant};

        let (probe_entered_tx, probe_entered_rx) = channel();
        let (probe_release, probe_release_rx) = channel();
        let (first_release, first_release_rx) = channel();
        let (second_release, second_release_rx) = channel();
        let (first_status_tx, first_status_rx) = channel();
        let (second_status_tx, second_status_rx) = channel();
        let panic_ran = Arc::new(AtomicBool::new(false));
        let mut owned = OwnedProbeThreads {
            lifetime: Arc::clone(lifetime),
            probe_release,
            first_release,
            second_release,
            first: None,
            second: None,
            finished: false,
        };
        let first_lifetime = Arc::clone(lifetime);
        let first_panic_ran = Arc::clone(&panic_ran);
        owned.first = Some(std::thread::spawn(move || {
            let driver = context_test_seams::Guard::new();
            let lifetime = first_lifetime;
            if matches!(mode, ProbeMode::QueryError) {
                driver.set_primary_error(Some(sys::CUresult::CUDA_ERROR_UNKNOWN));
            }
            let probe_lifetime = Arc::clone(&lifetime);
            driver.set_primary_probe_hook(move || {
                assert!(
                    probe_lifetime.state.try_lock().is_ok(),
                    "native probe must run outside the admission state mutex"
                );
                probe_entered_tx.send(()).unwrap();
                probe_release_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("release the owned first probe");
                if matches!(mode, ProbeMode::Panic) {
                    first_panic_ran.store(true, Ordering::SeqCst);
                    panic!("synthetic native-probe unwind");
                }
            });
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lifetime.admit()));
            let _ = first_status_tx.send(admission_status(&result));
            first_release_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("release the owned first lease");
            drop(result);
            let dispatch = driver.snapshot();
            drop(lifetime);
            dispatch
        }));
        probe_entered_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("first admission must enter its native probe");

        let second_lifetime = Arc::clone(lifetime);
        owned.second = Some(std::thread::spawn(move || {
            let driver = context_test_seams::Guard::new();
            let lifetime = second_lifetime;
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lifetime.admit()));
            let _ = second_status_tx.send(admission_status(&result));
            second_release_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("release the owned second lease");
            drop(result);
            let dispatch = driver.snapshot();
            drop(lifetime);
            dispatch
        }));
        let deadline = Instant::now() + Duration::from_secs(3);
        while lifetime.probe_waiters.load(Ordering::SeqCst) == 0 {
            assert!(
                Instant::now() < deadline,
                "second admission must wait behind the reserved probe"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let second = if matches!(mode, ProbeMode::OtherLoss) {
            lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
            Some(
                second_status_rx
                    .recv_timeout(Duration::from_secs(3))
                    .expect("published loss must wake the waiter before the first probe returns"),
            )
        } else {
            None
        };
        owned.probe_release.send(()).unwrap();
        let first = first_status_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("first admission must terminate");
        let second = second.unwrap_or_else(|| {
            second_status_rx
                .recv_timeout(Duration::from_secs(3))
                .expect("second admission must wake and terminate")
        });
        owned.first_release.send(()).unwrap();
        owned.second_release.send(()).unwrap();
        let first_native = owned.first.take().unwrap().join().unwrap();
        let second_native = owned.second.take().unwrap().join().unwrap();
        owned.finished = true;
        ProbePair {
            first,
            second,
            first_native,
            second_native,
            panic_ran: panic_ran.load(Ordering::SeqCst),
        }
    }

    #[test]
    fn concurrent_idle_admission_serializes_one_unlocked_probe() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let pair = run_contended_probe(&lifetime, ProbeMode::Healthy);

        assert_eq!(pair.first, Ok(()));
        assert_eq!(pair.second, Ok(()));
        assert_eq!(
            (
                pair.first_native.primary_queries,
                pair.first_native.id_queries
            ),
            (1, 1)
        );
        assert_eq!(
            (
                pair.second_native.primary_queries,
                pair.second_native.id_queries
            ),
            (0, 0)
        );
        assert_eq!(
            (
                pair.first_native.queries,
                pair.first_native.binds,
                pair.second_native.queries,
                pair.second_native.binds
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(!lifetime.state.lock().unwrap().probing);
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }

    #[test]
    fn failed_probe_wakes_waiters_and_preserves_terminal_refusal() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let pair = run_contended_probe(&lifetime, ProbeMode::QueryError);
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;

        assert_eq!(pair.first, Err(error as u32));
        assert_eq!(pair.second, Err(error as u32));
        assert_eq!(
            (
                pair.first_native.primary_queries,
                pair.first_native.id_queries
            ),
            (1, 0)
        );
        assert_eq!(
            (
                pair.second_native.primary_queries,
                pair.second_native.id_queries
            ),
            (0, 0)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(!lifetime.state.lock().unwrap().probing);
        assert_eq!(
            lifetime.admit().map(|_| ()).map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            (
                driver.snapshot().primary_queries,
                driver.snapshot().id_queries
            ),
            (0, 1)
        );
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }

    #[test]
    fn panicking_probe_wakes_waiters_and_terminally_refuses() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let pair = run_contended_probe(&lifetime, ProbeMode::Panic);
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;

        assert!(pair.panic_ran);
        assert_eq!(pair.first, Err(error as u32));
        assert_eq!(pair.second, Err(error as u32));
        assert_eq!(
            (
                pair.first_native.primary_queries,
                pair.first_native.id_queries
            ),
            (1, 0)
        );
        assert_eq!(
            (
                pair.second_native.primary_queries,
                pair.second_native.id_queries
            ),
            (0, 0)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(!lifetime.state.lock().unwrap().probing);
        assert_eq!(
            lifetime.admit().map(|_| ()).map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            (
                driver.snapshot().primary_queries,
                driver.snapshot().id_queries
            ),
            (0, 1)
        );
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }

    #[test]
    fn published_loss_wakes_waiters_before_the_probe_returns() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let pair = run_contended_probe(&lifetime, ProbeMode::OtherLoss);
        let error = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;

        assert_eq!(pair.first, Err(error as u32));
        assert_eq!(pair.second, Err(error as u32));
        assert_eq!(
            (
                pair.first_native.primary_queries,
                pair.first_native.id_queries
            ),
            (1, 0)
        );
        assert_eq!(
            (
                pair.second_native.primary_queries,
                pair.second_native.id_queries
            ),
            (0, 0)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(!lifetime.state.lock().unwrap().probing);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }

    #[test]
    fn poisoned_admission_state_refuses_without_native_queries() {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _state = lifetime.state.lock().unwrap();
            panic!("synthetic state-owner unwind");
        }));
        assert!(poisoned.is_err());
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;
        let refusal = lifetime.admit().map(|_| ()).map_err(|error| error.0 as u32);
        let repeated = lifetime.admit().map(|_| ()).map_err(|error| error.0 as u32);

        assert_eq!(refusal, Err(error as u32));
        assert_eq!(repeated, Err(error as u32));
        let state = lifetime
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            state
                .loss
                .as_ref()
                .map(|loss| loss.refusal().driver_status() as u32),
            Some(error as u32)
        );
        assert_eq!(state.active_leases, 0);
        assert!(!state.probing);
        drop(state);
        assert_eq!(
            (
                driver.snapshot().primary_queries,
                driver.snapshot().id_queries
            ),
            (0, 1)
        );
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
