use super::lifetime_event_cleanup_tests::seams as event_seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

pub(crate) mod seams {
    use crate::driver::{sys, DriverError};
    use core::cell::{Cell, RefCell};
    use core::ffi::{c_uint, c_void};

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Counts {
        pub(crate) attempts: usize,
        pub(crate) allocated: usize,
        pub(crate) free_attempts: usize,
        pub(crate) freed: usize,
    }

    #[derive(Clone, Copy, Default)]
    struct State {
        counts: Counts,
        error: Option<sys::CUresult>,
        free_error: Option<sys::CUresult>,
    }

    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
        static BUFFER: RefCell<Option<Box<[u64; 2]>>> = const { RefCell::new(None) };
        static ALLOCATE_HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
        static FREE_HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
    }

    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            assert!(BUFFER.with(|buffer| buffer.borrow().is_none()));
            STATE.set(Some(State::default()));
            Self
        }

        pub(crate) fn set_allocate_hook(&self, hook: impl FnMut() + 'static) {
            ALLOCATE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn set_free_hook(&self, hook: impl FnMut() + 'static) {
            FREE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn set_free_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.free_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn snapshot(&self) -> Counts {
            STATE.get().unwrap().counts
        }

        pub(crate) fn set_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.error = error;
            STATE.set(Some(state));
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            ALLOCATE_HOOK.with(|slot| *slot.borrow_mut() = None);
            FREE_HOOK.with(|slot| *slot.borrow_mut() = None);
            let buffer = BUFFER.with(|buffer| buffer.borrow_mut().take());
            drop(buffer);
            STATE.set(None);
        }
    }

    pub(crate) fn allocate(
        bytes: usize,
        _flags: c_uint,
    ) -> Option<Result<*mut c_void, DriverError>> {
        let mut state = STATE.get()?;
        assert_eq!(bytes, 16);
        state.counts.attempts += 1;
        STATE.set(Some(state));
        ALLOCATE_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        let result = match state.error {
            Some(error) => Err(DriverError(error)),
            None => {
                let mut buffer = Box::new([0u64; 2]);
                let pointer = buffer.as_mut_ptr().cast::<c_void>();
                BUFFER.with(|slot| {
                    assert!(slot.borrow().is_none());
                    *slot.borrow_mut() = Some(buffer);
                });
                state.counts.allocated += 1;
                Ok(pointer)
            }
        };
        STATE.set(Some(state));
        Some(result)
    }

    pub(crate) fn free(pointer: *mut c_void) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        BUFFER.with(|slot| {
            assert_eq!(
                slot.borrow().as_ref().unwrap().as_ptr().cast::<c_void>(),
                pointer.cast_const()
            )
        });
        state.counts.free_attempts += 1;
        STATE.set(Some(state));
        FREE_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        if let Some(error) = state.free_error {
            return Some(Err(DriverError(error)));
        }
        let buffer = BUFFER.with(|slot| slot.borrow_mut().take()).unwrap();
        drop(buffer);
        state.counts.freed += 1;
        STATE.set(Some(state));
        Some(Ok(()))
    }
}

fn tracked_context(lifetime: &Arc<PrimaryContextLifetime>) -> Arc<CudaContext> {
    CudaContext::new_tracked(Arc::clone(lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap()
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn lost_pinned_host_drop_disarms_sync_free_and_nested_event() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let allocation = unsafe { context.alloc_pinned::<u64>(2) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(host.snapshot().allocated, 1);
    assert_eq!(events.snapshot().created, 1);
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    let before = driver.snapshot();

    drop(allocation);

    assert_eq!(
        host.snapshot().freed,
        0,
        "lost pinned host storage must disarm native free_host"
    );
    assert_eq!(
        (events.snapshot().synchronized, events.snapshot().destroyed),
        (0, 0)
    );
    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries, before.id_queries)
    );
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn healthy_pinned_host_drop_synchronizes_and_frees_once() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let allocation = unsafe { context.alloc_pinned::<u64>(2) }
        .map_err(|error| error.0 as u32)
        .unwrap();

    let weak = Arc::downgrade(&lifetime);
    host.set_free_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime
            .state
            .try_lock()
            .expect("native free_host must run outside lifetime mutex");
        assert_eq!(
            state.active_leases, 1,
            "native pinned cleanup must retain admission"
        );
    });

    drop(allocation);

    assert_eq!((host.snapshot().allocated, host.snapshot().freed), (1, 1));
    assert_eq!(
        (
            events.snapshot().created,
            events.snapshot().synchronized,
            events.snapshot().destroyed
        ),
        (1, 1, 1)
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn pinned_host_allocation_oom_releases_its_event_and_is_retryable() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    host.set_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    let failed = unsafe { context.alloc_pinned::<u64>(2) }
        .map(|_| ())
        .map_err(|error| error.0 as u32);
    assert_eq!(failed, Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32));
    assert_eq!(
        (
            host.snapshot().attempts,
            host.snapshot().allocated,
            host.snapshot().freed
        ),
        (1, 0, 0)
    );
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 1)
    );
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    host.set_error(None);

    let retry = unsafe { context.alloc_pinned::<u64>(2) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    drop(retry);

    assert_eq!(
        (
            host.snapshot().attempts,
            host.snapshot().allocated,
            host.snapshot().freed
        ),
        (2, 1, 1)
    );
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (2, 2)
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn idle_pinned_drop_discovers_inactive_primary_before_sync_or_free() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let allocation = unsafe { context.alloc_pinned::<u64>(2) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    driver.set_primary_active(false);
    let before = driver.snapshot();
    drop(allocation);
    assert_eq!(host.snapshot().free_attempts, 0);
    assert_eq!(
        (events.snapshot().synchronized, events.snapshot().destroyed),
        (0, 0)
    );
    assert_eq!(
        lifetime.loss().unwrap().cause,
        crate::driver::PrimaryContextLossCause::PrimaryInactive
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.primary_queries + 1,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn pending_ordinary_oom_does_not_abandon_healthy_pinned_cleanup() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let allocation = unsafe { context.alloc_pinned::<u64>(2) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
    drop(allocation);
    assert_eq!(
        (host.snapshot().free_attempts, host.snapshot().freed),
        (1, 1)
    );
    assert_eq!(
        (events.snapshot().synchronized, events.snapshot().destroyed),
        (1, 1)
    );
    assert!(lifetime.loss().is_none());
    assert_eq!(
        context.check_err().map_err(|error| error.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
    );
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn native_pinned_cleanup_binding_failure_stops_before_sync_and_free() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let allocation = unsafe { context.alloc_pinned::<u64>(2) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    driver.set_current_query_error(Some(sys::CUresult::CUDA_ERROR_UNKNOWN));
    drop(allocation);
    driver.set_current_query_error(None);
    assert_eq!(host.snapshot().free_attempts, 0);
    assert_eq!(
        (events.snapshot().synchronized, events.snapshot().destroyed),
        (0, 0)
    );
    assert_eq!(
        lifetime.loss().unwrap().cause,
        crate::driver::PrimaryContextLossCause::DriverStatus(sys::CUresult::CUDA_ERROR_UNKNOWN)
    );
    let before = driver.snapshot();
    assert_eq!(
        context.bind_to_thread().map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_UNKNOWN as u32)
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.primary_queries,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn untracked_pinned_cleanup_keeps_legacy_sync_error_continuation() {
    for error in [
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
    ] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let host = seams::Guard::new();
        let context = CudaContext::new(0).map_err(|e| e.0 as u32).unwrap();
        let allocation = unsafe { context.alloc_pinned::<u64>(2) }
            .map_err(|e| e.0 as u32)
            .unwrap();
        events.set_synchronize_error(Some(error));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(allocation)));
        events.set_synchronize_error(None);
        assert!(result.is_ok());
        assert_eq!(
            (
                host.snapshot().free_attempts,
                host.snapshot().freed,
                events.snapshot().synchronized,
                events.snapshot().destroyed
            ),
            (1, 1, 1, 1)
        );
        assert!(context.primary_lifetime.is_none());
        assert_eq!(
            context.check_err().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
        drop(context);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn untracked_pinned_free_failure_records_once_without_panicking() {
    for error in [
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
        sys::CUresult::CUDA_ERROR_UNKNOWN,
    ] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let host = seams::Guard::new();
        let context = CudaContext::new(0).map_err(|e| e.0 as u32).unwrap();
        let allocation = unsafe { context.alloc_pinned::<u64>(2) }
            .map_err(|e| e.0 as u32)
            .unwrap();
        host.set_free_error(Some(error));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(allocation)));
        host.set_free_error(None);
        assert!(result.is_ok());
        assert_eq!(
            (
                host.snapshot().free_attempts,
                host.snapshot().freed,
                events.snapshot().synchronized,
                events.snapshot().destroyed
            ),
            (1, 0, 1, 1)
        );
        assert_eq!(
            context.check_err().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
        drop(context);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
