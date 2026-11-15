use super::lifetime_event_cleanup_tests::seams as event_seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

pub(crate) mod seams {
    use crate::driver::{sys, DriverError};
    use core::cell::Cell;

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Counts {
        pub(crate) allocation_attempts: usize,
        pub(crate) async_allocations: usize,
        pub(crate) sync_allocations: usize,
        pub(crate) async_frees: usize,
        pub(crate) sync_frees: usize,
        pub(crate) stream_synchronizes: usize,
    }

    #[derive(Clone, Copy, Default)]
    struct State {
        counts: Counts,
        live: Option<(sys::CUdeviceptr, usize, bool)>,
        allocate_error: Option<sys::CUresult>,
    }

    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
        static FREE_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
        static ALLOCATE_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
    }

    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            STATE.set(Some(State::default()));
            Self
        }

        pub(crate) fn set_allocate_hook(&self, hook: impl FnMut() + 'static) {
            ALLOCATE_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_allocate_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.allocate_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn set_free_hook(&self, hook: impl FnMut() + 'static) {
            FREE_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn snapshot(&self) -> Counts {
            STATE.get().unwrap().counts
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            FREE_HOOK.with(|slot| *slot.borrow_mut() = None);
            ALLOCATE_HOOK.with(|slot| *slot.borrow_mut() = None);
            STATE.set(None);
        }
    }

    pub(crate) fn allocate(bytes: usize, asynchronous: bool) -> Option<Result<sys::CUdeviceptr, DriverError>> {
        let mut state = STATE.get()?;
        assert!(state.live.is_none());
        assert_eq!(bytes, 16);
        state.counts.allocation_attempts += 1;
        STATE.set(Some(state));
        ALLOCATE_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        if let Some(error) = state.allocate_error {
            return Some(Err(DriverError(error)));
        }
        if asynchronous {
            state.counts.async_allocations += 1;
        } else {
            state.counts.sync_allocations += 1;
        }
        let pointer = (0xA100 + state.counts.async_allocations + state.counts.sync_allocations)
            as sys::CUdeviceptr;
        state.live = Some((pointer, bytes, asynchronous));
        STATE.set(Some(state));
        Some(Ok(pointer))
    }

    pub(crate) fn free(pointer: sys::CUdeviceptr, asynchronous: bool) -> bool {
        let Some(mut state) = STATE.get() else {
            return false;
        };
        let (owned, bytes, allocator_async) = state.live.unwrap();
        assert_eq!(owned, pointer);
        assert_eq!(bytes, 16);
        assert_eq!(allocator_async, asynchronous);
        if asynchronous {
            state.counts.async_frees += 1;
        } else {
            state.counts.sync_frees += 1;
        }
        state.live = None;
        STATE.set(Some(state));
        FREE_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        true
    }

    pub(crate) fn stream_synchronize(stream: sys::CUstream) -> bool {
        let Some(mut state) = STATE.get() else {
            return false;
        };
        assert!(stream.is_null());
        state.counts.stream_synchronizes += 1;
        STATE.set(Some(state));
        true
    }
}

pub(super) fn tracked_context(
    lifetime: &Arc<PrimaryContextLifetime>,
    asynchronous: bool,
) -> Arc<CudaContext> {
    let mut context = CudaContext::new_tracked(Arc::clone(lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    Arc::get_mut(&mut context).unwrap().has_async_alloc = asynchronous;
    context
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn lost_slice_disarms_all_native_free_and_nested_event_cleanup() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        assert_eq!(events.snapshot().created, 2);
        assert_eq!(
            memory.snapshot().async_allocations + memory.snapshot().sync_allocations,
            1
        );
        lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
        let before = driver.snapshot();

        drop(allocation);

        let freed = memory.snapshot();
        assert_eq!(
            freed.async_frees + freed.sync_frees,
            0,
            "lost slice must disarm every allocator's native free"
        );
        assert_eq!(freed.stream_synchronizes, 0);
        assert_eq!(events.snapshot().destroyed, 0);
        let retry = unsafe { stream.alloc::<u8>(16) }
            .map(|_| ())
            .map_err(|error| error.0 as u32);
        assert_eq!(
            retry,
            Err(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32)
        );
        assert_eq!(
            memory.snapshot().async_allocations + memory.snapshot().sync_allocations,
            1
        );
        let after = driver.snapshot();
        assert_eq!(
            (after.primary_queries, after.id_queries),
            (before.primary_queries, before.id_queries)
        );
        assert_eq!((after.queries, after.binds), (before.queries, before.binds));
        drop(stream);
        drop(context);
        drop(lifetime);
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
fn healthy_slice_free_preserves_allocator_symmetry_and_nested_cleanup() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        assert_eq!(events.snapshot().created, 2);
        let weak = Arc::downgrade(&lifetime);
        memory.set_free_hook(move || {
            let lifetime = weak.upgrade().unwrap();
            let active = {
                lifetime
                    .state
                    .try_lock()
                    .expect("native slice free must run outside the state mutex")
                    .active_leases
            };
            assert_eq!(active, 1, "slice native free must retain cleanup admission");
        });

        drop(allocation);

        let counts = memory.snapshot();
        assert_eq!(counts.async_allocations, usize::from(asynchronous));
        assert_eq!(counts.async_frees, usize::from(asynchronous));
        assert_eq!(counts.sync_allocations, usize::from(!asynchronous));
        assert_eq!(counts.sync_frees, usize::from(!asynchronous));
        assert_eq!(counts.stream_synchronizes, usize::from(!asynchronous));
        assert_eq!(events.snapshot().destroyed, 2);
        drop(stream);
        drop(context);
        drop(lifetime);
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
fn idle_slice_drop_discovers_inactive_primary_before_free_or_nested_cleanup() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        let before = driver.snapshot();
        driver.set_primary_active(false);

        drop(allocation);

        let freed = memory.snapshot();
        assert_eq!(freed.async_frees + freed.sync_frees, 0);
        assert_eq!(freed.stream_synchronizes, 0);
        assert_eq!(events.snapshot().destroyed, 0);
        let after = driver.snapshot();
        assert_eq!(
            (after.primary_queries, after.id_queries),
            (before.primary_queries + 1, before.id_queries)
        );
        assert_eq!((after.queries, after.binds), (before.queries, before.binds));
        assert_eq!(
            lifetime.loss().unwrap().error as u32,
            sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32
        );
        drop(stream);
        drop(context);
        drop(lifetime);
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
fn native_slice_cleanup_bind_failure_quarantines_before_free() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;
        driver.set_current_query_error(Some(error));
        let before = driver.snapshot();

        drop(allocation);

        let freed = memory.snapshot();
        assert_eq!(freed.async_frees + freed.sync_frees, 0);
        assert_eq!(freed.stream_synchronizes, 0);
        assert_eq!(events.snapshot().destroyed, 0);
        assert_eq!(driver.snapshot().queries, before.queries + 1);
        assert_eq!(lifetime.loss().unwrap().error as u32, error as u32);
        driver.set_current_query_error(None);
        let stopped = driver.snapshot();
        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            context.bind_to_thread().map_err(|error| error.0 as u32),
            Err(error as u32)
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
                stopped.primary_queries,
                stopped.id_queries,
                stopped.queries,
                stopped.binds
            )
        );
        drop(stream);
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
