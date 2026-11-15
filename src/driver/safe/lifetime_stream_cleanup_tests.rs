use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime};
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub(crate) mod seams {
    use crate::driver::{result::stream::StreamKind, sys, DriverError};
    use core::cell::Cell;

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Counts {
        pub(crate) created: usize,
        pub(crate) destroyed: usize,
        pub(crate) context_synchronizes: usize,
    }

    #[derive(Clone, Copy, Default)]
    struct State {
        counts: Counts,
        destroy_error: Option<sys::CUresult>,
    }

    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
        static DESTROY_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
    }

    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            STATE.set(Some(State::default()));
            Self
        }

        pub(crate) fn snapshot(&self) -> Counts {
            STATE.get().unwrap().counts
        }

        pub(crate) fn set_destroy_hook(&self, hook: impl FnMut() + 'static) {
            DESTROY_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_destroy_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.destroy_error = error;
            STATE.set(Some(state));
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            DESTROY_HOOK.with(|slot| *slot.borrow_mut() = None);
            STATE.set(None);
        }
    }

    pub(crate) fn create(kind: &StreamKind) -> Option<sys::CUstream> {
        let mut state = STATE.get()?;
        assert!(matches!(kind, StreamKind::NonBlocking));
        state.counts.created += 1;
        STATE.set(Some(state));
        Some((0x5700 + state.counts.created) as sys::CUstream)
    }

    pub(crate) fn destroy(stream: sys::CUstream) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        let handle = stream as usize;
        assert!(handle > 0x5700 && handle <= 0x5700 + state.counts.created);
        state.counts.destroyed += 1;
        STATE.set(Some(state));
        DESTROY_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        Some(match state.destroy_error {
            Some(error) => Err(DriverError(error)),
            None => Ok(()),
        })
    }

    pub(crate) fn context_synchronize() -> bool {
        let Some(mut state) = STATE.get() else {
            return false;
        };
        state.counts.context_synchronizes += 1;
        STATE.set(Some(state));
        true
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
fn lost_owned_stream_disarms_without_native_destroy() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!(streams.snapshot().created, 1);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1);
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    let before = driver.snapshot();
    let synchronized = streams.snapshot().context_synchronizes;

    drop(stream);

    assert_eq!(streams.snapshot().destroyed, 0, "a lost owned stream must disarm its native destroy");
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!(streams.snapshot().context_synchronizes, synchronized);
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries, before.id_queries));
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    assert_eq!(after.releases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn healthy_owned_stream_destroys_its_counted_handle_once() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (1, 0));
    assert_eq!(streams.snapshot().context_synchronizes, 1);
    let weak = Arc::downgrade(&lifetime);
    streams.set_destroy_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime.state.try_lock().expect("native stream cleanup must run outside the admission mutex");
        assert_eq!(state.active_leases, 1, "stream cleanup must hold admission through native destroy");
        assert!(!state.probing);
    });

    drop(stream);

    assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (1, 1));
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!(driver.snapshot().releases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]
#[test]
fn idle_owned_stream_drop_discovers_inactive_primary_before_destroy() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!(streams.snapshot().created, 1);
    let before = driver.snapshot();
    let synchronizes = streams.snapshot().context_synchronizes;
    driver.set_primary_active(false);

    drop(stream);

    let after = driver.snapshot();
    assert_eq!(streams.snapshot().destroyed, 0, "discovered loss must stop native stream cleanup");
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!(streams.snapshot().context_synchronizes, synchronizes);
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries + 1, before.id_queries));
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    assert_eq!(lifetime.state.lock().unwrap().loss.unwrap().error as u32, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]
#[test]
fn idle_owned_stream_drop_refuses_failed_primary_or_identity_queries() {
    for identity_error in [false, true] {
        let driver = context_test_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = tracked_context(&lifetime);
        let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
        let before = driver.snapshot();
        let synchronizes = streams.snapshot().context_synchronizes;
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;
        if identity_error {
            driver.set_context_id_error(Some(error));
        } else {
            driver.set_primary_error(Some(error));
        }

        drop(stream);

        let after = driver.snapshot();
        assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (1, 0));
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        assert_eq!(streams.snapshot().context_synchronizes, synchronizes);
        assert_eq!(after.primary_queries, before.primary_queries + 1);
        assert_eq!(after.id_queries, before.id_queries + usize::from(identity_error));
        assert_eq!((after.queries, after.binds), (before.queries, before.binds));
        assert_eq!(lifetime.state.lock().unwrap().loss.unwrap().error as u32, error as u32);
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(context);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}

#[cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]
#[test]
fn healthy_stream_destroy_error_is_recorded_without_panicking_or_latching_loss() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    let error = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    streams.set_destroy_error(Some(error));

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(stream)));

    assert!(outcome.is_ok());
    assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (1, 1));
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Err(error as u32));
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]
#[test]
fn borrowed_default_stream_drop_uses_no_native_cleanup_after_loss() {
    for per_thread in [false, true] {
        let driver = context_test_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = tracked_context(&lifetime);
        let stream = if per_thread { context.per_thread_stream() } else { context.default_stream() };
        lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
        let before = driver.snapshot();

        drop(stream);

        let after = driver.snapshot();
        assert_eq!((streams.snapshot().created, streams.snapshot().destroyed, streams.snapshot().context_synchronizes), (0, 0, 0));
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries, before.id_queries));
        assert_eq!((after.queries, after.binds), (before.queries, before.binds));
        assert_eq!(after.releases, 0);
        drop(context);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}
