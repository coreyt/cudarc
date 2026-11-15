use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

pub(crate) mod seams {
    use crate::driver::{sys, DriverError};
    use core::cell::Cell;

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Counts {
        pub(crate) create_attempts: usize,
        pub(crate) created: usize,
        pub(crate) destroyed: usize,
        pub(crate) synchronized: usize,
        pub(crate) queried: usize,
        pub(crate) recorded: usize,
    }

    #[derive(Clone, Copy, Default)]
    struct State {
        counts: Counts,
        create_error: Option<sys::CUresult>,
        fail_on_attempt: Option<usize>,
        destroy_error: Option<sys::CUresult>,
        query_error: Option<sys::CUresult>,
        record_error: Option<sys::CUresult>,
        synchronize_error: Option<sys::CUresult>,
    }

    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
        static DESTROY_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
        static CREATE_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
        static QUERY_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
        static RECORD_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
        static SYNCHRONIZE_HOOK: std::cell::RefCell<Option<std::boxed::Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
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

        pub(crate) fn fail_on_create_attempt(&self, attempt: usize, error: sys::CUresult) {
            let mut state = STATE.get().unwrap();
            state.create_error = Some(error);
            state.fail_on_attempt = Some(attempt);
            STATE.set(Some(state));
        }

        pub(crate) fn set_create_hook(&self, hook: impl FnMut() + 'static) {
            CREATE_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_destroy_hook(&self, hook: impl FnMut() + 'static) {
            DESTROY_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_destroy_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.destroy_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn set_query_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.query_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn set_query_hook(&self, hook: impl FnMut() + 'static) {
            QUERY_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_record_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.record_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn set_record_hook(&self, hook: impl FnMut() + 'static) {
            RECORD_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_synchronize_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.synchronize_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn set_synchronize_hook(&self, hook: impl FnMut() + 'static) {
            SYNCHRONIZE_HOOK.with(|slot| *slot.borrow_mut() = Some(std::boxed::Box::new(hook)));
        }

        pub(crate) fn set_create_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.create_error = error;
            state.fail_on_attempt = None;
            STATE.set(Some(state));
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            DESTROY_HOOK.with(|slot| *slot.borrow_mut() = None);
            CREATE_HOOK.with(|slot| *slot.borrow_mut() = None);
            QUERY_HOOK.with(|slot| *slot.borrow_mut() = None);
            RECORD_HOOK.with(|slot| *slot.borrow_mut() = None);
            SYNCHRONIZE_HOOK.with(|slot| *slot.borrow_mut() = None);
            STATE.set(None);
        }
    }

    pub(crate) fn create(_flags: sys::CUevent_flags) -> Option<Result<sys::CUevent, DriverError>> {
        let mut state = STATE.get()?;
        state.counts.create_attempts += 1;
        STATE.set(Some(state));
        CREATE_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        let failure = if state
            .fail_on_attempt
            .is_none_or(|attempt| attempt == state.counts.create_attempts)
        {
            state.create_error
        } else {
            None
        };
        let outcome = match failure {
            Some(error) => Err(DriverError(error)),
            None => {
                state.counts.created += 1;
                Ok((0xE700 + state.counts.created) as sys::CUevent)
            }
        };
        STATE.set(Some(state));
        Some(outcome)
    }

    pub(crate) fn query(event: sys::CUevent) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        let handle = event as usize;
        assert!(handle > 0xE700 && handle <= 0xE700 + state.counts.created);
        state.counts.queried += 1;
        STATE.set(Some(state));
        QUERY_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        Some(match state.query_error {
            Some(error) => Err(DriverError(error)),
            None => Ok(()),
        })
    }

    pub(crate) fn record(
        event: sys::CUevent,
        _stream: sys::CUstream,
    ) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        let handle = event as usize;
        assert!(handle > 0xE700 && handle <= 0xE700 + state.counts.created);
        state.counts.recorded += 1;
        STATE.set(Some(state));
        RECORD_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        Some(match state.record_error {
            Some(error) => Err(DriverError(error)),
            None => Ok(()),
        })
    }

    pub(crate) fn synchronize(event: sys::CUevent) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        let handle = event as usize;
        assert!(handle > 0xE700 && handle <= 0xE700 + state.counts.created);
        state.counts.synchronized += 1;
        STATE.set(Some(state));
        SYNCHRONIZE_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        Some(match state.synchronize_error {
            Some(error) => Err(DriverError(error)),
            None => Ok(()),
        })
    }

    pub(crate) fn destroy(event: sys::CUevent) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        let handle = event as usize;
        assert!(handle > 0xE700 && handle <= 0xE700 + state.counts.created);
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
fn lost_event_drop_disarms_without_native_destroy() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(
        (events.snapshot().create_attempts, events.snapshot().created),
        (1, 1)
    );
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    let before = driver.snapshot();

    drop(event);

    assert_eq!(
        events.snapshot().destroyed,
        0,
        "a lost event must disarm its native destroy"
    );
    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries, before.id_queries)
    );
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    assert_eq!(after.releases, 0);
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
fn healthy_event_drop_destroys_its_counted_handle_once() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 0)
    );
    let weak = Arc::downgrade(&lifetime);
    events.set_destroy_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime
            .state
            .try_lock()
            .expect("native cleanup must run outside the admission mutex");
        assert_eq!(
            state.active_leases, 1,
            "event cleanup must hold admission through native destroy"
        );
        assert!(!state.probing);
    });

    drop(event);

    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 1)
    );
    assert_eq!(driver.snapshot().releases, 0);
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
fn ordinary_event_create_oom_remains_retryable() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    events.set_create_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    let failed = context
        .new_event(None)
        .map(|_| ())
        .map_err(|error| error.0 as u32);
    assert_eq!(failed, Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32));
    assert_eq!(
        (
            events.snapshot().create_attempts,
            events.snapshot().created,
            events.snapshot().destroyed
        ),
        (1, 0, 0)
    );
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    events.set_create_error(None);

    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    drop(event);

    assert_eq!(
        (
            events.snapshot().create_attempts,
            events.snapshot().created,
            events.snapshot().destroyed
        ),
        (2, 1, 1)
    );
    assert!(lifetime.state.lock().unwrap().loss.is_none());
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
fn idle_event_drop_discovers_inactive_primary_before_destroy() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(events.snapshot().created, 1);
    let before = driver.snapshot();
    driver.set_primary_active(false);

    drop(event);

    let after = driver.snapshot();
    assert_eq!(
        events.snapshot().destroyed,
        0,
        "discovered loss must stop native event cleanup"
    );
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries + 1, before.id_queries)
    );
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    let loss = lifetime.state.lock().unwrap().loss.unwrap();
    assert_eq!(
        loss.refusal().driver_status() as u32,
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
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
fn idle_event_drop_refuses_failed_primary_or_identity_queries() {
    for identity_error in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime);
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let before = driver.snapshot();
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;
        if identity_error {
            driver.set_context_id_error(Some(error));
        } else {
            driver.set_primary_error(Some(error));
        }

        drop(event);

        let after = driver.snapshot();
        assert_eq!(
            (events.snapshot().created, events.snapshot().destroyed),
            (1, 0)
        );
        assert_eq!(after.primary_queries, before.primary_queries + 1);
        assert_eq!(
            after.id_queries,
            before.id_queries + usize::from(identity_error)
        );
        assert_eq!((after.queries, after.binds), (before.queries, before.binds));
        assert_eq!(
            lifetime
                .state
                .lock()
                .unwrap()
                .loss
                .unwrap()
                .refusal()
                .driver_status() as u32,
            error as u32
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
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
fn healthy_event_destroy_error_is_recorded_without_panicking_or_latching_loss() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime);
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let error = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    events.set_destroy_error(Some(error));

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(event)));

    assert!(outcome.is_ok());
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 1)
    );
    assert_eq!(
        context.check_err().map_err(|error| error.0 as u32),
        Err(error as u32)
    );
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
