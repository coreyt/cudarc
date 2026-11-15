#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::{
    lifetime_event_cleanup_tests::seams as event_seams,
    lifetime_slice_cleanup_tests::{seams as allocation_seams, tracked_context},
};
use crate::driver::{
    result::context_test_seams, sys, CudaSlice, CudaStream, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Operation {
    Memset,
    Htod,
    Dtoh,
    Dtod,
}

impl Operation {
    fn run(
        self,
        stream: &Arc<CudaStream>,
        allocation: &mut CudaSlice<u8>,
    ) -> Result<(), DriverError> {
        match self {
            Self::Memset => stream.memset_zeros(&mut allocation.slice_mut(..)),
            Self::Htod => stream.memcpy_htod(&[7u8; 16], allocation),
            Self::Dtoh => stream.memcpy_dtoh(&allocation.slice(..), &mut [0u8; 16]),
            Self::Dtod => {
                let (source, mut target) = allocation.split_at_mut(8);
                stream.memcpy_dtod(&source, &mut target)
            }
        }
    }
}

pub(crate) mod seams {
    use super::Operation;
    use crate::driver::{sys, DriverError};
    use std::cell::{Cell, RefCell};

    #[derive(Clone, Copy, Default)]
    struct State {
        calls: [usize; 4],
    }
    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
        static HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
    }
    pub(crate) struct Guard;
    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            STATE.set(Some(State::default()));
            Self
        }
        pub(crate) fn set_hook(&self, hook: impl FnMut() + 'static) {
            HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn calls(&self) -> [usize; 4] {
            STATE.get().unwrap().calls
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            HOOK.with(|slot| *slot.borrow_mut() = None);
            STATE.set(None);
        }
    }
    pub(crate) fn dispatch(
        operation: Operation,
        pointer: sys::CUdeviceptr,
        bytes: usize,
        stream: sys::CUstream,
    ) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        assert!(stream.is_null());
        assert!(pointer >= 0xA101 && pointer <= 0xA109);
        assert_eq!(
            bytes,
            if matches!(operation, Operation::Dtod) {
                8
            } else {
                16
            }
        );
        state.calls[operation as usize] += 1;
        STATE.set(Some(state));
        HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        Some(Ok(()))
    }
}

fn assert_admitted(lifetime: &PrimaryContextLifetime, expected: usize) {
    let state = lifetime
        .state
        .try_lock()
        .expect("memory use must dispatch outside the lifetime mutex");
    assert_eq!(
        state.active_leases, expected,
        "memory use must retain admission through native dispatch"
    );
    assert!(!state.probing);
}

fn check_native_admission(operation: Operation) {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let use_calls = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    use_calls.set_hook(move || assert_admitted(&weak.upgrade().unwrap(), 1));

    assert_eq!(
        operation
            .run(&stream, &mut allocation)
            .map_err(|error| error.0 as u32),
        Ok(())
    );

    let mut expected = [0; 4];
    expected[operation as usize] = 1;
    assert_eq!(use_calls.calls(), expected);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.loss().is_none());
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (2, 2)
    );
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn memset_holds_admission_through_native_dispatch() {
    check_native_admission(Operation::Memset);
}
#[test]
fn host_to_device_copy_holds_admission_through_native_dispatch() {
    check_native_admission(Operation::Htod);
}
#[test]
fn device_to_host_copy_holds_admission_through_native_dispatch() {
    check_native_admission(Operation::Dtoh);
}
#[test]
fn device_to_device_copy_holds_admission_through_native_dispatch() {
    check_native_admission(Operation::Dtod);
}

#[test]
fn known_loss_refuses_all_memory_use_without_native_calls() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let _memory = allocation_seams::Guard::new();
    let use_calls = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();
    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Err(lost as u32)
        );
    }
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
    assert_eq!(use_calls.calls(), [0; 4]);
    drop(allocation);
    assert_eq!(events.snapshot().destroyed, 0);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn memory_use_shares_an_outer_epoch_and_releases_its_own_lease() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let _memory = allocation_seams::Guard::new();
    let use_calls = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let weak = Arc::downgrade(&lifetime);
    use_calls.set_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));
    let before = driver.snapshot();
    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Ok(())
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
    }
    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries, before.id_queries)
    );
    assert_eq!(use_calls.calls(), [1; 4]);
    drop(outer);
    drop(allocation);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn preexisting_ordinary_oom_is_consumed_once_before_memory_use_and_can_retry() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let _memory = allocation_seams::Guard::new();
    let use_calls = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        context.record_err::<()>(Err(DriverError(oom)));
        let before = use_calls.calls();
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert_eq!(use_calls.calls(), before);
        assert!(lifetime.loss().is_none());
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Ok(())
        );
    }
    assert_eq!(use_calls.calls(), [1; 4]);
    drop(allocation);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn memory_pointer_guards_drop_while_outer_admission_remains_held() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let streams = super::lifetime_stream_cleanup_tests::seams::Guard::new();
    let use_calls = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let background = context
        .new_stream()
        .map_err(|error| error.0 as u32)
        .unwrap();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    events.set_record_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));

    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Ok(())
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    }

    assert_eq!(use_calls.calls(), [1; 4]);
    assert_eq!(events.snapshot().recorded, 5);
    assert_eq!(streams.snapshot().waits, 8);
    assert!(lifetime.loss().is_none());
    events.set_record_hook(|| {});
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    assert_eq!(events.snapshot().destroyed, 2);
    drop(background);
    assert_eq!(streams.snapshot().destroyed, 1);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
