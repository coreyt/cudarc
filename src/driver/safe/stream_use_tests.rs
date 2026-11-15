#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::{
    lifetime_event_cleanup_tests::seams as event_seams, lifetime_stream_cleanup_tests::seams,
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, CudaEvent, CudaStream, DriverError,
    PrimaryContextLifetime,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Synchronize,
    Wait,
}

fn run(operation: Operation, stream: &CudaStream, event: &CudaEvent) -> Result<(), DriverError> {
    match operation {
        Operation::Synchronize => stream.synchronize(),
        Operation::Wait => stream.wait(event),
    }
}

fn set_error(streams: &seams::Guard, operation: Operation, error: Option<sys::CUresult>) {
    match operation {
        Operation::Synchronize => streams.set_stream_synchronize_error(error),
        Operation::Wait => streams.set_wait_error(error),
    }
}

fn assert_admitted(lifetime: &PrimaryContextLifetime) {
    let (active, probing) = {
        let state = lifetime
            .state
            .try_lock()
            .expect("native stream use must run outside the lifetime mutex");
        (state.active_leases, state.probing)
    };
    assert_eq!(
        active, 1,
        "native stream use must retain admission through dispatch"
    );
    assert!(!probing);
}

#[test]
fn stream_sync_and_wait_hold_admission_through_native_dispatch() {
    for owned in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = if owned {
            context
                .new_stream()
                .map_err(|error| error.0 as u32)
                .unwrap()
        } else {
            context.default_stream()
        };
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let weak = Arc::downgrade(&lifetime);
        streams.set_stream_synchronize_hook(move || assert_admitted(&weak.upgrade().unwrap()));
        let weak = Arc::downgrade(&lifetime);
        streams.set_wait_hook(move || assert_admitted(&weak.upgrade().unwrap()));

        assert_eq!(stream.synchronize().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(stream.wait(&event).map_err(|error| error.0 as u32), Ok(()));

        assert_eq!(
            (
                streams.snapshot().stream_synchronizes,
                streams.snapshot().waits
            ),
            (1, 1)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        drop(event);
        assert_eq!(events.snapshot().destroyed, 1);
        drop(stream);
        assert_eq!(streams.snapshot().destroyed, usize::from(owned));
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn lost_same_lifetime_stream_use_refuses_all_native_dispatch() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = context
        .new_stream()
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();

    assert_eq!(
        stream.synchronize().map_err(|error| error.0 as u32),
        Err(lost as u32)
    );
    assert_eq!(
        stream.wait(&event).map_err(|error| error.0 as u32),
        Err(lost as u32)
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
    assert_eq!(
        (
            streams.snapshot().stream_synchronizes,
            streams.snapshot().waits
        ),
        (0, 0)
    );
    drop(event);
    drop(stream);
    assert_eq!(
        (events.snapshot().destroyed, streams.snapshot().destroyed),
        (0, 0)
    );
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn stream_native_terminal_result_is_shared_before_later_use() {
    for operation in [Operation::Synchronize, Operation::Wait] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = context.default_stream();
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        set_error(&streams, operation, Some(lost));
        let first = run(operation, &stream, &event).map_err(|error| error.0 as u32);
        let record = lifetime
            .state
            .lock()
            .unwrap()
            .loss
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.refusal().driver_status() as u32));
        let failed = driver.snapshot();
        let failed_calls = streams.snapshot();
        set_error(&streams, operation, None);
        let repeated = run(operation, &stream, &event).map_err(|error| error.0 as u32);
        let sibling_read = sibling.check_err().map_err(|error| error.0 as u32);
        let later_bind = sibling.bind_to_thread().map_err(|error| error.0 as u32);
        let after = driver.snapshot();
        assert_eq!(first, Err(lost as u32));
        assert_eq!(record, Some((17, None, lost as u32)));
        assert_eq!(repeated, Err(lost as u32));
        assert_eq!(sibling_read, Err(lost as u32));
        assert_eq!(later_bind, Err(lost as u32));
        assert_eq!(
            (
                after.primary_queries,
                after.id_queries,
                after.queries,
                after.binds
            ),
            (
                failed.primary_queries,
                failed.id_queries,
                failed.queries,
                failed.binds
            )
        );
        assert_eq!(
            (
                streams.snapshot().stream_synchronizes,
                streams.snapshot().waits
            ),
            (failed_calls.stream_synchronizes, failed_calls.waits)
        );
        drop(event);
        assert_eq!(events.snapshot().destroyed, 0);
        drop(stream);
        drop(context);
        drop(sibling);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn stream_use_oom_returns_directly_and_remains_retryable() {
    for operation in [Operation::Synchronize, Operation::Wait] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = context.default_stream();
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        set_error(&streams, operation, Some(oom));
        assert_eq!(
            run(operation, &stream, &event).map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        set_error(&streams, operation, None);
        assert_eq!(
            run(operation, &stream, &event).map_err(|error| error.0 as u32),
            Ok(())
        );
        drop(event);
        assert_eq!(events.snapshot().destroyed, 1);
        drop(stream);
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn stream_use_unwind_returns_admission_without_poisoning_loss_state() {
    for operation in [Operation::Synchronize, Operation::Wait] {
        let driver = context_test_seams::Guard::new();
        let _events = event_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = context.default_stream();
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let marker = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&marker);
        let weak = Arc::downgrade(&lifetime);
        let hook = move || {
            assert_admitted(&weak.upgrade().unwrap());
            observed.store(true, Ordering::SeqCst);
            panic!("synthetic admitted stream-use unwind");
        };
        match operation {
            Operation::Synchronize => streams.set_stream_synchronize_hook(hook),
            Operation::Wait => streams.set_wait_hook(hook),
        }
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(operation, &stream, &event)
        }));
        assert!(failed.is_err());
        assert!(marker.load(Ordering::SeqCst));
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        streams.set_stream_synchronize_hook(|| {});
        streams.set_wait_hook(|| {});
        assert_eq!(
            run(operation, &stream, &event).map_err(|error| error.0 as u32),
            Ok(())
        );
        drop(event);
        drop(stream);
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn untracked_stream_use_preserves_direct_errors_without_primary_probes() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    let stream = context.default_stream();
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    for operation in [Operation::Synchronize, Operation::Wait] {
        let error = sys::CUresult::CUDA_ERROR_UNKNOWN;
        set_error(&streams, operation, Some(error));
        assert_eq!(
            run(operation, &stream, &event).map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        set_error(&streams, operation, None);
    }
    assert_eq!(
        (
            driver.snapshot().primary_queries,
            driver.snapshot().id_queries
        ),
        (0, 0)
    );
    drop(event);
    assert_eq!(events.snapshot().destroyed, 1);
    drop(stream);
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
