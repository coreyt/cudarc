#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::lifetime_event_cleanup_tests::seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, CudaEvent, CudaStream, DriverError,
    PrimaryContextLifetime,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Query,
    LegacyQuery,
    Record,
    Synchronize,
}

fn run(operation: Operation, event: &CudaEvent, stream: &CudaStream) -> Result<bool, DriverError> {
    match operation {
        Operation::Query => event.try_is_complete(),
        Operation::LegacyQuery => Ok(event.is_complete()),
        Operation::Record => event.record(stream).map(|_| true),
        Operation::Synchronize => event.synchronize().map(|_| true),
    }
}

fn set_error(events: &seams::Guard, operation: Operation, error: Option<sys::CUresult>) {
    match operation {
        Operation::Query | Operation::LegacyQuery => events.set_query_error(error),
        Operation::Record => events.set_record_error(error),
        Operation::Synchronize => events.set_synchronize_error(error),
    }
}

fn assert_admitted(lifetime: &PrimaryContextLifetime) {
    let (active, probing) = {
        let state = lifetime
            .state
            .try_lock()
            .expect("native event use must run outside the lifetime mutex");
        (state.active_leases, state.probing)
    };
    assert_eq!(
        active, 1,
        "native event use must retain admission through dispatch"
    );
    assert!(!probing);
}

#[test]
fn lost_event_query_refuses_before_any_native_event_use() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = context.default_stream();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();

    let query = event.try_is_complete().map_err(|error| error.0 as u32);
    let legacy = event.is_complete();
    let record = event.record(&stream).map_err(|error| error.0 as u32);
    let sync = event.synchronize().map_err(|error| error.0 as u32);

    assert_eq!(query, Err(lost as u32));
    assert!(!legacy);
    assert_eq!(record, Err(lost as u32));
    assert_eq!(sync, Err(lost as u32));
    assert_eq!(
        (
            events.snapshot().queried,
            events.snapshot().recorded,
            events.snapshot().synchronized
        ),
        (0, 0, 0)
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
    drop(event);
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
fn healthy_event_query_record_and_sync_hold_admission_through_ffi() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = context.default_stream();
    let weak = Arc::downgrade(&lifetime);
    events.set_query_hook(move || assert_admitted(&weak.upgrade().unwrap()));
    let weak = Arc::downgrade(&lifetime);
    events.set_record_hook(move || assert_admitted(&weak.upgrade().unwrap()));
    let weak = Arc::downgrade(&lifetime);
    events.set_synchronize_hook(move || assert_admitted(&weak.upgrade().unwrap()));

    assert_eq!(
        event.try_is_complete().map_err(|error| error.0 as u32),
        Ok(true)
    );
    assert!(event.is_complete());
    assert_eq!(
        event.record(&stream).map_err(|error| error.0 as u32),
        Ok(())
    );
    assert_eq!(event.synchronize().map_err(|error| error.0 as u32), Ok(()));

    assert_eq!(
        (
            events.snapshot().queried,
            events.snapshot().recorded,
            events.snapshot().synchronized
        ),
        (2, 1, 1)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
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

#[test]
fn event_not_ready_and_other_query_errors_keep_the_typed_contract() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    for error in [
        sys::CUresult::CUDA_ERROR_NOT_READY,
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        sys::CUresult::CUDA_ERROR_UNKNOWN,
    ] {
        events.set_query_error(Some(error));
        let expected = if error == sys::CUresult::CUDA_ERROR_NOT_READY {
            Ok(false)
        } else {
            Err(error as u32)
        };
        assert_eq!(
            event.try_is_complete().map_err(|error| error.0 as u32),
            expected
        );
        assert!(!event.is_complete());
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    }
    events.set_query_error(None);
    assert_eq!(
        event.try_is_complete().map_err(|error| error.0 as u32),
        Ok(true)
    );
    drop(event);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn event_native_terminal_result_is_shared_before_later_event_use() {
    for operation in [
        Operation::Query,
        Operation::LegacyQuery,
        Operation::Record,
        Operation::Synchronize,
    ] {
        let driver = context_test_seams::Guard::new();
        let events = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = context.default_stream();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        set_error(&events, operation, Some(lost));
        let first = run(operation, &event, &stream).map_err(|error| error.0 as u32);
        let first_record = lifetime.state.lock().unwrap().loss.map(|loss| {
            (
                loss.recorded_id,
                loss.observed_id,
                loss.refusal().driver_status() as u32,
            )
        });
        let failed = driver.snapshot();
        let failed_events = events.snapshot();
        set_error(&events, operation, None);
        let repeated = event.try_is_complete().map_err(|error| error.0 as u32);
        let sibling_read = sibling.check_err().map_err(|error| error.0 as u32);
        let later_bind = sibling.bind_to_thread().map_err(|error| error.0 as u32);
        let after = driver.snapshot();
        let expected = if matches!(operation, Operation::LegacyQuery) {
            Ok(false)
        } else {
            Err(lost as u32)
        };
        assert_eq!(first, expected);
        assert_eq!(
            first_record,
            Some((17, None, lost as u32)),
            "native {operation:?} must publish first loss before handing its result back"
        );
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
                events.snapshot().queried,
                events.snapshot().recorded,
                events.snapshot().synchronized
            ),
            (
                failed_events.queried,
                failed_events.recorded,
                failed_events.synchronized
            )
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
fn event_use_oom_returns_directly_without_terminal_or_delayed_error() {
    for operation in [Operation::Query, Operation::Record, Operation::Synchronize] {
        let driver = context_test_seams::Guard::new();
        let events = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let event = context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = context.default_stream();
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        set_error(&events, operation, Some(oom));
        assert_eq!(
            run(operation, &event, &stream).map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        set_error(&events, operation, None);
        assert_eq!(
            run(operation, &event, &stream).map_err(|error| error.0 as u32),
            Ok(true)
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
fn untracked_event_query_preserves_native_and_delayed_error_behavior() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    context.record_err::<()>(Err(DriverError(oom)));
    let before = driver.snapshot();

    assert_eq!(
        event.try_is_complete().map_err(|error| error.0 as u32),
        Ok(true)
    );
    assert!(event.is_complete());

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
    assert_eq!(events.snapshot().queried, 2);
    assert_eq!(
        context.check_err().map_err(|error| error.0 as u32),
        Err(oom as u32)
    );
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    drop(event);
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
