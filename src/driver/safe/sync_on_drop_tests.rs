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
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime, SyncOnDrop,
};
use std::sync::Arc;

fn assert_admitted(lifetime: &PrimaryContextLifetime) {
    let active = {
        lifetime
            .state
            .try_lock()
            .expect("guard delegation must not retain the lifetime mutex")
            .active_leases
    };
    assert_eq!(active, 1);
}

#[test]
fn healthy_sync_on_drop_delegates_to_admitted_event_and_stream_calls() {
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
    let event = Some(
        context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap(),
    );
    let weak = Arc::downgrade(&lifetime);
    events.set_record_hook(move || assert_admitted(&weak.upgrade().unwrap()));
    let weak = Arc::downgrade(&lifetime);
    streams.set_stream_synchronize_hook(move || assert_admitted(&weak.upgrade().unwrap()));

    drop(SyncOnDrop::record_event(&event, &stream));
    drop(SyncOnDrop::sync_stream(&stream));

    assert_eq!(
        (
            events.snapshot().recorded,
            streams.snapshot().stream_synchronizes
        ),
        (1, 1)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.loss().is_none());
    drop(event);
    drop(stream);
    assert_eq!(
        (events.snapshot().destroyed, streams.snapshot().destroyed),
        (1, 1)
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn lost_sync_on_drop_refuses_native_work_even_during_owner_unwind() {
    for unwind in [false, true] {
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
        let event = Some(
            context
                .new_event(None)
                .map_err(|error| error.0 as u32)
                .unwrap(),
        );
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        lifetime.record_error(DriverError(lost));
        let before = driver.snapshot();
        let marker = std::cell::Cell::new(false);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _record = SyncOnDrop::record_event(&event, &stream);
            let _sync = SyncOnDrop::sync_stream(&stream);
            if unwind {
                marker.set(true);
                panic!("synthetic owner unwind with pending synchronization guards");
            }
        }));

        assert_eq!(result.is_err(), unwind);
        assert_eq!(marker.get(), unwind);
        assert_eq!(
            (
                events.snapshot().recorded,
                streams.snapshot().stream_synchronizes
            ),
            (0, 0)
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
            lifetime.loss().unwrap().refusal().driver_status() as u32,
            lost as u32
        );
        drop(event);
        drop(stream);
        assert_eq!(
            (events.snapshot().destroyed, streams.snapshot().destroyed),
            (0, 0)
        );
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn sync_on_drop_records_ordinary_oom_without_panicking_or_marking_loss() {
    for record in [false, true] {
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
        let event = Some(
            context
                .new_event(None)
                .map_err(|error| error.0 as u32)
                .unwrap(),
        );
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        if record {
            events.set_record_error(Some(oom));
        } else {
            streams.set_stream_synchronize_error(Some(oom));
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if record {
                drop(SyncOnDrop::record_event(&event, &stream));
            } else {
                drop(SyncOnDrop::sync_stream(&stream));
            }
        }));

        assert!(result.is_ok());
        assert_eq!(
            (
                events.snapshot().recorded,
                streams.snapshot().stream_synchronizes
            ),
            (usize::from(record), usize::from(!record))
        );
        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert!(lifetime.loss().is_none());
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(event);
        drop(stream);
        assert_eq!(
            (events.snapshot().destroyed, streams.snapshot().destroyed),
            (1, 1)
        );
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn sync_on_drop_association_error_is_transient_and_preserves_both_lifetimes() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let event_lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_context = CudaContext::new_tracked(Arc::clone(&event_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = CudaContext::new_tracked(Arc::clone(&stream_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event = Some(
        event_context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap(),
    );
    let stream = stream_context
        .new_stream()
        .map_err(|error| error.0 as u32)
        .unwrap();
    let before = driver.snapshot();

    drop(SyncOnDrop::record_event(&event, &stream));

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
    assert_eq!(events.snapshot().recorded, 0);
    assert!(event_lifetime.loss().is_none());
    assert!(stream_lifetime.loss().is_none());
    assert_eq!(
        event_context.check_err().map_err(|error| error.0 as u32),
        Ok(())
    );
    assert_eq!(
        stream_context.check_err().map_err(|error| error.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
    );
    assert_eq!(
        stream_context.check_err().map_err(|error| error.0 as u32),
        Ok(())
    );
    drop(event);
    drop(stream);
    assert_eq!(
        (events.snapshot().destroyed, streams.snapshot().destroyed),
        (1, 1)
    );
    drop(event_context);
    drop(stream_context);
    drop(event_lifetime);
    drop(stream_lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}
