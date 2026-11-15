#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::lifetime_event_cleanup_tests::seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

#[test]
fn equal_native_handles_with_distinct_lifetimes_refuse_record_before_dispatch() {
    for stream_lost in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = seams::Guard::new();
        let event_lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream_lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        assert!(!Arc::ptr_eq(&event_lifetime, &stream_lifetime));
        let event_context = CudaContext::new_tracked(Arc::clone(&event_lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream_context = CudaContext::new_tracked(Arc::clone(&stream_lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        assert!(!Arc::ptr_eq(&event_context, &stream_context));
        assert!(
            event_context == stream_context,
            "the native-handle equality shortcut must be exercised"
        );
        let event = event_context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = stream_context.default_stream();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if stream_lost {
            stream_lifetime.record_error(DriverError(lost));
        }
        let before = driver.snapshot();

        let result = event.record(&stream).map_err(|error| error.0 as u32);

        assert_eq!(
            result,
            Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
        );
        assert_eq!(events.snapshot().recorded, 0);
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
        assert!(event_lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(
            stream_lifetime
                .state
                .lock()
                .unwrap()
                .loss
                .map(|loss| loss.refusal().driver_status() as u32),
            stream_lost.then_some(lost as u32)
        );
        assert_eq!(event_lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(stream_lifetime.state.lock().unwrap().active_leases, 0);
        drop(event);
        assert_eq!(
            events.snapshot().destroyed,
            1,
            "association refusal must leave the healthy event owner usable for cleanup"
        );
        drop(stream);
        drop(event_context);
        drop(stream_context);
        drop(event_lifetime);
        drop(stream_lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (2, 2)
        );
    }
}

#[test]
fn missing_lifetime_on_either_side_refuses_record_before_dispatch() {
    for tracked_event in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let tracked = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let untracked = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
        assert!(tracked == untracked);
        let (event_context, stream_context) = if tracked_event {
            (&tracked, &untracked)
        } else {
            (&untracked, &tracked)
        };
        assert_ne!(
            event_context.primary_lifetime.is_some(),
            stream_context.primary_lifetime.is_some()
        );
        let event = event_context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = stream_context.default_stream();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if !tracked_event {
            lifetime.record_error(DriverError(lost));
        }
        let before = driver.snapshot();

        let result = event.record(&stream).map_err(|error| error.0 as u32);

        assert_eq!(
            result,
            Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
        );
        assert_eq!(events.snapshot().recorded, 0);
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
            lifetime
                .state
                .lock()
                .unwrap()
                .loss
                .map(|loss| loss.refusal().driver_status() as u32),
            (!tracked_event).then_some(lost as u32)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(event);
        assert_eq!(events.snapshot().destroyed, 1);
        drop(stream);
        drop(tracked);
        drop(untracked);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (2, 2)
        );
    }
}

#[test]
fn distinct_context_wrappers_sharing_one_lifetime_can_record() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert!(!Arc::ptr_eq(&event_context, &stream_context));
    assert!(Arc::ptr_eq(
        event_context.primary_lifetime.as_ref().unwrap(),
        stream_context.primary_lifetime.as_ref().unwrap()
    ));
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = stream_context.default_stream();
    let weak = Arc::downgrade(&lifetime);
    events.set_record_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let active = {
            lifetime
                .state
                .try_lock()
                .expect("native record must run outside the state mutex")
                .active_leases
        };
        assert_eq!(active, 1);
    });

    assert_eq!(
        event.record(&stream).map_err(|error| error.0 as u32),
        Ok(())
    );

    assert_eq!(events.snapshot().recorded, 1);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    drop(event);
    drop(stream);
    drop(event_context);
    drop(stream_context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn wholly_untracked_equal_handle_wrappers_preserve_record_behavior() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let event_context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    let stream_context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    assert!(!Arc::ptr_eq(&event_context, &stream_context));
    assert!(event_context == stream_context);
    assert!(event_context.primary_lifetime.is_none());
    assert!(stream_context.primary_lifetime.is_none());
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = stream_context.default_stream();

    assert_eq!(
        event.record(&stream).map_err(|error| error.0 as u32),
        Ok(())
    );

    assert_eq!(events.snapshot().recorded, 1);
    assert_eq!(
        (
            driver.snapshot().primary_queries,
            driver.snapshot().id_queries
        ),
        (0, 0)
    );
    drop(event);
    drop(stream);
    drop(event_context);
    drop(stream_context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}
