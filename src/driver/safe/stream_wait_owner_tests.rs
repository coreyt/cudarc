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
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

fn active(lifetime: &PrimaryContextLifetime) -> usize {
    let state = lifetime
        .state
        .try_lock()
        .expect("native wait must run outside both lifetime mutexes");
    assert!(!state.probing);
    state.active_leases
}

#[test]
fn lost_event_owner_refuses_wait_before_probing_the_healthy_stream_owner() {
    for stream_lost in [false, true] {
        let driver = context_test_seams::Guard::new();
        driver.enable_second_device();
        let events = event_seams::Guard::new();
        let streams = seams::Guard::new();
        let stream_lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let event_lifetime = PrimaryContextLifetime::new(1)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream_context = CudaContext::new_tracked(Arc::clone(&stream_lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let event_context = CudaContext::new_tracked(Arc::clone(&event_lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        assert_ne!(stream_context.cu_ctx, event_context.cu_ctx);
        let stream = stream_context.default_stream();
        let event = event_context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if stream_lost {
            stream_lifetime.record_error(DriverError(lost));
        } else {
            event_lifetime.record_error(DriverError(lost));
        }
        let before = driver.snapshot();

        let result = stream.wait(&event).map_err(|error| error.0 as u32);

        assert_eq!(result, Err(lost as u32));
        assert_eq!(streams.snapshot().waits, 0);
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
            stream_lifetime.state.lock().unwrap().loss.is_some(),
            stream_lost
        );
        assert_eq!(
            event_lifetime.state.lock().unwrap().loss.is_some(),
            !stream_lost
        );
        assert_eq!((active(&stream_lifetime), active(&event_lifetime)), (0, 0));
        drop(event);
        assert_eq!(events.snapshot().destroyed, usize::from(stream_lost));
        drop(stream);
        drop(stream_context);
        drop(event_context);
        drop(stream_lifetime);
        drop(event_lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (2, 2)
        );
    }
}

#[test]
fn healthy_cross_context_wait_holds_both_distinct_lifetimes() {
    let driver = context_test_seams::Guard::new();
    driver.enable_second_device();
    let events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let stream_lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_lifetime = PrimaryContextLifetime::new(1)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = CudaContext::new_tracked(Arc::clone(&stream_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_context = CudaContext::new_tracked(Arc::clone(&event_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_ne!(stream_context.cu_ctx, event_context.cu_ctx);
    let stream = stream_context.default_stream();
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak_stream = Arc::downgrade(&stream_lifetime);
    let weak_event = Arc::downgrade(&event_lifetime);
    streams.set_wait_hook(move || {
        assert_eq!(active(&weak_stream.upgrade().unwrap()), 1);
        assert_eq!(active(&weak_event.upgrade().unwrap()), 1);
    });

    assert_eq!(stream.wait(&event).map_err(|error| error.0 as u32), Ok(()));

    assert_eq!(streams.snapshot().waits, 1);
    assert_eq!((active(&stream_lifetime), active(&event_lifetime)), (0, 0));
    assert!(stream_lifetime.state.lock().unwrap().loss.is_none());
    assert!(event_lifetime.state.lock().unwrap().loss.is_none());
    drop(event);
    assert_eq!(events.snapshot().destroyed, 1);
    drop(stream);
    drop(stream_context);
    drop(event_context);
    drop(stream_lifetime);
    drop(event_lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}

#[test]
fn mixed_tracked_and_untracked_wait_owners_refuse_before_dispatch() {
    for tracked_stream in [false, true] {
        let driver = context_test_seams::Guard::new();
        driver.enable_second_device();
        let _events = event_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let tracked = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let untracked = CudaContext::new(1).map_err(|error| error.0 as u32).unwrap();
        let (stream_context, event_context) = if tracked_stream {
            (&tracked, &untracked)
        } else {
            (&untracked, &tracked)
        };
        let stream = stream_context.default_stream();
        let event = event_context
            .new_event(None)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let before = driver.snapshot();

        assert_eq!(
            stream.wait(&event).map_err(|error| error.0 as u32),
            Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
        );

        let after = driver.snapshot();
        assert_eq!(streams.snapshot().waits, 0);
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
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        drop(event);
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
fn same_lifetime_wait_deduplicates_admission_across_distinct_wrappers() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert!(!Arc::ptr_eq(&stream_context, &event_context));
    let stream = stream_context.default_stream();
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    streams.set_wait_hook(move || assert_eq!(active(&weak.upgrade().unwrap()), 1));
    let before = driver.snapshot();

    assert_eq!(stream.wait(&event).map_err(|error| error.0 as u32), Ok(()));

    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries + 1, before.id_queries + 1)
    );
    assert_eq!(streams.snapshot().waits, 1);
    drop(event);
    drop(stream);
    drop(stream_context);
    drop(event_context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn wholly_untracked_cross_context_wait_keeps_legacy_behavior() {
    let driver = context_test_seams::Guard::new();
    driver.enable_second_device();
    let _events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let stream_context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    let event_context = CudaContext::new(1).map_err(|error| error.0 as u32).unwrap();
    assert_ne!(stream_context.cu_ctx, event_context.cu_ctx);
    let stream = stream_context.default_stream();
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();

    assert_eq!(stream.wait(&event).map_err(|error| error.0 as u32), Ok(()));

    assert_eq!(streams.snapshot().waits, 1);
    assert_eq!(
        (
            driver.snapshot().primary_queries,
            driver.snapshot().id_queries
        ),
        (0, 0)
    );
    drop(event);
    drop(stream);
    drop(stream_context);
    drop(event_context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}

#[test]
fn terminal_native_wait_result_quarantines_both_admitted_owners() {
    let driver = context_test_seams::Guard::new();
    driver.enable_second_device();
    let events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let stream_lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_lifetime = PrimaryContextLifetime::new(1)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = CudaContext::new_tracked(Arc::clone(&stream_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_context = CudaContext::new_tracked(Arc::clone(&event_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = stream_context.default_stream();
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    streams.set_wait_error(Some(lost));

    assert_eq!(
        stream.wait(&event).map_err(|error| error.0 as u32),
        Err(lost as u32)
    );

    assert_eq!(
        stream_lifetime.state.lock().unwrap().loss.unwrap().error as u32,
        lost as u32
    );
    assert_eq!(
        event_lifetime.state.lock().unwrap().loss.unwrap().error as u32,
        lost as u32
    );
    let before = driver.snapshot();
    streams.set_wait_error(None);
    assert_eq!(
        event.try_is_complete().map_err(|error| error.0 as u32),
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
        (events.snapshot().queried, streams.snapshot().waits),
        (0, 1)
    );
    drop(event);
    assert_eq!(events.snapshot().destroyed, 0);
    drop(stream);
    drop(stream_context);
    drop(event_context);
    drop(stream_lifetime);
    drop(event_lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}

#[test]
fn cross_context_wait_oom_releases_both_owners_without_storing_errors() {
    let driver = context_test_seams::Guard::new();
    driver.enable_second_device();
    let _events = event_seams::Guard::new();
    let streams = seams::Guard::new();
    let stream_lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_lifetime = PrimaryContextLifetime::new(1)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = CudaContext::new_tracked(Arc::clone(&stream_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let event_context = CudaContext::new_tracked(Arc::clone(&event_lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = stream_context.default_stream();
    let event = event_context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    streams.set_wait_error(Some(oom));
    assert_eq!(
        stream.wait(&event).map_err(|error| error.0 as u32),
        Err(oom as u32)
    );
    assert_eq!((active(&stream_lifetime), active(&event_lifetime)), (0, 0));
    assert!(stream_lifetime.state.lock().unwrap().loss.is_none());
    assert!(event_lifetime.state.lock().unwrap().loss.is_none());
    assert_eq!(
        stream_context.check_err().map_err(|error| error.0 as u32),
        Ok(())
    );
    assert_eq!(
        event_context.check_err().map_err(|error| error.0 as u32),
        Ok(())
    );
    streams.set_wait_error(None);
    assert_eq!(stream.wait(&event).map_err(|error| error.0 as u32), Ok(()));
    drop(event);
    drop(stream);
    drop(stream_context);
    drop(event_context);
    drop(stream_lifetime);
    drop(event_lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}
