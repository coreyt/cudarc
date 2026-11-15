#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::{
    lifetime_event_cleanup_tests::seams as event_seams,
    lifetime_slice_cleanup_tests::{seams as slice_seams, tracked_context},
    lifetime_stream_cleanup_tests::seams as stream_seams,
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

fn assert_pending_oom_is_preserved(context: &CudaContext, lifetime: &PrimaryContextLifetime) {
    assert!(lifetime.loss().is_none());
    assert_eq!(
        context.check_err().map_err(|error| error.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
    );
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
}

#[test]
fn pending_ordinary_oom_does_not_abandon_completed_event() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
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
    context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));

    drop(event);

    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 1),
        "pending ordinary OOM must not abandon a healthy completed event"
    );
    assert_pending_oom_is_preserved(&context, &lifetime);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn pending_ordinary_oom_does_not_abandon_completed_stream() {
    let driver = context_test_seams::Guard::new();
    let streams = stream_seams::Guard::new();
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
    context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));

    drop(stream);

    assert_eq!(
        (streams.snapshot().created, streams.snapshot().destroyed),
        (1, 1),
        "pending ordinary OOM must not abandon a healthy completed stream"
    );
    assert_eq!(
        context
            .num_streams
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_pending_oom_is_preserved(&context, &lifetime);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

fn check_slice(asynchronous: bool, multistream: bool) {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = slice_seams::Guard::new();
    let streams = stream_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, asynchronous);
    let stream = if multistream {
        context
            .new_stream()
            .map_err(|error| error.0 as u32)
            .unwrap()
    } else {
        context.default_stream()
    };
    memory.set_expected_stream(stream.cu_stream);
    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));

    drop(allocation);

    let counts = memory.snapshot();
    assert_eq!(
        counts.async_frees + counts.sync_frees,
        1,
        "pending ordinary OOM must not abandon a healthy completed allocation"
    );
    assert_eq!(counts.stream_synchronizes, usize::from(!asynchronous));
    assert_eq!(streams.snapshot().waits, if multistream { 2 } else { 0 });
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (2, 2)
    );
    assert_pending_oom_is_preserved(&context, &lifetime);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn pending_ordinary_oom_does_not_abandon_completed_async_slice() {
    check_slice(true, false);
}
#[test]
fn pending_ordinary_oom_does_not_abandon_completed_sync_slice() {
    check_slice(false, false);
}

#[test]
fn pending_ordinary_oom_does_not_abandon_multistream_slice_or_its_waits() {
    check_slice(true, true);
}
