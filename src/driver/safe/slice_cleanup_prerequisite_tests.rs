#![cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]

use super::{lifetime_event_cleanup_tests::seams as event_seams, lifetime_slice_cleanup_tests::{seams as slice_seams, tracked_context}, lifetime_stream_cleanup_tests::seams as stream_seams};
use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime};
use std::sync::Arc;

fn check_cleanup_failure(wait: bool) {
    for error in [sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT, sys::CUresult::CUDA_ERROR_UNKNOWN] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = slice_seams::Guard::new();
        let streams = stream_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = tracked_context(&lifetime, wait);
        let sibling = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
        let stream = if wait { context.new_stream().map_err(|error| error.0 as u32).unwrap() } else { context.default_stream() };
        memory.set_expected_stream(stream.cu_stream);
        let allocation = unsafe { stream.alloc::<u8>(16) }.map_err(|error| error.0 as u32).unwrap();
        if wait { streams.set_wait_error(Some(error)); } else { memory.set_synchronize_error(Some(error)); }

        drop(allocation);

        assert_eq!(lifetime.loss().map(|loss| (loss.recorded_id, loss.observed_id, loss.error as u32)), Some((17, None, error as u32)), "actual native cleanup prerequisite failure must publish shared loss before abandoning memory");
        assert_eq!(memory.snapshot().async_frees + memory.snapshot().sync_frees, 0);
        assert_eq!(memory.snapshot().stream_synchronizes, usize::from(!wait));
        assert_eq!(streams.snapshot().waits, usize::from(wait));
        assert_eq!(events.snapshot().destroyed, 0);
        let stopped = driver.snapshot();
        streams.set_wait_error(None);
        memory.set_synchronize_error(None);
        for _ in 0..2 {
            assert_eq!(context.check_err().map_err(|error| error.0 as u32), Err(error as u32));
            assert_eq!(sibling.check_err().map_err(|error| error.0 as u32), Err(error as u32));
        }
        assert_eq!(sibling.bind_to_thread().map_err(|error| error.0 as u32), Err(error as u32));
        assert_eq!(unsafe { stream.alloc::<u8>(16) }.map(drop).map_err(|error| error.0 as u32), Err(error as u32));
        sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        assert_eq!(sibling.check_err().map_err(|error| error.0 as u32), Err(error as u32));
        assert_eq!(lifetime.loss().unwrap().error as u32, error as u32);
        let after = driver.snapshot();
        assert_eq!((after.primary_queries, after.id_queries, after.queries, after.binds), (stopped.primary_queries, stopped.id_queries, stopped.queries, stopped.binds));
        assert_eq!(memory.snapshot().allocation_attempts, 1);
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(stream);
        assert_eq!(streams.snapshot().destroyed, 0);
        drop(context);
        drop(sibling);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}

#[test]
fn native_cleanup_wait_failure_publishes_loss_before_abandonment() { check_cleanup_failure(true); }
#[test]
fn native_cleanup_sync_failure_publishes_loss_before_abandonment() { check_cleanup_failure(false); }

#[test]
fn ordinary_operation_wait_oom_remains_retryable_without_loss() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = slice_seams::Guard::new();
    let streams = stream_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime, true);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    memory.set_expected_stream(stream.cu_stream);
    let allocation = unsafe { stream.alloc::<u8>(16) }.map_err(|error| error.0 as u32).unwrap();
    let event = allocation.write.as_ref().unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    streams.set_wait_error(Some(oom));
    assert_eq!(stream.wait(event).map_err(|error| error.0 as u32), Err(oom as u32));
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    streams.set_wait_error(None);
    assert_eq!(stream.wait(event).map_err(|error| error.0 as u32), Ok(()));
    drop(allocation);
    assert_eq!(streams.snapshot().waits, 4);
    assert_eq!(memory.snapshot().async_frees, 1);
    assert_eq!(events.snapshot().destroyed, 2);
    drop(stream);
    assert_eq!(streams.snapshot().destroyed, 1);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn ordinary_operation_sync_oom_remains_retryable_without_loss() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = slice_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let allocation = unsafe { stream.alloc::<u8>(16) }.map_err(|error| error.0 as u32).unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    memory.set_synchronize_error(Some(oom));
    assert_eq!(stream.synchronize().map_err(|error| error.0 as u32), Err(oom as u32));
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    memory.set_synchronize_error(None);
    assert_eq!(stream.synchronize().map_err(|error| error.0 as u32), Ok(()));
    drop(allocation);
    assert_eq!(memory.snapshot().stream_synchronizes, 3);
    assert_eq!(memory.snapshot().sync_frees, 1);
    assert_eq!(events.snapshot().destroyed, 2);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}
