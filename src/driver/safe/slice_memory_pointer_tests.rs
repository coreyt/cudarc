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
    lifetime_stream_cleanup_tests::seams as stream_seams,
    slice_memory_use_tests::{seams, Operation},
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, PrimaryContextLifetime, PrimaryContextLossCause,
};
use std::sync::Arc;

fn check_wait_failure(operation: Operation, terminal: bool) {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let streams = stream_seams::Guard::new();
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
    let error = if terminal {
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    } else {
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
    };
    let weak = Arc::downgrade(&lifetime);
    streams.set_wait_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime
            .state
            .try_lock()
            .expect("pointer preparation native wait must be outside lifetime mutex");
        assert_eq!(state.active_leases, 2);
    });
    streams.set_wait_error(Some(error));

    let outcome = operation
        .run(&stream, &mut allocation)
        .map_err(|error| error.0 as u32);
    streams.set_wait_error(None);
    streams.set_wait_hook(|| {});

    assert_eq!(
        outcome,
        Err(error as u32),
        "a pointer-preparation error must stop the following memory dispatch"
    );
    assert_eq!(use_calls.calls(), [0; 4]);
    assert_eq!(streams.snapshot().waits, 1);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    if terminal {
        assert_eq!(
            lifetime
                .loss()
                .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
            Some((17, None, PrimaryContextLossCause::DriverStatus(error)))
        );
        let stopped = driver.snapshot();
        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(use_calls.calls(), [0; 4]);
        let after = driver.snapshot();
        assert_eq!(
            (
                after.primary_queries,
                after.id_queries,
                after.queries,
                after.binds
            ),
            (
                stopped.primary_queries,
                stopped.id_queries,
                stopped.queries,
                stopped.binds
            )
        );
    } else {
        assert!(lifetime.loss().is_none());
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Ok(())
        );
        let mut expected = [0; 4];
        expected[operation as usize] = 1;
        assert_eq!(use_calls.calls(), expected);
    }
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, usize::from(!terminal));
    assert_eq!(events.snapshot().destroyed, if terminal { 0 } else { 2 });
    drop(background);
    assert_eq!(streams.snapshot().destroyed, usize::from(!terminal));
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn memset_stops_after_native_pointer_wait_709() {
    check_wait_failure(Operation::Memset, true);
}
#[test]
fn memset_stops_after_native_pointer_wait_oom() {
    check_wait_failure(Operation::Memset, false);
}
#[test]
fn htod_stops_after_native_pointer_wait_709() {
    check_wait_failure(Operation::Htod, true);
}
#[test]
fn htod_stops_after_native_pointer_wait_oom() {
    check_wait_failure(Operation::Htod, false);
}
#[test]
fn dtoh_stops_after_native_pointer_wait_709() {
    check_wait_failure(Operation::Dtoh, true);
}
#[test]
fn dtoh_stops_after_native_pointer_wait_oom() {
    check_wait_failure(Operation::Dtoh, false);
}
#[test]
fn dtod_stops_after_native_pointer_wait_709() {
    check_wait_failure(Operation::Dtod, true);
}
#[test]
fn dtod_stops_after_native_pointer_wait_oom() {
    check_wait_failure(Operation::Dtod, false);
}

#[test]
fn pointer_association_201_stops_dispatch_without_marking_either_owner_lost() {
    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        let driver = context_test_seams::Guard::new();
        let _events = event_seams::Guard::new();
        let memory = allocation_seams::Guard::new();
        let streams = stream_seams::Guard::new();
        let use_calls = seams::Guard::new();
        let mut source_context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
        Arc::get_mut(&mut source_context).unwrap().has_async_alloc = false;
        let source_stream = source_context.default_stream();
        let background = source_context
            .new_stream()
            .map_err(|error| error.0 as u32)
            .unwrap();
        let mut allocation = unsafe { source_stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let target_context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let target_stream = target_context.default_stream();

        let outcome = operation
            .run(&target_stream, &mut allocation)
            .map_err(|error| error.0 as u32);

        assert_eq!(
            outcome,
            Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
        );
        assert_eq!(use_calls.calls(), [0; 4]);
        assert_eq!(streams.snapshot().waits, 0);
        assert!(lifetime.loss().is_none());
        assert_eq!(
            target_context.check_err().map_err(|error| error.0 as u32),
            Ok(())
        );
        assert_eq!(
            source_context.check_err().map_err(|error| error.0 as u32),
            Ok(())
        );
        drop(allocation);
        assert_eq!(memory.snapshot().sync_frees, 1);
        drop(background);
        drop(source_stream);
        drop(source_context);
        drop(target_stream);
        drop(target_context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (2, 2)
        );
    }
}
