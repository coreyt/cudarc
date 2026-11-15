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
    slice_memory_use_tests::{seams, Operation},
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
    PrimaryContextLossCause, PrimaryContextRefusal,
};
use std::sync::Arc;

fn check_terminal_native_result(operation: Operation, error: sys::CUresult) {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let use_calls = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    use_calls.set_error(Some(error));

    assert_eq!(
        operation
            .run(&stream, &mut allocation)
            .map_err(|error| error.0 as u32),
        Err(error as u32)
    );

    assert_eq!(
        lifetime
            .loss()
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
        Some((17, None, PrimaryContextLossCause::DriverStatus(error))),
        "terminal native memory result must publish its actual status before return"
    );
    let first = lifetime.loss();
    assert_eq!(
        first.unwrap().refusal(),
        PrimaryContextRefusal::DriverStatus(error)
    );
    let stopped = driver.snapshot();
    let dispatched = use_calls.calls();
    use_calls.set_error(None);
    for _ in 0..2 {
        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            sibling.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
    }
    for next in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        assert_eq!(
            next.run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Err(error as u32)
        );
    }
    assert_eq!(
        sibling.bind_to_thread().map_err(|error| error.0 as u32),
        Err(error as u32)
    );
    assert_eq!(
        unsafe { stream.alloc::<u8>(16) }
            .map(drop)
            .map_err(|error| error.0 as u32),
        Err(error as u32)
    );
    sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
    assert_eq!(
        sibling.check_err().map_err(|error| error.0 as u32),
        Err(error as u32)
    );
    assert_eq!(lifetime.loss(), first);
    assert_eq!(use_calls.calls(), dispatched);
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
    assert_eq!(memory.snapshot().allocation_attempts, 1);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 0);
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

#[test]
fn native_memset_709_publishes_first_loss_before_return() {
    check_terminal_native_result(
        Operation::Memset,
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    );
}
#[test]
fn native_memset_201_publishes_first_loss_before_return() {
    check_terminal_native_result(Operation::Memset, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}
#[test]
fn native_htod_709_publishes_first_loss_before_return() {
    check_terminal_native_result(
        Operation::Htod,
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    );
}
#[test]
fn native_htod_201_publishes_first_loss_before_return() {
    check_terminal_native_result(Operation::Htod, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}
#[test]
fn native_dtoh_709_publishes_first_loss_before_return() {
    check_terminal_native_result(
        Operation::Dtoh,
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    );
}
#[test]
fn native_dtoh_201_publishes_first_loss_before_return() {
    check_terminal_native_result(Operation::Dtoh, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}
#[test]
fn native_dtod_709_publishes_first_loss_before_return() {
    check_terminal_native_result(
        Operation::Dtod,
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    );
}
#[test]
fn native_dtod_201_publishes_first_loss_before_return() {
    check_terminal_native_result(Operation::Dtod, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

#[test]
fn native_memory_operation_oom_is_retryable_without_delayed_error_or_loss() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
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
    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        use_calls.set_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
        );
        assert!(lifetime.loss().is_none());
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        use_calls.set_error(None);
        assert_eq!(
            operation
                .run(&stream, &mut allocation)
                .map_err(|error| error.0 as u32),
            Ok(())
        );
    }
    assert_eq!(use_calls.calls(), [2; 4]);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn untracked_memory_errors_keep_their_native_return_contract() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let use_calls = seams::Guard::new();
    let mut context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    Arc::get_mut(&mut context).unwrap().has_async_alloc = false;
    let stream = context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    for operation in [
        Operation::Memset,
        Operation::Htod,
        Operation::Dtoh,
        Operation::Dtod,
    ] {
        for error in [
            sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
        ] {
            use_calls.set_error(Some(error));
            assert_eq!(
                operation
                    .run(&stream, &mut allocation)
                    .map_err(|error| error.0 as u32),
                Err(error as u32)
            );
            assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        }
    }
    assert_eq!(use_calls.calls(), [2; 4]);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(stream);
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
