#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
use super::tracked_pool_tests::seams;
use crate::driver::{
    result::context_test_seams, sys, CudaMemPool, DriverError, MemPoolProps,
    PrimaryContextLifetime, PrimaryContextLossCause,
};
use std::sync::Arc;
fn props() -> MemPoolProps {
    MemPoolProps {
        max_size: 0,
        release_threshold: 128,
    }
}
fn constructor_result(setter: bool, error: sys::CUresult) {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    if setter {
        native.set_set_error(Some(error));
    } else {
        native.set_create_error(Some(error));
    }
    assert_eq!(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    native.set_set_error(None);
    native.set_create_error(None);
    assert_eq!(
        lifetime
            .loss()
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
        Some((17, None, PrimaryContextLossCause::DriverStatus(error)))
    );
    assert_eq!(
        (
            native.snapshot().create_attempts,
            native.snapshot().created,
            native.snapshot().sets,
            native.snapshot().destroy_attempts
        ),
        (1, usize::from(setter), usize::from(setter), 0)
    );
    let first = lifetime.loss();
    let before = driver.snapshot();
    assert_eq!(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!(native.snapshot().create_attempts, 1);
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    assert_eq!(lifetime.loss(), first);
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
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn native_pool_create_709_publishes_before_return_or_partial_cleanup() {
    constructor_result(false, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn native_pool_create_201_publishes_before_return_or_partial_cleanup() {
    constructor_result(false, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

#[test]
fn native_pool_threshold_709_publishes_before_return_or_partial_cleanup() {
    constructor_result(true, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn native_pool_threshold_201_publishes_before_return_or_partial_cleanup() {
    constructor_result(true, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

fn operation_result(trim: bool, error: sys::CUresult) {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let pool = CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
        .map_err(|e| e.0 as u32)
        .unwrap();
    let result = if trim {
        native.set_trim_error(Some(error));
        pool.trim_to(0)
    } else {
        native.set_get_error(Some(error));
        pool.reserved_current().map(drop)
    };
    native.set_trim_error(None);
    native.set_get_error(None);
    assert_eq!(result.map_err(|e| e.0 as u32), Err(error as u32));
    assert_eq!(
        lifetime.loss().map(|loss| loss.cause),
        Some(PrimaryContextLossCause::DriverStatus(error))
    );
    let first = lifetime.loss();
    let calls = (native.snapshot().gets, native.snapshot().trims);
    let before = driver.snapshot();
    assert_eq!(
        pool.used_current().map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!(pool.trim_to(0).map_err(|e| e.0 as u32), Err(error as u32));
    assert_eq!(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!((native.snapshot().gets, native.snapshot().trims), calls);
    assert_eq!(native.snapshot().create_attempts, 1);
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    assert_eq!(lifetime.loss(), first);
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
    drop(pool);
    assert_eq!(native.snapshot().destroy_attempts, 0);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn native_pool_query_709_publishes_shared_first_loss() {
    operation_result(false, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn native_pool_query_201_publishes_shared_first_loss() {
    operation_result(false, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

#[test]
fn native_pool_trim_709_publishes_shared_first_loss() {
    operation_result(true, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn native_pool_trim_201_publishes_shared_first_loss() {
    operation_result(true, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

#[test]
fn ordinary_pool_query_and_trim_oom_remain_retryable() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let pool = CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
        .map_err(|e| e.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    native.set_get_error(Some(oom));
    assert_eq!(
        pool.reserved_current().map_err(|e| e.0 as u32),
        Err(oom as u32)
    );
    assert!(lifetime.loss().is_none());
    native.set_get_error(None);
    assert_eq!(pool.reserved_current().map_err(|e| e.0 as u32), Ok(4096));
    native.set_trim_error(Some(oom));
    assert_eq!(pool.trim_to(0).map_err(|e| e.0 as u32), Err(oom as u32));
    assert!(lifetime.loss().is_none());
    native.set_trim_error(None);
    assert_eq!(pool.trim_to(0).map_err(|e| e.0 as u32), Ok(()));
    assert_eq!((native.snapshot().gets, native.snapshot().trims), (2, 2));
    drop(pool);
    assert_eq!(native.snapshot().destroyed, 1);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
#[test]
fn untracked_pool_native_results_keep_their_return_contract() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let pool = CudaMemPool::create(0, &props())
        .map_err(|e| e.0 as u32)
        .unwrap();
    for error in [
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
    ] {
        native.set_get_error(Some(error));
        assert_eq!(
            pool.reserved_current().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        native.set_get_error(None);
        assert_eq!(pool.reserved_current().map_err(|e| e.0 as u32), Ok(4096));
        native.set_trim_error(Some(error));
        assert_eq!(pool.trim_to(0).map_err(|e| e.0 as u32), Err(error as u32));
        native.set_trim_error(None);
        assert_eq!(pool.trim_to(0).map_err(|e| e.0 as u32), Ok(()));
    }
    drop(pool);
    assert_eq!(native.snapshot().destroyed, 1);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (0, 0)
    );
}
