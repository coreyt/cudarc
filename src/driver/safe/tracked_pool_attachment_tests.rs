#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
use super::{
    lifetime_event_cleanup_tests::seams as event_seams,
    lifetime_slice_cleanup_tests::seams as memory_seams, tracked_pool_tests::seams,
};
use crate::driver::{
    result::context_test_seams, sys, AllocMode, CudaContext, CudaMemPool, MemPoolProps,
    PrimaryContextLifetime,
};
use std::sync::{Arc, Mutex};
fn props() -> MemPoolProps {
    MemPoolProps {
        max_size: 0,
        release_threshold: 128,
    }
}

#[test]
fn attachment_borrows_the_pool_token_and_keeps_private_allocator_ownership() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = memory_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    let pool = Arc::new(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    let entries = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&entries);
    let weak_entry = Arc::downgrade(&lifetime);
    driver.clear_current();
    driver.set_context_entry_hook(move || {
        let life = weak_entry.upgrade().unwrap();
        let state = life
            .state
            .try_lock()
            .expect("attachment binding must not hold the lifetime mutex");
        assert!(
            state.active_leases > 0,
            "attachment must keep admission through native binding"
        );
        log.lock().unwrap().push(state.active_leases);
    });
    let result = CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool));
    driver.set_context_entry_hook(|| {});
    assert!(
        result.is_ok(),
        "tracked pool attachment must construct a borrowed context wrapper"
    );
    let context = result.map_err(|e| e.0 as u32).unwrap();
    assert_eq!(entries.lock().unwrap().len(), 2);
    assert_eq!(context.alloc_mode(), AllocMode::Private);
    assert!(Arc::ptr_eq(context.mem_pool().unwrap(), &pool));
    assert_eq!(
        context
            .primary_lifetime()
            .map(|life| Arc::ptr_eq(life, &lifetime)),
        Some(true)
    );
    assert_eq!(driver.snapshot().retains, 1);
    let allocations = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&allocations);
    let weak_alloc = Arc::downgrade(&lifetime);
    native.set_hook(move |phase| {
        if phase == 5 {
            let life = weak_alloc.upgrade().unwrap();
            let state = life
                .state
                .try_lock()
                .expect("private allocation must not hold the lifetime mutex");
            log.lock().unwrap().push(state.active_leases);
        }
    });
    let stream = context.default_stream();
    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    assert_eq!(
        (
            native.snapshot().allocations,
            memory.snapshot().async_allocations,
            memory.snapshot().sync_allocations
        ),
        (1, 1, 0)
    );
    native.set_hook(|_| {});
    assert_eq!(*allocations.lock().unwrap(), vec![1]);
    drop(pool);
    drop(context);
    drop(lifetime);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        (
            native.snapshot().destroy_attempts,
            driver.snapshot().releases
        ),
        (0, 0)
    );
    drop(allocation);
    assert_eq!(
        (memory.snapshot().async_frees, events.snapshot().destroyed),
        (1, 2)
    );
    assert_eq!(native.snapshot().destroy_attempts, 0);
    drop(stream);
    assert_eq!(native.snapshot().destroyed, 1);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
    assert_eq!(driver.snapshot().default_pool_queries, 0);
}

#[test]
fn attachment_rejects_a_distinct_pool_generation_before_any_native_work() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let _native = seams::Guard::new();
    let first = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let second = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let pool = Arc::new(
        CudaMemPool::create_tracked(Arc::clone(&first), &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    let before = driver.snapshot();
    assert_eq!(
        CudaContext::new_with_mem_pool_tracked(Arc::clone(&second), Arc::clone(&pool))
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.initializations,
            after.device_queries,
            after.retains,
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.initializations,
            before.device_queries,
            before.retains,
            before.primary_queries,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    assert!(first.loss().is_none());
    assert!(second.loss().is_none());
    drop(pool);
    drop(first);
    drop(second);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}

#[test]
fn attachment_rejects_untracked_pool_and_legacy_context_rejects_tracked_pool() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let untracked = Arc::new(
        CudaMemPool::create(0, &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    let before = driver.snapshot();
    assert_eq!(
        CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&untracked))
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.initializations,
            after.device_queries,
            after.retains,
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.initializations,
            before.device_queries,
            before.retains,
            before.primary_queries,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    let tracked = Arc::new(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    let before = driver.snapshot();
    assert_eq!(
        CudaContext::new_with_mem_pool(0, Arc::clone(&tracked))
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.initializations,
            after.device_queries,
            after.retains,
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.initializations,
            before.device_queries,
            before.retains,
            before.primary_queries,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    assert!(lifetime.loss().is_none());
    drop(untracked);
    drop(tracked);
    drop(lifetime);
    assert_eq!(native.snapshot().destroyed, 2);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn attachment_native_binding_oom_retries_without_an_extra_primary_retain() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let pool = Arc::new(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    driver.clear_current();
    driver.set_current_bind_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    let result = CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool));
    driver.set_current_bind_error(None);
    assert_eq!(
        result.map(drop).map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
    );
    assert!(lifetime.loss().is_none());
    assert_eq!(
        (
            native.snapshot().destroy_attempts,
            driver.snapshot().retains,
            driver.snapshot().releases
        ),
        (0, 1, 0)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    let context = CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool))
        .map_err(|e| e.0 as u32)
        .unwrap();
    drop(context);
    assert_eq!(native.snapshot().destroy_attempts, 0);
    drop(pool);
    drop(lifetime);
    assert_eq!(native.snapshot().destroyed, 1);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn attachment_actual_terminal_binding_failure_latches_before_borrowed_wrapper_unwind() {
    for error in [
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
    ] {
        let driver = context_test_seams::Guard::new();
        let _legacy = super::mem_pool::seams::Reset::new();
        super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
        let native = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|e| e.0 as u32)
            .unwrap();
        let pool = Arc::new(
            CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
                .map_err(|e| e.0 as u32)
                .unwrap(),
        );
        driver.clear_current();
        driver.set_current_bind_error(Some(error));
        let result =
            CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool));
        driver.set_current_bind_error(None);
        assert_eq!(result.map(drop).map_err(|e| e.0 as u32), Err(error as u32));
        assert_eq!(
            lifetime.loss().map(|loss| loss.cause),
            Some(crate::driver::PrimaryContextLossCause::DriverStatus(error))
        );
        assert_eq!(
            (
                driver.snapshot().retains,
                driver.snapshot().releases,
                native.snapshot().destroy_attempts
            ),
            (1, 0, 0)
        );
        let before = driver.snapshot();
        assert_eq!(
            CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool))
                .map(drop)
                .map_err(|e| e.0 as u32),
            Err(error as u32)
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
        drop(pool);
        assert_eq!(native.snapshot().destroy_attempts, 0);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn attachment_known_loss_refuses_before_native_binding_or_primary_retain() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let pool = Arc::new(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(crate::driver::DriverError(lost));
    let before = driver.snapshot();
    assert_eq!(
        CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool))
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(lost as u32)
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.initializations,
            after.device_queries,
            after.retains,
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.initializations,
            before.device_queries,
            before.retains,
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
fn attachment_lost_last_descendant_disarms_all_native_children_and_releases_anchor_once() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = memory_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    let pool = Arc::new(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map_err(|e| e.0 as u32)
            .unwrap(),
    );
    let context = CudaContext::new_with_mem_pool_tracked(Arc::clone(&lifetime), Arc::clone(&pool))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let stream = context.default_stream();
    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    assert_eq!(native.snapshot().allocations, 1);
    assert_eq!(events.snapshot().created, 2);
    lifetime.record_error(crate::driver::DriverError(
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    ));
    let before = driver.snapshot();
    drop(pool);
    drop(context);
    drop(lifetime);
    drop(stream);
    assert!(weak.upgrade().is_some());
    drop(allocation);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        (
            memory.snapshot().async_frees,
            native.snapshot().destroy_attempts,
            events.snapshot().destroyed
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
    assert_eq!((after.retains, after.releases), (1, 1));
}
