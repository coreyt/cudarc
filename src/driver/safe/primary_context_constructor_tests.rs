#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use crate::driver::{result::context_test_seams, sys, CudaContext, PrimaryContextLifetime};
use std::sync::Arc;

#[test]
fn tracked_context_borrows_the_anchor_and_holds_construction_admission() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    assert_eq!((driver.snapshot().retains, driver.snapshot().id_queries), (1, 1));
    let weak = Arc::downgrade(&lifetime);
    driver.set_allocator_query_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime.state.try_lock().expect("allocator discovery must run outside the admission mutex");
        assert_eq!(state.active_leases, 1);
        assert!(!state.probing);
    });
    let weak = Arc::downgrade(&lifetime);
    driver.set_context_entry_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime.state.try_lock().expect("construction binding must run outside the admission mutex");
        assert_eq!(state.active_leases, 2, "binding shares the held construction epoch through a nested lease");
        assert!(!state.probing);
    });

    let outcome = CudaContext::new_tracked(Arc::clone(&lifetime));
    assert_eq!(outcome.as_ref().map(|_| ()).map_err(|error| error.0 as u32), Ok(()), "healthy tracked construction must be available");
    let first = outcome.unwrap();
    let second = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    assert!(Arc::ptr_eq(first.primary_lifetime.as_ref().unwrap(), &lifetime));
    assert!(Arc::ptr_eq(second.primary_lifetime.as_ref().unwrap(), &lifetime));
    let anchor = lifetime.anchor.as_ref().unwrap();
    assert_eq!((first.cu_device, first.cu_ctx, first.ordinal), (anchor.device, anchor.context, anchor.ordinal));
    assert_eq!((second.cu_device, second.cu_ctx, second.ordinal), (anchor.device, anchor.context, anchor.ordinal));
    let constructed = driver.snapshot();
    assert_eq!((constructed.retains, constructed.releases), (1, 0));
    assert_eq!((constructed.primary_queries, constructed.id_queries), (2, 3));
    assert_eq!((constructed.queries, constructed.binds), (2, 1));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(first);
    drop(second);
    let wrappers = driver.snapshot();
    assert_eq!((wrappers.releases, wrappers.queries, wrappers.binds), (0, 2, 1));
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn replaced_identity_refuses_construction_before_allocator_or_bind() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    driver.set_context_id(42);
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    let outcome = CudaContext::new_tracked(Arc::clone(&lifetime)).map(|_| ()).map_err(|error| error.0 as u32);
    assert_eq!(outcome, Err(lost as u32));
    let dispatch = driver.snapshot();
    assert_eq!((dispatch.retains, dispatch.releases), (1, 0));
    assert_eq!((dispatch.primary_queries, dispatch.id_queries), (1, 2));
    assert_eq!((dispatch.attribute_queries, dispatch.default_pool_queries, dispatch.queries, dispatch.binds), (0, 0, 0, 0));
    let loss = lifetime.state.lock().unwrap().loss.unwrap();
    assert_eq!((loss.recorded_id, loss.observed_id, loss.error as u32), (17, Some(42), lost as u32));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(lifetime);
    assert_eq!(driver.snapshot().releases, 1);
}

#[test]
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
fn allocator_failure_releases_construction_admission_without_releasing_anchor() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    driver.set_attribute_error(Some(oom));
    let outcome = CudaContext::new_tracked(Arc::clone(&lifetime)).map(|_| ()).map_err(|error| error.0 as u32);
    assert_eq!(outcome, Err(oom as u32));
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 0));
    assert_eq!((driver.snapshot().attribute_queries, driver.snapshot().queries, driver.snapshot().binds), (1, 0, 0));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    driver.set_attribute_error(None);
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    assert_eq!(driver.snapshot().attribute_queries, 2);
    drop(context);
    assert_eq!(driver.snapshot().releases, 0);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
fn allocator_unwind_returns_construction_admission_and_preserves_anchor() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let weak = Arc::downgrade(&lifetime);
    driver.set_allocator_query_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        {
            let state = lifetime.state.try_lock().unwrap();
            assert_eq!(state.active_leases, 1);
        }
        panic!("synthetic allocator-discovery unwind");
    });
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| CudaContext::new_tracked(Arc::clone(&lifetime))));
    assert!(outcome.is_err());
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 0));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    driver.set_allocator_query_hook(|| {});
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}
