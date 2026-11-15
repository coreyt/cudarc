use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime};
use std::sync::Arc;

fn attached_context(lifetime: &Arc<PrimaryContextLifetime>) -> Arc<CudaContext> {
    let mut context = context_test_seams::context();
    Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(Arc::clone(lifetime));
    context
}

#[test]
#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
fn tracked_bind_shares_an_existing_admission_epoch() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = attached_context(&lifetime);
    let lease = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let admitted = driver.snapshot();
    assert_eq!((admitted.primary_queries, admitted.id_queries), (1, 2));

    let result = context.bind_to_thread().map_err(|error| error.0 as u32);
    let bound = driver.snapshot();
    assert_eq!(result, Ok(()));
    assert_eq!((bound.primary_queries, bound.id_queries), (1, 2), "binding under a held lease must share its admitted epoch");
    assert_eq!((bound.queries, bound.binds), (1, 1));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
    drop(lease);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    assert_eq!(driver.snapshot().releases, 0);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
fn idle_tracked_bind_holds_admission_through_native_entry() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = attached_context(&lifetime);
    let weak = Arc::downgrade(&lifetime);
    driver.set_context_entry_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime.state.try_lock().expect("native entry must run outside the state mutex");
        assert_eq!(state.active_leases, 1, "the idle bind must retain admission through native dispatch");
        assert!(!state.probing);
    });

    assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Ok(()));
    let bound = driver.snapshot();
    assert_eq!((bound.primary_queries, bound.id_queries), (1, 2));
    assert_eq!((bound.queries, bound.binds), (1, 1));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
fn tracked_bind_refuses_published_loss_before_native_entry() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = attached_context(&lifetime);
    let lease = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();
    driver.set_context_entry_hook(|| panic!("published loss must prevent native context entry"));

    assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Err(lost as u32));
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries, before.id_queries));
    assert_eq!((after.queries, after.binds), (0, 0));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
    drop(lease);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}
