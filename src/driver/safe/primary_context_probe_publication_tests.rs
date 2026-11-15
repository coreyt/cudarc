use crate::driver::{result::context_test_seams, sys, DriverError, PrimaryContextLifetime};
use std::sync::Arc;

#[test]
#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
fn known_loss_during_state_query_prevents_the_next_identity_query() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let mut context = context_test_seams::context();
    Arc::get_mut(&mut context).unwrap().primary_lifetime = Some(Arc::clone(&lifetime));
    let weak = Arc::downgrade(&lifetime);
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    driver.set_primary_probe_hook(move || {
        weak.upgrade().unwrap().record_error(DriverError(lost));
    });

    let first = context.bind_to_thread().map_err(|error| error.0 as u32);
    let second = context.bind_to_thread().map_err(|error| error.0 as u32);
    let dispatch = driver.snapshot();
    drop(context);
    drop(lifetime);

    assert_eq!(first, Err(lost as u32));
    assert_eq!(second, Err(lost as u32));
    assert_eq!(dispatch.primary_queries, 1);
    assert_eq!(
        dispatch.id_queries, 1,
        "known loss must prevent an identity query after the in-flight state query returns"
    );
    assert_eq!((dispatch.queries, dispatch.binds), (0, 0));
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
