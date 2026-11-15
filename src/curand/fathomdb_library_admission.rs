use super::*;
use crate::driver::{
    result::context_test_seams, sys as driver_sys, CudaContext, PrimaryContextLifetime,
};

#[test]
fn constructor_refuses_cached_terminal_oom_before_native_work() {
    let driver = context_test_seams::Guard::new();
    let rng = result::partial_init_test_seams::Guard::new();
    rng.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    driver.set_context_id_error(Some(driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    assert!(lifetime.admit().is_err());
    let first_loss = lifetime.loss();
    let before = driver.snapshot();

    let outcome = CudaRng::new(42, context.default_stream());

    assert!(
        outcome.is_err(),
        "cached terminal OOM must refuse RNG construction"
    );
    let error = match outcome {
        Err(error) => error,
        Ok(_) => unreachable!(),
    };
    assert_eq!(
        format!("{error:?}"),
        "Driver(DriverError(CUDA_ERROR_OUT_OF_MEMORY))"
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.queries,
            after.binds,
            after.primary_queries,
            after.id_queries
        ),
        (
            before.queries,
            before.binds,
            before.primary_queries,
            before.id_queries
        )
    );
    assert_eq!((rng.snapshot().creates, rng.snapshot().seed_sets), (0, 0));
    assert_eq!(lifetime.loss(), first_loss);
}

#[test]
fn healthy_constructor_and_ordinary_native_oom_remain_distinct() {
    let _driver = context_test_seams::Guard::new();
    let rng = result::partial_init_test_seams::Guard::new();
    rng.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaRng::new(42, context.default_stream()).unwrap();
    drop(owner);
    assert_eq!(rng.snapshot().destroys, 1);
    rng.fail_create();
    assert!(CudaRng::new(42, context.default_stream()).is_err());
    assert!(
        lifetime.loss().is_none(),
        "ordinary RNG allocation failure must not publish loss"
    );
    assert!(lifetime.admit().is_ok());
}

#[test]
fn constructor_discovers_inactive_or_changed_primary_before_create() {
    for changed_id in [false, true] {
        let driver = context_test_seams::Guard::new();
        let rng = result::partial_init_test_seams::Guard::new();
        rng.allow_stream_setup();
        let lifetime = PrimaryContextLifetime::new(0).unwrap();
        let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
        if changed_id {
            driver.set_context_id(18);
        } else {
            driver.set_primary_active(false);
        }

        let outcome = CudaRng::new(42, context.default_stream());

        assert!(
            outcome.is_err(),
            "constructor must discover terminal identity loss"
        );
        assert_eq!(rng.snapshot().creates, 0);
        let loss = lifetime
            .loss()
            .expect("constructor publishes terminal identity loss");
        assert_eq!(
            loss.refusal().driver_status(),
            driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        );
        assert_eq!(loss.observed_id, if changed_id { Some(18) } else { None });
    }
}
