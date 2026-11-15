#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::lifetime_event_cleanup_tests::seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
    PrimaryContextLoss,
};
use std::sync::Arc;

fn calls(driver: &context_test_seams::Guard) -> [usize; 12] {
    let state = driver.snapshot();
    [
        state.initializations,
        state.device_queries,
        state.symbol_checks,
        state.attribute_queries,
        state.default_pool_queries,
        state.retains,
        state.releases,
        state.primary_queries,
        state.id_queries,
        state.queries,
        state.binds,
        state.error_string_queries,
    ]
}

#[test]
fn public_loss_observation_exposes_terminal_cleanup_oom_without_native_dispatch() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
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
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    driver.set_current_query_error(Some(oom));
    drop(event);
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 0)
    );
    driver.set_current_query_error(None);
    driver.set_primary_error(Some(sys::CUresult::CUDA_ERROR_UNKNOWN));
    driver.set_context_id(99);
    let before = calls(&driver);
    let expected = Some(PrimaryContextLoss {
        recorded_id: 17,
        observed_id: None,
        cause: crate::driver::PrimaryContextLossCause::DriverStatus(oom),
    });

    assert_eq!(
        lifetime.loss(),
        expected,
        "public observation must distinguish terminal native-cleanup OOM from ordinary OOM"
    );
    assert_eq!(lifetime.loss(), expected);
    assert_eq!(lifetime.recorded_id(), 17);
    context.record_err::<()>(Err(DriverError(
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    )));
    assert_eq!(lifetime.loss(), expected);
    assert_eq!(calls(&driver), before);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn public_recorded_identity_is_available_while_healthy_and_never_requeries() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    driver.set_context_id(99);
    driver.set_context_id_error(Some(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT));
    let before = calls(&driver);

    let observed =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lifetime.recorded_id()));

    assert_eq!(
        observed.ok(),
        Some(17),
        "recorded identity observation must be implemented without consulting the driver"
    );
    assert_eq!(lifetime.loss(), None);
    assert_eq!(calls(&driver), before);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn public_loss_observation_keeps_ordinary_oom_retryable_and_absent() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    context.record_err::<()>(Err(DriverError(oom)));
    let before = calls(&driver);

    assert_eq!(lifetime.loss(), None);
    assert_eq!(lifetime.recorded_id(), 17);

    assert_eq!(
        context.check_err().map_err(|error| error.0 as u32),
        Err(oom as u32)
    );
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    assert_eq!(lifetime.loss(), None);
    assert_eq!(calls(&driver), before);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn public_loss_observation_reports_synthetic_poison_refusal_without_ffi() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _state = lifetime.state.lock().unwrap();
        panic!("synthetic primary-state owner unwind");
    }));
    assert!(poisoned.is_err());
    let before = calls(&driver);
    let expected = Some(PrimaryContextLoss {
        recorded_id: 17,
        observed_id: None,
        cause: crate::driver::PrimaryContextLossCause::SyntheticUnknown,
    });

    assert_eq!(lifetime.loss(), expected);
    assert_eq!(lifetime.loss(), expected);
    assert_eq!(lifetime.recorded_id(), 17);

    assert_eq!(calls(&driver), before);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
