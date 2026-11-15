use super::*;
use crate::driver::{
    result::context_test_seams, sys as driver_sys, CudaContext, DriverError, PrimaryContextLifetime,
};

#[test]
fn drop_discovers_loss_without_forward_and_skips_destroy() {
    for changed_id in [false, true] {
        let driver = context_test_seams::Guard::new();
        let calls = result::partial_init_test_seams::Guard::new();
        calls.allow_stream_setup();
        let lifetime = PrimaryContextLifetime::new(0).unwrap();
        let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
        let owner = CudaBlas::new(context.default_stream()).unwrap();
        if changed_id {
            driver.set_context_id(18);
        } else {
            driver.set_primary_active(false);
        }

        drop(owner);

        assert_eq!(
            calls.snapshot().destroys,
            0,
            "no-forward close must discover loss before native library destruction"
        );
        let first = lifetime
            .loss()
            .expect("Drop must publish discovered identity loss");
        assert_eq!(
            first.refusal().driver_status(),
            driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        );
        assert_eq!(first.observed_id, if changed_id { Some(18) } else { None });
    }
}

#[test]
fn cached_terminal_oom_drop_performs_no_native_cleanup() {
    let driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaBlas::new(context.default_stream()).unwrap();
    driver.set_context_id_error(Some(driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    assert!(lifetime.admit().is_err());
    let first = lifetime.loss();
    let before = driver.snapshot();

    drop(owner);

    assert_eq!(calls.snapshot().destroys, 0);
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
    assert_eq!(lifetime.loss(), first);
}

#[test]
fn cleanup_binding_failure_publishes_first_loss_and_skips_destroy() {
    for set_current in [false, true] {
        for error in [
            driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            driver_sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
        ] {
            let driver = context_test_seams::Guard::new();
            let calls = result::partial_init_test_seams::Guard::new();
            calls.allow_stream_setup();
            let lifetime = PrimaryContextLifetime::new(0).unwrap();
            let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
            let owner = CudaBlas::new(context.default_stream()).unwrap();
            if set_current {
                driver.clear_current();
                driver.set_current_bind_error(Some(error));
            } else {
                driver.set_current_query_error(Some(error));
            }

            drop(owner);

            assert_eq!(
                calls.snapshot().destroys,
                0,
                "native cleanup binding uncertainty must stop destruction"
            );
            let first = lifetime
                .loss()
                .expect("actual cleanup binding failure is conservative terminal loss");
            assert_eq!(first.refusal().driver_status(), error);
            let before = driver.snapshot();
            driver.set_current_bind_error(None);
            driver.set_current_query_error(None);
            assert_eq!(context.bind_to_thread(), Err(DriverError(error)));
            context.record_err::<()>(Err(DriverError(
                driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            )));
            assert_eq!(lifetime.loss(), Some(first));
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
        }
    }
}

#[test]
fn healthy_tracked_drop_destroys_once() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaBlas::new(context.default_stream()).unwrap();

    drop(owner);

    assert_eq!(calls.snapshot().destroys, 1);
    assert!(lifetime.loss().is_none());
}

#[test]
fn preexisting_ordinary_oom_does_not_abandon_healthy_cleanup() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaBlas::new(context.default_stream()).unwrap();
    context.record_err::<()>(Err(DriverError(
        driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
    )));

    drop(owner);

    assert_eq!(
        calls.snapshot().destroys,
        1,
        "a preexisting ordinary OOM must not abandon healthy native cleanup"
    );
    assert!(
        lifetime.loss().is_none(),
        "pending ordinary OOM is not actual native cleanup-binding failure"
    );
}
