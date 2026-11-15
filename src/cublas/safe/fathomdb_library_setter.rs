use super::*;
use crate::driver::{result::context_test_seams, sys as driver_sys, CudaContext, DriverError, PrimaryContextLifetime};

#[test]
fn failed_setter_retains_original_stream_owner() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let mut owner = CudaBlas::new(context.default_stream()).unwrap();
    let old = Arc::downgrade(&owner.stream);
    let target = context.default_stream();
    calls.fail_stream_setup();

    let failed = unsafe { owner.set_stream(target.clone()) };

    assert_eq!(failed, Err(CublasError::Native(sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE)));
    assert!(old.upgrade().is_some(), "failed reassociation must retain the native handle's original stream owner");
    assert!(!Arc::ptr_eq(&owner.stream, &target));
    assert!(lifetime.loss().is_none(), "ordinary library setter error must not publish loss");
}

#[test]
fn setter_refuses_mixed_or_distinct_tracked_generations_before_native() {
    for mode in [0, 1, 2] {
        let driver = context_test_seams::Guard::new();
        let calls = result::partial_init_test_seams::Guard::new();
        calls.allow_stream_setup();
        let lifetime = PrimaryContextLifetime::new(0).unwrap();
        let tracked = CudaContext::new_tracked(lifetime.clone()).unwrap();
        let context = if mode == 2 { context_test_seams::context() } else { tracked };
        let mut owner = CudaBlas::new(context.default_stream()).unwrap();
        let original = owner.stream.clone();
        let other = if mode == 1 { context_test_seams::context() } else {
            CudaContext::new_tracked(PrimaryContextLifetime::new(0).unwrap()).unwrap()
        };
        let before = calls.snapshot().stream_sets;
        let native_before = driver.snapshot();

        let failed = unsafe { owner.set_stream(other.default_stream()) };

        assert_eq!(failed, Err(CublasError::ContextMismatch));
        assert_eq!(calls.snapshot().stream_sets, before, "association refusal precedes library FFI");
        let after = driver.snapshot();
        assert_eq!((after.queries, after.binds, after.primary_queries, after.id_queries),
            (native_before.queries, native_before.binds, native_before.primary_queries, native_before.id_queries));
        assert!(Arc::ptr_eq(&owner.stream, &original));
        assert!(lifetime.loss().is_none());
    }
}

#[test]
fn setter_refuses_cached_loss_without_native_work() {
    let driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let mut owner = CudaBlas::new(context.default_stream()).unwrap();
    let original = owner.stream.clone();
    driver.set_primary_active(false);
    assert!(lifetime.admit().is_err());
    let first = lifetime.loss();
    let before = driver.snapshot();
    let native_before = calls.snapshot().stream_sets;

    let failed = unsafe { owner.set_stream(context.default_stream()) };

    assert_eq!(failed, Err(CublasError::Driver(DriverError(driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED))));
    assert_eq!(calls.snapshot().stream_sets, native_before);
    assert!(Arc::ptr_eq(&owner.stream, &original));
    let after = driver.snapshot();
    assert_eq!((after.queries, after.binds, after.primary_queries, after.id_queries),
        (before.queries, before.binds, before.primary_queries, before.id_queries));
    assert_eq!(lifetime.loss(), first);
}

#[test]
fn setter_accepts_same_generation_and_changes_owner_after_success() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let mut owner = CudaBlas::new(context.default_stream()).unwrap();
    let old = Arc::downgrade(&owner.stream);
    let target = CudaContext::new_tracked(lifetime.clone()).unwrap().default_stream();

    assert_eq!(unsafe { owner.set_stream(target.clone()) }, Ok(()));

    assert!(Arc::ptr_eq(&owner.stream, &target));
    assert!(old.upgrade().is_none());
    assert_eq!(calls.snapshot().stream_sets, 2);
    assert!(lifetime.loss().is_none());
}
