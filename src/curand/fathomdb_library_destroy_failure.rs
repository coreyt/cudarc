use super::*;
use crate::driver::{result::context_test_seams, sys as driver_sys, CudaContext, DriverError,
    NativeLibrary, PrimaryContextLifetime, PrimaryContextLossCause, PrimaryContextRefusal};

#[test]
fn destroy_failure_records_actual_library_cause_and_stops_growth() {
    for status in [sys::curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE, sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED] {
        let driver = context_test_seams::Guard::new();
        let calls = result::partial_init_test_seams::Guard::new();
        calls.allow_stream_setup();
        let lifetime = PrimaryContextLifetime::new(0).unwrap();
        let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
        let owner = CudaRng::new(42, context.default_stream()).unwrap();
        calls.set_destroy_error(Some(result::CurandError::Native(status)));

        drop(owner);

        let first = lifetime.loss().expect("failed native library reclamation must freeze the generation");
        assert_eq!(first.cause, PrimaryContextLossCause::NativeLibraryCleanup { library: NativeLibrary::Curand, status: status as u32 });
        assert_eq!(first.refusal(), PrimaryContextRefusal::SyntheticUnknown);
        assert_eq!(calls.snapshot().destroys, 1);
        let before = driver.snapshot();
        calls.set_destroy_error(None);
        for _ in 0..3 {
            assert_eq!(CudaRng::new(42, context.default_stream()).map(|_| ()), Err(result::CurandError::Lost(first)));
        }
        assert_eq!(context.bind_to_thread(), Err(DriverError(driver_sys::CUresult::CUDA_ERROR_UNKNOWN)));
        assert_eq!(calls.snapshot().creates, 1, "a terminal generation cannot accumulate replacement handles");
        assert_eq!(calls.snapshot().destroys, 1);
        let after = driver.snapshot();
        assert_eq!((after.queries, after.binds, after.primary_queries, after.id_queries),
            (before.queries, before.binds, before.primary_queries, before.id_queries));
        assert_eq!(lifetime.loss(), Some(first));
    }
}

#[test]
fn partial_initialization_destroy_failure_keeps_actual_first_cause() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    calls.fail_seed();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let status = sys::curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE;
    calls.set_destroy_error(Some(result::CurandError::Native(status)));

    assert_eq!(CudaRng::new(42, context.default_stream()).map(|_| ()), Err(result::CurandError::Native(sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR)));

    let first = lifetime.loss().expect("partial initialization must publish failed native cleanup");
    assert_eq!(first.cause, PrimaryContextLossCause::NativeLibraryCleanup { library: NativeLibrary::Curand, status: status as u32 });
    assert_eq!(first.refusal(), PrimaryContextRefusal::SyntheticUnknown);
    assert_eq!(calls.snapshot().destroys, 1);
}

#[cfg(feature = "cublas")]
#[test]
fn first_library_cause_survives_other_library_drop_and_driver_observation() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let other_calls = crate::cublas::result::partial_init_test_seams::Guard::new();
    other_calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaRng::new(42, context.default_stream()).unwrap();
    let other = crate::cublas::CudaBlas::new(context.default_stream()).unwrap();
    let status = sys::curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE;
    calls.set_destroy_error(Some(result::CurandError::Native(status)));

    drop(owner);
    let first = lifetime.loss().expect("the first library failure must be published before returning");
    drop(other);
    context.record_err::<()>(Err(DriverError(driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED)));

    assert_eq!(first.cause, PrimaryContextLossCause::NativeLibraryCleanup { library: NativeLibrary::Curand, status: status as u32 });
    assert_eq!(lifetime.loss(), Some(first));
    assert_eq!(other_calls.snapshot().destroys, 0, "later library cleanup must refuse the first terminal generation");
    assert_eq!(context.check_err(), Err(DriverError(driver_sys::CUresult::CUDA_ERROR_UNKNOWN)));
}
