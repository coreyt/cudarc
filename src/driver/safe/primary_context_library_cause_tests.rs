#![cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]

use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, NativeLibrary, PrimaryContextLifetime, PrimaryContextLoss, PrimaryContextLossCause, PrimaryContextRefusal};
use std::sync::Arc;

fn calls(driver: &context_test_seams::Guard) -> [usize; 12] {
    let state = driver.snapshot();
    [state.initializations, state.device_queries, state.symbol_checks, state.attribute_queries, state.default_pool_queries, state.retains, state.releases, state.primary_queries, state.id_queries, state.queries, state.binds, state.error_string_queries]
}

fn check_library_publication(library: NativeLibrary, status: u32) {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let sibling = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let before = calls(&driver);

    lifetime.record_library_cleanup_loss(library, status);

    let loss = lifetime.loss();
    assert_eq!(loss.map(|loss| (loss.recorded_id, loss.observed_id, loss.refusal().driver_status() as u32)), Some((17, None, sys::CUresult::CUDA_ERROR_UNKNOWN as u32)), "tagged native library cleanup failure must publish terminal refusal");
    assert_eq!(loss.map(|loss| loss.cause), Some(PrimaryContextLossCause::NativeLibraryCleanup { library, status }));
    assert_eq!(loss.map(PrimaryContextLoss::refusal), Some(PrimaryContextRefusal::SyntheticUnknown));
    let diagnostic = format!("{:?}", loss);
    assert!(diagnostic.contains("NativeLibraryCleanup"));
    assert!(!diagnostic.contains("DriverStatus"));
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Err(sys::CUresult::CUDA_ERROR_UNKNOWN as u32));
    assert_eq!(sibling.check_err().map_err(|error| error.0 as u32), Err(sys::CUresult::CUDA_ERROR_UNKNOWN as u32));
    lifetime.record_loss(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED), Some(99));
    lifetime.record_library_cleanup_loss(if library == NativeLibrary::Cublas { NativeLibrary::Curand } else { NativeLibrary::Cublas }, status + 1);
    assert_eq!(lifetime.loss(), loss);
    assert_eq!(sibling.bind_to_thread().map_err(|error| error.0 as u32), Err(sys::CUresult::CUDA_ERROR_UNKNOWN as u32));
    assert_eq!(calls(&driver), before);
    drop(outer);
    drop(context);
    drop(sibling);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn cublas_cleanup_publication_establishes_shared_terminal_refusal() { check_library_publication(NativeLibrary::Cublas, 3); }
#[test]
fn curand_cleanup_publication_establishes_shared_terminal_refusal() { check_library_publication(NativeLibrary::Curand, 102); }

#[test]
fn library_observation_never_replaces_first_actual_driver_status() {
    for status in [sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY, sys::CUresult::CUDA_ERROR_UNKNOWN] {
        let driver = context_test_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
        lifetime.record_loss(DriverError(status), Some(23));
        let before = calls(&driver);
        let first = lifetime.loss();
        lifetime.record_library_cleanup_loss(NativeLibrary::Cublas, 3);
        lifetime.record_library_cleanup_loss(NativeLibrary::Curand, 102);
        assert_eq!(lifetime.loss(), first);
        assert_eq!(first.map(|loss| (loss.recorded_id, loss.observed_id, loss.refusal().driver_status() as u32)), Some((17, Some(23), status as u32)));
        assert_eq!(first.map(|loss| loss.cause), Some(PrimaryContextLossCause::DriverStatus(status)));
        assert_eq!(first.map(PrimaryContextLoss::refusal), Some(PrimaryContextRefusal::DriverStatus(status)));
        let synthetic = PrimaryContextLoss { recorded_id: 17, observed_id: None, cause: PrimaryContextLossCause::SyntheticUnknown };
        assert_eq!(synthetic.refusal(), PrimaryContextRefusal::SyntheticUnknown);
        assert_eq!(synthetic.refusal().driver_status() as u32, sys::CUresult::CUDA_ERROR_UNKNOWN as u32);
        assert_ne!(synthetic.refusal(), PrimaryContextRefusal::DriverStatus(sys::CUresult::CUDA_ERROR_UNKNOWN));
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Err(status as u32));
        assert_eq!(calls(&driver), before);
        drop(context);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}
