use super::*;
use crate::driver::{
    result::context_test_seams,
    safe::{
        lifetime_event_cleanup_tests::seams as event_seams,
        lifetime_slice_cleanup_tests::seams as slice_seams,
        lifetime_stream_cleanup_tests::seams as stream_seams,
    },
    sys as driver_sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::atomic::{AtomicBool, Ordering};

struct PointerGuardWitness {
    lifetime: Arc<PrimaryContextLifetime>,
    dropped: Arc<AtomicBool>,
}
impl Drop for PointerGuardWitness {
    fn drop(&mut self) {
        assert_eq!(
            self.lifetime.state.lock().unwrap().active_leases,
            1,
            "the checked call's lease must outlive caller pointer guards"
        );
        self.dropped.store(true, Ordering::SeqCst);
    }
}

fn raw_gemm(call: &CudaBlasLease<'_>) -> Result<(), CublasError> {
    let alpha = 1f32;
    let beta = 0f32;
    unsafe {
        result::gemm_strided_batched_ex(
            call.handle()?,
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            2,
            2,
            2,
            &alpha as *const f32 as _,
            0x1230 as _,
            sys::cudaDataType_t::CUDA_R_32F,
            2,
            4,
            0x1230 as _,
            sys::cudaDataType_t::CUDA_R_32F,
            2,
            4,
            &beta as *const f32 as _,
            0x1230 as _,
            sys::cudaDataType_t::CUDA_R_32F,
            2,
            4,
            1,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}

#[test]
fn default_stream_temporary_loss_before_admission_refuses_without_native() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let slices = slice_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    assert!(!context.is_managing_stream_synchronization());
    let owner = CudaBlas::new(stream.clone()).unwrap();
    let temporary = unsafe { stream.alloc::<f32>(4) }.unwrap();
    slices.set_synchronize_error(Some(driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    drop(temporary);
    assert_eq!(
        (
            slices.snapshot().sync_allocations,
            slices.snapshot().sync_frees
        ),
        (1, 0)
    );
    let first = lifetime.loss().unwrap();
    let before = driver.snapshot();

    let result = owner.admit().and_then(|call| raw_gemm(&call));

    assert_eq!(
        result,
        Err(CublasError::Driver(DriverError(
            first.refusal().driver_status()
        )))
    );
    assert_eq!(calls.snapshot().operations, 0);
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
    assert_eq!(lifetime.loss(), Some(first));
}

#[test]
fn default_stream_temporary_loss_inside_call_refuses_before_raw_dispatch() {
    let _driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let slices = slice_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    let owner = CudaBlas::new(stream.clone()).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));

    let result = (|| {
        let call = owner.admit()?;
        let _pointer_guards = PointerGuardWitness {
            lifetime: lifetime.clone(),
            dropped: dropped.clone(),
        };
        let temporary = unsafe { stream.alloc::<f32>(4) }.unwrap();
        slices.set_synchronize_error(Some(driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
        drop(temporary);
        raw_gemm(&call)
    })();

    assert_eq!(
        result,
        Err(CublasError::Driver(DriverError(
            driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        )))
    );
    assert_eq!(calls.snapshot().operations, 0);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert_eq!(slices.snapshot().sync_frees, 0);
}

fn wait_failure(error: driver_sys::CUresult) {
    let _driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let streams = stream_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    let owner = CudaBlas::new(stream.clone()).unwrap();
    let event = context.new_event(None).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    {
        let call = owner.admit().unwrap();
        let _pointer_guards = PointerGuardWitness {
            lifetime: lifetime.clone(),
            dropped: dropped.clone(),
        };
        streams.set_wait_error(Some(error));
        context.record_err(stream.wait(&event));

        assert_eq!(
            raw_gemm(&call),
            Err(CublasError::Driver(DriverError(error)))
        );

        assert_eq!(calls.snapshot().operations, 0);
        assert_eq!(streams.snapshot().waits, 1);
        assert_eq!(
            lifetime.loss().is_some(),
            error == driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        );
        if lifetime.loss().is_none() {
            streams.set_wait_error(None);
            assert_eq!(raw_gemm(&call), Ok(()));
            assert_eq!(calls.snapshot().operations, 1);
        }
    }
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
}

#[test]
fn pointer_wait_terminal_result_is_rechecked_inside_call() {
    wait_failure(driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn pointer_wait_ordinary_oom_is_returned_then_retryable() {
    wait_failure(driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);
}

#[test]
fn healthy_checked_raw_call_preserves_native_oom_and_retry() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaBlas::new(context.default_stream()).unwrap();
    {
        let call = owner.admit().unwrap();
        calls.fail_operation(true);
        assert_eq!(
            raw_gemm(&call),
            Err(CublasError::Native(
                sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED
            ))
        );
        assert!(lifetime.loss().is_none());
        calls.fail_operation(false);
        assert_eq!(raw_gemm(&call), Ok(()));
    }
    assert_eq!(calls.snapshot().operations, 2);
}

#[test]
fn prior_delayed_ordinary_oom_refuses_admission_once_then_retries() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let owner = CudaBlas::new(context.default_stream()).unwrap();
    context.record_err::<()>(Err(DriverError(
        driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
    )));

    assert_eq!(
        owner.admit().and_then(|call| raw_gemm(&call)),
        Err(CublasError::Driver(DriverError(
            driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
        )))
    );

    assert!(lifetime.loss().is_none());
    assert_eq!(calls.snapshot().operations, 0);
    assert_eq!(owner.admit().and_then(|call| raw_gemm(&call)), Ok(()));
    assert_eq!(calls.snapshot().operations, 1);
}
