use super::*;
use crate::driver::{
    result::context_test_seams,
    safe::{
        lifetime_event_cleanup_tests::seams as event_seams,
        lifetime_stream_cleanup_tests::seams as stream_seams,
    },
    sys as driver_sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use crate::driver::{CudaEvent, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use core::cell::Cell;

struct Operand {
    stream: Arc<CudaStream>,
    prepared: Cell<usize>,
    wait: Option<CudaEvent>,
}

impl Operand {
    fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            prepared: Cell::new(0),
            wait: None,
        }
    }
    fn prepare(&self, stream: &CudaStream) {
        self.prepared.set(self.prepared.get() + 1);
        if let Some(event) = &self.wait {
            stream.context().record_err(stream.wait(event));
        }
    }
}

impl DeviceSlice<f32> for Operand {
    fn len(&self) -> usize {
        4
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}
impl DevicePtr<f32> for Operand {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (driver_sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.prepare(stream);
        (0x1230, SyncOnDrop::Record(None))
    }
}
impl DevicePtrMut<f32> for Operand {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (driver_sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.prepare(stream);
        (0x1230, SyncOnDrop::Record(None))
    }
}

fn config() -> GemmConfig<f32> {
    GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_N,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: 2,
        n: 2,
        k: 2,
        alpha: 1.,
        lda: 2,
        ldb: 2,
        beta: 0.,
        ldc: 2,
    }
}

#[test]
fn safe_use_refuses_cached_loss_before_pointer_or_library_calls() {
    let driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    let owner = CudaBlas::new(stream.clone()).unwrap();
    let a = Operand::new(stream.clone());
    let b = Operand::new(stream.clone());
    let mut c = Operand::new(stream);
    driver.set_primary_active(false);
    assert!(lifetime.admit().is_err());
    let refused = Err(CublasError::Lost(lifetime.loss().unwrap()));

    assert_eq!(unsafe { owner.gemm(config(), &a, &b, &mut c) }, refused);
    assert_eq!(
        unsafe {
            owner.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: config(),
                    batch_size: 1,
                    stride_a: 4,
                    stride_b: 4,
                    stride_c: 4,
                },
                &a,
                &b,
                &mut c,
            )
        },
        refused
    );
    assert_eq!(
        unsafe {
            owner.gemv(
                GemvConfig {
                    trans: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: 2,
                    n: 2,
                    alpha: 1.,
                    lda: 2,
                    incx: 1,
                    beta: 0.,
                    incy: 1,
                },
                &a,
                &b,
                &mut c,
            )
        },
        refused
    );
    assert_eq!(
        unsafe { owner.asum(AsumConfig { n: 4, incx: 1 }, &a, &mut 0.) },
        refused
    );
    assert_eq!(
        owner.set_pointer_mode(sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST),
        refused
    );
    assert_eq!(owner.get_pointer_mode().map(|_| ()), refused);
    assert_eq!(
        (a.prepared.get(), b.prepared.get(), c.prepared.get()),
        (0, 0, 0)
    );
    assert_eq!(calls.snapshot().operations, 0);
}

#[test]
fn safe_gemm_rechecks_native_wait_failure_before_library_dispatch() {
    for error in [
        driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
    ] {
        let _driver = context_test_seams::Guard::new();
        let _events = event_seams::Guard::new();
        let streams = stream_seams::Guard::new();
        let calls = result::partial_init_test_seams::Guard::new();
        calls.allow_stream_setup();
        let lifetime = PrimaryContextLifetime::new(0).unwrap();
        let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
        let stream = context.default_stream();
        let owner = CudaBlas::new(stream.clone()).unwrap();
        let mut a = Operand::new(stream.clone());
        a.wait = Some(context.new_event(None).unwrap());
        let b = Operand::new(stream.clone());
        let mut c = Operand::new(stream);
        streams.set_wait_error(Some(error));

        assert_eq!(
            unsafe { owner.gemm(config(), &a, &b, &mut c) },
            Err(CublasError::Driver(DriverError(error)))
        );

        assert_eq!(streams.snapshot().waits, 1);
        assert_eq!(calls.snapshot().operations, 0);
        assert_eq!(
            lifetime.loss().is_some(),
            error == driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        );
        if lifetime.loss().is_none() {
            streams.set_wait_error(None);
            assert_eq!(unsafe { owner.gemm(config(), &a, &b, &mut c) }, Ok(()));
            assert_eq!(calls.snapshot().operations, 1);
        }
    }
}

#[test]
fn healthy_safe_gemm_preserves_native_error_and_retry() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    let owner = CudaBlas::new(stream.clone()).unwrap();
    let a = Operand::new(stream.clone());
    let b = Operand::new(stream.clone());
    let mut c = Operand::new(stream);
    calls.fail_operation(true);
    assert_eq!(
        unsafe { owner.gemm(config(), &a, &b, &mut c) },
        Err(CublasError::Native(
            sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED
        ))
    );
    assert!(lifetime.loss().is_none());
    calls.fail_operation(false);
    assert_eq!(unsafe { owner.gemm(config(), &a, &b, &mut c) }, Ok(()));
    assert_eq!(
        unsafe {
            owner.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: config(),
                    batch_size: 1,
                    stride_a: 4,
                    stride_b: 4,
                    stride_c: 4,
                },
                &a,
                &b,
                &mut c,
            )
        },
        Ok(())
    );
    assert_eq!(
        unsafe {
            owner.gemv(
                GemvConfig {
                    trans: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: 2,
                    n: 2,
                    alpha: 1.,
                    lda: 2,
                    incx: 1,
                    beta: 0.,
                    incy: 1,
                },
                &a,
                &b,
                &mut c,
            )
        },
        Ok(())
    );
    assert_eq!(
        unsafe { owner.asum(AsumConfig { n: 4, incx: 1 }, &a, &mut 0.) },
        Ok(())
    );
    assert_eq!(
        owner.set_pointer_mode(sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE),
        Ok(())
    );
    assert_eq!(
        owner.get_pointer_mode(),
        Ok(sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE)
    );
    assert_eq!(calls.snapshot().operations, 7);
    assert_eq!(
        (a.prepared.get(), b.prepared.get(), c.prepared.get()),
        (5, 4, 4)
    );
}
