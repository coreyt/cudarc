//! Safe abstractions around [crate::cublas::result] for doing gemm and gemv.
#![allow(clippy::too_many_arguments)]

use super::{result, result::CublasError, sys};
use crate::driver::CudaStream;
use std::sync::Arc;

mod asum;
mod gemm;
mod gemv;
mod grouped_gemm;

pub use asum::*;
pub use gemm::*;
pub use gemv::*;
pub use grouped_gemm::*;

/// Wrapper around [sys::cublasHandle_t]
///
/// 1. Create with [CudaBlas::new()]
/// 2. Execute gemm/gemv/gmm kernels with [Gemv], [Gemm] and [Gmm]. Both f32 and f64 are supported
///    for [Gemm] and [Gemv], f16 and bf16 are supported for [Gmm] if feature `half` is activated.
///
/// Note: This maintains a instance of [`Arc<CudaDevice>`], so will prevent the device
/// from being dropped.
#[derive(Debug)]
pub struct CudaBlas {
    pub(crate) handle: sys::cublasHandle_t,
    pub(crate) stream: Arc<CudaStream>,
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl CudaBlas {
    /// Creates a new cublas handle and sets the stream to the `device`'s stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::create_handle()?;
        let blas = Self { handle, stream };
        unsafe { result::set_stream(blas.handle, blas.stream.cu_stream() as _) }?;
        Ok(blas)
    }

    /// Returns a reference to the underlying cublas handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// Sets the handle's current to either the stream specified, or the device's default work
    /// stream.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CublasError> {
        self.stream = stream;
        unsafe { result::set_stream(self.handle, self.stream.cu_stream() as _) }
    }

    /// Set the handle's pointer mode.
    /// ref: <https://docs.nvidia.com/cuda/cublas/#cublassetpointermode>
    ///
    /// Some cublas functions require the pointer mode to be set to `cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE`
    /// when passing a device memory result buffer into the function, such as `cublas<t>asum()`.
    /// Otherwise the operation will panic with `SIGSEGV: invalid memory reference`,
    /// or one has to use a host memory reference, which has performance implications.
    pub fn set_pointer_mode(
        &self,
        pointer_mode: sys::cublasPointerMode_t,
    ) -> Result<(), CublasError> {
        unsafe {
            sys::cublasSetPointerMode_v2(self.handle, pointer_mode).result()?;
        }
        Ok(())
    }

    /// Get the handle's current pointer mode.
    /// ref: <https://docs.nvidia.com/cuda/cublas/#cublasgetpointermode>
    pub fn get_pointer_mode(&self) -> Result<sys::cublasPointerMode_t, CublasError> {
        unsafe {
            let mut mode = ::core::mem::MaybeUninit::uninit();
            sys::cublasGetPointerMode_v2(self.handle, mode.as_mut_ptr()).result()?;
            Ok(mode.assume_init())
        }
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            let _ = unsafe { result::destroy_handle(handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use crate::driver::CudaContext;

    use super::*;

    #[test]
    fn cublas_pointer_mode() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).unwrap();
        assert_eq!(
            blas.get_pointer_mode().unwrap(),
            sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            "The default pointer mode uses host pointers"
        );

        blas.set_pointer_mode(sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE)
            .unwrap();
        assert_eq!(
            blas.get_pointer_mode().unwrap(),
            sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            "We have set the mode to use device pointers"
        );
    }
}

#[cfg(test)]
mod fathomdb_context_lifetime {
    use super::*;
    use crate::driver::{result::context_test_seams, sys as driver_sys, CudaContext};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};

    #[test]
    fn fathomdb_blas_partial_initialization_releases_handle() {
        let driver = context_test_seams::Guard::new();
        let blas = result::partial_init_test_seams::Guard::new();
        let context = Arc::new(CudaContext {
            cu_device: 0,
            cu_ctx: 0xC7 as driver_sys::CUcontext,
            ordinal: 0,
            has_async_alloc: false,
            mem_pool: None,
            is_primary: true,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });

        let error = CudaBlas::new(context.default_stream()).unwrap_err();
        assert_eq!(
            error,
            CublasError(sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE)
        );
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = blas.snapshot();
        assert_eq!((calls.creates, calls.stream_sets), (1, 1));
        assert_eq!(
            calls.destroys, 1,
            "failed stream setup must release its acquired BLAS handle"
        );
    }

    fn synthetic_context() -> Arc<CudaContext> {
        Arc::new(CudaContext {
            cu_device: 0,
            cu_ctx: 0xC7 as driver_sys::CUcontext,
            ordinal: 0,
            has_async_alloc: false,
            mem_pool: None,
            is_primary: true,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        })
    }

    #[test]
    fn fathomdb_blas_failed_create_has_no_handle_to_destroy() {
        let driver = context_test_seams::Guard::new();
        let blas = result::partial_init_test_seams::Guard::new();
        blas.fail_create();
        let context = synthetic_context();

        let error = CudaBlas::new(context.default_stream()).unwrap_err();
        assert_eq!(
            error,
            CublasError(sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED)
        );
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = blas.snapshot();
        assert_eq!(
            (calls.creates, calls.stream_sets, calls.destroys),
            (1, 0, 0)
        );
    }

    #[test]
    fn fathomdb_blas_completed_owner_destroys_handle_once() {
        let driver = context_test_seams::Guard::new();
        let blas = result::partial_init_test_seams::Guard::new();
        blas.allow_stream_setup();
        let context = synthetic_context();

        let owner = CudaBlas::new(context.default_stream()).unwrap();
        let calls = blas.snapshot();
        assert_eq!(
            (calls.creates, calls.stream_sets, calls.destroys),
            (1, 1, 0)
        );
        drop(owner);
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = blas.snapshot();
        assert_eq!(
            (calls.creates, calls.stream_sets, calls.destroys),
            (1, 1, 1)
        );
    }

    #[test]
    fn fathomdb_blas_cleanup_error_does_not_unwind() {
        let driver = context_test_seams::Guard::new();
        let blas = result::partial_init_test_seams::Guard::new();
        blas.allow_stream_setup();
        blas.fail_destroy();
        let context = synthetic_context();
        let owner = CudaBlas::new(context.default_stream()).unwrap();

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = blas.snapshot();
        assert_eq!(
            (calls.creates, calls.stream_sets, calls.destroys),
            (1, 1, 1)
        );
        assert!(
            outcome.is_ok(),
            "native BLAS cleanup errors must not unwind Rust owners"
        );
    }
}
