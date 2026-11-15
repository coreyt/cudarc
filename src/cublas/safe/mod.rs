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
/// Retains its stream and context, keeping their ordinary Rust owners alive.
///
/// Tracked cleanup admits and binds before native destruction. A refused
/// generation abandons native cleanup while dropping ordinary Rust owners;
/// native destruction failure publishes its actual library cause once.
#[derive(Debug)]
pub struct CudaBlas {
    pub(crate) handle: sys::cublasHandle_t,
    pub(crate) stream: Arc<CudaStream>,
}

/// Borrowed admission for one raw cuBLAS operation.
///
/// Declare this before pointer guards so admission covers their completion and
/// unwind. Raw calls remain unsafe and require valid sizes, pointers and streams.
#[derive(Debug)]
pub struct CudaBlasLease<'a> {
    blas: &'a CudaBlas,
    #[cfg(feature = "std")]
    _lease: Option<crate::driver::PrimaryContextLease>,
}

impl CudaBlasLease<'_> {
    /// Rechecks cached terminal and delayed errors before one unsafe raw call.
    ///
    /// Call immediately before dispatch, after pointer preparation. This performs
    /// no native query; ordinary delayed errors are consumed once, while cached
    /// terminal loss remains sticky and preserves its authoritative cause.
    pub fn handle(&self) -> Result<sys::cublasHandle_t, CublasError> {
        #[cfg(feature = "std")]
        if self.blas.stream.ctx.primary_lifetime.is_some() {
            self.blas
                .stream
                .ctx
                .check_err()
                .map_err(|error| CudaBlas::admission_error(&self.blas.stream, error))?;
        }
        Ok(self.blas.handle)
    }
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl CudaBlas {
    /// Creates a new cublas handle and assigns `stream`.
    ///
    /// Tracked construction holds admission through initialization and unwind.
    /// Returns the original driver refusal or cached synthetic Lost cause before
    /// library calls, or the unchanged native library error from handle/stream initialization.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let _lease = Self::admit_stream(&stream)?;
        let ctx = stream.context();
        #[cfg(feature = "std")]
        let untracked = ctx.primary_lifetime.is_none();
        #[cfg(not(feature = "std"))]
        let untracked = true;
        if untracked {
            ctx.record_err(ctx.bind_to_thread());
        }
        let handle = result::create_handle()?;
        let blas = Self { handle, stream };
        unsafe { result::set_stream(blas.handle, blas.stream.cu_stream() as _) }?;
        Ok(blas)
    }

    /// Admits and binds one tracked raw operation before pointer preparation.
    ///
    /// Keep the returned guard through native dispatch and pointer-guard unwind.
    /// Returns the exact driver failure or cached synthetic Lost cause before
    /// calling native cuBLAS. Untracked owners retain raw caller responsibility.
    pub fn admit(&self) -> Result<CudaBlasLease<'_>, CublasError> {
        let _lease = Self::admit_stream(&self.stream)?;
        Ok(CudaBlasLease {
            blas: self,
            #[cfg(feature = "std")]
            _lease,
        })
    }

    #[cfg(feature = "std")]
    fn admission_error(stream: &CudaStream, error: crate::driver::DriverError) -> CublasError {
        match stream
            .ctx
            .primary_lifetime
            .as_ref()
            .and_then(|lifetime| lifetime.loss())
        {
            Some(loss) => match loss.refusal() {
                crate::driver::PrimaryContextRefusal::DriverStatus(status) => {
                    CublasError::Driver(crate::driver::DriverError(status))
                }
                _ => CublasError::Lost(loss),
            },
            None => CublasError::Driver(error),
        }
    }

    #[cfg(feature = "std")]
    fn admit_stream(
        stream: &CudaStream,
    ) -> Result<Option<crate::driver::PrimaryContextLease>, CublasError> {
        let ctx = stream.context();
        let Some(lifetime) = &ctx.primary_lifetime else {
            return Ok(None);
        };
        ctx.check_err()
            .map_err(|error| Self::admission_error(stream, error))?;
        let lease = lifetime
            .admit()
            .map_err(|error| Self::admission_error(stream, error))?;
        ctx.bind_to_thread()
            .map_err(|error| Self::admission_error(stream, error))?;
        Ok(Some(lease))
    }

    #[cfg(not(feature = "std"))]
    fn admit_stream(_stream: &CudaStream) -> Result<(), CublasError> {
        Ok(())
    }

    /// Returns a reference to the underlying cublas handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// Assigns the handle to `stream`, retaining its original stream on failure.
    ///
    /// If either context is tracked, both must share one lifetime. Returns
    /// ContextMismatch before native calls for incompatible owners, the exact
    /// driver admission/binding error or cached synthetic Lost cause, or the
    /// unchanged native setter error.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CublasError> {
        #[cfg(feature = "std")]
        match (
            &self.stream.ctx.primary_lifetime,
            &stream.ctx.primary_lifetime,
        ) {
            (None, None) => {}
            (Some(current), Some(target)) if Arc::ptr_eq(current, target) => {}
            _ => return Err(CublasError::ContextMismatch),
        }
        let _lease = Self::admit_stream(&self.stream)?;
        unsafe { result::set_stream(self.handle, stream.cu_stream as _) }?;
        self.stream = stream;
        Ok(())
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
        #[cfg(test)]
        if let Some(result) = result::partial_init_test_seams::set_pointer_mode(pointer_mode) {
            return result;
        }
        unsafe {
            sys::cublasSetPointerMode_v2(self.handle, pointer_mode).result()?;
        }
        Ok(())
    }

    /// Get the handle's current pointer mode.
    /// ref: <https://docs.nvidia.com/cuda/cublas/#cublasgetpointermode>
    pub fn get_pointer_mode(&self) -> Result<sys::cublasPointerMode_t, CublasError> {
        #[cfg(test)]
        if let Some(result) = result::partial_init_test_seams::get_pointer_mode() {
            return result;
        }
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
        if handle.is_null() {
            return;
        }
        #[cfg(feature = "std")]
        let _lease = match &self.stream.ctx.primary_lifetime {
            Some(lifetime) => match lifetime.admit() {
                Ok(lease) => Some(lease),
                Err(error) => {
                    self.stream.ctx.record_err::<()>(Err(error));
                    return;
                }
            },
            None => None,
        };
        #[cfg(feature = "std")]
        if self.stream.ctx.primary_lifetime.is_some() {
            if let Err(error) = self.stream.ctx.bind_for_cleanup() {
                self.stream.ctx.record_err::<()>(Err(error));
                return;
            }
        }
        if let Err(CublasError::Native(status)) = unsafe { result::destroy_handle(handle) } {
            #[cfg(feature = "std")]
            if let Some(lifetime) = &self.stream.ctx.primary_lifetime {
                lifetime.record_library_cleanup_loss(
                    crate::driver::NativeLibrary::Cublas,
                    status as u32,
                );
            }
            #[cfg(not(feature = "std"))]
            let _ = status;
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
            #[cfg(feature = "std")]
            primary_lifetime: None,
        });

        let error = CudaBlas::new(context.default_stream()).unwrap_err();
        assert_eq!(
            error,
            CublasError::Native(sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE)
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
            #[cfg(feature = "std")]
            primary_lifetime: None,
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
            CublasError::Native(sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED)
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

#[cfg(all(test, feature = "std"))]
mod fathomdb_library_admission;

#[cfg(all(test, feature = "std"))]
mod fathomdb_library_setter;

#[cfg(all(test, feature = "std"))]
mod fathomdb_library_drop;

#[cfg(all(test, feature = "std"))]
mod fathomdb_library_destroy_failure;

#[cfg(all(test, feature = "std"))]
mod fathomdb_raw_boundary;

#[cfg(all(test, feature = "std"))]
mod fathomdb_library_use;
