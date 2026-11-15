//! Safe abstractions around [crate::curand::result] with [CudaRng].

use super::{result, sys};
use crate::driver::{CudaStream, DevicePtrMut};
use std::sync::Arc;

/// Host side RNG that can fill [crate::driver::CudaSlice]/[crate::driver::CudaViewMut] with random values.
///
/// 1. Create:
/// ```rust
/// # use cudarc::{driver::*, curand::*};
/// let ctx = CudaContext::new(0).unwrap();
/// let stream = ctx.default_stream();
/// let rng = CudaRng::new(0, stream.clone()).unwrap();
/// ```
/// 2. Fill device memory:
/// ```rust
/// # use cudarc::{driver::*, curand::*};
/// # let ctx = CudaContext::new(0).unwrap();
/// # let stream = ctx.default_stream();
/// # let rng = CudaRng::new(0, stream.clone()).unwrap();
/// let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
/// rng.fill_with_uniform(&mut a_dev).unwrap();
/// ```
///
/// The three distributions are:
/// 1. Uniform - [CudaRng::fill_with_uniform()]
/// 2. Normal - [CudaRng::fill_with_normal()]
/// 3. LogNormal - [CudaRng::fill_with_log_normal()]
///
/// Tracked cleanup admits and binds before native destruction. A refused
/// generation abandons native cleanup while dropping ordinary Rust owners;
/// native destruction failure publishes its actual library cause once.
pub struct CudaRng {
    pub(crate) gen: sys::curandGenerator_t,
    pub(crate) stream: Arc<CudaStream>,
}

impl CudaRng {
    /// Constructs the RNG with the given `seed`. All calls run on `stream`.
    ///
    /// Tracked construction holds admission through initialization and unwind.
    /// Returns the original driver refusal or cached synthetic Lost cause before
    /// library calls, or the unchanged native library error from generator initialization.
    pub fn new(seed: u64, stream: Arc<CudaStream>) -> Result<Self, result::CurandError> {
        let _lease = Self::admit_stream(&stream)?;
        let ctx = &stream.ctx;
        #[cfg(feature = "std")]
        let untracked = ctx.primary_lifetime.is_none();
        #[cfg(not(feature = "std"))]
        let untracked = true;
        if untracked {
            ctx.record_err(ctx.bind_to_thread());
        }
        let gen = result::create_generator()?;
        let mut rng = Self { gen, stream };
        unsafe { result::set_stream(rng.gen, rng.stream.cu_stream as _) }?;
        rng.set_seed(seed)?;
        Ok(rng)
    }

    #[cfg(feature = "std")]
    fn admission_error(
        stream: &CudaStream,
        error: crate::driver::DriverError,
    ) -> result::CurandError {
        match stream
            .ctx
            .primary_lifetime
            .as_ref()
            .and_then(|lifetime| lifetime.loss())
        {
            Some(loss) => match loss.refusal() {
                crate::driver::PrimaryContextRefusal::DriverStatus(status) => {
                    result::CurandError::Driver(crate::driver::DriverError(status))
                }
                _ => result::CurandError::Lost(loss),
            },
            None => result::CurandError::Driver(error),
        }
    }

    #[cfg(feature = "std")]
    fn admit_stream(
        stream: &CudaStream,
    ) -> Result<Option<crate::driver::PrimaryContextLease>, result::CurandError> {
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
    fn admit_stream(_stream: &CudaStream) -> Result<(), result::CurandError> {
        Ok(())
    }

    /// Assigns the generator to `stream`, retaining its original stream on failure.
    ///
    /// If either context is tracked, both must share one lifetime. Returns
    /// ContextMismatch before native calls for incompatible owners, the exact
    /// driver admission/binding error or cached synthetic Lost cause, or the
    /// unchanged native setter error.
    ///
    /// # Safety
    /// Users must ensure this stream is properly synchronized.
    pub unsafe fn set_stream(
        &mut self,
        stream: Arc<CudaStream>,
    ) -> Result<(), result::CurandError> {
        #[cfg(feature = "std")]
        match (
            &self.stream.ctx.primary_lifetime,
            &stream.ctx.primary_lifetime,
        ) {
            (None, None) => {}
            (Some(current), Some(target)) if Arc::ptr_eq(current, target) => {}
            _ => return Err(result::CurandError::ContextMismatch),
        }
        let _lease = Self::admit_stream(&self.stream)?;
        unsafe { result::set_stream(self.gen, stream.cu_stream as _) }?;
        self.stream = stream;
        Ok(())
    }

    /// Re-seed the RNG.
    pub fn set_seed(&mut self, seed: u64) -> Result<(), result::CurandError> {
        unsafe { result::set_seed(self.gen, seed) }
    }

    pub fn set_offset(&mut self, offset: u64) -> Result<(), result::CurandError> {
        unsafe { result::set_offset(self.gen, offset) }
    }

    /// Fill the [crate::driver::CudaSlice]/[crate::driver::CudaViewMut] with data from a `Uniform` distribution
    pub fn fill_with_uniform<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
    ) -> Result<(), result::CurandError>
    where
        sys::curandGenerator_t: result::UniformFill<T>,
    {
        let num = dst.len();
        let (dst, _record_dst) = dst.device_ptr_mut(&self.stream);
        unsafe { result::UniformFill::fill(self.gen, dst as *mut T, num) }
    }

    /// Fill the [crate::driver::CudaSlice]/[crate::driver::CudaViewMut] with data from a `Normal(mean, std)` distribution.
    pub fn fill_with_normal<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        mean: T,
        std: T,
    ) -> Result<(), result::CurandError>
    where
        sys::curandGenerator_t: result::NormalFill<T>,
    {
        let num = dst.len();
        let (dst, _record_dst) = dst.device_ptr_mut(&self.stream);
        unsafe { result::NormalFill::fill(self.gen, dst as *mut T, num, mean, std) }
    }

    /// Fill the [crate::driver::CudaSlice]/[crate::driver::CudaViewMut] with data from a `LogNormal(mean, std)` distribution.
    pub fn fill_with_log_normal<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        mean: T,
        std: T,
    ) -> Result<(), result::CurandError>
    where
        sys::curandGenerator_t: result::LogNormalFill<T>,
    {
        let num = dst.len();
        let (dst, _record_dst) = dst.device_ptr_mut(&self.stream);
        unsafe { result::LogNormalFill::fill(self.gen, dst as *mut T, num, mean, std) }
    }
}

impl Drop for CudaRng {
    fn drop(&mut self) {
        let gen = std::mem::replace(&mut self.gen, std::ptr::null_mut());
        if gen.is_null() {
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
        if let Err(result::CurandError::Native(status)) = unsafe { result::destroy_generator(gen) }
        {
            #[cfg(feature = "std")]
            if let Some(lifetime) = &self.stream.ctx.primary_lifetime {
                lifetime.record_library_cleanup_loss(
                    crate::driver::NativeLibrary::Curand,
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

    use super::*;
    use crate::{
        curand::result::{LogNormalFill, NormalFill, UniformFill},
        driver::*,
    };
    use std::vec::Vec;

    fn gen_uniform<T: ValidAsZeroBits + Clone + Default + Unpin + DeviceRepr>(
        seed: u64,
        n: usize,
    ) -> Vec<T>
    where
        super::sys::curandGenerator_t: UniformFill<T>,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a_dev = stream.alloc_zeros::<T>(n).unwrap();
        let rng = CudaRng::new(seed, stream.clone()).unwrap();
        rng.fill_with_uniform(&mut a_dev).unwrap();
        stream.clone_dtoh(&a_dev).unwrap()
    }

    fn gen_normal<T: ValidAsZeroBits + Clone + Default + Unpin + DeviceRepr>(
        seed: u64,
        n: usize,
        mean: T,
        std: T,
    ) -> Vec<T>
    where
        super::sys::curandGenerator_t: NormalFill<T>,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a_dev = stream.alloc_zeros::<T>(n).unwrap();
        let rng = CudaRng::new(seed, stream.clone()).unwrap();
        rng.fill_with_normal(&mut a_dev, mean, std).unwrap();
        stream.clone_dtoh(&a_dev).unwrap()
    }

    fn gen_log_normal<T: ValidAsZeroBits + Clone + Default + Unpin + DeviceRepr>(
        seed: u64,
        n: usize,
        mean: T,
        std: T,
    ) -> Vec<T>
    where
        super::sys::curandGenerator_t: LogNormalFill<T>,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a_dev = stream.alloc_zeros::<T>(n).unwrap();
        let rng = CudaRng::new(seed, stream.clone()).unwrap();
        rng.fill_with_log_normal(&mut a_dev, mean, std).unwrap();
        stream.clone_dtoh(&a_dev).unwrap()
    }

    #[test]
    fn test_seed_reproducible() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
        let mut b_dev = a_dev.clone();

        let a_rng = CudaRng::new(0, stream.clone()).unwrap();
        let b_rng = CudaRng::new(0, stream.clone()).unwrap();

        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        b_rng.fill_with_uniform(&mut b_dev).unwrap();

        let a_host = stream.clone_dtoh(&a_dev).unwrap();
        let b_host = stream.clone_dtoh(&b_dev).unwrap();
        assert_eq!(a_host, b_host);
    }

    #[test]
    fn test_different_seeds_neq() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
        let mut b_dev = a_dev.clone();

        let a_rng = CudaRng::new(0, stream.clone()).unwrap();
        let b_rng = CudaRng::new(1, stream.clone()).unwrap();

        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        b_rng.fill_with_uniform(&mut b_dev).unwrap();

        let a_host = stream.clone_dtoh(&a_dev).unwrap();
        let b_host = stream.clone_dtoh(&b_dev).unwrap();
        assert_ne!(a_host, b_host);
    }

    #[test]
    fn test_set_offset() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut a_dev = stream.alloc_zeros::<f32>(10).unwrap();
        let mut a_rng = CudaRng::new(0, stream.clone()).unwrap();

        a_rng.set_seed(42).unwrap();
        a_rng.set_offset(0).unwrap();
        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        let a_host = stream.clone_dtoh(&a_dev).unwrap();

        a_rng.set_seed(42).unwrap();
        a_rng.set_offset(0).unwrap();
        a_rng.fill_with_uniform(&mut a_dev).unwrap();
        let b_host = stream.clone_dtoh(&a_dev).unwrap();

        assert_eq!(a_host, b_host);
    }

    const N: usize = 1000;

    #[test]
    fn test_uniform_f32() {
        let a = gen_uniform::<f32>(0, N);
        for i in 0..N {
            assert!(0.0 < a[i] && a[i] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_f64() {
        let a = gen_uniform::<f64>(0, N);
        for i in 0..N {
            assert!(0.0 < a[i] && a[i] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_u32() {
        let a = gen_uniform::<u32>(0, N);
        for i in 0..N {
            assert!(a[i] > 0);
        }
    }

    #[test]
    fn test_normal_f32() {
        let a = gen_normal::<f32>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_normal::<f32>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_normal_f64() {
        let a = gen_normal::<f64>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_normal::<f64>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_log_normal_f32() {
        let a = gen_log_normal::<f32>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_log_normal::<f32>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }

    #[test]
    fn test_log_normal_f64() {
        let a = gen_log_normal::<f64>(0, N, 0.0, 1.0);
        for i in 0..N {
            assert!(a[i] != 0.0);
        }

        let b = gen_log_normal::<f64>(0, N, -1.0, 2.0);
        for i in 0..N {
            assert_ne!(a[i], b[i]);
        }
    }
}

#[cfg(test)]
mod fathomdb_context_lifetime {
    use super::*;
    use crate::driver::result::context_test_seams;

    #[test]
    fn fathomdb_rng_partial_initialization_releases_generator() {
        let driver = context_test_seams::Guard::new();
        let rng = result::partial_init_test_seams::Guard::new();
        let context = context_test_seams::context();

        let error = match CudaRng::new(42, context.default_stream()) {
            Ok(_) => panic!("stream setup must return its forced error"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            result::CurandError::Native(sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR)
        );
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = rng.snapshot();
        assert_eq!((calls.creates, calls.stream_sets), (1, 1));
        assert_eq!(
            calls.destroys, 1,
            "failed stream setup must release its acquired RNG generator"
        );
    }

    fn expect_error(result: Result<CudaRng, result::CurandError>) -> result::CurandError {
        match result {
            Ok(_) => panic!("initialization must return its forced error"),
            Err(error) => error,
        }
    }

    #[test]
    fn fathomdb_rng_failed_create_has_no_generator_to_destroy() {
        let driver = context_test_seams::Guard::new();
        let rng = result::partial_init_test_seams::Guard::new();
        rng.fail_create();
        let context = context_test_seams::context();

        let error = expect_error(CudaRng::new(42, context.default_stream()));
        assert_eq!(
            error,
            result::CurandError::Native(sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED)
        );
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = rng.snapshot();
        assert_eq!(
            (
                calls.creates,
                calls.stream_sets,
                calls.seed_sets,
                calls.destroys
            ),
            (1, 0, 0, 0)
        );
    }

    #[test]
    fn fathomdb_rng_seed_failure_releases_generator() {
        let driver = context_test_seams::Guard::new();
        let rng = result::partial_init_test_seams::Guard::new();
        rng.allow_stream_setup();
        rng.fail_seed();
        let context = context_test_seams::context();

        let error = expect_error(CudaRng::new(42, context.default_stream()));
        assert_eq!(
            error,
            result::CurandError::Native(sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR)
        );
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = rng.snapshot();
        assert_eq!(
            (
                calls.creates,
                calls.stream_sets,
                calls.seed_sets,
                calls.destroys
            ),
            (1, 1, 1, 1)
        );
    }

    #[test]
    fn fathomdb_rng_completed_owner_destroys_generator_once() {
        let driver = context_test_seams::Guard::new();
        let rng = result::partial_init_test_seams::Guard::new();
        rng.allow_stream_setup();
        let context = context_test_seams::context();

        let owner = CudaRng::new(42, context.default_stream()).unwrap();
        let calls = rng.snapshot();
        assert_eq!(
            (
                calls.creates,
                calls.stream_sets,
                calls.seed_sets,
                calls.destroys
            ),
            (1, 1, 1, 0)
        );
        drop(owner);
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = rng.snapshot();
        assert_eq!(
            (
                calls.creates,
                calls.stream_sets,
                calls.seed_sets,
                calls.destroys
            ),
            (1, 1, 1, 1)
        );
    }

    #[test]
    fn fathomdb_rng_cleanup_error_does_not_unwind() {
        let driver = context_test_seams::Guard::new();
        let rng = result::partial_init_test_seams::Guard::new();
        rng.allow_stream_setup();
        rng.fail_destroy();
        let context = context_test_seams::context();
        let owner = CudaRng::new(42, context.default_stream()).unwrap();

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
        drop(context);

        let native = driver.snapshot();
        assert_eq!((native.queries, native.binds, native.releases), (2, 1, 1));
        let calls = rng.snapshot();
        assert_eq!(
            (
                calls.creates,
                calls.stream_sets,
                calls.seed_sets,
                calls.destroys
            ),
            (1, 1, 1, 1)
        );
        assert!(
            outcome.is_ok(),
            "native RNG cleanup errors must not unwind Rust owners"
        );
    }
}

#[cfg(all(test, feature = "std"))]
#[path = "fathomdb_library_admission.rs"]
mod fathomdb_library_admission;

#[cfg(all(test, feature = "std"))]
#[path = "fathomdb_library_setter.rs"]
mod fathomdb_library_setter;

#[cfg(all(test, feature = "std"))]
#[path = "fathomdb_library_drop.rs"]
mod fathomdb_library_drop;

#[cfg(all(test, feature = "std"))]
#[path = "fathomdb_library_destroy_failure.rs"]
mod fathomdb_library_destroy_failure;

#[cfg(all(test, feature = "std"))]
#[path = "fathomdb_library_use.rs"]
mod fathomdb_library_use;
