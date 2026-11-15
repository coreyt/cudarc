//! A thin wrapper around [sys] providing [Result]s with [CurandError].
//!
//! Two flavors of generation:
//! 1. Not generic: See [generate] for non-generic generation functions.
//! 2. Generic: See [UniformFill], [NormalFill], and [LogNormalFill] for generic generation functions.

use super::sys::{self};
use std::mem::MaybeUninit;

/// Failure from a native library call, tracked driver admission, or owner association.
///
/// Driver errors preserve the original status. Inspect the shared lifetime's
/// cached loss record to distinguish terminal refusal from ordinary OOM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CurandError {
    /// The unchanged native curand status.
    Native(sys::curandStatus_t),
    /// The original driver status from admission or context binding.
    Driver(crate::driver::DriverError),
    /// Cached terminal refusal with a synthetic driver-status projection.
    /// The authoritative cause and identities are retained without native queries.
    #[cfg(feature = "std")]
    Lost(crate::driver::PrimaryContextLoss),
    /// A stream belongs to a different tracked context generation.
    ContextMismatch,
}

impl From<crate::driver::DriverError> for CurandError {
    fn from(error: crate::driver::DriverError) -> Self {
        Self::Driver(error)
    }
}

impl sys::curandStatus_t {
    /// Transforms into a [Result] of [CurandError]
    pub fn result(self) -> Result<(), CurandError> {
        match self {
            sys::curandStatus_t::CURAND_STATUS_SUCCESS => Ok(()),
            _ => Err(CurandError::Native(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CurandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CurandError {}

/// Create new random number generator with the default pseudo rng type.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd).
pub fn create_generator() -> Result<sys::curandGenerator_t, CurandError> {
    create_generator_kind(sys::curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT)
}

/// Create new random number generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd).
pub fn create_generator_kind(
    kind: sys::curandRngType_t,
) -> Result<sys::curandGenerator_t, CurandError> {
    #[cfg(test)]
    if let Some(generator) = partial_init_test_seams::create_generator(kind) {
        return generator;
    }
    let mut generator = MaybeUninit::uninit();
    unsafe {
        sys::curandCreateGenerator(generator.as_mut_ptr(), kind).result()?;
        Ok(generator.assume_init())
    }
}

/// Set the seed value of the pseudo-random number generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gbcd2982aa3d53571b8ad12d8188b139b)
///
/// # Safety
/// The generator must be allocated and not already freed.
pub unsafe fn set_seed(generator: sys::curandGenerator_t, seed: u64) -> Result<(), CurandError> {
    #[cfg(test)]
    if let Some(result) = partial_init_test_seams::set_seed(generator, seed) {
        return result;
    }
    sys::curandSetPseudoRandomGeneratorSeed(generator, seed).result()
}

/// Set the offset value of the pseudo-random number generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb21ba987f85486e552797206451b0939)
///
/// # Safety
/// The generator must be allocated and not already freed.
pub unsafe fn set_offset(
    generator: sys::curandGenerator_t,
    offset: u64,
) -> Result<(), CurandError> {
    #[cfg(test)]
    if let Some(result) = partial_init_test_seams::offset() {
        return result;
    }
    sys::curandSetGeneratorOffset(generator, offset).result()
}

/// Set the current stream for CURAND kernel launches.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gc78c8d07c7acea4242e2a62bc41ff1f5)
///
/// # Safety
/// 1. The generator must be allocated and not already freed.
/// 2. The stream must be allocated and not already freed.
pub unsafe fn set_stream(
    generator: sys::curandGenerator_t,
    stream: sys::cudaStream_t,
) -> Result<(), CurandError> {
    #[cfg(test)]
    if let Some(result) = partial_init_test_seams::set_stream(generator, stream) {
        return result;
    }
    sys::curandSetStream(generator, stream).result()
}

/// Destroy an existing generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g8d82c56e2b869fef4f9929a775ee18d0).
///
/// # Safety
/// The generator must not have already been freed.
pub unsafe fn destroy_generator(generator: sys::curandGenerator_t) -> Result<(), CurandError> {
    #[cfg(test)]
    if let Some(result) = partial_init_test_seams::destroy_generator(generator) {
        return result;
    }
    sys::curandDestroyGenerator(generator).result()
}

pub mod generate {
    //! Functions to generate different distributions.

    use super::{
        sys::{self},
        CurandError,
    };

    /// Fills `out` with `num` f32 values in the range (0.0, 1.0].
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g5df92a7293dc6b2e61ea481a2069ebc2)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_f32(
        gen: sys::curandGenerator_t,
        out: *mut f32,
        num: usize,
    ) -> Result<(), CurandError> {
        #[cfg(test)]
        if let Some(result) = super::partial_init_test_seams::fill() {
            return result;
        }
        sys::curandGenerateUniform(gen, out, num).result()
    }

    /// Fills `out` with `num` f64 values in the range (0.0, 1.0].
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gbb08f0268f05c9d87eac2b4a2cf7fc24)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_f64(
        gen: sys::curandGenerator_t,
        out: *mut f64,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::curandGenerateUniformDouble(gen, out, num).result()
    }

    /// Fills `out` with `num` u32 values with all bits random.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gf18b3cbdf0b7d9e2335bada92610adac)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_u32(
        gen: sys::curandGenerator_t,
        out: *mut u32,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::curandGenerate(gen, out, num).result()
    }

    /// Fills `out` with `num` u64 values with all bits random.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gd8e23d144e88b8e638139db05ff798b3)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_u64(
        gen: sys::curandGenerator_t,
        out: *mut u64,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::curandGenerateLongLong(gen, out, num).result()
    }

    /// Fills `out` with `num` f32 values from a normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb9280e447ef04e1dec4611720bd0eb69)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn normal_f32(
        gen: sys::curandGenerator_t,
        out: *mut f32,
        num: usize,
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        #[cfg(test)]
        if let Some(result) = super::partial_init_test_seams::fill() {
            return result;
        }
        sys::curandGenerateNormal(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` f64 values from a normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g046759ff9b6bf8dafc9eaae04917dc8e)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn normal_f64(
        gen: sys::curandGenerator_t,
        out: *mut f64,
        num: usize,
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        sys::curandGenerateNormalDouble(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` f32 values from a log normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g3569cc960eb1a31357752fc813e21f49)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn log_normal_f32(
        gen: sys::curandGenerator_t,
        out: *mut f32,
        num: usize,
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        #[cfg(test)]
        if let Some(result) = super::partial_init_test_seams::fill() {
            return result;
        }
        sys::curandGenerateLogNormal(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` f64 values from a normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g300c31530c8b461ca89f1e0232a6f05f)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn log_normal_f64(
        gen: sys::curandGenerator_t,
        out: *mut f64,
        num: usize,
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        sys::curandGenerateLogNormalDouble(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` u32 values from a poisson distribution
    /// parameterized by `lambda`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g425c7c13db4444e6150d159bb1417f05)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn poisson_u32(
        gen: sys::curandGenerator_t,
        out: *mut u32,
        num: usize,
        lambda: f64,
    ) -> Result<(), CurandError> {
        sys::curandGeneratePoisson(gen, out, num, lambda).result()
    }
}

/// Fill with uniform distributed numbers of type `T`.
pub trait UniformFill<T> {
    /// # Safety
    /// This inherits the unsafe from methods in [generate].
    unsafe fn fill(self, out: *mut T, num: usize) -> Result<(), CurandError>;
}

impl UniformFill<f32> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut f32, num: usize) -> Result<(), CurandError> {
        generate::uniform_f32(self, out, num)
    }
}

impl UniformFill<f64> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut f64, num: usize) -> Result<(), CurandError> {
        generate::uniform_f64(self, out, num)
    }
}

impl UniformFill<u32> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut u32, num: usize) -> Result<(), CurandError> {
        generate::uniform_u32(self, out, num)
    }
}

impl UniformFill<u64> for sys::curandGenerator_t {
    unsafe fn fill(self, out: *mut u64, num: usize) -> Result<(), CurandError> {
        generate::uniform_u64(self, out, num)
    }
}

/// Fill with normally distributed numbers of type `T`.
pub trait NormalFill<T> {
    /// # Safety
    /// This inherits the unsafe from methods in [generate].
    unsafe fn fill(self, o: *mut T, n: usize, m: T, s: T) -> Result<(), CurandError>;
}

impl NormalFill<f32> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f32, n: usize, m: f32, s: f32) -> Result<(), CurandError> {
        generate::normal_f32(self, o, n, m, s)
    }
}

impl NormalFill<f64> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f64, n: usize, m: f64, s: f64) -> Result<(), CurandError> {
        generate::normal_f64(self, o, n, m, s)
    }
}

/// Fill with log normally distributed numbers of type `T`.
pub trait LogNormalFill<T> {
    /// # Safety
    /// This inherits the unsafe from methods in [generate].
    unsafe fn fill(self, o: *mut T, n: usize, m: T, s: T) -> Result<(), CurandError>;
}

impl LogNormalFill<f32> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f32, n: usize, m: f32, s: f32) -> Result<(), CurandError> {
        generate::log_normal_f32(self, o, n, m, s)
    }
}

impl LogNormalFill<f64> for sys::curandGenerator_t {
    unsafe fn fill(self, o: *mut f64, n: usize, m: f64, s: f64) -> Result<(), CurandError> {
        generate::log_normal_f64(self, o, n, m, s)
    }
}

#[cfg(test)]
pub(crate) mod partial_init_test_seams {
    use super::{sys, CurandError};
    use core::cell::Cell;

    const GENERATOR: usize = 0xA86;

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct State {
        pub(crate) creates: usize,
        pub(crate) stream_sets: usize,
        pub(crate) destroys: usize,
        pub(crate) seed_sets: usize,
        pub(crate) offset_sets: usize,
        pub(crate) fills: usize,
        create_error: Option<CurandError>,
        stream_error: Option<CurandError>,
        seed_error: Option<CurandError>,
        destroy_error: Option<CurandError>,
        operation_error: Option<CurandError>,
    }

    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
    }

    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            STATE.set(Some(State {
                creates: 0,
                stream_sets: 0,
                destroys: 0,
                seed_sets: 0,
                offset_sets: 0,
                fills: 0,
                create_error: None,
                stream_error: Some(CurandError::Native(
                    sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR,
                )),
                seed_error: None,
                destroy_error: None,
                operation_error: None,
            }));
            Self
        }

        pub(crate) fn fail_operation(&self, fail: bool) {
            let mut state = self.snapshot();
            state.operation_error = fail.then_some(CurandError::Native(
                sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED,
            ));
            STATE.set(Some(state));
        }

        pub(crate) fn snapshot(&self) -> State {
            STATE.get().unwrap()
        }

        pub(crate) fn fail_create(&self) {
            let mut state = self.snapshot();
            state.create_error = Some(CurandError::Native(
                sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED,
            ));
            STATE.set(Some(state));
        }

        pub(crate) fn fail_stream_setup(&self) {
            let mut state = self.snapshot();
            state.stream_error = Some(CurandError::Native(
                sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR,
            ));
            STATE.set(Some(state));
        }

        pub(crate) fn allow_stream_setup(&self) {
            let mut state = self.snapshot();
            state.stream_error = None;
            STATE.set(Some(state));
        }

        pub(crate) fn fail_seed(&self) {
            let mut state = self.snapshot();
            state.seed_error = Some(CurandError::Native(
                sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR,
            ));
            STATE.set(Some(state));
        }

        pub(crate) fn set_destroy_error(&self, error: Option<CurandError>) {
            let mut state = self.snapshot();
            state.destroy_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn fail_destroy(&self) {
            let mut state = self.snapshot();
            state.destroy_error = Some(CurandError::Native(
                sys::curandStatus_t::CURAND_STATUS_INTERNAL_ERROR,
            ));
            STATE.set(Some(state));
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            STATE.set(None);
        }
    }

    pub(crate) fn offset() -> Option<Result<(), CurandError>> {
        let mut state = STATE.get()?;
        state.offset_sets += 1;
        STATE.set(Some(state));
        Some(state.operation_error.map_or(Ok(()), Err))
    }

    pub(crate) fn fill() -> Option<Result<(), CurandError>> {
        let mut state = STATE.get()?;
        state.fills += 1;
        STATE.set(Some(state));
        Some(state.operation_error.map_or(Ok(()), Err))
    }

    pub(crate) fn create_generator(
        kind: sys::curandRngType_t,
    ) -> Option<Result<sys::curandGenerator_t, CurandError>> {
        let mut state = STATE.get()?;
        assert_eq!(kind, sys::curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT);
        state.creates += 1;
        STATE.set(Some(state));
        Some(match state.create_error {
            Some(error) => Err(error),
            None => Ok(GENERATOR as sys::curandGenerator_t),
        })
    }

    pub(crate) fn set_stream(
        generator: sys::curandGenerator_t,
        stream: sys::cudaStream_t,
    ) -> Option<Result<(), CurandError>> {
        let mut state = STATE.get()?;
        assert_eq!(generator as usize, GENERATOR);
        assert!(stream.is_null());
        state.stream_sets += 1;
        STATE.set(Some(state));
        Some(match state.stream_error {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }

    pub(crate) fn set_seed(
        generator: sys::curandGenerator_t,
        seed: u64,
    ) -> Option<Result<(), CurandError>> {
        let mut state = STATE.get()?;
        assert_eq!(generator as usize, GENERATOR);
        assert_eq!(seed, 42);
        state.seed_sets += 1;
        STATE.set(Some(state));
        Some(match state.seed_error {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }

    pub(crate) fn destroy_generator(
        generator: sys::curandGenerator_t,
    ) -> Option<Result<(), CurandError>> {
        let mut state = STATE.get()?;
        assert_eq!(generator as usize, GENERATOR);
        state.destroys += 1;
        STATE.set(Some(state));
        Some(match state.destroy_error {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }
}
