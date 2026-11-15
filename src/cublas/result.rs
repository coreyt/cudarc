use super::sys::{self};
use core::ffi::{c_int, c_longlong, c_void};
use core::mem::MaybeUninit;

/// Failure from a native library call, tracked driver admission, or owner association.
///
/// Driver errors preserve the original status. Inspect the shared lifetime's
/// cached loss record to distinguish terminal refusal from ordinary OOM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CublasError {
    /// The unchanged native cublas status.
    Native(sys::cublasStatus_t),
    /// The original driver status from admission or context binding.
    Driver(crate::driver::DriverError),
    /// Cached terminal refusal with a synthetic driver-status projection.
    /// The authoritative cause and identities are retained without native queries.
    #[cfg(feature = "std")]
    Lost(crate::driver::PrimaryContextLoss),
    /// A stream belongs to a different tracked context generation.
    ContextMismatch,
}

impl From<crate::driver::DriverError> for CublasError {
    fn from(error: crate::driver::DriverError) -> Self {
        Self::Driver(error)
    }
}

impl sys::cublasStatus_t {
    pub fn result(self) -> Result<(), CublasError> {
        match self {
            sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError::Native(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

/// Creates a handle to the cuBLAS library. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublascreate)
pub fn create_handle() -> Result<sys::cublasHandle_t, CublasError> {
    #[cfg(test)]
    if let Some(handle) = partial_init_test_seams::create_handle() {
        return handle;
    }
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cublasCreate_v2(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// Destroys a handle previously created with [create_handle()]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasdestroy)
///
/// # Safety
///
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cublasHandle_t) -> Result<(), CublasError> {
    #[cfg(test)]
    if let Some(result) = partial_init_test_seams::destroy_handle(handle) {
        return result;
    }
    sys::cublasDestroy_v2(handle).result()
}

/// Sets the stream cuBLAS will use. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream)
///
/// # Safety
///
/// `handle` and `stream` must be valid.
pub unsafe fn set_stream(
    handle: sys::cublasHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CublasError> {
    #[cfg(test)]
    if let Some(result) = partial_init_test_seams::set_stream(handle, stream) {
        return result;
    }
    sys::cublasSetStream_v2(handle, stream).result()
}

/// Single precision matrix vector multiplication. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv)
///
/// # Safety
///
/// - `a`, `x`, and `y` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemv(
    handle: sys::cublasHandle_t,
    trans: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: *const f32,
    y: *mut f32,
    incy: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy).result()
}

/// Double precision matrix vector multiplication. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv)
///
/// # Safety
///
/// - `a`, `x`, and `y` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemv(
    handle: sys::cublasHandle_t,
    trans: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: *const f64,
    y: *mut f64,
    incy: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy).result()
}

#[cfg(feature = "f16")]
/// Half precision matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn hgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const half::f16,
    a: *const half::f16,
    lda: c_int,
    b: *const half::f16,
    ldb: c_int,
    beta: *const half::f16,
    c: *mut half::f16,
    ldc: c_int,
) -> Result<(), CublasError> {
    // NOTE: for some reason cublasHgemm is only included in header files if using c++. Therefore it
    // is not included in the bindgen exports. So we manually link to the library & load the symbol here.
    static LIB: std::sync::OnceLock<libloading::Library> = std::sync::OnceLock::new();
    let lib = LIB
        .get_or_init(|| libloading::Library::new(libloading::library_filename("cublas")).unwrap());
    #[allow(non_snake_case)]
    let f: unsafe extern "C" fn(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const half::f16,
        A: *const half::f16,
        lda: c_int,
        B: *const half::f16,
        ldb: c_int,
        beta: *const half::f16,
        C: *mut half::f16,
        ldc: c_int,
    ) -> sys::cublasStatus_t = lib.get(b"cublasHgemm\0").map(|sym| *sym).unwrap();
    f(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

/// Single precision matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemm_v2(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

/// Double precision matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemm_v2(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

#[cfg(feature = "f16")]
/// Half precision batched matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn hgemm_strided_batched(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const half::f16,
    a: *const half::f16,
    lda: c_int,
    stride_a: c_longlong,
    b: *const half::f16,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const half::f16,
    c: *mut half::f16,
    ldc: c_int,
    stride_c: c_longlong,
    batch_size: c_int,
) -> Result<(), CublasError> {
    // NOTE: for some reason cublasHgemm is only included in header files if using c++. Therefore it
    // is not included in the bindgen exports. So we manually link to the library & load the symbol here.
    static LIB: std::sync::OnceLock<libloading::Library> = std::sync::OnceLock::new();
    let lib = LIB
        .get_or_init(|| libloading::Library::new(libloading::library_filename("cublas")).unwrap());
    #[allow(non_snake_case)]
    let f: unsafe extern "C" fn(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const half::f16,
        A: *const half::f16,
        lda: c_int,
        strideA: c_longlong,
        B: *const half::f16,
        ldb: c_int,
        strideB: c_longlong,
        beta: *const half::f16,
        C: *mut half::f16,
        ldc: c_int,
        strideC: c_longlong,
        batchCount: c_int,
    ) -> sys::cublasStatus_t = lib
        .get(b"cublasHgemmStridedBatched\0")
        .map(|sym| *sym)
        .unwrap();
    f(
        handle, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size,
    )
    .result()
}

/// Single precision batched matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_strided_batched(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    stride_a: c_longlong,
    b: *const f32,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
    stride_c: c_longlong,
    batch_size: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size,
    )
    .result()
}

/// Double precision batched matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemm_strided_batched(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    stride_a: c_longlong,
    b: *const f64,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
    stride_c: c_longlong,
    batch_size: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size,
    )
    .result()
}

/// Matmul with data types specified as parameters. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_ex(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: sys::cudaDataType,
    lda: c_int,
    b: *const c_void,
    b_type: sys::cudaDataType,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    c_type: sys::cudaDataType,
    ldc: c_int,
    compute_type: sys::cublasComputeType_t,
    algo: sys::cublasGemmAlgo_t,
) -> Result<(), CublasError> {
    sys::cublasGemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c,
        c_type,
        ldc,
        compute_type,
        algo,
    )
    .result()
}

/// Strided batched matmul with data types specified as parameters. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided_batched_ex(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: sys::cudaDataType,
    lda: c_int,
    stride_a: c_longlong,
    b: *const c_void,
    b_type: sys::cudaDataType,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const c_void,
    c: *mut c_void,
    c_type: sys::cudaDataType,
    ldc: c_int,
    stride_c: c_longlong,
    batch_count: c_int,
    compute_type: sys::cublasComputeType_t,
    algo: sys::cublasGemmAlgo_t,
) -> Result<(), CublasError> {
    sys::cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        stride_a,
        b,
        b_type,
        ldb,
        stride_b,
        beta,
        c,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    )
    .result()
}

/// Single precision sum of abosolute values. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-asum)
///
/// # Safety
///
/// - `x` must be valid device pointers that have not been freed.
/// - `result` can be a pointer to host memory, but must be not null.
/// - `n * incx` must be less than the size of `x`.
pub unsafe fn sasum(
    handle: sys::cublasHandle_t,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut f32,
) -> Result<(), CublasError> {
    sys::cublasSasum_v2(handle, n, x, incx, result).result()
}

/// Double precision sum of abosolute values. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-asum)
///
/// # Safety
///
/// - `x` must be valid device pointers that have not been freed.
/// - `result` can be a pointer to host memory, but must be not null.
/// - `n * incx` must be less than the size of `x`.
pub unsafe fn dasum(
    handle: sys::cublasHandle_t,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut f64,
) -> Result<(), CublasError> {
    sys::cublasDasum_v2(handle, n, x, incx, result).result()
}

#[cfg(test)]
pub(crate) mod partial_init_test_seams {
    use super::{sys, CublasError};
    use core::cell::Cell;

    const HANDLE: usize = 0xBA5;

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct State {
        pub(crate) creates: usize,
        pub(crate) stream_sets: usize,
        pub(crate) destroys: usize,
        create_error: Option<CublasError>,
        stream_error: Option<CublasError>,
        destroy_error: Option<CublasError>,
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
                create_error: None,
                stream_error: Some(CublasError::Native(
                    sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
                )),
                destroy_error: None,
            }));
            Self
        }

        pub(crate) fn snapshot(&self) -> State {
            STATE.get().unwrap()
        }

        pub(crate) fn fail_create(&self) {
            let mut state = self.snapshot();
            state.create_error = Some(CublasError::Native(
                sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED,
            ));
            STATE.set(Some(state));
        }

        pub(crate) fn fail_stream_setup(&self) {
            let mut state = self.snapshot();
            state.stream_error = Some(CublasError::Native(
                sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
            ));
            STATE.set(Some(state));
        }

        pub(crate) fn allow_stream_setup(&self) {
            let mut state = self.snapshot();
            state.stream_error = None;
            STATE.set(Some(state));
        }

        pub(crate) fn set_destroy_error(&self, error: Option<CublasError>) {
            let mut state = self.snapshot();
            state.destroy_error = error;
            STATE.set(Some(state));
        }

        pub(crate) fn fail_destroy(&self) {
            let mut state = self.snapshot();
            state.destroy_error = Some(CublasError::Native(
                sys::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
            ));
            STATE.set(Some(state));
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            STATE.set(None);
        }
    }

    pub(crate) fn create_handle() -> Option<Result<sys::cublasHandle_t, CublasError>> {
        let mut state = STATE.get()?;
        state.creates += 1;
        STATE.set(Some(state));
        Some(match state.create_error {
            Some(error) => Err(error),
            None => Ok(HANDLE as sys::cublasHandle_t),
        })
    }

    pub(crate) fn set_stream(
        handle: sys::cublasHandle_t,
        stream: sys::cudaStream_t,
    ) -> Option<Result<(), CublasError>> {
        let mut state = STATE.get()?;
        assert_eq!(handle as usize, HANDLE);
        assert!(stream.is_null());
        state.stream_sets += 1;
        STATE.set(Some(state));
        Some(match state.stream_error {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }

    pub(crate) fn destroy_handle(handle: sys::cublasHandle_t) -> Option<Result<(), CublasError>> {
        let mut state = STATE.get()?;
        assert_eq!(handle as usize, HANDLE);
        state.destroys += 1;
        STATE.set(Some(state));
        Some(match state.destroy_error {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }
}
