//! Safe CUDA wrappers for:
//!
//! | library | dynamic load | dynamic link | static link |
//! | --- | --- | --- | --- |
//! | [CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) | ✅ | ✅ | ❌ |
//! | [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) | ✅ | ✅ | ✅ |
//! | [cuRAND](https://docs.nvidia.com/cuda/curand/index.html) | ✅ | ✅ | ✅ |
//! | [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) | ✅ | ✅ | ✅ |
//! | [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api) | ✅ | ✅ | ✅ |
//! | [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) | ✅ | ✅ | ✅ |
//! | [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html) | ✅ | ✅ | ✅ |
//! | [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | ✅ | ✅ | ✅ |
//! | [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/) | ✅ | ✅ | ❌ |
//! | [cuFILE](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#introduction) | ✅ | ✅ | ✅ |
//! | [CUPTI](https://docs.nvidia.com/cupti/) | ✅ | ✅ | ✅ |
//! | [nvtx](https://nvidia.github.io/NVTX/) | ✅ | ✅ | ❌ |
//! | [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/index.html) | ✅ | ✅ | ❌ |
//!
//! CUDA Versions supported
//! - 11.4-11.8
//! - 12.0-12.9
//! - 13.0-13.4
//!
//! CUDNN versions supported:
//! - 9.12.0
//!
//! NCCL versions supported:
//! - 2.28.3
//!
//! # Configuring CUDA version
//!
//! Select cuda version with one of:
//! - `-F cuda-version-from-build-system`: At build time will get the cuda toolkit version using `nvcc`
//!     - `-F fallback-latest`: can be used to control behavior if this fails. default is not enabled, which will cause the build
//!       script to panic. if `-F fallback-latest` is enabled, we will use the highest bindings we have.
//! - `-F cuda-<major>0<minor>0` to build for a specific version of cuda
//!
//! # Configuring linking
//!
//! By default we use `-F dynamic-loading`, which will not require any libraries to be present at build time.
//!
//! You can also enable `-F dynamic-linking` or `-F static-linking` for your use case.
//!
//! # Getting started
//!
//! **See [driver] for more examples**
//!
//! At the core is the [driver] API, which exposes a bunch of structs, but the main ones are:
//!
//! 1. [`driver::CudaContext`] is a handle to a specific device ordinal (e.g. 0, 1, 2, ...)
//! 2. [`driver::CudaStream`] is how you submit work to a device
//! 3. [`driver::CudaSlice<T>`], which represents a [`Vec<T>`] on the device, can be allocated
//!    using the aforementioned [`driver::CudaStream`].
//!
//! Here is a table of similar concepts between CPU and Cuda:
//!
//! | Concept | CPU | Cuda |
//! | --- | --- | --- |
//! | Memory allocator | [`std::alloc::GlobalAlloc`] | [`driver::CudaContext`] |
//! | List of values on heap | [`Vec<T>`] | [`driver::CudaSlice<T>`] |
//! | Slice | `&[T]` | [`driver::CudaView<T>`] |
//! | Mutable Slice  | `&mut [T]` | [`driver::CudaViewMut<T>`] |
//! | Function | [`Fn`] | [`driver::CudaFunction`] |
//! | Calling a function | `my_function(a, b, c)` | [`driver::LaunchArgs::launch()`] |
//! | Thread | [`std::thread::Thread`] | [`driver::CudaStream`] |
//!
//! # Combining the different APIs
//!
//! All the highest level apis have been designed to work together.
//!
//! ## nvrtc
//!
//! [`nvrtc::compile_ptx()`] outputs a [`nvrtc::Ptx`], which can
//! be loaded into a device with [`driver::CudaContext::load_module()`].
//!
//! ## cublas
//!
//! [cublas::CudaBlas] can perform gemm operations using [`cublas::Gemm<T>`],
//! and [`cublas::Gemv<T>`]. Both of these traits can generically accept memory
//! allocated by the driver in the form of: [`driver::CudaSlice<T>`],
//! [`driver::CudaView<T>`], and [`driver::CudaViewMut<T>`].
//!
//! ## curand
//!
//! [curand::CudaRng] can fill a [`driver::CudaSlice<T>`] with random data, based on
//! one of its available distributions.
//!
//! # Combining safe/result/sys
//!
//! The result and sys levels are very inter-changeable for each API. However,
//! the safe apis don't necessarily allow you to mix in the result level. This
//! is to encourage going through the safe API when possible.
//!
//! **If you need some functionality that isn't present in the safe api, please
//! open a ticket.**
//!
//! # crate organization
//!
//! Each of the modules for the above is organized into three levels:
//! 1. A `safe` module which provides safe abstractions over the `result` module
//! 2. A `result` which is a thin wrapper around the `sys` module to ensure all functions return [Result]
//! 3. A `sys` module which contains the raw FFI bindings
//!
//! | API | Safe | Result | Sys |
//! | --- | --- | --- | --- |
//! | driver | [driver::safe] | [driver::result] | [driver::sys] |
//! | cublas | [cublas::safe] | [cublas::result] | [cublas::sys] |
//! | cublaslt | [cublaslt::safe] | [cublaslt::result] | [cublaslt::sys] |
//! | nvrtc | [nvrtc::safe] | [nvrtc::result] | [nvrtc::sys] |
//! | curand | [curand::safe] | [curand::result] | [curand::sys] |
//! | cudnn | [cudnn::safe] | [cudnn::result] | [cudnn::sys] |
//! | cufft | [cufft::safe] | [cufft::result] | [cufft::sys] |
//! | cusparse | - | [cusparse::result] | [cusparse::sys] |
//! | cusolver | [cusolver::safe] | [cusolver::result] | [cusolver::sys] |
//! | cusolvermg | [cusolvermg::safe] | [cusolvermg::result] | [cusolvermg::sys] |
//! | cupti | - | [cupti::result] | [cupti::sys] |
//! | nvtx | [nvtx::safe] | [nvtx::result] | [nvtx::sys] |
//! | cutensor | [cutensor::safe] | [cutensor::result] | [cutensor::sys] |

#![cfg_attr(feature = "no-std", no_std)]

#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;

#[cfg(feature = "cublas")]
pub mod cublas;
#[cfg(feature = "cublaslt")]
pub mod cublaslt;
#[cfg(any(
    feature = "cudnn-08009",
    feature = "cudnn-09010",
    feature = "cudnn-09021",
))]
pub mod cudnn;
#[cfg(all(
    feature = "cufft",
    not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
    ))
))]
pub mod cufft;
#[cfg(feature = "cufile")]
pub mod cufile;
#[cfg(feature = "cupti")]
pub mod cupti;
#[cfg(feature = "curand")]
pub mod curand;
#[cfg(feature = "cusolver")]
pub mod cusolver;
#[cfg(feature = "cusolvermg")]
pub mod cusolvermg;
#[cfg(feature = "cusparse")]
pub mod cusparse;
#[cfg(all(
    any(
        feature = "cutensor-02003",
        feature = "cutensor-02004",
        feature = "cutensor-02005",
        feature = "cutensor-02006",
    ),
    not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
    ))
))]
pub mod cutensor;
#[cfg(feature = "driver")]
pub mod driver;
#[cfg(any(
    feature = "nccl-02022",
    feature = "nccl-02024",
    feature = "nccl-02025",
    feature = "nccl-02026",
    feature = "nccl-02027",
    feature = "nccl-02028",
    feature = "nccl-02029",
    feature = "nccl-02030",
))]
pub mod nccl;
#[cfg(feature = "nvrtc")]
pub mod nvrtc;
#[cfg(feature = "nvtx")]
pub mod nvtx;
#[cfg(feature = "runtime")]
pub mod runtime;

pub mod types;

#[cfg(feature = "dynamic-loading")]
pub(crate) fn panic_no_lib_found<S: std::fmt::Debug>(lib_name: &str, choices: &[S]) -> ! {
    panic!("Unable to dynamically load the \"{lib_name}\" shared library - searched for library names: {choices:?}. Ensure that `LD_LIBRARY_PATH` has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.");
}

#[cfg(feature = "dynamic-loading")]
pub fn get_lib_name_candidates(lib_name: &str) -> std::vec::Vec<std::string::String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let pointer_width = if cfg!(target_pointer_width = "32") {
        "32"
    } else if cfg!(target_pointer_width = "64") {
        "64"
    } else {
        panic!("Unsupported target pointer width")
    };

    let major = env!("CUDA_MAJOR_VERSION");
    let minor = env!("CUDA_MINOR_VERSION");

    [
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{minor}{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}{minor}_0{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}0_{minor}{DLL_SUFFIX}"),
        // See issue #242
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_10{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_11{DLL_SUFFIX}"),
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_12{DLL_SUFFIX}"),
        // See issue #246
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_{major}0_0{DLL_SUFFIX}"),
        // See issue #260
        std::format!("{DLL_PREFIX}{lib_name}{pointer_width}_9{DLL_SUFFIX}"),
        // See issue #274
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.{major}"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.12"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.11"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.10"),
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.9"),
        // See issue #296
        std::format!("{DLL_PREFIX}{lib_name}{DLL_SUFFIX}.1"),
    ]
    .into()
}

pub use lib_major::{LibLoadError, LibVersion, LoadedLibrary, RejectReason};

/// Built-major library loading (FATHOMDB.md item 9): candidate names, version
/// lines and the loaded-library report.
// Each loader uses the part for its own library, so a build with only some
// library features leaves the rest unused.
#[allow(dead_code)]
mod lib_major {
    use core::fmt;
    use std::string::String;
    use std::sync::OnceLock;
    use std::vec::Vec;

    /// A library version as reported by the library's own version query.
    ///
    /// CUPTI before CUDA 13 reports an API version (for example 24); it is
    /// recorded as `major` = that number, `minor` = 0 and `patch` = `None`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct LibVersion {
        pub major: u32,
        pub minor: u32,
        pub patch: Option<u32>,
    }

    impl fmt::Display for LibVersion {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.patch {
                Some(p) => write!(f, "{}.{}.{}", self.major, self.minor, p),
                None => write!(f, "{}.{}", self.major, self.minor),
            }
        }
    }

    /// Why a candidate library name was not used, or (as
    /// [`LoadedLibrary::note`]) a caveat about the library that was used.
    #[derive(Clone, Debug)]
    pub enum RejectReason {
        /// The loader found no file of that name.
        NotFound,
        /// The loader found a file but could not load it.
        LoadFailed(String),
        /// The library reports a version outside the built major's line.
        WrongVersion(LibVersion),
        /// An object of that name was already loaded in the process and
        /// reports a version outside the built major's line. Another file of
        /// the same soname cannot be opened in this process.
        AlreadyLoadedWrongVersion(LibVersion),
        /// The library pulled in an in-scope dependency of another line.
        WrongMajorDependency { soname: String, version: LibVersion },
        /// The library exports no version query.
        NoVersionQuery,
    }

    impl fmt::Display for RejectReason {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                RejectReason::NotFound => write!(f, "not found"),
                RejectReason::LoadFailed(e) => write!(f, "load failed: {e}"),
                RejectReason::WrongVersion(v) => write!(f, "wrong version {v}"),
                RejectReason::AlreadyLoadedWrongVersion(v) => {
                    write!(f, "already loaded with wrong version {v}")
                }
                RejectReason::WrongMajorDependency { soname, version } => {
                    write!(f, "dependency {soname} has wrong version {version}")
                }
                RejectReason::NoVersionQuery => write!(f, "no version query"),
            }
        }
    }

    /// A library of the built CUDA major could not be loaded.
    ///
    /// `tried` lists every candidate name in the order tried, with the reason
    /// it was not used.
    #[derive(Clone, Debug)]
    pub struct LibLoadError {
        pub library: &'static str,
        pub cuda_major: u32,
        pub tried: Vec<(String, RejectReason)>,
    }

    // FC6-b RED stub: the base panic message (no major, no reasons).
    impl fmt::Display for LibLoadError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let names: Vec<&str> = self.tried.iter().map(|(n, _)| n.as_str()).collect();
            write!(
                f,
                "Unable to dynamically load the \"{}\" shared library - searched for library names: {names:?}.",
                self.library
            )
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for LibLoadError {}

    /// What was loaded for one library, and how.
    #[derive(Clone, Debug)]
    pub struct LoadedLibrary {
        pub library: &'static str,
        /// The built major this load was checked against.
        pub cuda_major: u32,
        /// The candidate name that was opened, or "linked at build time".
        pub opened_as: String,
        /// The canonical path of the loaded file, comparable with
        /// `/proc/self/maps`.
        pub path: Option<String>,
        /// `None` only when the library has no version query.
        pub version: Option<LibVersion>,
        /// For example [`RejectReason::NoVersionQuery`] for a cuFile without
        /// `cuFileGetVersion`.
        pub note: Option<RejectReason>,
    }

    // FC6-b RED stub.
    impl fmt::Display for LoadedLibrary {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}: unknown", self.library)
        }
    }

    // FC6-b RED stub.
    pub(crate) fn linked_report(library: &'static str, cuda_major: u32) -> LoadedLibrary {
        LoadedLibrary {
            library,
            cuda_major,
            opened_as: String::new(),
            path: None,
            version: None,
            note: None,
        }
    }

    /// The target a candidate list is generated for.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum TargetOs {
        Linux,
        Windows,
        Other {
            dll_prefix: &'static str,
            dll_suffix: &'static str,
        },
    }

    impl TargetOs {
        pub(crate) fn host() -> Self {
            if cfg!(target_os = "linux") {
                TargetOs::Linux
            } else if cfg!(target_os = "windows") {
                TargetOs::Windows
            } else {
                TargetOs::Other {
                    dll_prefix: std::env::consts::DLL_PREFIX,
                    dll_suffix: std::env::consts::DLL_SUFFIX,
                }
            }
        }

        fn affixes(self) -> (&'static str, &'static str) {
            match self {
                TargetOs::Linux => ("lib", ".so"),
                TargetOs::Windows => ("", ".dll"),
                TargetOs::Other {
                    dll_prefix,
                    dll_suffix,
                } => (dll_prefix, dll_suffix),
            }
        }
    }

    /// The candidate list of `ec1cac71`, with the target and version as
    /// parameters.
    pub(crate) fn base_candidates(
        os: TargetOs,
        pointer_width: u32,
        lib_name: &str,
        major: u32,
        minor: u32,
    ) -> Vec<String> {
        let (pre, suf) = os.affixes();
        let w = pointer_width;
        [
            std::format!("{pre}{lib_name}{suf}"),
            std::format!("{pre}{lib_name}{w}{suf}"),
            std::format!("{pre}{lib_name}{w}_{major}{suf}"),
            std::format!("{pre}{lib_name}{w}_{major}{minor}{suf}"),
            std::format!("{pre}{lib_name}{w}_{major}{minor}_0{suf}"),
            std::format!("{pre}{lib_name}{w}_{major}0_{minor}{suf}"),
            std::format!("{pre}{lib_name}{w}_10{suf}"),
            std::format!("{pre}{lib_name}{w}_11{suf}"),
            std::format!("{pre}{lib_name}{w}_12{suf}"),
            std::format!("{pre}{lib_name}{w}_{major}0_0{suf}"),
            std::format!("{pre}{lib_name}{w}_9{suf}"),
            std::format!("{pre}{lib_name}{suf}.{major}"),
            std::format!("{pre}{lib_name}{suf}.12"),
            std::format!("{pre}{lib_name}{suf}.11"),
            std::format!("{pre}{lib_name}{suf}.10"),
            std::format!("{pre}{lib_name}{suf}.9"),
            std::format!("{pre}{lib_name}{suf}.1"),
        ]
        .into()
    }

    // FC6-b RED stub: the base list.
    pub(crate) fn lib_candidates(
        os: TargetOs,
        pointer_width: u32,
        lib_name: &str,
        major: u32,
        minor: u32,
    ) -> Vec<String> {
        base_candidates(os, pointer_width, lib_name, major, minor)
    }

    /// The raw answer of a library's version query.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Observed {
        /// One integer (`cudaRuntimeGetVersion`, `cuptiGetVersion`,
        /// `curandGetVersion`, `cuFileGetVersion`).
        Int(i64),
        /// `*GetProperty` for major, minor and patch level.
        Triple(u32, u32, u32),
        /// `nvrtcVersion`.
        Pair(u32, u32),
        /// The library exports no version query.
        Absent,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Verdict {
        Accept,
        /// Accepted without a version query: only by a versioned name.
        AcceptWithoutQuery,
        Reject,
    }

    // FC6-b RED stub: accepts everything.
    pub(crate) fn version_verdict(_lib: &str, _major: u32, _observed: Observed) -> Verdict {
        Verdict::Accept
    }

    /// The memoized result of loading one library.
    pub(crate) struct LibCell<L> {
        cell: OnceLock<Result<(L, LoadedLibrary), LibLoadError>>,
    }

    // FC6-b RED stub: upstream's behaviour (a failure is retried, and the
    // presence probe loads independently).
    impl<L> LibCell<L> {
        pub(crate) const fn new() -> Self {
            Self {
                cell: OnceLock::new(),
            }
        }

        pub(crate) fn try_get(
            &self,
            load: impl FnOnce() -> Result<(L, LoadedLibrary), LibLoadError>,
        ) -> Result<&L, LibLoadError> {
            if let Some(Ok((l, _))) = self.cell.get() {
                return Ok(l);
            }
            match load() {
                Ok(v) => {
                    let _ = self.cell.set(Ok(v));
                    match self.cell.get() {
                        Some(Ok((l, _))) => Ok(l),
                        _ => unreachable!(),
                    }
                }
                Err(e) => Err(e),
            }
        }

        pub(crate) fn is_present(
            &self,
            load: impl FnOnce() -> Result<(L, LoadedLibrary), LibLoadError>,
        ) -> bool {
            load().is_ok()
        }

        pub(crate) fn report(&self) -> Option<LoadedLibrary> {
            match self.cell.get() {
                Some(Ok((_, r))) => Some(r.clone()),
                _ => None,
            }
        }
    }
}

/// T-UNIT of test plan § 3.4 (requirements CF-AC-11). Pure: every target is
/// a parameter, so the Windows names are checked on Linux too.
#[cfg(test)]
mod fathomdb_lib_major {
    use crate::lib_major::{
        lib_candidates, linked_report, version_verdict, LibCell, Observed, TargetOs, Verdict,
    };
    use crate::{LibLoadError, LibVersion, LoadedLibrary, RejectReason};
    use std::string::{String, ToString};
    use std::vec::Vec;

    const IN_SCOPE: [&str; 11] = [
        "cudart",
        "cublas",
        "cublasLt",
        "nvrtc",
        "cupti",
        "cufft",
        "cusolver",
        "cusolverMg",
        "cusparse",
        "curand",
        "cufile",
    ];
    const OUT_OF_SCOPE: [&str; 6] = ["cuda", "nvcuda", "cudnn", "nccl", "cutensor", "nvToolsExt"];
    const VERSIONS: [(u32, u32); 19] = [
        (11, 4),
        (11, 5),
        (11, 6),
        (11, 7),
        (11, 8),
        (12, 0),
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (12, 5),
        (12, 6),
        (12, 8),
        (12, 9),
        (13, 0),
        (13, 1),
        (13, 2),
        (13, 3),
        (13, 4),
    ];
    const OSES: [TargetOs; 2] = [TargetOs::Linux, TargetOs::Windows];

    /// The `ec1cac71` `get_lib_name_candidates`, with `env!` and the host
    /// constants replaced by parameters. This copy is the oracle; it is not
    /// shared with the code under test.
    fn ec1cac71_candidates(
        os: TargetOs,
        pointer_width: u32,
        lib_name: &str,
        major: u32,
        minor: u32,
    ) -> Vec<String> {
        let (dll_prefix, dll_suffix) = match os {
            TargetOs::Linux => ("lib", ".so"),
            TargetOs::Windows => ("", ".dll"),
            TargetOs::Other {
                dll_prefix,
                dll_suffix,
            } => (dll_prefix, dll_suffix),
        };
        let pointer_width = pointer_width.to_string();
        [
            std::format!("{dll_prefix}{lib_name}{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_{major}{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_{major}{minor}{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_{major}{minor}_0{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_{major}0_{minor}{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_10{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_11{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_12{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_{major}0_0{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{pointer_width}_9{dll_suffix}"),
            std::format!("{dll_prefix}{lib_name}{dll_suffix}.{major}"),
            std::format!("{dll_prefix}{lib_name}{dll_suffix}.12"),
            std::format!("{dll_prefix}{lib_name}{dll_suffix}.11"),
            std::format!("{dll_prefix}{lib_name}{dll_suffix}.10"),
            std::format!("{dll_prefix}{lib_name}{dll_suffix}.9"),
            std::format!("{dll_prefix}{lib_name}{dll_suffix}.1"),
        ]
        .into()
    }

    /// Design § 12.4's versioned names for library `lib` and built major
    /// `major` (12 or 13), in order. Windows CUPTI is exempt (`None`).
    fn design_versioned_names(os: TargetOs, lib: &str, major: u32) -> Option<Vec<String>> {
        let linux: &[&str] = match (lib, major) {
            ("cudart", 12) => &["libcudart.so.12"],
            ("cudart", 13) => &["libcudart.so.13"],
            ("cublas", 12) => &["libcublas.so.12"],
            ("cublas", 13) => &["libcublas.so.13"],
            ("cublasLt", 12) => &["libcublasLt.so.12"],
            ("cublasLt", 13) => &["libcublasLt.so.13"],
            ("nvrtc", 12) => &["libnvrtc.so.12"],
            ("nvrtc", 13) => &["libnvrtc.so.13"],
            ("cupti", 12) => &["libcupti.so.12"],
            ("cupti", 13) => &["libcupti.so.13"],
            ("cufft", 12) => &["libcufft.so.11"],
            ("cufft", 13) => &["libcufft.so.12"],
            ("cusolver", 12) => &["libcusolver.so.11"],
            ("cusolver", 13) => &["libcusolver.so.12"],
            ("cusolverMg", 12) => &["libcusolverMg.so.11"],
            ("cusolverMg", 13) => &["libcusolverMg.so.12"],
            ("cusparse", _) => &["libcusparse.so.12"],
            ("curand", 12) => &["libcurand.so.10"],
            ("curand", 13) => &["libcurand.so.10", "libcurand.so.11"],
            ("cufile", _) => &["libcufile.so.0"],
            _ => panic!("no design entry for {lib} {major}"),
        };
        let windows: &[&str] = match (lib, major) {
            ("cudart", 12) => &["cudart64_12.dll"],
            ("cudart", 13) => &["cudart64_13.dll"],
            ("cublas", 12) => &["cublas64_12.dll"],
            ("cublas", 13) => &["cublas64_13.dll"],
            ("cublasLt", 12) => &["cublasLt64_12.dll"],
            ("cublasLt", 13) => &["cublasLt64_13.dll"],
            ("nvrtc", 12) => &["nvrtc64_120_0.dll"],
            ("nvrtc", 13) => &["nvrtc64_130_0.dll"],
            ("cupti", _) => &[],
            ("cufft", 12) => &["cufft64_11.dll"],
            ("cufft", 13) => &["cufft64_12.dll"],
            ("cusolver", 12) => &["cusolver64_11.dll"],
            ("cusolver", 13) => &["cusolver64_12.dll"],
            ("cusolverMg", 12) => &["cusolverMg64_11.dll"],
            ("cusolverMg", 13) => &["cusolverMg64_12.dll"],
            ("cusparse", _) => &["cusparse64_12.dll"],
            ("curand", 12) => &["curand64_10.dll"],
            ("curand", 13) => &["curand64_10.dll", "curand64_11.dll"],
            ("cufile", _) => &[],
            _ => panic!("no design entry for {lib} {major}"),
        };
        let names = match os {
            TargetOs::Linux => linux,
            TargetOs::Windows if lib == "cupti" => return None,
            TargetOs::Windows => windows,
            TargetOs::Other { .. } => panic!("design § 12.4 names Linux and Windows only"),
        };
        Some(names.iter().map(|s| s.to_string()).collect())
    }

    fn unversioned_name(os: TargetOs, lib: &str) -> String {
        match os {
            TargetOs::Linux => std::format!("lib{lib}.so"),
            TargetOs::Windows => std::format!("{lib}64.dll"),
            TargetOs::Other { .. } => unreachable!(),
        }
    }

    fn minors(major: u32) -> impl Iterator<Item = u32> {
        VERSIONS
            .iter()
            .filter(move |(m, _)| *m == major)
            .map(|(_, n)| *n)
    }

    fn built() -> (u32, u32) {
        (
            env!("CUDA_MAJOR_VERSION").parse().unwrap(),
            env!("CUDA_MINOR_VERSION").parse().unwrap(),
        )
    }

    /// Every (os, lib, major, candidates) in scope of rules 1 and 2: the pure
    /// function for both majors and both OSes, plus `get_lib_name_candidates`
    /// when this build is 12.x or 13.x.
    fn in_scope_lists() -> Vec<(TargetOs, &'static str, u32, Vec<String>)> {
        let mut out = Vec::new();
        for os in OSES {
            for major in [12, 13] {
                for lib in IN_SCOPE {
                    if os == TargetOs::Windows && lib == "cupti" {
                        continue;
                    }
                    for minor in minors(major) {
                        out.push((os, lib, major, lib_candidates(os, 64, lib, major, minor)));
                    }
                }
            }
        }
        #[cfg(feature = "dynamic-loading")]
        {
            let (major, _) = built();
            let host = TargetOs::host();
            if (major == 12 || major == 13) && matches!(host, TargetOs::Linux | TargetOs::Windows) {
                for lib in IN_SCOPE {
                    if host == TargetOs::Windows && lib == "cupti" {
                        continue;
                    }
                    out.push((host, lib, major, crate::get_lib_name_candidates(lib)));
                }
            }
        }
        out
    }

    #[test]
    fn in_scope_candidates_never_name_another_major() {
        let mut bad = Vec::new();
        for (os, lib, major, cands) in in_scope_lists() {
            let versioned = design_versioned_names(os, lib, major).unwrap();
            let unversioned = unversioned_name(os, lib);
            for c in &cands {
                if !versioned.contains(c) && *c != unversioned {
                    bad.push(std::format!("{os:?} {lib} CUDA {major}: {c}"));
                }
            }
        }
        bad.dedup();
        for b in &bad {
            std::eprintln!("other-major or unknown name: {b}");
        }
        assert!(
            bad.is_empty(),
            "{} candidate names that are not the built major's versioned name or the unversioned name, e.g. {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
    }

    #[test]
    fn versioned_names_come_before_unversioned_names() {
        let mut bad = Vec::new();
        for (os, lib, major, cands) in in_scope_lists() {
            let versioned = design_versioned_names(os, lib, major).unwrap();
            let unversioned = unversioned_name(os, lib);
            let prefix: Vec<String> = cands.iter().take(versioned.len()).cloned().collect();
            if prefix != versioned {
                bad.push(std::format!(
                    "{os:?} {lib} CUDA {major}: starts with {prefix:?}, expected {versioned:?}"
                ));
            }
            if let Some(u) = cands.iter().position(|c| *c == unversioned) {
                if u < versioned.len() {
                    bad.push(std::format!(
                        "{os:?} {lib} CUDA {major}: {unversioned} at position {u}"
                    ));
                }
            }
        }
        bad.dedup();
        assert!(
            bad.is_empty(),
            "{} lists out of order, e.g. {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
    }

    #[test]
    fn out_of_scope_and_cuda11_candidates_equal_the_base() {
        let oses = [
            TargetOs::Linux,
            TargetOs::Windows,
            TargetOs::Other {
                dll_prefix: "lib",
                dll_suffix: ".dylib",
            },
        ];
        let mut checked = 0;
        for os in oses {
            for width in [32, 64] {
                for (major, minor) in VERSIONS {
                    for lib in IN_SCOPE.iter().chain(OUT_OF_SCOPE.iter()) {
                        let out_of_scope = OUT_OF_SCOPE.contains(lib)
                            || major == 11
                            || !matches!(os, TargetOs::Linux | TargetOs::Windows)
                            || (os == TargetOs::Windows && *lib == "cupti");
                        if !out_of_scope {
                            continue;
                        }
                        assert_eq!(
                            lib_candidates(os, width, lib, major, minor),
                            ec1cac71_candidates(os, width, lib, major, minor),
                            "{os:?} {width} {lib} {major}.{minor}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);
        #[cfg(feature = "dynamic-loading")]
        {
            let (major, minor) = built();
            let width = if cfg!(target_pointer_width = "32") {
                32
            } else {
                64
            };
            for lib in IN_SCOPE.iter().chain(OUT_OF_SCOPE.iter()) {
                let host = TargetOs::host();
                if OUT_OF_SCOPE.contains(lib)
                    || major == 11
                    || !matches!(host, TargetOs::Linux | TargetOs::Windows)
                    || (host == TargetOs::Windows && *lib == "cupti")
                {
                    assert_eq!(
                        crate::get_lib_name_candidates(lib),
                        ec1cac71_candidates(host, width, lib, major, minor),
                        "get_lib_name_candidates({lib})"
                    );
                }
            }
        }
    }

    #[test]
    fn version_lines_accept_their_major_and_reject_neighbours() {
        use Observed::{Absent, Int, Pair, Triple};
        // (library, observed, the line it belongs to: 11, 12, 13 or 14)
        let samples: &[(&str, Observed, u32)] = &[
            // design § 12.4 measured samples (12.6 apt / 12.8 wheel / cu13 wheel)
            ("cudart", Int(12060), 12),
            ("cudart", Int(12080), 12),
            ("cudart", Int(13000), 13),
            ("cublas", Triple(12, 6, 1), 12),
            ("cublas", Triple(12, 8, 4), 12),
            ("cublas", Triple(13, 1, 1), 13),
            ("cublasLt", Triple(12, 6, 1), 12),
            ("cublasLt", Triple(13, 0, 2), 13),
            ("curand", Int(10307), 12),
            ("curand", Int(10309), 12),
            ("curand", Int(10400), 13),
            ("cusparse", Triple(12, 5, 3), 12),
            ("cusparse", Triple(12, 5, 8), 12),
            ("cusparse", Triple(12, 6, 3), 13),
            ("cufile", Int(1110), 12),
            ("cufile", Int(1130), 12),
            ("cufile", Int(1150), 13),
            ("cupti", Int(24), 12),
            ("cupti", Int(26), 12),
            ("cupti", Int(130001), 13),
            // design § 12.4 Windows wheel table and query table
            ("nvrtc", Pair(12, 6), 12),
            ("nvrtc", Pair(13, 0), 13),
            ("cufft", Triple(11, 3, 0), 12),
            ("cufft", Triple(12, 0, 0), 13),
            ("cusolver", Triple(11, 7, 1), 12),
            ("cusolver", Triple(11, 4, 2), 12),
            ("cusolver", Triple(12, 0, 4), 13),
            ("cufile", Int(1080), 12),
            ("cufile", Int(1140), 12),
            ("curand", Int(10301), 12),
            ("cupti", Int(19), 12),
            // CUDA 11.x versions that share a soname or a query with 12.x
            ("cudart", Int(11080), 11),
            ("cublas", Triple(11, 11, 3), 11),
            ("cublasLt", Triple(11, 11, 3), 11),
            ("nvrtc", Pair(11, 8), 11),
            ("cufft", Triple(10, 9, 0), 11),
            ("cusolver", Triple(11, 4, 1), 11),
            ("cusparse", Triple(11, 7, 5), 11),
            ("curand", Int(10300), 11),
            ("cupti", Int(18), 11),
            ("cufile", Int(1050), 11),
            // a CUDA 14 line, for the libraries whose 13.x line is closed
            ("cudart", Int(14000), 14),
            ("cublas", Triple(14, 0, 0), 14),
            ("cublasLt", Triple(14, 0, 0), 14),
            ("nvrtc", Pair(14, 0), 14),
            ("cupti", Int(140000), 14),
        ];
        let mut bad = Vec::new();
        for &(lib, observed, line) in samples {
            for major in [12, 13] {
                let got = version_verdict(lib, major, observed);
                let want = if line == major {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                };
                if got != want {
                    bad.push(std::format!(
                        "{lib} {observed:?} (line {line}) for CUDA {major}: {got:?}, expected {want:?}"
                    ));
                }
            }
        }
        // no version query: cuFile is accepted for 12 with a note and
        // rejected for 13; cuSolverMg is accepted by its versioned name only
        for (lib, major, want) in [
            ("cufile", 12, Verdict::AcceptWithoutQuery),
            ("cufile", 13, Verdict::Reject),
            ("cusolverMg", 12, Verdict::AcceptWithoutQuery),
            ("cusolverMg", 13, Verdict::AcceptWithoutQuery),
            ("cublas", 12, Verdict::Reject),
            ("curand", 13, Verdict::Reject),
        ] {
            let got = version_verdict(lib, major, Absent);
            if got != want {
                bad.push(std::format!(
                    "{lib} Absent for CUDA {major}: {got:?}, expected {want:?}"
                ));
            }
        }
        assert!(bad.is_empty(), "{} wrong verdicts: {bad:#?}", bad.len());
    }

    #[test]
    fn load_error_names_library_major_and_tried_names() {
        let e = LibLoadError {
            library: "cublas",
            cuda_major: 13,
            tried: std::vec![
                ("libcublas.so.13".to_string(), RejectReason::NotFound),
                (
                    "libcublas.so".to_string(),
                    RejectReason::WrongVersion(LibVersion {
                        major: 12,
                        minor: 6,
                        patch: Some(4),
                    }),
                ),
            ],
        };
        let msg = e.to_string();
        for part in [
            "\"cublas\"",
            "CUDA 13",
            "libcublas.so.13 (not found)",
            "libcublas.so (wrong version 12.6.4)",
        ] {
            assert!(msg.contains(part), "{part:?} missing from {msg:?}");
        }
        let none = LibLoadError {
            library: "cufile",
            cuda_major: 12,
            tried: Vec::new(),
        }
        .to_string();
        assert!(
            none.contains("\"cufile\"") && none.contains("CUDA 12"),
            "{none:?}"
        );
    }

    #[test]
    fn windows_dll_names_match_nvidia_wheels() {
        // design § 12.4: DLLs of NVIDIA's win_amd64 wheels (cu12 12.6, cu13 13.0)
        let wheels: [(u32, &[(&str, &str)]); 2] = [
            (
                12,
                &[
                    ("cudart", "cudart64_12"),
                    ("cublas", "cublas64_12"),
                    ("cublasLt", "cublasLt64_12"),
                    ("nvrtc", "nvrtc64_120_0"),
                    ("cufft", "cufft64_11"),
                    ("cusolver", "cusolver64_11"),
                    ("cusolverMg", "cusolverMg64_11"),
                    ("cusparse", "cusparse64_12"),
                    ("curand", "curand64_10"),
                ],
            ),
            (
                13,
                &[
                    ("cudart", "cudart64_13"),
                    ("cublas", "cublas64_13"),
                    ("cublasLt", "cublasLt64_13"),
                    ("nvrtc", "nvrtc64_130_0"),
                    ("cufft", "cufft64_12"),
                    ("cusolver", "cusolver64_12"),
                    ("cusolverMg", "cusolverMg64_12"),
                    ("cusparse", "cusparse64_12"),
                    ("curand", "curand64_10"),
                ],
            ),
        ];
        let mut bad = Vec::new();
        for (major, dlls) in wheels {
            for (lib, dll) in dlls {
                for minor in minors(major) {
                    let cands = lib_candidates(TargetOs::Windows, 64, lib, major, minor);
                    let want = std::format!("{dll}.dll");
                    if cands.first() != Some(&want) {
                        bad.push(std::format!(
                            "{lib} {major}.{minor}: first candidate {:?}, wheel ships {want}",
                            cands.first()
                        ));
                    }
                }
            }
        }
        bad.dedup();
        assert!(bad.is_empty(), "{} mismatches: {bad:#?}", bad.len());
    }

    fn loaded(library: &'static str) -> LoadedLibrary {
        LoadedLibrary {
            library,
            cuda_major: 12,
            opened_as: std::format!("lib{library}.so.12"),
            path: None,
            version: None,
            note: None,
        }
    }

    fn failed(library: &'static str) -> LibLoadError {
        LibLoadError {
            library,
            cuda_major: 12,
            tried: std::vec![(std::format!("lib{library}.so.12"), RejectReason::NotFound)],
        }
    }

    /// A fixture loader: returns the scripted results in order and counts
    /// its calls.
    struct Script {
        library: &'static str,
        results: core::cell::RefCell<Vec<bool>>,
        calls: core::cell::Cell<usize>,
    }

    impl Script {
        fn new(library: &'static str, results: &[bool]) -> Self {
            Script {
                library,
                results: core::cell::RefCell::new(results.iter().rev().copied().collect()),
                calls: core::cell::Cell::new(0),
            }
        }

        fn load(&self) -> Result<(u32, LoadedLibrary), LibLoadError> {
            self.calls.set(self.calls.get() + 1);
            let ok = self.results.borrow_mut().pop().expect("script exhausted");
            if ok {
                Ok((7, loaded(self.library)))
            } else {
                Err(failed(self.library))
            }
        }
    }

    #[test]
    fn is_culib_present_agrees_with_try_culib() {
        let scripts: [&[bool]; 4] = [&[true], &[false], &[false, true], &[true, false]];
        let mut bad = Vec::new();
        for lib in IN_SCOPE {
            for script in scripts {
                for present_first in [false, true] {
                    let s = Script::new(lib, &[script, script, script].concat());
                    let cell: LibCell<u32> = LibCell::new();
                    let (present, tried) = if present_first {
                        let p = cell.is_present(|| s.load());
                        (p, cell.try_get(|| s.load()).is_ok())
                    } else {
                        let t = cell.try_get(|| s.load()).is_ok();
                        (cell.is_present(|| s.load()), t)
                    };
                    let again = cell.try_get(|| s.load()).is_ok();
                    if present != tried || tried != again {
                        bad.push(std::format!(
                            "{lib} {script:?} present_first={present_first}: is_present {present}, try {tried}, try again {again}"
                        ));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "{} disagreements: {bad:#?}", bad.len());
    }

    #[test]
    fn a_failed_load_is_memoized() {
        for lib in IN_SCOPE {
            let s = Script::new(lib, &[false, true, true]);
            let cell: LibCell<u32> = LibCell::new();
            let first = cell.try_get(|| s.load());
            let second = cell.try_get(|| s.load());
            assert_eq!(
                s.calls.get(),
                1,
                "{lib}: loader calls across two try_culib calls"
            );
            let (Err(a), Err(b)) = (first, second) else {
                panic!("{lib}: a failed load must stay failed");
            };
            assert_eq!(a.library, b.library);
            assert_eq!(a.tried.len(), b.tried.len());
            assert!(cell.report().is_none(), "{lib}: no report after a failure");
        }
    }

    #[test]
    fn the_report_says_linked_at_build_time_when_linked() {
        for lib in IN_SCOPE.iter().chain(["cuda"].iter()) {
            let r = linked_report(lib, 12);
            assert_eq!(r.library, *lib);
            assert_eq!(r.cuda_major, 12);
            assert_eq!(r.opened_as, "linked at build time");
            assert!(r.path.is_none() && r.version.is_none() && r.note.is_none());
            let text = r.to_string();
            assert!(
                text.contains(lib) && text.contains("linked at build time"),
                "{text:?}"
            );
        }
    }
}
