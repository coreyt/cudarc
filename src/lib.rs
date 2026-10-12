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

/// The candidate file names cudarc tries, in order, when it loads `lib_name`
/// at run time.
///
/// For the CUDA toolkit libraries of FATHOMDB.md item 9 (`cudart`, `cublas`,
/// `cublasLt`, `nvrtc`, `cupti`, `cufft`, `cusolver`, `cusolverMg`,
/// `cusparse`, `curand`, `cufile`) in a CUDA 12.x or 13.x build, the list is
/// the library's versioned names for the built major, then its unversioned
/// name (none for `cusolverMg`, which has no version query). It never names
/// another major's versioned file. Windows CUPTI, every other library (the
/// driver, cuDNN, NCCL, cuTENSOR, NVTX) and CUDA 11.x builds keep the list of
/// upstream cudarc 0.19.10.
///
/// # Panics
///
/// On a target whose pointer width is neither 32 nor 64 bits.
#[cfg(feature = "dynamic-loading")]
pub fn get_lib_name_candidates(lib_name: &str) -> std::vec::Vec<std::string::String> {
    let pointer_width = if cfg!(target_pointer_width = "32") {
        32
    } else if cfg!(target_pointer_width = "64") {
        64
    } else {
        panic!("Unsupported target pointer width")
    };
    let (major, minor) = lib_major::built_version();
    lib_major::lib_candidates(
        lib_major::TargetOs::host(),
        pointer_width,
        lib_name,
        major,
        minor,
    )
}

pub use lib_major::{LibLoadError, LibVersion, LoadedLibrary, RejectReason};

/// Built-major library loading (FATHOMDB.md item 9): candidate names, version
/// lines, the loader and the loaded-library report.
// Each loader uses the part for its own library, so a build with only some
// library features leaves the rest unused.
#[allow(dead_code)]
pub(crate) mod lib_major {
    use core::fmt;
    use std::string::{String, ToString};
    use std::sync::OnceLock;
    use std::vec::Vec;

    /// The libraries whose loading item 9 changes. Their soname or DLL name
    /// follows the CUDA toolkit's major version.
    pub(crate) const IN_SCOPE: [&str; 11] = [
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

    const LINKED: &str = "linked at build time";

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
        /// The loader found a file but could not load it, or its version
        /// query failed.
        LoadFailed(String),
        /// The library reports a version outside the built major's line.
        WrongVersion(LibVersion),
        /// An object of that name was already loaded in the process and
        /// reports a version outside the built major's line. Another file of
        /// the same soname cannot be opened in this process, so loading stops.
        AlreadyLoadedWrongVersion(LibVersion),
        /// Loading the library mapped an in-scope dependency of another line
        /// (Linux only).
        WrongMajorDependency { soname: String, version: LibVersion },
        /// The library exports no version query. A candidate without one is
        /// accepted only by its versioned name (cuSolverMg; cuFile before 1.8
        /// in a CUDA 12 build). As a note it marks such a load.
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
    /// it was not used. The error is memoized: later calls return a clone.
    #[derive(Clone, Debug)]
    pub struct LibLoadError {
        pub library: &'static str,
        pub cuda_major: u32,
        pub tried: Vec<(String, RejectReason)>,
    }

    impl fmt::Display for LibLoadError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "Unable to dynamically load the \"{}\" shared library for CUDA {}. Tried: ",
                self.library, self.cuda_major
            )?;
            if self.tried.is_empty() {
                write!(f, "no candidate names on this target")?;
            }
            for (i, (name, reason)) in self.tried.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{name} ({reason})")?;
            }
            write!(
                f,
                ". Ensure that `LD_LIBRARY_PATH` has the correct path to a CUDA {} copy of the library.",
                self.cuda_major
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
        /// `/proc/self/maps` (`None` if it could not be resolved, and when
        /// linked).
        pub path: Option<String>,
        /// `None` only when the library has no version query, and when
        /// linked.
        pub version: Option<LibVersion>,
        /// For example [`RejectReason::NoVersionQuery`] for a cuFile without
        /// `cuFileGetVersion`.
        pub note: Option<RejectReason>,
    }

    impl fmt::Display for LoadedLibrary {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.opened_as == LINKED {
                return write!(f, "{}: {LINKED} (CUDA {})", self.library, self.cuda_major);
            }
            write!(f, "{}: opened as {}, path ", self.library, self.opened_as)?;
            match &self.path {
                Some(p) => write!(f, "{p}")?,
                None => write!(f, "unknown")?,
            }
            match &self.version {
                Some(v) => write!(f, ", version {v}")?,
                None => write!(f, ", version none")?,
            }
            if let Some(n) = &self.note {
                write!(f, " ({n})")?;
            }
            write!(f, ", checked against CUDA {}", self.cuda_major)
        }
    }

    pub(crate) fn linked_report(library: &'static str, cuda_major: u32) -> LoadedLibrary {
        LoadedLibrary {
            library,
            cuda_major,
            opened_as: LINKED.to_string(),
            path: None,
            version: None,
            note: None,
        }
    }

    pub(crate) fn built_version() -> (u32, u32) {
        (
            env!("CUDA_MAJOR_VERSION").parse().unwrap_or(0),
            env!("CUDA_MINOR_VERSION").parse().unwrap_or(0),
        )
    }

    /// The report of a library under `dynamic-linking` or `static-linking`.
    pub(crate) fn linked_at_build_time(library: &'static str) -> LoadedLibrary {
        linked_report(library, built_version().0)
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

    /// The candidate list of upstream cudarc 0.19.10, with the target and
    /// version as parameters.
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
            // See issue #242
            std::format!("{pre}{lib_name}{w}_10{suf}"),
            std::format!("{pre}{lib_name}{w}_11{suf}"),
            std::format!("{pre}{lib_name}{w}_12{suf}"),
            // See issue #246
            std::format!("{pre}{lib_name}{w}_{major}0_0{suf}"),
            // See issue #260
            std::format!("{pre}{lib_name}{w}_9{suf}"),
            // See issue #274
            std::format!("{pre}{lib_name}{suf}.{major}"),
            std::format!("{pre}{lib_name}{suf}.12"),
            std::format!("{pre}{lib_name}{suf}.11"),
            std::format!("{pre}{lib_name}{suf}.10"),
            std::format!("{pre}{lib_name}{suf}.9"),
            // See issue #296
            std::format!("{pre}{lib_name}{suf}.1"),
        ]
        .into()
    }

    /// Whether `lib` is loaded by item 9's rules on this target and major.
    pub(crate) fn uses_checked_candidates(os: TargetOs, lib: &str, major: u32) -> bool {
        (major == 12 || major == 13)
            && IN_SCOPE.contains(&lib)
            && match os {
                TargetOs::Linux => true,
                // year-named CUPTI DLLs (design § 12.4 note 1)
                TargetOs::Windows => lib != "cupti",
                TargetOs::Other { .. } => false,
            }
    }

    /// The soname (Linux) or DLL (Windows) majors of `lib` for toolkit major
    /// `major` (design § 12.4).
    fn file_majors(lib: &str, major: u32) -> &'static [u32] {
        match (lib, major) {
            ("cudart" | "cublas" | "cublasLt" | "nvrtc" | "cupti", 12) => &[12],
            ("cudart" | "cublas" | "cublasLt" | "nvrtc" | "cupti", 13) => &[13],
            ("cufft" | "cusolver" | "cusolverMg", 12) => &[11],
            ("cufft" | "cusolver" | "cusolverMg", 13) => &[12],
            ("cusparse", _) => &[12],
            ("curand", 12) => &[10],
            // libcurand.so.11 is unconfirmed (PATHFINDER); accepted only
            // through the version check
            ("curand", 13) => &[10, 11],
            ("cufile", _) => &[0],
            _ => &[],
        }
    }

    pub(crate) fn versioned_candidates(
        os: TargetOs,
        pointer_width: u32,
        lib: &str,
        major: u32,
    ) -> Vec<String> {
        let majors = file_majors(lib, major);
        match os {
            TargetOs::Linux => majors
                .iter()
                .map(|n| std::format!("lib{lib}.so.{n}"))
                .collect(),
            // cuFile is not available on Windows
            TargetOs::Windows if lib == "cufile" => Vec::new(),
            TargetOs::Windows if lib == "nvrtc" => {
                std::vec![std::format!("nvrtc{pointer_width}_{major}0_0.dll")]
            }
            TargetOs::Windows => majors
                .iter()
                .map(|n| std::format!("{lib}{pointer_width}_{n}.dll"))
                .collect(),
            TargetOs::Other { .. } => Vec::new(),
        }
    }

    fn unversioned_candidate(os: TargetOs, pointer_width: u32, lib: &str) -> Option<String> {
        match (os, lib) {
            // no version query: versioned name only
            (_, "cusolverMg") => None,
            (TargetOs::Linux, _) => Some(std::format!("lib{lib}.so")),
            (TargetOs::Windows, "cufile") => None,
            (TargetOs::Windows, _) => Some(std::format!("{lib}{pointer_width}.dll")),
            (TargetOs::Other { .. }, _) => None,
        }
    }

    /// The candidates for `lib`: item 9's list where it applies (see
    /// [`crate::get_lib_name_candidates`]), otherwise the base list.
    pub(crate) fn lib_candidates(
        os: TargetOs,
        pointer_width: u32,
        lib: &str,
        major: u32,
        minor: u32,
    ) -> Vec<String> {
        if !uses_checked_candidates(os, lib, major) {
            return base_candidates(os, pointer_width, lib, major, minor);
        }
        let mut out = versioned_candidates(os, pointer_width, lib, major);
        out.extend(unversioned_candidate(os, pointer_width, lib));
        out
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

    fn int_version(v: i64) -> u32 {
        u32::try_from(v).unwrap_or(0)
    }

    pub(crate) fn decode_version(lib: &str, observed: Observed) -> Option<LibVersion> {
        let v = |major, minor, patch| {
            Some(LibVersion {
                major,
                minor,
                patch,
            })
        };
        match (lib, observed) {
            (_, Observed::Absent) => None,
            (_, Observed::Triple(a, b, c)) => v(a, b, Some(c)),
            (_, Observed::Pair(a, b)) => v(a, b, None),
            // 12060 = 12.6
            ("cudart" | "cuda", Observed::Int(n)) => {
                let n = int_version(n);
                v(n / 1000, n % 1000 / 10, None)
            }
            // 10307 = 10.3.7
            ("curand", Observed::Int(n)) => {
                let n = int_version(n);
                v(n / 1000, n % 1000 / 100, Some(n % 100))
            }
            // 1110 = 1.11
            ("cufile", Observed::Int(n)) => {
                let n = int_version(n);
                v(n / 1000, n % 1000 / 10, None)
            }
            // 130001 = 13.0.1 (xxyyzz from CUDA 13); before: an API version
            ("cupti", Observed::Int(n)) => {
                let n = int_version(n);
                if n >= 10000 {
                    v(n / 10000, n / 100 % 100, Some(n % 100))
                } else {
                    v(n, 0, None)
                }
            }
            (_, Observed::Int(n)) => v(int_version(n), 0, None),
        }
    }

    /// Whether a library of `lib` that reports `observed` belongs to the
    /// line of built major `major` (design § 12.4 table).
    pub(crate) fn version_verdict(lib: &str, major: u32, observed: Observed) -> Verdict {
        use Observed::{Absent, Int, Pair, Triple};
        if observed == Absent {
            return match (lib, major) {
                ("cusolverMg", _) | ("cufile", 12) => Verdict::AcceptWithoutQuery,
                _ => Verdict::Reject,
            };
        }
        let m = i64::from(major);
        let accept = match (lib, major, observed) {
            ("cudart", _, Int(v)) => v / 1000 == m,
            ("cublas" | "cublasLt", _, Triple(a, _, _)) => a == major,
            ("nvrtc", _, Pair(a, _)) => a == major,
            ("cupti", 12, Int(v)) => (19..130000).contains(&v),
            ("cupti", 13, Int(v)) => v / 10000 == 13,
            ("cufft", 12, Triple(a, _, _)) => a == 11,
            ("cufft", 13, Triple(a, _, _)) => a == 12,
            ("cusolver", 12, Triple(a, b, c)) => (a, b, c) >= (11, 4, 2) && a < 12,
            ("cusolver", 13, Triple(a, _, _)) => a == 12,
            ("cusparse", 12, Triple(a, b, _)) => a == 12 && b <= 5,
            ("cusparse", 13, Triple(a, b, _)) => (a, b) >= (12, 6),
            ("curand", 12, Int(v)) => (10301..=10399).contains(&v),
            ("curand", 13, Int(v)) => v >= 10400,
            ("cufile", 12, Int(v)) => (1080..=1140).contains(&v),
            ("cufile", 13, Int(v)) => v >= 1150,
            _ => false,
        };
        if accept {
            Verdict::Accept
        } else {
            Verdict::Reject
        }
    }

    /// The memoized result of loading one library: `try_culib`,
    /// `is_culib_present` and `loaded_library` of a loader share one cell, so
    /// they agree, and a failure is not retried.
    pub(crate) struct LibCell<L> {
        cell: OnceLock<Result<(L, LoadedLibrary), LibLoadError>>,
    }

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
            match self.cell.get_or_init(load) {
                Ok((l, _)) => Ok(l),
                Err(e) => Err(e.clone()),
            }
        }

        pub(crate) fn is_present(
            &self,
            load: impl FnOnce() -> Result<(L, LoadedLibrary), LibLoadError>,
        ) -> bool {
            self.try_get(load).is_ok()
        }

        pub(crate) fn report(&self) -> Option<LoadedLibrary> {
            match self.cell.get() {
                Some(Ok((_, r))) => Some(r.clone()),
                _ => None,
            }
        }
    }

    /// The version query of `lib`, if it has one.
    pub(crate) fn query_symbol(lib: &str) -> Option<&'static str> {
        Some(match lib {
            "cudart" => "cudaRuntimeGetVersion",
            "cublas" => "cublasGetProperty",
            "cublasLt" => "cublasLtGetProperty",
            "nvrtc" => "nvrtcVersion",
            "cupti" => "cuptiGetVersion",
            "cufft" => "cufftGetProperty",
            "cusolver" => "cusolverGetProperty",
            "cusparse" => "cusparseGetProperty",
            "curand" => "curandGetVersion",
            "cufile" => "cuFileGetVersion",
            _ => return None,
        })
    }

    /// An exported function of `lib`, for resolving the loaded file.
    pub(crate) fn address_symbol(lib: &str) -> &'static str {
        match lib {
            "cusolverMg" => "cusolverMgCreate",
            "cuda" => "cuDriverGetVersion",
            _ => query_symbol(lib).unwrap_or("cudaRuntimeGetVersion"),
        }
    }

    // a build with only the driver uses driver_report alone
    #[cfg(feature = "dynamic-loading")]
    #[allow(unused_imports)]
    pub(crate) use self::loader::{driver_report, load_culib, panic_lib_load_error};

    #[cfg(feature = "dynamic-loading")]
    mod loader {
        use super::{
            address_symbol, built_version, decode_version, lib_candidates, query_symbol,
            uses_checked_candidates, version_verdict, versioned_candidates, LibLoadError,
            LoadedLibrary, Observed, RejectReason, TargetOs, Verdict, IN_SCOPE,
        };
        use core::ffi::c_int;
        use libloading::Library;
        use std::string::{String, ToString};
        use std::vec::Vec;

        /// `culib()`'s panic on a failed load.
        pub(crate) fn panic_lib_load_error(e: &LibLoadError) -> ! {
            panic!("{e}")
        }

        #[repr(C)]
        struct CuFileStatus {
            err: c_int,
            cu_err: c_int,
        }

        /// Calls `lib`'s version query through `dlsym` on `handle` itself (never
        /// through the generated binding, which panics on a missing symbol).
        unsafe fn query_version(lib: &str, handle: &Library) -> Result<Observed, String> {
            type IntFn = unsafe extern "C" fn(*mut c_int) -> c_int;
            type U32Fn = unsafe extern "C" fn(*mut u32) -> c_int;
            type PairFn = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
            type PropFn = unsafe extern "C" fn(c_int, *mut c_int) -> c_int;
            type FileFn = unsafe extern "C" fn(*mut c_int) -> CuFileStatus;
            let Some(symbol) = query_symbol(lib) else {
                return Ok(Observed::Absent);
            };
            let failed = |s: c_int| std::format!("{symbol} returned status {s}");
            match lib {
                "cudart" | "curand" => {
                    let Ok(f) = handle.get::<IntFn>(symbol.as_bytes()) else {
                        return Ok(Observed::Absent);
                    };
                    let mut v = 0;
                    match f(&mut v) {
                        0 => Ok(Observed::Int(v.into())),
                        s => Err(failed(s)),
                    }
                }
                "cupti" => {
                    let Ok(f) = handle.get::<U32Fn>(symbol.as_bytes()) else {
                        return Ok(Observed::Absent);
                    };
                    let mut v = 0;
                    match f(&mut v) {
                        0 => Ok(Observed::Int(v.into())),
                        s => Err(failed(s)),
                    }
                }
                "nvrtc" => {
                    let Ok(f) = handle.get::<PairFn>(symbol.as_bytes()) else {
                        return Ok(Observed::Absent);
                    };
                    let (mut a, mut b) = (0, 0);
                    match f(&mut a, &mut b) {
                        0 => Ok(Observed::Pair(a as u32, b as u32)),
                        s => Err(failed(s)),
                    }
                }
                "cufile" => {
                    let Ok(f) = handle.get::<FileFn>(symbol.as_bytes()) else {
                        return Ok(Observed::Absent);
                    };
                    let mut v = 0;
                    let s = f(&mut v);
                    if s.err == 0 {
                        Ok(Observed::Int(v.into()))
                    } else {
                        Err(std::format!(
                            "{symbol} returned error {} (CUresult {})",
                            s.err,
                            s.cu_err
                        ))
                    }
                }
                _ => {
                    let Ok(f) = handle.get::<PropFn>(symbol.as_bytes()) else {
                        return Ok(Observed::Absent);
                    };
                    // libraryPropertyType: MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                    let mut v = [0; 3];
                    for (t, out) in v.iter_mut().enumerate() {
                        match f(t as c_int, out) {
                            0 => {}
                            s => return Err(failed(s)),
                        }
                    }
                    Ok(Observed::Triple(v[0] as u32, v[1] as u32, v[2] as u32))
                }
            }
        }

        #[cfg(target_os = "linux")]
        mod os {
            use core::ffi::{c_char, c_int, c_void, CStr};
            use libloading::Library;
            use std::string::{String, ToString};
            use std::vec::Vec;

            // glibc and musl
            const RTLD_NOLOAD: c_int = 4;

            #[repr(C)]
            struct DlInfo {
                dli_fname: *const c_char,
                dli_fbase: *mut c_void,
                dli_sname: *const c_char,
                dli_saddr: *mut c_void,
            }

            /// The leading fields of `struct dl_phdr_info`.
            #[repr(C)]
            struct DlPhdrInfo {
                dlpi_addr: usize,
                dlpi_name: *const c_char,
            }

            type PhdrCallback = unsafe extern "C" fn(
                info: *mut DlPhdrInfo,
                size: usize,
                data: *mut c_void,
            ) -> c_int;

            extern "C" {
                fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
                fn dl_iterate_phdr(callback: Option<PhdrCallback>, data: *mut c_void) -> c_int;
            }

            /// The object already loaded under `name` (a soname or a path),
            /// without loading anything.
            pub(super) unsafe fn already_loaded(name: &str) -> Option<Library> {
                use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
                UnixLibrary::open(Some(name), RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD)
                    .ok()
                    .map(Library::from)
            }

            /// The canonical path of the file that defines `symbol` in `lib`.
            pub(super) unsafe fn library_path(lib: &Library, symbol: &str) -> Option<String> {
                let addr = *lib.get::<*mut c_void>(symbol.as_bytes()).ok()?;
                let mut info = DlInfo {
                    dli_fname: core::ptr::null(),
                    dli_fbase: core::ptr::null_mut(),
                    dli_sname: core::ptr::null(),
                    dli_saddr: core::ptr::null_mut(),
                };
                if dladdr(addr, &mut info) == 0 || info.dli_fname.is_null() {
                    return None;
                }
                let name = CStr::from_ptr(info.dli_fname)
                    .to_string_lossy()
                    .into_owned();
                // realpath(3)
                Some(match std::fs::canonicalize(&name) {
                    Ok(p) => p.to_string_lossy().into_owned(),
                    Err(_) => name,
                })
            }

            unsafe extern "C" fn collect(
                info: *mut DlPhdrInfo,
                _size: usize,
                data: *mut c_void,
            ) -> c_int {
                let out = &mut *(data as *mut Vec<String>);
                let name = (*info).dlpi_name;
                if !name.is_null() {
                    let s = CStr::from_ptr(name).to_string_lossy();
                    if !s.is_empty() {
                        out.push(s.to_string());
                    }
                }
                0
            }

            /// The file names of every loaded object (the link map).
            pub(super) fn link_map() -> Vec<String> {
                let mut out: Vec<String> = Vec::new();
                unsafe {
                    dl_iterate_phdr(Some(collect), &mut out as *mut Vec<String> as *mut c_void)
                };
                out
            }
        }

        #[cfg(target_os = "windows")]
        mod os {
            use core::ffi::c_void;
            use libloading::Library;
            use std::string::String;

            const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;
            const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;

            #[link(name = "kernel32")]
            extern "system" {
                fn GetModuleHandleExW(
                    flags: u32,
                    name: *const u16,
                    module: *mut *mut c_void,
                ) -> i32;
                fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
            }

            /// The path of the module that defines `symbol` in `lib`.
            pub(super) unsafe fn library_path(lib: &Library, symbol: &str) -> Option<String> {
                let addr = *lib.get::<*mut c_void>(symbol.as_bytes()).ok()?;
                let mut module = core::ptr::null_mut();
                if GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                        | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    addr as *const u16,
                    &mut module,
                ) == 0
                {
                    return None;
                }
                let mut buf = [0u16; 32768];
                let n = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) as usize;
                if n == 0 || n >= buf.len() {
                    return None;
                }
                Some(String::from_utf16_lossy(&buf[..n]))
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        mod os {
            use libloading::Library;
            use std::string::String;

            pub(super) unsafe fn library_path(_lib: &Library, _symbol: &str) -> Option<String> {
                None
            }
        }

        fn open_error(e: &libloading::Error) -> RejectReason {
            use std::error::Error;
            // libloading's message is generic; the loader's text is the source
            let msg = match e.source() {
                Some(source) => std::format!("{e}: {source}"),
                None => e.to_string(),
            };
            // glibc's dlerror text; Windows ERROR_MOD_NOT_FOUND
            if msg.contains("No such file or directory") || msg.contains("(os error 126)") {
                RejectReason::NotFound
            } else {
                RejectReason::LoadFailed(msg)
            }
        }

        enum Judged {
            Accept(Observed),
            Reject(RejectReason),
        }

        /// Applies the version line of `lib` to an opened `handle`.
        unsafe fn judge(lib: &str, major: u32, versioned: bool, handle: &Library) -> Judged {
            let observed = match query_version(lib, handle) {
                Ok(o) => o,
                Err(msg) => return Judged::Reject(RejectReason::LoadFailed(msg)),
            };
            match version_verdict(lib, major, observed) {
                Verdict::Accept => Judged::Accept(observed),
                Verdict::AcceptWithoutQuery if versioned => Judged::Accept(observed),
                Verdict::AcceptWithoutQuery => Judged::Reject(RejectReason::NoVersionQuery),
                Verdict::Reject => Judged::Reject(match decode_version(lib, observed) {
                    Some(v) => RejectReason::WrongVersion(v),
                    None => RejectReason::NoVersionQuery,
                }),
            }
        }

        fn in_scope_library_of_file(file: &str) -> Option<&'static str> {
            IN_SCOPE.iter().copied().find(|l| {
                let stem = std::format!("lib{l}.so");
                file == stem
                    || file
                        .strip_prefix(stem.as_str())
                        .is_some_and(|r| r.starts_with('.'))
            })
        }

        /// Version-checks the in-scope objects that loading `lib` newly mapped.
        #[cfg(target_os = "linux")]
        unsafe fn wrong_dependency(
            lib: &str,
            major: u32,
            before: &[String],
        ) -> Option<RejectReason> {
            for path in os::link_map() {
                if before.contains(&path) {
                    continue;
                }
                let file = path.rsplit('/').next().unwrap_or(&path);
                let Some(dep) = in_scope_library_of_file(file) else {
                    continue;
                };
                if dep == lib {
                    continue;
                }
                let Some(handle) = os::already_loaded(&path) else {
                    continue;
                };
                let Ok(observed) = query_version(dep, &handle) else {
                    continue;
                };
                if version_verdict(dep, major, observed) == Verdict::Reject {
                    if let Some(version) = decode_version(dep, observed) {
                        return Some(RejectReason::WrongMajorDependency {
                            soname: file.to_string(),
                            version,
                        });
                    }
                }
            }
            None
        }

        enum Step {
            Use(Library, Observed),
            Next(RejectReason),
            Stop(RejectReason),
        }

        unsafe fn try_candidate(
            lib: &str,
            major: u32,
            checked: bool,
            versioned: bool,
            name: &str,
        ) -> Step {
            if !checked {
                // CUDA 11.x builds, Windows CUPTI, other targets: the first
                // name that opens, as upstream; the version is only reported
                return match Library::new(name) {
                    Ok(handle) => {
                        let observed = query_version(lib, &handle).unwrap_or(Observed::Absent);
                        Step::Use(handle, observed)
                    }
                    Err(e) => Step::Next(open_error(&e)),
                };
            }
            #[cfg(target_os = "linux")]
            if let Some(handle) = os::already_loaded(name) {
                return match judge(lib, major, versioned, &handle) {
                    Judged::Accept(observed) => Step::Use(handle, observed),
                    Judged::Reject(RejectReason::WrongVersion(v)) => {
                        Step::Stop(RejectReason::AlreadyLoadedWrongVersion(v))
                    }
                    Judged::Reject(r) => Step::Next(r),
                };
            }
            #[cfg(target_os = "linux")]
            let before = os::link_map();
            let handle = match Library::new(name) {
                Ok(h) => h,
                Err(e) => return Step::Next(open_error(&e)),
            };
            let observed = match judge(lib, major, versioned, &handle) {
                Judged::Accept(o) => o,
                Judged::Reject(r) => return Step::Next(r),
            };
            #[cfg(target_os = "linux")]
            if let Some(r) = wrong_dependency(lib, major, &before) {
                return Step::Next(r);
            }
            Step::Use(handle, observed)
        }

        /// Loads `lib` for the built CUDA major: the loader behind each
        /// in-scope `try_culib`.
        pub(crate) unsafe fn load_culib(
            lib: &'static str,
        ) -> Result<(Library, LoadedLibrary), LibLoadError> {
            let (major, minor) = built_version();
            let os = TargetOs::host();
            let width = if cfg!(target_pointer_width = "32") {
                32
            } else {
                64
            };
            let checked = uses_checked_candidates(os, lib, major);
            let versioned = if checked {
                versioned_candidates(os, width, lib, major)
            } else {
                Vec::new()
            };
            let mut tried = Vec::new();
            for name in lib_candidates(os, width, lib, major, minor) {
                match try_candidate(lib, major, checked, versioned.contains(&name), &name) {
                    Step::Use(handle, observed) => {
                        let report = LoadedLibrary {
                            library: lib,
                            cuda_major: major,
                            path: os::library_path(&handle, address_symbol(lib)),
                            opened_as: name,
                            version: decode_version(lib, observed),
                            note: (observed == Observed::Absent)
                                .then_some(RejectReason::NoVersionQuery),
                        };
                        return Ok((handle, report));
                    }
                    Step::Next(reason) => tried.push((name, reason)),
                    Step::Stop(reason) => {
                        tried.push((name, reason));
                        break;
                    }
                }
            }
            Err(LibLoadError {
                library: lib,
                cuda_major: major,
                tried,
            })
        }

        /// The driver's report: its file and `cuDriverGetVersion`.
        pub(crate) unsafe fn driver_report(handle: &Library, opened_as: &str) -> LoadedLibrary {
            type IntFn = unsafe extern "C" fn(*mut c_int) -> c_int;
            let version = handle
                .get::<IntFn>(b"cuDriverGetVersion")
                .ok()
                .and_then(|f| {
                    let mut v = 0;
                    (f(&mut v) == 0).then_some(v)
                })
                .and_then(|v| decode_version("cuda", Observed::Int(v.into())));
            LoadedLibrary {
                library: "cuda",
                cuda_major: built_version().0,
                opened_as: opened_as.to_string(),
                path: os::library_path(handle, address_symbol("cuda")),
                version,
                note: None,
            }
        }
    }
}

/// T-UNIT of test plan § 3.4 (requirements CF-AC-11). Pure: every target is
/// a parameter, so the Windows names are checked on Linux too.
#[cfg(test)]
mod fathomdb_lib_major {
    use crate::lib_major::{
        address_symbol, lib_candidates, linked_report, query_symbol, version_verdict, LibCell,
        Observed, TargetOs, Verdict,
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

    /// The checked-in binding file of each library whose loaded file the
    /// report resolves.
    fn binding_source(lib: &str) -> &'static str {
        match lib {
            "cuda" => include_str!("driver/sys/mod.rs"),
            "cudart" => include_str!("runtime/sys/mod.rs"),
            "cublas" => include_str!("cublas/sys/mod.rs"),
            "cublasLt" => include_str!("cublaslt/sys/mod.rs"),
            "nvrtc" => include_str!("nvrtc/sys/mod.rs"),
            "cupti" => include_str!("cupti/sys/mod.rs"),
            "cufft" => include_str!("cufft/sys/mod.rs"),
            "cusolver" => include_str!("cusolver/sys/mod.rs"),
            "cusolverMg" => include_str!("cusolvermg/sys/mod.rs"),
            "cusparse" => include_str!("cusparse/sys/mod.rs"),
            "curand" => include_str!("curand/sys/mod.rs"),
            "cufile" => include_str!("cufile/sys/mod.rs"),
            _ => panic!("no binding file for {lib}"),
        }
    }

    /// The `cuda-*` features of [VERSIONS] for which `lib`'s module exists
    /// (`src/lib.rs`: cuFFT is not built for CUDA 11.x).
    fn module_features(lib: &str) -> Vec<String> {
        VERSIONS
            .iter()
            .filter(|(major, _)| !(lib == "cufft" && *major == 11))
            .map(|(major, minor)| std::format!("cuda-{}{:03}", major, minor * 10))
            .collect()
    }

    /// The `cuda-*` features of `features` for which the binding file binds
    /// `symbol` (no `cfg` on the function, or a `cfg(any(...))` naming the
    /// feature). All of them are missing if the function is not bound.
    fn features_without(source: &str, symbol: &str, features: &[String]) -> Vec<String> {
        let decl = std::format!("pub unsafe fn {symbol}(");
        let Some(at) = source.find(&decl) else {
            return features.to_vec();
        };
        let before = source[..at].trim_end_matches([' ', '\n']);
        let prev = before.rsplit('\n').next().unwrap_or("");
        if !prev.starts_with("#[cfg(") {
            return Vec::new();
        }
        features
            .iter()
            .filter(|f| !prev.contains(&std::format!("feature = \"{f}\"")))
            .cloned()
            .collect()
    }

    #[test]
    fn report_address_symbols_are_bound_wherever_the_library_is() {
        let mut bad = Vec::new();
        for lib in IN_SCOPE.iter().copied().chain(["cuda"]) {
            let symbol = address_symbol(lib);
            let missing = features_without(binding_source(lib), symbol, &module_features(lib));
            if !missing.is_empty() {
                bad.push(std::format!("{lib}: {symbol} not bound for {missing:?}"));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
        // A cuFile before 1.8 has no `cuFileGetVersion` and is still accepted
        // for CUDA 12, so its path must come from another symbol.
        assert_ne!(Some(address_symbol("cufile")), query_symbol("cufile"));
    }
}
