//! T-DEV of test plan § 3.4 (requirements CF-AC-11): which CUDA library file
//! a process actually maps, read from `/proc/self/maps`.
//!
//! Every test is ignored. `fathomdb/scripts/t-dev-lib-major.sh` runs each one
//! as its own process, with the library setup of its case, and checks the
//! case's precondition first. Run by hand, a test only fails its own
//! precondition.

#![cfg(target_os = "linux")]
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[allow(unused_imports)]
use cudarc::RejectReason;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;

/// The distinct mapped file paths whose file name contains `pattern`.
fn mapped(pattern: &str) -> Vec<String> {
    let maps = std::fs::read_to_string("/proc/self/maps").expect("read /proc/self/maps");
    let mut out: Vec<String> = maps
        .lines()
        .filter_map(|l| l.split_whitespace().nth(5))
        .filter(|p| {
            p.rsplit('/')
                .next()
                .is_some_and(|name| name.contains(pattern))
        })
        .map(str::to_string)
        .collect();
    out.sort();
    out.dedup();
    out
}

fn print_maps(case: &str, pattern: &str) -> Vec<String> {
    let m = mapped(pattern);
    for p in &m {
        println!("MAPS {case} {p}");
    }
    if m.is_empty() {
        println!("MAPS {case} (no {pattern} mapping)");
    }
    m
}

fn env_dir(var: &str) -> String {
    let d = std::env::var(var)
        .unwrap_or_else(|_| panic!("precondition: {var} is not set (run t-dev-lib-major.sh)"));
    std::fs::canonicalize(&d)
        .unwrap_or_else(|e| panic!("precondition: {var}={d}: {e}"))
        .to_string_lossy()
        .into_owned()
}

/// Opens `name` with `dlopen` and closes it again; returns whether it opened.
fn open_and_close(name: &str) -> bool {
    let c = CString::new(name).unwrap();
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
    if h.is_null() {
        return false;
    }
    unsafe { dlclose(h) };
    true
}

/// (a) A 12.x build with an unversioned `libcublas.so` that resolves to the
/// CUDA 13 cuBLAS first on `LD_LIBRARY_PATH`. The build must map the
/// system's CUDA 12 `libcublas.so.12`, never `libcublas.so.13`.
#[cfg(feature = "cublas")]
#[test]
#[ignore = "T-DEV (a): run by fathomdb/scripts/t-dev-lib-major.sh"]
fn cuda12_build_skips_an_unversioned_cuda13_cublas() {
    let cu13 = env_dir("FATHOMDB_T_DEV_CU13_LIB");
    assert_eq!(
        env!("CUDA_MAJOR_VERSION"),
        "12",
        "precondition: a 12.x build"
    );
    let loaded = catch_unwind(AssertUnwindSafe(|| unsafe {
        cudarc::cublas::sys::culib();
    }));
    let m = print_maps("a", "libcublas.so");
    assert!(loaded.is_ok(), "cuBLAS did not load");
    let wrong: Vec<&String> = m.iter().filter(|p| p.contains("libcublas.so.13")).collect();
    assert!(
        wrong.is_empty(),
        "a CUDA 12 build mapped CUDA 13 cuBLAS: {wrong:?}"
    );
    let right: Vec<&String> = m
        .iter()
        .filter(|p| p.contains("libcublas.so.12") && !p.starts_with(&cu13))
        .collect();
    assert_eq!(
        right.len(),
        1,
        "expected one CUDA 12 cuBLAS mapping, got {m:?}"
    );
    // the report (new API, checked only at GREEN) names the mapped file
    let r = cudarc::cublas::sys::loaded_library().expect("a cuBLAS report after the load");
    println!("REPORT a {r}");
    assert_eq!(r.cuda_major, 12);
    assert_eq!(r.opened_as, "libcublas.so.12");
    assert_eq!(
        r.path.as_ref(),
        Some(right[0]),
        "report path vs /proc/self/maps"
    );
    assert_eq!(r.version.map(|v| v.major), Some(12));
}

/// (b) A 13.x build where no CUDA 13 cuBLAS is reachable and the unversioned
/// name resolves to CUDA 12.6 through the loader cache. No CUDA 12 cuBLAS
/// may stay mapped.
#[cfg(feature = "cublas")]
#[test]
#[ignore = "T-DEV (b): run by fathomdb/scripts/t-dev-lib-major.sh"]
fn cuda13_build_rejects_the_cuda12_cublas_from_the_loader_cache() {
    assert_eq!(
        env!("CUDA_MAJOR_VERSION"),
        "13",
        "precondition: a 13.x build"
    );
    let loaded = catch_unwind(AssertUnwindSafe(|| unsafe {
        cudarc::cublas::sys::culib();
    }));
    println!(
        "RESULT b culib() {}",
        if loaded.is_ok() {
            "returned"
        } else {
            "panicked"
        }
    );
    let m = print_maps("b", "libcublas");
    let wrong: Vec<&String> = m.iter().filter(|p| p.contains("libcublas.so.12")).collect();
    assert!(
        wrong.is_empty(),
        "a CUDA 13 build mapped CUDA 12 cuBLAS: {wrong:?}"
    );
    // the fallible check (new API, checked only at GREEN)
    let e = unsafe { cudarc::cublas::sys::try_culib() }
        .map(|_| ())
        .expect_err("no CUDA 13 cuBLAS is reachable");
    println!("RESULT b try_culib(): {e}");
    assert_eq!((e.library, e.cuda_major), ("cublas", 13));
    assert!(
        matches!(
            e.tried.as_slice(),
            [(n13, RejectReason::NotFound), (n, RejectReason::WrongVersion(v))]
                if n13 == "libcublas.so.13" && n == "libcublas.so" && v.major == 12
        ),
        "{:?}",
        e.tried
    );
    assert!(!unsafe { cudarc::cublas::sys::is_culib_present() });
}

/// (c) No acceptable cuBLAS at all: loading fails without a panic or an
/// abort.
#[cfg(feature = "cublas")]
#[test]
#[ignore = "T-DEV (c): run by fathomdb/scripts/t-dev-lib-major.sh"]
fn no_acceptable_cublas_is_a_typed_error_not_a_panic() {
    for name in ["libcublas.so", "libcublas.so.12", "libcublas.so.13"] {
        assert!(!open_and_close(name), "precondition: {name} is resolvable");
    }
    // RED (base API) called culib(), which panicked. GREEN calls the
    // fallible check, which must return the typed error instead.
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        cudarc::cublas::sys::try_culib().map(|_| ())
    }));
    let m = print_maps("c", "libcublas");
    let e = result
        .expect("the fallible check panicked")
        .expect_err("no cuBLAS is resolvable");
    println!("RESULT c try_culib(): {e}");
    assert!(m.is_empty(), "cuBLAS mapped: {m:?}");
    assert_eq!((e.library, e.cuda_major), ("cublas", 12));
    let names: Vec<&str> = e.tried.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["libcublas.so.12", "libcublas.so"]);
    assert!(e
        .tried
        .iter()
        .all(|(_, r)| matches!(r, RejectReason::NotFound)));
    assert!(e.to_string().contains("CUDA 12"), "{e}");
    assert!(!unsafe { cudarc::cublas::sys::is_culib_present() });
    assert!(cudarc::cublas::sys::loaded_library().is_none());
}

/// (d) A 12.x build in a process where the CUDA 13 cuFile is already
/// mapped: an earlier `dlopen("libcufile.so")` resolved the decoy to the
/// cu13 `libcufile.so.0`, which stays mapped after `dlclose`
/// (`DF_1_NODELETE`). The build must not use it.
// the cuFileGetVersion binding starts at cuda-12030
#[cfg(all(
    feature = "cufile",
    not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020"
    ))
))]
#[test]
#[ignore = "T-DEV (d): run by fathomdb/scripts/t-dev-lib-major.sh"]
fn cuda12_build_rejects_an_already_loaded_cuda13_cufile() {
    let cu13 = env_dir("FATHOMDB_T_DEV_CU13_LIB");
    assert_eq!(
        env!("CUDA_MAJOR_VERSION"),
        "12",
        "precondition: a 12.x build"
    );
    assert!(
        open_and_close("libcufile.so"),
        "precondition: libcufile.so opens"
    );
    let pre = print_maps("d-precondition", "libcufile.so");
    assert!(
        pre.iter().any(|p| p.starts_with(&cu13)),
        "precondition: the cu13 cuFile stays mapped after dlclose, got {pre:?}"
    );
    let version = catch_unwind(AssertUnwindSafe(|| unsafe {
        cudarc::cufile::sys::culib();
        let mut v = 0;
        cudarc::cufile::sys::cuFileGetVersion(&mut v);
        v
    }));
    println!("RESULT d cuFileGetVersion through culib(): {version:?}");
    print_maps("d", "libcufile.so");
    if let Ok(v) = version {
        assert!(
            v < 1150,
            "a CUDA 12 build uses the CUDA 13 cuFile (version {v})"
        );
    }
    // the fallible check (new API, checked only at GREEN)
    let e = unsafe { cudarc::cufile::sys::try_culib() }
        .map(|_| ())
        .expect_err("the cu13 cuFile is already loaded");
    println!("RESULT d try_culib(): {e}");
    assert!(
        matches!(
            e.tried.last(),
            Some((n, RejectReason::AlreadyLoadedWrongVersion(v)))
                if n == "libcufile.so.0" && (v.major, v.minor) == (1, 15)
        ),
        "{:?}",
        e.tried
    );
    let after = print_maps("d-after", "libcufile.so");
    assert!(
        after.iter().all(|p| p.starts_with(&cu13)) && !after.is_empty(),
        "only the cu13 cuFile stays mapped: {after:?}"
    );
}

/// (e) The driver loads when only `libcuda.so.1` exists (no `libcuda.so`
/// development link). G5b host only.
#[cfg(feature = "driver")]
#[test]
#[ignore = "T-DEV (e): G5b host only; run by fathomdb/scripts/t-dev-lib-major.sh"]
fn driver_loads_with_only_libcuda_so_1() {
    let cache = std::process::Command::new("/sbin/ldconfig")
        .arg("-p")
        .output()
        .expect("run ldconfig -p");
    let cache = String::from_utf8_lossy(&cache.stdout);
    assert!(
        !cache
            .lines()
            .any(|l| l.trim_start().starts_with("libcuda.so ")),
        "precondition: ldconfig -p lists libcuda.so"
    );
    let mut dirs: Vec<String> = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(str::to_string)
        .collect();
    for d in [
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ] {
        dirs.push(d.to_string());
    }
    for d in &dirs {
        let p = std::path::Path::new(d).join("libcuda.so");
        assert!(!p.exists(), "precondition: {} exists", p.display());
    }
    let loaded = catch_unwind(AssertUnwindSafe(|| unsafe {
        cudarc::driver::sys::culib();
    }));
    let m = print_maps("e", "libcuda.so");
    assert!(loaded.is_ok(), "the driver did not load");
    assert!(!m.is_empty(), "no libcuda mapping");
    // the driver report (new API, checked only at GREEN)
    let r = cudarc::driver::sys::loaded_library().expect("a driver report");
    println!("REPORT e {r}");
    assert!(
        r.path.as_ref().is_some_and(|p| m.contains(p)),
        "{r} vs {m:?}"
    );
}

/// (f) NVTX (out of scope of item 9) loads when only `libnvToolsExt.so.1`
/// is resolvable.
#[cfg(feature = "nvtx")]
#[test]
#[ignore = "T-DEV (f): run by fathomdb/scripts/t-dev-lib-major.sh"]
fn nvtx_loads_with_only_libnvtoolsext_so_1() {
    assert!(
        !open_and_close("libnvToolsExt.so"),
        "precondition: libnvToolsExt.so is resolvable"
    );
    let loaded = catch_unwind(AssertUnwindSafe(|| unsafe {
        cudarc::nvtx::sys::culib();
    }));
    let m = print_maps("f", "libnvToolsExt");
    assert!(loaded.is_ok(), "NVTX did not load");
    assert_eq!(m.len(), 1, "expected one NVTX mapping, got {m:?}");
}
