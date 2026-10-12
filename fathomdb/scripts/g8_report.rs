//! G8 report program (test plan § 3.4). Built by g8-common.sh as a scratch
//! binary that depends on this cudarc tree by path; it is not part of the
//! crate.
//!
//! Usage: g8_report [--preload PATH] [--not-mapped PREFIX] SPEC...
//!
//! SPEC is one of
//!   LIB:ok:DIR:VERSION  try_culib succeeds; the report's path is under DIR
//!                       and mapped (/proc/self/maps); its version starts
//!                       with VERSION
//!   LIB:err[:REASON[:TEXT]]
//!                       try_culib fails; some tried entry has REASON (the
//!                       RejectReason variant name) and contains TEXT
//! LIB: cudart cublas cublasLt nvrtc curand cusparse cusolver cufile.
//! --preload dlopens PATH first (RTLD_NOW | RTLD_GLOBAL) and keeps it.
//! --not-mapped fails if any mapped file of a SPEC's library lies under
//! PREFIX at the end.
//!
//! Prints one REPORT/ERROR line per SPEC and the matching maps lines; exits 1
//! on any failed expectation, never by a panic of the loader.

use cudarc::{LibLoadError, LoadedLibrary};
use std::ffi::{c_char, c_int, c_void, CString};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
}

type Probe = (
    unsafe fn() -> Result<(), LibLoadError>,
    fn() -> Option<LoadedLibrary>,
);

fn probe(lib: &str) -> Probe {
    macro_rules! p {
        ($m:path) => {{
            use $m as m;
            unsafe fn t() -> Result<(), LibLoadError> {
                m::try_culib().map(|_| ())
            }
            (t as unsafe fn() -> Result<(), LibLoadError>, m::loaded_library as fn() -> _)
        }};
    }
    match lib {
        "cudart" => p!(cudarc::runtime::sys),
        "cublas" => p!(cudarc::cublas::sys),
        "cublasLt" => p!(cudarc::cublaslt::sys),
        "nvrtc" => p!(cudarc::nvrtc::sys),
        "curand" => p!(cudarc::curand::sys),
        "cusparse" => p!(cudarc::cusparse::sys),
        "cusolver" => p!(cudarc::cusolver::sys),
        "cufile" => p!(cudarc::cufile::sys),
        _ => panic!("unknown library {lib}"),
    }
}

fn mapped(file_prefix: &str) -> Vec<String> {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    let mut v: Vec<String> = maps
        .lines()
        .filter_map(|l| l.split_whitespace().nth(5))
        .filter(|p| {
            p.rsplit('/')
                .next()
                .is_some_and(|f| f.starts_with(file_prefix))
        })
        .map(str::to_string)
        .collect();
    v.sort();
    v.dedup();
    v
}

fn canonical(p: &str) -> String {
    std::fs::canonicalize(p)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| p.to_string())
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut not_mapped = None;
    let mut ok = true;
    while let Some(a) = args.first().cloned() {
        match a.as_str() {
            "--preload" => {
                let path = CString::new(args[1].clone()).unwrap();
                let h = unsafe { dlopen(path.as_ptr(), 2 | 0x100) };
                println!("PRELOAD {} {}", args[1], if h.is_null() { "failed" } else { "ok" });
                ok &= !h.is_null();
                args.drain(..2);
            }
            "--not-mapped" => {
                not_mapped = Some(canonical(&args[1]));
                args.drain(..2);
            }
            _ => break,
        }
    }
    let mut libs = Vec::new();
    for spec in &args {
        let parts: Vec<&str> = spec.split(':').collect();
        let lib = parts[0];
        libs.push(lib.to_string());
        let (try_culib, report) = probe(lib);
        let result = std::panic::catch_unwind(|| unsafe { try_culib() });
        let Ok(result) = result else {
            println!("FAIL {spec}: the fallible check panicked");
            ok = false;
            continue;
        };
        let file = format!("lib{lib}.so");
        let maps = mapped(&file);
        for m in &maps {
            println!("MAPS {lib} {m}");
        }
        match (parts.get(1).copied(), result) {
            (Some("ok"), Ok(())) => {
                let r = report().expect("a report after a successful load");
                println!("REPORT {r}");
                let dir = canonical(parts.get(2).copied().unwrap_or(""));
                let version = parts.get(3).copied().unwrap_or("");
                let path = r.path.clone().unwrap_or_default();
                let v = r.version.map(|v| v.to_string()).unwrap_or_default();
                let checks = [
                    ("path under the expected directory", path.starts_with(&dir)),
                    ("path in /proc/self/maps", maps.contains(&path)),
                    ("version in the expected range", v.starts_with(version)),
                ];
                for (what, good) in checks {
                    if !good {
                        println!("FAIL {spec}: {what} ({path}, {v})");
                        ok = false;
                    }
                }
            }
            (Some("err"), Err(e)) => {
                println!("ERROR {e}");
                println!("TRIED {:?}", e.tried);
                if let Some(reason) = parts.get(2) {
                    let text = parts.get(3).copied().unwrap_or("");
                    let hit = e.tried.iter().any(|(_, r)| {
                        let d = format!("{r:?}");
                        d.starts_with(reason) && d.contains(text)
                    });
                    if !hit {
                        println!("FAIL {spec}: no tried entry {reason} containing {text:?}");
                        ok = false;
                    }
                }
            }
            (want, Ok(())) => {
                println!("FAIL {spec}: loaded ({:?}), expected {want:?}", report());
                ok = false;
            }
            (want, Err(e)) => {
                println!("FAIL {spec}: {e}, expected {want:?}");
                ok = false;
            }
        }
    }
    if let Some(prefix) = not_mapped {
        for lib in &libs {
            for m in mapped(&format!("lib{lib}.so")) {
                if m.starts_with(&prefix) {
                    println!("FAIL still mapped under {prefix}: {m}");
                    ok = false;
                }
            }
        }
    }
    println!("G8-REPORT {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
