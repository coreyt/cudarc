//! The loader functions at the end of every generated `sys/mod.rs`.

use proc_macro2::TokenStream;
use quote::quote;

/// The libraries loaded by FATHOMDB.md item 9's rules (`src/lib.rs`,
/// `lib_major::IN_SCOPE`).
const CHECKED: [&str; 11] = [
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

/// The loader tail for a module that loads `lib_names`: item 9's checked
/// loader for the CUDA toolkit libraries, the upstream loader plus a report
/// for the driver, and the upstream loader for everything else.
pub fn loader_tail(lib_names: &[String]) -> TokenStream {
    match lib_names.first().map(String::as_str) {
        Some("cuda") => driver_tail(lib_names),
        Some(lib) if CHECKED.contains(&lib) => checked_tail(lib),
        _ => base_tail(lib_names),
    }
}

fn checked_tail(lib: &str) -> TokenStream {
    quote! {
        #[cfg(feature = "dynamic-loading")]
        static CULIB: crate::lib_major::LibCell<::libloading::Library> = crate::lib_major::LibCell::new();

        #[doc = " Loads the library for the built CUDA major, once (FATHOMDB.md item 9). Candidates come from [`crate::get_lib_name_candidates`]; each is version-checked, and a failure is memoized and returned again by later calls."]
        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn try_culib() -> Result<&'static ::libloading::Library, crate::LibLoadError> {
            CULIB.try_get(|| crate::lib_major::load_culib(#lib))
        }

        #[doc = " Whether [`try_culib`] succeeds. Performs (and keeps) the load."]
        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn is_culib_present() -> bool {
            CULIB.is_present(|| crate::lib_major::load_culib(#lib))
        }

        #[doc = " [`try_culib`], panicking with the [`crate::LibLoadError`] message if the library cannot be loaded."]
        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn culib() -> &'static ::libloading::Library {
            try_culib().unwrap_or_else(|e| crate::lib_major::panic_lib_load_error(&e))
        }

        #[doc = " Whether the loaded library exports the symbol `name`, or why the library cannot be loaded."]
        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn is_symbol_present(name: &str) -> Result<bool, crate::LibLoadError> {
            Ok(try_culib()?
                .get::<*const ::core::ffi::c_void>(name.as_bytes())
                .is_ok())
        }

        #[doc = " What was loaded, after a successful load; \"linked at build time\" under `dynamic-linking` or `static-linking`."]
        pub fn loaded_library() -> Option<crate::LoadedLibrary> {
            #[cfg(feature = "dynamic-loading")]
            {
                CULIB.report()
            }
            #[cfg(not(feature = "dynamic-loading"))]
            {
                Some(crate::lib_major::linked_at_build_time(#lib))
            }
        }
    }
}

fn driver_tail(lib_names: &[String]) -> TokenStream {
    quote! {
        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn is_culib_present() -> bool {
            let lib_names = [#(#lib_names),*];
            let choices = lib_names
                .iter()
                .map(|l| crate::get_lib_name_candidates(l))
                .flatten();
            for choice in choices {
                if ::libloading::Library::new(choice).is_ok() {
                    return true;
                }
            }
            false
        }

        #[cfg(feature = "dynamic-loading")]
        static CULIB: OnceLock<(::libloading::Library, std::string::String)> = OnceLock::new();

        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn culib() -> &'static ::libloading::Library {
            &CULIB
                .get_or_init(|| {
                    let lib_names = std::vec![#(#lib_names),*];
                    let choices: std::vec::Vec<_> = lib_names
                        .iter()
                        .map(|l| crate::get_lib_name_candidates(l))
                        .flatten()
                        .collect();
                    for choice in choices.iter() {
                        if let Ok(lib) = ::libloading::Library::new(choice) {
                            return (lib, choice.clone());
                        }
                    }
                    crate::panic_no_lib_found(lib_names[0], &choices);
                })
                .0
        }

        #[doc = " The driver's file and `cuDriverGetVersion`, after it is loaded; \"linked at build time\" under `dynamic-linking` or `static-linking`."]
        pub fn loaded_library() -> Option<crate::LoadedLibrary> {
            #[cfg(feature = "dynamic-loading")]
            {
                let (lib, name) = CULIB.get()?;
                Some(unsafe { crate::lib_major::driver_report(lib, name) })
            }
            #[cfg(not(feature = "dynamic-loading"))]
            {
                Some(crate::lib_major::linked_at_build_time("cuda"))
            }
        }
    }
}

fn base_tail(lib_names: &[String]) -> TokenStream {
    quote! {
        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn is_culib_present() -> bool {
            let lib_names = [#(#lib_names),*];
            let choices = lib_names
                .iter()
                .map(|l| crate::get_lib_name_candidates(l))
                .flatten();
            for choice in choices {
                if ::libloading::Library::new(choice).is_ok() {
                    return true;
                }
            }
            false
        }

        #[cfg(feature = "dynamic-loading")]
        pub unsafe fn culib() -> &'static ::libloading::Library {
            static LIB: OnceLock<::libloading::Library> = OnceLock::new();
            LIB.get_or_init(|| {
                let lib_names = std::vec![#(#lib_names),*];
                let choices: std::vec::Vec<_> = lib_names
                    .iter()
                    .map(|l| crate::get_lib_name_candidates(l))
                    .flatten()
                    .collect();
                for choice in choices.iter() {
                    if let Ok(lib) = ::libloading::Library::new(choice) {
                        return lib;
                    }
                }
                crate::panic_no_lib_found(lib_names[0], &choices);
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::loader_tail;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// The `bindings_generator` directory. `file!()` is relative to the
    /// manifest directory in a cargo build, or absolute when the module is
    /// included by path from elsewhere.
    fn generator_dir() -> PathBuf {
        let here = Path::new(env!("CARGO_MANIFEST_DIR")).join(file!());
        here.parent().unwrap().parent().unwrap().to_path_buf()
    }

    /// Formats `tokens` as `merge::merge` does (rustfmt with
    /// bindings-fmt.toml).
    fn render(tokens: &str) -> String {
        let dir = generator_dir();
        let path = std::env::temp_dir().join(format!("loader-tail-{}.rs", std::process::id()));
        std::fs::write(&path, tokens).unwrap();
        let status = Command::new("rustfmt")
            .arg("--config-path")
            .arg(dir.join("bindings-fmt.toml"))
            .arg(&path)
            .status()
            .expect("run rustfmt");
        assert!(status.success(), "rustfmt failed");
        let out = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        out
    }

    #[test]
    fn template_loader_tail_matches_the_checked_in_cublas_loader() {
        let rendered = render(&loader_tail(&["cublas".to_string()]).to_string());
        let file = generator_dir().join("../src/cublas/sys/mod.rs");
        let checked_in = std::fs::read_to_string(&file).unwrap();
        assert!(
            checked_in.contains(rendered.trim()),
            "the template's cuBLAS loader tail is not in {}:\n{rendered}",
            file.display()
        );
    }
}
