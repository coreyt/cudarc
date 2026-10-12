//! The loader functions at the end of every generated `sys/mod.rs`.

use proc_macro2::TokenStream;
use quote::quote;

/// The loader tail for a module that loads `lib_names`.
pub fn loader_tail(lib_names: &[String]) -> TokenStream {
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
