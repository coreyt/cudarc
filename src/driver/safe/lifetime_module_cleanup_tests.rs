use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime};
use std::sync::Arc;

pub(crate) mod seams {
    use crate::driver::{sys, DriverError};
    use core::cell::{Cell, RefCell};
    use core::ffi::{c_void, CStr};

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Counts {
        pub(crate) load_attempts: usize,
        pub(crate) loaded: usize,
        pub(crate) unloaded: usize,
        pub(crate) function_attempts: usize,
        pub(crate) functions: usize,
    }

    std::thread_local! {
        static COUNTS: Cell<Option<Counts>> = const { Cell::new(None) };
        static UNLOAD_HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
        static LOAD_HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
        static FUNCTION_HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
        static LOAD_ERROR: Cell<Option<sys::CUresult>> = const { Cell::new(None) };
        static FUNCTION_ERROR: Cell<Option<sys::CUresult>> = const { Cell::new(None) };
        static UNLOAD_ERROR: Cell<Option<sys::CUresult>> = const { Cell::new(None) };
    }

    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(COUNTS.get().is_none());
            COUNTS.set(Some(Counts::default()));
            Self
        }

        pub(crate) fn set_load_hook(&self, hook: impl FnMut() + 'static) {
            LOAD_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn set_function_hook(&self, hook: impl FnMut() + 'static) {
            FUNCTION_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn set_load_error(&self, error: Option<sys::CUresult>) { LOAD_ERROR.set(error); }
        pub(crate) fn set_function_error(&self, error: Option<sys::CUresult>) { FUNCTION_ERROR.set(error); }
        pub(crate) fn set_unload_error(&self, error: Option<sys::CUresult>) { UNLOAD_ERROR.set(error); }

        pub(crate) fn set_unload_hook(&self, hook: impl FnMut() + 'static) {
            UNLOAD_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }

        pub(crate) fn snapshot(&self) -> Counts {
            COUNTS.get().unwrap()
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            UNLOAD_HOOK.with(|slot| *slot.borrow_mut() = None);
            LOAD_HOOK.with(|slot| *slot.borrow_mut() = None);
            FUNCTION_HOOK.with(|slot| *slot.borrow_mut() = None);
            LOAD_ERROR.set(None);
            FUNCTION_ERROR.set(None);
            UNLOAD_ERROR.set(None);
            COUNTS.set(None);
        }
    }

    pub(crate) fn load_data(image: *const c_void) -> Option<Result<sys::CUmodule, DriverError>> {
        let mut counts = COUNTS.get()?;
        assert!(!image.is_null());
        counts.load_attempts += 1;
        COUNTS.set(Some(counts));
        LOAD_HOOK.with(|slot| { if let Some(hook) = slot.borrow_mut().as_mut() { hook(); } });
        if let Some(error) = LOAD_ERROR.get() { return Some(Err(DriverError(error))); }
        counts.loaded += 1;
        COUNTS.set(Some(counts));
        Some(Ok((0x4D00 + counts.loaded) as sys::CUmodule))
    }

    pub(crate) fn get_function(module: sys::CUmodule, name: &CStr) -> Option<Result<sys::CUfunction, DriverError>> {
        let mut counts = COUNTS.get()?;
        assert!(module as usize > 0x4D00 && module as usize <= 0x4D00 + counts.loaded);
        assert_eq!(name.to_bytes(), b"entry");
        counts.function_attempts += 1;
        COUNTS.set(Some(counts));
        FUNCTION_HOOK.with(|slot| { if let Some(hook) = slot.borrow_mut().as_mut() { hook(); } });
        if let Some(error) = FUNCTION_ERROR.get() { return Some(Err(DriverError(error))); }
        counts.functions += 1;
        COUNTS.set(Some(counts));
        Some(Ok((0xF100 + counts.functions) as sys::CUfunction))
    }

    pub(crate) fn unload(module: sys::CUmodule) -> Option<Result<(), DriverError>> {
        let mut counts = COUNTS.get()?;
        assert!(module as usize > 0x4D00 && module as usize <= 0x4D00 + counts.loaded);
        counts.unloaded += 1;
        COUNTS.set(Some(counts));
        UNLOAD_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() { hook(); }
        });
        Some(match UNLOAD_ERROR.get() { Some(error) => Err(DriverError(error)), None => Ok(()) })
    }
}

fn tracked_context(lifetime: &Arc<PrimaryContextLifetime>) -> Arc<CudaContext> {
    CudaContext::new_tracked(Arc::clone(lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap()
}

const IMAGE: &str = ".version 6.0\n.target sm_52\n.address_size 64\n.visible .entry entry() { ret; }\n";

#[cfg(all(feature = "nvrtc", not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))))]
#[test]
fn lost_last_function_clone_disarms_module_unload() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let module = context.load_module(crate::nvrtc::Ptx::from_src(IMAGE)).map_err(|error| error.0 as u32).unwrap();
    let function = module.load_function("entry").map_err(|error| error.0 as u32).unwrap();
    let last = function.clone();
    assert_eq!((modules.snapshot().loaded, modules.snapshot().functions), (1, 1));
    drop(module);
    drop(function);
    assert_eq!(modules.snapshot().unloaded, 0);
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    let before = driver.snapshot();

    drop(last);

    assert_eq!(modules.snapshot().unloaded, 0, "the last lost function clone must disarm native module unload");
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries, before.id_queries));
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(all(feature = "nvrtc", not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))))]
#[test]
fn healthy_last_function_clone_unloads_its_counted_module_once() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let module = context.load_module(crate::nvrtc::Ptx::from_src(IMAGE)).map_err(|error| error.0 as u32).unwrap();
    let function = module.load_function("entry").map_err(|error| error.0 as u32).unwrap();
    let last = function.clone();
    drop(module);
    drop(function);
    assert_eq!(modules.snapshot().unloaded, 0);

    let weak = Arc::downgrade(&lifetime);
    modules.set_unload_hook(move || {
        let lifetime = weak.upgrade().unwrap();
        let state = lifetime.state.try_lock().expect("native unload must run outside lifetime mutex");
        assert_eq!(state.active_leases, 1, "native module cleanup must retain admission");
    });

    drop(last);

    assert_eq!((modules.snapshot().loaded, modules.snapshot().functions, modules.snapshot().unloaded), (1, 1, 1));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(all(feature = "nvrtc", not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))))]
#[test]
fn lost_module_refuses_function_lookup_without_native_dispatch() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let module = context.load_module(crate::nvrtc::Ptx::from_src(IMAGE)).map_err(|error| error.0 as u32).unwrap();
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    let before = driver.snapshot();

    let outcome = module.load_function("entry").map(|_| ()).map_err(|error| error.0 as u32);

    assert_eq!(outcome, Err(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32));
    assert_eq!(modules.snapshot().functions, 0);
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries, before.id_queries));
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    drop(module);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(all(feature = "nvrtc", not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
))))]
#[test]
fn idle_module_drop_discovers_inactive_primary_before_unload() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let module = context.load_module(crate::nvrtc::Ptx::from_src(IMAGE)).map_err(|error| error.0 as u32).unwrap();
    let before = driver.snapshot();
    driver.set_primary_active(false);

    drop(module);

    assert_eq!((modules.snapshot().loaded, modules.snapshot().unloaded), (1, 0));
    assert_eq!(lifetime.loss().unwrap().cause, crate::driver::PrimaryContextLossCause::PrimaryInactive);
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries + 1, before.id_queries));
    assert_eq!((after.queries, after.binds), (before.queries, before.binds));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(all(feature = "nvrtc", not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
))))]
#[test]
fn native_module_cleanup_binding_failure_quarantines_before_unload() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let module = context.load_module(crate::nvrtc::Ptx::from_src(IMAGE)).map_err(|error| error.0 as u32).unwrap();
    driver.set_current_query_error(Some(sys::CUresult::CUDA_ERROR_UNKNOWN));

    drop(module);

    assert_eq!((modules.snapshot().loaded, modules.snapshot().unloaded), (1, 0));
    assert_eq!(lifetime.loss().unwrap().cause, crate::driver::PrimaryContextLossCause::DriverStatus(sys::CUresult::CUDA_ERROR_UNKNOWN));
    driver.set_current_query_error(None);
    let stopped = driver.snapshot();
    assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Err(sys::CUresult::CUDA_ERROR_UNKNOWN as u32));
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries, after.queries, after.binds), (stopped.primary_queries, stopped.id_queries, stopped.queries, stopped.binds));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[cfg(all(feature = "nvrtc", not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
))))]
#[test]
fn preexisting_ordinary_oom_does_not_abandon_healthy_module_cleanup() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = tracked_context(&lifetime);
    let module = context.load_module(crate::nvrtc::Ptx::from_src(IMAGE)).map_err(|error| error.0 as u32).unwrap();
    context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));

    drop(module);

    assert_eq!((modules.snapshot().loaded, modules.snapshot().unloaded), (1, 1));
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32));
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}
