#![cfg(all(
    feature = "nvrtc",
    not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))
))]

use super::lifetime_module_cleanup_tests::seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

const IMAGE: &str =
    ".version 6.0\n.target sm_52\n.address_size 64\n.visible .entry entry() { ret; }\n";

fn assert_admitted(lifetime: &PrimaryContextLifetime) {
    let state = lifetime
        .state
        .try_lock()
        .expect("native module entry must run outside lifetime mutex");
    assert_eq!(
        state.active_leases, 1,
        "native module/function entry must retain admission"
    );
    assert!(!state.probing);
}

#[test]
fn module_construction_holds_admission_through_native_load() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    modules.set_load_hook(move || assert_admitted(&weak.upgrade().unwrap()));

    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();

    assert_eq!(
        (modules.snapshot().load_attempts, modules.snapshot().loaded),
        (1, 1)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn function_lookup_holds_admission_and_binds_before_native_entry() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    modules.set_function_hook(move || assert_admitted(&weak.upgrade().unwrap()));
    driver.clear_current();
    let before = driver.snapshot();

    let function = module
        .load_function("entry")
        .map_err(|error| error.0 as u32)
        .unwrap();

    let after = driver.snapshot();
    assert_eq!(
        (after.queries, after.binds),
        (before.queries + 1, before.binds + 1)
    );
    assert_eq!(
        (
            modules.snapshot().function_attempts,
            modules.snapshot().functions
        ),
        (1, 1)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(function);
    assert_eq!(modules.snapshot().unloaded, 0);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn known_loss_prevents_module_growth_before_binding_or_load() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();

    assert_eq!(
        context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map(drop)
            .map_err(|error| error.0 as u32),
        Err(lost as u32)
    );

    assert_eq!(modules.snapshot().load_attempts, 0);
    let after = driver.snapshot();
    assert_eq!(
        (
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            before.primary_queries,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn ordinary_module_and_function_native_oom_remain_retryable() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    modules.set_load_error(Some(oom));
    assert_eq!(
        context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map(drop)
            .map_err(|error| error.0 as u32),
        Err(oom as u32)
    );
    assert_eq!(modules.snapshot().loaded, 0);
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    modules.set_load_error(None);
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();
    modules.set_function_error(Some(oom));
    assert_eq!(
        module
            .load_function("entry")
            .map(drop)
            .map_err(|error| error.0 as u32),
        Err(oom as u32)
    );
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    modules.set_function_error(None);
    let function = module
        .load_function("entry")
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(
        (
            modules.snapshot().load_attempts,
            modules.snapshot().function_attempts
        ),
        (2, 2)
    );
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
