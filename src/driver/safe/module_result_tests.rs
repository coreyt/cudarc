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
    PrimaryContextLossCause, PrimaryContextRefusal,
};
use std::sync::Arc;
const IMAGE: &str =
    ".version 6.0\n.target sm_52\n.address_size 64\n.visible .entry entry() { ret; }\n";

fn check_native_entry(function: bool, error: sys::CUresult) {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let module = if function {
        Some(
            context
                .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
                .map_err(|e| e.0 as u32)
                .unwrap(),
        )
    } else {
        None
    };
    let outcome = if let Some(module) = &module {
        modules.set_function_error(Some(error));
        module
            .load_function("entry")
            .map(drop)
            .map_err(|e| e.0 as u32)
    } else {
        modules.set_load_error(Some(error));
        context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map(drop)
            .map_err(|e| e.0 as u32)
    };
    modules.set_load_error(None);
    modules.set_function_error(None);
    assert_eq!(outcome, Err(error as u32));
    assert_eq!(
        lifetime
            .loss()
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
        Some((17, None, PrimaryContextLossCause::DriverStatus(error))),
        "native module entry failure must publish its actual terminal status before return"
    );
    let first = lifetime.loss();
    assert_eq!(
        first.unwrap().refusal(),
        PrimaryContextRefusal::DriverStatus(error)
    );
    let stopped = driver.snapshot();
    let attempts = (
        modules.snapshot().load_attempts,
        modules.snapshot().function_attempts,
    );
    for _ in 0..2 {
        assert_eq!(
            context.check_err().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            sibling.check_err().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
    }
    assert_eq!(
        sibling.bind_to_thread().map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!(
        context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    if let Some(module) = &module {
        assert_eq!(
            module
                .load_function("entry")
                .map(drop)
                .map_err(|e| e.0 as u32),
            Err(error as u32)
        );
    }
    sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
    assert_eq!(lifetime.loss(), first);
    assert_eq!(
        (
            modules.snapshot().load_attempts,
            modules.snapshot().function_attempts
        ),
        attempts
    );
    let after = driver.snapshot();
    assert_eq!(
        (
            after.primary_queries,
            after.id_queries,
            after.queries,
            after.binds
        ),
        (
            stopped.primary_queries,
            stopped.id_queries,
            stopped.queries,
            stopped.binds
        )
    );
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 0);
    drop(context);
    drop(sibling);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn native_module_load_709_publishes_first_loss_before_return() {
    check_native_entry(false, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn native_module_load_201_publishes_first_loss_before_return() {
    check_native_entry(false, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

#[test]
fn native_module_function_709_publishes_first_loss_before_return() {
    check_native_entry(true, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}

#[test]
fn native_module_function_201_publishes_first_loss_before_return() {
    check_native_entry(true, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}

#[test]
fn native_unload_failure_freezes_generation_before_later_module_growth() {
    for error in [
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
        sys::CUresult::CUDA_ERROR_UNKNOWN,
    ] {
        let driver = context_test_seams::Guard::new();
        let modules = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|e| e.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|e| e.0 as u32)
            .unwrap();
        let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|e| e.0 as u32)
            .unwrap();
        let module = context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map_err(|e| e.0 as u32)
            .unwrap();
        let function = module
            .load_function("entry")
            .map_err(|e| e.0 as u32)
            .unwrap();
        let last = function.clone();
        drop(module);
        drop(function);
        assert_eq!(modules.snapshot().unloaded, 0);
        modules.set_unload_error(Some(error));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(last)));
        modules.set_unload_error(None);
        assert!(result.is_ok());
        assert_eq!(modules.snapshot().unloaded, 1);
        assert_eq!(
            lifetime
                .loss()
                .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
            Some((17, None, PrimaryContextLossCause::DriverStatus(error))),
            "actual unload failure must freeze the generation after one attempt"
        );
        let first = lifetime.loss();
        let stopped = driver.snapshot();
        for _ in 0..2 {
            assert_eq!(
                context.check_err().map_err(|e| e.0 as u32),
                Err(error as u32)
            );
            assert_eq!(
                sibling.check_err().map_err(|e| e.0 as u32),
                Err(error as u32)
            );
        }
        assert_eq!(
            sibling
                .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
                .map(drop)
                .map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(
            sibling.bind_to_thread().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        assert_eq!(lifetime.loss(), first);
        assert_eq!(
            (modules.snapshot().loaded, modules.snapshot().unloaded),
            (1, 1)
        );
        let after = driver.snapshot();
        assert_eq!(
            (
                after.primary_queries,
                after.id_queries,
                after.queries,
                after.binds
            ),
            (
                stopped.primary_queries,
                stopped.id_queries,
                stopped.queries,
                stopped.binds
            )
        );
        drop(context);
        drop(sibling);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn ordinary_module_entry_oom_remains_retryable_without_delayed_error() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    modules.set_load_error(Some(oom));
    assert_eq!(
        context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(oom as u32)
    );
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
    modules.set_load_error(None);
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|e| e.0 as u32)
        .unwrap();
    modules.set_function_error(Some(oom));
    assert_eq!(
        module
            .load_function("entry")
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(oom as u32)
    );
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
    modules.set_function_error(None);
    let function = module
        .load_function("entry")
        .map_err(|e| e.0 as u32)
        .unwrap();
    drop(function);
    drop(module);
    assert_eq!(
        (
            modules.snapshot().loaded,
            modules.snapshot().functions,
            modules.snapshot().unloaded
        ),
        (1, 1, 1)
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn untracked_native_module_errors_keep_legacy_return_and_drop_contract() {
    let driver = context_test_seams::Guard::new();
    let modules = seams::Guard::new();
    let context = CudaContext::new(0).map_err(|e| e.0 as u32).unwrap();
    for error in [
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
    ] {
        modules.set_load_error(Some(error));
        assert_eq!(
            context
                .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
                .map(drop)
                .map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
        modules.set_load_error(None);
        let module = context
            .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
            .map_err(|e| e.0 as u32)
            .unwrap();
        modules.set_function_error(Some(error));
        assert_eq!(
            module
                .load_function("entry")
                .map(drop)
                .map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
        modules.set_function_error(None);
        modules.set_unload_error(Some(error));
        drop(module);
        modules.set_unload_error(None);
        assert_eq!(
            context.check_err().map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
    }
    assert_eq!(modules.snapshot().unloaded, 2);
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
