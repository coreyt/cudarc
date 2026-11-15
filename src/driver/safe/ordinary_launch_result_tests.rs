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
use super::{
    lifetime_event_cleanup_tests::seams as event_seams,
    lifetime_module_cleanup_tests::seams as module_seams,
    lifetime_slice_cleanup_tests::{seams as memory_seams, tracked_context},
    lifetime_stream_cleanup_tests::seams as stream_seams,
    ordinary_launch_tests::seams as kernel_seams,
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr,
    DriverError, LaunchConfig, PrimaryContextLifetime, PrimaryContextLossCause, PushKernelArg,
};
use std::sync::Arc;
const IMAGE: &str =
    ".version 6.0\n.target sm_52\n.address_size 64\n.visible .entry entry() { ret; }\n";
fn run(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    allocation: &mut CudaSlice<u8>,
) -> Result<(), DriverError> {
    let mut view = allocation.slice_mut(..);
    let mut arguments = stream.launch_builder(function);
    arguments.arg(&mut view);
    unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }.map(drop)
}
fn check_native_kernel(error: sys::CUresult) {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let modules = module_seams::Guard::new();
    let memory = memory_seams::Guard::new();
    let kernels = kernel_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let stream = context.default_stream();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|e| e.0 as u32)
        .unwrap();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    kernels.set_error(Some(error));
    let result = run(&stream, &function, &mut allocation).map_err(|e| e.0 as u32);
    kernels.set_error(None);
    assert_eq!(result, Err(error as u32));
    assert_eq!(
        lifetime
            .loss()
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
        Some((17, None, PrimaryContextLossCause::DriverStatus(error))),
        "actual native kernel terminal result must publish before returning"
    );
    let first = lifetime.loss();
    let before = driver.snapshot();
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
        run(&stream, &function, &mut allocation).map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!(
        sibling.bind_to_thread().map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
    assert_eq!(lifetime.loss(), first);
    assert_eq!(kernels.launches(), 1);
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
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 0);
    assert_eq!(events.snapshot().destroyed, 0);
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 0);
    drop(stream);
    drop(context);
    drop(sibling);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
#[test]
fn native_kernel_709_publishes_shared_first_loss_before_return() {
    check_native_kernel(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
}
#[test]
fn native_kernel_201_publishes_shared_first_loss_before_return() {
    check_native_kernel(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);
}
#[test]
fn ordinary_kernel_oom_is_retryable_without_delayed_error_or_loss() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let modules = module_seams::Guard::new();
    let memory = memory_seams::Guard::new();
    let kernels = kernel_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|e| e.0 as u32)
        .unwrap();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    kernels.set_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    assert_eq!(
        run(&stream, &function, &mut allocation).map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
    );
    assert!(lifetime.loss().is_none());
    assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
    kernels.set_error(None);
    assert_eq!(
        run(&stream, &function, &mut allocation).map_err(|e| e.0 as u32),
        Ok(())
    );
    assert_eq!(kernels.launches(), 2);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
#[test]
fn untracked_kernel_terminal_errors_keep_the_native_return_contract() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let modules = module_seams::Guard::new();
    let memory = memory_seams::Guard::new();
    let kernels = kernel_seams::Guard::new();
    let mut context = CudaContext::new(0).map_err(|e| e.0 as u32).unwrap();
    Arc::get_mut(&mut context).unwrap().has_async_alloc = false;
    let stream = context.default_stream();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|e| e.0 as u32)
        .unwrap();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    for error in [
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
        sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
    ] {
        kernels.set_error(Some(error));
        assert_eq!(
            run(&stream, &function, &mut allocation).map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
    }
    kernels.set_error(None);
    assert_eq!(kernels.launches(), 2);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
    drop(stream);
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

fn check_post_argument_wait(terminal: bool) {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let modules = module_seams::Guard::new();
    let memory = memory_seams::Guard::new();
    let streams = stream_seams::Guard::new();
    let kernels = kernel_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let background = context.new_stream().map_err(|e| e.0 as u32).unwrap();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|e| e.0 as u32)
        .unwrap();
    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|e| e.0 as u32)
        .unwrap();
    let outer = lifetime.admit().map_err(|e| e.0 as u32).unwrap();
    let error = if terminal {
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    } else {
        sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
    };
    {
        let view = allocation.slice(..);
        let mut arguments = stream.launch_builder(&function);
        arguments.arg(&view);
        streams.set_wait_error(Some(error));
        let (_, guard) = view.device_ptr(&stream);
        drop(guard);
        streams.set_wait_error(None);
        let before = driver.snapshot();
        assert_eq!(
            unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }
                .map(drop)
                .map_err(|e| e.0 as u32),
            Err(error as u32)
        );
        assert_eq!(kernels.launches(), 0);
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
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
        if terminal {
            assert_eq!(
                lifetime.loss().unwrap().cause,
                PrimaryContextLossCause::DriverStatus(error)
            );
        } else {
            assert!(lifetime.loss().is_none());
            assert_eq!(context.check_err().map_err(|e| e.0 as u32), Ok(()));
            assert_eq!(
                unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }
                    .map(drop)
                    .map_err(|e| e.0 as u32),
                Ok(())
            );
            assert_eq!(kernels.launches(), 1);
        }
    }
    drop(outer);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, usize::from(!terminal));
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, usize::from(!terminal));
    drop(background);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
#[test]
fn actual_pointer_wait_709_after_arguments_refuses_before_kernel() {
    check_post_argument_wait(true);
}
#[test]
fn actual_pointer_wait_oom_after_arguments_is_returned_once_then_retries() {
    check_post_argument_wait(false);
}
