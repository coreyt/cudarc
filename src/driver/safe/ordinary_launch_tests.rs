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
    lifetime_slice_cleanup_tests::{seams as allocation_seams, tracked_context},
    lifetime_stream_cleanup_tests::seams as stream_seams,
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, LaunchConfig,
    PrimaryContextLifetime, PushKernelArg,
};
use std::sync::{Arc, Mutex};

pub(crate) mod seams {
    use crate::driver::{sys, DriverError};
    use core::ffi::{c_uint, c_void};
    use std::cell::{Cell, RefCell};
    #[derive(Clone, Copy, Default)]
    struct State {
        launches: usize,
        error: Option<sys::CUresult>,
    }
    std::thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
        static HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
    }
    pub(crate) struct Guard;
    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            STATE.set(Some(State::default()));
            Self
        }
        pub(crate) fn set_hook(&self, hook: impl FnMut() + 'static) {
            HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn set_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.error = error;
            STATE.set(Some(state));
        }
        pub(crate) fn launches(&self) -> usize {
            STATE.get().unwrap().launches
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            HOOK.with(|slot| *slot.borrow_mut() = None);
            STATE.set(None);
        }
    }
    pub(crate) fn launch(
        function: sys::CUfunction,
        grid: (c_uint, c_uint, c_uint),
        block: (c_uint, c_uint, c_uint),
        shared: c_uint,
        stream: sys::CUstream,
        parameters: &mut [*mut c_void],
    ) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        assert!(function as usize > 0xF100);
        assert_eq!(grid, (1, 1, 1));
        assert_eq!(block, (1024, 1, 1));
        assert_eq!(shared, 0);
        assert!(stream.is_null());
        assert_eq!(parameters.len(), 1);
        let pointer = unsafe { *parameters[0].cast::<sys::CUdeviceptr>() };
        assert!(pointer >= 0xA101 && pointer <= 0xA109);
        state.launches += 1;
        STATE.set(Some(state));
        HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook();
            }
        });
        Some(match state.error {
            Some(error) => Err(DriverError(error)),
            None => Ok(()),
        })
    }
}

const IMAGE: &str =
    ".version 6.0\n.target sm_52\n.address_size 64\n.visible .entry entry() { ret; }\n";

fn record_phase(lifetime: &PrimaryContextLifetime, phases: &Mutex<Vec<(u8, usize)>>, phase: u8) {
    let active = lifetime
        .state
        .try_lock()
        .expect("native launch boundary must not hold the lifetime mutex")
        .active_leases;
    phases.lock().unwrap().push((phase, active));
}

fn check_launch_epoch(outer: bool) {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let modules = module_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let streams = stream_seams::Guard::new();
    let kernels = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let background = context
        .new_stream()
        .map_err(|error| error.0 as u32)
        .unwrap();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|error| error.0 as u32)
        .unwrap();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let outer_lease = if outer {
        Some(lifetime.admit().map_err(|error| error.0 as u32).unwrap())
    } else {
        None
    };
    let phases = Arc::new(Mutex::new(Vec::new()));
    let weak = Arc::downgrade(&lifetime);
    let log = Arc::clone(&phases);
    streams.set_wait_hook(move || record_phase(&weak.upgrade().unwrap(), &log, 0));
    let weak = Arc::downgrade(&lifetime);
    let log = Arc::clone(&phases);
    kernels.set_hook(move || record_phase(&weak.upgrade().unwrap(), &log, 1));
    let weak = Arc::downgrade(&lifetime);
    let log = Arc::clone(&phases);
    events.set_record_hook(move || record_phase(&weak.upgrade().unwrap(), &log, 2));
    let before = driver.snapshot();

    let outcome = {
        let mut view = allocation.slice_mut(..);
        let mut arguments = stream.launch_builder(&function);
        arguments.arg(&mut view);
        unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }
            .map(drop)
            .map_err(|error| error.0 as u32)
    };

    streams.set_wait_hook(|| {});
    kernels.set_hook(|| {});
    events.set_record_hook(|| {});
    assert_eq!(outcome, Ok(()));
    let extra = usize::from(outer);
    assert_eq!(
        *phases.lock().unwrap(),
        vec![
            (0, 2 + extra),
            (0, 2 + extra),
            (1, 1 + extra),
            (2, 2 + extra)
        ],
        "one launch admission must span wait, native kernel and recording"
    );
    assert_eq!(kernels.launches(), 1);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, extra);
    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (
            before.primary_queries + usize::from(!outer),
            before.id_queries + usize::from(!outer)
        )
    );
    drop(outer_lease);
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
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
fn ordinary_launch_holds_one_epoch_through_wait_kernel_and_record() {
    check_launch_epoch(false);
}
#[test]
fn ordinary_launch_reuses_outer_epoch_without_new_primary_probes() {
    check_launch_epoch(true);
}

#[test]
fn known_loss_refuses_ordinary_launch_before_native_work() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let _modules = module_seams::Guard::new();
    let _memory = allocation_seams::Guard::new();
    let _streams = stream_seams::Guard::new();
    let kernels = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let module = context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|error| error.0 as u32)
        .unwrap();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    let before = driver.snapshot();
    let outcome = {
        let mut view = allocation.slice_mut(..);
        let mut arguments = stream.launch_builder(&function);
        arguments.arg(&mut view);
        unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }
            .map(drop)
            .map_err(|error| error.0 as u32)
    };
    assert_eq!(
        outcome,
        Err(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32)
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
    drop(allocation);
    drop(function);
    drop(module);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

fn check_function_stream_association(
    function_tracked: bool,
    stream_tracked: bool,
    function_lost: bool,
) {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let _modules = module_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let kernels = seams::Guard::new();
    let function_lifetime = if function_tracked {
        Some(
            PrimaryContextLifetime::new(0)
                .map_err(|error| error.0 as u32)
                .unwrap(),
        )
    } else {
        None
    };
    let stream_lifetime = if stream_tracked {
        Some(
            PrimaryContextLifetime::new(0)
                .map_err(|error| error.0 as u32)
                .unwrap(),
        )
    } else {
        None
    };
    let function_context = match &function_lifetime {
        Some(lifetime) => CudaContext::new_tracked(Arc::clone(lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap(),
        None => CudaContext::new(0).map_err(|error| error.0 as u32).unwrap(),
    };
    let stream_context = match &stream_lifetime {
        Some(lifetime) => tracked_context(lifetime, false),
        None => {
            let mut context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
            Arc::get_mut(&mut context).unwrap().has_async_alloc = false;
            context
        }
    };
    assert_eq!(function_context.cu_ctx(), stream_context.cu_ctx());
    let module = function_context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = stream_context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    if function_lost {
        function_lifetime
            .as_ref()
            .unwrap()
            .record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    }
    let before = driver.snapshot();
    let expected_function_loss = function_lifetime
        .as_ref()
        .and_then(|lifetime| lifetime.loss());

    let outcome = {
        let mut view = allocation.slice_mut(..);
        let mut arguments = stream.launch_builder(&function);
        arguments.arg(&mut view);
        unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }
            .map(drop)
            .map_err(|error| error.0 as u32)
    };

    let tracked = function_tracked || stream_tracked;
    assert_eq!(
        outcome,
        if tracked {
            Err(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT as u32)
        } else {
            Ok(())
        }
    );
    assert_eq!(kernels.launches(), usize::from(!tracked));
    if tracked {
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
    }
    assert_eq!(
        function_lifetime
            .as_ref()
            .and_then(|lifetime| lifetime.loss()),
        expected_function_loss
    );
    assert!(stream_lifetime
        .as_ref()
        .and_then(|lifetime| lifetime.loss())
        .is_none());
    assert_eq!(
        stream_context.check_err().map_err(|error| error.0 as u32),
        Ok(())
    );
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(function);
    drop(module);
    drop(stream);
    drop(function_context);
    drop(stream_context);
    drop(function_lifetime);
    drop(stream_lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (2, 2)
    );
}

#[test]
fn equal_native_handles_do_not_allow_distinct_tracked_function_stream_lifetimes() {
    check_function_stream_association(true, true, false);
    check_function_stream_association(true, true, true);
}
#[test]
fn missing_function_or_stream_lifetime_refuses_before_native_entry() {
    check_function_stream_association(true, false, false);
    check_function_stream_association(false, true, false);
}
#[test]
fn wholly_untracked_function_and_stream_keep_legacy_launch_contract() {
    check_function_stream_association(false, false, false);
}

#[test]
fn distinct_context_wrappers_sharing_one_lifetime_can_launch() {
    let driver = context_test_seams::Guard::new();
    let _events = event_seams::Guard::new();
    let modules = module_seams::Guard::new();
    let memory = allocation_seams::Guard::new();
    let kernels = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let function_context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream_context = tracked_context(&lifetime, false);
    assert!(!Arc::ptr_eq(&function_context, &stream_context));
    let module = function_context
        .load_module(crate::nvrtc::Ptx::from_src(IMAGE))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let function = module
        .load_function("entry")
        .map_err(|error| error.0 as u32)
        .unwrap();
    let stream = stream_context.default_stream();
    let mut allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    let outcome = {
        let mut view = allocation.slice_mut(..);
        let mut arguments = stream.launch_builder(&function);
        arguments.arg(&mut view);
        unsafe { arguments.launch(LaunchConfig::for_num_elems(1)) }
            .map(drop)
            .map_err(|error| error.0 as u32)
    };
    assert_eq!(outcome, Ok(()));
    assert_eq!(kernels.launches(), 1);
    assert!(lifetime.loss().is_none());
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    drop(function);
    drop(module);
    assert_eq!(modules.snapshot().unloaded, 1);
    drop(stream);
    drop(function_context);
    drop(stream_context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
