use super::*;
use crate::driver::{result::context_test_seams, safe::{lifetime_event_cleanup_tests::seams as event_seams, lifetime_stream_cleanup_tests::seams as stream_seams}, sys as driver_sys, CudaContext, DriverError, PrimaryContextLifetime};
use crate::driver::{CudaEvent, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use core::cell::Cell;

struct Operand {
    stream: Arc<CudaStream>,
    prepared: Cell<usize>,
    wait: Option<CudaEvent>,
}

impl Operand {
    fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream, prepared: Cell::new(0), wait: None }
    }
    fn prepare(&self, stream: &CudaStream) {
        self.prepared.set(self.prepared.get() + 1);
        if let Some(event) = &self.wait {
            stream.context().record_err(stream.wait(event));
        }
    }
}

impl DeviceSlice<f32> for Operand {
    fn len(&self) -> usize { 4 }
    fn stream(&self) -> &Arc<CudaStream> { &self.stream }
}
impl DevicePtr<f32> for Operand {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (driver_sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.prepare(stream);
        (0x1230, SyncOnDrop::Record(None))
    }
}
impl DevicePtrMut<f32> for Operand {
    fn device_ptr_mut<'a>(&'a mut self, stream: &'a CudaStream) -> (driver_sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.prepare(stream);
        (0x1230, SyncOnDrop::Record(None))
    }
}

fn fill(owner: &CudaRng, kind: usize, dst: &mut Operand) -> Result<(), result::CurandError> {
    match kind { 0 => owner.fill_with_uniform(dst), 1 => owner.fill_with_normal(dst, 0., 1.), _ => owner.fill_with_log_normal(dst, 0., 1.) }
}

#[test]
fn safe_use_refuses_cached_loss_before_pointer_or_library_calls() {
    let driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    let mut owner = CudaRng::new(42, stream.clone()).unwrap();
    let mut dst = Operand::new(stream);
    driver.set_primary_active(false);
    assert!(lifetime.admit().is_err());
    let refused = Err(result::CurandError::Lost(lifetime.loss().unwrap()));

    assert_eq!(owner.set_seed(42), refused);
    assert_eq!(owner.set_offset(0), refused);
    for kind in 0..3 { assert_eq!(fill(&owner, kind, &mut dst), refused); }
    assert_eq!(dst.prepared.get(), 0);
    assert_eq!((calls.snapshot().seed_sets, calls.snapshot().offset_sets, calls.snapshot().fills), (1, 0, 0));
}

#[test]
fn safe_fills_recheck_native_wait_failure_before_library_dispatch() {
    for kind in 0..3 {
        for error in [driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED, driver_sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY] {
            let _driver = context_test_seams::Guard::new();
            let _events = event_seams::Guard::new();
            let streams = stream_seams::Guard::new();
            let calls = result::partial_init_test_seams::Guard::new();
            calls.allow_stream_setup();
            let lifetime = PrimaryContextLifetime::new(0).unwrap();
            let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
            let stream = context.default_stream();
            let owner = CudaRng::new(42, stream.clone()).unwrap();
            let mut dst = Operand::new(stream);
            dst.wait = Some(context.new_event(None).unwrap());
            streams.set_wait_error(Some(error));

            assert_eq!(fill(&owner, kind, &mut dst), Err(result::CurandError::Driver(DriverError(error))));

            assert_eq!(calls.snapshot().fills, 0);
            assert_eq!(streams.snapshot().waits, 1);
            assert_eq!(lifetime.loss().is_some(), error == driver_sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
            if lifetime.loss().is_none() {
                streams.set_wait_error(None);
                assert_eq!(fill(&owner, kind, &mut dst), Ok(()));
                assert_eq!(calls.snapshot().fills, 1);
            }
        }
    }
}

#[test]
fn healthy_safe_fills_preserve_native_error_and_retry() {
    let _driver = context_test_seams::Guard::new();
    let calls = result::partial_init_test_seams::Guard::new();
    calls.allow_stream_setup();
    let lifetime = PrimaryContextLifetime::new(0).unwrap();
    let context = CudaContext::new_tracked(lifetime.clone()).unwrap();
    let stream = context.default_stream();
    let mut owner = CudaRng::new(42, stream.clone()).unwrap();
    let mut dst = Operand::new(stream);
    assert_eq!(owner.set_seed(42), Ok(()));
    assert_eq!(owner.set_offset(0), Ok(()));
    for kind in 0..3 {
        calls.fail_operation(true);
        assert_eq!(fill(&owner, kind, &mut dst), Err(result::CurandError::Native(sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED)));
        assert!(lifetime.loss().is_none());
        calls.fail_operation(false);
        assert_eq!(fill(&owner, kind, &mut dst), Ok(()));
    }
    assert_eq!(calls.snapshot().seed_sets, 2);
    assert_eq!(calls.snapshot().offset_sets, 1);
    assert_eq!(calls.snapshot().fills, 6);
    assert_eq!(dst.prepared.get(), 6);
}
