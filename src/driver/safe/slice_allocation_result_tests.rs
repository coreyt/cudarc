#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::{
    lifetime_event_cleanup_tests::seams as event_seams,
    lifetime_slice_cleanup_tests::{seams, tracked_context},
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::Arc;

#[test]
fn terminal_malloc_result_is_shared_before_any_later_allocation_or_probe() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let stream = context.default_stream();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        memory.set_allocate_error(Some(lost));

        let failed = unsafe { stream.alloc::<u8>(16) }
            .map(drop)
            .map_err(|error| error.0 as u32);

        let stopped = driver.snapshot();
        let first_record = lifetime
            .loss()
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.refusal().driver_status() as u32));
        memory.set_allocate_error(None);
        let first_read = context.check_err().map_err(|error| error.0 as u32);
        let sibling_read = sibling.check_err().map_err(|error| error.0 as u32);
        let later_bind = sibling.bind_to_thread().map_err(|error| error.0 as u32);
        let later_allocation = unsafe { stream.alloc::<u8>(16) }
            .map(drop)
            .map_err(|error| error.0 as u32);
        sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        let after_oom = sibling.check_err().map_err(|error| error.0 as u32);
        let after = driver.snapshot();
        assert_eq!(failed, Err(lost as u32));
        assert_eq!(
            first_record,
            Some((17, None, lost as u32)),
            "native malloc terminal result must publish before returning"
        );
        assert_eq!(first_read, Err(lost as u32));
        assert_eq!(sibling_read, Err(lost as u32));
        assert_eq!(later_bind, Err(lost as u32));
        assert_eq!(later_allocation, Err(lost as u32));
        assert_eq!(after_oom, Err(lost as u32));
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
        let counts = memory.snapshot();
        assert_eq!(counts.allocation_attempts, 1);
        assert_eq!(counts.async_allocations + counts.sync_allocations, 0);
        assert_eq!(counts.async_frees + counts.sync_frees, 0);
        assert_eq!(events.snapshot().create_attempts, 0);
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(stream);
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
fn ordinary_malloc_oom_remains_retryable_without_delayed_error() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        memory.set_allocate_error(Some(oom));
        assert_eq!(
            unsafe { stream.alloc::<u8>(16) }
                .map(drop)
                .map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert!(lifetime.loss().is_none());
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(events.snapshot().create_attempts, 0);
        memory.set_allocate_error(None);
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        drop(allocation);
        assert_eq!(memory.snapshot().allocation_attempts, 2);
        assert_eq!(
            memory.snapshot().async_frees + memory.snapshot().sync_frees,
            1
        );
        assert_eq!(
            (events.snapshot().created, events.snapshot().destroyed),
            (2, 2)
        );
        drop(stream);
        drop(context);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}

#[test]
fn untracked_malloc_terminal_error_keeps_its_return_contract() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let _events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let mut context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
        Arc::get_mut(&mut context).unwrap().has_async_alloc = asynchronous;
        let stream = context.default_stream();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        memory.set_allocate_error(Some(lost));
        assert_eq!(
            unsafe { stream.alloc::<u8>(16) }
                .map(drop)
                .map_err(|error| error.0 as u32),
            Err(lost as u32)
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert!(context.primary_lifetime.is_none());
        assert_eq!(memory.snapshot().allocation_attempts, 1);
        drop(stream);
        drop(context);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
