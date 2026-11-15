use super::lifetime_event_cleanup_tests::seams as event_seams;
use super::lifetime_slice_cleanup_tests::{seams, tracked_context};
use crate::driver::{result::context_test_seams, sys, PrimaryContextLifetime};

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn second_event_failure_releases_the_already_allocated_slice() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        events.fail_on_create_attempt(2, sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);

        let outcome = unsafe { stream.alloc::<u8>(16) }
            .map(|_| ())
            .map_err(|error| error.0 as u32);

        assert_eq!(outcome, Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32));
        let counts = memory.snapshot();
        assert_eq!(counts.async_allocations + counts.sync_allocations, 1);
        assert_eq!(
            counts.async_frees + counts.sync_frees,
            1,
            "fallible event construction must not strand the acquired allocation"
        );
        assert_eq!(
            (
                events.snapshot().create_attempts,
                events.snapshot().created,
                events.snapshot().destroyed
            ),
            (2, 1, 1)
        );
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        events.set_create_error(None);
        let retry = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        drop(retry);
        let retried = memory.snapshot();
        assert_eq!(retried.async_allocations + retried.sync_allocations, 2);
        assert_eq!(retried.async_frees + retried.sync_frees, 2);
        assert_eq!(
            (events.snapshot().created, events.snapshot().destroyed),
            (3, 3)
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

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn first_event_failure_releases_the_allocation_before_retry() {
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
        events.fail_on_create_attempt(1, oom);

        assert_eq!(
            unsafe { stream.alloc::<u8>(16) }
                .map(|_| ())
                .map_err(|error| error.0 as u32),
            Err(oom as u32)
        );

        assert_eq!(
            memory.snapshot().async_allocations + memory.snapshot().sync_allocations,
            1
        );
        assert_eq!(
            memory.snapshot().async_frees + memory.snapshot().sync_frees,
            1
        );
        assert_eq!(
            (events.snapshot().created, events.snapshot().destroyed),
            (0, 0)
        );
        assert!(lifetime.loss().is_none());
        events.set_create_error(None);
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        drop(allocation);
        assert_eq!(
            memory.snapshot().async_frees + memory.snapshot().sync_frees,
            2
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

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
#[test]
fn event_setup_unwind_releases_the_allocation_and_completed_child() {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let attempts = Arc::new(AtomicUsize::new(0));
        let marker = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&marker);
        events.set_create_hook(move || {
            if attempts.fetch_add(1, Ordering::SeqCst) == 1 {
                observed.store(true, Ordering::SeqCst);
                panic!("synthetic second-event setup unwind");
            }
        });

        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            stream.alloc::<u8>(16)
        }));

        assert!(failed.is_err());
        assert!(marker.load(Ordering::SeqCst));
        assert_eq!(
            memory.snapshot().async_allocations + memory.snapshot().sync_allocations,
            1
        );
        assert_eq!(
            memory.snapshot().async_frees + memory.snapshot().sync_frees,
            1
        );
        assert_eq!(
            (
                events.snapshot().create_attempts,
                events.snapshot().created,
                events.snapshot().destroyed
            ),
            (2, 1, 1)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(lifetime.loss().is_none());
        events.set_create_hook(|| {});
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        drop(allocation);
        assert_eq!(
            memory.snapshot().async_frees + memory.snapshot().sync_frees,
            2
        );
        assert_eq!(
            (events.snapshot().created, events.snapshot().destroyed),
            (3, 3)
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
