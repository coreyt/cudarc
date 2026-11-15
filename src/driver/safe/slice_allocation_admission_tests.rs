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
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

fn assert_admitted(lifetime: &PrimaryContextLifetime, expected: usize) {
    let (active, probing) = {
        let state = lifetime
            .state
            .try_lock()
            .expect("native allocation/child dispatch must run outside the lifetime mutex");
        (state.active_leases, state.probing)
    };
    assert_eq!(
        active, expected,
        "slice construction must retain admission through native work"
    );
    assert!(!probing);
}

#[test]
fn slice_allocation_holds_admission_through_malloc_and_both_events() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let phase = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&lifetime);
        let allocated = Arc::clone(&phase);
        memory.set_allocate_hook(move || {
            assert_admitted(&weak.upgrade().unwrap(), 1);
            assert_eq!(allocated.fetch_add(1, Ordering::SeqCst), 0);
        });
        let weak = Arc::downgrade(&lifetime);
        let child = Arc::clone(&phase);
        events.set_create_hook(move || {
            assert_admitted(&weak.upgrade().unwrap(), 2);
            let step = child.fetch_add(1, Ordering::SeqCst);
            assert!(step == 1 || step == 2);
        });
        let before = driver.snapshot();

        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();

        assert_eq!(phase.load(Ordering::SeqCst), 3);
        let after = driver.snapshot();
        assert_eq!(
            (after.primary_queries, after.id_queries),
            (before.primary_queries + 1, before.id_queries + 1)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(allocation);
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
fn slice_allocation_shares_the_outer_epoch_through_native_children() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let weak = Arc::downgrade(&lifetime);
    memory.set_allocate_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));
    let weak = Arc::downgrade(&lifetime);
    events.set_create_hook(move || assert_admitted(&weak.upgrade().unwrap(), 3));
    let before = driver.snapshot();

    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();

    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries, before.id_queries)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
    drop(allocation);
    drop(outer);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn lost_slice_allocation_refuses_before_native_or_owner_growth() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();

    assert_eq!(
        unsafe { stream.alloc::<u8>(16) }
            .map(drop)
            .map_err(|error| error.0 as u32),
        Err(lost as u32)
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
            before.primary_queries,
            before.id_queries,
            before.queries,
            before.binds
        )
    );
    assert_eq!(
        (
            memory.snapshot().allocation_attempts,
            events.snapshot().create_attempts
        ),
        (0, 0)
    );
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn native_allocation_oom_releases_admission_without_storing_error() {
    for asynchronous in [false, true] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let memory = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = tracked_context(&lifetime, asynchronous);
        let stream = context.default_stream();
        let weak = Arc::downgrade(&lifetime);
        memory.set_allocate_hook(move || assert_admitted(&weak.upgrade().unwrap(), 1));
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        memory.set_allocate_error(Some(oom));
        assert_eq!(
            unsafe { stream.alloc::<u8>(16) }
                .map(drop)
                .map_err(|error| error.0 as u32),
            Err(oom as u32)
        );
        assert_eq!(
            (
                memory.snapshot().allocation_attempts,
                events.snapshot().created
            ),
            (1, 0)
        );
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(lifetime.loss().is_none());
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        memory.set_allocate_error(None);
        let allocation = unsafe { stream.alloc::<u8>(16) }
            .map_err(|error| error.0 as u32)
            .unwrap();
        drop(allocation);
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
fn allocation_dispatch_unwind_releases_admission_before_retry() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let marker = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&marker);
    let weak = Arc::downgrade(&lifetime);
    memory.set_allocate_hook(move || {
        assert_admitted(&weak.upgrade().unwrap(), 1);
        observed.store(true, Ordering::SeqCst);
        panic!("synthetic admitted allocation dispatch unwind");
    });

    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        stream.alloc::<u8>(16)
    }));

    assert!(failed.is_err());
    assert!(marker.load(Ordering::SeqCst));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.loss().is_none());
    assert_eq!(
        memory.snapshot().async_allocations + memory.snapshot().sync_allocations,
        0
    );
    assert_eq!(events.snapshot().created, 0);
    memory.set_allocate_hook(|| {});
    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    drop(allocation);
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn child_setup_failure_retains_construction_admission_through_owner_cleanup() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = tracked_context(&lifetime, false);
    let stream = context.default_stream();
    let weak = Arc::downgrade(&lifetime);
    memory.set_free_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));
    events.fail_on_create_attempt(2, sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);

    assert_eq!(
        unsafe { stream.alloc::<u8>(16) }
            .map(drop)
            .map_err(|error| error.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
    );

    assert_eq!(
        memory.snapshot().async_frees + memory.snapshot().sync_frees,
        1
    );
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 1)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.loss().is_none());
    drop(stream);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn untracked_slice_allocation_preserves_native_owner_cleanup() {
    let driver = context_test_seams::Guard::new();
    let events = event_seams::Guard::new();
    let memory = seams::Guard::new();
    let context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    let stream = context.default_stream();
    let allocation = unsafe { stream.alloc::<u8>(16) }
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(
        (
            driver.snapshot().primary_queries,
            driver.snapshot().id_queries
        ),
        (0, 0)
    );
    drop(allocation);
    assert_eq!(memory.snapshot().sync_frees, 1);
    assert_eq!(events.snapshot().destroyed, 2);
    drop(stream);
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
