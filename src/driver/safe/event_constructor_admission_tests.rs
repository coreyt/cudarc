#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::lifetime_event_cleanup_tests::seams;
use crate::driver::{
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn assert_admitted(lifetime: &PrimaryContextLifetime, expected: usize) {
    let (active, probing) = {
        let state = lifetime
            .state
            .try_lock()
            .expect("native event construction must run outside the state mutex");
        (state.active_leases, state.probing)
    };
    assert_eq!(
        active, expected,
        "native event creation must retain admission"
    );
    assert!(!probing);
}

#[test]
fn event_construction_holds_admission_through_native_create() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    events.set_create_hook(move || assert_admitted(&weak.upgrade().unwrap(), 1));
    let before = driver.snapshot();

    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();

    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries + 1, before.id_queries + 1)
    );
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 0)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(event);
    assert_eq!(
        (events.snapshot().created, events.snapshot().destroyed),
        (1, 1)
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn event_construction_shares_an_outer_epoch_through_native_create() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let weak = Arc::downgrade(&lifetime);
    events.set_create_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));
    let before = driver.snapshot();

    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();

    let after = driver.snapshot();
    assert_eq!(
        (after.primary_queries, after.id_queries),
        (before.primary_queries, before.id_queries)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
    drop(event);
    drop(outer);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn lost_event_construction_refuses_before_native_growth() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
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
            .new_event(None)
            .map(|_| ())
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
        (events.snapshot().create_attempts, events.snapshot().created),
        (0, 0)
    );
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn event_construction_oom_returns_admission_and_remains_retryable() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let weak = Arc::downgrade(&lifetime);
    events.set_create_hook(move || assert_admitted(&weak.upgrade().unwrap(), 1));
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    events.set_create_error(Some(oom));

    assert_eq!(
        context
            .new_event(None)
            .map(|_| ())
            .map_err(|error| error.0 as u32),
        Err(oom as u32)
    );

    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    assert_eq!(
        (
            events.snapshot().create_attempts,
            events.snapshot().created,
            events.snapshot().destroyed
        ),
        (1, 0, 0)
    );
    events.set_create_error(None);
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(event);
    assert_eq!(
        (
            events.snapshot().create_attempts,
            events.snapshot().created,
            events.snapshot().destroyed
        ),
        (2, 1, 1)
    );
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn event_construction_unwind_returns_admission_before_retry() {
    let driver = context_test_seams::Guard::new();
    let events = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|error| error.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|error| error.0 as u32)
        .unwrap();
    let marker = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&marker);
    let weak = Arc::downgrade(&lifetime);
    events.set_create_hook(move || {
        assert_admitted(&weak.upgrade().unwrap(), 1);
        observed.store(true, Ordering::SeqCst);
        panic!("synthetic admitted event-create unwind");
    });

    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| context.new_event(None)));

    assert!(failed.is_err());
    assert!(marker.load(Ordering::SeqCst));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    assert_eq!(
        (events.snapshot().create_attempts, events.snapshot().created),
        (1, 0)
    );
    events.set_create_hook(|| {});
    let event = context
        .new_event(None)
        .map_err(|error| error.0 as u32)
        .unwrap();
    drop(event);
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
