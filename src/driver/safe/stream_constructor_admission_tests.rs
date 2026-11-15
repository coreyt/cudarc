#![cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]

use super::lifetime_stream_cleanup_tests::seams;
use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime};
use std::sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc};

fn assert_admitted(lifetime: &PrimaryContextLifetime, expected: usize) {
    let (active, probing) = {
        let state = lifetime.state.try_lock().expect("native constructor dispatch must run outside the state mutex");
        (state.active_leases, state.probing)
    };
    assert_eq!(active, expected, "native constructor dispatch must retain admission");
    assert!(!probing);
}

#[test]
fn stream_construction_holds_admission_through_native_sync_and_create() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let phase = Arc::new(AtomicUsize::new(0));
    let weak = Arc::downgrade(&lifetime);
    let synchronized = Arc::clone(&phase);
    streams.set_synchronize_hook(move || {
        assert_admitted(&weak.upgrade().unwrap(), 1);
        assert_eq!(synchronized.fetch_add(1, Ordering::SeqCst), 0);
    });
    let weak = Arc::downgrade(&lifetime);
    let created = Arc::clone(&phase);
    streams.set_create_hook(move || {
        assert_admitted(&weak.upgrade().unwrap(), 1);
        assert_eq!(created.fetch_add(1, Ordering::SeqCst), 1);
    });
    let before = driver.snapshot();

    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();

    let after = driver.snapshot();
    assert_eq!(phase.load(Ordering::SeqCst), 2);
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries + 1, before.id_queries + 1));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1);
    drop(stream);
    assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (1, 1));
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn stream_construction_shares_an_outer_epoch_through_native_dispatch() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
    let weak = Arc::downgrade(&lifetime);
    streams.set_synchronize_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));
    let weak = Arc::downgrade(&lifetime);
    streams.set_create_hook(move || assert_admitted(&weak.upgrade().unwrap(), 2));
    let before = driver.snapshot();

    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();

    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries), (before.primary_queries, before.id_queries));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
    drop(stream);
    drop(outer);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn lost_stream_construction_refuses_before_native_or_host_growth() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();

    let failed = context.new_stream().map(|_| ()).map_err(|error| error.0 as u32);

    assert_eq!(failed, Err(lost as u32));
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries, after.queries, after.binds), (before.primary_queries, before.id_queries, before.queries, before.binds));
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().context_synchronizes), (0, 0));
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn stream_construction_oom_returns_admission_and_remains_retryable() {
    for fail_sync in [false, true] {
        let driver = context_test_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
        let weak = Arc::downgrade(&lifetime);
        streams.set_synchronize_hook(move || assert_admitted(&weak.upgrade().unwrap(), 1));
        let weak = Arc::downgrade(&lifetime);
        streams.set_create_hook(move || assert_admitted(&weak.upgrade().unwrap(), 1));
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        if fail_sync { streams.set_context_synchronize_error(Some(oom)); } else { streams.set_create_error(Some(oom)); }

        assert_eq!(context.new_stream().map(|_| ()).map_err(|error| error.0 as u32), Err(oom as u32));

        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        streams.set_context_synchronize_error(None);
        streams.set_create_error(None);
        let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        drop(stream);
        assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (1, 1));
        drop(context);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}

#[test]
fn stream_construction_unwind_returns_admission_at_each_native_phase() {
    for panic_sync in [false, true] {
        let driver = context_test_seams::Guard::new();
        let streams = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
        let marker = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&marker);
        let weak = Arc::downgrade(&lifetime);
        let hook = move || {
            assert_admitted(&weak.upgrade().unwrap(), 1);
            observed.store(true, Ordering::SeqCst);
            panic!("synthetic admitted native-constructor unwind");
        };
        if panic_sync { streams.set_synchronize_hook(hook); } else { streams.set_create_hook(hook); }

        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| context.new_stream()));

        assert!(failed.is_err());
        assert!(marker.load(Ordering::SeqCst));
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!((streams.snapshot().created, streams.snapshot().destroyed), (0, 0));
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        streams.set_synchronize_hook(|| {});
        streams.set_create_hook(|| {});
        let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
        drop(stream);
        drop(context);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}
