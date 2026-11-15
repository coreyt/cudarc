#![cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]

use super::lifetime_stream_cleanup_tests::seams;
use crate::driver::{result::context_test_seams, sys, CudaContext, PrimaryContextLifetime};
use std::sync::{atomic::Ordering, Arc};

#[test]
fn failed_native_stream_create_rolls_back_owned_count_and_retries() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    streams.set_create_error(Some(oom));

    let failed = context.new_stream().map(|_| ()).map_err(|error| error.0 as u32);

    assert_eq!(failed, Err(oom as u32));
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (1, 0, 0));
    assert_eq!(streams.snapshot().context_synchronizes, 1);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0, "failed native construction must return its owned-stream reservation");
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    streams.set_create_error(None);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1);
    assert_eq!(streams.snapshot().context_synchronizes, 2, "retry must perform the first-stream synchronization again");
    drop(stream);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (2, 1, 1));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn failed_first_stream_synchronization_rolls_back_before_native_create() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    streams.set_context_synchronize_error(Some(oom));

    let failed = context.new_stream().map(|_| ()).map_err(|error| error.0 as u32);

    assert_eq!(failed, Err(oom as u32));
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (0, 0, 0));
    assert_eq!(streams.snapshot().context_synchronizes, 1);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    streams.set_context_synchronize_error(None);
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!(streams.snapshot().context_synchronizes, 2);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1);
    drop(stream);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (1, 1, 1));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn failed_additional_stream_preserves_existing_owned_count() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let existing = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    streams.set_create_error(Some(oom));

    let failed = context.new_stream().map(|_| ()).map_err(|error| error.0 as u32);

    assert_eq!(failed, Err(oom as u32));
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1, "rollback must preserve the already-owned stream");
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (2, 1, 0));
    assert_eq!(streams.snapshot().context_synchronizes, 1);
    streams.set_create_error(None);
    let additional = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 2);
    drop(additional);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1);
    drop(existing);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (3, 2, 2));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn stream_creation_unwind_returns_owned_count_before_retry() {
    let driver = context_test_seams::Guard::new();
    let streams = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    streams.set_create_hook(|| panic!("synthetic native-stream-dispatch unwind"));

    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| context.new_stream()));

    assert!(failed.is_err());
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (1, 0, 0));
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert!(lifetime.state.lock().unwrap().loss.is_none());
    streams.set_create_hook(|| {});
    let stream = context.new_stream().map_err(|error| error.0 as u32).unwrap();
    assert_eq!(streams.snapshot().context_synchronizes, 2);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 1);
    drop(stream);
    assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
    assert_eq!((streams.snapshot().create_attempts, streams.snapshot().created, streams.snapshot().destroyed), (2, 1, 1));
    drop(context);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}
