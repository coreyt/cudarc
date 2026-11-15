#![cfg(not(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080")))]

use super::{lifetime_event_cleanup_tests::seams as event_seams, lifetime_pinned_cleanup_tests::seams};
use crate::driver::{result::context_test_seams, sys, CudaContext, PrimaryContextLifetime};
use std::sync::{Arc, Mutex};

#[test]
fn pinned_construction_holds_admission_through_event_and_native_host_allocation() {
    let driver = context_test_seams::Guard::new(); let events = event_seams::Guard::new(); let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let phases = Arc::new(Mutex::new(Vec::new()));
    let weak = Arc::downgrade(&lifetime); let log = Arc::clone(&phases);
    events.set_create_hook(move || { let lifetime = weak.upgrade().unwrap(); let active = lifetime.state.try_lock().expect("native child event creation must be unlocked").active_leases; log.lock().unwrap().push((0,active)); });
    let weak = Arc::downgrade(&lifetime); let log = Arc::clone(&phases);
    host.set_allocate_hook(move || { let lifetime = weak.upgrade().unwrap(); let active = lifetime.state.try_lock().expect("native host allocation must be unlocked").active_leases; log.lock().unwrap().push((1,active)); });
    let allocation = unsafe { context.alloc_pinned::<u64>(2) }.map_err(|error| error.0 as u32).unwrap();
    events.set_create_hook(|| {}); host.set_allocate_hook(|| {});
    assert_eq!(*phases.lock().unwrap(), vec![(0,2),(1,1)], "pinned construction must retain admission until complete ownership");
    assert_eq!((events.snapshot().created,host.snapshot().allocated), (1,1));
    assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
    drop(allocation); assert_eq!((host.snapshot().freed,events.snapshot().destroyed), (1,1));
    drop(context); drop(lifetime); assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
}

#[test]
fn known_loss_prevents_pinned_owner_growth_before_any_native_child() {
    let driver = context_test_seams::Guard::new(); let events = event_seams::Guard::new(); let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    lifetime.record_error(crate::driver::DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED)); let before = driver.snapshot();
    let result = unsafe { context.alloc_pinned::<u64>(2) }.map(drop).map_err(|error| error.0 as u32);
    assert_eq!(result,Err(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED as u32));
    assert_eq!((events.snapshot().create_attempts,host.snapshot().attempts),(0,0));
    let after = driver.snapshot(); assert_eq!((after.primary_queries,after.id_queries,after.queries,after.binds),(before.primary_queries,before.id_queries,before.queries,before.binds));
    drop(context);drop(lifetime);assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
}

#[test]
fn host_allocation_unwind_destroys_its_completed_event_under_construction_admission() {
    let driver = context_test_seams::Guard::new(); let events = event_seams::Guard::new(); let host = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new())); let log = Arc::clone(&seen); let weak = Arc::downgrade(&lifetime);
    events.set_destroy_hook(move || { let lifetime = weak.upgrade().unwrap(); let active = lifetime.state.try_lock().expect("partial event cleanup must be unlocked").active_leases; log.lock().unwrap().push(active); });
    host.set_allocate_hook(|| panic!("synthetic host allocation unwind before native ownership"));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { context.alloc_pinned::<u64>(2) }));
    host.set_allocate_hook(|| {}); events.set_destroy_hook(|| {});
    assert!(result.is_err()); assert_eq!(*seen.lock().unwrap(),vec![2]);
    assert_eq!((host.snapshot().allocated,events.snapshot().created,events.snapshot().destroyed),(0,1,1));
    assert_eq!(lifetime.state.lock().unwrap().active_leases,0);assert!(lifetime.loss().is_none());
    drop(context);drop(lifetime);assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
}
