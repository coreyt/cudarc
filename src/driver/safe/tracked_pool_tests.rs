#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use crate::driver::{
    result::context_test_seams, sys, CudaContext, CudaMemPool, DriverError, MemPoolProps,
    PrimaryContextLifetime,
};
use std::sync::{Arc, Mutex};

pub(crate) mod seams {
    use crate::driver::{sys, DriverError};
    use core::ffi::c_void;
    use std::cell::{Cell, RefCell};
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Counts {
        pub(crate) create_attempts: usize,
        pub(crate) created: usize,
        pub(crate) sets: usize,
        pub(crate) gets: usize,
        pub(crate) trims: usize,
        pub(crate) destroy_attempts: usize,
        pub(crate) destroyed: usize,
        pub(crate) allocations: usize,
    }
    #[derive(Clone, Copy, Default)]
    struct State {
        counts: Counts,
        live: u64,
        threshold: u64,
        create_error: Option<sys::CUresult>,
        set_error: Option<sys::CUresult>,
        get_error: Option<sys::CUresult>,
        trim_error: Option<sys::CUresult>,
        destroy_error: Option<sys::CUresult>,
    }
    std::thread_local! {
        static STATE:Cell<Option<State>>=const{Cell::new(None)};
        static HOOK:RefCell<Option<Box<dyn FnMut(u8)>>>=const{RefCell::new(None)};
    }
    pub(crate) struct Guard;
    impl Guard {
        pub(crate) fn new() -> Self {
            assert!(STATE.get().is_none());
            STATE.set(Some(State::default()));
            Self
        }
        pub(crate) fn snapshot(&self) -> Counts {
            STATE.get().unwrap().counts
        }
        pub(crate) fn set_hook(&self, hook: impl FnMut(u8) + 'static) {
            HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        }
        pub(crate) fn set_create_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.create_error = error;
            STATE.set(Some(state));
        }
        pub(crate) fn set_set_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.set_error = error;
            STATE.set(Some(state));
        }
        pub(crate) fn set_get_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.get_error = error;
            STATE.set(Some(state));
        }
        pub(crate) fn set_trim_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.trim_error = error;
            STATE.set(Some(state));
        }
        pub(crate) fn set_destroy_error(&self, error: Option<sys::CUresult>) {
            let mut state = STATE.get().unwrap();
            state.destroy_error = error;
            STATE.set(Some(state));
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            HOOK.with(|slot| *slot.borrow_mut() = None);
            STATE.set(None);
        }
    }
    fn hook(phase: u8) {
        HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook(phase);
            }
        });
    }
    fn bit(pool: sys::CUmemoryPool, state: &State) -> u64 {
        let id = pool as usize;
        assert!(id > 0xB000 && id <= 0xB000 + state.counts.created);
        let bit = 1u64 << (id - 0xB001);
        assert_ne!(state.live & bit, 0);
        bit
    }
    pub(crate) fn create(
        props: *const sys::CUmemPoolProps,
    ) -> Option<Result<sys::CUmemoryPool, DriverError>> {
        let mut state = STATE.get()?;
        let props = unsafe { &*props };
        assert_eq!(
            props.allocType,
            sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED
        );
        assert_eq!(
            props.handleTypes,
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE
        );
        assert!(props.reserved.iter().all(|byte| *byte == 0));
        state.counts.create_attempts += 1;
        STATE.set(Some(state));
        hook(0);
        if let Some(error) = state.create_error {
            return Some(Err(DriverError(error)));
        }
        state.counts.created += 1;
        assert!(state.counts.created <= 16);
        state.live |= 1u64 << (state.counts.created - 1);
        STATE.set(Some(state));
        Some(Ok((0xB000 + state.counts.created) as sys::CUmemoryPool))
    }
    pub(crate) fn destroy(pool: sys::CUmemoryPool) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        let owned = bit(pool, &state);
        state.counts.destroy_attempts += 1;
        STATE.set(Some(state));
        hook(4);
        if let Some(error) = state.destroy_error {
            return Some(Err(DriverError(error)));
        }
        state.live &= !owned;
        state.counts.destroyed += 1;
        STATE.set(Some(state));
        Some(Ok(()))
    }
    pub(crate) fn set(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
        value: *mut c_void,
    ) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        bit(pool, &state);
        assert_eq!(
            attr,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
        );
        state.counts.sets += 1;
        STATE.set(Some(state));
        hook(1);
        if let Some(error) = state.set_error {
            return Some(Err(DriverError(error)));
        }
        state.threshold = unsafe { *value.cast::<u64>() };
        STATE.set(Some(state));
        Some(Ok(()))
    }
    pub(crate) fn get(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
        value: *mut c_void,
    ) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        bit(pool, &state);
        state.counts.gets += 1;
        STATE.set(Some(state));
        hook(2);
        if let Some(error) = state.get_error {
            return Some(Err(DriverError(error)));
        }
        let result = match attr {
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD => state.threshold,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT => 4096,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH => 8192,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT => 16,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH => 32,
            _ => panic!("unexpected pool attribute"),
        };
        unsafe {
            *value.cast::<u64>() = result;
        }
        Some(Ok(()))
    }
    pub(crate) fn trim(pool: sys::CUmemoryPool, _keep: usize) -> Option<Result<(), DriverError>> {
        let mut state = STATE.get()?;
        bit(pool, &state);
        state.counts.trims += 1;
        STATE.set(Some(state));
        hook(3);
        Some(match state.trim_error {
            Some(error) => Err(DriverError(error)),
            None => Ok(()),
        })
    }
    pub(crate) fn allocate(
        pool: sys::CUmemoryPool,
        bytes: usize,
        stream: sys::CUstream,
    ) -> Option<Result<sys::CUdeviceptr, DriverError>> {
        let mut state = STATE.get()?;
        bit(pool, &state);
        assert!(stream.is_null());
        state.counts.allocations += 1;
        STATE.set(Some(state));
        hook(5);
        Some(
            crate::driver::safe::lifetime_slice_cleanup_tests::seams::allocate(bytes, true)
                .expect("pool allocation requires counted slice ownership seam"),
        )
    }
}

fn props() -> MemPoolProps {
    MemPoolProps {
        max_size: 0,
        release_threshold: 128,
    }
}

#[test]
fn factory_tracks_one_retained_anchor_before_native_pool_ownership() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let phases = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&phases);
    let weak = Arc::downgrade(&lifetime);
    native.set_hook(move |phase| {
        let lifetime = weak.upgrade().unwrap();
        let active = lifetime
            .state
            .try_lock()
            .expect("pool construction native work must be unlocked")
            .active_leases;
        log.lock().unwrap().push((phase, active));
    });
    let result = CudaMemPool::create_tracked(Arc::clone(&lifetime), &props());
    assert!(
        result.is_ok(),
        "tracked pool factory must acquire native ownership, not report missing support"
    );
    let pool = result.map_err(|e| e.0 as u32).unwrap();
    native.set_hook(|_| {});
    assert_eq!(*phases.lock().unwrap(), vec![(0, 1), (1, 1)]);
    assert_eq!(
        (
            native.snapshot().create_attempts,
            native.snapshot().created,
            native.snapshot().sets
        ),
        (1, 1, 1)
    );
    assert_eq!(driver.snapshot().default_pool_queries, 0);
    let weak = Arc::downgrade(&lifetime);
    drop(lifetime);
    assert!(
        weak.upgrade().is_some(),
        "pool itself must retain its generation anchor"
    );
    assert_eq!(driver.snapshot().releases, 0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.snapshot().destroyed, 1);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn factory_native_create_oom_can_retry_without_latching_loss() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    native.set_create_error(Some(oom));
    assert_eq!(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(oom as u32)
    );
    assert_eq!(
        (
            native.snapshot().created,
            native.snapshot().destroy_attempts
        ),
        (0, 0)
    );
    assert!(lifetime.loss().is_none());
    native.set_create_error(None);
    let pool = CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
        .map_err(|e| e.0 as u32)
        .unwrap();
    drop(pool);
    assert_eq!(
        (
            native.snapshot().create_attempts,
            native.snapshot().created,
            native.snapshot().destroyed
        ),
        (2, 1, 1)
    );
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn factory_threshold_oom_releases_its_acquired_pool_once() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    super::mem_pool::seams::DRIVER_VERSION.set(Some(12060));
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    native.set_set_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    assert_eq!(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
    );
    assert_eq!(
        (
            native.snapshot().created,
            native.snapshot().sets,
            native.snapshot().destroy_attempts,
            native.snapshot().destroyed
        ),
        (1, 1, 1, 1)
    );
    assert!(lifetime.loss().is_none());
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn factory_known_loss_refuses_before_any_pool_native_growth() {
    let driver = context_test_seams::Guard::new();
    let _legacy = super::mem_pool::seams::Reset::new();
    let native = seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    lifetime.record_error(DriverError(lost));
    let before = driver.snapshot();
    assert_eq!(
        CudaMemPool::create_tracked(Arc::clone(&lifetime), &props())
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(lost as u32)
    );
    assert_eq!(native.snapshot().create_attempts, 0);
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
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn factory_context_lifetime_observation_returns_exact_anchor_without_native_calls() {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let before = driver.snapshot();
    assert_eq!(
        context
            .primary_lifetime()
            .map(|observed| Arc::ptr_eq(observed, &lifetime)),
        Some(true)
    );
    lifetime.record_error(DriverError(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED));
    assert_eq!(
        context
            .primary_lifetime()
            .map(|observed| Arc::ptr_eq(observed, &lifetime)),
        Some(true)
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
    drop(context);
    drop(lifetime);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}

#[test]
fn factory_untracked_context_observation_is_none_without_native_calls() {
    let driver = context_test_seams::Guard::new();
    let context = CudaContext::new(0).map_err(|e| e.0 as u32).unwrap();
    let before = driver.snapshot();
    assert!(context.primary_lifetime().is_none());
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
    drop(context);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
