#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]
use super::{
    lifetime_event_cleanup_tests::seams as events, lifetime_slice_cleanup_tests::seams as memory,
    lifetime_stream_cleanup_tests::seams as streams,
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, CudaEvent, CudaSlice, CudaStream, DriverError,
    PrimaryContextLifetime, PrimaryContextLossCause,
};
use std::sync::{atomic::Ordering, Arc};
#[derive(Clone, Copy, Debug)]
enum Kind {
    Event,
    Stream,
    AsyncSlice,
    SyncSlice,
}
const ALL: [Kind; 4] = [Kind::Event, Kind::Stream, Kind::AsyncSlice, Kind::SyncSlice];
enum Owner {
    Event(CudaEvent),
    Stream(Arc<CudaStream>),
    Slice(CudaSlice<u8>),
}
fn create(kind: Kind, ctx: &Arc<CudaContext>) -> Owner {
    match kind {
        Kind::Event => Owner::Event(ctx.new_event(None).map_err(|e| e.0 as u32).unwrap()),
        Kind::Stream => Owner::Stream(ctx.new_stream().map_err(|e| e.0 as u32).unwrap()),
        Kind::AsyncSlice | Kind::SyncSlice => Owner::Slice(
            unsafe { ctx.default_stream().alloc::<u8>(16) }
                .map_err(|e| e.0 as u32)
                .unwrap(),
        ),
    }
}
fn fault(
    kind: Kind,
    e: &events::Guard,
    s: &streams::Guard,
    m: &memory::Guard,
    error: Option<sys::CUresult>,
) {
    match kind {
        Kind::Event => e.set_destroy_error(error),
        Kind::Stream => s.set_destroy_error(error),
        Kind::AsyncSlice | Kind::SyncSlice => m.set_free_error(error),
    }
}
fn calls(e: &events::Guard, s: &streams::Guard, m: &memory::Guard) -> [usize; 10] {
    let e = e.snapshot();
    let s = s.snapshot();
    let m = m.snapshot();
    [
        e.create_attempts,
        e.destroyed,
        s.create_attempts,
        s.destroyed,
        s.waits,
        s.context_synchronizes,
        m.allocation_attempts,
        m.stream_synchronizes,
        m.async_frees,
        m.sync_frees,
    ]
}
fn cleanup_failure(kind: Kind, error: sys::CUresult) {
    let driver = context_test_seams::Guard::new();
    let e = events::Guard::new();
    let s = streams::Guard::new();
    let m = memory::Guard::new();
    let life = PrimaryContextLifetime::new(0)
        .map_err(|e| e.0 as u32)
        .unwrap();
    let mut ctx = CudaContext::new_tracked(Arc::clone(&life))
        .map_err(|e| e.0 as u32)
        .unwrap();
    Arc::get_mut(&mut ctx).unwrap().has_async_alloc = !matches!(kind, Kind::SyncSlice);
    let sibling = CudaContext::new_tracked(Arc::clone(&life))
        .map_err(|e| e.0 as u32)
        .unwrap();
    let owner = create(kind, &ctx);
    fault(kind, &e, &s, &m, Some(error));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
    fault(kind, &e, &s, &m, None);
    assert!(result.is_ok());
    assert_eq!(
        life.loss()
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.cause)),
        Some((17, None, PrimaryContextLossCause::DriverStatus(error))),
        "actual native {kind:?} cleanup failure must freeze before abandonment"
    );
    match kind {
        Kind::Event => assert_eq!(e.snapshot().destroyed, 1),
        Kind::Stream => assert_eq!(
            (
                s.snapshot().destroyed,
                ctx.num_streams.load(Ordering::Relaxed)
            ),
            (1, 0)
        ),
        Kind::AsyncSlice => assert_eq!((m.snapshot().async_frees, e.snapshot().destroyed), (1, 0)),
        Kind::SyncSlice => assert_eq!(
            (
                m.snapshot().sync_frees,
                m.snapshot().stream_synchronizes,
                e.snapshot().destroyed
            ),
            (1, 1, 0)
        ),
    };
    let first = life.loss();
    let before = driver.snapshot();
    let native = calls(&e, &s, &m);
    assert_eq!(ctx.check_err().map_err(|e| e.0 as u32), Err(error as u32));
    assert_eq!(ctx.check_err().map_err(|e| e.0 as u32), Err(error as u32));
    assert_eq!(
        sibling.bind_to_thread().map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!(
        ctx.new_event(None).map(drop).map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    assert_eq!(
        unsafe { ctx.default_stream().alloc::<u8>(16) }
            .map(drop)
            .map_err(|e| e.0 as u32),
        Err(error as u32)
    );
    sibling.record_err::<()>(Err(DriverError(
        sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
    )));
    assert_eq!(life.loss(), first);
    assert_eq!(calls(&e, &s, &m), native);
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
    drop(ctx);
    drop(sibling);
    drop(life);
    assert_eq!(
        (driver.snapshot().retains, driver.snapshot().releases),
        (1, 1)
    );
}
macro_rules! case {
    ($name:ident,$kind:ident,$error:ident) => {
        #[test]
        fn $name() {
            cleanup_failure(Kind::$kind, sys::CUresult::$error);
        }
    };
}
case!(
    actual_event_destroy_oom_stops_future_native_growth,
    Event,
    CUDA_ERROR_OUT_OF_MEMORY
);
case!(
    actual_event_destroy_201_stops_future_native_growth,
    Event,
    CUDA_ERROR_INVALID_CONTEXT
);
case!(
    actual_event_destroy_unknown_stops_future_native_growth,
    Event,
    CUDA_ERROR_UNKNOWN
);
case!(
    actual_stream_destroy_oom_stops_future_native_growth,
    Stream,
    CUDA_ERROR_OUT_OF_MEMORY
);
case!(
    actual_stream_destroy_201_stops_future_native_growth,
    Stream,
    CUDA_ERROR_INVALID_CONTEXT
);
case!(
    actual_stream_destroy_unknown_stops_future_native_growth,
    Stream,
    CUDA_ERROR_UNKNOWN
);
case!(
    actual_async_free_oom_stops_future_native_growth,
    AsyncSlice,
    CUDA_ERROR_OUT_OF_MEMORY
);
case!(
    actual_async_free_201_stops_future_native_growth,
    AsyncSlice,
    CUDA_ERROR_INVALID_CONTEXT
);
case!(
    actual_async_free_unknown_stops_future_native_growth,
    AsyncSlice,
    CUDA_ERROR_UNKNOWN
);
case!(
    actual_sync_free_oom_stops_future_native_growth,
    SyncSlice,
    CUDA_ERROR_OUT_OF_MEMORY
);
case!(
    actual_sync_free_201_stops_future_native_growth,
    SyncSlice,
    CUDA_ERROR_INVALID_CONTEXT
);
case!(
    actual_sync_free_unknown_stops_future_native_growth,
    SyncSlice,
    CUDA_ERROR_UNKNOWN
);
#[test]
fn actual_cleanup_709_preserves_existing_driver_origin() {
    for kind in ALL {
        cleanup_failure(kind, sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
    }
}
#[test]
fn pending_ordinary_oom_stays_pending_through_healthy_child_cleanup() {
    for kind in ALL {
        let driver = context_test_seams::Guard::new();
        let e = events::Guard::new();
        let s = streams::Guard::new();
        let m = memory::Guard::new();
        let life = PrimaryContextLifetime::new(0)
            .map_err(|e| e.0 as u32)
            .unwrap();
        let mut ctx = CudaContext::new_tracked(Arc::clone(&life))
            .map_err(|e| e.0 as u32)
            .unwrap();
        Arc::get_mut(&mut ctx).unwrap().has_async_alloc = !matches!(kind, Kind::SyncSlice);
        let owner = create(kind, &ctx);
        ctx.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        drop(owner);
        assert!(life.loss().is_none());
        assert_eq!(
            ctx.check_err().map_err(|e| e.0 as u32),
            Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32)
        );
        assert_eq!(ctx.check_err().map_err(|e| e.0 as u32), Ok(()));
        match kind {
            Kind::Event => assert_eq!(e.snapshot().destroyed, 1),
            Kind::Stream => assert_eq!(s.snapshot().destroyed, 1),
            Kind::AsyncSlice => {
                assert_eq!((m.snapshot().async_frees, e.snapshot().destroyed), (1, 2))
            }
            Kind::SyncSlice => assert_eq!(
                (
                    m.snapshot().sync_frees,
                    m.snapshot().stream_synchronizes,
                    e.snapshot().destroyed
                ),
                (1, 1, 2)
            ),
        };
        assert_eq!(
            ctx.new_event(None).map(drop).map_err(|e| e.0 as u32),
            Ok(())
        );
        drop(ctx);
        drop(life);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
#[test]
fn untracked_native_cleanup_failures_keep_the_delayed_error_contract() {
    for kind in ALL {
        for error in [
            sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
            sys::CUresult::CUDA_ERROR_UNKNOWN,
        ] {
            let driver = context_test_seams::Guard::new();
            let e = events::Guard::new();
            let s = streams::Guard::new();
            let m = memory::Guard::new();
            let mut ctx = CudaContext::new(0).map_err(|e| e.0 as u32).unwrap();
            Arc::get_mut(&mut ctx).unwrap().has_async_alloc = !matches!(kind, Kind::SyncSlice);
            let owner = create(kind, &ctx);
            fault(kind, &e, &s, &m, Some(error));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
            fault(kind, &e, &s, &m, None);
            assert!(result.is_ok());
            assert_eq!(ctx.check_err().map_err(|e| e.0 as u32), Err(error as u32));
            assert_eq!(ctx.check_err().map_err(|e| e.0 as u32), Ok(()));
            assert_eq!(ctx.bind_to_thread().map_err(|e| e.0 as u32), Ok(()));
            match kind {
                Kind::Event => assert_eq!(e.snapshot().destroyed, 1),
                Kind::Stream => assert_eq!(s.snapshot().destroyed, 1),
                Kind::AsyncSlice => {
                    assert_eq!((m.snapshot().async_frees, e.snapshot().destroyed), (1, 2))
                }
                Kind::SyncSlice => {
                    assert_eq!((m.snapshot().sync_frees, e.snapshot().destroyed), (1, 2))
                }
            };
            drop(ctx);
            assert_eq!(
                (driver.snapshot().retains, driver.snapshot().releases),
                (1, 1)
            );
        }
    }
}
