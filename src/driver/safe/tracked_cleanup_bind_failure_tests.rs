#![cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
)))]

use super::{
    lifetime_event_cleanup_tests::seams as event_seams,
    lifetime_stream_cleanup_tests::seams as stream_seams,
};
use crate::driver::{
    result::context_test_seams, sys, CudaContext, CudaEvent, CudaStream, DriverError,
    PrimaryContextLifetime,
};
use std::sync::{atomic::Ordering, Arc};

#[derive(Clone, Copy, Debug)]
enum OwnerKind {
    Event,
    Stream,
}

#[derive(Clone, Copy, Debug)]
enum FailurePoint {
    CurrentQuery,
    CurrentBind,
}

enum Owner {
    Event(CudaEvent),
    Stream(Arc<CudaStream>),
}

fn create_owner(kind: OwnerKind, context: &Arc<CudaContext>) -> Owner {
    match kind {
        OwnerKind::Event => Owner::Event(
            context
                .new_event(None)
                .map_err(|error| error.0 as u32)
                .unwrap(),
        ),
        OwnerKind::Stream => Owner::Stream(
            context
                .new_stream()
                .map_err(|error| error.0 as u32)
                .unwrap(),
        ),
    }
}

fn check_native_failure(error: sys::CUresult) {
    for kind in [OwnerKind::Event, OwnerKind::Stream] {
        for point in [FailurePoint::CurrentQuery, FailurePoint::CurrentBind] {
            let driver = context_test_seams::Guard::new();
            let events = event_seams::Guard::new();
            let streams = stream_seams::Guard::new();
            let lifetime = PrimaryContextLifetime::new(0)
                .map_err(|error| error.0 as u32)
                .unwrap();
            let context = CudaContext::new_tracked(Arc::clone(&lifetime))
                .map_err(|error| error.0 as u32)
                .unwrap();
            let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
                .map_err(|error| error.0 as u32)
                .unwrap();
            let owner = create_owner(kind, &context);
            assert_eq!(events.snapshot().created + streams.snapshot().created, 1);
            let before = driver.snapshot();
            match point {
                FailurePoint::CurrentQuery => driver.set_current_query_error(Some(error)),
                FailurePoint::CurrentBind => {
                    driver.clear_current();
                    driver.set_current_bind_error(Some(error));
                }
            }

            drop(owner);

            let failed = driver.snapshot();
            assert_eq!(
                events.snapshot().destroyed + streams.snapshot().destroyed,
                0
            );
            assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
            assert_eq!(failed.queries, before.queries + 1);
            assert_eq!(
                failed.binds,
                before.binds + usize::from(matches!(point, FailurePoint::CurrentBind))
            );
            driver.set_current_query_error(None);
            driver.set_current_bind_error(None);
            let first_read = context.check_err().map_err(|error| error.0 as u32);
            let second_read = context.check_err().map_err(|error| error.0 as u32);
            let sibling_read = sibling.check_err().map_err(|error| error.0 as u32);
            let later_bind = context.bind_to_thread().map_err(|error| error.0 as u32);
            let sibling_bind = sibling.bind_to_thread().map_err(|error| error.0 as u32);
            sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
            let after_oom = sibling.check_err().map_err(|error| error.0 as u32);
            let after = driver.snapshot();

            assert_eq!(first_read, Err(error as u32));
            assert_eq!(
                second_read,
                Err(error as u32),
                "native {point:?} failure during {kind:?} cleanup must survive error reads"
            );
            assert_eq!(sibling_read, Err(error as u32));
            assert_eq!(later_bind, Err(error as u32));
            assert_eq!(sibling_bind, Err(error as u32));
            assert_eq!(after_oom, Err(error as u32));
            assert_eq!(
                (
                    after.primary_queries,
                    after.id_queries,
                    after.queries,
                    after.binds
                ),
                (
                    failed.primary_queries,
                    failed.id_queries,
                    failed.queries,
                    failed.binds
                )
            );
            assert_eq!(
                lifetime.state.lock().unwrap().loss.unwrap().error as u32,
                error as u32
            );
            assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
            drop(context);
            drop(sibling);
            drop(lifetime);
            assert_eq!(
                (driver.snapshot().retains, driver.snapshot().releases),
                (1, 1)
            );
        }
    }
}

#[test]
fn native_cleanup_bind_failure_remains_shared_after_error_reads() {
    check_native_failure(sys::CUresult::CUDA_ERROR_UNKNOWN);
}

#[test]
fn native_cleanup_oom_is_classified_by_dispatch_origin() {
    check_native_failure(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);
}

#[test]
fn preexisting_ordinary_oom_does_not_become_terminal_cleanup_failure() {
    for kind in [OwnerKind::Event, OwnerKind::Stream] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let streams = stream_seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0)
            .map_err(|error| error.0 as u32)
            .unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let sibling = CudaContext::new_tracked(Arc::clone(&lifetime))
            .map_err(|error| error.0 as u32)
            .unwrap();
        let owner = create_owner(kind, &context);
        let error = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        context.record_err::<()>(Err(DriverError(error)));
        let before = driver.snapshot();

        drop(owner);

        let after = driver.snapshot();
        assert_eq!(
            events.snapshot().destroyed + streams.snapshot().destroyed,
            0
        );
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        assert_eq!((after.queries, after.binds), (before.queries, before.binds));
        assert_eq!(
            context.check_err().map_err(|error| error.0 as u32),
            Err(error as u32)
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(sibling.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(
            sibling.bind_to_thread().map_err(|error| error.0 as u32),
            Ok(())
        );
        let healthy = driver.snapshot();
        assert_eq!(healthy.queries, after.queries + 1);
        assert_eq!(
            events.snapshot().destroyed + streams.snapshot().destroyed,
            0
        );
        drop(context);
        drop(sibling);
        drop(lifetime);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
