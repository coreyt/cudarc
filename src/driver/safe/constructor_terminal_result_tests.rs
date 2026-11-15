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
    result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime,
};
use std::sync::{atomic::Ordering, Arc};

#[derive(Clone, Copy, Debug)]
enum NativePhase {
    StreamSync,
    StreamCreate,
    EventCreate,
}

fn set_native_error(
    phase: NativePhase,
    events: &event_seams::Guard,
    streams: &stream_seams::Guard,
    error: Option<sys::CUresult>,
) {
    match phase {
        NativePhase::StreamSync => streams.set_context_synchronize_error(error),
        NativePhase::StreamCreate => streams.set_create_error(error),
        NativePhase::EventCreate => events.set_create_error(error),
    }
}

fn construct_and_drop(phase: NativePhase, context: &Arc<CudaContext>) -> Result<(), DriverError> {
    match phase {
        NativePhase::StreamSync | NativePhase::StreamCreate => context.new_stream().map(drop),
        NativePhase::EventCreate => context.new_event(None).map(drop),
    }
}

#[test]
fn terminal_constructor_result_is_cached_before_later_native_dispatch() {
    for phase in [
        NativePhase::StreamSync,
        NativePhase::StreamCreate,
        NativePhase::EventCreate,
    ] {
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
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        set_native_error(phase, &events, &streams, Some(lost));

        let failed = construct_and_drop(phase, &context).map_err(|error| error.0 as u32);

        let failed_dispatch = driver.snapshot();
        let stream_calls = streams.snapshot();
        let event_calls = events.snapshot();
        let first_record = lifetime
            .state
            .lock()
            .unwrap()
            .loss
            .map(|loss| (loss.recorded_id, loss.observed_id, loss.refusal().driver_status() as u32));
        set_native_error(phase, &events, &streams, None);
        let first_read = context.check_err().map_err(|error| error.0 as u32);
        let second_read = context.check_err().map_err(|error| error.0 as u32);
        let sibling_read = sibling.check_err().map_err(|error| error.0 as u32);
        let later_bind = sibling.bind_to_thread().map_err(|error| error.0 as u32);
        let later_create = construct_and_drop(phase, &context).map_err(|error| error.0 as u32);
        sibling.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
        let after_oom = sibling.check_err().map_err(|error| error.0 as u32);
        let after = driver.snapshot();

        assert_eq!(failed, Err(lost as u32));
        assert_eq!(
            first_record,
            Some((17, None, lost as u32)),
            "native {phase:?} terminal result must publish shared loss before returning"
        );
        assert_eq!(first_read, Err(lost as u32));
        assert_eq!(second_read, Err(lost as u32));
        assert_eq!(sibling_read, Err(lost as u32));
        assert_eq!(later_bind, Err(lost as u32));
        assert_eq!(later_create, Err(lost as u32));
        assert_eq!(after_oom, Err(lost as u32));
        assert_eq!(
            (
                after.primary_queries,
                after.id_queries,
                after.queries,
                after.binds
            ),
            (
                failed_dispatch.primary_queries,
                failed_dispatch.id_queries,
                failed_dispatch.queries,
                failed_dispatch.binds
            )
        );
        assert_eq!(
            (
                streams.snapshot().context_synchronizes,
                streams.snapshot().create_attempts,
                streams.snapshot().created,
                streams.snapshot().destroyed
            ),
            (
                stream_calls.context_synchronizes,
                stream_calls.create_attempts,
                0,
                0
            )
        );
        assert_eq!(
            (
                events.snapshot().create_attempts,
                events.snapshot().created,
                events.snapshot().destroyed
            ),
            (event_calls.create_attempts, 0, 0)
        );
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
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

#[test]
fn ordinary_constructor_oom_remains_retryable_without_a_stored_error() {
    for phase in [
        NativePhase::StreamSync,
        NativePhase::StreamCreate,
        NativePhase::EventCreate,
    ] {
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
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        set_native_error(phase, &events, &streams, Some(oom));

        assert_eq!(
            construct_and_drop(phase, &context).map_err(|error| error.0 as u32),
            Err(oom as u32)
        );

        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(sibling.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert!(lifetime.state.lock().unwrap().loss.is_none());
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 0);
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        set_native_error(phase, &events, &streams, None);
        assert_eq!(
            construct_and_drop(phase, &context).map_err(|error| error.0 as u32),
            Ok(())
        );
        assert_eq!(events.snapshot().created + streams.snapshot().created, 1);
        assert_eq!(
            events.snapshot().destroyed + streams.snapshot().destroyed,
            1
        );
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
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
fn untracked_constructor_errors_preserve_the_returned_error_contract() {
    for phase in [
        NativePhase::StreamSync,
        NativePhase::StreamCreate,
        NativePhase::EventCreate,
    ] {
        let driver = context_test_seams::Guard::new();
        let events = event_seams::Guard::new();
        let streams = stream_seams::Guard::new();
        let context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
        let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
        set_native_error(phase, &events, &streams, Some(lost));

        assert_eq!(
            construct_and_drop(phase, &context).map_err(|error| error.0 as u32),
            Err(lost as u32)
        );

        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert!(context.primary_lifetime.is_none());
        assert_eq!(context.num_streams.load(Ordering::Relaxed), 0);
        set_native_error(phase, &events, &streams, None);
        drop(context);
        assert_eq!(
            (driver.snapshot().retains, driver.snapshot().releases),
            (1, 1)
        );
    }
}
