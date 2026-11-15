#![cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]

use super::lifetime_event_cleanup_tests::seams;
use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum Point { GetCurrent, SetCurrent }

fn force(driver: &context_test_seams::Guard, point: Point, error: Option<sys::CUresult>) {
    match point {
        Point::GetCurrent => driver.set_current_query_error(error),
        Point::SetCurrent => {
            if error.is_some() { driver.clear_current(); }
            driver.set_current_bind_error(error);
        }
    }
}

#[test]
fn native_terminal_bind_results_publish_first_loss_before_return() {
    for error in [sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT] {
        for point in [Point::GetCurrent, Point::SetCurrent] {
            let driver = context_test_seams::Guard::new();
            let events = seams::Guard::new();
            let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
            let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
            let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
            force(&driver, point, Some(error));
            let before = driver.snapshot();

            let failed = context.bind_to_thread().map_err(|error| error.0 as u32);

            let stopped = driver.snapshot();
            let first_record = lifetime.loss().map(|loss| (loss.recorded_id, loss.observed_id, loss.error as u32));
            force(&driver, point, None);
            let first_read = context.check_err().map_err(|error| error.0 as u32);
            let second_read = context.check_err().map_err(|error| error.0 as u32);
            let child = context.new_event(None).map(drop).map_err(|error| error.0 as u32);
            let nested = lifetime.admit().map(drop).map_err(|error| error.0 as u32);
            context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)));
            let after_oom = context.check_err().map_err(|error| error.0 as u32);
            let after = driver.snapshot();

            assert_eq!(failed, Err(error as u32));
            assert_eq!(stopped.queries, before.queries + 1);
            assert_eq!(stopped.binds, before.binds + usize::from(matches!(point, Point::SetCurrent)));
            assert_eq!(first_record, Some((17, None, error as u32)), "actual native {point:?} terminal result must be shared before return");
            assert_eq!(first_read, Err(error as u32));
            assert_eq!(second_read, Err(error as u32));
            assert_eq!(child, Err(error as u32));
            assert_eq!(nested, Err(error as u32));
            assert_eq!(after_oom, Err(error as u32));
            assert_eq!(events.snapshot().create_attempts, 0, "no following child/library-style native entry may proceed");
            assert_eq!((after.primary_queries, after.id_queries, after.queries, after.binds), (stopped.primary_queries, stopped.id_queries, stopped.queries, stopped.binds));
            assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
            drop(outer);
            drop(context);
            drop(lifetime);
            assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
        }
    }
}

#[test]
fn ordinary_native_bind_oom_remains_retryable_without_delayed_error() {
    for point in [Point::GetCurrent, Point::SetCurrent] {
        let driver = context_test_seams::Guard::new();
        let events = seams::Guard::new();
        let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
        let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
        let outer = lifetime.admit().map_err(|error| error.0 as u32).unwrap();
        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        force(&driver, point, Some(oom));

        assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Err(oom as u32));

        assert!(lifetime.loss().is_none());
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
        assert_eq!(lifetime.state.lock().unwrap().active_leases, 1);
        force(&driver, point, None);
        assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Ok(()));
        let event = context.new_event(None).map_err(|error| error.0 as u32).unwrap();
        assert_eq!(events.snapshot().created, 1);
        drop(event);
        drop(outer);
        drop(context);
        drop(lifetime);
        assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
    }
}

#[test]
fn untracked_terminal_bind_errors_keep_the_original_return_contract() {
    let driver = context_test_seams::Guard::new();
    let context = CudaContext::new(0).map_err(|error| error.0 as u32).unwrap();
    for error in [sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED, sys::CUresult::CUDA_ERROR_INVALID_CONTEXT] {
        driver.set_current_query_error(Some(error));
        assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Err(error as u32));
        assert_eq!(context.check_err().map_err(|error| error.0 as u32), Ok(()));
    }
    driver.set_current_query_error(None);
    drop(context);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}
