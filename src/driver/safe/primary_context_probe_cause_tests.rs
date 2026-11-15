#![cfg(not(any(
    feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060",
    feature = "cuda-11070", feature = "cuda-11080"
)))]

use crate::driver::{result::context_test_seams, sys, CudaContext, DriverError, PrimaryContextLifetime, PrimaryContextLossCause, PrimaryContextRefusal};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Observation { Inactive, ChangedIdentity, NativeStateError, NativeIdentityError }

fn check_observation(observation: Observation) {
    let driver = context_test_seams::Guard::new();
    let lifetime = PrimaryContextLifetime::new(0).map_err(|error| error.0 as u32).unwrap();
    let context = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let sibling = CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|error| error.0 as u32).unwrap();
    let lost = sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED;
    let (cause, observed_id, refusal, id_queries) = match observation {
        Observation::Inactive => {
            driver.set_primary_active(false);
            (PrimaryContextLossCause::PrimaryInactive, None, PrimaryContextRefusal::SyntheticContextDestroyed, 0)
        }
        Observation::ChangedIdentity => {
            driver.set_context_id(42);
            (PrimaryContextLossCause::IdentityChanged, Some(42), PrimaryContextRefusal::SyntheticContextDestroyed, 1)
        }
        Observation::NativeStateError => {
            driver.set_primary_error(Some(lost));
            (PrimaryContextLossCause::DriverStatus(lost), None, PrimaryContextRefusal::DriverStatus(lost), 0)
        }
        Observation::NativeIdentityError => {
            driver.set_context_id_error(Some(lost));
            (PrimaryContextLossCause::DriverStatus(lost), None, PrimaryContextRefusal::DriverStatus(lost), 1)
        }
    };
    let before = driver.snapshot();

    assert_eq!(context.bind_to_thread().map_err(|error| error.0 as u32), Err(lost as u32));

    let first = lifetime.loss().unwrap();
    assert_eq!(first.cause, cause, "successful probe observations must remain distinct from native driver errors");
    assert_eq!((first.recorded_id, first.observed_id), (17, observed_id));
    assert_eq!(first.refusal(), refusal);
    assert_eq!(first.refusal().driver_status() as u32, lost as u32);
    let stopped = driver.snapshot();
    assert_eq!(stopped.primary_queries, before.primary_queries + 1);
    assert_eq!(stopped.id_queries, before.id_queries + id_queries);
    assert_eq!((stopped.queries, stopped.binds), (before.queries, before.binds));
    driver.set_primary_active(true);
    driver.set_context_id(17);
    driver.set_primary_error(None);
    driver.set_context_id_error(None);
    lifetime.record_loss(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN), None);
    assert_eq!(lifetime.loss(), Some(first));
    assert_eq!(context.check_err().map_err(|error| error.0 as u32), Err(lost as u32));
    assert_eq!(sibling.bind_to_thread().map_err(|error| error.0 as u32), Err(lost as u32));
    let diagnostic = format!("{:?} {:?}", first.cause, first.refusal());
    if matches!(observation, Observation::Inactive | Observation::ChangedIdentity) {
        assert!(diagnostic.contains("SyntheticContextDestroyed"));
        assert!(!diagnostic.contains("DriverStatus"));
    } else {
        assert!(diagnostic.contains("DriverStatus"));
        assert!(!diagnostic.contains("Synthetic"));
    }
    let after = driver.snapshot();
    assert_eq!((after.primary_queries, after.id_queries, after.queries, after.binds, after.error_string_queries), (stopped.primary_queries, stopped.id_queries, stopped.queries, stopped.binds, stopped.error_string_queries));
    drop(context);
    drop(sibling);
    drop(lifetime);
    assert_eq!((driver.snapshot().retains, driver.snapshot().releases), (1, 1));
}

#[test]
fn inactive_primary_observation_is_not_a_native_driver_error() { check_observation(Observation::Inactive); }
#[test]
fn changed_identity_observation_is_not_a_native_driver_error() { check_observation(Observation::ChangedIdentity); }
#[test]
fn native_primary_probe_709_keeps_actual_driver_origin() { check_observation(Observation::NativeStateError); }
#[test]
fn native_identity_probe_709_keeps_actual_driver_origin() { check_observation(Observation::NativeIdentityError); }
