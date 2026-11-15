#![cfg(not(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080")))]

use super::{lifetime_event_cleanup_tests::seams as event_seams, lifetime_pinned_cleanup_tests::seams};
use crate::driver::{result::context_test_seams, sys, CudaContext, PrimaryContextLifetime, PrimaryContextLossCause};
use std::sync::Arc;

fn check_cleanup_result(free_failure: bool) {
    for error in [sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,sys::CUresult::CUDA_ERROR_UNKNOWN] {
        let driver=context_test_seams::Guard::new();let events=event_seams::Guard::new();let host=seams::Guard::new();
        let lifetime=PrimaryContextLifetime::new(0).map_err(|e|e.0 as u32).unwrap();
        let context=CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|e|e.0 as u32).unwrap();
        let allocation=unsafe{context.alloc_pinned::<u64>(2)}.map_err(|e|e.0 as u32).unwrap();
        if free_failure {host.set_free_error(Some(error));} else {events.set_synchronize_error(Some(error));}
        drop(allocation);
        assert_eq!(lifetime.loss().map(|loss|(loss.recorded_id,loss.observed_id,loss.cause)),Some((17,None,PrimaryContextLossCause::DriverStatus(error))),"actual pinned cleanup failure must freeze its native cause before abandonment");
        assert_eq!(host.snapshot().free_attempts,usize::from(free_failure));assert_eq!(host.snapshot().freed,0);
        assert_eq!((events.snapshot().synchronized,events.snapshot().destroyed),(1,0));
        host.set_free_error(None);events.set_synchronize_error(None);let before=driver.snapshot();
        for _ in 0..2 {assert_eq!(context.check_err().map_err(|e|e.0 as u32),Err(error as u32));}
        assert_eq!(unsafe{context.alloc_pinned::<u64>(2)}.map(drop).map_err(|e|e.0 as u32),Err(error as u32));
        assert_eq!(host.snapshot().attempts,1);let after=driver.snapshot();
        assert_eq!((after.primary_queries,after.id_queries,after.queries,after.binds),(before.primary_queries,before.id_queries,before.queries,before.binds));
        drop(context);drop(lifetime);assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
    }
}
#[test]
fn native_pinned_cleanup_sync_failure_freezes_before_free_or_later_growth(){check_cleanup_result(false);}
#[test]
fn native_pinned_cleanup_free_failure_freezes_before_later_growth(){check_cleanup_result(true);}

fn check_allocation_result(error:sys::CUresult){
    let driver=context_test_seams::Guard::new();let events=event_seams::Guard::new();let host=seams::Guard::new();
    let lifetime=PrimaryContextLifetime::new(0).map_err(|e|e.0 as u32).unwrap();let context=CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|e|e.0 as u32).unwrap();
    host.set_error(Some(error));
    assert_eq!(unsafe{context.alloc_pinned::<u64>(2)}.map(drop).map_err(|e|e.0 as u32),Err(error as u32));
    assert_eq!(lifetime.loss().map(|loss|loss.cause),Some(PrimaryContextLossCause::DriverStatus(error)));
    assert_eq!((host.snapshot().attempts,host.snapshot().allocated,host.snapshot().free_attempts),(1,0,0));
    assert_eq!((events.snapshot().created,events.snapshot().destroyed),(1,0));
    host.set_error(None);let before=driver.snapshot();
    assert_eq!(unsafe{context.alloc_pinned::<u64>(2)}.map(drop).map_err(|e|e.0 as u32),Err(error as u32));
    assert_eq!(host.snapshot().attempts,1);let after=driver.snapshot();assert_eq!((after.primary_queries,after.id_queries,after.queries,after.binds),(before.primary_queries,before.id_queries,before.queries,before.binds));
    drop(context);drop(lifetime);assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
}
#[test]
fn native_pinned_allocation_709_publishes_before_partial_event_cleanup(){check_allocation_result(sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);}
#[test]
fn native_pinned_allocation_201_publishes_before_partial_event_cleanup(){check_allocation_result(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT);}

#[test]
fn ordinary_pinned_data_access_sync_oom_remains_retryable(){
    let driver=context_test_seams::Guard::new();let events=event_seams::Guard::new();let host=seams::Guard::new();
    let lifetime=PrimaryContextLifetime::new(0).map_err(|e|e.0 as u32).unwrap();let context=CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|e|e.0 as u32).unwrap();
    let allocation=unsafe{context.alloc_pinned::<u64>(2)}.map_err(|e|e.0 as u32).unwrap();
    events.set_synchronize_error(Some(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
    assert_eq!(allocation.as_slice().map(|values|values.len()).map_err(|e|e.0 as u32),Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY as u32));
    assert!(lifetime.loss().is_none());assert_eq!(context.check_err().map_err(|e|e.0 as u32),Ok(()));
    events.set_synchronize_error(None);assert_eq!(allocation.as_slice().map(|values|values.len()).map_err(|e|e.0 as u32),Ok(2));
    drop(allocation);assert_eq!((host.snapshot().freed,events.snapshot().destroyed),(1,1));
    drop(context);drop(lifetime);assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
}

#[test]
fn native_pinned_data_sync_201_publishes_before_further_use(){
    let driver=context_test_seams::Guard::new();let events=event_seams::Guard::new();let host=seams::Guard::new();let lifetime=PrimaryContextLifetime::new(0).map_err(|e|e.0 as u32).unwrap();let context=CudaContext::new_tracked(Arc::clone(&lifetime)).map_err(|e|e.0 as u32).unwrap();let allocation=unsafe{context.alloc_pinned::<u64>(2)}.map_err(|e|e.0 as u32).unwrap();let error=sys::CUresult::CUDA_ERROR_INVALID_CONTEXT;events.set_synchronize_error(Some(error));let result=allocation.as_slice().map(|values|values.len()).map_err(|e|e.0 as u32);events.set_synchronize_error(None);assert_eq!(result,Err(error as u32));assert_eq!(lifetime.loss().map(|loss|loss.cause),Some(PrimaryContextLossCause::DriverStatus(error)));let before=driver.snapshot();assert_eq!(allocation.as_slice().map(|values|values.len()).map_err(|e|e.0 as u32),Err(error as u32));assert_eq!(events.snapshot().synchronized,1);let after=driver.snapshot();assert_eq!((after.primary_queries,after.id_queries,after.queries,after.binds),(before.primary_queries,before.id_queries,before.queries,before.binds));drop(allocation);assert_eq!(host.snapshot().free_attempts,0);assert_eq!(events.snapshot().destroyed,0);drop(context);drop(lifetime);assert_eq!((driver.snapshot().retains,driver.snapshot().releases),(1,1));
}
