//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaContext], [CudaStream], and more.

#[cfg(all(test, feature = "std"))]
pub(crate) mod constructor_terminal_result_tests;
pub(crate) mod core;
#[cfg(all(test, feature = "std"))]
pub(crate) mod event_constructor_admission_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod event_use_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod event_record_identity_tests;
pub(crate) mod external_memory;
pub(crate) mod graph;
pub(crate) mod launch;
#[cfg(all(test, feature = "std"))]
pub(crate) mod lifetime_event_cleanup_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod lifetime_stream_cleanup_tests;
pub(crate) mod mem_pool;
#[cfg(feature = "std")]
pub(crate) mod primary_context;
#[cfg(all(test, feature = "std"))]
pub(crate) mod primary_context_bind_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod primary_context_constructor_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod primary_context_probe_publication_tests;
pub(crate) mod profile;
#[cfg(all(test, feature = "std"))]
pub(crate) mod stream_construction_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod stream_constructor_admission_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod tracked_cleanup_bind_failure_tests;
pub(crate) mod unified_memory;

pub use self::core::{
    AllocMode, CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView,
    CudaViewMut, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, HostSlice, PinnedHostSlice,
    SyncOnDrop, ValidAsZeroBits,
};
pub use self::external_memory::{ExternalMemory, MappedBuffer};
pub use self::graph::CudaGraph;
pub use self::launch::{LaunchArgs, LaunchConfig, PushKernelArg};
pub use self::mem_pool::{CudaMemPool, MemPoolProps};
#[cfg(feature = "std")]
pub use self::primary_context::{PrimaryContextLease, PrimaryContextLifetime, PrimaryContextLoss};
pub use self::profile::{profiler_start, profiler_stop, Profiler};
pub use self::unified_memory::{UnifiedSlice, UnifiedView, UnifiedViewMut};
pub use crate::driver::result::DriverError;
