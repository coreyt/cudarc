//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaContext], [CudaStream], and more.

pub(crate) mod core;
pub(crate) mod external_memory;
pub(crate) mod graph;
pub(crate) mod launch;
pub(crate) mod mem_pool;
#[cfg(feature = "std")]
pub(crate) mod primary_context;
#[cfg(all(test, feature = "std"))]
pub(crate) mod primary_context_probe_publication_tests;
#[cfg(all(test, feature = "std"))]
pub(crate) mod primary_context_bind_tests;
pub(crate) mod profile;
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
