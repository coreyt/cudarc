use crate::driver::{
    result::{self, DriverError},
    sys::{self, CUfunc_cache_enum, CUfunction_attribute_enum},
};

use std::{
    ffi::CString,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    string::String,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
    sync::Arc,
    vec::Vec,
};

/// Represents a CUDA context on a certain device.
///
/// - [`CudaContext::new()`] retains the device's primary context.
/// - [`CudaContext::new_non_primary()`] creates an independent non-primary context.
/// - [`CudaContext::new_cig()`] creates a non-primary context with CiG (CUDA in Graphics) parameters (CUDA 12.050+).
/// - [`CudaContext::from_raw_context()`] wraps a pre-existing raw `CUcontext`.
///
/// This is the entrypoint to using any cuda calls, all objects maintain a pointer to `Arc<CudaContext>`
/// to ensure proper lifetimes.
///
/// # On thread safety
///
/// This object is thread safe and can be shared/used on multiple threads. All safe apis call
/// [CudaContext::bind_to_thread()] before doing work in a certain context.
#[derive(Debug)]
pub struct CudaContext {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_ctx: sys::CUcontext,
    pub(crate) ordinal: usize,
    pub(crate) has_async_alloc: bool,
    /// The pool every allocation of this context draws from, set only by
    /// [CudaContext::new_with_mem_pool] (FathomDB patch). Dropped after the
    /// context is released, so the pool outlives the context.
    pub(crate) mem_pool: Option<Arc<super::mem_pool::CudaMemPool>>,
    /// Whether this wraps a primary context (true) or a non-primary context (false).
    /// Primary contexts are released via `cuDevicePrimaryCtxRelease`, while non-primary
    /// contexts are destroyed via `cuCtxDestroy_v2`.
    pub(crate) is_primary: bool,
    pub(crate) num_streams: AtomicUsize,
    pub(crate) event_tracking: AtomicBool,
    pub(crate) error_state: AtomicU32,
}

// FATHOMDB PATCH BEGIN: private memory pool (see FATHOMDB.md, item 5).
/// The allocator a [CudaContext] uses, fixed when it is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocMode {
    /// Stream-ordered allocation (`cuMemAllocAsync` / `cuMemFreeAsync`) from
    /// the device's current memory pool.
    Default,
    /// Synchronous allocation (`cuMemAlloc` / `cuMemFree`).
    Synchronous,
    /// Stream-ordered allocation from the context's own pool
    /// (`cuMemAllocFromPoolAsync` / `cuMemFreeAsync`); see
    /// [CudaContext::new_with_mem_pool]. The device's current pool is neither
    /// used nor changed.
    Private,
}

/// A context's [AllocMode]: its own pool, if it has one, decides; otherwise
/// the device's stream-ordered or synchronous allocator.
fn alloc_mode_of(has_mem_pool: bool, has_async_alloc: bool) -> AllocMode {
    match (has_mem_pool, has_async_alloc) {
        (true, _) => AllocMode::Private,
        (false, true) => AllocMode::Default,
        (false, false) => AllocMode::Synchronous,
    }
}
// FATHOMDB PATCH END

// FATHOMDB PATCH BEGIN: allocator fallback (see FATHOMDB.md, items 1-3).
/// Whether this build applies the allocator fallback. It was measured only on
/// Jetson (Tegra), so it is limited to aarch64 Linux; every other target keeps
/// upstream 0.19.10's allocator selection and zero-length handling unchanged.
const SYNC_FALLBACK: bool = cfg!(all(target_os = "linux", target_arch = "aarch64"));

/// Whether a device should use stream-ordered allocation.
///
/// Without `sync_fallback` this is upstream's rule: stream-ordered allocation
/// whenever the pool attribute is positive, and `default_pool` is never
/// queried. With it:
///
/// Stream-ordered allocation (`cuMemAllocAsync`) draws from the device's
/// default memory pool. A driver can report memory-pool support yet be unable
/// to create that pool: on Jetson (Tegra) the pool needs one contiguous range
/// of about a third of device memory in the process address space, and in a
/// fragmented process `cuDeviceGetDefaultMemPool` and every `cuMemAllocAsync`
/// return `CUDA_ERROR_OUT_OF_MEMORY` while `cuMemAlloc` succeeds. Such a
/// device allocates synchronously instead.
///
/// `default_pool` is queried only when pools are supported. Only the results
/// that mean "this device has no usable default pool" select the synchronous
/// allocator: `CUDA_ERROR_OUT_OF_MEMORY` (measured above) and
/// `CUDA_ERROR_NOT_SUPPORTED` (the documented answer of a device without
/// pools). Any other error, for example a sticky error from earlier
/// asynchronous work, says nothing about the pool and is returned, so a
/// healthy device is never silently downgraded.
fn async_alloc_decision(
    sync_fallback: bool,
    memory_pools_supported: i32,
    default_pool: impl FnOnce() -> Result<sys::CUmemoryPool, DriverError>,
) -> Result<bool, DriverError> {
    if !sync_fallback || memory_pools_supported <= 0 {
        return Ok(memory_pools_supported > 0);
    }
    match default_pool() {
        Ok(_) => Ok(true),
        Err(DriverError(
            sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY | sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
        )) => Ok(false),
        Err(error) => Err(error),
    }
}

/// The allocator decision for each device, made once per process.
///
/// A buffer may outlive or move between [CudaContext] wrappers of one device
/// (`CudaContext::new` may be called repeatedly for the same primary context,
/// and [CudaSlice::leak] / [CudaStream::upgrade_device_ptr] move raw
/// pointers), and it is freed by whichever wrapper owns it last. NVIDIA
/// allows freeing across the two allocator families (see
/// [CudaStream::upgrade_device_ptr]), so this is not needed for a correct
/// free; keying the decision by device keeps every wrapper of a device on one
/// allocator, which is the configuration measured on the Jetson AGX Orin
/// (FATHOMDB.md, item 2). A cached `true` stays valid: on the measured Jetson
/// the default pool, once obtained, kept its address range and handle when
/// its primary context was released to refcount 0 or its non-primary context
/// was destroyed, and stayed usable on re-retain with every hole in the window
/// blocked (FathomDB repository,
/// `dev/plans/runs/0.8.27-slice-110-tegra/pool-teardown-evidence/`). A cached
/// `false` is conservative: if the address space later frees up, the device
/// stays on the slower but correct synchronous path. Used only on aarch64
/// Linux.
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "aarch64")),
    allow(dead_code)
)]
struct AllocModeByDevice(std::sync::Mutex<Vec<(sys::CUdevice, bool)>>);

#[cfg_attr(
    not(all(target_os = "linux", target_arch = "aarch64")),
    allow(dead_code)
)]
impl AllocModeByDevice {
    const fn new() -> Self {
        Self(std::sync::Mutex::new(Vec::new()))
    }

    /// Returns the recorded decision for `cu_device`, or runs `decide` and
    /// records its result if it succeeds. The lock is held across `decide`, so
    /// racing constructors cannot record different answers; a failed `decide`
    /// records nothing and the next wrapper probes again.
    fn get_or_decide(
        &self,
        cu_device: sys::CUdevice,
        decide: impl FnOnce() -> Result<bool, DriverError>,
    ) -> Result<bool, DriverError> {
        // Entries are pushed only after a successful `decide`, so a poisoned
        // lock still guards a consistent list.
        let mut decided = self.0.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(&(_, async_alloc)) = decided.iter().find(|(device, _)| *device == cu_device) {
            return Ok(async_alloc);
        }
        let async_alloc = decide()?;
        decided.push((cu_device, async_alloc));
        Ok(async_alloc)
    }

    /// The recorded decision for `cu_device`, if any.
    #[cfg(test)]
    fn decided(&self, cu_device: sys::CUdevice) -> Option<bool> {
        let decided = self.0.lock().unwrap_or_else(|poison| poison.into_inner());
        decided
            .iter()
            .find(|(device, _)| *device == cu_device)
            .map(|&(_, async_alloc)| async_alloc)
    }
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
static PROCESS_ALLOC_MODE: AllocModeByDevice = AllocModeByDevice::new();

/// The allocator decision for the device of a newly created or wrapped
/// context: on aarch64 Linux, the process-wide decision
/// ([async_alloc_decision] with the fallback). If it fails, `release_context`
/// runs so the constructor does not leak the context it created.
///
/// Every allocation and its matching free branch on the stored result, so all
/// wrappers of a device use one allocator.
///
/// # Safety
/// `cu_ctx` must be a live context created for `cu_device`; it is made current
/// on the calling thread.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn select_async_alloc(
    cu_device: sys::CUdevice,
    cu_ctx: sys::CUcontext,
    release_context: impl FnOnce(),
) -> Result<bool, DriverError> {
    PROCESS_ALLOC_MODE
        .get_or_decide(cu_device, || {
            let memory_pools_supported = result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )?;
            if memory_pools_supported > 0 {
                result::ctx::set_current(cu_ctx)?;
            }
            async_alloc_decision(SYNC_FALLBACK, memory_pools_supported, || {
                result::device::get_default_mem_pool(cu_device)
            })
        })
        .inspect_err(|_| release_context())
}

/// Upstream's allocator decision, unchanged: the pool attribute alone, per
/// context, with upstream's error handling (the context is not released).
///
/// # Safety
/// `cu_device` must be a valid device.
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
unsafe fn select_async_alloc(
    cu_device: sys::CUdevice,
    _cu_ctx: sys::CUcontext,
    _release_context: impl FnOnce(),
) -> Result<bool, DriverError> {
    let memory_pools_supported = result::device::get_attribute(
        cu_device,
        sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
    )?;
    async_alloc_decision(SYNC_FALLBACK, memory_pools_supported, || {
        unreachable!("upstream's rule never queries the default pool")
    })
}

/// Whether a synchronous request of `num_bytes` returns a null pointer without
/// a driver call. With the fallback, zero bytes do: `cuMemAlloc` rejects a
/// zero-byte request with `CUDA_ERROR_INVALID_VALUE`, whereas `cuMemAllocAsync`
/// succeeds with a null pointer, and matching that keeps zero-length buffers
/// working whichever path a context selected. Without it (upstream), every
/// request reaches `cuMemAlloc`.
fn sync_zero_bytes_is_null(sync_fallback: bool, num_bytes: usize) -> bool {
    sync_fallback && num_bytes == 0
}

/// Whether a synchronously allocated pointer is passed to `cuMemFree`. With
/// the fallback a null pointer is not; upstream frees every pointer.
fn sync_free_needed(sync_fallback: bool, cu_device_ptr: sys::CUdeviceptr) -> bool {
    !(sync_fallback && cu_device_ptr == 0)
}

/// Synchronous allocation, null for zero bytes with the fallback
/// ([sync_zero_bytes_is_null]).
///
/// # Safety
/// As for [result::malloc_sync].
unsafe fn malloc_sync_or_null(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    if sync_zero_bytes_is_null(SYNC_FALLBACK, num_bytes) {
        Ok(0)
    } else {
        result::malloc_sync(num_bytes)
    }
}
// FATHOMDB PATCH END

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        self.record_err(self.bind_to_thread());
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            if self.is_primary {
                self.record_err(unsafe { result::primary_ctx::release(self.cu_device) });
            } else {
                // Non-primary contexts (e.g., CiG) are destroyed directly.
                self.record_err(unsafe { sys::cuCtxDestroy_v2(ctx).result() });
            }
        }
    }
}

impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        self.cu_device == other.cu_device
            && self.cu_ctx == other.cu_ctx
            && self.ordinal == other.ordinal
    }
}
impl Eq for CudaContext {}

impl CudaContext {
    /// Creates a new context on the specified device ordinal.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        let cu_ctx = unsafe { result::primary_ctx::retain(cu_device) }?;
        // FATHOMDB PATCH: pool support alone does not guarantee a usable default pool.
        let has_async_alloc = unsafe {
            select_async_alloc(cu_device, cu_ctx, || {
                let _ = result::primary_ctx::release(cu_device);
            })
        }?;
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc,
            mem_pool: None,
            is_primary: true,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Creates a context on the primary context of device `ordinal` whose
    /// allocations all come from `pool` (`cuMemAllocFromPoolAsync`); they are
    /// freed with `cuMemFreeAsync`, which returns memory to the pool it came
    /// from. Its [CudaContext::alloc_mode] is [AllocMode::Private] (FathomDB
    /// patch; see `FATHOMDB.md`, item 5).
    ///
    /// The device's current memory pool is neither read nor changed, and on
    /// aarch64 Linux the device's process-wide allocator decision is neither
    /// read nor recorded, so every other context of the device, in this
    /// library or another, keeps its own allocator.
    ///
    /// The context holds `pool`, and every [CudaStream] and [CudaSlice] holds
    /// the context, so the pool is destroyed only after the last of them.
    /// A pointer taken out with [CudaSlice::leak] may be upgraded onto any
    /// context of the device with [CudaStream::upgrade_device_ptr], whose
    /// `# Safety` section states the conditions; `cuMemFreeAsync` returns it
    /// to this pool, and `cuMemFree` also releases it.
    ///
    /// An allocation made on a stream that is being captured into a graph is
    /// owned by the graph, not by `pool`: NVIDIA's `cuMemAllocFromPoolAsync`
    /// creates an allocation node during capture and uses the pool's
    /// properties only for the node's creation parameters. Such memory is
    /// outside the pool's `maxSize` and release threshold, and the pool's
    /// counters do not include it; callers who need a bound on captured
    /// memory must size their graphs.
    ///
    /// # Errors
    /// `CUDA_ERROR_INVALID_VALUE` if `pool` belongs to another device (no
    /// context is retained), and the driver errors of `cuInit`, `cuDeviceGet`,
    /// retaining the primary context and binding it to the calling thread.
    pub fn new_with_mem_pool(
        ordinal: usize,
        pool: Arc<super::mem_pool::CudaMemPool>,
    ) -> Result<Arc<Self>, DriverError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        if pool.cu_device != cu_device {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let cu_ctx = unsafe { result::primary_ctx::retain(cu_device) }?;
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc: true,
            mem_pool: Some(pool),
            is_primary: true,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Creates a new non-primary CUDA context on the specified device ordinal.
    ///
    /// Unlike [`CudaContext::new()`] which retains the device's primary context,
    /// this creates an independent context via `cuCtxCreate_v4` (CUDA 12.050+)
    /// or `cuCtxCreate_v3` (CUDA 11.040–12.040). On drop, the context is
    /// destroyed via `cuCtxDestroy_v2`.
    ///
    /// `flags` controls scheduling policy and other options — use 0 for defaults
    /// (`CU_CTX_SCHED_AUTO`). See [`sys::CUctx_flags`] for available flags.
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    ))]
    pub fn new_non_primary(ordinal: usize, flags: u32) -> Result<Arc<Self>, DriverError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;

        #[cfg(any(
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000",
            feature = "cuda-13010"
        ))]
        let cu_ctx = unsafe { result::ctx::create_v4(std::ptr::null_mut(), flags, cu_device) }?;

        #[cfg(not(any(
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000",
            feature = "cuda-13010"
        )))]
        let cu_ctx = unsafe { result::ctx::create_v3(flags, cu_device) }?;

        // FATHOMDB PATCH: pool support alone does not guarantee a usable default pool.
        let has_async_alloc = unsafe {
            select_async_alloc(cu_device, cu_ctx, || {
                let _ = sys::cuCtxDestroy_v2(cu_ctx);
            })
        }?;
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc,
            mem_pool: None,
            is_primary: false,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Creates a new CUDA context with CiG (CUDA in Graphics) parameters.
    ///
    /// This uses `cuCtxCreate_v4` to create a non-primary context that shares
    /// resources with a graphics API (e.g., D3D12). Requires CUDA 12.050+.
    ///
    /// `flags` controls scheduling policy and other options — use 0 for defaults.
    /// `cig_params` specifies the CiG shared data type and pointer.
    ///
    /// On drop, the context is destroyed via `cuCtxDestroy_v2`.
    #[cfg(any(
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    ))]
    pub fn new_cig(
        ordinal: usize,
        flags: u32,
        cig_params: &mut sys::CUctxCigParam,
    ) -> Result<Arc<Self>, DriverError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        let mut ctx_create_params = sys::CUctxCreateParams_st {
            execAffinityParams: std::ptr::null_mut(),
            numExecAffinityParams: 0,
            cigParams: cig_params as *mut sys::CUctxCigParam,
        };
        let cu_ctx = unsafe { result::ctx::create_v4(&mut ctx_create_params, flags, cu_device) }?;
        // FATHOMDB PATCH: pool support alone does not guarantee a usable default pool.
        let has_async_alloc = unsafe {
            select_async_alloc(cu_device, cu_ctx, || {
                let _ = sys::cuCtxDestroy_v2(cu_ctx);
            })
        }?;
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc,
            mem_pool: None,
            is_primary: false,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Wrap a pre-existing raw CUcontext (e.g., a CiG context created via `cuCtxCreate_v4`).
    ///
    /// The context must already be valid and will be made current on the calling thread.
    /// On drop, calls `cuCtxDestroy_v2` instead of `cuDevicePrimaryCtxRelease`.
    ///
    /// # Safety
    ///
    /// - `cu_ctx` must be a valid CUDA context that was created (not yet destroyed).
    /// - `cu_device` must be the device the context was created for.
    /// - The caller must not destroy or release the context after calling this function;
    ///   ownership is transferred to the returned `Arc<CudaContext>`. If the allocator
    ///   decision fails, the error is returned before the wrapper exists and ownership stays
    ///   with the caller (FathomDB patch). If binding the context to the calling thread fails
    ///   afterwards, the wrapper already owns it and destroys it when dropped, as upstream does.
    ///
    /// On aarch64 Linux the allocator is the one already chosen for `cu_device` in this
    /// process, if any (FathomDB patch; see [CudaContext::has_async_alloc]).
    pub unsafe fn from_raw_context(
        ordinal: usize,
        cu_device: sys::CUdevice,
        cu_ctx: sys::CUcontext,
    ) -> Result<Arc<Self>, DriverError> {
        // FATHOMDB PATCH: pool support alone does not guarantee a usable default pool.
        let has_async_alloc = select_async_alloc(cu_device, cu_ctx, || {})?;
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc,
            mem_pool: None,
            is_primary: false,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Returns whether this context wraps a primary context.
    ///
    /// Primary contexts are created via `cuDevicePrimaryCtxRetain` and released on drop.
    /// Non-primary contexts (e.g., CiG) are destroyed via `cuCtxDestroy_v2` on drop.
    pub fn is_primary(&self) -> bool {
        self.is_primary
    }

    /// Returns whether this context supports asynchronous memory allocation.
    ///
    /// It is `true` if the `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` attribute is greater
    /// than 0. On aarch64 Linux (FathomDB patch; see `FATHOMDB.md`, items 1-2) it additionally
    /// requires that the device's default memory pool can be obtained, and it is decided once
    /// per device per process, by the first context created for or wrapping that device;
    /// every later context of the device reuses it.
    /// Memory allocations performed through the default [CudaStream] will use `cuMemAllocAsync`
    /// over `cuMemAlloc` if this method returns `true`.
    pub fn has_async_alloc(&self) -> bool {
        self.has_async_alloc
    }

    /// Which allocator this context uses (FathomDB patch): [AllocMode::Private] for a context
    /// built with [CudaContext::new_with_mem_pool], otherwise [AllocMode::Default] or
    /// [AllocMode::Synchronous] as [CudaContext::has_async_alloc] says.
    pub fn alloc_mode(&self) -> AllocMode {
        alloc_mode_of(self.mem_pool.is_some(), self.has_async_alloc)
    }

    /// The pool this context allocates from, if it was built with
    /// [CudaContext::new_with_mem_pool] (FathomDB patch).
    pub fn mem_pool(&self) -> Option<&Arc<super::mem_pool::CudaMemPool>> {
        self.mem_pool.as_ref()
    }

    /// The number of devices available.
    pub fn device_count() -> Result<i32, DriverError> {
        result::init()?;
        result::device::get_count()
    }

    /// Get the `ordinal` index of the device this is on.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the name of this device.
    pub fn name(&self) -> Result<String, result::DriverError> {
        self.check_err()?;
        result::device::get_name(self.cu_device)
    }

    /// Get the UUID of this device.
    pub fn uuid(&self) -> Result<sys::CUuuid, result::DriverError> {
        self.check_err()?;
        result::device::get_uuid(self.cu_device)
    }

    /// Get the compute capability of this device as a (major,minor) tuple
    pub fn compute_capability(&self) -> Result<(i32, i32), result::DriverError> {
        self.check_err()?;
        let capability_major =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let capability_minor =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

        Ok((capability_major, capability_minor))
    }

    /// Get the total memory available on this device, in bytes.
    pub fn total_mem(&self) -> Result<usize, DriverError> {
        self.check_err()?;
        unsafe { result::device::total_mem(self.cu_device) }
    }

    /// Returns the free and total device memory in bytes as a `(free, total)` tuple.
    /// Note: this calls [CudaContext::bind_to_thread()] to ensure the query
    /// runs against this device's context.
    pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
        self.bind_to_thread()?;
        result::mem_get_info()
    }
    /// Get the underlying [sys::CUdevice] of this [CudaContext].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the device pointer**, as it is still
    /// owned by the [CudaContext].
    pub fn cu_device(&self) -> sys::CUdevice {
        self.cu_device
    }

    /// Get the underlying [sys::CUcontext] of this [CudaContext].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/release the context pointer**, as it is still
    /// owned by the [CudaContext].
    pub fn cu_ctx(&self) -> sys::CUcontext {
        self.cu_ctx
    }

    /// Binds this context to the calling thread. Calling this is key for thread safety.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        self.check_err()?;
        if match result::ctx::get_current()? {
            Some(curr_ctx) => curr_ctx != self.cu_ctx,
            None => true,
        } {
            unsafe { result::ctx::set_current(self.cu_ctx) }?;
        }
        Ok(())
    }

    /// Get the value of the specified attribute of the device in [CudaContext].
    pub fn attribute(&self, attrib: sys::CUdevice_attribute) -> Result<i32, result::DriverError> {
        self.check_err()?;
        unsafe { result::device::get_attribute(self.cu_device, attrib) }
    }

    /// Synchronize this context. Will only block CPU if you call [CudaContext::set_flags()] with
    /// [sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC].
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        result::ctx::synchronize()
    }

    /// Ensures calls to [CudaContext::synchronize()] block the calling thread.
    ///
    /// Sets [sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC]
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000"
    )))]
    pub fn set_blocking_synchronize(&self) -> Result<(), DriverError> {
        self.set_flags(sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)
    }

    /// Set flags for this context
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000"
    )))]
    pub fn set_flags(&self, flags: sys::CUctx_flags) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        result::ctx::set_flags(flags)
    }

    /// Gets the value of a context limit.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8)
    pub fn get_limit(&self, limit: sys::CUlimit) -> Result<usize, DriverError> {
        self.bind_to_thread()?;
        result::ctx::get_limit(limit)
    }

    /// Sets the value of a context limit.
    ///
    /// Common limits:
    /// - `CU_LIMIT_STACK_SIZE` - Stack size for each thread
    /// - `CU_LIMIT_PRINTF_FIFO_SIZE` - Size of printf buffer
    /// - `CU_LIMIT_MALLOC_HEAP_SIZE` - Heap size for malloc() in kernels
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a)
    pub fn set_limit(&self, limit: sys::CUlimit, value: usize) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        result::ctx::set_limit(limit, value)
    }

    /// Gets the L1/shared memory cache configuration preference.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360)
    pub fn get_cache_config(&self) -> Result<sys::CUfunc_cache, DriverError> {
        self.bind_to_thread()?;
        result::ctx::get_cache_config()
    }

    /// Sets the L1/shared memory cache configuration preference.
    ///
    /// Options:
    /// - `CU_FUNC_CACHE_PREFER_NONE` - No preference
    /// - `CU_FUNC_CACHE_PREFER_SHARED` - Prefer larger shared memory, smaller L1
    /// - `CU_FUNC_CACHE_PREFER_L1` - Prefer larger L1, smaller shared memory
    /// - `CU_FUNC_CACHE_PREFER_EQUAL` - Equal split between L1 and shared
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3)
    pub fn set_cache_config(&self, config: sys::CUfunc_cache) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        result::ctx::set_cache_config(config)
    }

    /// Whether multiple streams have been created in this context. If so,
    /// the [CudaSlice::read] and [CudaSlice::write] events will be activated.
    ///
    /// This only get's set to true by [CudaContext::new_stream()].
    pub fn is_in_multi_stream_mode(&self) -> bool {
        self.num_streams.load(Ordering::Relaxed) > 0
    }

    /// Whether event tracking is being managed by this context
    /// (via [CudaContext::enable_event_tracking()], which is the default behavior),
    /// or `false` if the user is manually managing stream synchronization
    /// (via [CudaContext::disable_event_tracking()]).
    pub fn is_event_tracking(&self) -> bool {
        self.event_tracking.load(Ordering::Relaxed)
    }

    /// Whether the context is automatically managing multiple stream synchronization.
    /// Both of these must be true:
    /// - [CudaContext::is_in_multi_stream_mode()]
    /// - [CudaContext::is_event_tracking()]
    pub fn is_managing_stream_synchronization(&self) -> bool {
        self.is_in_multi_stream_mode() && self.is_event_tracking()
    }

    /// When turned on, all [CudaSlice] **created after calling this function** will
    /// record usages using [CudaEvent] to ensure proper synchronization between streams.
    ///
    /// # Safety
    ///
    /// If [CudaContext::disable_event_tracking()] was called previously, then any
    /// [CudaSlice] created after that and before this current call won't have [CudaEvent]
    /// tracking their uses. Those [CudaSlice] will not manage their synchronization, even
    /// after this call.
    pub unsafe fn enable_event_tracking(&self) {
        self.event_tracking.store(true, Ordering::Relaxed);
    }

    /// When turned on, all [CudaSlice] **created after calling this function** will
    /// not track uses via [CudaEvent]s.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no [CudaSlice] is freed before a use on another stream is finished.
    /// - Ensure that a [CudaSlice] is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a [CudaSlice] is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.event_tracking.store(false, Ordering::Relaxed);
    }

    /// Checks to see if there have been any calls that stored an Err in a function
    /// that couldn't return a result (e.g. Drop calls).
    ///
    /// If there are any errors stored, this method will return the Err value, and
    /// then clear the stored error state.
    pub fn check_err(&self) -> Result<(), DriverError> {
        let error_state = self.error_state.swap(0, Ordering::Relaxed);
        if error_state == 0 {
            Ok(())
        } else {
            Err(result::DriverError(unsafe {
                std::mem::transmute::<u32, sys::cudaError_enum>(error_state)
            }))
        }
    }

    /// Records a result for later inspection when a Result can be returned.
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(err) = result {
            self.error_state.store(err.0 as u32, Ordering::Relaxed)
        }
    }
}

/// A lightweight synchronization primitive used to synchronize between [CudaStream]s.
///
/// - Create using [CudaContext::new_event()].
/// - Record a point of time in a stream using [CudaEvent::record()].
/// - Either call [CudaEvent::synchronize()] or [CudaStream::wait()] to use.
///
/// Note that calls to [CudaEvent::record()] will not change any **previous calls** to [CudaStream::wait()].
///
/// # Thread safety
/// This object is thread safe
#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) cu_event: sys::CUevent,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { result::event::destroy(self.cu_event) });
    }
}

impl CudaContext {
    /// Creates a new [CudaEvent] with no work recorded. If `flags` is None, the event is created with
    /// [sys::CUevent_flags::CU_EVENT_DISABLE_TIMING].
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let flags = flags.unwrap_or(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
        self.bind_to_thread()?;
        let cu_event = result::event::create(flags)?;
        Ok(CudaEvent {
            cu_event,
            ctx: self.clone(),
        })
    }
}

impl CudaEvent {
    /// The underlying cu_event object.
    ///
    /// # Safety
    /// Do not destroy this value
    pub fn cu_event(&self) -> sys::CUevent {
        self.cu_event
    }

    /// The context this was created in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Records the current amount of work in [CudaStream] into this event.
    ///
    /// **This does not affect any previous calls to [CudaStream::wait()]**
    ///
    /// If `stream` belongs to a different [CudaContext], this will fail with
    /// [sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1)
    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx != stream.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        unsafe { result::event::record(self.cu_event, stream.cu_stream) }
    }

    /// Will only block CPU thraed if [sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC] was used to create this event.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::event::synchronize(self.cu_event) }
    }

    /// The time between two events. `self` is the start event, and `end` is the end event.
    /// This is effectively `end - self`.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        if self.ctx != end.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        self.synchronize()?;
        end.synchronize()?;
        unsafe { result::event::elapsed(self.cu_event, end.cu_event) }
    }

    /// Returns `true` if all recorded work has been completed, `false` otherwise.
    ///
    /// This also returns `false` on CUDA errors. Use [Self::try_is_complete()] to handle errors.
    pub fn is_complete(&self) -> bool {
        unsafe { result::event::query(self.cu_event) }.is_ok()
    }

    /// Queries whether all recorded work has completed, without waiting.
    ///
    /// Returns `Ok(false)` only for [sys::CUresult::CUDA_ERROR_NOT_READY]. Other CUDA errors
    /// are returned to the caller, including errors from earlier asynchronous launches.
    /// An event that has not been recorded returns `Ok(true)`.
    pub fn try_is_complete(&self) -> Result<bool, DriverError> {
        match unsafe { result::event::query(self.cu_event) } {
            Ok(()) => Ok(true),
            Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
            Err(error) => Err(error),
        }
    }
}

/// A wrapper around [sys::CUstream] that you can schedule work on.
///
/// - Create with [CudaContext::new_stream()], [CudaContext::default_stream()], or [CudaStream::fork()].
/// - Wrap a stream owned by another library with [CudaContext::wrap_borrowed_stream()].
///
/// **Work done on this is asynchronous with respect to the host.**
///
/// See [CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
/// See [3. Stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
/// See [6.6. Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
/// See [Out-of-order execution](https://en.wikipedia.org/wiki/Out-of-order_execution)
/// See [Dependence analysis](https://en.wikipedia.org/wiki/Dependence_analysis)
#[derive(Debug)]
pub struct CudaStream {
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) owned: bool,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl PartialEq for CudaStream {
    fn eq(&self, other: &Self) -> bool {
        self.cu_stream == other.cu_stream && self.ctx == other.ctx
    }
}

impl Eq for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        self.ctx.record_err(self.ctx.bind_to_thread());
        let cu_stream = std::mem::replace(&mut self.cu_stream, std::ptr::null_mut());
        if !cu_stream.is_null() && cu_stream != (0x2 as _) {
            self.ctx.num_streams.fetch_sub(1, Ordering::Relaxed);
            self.ctx
                .record_err(unsafe { result::stream::destroy(cu_stream) });
        }
    }
}

impl CudaContext {
    /// Get's the default stream for this context (the null ptr stream). Note that context's
    /// on the same device can all submit to the same default stream from separate context objects.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            cu_stream: std::ptr::null_mut(),
            ctx: self.clone(),
            owned: false,
        })
    }

    /// Get's the per-thread stream handle. See https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior
    pub fn per_thread_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            // See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197
            cu_stream: 0x2 as _,
            ctx: self.clone(),
            owned: false,
        })
    }

    /// Wraps a CUDA stream owned by another library.
    ///
    /// Dropping the returned wrapper does not destroy the underlying stream.
    ///
    /// # Safety
    ///
    /// - `cu_stream` must be a valid CUDA stream belonging to this context.
    /// - The stream's owner must keep it alive while the returned wrapper, or
    ///   anything retaining it, may be used.
    /// - The caller is responsible for ordering work submitted through cudarc
    ///   with work submitted through the stream's owner.
    pub unsafe fn wrap_borrowed_stream(
        self: &Arc<Self>,
        cu_stream: sys::CUstream,
    ) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
            owned: false,
        })
    }

    /// Create a new [sys::CUstream_flags::CU_STREAM_NON_BLOCKING] stream.
    ///
    /// This will swap the calling context to multi stream mode [CudaContext::is_in_multi_stream_mode()].
    /// If the context is not already in multiple stream mode, then this function will also call [CudaContext::synchronize()].
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        self.bind_to_thread()?;
        let prev_num_streams = self.num_streams.fetch_add(1, Ordering::Relaxed);
        if prev_num_streams == 0 && self.is_event_tracking() {
            self.synchronize()?;
        }
        let cu_stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;
        Ok(Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
            owned: true,
        }))
    }

    /// Create a new [sys::CUstream_flags::CU_STREAM_NON_BLOCKING] stream with the
    /// specified priority.
    ///
    /// Lower numerical values indicate higher priority. Use
    /// [`result::stream::get_priority_range`] to query the valid range.
    ///
    /// This will swap the calling context to multi stream mode
    /// [`CudaContext::is_in_multi_stream_mode()`].
    pub fn new_stream_with_priority(
        self: &Arc<Self>,
        priority: i32,
    ) -> Result<Arc<CudaStream>, DriverError> {
        self.bind_to_thread()?;
        let prev_num_streams = self.num_streams.fetch_add(1, Ordering::Relaxed);
        if prev_num_streams == 0 && self.is_event_tracking() {
            self.synchronize()?;
        }
        let cu_stream = result::stream::create_with_priority(
            result::stream::StreamKind::NonBlocking,
            priority,
        )?;
        Ok(Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
            owned: true,
        }))
    }
}

impl CudaStream {
    /// Create's a new stream and then makes the new stream wait on `self`
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        self.ctx.bind_to_thread()?;
        self.ctx.num_streams.fetch_add(1, Ordering::Relaxed);
        let cu_stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;
        let stream = Arc::new(CudaStream {
            cu_stream,
            ctx: self.ctx.clone(),
            owned: true,
        });
        stream.join(self)?;
        Ok(stream)
    }

    /// The underlying cuda stream object
    /// # Safety
    /// Do not destroy this value.
    pub fn cu_stream(&self) -> sys::CUstream {
        self.cu_stream
    }

    /// The context the stream belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Will only block CPU if you call [CudaContext::set_flags()] with
    /// [sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad)
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::synchronize(self.cu_stream) }
    }

    /// Creates a new [CudaEvent] and records the current work in the stream to the event.
    pub fn record_event(
        &self,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let event = self.ctx.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }

    /// Waits for the work recorded in [CudaEvent] to be completed.
    ///
    /// You can record new work in `event` after calling this method without
    /// affecting this call.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f)
    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::stream::wait_event(
                self.cu_stream,
                event.cu_event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
    }

    /// Ensures this stream waits for the current workload in `other` to complete.
    /// This is shorthand for `self.wait(other.record_event())`
    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }
}

/// `Vec<T>` on a cuda device. You can allocate and modify this with [CudaStream].
///
/// This object is thread safe.
#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: Option<CudaEvent>,
    pub(crate) write: Option<CudaEvent>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) marker: PhantomData<*const T>,
}

unsafe impl<T> Send for CudaSlice<T> {}
unsafe impl<T> Sync for CudaSlice<T> {}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        let ctx = &self.stream.ctx;
        if ctx.is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                ctx.record_err(self.stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                ctx.record_err(self.stream.wait(write));
            }
        }
        if ctx.has_async_alloc {
            ctx.record_err(unsafe {
                result::free_async(self.cu_device_ptr, self.stream.cu_stream)
            });
        } else {
            ctx.record_err(self.stream.synchronize());
            // FATHOMDB PATCH: zero-length buffers hold a null pointer (see `malloc_sync_or_null`).
            if sync_free_needed(SYNC_FALLBACK, self.cu_device_ptr) {
                ctx.record_err(unsafe { result::free_sync(self.cu_device_ptr) });
            }
        }
    }
}

impl<T> CudaSlice<T> {
    /// The number of elements of `T` in this object.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of bytes in this object.
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// True if there are no elements in the object.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device ordinal this belongs to
    pub fn ordinal(&self) -> usize {
        self.stream.ctx.ordinal
    }

    /// The context this belongs to
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.ctx
    }

    /// The stream this object was allocated on and later will be dropped on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn try_clone(&self) -> Result<Self, result::DriverError> {
        self.stream.clone_dtod(self)
    }
}

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: Clone + Default + DeviceRepr> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = result::DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.stream.clone_dtoh(&value)
    }
}

/// `&[T]` on a cuda device. An immutable sub-view into a [CudaSlice] created by [CudaSlice::as_view()]/[CudaSlice::slice()].
#[derive(Debug)]
pub struct CudaView<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: &'a Option<CudaEvent>,
    pub(crate) write: &'a Option<CudaEvent>,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a [T]>,
}

impl<T> CudaSlice<T> {
    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            ptr: self.cu_device_ptr,
            len: self.len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        }
    }
}

impl<T> CudaView<'_, T> {
    /// The number of elements `T` in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn resize(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.len);
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }
}

/// `&mut [T]` on a cuda device. A mutable sub-view into a [CudaSlice] created by [CudaSlice::as_view_mut()]/[CudaSlice::slice_mut()].
#[derive(Debug)]
pub struct CudaViewMut<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: &'a Option<CudaEvent>,
    pub(crate) write: &'a Option<CudaEvent>,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a mut [T]>,
}

impl<T> CudaSlice<T> {
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            ptr: self.cu_device_ptr,
            len: self.len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        }
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// Number of elements `T` that are in this view.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Downgrade this to a `&[T]`
    pub fn as_view<'b>(&'b self) -> CudaView<'b, T> {
        CudaView {
            ptr: self.ptr,
            len: self.len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// Downgrate this into to a `&[T]`, consuming the current view and keeping the original lifetime 'a
    pub fn into_view(self) -> CudaView<'a, T> {
        CudaView {
            ptr: self.ptr,
            len: self.len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }
}

/// Marker trait to indicate that the type is valid
/// when all of its bits are set to 0.
///
/// # Safety
/// Not all types are valid when all bits are set to 0.
/// Be very sure when implementing this trait!
pub unsafe trait ValidAsZeroBits {}
unsafe impl ValidAsZeroBits for bool {}
unsafe impl ValidAsZeroBits for i8 {}
unsafe impl ValidAsZeroBits for i16 {}
unsafe impl ValidAsZeroBits for i32 {}
unsafe impl ValidAsZeroBits for i64 {}
unsafe impl ValidAsZeroBits for i128 {}
unsafe impl ValidAsZeroBits for isize {}
unsafe impl ValidAsZeroBits for u8 {}
unsafe impl ValidAsZeroBits for u16 {}
unsafe impl ValidAsZeroBits for u32 {}
unsafe impl ValidAsZeroBits for u64 {}
unsafe impl ValidAsZeroBits for u128 {}
unsafe impl ValidAsZeroBits for usize {}
unsafe impl ValidAsZeroBits for f32 {}
unsafe impl ValidAsZeroBits for f64 {}
#[cfg(feature = "f16")]
unsafe impl ValidAsZeroBits for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl ValidAsZeroBits for half::bf16 {}
unsafe impl<T: ValidAsZeroBits, const M: usize> ValidAsZeroBits for [T; M] {}
/// Implement `ValidAsZeroBits` for tuples if all elements are `ValidAsZeroBits`,
///
/// # Note
/// This will also implement `ValidAsZeroBits` for a tuple with one element
macro_rules! impl_tuples {
    ($t:tt) => {
        impl_tuples!(@ $t);
    };
    // the $l is in front of the reptition to prevent parsing ambiguities
    ($l:tt $(,$t:tt)+) => {
        impl_tuples!($($t),+);
        impl_tuples!(@ $l $(,$t)+);
    };
    (@ $($t:tt),+) => {
        unsafe impl<$($t: ValidAsZeroBits,)+> ValidAsZeroBits for ($($t,)+) {}
    };
}
impl_tuples!(A, B, C, D, E, F, G, H, I, J, K, L);

/// Something that can be copied to device memory and
/// turned into a parameter for [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because a struct should likely
/// be `#[repr(C)]` to be represented in cuda memory,
/// and not all types are valid.
pub unsafe trait DeviceRepr {}
unsafe impl DeviceRepr for bool {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for i128 {}
unsafe impl DeviceRepr for isize {}
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for u128 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}
#[cfg(feature = "f16")]
unsafe impl DeviceRepr for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl DeviceRepr for half::bf16 {}

#[cfg(feature = "f8")]
unsafe impl DeviceRepr for float8::F8E4M3 {}
#[cfg(feature = "f8")]
unsafe impl ValidAsZeroBits for float8::F8E4M3 {}

#[cfg(feature = "f8")]
unsafe impl DeviceRepr for float8::F8E5M2 {}
#[cfg(feature = "f8")]
unsafe impl ValidAsZeroBits for float8::F8E5M2 {}

#[cfg(feature = "f4")]
unsafe impl DeviceRepr for float4::F4E2M1 {}
#[cfg(feature = "f4")]
unsafe impl ValidAsZeroBits for float4::F4E2M1 {}

#[cfg(feature = "f4")]
unsafe impl DeviceRepr for float4::E8M0 {}
#[cfg(feature = "f4")]
unsafe impl ValidAsZeroBits for float4::E8M0 {}

#[cfg(feature = "f4")]
unsafe impl DeviceRepr for float4::F4E2M1x2 {}
#[cfg(feature = "f4")]
unsafe impl ValidAsZeroBits for float4::F4E2M1x2 {}

unsafe impl<const N: usize, T> DeviceRepr for [T; N] where T: DeviceRepr {}

/// Base trait for abstracting over [CudaSlice]/[CudaView]/[CudaViewMut].
///
/// Don't use this directly - use [DevicePtr]/[DevicePtrMut].
pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stream(&self) -> &Arc<CudaStream>;
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

/// A synchronization primitive to enable stream & event synchronization.
/// Primarily used with [DevicePtr] and [DevicePtrMut]
#[derive(Debug)]
#[must_use]
pub enum SyncOnDrop<'a> {
    /// Will record the stream's workload to the event on drop.
    Record(Option<(&'a CudaEvent, &'a CudaStream)>),
    /// Will call stream synchronize on drop.
    Sync(Option<&'a CudaStream>),
}

impl<'a> SyncOnDrop<'a> {
    /// Construct a [SyncOnDrop::Record] variant
    pub fn record_event(event: &'a Option<CudaEvent>, stream: &'a CudaStream) -> Self {
        SyncOnDrop::Record(event.as_ref().map(|e| (e, stream)))
    }
    /// Construct a [SyncOnDrop::Sync] variant
    pub fn sync_stream(stream: &'a CudaStream) -> Self {
        SyncOnDrop::Sync(Some(stream))
    }
}

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        match self {
            SyncOnDrop::Record(target) => {
                if let Some((event, stream)) = std::mem::take(target) {
                    if stream.ctx.is_managing_stream_synchronization() {
                        stream.ctx.record_err(event.record(stream));
                    }
                }
            }
            SyncOnDrop::Sync(target) => {
                if let Some(stream) = std::mem::take(target) {
                    stream.ctx.record_err(stream.synchronize());
                }
            }
        }
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T>: DeviceSlice<T> {
    /// Retrieve the device pointer with the intent to read the device memory
    /// associated with it.
    ///
    /// Implementations of this method should ensure `stream` waits for any previous
    /// writes of this memory before continuing (do not need to wait for any previous reads).
    ///
    /// The [SyncOnDrop] item of the return tuple should be dropped **after** the read of
    /// the [sys::CUdeviceptr] is scheduled.
    ///
    /// In most cases you can use like:
    /// ```ignore
    /// let (src, _record_src) = src.device_ptr(&stream);
    /// ```
    /// Which will drop the [SyncOnDrop] at the end of the scope.
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (
            self.cu_device_ptr,
            SyncOnDrop::record_event(&self.read, stream),
        )
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

impl<'a, T> CudaView<'a, T> {
    /// Identical behavior to [DevicePtr::device_ptr()], but the lifetime on the returned
    /// [SyncOnDrop], matches the lifetime of the view.
    pub fn view_ptr(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// Identical behavior to [DevicePtr::device_ptr()], but the lifetime on the returned
    /// [SyncOnDrop], matches the lifetime of the view.
    pub fn view_ptr(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    /// Retrieve the device pointer with the intent to modify the device memory
    /// associated with it.
    ///
    /// Implementations of this method should ensure `stream` waits for any previous
    /// reads/writes of this memory before continuing.
    ///
    /// The [SyncOnDrop] item of the return tuple should be dropped **after** the write of
    /// the [sys::CUdeviceptr] is scheduled.
    ///
    /// In most cases you can use like:
    /// ```ignore
    /// let (src, _record_src) = src.device_ptr_mut(&stream);
    /// ```
    /// Which will drop the [SyncOnDrop] at the end of the scope.
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                stream.ctx.record_err(stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (
            self.cu_device_ptr,
            SyncOnDrop::record_event(&self.write, stream),
        )
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                stream.ctx.record_err(stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.write, stream))
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// Identical behavior to [DevicePtrMut::device_ptr_mut()], but the lifetime on the returned
    /// [SyncOnDrop], matches the lifetime of the view.
    pub fn view_ptr_mut(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                stream.ctx.record_err(stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                stream.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.write, stream))
    }
}

/// Abstraction over `&[T]`, `&Vec<T>` and [`PinnedHostSlice<T>`].
pub trait HostSlice<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`. Otherwise
    /// You may run into device synchronization errors
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>);

    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`. Otherwise
    /// You may run into device synchronization errors
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>);
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
}

/// Rust side data that the `cuda` driver knows is pinned. This is different
/// than `Pin<Vec<T>>` mainly because cuda driver manages this memory and ensures
/// it is page locked.
///
/// Allocate this with [CudaContext::alloc_pinned()], and do device copies with
/// [CudaStream::clone_htod()]/[CudaStream::memcpy_htod()]/[CudaStream::memcpy_dtoh()]
#[derive(Debug)]
pub struct PinnedHostSlice<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: CudaEvent,
}

unsafe impl<T> Send for PinnedHostSlice<T> {}
unsafe impl<T> Sync for PinnedHostSlice<T> {}

impl<T> Drop for PinnedHostSlice<T> {
    fn drop(&mut self) {
        let ctx = &self.event.ctx;
        ctx.record_err(self.event.synchronize());
        ctx.record_err(unsafe { result::free_host(self.ptr as _) });
    }
}

impl CudaContext {
    /// Allocates page locked host memory with [sys::CU_MEMHOSTALLOC_WRITECOMBINED] flags.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)
    ///
    /// # Safety
    /// 1. This is unsafe because the memory is unset after this call.
    pub unsafe fn alloc_pinned<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<PinnedHostSlice<T>, DriverError> {
        self.alloc_pinned_with_flags(len, sys::CU_MEMHOSTALLOC_WRITECOMBINED)
    }

    /// Allocates page-locked host memory with the specified `flags`.
    ///
    /// `flags` may contain any combination of the following options:
    ///
    /// - [sys::CU_MEMHOSTALLOC_PORTABLE]
    /// - [sys::CU_MEMHOSTALLOC_DEVICEMAP]
    /// - [sys::CU_MEMHOSTALLOC_WRITECOMBINED]
    ///
    /// These flags are orthogonal and may be combined without restriction.
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)
    ///
    /// # Safety
    /// 1. This is unsafe because the memory is unset after this call.
    pub unsafe fn alloc_pinned_with_flags<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
        flags: u32,
    ) -> Result<PinnedHostSlice<T>, DriverError> {
        self.bind_to_thread()?;
        let num_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("Pinned host allocation size overflow");
        assert!(num_bytes < isize::MAX as usize);
        let event = self.new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;
        let ptr = result::malloc_host(num_bytes, flags)?;
        let ptr = ptr as *mut T;
        let allocation = PinnedHostSlice { ptr, len, event };
        assert!(!ptr.is_null());
        assert!(ptr.is_aligned());
        Ok(allocation)
    }
}

impl<T> PinnedHostSlice<T> {
    /// The context this was created in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.event.ctx
    }

    /// The number of elements `T` in this slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of bytes in this slice.
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ValidAsZeroBits> PinnedHostSlice<T> {
    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_ptr(&self) -> Result<*const T, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_ptr(&mut self) -> Result<*mut T, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_slice(&self) -> Result<&[T], DriverError> {
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], DriverError> {
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}

impl<T> HostSlice<T> for PinnedHostSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        stream.ctx.record_err(stream.wait(&self.event));
        (
            std::slice::from_raw_parts(self.ptr, self.len),
            SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        stream.ctx.record_err(stream.wait(&self.event));
        (
            std::slice::from_raw_parts_mut(self.ptr, self.len),
            SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
}

impl CudaStream {
    /// Allocates an empty [CudaSlice] with 0 length.
    pub fn null<T>(self: &Arc<Self>) -> Result<CudaSlice<T>, result::DriverError> {
        self.ctx.bind_to_thread()?;
        // FATHOMDB PATCH: a private pool also receives zero-byte requests (null, freeable).
        let cu_device_ptr = if let Some(pool) = &self.ctx.mem_pool {
            unsafe { result::mem_pool::alloc_async(pool.raw(), 0, self.cu_stream) }?
        } else if self.ctx.has_async_alloc {
            unsafe { result::malloc_async(self.cu_stream, 0) }?
        } else {
            unsafe { malloc_sync_or_null(0) }? // FATHOMDB PATCH
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len: 0,
            read: None,
            write: None,
            stream: self.clone(),
            marker: PhantomData,
        })
    }

    /// Allocates a [CudaSlice] with `len` elements of type `T`.
    /// # Safety
    /// This is unsafe because the memory is unset.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        self.ctx.bind_to_thread()?;
        // FATHOMDB PATCH: a private-pool context allocates only from its own pool.
        let cu_device_ptr = if let Some(pool) = &self.ctx.mem_pool {
            result::mem_pool::alloc_async(
                pool.raw(),
                len * std::mem::size_of::<T>(),
                self.cu_stream,
            )?
        } else if self.ctx.has_async_alloc {
            result::malloc_async(self.cu_stream, len * std::mem::size_of::<T>())?
        } else {
            malloc_sync_or_null(len * std::mem::size_of::<T>())? // FATHOMDB PATCH
        };
        let (read, write) = if self.ctx.is_event_tracking() {
            (
                Some(self.ctx.new_event(None)?),
                Some(self.ctx.new_event(None)?),
            )
        } else {
            (None, None)
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            read,
            write,
            stream: self.clone(),
            marker: PhantomData,
        })
    }

    /// Allocates a [CudaSlice] with `len` elements of type `T`. All values are zero'd out.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(len) }?;
        self.memset_zeros(&mut dst)?;
        Ok(dst)
    }

    /// Set's all the memory in `dst` to 0. `dst` can be a [CudaSlice] or [CudaViewMut]
    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        let num_bytes = dst.num_bytes();
        let (dptr, _record) = dst.device_ptr_mut(self);
        unsafe { result::memset_d8_async(dptr, 0, num_bytes, self.cu_stream) }?;
        Ok(())
    }

    /// Copy a `[T]`/`Vec<T>`/[`PinnedHostSlice<T>`] to a new [`CudaSlice`].
    #[deprecated = "Use clone_htod"]
    pub fn memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// Copy a `[T]`/`Vec<T>`/[`PinnedHostSlice<T>`] to a new [`CudaSlice`].
    pub fn clone_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// Copy a `[T]`/`Vec<T>`/[`PinnedHostSlice<T>`] into an existing [`CudaSlice`]/[`CudaViewMut`].
    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        self.ctx.bind_to_thread()?;
        let (src, _record_src) = unsafe { src.stream_synced_slice(self) };
        let (dst, _record_dst) = dst.device_ptr_mut(self);
        unsafe { result::memcpy_htod_async(dst, src, self.cu_stream) }
    }

    /// Copy a [`CudaSlice`]/[`CudaView`] to a new [`Vec<T>`].
    #[deprecated = "Use clone_dtoh"]
    pub fn memcpy_dtov<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len())
        };
        self.memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    /// Copy a [`CudaSlice`]/[`CudaView`] to a new [`Vec<T>`].
    pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len())
        };
        self.memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    /// Copy a [`CudaSlice`]/[`CudaView`] to a existing `[T]`/[`Vec<T>`]/[`PinnedHostSlice<T>`].
    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        self.ctx.bind_to_thread()?;
        let (src, _record_src) = src.device_ptr(self);
        let (dst, _record_dst) = unsafe { dst.stream_synced_mut_slice(self) };
        unsafe { result::memcpy_dtoh_async(dst, src, self.cu_stream) }
    }

    /// Copy a [`CudaSlice`]/[`CudaView`] to a existing [`CudaSlice`]/[`CudaViewMut`].
    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        self.ctx.bind_to_thread()?;

        let num_bytes = src.num_bytes();

        let src_ctx = src.stream().context();
        let dst_ctx = self.context();

        if src_ctx == dst_ctx {
            let (src_ptr, _record_src) = src.device_ptr(self);
            let (dst_ptr, _record_dst) = dst.device_ptr_mut(self);
            unsafe { result::memcpy_dtod_async(dst_ptr, src_ptr, num_bytes, self.cu_stream) }
        } else {
            // NOTE: Although we want the current stream to wait on src to be ready,
            // we can't use src.device_ptr(self). When `_record_src` is dropped,
            // we record an event from the src_stream onto dst_stream (i.e., self). This is not
            // allowed in CUDA, and will return a CUDA_ERROR_INVALID_HANDLE
            // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd
            let (src_ptr, _record_src) = src.device_ptr(src.stream());
            let (dst_ptr, _record_dst) = dst.device_ptr_mut(self);
            // NOTE: Although we can't record events on streams they weren't created on,
            // we can *wait* on events from any stream. We can leverage this and wait on
            // a src event.
            // OPTIM: Ideally we could wait on the src write_events, but we can artificially
            // insert an event which guarantees src is available.
            self.wait(&src.stream().record_event(None)?)?;
            unsafe {
                result::memcpy_peer_async(
                    dst_ctx.cu_ctx,
                    dst_ptr,
                    src_ctx.cu_ctx,
                    src_ptr,
                    num_bytes,
                    self.cu_stream,
                )
            }
        }
    }

    /// Copy a [`CudaSlice`]/[`CudaView`] to a new [`CudaSlice`].
    pub fn clone_dtod<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }
}

impl<T> CudaSlice<T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice(0..50);
    /// do_something(&view);
    /// ```
    ///
    /// Like a normal slice, borrow checking prevents the underlying [CudaSlice] from being dropped.
    /// ```rust,compile_fail
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let view = {
    ///     let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    ///     // cannot return view, since it borrows from slice
    ///     slice.slice(0..50)
    /// };
    /// do_something(&view);
    /// ```
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// Fallible version of [CudaSlice::slice()].
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        self.as_view().try_slice(bounds)
    }

    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// do_something(&mut view);
    /// ```
    ///
    /// Like a normal mutable slice, borrow checking prevents the underlying [CudaSlice] from being dropped.
    /// ```rust,compile_fail
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut view = {
    ///     let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    ///     // cannot return view, since it borrows from slice
    ///     slice.slice_mut(0..50)
    /// };
    /// do_something(&mut view);
    /// ```
    ///
    /// Like with normal mutable slices, one cannot mutably slice twice into the same [CudaSlice]:
    /// ```rust,compile_fail
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view1 = slice.slice_mut(0..50);
    /// // cannot borrow twice from slice
    /// let mut view2 = slice.slice_mut(50..100);
    /// do_something(view1, view2);
    /// ```
    /// If you need non-overlapping mutable views into a [CudaSlice], you can use [CudaSlice::split_at_mut()].
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [CudaSlice::slice_mut]
    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        to_range(bounds, self.len).map(|(start, end)| CudaViewMut {
            ptr: self.cu_device_ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        self.as_view().transmute(len)
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaViewMut {
                ptr: self.cu_device_ptr,
                len,
                read: &self.read,
                write: &self.write,
                stream: &self.stream,
                marker: PhantomData,
            },
        )
    }

    pub fn split_at(&self, mid: usize) -> (CudaView<'_, T>, CudaView<'_, T>) {
        self.as_view().split_at(mid)
    }

    /// Fallible version of [CudaSlice::split_at]. Returns `None` if `mid > self.len`.
    pub fn try_split_at(&self, mid: usize) -> Option<(CudaView<'_, T>, CudaView<'_, T>)> {
        self.as_view().try_split_at(mid)
    }

    /// Splits the [CudaSlice] into two at the given index, returning two [CudaViewMut] for the two halves.
    ///
    /// Panics if `mid > self.len`.
    ///
    /// This method can be used to create non-overlapping mutable views into a [CudaSlice].
    /// ```rust
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// // split the slice into two non-overlapping, mutable views
    /// let (mut view1, mut view2) = slice.split_at_mut(50);
    /// do_something(view1, view2);
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'_, T>, CudaViewMut<'_, T>) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [CudaSlice::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'_, T>, CudaViewMut<'_, T>)> {
        let length = self.len;
        (mid <= length).then(|| {
            let a = CudaViewMut {
                ptr: self.cu_device_ptr,
                len: mid,
                read: &self.read,
                write: &self.write,
                stream: &self.stream,
                marker: PhantomData,
            };
            let b = CudaViewMut {
                ptr: self.cu_device_ptr + (mid * std::mem::size_of::<T>()) as u64,
                len: length - mid,
                read: &self.read,
                write: &self.write,
                stream: &self.stream,
                marker: PhantomData,
            };
            (a, b)
        })
    }
}

impl<'a, T> CudaView<'a, T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaView};
    /// # fn do_something(view: &CudaView<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice(0..50);
    /// let mut view2 = view.slice(0..25);
    /// do_something(&view);
    /// ```
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> Self {
        self.try_slice(bounds).unwrap()
    }

    /// Fallible version of [CudaView::slice]
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<Self> {
        to_range(bounds, self.len).map(|(start, end)| self.resize(start, end))
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.ptr,
                len,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        self.try_split_at(mid).unwrap()
    }

    /// Fallible version of [CudaSlice::split_at].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at(&self, mid: usize) -> Option<(Self, Self)> {
        (mid <= self.len()).then(|| (self.resize(0, mid), self.resize(mid, self.len)))
    }

    /// Returns an iterarow over subviews of size `chunk_size`. Differs from [std::slice::ChunksExact],
    /// in that it asserts that the chunk_size must divide evenly into the length, instead of returning
    /// a remainder.
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = CudaView<'a, T>> + '_ {
        assert!(self.len.is_multiple_of(chunk_size));
        let num_chunks = self.len / chunk_size;
        (0..num_chunks).map(move |i| self.resize(i * chunk_size, (i + 1) * chunk_size))
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: &mut CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// let mut view2 = view.slice_mut(0..25);
    /// do_something(&mut view2);
    /// ```
    ///
    /// One cannot slice twice into the same [CudaViewMut]:
    /// ```rust,compile_fail
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// // cannot borrow twice from same view
    /// let mut view1 = slice.slice_mut(0..25);
    /// let mut view2 = slice.slice_mut(25..50);
    /// do_something(view1, view2);
    /// ```
    /// If you need non-overlapping mutable views into a [CudaViewMut], you can use [CudaViewMut::split_at_mut()].
    pub fn slice<'b>(&'b self, bounds: impl RangeBounds<usize>) -> CudaView<'b, T> {
        self.try_slice(bounds).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice]
    pub fn try_slice<'b>(&'b self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'b, T>> {
        to_range(bounds, self.len).map(move |(start, end)| self.as_view().resize(start, end))
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<'b, S>(&'b self, len: usize) -> Option<CudaView<'b, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.ptr,
                len,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    /// Creates a [CudaViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    pub fn slice_mut<'b>(&'b mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'b, T> {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [CudaViewMut::slice_mut]
    pub fn try_slice_mut<'b>(
        &'b mut self,
        bounds: impl RangeBounds<usize>,
    ) -> Option<CudaViewMut<'b, T>> {
        to_range(bounds, self.len).map(|(start, end)| CudaViewMut {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Splits the [CudaViewMut] into two at the given index.
    ///
    /// Panics if `mid > self.len`.
    ///
    /// This method can be used to create non-overlapping mutable views into a [CudaViewMut].
    /// ```rust
    /// # use cudarc::driver::safe::{CudaContext, CudaSlice, CudaViewMut};
    /// # fn do_something(view: CudaViewMut<u8>, view2: CudaViewMut<u8>) {}
    /// # let ctx = CudaContext::new(0).unwrap();
    /// # let stream = ctx.default_stream();
    /// let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
    /// let mut view = slice.slice_mut(0..50);
    /// // split the view into two non-overlapping, mutable views
    /// let (mut view1, mut view2) = view.split_at_mut(25);
    /// do_something(view1, view2);
    pub fn split_at_mut<'b>(&'b mut self, mid: usize) -> (CudaViewMut<'b, T>, CudaViewMut<'b, T>) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [CudaViewMut::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`
    pub fn try_split_at_mut<'b>(
        &'b mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'b, T>, CudaViewMut<'b, T>)> {
        let length = self.len;
        (mid <= length).then(|| {
            let a = CudaViewMut {
                ptr: self.ptr,
                len: mid,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            };
            let b = CudaViewMut {
                ptr: self.ptr + (mid * std::mem::size_of::<T>()) as u64,
                len: length - mid,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            };
            (a, b)
        })
    }

    /// Returns an iterarow over subviews of size `chunk_size`. Differs from [std::slice::ChunksExactMut],
    /// in that it asserts that the chunk_size must divide evenly into the length, instead of returning
    /// a remainder.
    pub fn chunks_exact_mut(self, chunk_size: usize) -> impl Iterator<Item = CudaViewMut<'a, T>> {
        assert!(self.len.is_multiple_of(chunk_size));
        let num_chunks = self.len / chunk_size;
        (0..num_chunks).map(move |i| CudaViewMut {
            ptr: self.ptr + (i * chunk_size * std::mem::size_of::<T>()) as u64,
            len: chunk_size,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute_mut<'b, S>(&'b mut self, len: usize) -> Option<CudaViewMut<'b, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaViewMut {
                ptr: self.ptr,
                len,
                read: self.read,
                write: self.write,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }
}

pub(super) fn to_range(range: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n + 1,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    (start <= end && end <= len).then_some((start, end))
}

/// Wrapper around [sys::CUmodule]. Create with [CudaContext::load_module()].
///
/// Call [CudaModule::load_function] to load a [CudaFunction].
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { result::module::unload(self.cu_module) });
    }
}

impl CudaContext {
    /// Dynamically load a compiled ptx into this context.
    ///
    /// - `ptx` contains the compiled ptx
    #[cfg(feature = "nvrtc")]
    pub fn load_module(
        self: &Arc<Self>,
        ptx: crate::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>, result::DriverError> {
        self.bind_to_thread()?;

        let cu_module = match ptx.0 {
            crate::nvrtc::PtxKind::Image(image) => unsafe {
                result::module::load_data(image.as_ptr() as *const _)
            },
            crate::nvrtc::PtxKind::Src(src) => {
                let c_src = CString::new(src).unwrap();
                unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
            }
            crate::nvrtc::PtxKind::File(path) => {
                let name_c = CString::new(path.to_str().unwrap()).unwrap();
                result::module::load(name_c)
            }
            crate::nvrtc::PtxKind::Binary(data) => unsafe {
                result::module::load_data(data.as_ptr() as *const _)
            },
        }?;
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
}

/// Wrapper around [sys::CUfunction]. Used by [CudaStream::launch_builder] to execute kernels.
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    /// Loads a function from the loaded module with the given name.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        let fn_name_c = CString::new(fn_name).unwrap();
        let cu_function = unsafe { result::module::get_function(self.cu_module, fn_name_c) }?;
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }

    /// Gets a global/constant symbol from the loaded module as a [CudaSlice<u8>].
    ///
    /// This can be used to access `__constant__` memory declared in CUDA kernels.
    /// The returned slice can be transmuted to the appropriate type via views.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In CUDA: __constant__ float my_const[4];
    /// let symbol = module.get_global("my_const", &stream)?;
    /// let mut symbol_view = symbol.as_view_mut();
    /// let mut symbol_f32 = unsafe { symbol_view.transmute_mut::<f32>(4).unwrap() };
    /// stream.memcpy_htod(&[1.0f32, 2.0, 3.0, 4.0], &mut symbol_f32)?;
    /// ```
    pub fn get_global<'a>(
        self: &'a Arc<Self>,
        name: &str,
        stream: &'a Arc<CudaStream>,
    ) -> Result<CudaViewMut<'a, u8>, DriverError> {
        let name_c =
            CString::new(name).map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
        let (cu_device_ptr, bytes) = unsafe { result::module::get_global(self.cu_module, name_c) }?;
        Ok(CudaViewMut {
            ptr: cu_device_ptr,
            len: bytes,
            read: &None,
            write: &None,
            stream,
            marker: PhantomData,
        })
    }
}

impl CudaFunction {
    /// Returns the raw CUDA Driver function handle.
    ///
    /// The returned handle remains valid only while this function and its
    /// owning module remain alive.
    pub fn cu_function(&self) -> sys::CUfunction {
        self.cu_function
    }

    pub fn occupancy_available_dynamic_smem_per_block(
        &self,
        num_blocks: u32,
        block_size: u32,
    ) -> Result<usize, result::DriverError> {
        let mut dynamic_smem_size: usize = 0;

        unsafe {
            sys::cuOccupancyAvailableDynamicSMemPerBlock(
                &mut dynamic_smem_size,
                self.cu_function,
                num_blocks as std::ffi::c_int,
                block_size as std::ffi::c_int,
            )
            .result()?
        };

        Ok(dynamic_smem_size)
    }

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_smem_size: usize,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> Result<u32, result::DriverError> {
        let mut num_blocks: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);

        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &mut num_blocks,
                self.cu_function,
                block_size as std::ffi::c_int,
                dynamic_smem_size,
                flags as std::ffi::c_uint,
            )
            .result()?
        };

        Ok(num_blocks as u32)
    }

    #[cfg(not(any(
        feature = "cuda-11070",
        feature = "cuda-11060",
        feature = "cuda-11050",
        feature = "cuda-11040"
    )))]
    pub fn occupancy_max_active_clusters(
        &self,
        config: crate::driver::LaunchConfig,
        stream: &CudaStream,
    ) -> Result<u32, result::DriverError> {
        let mut num_clusters: std::ffi::c_int = 0;

        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: stream.cu_stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            sys::cuOccupancyMaxActiveClusters(&mut num_clusters, self.cu_function, &cfg).result()?
        };

        Ok(num_clusters as u32)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        block_size_to_dynamic_smem_size: extern "C" fn(block_size: std::ffi::c_int) -> usize,
        dynamic_smem_size: usize,
        block_size_limit: u32,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> Result<(u32, u32), result::DriverError> {
        let mut min_grid_size: std::ffi::c_int = 0;
        let mut block_size: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);

        unsafe {
            sys::cuOccupancyMaxPotentialBlockSizeWithFlags(
                &mut min_grid_size,
                &mut block_size,
                self.cu_function,
                Some(block_size_to_dynamic_smem_size),
                dynamic_smem_size,
                block_size_limit as std::ffi::c_int,
                flags as std::ffi::c_uint,
            )
            .result()?
        };

        Ok((min_grid_size as u32, block_size as u32))
    }

    #[cfg(not(any(
        feature = "cuda-11070",
        feature = "cuda-11060",
        feature = "cuda-11050",
        feature = "cuda-11040"
    )))]
    pub fn occupancy_max_potential_cluster_size(
        &self,
        config: crate::driver::LaunchConfig,
        stream: &CudaStream,
    ) -> Result<u32, result::DriverError> {
        let mut cluster_size: std::ffi::c_int = 0;

        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: stream.cu_stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            sys::cuOccupancyMaxPotentialClusterSize(&mut cluster_size, self.cu_function, &cfg)
                .result()?
        };

        Ok(cluster_size as u32)
    }

    /// Get the value of a specific attribute of this [CudaFunction].
    ///
    /// See [CUDA docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b)
    pub fn get_attribute(
        &self,
        attribute: CUfunction_attribute_enum,
    ) -> Result<i32, result::DriverError> {
        self.module.ctx.bind_to_thread()?;
        unsafe { result::function::get_function_attribute(self.cu_function, attribute) }
    }

    /// Get the number of registers used per thread.
    pub fn num_regs(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS)
    }

    /// Get the size of statically-allocated shared memory in bytes.
    pub fn shared_size_bytes(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    }

    /// Get the size of constant memory in bytes used by this function.
    pub fn const_size_bytes(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    }

    /// Get the size of local memory in bytes used per thread.
    pub fn local_size_bytes(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    }

    /// Get the maximum number of threads per block for this function.
    pub fn max_threads_per_block(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Get the PTX virtual architecture version for which the function was compiled.
    pub fn ptx_version(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION)
    }

    /// Get the binary architecture version for which the function was compiled.
    pub fn binary_version(&self) -> Result<i32, result::DriverError> {
        self.get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION)
    }

    /// Set the value of a specific attribute of this [CudaFunction].
    pub fn set_attribute(
        &self,
        attribute: CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), result::DriverError> {
        unsafe { result::function::set_function_attribute(self.cu_function, attribute, value) }
    }

    /// Set the cache config of this [CudaFunction].
    pub fn set_function_cache_config(
        &self,
        attribute: CUfunc_cache_enum,
    ) -> Result<(), result::DriverError> {
        unsafe { result::function::set_function_cache_config(self.cu_function, attribute) }
    }
}

impl<T> CudaSlice<T> {
    /// Takes ownership of the underlying [sys::CUdeviceptr]. **It is up
    /// to the owner to free this value**.
    ///
    /// Drops the underlying host_buf if there is one.
    pub fn leak(self) -> sys::CUdeviceptr {
        let mut s = std::mem::ManuallyDrop::new(self);
        let ptr = s.cu_device_ptr;

        // Ensure pending operations are complete before resources are released.
        if s.stream.ctx.is_managing_stream_synchronization() {
            if let Some(read) = s.read.as_ref() {
                s.stream.ctx.record_err(s.stream.wait(read));
            }
            if let Some(write) = s.write.as_ref() {
                s.stream.ctx.record_err(s.stream.wait(write));
            }
        }

        // Manually drop fields that own resources.
        unsafe {
            std::ptr::drop_in_place(&mut s.read);
            std::ptr::drop_in_place(&mut s.write);
            std::ptr::drop_in_place(&mut s.stream);
        }

        ptr
    }
}

impl CudaStream {
    /// Creates a [CudaSlice] from a [sys::CUdeviceptr]. Useful in conjunction with
    /// [`CudaSlice::leak()`].
    ///
    /// The slice owns the memory: dropping it frees the memory with
    /// `cuMemFreeAsync` on this stream if [CudaContext::has_async_alloc] is
    /// `true` for this stream's context, and otherwise with `cuMemFree` after
    /// synchronizing this stream.
    ///
    /// # Safety
    /// - `cu_device_ptr` must be memory of this stream's device.
    /// - It must be the base of a live allocation of at least
    ///   `len * std::mem::size_of::<T>()` bytes, or 0 with `len == 0`.
    /// - Every access to the memory by other streams or by host code must be
    ///   ordered before the slice's free, for example by synchronizing them
    ///   before the drop.
    /// - The memory may come from either allocator family: `cuMemAlloc`, or
    ///   `cuMemAllocAsync` / `cuMemAllocFromPoolAsync` from any pool of the
    ///   device, including the [crate::driver::CudaMemPool] of another context
    ///   ([`CudaSlice::leak()`]). NVIDIA allows freeing across the families:
    ///   `cuMemFree` accepts `cuMemAllocAsync` and `cuMemAllocFromPoolAsync`
    ///   pointers, without implicit synchronization, and "Callers must ensure
    ///   that all accesses to these pointer have completed before invoking
    ///   ::cuMemFree" (CUDA 12.6 `cuda.h`, `cuMemFree`); memory from
    ///   `cudaMalloc()` may be freed with `cudaFreeAsync()` and memory from
    ///   `cudaMallocAsync()` with `cudaFree()` (CUDA Programming Guide,
    ///   stream-ordered memory allocation). `cuMemFreeAsync` returns pool
    ///   memory to the pool that owns it. Freeing pool memory with
    ///   `cuMemFree` (a context whose [CudaContext::has_async_alloc] is
    ///   `false`) releases it, but on driver 580 (measured) the pool's
    ///   `CU_MEMPOOL_ATTR_USED_MEM_CURRENT` is left unchanged, so prefer
    ///   `cuMemFreeAsync` for pool memory.
    /// - The memory may not be valid for type `T`, so some sort of memset
    ///   operation should be called on the memory.
    pub unsafe fn upgrade_device_ptr<T>(
        self: &Arc<Self>,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        let (read, write) = if self.ctx.is_event_tracking() {
            (
                Some(self.ctx.new_event(None).unwrap()),
                Some(self.ctx.new_event(None).unwrap()),
            )
        } else {
            (None, None)
        };
        CudaSlice {
            cu_device_ptr,
            len,
            read,
            write,
            stream: self.clone(),
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_event_try_is_complete() {
        use std::boxed::Box;
        use std::sync::mpsc::{channel, Receiver};
        use std::time::Duration;

        unsafe extern "C" fn wait_for_release(data: *mut std::ffi::c_void) {
            // The callback owns the receiver and makes no CUDA calls.
            let receiver = unsafe { Box::from_raw(data.cast::<Receiver<()>>()) };
            let _ = receiver.recv_timeout(Duration::from_secs(30));
        }

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let event = ctx.new_event(None).unwrap();
        assert_eq!(event.try_is_complete(), Ok(true));

        let (release, receiver) = channel();
        let receiver = Box::into_raw(Box::new(receiver));
        let launch = unsafe {
            sys::cuLaunchHostFunc(stream.cu_stream(), Some(wait_for_release), receiver.cast())
                .result()
        };
        if let Err(error) = launch {
            // A failed launch did not transfer ownership to the callback.
            unsafe { drop(Box::from_raw(receiver)) };
            panic!("cuLaunchHostFunc failed: {error:?}");
        }
        event.record(&stream).unwrap();
        let pending = event.try_is_complete();
        let legacy_complete = event.is_complete();
        release.send(()).unwrap();
        event.synchronize().unwrap();

        assert_eq!(pending, Ok(false));
        assert!(!legacy_complete);
        assert_eq!(event.try_is_complete(), Ok(true));
        assert!(event.is_complete());
    }

    #[test]
    fn test_event_try_is_complete_preserves_capture_error() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let event = ctx.new_event(None).unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        event.record(&stream).unwrap();
        let status = event.try_is_complete();
        let legacy_complete = event.is_complete();
        // Querying a captured event invalidates capture. End it before asserting.
        let _ = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );

        assert_eq!(
            status,
            Err(DriverError(sys::CUresult::CUDA_ERROR_CAPTURED_EVENT))
        );
        assert!(!legacy_complete);
    }

    #[test]
    fn test_transmutes() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut slice = stream.alloc_zeros::<u8>(100).unwrap();
        assert!(unsafe { slice.transmute::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute::<f32>(26) }.is_none());
        assert!(unsafe { slice.transmute_mut::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute_mut::<f32>(26) }.is_none());

        {
            let view = slice.slice(0..100);
            assert!(unsafe { view.transmute::<f32>(25) }.is_some());
            assert!(unsafe { view.transmute::<f32>(26) }.is_none());
        }

        {
            let mut view_mut = slice.slice_mut(0..100);
            assert!(unsafe { view_mut.transmute::<f32>(25) }.is_some());
            assert!(unsafe { view_mut.transmute::<f32>(26) }.is_none());
            assert!(unsafe { view_mut.transmute_mut::<f32>(25) }.is_some());
            assert!(unsafe { view_mut.transmute_mut::<f32>(26) }.is_none());
        }
    }

    #[test]
    fn test_threading() {
        let ctx1 = CudaContext::new(0).unwrap();
        let ctx2 = ctx1.clone();

        let thread1 = std::thread::spawn(move || {
            ctx1.bind_to_thread()?;
            ctx1.default_stream().alloc_zeros::<f32>(10)
        });
        let thread2 = std::thread::spawn(move || {
            ctx2.bind_to_thread()?;
            ctx2.default_stream().alloc_zeros::<f32>(10)
        });

        let _: crate::driver::CudaSlice<f32> = thread1.join().unwrap().unwrap();
        let _: crate::driver::CudaSlice<f32> = thread2.join().unwrap().unwrap();
    }

    #[test]
    fn test_post_build_arc_count() {
        let ctx = CudaContext::new(0).unwrap();
        assert_eq!(Arc::strong_count(&ctx), 1);
    }

    #[test]
    fn test_borrowed_stream_is_not_destroyed_on_drop() {
        let ctx = CudaContext::new(0).unwrap();
        ctx.bind_to_thread().unwrap();
        let raw = result::stream::create(result::stream::StreamKind::NonBlocking).unwrap();

        {
            let stream = unsafe { ctx.wrap_borrowed_stream(raw) };
            assert_eq!(stream.cu_stream(), raw);
            assert_eq!(stream.context(), &ctx);

            let data = stream.clone_htod(&[1_u32, 2, 3]).unwrap();
            stream.synchronize().unwrap();
            drop(data);
        }

        unsafe {
            result::stream::synchronize(raw).unwrap();
            result::stream::destroy(raw).unwrap();
        }
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let ctx = CudaContext::new(0).unwrap();
        assert_eq!(Arc::strong_count(&ctx), 1);
        let stream = ctx.default_stream();
        assert_eq!(Arc::strong_count(&ctx), 2);
        let t = stream.alloc_zeros::<f32>(1).unwrap();
        assert_eq!(Arc::strong_count(&ctx), 4);
        assert_eq!(Arc::strong_count(&stream), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&ctx), 2);
        assert_eq!(Arc::strong_count(&stream), 1);
        drop(stream);
        assert_eq!(Arc::strong_count(&ctx), 1);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let (free1, total1) = ctx.mem_get_info().unwrap();

        let t = stream.clone_htod(&[0.0f32; 5]).unwrap();
        let (free2, total2) = ctx.mem_get_info().unwrap();
        assert_eq!(total1, total2);
        assert!(free2 < free1);

        drop(t);
        ctx.synchronize().unwrap();

        let (free3, total3) = ctx.mem_get_info().unwrap();
        assert_eq!(total2, total3);
        assert!(free3 > free2);
        assert_eq!(free3, free1);
    }

    #[test]
    fn test_ctx_copy_to_views() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let smalls = [
            stream.clone_htod(&[-1.0f32, -0.8]).unwrap(),
            stream.clone_htod(&[-0.6, -0.4]).unwrap(),
            stream.clone_htod(&[-0.2, 0.0]).unwrap(),
            stream.clone_htod(&[0.2, 0.4]).unwrap(),
            stream.clone_htod(&[0.6, 0.8]).unwrap(),
        ];
        let mut big = stream.alloc_zeros::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.slice_mut(offset..offset + small.len());
            stream.memcpy_dtod(small, &mut sub).unwrap();
            offset += small.len();
        }

        assert_eq!(
            stream.clone_dtoh(&big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }

    #[test]
    fn test_leak_and_upgrade() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let a = stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let ptr = a.leak();
        let b = unsafe { stream.upgrade_device_ptr::<f32>(ptr, 3) };
        assert_eq!(stream.clone_dtoh(&b).unwrap(), &[1.0, 2.0, 3.0]);

        let ptr = b.leak();
        let c = unsafe { stream.upgrade_device_ptr::<f32>(ptr, 5) };
        assert_eq!(stream.clone_dtoh(&c).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// See https://github.com/chelsea0x3b/cudarc/issues/160
    #[test]
    fn test_slice_is_freed_with_correct_context() {
        let ctx0 = CudaContext::new(0).unwrap();
        let slice = ctx0.default_stream().clone_htod(&[1.0; 10]).unwrap();
        let ctx1 = CudaContext::new(0).unwrap();
        ctx1.bind_to_thread().unwrap();
        drop(ctx0);
        drop(slice);
        drop(ctx1);
    }

    /// See https://github.com/chelsea0x3b/cudarc/issues/161
    #[test]
    fn test_copy_uses_correct_context() {
        let ctx0 = CudaContext::new(0).unwrap();
        let _ctx1 = CudaContext::new(0).unwrap();
        let slice = ctx0.default_stream().clone_htod(&[1.0; 10]).unwrap();
        let _out = ctx0.default_stream().clone_dtoh(&slice).unwrap();
    }

    #[test]
    fn test_htod_copy_pinned() {
        let truth = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut pinned = unsafe { ctx.alloc_pinned::<f32>(10) }.unwrap();
        pinned.as_mut_slice().unwrap().clone_from_slice(&truth);
        assert_eq!(pinned.as_slice().unwrap(), &truth);
        let dst = stream.clone_htod(&pinned).unwrap();
        let host = stream.clone_dtoh(&dst).unwrap();
        assert_eq!(&host, &truth);
    }

    #[test]
    fn test_htod_copy_pinned_with_default_flags() {
        let truth = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut pinned = unsafe { ctx.alloc_pinned_with_flags::<f32>(10, 0) }.unwrap();
        pinned.as_mut_slice().unwrap().clone_from_slice(&truth);
        assert_eq!(pinned.as_slice().unwrap(), &truth);
        let dst = stream.clone_htod(&pinned).unwrap();
        let host = stream.clone_dtoh(&dst).unwrap();
        assert_eq!(&host, &truth);
    }

    #[test]
    #[should_panic(expected = "Pinned host allocation size overflow")]
    fn test_alloc_pinned_panics_on_size_overflow() {
        let ctx = CudaContext::new(0).unwrap();
        let _ = unsafe { ctx.alloc_pinned_with_flags::<u32>(usize::MAX, 0) };
    }

    #[test]
    fn test_default_pinned_host_reads_are_faster_than_write_combined() {
        fn timed_host_reads(values: &[u32], n_samples: usize) -> (std::time::Duration, u64) {
            let start = Instant::now();
            let mut sum = 0_u64;
            for _ in 0..n_samples {
                for value in black_box(values) {
                    sum = sum.wrapping_add(u64::from(*value));
                }
            }
            (start.elapsed(), black_box(sum))
        }

        let ctx = CudaContext::new(0).unwrap();
        let n = 1 << 20;
        let n_samples = 5;
        let mut write_combined =
            unsafe { ctx.alloc_pinned_with_flags::<u32>(n, sys::CU_MEMHOSTALLOC_WRITECOMBINED) }
                .unwrap();
        let mut default = unsafe { ctx.alloc_pinned_with_flags::<u32>(n, 0) }.unwrap();
        write_combined.as_mut_slice().unwrap().fill(1);
        default.as_mut_slice().unwrap().fill(1);

        let (write_combined_elapsed, write_combined_sum) =
            timed_host_reads(write_combined.as_slice().unwrap(), n_samples);
        let (default_elapsed, default_sum) =
            timed_host_reads(default.as_slice().unwrap(), n_samples);
        assert_eq!(write_combined_sum, default_sum);
        std::println!(
            "default pinned host reads: {default_elapsed:?}; write-combined host reads: {write_combined_elapsed:?}"
        );
        // The performance gap should be large, but leave margin for device and host variance.
        assert!(
            default_elapsed.as_secs_f32() * 2.0 < write_combined_elapsed.as_secs_f32(),
            "{default_elapsed:?} vs {write_combined_elapsed:?}"
        );
    }

    #[test]
    fn test_pinned_copy_is_faster() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let n = 100_000;
        let n_samples = 5;
        let not_pinned = std::vec![0.0f32; n];

        let start = Instant::now();
        for _ in 0..n_samples {
            let _ = stream.clone_htod(&not_pinned).unwrap();
            stream.synchronize().unwrap();
        }
        let unpinned_elapsed = start.elapsed() / n_samples;

        let pinned = unsafe { ctx.alloc_pinned::<f32>(n) }.unwrap();

        let start = Instant::now();
        for _ in 0..n_samples {
            let _ = stream.clone_htod(&pinned).unwrap();
            stream.synchronize().unwrap();
        }
        let pinned_elapsed = start.elapsed() / n_samples;

        // pinned memory transfer speed should be at least 2x faster, but this depends
        // on device
        assert!(
            pinned_elapsed.as_secs_f32() * 1.5 < unpinned_elapsed.as_secs_f32(),
            "{unpinned_elapsed:?} vs {pinned_elapsed:?}"
        );
    }

    #[test]
    fn test_primary_context_is_primary() {
        let ctx = CudaContext::new(0).unwrap();
        assert!(ctx.is_primary());
    }

    /// Helper to create a non-primary context for testing `from_raw_context`.
    /// Uses `cuCtxCreate_v4` (CUDA 12.050+) or `cuCtxCreate_v3` (CUDA 11.040–12.040).
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
    ))]
    fn create_non_primary_context() -> (sys::CUdevice, sys::CUcontext) {
        result::init().unwrap();
        let cu_device = result::device::get(0).unwrap();

        #[cfg(any(
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000",
            feature = "cuda-13010",
        ))]
        let cu_ctx = unsafe { result::ctx::create_v4(std::ptr::null_mut(), 0, cu_device) }
            .expect("cuCtxCreate_v4 failed");

        #[cfg(not(any(
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000",
            feature = "cuda-13010",
        )))]
        let cu_ctx =
            unsafe { result::ctx::create_v3(0, cu_device) }.expect("cuCtxCreate_v3 failed");

        assert!(!cu_ctx.is_null());
        (cu_device, cu_ctx)
    }

    #[test]
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
    ))]
    fn test_from_raw_context_creates_and_destroys() {
        let (cu_device, cu_ctx) = create_non_primary_context();

        let ctx = unsafe { CudaContext::from_raw_context(0, cu_device, cu_ctx) }.unwrap();
        assert!(!ctx.is_primary());
        // Verify the context is bound and usable.
        ctx.bind_to_thread().unwrap();
        // Drop should call cuCtxDestroy_v2, not primary_ctx::release.
        drop(ctx);
    }

    #[test]
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
    ))]
    fn test_from_raw_context_bind_to_thread() {
        let (cu_device, cu_ctx) = create_non_primary_context();

        let ctx = unsafe { CudaContext::from_raw_context(0, cu_device, cu_ctx) }.unwrap();

        // Verify bind_to_thread works from another thread.
        let ctx2 = ctx.clone();
        let handle = std::thread::spawn(move || {
            ctx2.bind_to_thread().unwrap();
            let stream = ctx2.default_stream();
            let data = stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
            let result = stream.clone_dtoh(&data).unwrap();
            assert_eq!(result, std::vec![1.0f32, 2.0, 3.0]);
        });
        handle.join().unwrap();
    }

    #[test]
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010",
    ))]
    fn test_new_non_primary_creates_and_destroys() {
        let ctx = CudaContext::new_non_primary(0, 0).unwrap();
        assert!(!ctx.is_primary());
        ctx.bind_to_thread().unwrap();
        drop(ctx);
    }

    #[test]
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010",
    ))]
    fn test_new_non_primary_htod_dtoh() {
        let ctx = CudaContext::new_non_primary(0, 0).unwrap();
        let stream = ctx.default_stream();
        let data = stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let result = stream.clone_dtoh(&data).unwrap();
        assert_eq!(result, std::vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010",
    ))]
    fn test_new_non_primary_cross_thread() {
        let ctx = CudaContext::new_non_primary(0, 0).unwrap();
        let ctx2 = ctx.clone();
        let handle = std::thread::spawn(move || {
            ctx2.bind_to_thread().unwrap();
            let stream = ctx2.default_stream();
            let data = stream.clone_htod(&[4.0f32, 5.0, 6.0]).unwrap();
            let result = stream.clone_dtoh(&data).unwrap();
            assert_eq!(result, std::vec![4.0f32, 5.0, 6.0]);
        });
        handle.join().unwrap();
    }
}

// FATHOMDB PATCH BEGIN: allocator fallback tests (see FATHOMDB.md, item 4).
#[cfg(test)]
mod fathomdb_alloc_fallback {
    use super::*;

    /// A primary context forced onto the synchronous allocator, as selected
    /// when the device's default memory pool is unavailable (or, off target,
    /// when pools are unsupported). `None` when no CUDA driver or device is
    /// present.
    fn sync_context() -> Option<Arc<CudaContext>> {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP fathomdb_alloc_fallback: no CUDA driver or device");
            return None;
        }
        let mut ctx = CudaContext::new(0).unwrap();
        Arc::get_mut(&mut ctx)
            .expect("fresh context")
            .has_async_alloc = false;
        Some(ctx)
    }

    fn pool_query_must_not_run() -> Result<sys::CUmemoryPool, DriverError> {
        panic!("the default pool must not be queried when pools are unsupported")
    }

    const ON_TARGET: bool = true;
    const OFF_TARGET: bool = false;

    #[test]
    fn off_target_selection_is_the_upstream_rule_whatever_the_pool_would_say() {
        use sys::CUresult::{
            CUDA_ERROR_ILLEGAL_ADDRESS, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY,
        };
        // Upstream 0.19.10: stream-ordered allocation iff the pool attribute
        // is positive, and the default pool is never queried.
        for supported in [-1, 0, 1, 2] {
            assert_eq!(
                async_alloc_decision(OFF_TARGET, supported, pool_query_must_not_run),
                Ok(supported > 0)
            );
        }
        for code in [
            CUDA_ERROR_OUT_OF_MEMORY,
            CUDA_ERROR_NOT_SUPPORTED,
            CUDA_ERROR_ILLEGAL_ADDRESS,
        ] {
            let mut queried = false;
            let decision = async_alloc_decision(OFF_TARGET, 1, || {
                queried = true;
                Err(DriverError(code))
            });
            assert_eq!(decision, Ok(true), "{code:?}");
            assert!(!queried, "off target the default pool must not be queried");
        }
    }

    #[test]
    fn off_target_zero_length_sync_handling_is_upstream() {
        // Upstream passes a zero-byte request to cuMemAlloc and frees every
        // synchronous pointer, null included.
        assert!(!sync_zero_bytes_is_null(OFF_TARGET, 0));
        assert!(sync_free_needed(OFF_TARGET, 0));
        assert!(sync_free_needed(OFF_TARGET, 0x1000));
    }

    #[test]
    fn on_target_zero_length_sync_buffers_are_null_and_never_freed() {
        assert!(sync_zero_bytes_is_null(ON_TARGET, 0));
        assert!(!sync_zero_bytes_is_null(ON_TARGET, 4));
        assert!(!sync_free_needed(ON_TARGET, 0));
        assert!(sync_free_needed(ON_TARGET, 0x1000));
    }

    #[test]
    fn async_alloc_requires_pool_support_and_an_obtainable_default_pool() {
        let pool = || Ok(std::ptr::null_mut());

        assert_eq!(async_alloc_decision(ON_TARGET, 1, pool), Ok(true));
        assert_eq!(async_alloc_decision(ON_TARGET, 2, pool), Ok(true));
        assert_eq!(
            async_alloc_decision(ON_TARGET, 0, pool_query_must_not_run),
            Ok(false)
        );
        assert_eq!(
            async_alloc_decision(ON_TARGET, -1, pool_query_must_not_run),
            Ok(false)
        );
    }

    #[test]
    fn only_pool_unavailable_errors_select_the_synchronous_allocator() {
        use sys::CUresult::{CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY};
        // The measured Tegra failure: pools are reported, the default pool
        // cannot be created.
        assert_eq!(
            async_alloc_decision(ON_TARGET, 1, || Err(DriverError(CUDA_ERROR_OUT_OF_MEMORY))),
            Ok(false)
        );
        // The device states it has no usable default pool.
        assert_eq!(
            async_alloc_decision(ON_TARGET, 1, || Err(DriverError(CUDA_ERROR_NOT_SUPPORTED))),
            Ok(false)
        );
    }

    #[test]
    fn unrelated_pool_query_errors_are_propagated_not_downgraded() {
        use sys::CUresult::{
            CUDA_ERROR_ILLEGAL_ADDRESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
            CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_UNKNOWN,
        };
        for code in [
            CUDA_ERROR_INVALID_CONTEXT,
            CUDA_ERROR_INVALID_DEVICE,
            // Sticky errors from earlier asynchronous work surface on any call.
            CUDA_ERROR_ILLEGAL_ADDRESS,
            CUDA_ERROR_LAUNCH_FAILED,
            CUDA_ERROR_UNKNOWN,
        ] {
            assert_eq!(
                async_alloc_decision(ON_TARGET, 1, || Err(DriverError(code))),
                Err(DriverError(code)),
                "{code:?} must not select an allocator"
            );
        }
    }

    #[test]
    fn every_wrapper_of_a_device_reuses_the_first_decision() {
        let modes = AllocModeByDevice::new();
        assert_eq!(modes.get_or_decide(0, || Ok(false)), Ok(false));
        // A later wrapper of device 0 whose own probe would say "async" must
        // still agree with the first, or buffers moved between wrappers with
        // `leak`/`upgrade_device_ptr` would be freed by the wrong API.
        assert_eq!(
            modes.get_or_decide(0, || panic!("device 0 was already decided")),
            Ok(false)
        );
        assert_eq!(modes.get_or_decide(0, || Ok(true)), Ok(false));
        // Devices are decided independently.
        assert_eq!(modes.get_or_decide(1, || Ok(true)), Ok(true));
        assert_eq!(modes.get_or_decide(1, || Ok(false)), Ok(true));
        assert_eq!(modes.get_or_decide(0, || Ok(true)), Ok(false));
    }

    #[test]
    fn a_failed_decision_is_not_remembered() {
        use sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS;
        let modes = AllocModeByDevice::new();
        assert_eq!(
            modes.get_or_decide(0, || Err(DriverError(CUDA_ERROR_ILLEGAL_ADDRESS))),
            Err(DriverError(CUDA_ERROR_ILLEGAL_ADDRESS))
        );
        assert_eq!(modes.get_or_decide(0, || Ok(true)), Ok(true));
        assert_eq!(modes.get_or_decide(0, || Ok(false)), Ok(true));
    }

    #[test]
    fn concurrent_wrappers_of_a_device_agree() {
        let modes = Arc::new(AllocModeByDevice::new());
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let modes = modes.clone();
                std::thread::spawn(move || modes.get_or_decide(7, || Ok(i % 2 == 0)).unwrap())
            })
            .collect();
        let decisions: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(
            decisions.iter().all(|&d| d == decisions[0]),
            "racing wrappers disagreed: {decisions:?}"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn wrappers_of_one_device_share_the_allocator() {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP fathomdb_alloc_fallback: no CUDA driver or device");
            return;
        }
        let first = CudaContext::new(0).unwrap();
        let second = CudaContext::new(0).unwrap();
        assert_eq!(first.has_async_alloc(), second.has_async_alloc());
        assert_eq!(
            PROCESS_ALLOC_MODE.get_or_decide(first.cu_device, || panic!("already decided")),
            Ok(first.has_async_alloc())
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn sync_zero_byte_request_is_a_null_pointer_without_a_driver_call() {
        // No CUDA library is loaded by this test; a driver call would panic
        // on hosts without one and fail with CUDA_ERROR_INVALID_VALUE on hosts
        // with one.
        assert_eq!(unsafe { malloc_sync_or_null(0) }, Ok(0));
    }

    #[test]
    fn new_context_selection_matches_default_pool_availability() {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP fathomdb_alloc_fallback: no CUDA driver or device");
            return;
        }
        let ctx = CudaContext::new(0).unwrap();
        let pools = ctx
            .attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
            .unwrap();
        let pool = unsafe { result::device::get_default_mem_pool(ctx.cu_device) };
        assert_eq!(
            Ok(ctx.has_async_alloc()),
            async_alloc_decision(SYNC_FALLBACK, pools, || pool)
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn sync_zero_length_allocations_are_null_and_free_cleanly() {
        let Some(ctx) = sync_context() else { return };
        let stream = ctx.default_stream();

        let null = stream.null::<f32>().unwrap();
        assert_eq!(null.cu_device_ptr, 0);
        assert!(null.is_empty());
        let unset = unsafe { stream.alloc::<f32>(0) }.unwrap();
        assert_eq!(unset.cu_device_ptr, 0);
        let zeros = stream.alloc_zeros::<f32>(0).unwrap();
        assert_eq!(zeros.cu_device_ptr, 0);
        assert!(stream.clone_dtoh(&zeros).unwrap().is_empty());
        let uploaded = stream.clone_htod(&Vec::<f32>::new()).unwrap();
        assert_eq!(uploaded.cu_device_ptr, 0);

        drop((null, unset, zeros, uploaded));
        stream.synchronize().unwrap();
        ctx.check_err().unwrap();
    }

    #[test]
    fn sync_context_round_trips_data() {
        let Some(ctx) = sync_context() else { return };
        let stream = ctx.default_stream();

        let device = stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert_ne!(device.cu_device_ptr, 0);
        assert_eq!(
            stream.clone_dtoh(&device).unwrap(),
            std::vec![1.0f32, 2.0, 3.0, 4.0]
        );
        let zeros = stream.alloc_zeros::<u8>(4).unwrap();
        assert_eq!(stream.clone_dtoh(&zeros).unwrap(), std::vec![0u8; 4]);

        drop((device, zeros));
        stream.synchronize().unwrap();
        ctx.check_err().unwrap();
    }

    /// Every constructor of an ordinary context of device 0 that these
    /// bindings have, with its name. `new_cig` needs a graphics queue,
    /// which a Linux host cannot give, so its result is usually an error.
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    fn every_ordinary_constructor() -> Vec<(&'static str, Result<Arc<CudaContext>, DriverError>)> {
        result::init().unwrap();
        let cu_device = result::device::get(0).unwrap();
        #[allow(unused_mut)]
        let mut made = std::vec![("new", CudaContext::new(0))];
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000",
            feature = "cuda-13010"
        ))]
        made.push(("new_non_primary", CudaContext::new_non_primary(0, 0)));
        #[cfg(any(
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000",
            feature = "cuda-13010"
        ))]
        {
            let mut cig = sys::CUctxCigParam {
                sharedDataType: sys::CUcigDataType::CIG_DATA_TYPE_D3D12_COMMAND_QUEUE,
                sharedData: std::ptr::null_mut(),
            };
            made.push(("new_cig", CudaContext::new_cig(0, 0, &mut cig)));
        }
        #[cfg(not(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040"
        )))]
        let cu_ctx = {
            let mut cu_ctx = std::ptr::null_mut();
            unsafe { sys::cuCtxCreate_v4(&mut cu_ctx, std::ptr::null_mut(), 0, cu_device) }
                .result()
                .unwrap();
            cu_ctx
        };
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040"
        ))]
        let cu_ctx = unsafe { result::ctx::create_v3(0, cu_device) }.unwrap();
        made.push(("from_raw_context", unsafe {
            CudaContext::from_raw_context(0, cu_device, cu_ctx)
        }));
        for (name, ctx) in &made {
            std::println!("{name}: {:?}", ctx.as_ref().map(|_| "context"));
        }
        made
    }

    /// CF-AC-05 (test plan § 3.3): off target, no constructor of an
    /// ordinary context queries the device's default pool.
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    #[test]
    fn off_target_constructors_never_query_the_default_pool() {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP fathomdb_alloc_fallback: no CUDA driver or device");
            return;
        }
        use result::device::seams::DEFAULT_POOL_QUERIES as QUERIES;
        result::init().unwrap();
        let cu_device = result::device::get(0).unwrap();
        let ctx = CudaContext::new(0).unwrap();

        // Positive control: a direct query is counted.
        QUERIES.set(0);
        unsafe { result::device::get_default_mem_pool(cu_device) }.unwrap();
        assert_eq!(QUERIES.get(), 1, "the counter was not reached");
        QUERIES.set(0);
        drop(ctx);

        let made = every_ordinary_constructor();
        assert_eq!(QUERIES.get(), 0, "default-pool queries");
        for (name, ctx) in &made {
            if *name != "new_cig" {
                assert!(ctx.is_ok(), "{name}: {ctx:?}");
            }
        }
    }

    /// CF-AC-05 (test plan § 3.3): off target, each constructor's choice of
    /// stream-ordered allocation is upstream's inline rule
    /// `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED > 0` (`core.rs:78, 150,
    /// 201, 238` at `ec1cac71`).
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    #[test]
    fn off_target_choice_equals_the_upstream_inline_rule() {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP fathomdb_alloc_fallback: no CUDA driver or device");
            return;
        }
        result::init().unwrap();
        let cu_device = result::device::get(0).unwrap();
        let pools = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )
        }
        .unwrap();
        for (name, ctx) in every_ordinary_constructor() {
            match ctx {
                Ok(ctx) => assert_eq!(ctx.has_async_alloc(), pools > 0, "{name}"),
                Err(e) => assert_eq!(name, "new_cig", "{name} failed: {e:?}"),
            }
        }
    }

    /// The result kind of each zero-length request on the synchronous path
    /// of a context of device 0, and the context's error state afterwards.
    /// The same text runs at `ec1cac71` and at the fork (test plan § 3.3).
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    fn zero_length_sync_outcomes() -> Option<Vec<String>> {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP zero_length_sync_outcomes: no CUDA driver or device");
            return None;
        }
        fn kind(result: Result<sys::CUdeviceptr, DriverError>) -> String {
            match result {
                Ok(0) => "Ok(null)".into(),
                Ok(_) => "Ok(non-null)".into(),
                Err(e) => std::format!("Err({:?})", e.0),
            }
        }
        let mut ctx = CudaContext::new(0).unwrap();
        Arc::get_mut(&mut ctx)
            .expect("fresh context")
            .has_async_alloc = false;
        let stream = ctx.default_stream();
        let mut outcomes = Vec::new();

        ctx.bind_to_thread().unwrap();
        let raw = unsafe { result::malloc_sync(0) };
        if let Ok(ptr) = raw {
            if ptr != 0 {
                unsafe { result::free_sync(ptr) }.unwrap();
            }
        }
        outcomes.push(std::format!("cuMemAlloc(0): {}", kind(raw)));
        let alloc = unsafe { stream.alloc::<u8>(0) };
        outcomes.push(std::format!(
            "alloc(0): {}",
            kind(alloc.as_ref().map(|s| s.cu_device_ptr).map_err(|e| *e))
        ));
        drop(alloc);
        let zeros = stream.alloc_zeros::<u8>(0);
        outcomes.push(std::format!(
            "alloc_zeros(0): {}",
            kind(zeros.as_ref().map(|s| s.cu_device_ptr).map_err(|e| *e))
        ));
        drop(zeros);
        let null = stream.null::<u8>();
        outcomes.push(std::format!(
            "null(): {}",
            kind(null.as_ref().map(|s| s.cu_device_ptr).map_err(|e| *e))
        ));
        drop(null);
        stream.synchronize().unwrap();
        outcomes.push(std::format!(
            "check_err: {:?}",
            ctx.check_err().map_err(|e| e.0)
        ));
        Some(outcomes)
    }

    /// CF-AC-05 (test plan § 3.3): off target, zero-length requests on the
    /// synchronous path give the result kinds that the same body
    /// ([zero_length_sync_outcomes]) gave at `ec1cac71`, recorded in
    /// `fathomdb/status.md` (FC6-a): every one reaches `cuMemAlloc(0)` and
    /// fails with `CUDA_ERROR_INVALID_VALUE`, and the context records no
    /// error.
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    #[test]
    fn off_target_zero_length_sync_behaviour_matches_the_base() {
        let Some(outcomes) = zero_length_sync_outcomes() else {
            return;
        };
        let base = [
            "cuMemAlloc(0): Err(CUDA_ERROR_INVALID_VALUE)",
            "alloc(0): Err(CUDA_ERROR_INVALID_VALUE)",
            "alloc_zeros(0): Err(CUDA_ERROR_INVALID_VALUE)",
            "null(): Err(CUDA_ERROR_INVALID_VALUE)",
            "check_err: Ok(())",
        ];
        assert_eq!(outcomes, base);
    }
}
// FATHOMDB PATCH END

// FATHOMDB PATCH BEGIN: private memory pool tests (see FATHOMDB.md, item 5).
#[cfg(test)]
mod fathomdb_private_pool {
    use super::*;
    use crate::driver::safe::mem_pool::{
        device_location, pool_props, seams, CudaMemPool, MemPoolProps,
    };

    fn have_device() -> bool {
        if !unsafe { sys::is_culib_present() } || CudaContext::device_count().unwrap_or(0) < 1 {
            std::eprintln!("SKIP fathomdb_private_pool: no CUDA driver or device");
            return false;
        }
        true
    }

    const SMALL_POOL: MemPoolProps = MemPoolProps {
        max_size: 64 << 20,
        release_threshold: 0,
    };

    #[test]
    fn a_context_with_its_own_pool_is_private_whatever_the_device_decided() {
        for has_async_alloc in [false, true] {
            assert_eq!(alloc_mode_of(true, has_async_alloc), AllocMode::Private);
        }
        assert_eq!(alloc_mode_of(false, true), AllocMode::Default);
        assert_eq!(alloc_mode_of(false, false), AllocMode::Synchronous);
    }

    #[test]
    fn pool_props_are_pinned_device_memory_on_the_ordinal() {
        let raw = pool_props(
            3,
            &MemPoolProps {
                max_size: 3 << 30,
                release_threshold: 0,
            },
        );
        assert_eq!(
            raw.allocType,
            sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED
        );
        assert_eq!(
            raw.handleTypes,
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE
        );
        assert_eq!(
            raw.location.type_,
            sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE
        );
        assert_eq!(location_id(&raw.location), 3);
        assert_eq!(
            location_bytes(raw.location),
            location_bytes(device_location(3))
        );
        assert!(raw.win32SecurityAttributes.is_null());
        #[cfg(feature = "cuda-12060")]
        assert_eq!(raw.maxSize, 3 << 30);
    }

    /// The location's `id`, which the 13020+ bindings hold in a union.
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    )))]
    fn location_id(location: &sys::CUmemLocation) -> i32 {
        // SAFETY: every variant of the union is plain integer data.
        unsafe { location.__bindgen_anon_1.id }
    }

    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    ))]
    fn location_id(location: &sys::CUmemLocation) -> i32 {
        location.id
    }

    fn location_bytes(location: sys::CUmemLocation) -> [u8; 8] {
        // SAFETY: CUmemLocation is 8 bytes of integers with no padding in every
        // binding (design D-8); the size is checked at compile time here.
        unsafe { std::mem::transmute::<sys::CUmemLocation, [u8; 8]>(location) }
    }

    /// Design D-8: the props start from zeroed memory (`reserved` "must be
    /// 0", CUDA 12.6 `cuda.h`), and `maxSize` is written on every binding
    /// that has it, from `cuda-12020` on.
    #[test]
    fn pool_props_are_zeroed_then_written() {
        for max_size in [0, 4096, (5 << 30) + 4096] {
            let raw = pool_props(
                2,
                &MemPoolProps {
                    max_size,
                    release_threshold: 7,
                },
            );
            assert!(
                raw.reserved.iter().all(|b| *b == 0),
                "reserved: {:?}",
                raw.reserved
            );
            assert_eq!(
                raw.allocType,
                sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED
            );
            assert_eq!(
                raw.handleTypes,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE
            );
            assert_eq!(
                location_bytes(raw.location),
                location_bytes(device_location(2))
            );
            assert!(raw.win32SecurityAttributes.is_null());
            #[cfg(not(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010"
            )))]
            assert_eq!(raw.maxSize, max_size, "maxSize");
            #[cfg(not(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050"
            )))]
            assert_eq!(raw.usage, 0, "usage");
            #[cfg(not(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000",
                feature = "cuda-13010",
                feature = "cuda-13020",
                feature = "cuda-13030"
            )))]
            assert_eq!(raw.gpuDirectRDMACapable, 0, "gpuDirectRDMACapable");
        }
    }

    /// CF-REQ-04: bindings older than `cuda-12020` have no `maxSize`, so a
    /// nonzero request is refused before any driver call, never ignored.
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    ))]
    #[test]
    fn max_size_without_binding_support_is_not_supported() {
        let _reset = seams::Reset::new();
        let refused = sys::CUresult::CUDA_ERROR_UNKNOWN;
        // Every call below stops at the counted entry instead of the driver.
        seams::FAIL_DRIVER_ENTRY.set(Some(refused));

        // Positive control: a request nothing refuses reaches the entry.
        assert_eq!(
            CudaMemPool::create(0, &MemPoolProps::default()).map(|_| ()),
            Err(DriverError(refused))
        );
        assert_eq!(seams::DRIVER_ENTRIES.get(), 1);
        seams::DRIVER_ENTRIES.set(0);

        assert_eq!(
            CudaMemPool::create(0, &SMALL_POOL).map(|_| ()),
            Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_SUPPORTED))
        );
        assert_eq!(seams::DRIVER_ENTRIES.get(), 0, "driver entries");
    }

    /// Design D-4: with `maxSize` in the bindings, a driver older than 12020
    /// reads those bytes as `reserved`, which must be 0, so a nonzero request
    /// is refused before the pool is created.
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )))]
    #[test]
    fn max_size_on_a_driver_older_than_12020_is_not_supported() {
        if !have_device() {
            return;
        }
        let _reset = seams::Reset::new();
        seams::DRIVER_VERSION.set(Some(12010));

        // Positive control: max_size 0 needs no support and is counted.
        let pool = CudaMemPool::create(0, &MemPoolProps::default()).unwrap();
        assert_eq!(seams::POOL_CREATES.get(), 1);
        drop(pool);
        seams::POOL_CREATES.set(0);

        assert_eq!(
            CudaMemPool::create(0, &SMALL_POOL).map(|_| ()),
            Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_SUPPORTED))
        );
        assert_eq!(seams::POOL_CREATES.get(), 0, "pools created");
    }

    /// CF-AC-04b, design D-4: a `cuda-13020`+ build on a driver older than
    /// 13020 never requests an attribute the driver does not know (9-14),
    /// because the read-back is gated on `cuDriverGetVersion`.
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    )))]
    #[test]
    fn attributes_newer_than_the_driver_are_never_requested() {
        if !have_device() {
            return;
        }
        let _reset = seams::Reset::new();
        let newer = &result::mem_pool::seams::NEWER_ATTRIBUTE_REQUESTS;
        let mut driver = 0;
        unsafe { sys::cuDriverGetVersion(&mut driver) }
            .result()
            .unwrap();
        if driver >= 13020 {
            std::eprintln!(
                "SKIP attributes_newer_than_the_driver_are_never_requested: driver {driver} knows attributes 9-14"
            );
            return;
        }

        // Positive control: with the version forced to 13020 the read-back
        // goes through the counted result-layer call (the real driver may
        // reject it; only the request matters here).
        seams::DRIVER_VERSION.set(Some(13020));
        let _ = CudaMemPool::create(0, &SMALL_POOL);
        assert!(newer.get() > 0, "the counter was never reached");
        seams::DRIVER_VERSION.set(None);
        newer.set(0);
        seams::READ_BACK_SKIPS.set(0);

        let pool = CudaMemPool::create(0, &SMALL_POOL);
        assert_eq!(newer.get(), 0, "requests for attributes above 8");
        assert_eq!(seams::READ_BACK_SKIPS.get(), 1, "gate branch runs");
        let pool = pool.unwrap();
        assert_eq!(pool.max_size(), None);
    }

    /// Design D-4: the driver may report a larger `MAX_POOL_SIZE` than
    /// requested; `max_size()` returns what it reports. A value below a
    /// nonzero request fails `create` with `CUDA_ERROR_INVALID_VALUE`, after
    /// the pool is destroyed.
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    )))]
    #[test]
    fn max_pool_size_read_back_uses_the_driver_value() {
        if !have_device() {
            return;
        }
        let _reset = seams::Reset::new();
        seams::DRIVER_VERSION.set(Some(13020));
        let request = SMALL_POOL.max_size as u64;

        for reported in [request, request + (2 << 20)] {
            seams::READ_BACK_VALUE.set(Some(reported));
            let pool = CudaMemPool::create(0, &SMALL_POOL).unwrap();
            assert_eq!(pool.max_size(), Some(reported));
        }
        // Positive control: each of the two pools was destroyed and counted.
        assert_eq!(seams::POOL_DESTROYS.get(), 2);
        seams::POOL_DESTROYS.set(0);
        seams::POOL_CREATES.set(0);

        seams::READ_BACK_VALUE.set(Some(request - 1));
        assert_eq!(
            CudaMemPool::create(0, &SMALL_POOL).map(|_| ()),
            Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))
        );
        assert_eq!(seams::POOL_CREATES.get(), 1, "pools created");
        assert_eq!(seams::POOL_DESTROYS.get(), 1, "pools destroyed");
    }

    /// Design D-4: a failed `MAX_POOL_SIZE` read leaves `create` successful,
    /// with `max_size()` unknown.
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    )))]
    #[test]
    fn a_failed_max_pool_size_read_leaves_create_successful() {
        if !have_device() {
            return;
        }
        let _reset = seams::Reset::new();
        seams::DRIVER_VERSION.set(Some(13020));
        seams::READ_BACK_ERROR.set(Some(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        let pool = CudaMemPool::create(0, &SMALL_POOL).unwrap();
        assert_eq!(pool.max_size(), None);
        assert_eq!(seams::POOL_DESTROYS.get(), 0);
    }

    /// CF-AC-04c: on a driver of 13020 or later the real read-back is at
    /// least the request. Ignored because driver 580 (13000) cannot run it;
    /// G5b runs it with `--ignored --exact`.
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    )))]
    #[test]
    #[ignore = "needs a driver of 13020 or later (test plan G5b)"]
    fn max_pool_size_reads_back_at_least_the_request() {
        let mut driver = 0;
        unsafe { sys::cuDriverGetVersion(&mut driver) }
            .result()
            .unwrap();
        assert!(
            driver >= 13020,
            "driver {driver} cannot report MAX_POOL_SIZE"
        );
        let pool = CudaMemPool::create(0, &SMALL_POOL).unwrap();
        let reported = pool.max_size().expect("MAX_POOL_SIZE read back");
        assert!(
            reported >= SMALL_POOL.max_size as u64,
            "reported {reported} < requested {}",
            SMALL_POOL.max_size
        );
    }

    /// Design D-4: the typed reads are 64 bits wide. A threshold above
    /// `u32::MAX`, set through the result layer, reads back whole.
    #[test]
    fn typed_reads_return_64_bit_values() {
        if !have_device() {
            return;
        }
        let pool = CudaMemPool::create(0, &SMALL_POOL).unwrap();
        let mut threshold: u64 = 1 << 33;
        unsafe {
            result::mem_pool::set_attribute(
                pool.raw(),
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
        }
        .unwrap();
        assert_eq!(pool.release_threshold(), Ok(1 << 33));
        assert_eq!(pool.reserved_current(), Ok(0));
        assert_eq!(pool.reserved_high(), Ok(0));
        assert_eq!(pool.used_current(), Ok(0));
        assert_eq!(pool.used_high(), Ok(0));
    }

    /// Design D-2: `create` owns the pool before setting the release
    /// threshold, so a failure there destroys it.
    #[test]
    fn create_destroys_the_pool_when_setting_the_threshold_fails() {
        if !have_device() {
            return;
        }
        let _reset = seams::Reset::new();
        // Positive control: dropping a created pool is counted.
        drop(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        assert_eq!(seams::POOL_DESTROYS.get(), 1);
        seams::POOL_DESTROYS.set(0);
        seams::POOL_CREATES.set(0);

        let failure = sys::CUresult::CUDA_ERROR_INVALID_VALUE;
        seams::FAIL_SET_THRESHOLD.set(Some(failure));
        assert_eq!(
            CudaMemPool::create(0, &SMALL_POOL).map(|_| ()),
            Err(DriverError(failure))
        );
        assert_eq!(seams::POOL_CREATES.get(), 1, "pools created");
        assert_eq!(seams::POOL_DESTROYS.get(), 1, "pools destroyed");
    }

    /// T-FAULT: `cuMemPoolCreate` failing with out-of-memory is returned as
    /// is, and there is no pool to destroy.
    #[test]
    fn a_pool_create_out_of_memory_returns_the_error_and_no_pool() {
        if !have_device() {
            return;
        }
        let _reset = seams::Reset::new();
        // Positive control: dropping a created pool is counted.
        drop(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        assert_eq!(seams::POOL_DESTROYS.get(), 1);
        seams::POOL_DESTROYS.set(0);

        let oom = sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
        seams::FAIL_POOL_CREATE.set(Some(oom));
        assert_eq!(
            CudaMemPool::create(0, &SMALL_POOL).map(|_| ()),
            Err(DriverError(oom))
        );
        assert_eq!(seams::POOL_DESTROYS.get(), 0, "pools destroyed");
    }

    /// CF-AC-03: a private-pool pointer leaked and upgraded onto an ordinary
    /// context of the same device is freed back into its pool.
    #[test]
    fn a_private_pointer_upgraded_onto_a_default_context_returns_to_its_pool() {
        if !have_device() {
            return;
        }
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let private = CudaContext::new_with_mem_pool(0, pool.clone()).unwrap();
        let private_stream = private.default_stream();
        let slice = private_stream.clone_htod(&[7u32; 4096]).unwrap();
        private_stream.synchronize().unwrap();
        let mut leaked = Leaked::new(&private);
        leaked.push(slice.leak());
        assert!(pool.used_current().unwrap() >= 4 * 4096);

        let ordinary = CudaContext::new(0).unwrap();
        let stream = ordinary.default_stream();
        let ptr = leaked.pop().unwrap();
        let upgraded = unsafe { stream.upgrade_device_ptr::<u32>(ptr, 4096) };
        assert_eq!(stream.clone_dtoh(&upgraded).unwrap(), std::vec![7u32; 4096]);
        drop(upgraded);
        stream.synchronize().unwrap();
        private_stream.synchronize().unwrap();

        assert_eq!(pool.used_current(), Ok(0));
        ordinary.check_err().unwrap();
        private.check_err().unwrap();
    }

    /// CF-AC-03: private-pool pointers freed with `cuMemFree` after their
    /// accesses completed are physically released: `trim_to(0)` leaves at
    /// most one chunk reserved, and free device memory recovers to within
    /// one chunk. On an integrated device (Tegra) free memory does not move,
    /// so the process's `VmRSS` must rise by two chunks while the pointers
    /// live and return to within one chunk instead (design D-2).
    /// `USED_MEM_CURRENT` is only printed: driver 580 leaves it stale after
    /// `cuMemFree` (design D-2).
    #[test]
    fn a_private_pointer_freed_synchronously_releases_its_memory() {
        if !have_device() {
            return;
        }
        const WORK: usize = 32 << 20;
        let (pool, stream) = private_stream(&MemPoolProps::default());
        let ctx = stream.context().clone();
        let chunk = reservation_step(&pool, &stream, WORK);
        assert_eq!(pool.reserved_current(), Ok(0));
        ctx.bind_to_thread().unwrap();
        let (free_before, _) = result::mem_get_info().unwrap();
        let rss_before = integrated(&ctx).then(vm_rss);

        let mut pointers = Leaked::new(&ctx);
        for _ in 0..3 {
            pointers.push(stream.alloc_zeros::<u8>(WORK).unwrap().leak());
        }
        stream.synchronize().unwrap();
        let reserved_live = pool.reserved_current().unwrap();
        let (free_live, _) = result::mem_get_info().unwrap();
        let rss_live = rss_before.map(|_| vm_rss());
        if let (Some(rss_before), Some(rss_live)) = (rss_before, rss_live) {
            assert!(
                reserved_live > chunk && rss_live >= rss_before + 2 * chunk,
                "precondition (integrated): reserved {reserved_live}, VmRSS {rss_before} -> {rss_live}, chunk {chunk}"
            );
        } else {
            assert!(
                reserved_live > chunk && free_live as u64 + 2 * chunk <= free_before as u64,
                "precondition: reserved {reserved_live}, free {free_before} -> {free_live}, chunk {chunk}"
            );
        }

        ctx.bind_to_thread().unwrap();
        while let Some(ptr) = pointers.pop() {
            unsafe { result::free_sync(ptr) }.unwrap();
        }
        stream.synchronize().unwrap();
        pool.trim_to(0).unwrap();
        let reserved = pool.reserved_current().unwrap();
        let (free_after, _) = result::mem_get_info().unwrap();
        std::println!(
            "USED_MEM_CURRENT after cuMemFree: {:?} (observation; design D-2)",
            pool.used_current()
        );
        if let (Some(rss_before), Some(rss_live)) = (rss_before, rss_live) {
            let rss_after = vm_rss();
            std::println!(
                "integrated: VmRSS {rss_before} -> live {rss_live} -> after {rss_after}; reserved live {reserved_live}, after {reserved}; cuMemGetInfo free {free_before} -> {free_live} -> {free_after} (observation)"
            );
            assert!(
                rss_after <= rss_before + chunk,
                "VmRSS {rss_before} -> {rss_after}, more than one chunk ({chunk}) still resident"
            );
        }
        assert!(reserved <= chunk, "reserved {reserved} > chunk {chunk}");
        if rss_before.is_none() {
            assert!(
                free_after as u64 + chunk >= free_before as u64,
                "free {free_before} -> {free_after}, more than one chunk ({chunk}) short"
            );
        }
        ctx.check_err().unwrap();
    }

    /// CF-AC-03: a `cuMemAlloc` pointer upgraded onto a private context is
    /// freed without error and never touches the private pool.
    #[test]
    fn a_synchronous_pointer_upgraded_onto_a_private_context_frees_cleanly() {
        if !have_device() {
            return;
        }
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let private = CudaContext::new_with_mem_pool(0, pool.clone()).unwrap();
        private.bind_to_thread().unwrap();
        let ptr = unsafe { result::malloc_sync(1 << 20) }.unwrap();

        let stream = private.default_stream();
        let mut upgraded = unsafe { stream.upgrade_device_ptr::<u8>(ptr, 1 << 20) };
        stream.memset_zeros(&mut upgraded).unwrap();
        drop(upgraded);
        stream.synchronize().unwrap();

        private.check_err().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
        assert_eq!(pool.used_high(), Ok(0));
        assert_eq!(pool.reserved_high(), Ok(0));
    }

    /// Design D-8: one helper builds the location from zeroed memory on every
    /// binding, the union's `id` included.
    #[test]
    fn device_location_is_the_ordinal_on_every_binding() {
        assert_eq!(std::mem::size_of::<sys::CUmemLocation>(), 8);
        for ordinal in [0, 1, 3, 255, 256, i32::MAX] {
            let location = device_location(ordinal);
            assert_eq!(
                location.type_,
                sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE
            );
            assert_eq!(location_id(&location), ordinal);
            let bytes = location_bytes(location);
            assert_eq!(
                bytes[..4],
                (sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE as u32).to_ne_bytes()
            );
            assert_eq!(bytes[4..], ordinal.to_ne_bytes());
        }
    }

    #[test]
    fn a_created_pool_carries_its_release_threshold() {
        if !have_device() {
            return;
        }
        let pool = CudaMemPool::create(
            0,
            &MemPoolProps {
                max_size: 64 << 20,
                release_threshold: 1 << 20,
            },
        )
        .unwrap();
        assert_eq!(pool.release_threshold(), Ok(1 << 20));
        assert_eq!(pool.used_current(), Ok(0));
        pool.trim_to(0).unwrap();
    }

    /// C9 at the cudarc level: the context allocates from its own pool and the
    /// device's current pool, which every other context uses, is untouched.
    #[test]
    fn a_private_context_allocates_from_its_pool_and_leaves_the_current_pool_alone() {
        if !have_device() {
            return;
        }
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let cu_device = result::device::get(0).unwrap();
        let current_before = unsafe { result::device::get_mem_pool(cu_device) }.unwrap();

        let ctx = CudaContext::new_with_mem_pool(0, pool.clone()).unwrap();
        assert_eq!(ctx.alloc_mode(), AllocMode::Private);
        assert!(ctx.has_async_alloc());
        assert!(Arc::ptr_eq(ctx.mem_pool().unwrap(), &pool));
        assert_ne!(current_before, pool.raw());

        let stream = ctx.default_stream();
        let slice = stream.alloc_zeros::<u8>(1 << 20).unwrap();
        stream.synchronize().unwrap();
        let used = pool.used_current().unwrap();
        assert!(used >= 1 << 20, "private pool used={used}");
        assert_eq!(stream.clone_dtoh(&slice).unwrap(), std::vec![0u8; 1 << 20]);

        drop(slice);
        stream.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
        // Release threshold 0: freed memory goes back at the synchronization.
        assert_eq!(pool.reserved_current(), Ok(0));
        ctx.check_err().unwrap();
        assert_eq!(
            unsafe { result::device::get_mem_pool(cu_device) },
            Ok(current_before)
        );
    }

    /// C5: a zero-byte request goes to `cuMemAllocFromPoolAsync`, which returns
    /// a null pointer that `cuMemFreeAsync` accepts.
    #[test]
    fn zero_length_private_allocations_are_null_and_free_cleanly() {
        if !have_device() {
            return;
        }
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let ctx = CudaContext::new_with_mem_pool(0, pool.clone()).unwrap();
        let stream = ctx.default_stream();

        let null = stream.null::<f32>().unwrap();
        assert_eq!(null.cu_device_ptr, 0);
        let unset = unsafe { stream.alloc::<f32>(0) }.unwrap();
        assert_eq!(unset.cu_device_ptr, 0);
        let zeros = stream.alloc_zeros::<f32>(0).unwrap();
        assert_eq!(zeros.cu_device_ptr, 0);

        drop((null, unset, zeros));
        stream.synchronize().unwrap();
        ctx.check_err().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
    }

    /// The pool outlives every allocation: slice → stream → context →
    /// `Arc<CudaMemPool>`. Dropping the caller's handle first changes nothing;
    /// the pool is released only after the context, and destroying it then
    /// succeeds.
    #[test]
    fn the_pool_is_destroyed_last_and_cleanly() {
        if !have_device() {
            return;
        }
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let ctx = CudaContext::new_with_mem_pool(0, pool.clone()).unwrap();
        let weak = Arc::downgrade(&pool);
        drop(pool);

        let stream = ctx.new_stream().unwrap();
        let slice = stream.alloc_zeros::<u32>(1024).unwrap();
        drop(ctx);
        assert!(
            weak.upgrade().is_some(),
            "the slice's context keeps the pool"
        );
        let ctx = stream.context().clone();
        drop(slice);
        stream.synchronize().unwrap();
        ctx.check_err().unwrap();
        drop(stream);

        let pool = weak.upgrade().unwrap();
        drop(ctx);
        let pool = Arc::try_unwrap(pool).expect("the context released the pool");
        assert_eq!(pool.used_current(), Ok(0));
        let raw = pool.raw();
        std::mem::forget(pool);
        assert_eq!(unsafe { result::mem_pool::destroy(raw) }, Ok(()));
    }

    #[test]
    fn a_pool_of_another_device_is_rejected() {
        if !have_device() {
            return;
        }
        let other_device = CudaContext::device_count().unwrap();
        let foreign = Arc::new(CudaMemPool {
            cu_device: other_device,
            pool: std::ptr::null_mut(),
            max_size: None,
        });
        assert_eq!(
            CudaContext::new_with_mem_pool(0, foreign.clone()).map(|_| ()),
            Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))
        );
        // The handle is fake; it must not reach cuMemPoolDestroy.
        std::mem::forget(Arc::try_unwrap(foreign).expect("the constructor kept no reference"));
    }

    /// Runs `inner` alone in a child test process, so the process-wide
    /// allocator table starts empty whatever this process already decided.
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    fn in_fresh_process(inner: &str) {
        let exe = std::env::current_exe().unwrap();
        let output = std::process::Command::new(exe)
            .args([
                inner,
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{inner} failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn a_private_context_never_consults_or_records_the_process_decision() {
        if !have_device() {
            return;
        }
        in_fresh_process(
            "driver::safe::core::fathomdb_private_pool::private_context_in_a_fresh_process",
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    #[ignore = "run by a_private_context_never_consults_or_records_the_process_decision"]
    fn private_context_in_a_fresh_process() {
        let cu_device = {
            result::init().unwrap();
            result::device::get(0).unwrap()
        };
        assert_eq!(PROCESS_ALLOC_MODE.decided(cu_device), None);
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let ctx = CudaContext::new_with_mem_pool(0, pool).unwrap();
        let slice = ctx.default_stream().alloc_zeros::<u8>(4096).unwrap();
        drop(slice);
        ctx.synchronize().unwrap();
        assert_eq!(PROCESS_ALLOC_MODE.decided(cu_device), None);
        // A later ordinary context still makes the device's own decision.
        let ordinary = CudaContext::new(0).unwrap();
        assert_eq!(
            PROCESS_ALLOC_MODE.decided(cu_device),
            Some(ordinary.has_async_alloc())
        );
        assert_ne!(ordinary.alloc_mode(), AllocMode::Private);
        assert!(ordinary.mem_pool().is_none());
    }

    #[test]
    fn ordinary_contexts_report_default_or_synchronous() {
        if !have_device() {
            return;
        }
        let ctx = CudaContext::new(0).unwrap();
        assert!(ctx.mem_pool().is_none());
        assert_eq!(
            ctx.alloc_mode(),
            if ctx.has_async_alloc() {
                AllocMode::Default
            } else {
                AllocMode::Synchronous
            }
        );
    }

    /// A pool on device 0 and a new stream of a private context using it.
    fn private_stream(props: &MemPoolProps) -> (Arc<CudaMemPool>, Arc<CudaStream>) {
        let pool = Arc::new(CudaMemPool::create(0, props).unwrap());
        let ctx = CudaContext::new_with_mem_pool(0, pool.clone()).unwrap();
        (pool, ctx.new_stream().unwrap())
    }

    /// Whether `ctx`'s device shares memory with the host (Tegra). There,
    /// `cuMemGetInfo` free memory does not show a release, and the tests use
    /// the pool attributes and [vm_rss] instead (design D-2).
    fn integrated(ctx: &CudaContext) -> bool {
        ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .unwrap()
            != 0
    }

    /// This process's resident set size in bytes, `VmRSS` of
    /// `/proc/self/status`. On Tegra, device memory the process holds is
    /// resident in it.
    fn vm_rss() -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let kib: u64 = status
            .lines()
            .find_map(|line| line.strip_prefix("VmRSS:"))
            .and_then(|value| value.trim().strip_suffix("kB"))
            .map(|kib| kib.trim().parse().unwrap())
            .expect("VmRSS in /proc/self/status");
        kib * 1024
    }

    /// Pointers taken out with `CudaSlice::leak`. Whatever is still held when
    /// the guard drops, also while a failing test unwinds, is freed with
    /// `cuMemFree` after `ctx` synchronizes. A failing test therefore leaves
    /// no pool in deferred destruction, which on Tegra would hold one of the
    /// process's few uncapped-pool address ranges (design D-2).
    struct Leaked {
        ctx: Arc<CudaContext>,
        pointers: Vec<sys::CUdeviceptr>,
    }

    impl Leaked {
        fn new(ctx: &Arc<CudaContext>) -> Self {
            Self {
                ctx: ctx.clone(),
                pointers: Vec::new(),
            }
        }

        fn push(&mut self, ptr: sys::CUdeviceptr) {
            self.pointers.push(ptr);
        }

        /// Hands a pointer back to the test, which then owns its free.
        fn pop(&mut self) -> Option<sys::CUdeviceptr> {
            self.pointers.pop()
        }
    }

    impl Drop for Leaked {
        fn drop(&mut self) {
            if self.pointers.is_empty() {
                return;
            }
            let _ = self.ctx.synchronize();
            for ptr in self.pointers.drain(..) {
                let _ = unsafe { result::free_sync(ptr) };
            }
        }
    }

    /// What [grow_until_failure] saw.
    struct Growth {
        /// The buffers that were allocated, still live.
        live: Vec<CudaSlice<u8>>,
        /// The error of the first failing request, if one failed.
        error: Option<DriverError>,
        /// `RESERVED_MEM_CURRENT` and `USED_MEM_CURRENT` right after the
        /// failure (or after the last request).
        reserved: u64,
        used: u64,
    }

    /// Allocates `size`-byte buffers on `stream` until a request fails or
    /// the requests would pass `cap + size` bytes in total, so no call asks
    /// for more than the cap plus one request (requirements CF-AC-02). The
    /// test plan's "M-2 limit" for `size` is `live.len() * size` when a
    /// request failed.
    fn grow_until_failure(
        pool: &CudaMemPool,
        stream: &Arc<CudaStream>,
        size: usize,
        cap: usize,
    ) -> Growth {
        let mut live = Vec::new();
        let mut error = None;
        for _ in 0..=cap / size {
            match unsafe { stream.alloc::<u8>(size) } {
                Ok(buffer) => live.push(buffer),
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        Growth {
            live,
            error,
            reserved: pool.reserved_current().unwrap(),
            used: pool.used_current().unwrap(),
        }
    }

    /// M-1 (test plan § 8): the step in `RESERVED_MEM_CURRENT` made by the
    /// first and the second allocation of each size in an empty pool.
    #[test]
    #[ignore = "measurement M-1 (test plan § 8); run alone with --nocapture"]
    fn measure_reservation_steps() {
        for size in [1usize, 1 << 20, 32 << 20, 256 << 20] {
            let (pool, stream) = private_stream(&MemPoolProps::default());
            let empty = pool.reserved_current().unwrap();
            let first = unsafe { stream.alloc::<u8>(size) }.unwrap();
            let one = pool.reserved_current().unwrap();
            let second = unsafe { stream.alloc::<u8>(size) }.unwrap();
            let two = pool.reserved_current().unwrap();
            stream.synchronize().unwrap();
            let used = pool.used_current().unwrap();
            let (free, total) = result::mem_get_info().unwrap();
            std::println!(
                "M-1 size={size} empty={empty} first_step={} second_step={} used_after_two={used} device_free={free} device_total={total}",
                one - empty,
                two - one
            );
            drop((first, second));
            stream.synchronize().unwrap();
            assert_eq!(pool.used_current(), Ok(0));
        }
    }

    /// M-2 (test plan § 8): one allocation size grown until the first
    /// failure in a pool capped at `SMALL_POOL.max_size`, at release
    /// thresholds 0 and the cap; reserved and used at the failure.
    #[test]
    #[ignore = "measurement M-2 (test plan § 8); run alone with --nocapture"]
    fn measure_max_size_semantics() {
        let cap = SMALL_POOL.max_size;
        let size = 1 << 20;
        for threshold in [0, cap as u64] {
            let (pool, stream) = private_stream(&MemPoolProps {
                max_size: cap,
                release_threshold: threshold,
            });
            let grown = grow_until_failure(&pool, &stream, size, cap);
            std::println!(
                "M-2 cap={cap} size={size} threshold={threshold} successes={} limit={} error={:?} reserved_at_failure={} used_at_failure={} max_size()={:?}",
                grown.live.len(),
                grown.live.len() * size,
                grown.error,
                grown.reserved,
                grown.used,
                pool.max_size()
            );
            drop(grown.live);
            stream.synchronize().unwrap();
        }
    }

    /// M-3 (test plan § 8): what driver 580 answers for the attributes that
    /// first appear in the 13.2 header (9-14) and the 13.4 header (15),
    /// asked directly through the result layer. An observation only: the
    /// driver-version gate never relies on it (design D-4).
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    )))]
    #[test]
    #[ignore = "observation M-3 (test plan § 8); run alone with --nocapture"]
    fn observe_new_attributes_on_driver_580() {
        use sys::CUmemPool_attribute::*;
        let pool = CudaMemPool::create(0, &SMALL_POOL).unwrap();
        let mut driver = 0;
        unsafe { sys::cuDriverGetVersion(&mut driver) }
            .result()
            .unwrap();
        #[allow(unused_mut)]
        let mut attributes = std::vec![
            CU_MEMPOOL_ATTR_ALLOCATION_TYPE,
            CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES,
            CU_MEMPOOL_ATTR_LOCATION_ID,
            CU_MEMPOOL_ATTR_LOCATION_TYPE,
            CU_MEMPOOL_ATTR_MAX_POOL_SIZE,
            CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED,
        ];
        #[cfg(not(any(feature = "cuda-13020", feature = "cuda-13030")))]
        attributes.push(CU_MEMPOOL_ATTR_LOCALITY_DOMAIN_ID);
        for attr in attributes {
            // Wider than any pool attribute, so no value can overrun it.
            let mut value = [0u64; 4];
            let status = unsafe {
                result::mem_pool::get_attribute(pool.raw(), attr, value.as_mut_ptr().cast())
            };
            std::println!(
                "M-3 driver={driver} attribute={} ({attr:?}) result={status:?} value={:#x}",
                attr as u32,
                value[0]
            );
        }
    }
    /// The step in `RESERVED_MEM_CURRENT` that one allocation of `bytes`
    /// makes in `pool`, which must hold nothing. The allocation is freed and
    /// the stream synchronized before returning (test plan § 3.1,
    /// "In-process chunk").
    fn reservation_step(pool: &CudaMemPool, stream: &Arc<CudaStream>, bytes: usize) -> u64 {
        stream.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0), "the pool must be empty");
        let before = pool.reserved_current().unwrap();
        let buffer = unsafe { stream.alloc::<u8>(bytes) }.unwrap();
        let step = pool.reserved_current().unwrap() - before;
        drop(buffer);
        stream.synchronize().unwrap();
        assert!(step > 0, "an allocation of {bytes} bytes reserved nothing");
        step
    }

    /// Waits until `stream` has no pending work without synchronizing it, so
    /// the pool's release threshold is not applied (CUDA 12.6 `cuda.h`,
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`: it applies at a synchronize).
    fn wait_idle_without_synchronizing(stream: &CudaStream) {
        loop {
            match unsafe { sys::cuStreamQuery(stream.cu_stream()) } {
                sys::CUresult::CUDA_SUCCESS => return,
                sys::CUresult::CUDA_ERROR_NOT_READY => std::thread::yield_now(),
                other => panic!("cuStreamQuery: {other:?}"),
            }
        }
    }

    /// The counters of a pool handle: reserved and used, current and high.
    fn counters(pool: sys::CUmemoryPool) -> [u64; 4] {
        use sys::CUmemPool_attribute::*;
        [
            CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        ]
        .map(|attr| {
            let mut value: u64 = 0;
            unsafe { result::mem_pool::get_attribute(pool, attr, (&mut value as *mut u64).cast()) }
                .unwrap();
            value
        })
    }

    /// The test plan's "M-2 limit" for `size`: measured as M-2 does, in a
    /// pool of its own with `props`, as the total of the requests that
    /// succeed before the first failure, which must be out-of-memory.
    fn m2_limit(props: &MemPoolProps, size: usize) -> usize {
        let (pool, stream) = private_stream(props);
        let grown = grow_until_failure(&pool, &stream, size, props.max_size);
        assert_eq!(
            grown.error,
            Some(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)),
            "the M-2 probe must stop at the cap"
        );
        let limit = grown.live.len() * size;
        drop(grown.live);
        stream.synchronize().unwrap();
        limit
    }

    /// CF-AC-02 isolation: the device's default pool and current pool are
    /// the same handles before the private pool exists, while it has live
    /// allocations (seen from the allocating thread and from another
    /// thread), and after it is gone.
    #[test]
    fn default_and_current_pool_are_untouched_while_live() {
        if !have_device() {
            return;
        }
        result::init().unwrap();
        let cu_device = result::device::get(0).unwrap();
        let handles = move || unsafe {
            (
                result::device::get_default_mem_pool(cu_device).unwrap() as usize,
                result::device::get_mem_pool(cu_device).unwrap() as usize,
            )
        };
        let before = handles();

        let (pool, stream) = private_stream(&SMALL_POOL);
        let live: Vec<_> = (0..4)
            .map(|_| stream.alloc_zeros::<u8>(1 << 20).unwrap())
            .collect();
        stream.synchronize().unwrap();
        assert!(pool.used_current().unwrap() >= 4 << 20, "live allocations");
        assert_ne!(before.0, pool.raw() as usize);
        assert_ne!(before.1, pool.raw() as usize);
        assert_eq!(handles(), before, "allocating thread, while live");
        let ctx = stream.context().clone();
        let other = std::thread::spawn(move || {
            ctx.bind_to_thread().unwrap();
            handles()
        })
        .join()
        .unwrap();
        assert_eq!(other, before, "second thread, while live");

        drop(live);
        stream.synchronize().unwrap();
        drop((stream, pool));
        assert_eq!(handles(), before, "after drop");
    }

    /// CF-AC-02 accounting: after every allocation is dropped and the
    /// context synchronized, the pool reports nothing used and, with release
    /// threshold 0, nothing reserved.
    #[test]
    fn accounting_returns_to_zero_after_drop_and_sync() {
        if !have_device() {
            return;
        }
        let (pool, a) = private_stream(&SMALL_POOL);
        let ctx = a.context().clone();
        let b = ctx.new_stream().unwrap();
        let mut live = Vec::new();
        for (stream, mib) in [(&a, 1), (&b, 3), (&a, 5), (&b, 7), (&a, 2)] {
            live.push(stream.alloc_zeros::<u8>(mib << 20).unwrap());
        }
        ctx.synchronize().unwrap();
        let used = pool.used_current().unwrap();
        assert!(used >= 18 << 20, "used {used}");
        assert!(pool.reserved_current().unwrap() >= used);

        drop(live);
        ctx.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0), "used");
        assert_eq!(pool.reserved_current(), Ok(0), "reserved");
        ctx.check_err().unwrap();
    }

    /// CF-AC-02 threshold: with one allocation live and more reserved than
    /// max(T, used) + one chunk, a synchronize brings reserved down to at
    /// most max(T, used) + one chunk.
    #[test]
    fn release_threshold_bounds_reserved_after_sync() {
        if !have_device() {
            return;
        }
        const WORK: usize = 32 << 20;
        const T: u64 = 16 << 20;
        let (pool, stream) = private_stream(&MemPoolProps {
            max_size: 0,
            release_threshold: T,
        });
        let chunk = reservation_step(&pool, &stream, WORK);

        let mut buffers: Vec<_> = (0..4)
            .map(|_| unsafe { stream.alloc::<u8>(WORK) }.unwrap())
            .collect();
        let kept = buffers.pop().unwrap();
        drop(buffers);
        wait_idle_without_synchronizing(&stream);
        let reserved_before = pool.reserved_current().unwrap();
        let used = pool.used_current().unwrap();
        assert!(
            reserved_before > T.max(used) + chunk,
            "precondition: reserved {reserved_before} > max(T {T}, used {used}) + chunk {chunk}"
        );

        stream.synchronize().unwrap();
        let reserved = pool.reserved_current().unwrap();
        let used = pool.used_current().unwrap();
        assert!(
            reserved <= T.max(used) + chunk,
            "reserved {reserved} > max(T {T}, used {used}) + chunk {chunk}"
        );
        drop(kept);
        stream.synchronize().unwrap();
    }

    /// CF-AC-02 trim: `trim_to(k)` sets a floor. With one allocation live
    /// and more reserved than max(k, used) + one chunk, it leaves
    /// k <= reserved <= max(k, used) + one chunk.
    #[test]
    fn trim_to_keeps_the_floor_and_releases_above_it() {
        if !have_device() {
            return;
        }
        const WORK: usize = 32 << 20;
        const K: u64 = 48 << 20;
        // Keep everything at a synchronize, so only the trim releases.
        let (pool, stream) = private_stream(&MemPoolProps {
            max_size: 0,
            release_threshold: u64::MAX,
        });
        let chunk = reservation_step(&pool, &stream, WORK);

        let mut buffers: Vec<_> = (0..4)
            .map(|_| unsafe { stream.alloc::<u8>(WORK) }.unwrap())
            .collect();
        let kept = buffers.pop().unwrap();
        drop(buffers);
        stream.synchronize().unwrap();
        let reserved_before = pool.reserved_current().unwrap();
        let used = pool.used_current().unwrap();
        assert!(
            reserved_before > K.max(used) + chunk,
            "precondition: reserved {reserved_before} > max(k {K}, used {used}) + chunk {chunk}"
        );

        pool.trim_to(K as usize).unwrap();
        stream.synchronize().unwrap();
        let reserved = pool.reserved_current().unwrap();
        let used = pool.used_current().unwrap();
        assert!(reserved >= K, "reserved {reserved} < k {K}");
        assert!(
            reserved <= K.max(used) + chunk,
            "reserved {reserved} > max(k {K}, used {used}) + chunk {chunk}"
        );
        drop(kept);
        stream.synchronize().unwrap();
        pool.trim_to(0).unwrap();
    }

    /// CF-AC-02 capacity: in a fresh pool, every request up to the M-2
    /// limit succeeds and keeps its data. The limit is at most the cap and
    /// within one chunk of it. On an integrated device (Tegra) only about a
    /// third of the cap is usable (design D-4), so there the limit must be
    /// at least one chunk and leave at least one chunk of the cap unusable,
    /// and it is printed as an M-2 measurement.
    #[test]
    fn requests_within_the_cap_succeed() {
        if !have_device() {
            return;
        }
        const SIZE: usize = 1 << 20;
        let cap = SMALL_POOL.max_size;
        let limit = m2_limit(&SMALL_POOL, SIZE);
        let (pool, stream) = private_stream(&SMALL_POOL);
        let chunk = reservation_step(&pool, &stream, SIZE) as usize;
        assert!(limit <= cap, "limit {limit} > cap {cap}");
        if integrated(stream.context()) {
            std::println!("M-2 integrated: cap={cap} size={SIZE} limit={limit} chunk={chunk}");
            assert!(
                limit >= chunk,
                "limit {limit} is less than one chunk ({chunk}) of the cap {cap}"
            );
            // Measured Orin/JetPack 6 behaviour (design D-4 table, fitted,
            // not documented by NVIDIA): a 64 MiB cap gives a 32 MiB limit.
            // A driver that serves the whole cap here is a finding to
            // record, not a pass.
            assert!(
                limit + chunk <= cap,
                "limit {limit} is within one chunk ({chunk}) of the cap {cap}: the measured Tegra limit (design D-4) no longer holds"
            );
        } else {
            assert!(
                limit + chunk > cap,
                "limit {limit} leaves more than a chunk ({chunk}) of the cap {cap} unusable"
            );
        }

        let buffers: Vec<_> = (0..limit / SIZE)
            .map(|i| {
                stream
                    .clone_htod(&std::vec![i as u8; SIZE])
                    .unwrap_or_else(|e| panic!("request {i} within the limit {limit}: {e:?}"))
            })
            .collect();
        for (i, buffer) in buffers.iter().enumerate() {
            assert!(
                stream
                    .clone_dtoh(buffer)
                    .unwrap()
                    .iter()
                    .all(|&b| b == i as u8),
                "buffer {i} lost its data"
            );
        }
        assert!(pool.used_current().unwrap() >= limit as u64);
        drop(buffers);
        stream.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
    }

    /// CF-AC-02 capacity: with the pool filled to the M-2 limit, the next
    /// request fails with out-of-memory. No more than the cap plus one chunk
    /// is ever requested.
    #[test]
    fn a_request_over_the_cap_is_out_of_memory() {
        if !have_device() {
            return;
        }
        const SIZE: usize = 1 << 20;
        let cap = SMALL_POOL.max_size;
        let limit = m2_limit(&SMALL_POOL, SIZE);
        let (pool, stream) = private_stream(&SMALL_POOL);
        let chunk = reservation_step(&pool, &stream, SIZE) as usize;
        assert!(
            limit + SIZE <= cap + chunk,
            "the requests would pass the cap {cap} plus one chunk {chunk}"
        );

        let buffers: Vec<_> = (0..limit / SIZE)
            .map(|_| unsafe { stream.alloc::<u8>(SIZE) }.unwrap())
            .collect();
        assert_eq!(
            unsafe { stream.alloc::<u8>(SIZE) }.map(|_| ()),
            Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))
        );
        drop(buffers);
        stream.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
        stream.context().check_err().unwrap();
    }

    /// CF-AC-02 capacity under concurrency (T-CONC): two host threads, two
    /// streams each, share one more request than the M-2 limit holds. At
    /// least one request fails with out-of-memory, the successful ones fit
    /// the limit, and each keeps its own data.
    #[test]
    fn exhaustion_under_two_threads_and_four_streams_fails_only_the_excess() {
        if !have_device() {
            return;
        }
        const SIZE: usize = 1 << 20;
        let cap = SMALL_POOL.max_size;
        let limit = m2_limit(&SMALL_POOL, SIZE);
        let (pool, probe) = private_stream(&SMALL_POOL);
        let chunk = reservation_step(&pool, &probe, SIZE) as usize;
        let requests = limit / SIZE + 1;
        assert!(
            requests * SIZE <= cap + chunk,
            "the requests would pass the cap {cap} plus one chunk {chunk}"
        );

        let ctx = probe.context().clone();
        let tickets = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let start = Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let (ctx, tickets, start) = (ctx.clone(), tickets.clone(), start.clone());
                std::thread::spawn(move || {
                    ctx.bind_to_thread().unwrap();
                    let streams = [ctx.new_stream().unwrap(), ctx.new_stream().unwrap()];
                    let mut done = Vec::new();
                    start.wait();
                    loop {
                        let ticket = tickets.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if ticket >= requests {
                            break;
                        }
                        let stream = &streams[ticket % 2];
                        done.push((ticket, stream.clone_htod(&std::vec![ticket as u8; SIZE])));
                    }
                    for (ticket, buffer) in &done {
                        if let Ok(buffer) = buffer {
                            let data = buffer.stream().clone_dtoh(buffer).unwrap();
                            assert!(
                                data.iter().all(|&b| b == *ticket as u8),
                                "buffer {ticket} lost its data"
                            );
                        }
                    }
                    done
                })
            })
            .collect();
        let done: Vec<_> = workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(done.len(), requests);

        let failures: Vec<_> = done.iter().filter_map(|(_, r)| r.as_ref().err()).collect();
        let succeeded = done.iter().filter(|(_, r)| r.is_ok()).count();
        assert!(!failures.is_empty(), "no request failed");
        for failure in &failures {
            assert_eq!(
                **failure,
                DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)
            );
        }
        assert!(
            succeeded * SIZE <= limit,
            "{succeeded} requests of {SIZE} bytes succeeded, above the limit {limit}"
        );
        std::println!(
            "T-CONC: {requests} requests, {succeeded} succeeded, {} failed, limit {limit}",
            failures.len()
        );
        drop(done);
        ctx.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
        ctx.check_err().unwrap();
    }

    /// A host-side gate a stream waits on (`cuLaunchHostFunc`).
    struct Latch {
        open: std::sync::Mutex<bool>,
        opened: std::sync::Condvar,
    }

    impl Latch {
        fn new() -> Self {
            Latch {
                open: std::sync::Mutex::new(false),
                opened: std::sync::Condvar::new(),
            }
        }

        fn open(&self) {
            *self.open.lock().unwrap_or_else(|p| p.into_inner()) = true;
            self.opened.notify_all();
        }
    }

    unsafe extern "C" fn wait_for_latch(latch: *mut core::ffi::c_void) {
        let latch = &*(latch as *const Latch);
        let mut open = latch.open.lock().unwrap_or_else(|p| p.into_inner());
        while !*open {
            open = latch.opened.wait(open).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// CF-AC-02 stream order (T-CONC): stream B is held by a host latch and
    /// then reads buffer X; X is freed on stream A; stream C allocates the
    /// same size and writes another pattern. Event tracking makes A's free
    /// wait for B's read, so B must still see X's pattern, whether or not C
    /// was given X's address.
    #[test]
    fn a_buffer_freed_on_one_stream_is_not_reused_before_its_reader_finishes() {
        if !have_device() {
            return;
        }
        const LEN: usize = 1 << 20;
        let (pool, a) = private_stream(&MemPoolProps::default());
        let ctx = a.context().clone();
        assert!(ctx.is_event_tracking());
        let b = ctx.new_stream().unwrap();
        let c = ctx.new_stream().unwrap();
        let x = a.clone_htod(&std::vec![0xABu8; LEN]).unwrap();
        a.synchronize().unwrap();
        let x_address = x.cu_device_ptr;
        let mut seen = unsafe { ctx.alloc_pinned::<u8>(LEN) }.unwrap();

        let latch = Arc::new(Latch::new());
        // If an enqueue below blocked the host, the latch would never open;
        // the watchdog opens it and the test fails instead of hanging.
        let watchdog_fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watchdog = {
            let (latch, fired, released) =
                (latch.clone(), watchdog_fired.clone(), released.clone());
            std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                while !released.load(std::sync::atomic::Ordering::SeqCst) {
                    if std::time::Instant::now() > deadline {
                        fired.store(true, std::sync::atomic::Ordering::SeqCst);
                        latch.open();
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            })
        };
        unsafe {
            result::stream::launch_host_function(
                b.cu_stream(),
                wait_for_latch,
                Arc::as_ptr(&latch) as *mut core::ffi::c_void,
            )
        }
        .unwrap();
        b.memcpy_dtoh(&x, &mut seen).unwrap();
        drop(x);
        let mut y = unsafe { c.alloc::<u8>(LEN) }.unwrap();
        {
            let (y_address, _record) = y.device_ptr_mut(&c);
            unsafe { result::memset_d8_async(y_address, 0xCD, LEN, c.cu_stream()) }.unwrap();
        }
        let reused = y.cu_device_ptr == x_address;
        assert_eq!(
            unsafe { sys::cuStreamQuery(b.cu_stream()) },
            sys::CUresult::CUDA_ERROR_NOT_READY,
            "B must still be held by the latch"
        );
        latch.open();
        released.store(true, std::sync::atomic::Ordering::SeqCst);
        watchdog.join().unwrap();
        assert!(
            !watchdog_fired.load(std::sync::atomic::Ordering::SeqCst),
            "an enqueue blocked the host while the latch was closed"
        );

        let read = seen.as_slice().unwrap();
        let wrong = read.iter().filter(|&&v| v != 0xAB).count();
        std::println!("stream order: C's buffer reuses X's address: {reused}");
        assert_eq!(wrong, 0, "B read {wrong} bytes that were not X's pattern");
        assert!(c.clone_dtoh(&y).unwrap().iter().all(|&v| v == 0xCD));
        drop(y);
        ctx.synchronize().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
        ctx.check_err().unwrap();
    }

    /// CF-AC-02 lifetime: with the context, stream and pool handles dropped,
    /// a live slice alone keeps the pool, whose counters still show the
    /// allocation; after the slice is dropped and the stream synchronized,
    /// they read 0, and the pool goes with the last handle.
    #[test]
    fn the_pool_outlives_its_context_while_slices_live() {
        if !have_device() {
            return;
        }
        let pool = Arc::new(CudaMemPool::create(0, &SMALL_POOL).unwrap());
        let raw = pool.raw();
        let weak = Arc::downgrade(&pool);
        let ctx = CudaContext::new_with_mem_pool(0, pool).unwrap();
        let stream = ctx.new_stream().unwrap();
        let slice = stream.alloc_zeros::<u8>(1 << 20).unwrap();
        stream.synchronize().unwrap();
        drop((stream, ctx));
        assert_eq!(
            weak.strong_count(),
            1,
            "only the slice's context holds the pool"
        );
        // Read through the raw handle, so no reference of the test keeps it.
        let used = counters(raw)[2];
        assert!(used >= 1 << 20, "used {used} while only the slice lives");

        let stream = slice.stream().clone();
        drop(slice);
        stream.synchronize().unwrap();
        let pool = weak.upgrade().unwrap();
        assert_eq!(pool.used_current(), Ok(0));
        stream.context().check_err().unwrap();
        drop((pool, stream));
        assert_eq!(weak.strong_count(), 0, "the pool outlived its last handle");
    }

    /// CF-AC-02, design D-6: exactly these are `Send + Sync`, checked at
    /// compile time.
    #[test]
    fn mem_pool_and_private_context_are_send_and_sync() {
        fn send_and_sync<T: Send + Sync>() {}
        send_and_sync::<CudaMemPool>();
        send_and_sync::<Arc<CudaMemPool>>();
        // The type of a context built with `new_with_mem_pool`.
        send_and_sync::<CudaContext>();
        send_and_sync::<Arc<CudaContext>>();
    }

    /// SplitMix64, a small seeded generator, so the property test needs no
    /// new dev-dependency.
    struct SplitMix(u64);

    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// One live buffer of the model test: its slice and its byte pattern.
    struct Held {
        slice: CudaSlice<u8>,
        pattern: u8,
    }

    /// Bytes at the start, middle and end of `held`, read synchronously.
    fn sampled(held: &Held) -> Vec<u8> {
        let len = held.slice.len();
        let stream = held.slice.stream();
        [0, len / 2, len - 16]
            .into_iter()
            .flat_map(|at| stream.clone_dtoh(&held.slice.slice(at..at + 16)).unwrap())
            .collect()
    }

    /// CF-AC-02 property test: 256 seeded cases (seed = case index) of 64
    /// operations each: allocate, free on a random stream, synchronize,
    /// trim. Every case uses one allocation size, so the pool cannot
    /// fragment, and a fresh pool capped at `SMALL_POOL.max_size`. The model
    /// predicts:
    /// - an allocation succeeds iff the live total plus its size is within
    ///   the M-2 limit for that size, and fails with out-of-memory otherwise;
    /// - after a synchronize, `USED_MEM_CURRENT` >= the live total;
    /// - every buffer still holds its pattern when it is freed;
    /// - the default pool's counters never change;
    /// - all counters are 0 after the case's drop and synchronize.
    #[test]
    fn pool_model_property() {
        if !have_device() {
            return;
        }
        const CASES: u64 = 256;
        const OPS: usize = 64;
        const SIZES: [usize; 3] = [1 << 20, 4 << 20, 32 << 20];
        let limits = SIZES.map(|size| m2_limit(&SMALL_POOL, size));
        result::init().unwrap();
        let cu_device = result::device::get(0).unwrap();
        let default_pool = unsafe { result::device::get_default_mem_pool(cu_device) }.unwrap();
        let oom = DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY);
        // Totals over all cases: allocations, predicted out-of-memory
        // failures, frees, frees on another stream, synchronizes, trims.
        let mut seen = [0usize; 6];

        for case in 0..CASES {
            let mut rng = SplitMix(case);
            let pick = rng.below(SIZES.len());
            let (size, limit) = (SIZES[pick], limits[pick]);
            let (pool, first) = private_stream(&SMALL_POOL);
            let ctx = first.context().clone();
            let streams = [first, ctx.new_stream().unwrap(), ctx.new_stream().unwrap()];
            let default_before = counters(default_pool);
            let mut live: Vec<Held> = Vec::new();

            for op in 0..OPS {
                let at = std::format!("case {case} (seed {case}) op {op}, size {size}");
                let live_total = live.len() * size;
                match rng.below(10) {
                    0..=3 => {
                        let stream = &streams[rng.below(streams.len())];
                        let fits = live_total + size <= limit;
                        match unsafe { stream.alloc::<u8>(size) } {
                            Ok(mut slice) => {
                                assert!(fits, "{at}: succeeded over the limit {limit}");
                                seen[0] += 1;
                                let pattern = (seen[0] % 251) as u8 + 1;
                                let (ptr, record) = slice.device_ptr_mut(stream);
                                unsafe {
                                    result::memset_d8_async(ptr, pattern, size, stream.cu_stream())
                                }
                                .unwrap();
                                drop(record);
                                live.push(Held { slice, pattern });
                            }
                            Err(e) => {
                                assert!(!fits, "{at}: failed within the limit {limit}: {e:?}");
                                assert_eq!(e, oom, "{at}");
                                seen[1] += 1;
                            }
                        }
                    }
                    4..=6 if !live.is_empty() => {
                        let held = live.swap_remove(rng.below(live.len()));
                        assert!(
                            sampled(&held).iter().all(|&b| b == held.pattern),
                            "{at}: a buffer lost its pattern {}",
                            held.pattern
                        );
                        seen[2] += 1;
                        let target = &streams[rng.below(streams.len())];
                        if Arc::ptr_eq(target, held.slice.stream()) {
                            drop(held);
                        } else {
                            seen[3] += 1;
                            target.join(held.slice.stream()).unwrap();
                            let ptr = held.slice.leak();
                            unsafe { result::free_async(ptr, target.cu_stream()) }.unwrap();
                        }
                    }
                    7 | 8 => {
                        seen[4] += 1;
                        ctx.synchronize().unwrap();
                        let used = pool.used_current().unwrap();
                        assert!(
                            used >= live_total as u64,
                            "{at}: used {used} below the live total {live_total}"
                        );
                    }
                    _ => {
                        seen[5] += 1;
                        pool.trim_to(rng.below(limit + 1)).unwrap();
                    }
                }
                assert_eq!(counters(default_pool), default_before, "{at}: default pool");
            }

            for held in &live {
                assert!(
                    sampled(held).iter().all(|&b| b == held.pattern),
                    "case {case}: a buffer lost its pattern at the end"
                );
            }
            drop(live);
            ctx.synchronize().unwrap();
            let [reserved, _, used, _] = counters(pool.raw());
            assert_eq!(
                (reserved, used),
                (0, 0),
                "case {case}: reserved and used after drop and synchronize"
            );
            ctx.check_err().unwrap();
            assert_eq!(counters(default_pool), default_before, "case {case}");
        }
        std::println!(
            "model: {} allocations, {} out-of-memory as predicted, {} frees ({} on another \
             stream), {} synchronizes, {} trims; limits {limits:?}",
            seen[0],
            seen[1],
            seen[2],
            seen[3],
            seen[4],
            seen[5]
        );
        assert!(seen[1] > 0, "no case reached the limit");
        assert!(seen[3] > 0, "no buffer was freed on another stream");
    }

    /// Design D-2: dropping the last pool handle while a leaked allocation
    /// is outstanding defers the destruction. The memory stays allocated
    /// until the pointer is freed with `cuMemFreeAsync`; then free device
    /// memory returns to less than half the allocation below where it
    /// started (not one chunk: the chunk of a 256 MiB request is 256 MiB,
    /// so that bound could not fail). On an
    /// integrated device (Tegra) free memory does not move, and the pool's
    /// attributes cannot be read once its handle is gone, so the process's
    /// `VmRSS` must stay at least half the allocation up while it is
    /// outstanding and fall back to less than half the allocation above its
    /// start after the free (design D-2).
    #[test]
    fn destroy_with_outstanding_allocations_is_deferred() {
        if !have_device() {
            return;
        }
        const SIZE: usize = 256 << 20;
        let ordinary = CudaContext::new(0).unwrap();
        let other = ordinary.new_stream().unwrap();
        let chunk = {
            let (pool, stream) = private_stream(&MemPoolProps::default());
            reservation_step(&pool, &stream, SIZE) as usize
        };
        ordinary.bind_to_thread().unwrap();
        let (free_before, _) = result::mem_get_info().unwrap();
        let rss_before = integrated(&ordinary).then(vm_rss);

        let (pool, stream) = private_stream(&MemPoolProps::default());
        let mut leaked = Leaked::new(&ordinary);
        leaked.push(stream.alloc_zeros::<u8>(SIZE).unwrap().leak());
        stream.synchronize().unwrap();
        drop(stream);
        assert_eq!(Arc::strong_count(&pool), 1);
        drop(pool);

        ordinary.bind_to_thread().unwrap();
        let (free_outstanding, _) = result::mem_get_info().unwrap();
        let rss_outstanding = rss_before.map(|_| vm_rss());
        if let (Some(rss_before), Some(rss_outstanding)) = (rss_before, rss_outstanding) {
            assert!(
                rss_outstanding >= rss_before + SIZE as u64 / 2,
                "the outstanding allocation was released with the pool: VmRSS {rss_before} -> {rss_outstanding}"
            );
        } else {
            assert!(
                free_outstanding + SIZE / 2 <= free_before,
                "the outstanding allocation was released with the pool: free {free_before} -> {free_outstanding}"
            );
        }
        let ptr = leaked.pop().unwrap();
        unsafe { result::free_async(ptr, other.cu_stream()) }.unwrap();
        other.synchronize().unwrap();
        let (free_after, _) = result::mem_get_info().unwrap();
        if let (Some(rss_before), Some(rss_outstanding)) = (rss_before, rss_outstanding) {
            let rss_after = vm_rss();
            std::println!(
                "integrated: VmRSS {rss_before} -> outstanding {rss_outstanding} -> after {rss_after}; cuMemGetInfo free {free_before} -> {free_outstanding} -> {free_after} (observation)"
            );
            assert!(
                rss_after < rss_before + SIZE as u64 / 2,
                "VmRSS {rss_before} -> {rss_after}: half the allocation ({}) or more is still resident",
                SIZE / 2
            );
        } else {
            std::println!(
                "discrete: cuMemGetInfo free {free_before} -> outstanding {free_outstanding} -> after {free_after}; chunk {chunk} (observation)"
            );
            assert!(
                free_after + SIZE / 2 > free_before,
                "free {free_before} -> {free_after}: half the allocation ({}) or more is still taken",
                SIZE / 2
            );
        }
        ordinary.check_err().unwrap();
    }

    /// Set in the child test process that [run_in_child] starts.
    const CHILD: &str = "FATHOMDB_CUDARC_TEST_CHILD";

    fn in_child() -> bool {
        std::env::var_os(CHILD).is_some()
    }

    /// Runs this module's test `name` alone in a child test process, with
    /// [CHILD] set. Returns its exit status and its stdout then stderr, each
    /// line prefixed with `child| `, so that printing them cannot be taken
    /// for the parent's own libtest lines.
    fn run_in_child(name: &str, ignored: bool) -> (std::process::ExitStatus, String) {
        let full = std::format!("driver::safe::core::fathomdb_private_pool::{name}");
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([full.as_str(), "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1");
        if ignored {
            command.arg("--ignored");
        }
        let output = command.output().unwrap();
        let text = [&output.stdout, &output.stderr]
            .into_iter()
            .flat_map(|stream| {
                String::from_utf8_lossy(stream)
                    .lines()
                    .map(|line| std::format!("child| {line}\n"))
                    .collect::<Vec<_>>()
            })
            .collect();
        (output.status, text)
    }

    /// Design D-2 (T-FAULT, lost context): in a child process, a private
    /// pool with live slices sees a device-side assert, a sticky error; then
    /// the slices, stream, context and pool are dropped. The child neither
    /// panics nor aborts, and exits with status 0.
    #[cfg(feature = "nvrtc")]
    #[test]
    #[ignore = "must be executed by itself"]
    fn drops_after_a_sticky_error_neither_panic_nor_abort() {
        if in_child() {
            use crate::driver::{LaunchConfig, PushKernelArg};
            let (pool, stream) = private_stream(&SMALL_POOL);
            let ctx = stream.context().clone();
            let inp = stream.clone_htod(&[1.0f32; 100]).unwrap();
            let mut out = stream.alloc_zeros::<f32>(100).unwrap();
            let ptx = crate::nvrtc::compile_ptx(
                "
    extern \"C\" __global__ void foo(float *out, const float *inp, const size_t numel) {
        assert(0);
    }",
            )
            .unwrap();
            let module = ctx.load_module(ptx).unwrap();
            let foo = module.load_function("foo").unwrap();
            let mut builder = stream.launch_builder(&foo);
            builder.arg(&mut out);
            builder.arg(&inp);
            builder.arg(&100usize);
            unsafe { builder.launch(LaunchConfig::for_num_elems(100)) }.unwrap();
            std::thread::sleep(std::time::Duration::from_secs(1));
            let sticky = stream
                .synchronize()
                .expect_err("a device-side assert must fail the stream");
            std::println!("CHILD sticky error: {sticky:?}");
            drop((inp, out));
            drop((foo, module));
            drop(stream);
            drop(ctx);
            drop(pool);
            std::println!("CHILD dropped everything");
            return;
        }
        let (status, output) =
            run_in_child("drops_after_a_sticky_error_neither_panic_nor_abort", true);
        let asserts = output.matches("Assertion `0` failed").count();
        let rest: Vec<_> = output
            .lines()
            .filter(|line| !line.contains("Assertion `0` failed"))
            .collect();
        std::println!(
            "child exited with {status}; {asserts} device-side assert lines; the rest:\n{}",
            rest.join("\n")
        );
        assert!(
            status.success()
                && output.contains("CHILD sticky error")
                && output.contains("CHILD dropped everything")
                && output.contains("1 passed")
                && !output.contains("panicked"),
            "child exited with {status}:\n{output}"
        );
    }

    /// Design D-9 (T-CAPTURE): an allocation and a free captured on a stream
    /// of a private context become graph allocation nodes. The graph
    /// instantiates and launches, graph memory (`cuDeviceGetGraphMemAttribute`,
    /// `USED_MEM_CURRENT`) rises, and neither the private pool's counters
    /// nor the default pool's move.
    #[test]
    fn capture_on_a_private_context_uses_graph_memory() {
        if !have_device() {
            return;
        }
        let (pool, stream) = private_stream(&SMALL_POOL);
        let ctx = stream.context().clone();
        // All accesses use one stream; no waits on events recorded outside
        // the capture (as upstream's graph tests).
        unsafe { ctx.disable_event_tracking() };
        let cu_device = ctx.cu_device;
        let default_pool = unsafe { result::device::get_default_mem_pool(cu_device) }.unwrap();
        let graph_used = || {
            let mut value: u64 = 0;
            unsafe {
                sys::cuDeviceGetGraphMemAttribute(
                    cu_device,
                    sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                    (&mut value as *mut u64).cast(),
                )
            }
            .result()
            .unwrap();
            value
        };
        let private_before = counters(pool.raw());
        let default_before = counters(default_pool);
        let graph_before = graph_used();

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        let buffer = stream.alloc_zeros::<u8>(4 << 20).unwrap();
        drop(buffer);
        let graph = stream
            .end_capture(sys::CUgraphInstantiate_flags(0))
            .unwrap()
            .expect("a captured graph");
        graph.launch().unwrap();
        stream.synchronize().unwrap();
        let graph_after = graph_used();

        std::println!("T-CAPTURE graph USED_MEM_CURRENT {graph_before} -> {graph_after}");
        assert!(
            graph_after > graph_before,
            "graph memory did not rise: {graph_before} -> {graph_after}"
        );
        assert_eq!(counters(pool.raw()), private_before, "private pool");
        assert_eq!(counters(default_pool), default_before, "default pool");
        drop(graph);
        stream.synchronize().unwrap();
        ctx.check_err().unwrap();
    }

    /// Design D-13 (T-MULTI): pools on devices 0 and 1 are independent;
    /// neither one's counters move when the other allocates.
    #[test]
    #[ignore = "must be executed with multiple gpus"]
    fn two_pools_on_two_devices_are_independent() {
        assert!(
            CudaContext::device_count().unwrap() >= 2,
            "needs two devices"
        );
        let streams = [0, 1].map(|ordinal| {
            let pool = Arc::new(CudaMemPool::create(ordinal, &SMALL_POOL).unwrap());
            let ctx = CudaContext::new_with_mem_pool(ordinal, pool.clone()).unwrap();
            (pool, ctx.new_stream().unwrap())
        });
        for (mine, other) in [(0, 1), (1, 0)] {
            let (pool, stream) = &streams[mine];
            let other_pool = &streams[other].0;
            let other_before = counters(other_pool.raw());
            let buffer = stream.alloc_zeros::<u8>(8 << 20).unwrap();
            stream.synchronize().unwrap();
            assert!(pool.used_current().unwrap() >= 8 << 20, "device {mine}");
            assert_eq!(
                counters(other_pool.raw()),
                other_before,
                "device {other}'s pool moved when device {mine} allocated"
            );
            drop(buffer);
            stream.synchronize().unwrap();
            assert_eq!(pool.used_current(), Ok(0));
        }
    }

    /// Design D-13 (T-MULTI): a real pool of device 1 is rejected for a
    /// context of device 0, and accepted for device 1.
    #[test]
    #[ignore = "must be executed with multiple gpus"]
    fn a_pool_of_device_1_is_rejected_for_device_0() {
        assert!(
            CudaContext::device_count().unwrap() >= 2,
            "needs two devices"
        );
        let pool = Arc::new(CudaMemPool::create(1, &SMALL_POOL).unwrap());
        assert_eq!(
            CudaContext::new_with_mem_pool(0, pool.clone()).map(|_| ()),
            Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))
        );
        assert_eq!(Arc::strong_count(&pool), 1, "the rejected pool was kept");
        let ctx = CudaContext::new_with_mem_pool(1, pool).unwrap();
        assert_eq!(ctx.alloc_mode(), AllocMode::Private);
    }

    /// Design D-2 (T-TEARDOWN): a child process that allocates from a
    /// private pool and exits without dropping anything exits cleanly.
    #[test]
    fn exit_with_a_live_pool_is_clean() {
        if in_child() {
            let (pool, stream) = private_stream(&SMALL_POOL);
            let slice = stream.alloc_zeros::<u8>(1 << 20).unwrap();
            stream.synchronize().unwrap();
            std::println!("CHILD live: used {}", pool.used_current().unwrap());
            let _keep = (pool, stream, slice);
            std::process::exit(0);
        }
        if !have_device() {
            return;
        }
        let (status, output) = run_in_child("exit_with_a_live_pool_is_clean", false);
        std::println!("child exited with {status}:\n{output}");
        assert!(
            status.success() && output.contains("CHILD live: used "),
            "child exited with {status}:\n{output}"
        );
    }
}
// FATHOMDB PATCH END
