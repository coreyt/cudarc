// FATHOMDB PATCH BEGIN: private memory pool primitive (see FATHOMDB-PATCH.md).
//! An explicit device memory pool ([CudaMemPool]) owned by the caller.

use crate::driver::{
    result::{self, DriverError},
    sys,
};

/// Properties of a [CudaMemPool].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemPoolProps {
    /// The pool's `maxSize` in bytes; 0 asks for the driver's default ("a
    /// system dependent value"). A nonzero value needs bindings with the field
    /// (`cuda-12020` and later) and a driver of 12.2 or later;
    /// [CudaMemPool::create] returns `CUDA_ERROR_NOT_SUPPORTED` otherwise,
    /// and never ignores it.
    pub max_size: usize,
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`: bytes of freed memory the pool may
    /// keep at a synchronization point (`u64::MAX` keeps everything).
    pub release_threshold: u64,
}

/// The driver's pool properties for `props`: pinned device memory on device
/// `ordinal`, no export handle types.
pub(crate) fn pool_props(ordinal: i32, props: &MemPoolProps) -> sys::CUmemPoolProps {
    // SAFETY: CUmemPoolProps is a plain C struct whose all-zero value means
    // "unset", and its reserved bytes must be zero.
    let mut raw: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
    raw.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    raw.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    raw.location = device_location(ordinal);
    set_max_size(&mut raw, props.max_size);
    raw
}

/// The location of device `ordinal`, built from zeroed memory so that no
/// struct literal depends on the binding's layout: from `cuda-13020` the
/// `id` field lives in an anonymous union.
pub(crate) fn device_location(ordinal: i32) -> sys::CUmemLocation {
    // SAFETY: CUmemLocation is a plain C struct of integers; all-zero is a
    // valid value that the writes below complete.
    let mut location: sys::CUmemLocation = unsafe { std::mem::zeroed() };
    location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    set_location_id(&mut location, ordinal);
    location
}

// Version gates name the old bindings that lack a field, never the new ones
// that have it, so a future `cuda-*` feature falls on the new side (design
// D-8).
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
fn set_location_id(location: &mut sys::CUmemLocation, ordinal: i32) {
    location.__bindgen_anon_1.id = ordinal;
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
fn set_location_id(location: &mut sys::CUmemLocation, ordinal: i32) {
    location.id = ordinal;
}

#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
)))]
fn set_max_size(raw: &mut sys::CUmemPoolProps, max_size: usize) {
    raw.maxSize = max_size;
}

#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
))]
fn set_max_size(_raw: &mut sys::CUmemPoolProps, _max_size: usize) {}

/// Whether the bindings have `CUmemPoolProps::maxSize` (`cuda-12020` on).
const BINDINGS_HAVE_MAX_SIZE: bool = cfg!(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
)));

/// The loaded driver's version (`cuDriverGetVersion`), for example 13000.
fn driver_version() -> Result<i32, DriverError> {
    #[cfg(test)]
    if let Some(version) = seams::DRIVER_VERSION.get() {
        return Ok(version);
    }
    let mut version = 0;
    unsafe { sys::cuDriverGetVersion(&mut version) }.result()?;
    Ok(version)
}

/// `CU_MEMPOOL_ATTR_MAX_POOL_SIZE` (a `cuuint64_t`, from the 13.2 header).
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
fn read_max_pool_size(pool: sys::CUmemoryPool) -> Result<Option<u64>, DriverError> {
    let mut value: u64 = 0;
    unsafe {
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_MAX_POOL_SIZE,
            (&mut value as *mut u64).cast(),
        )
    }?;
    Ok(Some(value))
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
fn read_max_pool_size(_pool: sys::CUmemoryPool) -> Result<Option<u64>, DriverError> {
    Ok(None)
}

/// A memory pool for one device (`cuMemPoolCreate`), created and owned by the
/// caller.
///
/// The pool is never made the device's current pool, so creating it changes
/// nothing for other users of the device. Allocations draw from it only
/// through a context built with
/// [crate::driver::CudaContext::new_with_mem_pool], which keeps the pool alive
/// for as long as the context and its streams and slices exist.
///
/// # Drop
/// Dropping the last reference destroys the pool (`cuMemPoolDestroy`).
#[derive(Debug)]
pub struct CudaMemPool {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) pool: sys::CUmemoryPool,
    /// `CU_MEMPOOL_ATTR_MAX_POOL_SIZE` as read back at creation.
    pub(crate) max_size: Option<u64>,
}

unsafe impl Send for CudaMemPool {}
unsafe impl Sync for CudaMemPool {}

impl CudaMemPool {
    /// Creates a pool of pinned device memory on device `ordinal` with
    /// `props`. Initializes the driver if needed; needs no current context.
    ///
    /// # Errors
    /// - `CUDA_ERROR_NOT_SUPPORTED` for a nonzero `max_size` when the bindings
    ///   predate `cuda-12020` (before any driver call) or when
    ///   `cuDriverGetVersion` reports a driver older than 12020 (before the
    ///   pool is created);
    /// - the driver error of `cuInit`, `cuDeviceGet`, `cuDriverGetVersion`,
    ///   `cuMemPoolCreate` (for example `CUDA_ERROR_OUT_OF_MEMORY` when the
    ///   process has no room for the pool's address range) or of setting the
    ///   release threshold; on the last, the pool is destroyed before
    ///   returning.
    pub fn create(ordinal: usize, props: &MemPoolProps) -> Result<Self, DriverError> {
        let not_supported = DriverError(sys::CUresult::CUDA_ERROR_NOT_SUPPORTED);
        if props.max_size > 0 && !BINDINGS_HAVE_MAX_SIZE {
            return Err(not_supported);
        }
        #[cfg(test)]
        seams::enter_driver()?;
        result::init()?;
        let cu_device = result::device::get(ordinal as i32)?;
        // A driver before 12.2 reads `maxSize` as part of `reserved`, which
        // must be 0.
        if props.max_size > 0 && driver_version()? < 12020 {
            return Err(not_supported);
        }
        let raw = pool_props(ordinal as i32, props);
        #[cfg(test)]
        seams::POOL_CREATES.set(seams::POOL_CREATES.get() + 1);
        let pool = unsafe { result::mem_pool::create(&raw) }?;
        // Owned from here, so an error below destroys the pool.
        let mut created = Self {
            cu_device,
            pool,
            max_size: None,
        };
        let mut threshold = props.release_threshold;
        unsafe {
            result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
        }?;
        created.max_size = read_max_pool_size(pool)?;
        Ok(created)
    }

    /// The pool's maximum size as the driver reports it
    /// (`CU_MEMPOOL_ATTR_MAX_POOL_SIZE`, read once at creation), or `None`
    /// when the bindings predate `cuda-13020`.
    pub fn max_size(&self) -> Option<u64> {
        self.max_size
    }

    /// Reads a 64-bit pool attribute: the release threshold or one of the
    /// reserved / used memory counters.
    ///
    /// # Errors
    /// The driver error of `cuMemPoolGetAttribute`, including
    /// `CUDA_ERROR_INVALID_VALUE` for an attribute that is not 64 bits wide.
    pub fn attribute(&self, attr: sys::CUmemPool_attribute) -> Result<u64, DriverError> {
        use sys::CUmemPool_attribute::*;
        if !matches!(
            attr,
            CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
                | CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
                | CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
                | CU_MEMPOOL_ATTR_USED_MEM_CURRENT
                | CU_MEMPOOL_ATTR_USED_MEM_HIGH
        ) {
            // The other attributes are 32-bit; the driver would write only
            // half of `value`.
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let mut value: u64 = 0;
        unsafe {
            result::mem_pool::get_attribute(self.pool, attr, (&mut value as *mut u64).cast())
        }?;
        Ok(value)
    }

    /// Releases unused memory until the pool holds at most `keep` bytes
    /// (`cuMemPoolTrimTo`).
    ///
    /// # Errors
    /// The driver error of `cuMemPoolTrimTo`.
    pub fn trim_to(&self, keep: usize) -> Result<(), DriverError> {
        unsafe { result::mem_pool::trim_to(self.pool, keep) }
    }

    /// The raw pool handle. It stays owned by this value.
    pub fn raw(&self) -> sys::CUmemoryPool {
        self.pool
    }
}

/// Test-only seams of [CudaMemPool::create]. Thread-local, so tests that run
/// in parallel do not see each other's settings.
#[cfg(test)]
pub(crate) mod seams {
    use crate::driver::{result::DriverError, sys::CUresult};
    use core::cell::Cell;

    std::thread_local! {
        /// Times `create` reached its first driver call.
        pub(crate) static DRIVER_ENTRIES: Cell<u32> = const { Cell::new(0) };
        /// When set, `create` returns this error at its first driver call,
        /// before calling the driver.
        pub(crate) static FAIL_DRIVER_ENTRY: Cell<Option<CUresult>> = const { Cell::new(None) };
        /// When set, replaces the result of `cuDriverGetVersion`.
        pub(crate) static DRIVER_VERSION: Cell<Option<i32>> = const { Cell::new(None) };
        /// Times `create` called `cuMemPoolCreate`.
        pub(crate) static POOL_CREATES: Cell<u32> = const { Cell::new(0) };
        /// Times `create` skipped the `MAX_POOL_SIZE` read-back because the
        /// driver is older than 13020.
        pub(crate) static READ_BACK_SKIPS: Cell<u32> = const { Cell::new(0) };
    }

    pub(crate) fn enter_driver() -> Result<(), DriverError> {
        DRIVER_ENTRIES.set(DRIVER_ENTRIES.get() + 1);
        match FAIL_DRIVER_ENTRY.get() {
            Some(err) => Err(DriverError(err)),
            None => Ok(()),
        }
    }

    /// Clears every seam now and again when dropped, so a failing test
    /// leaves nothing set for the next test on this thread.
    pub(crate) struct Reset;

    impl Reset {
        pub(crate) fn new() -> Self {
            reset();
            Reset
        }
    }

    impl Drop for Reset {
        fn drop(&mut self) {
            reset();
        }
    }

    fn reset() {
        DRIVER_ENTRIES.set(0);
        FAIL_DRIVER_ENTRY.set(None);
        DRIVER_VERSION.set(None);
        POOL_CREATES.set(0);
        READ_BACK_SKIPS.set(0);
        crate::driver::result::mem_pool::seams::NEWER_ATTRIBUTE_REQUESTS.set(0);
    }
}

impl Drop for CudaMemPool {
    fn drop(&mut self) {
        let _ = unsafe { result::mem_pool::destroy(self.pool) };
    }
}
// FATHOMDB PATCH END
