# cudarc-fathomdb design (FC2)

- Status: FC2 revision 3, 2026-10-12. Revision 1 (`8259c14`) was reviewed by
  R-FC-DES (APPROVE-WITH-FIXES); revision 2 fixes every finding
  (dispositions in [review.md](review.md)). Revision 3 (FC8 G6 fix round,
  findings F-G6-1 and F-G6-2 of `qualification/2026-10-12-G6-b29c41c.md`)
  adds the Tegra measurements: free memory (D-2), the usable fraction of a
  `maxSize` cap (D-4), the per-process budget of uncapped pools (D-2,
  D-9), and an upstream `cuCtxCreate_v4` defect (§ 10). No fork code
  changes for them; the fork's tests change (test plan revision 4).
- Inputs:
  - [requirements](requirements.md) revision 4 (§ 13 lists the changes made
    with this design: revision 3 with design revision 1, revision 4 with
    this revision);
  - [plan](plan.md) § 5 design topics;
  - the vendored delta at FathomDB `third_party/cudarc-0.19.7/`
    (`FATHOMDB-PATCH.md` and its two patches);
  - FathomDB Slice 30 design (`dev/plans/0.8.28/features/slice-30/design.md`
    § 3.1, § 4.5).
- Base: `ec1cac71` (v0.19.10). Upstream `main` equals that tag on 2026-10-12.

## 0. Sources

Primary NVIDIA sources, cited below by tag:

| Tag | Source |
| --- | --- |
| H12.6 | `/usr/local/cuda-12.6/include/cuda.h` (CUDA 12.6 header on `windchill3`) |
| DRV13.4 | CUDA Driver API 13.4, Stream Ordered Memory Allocator: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html> |
| TYPES13.4 | CUDA Driver API 13.4, data types: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html> |
| PG13.4 | CUDA Programming Guide 13.4.2, stream-ordered memory allocation: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html> |
| MVC | CUDA minor-version compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html> |
| RN13.0 | CUDA 13.0 release notes: <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html> |
| PROPS12.2 | `CUmemPoolProps` 12.2.1: <https://docs.nvidia.com/cuda/archive/12.2.1/cuda-driver-api/structCUmemPoolProps__v1.html> (12.1.1 has no `maxSize`) |
| PATHFINDER | NVIDIA cuda-pathfinder library catalog: <https://github.com/NVIDIA/cuda-python/blob/main/cuda_pathfinder/cuda/pathfinder/_dynamic_libs/descriptor_catalog.py> |

Local measurements (2026-10-12, read-only):

- `nm -D` of `libcuda.so.580.178.04`;
- `objdump -p` sonames;
- version queries called via ctypes, without creating a CUDA context, on
  the CUDA 12.6 apt libraries, the cu12 12.8 wheels and the cu13 wheels;
- `size_of`/`offset_of` of the bindings for every `cuda-*` feature.

Upstream context: issue #536 (asks for safe `cuMemPool*` wrappers, open);
PR #558 (a safe `CudaMemPool`, closed unmerged); PR #594 and PR #595 (open);
issue #603 (dynamic loading picks the unversioned system `libcublas.so` over
a loaded `libcublas.so.13`, open).

## 1. D-1 API shape

The pool lives at context level, not slice level. `CudaSlice` gains no
field, because the maintainer resisted new `CudaSlice` ownership variants
in #594. Names follow upstream conventions and the #536/#558 precedent.

```rust
// src/driver/safe/mem_pool.rs
pub struct MemPoolProps {        // Default: max_size 0, release_threshold 0
    pub max_size: usize,         // 0 = driver default (H12.6, PROPS12.2)
    pub release_threshold: u64,  // bytes kept at synchronize
}
pub struct CudaMemPool { /* cu_device, pool: CUmemoryPool */ }
impl CudaMemPool {
    pub fn create(ordinal: usize, props: &MemPoolProps) -> Result<Self, DriverError>;
    pub fn raw(&self) -> sys::CUmemoryPool;
    pub fn reserved_current(&self) -> Result<u64, DriverError>;
    pub fn reserved_high(&self) -> Result<u64, DriverError>;
    pub fn used_current(&self) -> Result<u64, DriverError>;
    pub fn used_high(&self) -> Result<u64, DriverError>;
    pub fn release_threshold(&self) -> Result<u64, DriverError>;
    pub fn max_size(&self) -> Option<u64>; // D-4
    pub fn trim_to(&self, min_bytes_to_keep: usize) -> Result<(), DriverError>;
}
// src/driver/safe/core.rs
pub enum AllocMode { Default, Synchronous, Private }
impl CudaContext {
    pub fn new_with_mem_pool(ordinal: usize, pool: Arc<CudaMemPool>)
        -> Result<Arc<Self>, DriverError>;
    pub fn alloc_mode(&self) -> AllocMode;
    pub fn mem_pool(&self) -> Option<&Arc<CudaMemPool>>;
}
```

**Changes from the vendored API (tree A), each an FC5 commit:**

| Change | Class | Serves |
| --- | --- | --- |
| typed reads replace `attribute(CUmemPool_attribute)` | `U` | CF-REQ-04 (no raw attribute values; 64-bit reads; no exhaustive match) |
| `max_size()` read-back | `U` | CF-REQ-04 / D-4 |
| `create` returns `NOT_SUPPORTED` for an unsupported nonzero `max_size` | `U` | CF-REQ-04 |
| `upgrade_device_ptr` `# Safety` text | `U` | CF-REQ-03 |

Nothing else is added. FathomDB reads pool counters through
`result::mem_pool::get_attribute` (`cuda_pool_policy.rs:1369-1384`), not
through `attribute()`, so removing it does not affect FathomDB. A setter for
the release threshold and a high-water reset are not added. No requirement
needs them, and `result::mem_pool::set_attribute` already serves callers.

Decisions:

- **Typed reads replace the vendored `attribute(CUmemPool_attribute)`.**
  Callers never pass a raw enum, so:
  - no exhaustive match on `CUmemPool_attribute` is needed (D-8);
  - every value is read into a 64-bit integer (D-4);
  - an attribute the loaded driver does not know is never requested.
- **`AllocMode` stays.** FathomDB's `doctor cuda-allocator` reports it
  (Slice 30 § 4.5). For the proposal (FC9) it is optional surface, and R-PROP
  decides whether to keep it.
- **`new_with_mem_pool` retains the primary context,** like
  `CudaContext::new`, and returns a new `Arc<CudaContext>`. It rejects a pool
  of another device with `CUDA_ERROR_INVALID_VALUE`, as the vendored code
  does. It needs no `cuCtxCreate_v4` (D-8).
- **IPC, managed pools, multi-location access and `cuMemPoolSetAccess`**
  are not exposed (D-13).
- **The marker feature `fathomdb-private-pool`** (item 6) is a no-op feature,
  class `L`.

## 2. D-2 Lifetime, destruction and leaked pointers

- **Ownership chain:** `CudaSlice` → `Arc<CudaStream>` → `Arc<CudaContext>` →
  `Option<Arc<CudaMemPool>>`. A slice therefore keeps the pool alive by
  construction.
- **Drop:** `CudaMemPool::drop` calls `cuMemPoolDestroy` and ignores its
  result, as upstream drops ignore driver results. If allocations or frees
  are still outstanding, the call returns at once and the pool is released
  "automatically once there are no more outstanding allocations" (DRV13.4,
  `cuMemPoolDestroy`). A pool that a `static` holds is never dropped, and the
  OS reclaims it at exit. T-TEARDOWN covers exit with pools alive.
- **Creation failure:** `create` builds the owning value right after
  `cuMemPoolCreate` succeeds. A failure from any later `set_attribute`
  therefore drops it and destroys the pool. T-FAULT covers this with a
  `cfg(test)` seam.

**Leaked pointers (CF-REQ-03).** NVIDIA allows freeing across the two
allocator families:

- `cuMemFree` accepts pointers from `cuMemAllocAsync` and
  `cuMemAllocFromPoolAsync`, without implicit synchronization: "Callers must
  ensure that all accesses to these pointer have completed before invoking
  ::cuMemFree" (H12.6, `cuMemFree`).
- PG13.4: "Memory allocated with cudaMalloc() can be freed with with
  cudaFreeAsync()" (NVIDIA's typo), and memory from `cudaMallocAsync` "can
  be freed with cudaFree()".

`cuMemFreeAsync` returns pool memory to whichever pool owns it. cudarc's
synchronous drop synchronizes the slice's stream before `cuMemFree`
(`core.rs:856-875`). So a private-pool pointer upgraded onto any context is
freed correctly, provided every access is ordered before the free.

Revision 1 of this design assumed a cross-family hazard, and so does the
rationale of vendored item 2 (`FATHOMDB-PATCH.md`). NVIDIA's text above
removes that premise. Item 2's per-device table stays as carried: it keeps
all wrappers of one device on one allocator, which is consistent and
measured on the Orin. `FATHOMDB.md` states the corrected rationale.

Design:

- **No new check.** `upgrade_device_ptr` keeps its behaviour.
- **The `# Safety` list** states the real preconditions, citing H12.6 and
  PG13.4:
  - the pointer belongs to this stream's device;
  - it is the base of an allocation of at least `len * size_of::<T>()`
    bytes, or 0 with `len == 0`;
  - every access by other streams or by host code is ordered before the
    slice's free;
  - it may come from either allocator family.
- **CF-AC-03** is a doc change with characterization tests. These tests are
  not RED, because no behaviour changes. On G1:
  - a private-pool pointer is leaked and upgraded onto a `CudaContext::new`
    context, and after drop and a synchronize the private pool's
    `USED_MEM_CURRENT` is 0;
  - a private-pool pointer is leaked and freed with `result::free_sync`
    after a synchronize. **Measured at FC5 (driver 580.178.04, RTX 3090):**
    `cuMemFree` of a pool allocation releases the memory (reserved bytes
    fall and `cuMemGetInfo` free memory rises), but the pool's
    `USED_MEM_CURRENT` is never decremented, for private and default pools
    alike, even after later alloc/free cycles. The test therefore asserts
    physical release: `trim_to(0)` brings reserved to at most one chunk,
    and free memory recovers to within one chunk. It records
    `USED_MEM_CURRENT` as an observation, not an assertion. The `# Safety`
    text states the caveat and advises `cuMemFreeAsync` for pool memory,
    because a stale `USED_MEM_CURRENT` misleads any consumer that reads it.
    FathomDB's `doctor cuda-allocator` is one such consumer.
  - a `result::malloc_sync` pointer is upgraded onto a private context and
    dropped, with no error and `check_err()` clean.

**Tegra: free memory is not a release signal (revision 3).** Measured on
the Jetson AGX Orin (JetPack 6, L4T R36.5.2, CUDA 12.6) during FC8:

- `cuMemGetInfo` free memory equals `/proc/meminfo` `MemAvailable`. Once
  the nvmap page pools are warm (`enable_page_pools=Y`), it does not move
  when pool memory is reserved or released: 96 MiB reserved and released
  left it at `52810706944 -> 52810706944`. Whether it moves in a fresh
  process depends on the page pools, so it is not a usable oracle either
  way. Upstream's `test_post_alloc_memory` fails on the Orin at
  `ec1cac71` for this reason.
- The pool attributes and the process's `VmRSS` are exact: 3 × 32 MiB
  gave `RESERVED_MEM_CURRENT` 100663296 and `VmRSS` +104 MiB, and after a
  synchronize at release threshold 0, reserved 0 and `VmRSS` back.

So on integrated devices (`CU_DEVICE_ATTRIBUTE_INTEGRATED` = 1) the tests
use the pool attributes and `VmRSS` to show release; discrete devices keep
`cuMemGetInfo` (test plan § 3.1). Consumers that judge pool memory on
Tegra should read the pool's counters, not `cuMemGetInfo`.

**Tegra: about four uncapped pools per process (revision 3).** On the
Orin, each pool created with `maxSize` 0 reserves about 21 GB of
contiguous GPU virtual address space, and a fresh process can hold about
four; the next uncapped `cuMemPoolCreate` returns
`CUDA_ERROR_OUT_OF_MEMORY`. The holders:

- the device's default pool, which the aarch64 fallback obtains on
  `CudaContext::new` (item 1);
- graph memory, after a stream capture allocates, for the rest of the
  process (D-9);
- every uncapped private pool, including one in deferred destruction
  because an allocation is still outstanding (above): its range stays
  taken until that allocation is freed.

With four uncapped private pools live, `cuDeviceGetDefaultMemPool` fails
and `CudaContext::new` picks `AllocMode::Synchronous`, which is the
fallback's intended behaviour. A fresh process picks `Default`. Capped
pools take less address space. The numbers are measured, not documented
by NVIDIA. The fork's tests free leaked pointers through a guard during
unwinding, so a failing test cannot leave a deferred-destroyed pool that
fails later tests (test plan § 3.1; finding F-G6-1).

## 3. D-3 Stream ordering and reuse

- **Frees are stream-ordered:** `CudaSlice::drop` frees with `cuMemFreeAsync`
  on the slice's stream.
- **Cross-stream use:** cudarc's event tracking makes the freeing stream
  wait on the slice's recorded read and write events before the free, as
  PG13.4 requires ("Events and/or stream synchronizing operations should be
  used"). With event tracking off, or after #591's unmanaged-sync case, the
  caller owns that ordering. This is upstream's existing contract, which the
  fork does not change.
- **Reuse policy:** the three `REUSE_*` attributes keep their driver
  defaults (all enabled; H12.6). Disabling opportunistic reuse adds run-to-run
  variance (PG13.4), and nothing needs it.
- **T-CONC:** N ≥ 2 streams allocate and free with cross-stream events, and
  the CF-AC-02 stream-order clause checks the result.

## 4. D-4 Capacity and exhaustion

- **`maxSize` exists in `CUmemPoolProps` from CUDA 12.2** (PROPS12.2; 12.1.1
  has no field). A nonzero value bounds the pool, and 0 means "a system
  dependent value" (DRV13.4, `cuMemPoolCreate`). NVIDIA does not say whether
  the bound applies to reserved or used bytes, or which error an excess
  returns. The only listed allocation failure is
  `CUDA_ERROR_OUT_OF_MEMORY`. The test plan measures this on G1 (UNCONFIRMED
  until measured).
- **Rule for `create`** (CF-REQ-04):
  - `max_size == 0`: no check.
  - `max_size > 0` with bindings older than 12020: return
    `DriverError(CUDA_ERROR_NOT_SUPPORTED)` and create nothing.
  - `max_size > 0` with 12020+ bindings: read `cuDriverGetVersion` first.
    Below 12020 return `CUDA_ERROR_NOT_SUPPORTED`, because an older driver
    reads those bytes as `reserved`, which must be 0.
  - Driver ≥ 13020 and bindings ≥ 13020: after creation, read
    `CU_MEMPOOL_ATTR_MAX_POOL_SIZE` (13, a `cuuint64_t`). The value "may be
    higher than what was initially passed" (TYPES13.4). If the read fails,
    `create` still succeeds and `max_size()` returns `None`. Only a
    successful read below a nonzero request makes `create` fail
    (`CUDA_ERROR_INVALID_VALUE`, after destroying the pool).
  - `max_size()` returns the read-back value cached at creation, or `None`.
- **The driver-version gate** rests on the headers, from which the bindings
  are generated:
  - Attributes 9–14 first appear in the 13.2 header (bindings 13020). They
    are requested only when the bindings are 13020+ **and**
    `cuDriverGetVersion` ≥ 13020.
  - Attribute 15 first appears in the 13.4 header (bindings 13040). It needs
    13040+ on both sides, and the fork never requests it.
  - The online Driver API pages are not used as evidence. The archived
    13.2.0 page renders as 13.1.1.
  - The gate is proved on G3 by a `cfg(test)` counter (CF-AC-04b), and on
    G5b by a real read-back (CF-AC-04c).
- **64-bit values:** attributes 4–8 and 13 are `cuuint64_t`; the others are
  `int` (H12.6, TYPES13.4). Typed reads use the matching width.
- **Tegra: about a third of the cap is usable (revision 3; measured, not
  documented).** On the Orin (JetPack 6, CUDA 12.6, `cuDriverGetVersion`
  12060), a pool with a nonzero `maxSize` serves about a third of it
  before `CUDA_ERROR_OUT_OF_MEMORY`, with reserved = used at the failure.
  The RTX 3090 on driver 580 serves the whole cap. One pool per process,
  1 MiB requests:

  | Cap (MiB) | Usable on the Orin (MiB) |
  | --- | --- |
  | 32–96 | 32 |
  | 97–192 | 64 |
  | 256 | 96 |
  | 512 | 192 |
  | 1024 | 352 |
  | 4096 | 1376 |

  The values fit `ceil(ceil(cap / 32 MiB) / 3) × 32 MiB`, a fitted
  formula with no NVIDIA source. Attributes 9 and 10 return
  `CUDA_ERROR_INVALID_VALUE` on this driver, so the reason cannot be read
  from the pool. A caller that sizes a pool on Tegra must size the cap at
  about three times the memory it needs, and the fork does not compensate:
  `maxSize` is passed to the driver unchanged. The capacity tests measure
  the limit in-process (M-2, test plan § 8).
- **Concurrent exhaustion:** each failing `cuMemAllocFromPoolAsync` returns
  `CUDA_ERROR_OUT_OF_MEMORY` to its caller and poisons no other stream. The
  fork surfaces the error unchanged, and T-CONC checks it with N ≥ 2 streams.

## 5. D-5 Release threshold and trimming

- **Release threshold:** at a stream, event or context synchronize, "the
  allocator will try to release memory back to the OS" down to the threshold
  (H12.6; PG13.4 says "device synchronize"). The default is 0, meaning all
  unused memory is released at every synchronize. That is the lowest
  footprint but adds re-reservation latency. FathomDB picks its own value
  (FathomDB D-14); the fork's default stays 0, as the driver's is.
- **`trim_to(k)` sets a floor:** "Releases memory back to the OS until the
  pool contains fewer than minBytesToKeep reserved bytes, or there is no
  more memory that the allocator can safely release". If more than k was
  reserved, the pool keeps "at least minBytesToKeep bytes reserved" (DRV13.4,
  `cuMemPoolTrimTo`). Frees whose completion the host has not observed count
  as outstanding.
- **Fragmentation.** The driver "cannot release OS allocations that back
  outstanding asynchronous allocations" (DRV13.4, TrimTo). Live
  allocations spread over several OS chunks therefore keep all those chunks
  reserved.
- **The CF-AC-02 bounds** are tested with the live set limited to one
  allocation. The test frees everything, synchronizes, makes one
  allocation, and then applies the threshold or trim. After a synchronize,
  with the precondition reserved > max(k, used):
  - k ≤ `RESERVED_MEM_CURRENT` ≤ max(k, `USED_MEM_CURRENT`) + one chunk;
  - for the threshold, the same bound with T.
- **Chunk size:** NVIDIA gives none ("may happen at different granularity",
  H12.6). The test plan measures it on G1, G5b and G6 as the step of
  `RESERVED_MEM_CURRENT` for allocations of 1 B, 1 MiB, 32 MiB and 256 MiB,
  and uses the largest step that a single allocation of the test's size
  produces.

## 6. D-6 Thread safety

- **`CudaMemPool` is `Send + Sync`.** It holds an immutable `CUmemoryPool`
  handle and a `CUdevice`. Every method calls a driver pool function that
  takes the handle by value, and the Driver API is thread-safe for distinct
  calls. `set_release_threshold` and `reset_high_watermarks` are
  last-writer-wins driver state, as with any shared pool.
- **`CudaContext` keeps upstream's `Send + Sync`.** The new
  `Option<Arc<CudaMemPool>>` field is immutable after construction.
- A compile-time test asserts exactly these impls (CF-AC-02).

## 7. D-7 Observability

The typed reads of § 1 cover:

- reserved and used, current and high-water;
- the release threshold;
- the maximum size.

`reset_high_watermarks` sets attributes 6 and 8 to 0, the only value the
driver allows (H12.6). FathomDB's `doctor cuda-allocator` and its receipts
read these.

## 8. D-8 CUDA 11.4 to 13.4

**cfg form.** Every version gate the fork adds names the old versions that
lack a feature. It never lists the new versions that have it, so a future
`cuda-13050` falls on the "new" side automatically:

```rust
#[cfg(not(any(feature = "cuda-11040", /* … */ feature = "cuda-13010")))]
//   → CUmemLocation has the union (13020+), and any later version
```

When several `cuda-*` features are enabled together (feature unification;
upstream issue #525), "old" wins in this form, while `build.rs` sets
`CUDA_MAJOR_VERSION` from the newest feature or `CUDARC_CUDA_VERSION`. A
mismatch gives a compile error (a missing field), never a silent wrong
layout. Item 9 takes its major only from `CUDA_MAJOR_VERSION`, the same
source `get_lib_name_candidates` uses.

Upstream's pattern lists the new versions instead
(`src/driver/safe/unified_memory.rs:186-221` names 13020, 13030 and 13040).
That pattern, and the vendored `set_max_size` list ending at 13020, rot as
versions are added. The vendored list silently drops `maxSize` at 13030 and
13040.

**Layouts** (measured; identical byte layout in every version):

- **`CUmemLocation`** is 8 bytes: `type_` at 0 and `id` at 4. From 13020 the
  `id` field is inside an anonymous union, and in 13040 that union also
  holds `localized { deviceId: u8, localityDomainId: u8 }`.
- **`CUmemPoolProps`** is 88 bytes:

  | Bindings | Fields added | `reserved` |
  | --- | --- | --- |
  | 11040–12010 | — | `[64]` |
  | 12020+ | `maxSize` (offset 24) | `[56]` |
  | 12060+ | `usage` (u16, offset 32) | `[54]` |
  | 13040 | `gpuDirectRDMACapable` (u8, offset 34) | `[53]` |

  The byte that 13040 adds lies in what a 13.0 driver reads as `reserved`.
  `zeroed()` fills it with 0, so the layouts are compatible.

**Construction:**

- one helper `fn device_location(ordinal: i32) -> sys::CUmemLocation`,
  starting from `zeroed()` and writing `type_` and `id` (or the union's `id`)
  under the cfg form above;
- `CUmemPoolProps` from `zeroed()` plus field writes, never a struct literal
  (H12.6: `reserved` "must be 0"; #558's review hit the literal problem);
- `maxSize` written under `cfg(not(any(11040…12010)))`.

**Bindings and matches:**

- no derive is relied on that the 13020+ bindings lack (`Hash` and `Eq` on
  `CUmemLocation`);
- no exhaustive match on `CUmemLocationType` or `CUmemPool_attribute`;
- typed reads (§ 1) name only the attributes they need.

**Removed APIs:**

- **Between bindings:** sixteen functions are declared in the 12090 bindings
  and absent from 13000; upstream cfg-switches the three it uses. The fork
  calls none of them.
- **In the vendored patches:** no deprecated function. None of the patches'
  functions carries `__CUDA_DEPRECATED` in H12.6.
- **Still to verify:** the 13.4 Driver API reference status of each fork
  function. FC3 records it, and R-FC-CODE greps for it (CF-AC-04a).

**Context creation:** not needed, because private contexts retain the
primary context.

**Upstream cfg gaps (recorded, not fixed):**

- `new_non_primary`, `new_cig` and the `create_v4` wrapper stop at 13010.
- The `from_raw_context` test helper stops at 12090.

FathomDB needs none of these, and fixing them would grow the delta beyond
items 1–9. The proposals may mention them. Test accounting lists those tests
as "not compiled (upstream cfg)" (CF-REQ-12).

## 9. D-9 Graph capture and PR #594

- **NVIDIA rule:** "During stream capture, this function results in the
  creation of an allocation node. In this case, the allocation is owned by
  the graph instead of the memory pool. The memory pool's properties are
  used to set the node's creation parameters" (H12.6, DRV13.4,
  `cuMemAllocFromPoolAsync`).
- **Consequences:**
  - a private-pool context that allocates on a capturing stream records a
    graph allocation node;
  - `maxSize` and the release threshold do not bound graph-owned memory;
  - a free during capture must name a graph allocation.
- **Decision:** document this on `new_with_mem_pool`. Graph memory is outside
  the pool's budget, and callers who need a bound on captured memory must
  size their graphs. No capture-specific code is added.
- **T-CAPTURE** checks three things, so it cannot pass vacuously:
  - alloc and free inside a capture on a private context instantiate and
    launch;
  - `cuDeviceGetGraphMemAttribute(CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT)` rises
    while the graph's allocation is live, a positive effect;
  - the private pool's `USED_MEM_CURRENT` and the default pool do not change
    from graph allocations.
- **Unconfirmed:** whether an allocation node accepts the parameters of a
  pool with nonzero `maxSize`. If capture fails, the test records the error
  and the design documents the limit. No capture-specific code is added.
- **Tegra risk:** graph memory needs its own virtual-address reservation,
  the same failure class as the default pool on a fragmented Orin. G6 runs
  T-CAPTURE; a failure there is recorded as a known Tegra limit for
  FathomDB.
- **Tegra budget (revision 3, measured at FC8):** T-CAPTURE passes on the
  Orin. After a capture allocates, graph memory holds one of the process's
  about four uncapped-pool address ranges until the process exits (D-2).
  So a Tegra process that captures and also uses the default pool has
  room for about two uncapped private pools.

**PR #594** (open, "capture-scoped CUDA graph memory pool"):

- **Conflict:** merged with items 1–6 in a scratch clone, it conflicts in
  exactly two hunks, the `CudaStream::null` and `CudaStream::alloc` bodies.
  Resolved in the order `route_allocation` → private pool → `has_async_alloc`
  → synchronous, it compiles.
- **Hazard:** its arena uses `malloc_async` on the device's current pool, so
  on a private context it escapes `maxSize`.
- **Status:** the maintainer called #594's shape into question
  (2026-09-22), and its author proposed a smaller owner field instead, with
  real pool support as a separate PR following #536.
- **Effect on FC9:**
  - **Timing:** FC9 must check #594's state when it starts. If an owner-field
    primitive has landed, D-1 is revisited for the proposal only: a pool
    slice could hold its pool through that field.
  - **Coordination with the #536 author** is an owner decision at FC9. It is
    recorded, not acted on.

## 10. D-12 Tegra (fork part)

- **Items 1–4 keep the predicate
  `cfg(all(target_os = "linux", target_arch = "aarch64"))`.**
  - The fallback acts only when `cuDeviceGetDefaultMemPool` returns
    `CUDA_ERROR_OUT_OF_MEMORY` or `CUDA_ERROR_NOT_SUPPORTED`.
  - On Grace, GB10 and other SBSA hosts the query succeeds, so their
    behaviour equals upstream apart from one query per device per process.
  - Narrowing with `CU_DEVICE_ATTRIBUTE_INTEGRATED` (18) would add a second
    runtime decision and still need the probe on Tegra, so it is not done.
- **Not measured:** non-Tegra aarch64 CUDA hosts. This is recorded in
  `FATHOMDB.md`.
- **Tegra P2P:** NVIDIA's Tegra app note says P2P calls are not supported,
  so D-13's multi-device tests are x86 only.
- **Upstream defect on Tegra (revision 3; F-G6-2): `cuCtxCreate_v4` with
  `ctxCreateParams = NULL`.** Upstream's `CudaContext::new_non_primary`
  and the helper of the `from_raw_context` tests (bindings 12050–13010)
  call `result::ctx::create_v4` with a null params pointer, that is
  `cuCtxCreate_v4(&ctx, NULL, flags, dev)`. In the Tegra 12.6 driver this
  segfaults; the same call with a zeroed `CUctxCreateParams` succeeds.
  Five upstream tests die with `SIGSEGV` on the Orin at
  `ec1cac71` and at `C` (test plan § 9). FathomDB never calls these
  constructors, so the fork does not patch them, which keeps the delta
  minimal. It is a candidate upstream issue (not posted, CF-REQ-10).
- **Early `cuInit`:** NVIDIA's Tegra app note and PG13.4 say nothing on
  ordering. The precondition stays FathomDB's (Slice 30 § 3.1): call
  `cuInit` before the address space fragments. Every fork constructor calls
  `result::init()` first, as upstream does.

## 11. D-13 Multi-device

- **One pool per device.** `new_with_mem_pool` rejects a pool of another
  device.
- **Access:** pool allocations "are accessible only from the device where
  the allocations are located", and peer access does not follow
  `cuCtxEnablePeerAccess` (PG13.4). The fork exposes no
  `cuMemPoolSetAccess`, so private-pool memory is single-device by design.
  The doc on `CudaMemPool` says so.
- **T-MULTI (G2, two RTX 3090s):** two pools on devices 0 and 1 are
  independent (their attributes do not interact), and a pool of device 1 is
  rejected for device 0. Upstream's ignored multi-GPU tests run with
  `--ignored --exact` (CF-REQ-12).

## 12. Fork-specific topics

### 12.1 D-F1 What a newer-binding build can test on driver 580

- **Functions:** every driver function the upstream driver tests and the
  fork call (97 symbols, plus `cuDriverGetVersion` and `cuPointerGetAttribute`)
  is exported by `libcuda.so.580.178.04` under its exact load name. Symbols
  load lazily, on first call.
- **Values:** every enum and attribute value the tests pass exists in CUDA
  13.0, except attributes 9–15 and location types 5 and 6, which only the
  gated code (§ 4) can request.
- **Layouts:** compatible (§ 8).
- **Test counts:** these go into `test-plan.md`, which requirements
  revision 4 names for them (CF-AC-04a, CF-AC-04b):
  - the per-test dependency table (functions, values and layouts, each with
    its introducing version);
  - the RUN/UNMEASURED map;
  - the 13.4 deprecation status of each fork function;
  - the per-module counts at `ec1cac71`. Upstream `--lib` totals, ignored
  tests included:

  | Feature set | 12060 | 13000/13010 | 13020–13040 |
  | --- | --- | --- | --- |
  | default | 75 | 73 | 69 |
  | driver-only | 28 | 26 | 23 |

- **G3:** every compiled test is RUN at all six binding versions on driver
  580. Tests missing at 13020+ are "not compiled (upstream cfg)".
- **G5a** (default features, 13.x wheels, driver 580):
  - RUN: the driver, NVRTC, cuBLAS, cuBLASLt and cuRAND tests;
  - UNMEASURED ("PTX JIT of 13.1+ output needs a newer driver";
    MVC: "Applications that compile device code to PTX will not work on
    older drivers"): upstream's `launch` and `unified_memory` tests, because
    they JIT PTX with no target architecture.

  Editing those tests to use CUBIN is forbidden (CF-REQ-12). The fork's own
  NVRTC-dependent tests compile to CUBIN for the device and run in G5a.
- **G5b:** runs all of them.

### 12.2 D-F2 Fork CI

- **Workflow:** `.github/workflows/fathomdb-ci.yaml` (class `W`), triggered
  on push to `fathomdb/**` and `ci/**`, and on `workflow_dispatch`. It
  mirrors upstream's four workflows.
- **aarch64:** the hosted `ubuntu-24.04-arm` runner (free for public
  repositories) runs check (both feature sets) and clippy for 19 features.
- **T-UNIT job:** Linux x86_64 and Windows, per CF-AC-11.
- **Hosted test job:** runs the `fathomdb_` modules with the default lock,
  then again with FathomDB's `Cargo.lock`. That lock is fetched from the
  FathomDB pin commit recorded in the workflow (`FATHOMDB_LOCK_SHA`), from
  the public repository by exact sha, and its SHA-256 is recorded. Cargo
  re-resolves entries that do not match the cudarc root. The job therefore
  asserts, from `cargo metadata`, that every resolved cudarc dependency's
  version equals FathomDB's lock entry, and fails otherwise.
- **CI receipt:** `gh api` export of the run (run id, `head_sha`, each job's
  conclusion) into `fathomdb/qualification/`.
- **Baseline lock (CF-REQ-12):** FC4 generates one `Cargo.lock` at
  `ec1cac71` and commits it as `fathomdb/qualification/Cargo.lock.baseline`
  (class `D`). Every baseline and `C` run copies it into place, and receipts
  record its SHA-256. Upstream ships no lock.

### 12.3 D-F3 cuBLAS workspace and PR #595

- **#595** (open) lets a `CudaBlas` own a caller-supplied `CudaSlice<u8>` as
  its workspace and re-binds it after `set_stream`. Its context check is
  `Arc::ptr_eq`. A workspace from a private context works only with a
  `CudaBlas` on a stream of that same `Arc<CudaContext>`, which is the
  desired behaviour.
- **Hazards** (input to FathomDB D-14 and Candle DK-4):
  - `CudaBlas` drop does not synchronize, and no write event is recorded on
    the workspace, so the workspace free can overtake in-flight GEMMs;
  - a captured graph bakes in the workspace address;
  - sizing is the caller's job.
- **Decision:** the fork does not carry #595. A pool-allocated cuBLAS
  workspace is possible and worthwhile once #595 lands. FathomDB D-14
  decides whether to measure it.

### 12.4 D-F4 Built-major library loading (item 9)

**Defects at `ec1cac71`:**

- the unversioned name comes first;
- other-major names are tried. A 12.x build tries `libcufft.so.12` (CUDA
  13's) before `libcufft.so.11`, and on Windows `cufft64_12.dll` before
  `cufft64_11.dll`. The same holds for cuSOLVER and cuSolverMg;
- cuFile never loads by its soname, because no `.so.0` candidate exists;
- Windows CUPTI cannot load at all, because its DLL names carry a year
  (`cupti64_2024.3.1.dll`);
- upstream issue #603 reports the unversioned-name case.

**Candidates** for library L and built major M (12 or 13), in this order:

1. L's versioned names for M (table below);
2. L's unversioned names (`libL.so`; `L64.dll`), accepted only after the
   version check.

| L | Linux, M = 12 | Linux, M = 13 | Windows, M = 12 | Windows, M = 13 |
| --- | --- | --- | --- | --- |
| cudart | `libcudart.so.12` | `.so.13` | `cudart64_12.dll` | `cudart64_13.dll` |
| cublas | `libcublas.so.12` | `.so.13` | `cublas64_12.dll` | `cublas64_13.dll` |
| cublasLt | `libcublasLt.so.12` | `.so.13` | `cublasLt64_12.dll` | `cublasLt64_13.dll` |
| nvrtc | `libnvrtc.so.12` | `.so.13` | `nvrtc64_120_0.dll` | `nvrtc64_130_0.dll` |
| cupti | `libcupti.so.12` | `.so.13` | unchanged from upstream (note 1) | same |
| cufft | `libcufft.so.11` | `.so.12` | `cufft64_11.dll` | `cufft64_12.dll` |
| cusolver | `libcusolver.so.11` | `.so.12` | `cusolver64_11.dll` | `cusolver64_12.dll` |
| cusolverMg | `libcusolverMg.so.11` | `.so.12` | `cusolverMg64_11.dll` | `cusolverMg64_12.dll` |
| cusparse | `libcusparse.so.12` | `.so.12` | `cusparse64_12.dll` | `cusparse64_12.dll` |
| curand | `libcurand.so.10` | `.so.10`, then `.so.11` (note 2) | `curand64_10.dll` | `curand64_10.dll`, `curand64_11.dll` |
| cufile | `libcufile.so.0` | `.so.0` | not available | not available |

Notes:

1. Windows CUPTI keeps upstream's candidates, which cannot load it either.
   This is a known upstream gap and not a regression. Enumerating
   year-named DLLs is out of scope.
2. PATHFINDER lists `libcurand.so.11` for CUDA 13.4, but the 13.4 release
   notes ship cuRAND 10.4.4. It is accepted only through the version check
   (UNCONFIRMED name).

**The Windows names were confirmed from NVIDIA's Windows wheels**
(win_amd64, downloaded from PyPI without installing, 2026-10-12):

| Wheels | DLLs |
| --- | --- |
| cu12 12.6: runtime 12.6.77, cuBLAS 12.6.4.1, NVRTC 12.6.85, CUPTI 12.6.80, cuFFT 11.3.0.4, cuSOLVER 11.7.1.2, cuSPARSE 12.5.4.2, cuRAND 10.3.7.77 | `cudart64_12`, `cublas64_12`, `cublasLt64_12`, `nvrtc64_120_0`, `cupti64_2024.3.2`, `cufft64_11`, `cusolver64_11`, `cusolverMg64_11`, `cusparse64_12`, `curand64_10` |
| cu13 13.0: runtime 13.0.96, cuBLAS 13.0.2.14, NVRTC 13.0.88, CUPTI 13.0.85, cuFFT 12.0.0.61, cuSOLVER 12.0.4.66, cuSPARSE 12.6.3.3, cuRAND 10.4.0.35 | `cudart64_13`, `cublas64_13`, `cublasLt64_13`, `nvrtc64_130_0`, `cupti64_2025.3.1`, `cufft64_12`, `cusolver64_12`, `cusolverMg64_12`, `cusparse64_12`, `curand64_10` |

All names end in `.dll`. The cu13 wheels place their DLLs under
`nvidia/cu13/bin/x86_64/`, which affects users' `PATH` but not the names.
`curand64_11.dll` is not shipped in 13.0 and stays an accepted-by-version
candidate only.

**The version check** runs `dlsym` on the candidate's own handle, never
through the generated binding, which panics. A library is accepted if its
query reports the built major's line:

| L | Query (no handle or context needed) | Accept for M = 12 | Accept for M = 13 |
| --- | --- | --- | --- |
| cudart | `cudaRuntimeGetVersion` | v / 1000 = 12 | v / 1000 = 13 |
| cublas | `cublasGetProperty(MAJOR)` | 12 | 13 |
| cublasLt | `cublasLtGetProperty(MAJOR)` | 12 | 13 |
| nvrtc | `nvrtcVersion` | major 12 | major 13 |
| cupti | `cuptiGetVersion` | 19 ≤ v < 130000 (12.0 is 19 and 11.8 is 18, per the 12.6 `cupti_version.h`) | v / 10000 = 13 |
| cufft | `cufftGetProperty(MAJOR)` | 11 (CUDA 11.x ships `.so.10`) | 12 |
| cusolver | `cusolverGetProperty` | 11.4.2 ≤ v < 12 | 12 |
| cusparse | `cusparseGetProperty(MAJOR, MINOR)` | 12.0–12.5 | 12.6 or later |
| curand | `curandGetVersion` | 10301–10399 | ≥ 10400 |
| cufile | `cuFileGetVersion`, if the symbol exists (cuFile 1.8 / CUDA 12.3 on) | 1080–1140 with the symbol; no symbol: accepted, reason `NoVersionQuery` | ≥ 1150; no symbol: rejected |
| cusolverMg | none exported | versioned name only (known gap below) | versioned name only |

Lower bounds keep CUDA 11.x libraries out of 12.x builds. CUDA 11.x shares
some sonames with 12.x: cuSOLVER and cuSolverMg `.so.11`, cuRAND `.so.10`
and cuFile `.so.0`. The bounds use the earliest CUDA 12 versions on PyPI
(cuSOLVER 11.4.2, cuRAND 10.3.1) and the CUPTI header list.

Known gaps:

- **cuSolverMg `.so.11`** exists in both CUDA 11.x and 12.x and exports no
  version query, so a CUDA 11 cuSolverMg on the path of a 12.x build is not
  detected. FathomDB does not use cuSolverMg.
- **cuFile before 1.8** (CUDA 11.x to 12.2) exports no `cuFileGetVersion`.
  This was measured on 2026-10-12 from NVIDIA's Ubuntu 22.04 apt packages,
  extracted without installing:

  | Package | `cuFileGetVersion` | Value |
  | --- | --- | --- |
  | `libcufile-12-0` 1.5.1.14 | absent | — |
  | `libcufile-12-3` 1.8.1.2 | present | 1080 |
  | `libcufile-12-5` 1.10.1.7 | present | 1100 |

  The encoding is major × 1000 + minor × 10.

  A cuFile without the symbol is accepted for M = 12 with reason
  `NoVersionQuery`, in the report and in its documentation. CUDA 11.x
  cuFile cannot be told apart from 12.0–12.2 cuFile, which is a known gap.
  A cuFile without the symbol is rejected for M = 13, since every 13.x
  cuFile is 1.15 or later.
- **Future majors:** the 13.x ranges are open-ended, so a future CUDA 14
  library that keeps the same soname would be accepted by a 13.x build.
  This trade-off is accepted and noted for the second proposal (FC9b).

- **Measured samples** (12.6 apt / 12.8 wheel / cu13 wheel):
  - cudart: 12060 / 12080 / 13000
  - cuBLAS: [12,6,1] / [12,8,4] / [13,1,1]
  - cuRAND: 10307 / 10309 / 10400
  - cuSPARSE: 12.5.3 / 12.5.8 / 12.6.3
  - cuFile: 1110 / 1130 / 1150
  - CUPTI: 24 / 26 / 130001

  The CUDA 13 CUPTI header defines the `xxyyzz` format with `xx` = CUDA
  major.
- **Same-soname libraries are checked by line, and the other major's line is
  rejected** (requirements revision 3 rule 3, with this evidence):
  - **NVIDIA guarantees ABI only within a major.** RN13.0: "guarantees ABI
    stability within the major version series"; MVC: "provided they are all
    from the same major release family".
  - **CUDA 13 dropped pre-Turing GPUs.** RN13.0: cuSPARSE "Dropped support
    for pre-Turing architectures"; cuFFT "Removed support for Maxwell,
    Pascal, and Volta". The cu13 cuRAND and cuSPARSE fatbins contain only
    sm_75 and newer.
  - **Different dependencies.** The cu13 `libcusparse.so.12` needs
    `libnvJitLink.so.13`, and 13.x needs driver ≥ 580.
- **Cost of rejecting:** a default 12.x build that today uses a torch-cu13
  `libcurand.so.10` on a Turing+ GPU with driver ≥ 580 would now get the
  CUDA 12 cuRAND, if one is installed, or a typed error. That trade is
  accepted, because REQ31-15 forbids loading another major. Out-of-range
  future versions within a major's line (for example cuRAND 10.5 under
  13.5) are accepted for M = 13, since the 13.x ranges are open-ended.
- **Rejected handles are closed.** Measured: after `dlclose`,
  `/proc/self/maps` no longer lists cuBLAS and cuBLASLt, cuSPARSE and
  nvJitLink, cuRAND, cuFFT, or the cuSOLVER chain.
- **Already-loaded objects.** glibc returns an already-loaded object for a
  bare soname. That object might have been loaded earlier in the process by
  another component (for example PyTorch's cu13 `libcurand.so.10`), or kept
  by `DF_1_NODELETE` (`libcufile.so.0`, which stays mapped after
  `dlclose`). So before opening each candidate name, the loader probes it
  with `RTLD_NOLOAD`:
  - if an object is already loaded, its version is checked;
  - if that object belongs to the other major's line, loading fails with
    `RejectReason::AlreadyLoadedWrongVersion`. Another file of the same
    soname cannot be opened in that process.

  This is a limit, not something to work around, and it is documented.
  T-DEV and G8 expect the typed error in these cases (§ 13).
- **Transitive dependencies (Linux).** An accepted library's own
  dependencies may pull in a wrong-line library through the loader. For
  example, CUDA 12's `libcusolver.so.11` needs `libcusparse.so.12`, which
  can resolve to the cu13 copy.
  - After an accepted load, the loader compares the link map before and
    after (`dl_iterate_phdr`, an `extern` declaration).
  - For every newly mapped object whose soname is in scope, it runs that
    library's version check.
  - A wrong line fails the load with `RejectReason::WrongMajorDependency`,
    naming the dependency. The handle is closed.
  - Windows has no equivalent check, which is a known gap.

**API** (upstream-shaped, in each `src/<lib>/sys/mod.rs`, and in
`bindings_generator/src/merge.rs` so regeneration keeps it):

```rust
pub unsafe fn try_culib() -> Result<&'static libloading::Library, crate::LibLoadError>; // memoized
pub unsafe fn culib() -> &'static libloading::Library;      // try_culib() or panic, as today
pub unsafe fn is_culib_present() -> bool;                   // try_culib().is_ok()
pub unsafe fn is_symbol_present(name: &str) -> Result<bool, crate::LibLoadError>;
pub fn loaded_library() -> Option<crate::LoadedLibrary>;    // after a load
```

```rust
// src/lib.rs
#[derive(Clone, Debug)]
pub struct LibLoadError { pub library: &'static str, pub cuda_major: u32,
                          pub tried: Vec<(String, RejectReason)> }
#[derive(Clone, Debug)]
pub enum RejectReason { NotFound, LoadFailed(String), WrongVersion(LibVersion),
                        AlreadyLoadedWrongVersion(LibVersion),
                        WrongMajorDependency { soname: String, version: LibVersion } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LibVersion { pub major: u32, pub minor: u32, pub patch: Option<u32> }
#[derive(Clone, Debug)]
pub struct LoadedLibrary {
    pub library: &'static str,
    pub cuda_major: u32,              // the built major this load was checked against
    pub opened_as: String,            // candidate name that was opened
    pub path: Option<String>,         // canonical path (realpath of dli_fname)
    pub version: Option<LibVersion>,  // None only for NoVersionQuery
    pub note: Option<RejectReason>,   // e.g. NoVersionQuery for an old cuFile
}
pub fn get_lib_name_candidates(lib_name: &str) -> Vec<String>;   // unchanged signature
fn lib_candidates(os: TargetOs, ptr_width: u32, lib: &str, major: u32, minor: u32) -> Vec<String>; // pure
```

- **Resolved path:**
  - **Linux:** `dladdr` of an exported symbol gives `dli_fname`, and the
    report canonicalizes it with `realpath`, so it compares directly with
    `/proc/self/maps`. The symbol is the version query, or any exported
    function for cuSolverMg, for example `cusolverMgCreate`.
  - **Windows:** `GetModuleFileNameW` of the module handle.

  All of these are declared with `extern` blocks, so no new crate
  dependency is added.
- **Memoization** (a documented change from upstream, listed in
  `FATHOMDB.md` and in the second proposal):
  - `try_culib` stores `OnceLock<Result<Library, LibLoadError>>`, so a
    failure is memoized and not retried. Upstream's panicking `get_or_init`
    retries on the next call.
  - `LibLoadError` is `Clone`.
  - `is_culib_present()` now performs and keeps the real load. Upstream
    probes the candidates and drops the handle.
- **Driver:** `crate::driver::sys::loaded_library()` reports the driver's
  file and `cuDriverGetVersion` without changing how it loads.
- **Linked builds:** under `dynamic-linking` or `static-linking`, the report
  says "linked at build time".
- **Scope of the changed output:** `get_lib_name_candidates` keeps its
  signature. Its output changes only for in-scope libraries on 12.x and
  13.x builds, and that change is listed in `FATHOMDB.md` and in the second
  proposal. Out-of-scope names and 11.x builds return exactly the base
  output (T-UNIT).
- **`no-std`:** the report types use `no_std_compat` like the rest of the
  crate.
- **Files changed for item 9:**
  - `src/lib.rs`;
  - the `sys/mod.rs` loaders of the 11 in-scope libraries: `src/runtime`,
    `cublas`, `cublaslt`, `nvrtc`, `cupti`, `cufft`, `cusolver`,
    `cusolvermg`, `cusparse`, `curand` and `cufile`;
  - `src/driver/sys/mod.rs`, for `loaded_library()` only;
  - the template `bindings_generator/src/merge.rs`.

  The cuDNN, NCCL, cuTENSOR and NVTX loaders are unchanged. A unit test in
  the generator renders the template's loader tail for one library and
  compares it with the checked-in file. If the generator crate cannot host
  that test, FC3 moves it to a fork-CI script.

## 13. Requirements revision 3 (made in this commit)

- **CF-AC-02 trim:** k ≤ reserved ≤ max(k, used) + one granule, after a
  synchronize, with the precondition (§ 5).
- **CF-REQ-11 rule 1:** add cuFile's `.so.0`. Versioned-first ordering also
  fixes the base's wrong-major preference for cuFFT, cuSOLVER and
  cuSolverMg.
- **CF-REQ-11 rule 2:** CUPTI is verifiable on Linux (`cuptiGetVersion`).
  The "versioned name only" set is `cusolverMg`. Windows CUPTI stays
  upstream's.
- **CF-REQ-11 rule 3:** same-soname libraries are checked by version line,
  and the other major's line is rejected, with the § 12.4 evidence. This
  replaces revision 2's "reported, not rejected", which rested on an ABI
  claim NVIDIA does not make. G8's cu13 cuRAND/cuSPARSE sub-cell now expects
  rejection, and then either the CUDA 12 library or the typed error.
- **CF-AC-04c G5a:** upstream tests that JIT PTX are UNMEASURED there.
  CUBIN applies only to fork-owned tests.
- **CF-REQ-12:** the baseline uses the committed
  `fathomdb/qualification/Cargo.lock.baseline`.
- **CF-REQ-11 table:** the claim that the cu13 cuRAND and cuSPARSE are
  "ABI-compatible successors" is removed.

### Requirements revision 4 (made with design revision 2)

- **CF-REQ-03 / CF-AC-03:** the hazard is removed (§ 2). The requirement is
  now a doc correction backed by characterization tests, which need no RED.
- **CF-AC-02:** bounds hold with the live set limited to one allocation,
  plus one chunk.
- **CF-AC-04a/04b:** the per-test dependency table and the 13.4 status list
  live in `test-plan.md`.
- **CF-REQ-11:** lower bounds for 12.x; already-loaded and dependency
  checks; cuFile without a query symbol; Windows CUPTI exempt in T-UNIT;
  structured report.
- **G8:** with the cu13 cuRAND/cuSPARSE first on the path, they are
  rejected. Another sub-cell preloads cu13 `libcurand.so.10` and expects
  `AlreadyLoadedWrongVersion`. A third gives CUDA 12 cuSOLVER with cu13
  cuSPARSE first and expects `WrongMajorDependency`.

## 14. Open items for the test plan (FC3)

1. Measure maxSize semantics on G1: reserved versus used, and the error on
   excess.
2. Measure the allocation granule on G1, G5b and G6.
3. Check that driver 580 rejects attributes 9–15. This is an observation
   only; the gate never requests them.
4. Done at FC2: the Windows DLL names were confirmed from NVIDIA's wheels
   (§ 12.4).
5. Pick G5b's driver branch from what Ubuntu 24.04 and NVIDIA's repository
   offer for the T4.
6. Record the 13.4 Driver API deprecation status of each fork function.
7. Done at FC2: `cuFileGetVersion` exists from cuFile 1.8 (CUDA 12.3)
   (§ 12.4).
8. Measure the chunk step at the four allocation sizes (§ 5).
