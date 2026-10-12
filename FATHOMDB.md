# cudarc-fathomdb: differences from upstream cudarc

`cudarc-fathomdb` is FathomDB's fork of
[cudarc](https://github.com/chelsea0x3b/cudarc). It is not the upstream
crate. This file lists everything it changes, so that a consumer can see
what differs from upstream and when each change can go away.

- **Base:** upstream tag `v0.19.10`, commit `ec1cac71`.
- **Source:** [coreyt/cudarc](https://github.com/coreyt/cudarc), branch
  `fathomdb/0.19.10`. The fork's plan, requirements, design, test plan and
  status records are in that branch's `fathomdb/` directory, which is not
  part of the published crate.
- **Licence and authorship:** unchanged from upstream (MIT OR Apache-2.0).

## How to depend on it

The package is named `cudarc-fathomdb`, but the library is still named
`cudarc`, so code keeps writing `cudarc::...`. Rename the dependency back to
`cudarc`:

```toml
[dependencies]
cudarc = { package = "cudarc-fathomdb", version = "0.19.10" }
```

The feature names are upstream's, plus `fathomdb-private-pool` (item 6).

## Items

Class: `U` upstream-shaped code, proposable upstream; `L` local-only code;
`P` packaging; `W` fork CI.

| # | Item | Class | Upstream status | Removal path |
| --- | --- | --- | --- | --- |
| 1 | Allocator fallback: synchronous `cuMemAlloc` when the default pool is unusable, aarch64 Linux only | `L` | local only | drop when no FathomDB artifact needs an optional default pool on aarch64 Linux, or upstream ships an equivalent |
| 2 | Per-device fallback decision | `L` | local only | with item 1 |
| 3 | Zero-length synchronous allocation | `L` | local only | with item 1 |
| 4 | Fallback tests | `L` | local only | with item 1 |
| 5 | Private memory pool: `CudaMemPool`, `CudaContext::new_with_mem_pool`, `AllocMode::Private`, tests | `U` | not upstream; an upstream proposal (`proposal/mem-pool`) is planned and not posted | drop when an upstream release ships an equivalent; FathomDB switches to the upstream names |
| 6 | Marker feature `fathomdb-private-pool` | `L` | local only | with item 5 |
| 7 | Fork CI workflow (not in the published crate) | `W` | not applicable | delete with the branch |
| 8 | Packaging as `cudarc-fathomdb` (`[lib] name = "cudarc"`), `exclude`, README notice, this file | `P` | not applicable | when FathomDB consumes upstream cudarc again |
| 9 | Built-major library loading: per library, try only the built major's versioned names first, verify unversioned names by the library's version query, and report the loaded file and version | `U` | not upstream; a second upstream proposal (`proposal/lib-major`) is planned and not posted | drop when upstream ships an equivalent |

## Items 1–4: allocator fallback (aarch64 Linux only)

All four are compiled only for `cfg(all(target_os = "linux", target_arch =
"aarch64"))`. On every other target, allocator selection and zero-length
handling are upstream's.

- **Item 1.** A context uses stream-ordered allocation when the device
  reports memory-pool support, as upstream does, and also only if
  `cuDeviceGetDefaultMemPool` succeeds. If that query returns
  `CUDA_ERROR_OUT_OF_MEMORY` or `CUDA_ERROR_NOT_SUPPORTED`, the device
  allocates synchronously (`cuMemAlloc` / `cuMemFree`). Any other error is
  returned, not treated as "no pool". This was measured on Jetson (Tegra),
  where a fragmented process cannot create the default pool while
  `cuMemAlloc` still succeeds.
- **Item 2.** The decision is made once per device per process, by the first
  context created for or wrapping that device, and every later context of
  the device reuses it.

  Corrected rationale: an earlier version of this record justified the
  per-device decision by a hazard in freeing a `cuMemAlloc` buffer with
  `cuMemFreeAsync` (or the reverse). NVIDIA allows both: `cuMemFree` accepts
  `cuMemAllocAsync` and `cuMemAllocFromPoolAsync` pointers (CUDA 12.6
  `cuda.h`, `cuMemFree`), and the CUDA Programming Guide allows
  `cudaMalloc` memory to be freed with `cudaFreeAsync` and `cudaMallocAsync`
  memory with `cudaFree`. The item is kept for a different reason: it keeps
  all wrappers of one device on one allocator, which is consistent and was
  measured on the Orin.
- **Item 3.** With the fallback, a zero-byte synchronous request returns a
  null pointer without a driver call, and a null pointer is not passed to
  `cuMemFree`. This matches `cuMemAllocAsync`, which succeeds with a null
  pointer, whereas `cuMemAlloc` rejects zero bytes with
  `CUDA_ERROR_INVALID_VALUE`.
- **Item 4.** The tests of items 1–3.

Not measured: aarch64 Linux CUDA hosts other than Tegra (for example Grace or
other SBSA systems). On those, the fallback acts only if the default-pool
query fails with one of the two errors above.

## Item 5: private memory pool

The pool uses upstream's result-layer `driver::result::mem_pool` and adds no
new bindings. What it adds:

- `CudaMemPool::create(ordinal, &MemPoolProps)` creates a pool of pinned
  device memory with a `max_size` (0 asks for the driver's default) and a
  release threshold. Dropping the last reference destroys the pool.
- `CudaContext::new_with_mem_pool` builds a context on the device's primary
  context whose allocations all come from that pool
  (`cuMemAllocFromPoolAsync`, freed with `cuMemFreeAsync`).
  `CudaContext::alloc_mode()` reports `AllocMode::Private` for it, and
  `CudaContext::mem_pool()` returns the pool. The device's current memory
  pool is neither read nor changed, so every other context of the device
  keeps its own allocator. A pool of another device is rejected with
  `CUDA_ERROR_INVALID_VALUE`.
- Typed reads of the pool's 64-bit counters (`reserved_current`,
  `reserved_high`, `used_current`, `used_high`, `release_threshold`),
  `trim_to` and `raw`.

Behaviour to know about:

- **`max_size` is never silently ignored.** A nonzero `max_size` needs
  bindings from `cuda-12020` and a driver of 12.2 or later; otherwise
  `create` returns `CUDA_ERROR_NOT_SUPPORTED`.
- **Read-back of the maximum size.** `CudaMemPool::max_size()` is the
  driver's `CU_MEMPOOL_ATTR_MAX_POOL_SIZE`, read at creation only when the
  bindings are `cuda-13020` or later and `cuDriverGetVersion` reports 13020
  or later. Pool attributes 9–15 are never requested otherwise. Driver
  580.178.04 (CUDA 13.0) rejects every pool attribute above 8 with
  `CUDA_ERROR_INVALID_VALUE` (measured). A failed read leaves
  `max_size()` `None`; a reported size below a nonzero request fails
  `create` with `CUDA_ERROR_INVALID_VALUE` after the pool is destroyed.
- **Freeing pool memory with `cuMemFree`.** A pointer taken out with
  `CudaSlice::leak` may be upgraded onto any context of the device with
  `CudaStream::upgrade_device_ptr`, whose `# Safety` section states the
  conditions. `cuMemFreeAsync` returns the memory to its pool. `cuMemFree`
  (a context without stream-ordered allocation) also releases it, but on
  driver 580.178.04 the pool's `CU_MEMPOOL_ATTR_USED_MEM_CURRENT` is left
  unchanged after a synchronous `cuMemFree` (measured on an RTX 3090), so
  prefer `cuMemFreeAsync` for pool memory.
- **Graph capture is outside the pool's budget.** An allocation made on a
  stream that is being captured into a graph is owned by the graph: it is
  outside the pool's `maxSize` and release threshold, and the pool's
  counters do not include it. On driver 580.178.04, a captured allocation
  from a pool with a nonzero `maxSize` instantiated and launched, and the
  pool's counters did not change (measured).
- **Reservation granularity (measured, RTX 3090, driver 580.178.04).** A
  pool reserves 32 MiB for its first allocation of up to 32 MiB, and larger
  allocations reserve their own size. With `maxSize` 64 MiB, 64 requests of
  1 MiB succeed and the 65th returns `CUDA_ERROR_OUT_OF_MEMORY`, so the cap
  bounds reserved memory.

## Tegra notes

Measured on a Jetson AGX Orin (JetPack 6, L4T R36.5.2, CUDA 12.6). NVIDIA
does not document these values.

- **A `maxSize` cap gives about a third of the cap.** A pool with a nonzero
  `max_size` returns `CUDA_ERROR_OUT_OF_MEMORY` after about a third of it
  (64 MiB cap: 32 MiB; 512 MiB: 192 MiB; 4096 MiB: 1376 MiB). The fork
  passes `max_size` to the driver unchanged, so size the cap at about three
  times the memory needed. On an RTX 3090 the whole cap is usable.
- **About four uncapped pools per process.** Each pool with `max_size` 0
  takes a large range of GPU virtual address space, and a fresh process has
  room for about four. The device's default pool (which item 1 obtains on
  `CudaContext::new`) takes one, and graph memory takes one for the rest of
  the process once a graph with a memory allocation is instantiated. A pool
  dropped while an allocation is outstanding keeps its range until the
  allocation is freed. When no range is left, `CudaContext::new` selects
  synchronous allocation (item 1). Capped pools take less.
- **`cuMemGetInfo` is not a release signal.** Free memory is the kernel's
  `MemAvailable` and barely moves when device memory is allocated or freed.
  Read the pool's counters (`reserved_current`, `used_current`) or the
  process's resident set instead.
- **Upstream `CudaContext::new_non_primary` crashes.** It calls
  `cuCtxCreate_v4` with a null parameter pointer, which segfaults in this
  driver; the same call with zeroed parameters succeeds. This is upstream
  behaviour, unchanged by the fork, and FathomDB does not call it.

## Item 6: marker feature

`fathomdb-private-pool` enables nothing. A dependent that enables it fails to
resolve against an unpatched cudarc, instead of failing later on a missing
symbol.

## Item 9: built-major library loading

Item 9 changes how the CUDA toolkit libraries are found under
`dynamic-loading` and `fallback-dynamic-loading`. The in-scope libraries are
`cudart`, `cublas`, `cublasLt`, `nvrtc`, `cupti`, `cufft`, `cusolver`,
`cusolverMg`, `cusparse`, `curand` and `cufile`. The driver, cuDNN, NCCL,
cuTENSOR and NVTX load as upstream does.

Upstream tries a long list of names across CUDA majors, so a CUDA 12 build
can bind a CUDA 13 library found under an unversioned name, or the reverse.
NVIDIA guarantees ABI stability only within one major.

Behaviour changes from upstream v0.19.10 that apply to the in-scope
libraries in CUDA 12.x and 13.x builds (Windows CUPTI excepted):

- **`get_lib_name_candidates` returns a different list for the in-scope
  libraries.** Its signature is unchanged. The list is the library's
  versioned names for the built major, then its unversioned name. It never
  names another major's versioned file. Additional details:
  - cuFile's `libcufile.so.0` is now tried;
  - cuSolverMg has no unversioned candidate (it has no version query);
  - Windows cuFile has no candidates;
  - Windows CUPTI keeps upstream's candidates.

  Its output for the other libraries, and for CUDA 11.x builds, is
  upstream's.
- **Unversioned names are version-checked.** A library opened by an
  unversioned name (for example `libcublas.so`) is accepted only if its own
  version query reports the built major's line. Otherwise it is closed and
  loading continues.
- **cuSPARSE, cuRAND and cuFile are version-checked even by their versioned
  name**, because their soname is the same in CUDA 12 and 13. A cuFile
  without `cuFileGetVersion` (before cuFile 1.8) is accepted for 12.x with the
  note `NoVersionQuery` and rejected for 13.x.
- **On Linux, an object already loaded under the soname** is checked first.
  If it reports another line, loading fails with
  `AlreadyLoadedWrongVersion`. In-scope dependencies that an accepted
  library pulls in are checked too, and a wrong one fails the load with
  `WrongMajorDependency`.

In CUDA 11.x builds (and for Windows CUPTI) the candidate list is
upstream's and the first name that opens is used, with no version check, as
upstream does. The changes below apply to every in-scope library in every
build, CUDA 11.x included:

- **A typed, non-panicking path.** Each in-scope `sys` module adds
  `try_culib()`, which returns `Result<&Library, LibLoadError>`, and
  `is_symbol_present(name)`. `LibLoadError` names the library, the CUDA
  major it needs, and every name tried with the reason it was rejected.
  `culib()` still panics on failure, now with the `LibLoadError` message,
  so the panic text differs from upstream's.
- **Memoization.** A failed load is memoized and returned again by later
  calls; it is not retried in the process. Upstream retries on the next call.
  `LibLoadError` is `Clone`.
- **`is_culib_present()` performs and keeps the real load**, and gives the
  same answer as `try_culib()`. Upstream probes the candidates and drops the
  handle.
- **Loaded-library report.** Each in-scope `sys` module and `driver::sys`
  add `loaded_library()`, which returns a `LoadedLibrary`, available after a
  successful load. It gives the major checked against, the name opened, the
  canonical path of the file loaded (comparable with `/proc/self/maps`), and
  the version (or the reason there is none). To fill in the version, the
  loader calls the library's version query (through `dlsym` on the loaded
  handle) during the load, in CUDA 11.x builds too, where the result is only
  reported. Under `dynamic-linking` or
  `static-linking` the report says "linked at build time". The driver's
  report does not change how the driver loads.
- **New public types** at the crate root: `LibLoadError`, `LibVersion`,
  `LoadedLibrary` and `RejectReason`.

Known gaps:

- cuSolverMg `.so.11` exists in CUDA 11.x and 12.x and has no version query,
  so a CUDA 11 cuSolverMg on the path of a 12.x build is not detected.
- CUDA 11.x cuFile cannot be told apart from 12.0–12.2 cuFile (neither
  exports `cuFileGetVersion`).
- The already-loaded and dependency checks are Linux only.
- On Windows, the code that resolves the loaded file's path
  (`GetModuleHandleExW`, `GetModuleFileNameW`) is compile-checked by fork CI
  only; it has not been run.
