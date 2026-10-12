# Probe receipt: uncapped pools on the Orin (R-FC-CODE-2 L1, L2)

Evidence for design D-2 ("Tegra: about four uncapped pools per process")
and the matching `FATHOMDB.md` Tegra note. It replaces two statements the
review found unmeasured: the virtual address space per uncapped pool (L1)
and the allocator `CudaContext::new` selects once uncapped pools hold all
of it (L2).

- **Host:** Jetson AGX Orin, `Linux 5.15.199-tegra aarch64`, L4T R36.5.2
  (`# R36 (release), REVISION: 5.2, GCID: 46426093`), `/usr/local/cuda` →
  `/usr/local/cuda-12.6`, `cuDriverGetVersion` 12060; `rustc 1.95.0
  (59807616e 2026-04-14)`.
- **Probe:** `uncapped-pools-probe.rs` (SHA-256
  `e7e84a328169b56abf5453b7ef1b0883f2785c1e89fb9a5414a1a800dabcf74a`)
  with `uncapped-pools-probe.Cargo.toml` (both beside this receipt). It
  depends on the fork by path, driver-only features at `cuda-12060`; the
  fork tree was the library code of `8528fc2` (the R-FC-CODE-2 round
  changes only the fork test module and `FATHOMDB.md`). Binary SHA-256
  `fbe2aa66aa896b0a8fe84a6055824fe542019ac0f2389b2b9334399adefa7ae9`.
- **Environment:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`,
  no `LD_LIBRARY_PATH`. Each run is a fresh process, bracketed by the
  Orin guard (`head -1 /etc/nv_tegra_release`, `readlink -f
  /usr/local/cuda`); all guard lines equal, 2026-10-12T12:13 local.
- **Output:** `2026-10-12-uncapped-pools-probe.log` (beside this receipt,
  6294 bytes, SHA-256
  `f9db1cc7bc7fa7eb3230521a946c92397dd9073dfaa09f367c66ac188bed222e`):
  `exhaust` k = 3, `fresh` k = 2. The runs agree in every `VmSize` value
  and outcome; `VmRSS` differs by a few KiB. Quoted below: the first run
  of each mode.

## Results

`exhaust` (no context first; uncapped `CudaMemPool::create(0, max_size 0)`
until it fails):

```text
before cuInit: VmSize 2998272 VmRSS 946176
mode exhaust; driver 12060; integrated 1; start VmSize 66029326336 VmRSS 5824512
pool 1: ok; VmSize 66029326336 -> 66029326336 (+0 bytes, 0.00 GiB); VmRSS 5824512 -> 5824512
pool 2: ok; VmSize 66029326336 -> 66029502464 (+176128 bytes, 0.00 GiB); VmRSS 5824512 -> 5824512
pool 3: ok; VmSize 66029502464 -> 88007655424 (+21978152960 bytes, 20.47 GiB); VmRSS 5824512 -> 5824512
pool 4: ok; VmSize 88007655424 -> 109985808384 (+21978152960 bytes, 20.47 GiB); VmRSS 5824512 -> 6381568
pool 5: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory"); VmSize 109985808384 -> 109985984512 (+176128 bytes)
with 4 uncapped pools live: cuDeviceGetDefaultMemPool -> Err(DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")); VmSize 109985984512 -> 109985984512
with 4 uncapped pools live: CudaContext::new(0).alloc_mode() = Synchronous
after dropping the pools: CudaContext::new(0).alloc_mode() = Synchronous (process-wide decision, fixed by the first new)
```

`fresh` (control: `CudaContext::new` first, then the same loop):

```text
fresh process: CudaContext::new(0).alloc_mode() = Default; VmSize 66029326336 -> 66124152832 (+94826496 bytes)
pool 1: ok; VmSize 66124152832 -> 66124152832 (+0 bytes, 0.00 GiB); VmRSS 108879872 -> 108879872
pool 2: ok; VmSize 66124152832 -> 88102477824 (+21978324992 bytes, 20.47 GiB); VmRSS 108879872 -> 108879872
pool 3: ok; VmSize 88102477824 -> 110080630784 (+21978152960 bytes, 20.47 GiB); VmRSS 108879872 -> 108879872
pool 4: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory"); VmSize 110080630784 -> 110080630784 (+0 bytes)
with the context's default pool: 3 uncapped private pools created
```

## Reading

- **L1.** An uncapped pool that does not fit in the range `cuInit` maps
  adds 21978152960 bytes (20.47 GiB, about 22 GB) to `VmSize`.
  `cuInit` itself maps 66026328064 bytes, room for the first two pools,
  which add nothing visible. A fresh process holds four uncapped pools
  (or the device default pool and three private ones); the next
  `cuMemPoolCreate` returns `CUDA_ERROR_OUT_OF_MEMORY`. `VmRSS` does not
  move: the cost is address space, not memory.
- **L2.** With four uncapped private pools live, `cuDeviceGetDefaultMemPool`
  returns `CUDA_ERROR_OUT_OF_MEMORY` and the fork's `CudaContext::new`
  selects `AllocMode::Synchronous` (item 1's fallback). The decision is
  per process and device: it stays `Synchronous` after the pools are
  dropped. A fresh process selects `Default`.
- NVIDIA does not document these values; they are measured on this
  driver only.
