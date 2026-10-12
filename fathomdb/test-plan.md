# cudarc-fathomdb test plan (FC3)

- Status: FC3 draft for review R-FC-TEST, 2026-10-12.
- Inputs: [requirements](requirements.md) revision 4,
  [design](design.md) revision 2, [plan](plan.md) §§ 6–8.
- Every test traces to a CF-AC and names its cell, command and expected
  result. Test names below are binding: FC4–FC6 must use them.

## 1. Conventions

### 1.1 Environment and guard

Every GPU run on `windchill3` uses:

```bash
export CUDA_VISIBLE_DEVICES=0        # one RTX 3090; G2 uses 0,1; the K620 (2) never
export CUDARC_CUDA_VERSION=<v>       # equal to the -F cuda-<v> feature (issue #525)
```

- **Guard:** the host guard of requirements § 2, run as the first and last
  commands of every run:
  - `nvidia-smi --query-gpu=driver_version --format=csv,noheader`;
  - `readlink -f /usr/local/cuda`;
  - on the Orin, `cat /etc/nv_tegra_release` instead of `nvidia-smi`.

  Any change stops the work.
- **Toolkit libraries** come from:
  - the 12.6 system toolkit, by default;
  - scratch venvs under `~/fathomdb-forks/venvs/<name>`, reached by a
    per-command `LD_LIBRARY_PATH` naming exactly one CUDA library
    directory. A venv is never the FathomDB `.venv`.
- **Lockfile:** every run (baseline and `C`) copies
  `fathomdb/qualification/Cargo.lock.baseline` into place. FC4 generates it
  at `ec1cac71` with the toolchain of record. Receipts record its SHA-256
  and `cargo tree -e normal`.

### 1.2 Runs, baseline and receipts

- **Baseline:** for each cell, the upstream tests at `ec1cac71`, from a
  detached worktree `../cudarc-baseline`, with the cell's
  `CUDARC_CUDA_VERSION` and upstream feature set.
- **Runs:** baseline and `C` each run k = 3 times. The rules of
  requirements CF-REQ-12 classify each test: baseline fact, regression,
  flaky (must be named in § 9), or pass.
- **Ignored tests the cell can run** are run as separate processes with
  `-- --ignored --exact <name>`. These are the "must be executed by itself"
  tests and, on G2, the multi-GPU tests.
- **Counting:** `fathomdb/scripts/count-tests.py` (class `D`, written TDD at
  FC4 with its own fixture tests) parses cargo's output. It compares every
  binary's "N passed; M failed; K ignored; F filtered out" with the expected
  counts of § 2, and fails on any mismatch. Receipts embed its output.
- **Receipt file:**
  `fathomdb/qualification/<YYYY-MM-DD>-<cell>-<short sha>.md`, with the
  fields of plan § 7, plus:
  - the k-run matrix;
  - the host-dependent values (§ 9);
  - for item-9 cells, the loaded-library report and the matching
    `/proc/self/maps` lines.

## 2. Expected counts

### 2.1 Upstream tests at `ec1cac71`

The full name lists are committed in `fathomdb/test-inventory/`
(`upstream-def-<v>.txt`, default features; `upstream-drv-<v>.txt`, the
driver-only set `--no-default-features -F std,driver,dynamic-loading,cuda-<v>`).
They were listed with CPU-only `cargo test --lib -- --list` on 2026-10-12,
and R-FC-DES reproduced one of them. Ignored tests are included.

| Module | def 12060 | def 13000/13010 | def 13020–13040 | drv 12060 | drv 13000/13010 | drv 13020–13040 |
| --- | --- | --- | --- | --- | --- | --- |
| `driver::result::tests` | 3 | 3 | 3 | 3 | 3 | 3 |
| `driver::safe::core::tests` | 23 | 21 | 18 | 23 | 21 | 18 |
| `driver::safe::graph::tests` | 2 | 2 | 2 | 2 | 2 | 2 |
| `driver::safe::launch::tests` | 14 | 14 | 13 | — | — | — |
| `driver::safe::unified_memory::tests` | 6 | 6 | 6 | — | — | — |
| `nvrtc::{result,safe}` | 9 | 9 | 9 | — | — | — |
| `cublas` (asum, gemm, gemv, safe) | 7 | 7 | 7 | — | — | — |
| `cublaslt::safe` | 1 | 1 | 1 | — | — | — |
| `curand::safe` | 10 | 10 | 10 | — | — | — |
| **Total** | **75** | **73** | **69** | **28** | **26** | **23** |

Tests missing at newer bindings are "not compiled (upstream cfg)", never
UNMEASURED:

- `test_from_raw_context_{creates_and_destroys,bind_to_thread}` exist
  only up to 12090;
- `test_new_non_primary_{creates_and_destroys,htod_dtoh,cross_thread}` and
  `test_peer_memcpy_waits_for_src_stream` exist only up to 13010.

Upstream ignored tests (5):

- `peer_transfer_contexts` and `re_associate_context_for_memory_op`, which
  need two GPUs;
- `test_post_alloc_memory` and `test_device_side_assert`, which must run
  by themselves;
- `test_peer_memcpy_waits_for_src_stream`, which needs two GPUs.

### 2.2 Fork tests

The vendored tests (tree A) are listed in
`fathomdb/test-inventory/vendored-drv-12060-x86_64.txt`: 11 in
`fathomdb_alloc_fallback` and 8 in `fathomdb_private_pool` on x86_64. Five
more compile only on aarch64:

- `wrappers_of_one_device_share_the_allocator`;
- `sync_zero_byte_request_is_a_null_pointer_without_a_driver_call`;
- `sync_zero_length_allocations_are_null_and_free_cleanly`;
- `a_private_context_never_consults_or_records_the_process_decision`;
- `private_context_in_a_fresh_process`, the ignored helper of the one
  above.

After FC6 the fork modules hold the vendored tests plus the § 3 tests. FC6
commits `fathomdb/test-inventory/fork-<set>-<v>-<arch>.txt` for every cell's
build. The count script uses those files, and R-FC-CODE checks that each
equals the § 3 catalogue for that cfg.

## 3. Fork test catalogue

Kinds: P = pure (no GPU), D = device, F = fault seam (`cfg(test)`),
X = fresh process, C = compile-time. Each "RED at" names the commit where
the test must fail before its GREEN.

### 3.1 Pool: `driver::safe::core::fathomdb_private_pool` (CF-AC-02, 04, 06)

| Test | Kind | CF-AC | Cells | Expected | RED at |
| --- | --- | --- | --- | --- | --- |
| vendored 8 (inventory) | P/D | 02 | G1, G3 (≤13010), G6 | pass | ported at FC4 (RED = compile failure on bare `ec1cac71`) |
| `default_and_current_pool_are_untouched_while_live` | D | 02 isolation | G1, G3, G6 | handles equal before, during (two threads) and after | FC6 (the during and second-thread reads are new) |
| `accounting_returns_to_zero_after_drop_and_sync` | D | 02 accounting | G1, G3, G6 | used = reserved = 0 | FC6 |
| `release_threshold_bounds_reserved_after_sync` | D | 02 threshold | G1, G3, G5b, G6 | precondition holds; reserved ≤ max(T, used) + chunk | FC6 |
| `trim_to_keeps_the_floor_and_releases_above_it` | D | 02 trim | G1, G3, G5b, G6 | k ≤ reserved ≤ max(k, used) + chunk | FC6 |
| `requests_within_the_cap_succeed` | D | 02 capacity | G1, G3, G5b, G6 | all succeed | FC6 |
| `a_request_over_the_cap_is_out_of_memory` | D | 02 capacity | G1, G3, G5b, G6 | `CUDA_ERROR_OUT_OF_MEMORY`; request ≤ cap + chunk | FC6 |
| `exhaustion_under_four_streams_fails_only_the_excess` | D | 02 capacity | G1, G3, G6 | N = 4; each failing call gets OOM; successful buffers verify | FC6 |
| `a_buffer_freed_on_one_stream_is_not_reused_before_its_reader_finishes` | D | 02 stream order | G1, G3, G6 | the reader sees its data; the reuse address is checked after an event dependency | FC6 |
| `the_pool_outlives_its_context_while_slices_live` | D | 02 lifetime | G1, G3, G6 | extends the vendored `the_pool_is_destroyed_last_and_cleanly` with attribute reads while only the slice holds the pool | FC6 |
| `mem_pool_is_send_and_sync` | C | 02 Send/Sync | CI | compiles; `static_assertions`-style check without a new crate | FC6 |
| `pool_model_property` | D | 02 property | G1, G6 | 256 seeded cases × 64 operations (alloc, free on a random stream, sync, trim) against a model; invariants: used ≤ reserved ≤ cap + chunk; all zero after drop and sync | FC6 |
| `device_location_is_the_ordinal_on_every_binding` | P | 04 (D-8) | CI (19 features) | `type_` = DEVICE, id = ordinal through the union where present | FC5 (fails to compile at 13020+ on the vendored literal) |
| `pool_props_are_zeroed_then_written` | P | 04 (D-8) | CI (19 features) | all `reserved` bytes 0; `maxSize` set from 12020; `usage` 0 | FC5 (vendored `set_max_size` drops 13030/13040) |
| `max_size_without_binding_support_is_not_supported` | P | 04 | CI (11040–12010) | `create` returns `CUDA_ERROR_NOT_SUPPORTED` before any driver call | FC5 |
| `max_size_on_a_driver_older_than_12020_is_not_supported` | F | 04 (D-4) | G1, G3 | a `cfg(test)` driver-version override of 12010 makes `create` return `NOT_SUPPORTED` with no pool created | FC5 |
| `attributes_newer_than_the_driver_are_never_requested` | F+D | 04b (gate) | G3 (13020–13040) | the counter of attribute requests > 8 is 0 on driver 13000, **and** a counter shows the gate branch was evaluated | FC5 |
| `max_pool_size_reads_back_at_least_the_request` | D | 04c | G5b (13020–13040) | read-back ≥ request; `max_size()` is `Some` | FC5 |
| `a_failed_max_pool_size_read_leaves_create_successful` | F | 04 (D-4) | G1 | with a seam-forced read error, `create` succeeds and `max_size()` is `None` | FC5 |
| `typed_reads_return_64_bit_values` | D | 04 (D-4) | G1, G6 | a threshold of 2^33 round-trips through `set_attribute` and `release_threshold()` | FC5 |
| `create_destroys_the_pool_when_setting_the_threshold_fails` | F | 02 (D-2) | G1 | seam fails `set_attribute`; a pool-count seam shows destroy called once | FC5 |
| `destroy_with_outstanding_allocations_is_deferred` | D | 02 (D-2) | G1, G6 | drop the pool handle with a live leaked pointer; free it later with `free_async`; no error | FC6 |
| `capture_on_a_private_context_uses_graph_memory` | D | T-CAPTURE | G1, G5b, G6 | graph memory `USED_MEM_CURRENT` rises; private pool and default pool unchanged; capture error recorded if any | FC6 (outcome-recording test, not RED) |
| `two_pools_on_two_devices_are_independent` | D | T-MULTI | G2 | attributes of pool 0 do not move when pool 1 allocates | FC6 (no RED: independence check) |
| `a_pool_of_device_1_is_rejected_for_device_0` | D | T-MULTI | G2 | `CUDA_ERROR_INVALID_VALUE` with a real device-1 pool | FC6 |
| `exit_with_a_live_pool_is_clean` | X | T-TEARDOWN | G1, G6 | child process allocates and exits without dropping; exit status 0 | FC6 |

The model test uses a small seeded generator inside the test (seed = case
index, printed on failure), so no new dev-dependency is added. That keeps
the proposal clean (plan § 6.1 left the `proptest` choice to FC3).

**Measuring the "chunk"** (design D-5) happens before the bound tests in
each cell. `measure_reservation_steps` (D, an ignored test run first with
`--ignored --exact`) prints the reservation step for 1 B, 1 MiB, 32 MiB and
256 MiB. The bound tests read the value from `CUDARC_TEST_POOL_CHUNK`, set
from that output. The receipt records both.

**Not tested in the fork:** fork-after-init and lost-context teardown.
FathomDB's T-TEARDOWN-PRODUCT owns them (FathomDB D-15). The fork has no
`libc` dependency for `fork(2)`, and inducing a lost context needs product
code. Recorded here so R-FC-TEST can rule on it.

### 3.2 Leaked pointers: `fathomdb_private_pool` (CF-AC-03)

| Test | Kind | Cells | Expected |
| --- | --- | --- | --- |
| `a_private_pointer_upgraded_onto_a_default_context_returns_to_its_pool` | D | G1, G6 | private `USED_MEM_CURRENT` = 0 after drop and sync; `check_err()` clean |
| `a_private_pointer_freed_synchronously_returns_to_its_pool` | D | G1, G6 | same, after `result::free_sync` following a synchronize |
| `a_synchronous_pointer_upgraded_onto_a_private_context_frees_cleanly` | D | G1, G6 | no error; private pool unchanged |

These are characterization tests with no RED (requirements CF-AC-03). The
`# Safety` text is checked by R-FC-CODE.

### 3.3 Allocator fallback: `fathomdb_alloc_fallback` (CF-AC-05)

| Test | Kind | Cells | Expected | RED at |
| --- | --- | --- | --- | --- |
| vendored 11 + 5 aarch64-only (inventory) | P/D/X | G1 (x86 subset), G6 (all) | pass | ported at FC4 |
| `off_target_constructors_never_query_the_default_pool` | F+D | G1 | the `cuDeviceGetDefaultMemPool` call counter is 0 after `new`, `new_non_primary` (≤ 13010), `new_cig` (≤ 13010) and `from_raw_context` | FC6 |
| `off_target_choice_equals_the_upstream_inline_rule` | D | G1 | `has_async_alloc()` == (`MEMORY_POOLS_SUPPORTED` > 0) for each constructor | FC6 (no RED: equality check) |
| `off_target_zero_length_sync_behaviour_matches_the_base` | D | G1 + baseline run | same result as the base on a forced-sync path: `cuMemAlloc(0)` error kind recorded at the base and at `C` | FC6 (compared against the base, not RED) |

### 3.4 Library loading: `fathomdb_lib_major` in `src/lib.rs` (CF-AC-11)

**T-UNIT** (P, the T-UNIT job: 19 features × Linux x86_64 and Windows, with
every library feature enabled). Exactly 6 tests per feature, so the
expected count is 6 per feature: 114 per OS.

| Test | Checks |
| --- | --- |
| `in_scope_candidates_never_name_another_major` | for 12.x/13.x features, every in-scope library on both target OSes (pure function), Windows CUPTI exempt |
| `versioned_names_come_before_unversioned_names` | same scope |
| `out_of_scope_and_cuda11_candidates_equal_the_base` | the candidates for `cuda`, `nvcuda`, `cudnn`, `nccl`, `cutensor` and `nvToolsExt` (and all libraries under 11.x features) equal a verbatim copy of the `ec1cac71` function kept in the test module as the oracle |
| `version_lines_accept_their_major_and_reject_neighbours` | table-driven: the measured samples (design § 12.4) accepted for their line; the CUDA 11 versions (cuSOLVER 11.4.1, cuRAND 10.3.0, CUPTI 18, cuFile with no query for M = 13) and the other major's line rejected |
| `load_error_names_library_major_and_tried_names` | message content of `LibLoadError` |
| `windows_dll_names_match_nvidia_wheels` | the § 12.4 Windows table for both majors |

**T-DEV** (D, G1; `tests/fathomdb_lib_major.rs`, ignored tests run by
`fathomdb/scripts/t-dev-lib-major.sh`). Each case runs in its own process
with its own `LD_LIBRARY_PATH` and reads `/proc/self/maps`:

| Case | Build | Library setup | At `ec1cac71` | At `C` |
| --- | --- | --- | --- | --- |
| (a) | 12060 | dir first on path with `libcublas.so` → cu13 `libcublas.so.13` | maps `libcublas.so.13` (RED) | maps the system `libcublas.so.12`; report names it |
| (b) | 13000 | no CUDA 13 cuBLAS reachable; loader cache gives 12.6 for `libcublas.so` | maps `libcublas.so.12` (RED) | typed error; no `libcublas.so.12` mapping |
| (c) | 12060 | no cuBLAS reachable at all (empty dir first, cache bypassed with an `ld.so` `--inhibit-cache` run of the test binary) | panic (RED) | typed error; exit 0 |
| (d) | 12060 | cu13 `libcufile.so.0` first | maps cu13 cuFile (RED) | `AlreadyLoadedWrongVersion` |
| (e) | 12060 | only `libcuda.so.1` reachable for the driver (lab VM, G5b host) | loads | loads (unchanged) |

The scratch venv `~/fathomdb-forks/venvs/cu13-130` pins:
`nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-nvrtc==13.0.88`,
`nvidia-cublas==13.0.2.14`, `nvidia-curand==10.4.0.35`,
`nvidia-cusparse==12.6.3.3`, `nvidia-cusolver==12.0.4.66`,
`nvidia-cufile==1.15.1.6` and `nvidia-nvjitlink==13.0.88`. FC4 resolves
exact patch versions where PyPI differs and records them.

**G8** (D, `windchill3`, venvs `cu12-121`, `cu12-124`, `cu12-129`,
`cu13-130`):

| Sub-cell | Build | Expected |
| --- | --- | --- |
| G8-a | 12060 + cuBLAS 12.4.x, 12.9.x | upstream cuBLAS tests pass; report path in the venv; version in range |
| G8-b | 12060 + cuBLAS 12.1.x | cuBLAS tests pass, or each missing symbol is a typed error via `is_symbol_present`; no panic |
| G8-c | 13000 + cuBLAS 13.0.x | cuBLAS tests pass |
| G8-d | 12060, cu13 cuRAND and cuSPARSE first | both rejected and unmapped; system 12.6 copies mapped and reported |
| G8-e | 12060, process `dlopen`s cu13 `libcurand.so.10` first | `AlreadyLoadedWrongVersion` |
| G8-f | 12060, cuSOLVER with cu13 cuSPARSE first | `WrongMajorDependency` naming cuSPARSE |

## 4. Cells and commands

`<F>` is the cell's feature list. Every command runs k = 3 times at
`ec1cac71` (upstream tests only) and at `C`.

| Cell | Host | Builds | Command |
| --- | --- | --- | --- |
| G1 | `windchill3`, RTX 3090, driver 580, 12.6 | 12060, default features | `cargo test --lib -F cuda-12060 -- --test-threads=1` plus the ignored tests one by one |
| G2 | same, `CUDA_VISIBLE_DEVICES=0,1` | 12060 | the T-MULTI tests and upstream's multi-GPU ignored tests, with `--ignored --exact` |
| G3 | same | 12060, 13000, 13010, 13020, 13030, 13040; driver-only | `cargo test --lib --no-default-features -F std,driver,dynamic-loading,cuda-<v> -- --test-threads=1` |
| G4 | same + `cu13-130` venv | 13000, default features | `LD_LIBRARY_PATH=<venv>/…/nvidia/cu13/lib cargo test --lib -F cuda-13000 -- --test-threads=1 nvrtc:: cublas:: cublaslt:: curand::` |
| G5a | same + `cu13-13<m>` venvs (nearest PyPI minor, recorded) | 13010–13040, default features | as G4 with the whole `--lib` set; § 7 lists the UNMEASURED tests |
| G5b | GCP Spot T4 (`scripts/gcp-cuda-lab.sh`, `--max-run 2h`), newer driver | 13010–13040, default features, apt toolkits 13.x | full `--lib`, ignored tests one by one |
| G6 | Orin, JetPack 6, 12.6 | 12060, default features, `CUDA_COMPUTE_CAP=87` | as G1, plus every X test (fresh process) |
| G8 | as § 3.4 | 12060, 13000 | the per-sub-cell scripts in `fathomdb/scripts/g8-*.sh` |
| CI | GitHub-hosted (plan § 1.4) | 19 features | check/clippy/fmt/T-UNIT/hosted test/dry run |

**G5b session:**

1. Record the VM creation time.
2. Run `apt-cache policy` for the `nvidia-driver-*-server` and
   `nvidia-headless-no-dkms-*-server` branches, and pick the newest branch
   that offers `cuDriverGetVersion` ≥ 13040. If none reaches 13040, use the
   newest, and record each build whose version exceeds the driver as
   UNMEASURED with the driver version.
3. Install toolkits 13.1–13.4 with toolkit packages only, as in the T4
   trial.
4. Run the cells.
5. Pull receipts and destroy the VM.

The whole session stays within 2 hours of wall clock. The receipt records
the creation and deletion times.

## 5. Per-test dependencies (requirements CF-AC-04b; design D-F1)

All functions below are exported by `libcuda.so.580.178.04`. The version
given is the first cudarc binding that declares the function or value;
the bindings start at 11040.

| Test group (membership) | Driver functions | Values | Layouts |
| --- | --- | --- | --- |
| upstream `core::tests`, `result::tests`, `graph::tests` (inventory) | `cuInit`, `cuDeviceGet{,Count,Attribute}`, `cuDevicePrimaryCtx{Retain,Release_v2}`, `cuCtx{Get,Set}Current`, `cuCtxSynchronize`, `cuCtxDestroy_v2`, `cuStream{Create,Destroy_v2,Synchronize,WaitEvent}`, `cuLaunchHostFunc`, `cuEvent{Create,Record,Query,Synchronize,Destroy_v2}`, `cuMem{AllocAsync,FreeAsync,Alloc_v2,Free_v2,setD8Async,cpyHtoDAsync_v2,cpyDtoHAsync_v2,cpyDtoDAsync_v2,cpyPeerAsync,GetInfo_v2,HostAlloc,FreeHost}`, `cuStream{BeginCapture_v2,EndCapture,IsCapturing}`, `cuGraph{InstantiateWithFlags,Launch,Destroy,ExecDestroy}` (all ≤ 11040); `cuCtxCreate_v4` (12050+) for the non-primary tests | `MEMORY_POOLS_SUPPORTED` 115; event and stream flags; `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`; graph instantiate flags 1, 4 (12000+) and 8 (11070+) | none |
| vendored and new pool tests | the above, plus `cuMemPool{Create,Destroy,GetAttribute,SetAttribute,TrimTo}`, `cuMemAllocFromPoolAsync`, `cuDevice{GetDefaultMemPool,GetMemPool}`, `cuDriverGetVersion` (≤ 11040) | pool attributes 4–8 (≤ 11040); 13 `MAX_POOL_SIZE` (13020+, gated); `CU_MEM_LOCATION_TYPE_DEVICE` 1; `CU_MEM_ALLOCATION_TYPE_PINNED` 1; `CU_MEM_HANDLE_TYPE_NONE` 0 | `CUmemPoolProps` (88 B); `CUmemLocation` (8 B; union from 13020) |
| T-CAPTURE | the above, plus `cuDeviceGetGraphMemAttribute` (≤ 11040) | `CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT` | none |
| fallback tests | `cuDeviceGetDefaultMemPool`, `cuMemAlloc_v2`, `cuMemFree_v2` and the core set | `MEMORY_POOLS_SUPPORTED`; `INTEGRATED` 18 (not used) | none |
| item 9 T-DEV and G8 | none new in the driver; library version queries per design § 12.4 | — | — |
| `launch`, `unified_memory` (default features) | `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuMemAllocManaged`, `cuMemAdvise_v2`/`cuMemPrefetchAsync_v2` (12020+) and the core set | managed attach flags; location types 1–4 (12020+) | `CUmemLocation` |

On driver 580 (`cuDriverGetVersion` 13000):

- Every function and value above is known, except attribute 13 and
  location types 5 and 6 (13020+). The gate keeps attribute 13 from being
  requested, and nothing requests the location types.
- So G3 runs every compiled test at every binding version.
- What depends on driver 580 is PTX JIT of NVRTC 13.1+ output (§ 7).

## 6. Driver API deprecation status (CF-AC-04a)

NVIDIA's Driver API 13.4 deprecated list was read on 2026-10-12
(<https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/deprecated.html>,
header "v13.4"). It lists mostly texture and surface references,
Direct3D/OpenGL interop, legacy launch and the `cuParam*` family. **None
of the functions the fork calls appears on it.** The fork's functions are
§ 5's pool, device, context, stream and allocation functions, plus
`cuDriverGetVersion`.

R-FC-CODE greps the fork diff for every name on that list, and the grep
must be empty.

## 7. RUN/UNMEASURED map

| Test set | G1 | G3 12060–13010 | G3 13020–13040 | G4 | G5a | G5b | G6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| upstream driver tests | RUN | RUN | RUN | — | RUN | RUN | RUN |
| upstream `nvrtc`, `cublas`, `cublaslt`, `curand` | RUN | — | — | RUN | RUN | RUN | RUN |
| upstream `launch`, `unified_memory` | RUN | — | — | — | UNMEASURED: PTX JIT of 13.1+ output on driver 580 (MVC) | RUN | RUN |
| two-GPU upstream tests | in G2 | — | — | — | — | UNMEASURED: one GPU | UNMEASURED: one GPU |
| fork pool tests | RUN | RUN | RUN after FC5 (D-8) | — | RUN | RUN | RUN |
| MAX_POOL_SIZE read-back | — | — | gate test only | — | gate test only | RUN | — |
| fork aarch64-only tests | not compiled (x86) | same | same | — | same | same | RUN |

Every UNMEASURED entry is recorded with the reason shown. "—" means the
cell does not include that set.

## 8. Measurements before the tests

| ID | Where | What | Used by |
| --- | --- | --- | --- |
| M-1 | G1, G5b, G6 | reservation steps at four sizes (§ 3.1) | the bound tests' chunk |
| M-2 | G1 | `maxSize` semantics: grow until OOM with one allocation size; record reserved and used at the failure; repeat with release threshold 0 and with T = cap | design D-4; doc on `MemPoolProps::max_size` |
| M-3 | G1 | `cuMemPoolGetAttribute` with attributes 9–15 on driver 580, called directly through `result::mem_pool::get_attribute` in an ignored test | an observation for design D-4; the gate never relies on it |
| M-4 | G5b | the chosen driver branch's `cuDriverGetVersion` | G5b's RUN/UNMEASURED |

## 9. Host-dependent and flaky tests

- **Host-dependent (closed list):** `test_pinned_copy_is_faster` and
  `test_default_pinned_host_reads_are_faster_than_write_combined`. They
  always run, and the receipt records the timing pair printed on failure.
- **Flaky:** none are named now. A test that is flaky at the baseline is
  added here only with R-FC-TEST approval and its baseline runs attached.

## 10. TDD order (FC4–FC7)

| Step | RED | GREEN |
| --- | --- | --- |
| FC4-W | — | commit `W` (fork CI) and the baseline lock (class `D`) |
| FC4-port | the vendored tests on bare `ec1cac71` (compile failure) | items 1–6 as `L`/`U` commits; tree A check (CF-AC-01) |
| FC5-a | `device_location_is_the_ordinal_on_every_binding` and `pool_props_are_zeroed_then_written` at 13020+ | the D-8 helpers |
| FC5-b | the max-size, gate and read-back tests | D-4 |
| FC5-c | `create_destroys_the_pool_when_setting_the_threshold_fails`; typed reads | D-2 and D-7 |
| FC5-d | — (doc) | `upgrade_device_ptr` `# Safety`; § 3.2 characterization tests |
| FC6-a | the remaining § 3.1 and § 3.3 tests that have a RED | fixes, if any fail |
| FC6-b | § 3.4 T-UNIT, then T-DEV (a)–(d) | item 9 |
| FC7 | — | packaging `P` |

Each RED sha and its failing output go into `status.md`, and the reviewer
re-runs it. Tests marked "no RED" are listed there with their reason.
