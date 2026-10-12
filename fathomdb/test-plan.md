# cudarc-fathomdb test plan (FC3)

- Status: FC3 revision 3, 2026-10-12.
  - Revision 1 (`936c25e`) was reviewed by R-FC-TEST (APPROVE-WITH-FIXES).
  - Revision 2 fixes every finding; dispositions are in
    [review.md](review.md).
  - Revision 3 (R-FC-CODE fix round): § 3.4 T-UNIT gains
    `report_address_symbols_are_bound_wherever_the_library_is` (finding L3),
    so T-UNIT has 10 tests per feature; the T-DEV (d) setup states the
    in-process preload (finding L4).
  - Revision 4 (FC8 G6 fix round; findings F-G6-1, F-G6-2, F-G5a-1 of
    the G6 receipt `2026-10-12-G6-b29c41c.md`): § 3.1 and § 3.2 give the
    release oracle on Tegra (pool attributes and process VmRSS, not
    `cuMemGetInfo`) and the Tegra branch of
    `requests_within_the_cap_succeed`; § 3.1 adds the leaked-pointer
    guard; § 4 runs G6 one process per test; § 8 gives M-2 per platform
    with the Orin table; § 9 adds the G6 platform list. The § 9 list
    amends the "closed list" rule and is **pending R-FC-CODE-2**.
- Inputs: [requirements](requirements.md) revision 6,
  [design](design.md) revision 3, [plan](plan.md) §§ 1, 6–9.
- Tracing: every test traces to a CF-AC, or, where marked "design", to the
  design topic it serves. Each test names its cell, command and expected
  result.
- Names: the test names below are binding. FC4–FC6 must use them.

## 1. Conventions

### 1.1 Environment and guard

Every GPU run on `windchill3` uses:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # 0, 1 = RTX 3090 (41:00, 42:00); 2 = K620 (61:00)
export CUDA_VISIBLE_DEVICES=0         # G2 uses 0,1; the K620 is never visible
export CUDARC_CUDA_VERSION=<v>        # equal to the -F cuda-<v> feature (issue #525)
```

- **Device check:** the receipt logs `nvidia-smi --query-gpu=index,name,pci.bus_id`
  and the device name the tests saw.
- **Guard** (requirements § 2), run as the first and last commands of every
  run:
  - `nvidia-smi --query-gpu=driver_version --format=csv,noheader`;
  - `readlink -f /usr/local/cuda`;
  - on the Orin, `cat /etc/nv_tegra_release` instead of `nvidia-smi`.

  Any change stops the work.
- **Toolkit libraries** come from:
  - the 12.6 system toolkit, by default;
  - scratch venvs under `~/fathomdb-forks/venvs/<name>`, reached by a
    per-command `LD_LIBRARY_PATH` naming exactly one CUDA library
    directory (two for T-DEV (a), § 3.4). A venv is never the FathomDB
    `.venv`.
- **Lockfile:** every run, baseline and `C`, copies
  `fathomdb/qualification/Cargo.lock.baseline` into place. FC4 generates it
  at `ec1cac71`. Receipts record its SHA-256 and `cargo tree -e normal`.

### 1.2 Runs, baseline and receipts

- **Baseline:** for each cell, the upstream tests at `ec1cac71`, from a
  detached worktree `../cudarc-baseline`, with the cell's
  `CUDARC_CUDA_VERSION` and upstream feature set.
  - The baseline records its loaded-library map (`/proc/<pid>/maps` lines
    for CUDA libraries, captured by a wrapper script). The base may load a
    different CUDA library than `C`; for example, a 13.x base build resolves
    unversioned `libcublas.so` to 12.6 through the loader cache.
  - The receipt explains every baseline-vs-`C` difference against those
    maps.
- **Runs:** baseline and `C` each run k = 3 times. The rules of
  requirements CF-REQ-12 classify each test: baseline fact, regression,
  flaky (must be named in § 9), or pass.
- **Ignored tests the cell can run** are run as separate processes with
  `-- --ignored --exact <name>`. That covers every cell, including G3.
- **Name sets and counts:**
  - `fathomdb/scripts/count-tests.py` (class `D`, written TDD at FC4 with
    its own fixture tests) parses cargo's output.
  - It compares the **set of test names** run per binary and per module,
    and each name's outcome, with the expected list for that cell's
    feature set, OS, architecture and filter. It also checks cargo's
    "filtered out" count.
  - Any missing, extra or unexpected-outcome name fails the run.
  - Receipts embed its output.
- **Expected lists:**
  - **Upstream:** `fathomdb/test-inventory/upstream-*.txt` (§ 2.1).
  - **Fork:** generated from `fathomdb/test-inventory/fork-catalogue.toml`,
    never from the build. The catalogue is a machine-readable copy of § 3:
    each row gives the name, module, `cfg` condition (features, arch, OS),
    kind and cells. FC4 commits it as its first step, and R-FC-CODE checks
    it against § 3. A test that the code silently cfg's out is therefore
    missing from the run and fails the comparison.
- **Receipt file:**
  `fathomdb/qualification/<YYYY-MM-DD>-<cell>-<short sha>.md`, with the
  fields of plan § 7, plus:
  - the k-run matrix;
  - the host-dependent values (§ 9);
  - the baseline and `C` loaded-library maps;
  - for item-9 cells, the loaded-library report.

## 2. Expected counts

### 2.1 Upstream tests at `ec1cac71`

The full name lists are committed in `fathomdb/test-inventory/`:

- `upstream-def-<v>.txt`: default features;
- `upstream-drv-<v>.txt`: the driver-only set
  `--no-default-features -F std,driver,dynamic-loading,cuda-<v>`.

They were listed with CPU-only `cargo test --lib -- --list`, and R-FC-TEST
re-listed five sets with identical results. Ignored tests are included.

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

- `test_from_raw_context_{creates_and_destroys,bind_to_thread}` exist only
  up to 12090;
- `test_new_non_primary_{creates_and_destroys,htod_dtoh,cross_thread}` and
  `test_peer_memcpy_waits_for_src_stream` exist only up to 13010.

Upstream ignored tests (5):

| Test | Location | Why ignored | Where run |
| --- | --- | --- | --- |
| `peer_transfer_contexts` | `result.rs:1644` | two GPUs | G2 |
| `re_associate_context_for_memory_op` | `result.rs:1673` | two GPUs | G2 |
| `test_post_alloc_memory` | `core.rs:2781` | must run by itself | alone in every cell where it compiles |
| `test_device_side_assert` | `launch.rs:794` | must run by itself | alone in every cell where it compiles |
| `test_peer_memcpy_waits_for_src_stream` | `launch.rs:835` | two GPUs | G2 |

### 2.2 Fork tests

The vendored tests (tree A) on x86_64 are listed in
`fathomdb/test-inventory/vendored-drv-12060-x86_64.txt`: 11 in
`fathomdb_alloc_fallback` and 8 in `fathomdb_private_pool`. Five more
compile only on aarch64:

- **In `fathomdb_alloc_fallback` (3):**
  - `wrappers_of_one_device_share_the_allocator`;
  - `sync_zero_byte_request_is_a_null_pointer_without_a_driver_call`;
  - `sync_zero_length_allocations_are_null_and_free_cleanly`.
- **In `fathomdb_private_pool` (2):**
  - `a_private_context_never_consults_or_records_the_process_decision`;
  - `private_context_in_a_fresh_process`, its ignored helper.

## 3. Fork test catalogue

Kinds:

| Kind | Meaning |
| --- | --- |
| P | pure, no GPU |
| D | device |
| F | fault seam (`cfg(test)`) |
| X | fresh process |
| C | compile-time |

- **"RED at"** names the commit at which the test fails behaviourally
  before its GREEN. A compile failure of new API does not count as a RED
  (requirements § 3).
- **"No RED"** is stated with its reason.
- **Counters** (F) always have a positive control: the same test makes one
  direct call through the counted seam, asserts the counter moved, and
  resets it before the measured part.

### 3.1 Pool: `driver::safe::core::fathomdb_private_pool`

**In-process chunk.** Every bound test measures its own chunk first. It
allocates its working size once, reads the step in `RESERVED_MEM_CURRENT`,
frees everything and synchronizes, and uses that step as "chunk" for the
rest of the test. It then asserts the precondition
reserved_before > max(T or k, used) + chunk, so that a no-op threshold or
trim fails the bound. No environment variable is involved.
`measure_reservation_steps` (an ignored D test) still prints the steps for
1 B, 1 MiB, 32 MiB and 256 MiB into the receipt (M-1).

**Release oracle on Tegra (revision 4).** Tests that assert that memory
was released branch at run time on the device's
`CU_DEVICE_ATTRIBUTE_INTEGRATED` (a shared test helper):

- **Discrete GPUs (attribute 0)** keep `cuMemGetInfo` free memory, with
  the assertions of revision 3 unchanged.
- **Integrated GPUs (attribute 1, the Orin)** use the pool attributes
  (`RESERVED_MEM_CURRENT`, `USED_MEM_CURRENT`) where the pool is still
  alive, plus the process's `VmRSS` from `/proc/self/status`. On the Orin
  (JetPack 6, 12.6), `cuMemGetInfo` free memory is `/proc/meminfo`
  `MemAvailable`, and once the nvmap page pools are warm
  (`enable_page_pools=Y`) it does not move: 96 MiB reserved and then
  released left it at `52810706944 -> 52810706944` (G6 receipt
  `2026-10-12-G6-b29c41c.md`). The pool attributes and `VmRSS` are exact:
  3 × 32 MiB gave `RESERVED_MEM_CURRENT` 100663296 and `VmRSS` +104 MiB,
  and after a synchronize at threshold 0, reserved 0 and `VmRSS` back
  (FC8 G6 root-cause probes, 2026-10-12; the new G6 receipt reproduces
  them with the tests' printed values). Upstream's
  `test_post_alloc_memory`, which asserts `cuMemGetInfo` movement, fails
  on the Orin at `ec1cac71` for the same reason (§ 9).

**Leaked pointers are guarded (revision 4).** Every test in this module
that takes a pointer out with `CudaSlice::leak` and can panic before
freeing it holds the pointer in a guard that frees it during unwinding.
A failing test therefore cannot leave a pool in deferred destruction
behind. On the Orin each uncapped pool holds about 21 GB of GPU virtual
address space and a process gets about four (design D-2), so one leaked
pool made later tests fail with `CUDA_ERROR_OUT_OF_MEMORY` (F-G6-1).
`pool_model_property` frees each leaked pointer in the next statement,
with no panic point between, and needs no guard.

| Test | Kind | Serves | Cells | Expected | RED at |
| --- | --- | --- | --- | --- | --- |
| vendored 8 (+ 2 aarch64) | P/D/X | CF-AC-02 | G1, G3 (12060–13010; 13020+ after FC5-a), G6 (all 10) | pass | FC4-port: the vendored tests on bare `ec1cac71` fail because `CudaMemPool` is missing (a compile failure; the port is the exception plan § 9 FC4 names) |
| `default_and_current_pool_are_untouched_while_live` | D | CF-AC-02 isolation | G1, G3, G6 | handles equal before, during (from the allocating thread and a second thread) and after | no RED: the vendored code already isolates; characterization |
| `accounting_returns_to_zero_after_drop_and_sync` | D | CF-AC-02 accounting | G1, G3, G6 | used = reserved = 0 | no RED: characterization |
| `release_threshold_bounds_reserved_after_sync` | D | CF-AC-02 threshold | G1, G3, G5b, G6 | precondition holds; reserved ≤ max(T, used) + chunk | no RED: the driver enforces it; characterization |
| `trim_to_keeps_the_floor_and_releases_above_it` | D | CF-AC-02 trim | G1, G3, G5b, G6 | precondition holds; k ≤ reserved ≤ max(k, used) + chunk | no RED: characterization |
| `requests_within_the_cap_succeed` | D | CF-AC-02 capacity | G1, G3, G5b, G6 | each allocation up to the M-2 limit succeeds. **Discrete:** unchanged; limit ≤ cap and limit + chunk > cap. **Tegra:** every allocation within the measured usable limit succeeds, limit ≤ cap, and limit ≥ one chunk; the limit is printed as an M-2 measurement (§ 8, Orin table) | no RED on discrete (characterization). Tegra branch: RED is the revision-3 test on G6 at `b29c41c` (`limit 33554432 leaves more than a chunk (33554432) of the cap 67108864 unusable`) |
| `a_request_over_the_cap_is_out_of_memory` | D | CF-AC-02 capacity | G1, G3, G5b, G6 | `CUDA_ERROR_OUT_OF_MEMORY`; total requested ≤ cap + one chunk | no RED: characterization |
| `exhaustion_under_two_threads_and_four_streams_fails_only_the_excess` | D | CF-AC-02 capacity (T-CONC) | G1, G3, G6 | 2 host threads × 2 streams each; total requested ≤ cap + one chunk; at least one call fails with OOM; sum of successful sizes ≤ the M-2 limit; every successful buffer verifies its data pattern | no RED: characterization |
| `a_buffer_freed_on_one_stream_is_not_reused_before_its_reader_finishes` | D | CF-AC-02 stream order (T-CONC) | G1, G3, G6 | see "stream-order test" below | no RED: characterization of event tracking |
| `the_pool_outlives_its_context_while_slices_live` | D | CF-AC-02 lifetime | G1, G3, G6 | with only the slice holding the pool, `used_current()` reads > 0; after drop and sync, 0 | no RED: characterization |
| `mem_pool_and_private_context_are_send_and_sync` | C | CF-AC-02 Send/Sync (D-6) | fork pure-test job | `CudaMemPool` and `CudaContext` (built with a pool) are `Send + Sync`, by trait-bound functions with no new crate | no RED: compile-time assertion |
| `pool_model_property` | D | CF-AC-02 property | G1, G3 (13000), G6 | see "model test" below | no RED: characterization |
| `device_location_is_the_ordinal_on_every_binding` | P | CF-AC-04 (D-8) | fork pure-test job (19 features) | `type_` = DEVICE, id = ordinal through the union where present | FC5-a: at 13020+ the vendored literal fails to compile, so the RED is the fork test module's compile failure at those features. Recorded as the D-8 defect, not new API |
| `pool_props_are_zeroed_then_written` | P | CF-AC-04 (D-8) | fork pure-test job (19 features) | every `reserved` byte is 0; `maxSize` is set on every feature from 12020; `usage` is 0 | FC5-a: at 13030 and 13040 `maxSize` stays 0 with the vendored cfg list, a behavioural RED once the location fix lets 13030 compile (ordering in § 10) |
| `max_size_without_binding_support_is_not_supported` | P | CF-AC-04 | fork pure-test job (11040–12010) | `create` returns `CUDA_ERROR_NOT_SUPPORTED` before any driver call (counter with positive control) | FC5-b: the vendored code silently ignores it |
| `max_size_on_a_driver_older_than_12020_is_not_supported` | F | CF-AC-04 (D-4) | G1, G3 | a `cfg(test)` driver-version override of 12010 makes `create` return `NOT_SUPPORTED`; the pool-create counter (with positive control) is 0 | FC5-b: the vendored code creates the pool |
| `attributes_newer_than_the_driver_are_never_requested` | F+D | CF-AC-04b (gate) | G3 (13020–13040) | `max_size > 0`; the result-layer counter of requests for attributes > 8 (with positive control) is 0 on driver 13000; a second counter shows the gate branch ran | FC5-b, staged: an ungated read-back is committed first, so this test is RED on G3 13020 on driver 580 (counter > 0); the gate commit turns it GREEN |
| `max_pool_size_read_back_uses_the_driver_value` | F | CF-AC-04 (D-4) | G3 (13020+; the read-back path exists only from the 13020 bindings) | `cfg(test)` seams set the driver version to 13020 and the read-back value. A value ≥ the request gives `max_size() == Some(value)`; a lower value makes `create` fail with `INVALID_VALUE` after destroying the pool | FC5-b |
| `max_pool_size_reads_back_at_least_the_request` | D | CF-AC-04c | G5b (13020–13040) | real read-back ≥ request; `max_size()` is `Some` | no RED at FC5 (needs a newer driver); observed at FC8 on G5b |
| `a_failed_max_pool_size_read_leaves_create_successful` | F | CF-AC-04 (D-4) | G3 (13020+) | a seam-forced read error: `create` succeeds and `max_size()` is `None` | FC5-b |
| `typed_reads_return_64_bit_values` | D | CF-AC-04 (D-4) | G1, G6 | a threshold of 2^33 round-trips through `result::mem_pool::set_attribute` and `release_threshold()` | no RED: the vendored `attribute()` already reads a `u64`; characterization |
| `create_destroys_the_pool_when_setting_the_threshold_fails` | F | CF-AC-02 (D-2) | G1 | the seam fails `set_attribute`; the destroy counter (with positive control) is 1 | no RED: the vendored `create` already owns the pool before `set_attribute`; characterization |
| `a_pool_create_out_of_memory_returns_the_error_and_no_pool` | F | CF-AC-02 (T-FAULT) | G1 | the seam makes `cuMemPoolCreate` return OOM; `create` returns `CUDA_ERROR_OUT_OF_MEMORY` and the destroy counter is 0 | no RED: characterization |
| `destroy_with_outstanding_allocations_is_deferred` | D | design D-2 | G1, G6 | drop the pool handle with a live leaked pointer (held by the guard), free it with `free_async`, synchronize. **Discrete:** `cuMemGetInfo` free memory is at least half the allocation below its start while outstanding, and returns to within one chunk of its starting value. **Tegra:** `VmRSS` is at least half the allocation above its start while outstanding, and returns to within half the allocation of its start (the pool is gone, so its attributes cannot be read; half the allocation, not one chunk, because the chunk of a 256 MiB request is 256 MiB) | no RED on discrete (characterization). Tegra branch: RED is the revision-3 test on G6 at `b29c41c` (`the outstanding allocation was released with the pool: free 52818878464 -> 52818878464`) |
| `drops_after_a_sticky_error_neither_panic_nor_abort` | X | design D-2 (T-FAULT, lost context) | G1, G5b (ignored, run alone) | child process: private pool, live slices, then a device-side assert (as upstream's `test_device_side_assert`); then drops slices, stream, context and pool; exit status 0 and no panic | no RED: characterization |
| `capture_on_a_private_context_uses_graph_memory` | D | design D-9 (T-CAPTURE) | G1, G5b: success required; G6: success or recorded Tegra limit | capture instantiates and launches; graph-memory `USED_MEM_CURRENT` rises; private and default pool counters are unchanged | no RED: characterization. A capture failure on G1 or G5b is a finding needing a design disposition |
| `two_pools_on_two_devices_are_independent` | D | design D-13 (T-MULTI) | G2 | `#[ignore = "must be executed with multiple gpus"]`; pool 0 counters do not move when pool 1 allocates | no RED: characterization |
| `a_pool_of_device_1_is_rejected_for_device_0` | D | design D-13 (T-MULTI) | G2 | same `#[ignore]`; `CUDA_ERROR_INVALID_VALUE` with a real device-1 pool | no RED: the vendored code rejects already |
| `exit_with_a_live_pool_is_clean` | X | design D-2 (T-TEARDOWN) | G1, G6 | child process allocates and exits without dropping; exit status 0 | no RED: characterization |
| `measure_reservation_steps` | D | M-1 | G1, G5b, G6 (ignored) | prints steps | measurement |
| `measure_max_size_semantics` | D | M-2 | G1 (ignored) | prints reserved and used at the first OOM, at thresholds 0 and cap | measurement |
| `observe_new_attributes_on_driver_580` | D | M-3 | G3 13020+ (ignored; the 12060 bindings have no attributes 9–15) | calls `result::mem_pool::get_attribute` for 9–15 directly and prints each result | observation; the gate never relies on it |

**Stream-order test.**

1. Stream B waits on a host latch: a `cuLaunchHostFunc` callback that
   blocks until the test releases it. The function exists in every binding.
2. B enqueues a device-to-host copy of buffer X.
3. X is dropped, and so freed, on stream A.
4. Stream C allocates the same size and writes a different pattern.
5. The latch is released.
6. B's copy must hold X's original pattern. The test records whether C's
   address equals X's: reuse may happen only after B's work.

Without the event dependency B would read C's pattern, so the test fails if
event tracking breaks.

**Model test** (`pool_model_property`):

- **Cases:** 256 seeded cases (seed = case index, printed on failure), each
  with 64 operations: allocate size s, free on a random stream,
  synchronize, trim. A small in-test generator is used, so no new
  dev-dependency is added.
- **The model predicts:**
  - success iff the model's live total plus s is within the M-2 limit;
  - after a synchronize, `USED_MEM_CURRENT` ≥ the sum of live sizes;
  - every buffer's per-buffer pattern is intact when it is freed;
  - the default pool's counters are unchanged throughout;
  - all counters are 0 after drop and synchronize.
- **Why it catches bugs:** allocations routed to the wrong pool, lost frees
  and corrupted buffers all fail.

**Fork-after-init** is not tested in the fork. FathomDB's
T-TEARDOWN-PRODUCT owns it (Slice 31 plan D-15, Orin). The fork's lost-context
case is `drops_after_a_sticky_error_neither_panic_nor_abort`.

### 3.2 Leaked pointers: `fathomdb_private_pool` (CF-AC-03)

| Test | Kind | Cells | Expected |
| --- | --- | --- | --- |
| `a_private_pointer_upgraded_onto_a_default_context_returns_to_its_pool` | D | G1, G6 | private `USED_MEM_CURRENT` = 0 after drop and sync; `check_err()` clean |
| `a_private_pointer_freed_synchronously_releases_its_memory` | D | G1, G6 | after `result::free_sync` following a synchronize: `trim_to(0)` leaves reserved ≤ one chunk; `USED_MEM_CURRENT` printed as an observation (stale on driver 580, design D-2). **Discrete:** `cuMemGetInfo` free memory falls by at least two chunks while live and recovers within one chunk. **Tegra:** `VmRSS` rises by at least two chunks while live and returns to within one chunk of its start (§ 3.1, release oracle); the pointers are held by the guard. Tegra branch RED: the revision-3 test on G6 at `b29c41c` (`precondition: reserved 100663296, free 52810706944 -> 52810706944, chunk 33554432`) |
| `a_synchronous_pointer_upgraded_onto_a_private_context_frees_cleanly` | D | G1, G6 | no error; private pool unchanged |

These are characterization tests with no RED (requirements CF-AC-03).
R-FC-CODE checks the `# Safety` text.

### 3.3 Allocator fallback: `fathomdb_alloc_fallback` (CF-AC-05)

| Test | Kind | Cells | Expected | RED at |
| --- | --- | --- | --- | --- |
| vendored 11 (+ 3 aarch64) | P/D/X | G1 (11), G6 (14) | pass | FC4-port |
| `off_target_constructors_never_query_the_default_pool` | F+D | G1 | the `cuDeviceGetDefaultMemPool` counter (with positive control) is 0 after `new`, `new_non_primary` (≤ 13010), `new_cig` (≤ 13010) and `from_raw_context` | no RED: the vendored x86 path already skips the query; characterization |
| `off_target_choice_equals_the_upstream_inline_rule` | D | G1 | `has_async_alloc()` == (`MEMORY_POOLS_SUPPORTED` > 0) for each constructor | no RED: equality check |
| `off_target_zero_length_sync_behaviour_matches_the_base` | D | G1 + baseline | on the synchronous path, the result kind of `cuMemAlloc(0)` at `C` equals the one recorded at the base | no RED: compared with the base |

### 3.4 Library loading (CF-AC-11)

All REDs here use only base API: the public `get_lib_name_candidates`,
`culib()`, `is_culib_present()` and `/proc/self/maps`. New API (`try_culib`,
the report) is checked only in GREEN.

**T-UNIT** (P; module `fathomdb_lib_major` in `src/lib.rs`; the T-UNIT
job runs 19 features × Linux x86_64 and Windows, with every library
feature enabled). Exactly 10 tests per feature, so the expected set is 10
names per feature, 190 per OS. All compile under 11.x features, `no-std`
and Windows: they use only `std::` through `no_std_compat`, and pure
functions parameterized by OS.

| Test | Checks | RED at |
| --- | --- | --- |
| `in_scope_candidates_never_name_another_major` | for 12.x/13.x features, every in-scope library on both OSes (Windows CUPTI exempt), through `get_lib_name_candidates` and the pure function | FC6-b: the base output contains `.so.12` for 13.x cuBLAS and `cufft64_12.dll` for 12.x |
| `versioned_names_come_before_unversioned_names` | same scope | FC6-b: the base output has the unversioned name first |
| `out_of_scope_and_cuda11_candidates_equal_the_base` | candidates for `cuda`, `nvcuda`, `cudnn`, `nccl`, `cutensor`, `nvToolsExt`, and every library for 11.x majors, equal a parameterized copy of the `ec1cac71` function (major, minor and OS as parameters instead of `env!` and host constants) | no RED: it guards against regressions |
| `version_lines_accept_their_major_and_reject_neighbours` | table-driven: the measured samples of design § 12.4 accepted for their line; CUDA 11 versions (cuSOLVER 11.4.1, cuRAND 10.3.0, CUPTI 18) and the other major's line rejected; a cuFile with no query accepted for 12 and rejected for 13 | FC6-b (pure function; written with its stub returning "accept") |
| `load_error_names_library_major_and_tried_names` | `LibLoadError` message content | FC6-b |
| `windows_dll_names_match_nvidia_wheels` | design § 12.4's Windows table for both majors | FC6-b |
| `is_culib_present_agrees_with_try_culib` | rule 4, for every in-scope library, against a fixture loader that returns chosen results | FC6-b |
| `a_failed_load_is_memoized` | rule 4 / design L2: the fixture loader is called once across two `try_culib` calls that fail | FC6-b |
| `the_report_says_linked_at_build_time_when_linked` | rule 5, the pure formatting path for `dynamic-linking`/`static-linking` | FC6-b |
| `report_address_symbols_are_bound_wherever_the_library_is` | rule 5: the symbol whose address resolves each in-scope library's file (and the driver's) is bound in the checked-in bindings for every `cuda-*` feature that builds the module; cuFile's is not `cuFileGetVersion`, which a pre-1.8 cuFile accepted for 12.x lacks | R-FC-CODE L3 (`480db3f`): cuFile's symbol was `cuFileGetVersion`, unbound for 11040–12020 |

**T-DEV** (D, G1; integration test `tests/fathomdb_lib_major.rs`, whose
ignored tests are run by `fathomdb/scripts/t-dev-lib-major.sh`). Each case
runs its own binary directly, not re-spawned through `current_exe`, with
its own environment, and reads `/proc/self/maps`. The RED body calls only
`culib()` or the upstream wrapper (for example
`cublas::sys::cublasGetProperty`) and then inspects the maps.

| Case | Build | Library setup | Precondition asserted | At `ec1cac71` | At `C` |
| --- | --- | --- | --- | --- | --- |
| (a) | 12060 | decoy dir holding `libcublas.so` → cu13 `libcublas.so.13`, then the cu13 venv's `nvidia/cu13/lib`, first on `LD_LIBRARY_PATH` (cu13 cuBLAS has `RUNPATH $ORIGIN` and needs `libcublasLt.so.13`) | `dlopen("libcublas.so")` in a helper process maps `.so.13` | maps `libcublas.so.13` (RED) | maps the system `libcublas.so.12`; report names it |
| (b) | 13000 | no CUDA 13 cuBLAS reachable; the loader cache gives 12.6 for `libcublas.so` | `ldconfig -p` lists `libcublas.so` → 12.6 | maps `libcublas.so.12` (RED) | typed error; no `libcublas.so.12` mapping |
| (c) | 12060 | binary run as `ld-linux-x86-64.so.2 --inhibit-cache <test binary>` with an empty dir first | no cuBLAS resolvable (helper probe) | panics (RED) | typed error; exit 0 |
| (d) | 12060 | decoy dir holding `libcufile.so` → cu13 `libcufile.so.0`, first on the path; the test loads the decoy in-process first, then closes it (it stays mapped: `DF_1_NODELETE`) | helper maps the cu13 cuFile | maps cu13 cuFile (RED) | `AlreadyLoadedWrongVersion` (the object stays mapped: `DF_1_NODELETE`) |
| (e) | 12060 | driver only as `libcuda.so.1` (G5b host) | `ldconfig -p` has no `libcuda.so` entry and no `libcuda.so` file exists in the loader paths | loads | loads (unchanged); no RED |
| (f) | 12060 | `libnvToolsExt.so.1` only, for `nvtx` | the helper finds it | loads | loads (unchanged); no RED |

Case (d) setup (revision 3): the in-process preload is how FC6-b ran it.
R-FC-CODE found it a faithful reading of R-FC-DES M2 (the already-loaded
check). Without the preload, `C` never reaches the decoy:
`libcufile.so.0` comes first and resolves through the loader cache, so `C`
loads and accepts the system cuFile 1.11 (`/usr/local/cuda-12.6/.../
libcufile.so.1.11.1`), and nothing is mapped from the venv.

**G8** (D, `windchill3`; venvs `cu12-121`, `cu12-124`, `cu12-129`,
`cu13-130`; builds with
`-F cuda-<v>,cublas,cublaslt,curand,cusparse,cusolver,cufile`, which turns
on the non-default library features). For every sub-cell the report path
is in the expected directory and the reported version is in the expected
range, cross-checked against `/proc/self/maps`.

| Sub-cell | Build | Expected |
| --- | --- | --- |
| G8-a | 12060 + cuBLAS 12.4.x, 12.9.x | upstream cuBLAS tests pass; path in the venv; version 12.4.x / 12.9.x |
| G8-b | 12060 + cuBLAS 12.1.x | path in the venv; version 12.1.x; cuBLAS tests pass, or each missing symbol is a typed error through `is_symbol_present`; no panic |
| G8-c | 13000 + cuBLAS 13.0.x | path in the venv; version 13.0.x; cuBLAS tests pass |
| G8-d | 12060, cu13 cuRAND and cuSPARSE first | both rejected and unmapped; system 12.6 copies mapped and reported. With the system copies hidden (`--inhibit-cache`), typed error |
| G8-e | 12060, the process first `dlopen`s cu13 `libcurand.so.10` | `AlreadyLoadedWrongVersion` |
| G8-f | 12060, cuSOLVER with cu13 cuSPARSE first | `WrongMajorDependency` naming cuSPARSE |

The scratch venv `cu13-130` pins:

- `nvidia-cuda-runtime==13.0.96`
- `nvidia-cuda-nvrtc==13.0.88`
- `nvidia-cublas==13.0.2.14`
- `nvidia-curand==10.4.0.35`
- `nvidia-cusparse==12.6.3.3`
- `nvidia-cusolver==12.0.4.66`
- `nvidia-cufile==1.15.1.6`
- `nvidia-nvjitlink==13.0.88`

R-FC-TEST confirmed each of these on PyPI. The cu12 venvs pin the newest
patch release of `nvidia-cublas-cu12` 12.1, 12.4 and 12.9, recorded at
FC4.

**Generator template** (design D-F4, R-FC-DES L10):
`template_loader_tail_matches_the_checked_in_cublas_loader` is a P test in
the `bindings_generator` crate, run by the fork pure-test job. If that
crate cannot host it, `fathomdb/scripts/check-loader-template.sh` runs in
fork CI instead. Either way, the template is rendered for cuBLAS and
compared with `src/cublas/sys/mod.rs`'s loader tail. RED at FC6-b: the
template is edited after the files.

### 3.5 Packaging and proposals

| AC | Command | Expected |
| --- | --- | --- |
| CF-AC-07a (FC8, at `C`) | `cargo package --list --allow-dirty=false` and `cargo publish --dry-run` from a detached checkout of `C` | no path starting with `fathomdb/`; no `.github/workflows/fathomdb-ci.yaml`; dry run succeeds; the file list is saved as `qualification/package-list-<C>.txt` |
| CF-AC-07b (FC10) | the same commands at the R-FC-PUB commit | file list byte-identical to 07a's |
| CF-AC-08 | on `ci/proposal-mem-pool`: the hosted job runs the proposal's pool tests by name, with the expected names listed in `fathomdb/proposal/test-names.txt`; G1's host runs the proposal's device tests k = 3 | names equal; pass; T-PERF numbers recorded (`fathomdb/proposal/benchmark.md`: default, private and synchronous allocation on RTX 3090 and Orin) |
| CF-AC-08b | on `ci/proposal-lib-major`: T-UNIT by name (`fathomdb/proposal-lib-major/test-names.txt`), Linux and Windows; G8 re-run against the proposal build | names equal; G8 expectations as § 3.4 |

## 4. Cells and commands

Every command runs k = 3 times at `ec1cac71` (upstream tests only) and at
`C`. "Ignored one by one" means each ignored test the cell can run is run
alone with `-- --ignored --exact <name>`.

| Cell | Host | Builds | Command |
| --- | --- | --- | --- |
| G1 | `windchill3`, RTX 3090, driver 580, 12.6 | 12060, default features | `cargo test --lib -F cuda-12060 -- --test-threads=1`; ignored one by one |
| G2 | same, `CUDA_VISIBLE_DEVICES=0,1` | 12060 | the multi-GPU ignored tests (§ 2.1, § 3.1) one by one |
| G3 | same | 12060, 13000, 13010, 13020, 13030, 13040; driver-only | `cargo test --lib --no-default-features -F std,driver,dynamic-loading,cuda-<v> -- --test-threads=1`; ignored one by one |
| G4 | same + `cu13-130` venv | 13000, default features | `LD_LIBRARY_PATH=<venv>/…/nvidia/cu13/lib cargo test --lib -F cuda-13000 -- --test-threads=1`, the whole set (NVRTC 13.0 PTX JITs on driver 580); ignored one by one |
| G5a | same + `cu13-13<m>` venvs (nearest PyPI minor, recorded) | 13010–13040, default features | as G4; § 7 lists the UNMEASURED tests |
| G5b | GCP Spot T4 (session below) | 13010–13040, default features | the cross-built test binaries run directly with `LD_LIBRARY_PATH=/usr/local/cuda-13.<m>/lib64` for each build; ignored one by one; T-DEV (e) |
| G6 | Orin, JetPack 6, 12.6 | 12060, default features | every `--lib` test as its own process: `<test binary> --exact <name> --test-threads=1`, and each ignored test with `--ignored --exact <name>` (revision 4: one upstream test segfaults the process on the Orin, F-G6-2, § 9); then, at `C`, the fork modules in one process in sequence, `<test binary> fathomdb_ --test-threads=1`, which also covers every X test's parent |
| G7 | Orin, JetPack 7.2 | 13020/13040 | UNMEASURED, owner ruling Q5 (requirements CF-AC-05b) |
| G8 | as § 3.4 | 12060, 13000 | `fathomdb/scripts/g8-*.sh` per sub-cell |
| fork pure-test job | GitHub-hosted, Linux x86_64 | 19 features, `--no-default-features -F std,driver,dynamic-loading,cuda-<v>` | `cargo test --lib -- fathomdb_private_pool:: --test-threads=1`, P tests only (device tests return early without a device and are excluded by name); expected names per feature from the catalogue. The generator template test also runs here |
| T-UNIT job | GitHub-hosted, Linux x86_64 and Windows | 19 features, every library feature | `cargo test --lib -- fathomdb_lib_major::`; 10 names per feature |
| hosted test job, publish dry run, check, clippy, fmt | plan § 1.4 | — | — |

**The fork pure-test job is new** in revision 2 (R-FC-TEST H4). The fork
plan § 1.4 adds it in the same commit.

**G5b session.** Each session stays within 2 hours of wall clock, and the
receipt records VM creation and deletion times.

- **Driver** (named now; R-FC-TEST M13): Ubuntu 24.04's
  `nvidia-headless-no-dkms-615-server-open` 615.71.09, which supports
  Turing (open kernel modules). Prebuilt modules exist for the GCP kernel
  `7.0.0-1014-gcp` (`linux-modules-nvidia-615-server-open-7.0.0-1014-gcp`).
  - The session first installs `linux-image-7.0.0-1014-gcp` if the image
    boots an older kernel. That kernel change and the driver install share
    one reboot.
  - `scripts/gcp-lab/remote-driver.sh` in FathomDB gains an `--open`
    variant before the session (TDD in FathomDB).
- **Fallback:** if 615 cannot load, 595.91.07 (prebuilt for
  `7.0.0-1011..1014-gcp`) runs the 13010 and 13020 builds. The 13030 and
  13040 builds then **fail CF-AC-04c**. They are not recorded as
  UNMEASURED, and the owner decides what follows.
- **Toolkits:** runtime library packages only (`cuda-cudart-13-<m>`,
  `libcublas-13-<m>`, `libcurand-13-<m>`, `cuda-nvrtc-13-<m>`) for m = 1–4.
  Test binaries are cross-built on `windchill3` (`cargo test --no-run`,
  CPU-only; G5b needs no nvcc) and copied with `gcp-cuda-lab.sh push`.
- **Budget:**

  | Step | Minutes |
  | --- | --- |
  | VM and SSH | 5 |
  | kernel and driver plus reboot | 15 |
  | runtime packages | 10 |
  | tests (4 builds × k = 3, plus ignored and T-DEV (e)) | 60 |
  | receipts and teardown | 10 |
  | **Total** | **100** |

- **If a session cannot finish,** the receipt records what ran. A second
  session needs the owner's approval, because the owner set a 2-hour limit
  per session.

## 5. Per-test dependencies (requirements CF-AC-04b; design D-F1)

All functions below are exported by `libcuda.so.580.178.04`. The version
given is the first cudarc binding that declares the function or value;
the bindings start at 11040.

| Test group (membership) | Driver functions | Values | Layouts |
| --- | --- | --- | --- |
| upstream `core::tests`, `result::tests`, `graph::tests` (inventory) | `cuInit`, `cuDeviceGet{,Count,Attribute}`, `cuDevicePrimaryCtx{Retain,Release_v2}`, `cuCtx{Get,Set}Current`, `cuCtxSynchronize`, `cuCtxDestroy_v2`, `cuStream{Create,Destroy_v2,Synchronize,WaitEvent}`, `cuLaunchHostFunc`, `cuEvent{Create,Record,Query,Synchronize,Destroy_v2}`, `cuMem{AllocAsync,FreeAsync,Alloc_v2,Free_v2,setD8Async,cpyHtoDAsync_v2,cpyDtoHAsync_v2,cpyDtoDAsync_v2,cpyPeerAsync,GetInfo_v2,HostAlloc,FreeHost}`, `cuStream{BeginCapture_v2,EndCapture,IsCapturing}`, `cuGraph{InstantiateWithFlags,Launch,Destroy,ExecDestroy}` (all ≤ 11040); `cuCtxCreate_v4` (12050+) for the non-primary tests | `MEMORY_POOLS_SUPPORTED` 115; event and stream flags; `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`; graph instantiate flags 1, 4 (12000+) and 8 (11070+) | none |
| fork pool tests (§ 3.1, § 3.2) | the above, plus `cuMemPool{Create,Destroy,GetAttribute,SetAttribute,TrimTo}`, `cuMemAllocFromPoolAsync`, `cuDevice{GetDefaultMemPool,GetMemPool}`, `cuDriverGetVersion` (≤ 11040) | pool attributes 4–8 (≤ 11040); 13 `MAX_POOL_SIZE` (13020+, gated); `CU_MEM_LOCATION_TYPE_DEVICE` 1; `CU_MEM_ALLOCATION_TYPE_PINNED` 1; `CU_MEM_HANDLE_TYPE_NONE` 0 | `CUmemPoolProps` (88 B); `CUmemLocation` (8 B; union from 13020) |
| `capture_on_a_private_context_uses_graph_memory` | the above, plus `cuDeviceGetGraphMemAttribute` (≤ 11040) | `CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT` | none |
| `drops_after_a_sticky_error_neither_panic_nor_abort` | the above, plus `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel` (≤ 11040; NVRTC for the assert kernel) | — | none |
| fallback tests (§ 3.3) | `cuDeviceGetDefaultMemPool`, `cuMemAlloc_v2`, `cuMemFree_v2` and the core set | `MEMORY_POOLS_SUPPORTED`; `INTEGRATED` 18 (not used) | none |
| item-9 tests (§ 3.4) | nothing new in the driver; library version queries per design § 12.4 | — | — |
| upstream `launch`, `unified_memory` (default features) | `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuMemAllocManaged`, `cuMemAdvise_v2` and `cuMemPrefetchAsync_v2` (12020+), and the core set | managed attach flags; location types 1–4 (12020+) | `CUmemLocation` |

On driver 580 (`cuDriverGetVersion` 13000):

- Every function and value above is known, except attribute 13 and
  location types 5 and 6 (13020+). The gate keeps attribute 13 from being
  requested, and nothing requests the location types.
- So G3 runs every compiled test at every binding version.
- What depends on the driver is PTX JIT of NVRTC 13.1+ output (§ 7).

## 6. Driver API deprecation status (CF-AC-04a)

NVIDIA's Driver API 13.4 deprecated list was read on 2026-10-12
(<https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/deprecated.html>,
header "v13.4"), and R-FC-TEST re-checked it. It lists mostly texture and
surface references, Direct3D/OpenGL interop, legacy launch and the
`cuParam*` family. **None of the functions the fork calls appears on it.**
The fork's functions are § 5's pool, device, context, stream and
allocation functions, plus `cuDriverGetVersion` and
`cuDeviceGetGraphMemAttribute`. R-FC-CODE greps the fork diff for every
name on the list, and the grep must be empty.

## 7. RUN/UNMEASURED map

| Test set | G1 | G3 12060–13010 | G3 13020–13040 | G4 | G5a | G5b | G6 | G7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upstream driver tests | RUN | RUN | RUN | RUN | RUN | RUN | RUN | UNMEASURED (Q5) |
| upstream `nvrtc`, `cublas`, `cublaslt`, `curand` | RUN | — | — | RUN | RUN | RUN | RUN | UNMEASURED (Q5) |
| upstream `launch`, `unified_memory` | RUN | — | — | RUN | UNMEASURED: PTX JIT of 13.1+ output on driver 580 (MVC) | RUN | RUN | UNMEASURED (Q5) |
| two-GPU tests | in G2 | — | — | — | — | UNMEASURED: one GPU | UNMEASURED: one GPU | — |
| fork pool tests | RUN | RUN | RUN after FC5-a | RUN | RUN | RUN | RUN | UNMEASURED (Q5) |
| gate test (`attributes_newer_…`) | — | — | RUN | — | — | — | — | — |
| real `MAX_POOL_SIZE` read-back | — | — | — | — | — | RUN | — | — |
| fork aarch64-only tests | not compiled | not compiled | not compiled | not compiled | not compiled | not compiled | RUN | UNMEASURED (Q5) |

Every UNMEASURED entry is recorded with the reason shown. "—" means the
cell does not include that set.

## 8. Measurements before the tests

| ID | Where | What | Used by |
| --- | --- | --- | --- |
| M-1 | G1, G5b, G6 | reservation steps at four sizes (`measure_reservation_steps`) | receipts; the bound tests measure their own chunk in-process |
| M-2 | G1, G6 (per platform) | `measure_max_size_semantics`: grow until OOM with one allocation size; record reserved and used at the failure; repeat at release threshold 0 and T = cap. `requests_within_the_cap_succeed` also prints its limit on Tegra | the capacity tests' "M-2 limit"; the doc on `MemPoolProps::max_size` |
| M-3 | G1 | `observe_new_attributes_on_driver_580` | an observation for design D-4 |
| M-4 | G5b | the 615 driver's `cuDriverGetVersion` | CF-AC-04c (must be ≥ each build) |

**M-2 per platform (revision 4).**

- **RTX 3090, driver 580.178.04:** with `maxSize` 64 MiB and 1 MiB
  requests, 64 succeed and the 65th returns `CUDA_ERROR_OUT_OF_MEMORY`
  (limit = cap) at both thresholds (G1 receipt).
- **Jetson AGX Orin, JetPack 6 (L4T R36.5.2), CUDA 12.6
  (`cuDriverGetVersion` 12060):** the usable limit is about a third of the
  cap. With a 64 MiB cap, 32 requests of 1 MiB succeed, at reserved = used
  = 32 MiB, at both thresholds (G6 receipt `b29c41c`). Over a range of caps
  (one pool per process, 1 MiB requests; FC8 G6 root-cause probe
  `m2.c`, reproduced in the new G6 receipt):

  | Cap (MiB) | Usable (MiB) |
  | --- | --- |
  | 32–96 | 32 |
  | 97–192 | 64 |
  | 256 | 96 |
  | 512 | 192 |
  | 1024 | 352 |
  | 4096 | 1376 |

  These fit `ceil(ceil(cap / 32 MiB) / 3) × 32 MiB`. That formula is fitted
  to the measurements; NVIDIA does not document it. M-1 on the Orin equals
  the RTX 3090's (a 32 MiB first step for requests up to 32 MiB).
  Attributes 9 and 10 return `CUDA_ERROR_INVALID_VALUE` on this driver.
  Design D-4 records the consequence for callers.

## 9. Host-dependent and flaky tests

- **Host-dependent (closed list):** `test_pinned_copy_is_faster` and
  `test_default_pinned_host_reads_are_faster_than_write_combined`. They
  always run, and the receipt records the timing pair printed on failure.
- **G6 platform list (revision 4; pending R-FC-CODE-2).** The closed
  host-dependent list above stays exactly two tests. Separately, these
  tests fail on the Orin (G6) at `ec1cac71` and at `b29c41c`, each run as
  its own process (`--exact`; ignored ones with `--ignored --exact`).
  Evidence: `qualification/2026-10-12-G6-pertest-ec1cac71.tsv` and
  `qualification/2026-10-12-G6-pertest-b29c41c.tsv` (one row per test:
  name, kind, exit status, libtest result).

  | Test | Cause on the Orin |
  | --- | --- |
  | `driver::safe::core::tests::test_from_raw_context_bind_to_thread` | upstream defect: `cuCtxCreate_v4` with `ctxCreateParams = NULL` segfaults in the Tegra 12.6 driver (exit 139); a zeroed params struct succeeds (design § 10) |
  | `driver::safe::core::tests::test_from_raw_context_creates_and_destroys` | same `cuCtxCreate_v4(NULL)` segfault |
  | `driver::safe::core::tests::test_new_non_primary_creates_and_destroys` | same |
  | `driver::safe::core::tests::test_new_non_primary_htod_dtoh` | same |
  | `driver::safe::core::tests::test_new_non_primary_cross_thread` | same |
  | `driver::safe::unified_memory::tests::test_unified_memory_host` | `CUDA_ERROR_NOT_PERMITTED` on Tegra |
  | `driver::result::tests::peer_transfer_contexts` (ignored) | one device: `CUDA_ERROR_INVALID_DEVICE` |
  | `driver::result::tests::re_associate_context_for_memory_op` (ignored) | one device: `CUDA_ERROR_INVALID_DEVICE` |
  | `driver::safe::launch::tests::test_peer_memcpy_waits_for_src_stream` (ignored) | one device: `CUDA_ERROR_INVALID_DEVICE` |
  | `driver::safe::core::tests::test_post_alloc_memory` (ignored) | the Tegra free-memory oracle: `cuMemGetInfo` does not move (§ 3.1) |

  - They are always run and recorded, never skipped or filtered
    (CF-REQ-12). Each is a baseline fact in the sense of CF-REQ-12; this
    list only makes the G6 set explicit. A G6 result at `C` that differs
    from the baseline for any of them is a finding.
  - The fork's T-MULTI tests (`two_pools_on_two_devices_are_independent`,
    `a_pool_of_device_1_is_rejected_for_device_0`) are G2 tests, ignored
    and multi-GPU only. They are not G6 tests: the Orin has one device.
    G6 runs them with `--ignored --exact` and records the failure.
  - **Cascade, not a platform failure:**
    `max_size_on_a_driver_older_than_12020_is_not_supported` stays a G1
    and G3 test (§ 3.1). It failed on the Orin only in the fork-module
    sequence at `b29c41c`, with `CUDA_ERROR_OUT_OF_MEMORY` on pool
    creation after earlier failing tests had left pools in deferred
    destruction (§ 3.1, leaked-pointer guard; design D-2). It passed
    alone. `release_threshold_bounds_reserved_after_sync` and
    `trim_to_keeps_the_floor_and_releases_above_it` failed the same way
    and are G6 tests that must pass.
- **Flaky:** none are named now. A test that is flaky at the baseline is
  added here only with R-FC-TEST approval and its baseline runs attached.

## 10. TDD order (FC4–FC7)

| Step | RED | GREEN |
| --- | --- | --- |
| FC4-0 | `count-tests.py` fixture tests | the script; `fork-catalogue.toml`; baseline lock |
| FC4-W | — | commit `W` (fork CI, including the fork pure-test and T-UNIT jobs, which have no tests to run yet) |
| FC4-port | the vendored tests on bare `ec1cac71` (the port's documented exception) | items 1–6 as `L`/`U` commits; tree A check (CF-AC-01) |
| FC5-a | `device_location_is_the_ordinal_on_every_binding` (13020+ compile failure of the vendored literal; the vendored test `pool_props_are_pinned_device_memory_on_the_ordinal` is edited in the same step to use the helper) | the D-8 location helper. Then `pool_props_are_zeroed_then_written` is RED at 13030/13040 (behaviour: `maxSize` stays 0), and the props fix turns it GREEN |
| FC5-b | `max_size_without_binding_support…`, `max_size_on_a_driver_older…`, an ungated read-back commit, then `attributes_newer_than_the_driver…` (RED on G3 13020), `max_pool_size_read_back_uses_the_driver_value`, `a_failed_max_pool_size_read…` | D-4: the not-supported paths, the gate and the read-back handling |
| FC5-c | — (characterization: the vendored code already behaves) | typed reads replace `attribute()`; `typed_reads_…` and `create_destroys_…` added |
| FC5-d | — (doc) | the `upgrade_device_ptr` `# Safety` text; § 3.2 tests |
| FC6-a | — (§ 3.1 and § 3.3 characterization tests; any failure is a defect handled RED/GREEN) | — |
| FC6-b | § 3.4 T-UNIT REDs, the template test, then T-DEV (a)–(d) | item 9 |
| FC7 | — | packaging `P` |

`status.md` gets each RED sha with its failing output, and the reviewer
re-runs it. Tests marked "no RED" are listed there with their reason.
