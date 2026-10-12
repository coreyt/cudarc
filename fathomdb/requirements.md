# cudarc-fathomdb requirements (FC1)

- Status: FC1 draft for review R-FC-REQ, 2026-10-12. It finalizes
  [plan](plan.md) § 4 (revision 3). Where this file and plan § 4 differ, this
  file governs from R-FC-REQ approval on.
- Branch: `fathomdb/0.19.10` of `coreyt/cudarc`, cut from upstream tag
  `v0.19.10` (`ec1cac71`).
- Consumer: FathomDB 0.8.28 Slice 31. FathomDB requirement IDs (`REQ31-nn`)
  are given in brackets.

## 1. Scope

The fork carries items 1–9 of plan § 3 and nothing else:

- the aarch64 Linux allocator fallback (items 1–4);
- the private memory pool and its marker feature (items 5–6);
- the fork CI workflow (item 7);
- packaging as `cudarc-fathomdb` (item 8);
- built-major library loading with a loaded-library report (item 9).

Out of scope:

- new bindings, and regenerating upstream's bindings;
- cuDNN, NCCL and cuTENSOR loading changes. Those libraries follow their own
  version lines (cuDNN 9, NCCL 2, cuTENSOR 2), not the CUDA toolkit's;
- any upstream PR, issue or comment (CF-REQ-10);
- Tegra on CUDA 13 (FathomDB owner ruling Q5).

## 2. Hosts and cells

Acceptance criteria name the cells of plan § 6.2:

- G1–G4 and G8 run on `windchill3` (RTX 3090, driver 580.178.04);
- G6 runs on the Jetson AGX Orin (JetPack 6, CUDA 12.6);
- G5 runs on a GCP Tesla T4 with a newer driver (owner ruling Q-F1,
  2026-10-12; CF-AC-04c);
- G7 is UNMEASURED (owner ruling Q5).

The rules of plan § 1.6 bind every GPU run.

A cloud T4 trial on 2026-10-12 built and tested this branch at `c075191`
with the 12.6 and 13.0 toolkits. That trial informed CF-REQ-11 and CF-REQ-12,
but it is not qualification evidence. See FathomDB
`dev/plans/0.8.28/features/slice-31/research/cloud-gpu-test-options.md`
§ "Fork build trial on a T4".

## 3. Requirements

Each requirement has acceptance criteria (CF-AC). Every CF-AC names a cell
or CI job and a receipt (plan § 7). "RED" means a test that fails at the
named base before the change and passes after it.

### CF-REQ-01 Base and delta [REQ31-01]

The branch starts at `ec1cac71`. Items 1–6 are re-applied as small commits
that reproduce the vendored FathomDB delta. Each fix and the packaging
change is a separate commit.

- **CF-AC-01.** Tree A is both vendored patch files
  (`fathomdb-alloc-fallback.patch`, then `fathomdb-private-pool.patch`, from
  FathomDB `third_party/cudarc-0.19.7/`) applied to `ec1cac71`, with the
  manual resolutions recorded in `status.md`:
  - the marker feature added by hand;
  - the `CudaSlice::drop` hunk placed under #591's gate.

  At the last item commit (before any FC5 fix),
  `git diff <treeA> HEAD -- . ':(exclude)fathomdb' ':(exclude).github/workflows/fathomdb-ci.yaml'`
  is empty. Every later code commit is listed in `status.md` with its class
  and purpose.

### CF-REQ-02 Pool isolation and correctness [REQ31-02]

The private pool must:

- never read or change the device's default pool or current pool;
- free allocations in stream order;
- for allocations owned by `CudaSlice`, outlive every allocation by
  construction (the `Arc` chain slice → stream → context → pool);
- enforce `maxSize` where supported (CF-REQ-04);
- keep the release threshold and trim within their bounds;
- serve zero-length requests;
- be `Send` or `Sync` only where design D-6 justifies it;
- be correct across multiple streams with event tracking.

**CF-AC-02.** On G1 (12060 bindings), G4 (13000 bindings) and G6:

- the default-pool handle (`cuDeviceGetDefaultMemPool`) and the current pool
  (`cuDeviceGetMemPool`) are identical before and after creating, using and
  dropping a private pool;
- after drop and a synchronize, `USED_MEM_CURRENT` and
  `RESERVED_MEM_CURRENT` are both 0;
- with release threshold T, after a synchronize, `RESERVED_MEM_CURRENT` ≤
  max(T, `USED_MEM_CURRENT`), rounded up to the allocation granularity;
- after `trim_to(k)`, `RESERVED_MEM_CURRENT` ≤ max(k, `USED_MEM_CURRENT`),
  with the same rounding;
- a request that would exceed the cap returns
  `DriverError(CUDA_ERROR_OUT_OF_MEMORY)`, also with N concurrent streams
  (N is set in the test plan);
- every exhaustion test requests at most cap + one granule (plan § 6.1);
- the model-based property test passes the case budget the test plan sets.

### CF-REQ-03 Leaked pointers [REQ31-02]

A pointer leaked from a private-pool context (`CudaSlice::leak`) must never
silently become valid input to another context's allocator. Design D-2
either rejects that case or states it as an explicit safety precondition.
The `upgrade_device_ptr` documentation (`src/driver/safe/core.rs`, the
`# Safety` list above `pub unsafe fn upgrade_device_ptr`) must stop implying
that any valid allocation on the device is acceptable.

**CF-AC-03.** The test or doc test chosen in FC2 is RED at the commit before
the fix. The corrected `# Safety` text names the pool or context the pointer
must come from.

### CF-REQ-04 Every CUDA feature [REQ31-03]

- The crate compiles for all 19 `cuda-*` features with the default and the
  `no-std` feature sets, on Linux x86_64, Linux aarch64 and Windows.
- `max_size` takes effect only where both the bindings and the loaded driver
  support it. Elsewhere a nonzero value returns a typed error and is never
  silently ignored. Today it is silently ignored below 12020 (plan § 2).
- No removed or deprecated Driver API is called.
- The `CUmemLocation` layout change (the `id` field moves into a union in the
  13.2 bindings) and the `CUmemPoolProps` layout are handled without struct
  literals (design D-8).
- Behaviour that depends on the driver is selected from `cuDriverGetVersion`
  at run time, never from the binding cfg alone (D-4).

**CF-AC-04a.** Fork CI (plan § 1.4) passes on the qualified commit. Its CI
receipt's `head_sha` equals that commit, and it covers:

- 19 features × {Linux x86_64, Windows} × {default, `no-std`};
- aarch64 Linux check and clippy for all 19 features;
- clippy `-D warnings` and fmt;
- the hosted test job;
- the name-agnostic publish dry run.

**CF-AC-04c (G5, GCP T4, sm_75).** The full `--lib` test set passes on a
newer driver for builds with `CUDARC_CUDA_VERSION` and `-F cuda-<v>` equal
to 13010, 13020, 13030 and 13040, with default features.

- Each build runs on a driver whose `cuDriverGetVersion` is at least that
  build's version. The receipt records:
  - the driver package and version;
  - `cuDriverGetVersion`;
  - the toolkit whose libraries were loaded (CF-REQ-11 report).
- One driver that supports 13.4 may serve all four builds. The test plan
  (FC3) names the branch after checking which branches Ubuntu 24.04 or
  NVIDIA's repository offers for the T4.
- The `MAX_POOL_SIZE` read-back (D-4) is exercised on the 13020+ builds.
- Tests that need hardware the T4 lacks (two GPUs, sm_80+) are recorded
  UNMEASURED with that reason, and CF-REQ-12 applies.

**CF-AC-04b.** The G3 test set passes on driver 580 for builds with
`CUDARC_CUDA_VERSION` = 12060, 13000, 13010, 13020, 13030 and 13040.

- Design D-F1 lists what each test depends on, each item with the CUDA
  version that introduced it:
  - every driver function it calls, checked against `nm -D` of
    `libcuda.so.580.178.04`;
  - every enum, attribute and flag value it passes;
  - every struct layout it relies on.
- A test that depends on something driver 580 lacks is recorded UNMEASURED,
  never counted as passed.
- G3 also proves the driver-version gate: a 13020+ build running on the
  13000 driver never requests `MAX_POOL_SIZE` or any other attribute newer
  than the driver.

### CF-REQ-05 Allocator fallback scope [REQ31-05]

Items 1–4 act only on `cfg(all(target_os = "linux", target_arch = "aarch64"))`.
On every other target, allocation behaviour equals upstream v0.19.10.
Design D-12 decides whether the predicate is narrowed with
`CU_DEVICE_ATTRIBUTE_INTEGRATED`, since it also matches Grace, GB10 and
other SBSA hosts.

**CF-AC-05.** On G1 (x86_64):

- `SYNC_FALLBACK` is `false`;
- every constructor's choice of stream-ordered allocation equals upstream's
  `select_async_alloc`, and a real x86_64 GPU test proves it.

The vendored tree only proved this from a copy with the cfg predicate
flipped. On G6, the fallback's fresh-process tests (T-PROC) pass.

### CF-REQ-05b Tegra [REQ31-04]

The fork's tests pass on the Jetson AGX Orin (JetPack 6, CUDA 12.6,
`CUDARC_CUDA_VERSION=12060`).

**CF-AC-05b.** A G6 receipt covers the § 6.2 Orin set. The Tegra on CUDA 13
cell (G7) reads UNMEASURED and cites owner ruling Q5.

### CF-REQ-06 Minimal delta, upstream APIs [REQ31-06]

- Pool code uses upstream's result-layer `driver::result::mem_pool`
  (`create`, `destroy`, `trim_to`, `get_attribute`, `set_attribute`,
  `alloc_async`; upstream #544) and adds no new bindings.
- No upstream function is duplicated.
- Every carried item has an upstream status and a removal path in
  `FATHOMDB.md`.

**CF-AC-06.** The `FATHOMDB.md` table covers items 1–9. R-FC-DES and
R-FC-CODE each record "no duplicated upstream functionality", or list the
duplicates with their dispositions.

### CF-REQ-07 Packaging [REQ31-06, REQ31-08]

- The crate is published as `cudarc-fathomdb` 0.19.10 with
  `[lib] name = "cudarc"`.
- It keeps the upstream licences (MIT OR Apache-2.0) and attribution, and
  the README carries a fork notice.
- The package contains no `fathomdb/` path and no fork-CI file.
- It is published from the qualified commit (plan § 1.5).

**CF-AC-07a (FC8).** `cargo package --list` and `cargo publish --dry-run`
receipts at `C` show no `fathomdb/` path and no `fathomdb-ci.yaml`.

**CF-AC-07b (FC10).** The same receipts at the commit R-FC-PUB approves.
That commit equals `C` (or a restarted `C`, plan § 1.5), and its file list is
identical to CF-AC-07a's.

### CF-REQ-08 First proposal: the pool primitive [REQ31-09]

- The proposal is item 5 only, without the marker feature.
- It lives on `proposal/mem-pool`, cut from `upstream/main`, in upstream
  style.
- Its tests skip cleanly without a device.
- It brings docs, an example and a benchmark.
- It coexists with upstream PR #594.
- Issue and PR text are drafted in `fathomdb/proposal/`.
- It is not posted.

**CF-AC-08.**

- Fork CI passes on `ci/proposal-mem-pool`, with a CI receipt.
- The GPU set passes on G1 (12.6).
- `git diff --stat upstream/main..proposal/mem-pool` lists no `fathomdb/`,
  `FATHOMDB.md` or `fathomdb-ci.yaml` path.
- No proposal commit carries a `Fork-Class:` trailer or the marker feature.
- R-PROP approves.

### CF-REQ-08b Second proposal: built-major library loading [REQ31-09]

- The proposal is item 9 only, with the loaded-library report.
- It lives on `proposal/lib-major`, cut from `upstream/main`, in upstream
  style, and is independent of `proposal/mem-pool`.
- Its tests run without a device wherever possible.
- Issue and PR text are drafted in `fathomdb/proposal-lib-major/`. They cite
  the cross-major hazard and the upstream issues the old candidate names
  serve (#242, #246, #260, #274, #296).
- It is not posted.

**CF-AC-08b.**

- Fork CI passes on `ci/proposal-lib-major`, including Windows, with a CI
  receipt.
- G8 is re-run against the proposal build.
- The same path and trailer checks as CF-AC-08 pass.
- R-PROP2 approves.

### CF-REQ-09 Process [REQ31-11]

TDD, commit classes, reviews and receipts follow plan §§ 1.3, 7 and 9.

**CF-AC-09.**

- `fathomdb/status.md` lists the red and green shas for each step, and the
  reviewer re-runs each red.
- `fathomdb/review.md` records every plan § 9 review with model `opus`.
- The plan § 1.3 class check passes over `ec1cac71..C`.

### CF-REQ-10 Scope safety [REQ31-12]

**CF-AC-10.**

- There is no upstream PR, issue or comment.
- There is no PR or issue object on the fork.
- Nothing is pushed to `upstream`.
- Every receipt shows the host guard unchanged (plan § 1.6).

### CF-REQ-11 Built-major library loading [REQ31-15]

The rule is defined per library, because CUDA library sonames do not all
follow the toolkit major. These were measured on 2026-10-12: CUDA 12.6 apt
libraries on `windchill3`, and NVIDIA's cu13 PyPI wheels.

| Library | CUDA 12 soname | CUDA 13 soname | Name tells major? |
| --- | --- | --- | --- |
| `cudart`, `cublas`, `cublasLt`, `nvrtc`, `cupti` | 12 | 13 | yes |
| `cufft`, `cusolver`, `cusolverMg` | 11 | 12 | yes |
| `cusparse` | 12 | 12 | no |
| `curand` | 10 | 10 | no |
| `cufile` | 0 | 0 | no |
| `nvToolsExt` (NVTX) | 1 | none (note 1) | no |

Note 1: NVTX 3 is header-only; the cu13 wheel ships only
`libnvtx3interop.so.1`.

Requirements:

1. **Expected names only.** A build for CUDA toolkit major M tries only the
   expected versioned names for that library and major. On Linux that is the
   table's soname. On Windows it is the matching DLL names, which design
   D-F4 lists. It never tries another major's versioned name, so for
   example a 13.x build never tries `libcublas.so.12`.
2. **Unversioned and shared names are verified.** The unversioned name
   (`libcublas.so`), and any name the table marks "no", is accepted only
   after the loaded library's own version query reports CUDA major M, or the
   library-version range D-F4 sets for major M. Examples of version queries:
   `cublasGetProperty`, `nvrtcVersion`, `curandGetVersion`,
   `cusparseGetProperty`, `cudaRuntimeGetVersion`. For cuSPARSE, cuRAND and
   cuFile this check is the only safeguard against a wrong-major library.
3. **Typed failure.** When no acceptable library is found, the failure names
   the library, the major it needs, and the names it tried. It never loads a
   wrong-major library instead. Upstream panics (`panic_no_lib_found`).
   Whether the fork keeps the panic or returns an error is decided in D-F4
   and must stay upstream-shaped.
4. **Loaded-library report.** For each loaded library, a public API exposes
   the resolved file that was loaded and the version it reports
   (FathomDB D-18).
5. **The driver is exempt.** `libcuda.so.1` / `nvcuda.dll` keep upstream's
   loading, because the driver is backward compatible across majors.

**CF-AC-11.**

- **T-UNIT**, for every `cuda-*` feature and every library in the table:
  - the candidate list contains no versioned name of another CUDA major;
  - the version check accepts major M and rejects M ± 1.
  - This runs in the hosted test job, with no GPU.
- **T-DEV (G1).** With a decoy wrong-major library first on the search path,
  it is not used, and the report names the right file. This is RED at
  `ec1cac71`: the T4 trial showed that stock v0.19.10 opens the unversioned
  `libcublas.so` first.
- **G8.** A 12060 build against PyPI cuBLAS 12.1.x, 12.4.x and 12.9.x, and a
  13000 build against cuBLAS 13.0.x, each record the loaded file (from the
  report, cross-checked against `/proc/self/maps`) and its reported version.
  On every host the 12060 build either loads a CUDA 12 cuRAND (which, by
  name, could be a CUDA 13 one) or rejects it by version.

### CF-REQ-12 Honest test accounting [REQ31-11]

Every receipt accounts for every test in its set by name, as passed, failed,
ignored or UNMEASURED with a reason. Some upstream tests encode a host
performance assumption rather than a correctness property. Two failed on a
cloud Tesla T4 in 6 of 6 runs under both toolkits:

- `test_pinned_copy_is_faster`;
- `test_default_pinned_host_reads_are_faster_than_write_combined`.

Such tests:

- are listed by name in the test plan as host-dependent;
- are run, never skipped, and recorded with their measured values;
- count as failures only on a host where the test plan expects them to pass
  (G1 and G6, if they pass there at baseline `ec1cac71`).

A test that fails at baseline on a host is a baseline fact and is recorded
as such, never as a regression. A test that passes at baseline and fails
after a fork commit is a regression.

**CF-AC-12.** For each cell, the receipt includes the same test set run at
`ec1cac71` on the same host and build. Every difference between baseline
and `C` is explained in the receipt. No test is filtered out silently: any
filter, or `--skip` used, is printed in the receipt.

## 4. Traceability

| FathomDB | Fork requirements |
| --- | --- |
| REQ31-01 | CF-REQ-01 |
| REQ31-02 | CF-REQ-02, CF-REQ-03 |
| REQ31-03 | CF-REQ-04 |
| REQ31-04 | CF-REQ-05b |
| REQ31-05 | CF-REQ-05 |
| REQ31-06 | CF-REQ-06, CF-REQ-07 |
| REQ31-08 | CF-REQ-07 |
| REQ31-09 | CF-REQ-08, CF-REQ-08b |
| REQ31-11 | CF-REQ-09, CF-REQ-12 |
| REQ31-12 | CF-REQ-10 |
| REQ31-15 | CF-REQ-11 |

## 5. Changes from plan § 4 draft

- **CF-REQ-11:** "major M" is now defined per library, from measured sonames.
  cuRAND keeps `.so.10` and cuSPARSE keeps `.so.12` across CUDA 12 and 13, so
  a name-only rule would reject the right cuRAND or accept the wrong
  cuSPARSE. The version check is now mandatory for those libraries. cuDNN,
  NCCL and cuTENSOR are out of scope. The failure must be typed and must name
  the library, the major and the names tried.
- **CF-REQ-12** is new: per-test accounting, a same-host baseline at
  `ec1cac71`, and host-dependent upstream timing tests (T4 evidence).
- **CF-AC-02** also checks the current pool (`cuDeviceGetMemPool`) and bounds
  exhaustion requests.
- **CF-AC-05** requires a real x86_64 GPU proof that allocation selection
  equals upstream. The vendored tree had only a flipped-predicate copy.
- **CF-REQ-04** requires driver-dependent behaviour to be selected at run
  time from `cuDriverGetVersion`.

## 6. Owner rulings

- **Q-F1 Cloud venue for G5 (ruled 2026-10-12): run it on a GCP T4 with a
  newer driver.** CUDA 13.1–13.4 builds need a driver newer than R580, which
  `windchill3` can never take (Q4, Q11). The driver change happens only on a
  lab VM (FathomDB `scripts/gcp-cuda-lab.sh`: Spot, labeled, capped by
  `--max-run-duration`, deleted on termination), never on `windchill3`. Each
  T4 session has a 2-hour wall-clock limit, and the receipt records the VM
  creation and deletion times.
