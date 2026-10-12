# cudarc-fathomdb requirements (FC1)

- Status: FC1 revision 4, 2026-10-12.
  - Revision 1 (`f774dc3`) was reviewed by R-FC-REQ (APPROVE-WITH-FIXES).
  - Revision 2 (`761d7a4`) fixed every finding.
  - Revision 3 (`8259c14`) applied the FC2 research.
  - Revision 4 applies the R-FC-DES findings ([design](design.md) § 13).

  Dispositions are in [review.md](review.md). The file finalizes
  [plan](plan.md) § 4. Where it and plan § 4 differ, this file governs.
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
- loading changes for the driver (`cuda`, `nvcuda`) and for libraries that
  follow their own version lines rather than the CUDA toolkit's: cuDNN (9),
  NCCL (2), cuTENSOR (2) and NVTX (`nvToolsExt`, 1). Their candidate names
  stay exactly as at `ec1cac71` (CF-AC-11, T-UNIT);
- loading changes for CUDA 11.x builds (`cuda-11040` … `cuda-11080`). They
  keep upstream's candidate names, because FathomDB's library floor is CUDA
  12 (FathomDB ruling Q10);
- any upstream PR, issue or comment (CF-REQ-10);
- Tegra on CUDA 13 (FathomDB owner ruling Q5).

## 2. Hosts, cells and guards

Acceptance criteria name the cells of plan § 6.2:

- G1–G4 and G8 run on `windchill3` (RTX 3090, driver 580.178.04);
- G5 has two required parts (owner ruling Q-F1, § 6): G5a on `windchill3`
  (driver 580) and G5b on a GCP Tesla T4 with a driver newer than R580;
- G6 runs on the Jetson AGX Orin (JetPack 6, CUDA 12.6);
- G7 is UNMEASURED (owner ruling Q5).

The rules of plan § 1.6 bind every GPU run. The host guard is recorded at the
start and end of each run:

- **`windchill3` and lab VMs:** the driver version from `nvidia-smi` and
  `readlink -f /usr/local/cuda`. On a lab VM the driver is expected to be
  the one installed for the cell.
- **Orin**, which has no `nvidia-smi`: the L4T release
  (`/etc/nv_tegra_release`), the JetPack version, and
  `readlink -f /usr/local/cuda`.

A cloud T4 trial on 2026-10-12 built and tested this branch at `c075191`,
whose code tree equals `ec1cac71`, with the 12.6 and 13.0 toolkits. It
informed CF-REQ-11 and CF-REQ-12, but it is not qualification evidence. See
FathomDB `dev/plans/0.8.28/features/slice-31/research/cloud-gpu-test-options.md`
§ "Fork build trial on a T4".

## 3. Requirements

Each requirement has acceptance criteria (CF-AC). Every CF-AC names a cell
or CI job and a receipt (plan § 7). "RED" means a test that fails at the
named base and passes after the change. A RED for a behaviour change must
fail on the behaviour itself, not only because a new API does not compile
at the base.

### CF-REQ-01 Base and delta [REQ31-01]

The branch starts at `ec1cac71`. Items 1–6 are re-applied as small commits
that reproduce the vendored FathomDB delta. Each fix and the packaging
change is a separate commit.

**CF-AC-01.** Tree A is both vendored patch files
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

**CF-AC-02.** On G1 (12060 bindings, default features), G3 (13000 bindings,
driver-only feature set) and G6, each clause has its own test:

- **Isolation.** The default-pool handle (`cuDeviceGetDefaultMemPool`) and
  the current pool (`cuDeviceGetMemPool`) are read before creation, while
  private-pool allocations are live (from the allocating thread and from a
  second thread), and after drop. All reads are equal.
- **Accounting.** After drop and a synchronize, `USED_MEM_CURRENT` and
  `RESERVED_MEM_CURRENT` are both 0.
- **Live set for the bounds.** The driver cannot release an OS chunk that
  backs a live allocation. So the threshold and trim tests first free
  everything, synchronize, and leave exactly one live allocation. "One
  chunk" is the reservation step `test-plan.md` measures for that
  allocation size (design D-5).
- **Threshold.** The test first drives `RESERVED_MEM_CURRENT` above
  max(T, `USED_MEM_CURRENT`) and asserts that precondition. After a
  synchronize, `RESERVED_MEM_CURRENT` ≤ max(T, `USED_MEM_CURRENT`) + one
  chunk.
- **Trim.** `trim_to(k)` sets a floor, not a ceiling: the pool keeps at
  least k bytes reserved if it had more (DRV13.4, `cuMemPoolTrimTo`). With
  the precondition `RESERVED_MEM_CURRENT` > max(k, `USED_MEM_CURRENT`)
  before the call, after it and a synchronize:
  k ≤ `RESERVED_MEM_CURRENT` ≤ max(k, `USED_MEM_CURRENT`) + one chunk.
- **Capacity.**
  - Requests whose total is at most the cap succeed.
  - A request that would exceed the cap returns
    `DriverError(CUDA_ERROR_OUT_OF_MEMORY)`.
  - The same holds with N ≥ 2 concurrent streams.
  - No exhaustion test requests more than cap + one granule (plan § 6.1).
- **Zero length.** A zero-length allocation succeeds, and its drop is a
  no-op.
- **Stream order.** A buffer freed on stream A while stream B still reads it
  through an event dependency is not reused before B completes.
- **Lifetime.** Dropping the context and pool handles while `CudaSlice`s are
  live leaves the pool alive until the last slice drops. This is checked
  through the pool attributes, not by assumption.
- **`Send`/`Sync`.** A compile-time test asserts exactly the impls that D-6
  justifies.
- **Property test.** The model-based test runs at least 256 cases. The test
  plan may raise that number and must state the seed policy.

### CF-REQ-03 Leaked pointers [REQ31-02]

NVIDIA allows freeing across the two allocator families. `cuMemFree`
accepts `cuMemAllocAsync` and `cuMemAllocFromPoolAsync` pointers once all
accesses have completed, and `cuMemFreeAsync` accepts `cuMemAlloc` pointers
(CUDA 12.6 `cuda.h`, `cuMemFree`; Programming Guide 13.4). A pointer leaked
from a private-pool context (`CudaSlice::leak`) and upgraded onto any
context of the same device is therefore freed correctly. The `# Safety`
list above `pub unsafe fn upgrade_device_ptr`
(`src/driver/safe/core.rs:2590-2599` at `ec1cac71`) must state the real
preconditions:

- same device;
- an allocation base of at least `len * size_of::<T>()` bytes, or 0 with
  `len == 0`;
- every access ordered before the free;
- either allocator family.

It must cite NVIDIA.

**CF-AC-03.** The corrected `# Safety` text is present. Characterization
tests on G1 pass (design D-2):

- a private-pool pointer upgraded onto a `CudaContext::new` context;
- a private-pool pointer freed with `result::free_sync` after a
  synchronize;
- a `result::malloc_sync` pointer upgraded onto a private context.

After each, the private pool's `USED_MEM_CURRENT` returns to 0 and
`check_err()` is clean. No behaviour changes, so these tests need no RED;
the TDD exception is recorded in `status.md`.

### CF-REQ-04 Every CUDA feature [REQ31-03]

- The crate compiles for all 19 `cuda-*` features with the default and the
  `no-std` feature sets, on Linux x86_64, Linux aarch64 and Windows.
- `max_size` takes effect only where both the bindings and the loaded driver
  support it. Elsewhere a nonzero value returns a typed error and is never
  silently ignored. In the vendored delta, it is ignored below 12020 and,
  because the `set_max_size` cfg list stops at `cuda-13020`, also for 13030
  and 13040. A compile failure at 13020+ hides that today (plan § 2).
- No removed or deprecated Driver API is called.
- The `CUmemLocation` layout change (the `id` field moves into a union in the
  13020+ bindings) and the `CUmemPoolProps` layout are handled without struct
  literals (design D-8).
- Behaviour that depends on the driver is selected at run time from
  `cuDriverGetVersion`, never from the binding cfg alone (D-4).

**CF-AC-04a.** Fork CI (plan § 1.4) passes on the qualified commit. Its CI
receipt's `head_sha` equals that commit, and it covers:

- 19 features × {Linux x86_64, Linux aarch64, Windows} × {default, `no-std`};
- aarch64 Linux clippy for all 19 features;
- clippy `-D warnings` and fmt;
- the hosted test job and the T-UNIT job (CF-AC-11);
- the name-agnostic publish dry run.

Removed or deprecated APIs: `test-plan.md` lists every Driver API function
the fork calls, with its deprecation status in the CUDA 13.4 Driver API
reference (design D-F1).
R-FC-CODE records a grep of the fork diff showing that none of them is
deprecated or removed.

**CF-AC-04b (G3).** The G3 test set passes on driver 580 for builds with
`CUDARC_CUDA_VERSION` = 12060, 13000, 13010, 13020, 13030 and 13040.

- `test-plan.md` lists what each test depends on (design D-F1), each item
  with the CUDA version that introduced it:
  - every driver function it calls, checked against `nm -D` of
    `libcuda.so.580.178.04`;
  - every enum, attribute and flag value it passes;
  - every struct layout it relies on.
- A test that depends on something driver 580 lacks is recorded UNMEASURED,
  never counted as passed.
- G3 also proves the driver-version gate: a 13020+ build running on the
  13000 driver never requests `MAX_POOL_SIZE` or any other attribute newer
  than the driver. A `cfg(test)` counter on the attribute calls shows this,
  and the test fails if the counter is never reached.

**CF-AC-04c (G5).** Builds with `CUDARC_CUDA_VERSION` and `-F cuda-<v>` equal
to 13010, 13020, 13030 and 13040, with default features, are tested in two
required parts.

**G5a (`windchill3`, RTX 3090, driver 580).**

- CUDA 13.x libraries come from a scratch venv with pinned NVIDIA wheels for
  each build's minor, reached by `LD_LIBRARY_PATH`. If a minor's wheels are
  not on PyPI, the nearest available 13.x minor is used and recorded.
- The fork's own NVRTC-dependent tests compile to CUBIN for sm_86, so they
  need no PTX JIT.
- Every test that needs neither PTX JIT of 13.1+ output nor driver
  behaviour newer than 13.0 must pass.
- Upstream tests that JIT PTX (`launch` and `unified_memory`) cannot be
  changed to CUBIN, because CF-REQ-12 forbids editing upstream tests. G5a
  records them, and any other test D-F1 marks, UNMEASURED with the reason.

**G5b (GCP T4, sm_75, newer driver).** The full `--lib` test set passes.

- Each build runs on a driver whose `cuDriverGetVersion` is at least that
  build's version. The receipt records:
  - the driver package and version;
  - `cuDriverGetVersion`;
  - the CUDA libraries loaded (the CF-REQ-11 report).
- One driver that supports 13.4 may serve all four builds. The test plan
  (FC3) names the branch after checking which branches Ubuntu 24.04 or
  NVIDIA's repository offers for the T4.
- This cell measures what driver 580 cannot:
  - NVRTC 13.1+ output JIT-loaded by the driver;
  - the `MAX_POOL_SIZE` read-back on a driver that supports it (D-4, 13020+
    builds);
  - the `CUmemLocation` union path on a driver that knows it.
- Tests that need hardware the T4 lacks (two GPUs, sm_80+) are recorded
  UNMEASURED with that reason. CF-REQ-12 applies.

### CF-REQ-05 Allocator fallback scope [REQ31-05]

Items 1–4 act only on `cfg(all(target_os = "linux", target_arch = "aarch64"))`.
On every other target, allocation behaviour equals upstream v0.19.10, in
which each constructor computes `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED > 0`
inline (`src/driver/safe/core.rs:78,150,201,238` at `ec1cac71`). Design D-12
decides whether the predicate is narrowed with
`CU_DEVICE_ATTRIBUTE_INTEGRATED`, since it also matches Grace, GB10 and
other SBSA hosts.

**CF-AC-05.**

- **On G1 (x86_64):**
  - `SYNC_FALLBACK` is `false`;
  - a `cfg(test)` counter shows that no constructor calls
    `cuDeviceGetDefaultMemPool`;
  - each constructor's stream-ordered choice equals the upstream inline rule
    for that device;
  - a zero-length synchronous allocation behaves as at `ec1cac71`, checked by
    running the same test at the base (CF-REQ-12).
- **On G6:** the fallback's fresh-process tests (T-PROC) pass.

### CF-REQ-05b Tegra [REQ31-04]

The fork's tests pass on the Jetson AGX Orin (JetPack 6, CUDA 12.6,
`CUDARC_CUDA_VERSION=12060`).

**CF-AC-05b.** A G6 receipt covers the plan § 6.2 G6 test set. The Tegra on
CUDA 13 cell (G7) reads UNMEASURED and cites owner ruling Q5.

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

### CF-REQ-07 Packaging [REQ31-06, REQ31-08, REQ31-14]

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

### CF-REQ-08 First proposal: the pool primitive [REQ31-09, REQ31-13]

- The proposal is item 5 only, without the marker feature.
- It lives on `proposal/mem-pool`, cut from `upstream/main`, in upstream
  style.
- Its tests skip cleanly without a device.
- It brings docs, an example and a benchmark (T-PERF).
- It coexists with upstream PR #594.
- Issue and PR text are drafted in `fathomdb/proposal/`.
- It is not posted.

**CF-AC-08.**

- Fork CI passes on `ci/proposal-mem-pool`, with a CI receipt. Its hosted
  test job runs the proposal's own tests by name, with a nonzero count.
- The GPU set passes on G1's host and build, against `proposal/mem-pool`.
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
  serve:
  - #242: Windows fixed-major DLLs;
  - #246: `nvrtc64_120_0`;
  - #260: `cudnn64_9`;
  - #274: Linux versioned sonames without dev links;
  - #296: `.so.1`, as for the driver and NVTX.
- It is not posted.

**CF-AC-08b.**

- Fork CI passes on `ci/proposal-lib-major`, including Windows, with a CI
  receipt. Its T-UNIT job meets CF-AC-11's count rule.
- G8 is re-run against the proposal build.
- The same path and trailer checks as CF-AC-08 pass.
- R-PROP2 approves.

### CF-REQ-09 Process [REQ31-11, REQ31-14]

TDD, commit classes, reviews and receipts follow plan §§ 1.3, 7 and 9.

**CF-AC-09.**

- `fathomdb/status.md` lists the red and green shas for each step, and the
  reviewer re-runs each red.
- `fathomdb/review.md` records every plan § 9 review with model `opus`.
- The plan § 1.3 class check passes over `ec1cac71..C`.
- R-FC-VERIFY checks CF-AC-01 to 06, 07a and 09 to 12 at `C`.

### CF-REQ-10 Scope safety [REQ31-12]

**CF-AC-10.**

- There is no upstream PR, issue or comment.
- There is no PR or issue object on the fork.
- Nothing is pushed to `upstream`.
- Every receipt shows the host guard (§ 2) unchanged.

### CF-REQ-11 Built-major library loading [REQ31-15]

The in-scope libraries are those cudarc loads through
`get_lib_name_candidates` (`src/lib.rs:204` at `ec1cac71`), minus the
out-of-scope ones in § 1. Their soname majors were measured on 2026-10-12
from the CUDA 12.6 apt libraries on `windchill3` and NVIDIA's cu13 PyPI
wheels:

| Library | CUDA 12 soname | CUDA 13 soname | Name tells major? |
| --- | --- | --- | --- |
| `cudart`, `cublas`, `cublasLt`, `nvrtc`, `cupti` | 12 | 13 | yes |
| `cufft`, `cusolver`, `cusolverMg` | 11 | 12 | yes |
| `cusparse` | 12 | 12 | no |
| `curand` | 10 | 10 | no |
| `cufile` | 0 | 0 | no |

For cuSPARSE, cuRAND and cuFile, the soname does not tell CUDA 12 from CUDA
13. NVIDIA guarantees ABI stability only within one major (CUDA 13.0 release
notes; minor-version compatibility guide). CUDA 13's cuSPARSE and cuFFT
drop pre-Turing GPUs. The cu13 cuRAND and cuSPARSE contain only sm_75+ code,
and the cu13 cuSPARSE needs `libnvJitLink.so.13`. Design D-F4 (§ 12.4)
gives the evidence.

These rules apply to CUDA 12.x and 13.x builds:

1. **Expected versioned names first.** For library L and toolkit major M,
   the candidates are L's versioned names for M (the table's soname on
   Linux, including cuFile's `.so.0`, which `ec1cac71` never tries; the
   DLL names D-F4 lists on Windows). They come before any
   unversioned name, and no versioned name of another major is ever tried.
   For example, a 13.x build never tries `libcublas.so.12`.
2. **Unversioned names are verified.** An unversioned name (`libcublas.so`,
   `cublas64.dll`) is accepted only if the loaded library's version query
   reports major M (or the library-version range D-F4 sets for M). If it
   does not, the handle is closed and loading continues. A library with no
   usable version query is never accepted by an unversioned name, only by
   its versioned name. D-F4 gives a per-library table: the query and the
   accepted range, or "versioned name only". "Versioned name only" applies
   to `cusolverMg`, which exports no version query. The check calls the
   query through `dlsym` on the candidate's own handle, so it works even
   where the binding is cfg-limited (`cuFileGetVersion` before 12030).
   Windows CUPTI keeps upstream's candidates (year-named DLLs; D-F4).
   For 12.x the accepted lines have lower bounds, so CUDA 11.x libraries
   that share a soname (cuSOLVER, cuRAND, cuFile, CUPTI) are rejected.
   cuSolverMg `.so.11` is a known gap (D-F4). A cuFile without
   `cuFileGetVersion` is accepted for 12.x with reason `NoVersionQuery` and
   rejected for 13.x.
3. **Same-soname libraries are checked by version line.** For cuSPARSE,
   cuRAND and cuFile, a library found by its versioned name is also
   version-checked against the built major's line (D-F4 ranges). A library
   of the other major's line is rejected and closed, on the evidence above.
   The 13.x ranges are open-ended, so later 13.x minors are accepted.
   - **Already loaded.** An object already loaded under that soname (by
     another component, or kept by `DF_1_NODELETE`) is checked first. If it
     is wrong, loading fails with a typed error, because the process cannot
     open another file of that soname.
   - **Dependencies (Linux).** Libraries that an accepted library pulls in
     and that are in scope are version-checked. A wrong one fails the load
     with a typed error naming the dependency.
4. **A typed, non-panicking path.** For each in-scope library, there is a
   fallible check that applies exactly the acceptance rule `culib()` uses
   and returns a typed error naming:
   - the library;
   - the major it needs;
   - every name tried, and why each was rejected.

   `is_culib_present()` gives the same answer as that check. A consumer
   that runs the check first never reaches upstream's panic. A symbol that
   the loaded library lacks also surfaces through a fallible check, never
   only as a panic or abort. D-F4 decides the API shape, which must stay
   upstream-shaped.
5. **Loaded-library report.** A public API exposes, for each in-scope
   library loaded under `dynamic-loading`/`fallback-dynamic-loading`:
   - the CUDA major it was checked against;
   - the name opened;
   - the canonical path of the file actually loaded, comparable with
     `/proc/self/maps`;
   - its structured version (major, minor, patch), or "none" with the
     reason.

   The report also covers the driver (resolved file and
   `cuDriverGetVersion`), without changing how the driver loads. Under
   `dynamic-linking` or `static-linking` the report says "linked at build
   time" (`build.rs:83-100`).
6. **Public surface.** `get_lib_name_candidates` (`pub`, `src/lib.rs:204`)
   keeps its signature. Its output for out-of-scope names, and for CUDA
   11.x builds, is unchanged. Any change to its output for in-scope names
   is an intended behaviour change, listed in `FATHOMDB.md` and in the
   second proposal.
7. **Testable without the target.** Candidate generation is a pure function
   of (target OS, pointer width, library, major, minor), so the Windows
   names are unit-tested on Linux as well.

**CF-AC-11.**

- **T-UNIT job** (fork CI, Linux x86_64 and Windows; plan § 1.4). For each
  of the 19 `cuda-*` features, with every library feature enabled, the job
  runs the item-9 test module by name. Its receipt asserts a nonzero count
  per feature, equal to the expected count in the test plan. The tests
  check:
  - for 12.x and 13.x features, every in-scope library on Linux and Windows
    except Windows CUPTI (which keeps upstream's candidates): the candidate
    list has no other major's versioned name, and versioned names come
    first;
  - the version-acceptance function accepts the built major's line and
    rejects the lines of M − 1 and M + 1 for every library with a query,
    including the CUDA 11.x versions that share a soname;
  - candidate lists for `cuda`, `nvcuda`, `cudnn`, `nccl`, `cutensor` and
    `nvToolsExt`, and for every library under 11.x features, equal those
    at `ec1cac71`;
  - the typed error's message names the library, the major and the names
    tried.
- **T-DEV (G1)**, observed through `/proc/self/maps`, not through the new
  API. The CUDA 13 cuBLAS comes from a scratch cu13 venv (as G4).
  - (a) A 12060 build, with a directory first on `LD_LIBRARY_PATH` holding
    an unversioned `libcublas.so` → cu13 `libcublas.so.13`. At `ec1cac71`
    the process maps `libcublas.so.13`: RED. After the change it maps the
    system's `libcublas.so.12`, and the report names it.
  - (b) A 13000 build where no CUDA 13 cuBLAS is reachable, and the
    unversioned name resolves to 12.6 through the loader cache. At
    `ec1cac71` the process maps `libcublas.so.12`: RED. After the change the
    fallible check returns the typed error, and no `libcublas.so.12`
    mapping remains.
  - (c) No acceptable library at all: the check returns the typed error,
    and the process neither panics nor aborts.
  - (d) A 12060 build with cu13 `libcufile.so.0` first on the path: it is
    rejected, stays mapped (`DF_1_NODELETE`), and the check returns
    `AlreadyLoadedWrongVersion`.
- **Driver with only `libcuda.so.1`** (a lab VM or container where no
  `libcuda.so` dev link exists). The driver loads at `C` as at `ec1cac71`.
- **G8 (`windchill3`, scratch venvs).** For each sub-cell, the loaded file
  is the venv's (from the report, cross-checked against `/proc/self/maps`)
  and its reported version is in the expected range:
  - 12060 build with cuBLAS 12.4.x and 12.9.x: the cuBLAS tests pass;
  - 12060 build with cuBLAS 12.1.x: the cuBLAS tests pass, or each missing
    symbol is a typed error. The receipt names which. No panic or abort;
  - 13000 build with cuBLAS 13.0.x: the cuBLAS tests pass;
  - 12060 build with the cu13 wheels' `libcurand.so.10` and
    `libcusparse.so.12` first on the path: both are rejected by version
    (rule 3) and unmapped. The unversioned names then resolve to the
    system's CUDA 12.6 copies, which `/proc/self/maps` shows and the report
    names. With no CUDA 12 copy reachable, the check returns the typed
    error.
  - 12060 build in a process that first `dlopen`s the cu13
    `libcurand.so.10`: the check returns `AlreadyLoadedWrongVersion`.
  - 12060 build loading cuSOLVER with the cu13 `libcusparse.so.12` first on
    the path: the check returns `WrongMajorDependency` naming cuSPARSE.

### CF-REQ-12 Honest test accounting [REQ31-11]

Every receipt accounts for every test in its set by name, as passed, failed,
ignored or UNMEASURED with a reason.

- **Baseline.** For each cell, the baseline is the upstream tests at
  `ec1cac71`, built with the same `CUDARC_CUDA_VERSION`, the same upstream
  feature set and the same `Cargo.lock`
  (`fathomdb/qualification/Cargo.lock.baseline`, committed at FC4, because
  upstream ships no lockfile), on the same host. Tests the fork
  adds have no baseline. They must pass at `C`.
- **Run count.** Baseline and `C` each run k ≥ 3 times. A test that fails
  in every baseline run is a baseline fact. A test that passes in every
  baseline run and fails in any run at `C` is a regression. Any other
  pattern is flaky; the test plan must name it, or it blocks.
- **Host-dependent tests: a closed list.** Exactly two upstream tests assert
  a host performance relation rather than a correctness property:
  - `test_pinned_copy_is_faster`;
  - `test_default_pinned_host_reads_are_faster_than_write_combined`.

  On a cloud Tesla T4, each failed in all 8 runs: per toolkit (12.6 and
  13.0), one full run and three reruns. They are always run, and their
  measured values are recorded. Adding a test to this list needs baseline
  evidence and R-FC-TEST approval.
- **No hiding.**
  - The fork never edits, removes, `#[ignore]`s or filters an upstream test.
    R-FC-CODE checks the upstream test diff.
  - Ignored upstream tests that a cell can run are run with `--ignored`
    and `--exact`. Examples: the multi-GPU tests on G2, and "must be
    executed by itself" tests run one per process.
  - Each cell's test set and expected per-module counts are fixed in
    `test-plan.md`. The receipt checks cargo's per-module counts, including
    "filtered out", against them.

**CF-AC-12.** Each cell's receipt contains:

- the baseline and `C` results, k ≥ 3 runs each;
- per-module counts matching `test-plan.md`;
- the closed host-dependent list with measured values;
- an explanation for every difference between baseline and `C`.

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
| REQ31-13 | CF-REQ-08 (T-PERF, the proposal benchmark) |
| REQ31-14 | CF-REQ-07 (CF-AC-07a), CF-REQ-09 (class check), CF-AC-04a (CI receipt) |
| REQ31-15 | CF-REQ-11 |

## 5. Plan changes this file requires

Made in the same commit as this revision, in [plan](plan.md):

- § 1.4: the T-UNIT job (CF-AC-11); aarch64 × {default, `no-std`};
- § 6.2: G5 split into G5a (`windchill3`) and G5b (GCP T4) (Q-F1);
- § 8: a row for item 9 code (T-UNIT job, T-DEV, G8, fork CI);
- § 9: FC8 and R-FC-VERIFY cover CF-AC-01 to 06, 07a and 09 to 12.

## 6. Owner rulings

- **Q-F1 Venue for G5 (ruled 2026-10-12): run it locally on `windchill3`
  and, in addition, on a GCP T4 with a newer driver.**
  - **Rationale.** Under minor-version compatibility, CUDA 13.x libraries
    already run on driver 580. The T4 trial loaded cuBLAS 13.1.1.3 on
    580.178.04, and its tests passed. What driver 580 cannot do:
    - JIT-load PTX that NVRTC 13.1+ produced;
    - support behaviour that newer drivers add, such as the `MAX_POOL_SIZE`
      read-back on 13.2+.

    `windchill3` can never take a newer driver (Q4, Q11).
  - **Where the driver changes.** Only on a lab VM (FathomDB
    `scripts/gcp-cuda-lab.sh`: Spot, labeled, capped by
    `--max-run-duration`, deleted on termination), never on `windchill3`.
  - **Limits.** Each T4 session has a 2-hour wall-clock limit, and the
    receipt records the VM creation and deletion times.
  - **Both venues are required (owner clarification, 2026-10-12).** G5a runs
    locally on `windchill3` with 13.x wheels in a scratch venv and NVRTC
    compiling to CUBIN, covering everything driver 580 can run. G5b runs on
    the T4 in addition to G5a, never instead of it.
