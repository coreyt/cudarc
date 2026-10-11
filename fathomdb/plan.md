# cudarc-fathomdb fork plan: the private memory pool on cudarc v0.19.10

- Repository: `coreyt/cudarc` (a GitHub fork of `chelsea0x3b/cudarc`)
- Branch: `fathomdb/0.19.10`, cut from upstream tag `v0.19.10` (`ec1cac71`)
- Owner: Corey Thompson. Consumer: FathomDB 0.8.28 Slice 31
  (`fathomadb/fathomdb`, `dev/plans/0.8.28/features/slice-31/plan.md` on
  `release/0.8.28`).
- Status: plan revision 2 (2026-10-11), with FathomDB Slice 31 plan
  revision 9. Revision 1 was reviewed by R0-rev8 (serving as R-FC-PLAN,
  APPROVE-WITH-FIXES; findings and dispositions in `fathomdb/review.md`).

This directory (`fathomdb/`) is the fork's own engineering record. Fork work
is planned, specified, designed, test-planned, coded (test first), reviewed
and qualified here, in this repository. FathomDB's Slice 31 tracks this
work and consumes its result. It does not re-run cudarc's own tests. Where
this plan and the FathomDB plan disagree about fork work, this plan governs;
where they disagree about FathomDB, the FathomDB plan governs.

## 1. Fork policy (every branch of `coreyt/cudarc`)

### 1.1 Remotes and branches

- `origin` is `coreyt/cudarc`. `upstream` is `chelsea0x3b/cudarc`, fetch only.
  Its push URL is set to `DISABLED-never-push-upstream` in every clone.
- `main` mirrors upstream. It only fast-forwards to an upstream tag or
  `upstream/main`, and never carries a fork commit.
- `fathomdb/<upstream version>` is the product branch FathomDB consumes,
  cut from that upstream tag. It is never rewritten or force-pushed.
- `proposal/<topic>` holds the tidy upstream proposal, cut from
  `upstream/main`. It contains upstream-shaped commits only.
- `ci/<topic>` is a throwaway branch: a proposal branch plus the fork-CI
  (`W`) commit, used only to run CI on the proposal (§ 1.4). It is deleted after
  each use and is never the base of anything.

### 1.2 Never create GitHub pull requests or issues from the forks

Because this repository is a GitHub fork, `gh pr create` and the web UI
default the base repository to `chelsea0x3b/cudarc`; one slip posts an
upstream PR. Therefore:

- never run `gh pr create`, `gh issue create` or `gh repo set-default` in a
  fork clone, and never open a PR from the web UI, even fork-internal;
- review is done in local clones by Opus review subagents (§ 9), and CI runs
  on push (§ 1.4), so no PR object is needed;
- the proposal (FC9) is a branch plus drafted text in `fathomdb/proposal/`.
  Posting it upstream is a later owner action, outside this plan.

### 1.3 Commit classes

Each commit on a `fathomdb/*` branch belongs to exactly one class, recorded
as a `Fork-Class:` trailer. Commits on `proposal/*` carry no trailer: they
are upstream commits (CF-AC-08 checks this). The letter `C` is never a
class; it is reserved for the qualified commit of the pin chain (§ 1.5).

| Class | Content | May touch `fathomdb/` | Proposable upstream |
| --- | --- | --- | --- |
| `U` | upstream-shaped code: the pool primitive, its fixes and tests | no | yes, re-authored on `proposal/mem-pool` |
| `L` | local-only code: the aarch64 allocator fallback (items 1–4), the marker feature (item 6) | no | no |
| `P` | packaging: crate rename, `exclude`, README fork notice, `FATHOMDB.md` | no | no |
| `W` | fork CI workflow (`.github/workflows/fathomdb-ci.yaml`) | no | no |
| `D` | this directory only: plans, requirements, design, test plan, reviews, status, qualification receipts, proposal drafts | only | no |

A `D` commit touches nothing outside `fathomdb/`; every other commit touches
nothing inside it. R-FC-CODE checks this, and that every commit since
`ec1cac71` has exactly one trailer, with
`git log --format='%h %(trailers:key=Fork-Class,valueonly)' --name-only ec1cac71..HEAD`.

### 1.4 Fork CI

Upstream's workflows run only `on: pull_request`, so a push to the fork runs
nothing, and § 1.2 forbids PRs. Commit `W` adds
`.github/workflows/fathomdb-ci.yaml`, triggered on push to `fathomdb/**` and
`ci/**` and on `workflow_dispatch`. It mirrors upstream's four workflows:

- `cargo-check.yaml`: all 19 `cuda-*` features × {ubuntu, windows} × {default
  feature set, `no-std` feature set}, with upstream's feature lists;
- `cargo-clippy.yaml` (upstream already runs `-- -D warnings`),
  `cargo-fmt.yaml` and `cargo-publish-dry-run.yaml`.

It adds:

- **trigger and pinning:** the push and `workflow_dispatch` triggers, and
  `CUDARC_CUDA_VERSION` equal to the feature in every cell (upstream issue
  #525);
- **aarch64 Linux:** check and clippy for all 19 features on an aarch64 Linux
  runner (`ubuntu-24.04-arm`), because items 1–3 compile only there; if that
  runner is unavailable, `cargo check --target aarch64-unknown-linux-gnu`
  on ubuntu (no CUDA needed). FC2 D-F2 picks one;
- **a hosted test job:**
  `cargo test --lib --no-default-features -F std,driver,dynamic-loading,cuda-12060 -- fathomdb_ --test-threads=1`
  on ubuntu x86_64 and aarch64. The pure tests run there (for example
  `SYNC_FALLBACK == false` on x86_64), and device tests return early
  without a GPU. The job also runs once with FathomDB's `Cargo.lock`
  seeded, as FathomDB's retired `test_vendored_cudarc.sh` did; FC2 D-F2
  says where that lock comes from (the latest FathomDB pin commit);
- **a name-agnostic publish dry run** (`cargo publish --dry-run`, no `-p`),
  so it works before the packaging commit and on `ci/proposal-*`, where the
  crate is still `cudarc`.

GitHub-hosted runners have no GPU and no CUDA toolkit. cudarc's bindings are
pre-generated and loaded dynamically, so these jobs need neither; the GPU
cells run locally (§ 6). The hosted Windows jobs replace the local Windows VM
for this crate.

**CI evidence.** Actions logs expire, and `workflow_dispatch` cannot target
an arbitrary sha. The qualified commit is therefore pushed as the branch tip
(or as a `ci/qualify-<short sha>` ref) before any receipt commit lands on top.
The run is exported with `gh api` (run id, `head_sha`, each job's name and
conclusion) into a CI receipt under `fathomdb/qualification/`. Its
`head_sha` must equal the qualified commit.

### 1.5 Pins, receipts and publication

- **A pin names a code commit.** FathomDB and the Candle fork pin the
  qualified code commit. Its receipts are `D` commits after it that name
  it. A later tip may still be pinned instead if
  `git diff <qualified> <tip> -- . ':(exclude)fathomdb'` is empty.
- **The pin chain.** A cudarc code commit `C` is qualified (R-FC-VERIFY).
  The Candle fork then bumps to `C` and is qualified. FathomDB then pins `C`
  and the Candle commit together. Any later code commit here restarts the
  chain from R-FC-CODE for the changed area. § 8 lists which cells must
  re-run.
- **Publication** (FC10) is from the qualified commit `C` itself, from a
  detached checkout. If metadata must change first, the diff from `C` may
  touch only `Cargo.toml` package metadata and `FATHOMDB.md`. That makes a
  new code commit, which restarts the chain (§ 8) and becomes the new `C`.
  `FATHOMDB.md` ships in the crate, so editing it is a code-tree change,
  not a `D` change. `cargo package --list` must show no `fathomdb/` path and
  no fork-CI file (CF-AC-07a at FC8, CF-AC-07b at FC10). `cargo publish` is
  the owner's action. The first publish of a new crate name needs the
  crates.io token set up for it (FathomDB memory
  `0.8.27-slice132-rust-sdk-lessons.md`). The crate name is `cudarc-fathomdb` (owner
  ruling, 2026-10-11; unclaimed that day, crates.io API returned 404).

### 1.6 Host safety

These are owner rulings Q4 and Q6 for FathomDB Slice 31, and they bind this
fork's work.

- **`windchill3` driver and toolkit.** Never install, update or remove an
  NVIDIA driver on `windchill3`; it stays 580.178.04, and R615 would disable
  the host. Never change `/usr/local/cuda`, which stays 12.6.
- **The CUDA 13.0 toolkit,** if the owner approves its download, is
  installed toolkit-only and user-local in `$HOME/opt/cuda-13.0`, and
  selected per command.
- **Every GPU run starts and ends with a guard**
  (`nvidia-smi --query-gpu=driver_version --format=csv,noheader` and
  `readlink -f /usr/local/cuda`). Any change stops the work.
- **Device selection.** Pin `CUDA_VISIBLE_DEVICES` to one RTX 3090, except
  for the multi-device tests.
- **The Jetson AGX Orin** (`10.83.10.13`, ssh/scp approved) stays on
  JetPack 6 / CUDA 12.6. Tegra on CUDA 13 is UNMEASURED (Q5).

## 2. Starting state (verified 2026-10-11)

| Fact | Evidence |
| --- | --- |
| Fork `main` = upstream tag `v0.19.10` = `ec1cac71`; no other fork branches or tags before this plan | `git ls-remote origin`; `git describe --tags ec1cac71` |
| 19 CUDA features: `cuda-11040`…`11080`, `12000`…`12060`, `12080`, `12090`, `13000`…`13040` | `Cargo.toml`, `.github/workflows/cargo-check.yaml` |
| Upstream CI triggers only on `pull_request` | `.github/workflows/*.yaml` |
| GitHub Actions enabled on the fork, all actions allowed | `gh api repos/coreyt/cudarc/actions/permissions` |
| Driver functions resolve lazily, one `OnceLock` per function (`load::<_F>("name")`), so a binary built for newer bindings runs on an older driver until it calls a function that driver lacks | `src/driver/sys/mod.rs` (e.g. `cuMemPoolCreate`) |
| Version selection: `CUDARC_CUDA_VERSION`, else the highest `cuda-*` feature, else `nvcc --version`; `build.rs` emits the env-pinned cfg in addition to a differing feature cfg | `build.rs`; upstream issue #525 |
| The FathomDB delta to carry: vendored `third_party/cudarc-0.19.7` on FathomDB `release/0.8.28`, two patch files (598 + 676 lines), changes marked `FATHOMDB PATCH` in `src/driver/safe/core.rs`, `mod.rs`, `Cargo.toml` and new `mem_pool.rs`; test modules `fathomdb_alloc_fallback` and `fathomdb_private_pool` in `core.rs` | `FATHOMDB-PATCH.md` there |
| v0.19.7 → v0.19.10: 15 upstream commits, #591 and #602 fixes, one break (#600), CUDA 13.3/13.4; no upstream pool work; PR #594 (graph memory pool) overlaps items 1–3 and 5 | FathomDB Slice 31 plan § 2.1 |
| Known defects in the carried delta (both bases) | `mem_pool.rs` fails to compile with `cuda-13020`+ (`CUmemLocation.id` moved into a union); `set_max_size` cfg list stops at `cuda-13020`; below 12020 `max_size` is silently ignored; x86 dead-code warning (`in_fresh_process`); `upgrade_device_ptr` safety doc (`core.rs:2797-2799` in the vendored tree) is false once a private-pool context exists on the device |

## 3. Items carried on `fathomdb/0.19.10`

These are recorded in `FATHOMDB.md` at the repository root, which ships in
the crate so consumers can see what differs from upstream.

| # | Item | Class | Upstream status | Removal path |
| --- | --- | --- | --- | --- |
| 1 | Allocator fallback: synchronous `cuMemAlloc` when the default pool is unusable, aarch64 Linux only | `L` | local only | drop when no FathomDB artifact needs an optional default pool on aarch64 Linux, or upstream ships an equivalent |
| 2 | Per-device fallback decision | `L` | local only | with item 1 |
| 3 | Zero-length synchronous allocation | `L` | local only | with item 1 |
| 4 | Fallback tests | `L` | local only | with item 1 |
| 5 | Private memory pool: `CudaMemPool`, `CudaContext::new_with_mem_pool`, `AllocMode::Private`, tests | `U` | proposal prepared (FC9), not posted | drop when an upstream release ships an equivalent; FathomDB switches to the upstream names |
| 6 | Marker feature `fathomdb-private-pool` | `L` | local only | with item 5 |
| 7 | Fork CI workflow | `W` | not applicable | delete with the branch |
| 8 | Packaging as `cudarc-fathomdb` (`[lib] name = "cudarc"`), `exclude`, README notice | `P` | not applicable | when FathomDB consumes upstream cudarc again |

## 4. Requirements (draft; finalized in FC1 as `fathomdb/requirements.md`, reviewed by R-FC-REQ)

FathomDB REQ31 IDs are noted in brackets. Every acceptance criterion names a
host and a receipt (§ 7).

| Requirement | Acceptance criteria |
| --- | --- |
| **CF-REQ-01 Base and delta** [REQ31-01]. The branch starts at `ec1cac71`; items 1–6 are re-applied as small commits that reproduce the vendored delta; fixes and packaging are separate commits. | **CF-AC-01:** tree A = both vendored patch files applied to `ec1cac71`, with the recorded manual resolutions (marker feature hand-added; the `CudaSlice::drop` hunk under #591's gate). The branch's tree at the last item commit equals tree A, ignoring `fathomdb/` and `.github/workflows/fathomdb-ci.yaml` (`git diff` empty). Every later code commit is listed with its class and purpose. |
| **CF-REQ-02 Pool isolation and correctness** [REQ31-02]. The private pool never reads or changes the device's default or current pool; frees are stream-ordered; for allocations owned by `CudaSlice` the pool outlives every allocation by construction; `maxSize` is enforced; release threshold and trim obey their bounds; zero-length requests work; `Send`/`Sync` only where justified; multi-stream use with event tracking is correct. | **CF-AC-02:** on RTX 3090 (12.6 bindings; 13000 bindings) and Orin (12.6): default-pool handle identical before and after; after drop + sync, `used_cur` = `reserved_cur` = 0; with threshold T, after sync `reserved_cur` ≤ max(T, `used_cur`), rounded up to the allocation granularity; after `trim_to(k)`, `reserved_cur` ≤ max(k, `used_cur`); a request beyond the cap returns `DriverError(CUDA_ERROR_OUT_OF_MEMORY)` under N concurrent streams; the property test passes its stated case budget. |
| **CF-REQ-03 Leaked pointers** [REQ31-02]. A pointer leaked from a private-pool context is never valid input to another context's allocator; the hazard is either rejected or an explicit safety precondition; `upgrade_device_ptr`'s doc is corrected. | **CF-AC-03:** the doc test or unit test chosen in FC2 fails on the base and passes after the fix; the doc no longer claims device-wide validity. |
| **CF-REQ-04 Every CUDA feature** [REQ31-03]. The crate compiles for all 19 features, with the default and `no-std` feature sets, on Linux x86_64, Linux aarch64 and Windows; `max_size` exists only where the bindings support it, or a nonzero value on older bindings returns a typed error; no removed or deprecated Driver API is used; `CUmemLocation` and `CUmemPoolProps` layout differences are handled (FC2 D-8). | **CF-AC-04a:** fork CI (§ 1.4) green on the qualified commit, with a CI receipt whose `head_sha` is that commit: 19 features × {Linux x86_64, Windows} × 2 feature sets, plus aarch64 Linux check and clippy for all 19 features, clippy `-D warnings`, fmt, the hosted test job, and the publish dry run. **CF-AC-04b:** the G3 test set passes on driver 580 when built with `CUDARC_CUDA_VERSION` = 12060, 13000, 13010, 13020, 13030 and 13040. D-F1 lists everything each test depends on: driver functions, enum, attribute and flag values, and struct layouts, each with the CUDA version that introduced it, checked against driver 580. Any test that depends on something driver 580 lacks is recorded UNMEASURED, never passed. G3 also proves D-4's driver-version gate: a 13.2+ build on the 13.0 driver never requests `MAX_POOL_SIZE` or another post-13.0 attribute. |
| **CF-REQ-05 Allocator fallback scope** [REQ31-05]. Items 1–4 act only on aarch64 Linux; elsewhere behaviour equals upstream. | **CF-AC-05:** on x86_64, `SYNC_FALLBACK` is `false` and allocation follows upstream `select_async_alloc`; on the Orin, the fallback's fresh-process tests pass. |
| **CF-REQ-05b Tegra** [REQ31-04]. The fork's tests pass on the Jetson AGX Orin (JetPack 6, CUDA 12.6, `CUDARC_CUDA_VERSION=12060`). | **CF-AC-05b:** Orin receipt for the § 6.2 Orin set; the Tegra/CUDA 13 cell reads UNMEASURED, citing owner ruling Q5. |
| **CF-REQ-06 Minimal delta, upstream APIs** [REQ31-06]. Uses upstream's result-layer `mem_pool` (#544) rather than new bindings; no duplicated upstream function; every item has an upstream status and removal path. | **CF-AC-06:** `FATHOMDB.md` table complete; R-FC-DES and R-FC-CODE record no duplicated upstream functionality. |
| **CF-REQ-07 Packaging** [REQ31-06, REQ31-08]. Published as `cudarc-fathomdb` 0.19.10 with `[lib] name = "cudarc"`, upstream licences and attribution, a README fork notice, and no `fathomdb/` or fork-CI files in the package; published from the qualified commit. | **CF-AC-07a (FC8):** `cargo package --list` and `cargo publish --dry-run` receipts at `C`, showing no `fathomdb/` path and no `fathomdb-ci.yaml`. **CF-AC-07b (FC10):** the same receipts at the R-FC-PUB commit, which equals `C` (or a restarted `C`, § 1.5), with a file list identical to 07a's. |
| **CF-REQ-08 Proposal** [REQ31-09]. Pool primitive only (item 5 without the marker), on `proposal/mem-pool` from `upstream/main`, in upstream style; tests skip cleanly without a device; docs, example and benchmark; coexists with PR #594; issue and PR text drafted. Not posted. | **CF-AC-08:** fork CI green on `ci/proposal-mem-pool` (CI receipt); GPU set passes on RTX 3090 with 12.6; `git diff --stat upstream/main..proposal/mem-pool` lists no `fathomdb/`, `FATHOMDB.md` or `fathomdb-ci.yaml` path; no proposal commit carries a `Fork-Class:` trailer or the marker feature; R-PROP approves. |
| **CF-REQ-09 Process** [REQ31-11]. TDD, commit classes, reviews and receipts as this plan says. | **CF-AC-09:** `fathomdb/status.md` lists red and green shas per step, each red re-run by the reviewer; `fathomdb/review.md` records every § 9 review with model `opus`; § 1.3 class check passes. |
| **CF-REQ-10 Scope safety** [REQ31-12]. | **CF-AC-10:** no upstream PR, issue or comment; no PR or issue object on the fork; no push to `upstream`; host guard logs unchanged in every receipt. |

## 5. Design topics (FC2 writes `fathomdb/design.md`, citing primary NVIDIA documentation; R-FC-DES)

These moved here from FathomDB Slice 31 plan § 6, keeping their numbers;
FathomDB keeps D-8W (workspace CUDA version), D-10, D-11, D-12 (product), D-14,
D-15 and D-16. Slice 30's pool design in
FathomDB (`dev/plans/0.8.28/features/slice-30/design.md` § 3.1, § 4.5) is the
baseline.

- **D-1 API shape for upstream.** The `CudaMemPool` constructor and
  properties: device location, `maxSize` and release threshold, with no IPC
  handles. Naming follows upstream conventions. Decide between `AllocMode`
  and a narrower surface.
- **D-2 Lifetime and destruction.** The `Arc` chain slice → stream →
  context → pool, and `cuMemPoolDestroy` with allocations outstanding (cite
  its deferred release). Also process exit and statics. The leak/upgrade
  hazard (CF-REQ-03): either reject it or make it a stated safety
  precondition.
- **D-3 Stream ordering and reuse.** Cross-stream allocation and free, the
  `CU_MEMPOOL_ATTR_REUSE_*` defaults, and the interaction with event
  tracking and #591.
- **D-4 Capacity and exhaustion.** Whether `maxSize` is available depends on
  both the bindings and the driver. A 12020+ build may load an older libcuda
  that keeps `maxSize` in reserved bytes, so check `cuDriverGetVersion` at
  creation and read back `MAX_POOL_SIZE` on 13.2+. `attribute()` must accept
  the 64-bit attributes. Behaviour under concurrent exhaustion.
- **D-5 Release threshold and trimming.** Threshold 0 versus higher,
  `trim_to`, and the latency effect.
- **D-6 Thread safety.** Justify `Send`/`Sync` per type.
- **D-7 Observability.** Typed reads of the reserved and used, current and
  high-water attributes.
- **D-8 CUDA 11.4 → 13.4.**
  - **Struct layouts:** one cfg-gated `CUmemLocation` constructor (plain
    `id` ≤ 13.1, union ≥ 13.2); build `CUmemPoolProps` from `zeroed()` then
    field writes, never a literal.
  - **Bindings and matches:** no derives the 13.2+ bindings lack; no
    exhaustive matches on `CUmemLocationType` or `CUmemPool_attribute`;
    read attributes as raw integers.
  - **`max_size`:** present only on 12020+ builds, or a typed error for a
    nonzero value on older builds.
  - **cfg form:** one that cannot silently drop new versions.
  - **Context creation:** `cuCtxCreate_v4` only if a non-primary context is
    needed.
- **D-9 Graph capture.** Private-pool allocation during stream capture, and
  coexistence with PR #594.
- **D-12 (fork part) Tegra.** Items 1–4 on the new base.
  `cfg(aarch64, linux)` also matches Thor, Grace and SBSA: decide whether to
  gate on `CU_DEVICE_ATTRIBUTE_INTEGRATED`. The early-`cuInit`
  precondition.
- **D-13 Multi-device.** One pool per device, `cuMemPoolSetAccess` and peer
  access, and ordinal checks (`windchill3` has two RTX 3090s).
- **D-F1 What a newer-binding build can test on driver 580.** For each G3
  test, list:
  - every driver function it calls, confirmed exported by
    `libcuda.so.580.178.04` (`nm -D`);
  - every enum, attribute or flag value it passes (for example pool
    attributes 9–14 including `MAX_POOL_SIZE`, which arrived in 13.2;
    attribute 15 in 13.4; `CU_MEM_LOCATION_TYPE_*` additions in 13.4);
  - every struct whose layout it relies on (`CUmemLocation`,
    `CUmemPoolProps`), with `size_of` compared across the 13000 and 13040
    bindings.

  Give each its introducing CUDA version, as reported by the driver's
  `cuDriverGetVersion` (13000 on driver 580). Map each test to RUN or
  UNMEASURED per binding version. The design must make driver-dependent
  behaviour depend on `cuDriverGetVersion`, never only on the binding cfg
  (D-4).
- **D-F2 Fork CI design.** The § 1.4 workflow, the aarch64 runner choice,
  the hosted test job and its FathomDB-lock run, the CI receipt, and why
  it is a fork-only commit.
- **D-F3 cuBLAS workspace and upstream PR #595.** PR #595, "feat(cublas):
  support a retained caller-owned workspace", was open on 2026-08-20.
  Decide how the private pool coexists with it, and whether a cuBLAS
  workspace allocated from the pool is possible and worthwhile. This is
  input to FathomDB D-14 and to Candle DK-4, and to the proposal (FC9)
  alongside #594.

## 6. Test plan topics (FC3 writes `fathomdb/test-plan.md`; R-FC-TEST)

### 6.1 Test kinds

Tests follow cudarc's style: `#[cfg(test)] mod tests` beside the code, and
device tests return early when no device is present.

- **T-UNIT** pure tests: props construction, cfg selection, error mapping.
- **T-DEV** device tests:
  - create/drop, alloc/free, default pool unchanged, zero length;
  - cap exhaustion, threshold and trim bounds (the CF-AC-02 inequalities);
  - attributes including 64-bit;
  - the leak/upgrade precondition.
- **T-PROP** proptest against a model: random alloc, free, cross-stream
  free, sync and trim. Invariants: `used_cur` ≤ `reserved_cur` ≤ cap, and
  all zero after drop + sync. Case budget stated. Whether `proptest` is a
  dev-dependency here or only on the FathomDB side is decided in FC3.
- **T-CONC** N threads × M streams; concurrent exhaustion; drop while other
  streams are live.
- **T-FAULT** `cfg(test)` seams in the result layer:
  - `cuMemPoolCreate` OOM;
  - `set_attribute` failure (create destroys the pool);
  - OOM mid-sequence;
  - destroy with allocations outstanding;
  - sticky error or lost context before drop.
- **T-CAPTURE** private-pool allocation during stream capture.
- **T-MULTI** two RTX 3090s.
- **T-TEARDOWN** exit with pools alive; fork after init; lost-context
  teardown.
- **T-PROC** fresh-process tests for the fallback's per-device table.
- **T-MATRIX** fork CI (§ 1.4).
- **T-PERF** the proposal benchmark (default, private, synchronous) on
  RTX 3090 and Orin.

Exhaustion tests request at most cap + one granule, far below physical free
memory. On JetPack 7.2 (L4T 39.2, the G7 / Q5 follow-up), NVIDIA known
issue 5699079 says an excessively large allocation may reboot the device,
so G7 must keep that bound.

### 6.2 GPU cells (receipts in `fathomdb/qualification/`)

| # | Host | Build | Needs | Test set |
| --- | --- | --- | --- | --- |
| G1 | `windchill3` RTX 3090, driver 580 | `CUDARC_CUDA_VERSION=12060`, `-F cuda-12060` | present | all `driver::` tests incl. T-DEV, T-PROP, T-CONC, T-FAULT, T-CAPTURE, T-TEARDOWN |
| G2 | same, both 3090s | 12060 | present | T-MULTI |
| G3 | same | 12060, 13000, 13010, 13020, 13030, 13040 bindings, driver-only feature set (`--no-default-features -F std,driver,dynamic-loading,cuda-<v>`) | nothing installed | T-DEV, T-CONC, T-FAULT, the D-4 driver-version gate, and every other test D-F1 maps to RUN (CF-AC-04b) |
| G4 | same | 13000, default features | the 13.0 runtime libraries (NVRTC, cuBLAS): the user-local 13.0 toolkit if the owner approves its download (FathomDB W0 step 4), or else the NVIDIA cu13 wheels from PyPI installed into a scratch virtual environment owned by this work (never the FathomDB primary `.venv`, whose cuBLAS is 13.1.1), pinned to 13.0.x (`nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-nvrtc==13.0.88`, `nvidia-cublas==13.0.2.14`) and reached by `LD_LIBRARY_PATH=<venv>/lib/python3.*/site-packages/nvidia/cu13/lib` for that command only; `libcuda.so.1` stays the system driver's | the toolkit-dependent tests (NVRTC- or cuBLAS-using) that D-F1 identifies; UNMEASURED only if neither source is available |
| G5 | same | 13010–13040 with default features | R615 driver (never — Q4) | UNMEASURED |
| G6 | Orin, JetPack 6 | 12060 | present | all `driver::` tests incl. T-PROC and the fallback tests |
| G7 | Orin, JetPack 7.2.x | 13020 / 13040 | reflash (not in Slice 31 — Q5) | UNMEASURED (known issue 5699079 bounds its exhaustion tests) |

The exact command for each cell is written in `test-plan.md`. One example,
for G3:

```bash
CUDA_VISIBLE_DEVICES=0 CUDARC_CUDA_VERSION=13020 \
  cargo test --no-default-features -F std,driver,dynamic-loading,cuda-13020 \
  -- driver:: --test-threads=1
```

`--test-threads` is set per test kind in `test-plan.md`; tests that need a
fresh process (T-PROC) run one per process.

## 7. Receipts and status

- `fathomdb/status.md`: one line per step: date, step, red sha (failing
  test), green sha, reviewer's red re-run result.
- `fathomdb/review.md`: per review, the model, reviewed sha, findings with
  file:line and failure scenario, and their dispositions.
- `fathomdb/qualification/<YYYY-MM-DD>-<cell>-<short sha>.md`: one receipt
  per run. It records the code sha, host, `uname -m`, driver and
  `readlink -f /usr/local/cuda` before and after, toolkit path,
  `CUDARC_CUDA_VERSION`, features, the `Cargo.lock` SHA-256 and `cargo tree
  -e normal` output (the fork commits no lockfile; upstream ignores it), the
  exact command, the test count and
  names (passed, failed, ignored, UNMEASURED with reason) and a SHA-256 of the
  full log. Logs up to 1 MiB are committed beside the receipt; larger logs
  stay on the host and the receipt names their path.
- CI receipts (§ 1.4) live in the same directory.
- All of these are `D` commits on `fathomdb/0.19.10`. FathomDB links them by
  permalink at the receipt commit's sha.
- The R0-rev8 review of plan revision 1 is recorded in `fathomdb/review.md`
  as R-FC-PLAN.

## 8. Re-qualification after a code change

| Change touches | Re-run |
| --- | --- |
| `fathomdb/` only | nothing |
| item 1–4 code | G1, G6 (the allocator fallback tests and T-PROC), fork CI |
| item 5 code or `mem_pool.rs` | G1–G4 for the touched test kinds, G6, fork CI |
| cfg or bindings use (D-8) | fork CI, G1, G3, G6 |
| packaging (`P`), including `FATHOMDB.md` | fork CI; CF-AC-07a; G1 smoke; restarts the chain (§ 1.5) |
| fork CI workflow (`W`) only | fork CI |
| FathomDB's `Cargo.lock` changes the resolved dependency set of `cudarc-fathomdb` (FathomDB's D-16 guard, owner ruling Q8 option (b), fails until this re-run is recorded) | the hosted test job with FathomDB's lock; G1 and G6 if any dependency of the pool path changed |

## 9. Work packages and reviews

Writers: the FathomDB Slice 31 main thread writes documents and creates
branches. One `implementer` subagent at a time writes code in this clone
(`/home/coreyt/projects/fathomdb-forks/cudarc`); the proposal uses a
separate worktree of it (`/home/coreyt/projects/fathomdb-forks/cudarc-proposal`).
No two writers run in one checkout. The main thread pushes explicit refs
after verifying from git; it never force-pushes.

Every review is an independent Opus subagent (`model: "opus"`, high effort,
read-only, general-purpose type), never codex. It re-runs each RED sha,
reports findings with file:line and a failure scenario, and is recorded in
`fathomdb/review.md`. Findings are fixed RED then GREEN, or given a recorded
disposition, before the blocked step starts.

| WP | Content | Output | Review | Blocks |
| --- | --- | --- | --- | --- |
| FC0 | Main thread: branch from `ec1cac71`, upstream push disabled, this plan | `fathomdb/plan.md` | R-FC-PLAN (with FathomDB R0-rev8) | FC1 |
| FC1 | Main thread: final requirements from § 4 | `fathomdb/requirements.md` | R-FC-REQ | FC2 |
| FC2 | Main thread, with a research subagent for citations: design D-1…D-9, D-12, D-13, D-F1, D-F2 | `fathomdb/design.md` | R-FC-DES | FC3 |
| FC3 | Main thread: test plan, each test traced to a CF-AC with host, command and expected result | `fathomdb/test-plan.md` | R-FC-TEST | FC4 |
| FC4 | Implementer: commit `W` (fork CI) first, then port the vendored tests (they fail to compile on bare `ec1cac71`: RED), then items 1–4 and 5–6 as small `L`/`U` commits (GREEN); record tree A | commits; CF-AC-01 output | — | FC5 |
| FC5 | Implementer: RED then GREEN fixes. The `CUmemLocation` constructor, `CUmemPoolProps` from `zeroed()`, the `max_size` cfg rule and the `set_max_size` cfg form. 64-bit attributes, the driver-version check, raw attribute reads and no exhaustive matches. The `upgrade_device_ptr` precondition and doc, and the dead-code warning. | `U` commits | — | FC6 |
| FC6 | Implementer: new tests from the test plan, RED first, then code | `U`/`L` commits | — | FC7 |
| FC7 | Implementer: packaging commit `P` (`cudarc-fathomdb`, `[lib] name = "cudarc"`, `exclude`, README notice, `FATHOMDB.md`) | `P` commit | R-FC-CODE (whole branch, class check, RED re-runs, fork CI) | FC8 |
| FC8 | Main thread: push the qualified commit `C` as the tip (§ 1.4); CI receipt; run G1–G4, G6 (optionally a Claude agent on the Orin); CF-AC-07a package receipts; receipts committed after `C` | `fathomdb/qualification/` | R-FC-VERIFY (CF-AC-01…06, 07a, 09, 10 at `C`) | the Candle fork's FK2; FathomDB R-PIN |
| FC9 | Implementer in `cudarc-proposal`: `proposal/mem-pool` from `upstream/main`, item 5 only, re-authored in upstream style, RED first; example and benchmark (T-PERF); `ci/proposal-mem-pool` run then deleted. Main thread writes `fathomdb/proposal/issue.md` and `pr.md` (motivation, API, #594 coexistence, benchmark) | proposal branch and drafts | R-PROP (CF-AC-08) | owner sign-off, later |
| FC10 | Main thread: publish preparation from a detached checkout of `C` (§ 1.5); `cargo package --list`, `cargo publish --dry-run` (CF-AC-07b); crates.io token for the new name | dry-run receipt | R-FC-PUB, then owner R-PUB | owner `cargo publish` |

FC9 may start once R-FC-CODE approves. FC10 waits for FathomDB's R-VERIFY,
so the published crate is the one FathomDB qualified.

## 10. What FathomDB consumes from this fork

FathomDB Slice 31 records these in its `status.md` and `review.md`, and
R-VERIFY checks them:

- the qualified code commit and the pin-chain check;
- R-FC-CODE and R-FC-VERIFY verdicts with no open finding;
- the receipt list (G1–G4, G6), with G5 and G7 UNMEASURED by owner rulings
  Q4 and Q5;
- after publication, the published version and its `.cargo_vcs_info.json`
  sha, equal to `C`.

## 11. Risks

- **Accidental upstream PR or push.** Mitigated by § 1.1 (push URL disabled)
  and § 1.2 (no PR objects at all).
- **A 13.2+ build passing only because a test never ran.** CF-AC-04b
  requires per-test RUN/UNMEASURED accounting from D-F1.
- **Upstream PR #594** overlaps items 1–3 and 5 (D-9, FC9); **PR #595**
  (cuBLAS caller-owned workspace) bears on D-F3 and FC9.
- **`maxSize` on a driver older than the bindings** (D-4).
- **Hosted CI drift from upstream's workflow.** The fork workflow is diffed
  against upstream's four workflows in R-FC-CODE.
- **Licensing.** cudarc is MIT OR Apache-2.0; attribution is kept in the
  README and `FATHOMDB.md`.
