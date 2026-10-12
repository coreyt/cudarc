# cudarc-fathomdb status

Per-step record (plan § 7, requirements CF-AC-09). One row per step. The
reviewer re-runs each RED; the last column holds that result.

| Date | Step | RED sha | RED failure | GREEN sha | Reviewer RED re-run |
| --- | --- | --- | --- | --- | --- |
| 2026-10-12 | FC4-0 `count-tests.py` | `86b8487` | 35 of 35 fixture tests error with `NotImplementedError` (stub script) | `5fc550e` (35 pass) | pending (R-FC-CODE) |
| 2026-10-12 | FC4-0 `names --with-binary` | `9f2bd1c` | 1 of 36 errors: `python3 -m unittest: error: unrecognized arguments: --with-binary` | `cafb294` (36 pass) | pending (R-FC-CODE) |
| 2026-10-12 | FC4-W fork CI | — (test plan § 10: no RED) | — | `4869fab` | — |
| 2026-10-12 | FC4-port items 1–6 | not committed (plan § 9 FC4 exception; see below) | vendored tests on bare `ec1cac71`: 37 compile errors | `9f02c3a`, `ce0b56e`, `3fbccda` (19 of 19 on RTX 3090) | pending (R-FC-CODE) |
| 2026-10-12 | FC5-a `device_location_is_the_ordinal_on_every_binding` | `dfff1a1` | cuda-13020+: crate does not build, `error[E0560]` no field `id` (the D-8 defect) | `6f4c67a` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-a `pool_props_are_zeroed_then_written` | `49ef889` | cuda-13030/13040: `maxSize` left 0, right 4096 | `e37097d` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-a `in_fresh_process` dead code | — (warning fix; no behaviour) | — | `cf5d145` | — |
| 2026-10-12 | FC5-b `max_size_without_binding_support_is_not_supported` | `536ec11` | cuda-12010: left `Err(CUDA_ERROR_UNKNOWN)`, right `Err(CUDA_ERROR_NOT_SUPPORTED)` | `4dc1ec7` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-b `max_size_on_a_driver_older_than_12020_is_not_supported` | `536ec11` | cuda-12060, RTX 3090: left `Ok(())`, right `Err(CUDA_ERROR_NOT_SUPPORTED)` | `4dc1ec7` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-b `attributes_newer_than_the_driver_are_never_requested` | `5a342fe` (after the ungated read-back `7889643`) | G3 cuda-13020, driver 580: attribute requests above 8, left 1, right 0 | `2f936a5` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-b `max_pool_size_read_back_uses_the_driver_value` | `0314e34` | cuda-13020: left `Ok(())`, right `Err(CUDA_ERROR_INVALID_VALUE)` | `f0e72b3` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-b `a_failed_max_pool_size_read_leaves_create_successful` | `0314e34` | cuda-13020: `unwrap()` on `Err(CUDA_ERROR_INVALID_VALUE)` | `f0e72b3` | pending (R-FC-CODE) |
| 2026-10-12 | FC5-b `max_pool_size_reads_back_at_least_the_request` | — (no RED at FC5: needs a driver ≥ 13020; ignored, G5b) | — | `c44d3b7` | — |
| 2026-10-12 | FC5-c typed reads; 3 characterization tests | — (no RED: the vendored code already behaves) | — | `75f13fa`, `a11a045` | — |
| 2026-10-12 | FC5-d `# Safety` text; 2 of 3 § 3.2 tests | — (no RED: doc and characterization) | — | `9602202` | — |

## FC4-0 (class D)

- **RED `86b8487`.** `fathomdb/scripts/test_count_tests.py` plus a stub
  `count-tests.py`. Command and result:

  ```text
  $ python3 -I -m unittest discover -s fathomdb/scripts -p 'test_*.py'
  Ran 35 tests in 0.007s
  FAILED (errors=35)
  ```

  Every error is
  `NotImplementedError: count-tests.py is not implemented yet`.
- **GREEN `5fc550e`.** `count-tests.py` (subcommands `expect`, `names`,
  `check`, `unlanded`), `fathomdb/test-inventory/fork-catalogue.toml` and
  `fathomdb/qualification/Cargo.lock.baseline`. The same command gives
  `Ran 35 tests … OK`; `python3 -I -m pytest fathomdb/scripts/test_count_tests.py`
  gives `35 passed`.
- **`ca88a76`** removes `fathomdb/scripts/__pycache__/`, committed by
  mistake in `86b8487` and `5fc550e`, and adds `fathomdb/.gitignore`.
  History is not rewritten.
- **RED `9f2bd1c` / GREEN `cafb294`.** `names --with-binary` prints the
  binary label of each name, so a CI job fails on a cell test that lives
  in a binary it does not run. RED: `ERROR:
  test_names_cli_can_print_the_binary_of_each_name … unrecognized
  arguments: --with-binary` (35 pass, 1 error). GREEN: 36 pass.
- **Catalogue.** 71 named tests (test plan §§ 3.1–3.4): 24 vendored rows
  with `landed = true` (11 + 3 aarch64 in `fathomdb_alloc_fallback`, 8 + 2
  aarch64 in `fathomdb_private_pool`), 47 rows with `landed = false`, and
  the six T-DEV cases of § 3.4 under `[[unnamed]]`, because the test plan
  does not name their functions. The five upstream `#[ignore]` tests of
  § 2.1 are listed under `[[upstream_ignored]]`. The cfg of a row that has
  not landed is provisional; the commit that adds the test makes it exact
  and sets `landed = true`.
- **Baseline lock.** `git worktree add --detach ../cudarc-baseline ec1cac71`,
  then `cargo generate-lockfile` there ("Locking 27 packages to latest
  compatible versions"), copied to
  `fathomdb/qualification/Cargo.lock.baseline`.
  - SHA-256 `3d399b4e43bfa2c4078f2ff607145b6a35216041054225905a74fc1094747923`
  - Toolchain: `rustc 1.95.0 (59807616e 2026-04-14)`,
    `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
  - `cargo tree -e normal`, driver-only and default 12060:
    `cudarc v0.19.10` → `libloading v0.9.0` → `cfg-if v1.0.5`.

## FC4-W (class W) `4869fab`

`.github/workflows/fathomdb-ci.yaml`, push to `fathomdb/**` and `ci/**`
plus `workflow_dispatch`. It has not run yet: CI runs when the main thread
pushes.

- **actionlint** 1.7.12 with shellcheck: `actionlint
  .github/workflows/fathomdb-ci.yaml` exits 0, no findings.
- **Upstream mirror.** `cargo-check` (19 features × {ubuntu-latest,
  windows-latest} × {default, `no-std`} sets, upstream's feature lists),
  `clippy` (upstream's single cuda-13040 command, `-D warnings`),
  `cargo-fmt`, `cargo-publish-dry-run` (`-F cuda-13040`, no `-p`).
  `CUDARC_CUDA_VERSION` equals the feature in every cell.
  - Deviation: upstream uses `actions/checkout@v2`, `actions-rs/toolchain@v1`
    and `actions-rs/cargo@v1` (archived, Node 12). The fork uses
    `actions/checkout@v4`, `dtolnay/rust-toolchain@stable` and plain
    `cargo` commands with the same arguments.
- **Fork additions.** `cargo-check-aarch64` and `clippy-aarch64`
  (`ubuntu-24.04-arm`, 19 features); `hosted-test`; `pure-test`; `t-unit`.
- **`hosted-test`.** x86_64 and aarch64; driver-only cuda-12060;
  `-- fathomdb_ --test-threads=1`. Run 1 copies `Cargo.lock.baseline`;
  run 2 fetches
  `https://raw.githubusercontent.com/fathomadb/fathomdb/4c327cd8a90e21d431246b566eb2cba73a880619/Cargo.lock`
  (`FATHOMDB_LOCK_SHA` = `origin/release/0.8.28` on 2026-10-12), prints its
  SHA-256 (the git blob at that sha hashes to
  `5b9e680660c861f44c0476576adbfb686eb3cb3854a3fc928f0f03ed5d2fac52`), and
  asserts from `cargo metadata` that every resolved dependency of cudarc
  equals FathomDB's lock entry (inline Python in the workflow, not under
  `fathomdb/`). Both runs use `--locked` and are checked by
  `count-tests.py check` and `unlanded`.
- **`pure-test` and `t-unit`.** The names come from catalogue cells `pure`
  and `t-unit`. None have landed, so the run step prints "expected names:
  none yet …" and passes. Each job also lists its module
  (`fathomdb_private_pool::`, `fathomdb_lib_major::`) and fails if the
  listing differs from the landed catalogue rows, or if it contains a test
  whose row says `landed = false`. Once a row lands, a missing test fails
  the run, and a cell test in a binary other than `unittests:cudarc`
  fails the job until a step for that binary is added.
- **Local simulation at `3fbccda`** (scratch copy, `CUDA_VISIBLE_DEVICES=`
  empty, so device tests take their SKIP path):
  - hosted run 1: 19 passed, 28 filtered out, `count-tests: PASS`,
    `unlanded: PASS`, 8 SKIP lines;
  - hosted run 2: the lock assertion printed `ok cfg-if 1.0.4`,
    `ok libloading 0.9.0`, `PASS: 2 resolved dependencies equal FathomDB's
    lock`. cfg-if resolved to FathomDB's 1.0.4 rather than the baseline's
    1.0.5. The tests gave `count-tests: PASS`. A tampered lock
    (`libloading 0.8.9`) made the assertion exit 1;
  - pure-test, cuda-12060 and cuda-11040: module listing 8 of 8,
    `count-tests: PASS`, "expected names: none yet". With a scratch
    catalogue edit (one real name in cell `pure`, plus one landed name
    that does not exist), the run step failed with
    `FAIL unittests:cudarc: missing …::not_written_yet`.
- **Expected CI failures at this tip** (known defects, fixed at FC5; plan
  § 2, last row):
  - every job building cuda-13020, 13030 or 13040 fails to compile:
    `` error[E0560]: struct `CUmemLocation_st` has no field named `id` ``
    at `src/driver/safe/mem_pool.rs:31:9`. That covers `cargo-check`,
    the `cargo-check-aarch64`/`clippy-aarch64`/`pure-test`/`t-unit` cells
    of those three features, `clippy` (cuda-13040) and
    `cargo-publish-dry-run` (cuda-13040);
  - x86_64 clippy with `--all-targets -D warnings` also fails on
    `` error: function `in_fresh_process` is never used ``
    (`src/driver/safe/core.rs:3930:8`). This was observed locally at
    cuda-12060.
- **Not covered yet.** The `ci/proposal-*` adaptations (the proposal's own
  test names in the hosted jobs, plan § 1.4) belong to FC9 and FC9b.

## FC4-port (classes L and U)

- **RED (not committed; plan § 9 FC4 exception).** A scratch copy of
  `ec1cac71` (`git archive ec1cac71 | tar -x`), with only the two vendored
  test modules appended to `src/driver/safe/core.rs`. They were taken from
  the `+// FATHOMDB PATCH BEGIN: … tests` blocks of both patches. Then:

  ```text
  $ CUDARC_CUDA_VERSION=12060 cargo test --locked --lib --no-default-features -F std,driver,dynamic-loading,cuda-12060 --no-run
  error: could not compile `cudarc` (lib test) due to 37 previous errors; 2 warnings emitted
  ```

  Errors by kind, verbatim:

  ```text
       10 error[E0425]: cannot find function `async_alloc_decision` in this scope
        6 error[E0433]: cannot find type `AllocMode` in this scope
        4 error[E0599]: no function or associated item named `new_with_mem_pool` found for struct `core::CudaContext` in the current scope
        4 error[E0425]: cannot find function `sync_free_needed` in this scope
        3 error[E0433]: cannot find type `AllocModeByDevice` in this scope
        3 error[E0425]: cannot find function `sync_zero_bytes_is_null` in this scope
        3 error[E0425]: cannot find function `alloc_mode_of` in this scope
        1 error[E0599]: no method named `mem_pool` found for struct `std::sync::Arc<core::CudaContext>` in the current scope
        1 error[E0599]: no method named `alloc_mode` found for struct `std::sync::Arc<core::CudaContext>` in the current scope
        1 error[E0432]: unresolved import `crate::driver::safe::mem_pool`
        1 error[E0425]: cannot find value `SYNC_FALLBACK` in this scope
  ```

  Log SHA-256
  `84a36633f58cd51a6b7dae68cc337433d145dd883024e2eafcf9177d46474469`
  (11675 bytes, kept on the host in the session scratchpad).
- **GREEN commits.** Each passed `cargo check --locked --lib -F cuda-12060`
  and `cargo check --locked --lib --no-default-features -F
  std,driver,dynamic-loading,cuda-12060` with no warnings:
  - `9f02c3a` (L) items 1–4: `git apply -3 fathomdb-alloc-fallback.patch`,
    unchanged;
  - `ce0b56e` (U) item 5: `git apply -3 --exclude=Cargo.toml
    fathomdb-private-pool.patch`, unchanged. Also checked: cuda-13000
    builds; cuda-13020 fails as listed under FC4-W;
  - `3fbccda` (L) item 6: the patch's four `Cargo.toml` lines added by hand
    directly after `fallback-latest = []`, the line the hunk follows.
- **Manual resolutions (CF-AC-01).**
  - Marker feature: hand-added as above. The patch's `Cargo.toml` hunk
    targets the crates.io-normalized manifest and does not apply to the
    git tree.
  - `CudaSlice::drop`: the fallback patch's hunk only changes the
    synchronous branch (`if sync_free_needed(SYNC_FALLBACK, …)` around
    `free_sync`). It applied with an offset after #591's
    `if ctx.is_managing_stream_synchronization() { … }` block (upstream
    `70c5e46`). The gate is unchanged and the event waits stay inside it.
    No hand edit was needed.
- **Tree A check (CF-AC-01).** Run in
  `/tmp/claude-1000/-home-coreyt-projects-fathomdb/a9720f76-333b-48d0-ad12-5233c37b0381/scratchpad/`:

  ```text
  git archive ec1cac71 | tar -x -C treeA
  cd treeA
  git apply ../patches/fathomdb-alloc-fallback.patch
  git apply --exclude=Cargo.toml ../patches/fathomdb-private-pool.patch
  # marker feature: the same four lines after `fallback-latest = []`
  git archive 3fbccda | tar -x -C ../branchtree
  diff -r -x fathomdb -x fathomdb-ci.yaml -x target -x Cargo.lock treeA branchtree
  ```

  Result: no output, exit 0. `git apply -3` refuses to run outside a
  repository (`error: '--3way' outside a repository`), so tree A used plain
  `git apply`; both patches applied cleanly with offsets only. Also checked:
  tree A as a git tree object (`3827e1e2cab3ce24c6067fb1b60ac7ad27f699d3`)
  against the branch,
  `git diff 3827e1e 3fbccda -- . ':(exclude)fathomdb' ':(exclude).github/workflows/fathomdb-ci.yaml'`,
  is empty.
- **GREEN on the GPU (G1 setup, `windchill3`).**
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` (index 0 = NVIDIA
  GeForce RTX 3090, 00000000:41:00.0), `CUDARC_CUDA_VERSION` = the feature,
  `Cargo.lock.baseline` copied in, `--locked`, code at `3fbccda`. Each run
  was checked by `count-tests.py check` against `expect … --filter
  fathomdb_`.

  | Build | Command (after `cargo test --locked --lib`) | alloc_fallback | private_pool | Summary | count-tests |
  | --- | --- | --- | --- | --- | --- |
  | default 12060 | `-F cuda-12060 -- fathomdb_ --test-threads=1` | 11/11 | 8/8 | 19 passed; 0 failed; 0 ignored; 75 filtered out | PASS |
  | driver-only 12060 | `--no-default-features -F std,driver,dynamic-loading,cuda-12060 -- fathomdb_ --test-threads=1 --nocapture` | 11/11 | 8/8 | 19 passed; 0 failed; 0 ignored; 28 filtered out | PASS |
  | driver-only 13000 | `--no-default-features -F std,driver,dynamic-loading,cuda-13000 -- fathomdb_ --test-threads=1 --nocapture` | 11/11 | 8/8 | 19 passed; 0 failed; 0 ignored; 26 filtered out | PASS |

  - The `--nocapture` runs printed no `SKIP` line, so every device test
    ran on the device. A `--nocapture` rerun of the default-features build
    also printed none.
  - Log SHA-256: default 12060 `71b1a8d0…dd7`, driver-only 12060
    `29544b8e…d6c`, driver-only 13000 `0ed1736b…7b4`. The logs are kept on
    the host in the session scratchpad. These runs are FC4 evidence, not
    FC8 receipts.
  - Guard before (2026-10-12T07:41:16-05:00) and after
    (2026-10-12T07:41:48-05:00): driver `580.178.04` on all three GPUs;
    `/usr/local/cuda` → `/usr/local/cuda-12.6`. Unchanged.
  - Not run here: 13020+ (FC5-a), the aarch64-only tests (G6), and k = 3
    or baseline comparisons (FC8).

## FC5 (classes U and D)

Host `windchill3`. Every GPU run used `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0` (index 0 = NVIDIA GeForce RTX 3090,
00000000:41:00.0), `CUDARC_CUDA_VERSION` = the feature,
`Cargo.lock.baseline` in place and `--locked`. The guard was taken before
and after every GPU run (first 2026-10-12T07:53:19-05:00, last
2026-10-12T08:05:49-05:00): driver `580.178.04` on all three GPUs,
`/usr/local/cuda` → `/usr/local/cuda-12.6`, never changed. Logs are kept on
the host in the session scratchpad.

### FC5-a (D-8)

- **RED `dfff1a1`.** `device_location_is_the_ordinal_on_every_binding`;
  the vendored `pool_props_are_pinned_device_memory_on_the_ordinal` now
  reads the id through a union-aware test helper and compares the
  location's bytes with `device_location(3)` (the edit test plan § 10
  names). Driver-only `cargo test --no-run`:
  - cuda-13020: `` error[E0560]: struct `CUmemLocation_st` has no field
    named `id` `` at `src/driver/safe/mem_pool.rs:31:9`, the D-8 defect,
    plus `` error[E0432]: unresolved import
    `crate::driver::safe::mem_pool::device_location` ``;
  - cuda-12060: only the E0432 import error (new API, not counted as RED).
- **GREEN `6f4c67a`.** `device_location(ordinal)` from `zeroed()`; the
  union is written under `cfg(not(any(cuda-11040 … cuda-13010)))`.
- **RED `49ef889`.** `pool_props_are_zeroed_then_written`, behavioural
  (pure, no GPU): at cuda-13030 and cuda-13040,
  `` assertion `left == right` failed: maxSize  left: 0  right: 4096 ``;
  passes at 12010, 12060 and 13020.
- **GREEN `e37097d`.** `maxSize` written under
  `cfg(not(any(cuda-11040 … cuda-12010)))`.
- **`cf5d145`.** `in_fresh_process` gets its callers' cfg
  (`all(target_os = "linux", target_arch = "aarch64")`), which removes
  the x86_64 `function in_fresh_process is never used` error under
  `-D warnings`. Nothing it checks changes.

### FC5-b (D-4)

- **`cfg(test)` seams** (thread-local, cleared by a guard at the start and
  end of each test): a counter and forced error at `create`'s first
  driver call; a `cuDriverGetVersion` override; `cuMemPoolCreate` and
  `cuMemPoolDestroy` counters; forced `cuMemPoolCreate` and threshold
  failures; forced read-back value and error; a gate-skip counter; and,
  in `result::mem_pool`, a counter of `get_attribute`/`set_attribute`
  requests for attributes above 8. Each counter test has a positive
  control. Seams are added in the RED commit of the test that first
  uses them; they do not exist in non-test builds.
- **RED `536ec11` / GREEN `4dc1ec7`.** Both REDs are behavioural (table
  above). GREEN: `max_size > 0` returns `CUDA_ERROR_NOT_SUPPORTED` before
  any driver call when the bindings predate cuda-12020, and before
  `cuMemPoolCreate` when `cuDriverGetVersion` < 12020.
- **Staged gate.** `7889643` reads `MAX_POOL_SIZE` after every create on
  cuda-13020+ bindings, ungated, propagating errors. On driver 580 that
  fails every create of a cuda-13020 build with
  `CUDA_ERROR_INVALID_VALUE` (5 of 11 pool tests failed), which also
  answers design § 14 item 3 for attribute 13: driver 580 rejects it.
  RED `5a342fe` then fails on G3 cuda-13020 with
  `requests for attributes above 8  left: 1  right: 0`. GREEN `2f936a5`
  reads the attribute only when `cuDriverGetVersion` ≥ 13020.
  Attribute 15 is never requested.
- **RED `0314e34` / GREEN `f0e72b3`.** Read-back handling: a failed read
  leaves `create` Ok and `max_size()` `None`; a value below a nonzero
  request fails `create` with `CUDA_ERROR_INVALID_VALUE` after the pool is
  destroyed (destroy counter 1); otherwise `max_size()` is the driver's
  value.
- **`c44d3b7`.** `max_pool_size_reads_back_at_least_the_request`,
  `#[ignore]`d, compiled from cuda-13020. No RED at FC5. Run by hand on
  driver 580 it fails its precondition (`driver 13000 cannot report
  MAX_POOL_SIZE`), so it can never pass on the wrong driver.

### FC5-c (typed reads)

- **`75f13fa`.** `reserved_current`, `reserved_high`, `used_current`,
  `used_high`, `release_threshold` replace `attribute()`; no setters, no
  `ordinal()`. This removes `attribute()` from the vendored API (design § 1
  change table). The vendored tests call the typed read of the same
  attribute instead; what they check is unchanged.
- **`a11a045`.** `typed_reads_return_64_bit_values`,
  `create_destroys_the_pool_when_setting_the_threshold_fails`,
  `a_pool_create_out_of_memory_returns_the_error_and_no_pool`. No RED:
  characterization, the vendored code already behaves. Mutation check,
  not committed: forgetting the owned pool on the threshold failure made
  the second test fail with `pools destroyed  left: 0  right: 1`.

### FC5-d (CF-REQ-03)

- **`9602202`.** The `upgrade_device_ptr` `# Safety` list states the four
  preconditions of design § 2 and cites CUDA 12.6 `cuda.h` (`cuMemFree`)
  and the CUDA Programming Guide (stream-ordered memory allocation). The
  vendored "matching allocator" bullet is removed. Behaviour unchanged.
  Tests `a_private_pointer_upgraded_onto_a_default_context_returns_to_its_pool`
  and `a_synchronous_pointer_upgraded_onto_a_private_context_frees_cleanly`:
  no RED (characterization), pass.
- **Finding, test not landed.**
  `a_private_pointer_freed_synchronously_returns_to_its_pool` fails as
  specified on driver 580.178.04: after `result::free_sync` (`cuMemFree`)
  of a private-pool pointer and a synchronize, `USED_MEM_CURRENT` stays at
  the freed size (`left: Ok(1048576)  right: Ok(0)`, cuda-12060 and
  cuda-13040). Probe runs, not committed:
  - private pool, 48 MiB + 1 MiB live: used 51380224, reserved 67108864;
    after `cuMemFree` of each and a context synchronize, reserved fell to
    33554432 and then 0, and `cuMemGetInfo` free memory rose by 32 MiB
    per step, but used stayed 51380224;
  - polling for 1 s and a later allocate/free cycle did not change it;
  - the default pool behaves the same: used 0 → 1048576 after the
    allocation, still 1048576 after `cuMemFree`.

  So `cuMemFree` does release the memory, as design § 2 and NVIDIA say,
  but this driver does not decrement the pool's `USED_MEM_CURRENT`. The
  test's assertion (requirements CF-AC-03, test plan § 3.2: "same, after
  `result::free_sync`") cannot hold here. The catalogue row stays
  `landed = false`. Needs a design and test-plan disposition (for example
  asserting reserved and `cuMemGetInfo` instead, and documenting the
  counter). Possibly related: a context of a device without async
  allocation frees an upgraded pool pointer with `cuMemFree` too.

### Deviations

- The two read-back seam tests
  (`max_pool_size_read_back_uses_the_driver_value`,
  `a_failed_max_pool_size_read_leaves_create_successful`) compile only
  from cuda-13020, so the catalogue lists cell G3, not the test plan's G1:
  G1 builds cuda-12060, whose bindings have no `MAX_POOL_SIZE`, so the
  read-back they test does not exist there. Test plan § 3.1 needs the
  same edit.
- Adding `max_size` to `CudaMemPool` made the vendored test
  `a_pool_of_another_device_is_rejected` add `max_size: None` to its
  struct literal (`7889643`).
- `attributes_newer_than_the_driver_are_never_requested` uses the real
  driver version and prints SKIP on a driver ≥ 13020, where the gate opens.

### Verification at `265ce42`

- `cargo fmt --check`: clean.
- 19 features, driver-only (`--no-default-features -F
  std,driver,dynamic-loading,cuda-<v>`): `cargo check --locked --lib`
  passes with no warnings; `cargo clippy --locked --lib -- -D warnings`
  passes; `cargo clippy --locked --lib --tests -- -D warnings` fails at
  every feature with exactly the five upstream errors in
  `src/driver/safe/launch.rs` (unused imports at 276 and 280, unused
  `SIN_CU`, `TEST_KERNELS`, `SLOW_KERNELS`). The base `ec1cac71` gives the
  same five at 12060 and 13040 (checked from a `git archive` copy), so
  they are upstream, not fork. Fork CI's clippy command
  (`--all-targets` with `CLIPPY_FEATURES`) passes at cuda-13040, 12060 and
  11040 on x86_64.
- Default features, cuda-12060 and cuda-13040: `cargo check --locked
  --lib` and `cargo clippy --locked --lib --tests -- -D warnings` pass.
- Pure-test job, simulated locally for all 19 features (both steps,
  `CUDA_VISIBLE_DEVICES=` empty): every listing passes `unlanded` and
  `check --mode list` (16 tests in `fathomdb_private_pool` up to
  cuda-13010, 20 from cuda-13020); the cell-`pure` run passes with 3
  names up to cuda-12010 and 2 from cuda-12020, `count-tests: PASS`.
- Hosted-test job, simulated at cuda-12060 without a GPU: 27 passed, 28
  filtered out, `count-tests: PASS`, `unlanded: PASS`.
- GPU, `-- fathomdb_ --test-threads=1 --nocapture`, each checked by
  `count-tests.py check` against `expect --filter fathomdb_`; no SKIP line
  in any run:

  | Build | alloc_fallback | private_pool | Summary | count-tests |
  | --- | --- | --- | --- | --- |
  | default 12060 | 11/11 | 16/16 | 27 passed; 0 ignored; 75 filtered out | PASS |
  | driver-only 12060 | 11/11 | 16/16 | 27 passed; 0 ignored; 28 filtered out | PASS |
  | driver-only 13000 | 11/11 | 16/16 | 27 passed; 0 ignored; 26 filtered out | PASS |
  | driver-only 13020 | 11/11 | 20/20 | 30 passed; 1 ignored; 23 filtered out | PASS |
  | driver-only 13040 | 11/11 | 20/20 | 30 passed; 1 ignored; 23 filtered out | PASS |

  The ignored test is `max_pool_size_reads_back_at_least_the_request`.
  The FC4-W "expected CI failures" (13020+ compile error, x86_64
  `in_fresh_process`) are fixed.

## Code commits since `ec1cac71` (CF-AC-01)

| Commit | Class | Purpose |
| --- | --- | --- |
| `4869fab` | W | fork CI workflow |
| `9f02c3a` | L | items 1–4 (allocator fallback and its tests) |
| `ce0b56e` | U | item 5 (private memory pool and its tests) |
| `3fbccda` | L | item 6 (marker feature); last item commit, equal to tree A |
| `dfff1a1` | U | FC5-a RED: `device_location` test |
| `6f4c67a` | U | FC5-a GREEN: `device_location` helper (D-8) |
| `49ef889` | U | FC5-a RED: `pool_props_are_zeroed_then_written` |
| `e37097d` | U | FC5-a GREEN: `maxSize` cfg names the old bindings |
| `cf5d145` | U | `in_fresh_process` cfg-gated to aarch64 Linux |
| `536ec11` | U | FC5-b RED: unsupported `max_size` tests and seams |
| `4dc1ec7` | U | FC5-b GREEN: `NOT_SUPPORTED` paths |
| `7889643` | U | FC5-b staging: ungated `MAX_POOL_SIZE` read-back, `max_size()` |
| `5a342fe` | U | FC5-b RED: gate test and attribute counter |
| `2f936a5` | U | FC5-b GREEN: driver-version gate |
| `0314e34` | U | FC5-b RED: read-back handling tests |
| `f0e72b3` | U | FC5-b GREEN: read-back handling |
| `c44d3b7` | U | G5b read-back test (ignored) |
| `75f13fa` | U | FC5-c typed reads replace `attribute()` |
| `a11a045` | U | FC5-c characterization tests |
| `9602202` | U | FC5-d `# Safety` text and two § 3.2 tests |
| `265ce42` | U | seam modules moved last (clippy `items_after_test_module`) |
