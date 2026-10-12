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
| 2026-10-12 | FC6-a measurements M-1, M-2, M-3 | — (measurement) | — | `6c37447` | — |
| 2026-10-12 | FC6-a § 3.1, 16 tests | — (no RED: characterization; none failed) | — | `32f49c7`, `51a2073` | — |
| 2026-10-12 | FC6-a § 3.2 revised test; `# Safety` caveat | — (no RED: characterization and doc) | — | `6db7d69` | — |
| 2026-10-12 | FC6-a § 3.3, 3 tests | — (no RED: characterization; base comparison) | — | `bfdff00` | — |
| 2026-10-12 | FC6-b T-UNIT, 8 tests of `fathomdb_lib_major` | `17f3fa0` | 8 of 9 fail behaviourally against base-equivalent stubs (verbatim below) | `8470cc2` (9 of 9) | pending (R-FC-CODE) |
| 2026-10-12 | FC6-b `out_of_scope_and_cuda11_candidates_equal_the_base` | — (no RED: regression guard; passes at `17f3fa0` and `8470cc2`) | — | `17f3fa0` | — |
| 2026-10-12 | FC6-b T-DEV (a)–(d) | `8aa03b5` (run on a `git archive` of `068779b`, pre-item-9) | (a) maps cu13 `libcublas.so.13`; (b) maps 12.6 `libcublas.so.12`; (c) `culib()` panics; (d) `culib()` binds the cu13 cuFile (1150) | `8470cc2` (run at `7a84fce`) | pending (R-FC-CODE) |
| 2026-10-12 | FC6-b T-DEV (f) | — (no RED: loads at base and at `C`) | — | `8aa03b5` | — |
| 2026-10-12 | FC6-b T-DEV (e) | — (no RED; G5b only) | — | pending (G5b) | — |
| 2026-10-12 | FC6-b `template_loader_tail_matches_the_checked_in_cublas_loader` | `91b1317` | the template renders the base tail, not in `src/cublas/sys/mod.rs` | `160a0c4` | pending (R-FC-CODE) |
| 2026-10-12 | FC6-b count-tests: other binaries, `--binary` | `43bff6d` | 2 of 38 fail: unexpected sections `test:fathomdb_lib_major`, `unittests:bindings_generator` | `8889071` (38 pass) | pending (R-FC-CODE) |
| 2026-10-12 | FC6-b count-tests: `- should panic` lines | `2e5de53` | 1 of 39 fails: `{} != {'…test_alloc_pinned_panics_on_size_overflow': 'ok'}` | `5e28fa7` (39 pass) | pending (R-FC-CODE) |
| 2026-10-12 | FC6-b pool doc (D-2, D-9) | — (doc) | — | `e6e5e0f` | — |
| 2026-10-12 | FC7 packaging (item 8) | — (packaging; test plan § 10: no RED) | — | `d04ea18` | — |

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

## FC6-a (classes U, L and D)

Test plan § 10 row FC6-a: the § 3.1 and § 3.3 characterization tests and
the revised § 3.2 test. None has a RED, and **no test failed against the
code**, so no product code changed. Host `windchill3`; every GPU run used
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0` (NVIDIA GeForce RTX
3090, 00000000:41:00.0) or `0,1` for G2 (adds 00000000:42:00.0; the K620,
index 2, was never visible), `CUDARC_CUDA_VERSION` = the feature,
`Cargo.lock.baseline` in place (SHA-256 `3d399b4e…747923`) and `--locked`.
The guard was taken before and after every GPU run (first
2026-10-12T08:16:01-05:00, last 2026-10-12T08:40:41-05:00): driver
`580.178.04` on all three GPUs, `/usr/local/cuda` → `/usr/local/cuda-12.6`,
never changed. Logs are kept on the host in the session scratchpad.

### Measurements (M-1, M-2, M-3)

Landed first (`6c37447`) as ignored tests that only print. Verbatim, from
the default-12060 run at `51a2073` (the driver-only 12060, 13000 and 13040
runs printed identical lines):

```text
M-1 size=1 empty=0 first_step=33554432 second_step=0 used_after_two=2 device_free=24983568384 device_total=25296044032
M-1 size=1048576 empty=0 first_step=33554432 second_step=0 used_after_two=2097152 device_free=24983568384 device_total=25296044032
M-1 size=33554432 empty=0 first_step=33554432 second_step=33554432 used_after_two=67108864 device_free=24950013952 device_total=25296044032
M-1 size=268435456 empty=0 first_step=268435456 second_step=268435456 used_after_two=536870912 device_free=24480251904 device_total=25296044032
M-2 cap=67108864 size=1048576 threshold=0 successes=64 limit=67108864 error=Some(DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")) reserved_at_failure=67108864 used_at_failure=67108864 max_size()=None
M-2 cap=67108864 size=1048576 threshold=67108864 successes=64 limit=67108864 error=Some(DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")) reserved_at_failure=67108864 used_at_failure=67108864 max_size()=None
```

M-3, driver-only cuda-13040 (cuda-13020 printed the same for 9–14; it has
no attribute 15):

```text
M-3 driver=13000 attribute=9 (CU_MEMPOOL_ATTR_ALLOCATION_TYPE) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
M-3 driver=13000 attribute=10 (CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
M-3 driver=13000 attribute=11 (CU_MEMPOOL_ATTR_LOCATION_ID) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
M-3 driver=13000 attribute=12 (CU_MEMPOOL_ATTR_LOCATION_TYPE) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
M-3 driver=13000 attribute=13 (CU_MEMPOOL_ATTR_MAX_POOL_SIZE) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
M-3 driver=13000 attribute=14 (CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
M-3 driver=13000 attribute=15 (CU_MEMPOOL_ATTR_LOCALITY_DOMAIN_ID) result=Err(DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")) value=0x0
```

- **M-1.** A pool reserves 32 MiB for its first allocation up to 32 MiB;
  smaller allocations share that chunk. Above 32 MiB the step equals the
  allocation (256 MiB → 256 MiB). `USED_MEM_CURRENT` counts the exact bytes
  requested (two 1-byte allocations read 2).
- **M-2.** With `maxSize` 64 MiB and 1 MiB requests, 64 succeed and the
  65th fails with `CUDA_ERROR_OUT_OF_MEMORY`, at reserved = used = 64 MiB,
  at both thresholds. So the cap bounds reserved memory, and the "M-2
  limit" for 1 MiB is exactly the cap. `grow_until_failure` is the M-2
  procedure; the capacity tests and the model test call it (through
  `m2_limit`) in-process instead of using a constant.
- **M-3.** Driver 580 rejects every attribute above 8 with
  `CUDA_ERROR_INVALID_VALUE`. This is an observation only; the gate never
  relies on it (design D-4).

### Tests landed

- **§ 3.1 (`32f49c7`, U).** The 16 remaining tests. Notes:
  - threshold and trim measure their chunk in-process
    (`reservation_step`: 32 MiB for a 32 MiB working size) and assert the
    precondition reserved > max(T or k, used) + chunk before acting
    (4 × 32 MiB reserved, one live). The threshold test reads its
    precondition after `cuStreamQuery` reports the stream idle, so no
    synchronize applies the threshold early;
  - the capacity tests derive the limit with `m2_limit` in a pool of their
    own, then work in a fresh pool. `requests_within_the_cap_succeed` also
    asserts limit ≤ cap and limit + chunk > cap;
  - T-CONC printed `65 requests, 64 succeeded, 1 failed, limit 67108864`;
  - stream order: with the latch held, C's buffer did not reuse X's
    address and B read X's pattern. **Mutation check, not committed:** with
    event tracking disabled, C reused X's address and B read 1048576 bytes
    of C's pattern (test FAILED), so the test is not vacuous;
  - `pool_model_property`: 256 cases (seed = case index) × 64 operations.
    Each case uses one size from {1, 4, 32} MiB, so the capped pool cannot
    fragment and "success iff live + s ≤ limit" is exact. A free on another
    stream is `join` + `leak` + `cuMemFreeAsync`. Totals: `5732 allocations,
    814 out-of-memory as predicted, 4100 frees (2778 on another stream),
    3245 synchronizes, 2493 trims; limits [67108864, 67108864, 67108864]`.
    It takes about 48 s in a debug build;
  - T-CAPTURE succeeded on G1 (required): alloc and free captured on a
    private context with `maxSize` 64 MiB instantiate and launch; graph
    `USED_MEM_CURRENT` rose `0 -> 33554432`; private and default pool
    counters were unchanged. This also answers design D-9's unconfirmed
    point for driver 580: an allocation node accepts the parameters of a
    pool with nonzero `maxSize`;
  - `drops_after_a_sticky_error_neither_panic_nor_abort` (ignored, needs
    `nvrtc`): the child printed `CHILD sticky error:
    DriverError(CUDA_ERROR_ASSERT, "device-side assert triggered")` and
    `CHILD dropped everything`, and exited with status 0;
  - `exit_with_a_live_pool_is_clean`: the child printed `CHILD live: used
    1048576` and `process::exit(0)` gave status 0.
- **§ 3.2 (`6db7d69`, U).**
  `a_private_pointer_freed_synchronously_releases_its_memory` per the
  revised row: three 32 MiB pointers freed with `result::free_sync` after a
  synchronize; `trim_to(0)` left reserved ≤ one chunk and `cuMemGetInfo`
  recovered to within one chunk (preconditions: more than one chunk
  reserved and at least two chunks taken). `USED_MEM_CURRENT` printed
  `Ok(100663296)` (cuda-12060 and 13040): stale, as design D-2 says. The
  `upgrade_device_ptr` `# Safety` text gains the measured caveat and the
  advice to prefer `cuMemFreeAsync` for pool memory.
- **§ 3.3 (`bfdff00`, L).** A `cfg(test)` thread-local counter in
  `result::device::get_default_mem_pool`.
  - `off_target_constructors_never_query_the_default_pool`: positive
    control 1, then 0 queries across `new`, `new_non_primary`, `new_cig`
    and `from_raw_context`. On Linux `new_cig` returns
    `CUDA_ERROR_NOT_SUPPORTED` (it needs a D3D12 queue), so that arm builds
    no context on G1; the other three must succeed. **Mutation check, not
    committed:** a default-pool query in the off-target
    `select_async_alloc` failed the test with `left: 3, right: 0`.
  - `off_target_choice_equals_the_upstream_inline_rule`: `has_async_alloc()`
    equals `MEMORY_POOLS_SUPPORTED > 0` for each context built.
  - `off_target_zero_length_sync_behaviour_matches_the_base`: base run
    first. `git archive` of `../cudarc-baseline` (`ec1cac71`, left
    unmodified) into the scratchpad, a probe module appended to
    `src/driver/safe/core.rs` whose `zero_length_sync_outcomes` has the same
    text as the fork's (rustfmt wraps one line differently; diff of the
    probe against the base file, SHA-256 `81a42696…df20e39`), run
    `-F cuda-12060 -- --exact
    driver::safe::core::fathomdb_base_probe::base_zero_length_sync_outcomes`
    on the same GPU:

    ```text
    BASE cuMemAlloc(0): Err(CUDA_ERROR_INVALID_VALUE)
    BASE alloc(0): Err(CUDA_ERROR_INVALID_VALUE)
    BASE alloc_zeros(0): Err(CUDA_ERROR_INVALID_VALUE)
    BASE null(): Err(CUDA_ERROR_INVALID_VALUE)
    BASE check_err: Ok(())
    test driver::safe::core::fathomdb_base_probe::base_zero_length_sync_outcomes ... ok
    ```

    The fork test asserts these five lines as constants, and it passes. So
    the test can be expressed against base API (`has_async_alloc` is a
    `pub(crate)` field at the base too).
- **`51a2073` (U).** The child-process tests printed the child's own
  libtest lines, which `count-tests.py` then counted as the parent's under
  `--nocapture` (count-tests FAIL on a passing run). The child's lines are
  now prefixed `child| `. Nothing asserted changed.
- **Catalogue (D):** `1b9116a`, `666634f`, `1062281`, `b6d99bc`. Every
  § 3.1–3.3 row is now `landed = true`; only the ten item-9 rows (§ 3.4)
  remain unlanded.

### Runs at `51a2073`

Each run is `cargo test --locked --lib <features> -- fathomdb_
--test-threads=1 --nocapture`, then each ignored fork test the cell can run
alone with `-- --ignored --exact <name> --nocapture`, each checked by
`count-tests.py check` against `expect` for the same set, version and
filter. No run printed a `SKIP` line.

| Cell | Main run | count-tests | Ignored, one by one |
| --- | --- | --- | --- |
| G1 default 12060 | 45 passed; 0 failed; 5 ignored; 75 filtered out | PASS | `drops_after_a_sticky_error_…`, `measure_max_size_semantics`, `measure_reservation_steps`: 1 passed each, PASS |
| driver-only 12060 | 45 passed; 0 failed; 4 ignored; 28 filtered out | PASS | `measure_max_size_semantics`, `measure_reservation_steps`: PASS |
| driver-only 13000 | 45 passed; 0 failed; 4 ignored; 26 filtered out | PASS | the same two: PASS |
| driver-only 13040 | 48 passed; 0 failed; 6 ignored; 23 filtered out | PASS | the two measurements and `observe_new_attributes_on_driver_580`: PASS. `max_pool_size_reads_back_at_least_the_request` fails its precondition as designed (`driver 13000 cannot report MAX_POOL_SIZE`; G5b only) |
| G2 default 12060, `CUDA_VISIBLE_DEVICES=0,1` | — | — | `two_pools_on_two_devices_are_independent`, `a_pool_of_device_1_is_rejected_for_device_0`: 1 passed each, PASS |

Main-run log SHA-256: default 12060 `a22ab61e…8f191`, driver-only 12060
`eb43aac5…f9050`, 13000 `e77b443e…20432`, 13040 `5f183b0d…4d720`.

Also checked at `51a2073`:

- hosted-job simulation (driver-only 12060, `CUDA_VISIBLE_DEVICES=` empty):
  45 passed, 4 ignored, `count-tests: PASS`, 31 SKIP lines;
- `cargo fmt --check` clean; `cargo clippy --locked --lib --tests -- -D
  warnings` clean with default features at cuda-12060 and cuda-13040;
  driver-only `--tests` clippy at 11040, 12060 and 13040 shows only the five
  upstream `launch.rs` errors listed under FC5;
- `cargo check --locked --lib`, driver-only, all 19 features: no
  diagnostics. `cargo check --lib --tests`, driver-only, all 19 features:
  no fork diagnostics;
- module listings (`-- fathomdb_ --list`) at driver-only 11040, 12010,
  12050, 13010 and 13030, and full listings at default 12060 and
  driver-only 12060, 13000, 13020 and 13040: `count-tests unlanded: PASS`
  and `check --mode list: PASS`;
- `python3 -I fathomdb/scripts/test_count_tests.py`: OK.

### Deviations

- `observe_new_attributes_on_driver_580` compiles from cuda-13020 and runs
  in G3 (13020–13040), not G1: the cuda-12060 bindings that G1 builds have
  no attributes 9–15. Attribute 15 is asked only from cuda-13040. Test plan
  § 3.1 needs the same edit.
- The two X tests run their child role in the same test function, chosen
  by the environment variable `FATHOMDB_CUDARC_TEST_CHILD`, because the
  catalogue names are binding and a separate helper test would add a name.
- `drops_after_a_sticky_error_neither_panic_nor_abort` compiles only with
  `nvrtc` (catalogue cfg `features = ["driver", "nvrtc"]`), so the
  driver-only cells do not run it.
- The three § 3.3 tests compile off aarch64 Linux only; their catalogue cfg
  is `arch = ["x86_64"]`.
- The model test fixes one allocation size per case (see above); the test
  plan's "allocate size s" does not say whether sizes mix within a case.
- The aarch64 build was not compiled locally (no aarch64 target installed);
  fork CI's aarch64 jobs cover it. The new private-pool tests have no arch
  cfg; the new fallback items are all cfg'd off aarch64 Linux.

### Notes for the orchestrator (not changed; outside FC6-a)

- The doc of `CudaContext::new_with_mem_pool` still says a leaked pointer
  "must be freed with `cuMemFreeAsync`, so it may be upgraded only onto a
  context whose `has_async_alloc` is `true`". That contradicts design D-2
  and the corrected `upgrade_device_ptr` `# Safety` text.
- Design D-9 decides to document on `new_with_mem_pool` that graph memory
  is outside the pool's budget; that doc is not present.

## FC6-b (classes U, D and W)

Test plan § 10 row FC6-b and § 3.4: the T-UNIT REDs, the T-DEV REDs, item 9,
then the template test. Host `windchill3`; every GPU run used
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` (NVIDIA GeForce RTX 3090,
00000000:41:00.0), `CUDARC_CUDA_VERSION` = the feature and
`Cargo.lock.baseline` in place. The guard was taken before and after every
GPU run (first 2026-10-12T09:07:12-05:00, last 2026-10-12T09:34:29-05:00):
driver `580.178.04` on all three GPUs, `/usr/local/cuda` →
`/usr/local/cuda-12.6`, never changed. Logs are kept on the host in the
session scratchpad.

### Scratch venvs (`~/fathomdb-forks/venvs`, PyPI, `pip install --no-deps`)

| Venv | Pins |
| --- | --- |
| `cu12-121` | `nvidia-cublas-cu12==12.1.3.1` (newest 12.1 patch on PyPI) |
| `cu12-124` | `nvidia-cublas-cu12==12.4.5.8` (newest 12.4 patch) |
| `cu12-129` | `nvidia-cublas-cu12==12.9.2.10` (newest 12.9 patch) |
| `cu13-130` | the eight test plan § 3.4 pins, exactly |

`--no-deps` keeps each venv to its pins (the wheels' own dependencies would
add unpinned packages).

### T-UNIT

- **RED `17f3fa0`.** The nine tests plus compile-only stubs that reproduce
  the base (`lib_candidates` = the `ec1cac71` list, `version_verdict`
  accepts everything, `LibCell` retries a failure and probes presence
  separately as upstream's `is_culib_present` does, upstream's panic text,
  an empty linked-build report). `get_lib_name_candidates` and the loaders
  were unchanged. `cargo test --locked --lib -F
  cuda-12060,$CHECK_DEFAULT_FEATURES -- fathomdb_lib_major::` (the same at
  cuda-11040 and cuda-13000):

  ```text
  test result: FAILED. 1 passed; 8 failed; 0 ignored; 0 measured; 151 filtered out
  "CUDA 13" missing from "Unable to dynamically load the \"cublas\" shared library - searched for library names: [\"libcublas.so.13\", \"libcublas.so\"]."
  a_failed_load_is_memoized: left: 2 right: 1 (cudart: loader calls across two try_culib calls)
  44 disagreements: ["cudart [false, true] present_first=false: is_present true, try false, try again false", …]
  the_report_says_linked_at_build_time_when_linked: left: "" right: "linked at build time"
  67 wrong verdicts: ["cudart Int(12060) (line 12) for CUDA 13: Accept, expected Reject", …]
  126 mismatches: ["cudart 12.0: first candidate Some(\"cudart.dll\"), wheel ships cudart64_12.dll", …]
  357 lists out of order, e.g. ["Linux cudart CUDA 12: starts with [\"libcudart.so\"], expected [\"libcudart.so.12\"]", …]
  4314 candidate names that are not the built major's versioned name or the unversioned name, …
  ```

  The test plan's named witnesses are among the dumped names: `Linux cublas
  CUDA 13: libcublas.so.12` (6 lines at cuda-13000, one of them from the
  base `get_lib_name_candidates("cublas")` of that build) and `Windows cufft
  CUDA 12: cufft64_12.dll` (18 lines). Log SHA-256: 12060 `409f550f…26d8d`,
  13000 `96222190…20527`, 11040 `f19d6492…dc960`.
- **GREEN `8470cc2`.** 9 of 9 pass. Fork CI's T-UNIT job simulated on Linux
  for all 19 features at `7a84fce` (both steps: listing with `unlanded` and
  `check --mode list`, then the run by name with `check`): 9 names and
  9 passed per feature, `count-tests: PASS` for every step, 171 in total.
  The Windows half runs only in fork CI.

### T-DEV (G1)

`fathomdb/scripts/t-dev-lib-major.sh` (cases a–d and f; e needs G5b). The
base is a `git archive` of `068779b` (the fork before item 9: the loaders
and `get_lib_name_candidates` equal `ec1cac71`; the T-DEV test file at
`8aa03b5` uses only base API), built with its own target directory.

| Case | Test | Before item 9 (`068779b`) | At `C` (`7a84fce`) |
| --- | --- | --- | --- |
| (a) | `cuda12_build_skips_an_unversioned_cuda13_cublas` | FAILED: `MAPS a …/venvs/cu13-130/…/nvidia/cu13/lib/libcublas.so.13` | ok: `MAPS a /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so.12.6.1.4`; `REPORT a cublas: opened as libcublas.so.12, path /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so.12.6.1.4, version 12.6.1, checked against CUDA 12` |
| (b) | `cuda13_build_rejects_the_cuda12_cublas_from_the_loader_cache` | FAILED: `RESULT b culib() returned`; `MAPS b /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so.12.6.1.4` (and `libcublasLt.so.12.6.1.4`) | ok: `MAPS b (no libcublas mapping)`; `Tried: libcublas.so.13 (not found), libcublas.so (wrong version 12.6.1)` |
| (c) | `no_acceptable_cublas_is_a_typed_error_not_a_panic` | FAILED: `culib()` panicked with the base message (17 names) | ok: `Tried: libcublas.so.12 (not found), libcublas.so (not found)`; no mapping; exit 0 |
| (d) | `cuda12_build_rejects_an_already_loaded_cuda13_cufile` | FAILED: `RESULT d cuFileGetVersion through culib(): Ok(1150)`; `a CUDA 12 build uses the CUDA 13 cuFile (version 1150)` | ok: `Tried: libcufile.so.0 (already loaded with wrong version 1.15)`; `MAPS d-after …/nvidia/cu13/lib/libcufile.so.0` only (NODELETE) |
| (f) | `nvtx_loads_with_only_libnvtoolsext_so_1` | ok: `MAPS f /usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvToolsExt.so.1.0.0` | ok: the same |
| (e) | `driver_loads_with_only_libcuda_so_1` | pending: G5b | pending: G5b |

Preconditions (helper process `t-dev-probe.py`, logged as `HELPER` lines):
(a) `dlopen(libcublas.so)` maps the cu13 `libcublas.so.13` and
`libcublasLt.so.13`; (b) `ldconfig -p`: `libcublas.so` →
`libcublas.so.12.6.1.4`, and `libcublas.so.13` fails; (c) under `ld-linux
--inhibit-cache` with an empty directory, `libcublas.so`, `.so.12` and
`.so.13` all fail; (d) `dlopen(libcufile.so)` maps the cu13
`libcufile.so.0`; (f) under `--inhibit-cache`, `libnvToolsExt.so.1` opens
and `libnvToolsExt.so` fails. Run logs: RED `1f449a1e…2a752`, GREEN
`f7707f52…c45f3`.

- **Case (d) reading.** The test opens the decoy `libcufile.so` in-process
  first, then closes it, and the cu13 `libcufile.so.0` stays mapped
  (`DF_1_NODELETE`), which is the state the test plan's `C` column
  (`AlreadyLoadedWrongVersion`, "the object stays mapped") needs. Without
  that preload the decoy is never reached at `C`: `.so.0` comes first and
  resolves through the loader cache, so `C` loads the system cuFile
  (observed with the G8 report program and the decoy directory alone:
  `REPORT cufile: opened as libcufile.so.0, path
  /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcufile.so.1.11.1,
  version 1.11`, nothing mapped from the venv). Needs a test-plan wording
  confirmation (R-FC-CODE).
- **Case (c) GREEN body.** RED calls `culib()` (base API); `culib()` still
  panics by design, so GREEN calls `try_culib()` and asserts the typed error
  (test plan § 3.4: new API only at GREEN). Cases (a), (b) and (d) keep
  their RED assertions and add new-API ones.
- **Missed at RED:** `8aa03b5` compiled case (d) on every feature, but the
  `cuFileGetVersion` binding starts at cuda-12030, so `cargo check
  --all-targets` failed at cuda-11040–12020 until `8470cc2` gated it.

### Template test

`template_loader_tail_matches_the_checked_in_cublas_loader` is in
`bindings_generator/src/loader_tail.rs` (module `loader_tail::tests`). The
generator crate cannot be built on `windchill3`: `openssl-sys` (through
`reqwest`) needs the OpenSSL headers, which are not installed, and the
host rules forbid apt. So the test ran in a scratch crate that includes
`loader_tail.rs` by `#[path]` with the generator's `proc-macro2 1.0.94` and
`quote 1.0.40`, and fork CI's new `pure-test-generator` job (`7a84fce`)
runs it in the generator crate. `merge.rs`/`main.rs` (`91b1317`) are
therefore compiled only by CI.

- **RED `91b1317`** (the tail moved, unchanged, into `loader_tail`): `test
  result: FAILED. 0 passed; 1 failed`, `the template's cuBLAS loader tail is
  not in …/src/cublas/sys/mod.rs`. Positive control: the same rendering is
  contained in `068779b:src/cublas/sys/mod.rs`.
- **GREEN `160a0c4`.** 1 passed. A one-off check (not committed) rendered
  the template for all sixteen modules and found each tail in its
  checked-in file.

### G8 (`fathomdb/scripts/g8-*.sh`, FC6 evidence; receipts at FC8)

`g8_report.rs` is built as a scratch binary against the tree; it calls
`try_culib` under `catch_unwind` and cross-checks the report with
`/proc/self/maps`. Runs at `7a84fce`, all `PASS`:

| Sub-cell | Setup | Result |
| --- | --- | --- |
| G8-a | 12060, `cu12-124` | `REPORT cublas: opened as libcublas.so.12, path ~/fathomdb-forks/venvs/cu12-124/…/nvidia/cublas/lib/libcublas.so.12, version 12.4.5`; cuBLASLt the same; upstream `cublas` tests 8 passed |
| G8-a | 12060, `cu12-129` | version 12.9.2 from the venv; 8 passed |
| G8-b | 12060, `cu12-121` | version 12.1.3 from the venv; 8 passed (no missing symbol, no panic) |
| G8-c | 13000, `cu13-130` | `opened as libcublas.so.13 … version 13.0.2, checked against CUDA 13`; 8 passed |
| G8-d | 12060, cu13 lib dir first | `curand: opened as libcurand.so, path /usr/local/cuda-12.6/…/libcurand.so.10.3.7.68, version 10.3.7`; `cusparse: … libcusparse.so.12.5.3.3, version 12.5.3`; nothing mapped from the venv. With `--inhibit-cache`: `libcurand.so.10 (wrong version 10.4.0), libcurand.so (not found)` and `libcusparse.so.12 (wrong version 12.6.3), libcusparse.so (not found)` |
| G8-e | 12060, cu13 `libcurand.so.10` preloaded | `libcurand.so.10 (already loaded with wrong version 10.4.0)` |
| G8-f | 12060, cu13 lib dir first, cuSOLVER | `libcusolver.so.11 (dependency libcusparse.so.12 has wrong version 12.6.3), libcusolver.so (dependency libcusparse.so.12 has wrong version 12.6.3)`; nothing mapped from the venv |

### Runs at `7a84fce` (k = 1, FC6 evidence)

| Run | Result | count-tests (`expect --binary unittests:cudarc`) |
| --- | --- | --- |
| G1 default 12060, `--lib -- --test-threads=1` | 124 passed; 0 failed; 10 ignored | PASS (`fathomdb_lib_major` 9 of 9) |
| default 13000 with `cu13-130` (G4 setup) | 122 passed; 0 failed; 10 ignored | PASS |
| hosted-test run 1 simulation (driver-only 12060, no GPU) | 54 passed; 4 ignored; 28 filtered out | PASS; `unlanded` PASS |
| pure-test simulation, 19 features | every step `count-tests: PASS` (4 names to 12010, 3 from 12020) | — |

Log SHA-256: G1 `203f1f87…213c8`, 13000 `d32d6d08…7ac7`.

### Checks at `5e28fa7`

| Check | Result |
| --- | --- |
| `cargo fmt --check` | clean |
| `cargo check --locked --lib`, 19 features × {driver-only, default} | 38 of 38, no warnings |
| `cargo check --all-targets -F cuda-<v>,$CHECK_DEFAULT_FEATURES`, 19 features | 19 of 19, no warnings |
| `cargo check --all-targets --no-default-features -F cuda-<v>,$CHECK_NO_STD_FEATURES`, 19 features | 19 of 19, no warnings |
| `cargo clippy --locked --lib --tests -- -D warnings`, default 12060 and 13040 | clean |
| fork CI clippy (`$CLIPPY_FEATURES`, `--all-targets`), 11040, 12060, 13000, 13040 | clean |
| `cargo check --lib`, `dynamic-linking` with the G8 libraries, 12060 | clean |
| `python3 -I fathomdb/scripts/test_count_tests.py` | 39 tests OK |
| `actionlint`, `shellcheck -x` on the new scripts | clean |
| Windows and aarch64 | not compiled locally (no targets installed); fork CI |

### Deviations

- **`RejectReason::NoVersionQuery`** is added: design § 12.4 names the
  reason (cuFile without `cuFileGetVersion`, the `note` field) but its
  enum sketch omits it.
- **cuSolverMg** has no unversioned candidate ("versioned name only").
  **Windows cuFile** has no candidates ("not available"), so its error
  lists no names.
- **Resolved path:** `dladdr` and `dl_iterate_phdr` (Linux),
  `GetModuleHandleExW` (from the symbol address) and `GetModuleFileNameW`
  (Windows) through `extern` blocks; canonicalization through
  `std::fs::canonicalize`, which is `realpath(3)`. No new crate.
- **The already-loaded probe** is Linux only (`RTLD_NOLOAD`), as is the
  dependency check (design: Windows gap). A dependency without a decodable
  version is not judged.
- **A `W` commit** (`7a84fce`): the template test lives in another binary,
  so fork CI needs a job for it; the per-feature pure-test steps skip it.
- **count-tests** (`D`, TDD): sections of binaries other than the lib with
  no selected test are omitted, `expect --binary` selects binaries, and
  `- should panic` lines parse. FC8 receipts of unfiltered `cargo test
  --lib` runs must use `--binary unittests:cudarc`.
- **G8 report program** is a scratch binary, not a crate test, so no test
  name was added. The G8 scripts check the upstream tests' outcomes but not
  their names (FC8).
- **Untracked file** `log_matmul_f32` in the checkout root, written by the
  upstream `cublaslt::safe::tests::test_matmul_f32` with cuBLASLt 13.0
  logging; left in place (deleting needs approval).

### Notes for the orchestrator

- `get_lib_name_candidates`'s doc and the generated loader docs cite
  `FATHOMDB.md` item 9, which FC7 writes.
- Memoization and `is_culib_present` changes (design L2) are visible to
  callers: a failed load is never retried in the process, and
  `is_culib_present()` keeps the library loaded.

## FC7 (class P)

`d04ea18` (`P`) touches only `Cargo.toml`, `README.md` and the new
`FATHOMDB.md`:

- `Cargo.toml`: package `cudarc-fathomdb` 0.19.10, `[lib] name = "cudarc"`;
  licence and authors unchanged; description, homepage, documentation and
  repository name the fork; `exclude = ["/fathomdb/",
  "/.github/workflows/fathomdb-ci.yaml"]` (upstream had neither `include`
  nor `exclude`). The marker-feature comment cites `FATHOMDB.md` item 6.
- `README.md`: fork notice above the unchanged upstream text.
- `FATHOMDB.md`: the items 1–9 table (plan § 3), sections for items 1–4, 5,
  6 and 9 (the anchor the item-9 code docs cite), the corrected item-2
  rationale (design § 2), the behaviour changes of design § 12.4 and
  requirements CF-REQ-11 item 6, the measured facts of FC5 and FC6-a
  (driver 580 leaves `USED_MEM_CURRENT` stale after `cuMemFree`; captured
  allocations are outside the pool budget; attributes 9–15 gated on
  bindings and driver 13020; reservation step and cap), and how to depend
  on the package. Upstream status for items 5 and 9 reads "planned and not
  posted": FC9 and FC9b have not run.

Name dependence: `build.rs`, `bindings_generator`, the examples, the tests
and the doc tests use the library name `cudarc`, which `[lib]` keeps; no
`CARGO_PKG_NAME` use. The lib test binary is still
`target/debug/deps/cudarc-*`, so `unittests:cudarc` labels hold.

### Checks at `d04ea18`

| Check | Result |
| --- | --- |
| `cargo package --list` (main checkout) | refuses: `log_matmul_f32` is untracked and not ignored, so cargo counts it as an uncommitted package file |
| `cargo package --list` (scratch `git worktree add --detach d04ea18`, removed after) | 135 files; no `fathomdb/` path; no `fathomdb-ci.yaml`; `FATHOMDB.md`, `LICENSE-APACHE`, `LICENSE-MIT`, `README.md` present; list SHA-256 `83db814a…f6ddac7` |
| `cargo publish --dry-run -F cuda-13040` (same worktree) | `Packaged 135 files, 7.7MiB (633.4KiB compressed)`, verify build ok, `warning: aborting upload due to dry run`, exit 0 |
| `cargo publish --dry-run` without a feature | fails in `build.rs` (`Must specify one of the following features`), as upstream does with no `nvcc` on `PATH`; fork CI passes `-F cuda-13040` |
| `cargo fmt --all -- --check` | clean |
| fork CI clippy (`--all-targets`, `$CLIPPY_FEATURES`), 13040 and 12060 | clean |
| `cargo check --lib`, default features, 11040, 12060, 13000, 13040 | clean |
| `python3 -I fathomdb/scripts/test_count_tests.py` | 39 tests OK |
| `git diff 5b23a87 HEAD --stat -- . ':(exclude)fathomdb'` | `Cargo.toml`, `FATHOMDB.md`, `README.md` only (`Cargo.lock` is gitignored) |

### `--locked` and the baseline lock (open)

`fathomdb/qualification/Cargo.lock.baseline` names the root package
`cudarc`. With the rename, `cp …/Cargo.lock.baseline Cargo.lock` followed by
any `--locked` command fails: `error: cannot update the lock file … because
--locked was passed`. Fork CI's jobs that copy the baseline (lines 221, 343
and 459 of the workflow) and `g8-common.sh` and `t-dev-lib-major.sh` will
fail at `d04ea18` until this is resolved. Regenerating with `--offline`
changes exactly one line (`name = "cudarc"` → `name = "cudarc-fathomdb"`,
line 24); every dependency entry is byte-identical. The baseline was not
changed here: baseline runs at `ec1cac71` still need the `cudarc` root, so
the choice (a second lock, a rename at copy time, or replacing the file) is
a design decision. The runs below therefore used `--offline` with that
one-line-different lock instead of `--locked`.

### G1 smoke (plan § 8, packaging row)

Default features, cuda-12060, `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0` (NVIDIA GeForce RTX 3090, 00000000:41:00.0),
`cargo test --offline --lib -F cuda-12060 -- fathomdb_ --test-threads=1`:
`54 passed; 0 failed; 5 ignored; 0 measured; 75 filtered out`;
`count-tests check` against `expect --set def --version 12060 --filter
fathomdb_ --binary unittests:cudarc`: PASS; `unlanded`: PASS. Log SHA-256
`dae68dee…9616b05`. Guard before 2026-10-12T09:51:27-05:00 and after
2026-10-12T09:52:19-05:00: driver `580.178.04` on all three GPUs,
`/usr/local/cuda` → `/usr/local/cuda-12.6`, unchanged.

### Hosted-test run 2 (FathomDB's lock), simulated

FathomDB's `Cargo.lock` at `FATHOMDB_LOCK_SHA` (`4c327cd8`) read from the
local FathomDB clone with `git show` (not fetched; SHA-256 `5b9e6806…2fac52`),
then the workflow's steps: `cargo metadata` (which rewrites the lock with a
`cudarc-fathomdb` root), the workflow's Python check (`FathomDB's entry:
cudarc 0.19.7`, `PASS: 2 resolved dependencies equal FathomDB's lock`),
`cargo tree --locked`, and `cargo test --locked --lib --no-default-features
-F std,driver,dynamic-loading,cuda-12060 -- fathomdb_ --test-threads=1`
without a GPU: `54 passed; 0 failed; 4 ignored; 28 filtered out`,
`count-tests` PASS, `unlanded` PASS. The rename does not affect this step:
the lock is re-resolved before any `--locked` command, and the check
already accepts `cudarc` or `cudarc-fathomdb`. FathomDB itself still
vendors the crate as `cudarc` 0.19.7; when it switches to this package its
lock gains a `cudarc-fathomdb` entry, which the check also accepts.

### Notes for the orchestrator

- The `--locked` break above needs a decision before the push.
- `examples/07-build-workflow/Cargo.toml` (its own package, not in the
  crate) depends on `cudarc = { path = "../.." }`, which no longer resolves
  under the new package name; it needs `package = "cudarc-fathomdb"`. Not
  changed: outside the `P` file list.
- Source comments still cite `FATHOMDB-PATCH.md` (`core.rs`, `mem_pool.rs`)
  and "upstream 0.19.7" (`core.rs`, `SYNC_FALLBACK`). `FATHOMDB.md` says such
  citations refer to it; the comments themselves are code-tree changes.

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
| `6c37447` | U | FC6-a measurement tests M-1, M-2, M-3 |
| `32f49c7` | U | FC6-a § 3.1 characterization tests |
| `6db7d69` | U | FC6-a § 3.2 test; `# Safety` caveat on `cuMemFree` |
| `bfdff00` | L | FC6-a § 3.3 tests and the default-pool query counter |
| `51a2073` | U | child-process output prefixed |
| `17f3fa0` | U | FC6-b RED: T-UNIT tests and base-equivalent stubs |
| `8aa03b5` | U | FC6-b RED: T-DEV tests (base API) |
| `8470cc2` | U | FC6-b GREEN: item 9 (lib.rs, 11 loaders, driver report); T-DEV new-API assertions |
| `91b1317` | U | FC6-b RED: generator `loader_tail` (unchanged tail) and template test |
| `160a0c4` | U | FC6-b GREEN: item 9 tails in the generator template |
| `e6e5e0f` | U | `new_with_mem_pool` doc: leaked pointers (D-2), graph capture (D-9) |
| `7a84fce` | W | fork CI: `pure-test-generator` job; pure-test skips generator rows |
| `d04ea18` | P | item 8: `cudarc-fathomdb` package, `exclude`, README notice, `FATHOMDB.md` |
