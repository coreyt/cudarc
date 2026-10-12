# cudarc-fathomdb status

Per-step record (plan § 7, requirements CF-AC-09). One row per step. The
reviewer re-runs each RED; the last column holds that result.

| Date | Step | RED sha | RED failure | GREEN sha | Reviewer RED re-run |
| --- | --- | --- | --- | --- | --- |
| 2026-10-12 | FC4-0 `count-tests.py` | `86b8487` | 35 of 35 fixture tests error with `NotImplementedError` (stub script) | `5fc550e` (35 pass) | pending (R-FC-CODE) |
| 2026-10-12 | FC4-0 `names --with-binary` | `9f2bd1c` | 1 of 36 errors: `python3 -m unittest: error: unrecognized arguments: --with-binary` | `cafb294` (36 pass) | pending (R-FC-CODE) |
| 2026-10-12 | FC4-W fork CI | — (test plan § 10: no RED) | — | `4869fab` | — |
| 2026-10-12 | FC4-port items 1–6 | not committed (plan § 9 FC4 exception; see below) | vendored tests on bare `ec1cac71`: 37 compile errors | `9f02c3a`, `ce0b56e`, `3fbccda` (19 of 19 on RTX 3090) | pending (R-FC-CODE) |

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

## Code commits since `ec1cac71` (CF-AC-01)

| Commit | Class | Purpose |
| --- | --- | --- |
| `4869fab` | W | fork CI workflow |
| `9f02c3a` | L | items 1–4 (allocator fallback and its tests) |
| `ce0b56e` | U | item 5 (private memory pool and its tests) |
| `3fbccda` | L | item 6 (marker feature); last item commit, equal to tree A |
