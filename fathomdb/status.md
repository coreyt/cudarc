# cudarc-fathomdb status

Per-step record (plan § 7, requirements CF-AC-09). One row per step. The
reviewer re-runs each RED; the last column holds that result.

| Date | Step | RED sha | RED failure | GREEN sha | Reviewer RED re-run |
| --- | --- | --- | --- | --- | --- |
| 2026-10-12 | FC4-0 `count-tests.py` | `86b8487` | 35 of 35 fixture tests error with `NotImplementedError` (stub script) | `5fc550e` (35 pass) | not re-run by a reviewer |
| 2026-10-12 | FC4-0 `names --with-binary` | `9f2bd1c` | 1 of 36 errors: `python3 -m unittest: error: unrecognized arguments: --with-binary` | `cafb294` (36 pass) | not re-run by a reviewer |
| 2026-10-12 | FC4-W fork CI | — (test plan § 10: no RED) | — | `4869fab` | — |
| 2026-10-12 | FC4-port items 1–6 | not committed (plan § 9 FC4 exception; see below) | vendored tests on bare `ec1cac71`: 37 compile errors | `9f02c3a`, `ce0b56e`, `3fbccda` (19 of 19 on RTX 3090) | not re-run by a reviewer |
| 2026-10-12 | FC5-a `device_location_is_the_ordinal_on_every_binding` | `dfff1a1` | cuda-13020+: crate does not build, `error[E0560]` no field `id` (the D-8 defect) | `6f4c67a` | R-FC-CODE: fails at `dfff1a1` for the stated reason, passes at `6f4c67a` (review.md, R-FC-CODE: "two FC5-a REDs") |
| 2026-10-12 | FC5-a `pool_props_are_zeroed_then_written` | `49ef889` | cuda-13030/13040: `maxSize` left 0, right 4096 | `e37097d` | R-FC-CODE: fails at `49ef889` for the stated reason, passes at `e37097d` (review.md, R-FC-CODE: "two FC5-a REDs") |
| 2026-10-12 | FC5-a `in_fresh_process` dead code | — (warning fix; no behaviour) | — | `cf5d145` | — |
| 2026-10-12 | FC5-b `max_size_without_binding_support_is_not_supported` | `536ec11` | cuda-12010: left `Err(CUDA_ERROR_UNKNOWN)`, right `Err(CUDA_ERROR_NOT_SUPPORTED)` | `4dc1ec7` | not listed separately: review.md (R-FC-CODE) names "the D-4 gate" only, without its commits |
| 2026-10-12 | FC5-b `max_size_on_a_driver_older_than_12020_is_not_supported` | `536ec11` | cuda-12060, RTX 3090: left `Ok(())`, right `Err(CUDA_ERROR_NOT_SUPPORTED)` | `4dc1ec7` | not listed separately: review.md (R-FC-CODE) names "the D-4 gate" only, without its commits |
| 2026-10-12 | FC5-b `attributes_newer_than_the_driver_are_never_requested` | `5a342fe` (after the ungated read-back `7889643`) | G3 cuda-13020, driver 580: attribute requests above 8, left 1, right 0 | `2f936a5` | R-FC-CODE: "the D-4 gate" fails at RED and passes at GREEN (review.md); this counter test is the gate's G3 proof (design D-4). review.md does not give the commits |
| 2026-10-12 | FC5-b `max_pool_size_read_back_uses_the_driver_value` | `0314e34` | cuda-13020: left `Ok(())`, right `Err(CUDA_ERROR_INVALID_VALUE)` | `f0e72b3` | not listed separately: review.md (R-FC-CODE) names "the D-4 gate" only, without its commits |
| 2026-10-12 | FC5-b `a_failed_max_pool_size_read_leaves_create_successful` | `0314e34` | cuda-13020: `unwrap()` on `Err(CUDA_ERROR_INVALID_VALUE)` | `f0e72b3` | not listed separately: review.md (R-FC-CODE) names "the D-4 gate" only, without its commits |
| 2026-10-12 | FC5-b `max_pool_size_reads_back_at_least_the_request` | — (no RED at FC5: needs a driver ≥ 13020; ignored, G5b) | — | `c44d3b7` | — |
| 2026-10-12 | FC5-c typed reads; 3 characterization tests | — (no RED: the vendored code already behaves) | — | `75f13fa`, `a11a045` | — |
| 2026-10-12 | FC5-d `# Safety` text; 2 of 3 § 3.2 tests | — (no RED: doc and characterization) | — | `9602202` | — |
| 2026-10-12 | FC6-a measurements M-1, M-2, M-3 | — (measurement) | — | `6c37447` | — |
| 2026-10-12 | FC6-a § 3.1, 16 tests | — (no RED: characterization; none failed) | — | `32f49c7`, `51a2073` | no RED; R-FC-CODE re-ran "the FC6-a mutation" (review.md), without saying which of the FC6-a mutation checks (§ 3.1 stream order, § 3.3 off-target query) |
| 2026-10-12 | FC6-a § 3.2 revised test; `# Safety` caveat | — (no RED: characterization and doc) | — | `6db7d69` | — |
| 2026-10-12 | FC6-a § 3.3, 3 tests | — (no RED: characterization; base comparison) | — | `bfdff00` | no RED; R-FC-CODE re-ran "the FC6-a mutation" (review.md), without saying which of the FC6-a mutation checks (§ 3.1 stream order, § 3.3 off-target query) |
| 2026-10-12 | FC6-b T-UNIT, 8 tests of `fathomdb_lib_major` | `17f3fa0` | 8 of 9 fail behaviourally against base-equivalent stubs (verbatim below) | `8470cc2` (9 of 9) | R-FC-CODE: fails at `17f3fa0`, passes at `8470cc2`, at 12060 and 13000 (review.md) |
| 2026-10-12 | FC6-b `out_of_scope_and_cuda11_candidates_equal_the_base` | — (no RED: regression guard; passes at `17f3fa0` and `8470cc2`) | — | `17f3fa0` | — |
| 2026-10-12 | FC6-b T-DEV (a)–(d) | `8aa03b5` (run on a `git archive` of `068779b`, pre-item-9) | (a) maps cu13 `libcublas.so.13`; (b) maps 12.6 `libcublas.so.12`; (c) `culib()` panics; (d) `culib()` binds the cu13 cuFile (1150) | `8470cc2` (run at `7a84fce`) | R-FC-CODE: (a), (b) and (d) on the GPU, each failing at `068779b` for the stated reason and passing at `56bb4b7` (review.md); (c) not re-run by a reviewer |
| 2026-10-12 | FC6-b T-DEV (f) | — (no RED: loads at base and at `C`) | — | `8aa03b5` | — |
| 2026-10-12 | FC6-b T-DEV (e) | — (no RED; G5b only) | — | pending (G5b) when written; done: G5b at `b777f8d`, 3 of 3 pass (`qualification/2026-10-12-G5b-TDEV-b777f8d.md`) | — |
| 2026-10-12 | FC6-b `template_loader_tail_matches_the_checked_in_cublas_loader` | `91b1317` | the template renders the base tail, not in `src/cublas/sys/mod.rs` | `160a0c4` | R-FC-CODE: fails at `91b1317`, passes at `160a0c4` (review.md) |
| 2026-10-12 | FC6-b count-tests: other binaries, `--binary` | `43bff6d` | 2 of 38 fail: unexpected sections `test:fathomdb_lib_major`, `unittests:bindings_generator` | `8889071` (38 pass) | not re-run by a reviewer |
| 2026-10-12 | FC6-b count-tests: `- should panic` lines | `2e5de53` | 1 of 39 fails: `{} != {'…test_alloc_pinned_panics_on_size_overflow': 'ok'}` | `5e28fa7` (39 pass) | not re-run by a reviewer |
| 2026-10-12 | FC6-b pool doc (D-2, D-9) | — (doc) | — | `e6e5e0f` | — |
| 2026-10-12 | FC7 packaging (item 8) | — (packaging; test plan § 10: no RED) | — | `d04ea18` | — |
| 2026-10-12 | FC7 `use-baseline-lock.py`, 9 fixture tests | `ace64e1` | 7 of 9 fail against a stub (`NotImplementedError`) | `9aef7c0` (9 pass) | R-FC-CODE: fails at `ace64e1`, passes at `9aef7c0` (review.md) |
| 2026-10-12 | FC7 `test_crlf_line_endings_are_kept` | `0c66c4d` | 1 of 10 fails: `expected exactly one root entry (name = "cudarc", no source), found 0` | `27fa9c5` (10 pass) | R-FC-CODE: fails at `0c66c4d`, passes at `27fa9c5` (review.md, "its CRLF case") |
| 2026-10-12 | R-FC-CODE L3 `report_address_symbols_are_bound_wherever_the_library_is` | `480db3f` | 9 passed, 1 failed at 12060, 11040 and 13000: `cufile: cuFileGetVersion not bound for [cuda-11040 … cuda-12020]` | `3647a60` (10 of 10) | R-FC-VERIFY: `480db3f` 9 passed, 1 failed; `3647a60` 10 passed |
| 2026-10-12 | FC8-G6 spec: test plan rev 4, requirements rev 6, design rev 3 (F-G6-1, F-G6-2, F-G5a-1) | — (documents) | — | `fe8e1b4` | — |
| 2026-10-12 | FC8-G6 Tegra release oracle and cap bound: `a_private_pointer_freed_synchronously_releases_its_memory`, `destroy_with_outstanding_allocations_is_deferred`, `requests_within_the_cap_succeed` | `fe8e1b4` (code of `b29c41c`; Orin) | `precondition: reserved 100663296, free 53138673664 -> 53138673664, chunk 33554432`; `the outstanding allocation was released with the pool: free 53139124224 -> 53139124224`; `limit 33554432 leaves more than a chunk (33554432) of the cap 67108864 unusable` | `ba537a2` (Orin: 3 of 3 alone, k = 3; RTX 3090: discrete path unchanged, pass) | RED at `fe8e1b4` not re-run by a reviewer. R-FC-CODE-2 ran its own Orin mutations of the release oracle (free and `free_async` removed); both fail (review.md) |
| 2026-10-12 | FC8-G6 leaked-pointer guard | `fe8e1b4` (Orin, fork modules in sequence) | 50 passed, 6 failed: the 3 above plus 3 out-of-memory cascade failures | `ba537a2` (56 passed; with mutations M1/M3, only the 2 mutated tests fail) | not re-run by a reviewer; R-FC-CODE-2 judged the guard sound from the code (inferred, review.md) |
| 2026-10-12 | FC8-G6 FATHOMDB.md Tegra notes | — (packaging doc) | — | `8528fc2` | — |
| 2026-10-12 | FC8-G6 catalogue: `ignored` restored on 3 rows | `fe8e1b4` (catalogue edit wrote the new `red` text over `ignored = false`) | G1 at `8528fc2`: `count-tests: FAIL`, `expected ignored, got ok` for 2 tests | `9702b51` (`count-tests: PASS`) | not re-run by a reviewer |
| 2026-10-12 | R-FC-CODE-2 M1 Tegra upper bound in `requests_within_the_cap_succeed` | mutation `raw.maxSize = max_size * 2` on `5fce665`'s test (Orin) | `limit 67108864 is within one chunk (33554432) of the cap 67108864: the measured Tegra limit (design D-4) no longer holds` (the `8528fc2` test passes the same mutation) | `5fce665` (Orin k = 3: `limit=33554432 chunk=33554432`, pass; RTX 3090 k = 3 pass) | R-FC-VERIFY (Orin): the mutation fails, the unmutated test passes |
| 2026-10-12 | R-FC-CODE-2 M2 discrete post-free bound in `destroy_with_outstanding_allocations_is_deferred` | leak mutation (`free_async` removed) on `84ede22`'s test (RTX 3090), k = 3 | `free 25017122816 -> 24748687360: half the allocation (134217728) or more is still taken` (the `8528fc2` test passes the same mutation) | `84ede22` (RTX 3090 k = 3: free `25017122816 -> 24748687360 -> 25017122816`, pass) | R-FC-VERIFY (RTX 3090): the mutation fails, the unmutated test passes |
| 2026-10-12 | R-FC-CODE-2 L1/L2 uncapped-pool probe; L3 FATHOMDB.md | — (measurement and docs) | — | `393f165` (probe), `b777f8d` (FATHOMDB.md) | — |

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

### `--locked` and the baseline lock (resolved below: FC7 follow-up)

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

- The `--locked` break above: resolved by renaming at copy time (main
  thread decision; FC7 follow-up below).
- `examples/07-build-workflow/Cargo.toml` no longer resolved under the new
  package name; fixed in the FC7 follow-up (`b8f9777`).
- Source comments still cite `FATHOMDB-PATCH.md` (`core.rs`, `mem_pool.rs`)
  and "upstream 0.19.7" (`core.rs`, `SYNC_FALLBACK`). `FATHOMDB.md` says such
  citations refer to it; the comments themselves are code-tree changes.

### FC7 follow-up: baseline lock renamed at copy time

Main-thread decision: `fathomdb/qualification/Cargo.lock.baseline` stays
byte-identical (SHA-256 `3d399b4e…747923`, unchanged); runs at
`ec1cac71` need its `cudarc` root.

- **Helper** `fathomdb/scripts/use-baseline-lock.py [--baseline LOCK]
  [DEST]` (Python 3.11+, `-I`, standard library). It copies the baseline to
  DEST (default `Cargo.lock`) and sets the one root entry (`name =
  "cudarc"`, no `source` line) to the `[package] name` of the `Cargo.toml`
  next to DEST. Zero or several candidates exit nonzero and write nothing.
  CRLF line endings are kept. It reports on stderr.
- **Oracle.** The fixture `fathomdb/scripts/fixtures/Cargo.lock.cudarc-fathomdb`
  (SHA-256 `d20e4d90…fe404ca`) is what cargo 1.95.0 wrote offline
  when it updated the baseline for the renamed root (`cp` baseline, then
  `cargo metadata --offline`, on a copy of the tree at `d04ea18`). It differs
  from the baseline in line 24 only. `cargo generate-lockfile --offline` is
  not the oracle: on the same copy it re-resolved from the local cache and
  changed `wasip2` (`1.0.3+wasi-0.2.9` → `1.0.4+wasi-0.2.12`, lines 204 and
  206) as well as line 24.
- **RED `ace64e1`** (stub raising `NotImplementedError`): `Ran 9 tests`,
  `FAILED (failures=7)`. The two that pass against the stub only check a
  nonzero exit and that the baseline is not written. **GREEN `9aef7c0`:**
  9 of 9.
- **RED `0c66c4d`** (CRLF, for the Windows T-UNIT checkout):
  `AssertionError: 1 != 0 : use-baseline-lock: expected exactly one root
  entry (name = "cudarc", no source), found 0`; 1 of 10 fails. **GREEN
  `27fa9c5`:** 10 of 10.
- **Consumers.** `7475fa7` (`W`) changes the three steps of
  `fathomdb-ci.yaml` that copied the baseline (hosted-test run 1,
  pure-test, T-UNIT); the Linux steps log the SHA-256 of both files.
  `3174517` (`D`) changes `g8-common.sh` and `t-dev-lib-major.sh`. Both run
  the helper from this checkout's `scripts/` directory, so a `--src` or
  `G8_SRC` tree without `fathomdb/` (`ec1cac71`) still works. No other
  file copied the baseline (grep).
- **Example 07** (`b8f9777`, `P`): `cudarc = { package = "cudarc-fathomdb",
  path = "../.." }`. Before the change, `cargo metadata --offline
  --manifest-path examples/07-build-workflow/Cargo.toml` failed with `no
  matching package named cudarc found`. After it, `cargo metadata` (online:
  `--offline` cannot resolve `bindgen 0.66.1`'s dependencies from this
  host's cache) resolves the dependency `cudarc` to `cudarc-fathomdb@0.19.10`.
  That run rewrote the example's tracked `Cargo.lock` (upstream's, still at
  cudarc 0.9.14), so the file was restored with `git restore`. Not built:
  its `build.rs` compiles CUDA.

Checks at `b8f9777`:

| Check | Result |
| --- | --- |
| helper at HEAD, then `diff` against the baseline | line 24 only; copy SHA-256 equals the fixture's |
| `cargo tree -e normal --locked`, driver-only 12060 | `cudarc-fathomdb v0.19.10` → `libloading v0.9.0` → `cfg-if v1.0.5` |
| `cargo check --locked --lib -F cuda-12060` | ok |
| hosted-test run 1 simulated (helper, `--locked`, driver-only 12060, no GPU) | 54 passed; 4 ignored; 28 filtered out; `count-tests: PASS` |
| helper in a `git archive ec1cac71` tree (no `fathomdb/`) | root `cudarc`; `cmp` byte-identical to the baseline; `cargo tree --locked --offline` ok |
| `test_use_baseline_lock.py` / `test_count_tests.py` | 10 OK / 39 OK |
| `ruff check` on the two new Python files | clean |
| `cargo fmt --all -- --check` | clean |
| fork CI clippy, 13040, `--locked` | clean |
| `actionlint .github/workflows/fathomdb-ci.yaml` | clean (bare `actionlint` also reports upstream's four workflows for old actions; unchanged upstream files) |
| `shellcheck -x fathomdb/scripts/*.sh` | clean |
| T-DEV case f (`t-dev-lib-major.sh f`, RTX 3090 index 0) | `use-baseline-lock: … (root cudarc-fathomdb)`; `CASE f nvtx_loads_with_only_libnvtoolsext_so_1: test ok (exit 0)`; guard before 2026-10-12T09:57:35-05:00 and after 09:57:43-05:00, driver `580.178.04`, `/usr/local/cuda` → `/usr/local/cuda-12.6` |

The helper's self-tests are not run by fork CI (nor are `count-tests`'s).

## R-FC-CODE fix round (classes L, U, D, P)

R-FC-CODE gave APPROVE-WITH-FIXES on `56bb4b7`; the review and its
dispositions are recorded in `review.md` by the main thread.

| Commit | Class | Finding | Change |
| --- | --- | --- | --- |
| `ccdc7b9` | L | M1, L1 | `core.rs`: the per-device decision's doc no longer claims a cross-family free hazard (it keeps one allocator per device, the measured configuration; NVIDIA allows cross-family frees, see the `upgrade_device_ptr` `# Safety` text); the Orin teardown measurements are cited where they are (FathomDB repository, `dev/plans/runs/0.8.27-slice-110-tegra/pool-teardown-evidence/`); item 1–4 markers cite `FATHOMDB.md`; "upstream 0.19.7" → 0.19.10 (2 places). Comments only |
| `1663adc` | U | L1 | `mem_pool.rs:1`, `core.rs` pool markers and the `new_with_mem_pool` doc cite `FATHOMDB.md`, item 5. Comments only. `grep -rn -E 'FATHOMDB-PATCH\|0\.19\.7' src` is now empty |
| `480db3f` | U | L3 RED | T-UNIT test `report_address_symbols_are_bound_wherever_the_library_is` (pure: reads the checked-in bindings with `include_str!`); `query_symbol`/`address_symbol` move, unchanged, out of the dynamic-loading `loader` module so the test can call them |
| `4ec4cf3` | D | L3 | catalogue row (cell `t-unit`, landed); test plan revision 3: § 3.4 lists the test, T-UNIT has 10 names per feature, 190 per OS. The CI expectations come from the catalogue, so the workflow is unchanged |
| `3647a60` | U | L3 GREEN | cuFile's address symbol is `cuFileDriverOpen` (bound for every cuda feature, no `cfg`) |
| `f37c698` | D | (found while verifying L3) | `g8-common.sh` wrote `cudarc = { path = … }` into the G8 report project, which stopped resolving at `d04ea18`; it now reads the package name from `$G8_SRC/Cargo.toml` |
| `88acf7f` | D | L5 | plan revision 5: § 1.4 lists fork CI's drift from upstream's workflows |
| `59e8f08` | D | L4 | test plan T-DEV (d) setup states the in-process decoy preload; ruling and no-preload outcome recorded |
| `b29c41c` | P | L2 | `FATHOMDB.md`: what item 9 changes in CUDA 11.x builds (shared loader: memoization, loading `is_culib_present`, panic text, version query at load, new API; candidates and first-that-opens acceptance unchanged); Windows path code compile-checked only; the `FATHOMDB-PATCH.md` note removed |

### L3 RED and GREEN

`cargo test --locked --lib -F cuda-<v>,$CHECK_DEFAULT_FEATURES --
fathomdb_lib_major::`, no GPU:

- RED `480db3f`, at 12060, 11040 and 13000 alike:
  `test fathomdb_lib_major::report_address_symbols_are_bound_wherever_the_library_is ... FAILED`,
  `"cufile: cuFileGetVersion not bound for [\"cuda-11040\", \"cuda-11050\", \"cuda-11060\", \"cuda-11070\", \"cuda-11080\", \"cuda-12000\", \"cuda-12010\", \"cuda-12020\"]"`,
  `test result: FAILED. 9 passed; 1 failed`. Log SHA-256: 12060
  `6e92830d…986d3c8`, 11040 `a5567e45…6abc6`, 13000 `82a1934f…2f22`.
- GREEN `3647a60`: 10 passed at 11040, 12060 and 13000;
  `count-tests check` against `expect --set def --features
  $CHECK_DEFAULT_FEATURES --filter fathomdb_lib_major:: --binary
  unittests:cudarc` PASS and `unlanded` PASS at each.
- **Not covered:** the path of a cuFile without `cuFileGetVersion` (before
  1.8). No such cuFile is on this host (12.6 ships 1.11), so the test checks
  the symbol choice, not `dladdr` on a pre-1.8 file. With the system cuFile
  1.11, the G8 report binary at `3647a60` printed `REPORT cufile: opened as
  libcufile.so.0, path
  /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcufile.so.1.11.1, version
  1.11, checked against CUDA 12`, `G8-REPORT PASS` (path now resolved
  through `cuFileDriverOpen`).

### Checks at `b29c41c`

| Check | Result |
| --- | --- |
| G1, items 1–4: `cargo test --locked --lib -F cuda-12060 -- fathomdb_alloc_fallback --test-threads=1`, RTX 3090 index 0 | 14 passed; 0 failed; `count-tests` PASS, `unlanded` PASS; log SHA-256 `f48f8039…8d7903` |
| G1, all fork modules, same build, `-- fathomdb_` | 55 passed; 0 failed; 5 ignored; 75 filtered out; `count-tests` PASS |
| T-DEV case (d) (`t-dev-lib-major.sh d`) | `CASE d cuda12_build_rejects_an_already_loaded_cuda13_cufile: test ok (exit 0)` |
| G8-e (`g8-e.sh`, after `f37c698`) | `G8 g8-e: PASS`; report project tree `cudarc-fathomdb v0.19.10` |
| `cargo fmt --all -- --check` | clean |
| fork CI clippy (`--all-targets`, `$CLIPPY_FEATURES`, `--locked`), 13040 and 12060 | clean |
| `cargo check --locked --lib`, default features, 11040, 12060, 13000, 13040 | clean |
| `test_count_tests.py` / `test_use_baseline_lock.py` | OK / OK |
| `actionlint` (fork workflow), `shellcheck -x fathomdb/scripts/*.sh` | clean |
| `cargo package --list` (scratch `git worktree add --detach b29c41c`, removed after) | 135 files; no `fathomdb/` path; no `fathomdb-ci.yaml`; `FATHOMDB.md` present; list identical to `d04ea18`'s (SHA-256 `83db814a…f6ddac7`) |

Guards (driver `580.178.04`, `/usr/local/cuda` → `/usr/local/cuda-12.6`,
unchanged each time): T-DEV (d) ended 2026-10-12T10:18:19-05:00; G8-e ended
10:19:00; cuFile report 10:19:05 before and after; items 1–4 run
10:20:19–10:20:23; all-modules run 10:20:30–10:21:20.

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
| `7475fa7` | W | fork CI: baseline lock through `use-baseline-lock.py` |
| `b8f9777` | P | example 07 depends on `cudarc-fathomdb` by path |
| `ccdc7b9` | L | item 2 rationale; FATHOMDB.md pointers in fallback code (comments) |
| `1663adc` | U | FATHOMDB.md pointers in pool code (comments) |
| `480db3f` | U | L3 RED: report address-symbol test |
| `3647a60` | U | L3 GREEN: cuFile address symbol `cuFileDriverOpen` |
| `b29c41c` | P | FATHOMDB.md: item 9 in CUDA 11.x builds; Windows path code compile-checked only |
| `ba537a2` | U | FC8 G6 fix round: Tegra release oracle and capacity bound; leaked-pointer guard |
| `8528fc2` | P | FATHOMDB.md Tegra notes |
| `5fce665` | U | R-FC-CODE-2 M1: Tegra upper bound in `requests_within_the_cap_succeed` |
| `84ede22` | U | R-FC-CODE-2 M2: half-allocation post-free bound (discrete) in `destroy_with_outstanding_allocations_is_deferred` |
| `b777f8d` | P | R-FC-CODE-2 L2/L3: FATHOMDB.md Tegra notes match the cap table and probe; code commit `C'` |

## FC8 qualification at `C = b29c41c` (all cells except G5b)

Run 2026-10-12 by an implementer subagent on `windchill3` and the Orin. The
tree tested is a `git archive` of the branch tip `5249683`, whose code diff
to `b29c41c` is empty (`git diff b29c41c 5249683 -- . ':(exclude)fathomdb'`).
Every cell ran k = 3 at `C` and, where requirements CF-REQ-12 demands one,
k = 3 at `ec1cac71` (a `git archive`), with `Cargo.lock.baseline` through
`use-baseline-lock.py` and `--locked`. Receipts are in `qualification/`;
logs over 1 MiB, and the FC8 scratch tools (`run-cell.sh`, the
`fdbmaps.so` maps preload, `analyze.py`, `mkreceipt.py`), are kept on
`windchill3` in `~/fathomdb-forks/fc8-logs/`.

| Cell | Result | Receipt | UNMEASURED / open |
| --- | --- | --- | --- |
| CI run 38269933260 (`head_sha` `5249683`) | PASS: 139 of 139 jobs `success`; T-UNIT 10 per feature on 38 jobs | `2026-10-12-CI-5249683.md` | — |
| G1 (12060, default) | PASS; `test_pinned_copy_is_faster` (host-dependent) failed once at each commit | `2026-10-12-G1-b29c41c.md` | none |
| G1 T-DEV (a)–(d), (f) | PASS, k = 3 | `2026-10-12-G1-TDEV-b29c41c.md` | (e): G5b |
| G2 (T-MULTI, 0,1) | PASS | `2026-10-12-G2-b29c41c.md` | none |
| G3 12060, 13000, 13010 | PASS | `2026-10-12-G3-<v>-b29c41c.md` | none |
| G3 13020, 13030, 13040 | PASS; driver-version gate holds | `2026-10-12-G3-<v>-b29c41c.md` | `max_pool_size_reads_back_at_least_the_request` (G5b) |
| G4 (13000, default, CUDA 13.0 wheels) | PASS; baseline loaded CUDA 12.6 libraries | `2026-10-12-G4-b29c41c.md` | none |
| G5a 13010–13040 (default, 13.1–13.4 wheels) | PASS for the G5a set; finding F-G5a-1 open | `2026-10-12-G5a-<v>-b29c41c.md` | per build: 15 upstream `launch`/`unified_memory` tests, `test_device_side_assert`, `drops_after_a_sticky_error_neither_panic_nor_abort` (13.1+ PTX JIT on driver 580: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`); from 13020 also `max_pool_size_reads_back_at_least_the_request` |
| G6 (Orin, 12060) | **FAIL, stopped**: findings F-G6-1, F-G6-2 | `2026-10-12-G6-b29c41c.md` | 46 upstream tests and the 10 `fathomdb_lib_major` tests never reached (crash, F-G6-2) |
| G8 a–f | PASS | `2026-10-12-G8-b29c41c.md` | none |
| CF-AC-07a | PASS: 135 files, list SHA-256 `83db814a…f6ddac7` at `5249683` and at `b29c41c` (unchanged); no `fathomdb/`, no `fathomdb-ci.yaml`; dry run ok | `2026-10-12-CF-AC-07a-b29c41c.md`, `package-list-b29c41c.txt` | `.cargo_vcs_info.json` names `5249683`; FC10 packages `C` itself |
| G7 | UNMEASURED (owner ruling Q5) | — | — |

Guards: `windchill3` first 2026-10-12T10:33:04-05:00, last
11:11:21-05:00, 852 guard lines in the runner logs (plus the T-DEV and G8
script guards), all `driver=580.178.04 cuda=/usr/local/cuda-12.6`. Orin
first 10:31:48-05:00, last 11:11:21-05:00, all `R36 (release), REVISION:
5.2` with `/usr/local/cuda-12.6`. No guard changed. The Orin work directory
`~/fdb-fork-qual/` was removed afterwards.

Scratch venvs added for G5a (`~/fathomdb-forks/venvs`, `pip install
--no-deps`, pins = NVIDIA's `cuda-toolkit` metapackage components on PyPI):

| Venv | Pins |
| --- | --- |
| `cu13-131` | `nvidia-cuda-runtime==13.1.80`, `nvidia-cuda-nvrtc==13.1.115`, `nvidia-cublas==13.2.2.2`, `nvidia-curand==10.4.1.81`, `nvidia-nvjitlink==13.1.115` (`cuda-toolkit==13.1.2.0`) |
| `cu13-132` | `nvidia-cuda-runtime==13.2.86`, `nvidia-cuda-nvrtc==13.2.86`, `nvidia-cublas==13.4.1.3`, `nvidia-curand==10.4.2.66`, `nvidia-nvjitlink==13.2.86` (`cuda-toolkit==13.2.2`) |
| `cu13-133` | `nvidia-cuda-runtime==13.3.29`, `nvidia-cuda-nvrtc==13.3.33`, `nvidia-cublas==13.6.0.2`, `nvidia-curand==10.4.3.29`, `nvidia-nvjitlink==13.3.33` (`cuda-toolkit==13.3.1`) |
| `cu13-134` | `nvidia-cuda-runtime==13.4.92`, `nvidia-cuda-nvrtc==13.4.92`, `nvidia-cublas==13.8.0.4`, `nvidia-curand==10.4.4.72`, `nvidia-nvjitlink==13.4.92` (`cuda-toolkit==13.4.2`) |

### FC8 findings (open; no disposition made here)

- **F-G6-1.** On the Orin, six fork tests fail at `C` in every run:
  `a_private_pointer_freed_synchronously_releases_its_memory`,
  `destroy_with_outstanding_allocations_is_deferred` and
  `requests_within_the_cap_succeed` (also alone), and
  `max_size_on_a_driver_older_than_12020_is_not_supported`,
  `release_threshold_bounds_reserved_after_sync` and
  `trim_to_keeps_the_floor_and_releases_above_it` (only in sequence,
  `CUDA_ERROR_OUT_OF_MEMORY`). Observations: `cuMemGetInfo` free memory does
  not move on Tegra (upstream `test_post_alloc_memory` fails the same way at
  both commits), and M-2 gives a 32 MiB limit for a 64 MiB cap on the Orin.
  Verbatim messages in the G6 receipt. The old `cuMemGetInfo` oracle was
  nondeterministic on the Orin, depending on page-pool warmth: the two
  release tests passed alone in `2026-10-12-G6-pertest-b29c41c.tsv` but
  failed alone in the RED runs (R-FC-CODE-2 finding L4).
- **F-G6-2.** Upstream `test_from_raw_context_bind_to_thread` segfaults on
  the Orin at `ec1cac71` and at `C` (alone too, and without the maps
  preload), which ends every `--lib` main run; the tests after it are not
  measured on the Orin.
- **F-G5a-1.** CF-AC-04c says the fork's NVRTC-dependent tests compile to
  CUBIN; `drops_after_a_sticky_error_neither_panic_nor_abort` compiles PTX
  (`nvrtc::compile_ptx`), so it fails on G5a with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` and is recorded UNMEASURED. The
  catalogue lists it for G1 and G5b only.

### Left for G5b (GCP T4)

- The full `--lib` set for 13010–13040 with default features (CF-AC-04c),
  including every test recorded UNMEASURED in the G5a receipts.
- `max_pool_size_reads_back_at_least_the_request` (13020+; M-4
  `cuDriverGetVersion` ≥ each build).
- `drops_after_a_sticky_error_neither_panic_nor_abort`,
  `release_threshold_bounds_reserved_after_sync`,
  `trim_to_keeps_the_floor_and_releases_above_it`,
  `requests_within_the_cap_succeed`,
  `a_request_over_the_cap_is_out_of_memory`,
  `capture_on_a_private_context_uses_graph_memory` and
  `measure_reservation_steps` (catalogue cell G5b; M-1).
- T-DEV (e) `driver_loads_with_only_libcuda_so_1`.
- After G5b: R-FC-VERIFY over CF-AC-01…06, 07a and 09…12 at `C`.
- **Code commit now `C' = b777f8d`** (R-FC-CODE-2 fix round, below): G5b
  runs at `b777f8d`, and every item above is owed at `b777f8d`, not at
  `b29c41c` or `8528fc2`.

## FC8 G6 fix round (findings F-G6-1, F-G6-2, F-G5a-1; classes D, U, P)

Dispositions decided by the main thread; implemented 2026-10-12 by an
implementer subagent on `windchill3` and the Orin. New code commit
**`8528fc2`**; the pin chain restarts there (plan § 1.5), so `b29c41c` is
no longer the commit to pin.

| Commit | Class | Files | Purpose |
| --- | --- | --- | --- |
| `fe8e1b4` | D | `test-plan.md`, `requirements.md`, `design.md`, `fork-catalogue.toml`, `qualification/2026-10-12-G6-pertest-{ec1cac71,b29c41c}.tsv` | test plan rev 4 (Tegra oracle, cap-test Tegra branch, guard, G6 one process per test, M-2 per platform, § 9 G6 platform list pending R-FC-CODE-2); requirements rev 6 (CF-AC-04c CUBIN statement scoped to the G5a set, F-G5a-1); design rev 3 (Tegra free memory, usable cap fraction, uncapped-pool budget, upstream `cuCtxCreate_v4(NULL)` defect) |
| `ba537a2` | U | `src/driver/safe/core.rs` (fork test module only) | `integrated`/`vm_rss` helpers and the `Leaked` guard; Tegra branches of the three tests; guard in the two other tests that leak and can panic before freeing |
| `8528fc2` | P | `FATHOMDB.md` | "Tegra notes" section |
| `9702b51` | D | `fork-catalogue.toml` | restores `ignored = false` on the three rows `fe8e1b4` broke |
| this commit | D | `qualification/2026-10-12-G6-8528fc2.{md,log}`, `status.md`, `design.md`, `test-plan.md` | G6 receipt; probe precision in design D-2/D-9 and test plan § 3.1 |

Discrete path unchanged: in `git diff -w fe8e1b4 ba537a2 --
src/driver/safe/core.rs`, all five discrete assertions
(`free_live as u64 + 2 * chunk <= free_before as u64`,
`free_after as u64 + chunk >= free_before as u64`, `limit + chunk > cap`,
`free_outstanding + SIZE / 2 <= free_before`,
`free_after + chunk >= free_before`) and their messages are unchanged
context lines; they moved only into `else` / `is_none()` branches. The ten
removed lines are the old leak/collect statements and doc lines.
`pool_model_property` keeps its bare `leak()`: it frees the pointer in the
next statement, with no panic point between.

### Verification at `8528fc2`

- `cargo fmt --check`: clean.
- Fork CI clippy (`--no-default-features --features
  "cuda-<v>,$CLIPPY_FEATURES" --all-targets -- -D warnings`, the
  workflow's `CLIPPY_FEATURES`): pass at 13040 and 12060 on x86_64, and at
  12060 and 13040 natively on the Orin (aarch64).
- `cargo check --locked --lib`, driver-only, at 11040, 12060, 13000 and
  13040, and default features at 12060 and 13040: pass, no warnings.
- `python3 -I -m unittest discover -s fathomdb/scripts -p 'test_*.py'`
  (count-tests and use-baseline-lock self-tests): 49 tests OK.
- **G1 (RTX 3090 `41:00`, 12060 default, k = 1):** `cargo test --locked
  --lib -F cuda-12060 -- fathomdb_ --test-threads=1 --nocapture`: 55
  passed, 0 failed, 5 ignored, 75 filtered out, `count-tests: PASS`
  (catalogue at `9702b51`); no `integrated` line printed (discrete path).
  Alone with `--ignored --exact`: `drops_after_a_sticky_error_…`,
  `measure_max_size_semantics`, `measure_reservation_steps`: 1 passed
  each, `count-tests: PASS`. Guards 11:45:53, 11:46:46, 11:46:48 and
  11:55:13: `580.178.04`, `/usr/local/cuda-12.6`.
- **G6 (Orin):** receipt `qualification/2026-10-12-G6-8528fc2.md`. RED,
  mutations and GREEN as above; fork modules in sequence k = 3: 56 passed
  each, `count-tests: PASS`; the six F-G6-1 tests alone k = 3: 18 of 18
  pass; every `--lib` test as its own process, k = 3 at `ec1cac71` and at
  `8528fc2`: only the § 9 G6 platform list fails (identically at both
  commits in all runs) plus the two T-MULTI tests (G2 only, recorded).

### Open after this round

- R-FC-CODE-2: review of `fe8e1b4..9702b51` and the receipt commit,
  including the § 9 G6 platform list amendment (pending approval).
- The restarted chain at `8528fc2` (plan § 8: item 5 tests and `P`): fork
  CI on `8528fc2` pushed as the tip (CI receipt), CF-AC-07a at `8528fc2`
  (FATHOMDB.md is in the package), and G3 for the touched test kinds
  (`requests_within_the_cap_succeed` is a G3 test) were not run here. The
  push is the main thread's.
- G5b (unchanged by this round).

## R-FC-CODE-2 fix round (findings M1, M2, L1–L4; classes D, U, P)

Dispositions decided by the main thread; implemented 2026-10-12 by an
implementer subagent on `windchill3` and the Orin. New code commit
**`C' = b777f8d`**; the pin chain restarts there (plan § 1.5), so
`8528fc2` is no longer the commit to pin. `git diff 8528fc2 b777f8d -- .
':(exclude)fathomdb'` touches only the fork test module in
`src/driver/safe/core.rs` (4 hunks, all in `mod fathomdb_private_pool`)
and `FATHOMDB.md`.

| Commit | Class | Files | Purpose |
| --- | --- | --- | --- |
| `393f165` | D | `design.md`, `test-plan.md`, `fork-catalogue.toml`, `qualification/uncapped-pools-probe.{rs,Cargo.toml}`, `qualification/2026-10-12-uncapped-pools-probe.{md,log}` | L1/L2 probe and evidence; design rev 4 (D-2 measured: 20.47 GiB VA per uncapped pool, four per process, `Synchronous` with four live); test plan rev 5 rows and catalogue `red` text for M1/M2 |
| `5fce665` | U | `src/driver/safe/core.rs` (fork test module) | M1: Tegra `limit + chunk <= cap` in `requests_within_the_cap_succeed`, labelled measured Orin/JetPack 6 behaviour (design D-4) |
| `84ede22` | U | `src/driver/safe/core.rs` (fork test module) | M2: discrete post-free bound `free_after + SIZE / 2 > free_before` in `destroy_with_outstanding_allocations_is_deferred` (was one chunk = 256 MiB, which could not fail). **Strengthens a discrete assertion; accepted by the main thread as a pre-existing defect fix.** The test also prints the discrete `cuMemGetInfo` values |
| `b777f8d` | P | `FATHOMDB.md` | L3: cap note consistent with the D-4 table (about a third for large caps; 32–96 MiB caps give 32 MiB); L2: `Synchronous` selection measured, per process. **Code commit `C'`** |
| this commit | D | `status.md`, receipts `qualification/2026-10-12-*-b777f8d.{md,log}`, `qualification/2026-10-12-CI-863ee48.md`, `qualification/package-list-b777f8d.txt`, `qualification/2026-10-12-R-FC-CODE-2-red.log` | L4 sentence (F-G6-1), RED evidence, re-qualification at `C'` |

### RED and GREEN (log `qualification/2026-10-12-R-FC-CODE-2-red.log`, SHA-256 `d61ddda2600059d3e7ec32ccc839481936bebba69452c0c70e4b0c0e9ab9999e`)

Each test alone (`--exact`, `--nocapture`), 12060 bindings, own target
directory and binary per tree (binary SHA-256 values in the log).

- **M1 (Orin):** with the mutation `raw.maxSize = max_size * 2`, the new
  test fails: `M-2 integrated: cap=67108864 size=1048576 limit=67108864
  chunk=33554432`, `limit 67108864 is within one chunk (33554432) of the
  cap 67108864: the measured Tegra limit (design D-4) no longer holds`.
  The `8528fc2` test with the same mutation passes (`limit=67108864`).
  Without the mutation, k = 3: `limit=33554432 chunk=33554432`, `1 passed`
  each. RTX 3090 discrete path k = 3: `1 passed` each.
- **M2 (RTX 3090 `41:00`):** with `free_async` removed (pointer leaked),
  k = 3, the new test fails each time: `discrete: cuMemGetInfo free
  25017122816 -> outstanding 24748687360 -> after 24748687360; chunk
  268435456`, `free 25017122816 -> 24748687360: half the allocation
  (134217728) or more is still taken`. The `8528fc2` test with the same
  mutation passes. Without the mutation, k = 3: `free 25017122816 ->
  outstanding 24748687360 -> after 25017122816`, `1 passed` each. Free
  memory deltas: −268435456 (the full 256 MiB) while outstanding and 0
  after the free in all three GREEN runs, so the 128 MiB margin is far
  from `cuMemGetInfo` noise here.

### Re-qualification at `C' = b777f8d` (plan § 7 receipts)

Trees are `git archive`s of `b777f8d` with `Cargo.lock.baseline` through
`use-baseline-lock.py` and `--locked`. Each cell built in its own
`CARGO_TARGET_DIR`; the test-binary SHA-256 values all differ (in each
receipt). Baseline runs at `ec1cac71` were **not** re-run: they are
unchanged by this round, and each receipt cites the earlier baseline
receipt (`*-b29c41c.md`; G6: `2026-10-12-G6-8528fc2.md`).

| Cell | Result | Receipt | UNMEASURED / open |
| --- | --- | --- | --- |
| G1 (12060, default) | PASS: main 125 passed, 0 failed, 10 ignored ×3; ignored alone (`test_post_alloc_memory`, `test_device_side_assert`, `drops_after_a_sticky_error_…`, `measure_max_size_semantics`, `measure_reservation_steps`) 15 of 15; count-tests PASS in every run | `2026-10-12-G1-b777f8d.md` | none |
| G3 12060 | PASS: 79 passed, 1 failed (host-dependent `test_pinned_copy_is_faster`), 7 ignored ×3 | `2026-10-12-G3-12060-b777f8d.md` | none |
| G3 13000, 13010 | PASS: 77 passed, 1 failed (same host-dependent test), 7 ignored ×3 | `2026-10-12-G3-<v>-b777f8d.md` | none |
| G3 13020, 13030, 13040 | PASS: 77 passed, 1 failed (same), 9 ignored ×3; driver-version gate tests pass | `2026-10-12-G3-<v>-b777f8d.md` | `max_pool_size_reads_back_at_least_the_request` (G5b) |
| G4 (13000, default, CUDA 13.0 wheels) | PASS: 122 passed, 1 failed (same host-dependent test), 10 ignored ×3; ignored alone 15 of 15 | `2026-10-12-G4-b777f8d.md` | none |
| G6 (Orin, 12060) | PASS: fork modules in sequence 56 passed ×3 (count-tests PASS); six F-G6-1 tests alone 18 of 18; one process per test ×3: only the § 9 G6 platform list (10), the two T-MULTI tests and the host-dependent `test_pinned_copy_is_faster` fail, identically in all runs | `2026-10-12-G6-b777f8d.md` | none |
| CF-AC-07a | PASS: 135 files, list byte-identical to `package-list-b29c41c.txt` (SHA-256 `83db814a…f6ddac7`); dry run ok; `.cargo_vcs_info.json` names `b777f8d` | `2026-10-12-CF-AC-07a-b777f8d.md`, `package-list-b777f8d.txt` | — |
| CI run 38281195051 (`head_sha` `863ee48`, code tree `8528fc2`) | success, 139 of 139 jobs; **superseded** by `C'` | `2026-10-12-CI-863ee48.md` | — |
| CI at `C'` | **pending** when written: made after the main thread pushes the new tip. **Done:** run 38286272393 at `35d3fec` (code tree `C'`) | `2026-10-12-CI-35d3fec.md` | — |

In every G3/G4 cell, the host-dependent `test_pinned_copy_is_faster`
failed in the three main runs and passed in the three `timing-*` runs
(values in the receipts), the same pattern as at `b29c41c`. Every count-tests
FAIL line in those cells is that host-dependent outcome; no
non-host-dependent problem. On the Orin it failed in all three per-test
runs (it passed in all three at `8528fc2`, failed once at `b29c41c`).

Not run in this round (unchanged areas): G2, G5a, G8, G1 T-DEV. G5b is
pending owner approval (list above names `C'`). **Done since:** G5b ran
at `C'` with the owner's approval; see "FC8 G5b at `C' = b777f8d`" below.

### Checks at `b777f8d`

- `cargo fmt --check`: clean.
- Fork CI clippy (`--no-default-features --features
  "cuda-<v>,$CLIPPY_FEATURES" --all-targets -- -D warnings`, the
  workflow's `CLIPPY_FEATURES`, `--locked`): pass at 13040 and 12060 on
  x86_64, and at 12060 and 13040 natively on the Orin.
- `cargo check --locked --lib`, driver-only, at 11040, 12060, 13000 and
  13040, and default features at 12060 and 13040: pass, no warnings.
- `python3 -I -m unittest discover -s fathomdb/scripts -p 'test_*.py'`
  (count-tests and use-baseline-lock self-tests): 49 tests OK.

### Guards

- `windchill3` (`nvidia-smi … driver_version`, `readlink -f
  /usr/local/cuda`): first 2026-10-12T12:11:46-05:00, last
  12:39:11-05:00; every value `580.178.04`, `/usr/local/cuda-12.6`.
- Orin (`head -1 /etc/nv_tegra_release`, `readlink -f /usr/local/cuda`):
  first 12:09:23-05:00, last 12:39:06-05:00; every value `R36 (release),
  REVISION: 5.2` and `/usr/local/cuda-12.6`. Work directory
  `~/fdb-rfc2/` removed at the end.
- No guard changed.

### Open after this round

- R-FC-CODE-2 re-review of `393f165..` this commit (the reviewer records
  it in `review.md`). **Done:** `review.md`, R-FC-CODE-2.
- CI receipt at `C'` after the push (main thread). **Done:**
  `qualification/2026-10-12-CI-35d3fec.md`.
- G5b at `C'` (pending owner approval), then R-FC-VERIFY at `C'`.
  **Done:** G5b, see "FC8 G5b at `C' = b777f8d`" below; R-FC-VERIFY
  returned APPROVE-WITH-NOTES with no code defect.

## FC8 G5b at `C' = b777f8d` (GCP T4; class D)

The G5b cell passed, with T-DEV (e) needing the owner-approved setup
described below. The main thread recorded this section on 2026-10-12.

### Sessions

| Session | VM | Kind | Zone | Created | Deleted | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `fdb-lab-g5b-1` | Spot T4 | us-central1-a | 18:05:00Z | 18:09:37Z, preempted (operation `compute.instances.preempted`) | the driver step was cut off; no tests ran |
| 2 | (no VM) | on-demand T4 | 35 T4 zones worldwide | — | — | "not enough resources" in every zone; nothing created |
| 3 | `fdb-lab-g5b-3` | Spot T4 | us-central1-a | 19:58:00Z | 20:12:57Z (destroyed) | ran to completion |

- **Owner rulings:**
  - session 2 as on-demand;
  - then the order Spot T4, on-demand T4, L4.
- **Lab script changes in FathomDB `release/0.8.28`:**
  - `bc9037c48` adds `--on-demand`;
  - `0a6620ffe` adds an `l4` profile.
- **Session 3 timeline:**
  - kernel `7.0.0-1014-gcp` and driver 615.71.09 open
    (`nvidia-headless-no-dkms-615-server-open`), installed with one reboot;
    `driver.json` method `verified-after-reboot`;
  - runner 20:03:50–20:09:38Z;
  - T-DEV (e) re-run 20:11:43–20:11:47Z.
- **M-4:** `cuDriverGetVersion` = 13040, which is at least every build.

### Results (k = 3 per build, C' and the `ec1cac71` baseline)

- **Receipts:** `fathomdb/qualification/2026-10-12-G5b-{13010,13020,13030,13040,TDEV}-b777f8d.md`
  and their `.log` files. They are generated by the G5b analyser from the
  pulled results; the manifest check passed (`out.tar.gz` SHA-256
  `8d54dcf9…2dfb`).
- **Main `--lib` runs:**
  - C': 121 passed, 2 failed, 10 ignored at 13010 and 11 ignored at
    13020+;
  - baseline: 66 passed / 2 failed at 13010 and 63 passed / 2 failed at
    13030/13040;
  - in every run the 2 failures are exactly the two closed-list
    host-dependent timing tests (`test_pinned_copy_is_faster`,
    `test_default_pinned_host_reads_are_faster_than_write_combined`), as
    in the earlier T4 trial.
- **Ignored tests, one by one** (`--ignored --exact`): every one passes,
  including:
  - `drops_after_a_sticky_error_neither_panic_nor_abort`, the PTX that G5a
    could not JIT on driver 580;
  - `max_pool_size_reads_back_at_least_the_request` at 13020–13040, the
    real read-back that driver 580 rejects;
  - `measure_reservation_steps` (M-1);
  - `test_post_alloc_memory` and `test_device_side_assert`.
- **Not run:** the two T-MULTI tests and the three upstream two-GPU tests,
  because the VM has one GPU (test plan § 7, UNMEASURED: one GPU).
- **G5a cross-check:** the 15 `launch`/`unified_memory` tests that are
  UNMEASURED on G5a pass in the G5b main runs.
- **T-DEV (e)** `driver_loads_with_only_libcuda_so_1`: 3 of 3 pass.
  - Ubuntu's `libnvidia-compute-615-server` ships the unversioned
    `libcuda.so`, so the first three attempts recorded
    PRECONDITION-NOT-MET.
  - With the owner's approval, the main thread moved it aside on the lab VM
    with `dpkg-divert` and re-ran only (e). The receipt records this.

### Open

R-FC-VERIFY over CF-AC-01…06, 07a and 09…12 at `C' = b777f8d`. Its CI
receipt is `2026-10-12-CI-35d3fec.md`: 139/139, and T-UNIT 10 per job on
all 38 jobs.
