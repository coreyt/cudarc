# cudarc-fathomdb reviews

Every review is an independent Opus subagent, read-only (plan § 9).

## R-FC-PLAN (served by FathomDB R0-rev8), 2026-10-11

- Model: `opus` (Claude Opus), high effort, general-purpose, read-only.
- Reviewed: `fathomdb/plan.md` revision 1 at `74a1665`, together with
  FathomDB Slice 31 plan revision 8 (`4a33e887e`) and the Candle fork plan
  (`37bc0b4e`).
- Verdict: APPROVE-WITH-FIXES. All findings below are fixed in plan
  revision 2; none was deferred.

| Finding | Defect (short) | Disposition in revision 2 |
| --- | --- | --- |
| H2 | CF-AC-07 package receipt claimed at FC8 but produced only at FC10 | split into CF-AC-07a (FC8, at `C`) and CF-AC-07b (FC10, identical file list) |
| M1 | fork CI ran no `cargo test`; no lockfile recorded; FathomDB lock drift untested | hosted `fathomdb_` test job (x86_64 + aarch64, once with FathomDB's lock); receipts record `Cargo.lock` SHA-256 and `cargo tree`; § 8 row for FathomDB lock changes |
| M2 | no aarch64 compile in fork CI | aarch64 Linux check + clippy for all 19 features; counted in CF-AC-04a |
| M3 | RUN/UNMEASURED looked only at functions | D-F1 covers enum/attribute/flag values and struct layouts; G3 proves D-4's driver-version gate |
| M5 | published commit could differ from the qualified one | publication from `C` itself; metadata change restarts the chain; `.cargo_vcs_info.json` must equal `C` |
| M6 | PR #595 (cuBLAS caller-owned workspace) unmentioned | new D-F3; risk entry |
| M9 | proposal hygiene unenforced; trailers on upstream commits; dry run named the renamed crate | `proposal/*` exempt from trailers; CF-AC-08 diff and trailer assertions; name-agnostic dry run |
| M11 | no CI receipt; CI might never run on `C` | `C` pushed as tip (or `ci/qualify-*` ref) before receipts; CI receipt with `head_sha == C` |
| M12 | class `C` collided with the qualified commit `C` | class renamed `W`; `C` reserved for the pin |
| L1 | clippy `-D warnings` wrongly listed as a difference from upstream | corrected list of differences |
| L7 | G4 needlessly depended on the runfile | G4 may use cu13 PyPI wheels in a scratch virtual environment |
| L8 | CF-AC-04b named 12060 but G3 omitted it; 13010/13030 omitted | G3 builds 12060, 13000, 13010, 13020, 13030, 13040 |
| L9 | JetPack 7 known issue 5699079 dropped | restored in § 6.1 and G7 |
| L10 | crate-name and token bootstrap unchecked | name unclaimed on 2026-10-11 (404); token step in § 1.5 and FC10 |
| L13 | re-qualification table gaps | rows added for cfg/bindings (G1, G6), `P`, `W`, FathomDB lock |
| L14 | links not permalinks | receipts linked by permalink at the receipt commit |

## R-FC-REQ, 2026-10-12

- Model: `opus` (Claude Opus), high effort, general-purpose, read-only.
- Reviewed: `fathomdb/requirements.md` revision 1 at `f774dc3`, against plan
  revision 3, FathomDB Slice 31 plan §§ 5, 7 and 15, the vendored
  `FATHOMDB-PATCH.md` and patches, the T4 trial receipts, `src/` at
  `ec1cac71` and the host's sonames.
- Verdict: APPROVE-WITH-FIXES. Every finding is fixed in requirements
  revision 2 and plan revision 4; none was deferred.

| Finding | Defect (short) | Disposition |
| --- | --- | --- |
| H1 | item-9 unit tests would run in the one-feature, `fathomdb_`-filtered hosted job, so 0 tests ran | dedicated T-UNIT job: 19 features, every library feature, module by name, count asserted per feature (CF-AC-11; plan § 1.4); proposal jobs run their own tests by name (CF-AC-08, 08b) |
| H2 | G8 only recorded results; the cuRAND clause was always true | per-sub-cell assertions: loaded file is the venv's, version in range; 12.1.x outcome stated (pass or typed error, never panic); cu13 cuRAND/cuSPARSE cell added |
| H3 | out-of-scope libraries and the driver share `get_lib_name_candidates`; nothing would catch a broken `.so.1` or cuDNN `.so.9` | T-UNIT asserts candidate lists equal `ec1cac71` for `cuda`, `nvcuda`, `cudnn`, `nccl`, `cutensor`, `nvToolsExt` and all 11.x features; driver-with-only-`libcuda.so.1` case |
| H4 | the T-DEV decoy was unspecified, so RED could be vacuous | decoys (a) and (b) named; RED observed through `/proc/self/maps`; cu13 cuBLAS source named |
| M1 | some libraries have no usable version query | rule 2: such libraries are "versioned name only"; D-F4 per-library table; `cusolverMg`, `cufile` < 12030 and `cupti` named |
| M2 | rejecting same-soname libraries was unjustified and would break working setups | rule 3: same-soname libraries accepted and reported; rejection needs cited evidence; NVTX moved out of scope |
| M3 | 11.x features had no oracle | 11.x keeps upstream candidates (FathomDB Q10 floor); T-UNIT asserts equality |
| M4 | "typed failure" allowed a panic; `is_culib_present` disagreed with `culib()` | rule 4: fallible check with a typed error; `is_culib_present` agrees; missing symbols fallible; T-UNIT on message, T-DEV (c) |
| M5 | Windows names untested | rule 7: candidate generation is a pure function of target parameters; T-UNIT covers Windows names on Linux and runs on Windows |
| M6 | CF-AC-02 named G4 for 13000 pool tests | G3 (13000, driver-only feature set) |
| M7 | CF-AC-02 could pass vacuously; clauses uncovered | preconditions for threshold and trim; success up to the cap; live and second-thread handle checks; N ≥ 2; ≥ 256 property cases; one test per clause |
| M8 | `select_async_alloc` is not upstream; the equality check was trivial on RTX 3090 | upstream inline rule cited (`core.rs:78,150,201,238`); `cfg(test)` counter shows no default-pool query on x86_64; zero-length behaviour checked against base |
| M9 | CF-REQ-12 loopholes | baseline defined (same version, upstream features, lock, host); k ≥ 3 runs; closed host-dependent list; no edits, ignores or filters of upstream tests; ignored tests run where the cell can |
| M10 | printing a filter did not account for tests | fixed per-module counts in `test-plan.md`, checked in each receipt |
| M11 | Q-F1 premise overstated | rationale restated (PTX JIT and post-13.0 driver behaviour); owner then ruled both G5a (local) and G5b (T4) required |
| M12 | R-FC-VERIFY omitted CF-AC-11/12; no item-9 re-qualification row | CF-AC-09 states the verify set; plan § 8 item-9 row; § 9 FC8 updated |
| M13 | deprecated-API and aarch64 `no-std` clauses had no AC | aarch64 × both feature sets in CF-AC-04a; D-F1 deprecation column and R-FC-CODE grep |
| L1 | `max_size` gap wording | 13030/13040 gap named; "vendored delta" stated |
| L2 | "6 of 6" was wrong | each test failed in all 8 runs (per toolkit: full run + 3 reruns) |
| L3 | wrong section reference | "plan § 6.2 G6 test set" |
| L4 | versioned-first and closing rejected handles not required | rules 1 and 2 |
| L5 | `get_lib_name_candidates` is public | rule 6 |
| L6 | traceability missed REQ31-13 and REQ31-14 | rows added |
| L7 | Orin host guard undefined | § 2 defines it (L4T, JetPack, `/usr/local/cuda`) |
| L8 | report scope undefined for linked builds and the driver | rule 5 |
| L9 | "G1 (12.6)" ambiguous for the proposal | "G1's host and build, against `proposal/mem-pool`" |

## R-FC-DES, 2026-10-12

- Model: `opus` (Claude Opus), general-purpose, read-only.
- Reviewed: `fathomdb/design.md` revision 1 at `8259c14`, with requirements
  revision 3, plan § 5, the vendored delta, FathomDB Slice 30 design
  §§ 3.1 and 4.5, the Slice 31 plan, `src/` at `ec1cac71`, and the FC2
  research notes and scratch evidence.
- Verdict: APPROVE-WITH-FIXES. No duplicated upstream functionality
  (CF-AC-06). Every finding is fixed in design revision 2 and requirements
  revision 4; none was deferred.

| Finding | Defect (short) | Disposition |
| --- | --- | --- |
| H1 | the cross-family free hazard is false: `cuMemFree` accepts pool pointers (CUDA 12.6 `cuda.h`, `cuMemFree`), PG13.4 allows both directions; the debug check would reject valid input | debug check and `should_panic` RED dropped; CF-REQ-03 is a cited doc correction with characterization tests; item 2's rationale corrected in `FATHOMDB.md` (item kept: one allocator per device, measured) |
| M1 | 12.x ranges admitted CUDA 11.x libraries sharing a soname | lower bounds (cuSOLVER ≥ 11.4.2, cuRAND ≥ 10301, CUPTI ≥ 19 from the 12.6 header); cuSolverMg `.so.11` a recorded gap; M−1 cases in T-UNIT |
| M2 | glibc reuses an already-loaded object for a bare soname (NODELETE cuFile; a library another component loaded) | `RTLD_NOLOAD` probe; `AlreadyLoadedWrongVersion`; T-DEV (d) and a preloaded G8 sub-cell |
| M3 | transitive dependencies unchecked (`libcusolver.so.11` needs `libcusparse.so.12`) | Linux link-map diff after load; `WrongMajorDependency`; G8 sub-cell; Windows a recorded gap |
| M4 | cuFile without `cuFileGetVersion` unspecified | accepted for 12.x with `NoVersionQuery`, rejected for 13.x; FC3 measures 12.3/12.5 wheels |
| M5 | trim/threshold upper bounds ignored fragmentation | live set limited to one allocation; bound + one chunk; chunk step measured at four sizes |
| M6 | per-test dependency table and 13.4 status list promised in design, absent | CF-AC-04a/04b now name `test-plan.md` |
| M7 | gate evidence misattributed (archived 13.2 page renders as 13.1.1); attribute 15 gated at 13020; read failure unspecified | gate rests on headers; attribute 15 needs 13040 and is never requested; read error gives `None`, only a successful low read fails `create` |
| L1 | report version unstructured; path not canonical; cuSolverMg has no query symbol | structured `LibVersion`, `cuda_major`, `opened_as`, canonical path; any exported symbol for `dladdr` |
| L2 | memoization and `is_culib_present` changes unspecified | `Clone` error, memoized failure, `LoadFailed`; behaviour changes listed for `FATHOMDB.md` and the proposal |
| L3 | Windows CUPTI contradicts T-UNIT | exempted explicitly |
| L4 | T-CAPTURE could pass vacuously; Tegra graph-memory risk unmentioned | positive graph-memory `USED_MEM_CURRENT` assertion; failure outcome stated; Tegra risk added |
| L5 | FathomDB-lock run may silently re-resolve | `cargo metadata` assertion against the lock; lock SHA-256 recorded |
| L6 | unrequested API growth | setters and `ordinal()` dropped; FC5 API change table with class and requirement |
| L7 | multi-feature cfg mismatch | noted: compile error, never silent; item 9 major from `CUDA_MAJOR_VERSION` |
| L8 | aarch64 RED seeded a shared table | moot (H1) |
| L9 | open-ended 13.x ranges accept a future same-soname CUDA 14 library | recorded trade-off, FC9b note |
| L10 | hand edits vs template could drift; files unnamed | files named; template-vs-file unit test (or fork-CI script) |

## R-FC-TEST, 2026-10-12

- Model: `opus` (Claude Opus), general-purpose, read-only.
- Reviewed: `fathomdb/test-plan.md` and `fathomdb/test-inventory/*` at
  `936c25e`, against requirements revision 4, design revision 2, plan
  §§ 1 and 6–9, the earlier review dispositions, the vendored patches,
  `src/` at `ec1cac71`, the `windchill3` loader cache and PyPI.
- Verdict: APPROVE-WITH-FIXES. Inventories verified correct. Every finding
  is fixed in test plan revision 2 and requirements revision 5; none was
  deferred.

| Finding | Defect (short) | Disposition |
| --- | --- | --- |
| H1 | bound tests could pass a no-op (weak precondition, hand-set chunk) | each test measures its chunk in-process; precondition reserved > max + chunk; no env var |
| H2 | T-DEV (d) RED false: the base never tries `.so.0` and maps 12.6 cuFile via the cache | decoy `libcufile.so` → cu13 `.so.0`; requirements revision 5 (d) |
| H3 | T-DEV (a) RED false: cu13 cuBLAS `RUNPATH $ORIGIN` needs `libcublasLt.so.13` beside the symlink | decoy dir then the cu13 venv lib dir; precondition probe; requirements revision 5 (a) |
| H4 | pure tests named "CI" ran in no job | new fork pure-test job (19 features, names checked); plan § 1.4 |
| M1 | property test predicted nothing | the model predicts success vs the M-2 limit, used ≥ live sum, per-buffer patterns, default pool unchanged; G3 13000 added |
| M2 | stream-order test could not observe premature reuse | `cuLaunchHostFunc` latch; reuse by a third stream with a different pattern |
| M3 | exhaustion assertions too weak; not multi-threaded | at least one OOM; successful sum ≤ limit; total ≤ cap + chunk; 2 threads × 2 streams |
| M4 | FC5-c REDs false (vendored code already correct) | marked characterization, no RED |
| M5 | gate and read-back REDs unobservable at FC5 | staged ungated read-back then gate (RED on G3 13020); seam-driven read-back test on G1; real read-back observed at G5b |
| M6 | counters without positive control | every counter test makes one counted call first |
| M7 | T-CAPTURE passed on failure | success required on G1/G5b; only G6 may record a Tegra limit |
| M8 | lost-context and `cuMemPoolCreate` OOM deferred without reason | `drops_after_a_sticky_error_neither_panic_nor_abort` (by itself, as upstream's device-side assert); OOM seam test; fork-after-init stays with FathomDB D-15 |
| M9 | totals only; fork inventories generated from the build | name-set comparison per module; fork expectations from `fork-catalogue.toml` |
| M10 | G4 filtered out `launch`/`unified_memory` | G4 runs the whole `--lib` |
| M11 | G3 omitted ignored tests; T-MULTI not ignored | ignored one by one in every cell; T-MULTI `#[ignore = "must be executed with multiple gpus"]` |
| M12 | baselines load different libraries; G5b per-build paths missing | baseline maps recorded and explained; per-build `LD_LIBRARY_PATH` on G5b |
| M13 | G5b driver unnamed; fallback contradicted CF-AC-04c; no budget | 615.71.09 open (prebuilt for `7.0.0-1014-gcp`); 595 fallback fails 04c for 13030/13040; budget 100 min; cross-built binaries; second session needs owner approval |
| M14 | several ACs and dispositions without tests | template test; CF-AC-07a/07b commands; CF-AC-08/08b name lists and T-PERF; `is_culib_present` agreement, memoized failure and linked-build report tests |
| M15 | T-DEV REDs relied on new API | REDs use only base API plus `/proc/self/maps` |
| L1 | aarch64 test split wrong | 3 + 2 |
| L2 | T-DEV (e) had no precondition | `ldconfig -p` and file checks |
| L3 | `CUDA_COMPUTE_CAP` unused by cudarc | dropped |
| L4 | no G7 row; gate-cell inconsistency | G7 row (Q5); map and catalogue agree |
| L5 | weak destroy check; `CudaContext` Send/Sync; oracle could not cover 11.x | `cuMemGetInfo` recovery; `CudaContext` added; parameterized copy of the base function |
| L6 | design-only tests claimed CF-AC tracing | marked "design" |
| L7 | device order by enumeration | `CUDA_DEVICE_ORDER=PCI_BUS_ID`; device name logged |
| L8 | vendored test breaks at 13020+; RED ordering | FC5-a edits it with the helper; location fix before the 13030 `maxSize` RED |
| L9 | G8 sub-cells incomplete; features unspecified | completed; `-F …,cusparse,cusolver,cufile` |
