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
