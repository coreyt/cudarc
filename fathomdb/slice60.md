# Slice60 cudarc maintenance and qualification

Status: SOURCE_AND_CPU_ACCEPTED; input343b04fa73f85ee78754d6ee2baa07b959cc9ccc.
FathomDB release0.8.30 seq409 commissions this owning-fork work. Historical
plan/requirements/design/test-plan remain authoritative outside this scoped
amendment. Direct Astra-medium reviews this amendment before changes; direct
Sol6.2-high reviews code and direct Terra verifies actual affected native results.

## Needs, requirements and acceptance

Keep upstreamv0.19.10/ec1cac71 as base after comparing published upstream0.19.10.
Retain items1–5,7–9 plus all qualified .29 shared-lifetime/library guards.
Actual owning source must preserve allocator fallback, private isolation and
accounting, one per-device irreversible token, first typed cause, ordinary OOM
retry, healthy release and no stale nested cleanup after loss. Balanced final
primary release is distinct from forbidden post-loss binding/free/destroy.

Remove unused fathomdb-private-pool marker independently of pool primitives:
FathomDB tegra-pool and Candle use package identity and no marker edge. First
update FATHOMDB.md item6 removal policy to permit independently qualified marker
removal. Then remove only empty Cargo feature and corresponding stale text.
Feature metadata cleanup changes no owning allocator implementation; existing
pool/lifetime controls are preservation, not invented semantic RED. Package
and source graphs must show identical implementation features except marker
absence, and no active upstream cudarc substitution. No new crate version or
publication is selected:0.20.0 remains unpublished candidate.

## Design and exact qualification

FATHOMDB.md must inventory .29 primary lifetime and library cleanup deltas,
including their removal conditions; current item table omits them despite intro
mention. Retain actual source, no new wrapper/lifetime implementation. Fork
commit classes apply: D only fathomdb/, P packaging/docs, L marker metadata;
U only a newly demonstrated owning behavioral defect. Keep no upstream writes.

Reuse qualified312 selected CPU controls for unchanged343b04fa source and run
changed-feature resolution, targeted owning tests and library Werror/fmt. Actual
owning nested-slice test is lost_slice_disarms_all_native_free_and_nested_event_cleanup
with paired healthy control; tracked pools/BLAS/RNG have their existing deep
suites. Record exact selected names/results, no nonzero-count inference.

Native controls use CUDA12.6 only on exclusively leased RTX3090 UUID
GPU-5f9cfc90-2be1-06a7-ce39-5a6d294b209b and actual AGXOrin64 at10.83.10.13.
Run actual context/pool isolation/healthy-release and finite OOM/retry controls;
Orin private pool proof cannot be replaced by3090. Existing locks are opened
without create/unlink. Preserve R580.178.04/primaryCUDA12.6 and OrinJetPack6.
No K620, driver/toolkit mutation, paid venue or CPU-as-GPU proof.

The source candidate is qualified before Candle adopts it; receipts after code
are D and identify exact code commit. FATHOMDB.md is packaged code-tree metadata.
A source-only marker change does not invalidate unchanged native implementation
credit, but final combination receives actual native preservation controls.
Shared terminal loss, owning destructors, typed native/synthetic provenance and
healthy/OOM controls remain required at final consumed pins.

Actual reset source/protocol, completed-construction quiescence,120s child/
30min venue,8GiB GPU/16GiB ownedRSS bounds and exact commands are prepared and
reviewed in product before parent obtains any remaining seq377 fault authority.
No reset is authorized merely by this fork document. Original expected-crash
history remains until a real positive successor qualifies. New defects require
actual semantic RED commit before GREEN; no missing-API/compile/GPU failure RED.

Return exact clean code/receipt heads, native subject hashes, actual exits and
limits to product. No registry publication, upstream issue/PR/default change or
force push. Parent owns final worktree integration/removal.

## Consolidated FIX-1 lifetime acceptance

Retain seq376 frozen finite pre-loss native live set until process restart,
known byte/count ceilings and no post-loss growth. Opaque native host bytes
have no hard cap;16GiBRSS is a session-stop bound only. Post-loss diagnostics
are cached/unavailable, never native pool queries. Owning dispatch spies prove
no new allocation/free/trim/nested/library work; real children prove typed
single/batch/rerank same-cause refusals, no-forward first detection, lastclone
and normal post-drop markers/exit0/fresh process. No native-reclamation claim
from Rust close alone. Retain historical signal11 case unexecuted.

Separate healthy final-owner control is
tracked_pool_cleanup_tests::cleanup_healthy_last_owner_holds_admission_through_one_native_destroy;
lost control cleanup_known_loss_disarms_last_pool_owner_without_native_destroy
requires0 destroy. Their actual production dispatcher, Weak/strong ownership
and balanced primary release join native
destroy_with_outstanding_allocations_is_deferred256MiB resource/release/RSS
oracle and Candle positive Weak-last-owner/post-drop normal-exit child. Never
query a destroyed pool. OrdinaryOOM product control preserves same allocator/
token/recorded generation through status2 -> healthy USEDrelease ->4MiB retry.
Native reset cases remain HELDseq377 until reviewed concrete product protocol
and actor/venue/resource authority; no additional blanket approval is invented.

## Actual marker candidate

Code pin7948f520e122cc0ae1d302a5487bfd383486d327 is staged on the owner
branch fathomdb/0.8.30-slice60-cudarc. Direct Sol6.2-high source PASS confirms
all remaining manifest fields/features and src/build.rs unchanged. PolicyP
precedes markerL; the obsolete comment is removed. Direct Terra independently
compiled this pin with libloading0.9.0, Rust1.95, CUDA12060 and hidden GPUs;
seven owning filters executed149 tests,0failed/0ignored. This includes paired
lost/healthy nested slice, final pool, typed first cause and ordinaryOOM retry.
This is preservation, not semanticRED or real-reset proof. Native pools,
installed consumers and final graph/native/reset qualification remain OPEN.
Raw logs/list/features/actual exits and lock/binary identities:
/mnt/data/tmp2/fathomdb-0830-slice60/terra-cudarc-evidence/.
Binary e0be4188d359904c39b1911ea450800864824d1dcc84a1bd5173739c7bffb42f;
lock3704e868ea653b3309009b41c9bb1d1ed3583c2f26004c7bebfed955414060ab.
Baseline146unique selector matches/149 executions usedlibloading0.9.1 and is
retained separately; its dependency inputs are not relabeled as the candidate.
