# Slice60 cudarc maintenance and qualification

Status: QUALIFIED_AND_PUBLISHED; input343b04fa73f85ee78754d6ee2baa07b959cc9ccc.
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
absence, and no active upstream cudarc substitution. Registry0.20.0 was already published from343b04fa; matching marker metadata
requires a fresh candidate as selected in the delivery correction below.

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


## Registry identity correction

Registry0.20.0 is already published from343b04fa; its source cannot change.
The selected marker removal requires fresh candidate0.21.0 for matching
registry metadata. Read-only exact-version (including yanked) and ownership
preflight precedes this bump. Normalized package staging and an external
no-root-patch Candle CUDA consumer must resolve one cudarc with identical
source; public-registry proof remains OPEN until parent publication disposition.
No registry write is authorized by this amendment.


## Append-only commit-class correction

Commit `8e2f2b39777055e936791db52fb4a11d5ded8617` omitted its required
`Fork-Class: P` trailer. This is historical noncompliance. Actual path audit
(`git diff-tree --no-commit-id --name-only -r`) shows only `Cargo.toml` and
packaged `FATHOMDB.md`, both permitted P paths under this fork's own plan;
no `fathomdb/` owning docs were mixed into it. Parent agent coordination directed this narrow
append-only correction, preserving original refs and measured bindings.
It does not claim the missing trailer existed. Future commits retain the
existing trailer/path rules and normal hooks; no history is rewritten.


## Historical pre-final exact-pin qualification

Code candidate `8e2f2b39777055e936791db52fb4a11d5ded8617` (0.21.0)
retains byte-identical owning Rust/tests/build.rs versus the old qualified
343b04fa source; marker/version/packaged notice are the only later changes.
Required fresh owning CPU qualification nevertheless ran once against actual
Fathom lock f02a722f51e298525db9670045908cbfb22d26ebd2ad89b7369396ec6cc4072f:
323 x86 and322 native ARM selected tests, all passed, zero ignored. Exact22
normal package identities/checksums and10 root features match both product
graphs; seven non-root feature sets differ and are explicitly recorded, not
claimed equal. Source/test/build identity, original setup/operator failures,
corrected remaining-only execution and actual binary hashes are retained at
`/mnt/data/tmp2/fathomdb-0830-slice60/terra-exact-lock/`. Product's reviewed
lock-set record preserves the previous native qualification unchanged.

A real owning outstanding-allocation/deferred-destruction child passed on the
selected3090. Existing native/Candle/model controls exercised this actual0.21
source separately from the CPU lifetime/known-loss seams. No actual reset is
claimed. A checksum-bound normalized0.21 archive compiled and ran through an
independent no-root-patch staged-registry consumer, on the3090 and in a real
driverless CPU container; exact known F32 output and post-drop exit0 passed.
This proves the staged package route, not public registry availability.
Publication, actual Orin reset/private-pool fault successors, final installed
consumers and full product gate remain separately recorded OPEN until executed.

## Final Slice60 closure

Final CODE3509cd6bd83b44204be1880e833474ab0f75e060 retains unchanged owning
Rust/tests/build behavior. Fresh actual final-product-lock qualification passed
323x86 and322nativeARM selected controls, zero ignored, under
`/mnt/data/tmp2/fathomdb-0830-slice60/terra-final-exact-lock/`. Both venues
passed the final owning/model/installed functional controls and seven actual
process-primary fault children plus seven fresh recovery children each under
owner411/412. Orin used the real3GiB private pool; typed synthetic709 is not
a claimed native709 return. Ordinary Orin OOM and the owner-requested one
full-card3090 Default OOM retry passed actual native2/cleanup/same-generation
retry. Original8GiB probe and clone-monitor limitations retain their failures.
All selected children ended and leases freed. Product strict gate on f020b84
passed241registered/238ran+passed/3explicitlive-controlskips, exit0.

Owner seq413 authorized and normal Cargo published cudarc-fathomdb0.21.0
from the clean3509 checkout. Actual public archive SHA256 is
`6eaa637ddf017eee725e50219745293994400c7b214886bcd5a8b10a4dc18413`,
byte-identical to the staged candidate. Earlier records above retain their
original source/lock and held-at-that-time meanings; they are not relabelled.
Final public consumer delivery and full shared cohort outcome are recorded in
FathomDB Slice60 status and `evidence/public-registry-downloads.json` under
the task evidence directory. No future50 API or0.8.30 shipping artifact is
qualified by these development artifacts.
