# Selected library lifetime admission

Proposed 2026-11-15 against `53e77220d1187b709da5126b8e8c551068d531c2`.
This focused amendment owns cuBLAS/cuRAND admission and stream ownership;
`design.md`, `requirements.md` and `test-plan.md` retain the fork-wide contract.
Independent design review precedes production edits. Actual resets remain held.

## Source fit and public contract

Selected Candle K9 `182ceab0547348993b6dbec4b1aed610dcc8b23f` F32 GEMM
in `candle-core/src/cuda_backend/mod.rs:2534-2563` clones the destination stream,
obtains input/output pointer guards, then immediately invokes unsafe raw
`gemm_strided_batched_ex(*cublas.handle(), ...)`. Driver C53
`core.rs:1863-1869,1937-1947` pointer preparation records fallible stream waits
but still returns pointers; a native terminal wait publishes shared loss and
stores its delayed error. There is no K9 check between that preparation and
raw cuBLAS. An outer110 lease coordinates quiescent reset but cannot by itself
stop dispatch after an error discovered inside that admitted operation.

A necessary Candle boundary repair is therefore proposed: expose
`CudaBlas::admit() -> Result<CudaBlasLease<'_>, CublasError>`. The borrowed guard
holds BLAS/context admission and binding across one caller operation. Its
`handle() -> Result<sys::cublasHandle_t, CublasError>` rechecks the tracked
context's cached loss/delayed error without native dispatch. K9 declares this
guard before preparing device pointers, then calls `guard.handle()?` immediately
as the first argument of its existing raw GEMM call. The guard drops after all
pointer guards on return or unwind. Preserve every existing GEMM dimension,
stride, datatype, scalar, compute-type choice and algorithm argument exactly;
no numerical policy/API expansion. The existing raw handle escape remains
unsafe caller responsibility.110 outer admission still spans the full model
construction/forward/unwind. No production Candle edit before independent
review of this amendment.

RNG construction is required even without random model operations: Candle
creates BLAS then RNG. Ordinary safe BLAS/RNG methods need admission before
pointer preparation plus a cached/delayed-error recheck immediately before
library dispatch, retaining admission through pointer-guard destruction.

Keep existing safe method/result type names. Change `CublasError` and
`CurandError` from tuple wrappers to enums with `Native(status)`,
`Driver(DriverError)` and `ContextMismatch` variants. Raw result wrappers return
`Native`; safe admission/binding returns the exact `Driver` status, including
cached terminal OOM. This is a public source change with no compatibility shim.
Exact K9 `182ceab0547348993b6dbec4b1aed610dcc8b23f` uses only these error
types and typed From conversions, which need no Candle edit. Product
`candle_bge.rs` currently reads `CublasError.0` for ARCH_MISMATCH/NOT_SUPPORTED
and has three tuple test constructors. Pin adoption must migrate those exact
Native matches, preserving their statuses, and classify Driver variants through
the existing driver/shared-loss policy. Consumers
query shared cached `loss()` to distinguish terminal from ordinary OOM; native
allocation failure alone never publishes loss.

## Operations and ownership

- Tracked construction acquires the context's lifetime lease before binding or
  library handle creation; retain it through stream/seed setup and partial-owner
  Drop. Immediately wrap each successful create as at baseline. Propagate tracked
  binding failure before library FFI. Preserve untracked constructor behavior.
  Depend on the independently qualified core normal-binding repair: an actual
  terminal709 bind failure after healthy admission publishes the first loss
  before returning Driver; a later call refuses without native work. Ordinary
  bind OOM remains retryable. Libraries must not duplicate core publication.
- Safe use and setters acquire admission before binding, pointer acquisition or
  library FFI and retain it through pointer-guard drops. Cached Lost performs no
  native work. Cover existing safe BLAS and RNG seed/offset/fill methods without
  expanding supported model dtypes or APIs.
- When either old or target stream context is tracked, set_stream requires both
  to share the same lifetime (Arc identity), refusing mismatch before native
  work. The old context/lifetime remains authoritative for the native handle.
  Replace the stored stream only after successful native reassociation, with
  the lease held. Both-untracked retains its existing unsafe caller contract.
- Drop disarms its own handle first. Tracked Drop acquires admission, uses the
  existing conservative CudaContext::bind_for_cleanup, and destroys only on
  success, with admission across both. The core owner exposes that existing
  helper crate-wide; no state mutex spans FFI. Lost or failed cleanup binding
  skips native destroy. Rust descendants drop normally. Untracked Drop retains
  existing destruction behavior; no destructor unwinds.
- Library statuses are not fabricated into driver loss. Cleanup binding
  uncertainty uses existing core first-loss publication. Later refusal cannot
  overwrite it. Ordinary OOM remains retryable.

## Tests and order

Use owned test FFI seams; no GPU/reset/model execution is implied. Each changed
behavior receives actual semantic RED/output before minimal GREEN. Existing
partial-owner cleanup is correct reuse, never manufactured RED. Existing
assertions migrate mechanically from tuple construction to Native only.

1. Constructors: inactive/changed identity and cached terminal OOM refuse before
   bind/create; healthy tracked and ordinary native OOM controls. RNG includes
   stream/seed partial initialization and first-owner unwind. Paired normal-bind
   controls start with healthy admission, force actual709 binding failure, and
   require exact Driver return, first-loss publication, zero library FFI, then
   zero-native subsequent refusal; ordinary bind OOM publishes no loss and
   succeeds on retry after removing the fault. The shared core repair owns the
   corresponding actual RED/GREEN before library integration.
2. Use: tracked BLAS pointer-mode/GEMM and RNG seed/offset/fills refuse before
   pointer acquisition/FFI after loss. Healthy counters/native error controls
   prevent always-refuse. Share one admission helper per library.
3. Stream setter: failed native setter retains original Arc; mixed/distinct
   tracked owners refuse before FFI; same generation succeeds. Cover both BLAS
   and RNG, including premature release of the old Rust stream owner.
4. Drop: no-forward-close discovers loss and skips destroy; cached Lost skips
   native cleanup; healthy destroys once; cleanup-binding failure publishes
   first loss and skips destroy. Existing untracked partial-failure/nonpanic
   controls remain passing.

Root authorizes bounded native/build/test leases. Focused new selectors use
`fathomdb_library_admission` modules. Final independent implementation review
covers this coherent library family and raw outputs before integration.

## Checked raw boundary test intent (proposed amendment)

Use actual stream-wait seams to force native709 during pointer preparation
after healthy admission. Require preserved Driver status/first loss, zero raw
GEMM calls, and normal Rust guard unwind under the lease. Pair with ordinary
wait-OOM refusal followed by successful retry, healthy raw-GEMM call counters,
and a preserved native library-error control. Also exercise cached Lost before
checked admission and loss first published between admission and handle check.
This boundary receives its own semantic RED before minimal GREEN; compiler
absence of a new API is preparation, never a substitute for behavior RED.
Candle qualification must retain its exact F32 compute/algo argument behavior.

## Native cleanup failure cause (proposed amendment)

A tracked native library destroy failure leaves reclamation uncertain. Keeping
its generation healthy permits repeated new handles/residue and violates the
selected frozen-live-set/no-growth policy. Publish terminal loss before Drop
returns, retaining the actual cuBLAS/cuRAND domain and status. A native OPERATION
allocation failure remains retryable; a preexisting ordinary delayed OOM is not
an observed cleanup failure. Untracked destruction retains legacy behavior.

The core owner supplies one first-writer publication path; libraries do not own
another loss store or edit driver files. Proposed public representation:

- `PrimaryContextLoss` retains recorded/observed identities and an authoritative
  `cause: PrimaryContextLossCause`, replacing its driver-only `error` field.
- Cause variants are `DriverStatus(CUresult)`,
  `NativeLibraryCleanup { library: NativeLibrary, status: u32 }`, and
  `SyntheticUnknown` for existing poison/probe-unwind cases without a native
  status. NativeLibrary has cuBLAS/cuRAND domain tags independent of library
  feature selection; the numeric value is the unchanged native enum status.
- Pure cached `loss.refusal()` derives `PrimaryContextRefusal::DriverStatus`
  for an original driver cause or `SyntheticUnknown` for library/synthetic
  causes. Its driver-status projection is UNKNOWN only in the latter case;
  documentation and diagnostics explicitly label that projection SYNTHETIC,
  never a measured driver UNKNOWN. An actual driver UNKNOWN remains a distinct
  DriverStatus cause. No native query participates in reading/formatting loss.
- Existing driver-returning operations may return the documented synthetic
  UNKNOWN projection; callers inspect the authoritative cached cause. Library
  safe errors add `Lost(PrimaryContextLoss)` for a synthetic terminal refusal,
  preserving the real cause instead of reporting it as a measured Driver
  error. Original driver failures keep the existing Driver variant/status.
- A crate-visible core publication method accepts the tagged native cleanup
  cause after actual destroy returns Err. It records once under the existing
  state lock, with no FFI under that lock. Any later driver/library observation
  preserves the first cross-domain cause. No constructor or operation native
  OOM is reclassified merely by sharing a library status value.

Paired semantic tests precede implementation: failed tracked BLAS/RNG destroy
(including allocation-status failure) freezes the generation after one native
attempt; repeated construction/use/drop performs no native dispatch or growth;
assert exact domain/status and explicit synthetic refusal. Include partial
stream/seed unwind, successful destroy, native OPERATION OOM retry, preexisting
ordinary delayed OOM with successful cleanup, driver-first and library-first
loss ordering across BLAS/RNG/driver observations, and unchanged untracked
nonpanic cleanup. Existing driver-only loss assertions migrate mechanically to
the authoritative DriverStatus cause, preserving every expected status and
identity. The core representation and this amendment require the same independent
Astra review before public/core/API implementation. The earlier library Drop
GREEN remains partial until this failure bound is closed and qualified.
