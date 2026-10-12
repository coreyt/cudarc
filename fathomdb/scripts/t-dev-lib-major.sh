#!/usr/bin/env bash
# T-DEV of test plan § 3.4 (requirements CF-AC-11), cell G1 (windchill3).
#
# Builds tests/fathomdb_lib_major.rs and runs each case's ignored test as its
# own process, directly (not through current_exe), with the case's library
# setup. Before each case a helper process checks the case's precondition;
# a failed precondition stops the script. The host guard (test plan § 1.1)
# is taken first and last; any change stops the script.
#
# Usage: t-dev-lib-major.sh [--src DIR] [--out DIR] [CASE...]
#   --src DIR  the cudarc tree to build (default: this checkout). Use a
#              `git archive` of another commit to run the same cases there.
#   --out DIR  logs and decoy directories (default: a new mktemp dir)
#   CASE       a b c d e f (default: a b c d f; e needs the G5b host)
#
# Environment: FATHOMDB_CU13_VENV (default ~/fathomdb-forks/venvs/cu13-130),
# the scratch venv with the test plan § 3.4 cu13 pins.
#
# Output: one "CASE <x> <test name>: test ok|FAILED" line per case, plus the
# test's MAPS/RESULT lines. A FAILED test is reported, not an error: at a
# commit before item 9 the cases a-d are expected to fail (RED).

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="$(cd "$here/../.." && pwd)"
out=""
cases=()
while [ $# -gt 0 ]; do
  case "$1" in
    --src) src="$(cd "$2" && pwd)"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    a|b|c|d|e|f) cases+=("$1"); shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ ${#cases[@]} -gt 0 ] || cases=(a b c d f)
[ -n "$out" ] || out="$(mktemp -d)"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

venv="${FATHOMDB_CU13_VENV:-$HOME/fathomdb-forks/venvs/cu13-130}"
cu13="$venv/lib/python3.12/site-packages/nvidia/cu13/lib"
ldso=/lib64/ld-linux-x86-64.so.2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

guard() {
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | tr '\n' ' ')cuda=$(readlink -f /usr/local/cuda)"
}
guard_before="$(guard)"
echo "GUARD before $(date -Iseconds) $guard_before"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader | sed 's/^/DEVICE /'
echo "SRC $src ($(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
echo "OUT $out"

# A helper process (t-dev-probe.py): dlopen each name ("!name": must not
# open), print the mapped files matching the pattern; exit 1 on a surprise.
probe() { python3 -I "$here/t-dev-probe.py" "$@"; }
probe_inhibit() {  # probe_inhibit <LD_LIBRARY_PATH> <pattern> <names...>
  local llp="$1"; shift
  env LD_LIBRARY_PATH="$llp" "$ldso" --inhibit-cache "$(readlink -f "$(command -v python3)")" -I "$here/t-dev-probe.py" "$@"
}

build() {  # build <cuda version> <features> -> prints the test binary path
  local v="$1" feats="$2"
  (
    cd "$src"
    python3 -I "$here/use-baseline-lock.py" Cargo.lock
    CUDARC_CUDA_VERSION="$v" cargo test --locked -F "$feats" --test fathomdb_lib_major \
      --no-run --message-format=json 2>"$out/build-$v.log" |
      python3 -I -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m["target"]["name"] == "fathomdb_lib_major" and m.get("executable"):
        print(m["executable"])'
  )
}

need12=0; need13=0
for c in "${cases[@]}"; do
  case "$c" in b) need13=1 ;; *) need12=1 ;; esac
done
bin12=""; bin13=""
[ $need12 = 1 ] && bin12="$(build 12060 cuda-12060,cufile,nvtx)" && echo "BIN 12060 $bin12"
[ $need13 = 1 ] && bin13="$(build 13000 cuda-13000)" && echo "BIN 13000 $bin13"

run_case() {  # run_case <case> <test name> <binary> [env/loader prefix...]
  local c="$1" name="$2" bin="$3"; shift 3
  local log="$out/case-$c.log" status=0
  "$@" "$bin" --ignored --exact "$name" --nocapture --test-threads=1 >"$log" 2>&1 || status=$?
  grep -oE '(MAPS|RESULT|REPORT) .*|panicked at .*' "$log" | sed "s/^/  /" || true
  grep -A1 'panicked at' "$log" | grep -vE 'panicked at|^--$' | sed "s/^/  message: /" || true
  local summary
  summary="$(grep -E '^test result: ' "$log" || true)"
  case "$summary" in
    "test result: ok. 1 passed;"*) echo "CASE $c $name: test ok (exit $status)" ;;
    "test result: FAILED. 0 passed; 1 failed;"*) echo "CASE $c $name: test FAILED (exit $status)" ;;
    *) echo "CASE $c $name: no test result (exit $status); see $log" ;;
  esac
}

for c in "${cases[@]}"; do
  echo "== case $c"
  case "$c" in
    a)
      d="$out/decoy-a"; mkdir -p "$d"; ln -sfn "$cu13/libcublas.so.13" "$d/libcublas.so"
      LD_LIBRARY_PATH="$d:$cu13" probe libcublas libcublas.so | tee "$out/helper-a.txt"
      grep -q "maps $cu13/libcublas.so.13" "$out/helper-a.txt" ||
        { echo "precondition (a) failed: libcublas.so does not map .so.13"; exit 1; }
      run_case a cuda12_build_skips_an_unversioned_cuda13_cublas "$bin12" \
        env LD_LIBRARY_PATH="$d:$cu13" FATHOMDB_T_DEV_CU13_LIB="$cu13"
      ;;
    b)
      entry="$(/sbin/ldconfig -p | awk '$1 == "libcublas.so" {print $NF; exit}')"
      echo "HELPER ldconfig -p: libcublas.so => $entry -> $(readlink -f "$entry")"
      readlink -f "$entry" | grep -q 'libcublas.so.12' ||
        { echo "precondition (b) failed: libcublas.so is not 12.x in the cache"; exit 1; }
      probe libcublas '!libcublas.so.13' ||
        { echo "precondition (b) failed: a CUDA 13 cuBLAS is reachable"; exit 1; }
      run_case b cuda13_build_rejects_the_cuda12_cublas_from_the_loader_cache "$bin13" env
      ;;
    c)
      d="$out/empty-c"; mkdir -p "$d"
      probe_inhibit "$d" libcublas '!libcublas.so' '!libcublas.so.12' '!libcublas.so.13' ||
        { echo "precondition (c) failed: a cuBLAS is resolvable"; exit 1; }
      run_case c no_acceptable_cublas_is_a_typed_error_not_a_panic "$bin12" \
        env LD_LIBRARY_PATH="$d" "$ldso" --inhibit-cache
      ;;
    d)
      d="$out/decoy-d"; mkdir -p "$d"; ln -sfn "$cu13/libcufile.so.0" "$d/libcufile.so"
      LD_LIBRARY_PATH="$d" probe libcufile libcufile.so | tee "$out/helper-d.txt"
      grep -q "maps $cu13/libcufile.so.0" "$out/helper-d.txt" ||
        { echo "precondition (d) failed: libcufile.so does not map the cu13 cuFile"; exit 1; }
      run_case d cuda12_build_rejects_an_already_loaded_cuda13_cufile "$bin12" \
        env LD_LIBRARY_PATH="$d" FATHOMDB_T_DEV_CU13_LIB="$cu13"
      ;;
    e)
      run_case e driver_loads_with_only_libcuda_so_1 "$bin12" env
      ;;
    f)
      d="$out/nvtx-f"; mkdir -p "$d"
      ln -sfn "$(readlink -f /usr/local/cuda/lib64/libnvToolsExt.so.1)" "$d/libnvToolsExt.so.1"
      probe_inhibit "$d" libnvToolsExt libnvToolsExt.so.1 '!libnvToolsExt.so' ||
        { echo "precondition (f) failed"; exit 1; }
      run_case f nvtx_loads_with_only_libnvtoolsext_so_1 "$bin12" \
        env LD_LIBRARY_PATH="$d" "$ldso" --inhibit-cache
      ;;
  esac
done

guard_after="$(guard)"
echo "GUARD after $(date -Iseconds) $guard_after"
if [ "$guard_before" != "$guard_after" ]; then
  echo "GUARD CHANGED: stop" >&2
  exit 3
fi
