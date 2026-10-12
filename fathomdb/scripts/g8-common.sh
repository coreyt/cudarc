# shellcheck shell=bash disable=SC2034
# Sourced by g8-a.sh … g8-f.sh (test plan § 3.4 G8, cell G8 on windchill3).
#
# Environment:
#   G8_SRC   the cudarc tree to test (default: this checkout)
#   G8_OUT   logs and the scratch report project (default: a new mktemp dir)
#   FATHOMDB_VENVS  the scratch venvs (default ~/fathomdb-forks/venvs)
set -euo pipefail

g8_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G8_SRC="$(cd "${G8_SRC:-$g8_here/../..}" && pwd)"
G8_OUT="${G8_OUT:-$(mktemp -d)}"
mkdir -p "$G8_OUT"
G8_OUT="$(cd "$G8_OUT" && pwd)"
venvs="${FATHOMDB_VENVS:-$HOME/fathomdb-forks/venvs}"
site=lib/python3.12/site-packages/nvidia
cu13_lib="$venvs/cu13-130/$site/cu13/lib"
sys_lib="$(readlink -f /usr/local/cuda)/targets/x86_64-linux/lib"
ldso=/lib64/ld-linux-x86-64.so.2
G8_FEATURES=cublas,cublaslt,curand,cusparse,cusolver,cufile
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
g8_fail=0

g8_guard() {
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | tr '\n' ' ')cuda=$(readlink -f /usr/local/cuda)"
}
g8_guard_before="$(g8_guard)"
echo "GUARD before $(date -Iseconds) $g8_guard_before"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader | sed 's/^/DEVICE /'
echo "SRC $G8_SRC ($(git -C "$G8_SRC" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
echo "OUT $G8_OUT"

g8_end() {
  local after
  after="$(g8_guard)"
  echo "GUARD after $(date -Iseconds) $after"
  if [ "$g8_guard_before" != "$after" ]; then
    echo "GUARD CHANGED: stop" >&2
    exit 3
  fi
  echo "G8 $(basename "$0" .sh): $([ $g8_fail = 0 ] && echo PASS || echo FAIL)"
  exit $g8_fail
}

# g8_build_report <cuda version> -> prints the report binary path
g8_build_report() {
  local v="$1" d="$G8_OUT/g8-report-$1" pkg
  # The tree's package name: `cudarc-fathomdb` from d04ea18, `cudarc` before.
  pkg="$(python3 -I -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["package"]["name"])' "$G8_SRC/Cargo.toml")"
  mkdir -p "$d"
  cat >"$d/Cargo.toml" <<TOML
[package]
name = "g8_report"
version = "0.0.0"
edition = "2021"

[[bin]]
name = "g8_report"
path = "$g8_here/g8_report.rs"

[dependencies]
cudarc = { package = "$pkg", path = "$G8_SRC", default-features = false, features = ["std", "driver", "dynamic-loading", "runtime", "nvrtc", "cuda-$v", "cublas", "cublaslt", "curand", "cusparse", "cusolver", "cufile"] }

[workspace]
TOML
  (
    cd "$d"
    CUDARC_CUDA_VERSION="$v" CARGO_TARGET_DIR="$d/target" cargo build --offline -q >"$d/build.log" 2>&1 ||
      { cat "$d/build.log" >&2; exit 1; }
    cargo tree --offline -e normal >"$d/tree.txt" 2>/dev/null || true
  )
  echo "$d/target/debug/g8_report"
}

# g8_report <label> [env assignments and/or loader prefix ...] -- <report args...>
g8_report() {
  local label="$1"; shift
  local pre=()
  while [ "$1" != "--" ]; do pre+=("$1"); shift; done
  shift
  local log="$G8_OUT/$label.log" status=0
  echo "== $label: ${pre[*]} $*"
  "${pre[@]}" "$G8_BIN" "$@" >"$log" 2>&1 || status=$?
  sed 's/^/  /' "$log"
  [ $status = 0 ] || g8_fail=1
}

# g8_upstream_tests <label> <cuda version> <filter> [env assignments ...]
# The upstream tests matching <filter> (unchanged) in this tree, with the
# cell's libraries; a failure or panic fails the sub-cell. (FC8 receipts add
# the count-tests name check.)
g8_upstream_tests() {
  local label="$1" v="$2" filter="$3"; shift 3
  local log="$G8_OUT/$label.log" status=0
  echo "== $label: cargo test --lib -F cuda-$v,$G8_FEATURES -- $filter ($*)"
  (
    cd "$G8_SRC"
    python3 -I "$g8_here/use-baseline-lock.py" Cargo.lock
    env "$@" CUDARC_CUDA_VERSION="$v" cargo test --locked --lib -F "cuda-$v,$G8_FEATURES" \
      -- "$filter" --test-threads=1 >"$log" 2>&1
  ) || status=$?
  grep -E '^test .* \.\.\. |^test result' "$log" | sed 's/^/  /'
  if grep -q 'panicked at' "$log"; then
    grep -A2 'panicked at' "$log" | sed 's/^/  panic: /'
  fi
  [ $status = 0 ] || g8_fail=1
}
