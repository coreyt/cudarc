#!/usr/bin/env bash
# G8-b: 12060 build with cuBLAS 12.1.x: the cuBLAS tests pass, or each
# missing symbol is a typed error through is_symbol_present; no panic.
# shellcheck source=g8-common.sh source-path=SCRIPTDIR
source "$(dirname "$0")/g8-common.sh"
G8_BIN="$(g8_build_report 12060)"
dir="$venvs/cu12-121/$site/cublas/lib"
g8_report b-report env LD_LIBRARY_PATH="$dir" -- "cublas:ok:$dir:12.1." "cublasLt:ok:$dir:12.1."
g8_upstream_tests b-cublas-tests 12060 cublas LD_LIBRARY_PATH="$dir"
g8_end
