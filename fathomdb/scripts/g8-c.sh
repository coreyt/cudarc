#!/usr/bin/env bash
# G8-c: 13000 build with cuBLAS 13.0.x from the cu13-130 venv.
# shellcheck source=g8-common.sh source-path=SCRIPTDIR
source "$(dirname "$0")/g8-common.sh"
G8_BIN="$(g8_build_report 13000)"
g8_report c-report env LD_LIBRARY_PATH="$cu13_lib" -- "cublas:ok:$cu13_lib:13.0." "cublasLt:ok:$cu13_lib:13.0."
g8_upstream_tests c-cublas-tests 13000 cublas LD_LIBRARY_PATH="$cu13_lib"
g8_end
