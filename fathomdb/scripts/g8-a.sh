#!/usr/bin/env bash
# G8-a: 12060 build with cuBLAS 12.4.x and 12.9.x from the scratch venvs.
# shellcheck source=g8-common.sh source-path=SCRIPTDIR
source "$(dirname "$0")/g8-common.sh"
G8_BIN="$(g8_build_report 12060)"
for venv in cu12-124:12.4. cu12-129:12.9.; do
  name="${venv%%:*}" version="${venv#*:}"
  dir="$venvs/$name/$site/cublas/lib"
  g8_report "a-$name-report" env LD_LIBRARY_PATH="$dir" -- \
    "cublas:ok:$dir:$version" "cublasLt:ok:$dir:$version"
  g8_upstream_tests "a-$name-cublas-tests" 12060 cublas LD_LIBRARY_PATH="$dir"
done
g8_end
