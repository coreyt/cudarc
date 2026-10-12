#!/usr/bin/env bash
# G8-f: 12060 build; cuSOLVER with the cu13 cuSPARSE first on the path.
# shellcheck source=g8-common.sh source-path=SCRIPTDIR
source "$(dirname "$0")/g8-common.sh"
G8_BIN="$(g8_build_report 12060)"
g8_report f-cusolver env LD_LIBRARY_PATH="$cu13_lib" -- --not-mapped "$cu13_lib" \
  "cusolver:err:WrongMajorDependency:libcusparse.so.12"
g8_end
