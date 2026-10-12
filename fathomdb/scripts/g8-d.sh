#!/usr/bin/env bash
# G8-d: 12060 build with the cu13 cuRAND and cuSPARSE first on the path:
# both rejected and unmapped, the system 12.6 copies loaded and reported;
# with the system copies hidden (--inhibit-cache), the typed error.
# shellcheck source=g8-common.sh source-path=SCRIPTDIR
source "$(dirname "$0")/g8-common.sh"
G8_BIN="$(g8_build_report 12060)"
g8_report d-cu13-first env LD_LIBRARY_PATH="$cu13_lib" -- --not-mapped "$cu13_lib" \
  "curand:ok:$sys_lib:10.3." "cusparse:ok:$sys_lib:12.5."
g8_report d-system-hidden env LD_LIBRARY_PATH="$cu13_lib" "$ldso" --inhibit-cache -- --not-mapped "$cu13_lib" \
  "curand:err:WrongVersion:major: 10, minor: 4" "cusparse:err:WrongVersion:major: 12, minor: 6"
g8_end
