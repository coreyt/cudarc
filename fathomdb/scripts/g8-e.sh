#!/usr/bin/env bash
# G8-e: 12060 build; the process first dlopens the cu13 libcurand.so.10.
# shellcheck source=g8-common.sh source-path=SCRIPTDIR
source "$(dirname "$0")/g8-common.sh"
G8_BIN="$(g8_build_report 12060)"
g8_report e-preloaded env -- --preload "$cu13_lib/libcurand.so.10" \
  "curand:err:AlreadyLoadedWrongVersion:major: 10, minor: 4"
g8_end
