#!/bin/bash
# Launch the newest built DrewsChessMachine (Debug or Release, whichever was built last),
# printing which binary, its build and the launch time first
# (scripts/dcm_launcher_common.sh).
source "$(dirname "$0")/scripts/dcm_launcher_common.sh"
dcm_refuse_if_running
dcm_resolve_binary latest
dcm_print_banner
exec "$DCM_BINARY" "$@"
