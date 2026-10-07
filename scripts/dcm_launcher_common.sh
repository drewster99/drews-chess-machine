# Shared by the repo-root launchers (run_debug.sh, run_release.sh,
# run_latest.sh, run_uci.sh). Sourced, not run.
#
# One place resolves which binary a launcher runs and says so before it
# runs: every launch prints, to stderr, the launcher, the configuration and
# path of the binary, the binary's own `--version` line (build counter,
# commit, configuration, build time) and the launch time. Stderr, so a UCI
# engine's stdout carries nothing but the protocol.

DCM_PROG="$(basename "$0")"
DCM_DERIVED_DATA="$HOME/Library/Developer/Xcode/DerivedData"

# Refuse to launch a second DrewsChessMachine instance — macOS LaunchServices
# would normally refocus the running app, but execing the binary directly
# bypasses that and silently spawns a concurrent process, which wastes the
# GPU and skews any ongoing training. Override with DCM_FORCE_LAUNCH=1 if
# you truly need two instances.
dcm_refuse_if_running() {
    if [ "${DCM_FORCE_LAUNCH:-0}" != "1" ] && pgrep -x DrewsChessMachine >/dev/null 2>&1; then
        echo "$DCM_PROG: DrewsChessMachine is already running (set DCM_FORCE_LAUNCH=1 to override)" >&2
        exit 10
    fi
}

# dcm_resolve_binary <Debug|Release|latest>
# Sets DCM_BINARY and DCM_CONFIGURATION to the newest (by modification time)
# built binary of that configuration across every DerivedData folder of the
# project; `latest` takes the newest of both. Xcode can leave several
# DrewsChessMachine-* folders behind, so a bare glob could name more than one
# binary — and exec would run the first with the rest as its arguments.
dcm_resolve_binary() {
    local wanted="$1"
    local configurations
    case "$wanted" in
        Debug|Release) configurations="$wanted" ;;
        latest) configurations="Debug Release" ;;
        *) echo "$DCM_PROG: unknown configuration '$wanted'" >&2; exit 1 ;;
    esac
    DCM_BINARY=""
    DCM_CONFIGURATION=""
    local newest_mtime=0 configuration path mtime
    shopt -s nullglob
    for configuration in $configurations; do
        for path in "$DCM_DERIVED_DATA"/DrewsChessMachine-*/Build/Products/"$configuration"/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine; do
            [ -x "$path" ] || continue
            mtime=$(stat -f %m "$path") || { echo "$DCM_PROG: cannot stat $path" >&2; exit 1; }
            if [ "$mtime" -gt "$newest_mtime" ]; then
                newest_mtime="$mtime"
                DCM_BINARY="$path"
                DCM_CONFIGURATION="$configuration"
            fi
        done
    done
    shopt -u nullglob
    if [ -z "$DCM_BINARY" ]; then
        echo "$DCM_PROG: no built DrewsChessMachine binary found for $wanted under $DCM_DERIVED_DATA" >&2
        exit 1
    fi
}

# Print which binary is about to run, its build, and when. A binary built
# before `--version` existed refuses the flag (exit 2, before any window or
# GPU setup); that is reported rather than hidden.
dcm_print_banner() {
    local version
    if ! version=$("$DCM_BINARY" --version 2>/dev/null); then
        version="version unavailable (this binary predates --version)"
    fi
    {
        echo "$DCM_PROG: launching $DCM_CONFIGURATION binary (file modified $(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$DCM_BINARY"))"
        echo "$DCM_PROG:   $DCM_BINARY"
        echo "$DCM_PROG:   $version"
        echo "$DCM_PROG:   launch time $(date '+%Y-%m-%dT%H:%M:%S%z')"
    } >&2
}
