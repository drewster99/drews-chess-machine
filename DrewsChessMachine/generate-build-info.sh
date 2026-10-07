#!/bin/bash
# Generates BuildInfo.swift with a monotonic per-build counter plus
# current commit metadata. Invoked as an Xcode Run Script build phase
# that runs BEFORE the Compile Sources phase, with `alwaysOutOfDate`
# set so every build (including incremental ones) bumps the counter.
#
# The counter lives at `DrewsChessMachine/build_counter.txt` next to
# this script. If it's missing on first run, it seeds from the repo's
# commit count so the number stays close to prior committed builds.
#
# Build identity (what a saved model's lineage record names as the code
# that wrote it):
#
# - `gitDirty` says whether the compiled project differs from HEAD. Its
#   scope is `DrewsChessMachine/` only — the Xcode project, scheme, test
#   plan, app and test sources and the local packages — because nothing
#   outside it is compiled; an edit under `documentation/` or
#   `experiments/` changes no binary and must not make a build "dirty".
#   The two files this script writes (`build_counter.txt` and
#   `App/BuildInfo.swift`) are tracked and change on every build, so they
#   are excluded; otherwise every build would be dirty, which is what the
#   flag said for every build before this scope existed. Staged and
#   unstaged changes both count, and so does any untracked, non-ignored
#   file under the scope: the project uses synchronized root groups, so an
#   untracked `.swift` file there is compiled.
# - `gitDiffSHA256` identifies the code that was built, so two dirty builds
#   of the same commit can be told apart, and the same code built on two
#   Macs reads as the same code. It is the SHA-256 of the git tree id of
#   the scope as built: the working tree's `DrewsChessMachine/` (tracked
#   and untracked, non-ignored files, without the two generated files)
#   staged into a temporary index started from HEAD, then `write-tree`. A
#   tree id is content-addressed — every file's path, mode and bytes, and
#   nothing else — so it does not depend on what the real index holds
#   (a staged new file and the same file untracked are one identity), on
#   diff options (context, hunk joining, file order, path quoting,
#   algorithm, prefixes), or on line-ending and file-mode settings, which
#   are pinned for the staging. Dirty is that tree differing from HEAD's
#   scope tree built the same way. It equals the hash of the scope's tree
#   once that code is committed. `nil` exactly when the build is clean.
#   Staging writes the blobs and trees into the repository's object store,
#   as `git add` does; they are unreferenced until committed, and git's own
#   garbage collection removes them. The real index is never touched.
# - `xcodeBuild`, `sdkBuild` and `configuration` come from Xcode's build
#   settings. An empty or missing value fails the build instead of
#   writing a blank field, so a renamed build setting can never silently
#   produce an unidentified toolchain.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/DrewsChessMachine/App/BuildInfo.swift"
COUNTER="$SCRIPT_DIR/build_counter.txt"
REPO_ROOT="$SCRIPT_DIR/.."

for SETTING in XCODE_PRODUCT_BUILD_VERSION SDK_PRODUCT_BUILD_VERSION CONFIGURATION; do
    if [ -z "${!SETTING}" ]; then
        echo "error: generate-build-info.sh: build setting $SETTING is empty or missing; refusing to write BuildInfo.swift" >&2
        exit 1
    fi
done

# Paths are relative to the repository root, where every git command runs.
SCOPE="DrewsChessMachine"
GENERATED_COUNTER="DrewsChessMachine/build_counter.txt"
GENERATED_BUILD_INFO="DrewsChessMachine/DrewsChessMachine/App/BuildInfo.swift"
# `DrewsChessMachine/.claude` holds per-user tool settings, never compiled;
# the repository's .gitignore anchors `.claude/*` at the root only, so it is
# left out here instead.
LOCAL_TOOL_SETTINGS="DrewsChessMachine/.claude"
PATHSPEC=("$SCOPE" ":(exclude)$GENERATED_COUNTER" ":(exclude)$GENERATED_BUILD_INFO" ":(exclude)$LOCAL_TOOL_SETTINGS")

# The settings that decide which files and which bytes and mode a
# working-tree file is staged as, pinned so neither a user's nor the
# repository's configuration changes the tree: no line-ending conversion,
# the executable bit read from the file system, symbolic links staged as
# links, and no user-wide excludes file (`core.excludesFile`, which `git
# add -A` would otherwise honor, so an untracked file one Mac ignores
# globally would make the same code dirty on another). The repository's
# own .gitignore files still apply.
git_pinned() {
    git -C "$REPO_ROOT" -c core.autocrlf=false -c core.safecrlf=false -c core.eol=lf \
        -c core.fileMode=true -c core.symlinks=true -c core.excludesFile=/dev/null "$@"
}

# Two temporary indexes, never the repository's own: one holds HEAD, the
# other HEAD with the working tree's scope staged over it.
STAGING_DIR=$(mktemp -d "${TMPDIR:-/tmp}/dcm-build-info.XXXXXX")
HEAD_INDEX="$STAGING_DIR/head.index"
BUILT_INDEX="$STAGING_DIR/built.index"
remove_staging() {
    rm -f "$HEAD_INDEX" "$HEAD_INDEX.lock" "$BUILT_INDEX" "$BUILT_INDEX.lock"
    rmdir "$STAGING_DIR"
}
trap remove_staging EXIT

# Set SCOPE_TREE to the tree id of `DrewsChessMachine/` in the index file
# $1, built from HEAD and — when $2 is "built" — the working tree staged
# over it, without the two generated files. Each step is checked here:
# `set -e` does not reach into a function called as an `if` condition.
SCOPE_TREE=""
build_scope_tree() {
    local INDEX="$1" SOURCE="$2" TREE
    GIT_INDEX_FILE="$INDEX" git_pinned read-tree HEAD || return 1
    if [ "$SOURCE" = "built" ]; then
        GIT_INDEX_FILE="$INDEX" git_pinned add -A -- "${PATHSPEC[@]}" || return 1
    fi
    GIT_INDEX_FILE="$INDEX" git_pinned update-index --force-remove -- \
        "$GENERATED_COUNTER" "$GENERATED_BUILD_INFO" || return 1
    TREE=$(GIT_INDEX_FILE="$INDEX" git_pinned write-tree) || return 1
    SCOPE_TREE=$(git_pinned rev-parse --verify "$TREE:$SCOPE") || return 1
}

if ! build_scope_tree "$HEAD_INDEX" head; then
    echo "error: generate-build-info.sh: could not read HEAD's $SCOPE/ tree" >&2
    exit 1
fi
HEAD_SCOPE_TREE="$SCOPE_TREE"
if ! build_scope_tree "$BUILT_INDEX" built; then
    echo "error: generate-build-info.sh: could not stage the working tree's $SCOPE/ into a temporary index" >&2
    exit 1
fi
BUILT_SCOPE_TREE="$SCOPE_TREE"

if [ "$BUILT_SCOPE_TREE" != "$HEAD_SCOPE_TREE" ]; then
    GIT_DIRTY="true"
    GIT_DIFF_SHA256=$(printf '%s' "$BUILT_SCOPE_TREE" | shasum -a 256 | cut -d ' ' -f 1)
    if ! [[ "$GIT_DIFF_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "error: generate-build-info.sh: could not hash the built tree $BUILT_SCOPE_TREE (got '$GIT_DIFF_SHA256')" >&2
        exit 1
    fi
    GIT_DIFF_SWIFT="\"$GIT_DIFF_SHA256\""
else
    GIT_DIRTY="false"
    GIT_DIFF_SWIFT="nil"
fi

if [ -f "$COUNTER" ]; then
    CURRENT=$(cat "$COUNTER" 2>/dev/null || echo "0")
    case "$CURRENT" in
        ''|*[!0-9]*) CURRENT=0 ;;
    esac
    BUILD_NUM=$((CURRENT + 1))
else
    BUILD_NUM=$(git -C "$REPO_ROOT" rev-list --count HEAD 2>/dev/null || echo "0")
    BUILD_NUM=$((BUILD_NUM + 1))
fi
echo "$BUILD_NUM" > "$COUNTER"

BUILD_DATE=$(date +%Y-%m-%d)
BUILD_TIMESTAMP=$(date +"%Y-%m-%dT%H:%M:%S%z")
GIT_HASH=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

cat > "$OUTPUT" << EOF
// Auto-generated by generate-build-info.sh — do not edit manually.
// Regenerated on every Xcode build (Run Script phase before Compile Sources).
enum BuildInfo {
    static let buildNumber = $BUILD_NUM
    static let buildDate = "$BUILD_DATE"
    static let buildTimestamp = "$BUILD_TIMESTAMP"
    static let gitHash = "$GIT_HASH"
    static let gitBranch = "$GIT_BRANCH"
    /// Whether the compiled project (\`DrewsChessMachine/\`, without this
    /// file and the build counter) differed from \`gitHash\` at build time.
    static let gitDirty = $GIT_DIRTY
    /// SHA-256 of the git tree id of that scope as built (the working
    /// tree's files, tracked and untracked, staged over \`gitHash\`); nil
    /// exactly when \`gitDirty\` is false.
    static let gitDiffSHA256: String? = $GIT_DIFF_SWIFT
    /// Xcode's build version (\`XCODE_PRODUCT_BUILD_VERSION\`).
    static let xcodeBuild = "$XCODE_PRODUCT_BUILD_VERSION"
    /// The SDK's build version (\`SDK_PRODUCT_BUILD_VERSION\`).
    static let sdkBuild = "$SDK_PRODUCT_BUILD_VERSION"
    /// The build configuration (\`CONFIGURATION\`), e.g. Debug or Release.
    static let configuration = "$CONFIGURATION"

    /// One-line human-readable summary, e.g. "build 237 (abc1234*) 2026-04-17".
    /// Asterisk suffix on gitHash: the compiled project (\`gitDirty\`'s
    /// scope) differed from \`gitHash\` at build time.
    static var summary: String {
        let dirtyMarker = gitDirty ? "*" : ""
        return "build \(buildNumber) (\(gitHash)\(dirtyMarker)) \(buildDate)"
    }
}
EOF
