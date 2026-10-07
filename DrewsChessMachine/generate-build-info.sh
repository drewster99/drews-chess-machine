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
#   unstaged changes both count (`git diff HEAD`), and so does any
#   untracked, non-ignored file under the scope: the project uses
#   synchronized root groups, so an untracked `.swift` file there is
#   compiled.
# - `gitDiffSHA256` identifies the uncommitted code itself, so two dirty
#   builds of the same commit can be told apart. It is the SHA-256 of
#   `git diff --binary HEAD` over the same scope, followed by every
#   untracked file in byte-sorted path order, each framed as
#   `<path byte length>\n<path>\n<content byte length>\n<content>` so no
#   two different sets of files hash the same stream. A symbolic link's
#   content is its target text, as git would store it. `nil` exactly when
#   the build is clean.
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
PATHSPEC=("$SCOPE" ":(exclude)$GENERATED_COUNTER" ":(exclude)$GENERATED_BUILD_INFO")

# Output options pinned so a user's git configuration (color, external
# diff drivers, text conversion, rename detection, prefixes) cannot
# change the bytes that are hashed.
diff_against_head() {
    git -C "$REPO_ROOT" diff --binary --no-ext-diff --no-textconv --no-color --no-renames \
        --src-prefix=a/ --dst-prefix=b/ HEAD -- "${PATHSPEC[@]}"
}

untracked_files() {
    git -C "$REPO_ROOT" ls-files -z --others --exclude-standard -- "${PATHSPEC[@]}" | LC_ALL=C sort -z
}

# Exit status 0 = no difference, 1 = difference; anything else is a git
# failure, which fails the build rather than guessing either way.
set +e
git -C "$REPO_ROOT" diff --quiet HEAD -- "${PATHSPEC[@]}"
DIFF_STATUS=$?
set -e
case "$DIFF_STATUS" in
    0) TRACKED_DIRTY="false" ;;
    1) TRACKED_DIRTY="true" ;;
    *) echo "error: generate-build-info.sh: git diff failed with status $DIFF_STATUS" >&2; exit 1 ;;
esac

UNTRACKED_COUNT=$(untracked_files | tr -cd '\0' | wc -c | tr -d ' ')

if [ "$TRACKED_DIRTY" = "true" ] || [ "$UNTRACKED_COUNT" -gt 0 ]; then
    GIT_DIRTY="true"
    GIT_DIFF_SHA256=$(
        set -e
        {
            diff_against_head
            untracked_files | while IFS= read -r -d '' FILE_PATH; do
                FULL_PATH="$REPO_ROOT/$FILE_PATH"
                PATH_BYTES=$(printf '%s' "$FILE_PATH" | wc -c | tr -d ' ')
                if [ -L "$FULL_PATH" ]; then
                    LINK_TARGET=$(readlink "$FULL_PATH")
                    CONTENT_BYTES=$(printf '%s' "$LINK_TARGET" | wc -c | tr -d ' ')
                    printf '%s\n%s\n%s\n%s' "$PATH_BYTES" "$FILE_PATH" "$CONTENT_BYTES" "$LINK_TARGET"
                elif [ -f "$FULL_PATH" ]; then
                    CONTENT_BYTES=$(wc -c < "$FULL_PATH" | tr -d ' ')
                    printf '%s\n%s\n%s\n' "$PATH_BYTES" "$FILE_PATH" "$CONTENT_BYTES"
                    cat "$FULL_PATH"
                else
                    echo "error: generate-build-info.sh: untracked path $FILE_PATH is neither a regular file nor a symbolic link" >&2
                    exit 1
                fi
            done
        } | shasum -a 256 | cut -d ' ' -f 1
    )
    if ! [[ "$GIT_DIFF_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "error: generate-build-info.sh: could not hash the uncommitted changes (got '$GIT_DIFF_SHA256')" >&2
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
    /// SHA-256 of the uncommitted changes under that scope (tracked diff
    /// plus framed untracked files); nil exactly when \`gitDirty\` is false.
    static let gitDiffSHA256: String? = $GIT_DIFF_SWIFT
    /// Xcode's build version (\`XCODE_PRODUCT_BUILD_VERSION\`).
    static let xcodeBuild = "$XCODE_PRODUCT_BUILD_VERSION"
    /// The SDK's build version (\`SDK_PRODUCT_BUILD_VERSION\`).
    static let sdkBuild = "$SDK_PRODUCT_BUILD_VERSION"
    /// The build configuration (\`CONFIGURATION\`), e.g. Debug or Release.
    static let configuration = "$CONFIGURATION"

    /// One-line human-readable summary, e.g. "build 237 (abc1234*) 2026-04-17".
    /// Asterisk suffix on gitHash indicates a dirty working tree at build time.
    static var summary: String {
        let dirtyMarker = gitDirty ? "*" : ""
        return "build \(buildNumber) (\(gitHash)\(dirtyMarker)) \(buildDate)"
    }
}
EOF
