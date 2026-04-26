#!/usr/bin/env python3
"""
apply_code_proposal.py — apply a code-change proposal, build, and verify.

Usage: apply_code_proposal.py <folder>

Reads <folder>/code_proposal.json (written by the autotrain skill from the
code-proposer subagent's response) and applies it to the working tree, then
runs an Xcode build to verify nothing broke. On any failure, fully reverts the
working tree back to HEAD for the touched files so the next training run
operates on the unmodified codebase.

Schema of code_proposal.json:
{
  "change_details": "<≤60 words rationale>",
  "files": {
    "<path-relative-to-repo-root>": "<full new file content as a string>",
    ...
  },
  "rationale": "<longer mechanism explanation>"
}

We use full-file replacement instead of unified diffs for two reasons:
  1. Subagents emit textually-clean full-file content far more reliably than
     diffs that apply cleanly to specific line numbers.
  2. The allowlist gate naturally restricts blast radius to listed files —
     a touched-but-not-listed file is impossible.

ALLOWLIST: an explicit set of files this script will overwrite. Anything else
in the proposal is rejected before we touch the disk. The allowlist is intended
to keep code changes scoped to training-loop behavior, not architecture or
tests. Architecture-changing files (ChessNetwork.swift, BoardEncoder.swift,
PolicyEncoding.swift) are explicitly OUT.

Exit codes:
  0  — proposal applied, build succeeded, all good.
  2  — schema invalid or proposal touches a forbidden file (working tree
       unchanged).
  3  — build failed after applying the proposal; we reverted the touched
       files and the tree is back to HEAD.
  4  — internal error (couldn't run git, couldn't write file, etc.); tree
       state may be partially mutated, manual inspection needed.

The skill's step-8-equivalent for code-change iterations should:
  - On exit 0: continue to training (step 6) as usual.
  - On exit 2: stub-reject the iteration (don't run training, don't commit).
  - On exit 3: stub-reject the iteration (don't run training, don't commit;
               the tree is already reverted).
  - On exit 4: HALT and surface the error — don't try to recover.

The script writes a build log to <folder>/build.log and a status file to
<folder>/code_apply_status.json with the verdict, so the skill can read it
back without re-running anything.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Files this script is allowed to overwrite. Keep tight.
# Rationale per file:
#   ChessTrainer.swift  — training-loop body, loss computation, the BN/SGD
#                          step machinery. The natural place for code-level
#                          training-stability tweaks.
#   ContentView.swift   — session-lifecycle wiring: stat thresholds, alarm
#                          thresholds, controller coupling. Touch-prone for
#                          observability fixes that the proposer might want
#                          to land alongside a training-loop tweak.
ALLOWED_FILES = frozenset({
    "DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift",
    "DrewsChessMachine/DrewsChessMachine/ContentView.swift",
})

# Files explicitly forbidden — we name them so an accidental allowlist
# expansion still won't sweep these in. Architecture and tests are off-limits.
FORBIDDEN_FILES = frozenset({
    "DrewsChessMachine/DrewsChessMachine/ChessNetwork.swift",
    "DrewsChessMachine/DrewsChessMachine/BoardEncoder.swift",
    "DrewsChessMachine/DrewsChessMachine/PolicyEncoding.swift",
    # Tests are also untouchable per the skill's invariants.
    "DrewsChessMachine/DrewsChessMachineTests/",
})

XCODEPROJ = "DrewsChessMachine/DrewsChessMachine.xcodeproj"
BUILD_SCHEME = "DrewsChessMachine"


def write_status(folder: Path, verdict: str, detail: str):
    (folder / "code_apply_status.json").write_text(
        json.dumps({"verdict": verdict, "detail": detail}, indent=2)
    )


def is_forbidden(rel_path: str) -> bool:
    if rel_path in FORBIDDEN_FILES:
        return True
    # Prefix match for directories (e.g. tests).
    return any(rel_path.startswith(f) for f in FORBIDDEN_FILES if f.endswith("/"))


def revert_files(paths):
    """git checkout HEAD -- <paths> for each path. Best-effort: log failures
    but don't raise — caller handles overall verdict."""
    for p in paths:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "checkout", "HEAD", "--", p],
            check=False, capture_output=True,
        )


def run_xcode_build(log_path: Path):
    """Build via xcrun xcodebuild. Returns (ok, last_50_lines).

    We don't use xcode-mcp-server here because this script is invoked by the
    autotrain skill (which uses Bash), not by Claude directly. The MCP tool is
    Claude-facing. xcrun xcodebuild from the CLI builds the same target with
    the same toolchain — close enough for a verification gate.
    """
    cmd = [
        "xcrun", "xcodebuild",
        "-project", XCODEPROJ,
        "-scheme", BUILD_SCHEME,
        "-configuration", "Debug",
        "-quiet",
        "build",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        log_path.write_text("xcodebuild: TIMEOUT after 600s\n")
        return False, "TIMEOUT"
    log = (result.stdout or "") + "\n--- stderr ---\n" + (result.stderr or "")
    log_path.write_text(log)
    tail = "\n".join(log.strip().split("\n")[-50:])
    return result.returncode == 0, tail


def main(argv):
    if len(argv) < 2:
        print("usage: apply_code_proposal.py <folder>", file=sys.stderr)
        return 2
    folder = Path(argv[1]).resolve()
    if not folder.is_dir():
        print(f"folder not found: {folder}", file=sys.stderr)
        return 2

    proposal_path = folder / "code_proposal.json"
    if not proposal_path.is_file():
        print(f"missing {proposal_path}", file=sys.stderr)
        write_status(folder, "schema_error", "code_proposal.json missing")
        return 2

    try:
        proposal = json.loads(proposal_path.read_text())
    except json.JSONDecodeError as e:
        write_status(folder, "schema_error", f"invalid JSON: {e}")
        return 2

    files = proposal.get("files")
    if not isinstance(files, dict) or not files:
        write_status(folder, "schema_error", "no `files` map in proposal")
        return 2

    # Allowlist enforcement BEFORE writing anything.
    for rel in files.keys():
        if is_forbidden(rel):
            write_status(folder, "forbidden", f"forbidden path: {rel}")
            return 2
        if rel not in ALLOWED_FILES:
            write_status(folder, "forbidden",
                         f"path not in allowlist: {rel} (allowed: "
                         f"{sorted(ALLOWED_FILES)})")
            return 2

    # Capture original content so we can restore on build failure. We use git
    # checkout (cleaner) rather than caching bytes in memory — git already
    # tracks HEAD content authoritatively.
    touched_paths = list(files.keys())

    # Sanity: every touched file must already be tracked in git, otherwise we
    # can't `git checkout` to revert it.
    for rel in touched_paths:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", rel],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            write_status(folder, "schema_error",
                         f"file not tracked in git, refusing to overwrite: {rel}")
            return 2

    # Save the patch trail BEFORE applying — even if everything fails we want
    # the trail. The patch is "what the proposer asked for", not "what landed".
    code_patch_dir = folder / "code_patch"
    code_patch_dir.mkdir(exist_ok=True)
    for rel, content in files.items():
        out = code_patch_dir / rel.replace("/", "__")
        out.write_text(content)

    # Apply.
    try:
        for rel, content in files.items():
            target = REPO_ROOT / rel
            target.write_text(content)
    except OSError as e:
        # Partial write — try to revert what we touched.
        revert_files(touched_paths)
        write_status(folder, "io_error", f"write failed: {e}; tree reverted")
        return 4

    # Build.
    build_log = folder / "build.log"
    ok, tail = run_xcode_build(build_log)
    if not ok:
        # Revert before giving up so the next training run uses HEAD.
        revert_files(touched_paths)
        write_status(folder, "build_failed",
                     f"build failed; tree reverted. Tail:\n{tail}")
        return 3

    write_status(folder, "applied",
                 f"code change applied; build OK; touched {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
