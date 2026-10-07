#!/usr/bin/env python3
"""Read-only audit of model files that ran an fp32 policy tail but record none
(POLICY_TAIL_ARCHITECTURE_PLAN.md, PT-D3 rule 2).

From `da159208` (2026-09-28 21:55) every bf16 / fp16 network computed its policy
head's tail in fp32 at one of two boundaries, `fp32_from_pre_bn` or
`mixed_final_projection`. Format v12 makes that an architecture field. A file written
before v12 resolves to the tail it records (its flat `trainer_policy_tail_precision`
key, else its lineage `configuration.policy_tail_precision`), and one recording
neither to `mixed_final_projection` (rule 1). This tool lists every file of the second
kind written while an fp32 tail existed, with the evidence for the tail it actually
ran, so the owner can approve a one-time header edit (rule 3: add only the flat key).
It never writes into the model store: every model file is opened read-only.

Evidence, strongest first:
  0. the file's own `notes`, where a `--new-model` mint from `4582aca0` on states the
     tail its batch-norm warm-up ran under;
  1. the writing process's session log (`~/Library/Logs/DrewsChessMachine`): the line
     `[REPLAY] trainer policy tail precision: X` (builds f6fdd88b..4582aca0) or
     `[NUMERICS] policy_tail_precision=X source=...` (4582aca0 on), in the log whose
     save lines name the file (the newest such log, the file's last writer);
  2. the experiment launch commands in the repository (`experiments/*/README.md`,
     `experiments/**/*.sh`): `--policy-tail-precision X` on the command whose
     `--out-model` names the file's run, or no flag on it;
  3. the build's default, from the file's `built_by_git` / lineage build commit:
     an ancestor of `f6fdd88b` (no flag existed) ran `fp32_from_pre_bn` with certainty;
     from `f6fdd88b` to `de0f22be` the default was `fp32_from_pre_bn` and only
     `--replay-corpus` (and the numerics audit) took the flag, so a corpus-replay file
     of those builds needs a launch line; from `de0f22be` the default was
     `mixed_final_projection`, with the flag on corpus replay, and from `4582aca0` on
     every mode.
Sources that disagree make the file *conflicting*; a file whose tail no source settles
is *missing*. Both are listed and must be left untouched.

It also lists every record that cites a candidate by its whole-file SHA-256, which the
flat-key edit changes: `source_sha256` in any model file's `derivation_history`, and
any 64-hex token in the repository's experiment and documentation records or in the
store's text files (results.json `lineage.checkpoint_sha256` included). Lineage
`parent` references cite `content_sha256`, which covers the tensors only and does not
change.

Usage:
  python3 -I scripts/policy_tail_audit.py [--store DIR] [--logs DIR] [--repo DIR]
      [--report FILE.md] [--json FILE.json] [--no-hash]
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcm_arch  # noqa: E402

FP32_TAIL_COMMIT = "da159208"       # head-numerics fix: the fp32 policy tail exists
FLAG_COMMIT = "f6fdd88b"            # --policy-tail-precision (corpus replay, numerics audit)
MIXED_DEFAULT_COMMIT = "de0f22be"   # mixed_final_projection becomes the default
PROCESS_FLAG_COMMIT = "4582aca0"    # the flag applies to every mode; trainer files record it
FROM_PRE_BN = "fp32_from_pre_bn"
MIXED = "mixed_final_projection"
TAILS = (FROM_PRE_BN, MIXED)
UNTRAINED_CREATORS = {"new-model", "derive-model"}
REPLAY_CREATOR = "replay"
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
LOG_NAME = re.compile(r"^dcm_log_(\d{8})-\d{6}(?:-\d+)?\.txt$")
REPLAY_TAIL_LINE = re.compile(r"\[REPLAY\] trainer policy tail precision: (\S+)")
NUMERICS_TAIL_LINE = re.compile(r"\[NUMERICS\] policy_tail_precision=(\S+) source=(\S+)")
SAVED_FILE = re.compile(r"-> (\S+\.safetensors)")
NOTES_TAIL = re.compile(r"BN warm-up under policy tail precision (\S+)")
SAVED_SESSION = re.compile(r"\[CHECKPOINT\] Saved session \([^)]*\): (\S+\.dcmsession)")
TEXT_SUFFIXES = (".md", ".json", ".jsonl", ".csv", ".txt", ".html", ".js", ".tsv")
MAX_TEXT_BYTES = 64 * 1024 * 1024


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


class Commits:
    """The ancestry questions the build-default rule asks, cached per commit."""

    def __init__(self, repo):
        self.repo = repo
        self.cache = {}
        self.anchor = {name: self._resolve(name) for name in
                       (FP32_TAIL_COMMIT, FLAG_COMMIT, MIXED_DEFAULT_COMMIT, PROCESS_FLAG_COMMIT)}
        missing = [name for name, full in self.anchor.items() if full is None]
        if missing:
            raise SystemExit(f"error: anchor commit(s) {missing} not found in {repo}")

    def _resolve(self, short):
        result = git(self.repo, "rev-parse", "--verify", "--quiet", f"{short}^{{commit}}")
        return result.stdout.strip() if result.returncode == 0 else None

    def _contains(self, commit, anchor):
        """True when `commit` has `anchor` in its history (anchor is an ancestor or itself)."""
        return git(self.repo, "merge-base", "--is-ancestor", self.anchor[anchor], commit).returncode == 0

    def era(self, short):
        """(full hash, era) for a build commit; era is None for an unknown commit."""
        if short in self.cache:
            return self.cache[short]
        full = self._resolve(short)
        if full is None:
            value = (None, None)
        elif not self._contains(full, FP32_TAIL_COMMIT):
            value = (full, "before-fp32-tail")
        elif not self._contains(full, FLAG_COMMIT):
            value = (full, "no-flag")
        elif not self._contains(full, MIXED_DEFAULT_COMMIT):
            value = (full, "flag-window")
        elif not self._contains(full, PROCESS_FLAG_COMMIT):
            value = (full, "mixed-default-replay-flag")
        else:
            value = (full, "mixed-default-process-flag")
        self.cache[short] = value
        return value


def commit_time(repo, short):
    result = git(repo, "log", "-1", "--format=%ct", short)
    if result.returncode != 0:
        raise SystemExit(f"error: no commit time for {short}")
    return int(result.stdout.strip())


def model_files(store):
    for sub in ("Models", "Sessions"):
        base = os.path.join(store, sub)
        for dirpath, _dirs, files in os.walk(base):
            for name in sorted(files):
                if name.endswith(".safetensors"):
                    yield sub, os.path.join(dirpath, name)


def derivation_sources(text):
    """Every `source_sha256` in a derivation_history JSON array (as stored)."""
    found = []
    try:
        history = json.loads(text)
    except ValueError:
        return found
    if isinstance(history, list):
        for entry in history:
            if isinstance(entry, dict) and isinstance(entry.get("source_sha256"), str):
                found.append(entry["source_sha256"])
    return found


def describe(sub, path):
    md = dcm_arch.read_metadata(path)
    arch = json.loads(md["architecture"]) if "architecture" in md else {}
    lineage = json.loads(md["dcm_lineage"]) if "dcm_lineage" in md else None
    configured = None
    build_git = md.get("built_by_git")
    git_dirty = md.get("git_dirty")
    run_id = None
    sources = derivation_sources(md["derivation_history"]) if "derivation_history" in md else []
    if lineage is not None:
        configuration = lineage.get("configuration")
        if isinstance(configuration, dict):
            configured = configuration.get("policy_tail_precision")
        build = lineage.get("build") or {}
        build_git = build_git or build.get("git_hash")
        if build.get("git_dirty") is not None:
            git_dirty = str(build["git_dirty"]).lower()
        run_id = (lineage.get("run") or {}).get("lineage_run_id")
        history = lineage.get("derivation_history")
        if isinstance(history, list):
            sources += derivation_sources(json.dumps(history))
    return dict(
        sub=sub, path=path, name=os.path.basename(path),
        session=os.path.basename(os.path.dirname(path)) if sub == "Sessions" else None,
        version=md.get("dcm_format_version"), dtype=arch.get("compute_data_type"),
        stated=arch.get("policy_tail_precision"), flat=md.get("trainer_policy_tail_precision"),
        configured=configured, creator=md.get("creator"), model_id=md.get("model_id"),
        created=int(md.get("created_at_unix") or 0), training_step=md.get("training_step"),
        build_git=build_git, build_number=md.get("built_by_build"), git_dirty=git_dirty,
        run_id=run_id, notes=md.get("notes"), cites=sorted(set(sources)),
        size=os.path.getsize(path))


def run_stem(name):
    """The run a checkpoint name belongs to: the out-model name without its
    `-latest` / `-step<N>` suffix and extension."""
    stem = name[:-len(".safetensors")] if name.endswith(".safetensors") else name
    stem = re.sub(r"-step\d+$", "", stem)
    stem = re.sub(r"-latest$", "", stem)
    return stem


def scan_logs(logs_dir, since_day):
    """name -> [(log, values)] for every file a log's save lines name, and
    session folder -> [(log, values)]; values = the tails the log states."""
    by_file, by_session = {}, {}
    for log in sorted(os.listdir(logs_dir)):
        match = LOG_NAME.match(log)
        if not match or match.group(1) < since_day:
            continue
        values, names, sessions = set(), set(), set()
        with open(os.path.join(logs_dir, log), "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "policy tail precision" in line or "policy_tail_precision=" in line:
                    for regex in (REPLAY_TAIL_LINE, NUMERICS_TAIL_LINE):
                        hit = regex.search(line)
                        if hit:
                            values.add(hit.group(1))
                if "-> " in line and (".safetensors" in line) and ("[REPLAY]" in line or "[VS-UCI]" in line):
                    for hit in SAVED_FILE.finditer(line):
                        names.add(os.path.basename(hit.group(1)))
                if "[CHECKPOINT] Saved session" in line:
                    hit = SAVED_SESSION.search(line)
                    if hit:
                        sessions.add(os.path.basename(hit.group(1)))
        for name in names:
            by_file.setdefault(name, []).append((log, sorted(values)))
        for session in sessions:
            by_session.setdefault(session, []).append((log, sorted(values)))
    return by_file, by_session


def scan_experiment_launches(repo):
    """run stem -> [(file, value or None)] from every launch command in the
    experiments that names an --out-model."""
    launches = {}
    root = os.path.join(repo, "experiments")
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not (name.endswith(".sh") or name == "README.md"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            joined = re.sub(r"\\\n\s*", " ", text)
            for line in joined.splitlines():
                if "--out-model" not in line:
                    continue
                out = re.search(r"--out-model\s+\"?([^\s\"]+)", line)
                if not out:
                    continue
                tail = re.search(r"--policy-tail-precision\s+(\S+)", line)
                value = tail.group(1).strip("`\"'") if tail else None
                stem = run_stem(os.path.basename(out.group(1)))
                launches.setdefault(stem, []).append((os.path.relpath(path, repo), value))
    return launches


def scan_hex_citations(roots, skip_dirs):
    """64-hex token -> sorted list of files mentioning it, over text files."""
    found = {}
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if os.path.join(dirpath, d) not in skip_dirs and not d.startswith(".")]
            for name in files:
                if not name.endswith(TEXT_SUFFIXES):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > MAX_TEXT_BYTES:
                        continue
                    with open(path, "r", encoding="utf-8", errors="replace") as handle:
                        text = handle.read()
                except OSError:
                    continue
                for token in set(HEX64.findall(text)):
                    found.setdefault(token, set()).add(path)
    return {token: sorted(paths) for token, paths in found.items()}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decide(row, era, log_evidence, launch_evidence):
    """(proposed tail or None, status, reasons)."""
    reasons = []
    build_default = None
    certain = False
    if era is None:
        reasons.append("build commit unknown" if row["build_git"] else "no build recorded")
    elif era == "no-flag":
        build_default, certain = FROM_PRE_BN, True
        reasons.append(f"build predates {FLAG_COMMIT}: no flag existed, fp32_from_pre_bn")
    elif era == "flag-window":
        build_default = FROM_PRE_BN
        if row["creator"] == REPLAY_CREATOR:
            reasons.append(f"build in {FLAG_COMMIT}..{MIXED_DEFAULT_COMMIT}: default fp32_from_pre_bn, "
                           "corpus replay could pass the flag")
        else:
            certain = True
            reasons.append(f"build in {FLAG_COMMIT}..{MIXED_DEFAULT_COMMIT}: only --replay-corpus took the flag, "
                           "so this path ran the default fp32_from_pre_bn")
    elif era == "mixed-default-replay-flag":
        build_default = MIXED
        if row["creator"] == REPLAY_CREATOR:
            reasons.append(f"build in {MIXED_DEFAULT_COMMIT}..{PROCESS_FLAG_COMMIT}: default mixed, corpus replay "
                           "could pass the flag")
        else:
            certain = True
            reasons.append(f"build in {MIXED_DEFAULT_COMMIT}..{PROCESS_FLAG_COMMIT}: only --replay-corpus took the "
                           "flag, so this path ran the default mixed_final_projection")
    elif era == "mixed-default-process-flag":
        build_default = MIXED
        reasons.append(f"build from {PROCESS_FLAG_COMMIT}: default mixed, any mode could pass the flag")
    if row.get("git_dirty") == "true":
        reasons.append("build was dirty (uncommitted changes on top of its commit)")

    logged = sorted({value for _log, values in log_evidence for value in values})
    last_log_values = log_evidence[-1][1] if log_evidence else []
    explicit_launch = sorted({value for _file, value in launch_evidence if value is not None})
    unflagged_launch = any(value is None for _file, value in launch_evidence)

    if len(last_log_values) > 1:
        return None, "conflicting", reasons + [f"writer log states several tails {last_log_values}"]
    if len(explicit_launch) > 1:
        return None, "conflicting", reasons + [f"launch commands state several tails {explicit_launch}"]
    candidates = []
    noted = NOTES_TAIL.search(row.get("notes") or "")
    if noted:
        candidates.append(("notes", noted.group(1)))
    if last_log_values:
        candidates.append(("log", last_log_values[0]))
        if len(logged) > 1:
            reasons.append(f"earlier writer logs of this name state {logged}; the last writer's is used")
    if explicit_launch:
        candidates.append(("launch", explicit_launch[0]))
    if unflagged_launch and not explicit_launch and build_default is not None:
        candidates.append(("launch-without-flag", build_default))
    if certain:
        candidates.append(("build", build_default))
    values = {value for _source, value in candidates}
    if len(values) > 1:
        return None, "conflicting", reasons + [f"sources disagree: {candidates}"]
    if values:
        value = values.pop()
        if value not in TAILS:
            return None, "conflicting", reasons + [f"unknown tail token {value!r}"]
        kinds = sorted({source for source, _value in candidates})
        status = "certain-by-build" if kinds == ["build"] else "evidence"
        return value, status, reasons + [f"settled by {', '.join(kinds)}"]
    if build_default is not None and era != "flag-window":
        return build_default, "build-default", reasons + ["no launch evidence found; the build default"]
    return None, "missing", reasons + ["no log or launch evidence settles it"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", default=os.path.expanduser("~/Library/Application Support/DrewsChessMachine"))
    parser.add_argument("--logs", default=os.path.expanduser("~/Library/Logs/DrewsChessMachine"))
    parser.add_argument("--repo", default=os.path.dirname(here))
    parser.add_argument("--report", default=None)
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--no-hash", action="store_true", help="skip whole-file hashing (no citation matching)")
    args = parser.parse_args()
    stamp = datetime.date.today().isoformat()
    report_path = args.report or os.path.join(args.repo, "documentation", "plans-active", f"POLICY_TAIL_AUDIT_{stamp}.md")
    json_path = args.json_out or os.path.splitext(report_path)[0] + ".json"
    for output in (report_path, json_path):
        if os.path.realpath(output).startswith(os.path.realpath(args.store) + os.sep):
            raise SystemExit("error: refusing to write into the model store")
        if os.path.exists(output):
            raise SystemExit(f"error: {output} exists; this tool never overwrites")

    commits = Commits(args.repo)
    fp32_tail_time = commit_time(args.repo, FP32_TAIL_COMMIT)

    print("reading headers…", file=sys.stderr)
    rows, unreadable = [], []
    for sub, path in model_files(args.store):
        try:
            rows.append(describe(sub, path))
        except (dcm_arch.ArchitectureError, ValueError, OSError) as error:
            unreadable.append((path, str(error)))

    recorded = [r for r in rows if r["stated"] or r["flat"] or r["configured"]]
    unrecorded = [r for r in rows if not (r["stated"] or r["flat"] or r["configured"])]
    fp32_unrecorded = [r for r in unrecorded if r["dtype"] == "float32"]
    reduced_unrecorded = [r for r in unrecorded if r["dtype"] != "float32"]
    before = []
    candidates = []
    for row in reduced_unrecorded:
        full, era = commits.era(row["build_git"]) if row["build_git"] else (None, None)
        row["build_commit"] = full
        row["era"] = era
        if era == "before-fp32-tail" or (era is None and row["created"] < fp32_tail_time):
            before.append(row)
        else:
            candidates.append(row)

    print(f"scanning logs for {len(candidates)} candidates…", file=sys.stderr)
    since = datetime.datetime.fromtimestamp(fp32_tail_time).strftime("%Y%m%d")
    by_file, by_session = scan_logs(args.logs, since)
    launches = scan_experiment_launches(args.repo)

    for row in candidates:
        if row["session"]:
            log_evidence = by_session.get(row["session"], [])
        else:
            log_evidence = by_file.get(row["name"], [])
        launch_evidence = launches.get(run_stem(row["name"]), [])
        row["log_evidence"] = log_evidence
        row["launch_evidence"] = launch_evidence
        row["proposed"], row["status"], row["reasons"] = decide(row, row["era"], log_evidence, launch_evidence)
        row["untrained"] = row["creator"] in UNTRAINED_CREATORS
        if row["session"]:
            trainer = next((r for r in rows if r["session"] == row["session"] and r["name"] == "trainer.safetensors"), None)
            row["session_trainer_tail"] = (trainer or {}).get("flat") or (trainer or {}).get("configured") \
                or (trainer or {}).get("stated")

    citations = {}
    if not args.no_hash:
        print("hashing candidates…", file=sys.stderr)
        for row in candidates:
            row["sha256"] = file_sha256(row["path"])
        derivation_citations = {}
        for row in rows:
            for source in row["cites"]:
                derivation_citations.setdefault(source, []).append(row["path"])
        skip = {os.path.join(args.store, d) for d in ("Models", "Sessions", "Corpora", "FrozenBuilds", "Backups")}
        hex_citations = scan_hex_citations(
            [os.path.join(args.repo, "experiments"), os.path.join(args.repo, "documentation"), args.store], skip)
        for row in candidates:
            cited = {}
            for path in derivation_citations.get(row["sha256"], []):
                cited.setdefault("derivation_history source_sha256", []).append(path)
            for path in hex_citations.get(row["sha256"], []):
                cited.setdefault("text record", []).append(path)
            if cited:
                citations[row["path"]] = cited
            row["citations"] = cited

    write_json(json_path, args, rows, recorded, fp32_unrecorded, before, candidates, unreadable)
    write_report(report_path, json_path, args, rows, recorded, fp32_unrecorded, before, candidates, unreadable, citations)
    print(report_path)


def rel(args, path):
    return os.path.relpath(path, args.store)


def when(created):
    return datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M") if created else "?"


def write_json(path, args, rows, recorded, fp32_unrecorded, before, candidates, unreadable):
    payload = dict(
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        store=args.store, scanned=len(rows), unreadable=unreadable,
        candidates=[{k: (rel(args, v) if k == "path" else v) for k, v in row.items()} for row in candidates])
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


def count(rows, key):
    totals = {}
    for row in rows:
        totals[row[key]] = totals.get(row[key], 0) + 1
    return totals


def write_report(path, json_path, args, rows, recorded, fp32_unrecorded, before, candidates, unreadable, citations):
    trained = [r for r in candidates if not r["untrained"]]
    untrained = [r for r in candidates if r["untrained"]]
    recorded_values = count([dict(v=r["stated"] or r["flat"] or r["configured"]) for r in recorded], "v")
    lines = []
    out = lines.append
    out(f"# Policy tail audit ({datetime.date.today().isoformat()})")
    out("")
    out("Read-only audit for `POLICY_TAIL_ARCHITECTURE_PLAN.md` PT-D3 rule 2, written by "
        "`scripts/policy_tail_audit.py`. No model file was modified. **Nothing below is to be written "
        "until the owner approves it** (rule 3: the edit adds only the flat `trainer_policy_tail_precision` "
        "key; version, architecture string and every other key stay byte-identical).")
    out("")
    out(f"Machine-readable companion: `{os.path.relpath(json_path, args.repo)}`.")
    out("")
    out("## Headline numbers")
    out("")
    out(f"- Model files scanned (`Models/` + `Sessions/`): **{len(rows)}**"
        + (f" ({len(unreadable)} unreadable, listed at the end)" if unreadable else ""))
    out(f"- Recording a tail: **{len(recorded)}** "
        + ", ".join(f"{v} {k}" for k, v in sorted(recorded_values.items(), key=lambda kv: str(kv[0]))))
    out(f"- fp32 files recording none: **{len(fp32_unrecorded)}** (resolve to `does_not_apply`; not candidates)")
    out(f"- bf16 / fp16 files recording none, built before `{FP32_TAIL_COMMIT}` (no fp32 tail existed; rule 1 "
        f"gives them `mixed_final_projection`): **{len(before)}**")
    out(f"- **Candidates** (bf16 / fp16, written from `{FP32_TAIL_COMMIT}`, recording no tail): **{len(candidates)}** "
        f"— {count(candidates, 'sub')}; format versions {dict(sorted(count(candidates, 'version').items()))}")
    out(f"  - trained: **{len(trained)}**; untrained (`new-model` / `derive-model`, no training under any "
        f"tail): **{len(untrained)}**")
    statuses = count(candidates, "status")
    out("  - by status: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    proposed = count([r for r in candidates if r["proposed"]], "proposed")
    out("  - proposed tails (where settled): " + ", ".join(f"{k} {v}" for k, v in sorted(proposed.items())))
    cited = [r for r in candidates if r.get("citations")]
    out(f"  - cited by whole-file SHA-256 somewhere: **{len(cited)}**"
        + ("" if not args.no_hash else " (hashing skipped: --no-hash)"))
    out("")
    out("Status meanings: `evidence` — the file's notes, the writer's session log or the experiment launch "
        "command settles the tail (a launch without the flag settles it to the build default); `certain-by-build` — the build could run only one tail on this path; `build-default` — the "
        "flag existed but no launch evidence was found, so the build default is proposed (weaker: the owner "
        "should confirm); `conflicting` / `missing` — left untouched under rule 3.")
    out("")
    out("Anchor commits: " + ", ".join(f"`{c}`" for c in (FP32_TAIL_COMMIT, FLAG_COMMIT, MIXED_DEFAULT_COMMIT,
                                                                PROCESS_FLAG_COMMIT))
        + " (fp32 tail; flag on corpus replay; mixed default; process-wide flag recorded on trainer files).")
    out("")

    def evidence_text(row):
        parts = []
        if row["log_evidence"]:
            log, values = row["log_evidence"][-1]
            parts.append(f"log `{log}`: {', '.join(values) if values else 'no tail line'}"
                         + (f" (+{len(row['log_evidence']) - 1} earlier writer logs)" if len(row["log_evidence"]) > 1 else ""))
        for file, value in sorted(set(row["launch_evidence"])):
            parts.append(f"`{file}`: {value or 'no flag'}")
        return "; ".join(parts) if parts else "—"

    out("## Trained candidates")
    out("")
    out("Grouped by run (the out-model stem); every file of a group is listed in the JSON companion with its "
        "own evidence. A group's row counts its files per proposed tail and status.")
    out("")
    groups = {}
    for row in trained:
        key = row["session"] or run_stem(row["name"])
        groups.setdefault(key, []).append(row)
    out("| run | files | versions | build | created | proposed (status) | evidence | reasons |")
    out("|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        group = groups[key]
        versions = ",".join(sorted({r["version"] or "?" for r in group}, key=lambda v: (len(v), v)))
        builds = ",".join(sorted({f"{r['build_git'] or '?'}" + (f" (b{r['build_number']})" if r.get('build_number') else "")
                                  for r in group}))
        created = f"{when(min(r['created'] for r in group))} … {when(max(r['created'] for r in group))}"
        outcome = {}
        for r in group:
            label = f"{r['proposed'] or '—'} ({r['status']})"
            outcome[label] = outcome.get(label, 0) + 1
        outcome_text = "<br>".join(f"{k}: {v}" for k, v in sorted(outcome.items()))
        evidence = sorted({evidence_text(r) for r in group})
        reasons = sorted({"; ".join(r["reasons"]) for r in group})
        session_note = ""
        if group[0]["session"]:
            session_note = f" — session trainer file records `{group[0].get('session_trainer_tail')}` (rule 4 resolves it at load)"
        out(f"| `{key}`{session_note} | {len(group)} | {versions} | {builds} | {created} | {outcome_text} | "
            f"{'<br>'.join(evidence)} | {'<br>'.join(reasons)} |")
    out("")
    out("## Untrained candidates (listed separately)")
    out("")
    out("No training ran under any tail. For a bf16 / fp16 mint the batch-norm running statistics come from a "
        "warm-up forward pass under the build's tail, so the tail can differ in their last bits only.")
    out("")
    out("| file | creator | version | build | created | proposed (status) | notes |")
    out("|---|---|---|---|---|---|---|")
    for row in sorted(untrained, key=lambda r: r["name"]):
        out(f"| `{rel(args, row['path'])}` | {row['creator']} | {row['version']} | {row['build_git'] or '?'} | "
            f"{when(row['created'])} | {row['proposed'] or '—'} ({row['status']}) | {(row['notes'] or '').replace('|', '/')} |")
    out("")
    out("## Whole-file SHA-256 citations")
    out("")
    if args.no_hash:
        out("Not computed (`--no-hash`).")
    elif not cited:
        out("None found.")
    else:
        out("The flat-key edit changes each file's whole-file SHA-256 (its `content_sha256`, over the tensors, "
            "does not change). In-file `derivation_history` values are history and are never rewritten (rule 3): "
            "approve these files knowing the citation names the pre-edit bytes, or exclude them.")
        out("")
        out("| candidate | sha256 | cited by |")
        out("|---|---|---|")
        for row in sorted(cited, key=lambda r: r["name"]):
            refs = []
            for kind, paths in sorted(row["citations"].items()):
                for p in paths:
                    shown = os.path.relpath(p, args.store) if p.startswith(args.store) else os.path.relpath(p, args.repo)
                    refs.append(f"{kind}: `{shown}`")
            out(f"| `{rel(args, row['path'])}` | `{row['sha256'][:16]}…` | {'<br>'.join(refs)} |")
    out("")
    out("## Conflicting or missing")
    out("")
    unsettled = [r for r in candidates if r["status"] in ("conflicting", "missing")]
    if not unsettled:
        out("None.")
    else:
        out("| file | status | evidence | reasons |")
        out("|---|---|---|---|")
        for row in sorted(unsettled, key=lambda r: r["name"]):
            out(f"| `{rel(args, row['path'])}` | {row['status']} | {evidence_text(row)} | {'; '.join(row['reasons'])} |")
    if unreadable:
        out("")
        out("## Unreadable files")
        out("")
        for p, error in unreadable:
            out(f"- `{rel(args, p)}`: {error}")
    out("")
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
