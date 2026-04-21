---
name: autotrain
description: Run one iteration of the automated chess-training parameter-tuning loop. Propose a parameter change, run a time-limited training run, analyze results vs. a stated improvement goal, and accept (commit+push) or reject the change. Intended to be driven by `/loop /autotrain`. Use when the user says "autotrain", "tune the chess params automatically", or similar.
---

# autotrain — one iteration of the chess-training parameter-tuning loop

This skill runs **one** iteration of a propose → run → analyze → accept/reject loop for tuning the training hyperparameters of the chess engine in this repo. Invoke via `/loop /autotrain` to run continuously.

Repo root (referred to below as `$ROOT`): `/Users/andrew/cursor/drews-chess-machine`.

The app is launched via `$ROOT/run_debug.sh` (a thin wrapper that locates the Debug `.app` under DerivedData and execs it with any extra args).

## State layout

- `$ROOT/parameters.json` — current best parameters. The propose step mutates a copy of this.
- `$ROOT/results.json` — results from the current-best training run. The analyzer compares new results against this.
- `$ROOT/experiments/goal.txt` — persisted improvement goal (a free-form human description of what "better" means). Prompt the user once on first run if missing.
- `$ROOT/experiments/<UTC-timestamp>/` — per-iteration folder. Contains:
  - `proposal.json` — proposer subagent's raw structured response
  - `proposal.md` — human-readable description (copy of `change_details`)
  - `parameters.json` — the new parameters to test (copy of `proposal.parameters`)
  - `previous_result.json` — snapshot of `$ROOT/results.json` at iteration start
  - `result.json` — output of the training run
  - `analysis.json` — analyzer subagent's structured response
  - `run.log` — stdout/stderr of the app run

## Tools

- `$ROOT/.claude/skills/autotrain/run_training.sh <params-json> <time-limit-seconds> <output-json> [log-file]`
  Thin wrapper around `$ROOT/run_debug.sh` that passes the right CLI flags, waits for exit, and verifies the output file. Flag names are configurable via env vars `DCM_PARAMS_FLAG`, `DCM_TIME_FLAG`, `DCM_OUT_FLAG` if they ever change.
- `$ROOT/.claude/skills/autotrain/regen_dashboard.py`
  Scans `experiments/*/` and rewrites `$ROOT/experiment_results.js` (and creates `$ROOT/experiment_results.html` the first time). The HTML page auto-polls the `.js` sidecar every 15 s via a cache-busted `<script>` reinjection — new rows are appended in place without reloading, so scroll position is preserved. Invoke as `python3 $ROOT/.claude/skills/autotrain/regen_dashboard.py`.
- `$ROOT/.claude/skills/autotrain/summarize_results.py <path-to-results.json>`
  Emits a compact (~1-5 KB) JSON digest of a results.json — arena scoreboards, per-metric trajectories (first/min/median/mean/max/final), collapse signals, candidate-probe first/mid/last, build-number stamp. Used to keep subagent prompts small; the raw file is ~1 MB / ~300k tokens and you should never paste it into a subagent directly.
- `$ROOT/.claude/skills/autotrain/validate_params.py <path-to-parameters.json>`
  Sanity-bound check on a proposed parameters.json. Bounds are intentionally very wide — the point is to catch proposer hallucinations (`learning_rate: 5.0`, fractional worker counts, capacity > 1e8), not to gatekeep tuning. Exits 0 on valid, 1 on violations (printed to stderr). Invoked by the skill after step 5's proposal lands; a violation rejects the iteration before any training run happens.

## Iteration procedure

Work through these steps in order. Do each step; don't skip.

### 0. Confirm branch with user

Run `git rev-parse --abbrev-ref HEAD` to read the current branch.

- **If the branch is literally `experiments`**, proceed without asking — that branch is the designated autotrain scratch branch, confirmation would be pure friction. Log a one-liner like `autotrain: branch=experiments, proceeding without prompt` and continue.
- **On any other branch** (including `main`), show the branch name and ask: "autotrain will commit and push accepted iterations to **<branch>** — proceed?" Wait for confirmation before continuing. If the user says switch branches, let them do that and re-invoke.

Once the user has confirmed a non-`experiments` branch once in a given session, you may skip this prompt for subsequent `/loop` iterations **in that same session**, but resume asking if the branch ever changes under you.

### 0.5. Bail if the GPU is busy

Before doing any work, check for conflicts so a `/loop` tick doesn't step on a manual run:

- `pgrep -x DrewsChessMachine` — any match means the app is already running (either the user launched it manually or a prior iteration is still in-flight). Do nothing: print `autotrain: DrewsChessMachine already running — skipping iteration` and end. `/loop` will retry on the next tick.
- `$ROOT/.dcm.training.lock` — if the file exists, read the first line as a PID and run `kill -0 <pid>` to check liveness. Live PID → skip (same message, but mention the PID). Dead PID (stale lock from a crash) → leave it; `run_training.sh` will clear it on next use.

`run_training.sh` itself enforces the same guards and exits with status **10** specifically for "skip iteration" (distinct from real failures at status ≠ 0). If you see exit code 10 from a training run, treat it as a skip, not a failure — don't write a FAILED analysis stub; just log and end.

### 0.6. Replicate-or-halt check

Walk `experiments/*/` newest → oldest and compute two numbers:

- **`failure_streak`** — trailing count of iterations classified as `regressed` (dashboard status `REJECTED`) or `FAILED`. Walking newest → oldest:
  - `regressed` / `FAILED` → add 1 to the streak.
  - `neutral` (dashboard status `NEUTRAL`) → transparent: skip over without adding or resetting. A neutral result isn't a failure, but it isn't progress either; the streak pauses in place.
  - `IN_PROGRESS` → transparent (same as before; not yet a decision).
  - `improved` (status `ACCEPTED`) or `SEED` → stop walking and reset to 0.

  The dashboard already exposes this as `window.AGGREGATES.failure_streak` in `experiment_results.js`.
- **`trailing_replicates`** — trailing count of iterations whose folder name ends in `-replicate` (walking newest backward until the first non-replicate folder breaks the count). Resets implicitly the moment a non-replicate iteration happens.

Then decide mode for this iteration:

- **`trailing_replicates ≥ 3`** → **HALT**. We've run the baseline verbatim three times in a row without an accept. Either the baseline is genuinely unreproducible or the analyzer's goal can't be satisfied by the current best. Print `autotrain: halted — 3 consecutive replicates did not promote, manual intervention recommended`, include pointers to those three folders, and end. Do NOT enter another replicate or a normal iteration.
- **`failure_streak ≥ 15`** (and `trailing_replicates < 3`) → enter **REPLICATE mode**. The proposer has been failing to find improvements for 15 iterations; rather than ask it to try again, re-run the current-best parameters verbatim to probe whether the "best" result is actually reproducible. Set a local flag `replicate_mode = true` that step 4 and step 5 observe. Rationale: if the baseline reproduces, the proposer is the problem (analyzer confirms rejections are real); if the baseline does *not* reproduce, the "current best" was noise-lucky and we may get a free accept from a decent replicate.
- **`failure_streak` 10–14** → proceed normally but note it in the iteration's summary line (`... (streak=12, replicate at 15)`).
- **`failure_streak` 5–9** → proceed normally but note it (`... (streak=7, watch)`).
- **`failure_streak` < 5** → proceed silently.

### 1. Check working tree is clean enough to commit

Run `git status --porcelain`. If there are staged changes unrelated to autotrain, stop and tell the user — we will commit on an accepted improvement and don't want to sweep unrelated work into the commit. Unstaged build-counter / BuildInfo.swift drift (from the pre-Compile script phase) is expected and fine to leave alone — don't stage them.

### 2. Load or create the improvement goal

If `$ROOT/experiments/goal.txt` doesn't exist, ask the user: "What's the improvement goal for this autotrain run? (e.g., 'prevent policy collapse — keep pEnt well above 5.0 after 15 minutes of training from scratch')." Save the answer to the file. Read the goal from the file.

### 3. Seed if needed

A seed run is needed whenever `$ROOT/results.json` is missing (either first-ever run, or results got deleted). Handle `parameters.json` carefully:

  a. **If `$ROOT/parameters.json` does not exist**, write the default parameters block (reproduced at the bottom of this file) to `$ROOT/parameters.json`. **If it already exists, leave it alone** — the user may have hand-tuned values they don't want clobbered with defaults.
  b. Create a test folder `experiments/<timestamp>-seed/` and copy the current `$ROOT/parameters.json` in there.
  c. Run `run_training.sh` with **300 seconds** (5-minute seeding run) outputting to the test folder's `result.json`.
  d. Copy the seed's `result.json` to `$ROOT/results.json`. Do **not** re-copy parameters back to root — they're already there.
  e. Run `regen_dashboard.py` so the dashboard shows the seed row.
  f. Skip straight to step 8 (commit) with commit message `autotrain: seed baseline`, then end this iteration.

If `$ROOT/results.json` exists but `$ROOT/parameters.json` is missing, that's an inconsistent state — stop and tell the user; don't invent parameters to match a prior result.

### 4. Create the test folder

Timestamp = UTC `YYYYMMDD-HHMMSS`.

- **Normal mode**: Folder = `$ROOT/experiments/<timestamp>/`.
- **Replicate mode** (set in step 0.6): Folder = `$ROOT/experiments/<timestamp>-replicate/`. The suffix is how the dashboard, the streak counter, and the next iteration's trailing-replicate count identify replicate iterations.

Regardless of mode:

- Copy `$ROOT/results.json` → `<folder>/previous_result.json`.
- Copy `$ROOT/parameters.json` → `<folder>/previous_parameters.json`.
- Copy `$ROOT/experiments/goal.txt` → `<folder>/goal.txt` so iterations are pinned to the goal at the time of the iteration. If the user edits `goal.txt` mid-loop, older history entries still reflect the goal they were actually judged against.

### 5. Propose a change (subagent)

**Replicate-mode shortcut** (if step 0.6 set `replicate_mode = true`): do NOT spawn a proposer subagent. Instead:
- Copy `$ROOT/parameters.json` → `<folder>/parameters.json` verbatim (no change).
- Write `<folder>/proposal.json`:
  ```json
  {
    "change_details": "Replicate mode — re-running current best parameters verbatim to probe baseline reproducibility after a long non-accept streak.",
    "parameters": <contents of $ROOT/parameters.json>,
    "mode": "replicate"
  }
  ```
- Write `<folder>/proposal.md` with the `change_details` text.
- Write `<folder>/training_time.txt` with `600` (use the full cap; a short replicate wouldn't be informative).
- Run `validate_params.py` as a sanity check (should trivially pass since the baseline already validates). If it somehow fails, that's a real problem — stub-reject and halt the replicate cascade for user attention.
- Run `regen_dashboard.py`.
- Skip the rest of step 5 and go straight to step 6.

**Normal mode** (from here on):

**Before spawning** the subagent, assemble the context inputs. **Never paste the raw `results.json`** into a subagent prompt — it's ~1 MB / ~300k tokens. Use the summarizer.

  a. `latest_summary` = stdout of `python3 $ROOT/.claude/skills/autotrain/summarize_results.py $ROOT/results.json`.
  b. `recent_history` = up to the **10 most recent** subfolders of `$ROOT/experiments/`, newest last. For each entry include:
       - `timestamp` — folder name.
       - `status` — `ACCEPTED` / `REJECTED` / `SEED` / `FAILED`.
       - `goal` — contents of `<folder>/goal.txt` if present, else `null`. Distinguishes iterations judged under an older goal from ones judged under the current goal.
       - `change_details` — the proposer's original rationale (from `proposal.json`). Continuity of prior reasoning.
       - `changed_params_diff` — the params-vs-previous diff.
       - `analysis_commentary` — the analyzer's rebuttal (from `analysis.json`).
       - `training_time_seconds` — from `<folder>/training_time.txt` if present, else `null`.
       - `summary` — summarizer output on `<folder>/result.json` if valid, else `null`.
       - `result_json_path` — absolute path to `<folder>/result.json`.
       - `parameters_json_path` — absolute path to `<folder>/parameters.json`.
       - `previous_parameters_json_path` — absolute path to `<folder>/previous_parameters.json`.
     Skip the current iteration's folder. Skip corrupted or in-progress folders silently.
  c. `exploration_mode` — set to `true` when the **current iteration number is a multiple of 10**, else `false`. Iteration number = count of existing non-seed experiment folders **plus 1** (counting the one we're about to run). So iterations 10, 20, 30, … get explore mode. Combats local-minimum ruts.

Then spawn a general-purpose subagent with this prompt (pass as a fenced JSON block):
```json
{
  "improvement_goal": "<contents of goal.txt>",
  "current_parameter_configuration": <contents of $ROOT/parameters.json>,
  "latest_results_summary": <latest_summary>,
  "latest_results_json_path": "<absolute path to $ROOT/results.json>",
  "recent_history": <recent_history>,
  "training_time_seconds_max": 600,
  "exploration_mode": <boolean>
}
```

Instructions embedded in the prompt:
- The summary is a digest. For detail not in it, run `jq` or `python3 -c "..."` via Bash against the path. **Do not use the Read tool on the JSON** — it's ~1 MB.
- Example: `jq '.stats | map(.policy_entropy) | [.[0], min, max, .[-1]]' <path>`.
- **If `exploration_mode` is true**: propose a **bolder or orthogonal change** than recent history — change a parameter you haven't touched recently, try a larger magnitude, or explore an axis the goal hasn't been examined against. Still respect physical bounds and the goal.
- **If `exploration_mode` is false**: propose an incremental, low-risk change aimed directly at the goal.
- Return JSON of exactly this shape and **nothing else** (no markdown, no prose, no leading comment):
  ```json
  {
    "change_details": "<BRIEF rationale, 1-2 sentences, <= 60 words>",
    "parameters": { ... full parameters object, every key from input preserved exactly ... },
    "training_time_seconds": <integer in [60, training_time_seconds_max], OPTIONAL>,
    "training_time_rationale": "<BRIEF, one short sentence, <= 20 words, OPTIONAL>"
  }
  ```
- **Keep `change_details` brief**: 1-2 sentences, under 60 words. Don't re-explain the overall strategy or restate prior history — the reader has all of it. Just state the change and its expected mechanism.
- Preserve every key in the input parameters object. Never drop or rename a key.
- Respect physical bounds: `replay_ratio_target > 0`, integer worker counts, non-negative decay values, positive batch sizes, etc. A separate validator enforces wide bounds on the server side — stay well within them.
- Only set `training_time_seconds` if you have a specific reason. If you do, include a brief `training_time_rationale`.

**After** the subagent returns:
1. Parse the JSON. If parsing fails or required keys (`change_details`, `parameters`) are missing, retry once with a terser reminder of the schema. If the retry also fails, write a stub `analysis.json` with `{"is_result_improved": false, "analysis_commentary": "proposer returned invalid JSON twice — skipping iteration"}`, run `regen_dashboard.py`, and jump to step 8 (reject).
2. If `training_time_seconds` is present, clamp to `[60, 600]`. If absent, use 600.
3. Save the full raw JSON response to `<folder>/proposal.json`.
4. Write `change_details` to `<folder>/proposal.md`.
5. Write the `parameters` object to `<folder>/parameters.json`.
6. Write the chosen (post-clamp) training time to `<folder>/training_time.txt` as a plain integer on one line.
7. **Sanity-bound validation**: run `python3 $ROOT/.claude/skills/autotrain/validate_params.py <folder>/parameters.json`. If it exits non-zero, the proposal has an out-of-bounds value. Capture the violation message, write a stub `analysis.json` with `{"is_result_improved": false, "analysis_commentary": "proposal failed bounds check: <violations>"}`, run `regen_dashboard.py`, and jump to step 8 (reject). This iteration counts toward the failure streak (by design — if the proposer keeps hallucinating bad values, the loop should halt).

Then run `regen_dashboard.py` so the dashboard shows this iteration as `IN_PROGRESS` while training runs.

### 6. Run training

Read the training time from `<folder>/training_time.txt` (fall back to 600 if the file is missing, e.g. during the seed path).

Invoke `run_training.sh <folder>/parameters.json <training_time> <folder>/result.json <folder>/run.log`. `run_training.sh` enforces its own hard cap at 600 s — anything larger gets clamped down.

If the script exits non-zero or `result.json` is missing/invalid, write a stub `analysis.json` with `{"is_result_improved": false, "analysis_commentary": "training run failed: <reason>"}`, run `regen_dashboard.py`, and jump to step 8 (reject). Exit status `10` from `run_training.sh` is not a failure — it means "skip iteration", handled in step 0.5.

### 7. Analyze (subagent)

**Before spawning**, summarize both results files. **Never paste raw `result.json` / `previous_result.json` into the prompt** (same size concern as step 5):

  a. `previous_summary` = stdout of `python3 $ROOT/.claude/skills/autotrain/summarize_results.py <folder>/previous_result.json`.
  b. `new_summary` = stdout of `python3 $ROOT/.claude/skills/autotrain/summarize_results.py <folder>/result.json`.

Spawn a general-purpose subagent with this prompt:
```json
{
  "improvement_goal": "<contents of goal.txt>",
  "training_time_seconds": <contents of <folder>/training_time.txt as an integer>,
  "change_proposal": <contents of proposal.json>,
  "previous_results_summary": <previous_summary>,
  "previous_results_json_path": "<absolute path to <folder>/previous_result.json>",
  "new_results_summary": <new_summary>,
  "new_results_json_path": "<absolute path to <folder>/result.json>"
}
```

Instructions embedded in the prompt:
- The summaries are digests of ~1 MB JSON files. For specific details not in the summaries, use `jq` or `python3` via Bash against the paths. **Do not Read the JSON files** — they'll blow your context.
- Judge improvement strictly against `improvement_goal`. Do not invent a different goal. If the goal isn't moving but metrics unrelated to the goal look better, that's not an improvement.
- If `training_time_seconds` is unusually short (say < 180 s) and the results are inconclusive, prefer `is_result_improved: false` and say so in the commentary — shorter runs shouldn't ratchet progress.
- If the two summaries have **different `build_number`** values, the app code was rebuilt between runs. Flag this prominently in the commentary and lean toward `is_result_improved: false` unless the goal metric moved by a clearly-larger margin than plausible build-drift noise.
- Classify the run into **one of three buckets** (this is how the skill decides whether to accept, do nothing, or count against the failure streak):
  - **`improved`** — the goal metric(s) moved in the right direction by clearly more than plausible noise. Ratcheted progress. This iteration will be promoted to root and committed.
  - **`neutral`** — the run was stable and within plausible noise of the baseline: didn't clearly help, didn't regress, collapse signals are no worse than before. The baseline holds; no changes are ratcheted. A neutral is NOT a failure.
  - **`regressed`** — the goal metric(s) moved in the wrong direction, OR collapse signals got worse (pEnt fell below 5.0 when it previously held, top1_legal_fraction fell, vAbs saturated toward 1.0, gNorm exploded, etc.). This counts against the failure streak.
- Return JSON of exactly this shape and nothing else:
  ```json
  {
    "analysis_commentary": "<BRIEF, 2-3 sentences, <= 80 words; cite 1-3 specific metrics that drove the decision>",
    "classification": "improved"
  }
  ```
- **Keep `analysis_commentary` brief**: under 80 words, 2-3 sentences. Don't restate the proposal or the full trajectory. Just: what metric(s) moved, by how much, and which classification that justifies.
- **Always include a brief note on promotion activity**: state whether the trainer was promoted during the session (e.g., "promoted 1× at arena 2" or "no promotion"). The promotion count is in `new_summary.arenas.promoted` / `.count`; fetch specifics from the JSON path if needed. This is separate from the classification decision — a promotion isn't automatically an improvement, but the reader wants the fact visible at a glance.
- Do NOT return the legacy `is_result_improved` field; use `classification` only.

Save the subagent's JSON response to `<folder>/analysis.json`.

### 8. Accept / neutral / reject

Regardless of outcome, run `regen_dashboard.py` first so the HTML dashboard reflects the final status.

Read `classification` from `<folder>/analysis.json` and branch:

- **`improved` (ACCEPTED)**: copy `<folder>/parameters.json` → `$ROOT/parameters.json` and `<folder>/result.json` → `$ROOT/results.json`. Stage:
  - `$ROOT/parameters.json`, `$ROOT/results.json`
  - the whole test folder under `experiments/`
  - `$ROOT/experiment_results.js` (and `$ROOT/experiment_results.html` if newly created on this iteration)
  - any previously-uncommitted test folders and the current dashboard state from prior non-accepted iterations (those are legitimate evidence and ride along on the first subsequent accept)

  Commit with message:
  ```
  autotrain: accept <timestamp> — <first-line of change_details>
  ```
  Then `git push origin <branch-confirmed-in-step-0>`. (Per conversation authorization on 2026-04-21, autotrain is permitted to commit+push on accepted iterations to the user-confirmed branch — do not ask each time.)

- **`neutral` (NEUTRAL)**: do **not** touch root `parameters.json` or `results.json`. Do NOT commit this iteration (it'll ride along on the next accept). Log a brief line noting neutrality — the baseline held, the proposed change was safe but not an improvement. Crucially, a neutral does **not** count against the failure streak (see step 0.6).

- **`regressed` (REJECTED)**: do not touch root files. Leave the test folder in place — it's evidence of a regression and feeds the proposer's future history. Do not commit. Log a brief line. This counts against the failure streak.

- **`FAILED`** (from step 5's bounds-violation stub or step 6's training-failure stub): treated like REGRESSED for the streak counter; no commit. The stub `analysis.json` will carry a failure-shaped commentary, not a real `classification`.

Legacy back-compat: if `analysis.json` has the old `is_result_improved` field instead of `classification` (from pre-trichotomy iterations), treat `true` as `improved` and `false` as `regressed` — no neutrals in old data.

### 9. End iteration

Print a one-line summary: `autotrain <timestamp>: ACCEPTED|REJECTED — <change_details first line>`. That's it — `/loop` will re-invoke.

## Default parameters (for seeding)

```json
{
  "entropy_bonus": 2.5e-3,
  "grad_clip_max_norm": 25.0,
  "weight_decay": 5.0e-4,
  "K": 6.0,
  "learning_rate": 7.5e-5,
  "draw_penalty": 0.15,
  "self_play_start_tau": 1.2,
  "self_play_target_tau": 0.35,
  "self_play_tau_decay_per_ply": 0.045,
  "arena_start_tau": 0.95,
  "arena_target_tau": 0.22,
  "arena_tau_decay_per_ply": 0.038,
  "replay_ratio_target": 1.25,
  "replay_ratio_auto_adjust": true,
  "self_play_workers": 6,
  "training_step_delay_ms": 75,
  "training_batch_size": 2048,
  "replay_buffer_capacity": 500000,
  "replay_buffer_min_positions_before_training": 10000,
  "arena_promote_threshold": 0.58,
  "arena_games_per_tournament": 100,
  "arena_auto_interval_sec": 1200,
  "candidate_probe_interval_sec": 20
}
```

## Invariants

- Never modify tests.
- Never run `git reset`, `git stash`, or `git rebase`.
- Never force-push.
- Never skip commit hooks.
- The per-iteration time limit is hard-capped at 600 seconds (10 min). If a subagent requests a longer run, clamp it.
- Only push to the branch the user confirmed in step 0 of the current session. If branch changed mid-loop, re-confirm.
- If `git status` shows unexpected staged changes at iteration start, stop and surface them — don't sweep them into an autotrain commit.
