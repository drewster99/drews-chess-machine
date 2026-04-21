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

Timestamp = UTC `YYYYMMDD-HHMMSS`. Folder = `$ROOT/experiments/<timestamp>/`. Copy `$ROOT/results.json` to `<folder>/previous_result.json` **and** `$ROOT/parameters.json` to `<folder>/previous_parameters.json` (the dashboard's diff column uses the latter).

### 5. Propose a change (subagent)

Spawn a general-purpose subagent with this structured prompt. Do NOT delegate the decision of what "better" means — that's encoded in the goal.

Request JSON (pass inside the prompt as a fenced block):
```json
{
  "improvement_goal": "<contents of goal.txt>",
  "current_parameter_configuration": <contents of $ROOT/parameters.json>,
  "latest_results": <contents of $ROOT/results.json>
}
```

Response must be JSON of exactly this shape:
```json
{
  "change_details": "<one-paragraph rationale>",
  "parameters": { ... full parameters object with every key preserved ... }
}
```

Instruct the subagent: preserve all parameter keys exactly (never drop or rename a key), respect obvious physical bounds (e.g. `replay_ratio_target > 0`, integer workers, non-negative decay). Return only the JSON — no prose around it.

Save the raw JSON to `<folder>/proposal.json`. Write `change_details` to `<folder>/proposal.md`. Write the `parameters` object to `<folder>/parameters.json`.

Then run `regen_dashboard.py` so the dashboard shows this iteration as `IN_PROGRESS` while training runs.

### 6. Run training

Invoke `run_training.sh <folder>/parameters.json 600 <folder>/result.json <folder>/run.log`. Time cap is hard-capped at 600 seconds (10 min) regardless of anything else. If the script exits non-zero or `result.json` is missing/invalid, write a stub `analysis.json` with `{"is_result_improved": false, "analysis_commentary": "training run failed: <reason>"}`, run `regen_dashboard.py`, and jump to step 8 (reject).

### 7. Analyze (subagent)

Spawn a general-purpose subagent with this prompt:
```json
{
  "improvement_goal": "<contents of goal.txt>",
  "previous_results": <contents of previous_result.json>,
  "change_proposal": <contents of proposal.json>,
  "new_results": <contents of result.json>
}
```

Response must be JSON of exactly this shape:
```json
{
  "analysis_commentary": "<a few sentences explaining the comparison>",
  "is_result_improved": true
}
```

Save to `<folder>/analysis.json`.

### 8. Accept or reject

Regardless of outcome, run `regen_dashboard.py` first so the HTML dashboard reflects the final status.

- **Accepted** (`is_result_improved: true`): copy `<folder>/parameters.json` → `$ROOT/parameters.json` and `<folder>/result.json` → `$ROOT/results.json`. Stage:
  - `$ROOT/parameters.json`, `$ROOT/results.json`
  - the whole test folder under `experiments/`
  - `$ROOT/experiment_results.js` (and `$ROOT/experiment_results.html` if newly created on this iteration)
  - any previously-uncommitted test folders and the current dashboard state from prior rejected iterations (those are legitimate evidence and ride along on the first subsequent accept)
  
  Commit with message:
  ```
  autotrain: accept <timestamp> — <first-line of change_details>
  ```
  Then `git push origin <branch-confirmed-in-step-0>`. (Per conversation authorization on 2026-04-21, autotrain is permitted to commit+push on accepted iterations to the user-confirmed branch — do not ask each time.)

- **Rejected** (`is_result_improved: false`): do not touch root `parameters.json` or `results.json`. Leave the test folder in place — it's evidence. Leave the regenerated `experiment_results.js` uncommitted; it will be swept into the next accepted iteration's commit. Do not commit this iteration. Log a brief line to the user.

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
