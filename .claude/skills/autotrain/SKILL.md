---
name: autotrain
description: Run one iteration of the automated chess-training parameter-tuning loop. Propose a parameter change, run a time-limited training run, analyze results vs. a stated improvement goal, and accept (commit+push) or reject the change. Intended to be driven by `/loop /autotrain`. Use when the user says "autotrain", "tune the chess params automatically", or similar.
---

# autotrain — one iteration of the chess-training parameter-tuning loop

This skill runs **one** iteration of a propose → run → analyze → accept/reject loop for tuning the training hyperparameters of the chess engine in this repo. Invoke via `/loop /autotrain` to run continuously.

Repo root (referred to below as `$ROOT`): `/Users/andrew/cursor/drews-chess-machine`.

The app is launched via `$ROOT/run_latest.sh` (a thin wrapper that picks the most recently built Debug or Release `.app` under DerivedData and execs it with any extra args).

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
  Thin wrapper around `$ROOT/run_latest.sh` that passes the right CLI flags, waits for exit, and verifies the output file. Flag names are configurable via env vars `DCM_PARAMS_FLAG`, `DCM_TIME_FLAG`, `DCM_OUT_FLAG` if they ever change.
- `$ROOT/.claude/skills/autotrain/regen_dashboard.py`
  Scans `experiments/*/` and rewrites `$ROOT/experiment_results.js` (and creates `$ROOT/experiment_results.html` the first time). The HTML page auto-polls the `.js` sidecar every 15 s via a cache-busted `<script>` reinjection — new rows are appended in place without reloading, so scroll position is preserved. Invoke as `python3 $ROOT/.claude/skills/autotrain/regen_dashboard.py`.
- `$ROOT/.claude/skills/autotrain/summarize_results.py <path-to-results.json>`
  Emits a compact (~1-5 KB) JSON digest of a results.json — arena scoreboards, per-metric trajectories (first/min/median/mean/max/final), collapse signals, candidate-probe first/mid/last, build-number stamp. Used to keep subagent prompts small; the raw file is ~1 MB / ~300k tokens and you should never paste it into a subagent directly.
- `$ROOT/.claude/skills/autotrain/validate_params.py <path-to-parameters.json>`
  Sanity-bound check on a proposed parameters.json. Bounds are intentionally very wide — the point is to catch proposer hallucinations (`learning_rate: 5.0`, fractional worker counts, capacity > 1e8), not to gatekeep tuning. Exits 0 on valid, 1 on violations (printed to stderr). Invoked by the skill after step 5's proposal lands; a violation rejects the iteration before any training run happens.
- `$ROOT/.claude/skills/autotrain/apply_code_proposal.py <folder>`
  CODE-CHANGE mode only (step 5b). Reads `<folder>/code_proposal.json`, validates touched files against an allowlist (`ChessTrainer.swift`, `ContentView.swift`, `MPSChessPlayer.swift`, `ReplayBuffer.swift`), overwrites them with the proposer's content, runs `xcrun xcodebuild` to verify, and reverts the working tree if the build fails. Writes `code_apply_status.json` and `build.log` next to the proposal. Exit codes: 0 = applied + built; 2 = schema/forbidden-file (tree unchanged); 3 = build failed (tree reverted); 4 = io error (manual intervention). Files explicitly forbidden from overwrite: `ChessNetwork.swift`, `ChessMPSNetwork.swift`, `BoardEncoder.swift`, `PolicyEncoding.swift`, `ChessGameEngine.swift`, `MoveGenerator.swift`, `ChessMove.swift`, `ReplayRatioController.swift`, anything under `DrewsChessMachineTests/`.
  ***** NOTE ON ABOVE - THIS SHOULD USE xcode-mcp-server INSTEAD OF XCODEBUILD 
  ***** Running xcode build will by default create a local .build folder. this 
  ***** is problematic but the runner scripts look in ~/Library/Developer/Xcode/DerivedData
  ***** for the Debug and Release executables and never the local .build folder.
  ***** Thus, a code change could be made but not really actually tested until the app
  ***** is rebuilt with Xcode in the more normal way.
  ***** TODO: Update this item above to clarify the build process and express the importance
  ***** of using xcode-mcp-server rather than xcodebuidl (xcode-mcp-server instructs Xcode
  ***** to do the build).

## Iteration procedure

Work through these steps in order. Do each step; don't skip.

### 0. Confirm branch with user

Run `git rev-parse --abbrev-ref HEAD` to read the current branch.

- **If the branch is literally `experiments`**, proceed without asking — that branch is the designated autotrain scratch branch, confirmation would be pure friction. Log a one-liner like `autotrain: branch=experiments, proceeding without prompt` and continue.
- **On ANY other branch** (including `main`), show the branch name and ask: "autotrain will commit and push accepted iterations to **<branch>** — proceed?" Wait for confirmation before continuing. If the user says switch branches, let them do that and re-invoke.

Once the user has confirmed a non-`experiments` branch once in a given session, you may skip this prompt for subsequent `/loop` iterations **in that same session**, but resume asking if the branch ever changes under you.

### 0.5. Bail if DrewsChessMachine is already running

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

- **`failure_streak ≥ 25`** → enter **REPLICATE mode**. The proposer has been failing to find improvements for 25 iterations; rather than ask it to try again, re-run the current-best parameters verbatim to probe whether the "best" result is actually reproducible. Set a local flag `replicate_mode = true` that step 4 and step 5 observe. Rationale: if the baseline reproduces, the proposer is the problem (analyzer confirms rejections are real); if the baseline does *not* reproduce, the "current best" was noise-lucky and we may get a free accept from a decent replicate. There is no HALT — keep cycling through replicates and normal iterations indefinitely; if `trailing_replicates ≥ 3`, force the next iteration into **normal mode** (skip replicate even if `failure_streak ≥ 25`) so the proposer gets another shot.
- **`failure_streak` 15–24** → proceed normally but note it in the iteration's summary line (`... (streak=22, replicate at 2)`).
- **`failure_streak` 9–14** → proceed normally but note it (`... (streak=10, watch)`).
- **`failure_streak` < 9** → proceed silently.

### 0.7. Code-iteration cadence (every 200 normal iterations)

`regen_dashboard.py` exposes `window.AGGREGATES.code_iteration_due` and `iterations_since_codechange`. The cadence counts only `mode=normal` iterations since the last `-codechange` folder (replicates and seeds are transparent), so the proposer's tuning history isn't artificially shortened by replicate cascades.

If `code_iteration_due` is `true` AND `failure_streak < 9`, route this iteration into **CODE-CHANGE mode** instead of normal. Set a local flag `code_change_mode = true` that step 4 and step 5b observe. Folder suffix becomes `-codechange`.

**Why the `failure_streak < 9` gate:** code surgery on top of unstable parameter dynamics is a recipe for chaos — one variable at a time. If we're in a watch-zone streak, defer the code-change iteration until the next normal accept resets the streak; the cadence counter keeps ticking, so we'll just take the code-change shot a few iterations later.

If we route into code-change mode, also REQUIRE that `git status --porcelain` shows no unstaged changes to the files in the code-proposal allowlist (`ChessTrainer.swift`, `ContentView.swift`). If the user is mid-edit on those files, defer code-change mode by one iteration (run a normal proposal instead) and log the deferral. The cadence counter doesn't advance from a deferred attempt — it'll trigger again next iteration.

### 1. Check working tree is clean enough to commit

Run `git status --porcelain`. If there are staged changes unrelated to autotrain, stop and tell the user — we will commit on an accepted improvement and don't want to sweep unrelated work into the commit. Unstaged build-counter / BuildInfo.swift drift (from the pre-Compile script phase) is expected and fine to leave alone — don't stage them.

### 2. Load or create the improvement goal

If `$ROOT/experiments/goal.txt` doesn't exist, ask the user: "What's the improvement goal for this autotrain run? (e.g., 'prevent policy collapse — keep pEnt well above 5.0 after 15 minutes of training from scratch')." Save the answer to the file. Read the goal from the file.

### 3. Seed if needed

A seed run is needed whenever `$ROOT/results.json` is missing (either first-ever run, or results got deleted). Handle `parameters.json` carefully:

  a. **If `$ROOT/parameters.json` does not exist**, run `"$ROOT/run_latest.sh" --show-default-parameters > "$ROOT/parameters.json"` to seed it from the app's canonical defaults. (`--show-default-parameters` is sub-second, never opens the GUI, and writes a flat snake_case JSON object to stdout matching the format the autotrain proposer + `validate_params.py` expect.) **If `parameters.json` already exists, leave it alone** — the user may have hand-tuned values they don't want clobbered with defaults.
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
- **Code-change mode** (set in step 0.7): Folder = `$ROOT/experiments/<timestamp>-codechange/`. The suffix is how the dashboard and the `iterations_since_codechange` counter identify code-change iterations. Step 5b owns this folder.

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
- Write `<folder>/training_time.txt` with `600` (match the baseline's training time; a short replicate wouldn't be informative, and going *longer* than baseline would confound reproducibility signal with extra training).
- Run `validate_params.py` as a sanity check (should trivially pass since the baseline already validates). If it somehow fails, that's a real problem — stub-reject and halt the replicate cascade for user attention.
- Run `regen_dashboard.py`.
- Skip the rest of step 5 and go straight to step 6.

### 5b. Code-change mode (if `code_change_mode = true` from step 0.7)

This is a one-shot detour that runs IN PLACE OF the parameter-proposer subagent. It produces a Swift code change instead of a parameter delta. After it runs (whether the change applied or not), we proceed to step 6 (training) and step 7 (analyze) as usual — analysis judges the resulting training run against `improvement_goal` exactly like a normal iteration.

**Setup** (matches step 5's normal-mode preamble):

  a. `current_best_summary` = stdout of `summarize_results.py $ROOT/results.json`.
  b. `recent_history` = last 10 iterations, same shape as step 5.
  c. Copy `$ROOT/parameters.json` → `<folder>/parameters.json` verbatim. Code-change iterations don't change params.
  d. Read the current full content of every file in the allowlist (`DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift`, `DrewsChessMachine/DrewsChessMachine/ContentView.swift`, `DrewsChessMachine/DrewsChessMachine/MPSChessPlayer.swift`, `DrewsChessMachine/DrewsChessMachine/ReplayBuffer.swift`) and assemble `current_files`: `{ <relpath>: <full content as string> }`.

**Spawn a code-proposer subagent** with this prompt:
```json
{
  "improvement_goal": "<contents of goal.txt>",
  "current_best_parameters": <contents of $ROOT/parameters.json>,
  "current_best_results_summary": <current_best_summary>,
  "current_best_results_json_path": "<absolute path to $ROOT/results.json>",
  "recent_history": <recent_history>,
  "current_files": <current_files>,
  "allowed_files": [
    "DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift",
    "DrewsChessMachine/DrewsChessMachine/ContentView.swift"
  ],
  "forbidden_files": [
    "DrewsChessMachine/DrewsChessMachine/ChessNetwork.swift",
    "DrewsChessMachine/DrewsChessMachine/BoardEncoder.swift",
    "DrewsChessMachine/DrewsChessMachine/PolicyEncoding.swift",
    "DrewsChessMachine/DrewsChessMachineTests/"
  ]
}
```

Instructions embedded in the prompt:
- This is a **CODE change**, not a parameter change. Touch Swift code in the allowlist files only. Do NOT propose a parameter change in the same iteration.
- Do not change the network architecture, board encoding, or policy encoding (the forbidden files). Do not modify tests. Do not change file paths or module structure.
- Prefer **small, targeted changes** with a clear mechanism: tweak a constant, change a threshold, adjust a smoothing factor, fix a subtle bug, change a heuristic. Multi-line surgery within one function is fine. Avoid sweeping refactors.
- The change must be **orthogonal to the parameters in `current_best_parameters`** — if the same effect is achievable by tuning an existing parameter, don't bake it into code.
- Architecture is immutable: same network shape, same input encoding, same policy head.
- Return JSON of exactly this shape (no markdown, no surrounding prose):
  ```json
  {
    "change_details": "<≤60 words rationale, like a commit subject>",
    "rationale": "<longer mechanism explanation, ≤200 words>",
    "files": {
      "<relpath>": "<FULL new file content as a single string — every line, including imports and unchanged code>"
    }
  }
  ```
- The `files` map is **full-file replacement** — emit the complete new contents of every file you want to change. Do not emit a unified diff. The applier writes your strings verbatim to disk and runs the build, so any syntax error you emit will trip the build gate.
- A code-change run is judged the same way a parameter run is: the resulting training run's metrics, against `improvement_goal`. So your change should plausibly affect training-run behavior, not just refactor for style.

**After** the subagent returns:
1. Parse the JSON. If parsing fails or required keys (`change_details`, `files`) are missing, retry once. If retry fails, write a stub `analysis.json` with `{"classification": "regressed", "analysis_commentary": "code proposer returned invalid JSON twice — skipping iteration"}`, run `regen_dashboard.py`, and jump to step 8 (reject).
2. Write `<folder>/code_proposal.json` with the full subagent JSON.
3. Write `<folder>/proposal.json` with `{"change_details": "<from code_proposal>", "parameters": <baseline params>, "mode": "codechange"}` so the dashboard and history-builder treat this row uniformly.
4. Write `<folder>/proposal.md` with the `change_details` text.
5. Write `<folder>/training_time.txt` with `900` (default).
6. **Apply and verify**: run `python3 $ROOT/.claude/skills/autotrain/apply_code_proposal.py <folder>`. Read `<folder>/code_apply_status.json`:
   - Exit 0 / verdict `applied`: build succeeded, working tree now contains the proposed change. Continue to step 6 (training).
   - Exit 2 / verdict `schema_error` or `forbidden`: stub-reject — write `analysis.json` with `{"classification": "regressed", "analysis_commentary": "code proposal rejected by allowlist/schema: <detail>"}`, run `regen_dashboard.py`, jump to step 8.
   - Exit 3 / verdict `build_failed`: stub-reject — write `analysis.json` with `{"classification": "regressed", "analysis_commentary": "code proposal failed build (tree reverted): <last lines>"}`, run `regen_dashboard.py`, jump to step 8.
   - Exit 4 / verdict `io_error`: HALT. Print `autotrain: code-apply io error, manual intervention required` and end the iteration. Do NOT continue to training.
7. Run `regen_dashboard.py`. The dashboard row will be marked `IN_PROGRESS` with mode=codechange.

Then proceed to step 6 (run training) — the existing pre-build hook on the Xcode project will pick up the modified Swift files when `run_training.sh` launches the app, and `result.json`'s `build_number` will reflect the new build.

**Step 8 handling for code-change iterations** is in step 8's "Code-change mode" subsection below.

**Normal mode** (from here on):

**Before spawning** the subagent, assemble the context inputs. **Never paste the raw `results.json`** into a subagent prompt — it's ~1 MB / ~300k tokens. Use the summarizer.

  a. `current_best_summary` = stdout of `python3 $ROOT/.claude/skills/autotrain/summarize_results.py $ROOT/results.json`.
  b. `recent_history` = up to the **10 most recent** subfolders of `$ROOT/experiments/`, newest last. For each entry include:
       - `timestamp` — folder name.
       - `status` — `ACCEPTED` / `REJECTED` / `SEED` / `FAILED`.
       - `goal` — contents of `<folder>/goal.txt` if present, else `null`. Distinguishes iterations judged under an older goal from ones judged under the current goal.
       - `change_details` — the proposer's original rationale (from `proposal.json`). Continuity of prior reasoning.
       - `changed_params_diff` — the params-vs-previous diff.
       - `analysis_commentary` — the analyzer's rebuttal (from `analysis.json`).
       - `training_elapsed_seconds` — from `<folder>/result.json` under the same-named key (actual wall-clock time the run lasted, written by the app). Null only for older runs that completed before this field was introduced; in that case fall back to `<folder>/training_time.txt` and then to the iteration's `proposal.json["training_time_seconds"]` budget.
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
  "current_best_parameters": <contents of $ROOT/parameters.json>,
  "current_best_results_summary": <current_best_summary>,
  "current_best_results_json_path": "<absolute path to $ROOT/results.json>",
  "recent_history": <recent_history>,
  "training_time_seconds_max": null,
  "training_time_seconds_default": 900,
  "exploration_mode": <boolean>
}
```

Instructions embedded in the prompt:
- **`current_best_parameters` is the baseline you are trying to beat.** It is the cumulative result of every previously-accepted iteration — each `ACCEPTED` entry in `recent_history` contributed to it, `REJECTED` / `FAILED` / `NEUTRAL` entries did not. `current_best_results_summary` is the training result that was recorded when this baseline was last ratcheted. Your proposal will be judged as an improvement over this baseline, not over the most recent iteration's result.
- The summary is a digest. For detail not in it, run `jq` or `python3 -c "..."` via Bash against the path. **Do not use the Read tool on the JSON** — it's ~1 MB.
- Example: `jq '.stats | map(.policy_entropy) | [.[0], min, max, .[-1]]' <path>`.
- **If `exploration_mode` is true**: propose a **bolder or orthogonal change** than recent history — change a parameter you haven't touched recently, try a larger magnitude, or explore an axis the goal hasn't been examined against. Still respect physical bounds and the goal.
- **If `exploration_mode` is false**: propose an incremental, low-risk change aimed directly at the goal.
- Return JSON of exactly this shape and **nothing else** (no markdown, no prose, no leading comment):
  ```json
  {
    "change_details": "<BRIEF rationale, 1-2 sentences, <= 60 words>",
    "parameters": { ... full parameters object, every key from input preserved exactly ... },
    "training_time_seconds": <integer ≥ 60, no upper bound, OPTIONAL>,
    "training_time_rationale": "<BRIEF, one short sentence, <= 20 words, OPTIONAL>"
  }
  ```
- **Keep `change_details` brief**: 1-2 sentences, under 60 words. Don't re-explain the overall strategy or restate prior history — the reader has all of it. Just state the change and its expected mechanism.
- Preserve every key in the input parameters object. Never drop or rename a key.
- Respect physical bounds: `replay_ratio_target > 0`, integer worker counts, non-negative decay values, positive batch sizes, etc. A separate validator enforces wide bounds on the server side — stay well within them.
- **Parameter coupling & budget constraints** (the validator will reject proposals that violate these; stay well inside them):
  - `lr_warmup_steps` must be **≤ ⅓ of `training_steps`** from the latest run's summary (see `derived_budget.recommended_lr_warmup_max`). Above that the lr ramp never finishes in a 10-minute window, so the configured `learning_rate` is never actually exercised and the result looks like stalled learning even when the parameters are otherwise fine. The validator hard-caps at 50% of `training_steps`.
  - `replay_buffer_min_positions_before_training` eats wall-clock before any SGD step — larger values delay the first probe and reduce the number of training steps that fit in the window. Don't raise it unless you have a specific reason related to replay diversity.
  - `{training_batch_size, learning_rate, weight_decay}` are coupled through SGD noise and update magnitude. Scaling batch requires scaling lr in the same direction (linear for SGD, √-batch for Adam); `weight_decay` per-epoch also couples via the number of update steps per epoch. Don't change batch alone. The repo has `sqrt_batch_scaling_lr` which the app can apply automatically — keep that flag on unless you've thought hard about why not.
- **There is no upper cap on `training_time_seconds`.** The user's standing instruction is **10-hour sessions** (`training_time_seconds: 36000`). Default to 36000 for every proposal unless you have a specific short-range reason (e.g. testing a parameter that historically collapses fast — then 5400–7200s is acceptable). The autotrain monitoring loop is built to ride alongside long sessions: a 5-minute cron polls `[STATS]` health and only intervenes if a hard-reject criterion has been firmly tripped for ≥2 consecutive ticks. Short windows starve the analyzer of late-trajectory data. Include a brief `training_time_rationale` whenever you go below 36000s.

**After** the subagent returns:
1. Parse the JSON. If parsing fails or required keys (`change_details`, `parameters`) are missing, retry once with a terser reminder of the schema. If the retry also fails, write a stub `analysis.json` with `{"is_result_improved": false, "analysis_commentary": "proposer returned invalid JSON twice — skipping iteration"}`, run `regen_dashboard.py`, and jump to step 8 (reject).
2. If `training_time_seconds` is present, clamp to a minimum of 60 (no upper bound). If absent, use **36000** (10 hours, the standing default).
3. Save the full raw JSON response to `<folder>/proposal.json`.
4. Write `change_details` to `<folder>/proposal.md`.
5. Write the `parameters` object to `<folder>/parameters.json`.
6. Write the chosen (post-clamp) training time to `<folder>/training_time.txt` as a plain integer on one line.
7. **Sanity-bound validation**: run `python3 $ROOT/.claude/skills/autotrain/validate_params.py <folder>/parameters.json --latest-result $ROOT/results.json`. If it exits non-zero, the proposal has an out-of-bounds value **or** a budget-coupling violation (e.g. `lr_warmup_steps` larger than half the latest run's `training_steps`). Capture the violation message, write a stub `analysis.json` with `{"is_result_improved": false, "analysis_commentary": "proposal failed bounds check: <violations>"}`, run `regen_dashboard.py`, and jump to step 8 (reject). This iteration counts toward the failure streak (by design — if the proposer keeps hallucinating bad values, the loop should halt).

Then run `regen_dashboard.py` so the dashboard shows this iteration as `IN_PROGRESS` while training runs.

### 6. Run training

Read the training time from `<folder>/training_time.txt` (fall back to 600 if the file is missing, e.g. during the seed path; the standing default for actual iterations is 36000 = 10 hours).

Invoke `run_training.sh <folder>/parameters.json <training_time> <folder>/result.json <folder>/run.log`. There is **no upper cap** on training time — the wrapper passes whatever value you give it through to the app.

#### 6a. In-flight monitoring (every wakeup while the run is in flight)

While the training run is in flight, every cron / wakeup tick must do **both** of these — `pgrep` alone is insufficient. The 5850s run on 2026-05-01 wasted ~35 minutes of GPU because polling only checked liveness, not health.

1. **Health check.** Read the most recent `[STATS]` line from the active session log:
   ```bash
   ls -t ~/Library/Logs/DrewsChessMachine/dcm_log_*.txt | head -1 | xargs grep "[STATS] elapsed=" | tail -1
   ```
   Parse and report all six health metrics in one line: `pEnt`, `gNorm`, `pLogitAbsMax`, `legalMass`, `top1Legal`, `vAbs`. Don't skim — these are the same fields the analyzer keys off in step 7, so you should be able to predict the eventual classification from them.

   **Positive-health bands** — when reporting metrics, label each as `in-band` / `watch` / `out-of-band` so trends are visible across ticks rather than a vague "healthy". For a network training from random init through self-play (no MCTS, no opening book, no human data).

   **NOTE on `pEnt` after the legal-mask change (commit acc5340 + threshold recal 2f95f21):** `pEnt` in `[STATS]` is now computed over the **post-mask softmax — legal moves only**. Theoretical max is `log(legal_count)`, which averages ~3.4 nats (ln(30) ≈ 3.4) over a typical chess game; uniform-random init lands near that ceiling. The in-repo alarm floor is now **1.0** (≈ 2.7 effective legal moves), and the critical floor is **0.5** (≈ 1.6 effective). The old "5.0 floor" derived from the unmasked log(4864) ≈ 8.49 ceiling is OBSOLETE — do not use it.

   | Metric | In-band | Watch | Out-of-band |
   |---|---|---|---|
   | `pEnt` (post-mask, legal-only) | 1.5–3.4 (ceiling ≈ ln(legal_count)) | 1.0–1.5 trending down | < 1.0 (alarm floor; < 0.5 = critical) |
   | `gNorm` median | 20–60 | 60–100 | sustained > 100, OR trajectory monotone-rising over ≥3 ticks |
   | `pLogitAbsMax` | 5–25 | 25–40 | > 40 (>50 = severe blowup, hard kill) |
   | `legalMass` | rising over time, ≥0.05 by 30min, ≥0.20 by 2h | flat at 0.002–0.005 past 30min | < 0.002 with `top1Legal=0` past 60min |
   | `top1Legal` | rising, ≥0.05 by 1h | 0 still at 1h–2h | 0 throughout 2h+ run AND legalMass < 0.005 |
   | `vAbs` | 0.15–0.50 | 0.50–0.85 | > 0.85 (tanh saturated) OR < 0.05 throughout |

   Also surface (when present in the line):
   - **diversity** — `unique=200/200(100%)` and `diverge ≥ 1.5` is in-band; <100 unique or diverge=1.0 means self-play exploration is collapsing.
   - **avgLen** — 200–500 ply in-band; <100 means games end too quickly to learn from, >800 means networks are shuffling.
   - **ratio cur vs target** — within ±0.10 of target is in-band; persistent low ratio means trainer is starving.

   Across ticks, the **shape of progress** matters more than any single value: pEnt should drift down monotonically without cliffs, gNorm should trend down (not up) as the loss landscape smooths, legalMass should rise.

2. **Early-kill if any hard-reject signal trips mid-run.** Compute these from the `[STATS]` line; kill the run with `kill -SIGUSR1 <pid>` (NOT bare `kill` — bare SIGTERM races the dispatch handler; SIGUSR1 routes through `EarlyStopCoordinator` and writes a complete `result.json` with `termination_reason: "SIGUSR1-user-requested"` before exit). The user's instruction is **explicit** on this — never use bare `kill`. Hard-reject criteria (any one ⇒ kill):

   - `pLogitAbsMax > 50` — softmax has crystallized; nothing recoverable.
   - `pEnt < 1.0` (post-mask alarm floor) AND elapsed > 60 min — long past warmup, legal-only entropy is below the alarm.
   - `legalMass < 0.005` AND `top1Legal == 0` AND elapsed > 60 min — equivalent of the Forward Pass demo showing 100% mass on illegals.
   - `gNorm > 300` for two consecutive checks — gradient norm is in runaway, not transient.

   **Be conservative on long sessions.** The user has explicitly directed: "DO NOT terminate any session unless you are CERTAIN it is unrecoverable." For a 10-hour session, a slow legal-mass climb at 90 min is not grounds to kill — let the run breathe. Only fire if a criterion is unambiguously tripped AND has been tripped for at least 2 consecutive 5-minute ticks. Killing early saves GPU per bad iteration; killing too early throws away potentially-useful trajectory data.

   These mirror H2/H3/H4/H6 in the analyzer's hard-reject criteria. The iteration will still go through the analyzer normally and be classified `regressed` from the partial `result.json`.

   **Before killing, take a UI screenshot.** Call `mcp__xcode-mcp-server__take_app_screenshot` with `app_name: "Drew's Chess Machine"` (display name has spaces and an apostrophe; the binary `DrewsChessMachine` substring does NOT match — the matcher uses the AppKit display name, not the executable name). Save the returned screenshot path into the iteration folder for context — copy it to `<folder>/pre_kill_screenshot.png` so the post-mortem analyzer has it. If the screenshot call fails (app already exited, accessibility denied, etc.), log the error and proceed with the kill anyway — the screenshot is a nice-to-have, not a precondition.

   Then log a one-line summary to the conversation: `autotrain: early-kill on H<N> at <elapsed>s — <metric>=<value>`. Then `kill -SIGUSR1 <pid>`. Then proceed to step 7 with the partial result.

   **Manual kills outside the H1-H7 flow** (user asks "kill this run", a launcher bug like the osascript-activation failure, or any other case where you SIGUSR1 without going on to step 7's real analyzer subagent) MUST still leave the iteration in a non-IN_PROGRESS state on the dashboard. The dashboard keys IN_PROGRESS off "no `analysis.json` present" — leaving the file absent means the row sticks at IN_PROGRESS forever. After the SIGUSR1 lands and the wrapper exits, write a stub `<folder>/analysis.json` with `{"classification": "regressed", "analysis_commentary": "<one-sentence reason for the manual kill, e.g. 'user-directed kill so the new build could be tested' or 'launcher osascript activation failed; not a parameter result'>"}` and run `regen_dashboard.py`. Mark it as REJECTED so it's distinct from real analyzer rejections in the commentary text.

3. **Report briefly.** Even when nothing is wrong, surface the six metrics on every wakeup so trends are visible across ticks. Don't reduce these to a vague "healthy" — the user reads the line.

#### 6b. Exit-code handling

Exit-code handling:
- **`0`** — clean training run, ran to the timer. `result.json` will have `"termination_reason": "timer_expired"`. Continue to step 7.
- **`10`** — skip iteration cleanly (GPU busy / lock conflict). Not a failure; just log and end. Handled in step 0.5.
- **`11`** — early bail on legal-mass collapse. The app wrote a valid `result.json` with `"termination_reason": "legal_mass_collapse"` before exiting. Treat as a successfully-completed-but-collapsed run: continue to step 7 normally (do NOT write a failure stub). The analyzer will surface the termination reason and classify as `regressed`.
- **Any other non-zero exit, OR `result.json` missing/invalid** — real failure. Write a stub `analysis.json` with `{"classification": "regressed", "analysis_commentary": "training run failed: <reason>"}`, run `regen_dashboard.py`, and jump to step 8 (reject).

### 7. Analyze (subagent)

**Before spawning**, summarize both results files. **Never paste raw `result.json` / `previous_result.json` into the prompt** (same size concern as step 5):

  a. `previous_summary` = stdout of `python3 $ROOT/.claude/skills/autotrain/summarize_results.py <folder>/previous_result.json`.
  b. `new_summary` = stdout of `python3 $ROOT/.claude/skills/autotrain/summarize_results.py <folder>/result.json`.

Spawn a general-purpose subagent with this prompt:
```json
{
  "improvement_goal": "<contents of goal.txt>",
  "training_elapsed_seconds": <contents of <folder>/result.json's `training_elapsed_seconds` field as an integer; null if the run pre-dates the field>,
  "change_proposal": <contents of proposal.json>,
  "previous_results_summary": <previous_summary>,
  "previous_results_json_path": "<absolute path to <folder>/previous_result.json>",
  "new_results_summary": <new_summary>,
  "new_results_json_path": "<absolute path to <folder>/result.json>"
}
```

Instructions embedded in the prompt:
- The summaries are digests of ~1 MB JSON files. For specific details not in the summaries, use `jq` or `python3` via Bash against the paths. **Do not Read the JSON files** — they'll blow your context.
- **`termination_reason`** is in `new_summary.termination_reason` (and `previous_summary.termination_reason`). Two values exist:
  - `"timer_expired"` — the training run ran to the requested time limit.
  - `"legal_mass_collapse"` — the app early-bailed because `illegal_mass_sum` stayed at ≥ `legal_mass_collapse_threshold` for `legal_mass_collapse_no_improvement_probes` consecutive 60 s probes (after a `legal_mass_collapse_grace_seconds` post-training-start grace period — currently 180 s). This is a definitive collapse — the run did NOT complete the requested window. **Always classify the new run as `regressed` when `termination_reason == "legal_mass_collapse"`**, and lead the commentary with the bail (e.g. "early-bail at <elapsed>s on legal-mass collapse"). Never call this `improved` or `neutral` no matter what other metrics look like — the run did not run long enough to be a fair comparison.
- Judge improvement strictly against `improvement_goal`. Do not invent a different goal. If the goal isn't moving but metrics unrelated to the goal look better, that's not an improvement.
- If `training_elapsed_seconds` is unusually short (say < 180 s) and the results are inconclusive, prefer `is_result_improved: false` and say so in the commentary — shorter runs shouldn't ratchet progress.
- If the two summaries have **different `build_number`** values, the app code was rebuilt between runs. Flag this prominently in the commentary and lean toward `is_result_improved: false` unless the goal metric moved by a clearly-larger margin than plausible build-drift noise.
- Classify the run into **one of three buckets** (this is how the skill decides whether to accept, do nothing, or count against the failure streak):
  - **`improved`** — the goal metric(s) moved in the right direction by clearly more than plausible noise. Ratcheted progress. This iteration will be promoted to root and committed.
  - **`neutral`** — the run was stable and within plausible noise of the baseline: didn't clearly help, didn't regress, no hard-reject criteria tripped. The baseline holds; no changes are ratcheted. A neutral is NOT a failure.
  - **`regressed`** — any hard-reject criterion below is tripped, OR ≥2 soft-reject criteria are tripped, OR the goal metric(s) moved in the wrong direction by clearly more than plausible noise. This counts against the failure streak.

**HARD-REJECT CRITERIA (any single one ⇒ classify `regressed`):**

Read these directly off `new_summary.collapse_signals` and `new_summary.arenas`. Each exposes a bool you key off; you don't need to recompute from the trajectory.

  H1. `termination_reason == "legal_mass_collapse"` — early-bail.
  H2. `pEnt_final_below_5 == true` **AND** `final_top1_legal_fraction < 0.01` — final policy entropy below the 5.0 alarm floor (≈ exp(5)=148 effective moves vs ~30–40 legal moves) **paired with no real legal-move preference forming** (top-1 legal at <1% mass = barely above uniform 1/4864 ≈ 0.0002). Concentration alone isn't collapse — concentration *onto illegal moves* is. If pEnt<5 but the network has a legitimate legal-move signal (top-1 legal ≥ 1%, ≥50× uniform), the narrowing is "starting to learn" rather than "collapsing", and the run shouldn't be hard-rejected on this rule. The pure "pEnt<5" signal still triggers a soft-reject (S1) when paired with another mid-run concern.
  H3. `policy_logit_severe_blowup == true` (i.e. `max_policy_logit_abs_max > 50`) — softmax has become a delta function. One logit dwarfs the rest by ≥e^50 ratio. Gradients through other classes are dead.
  H4. `late_probe_collapsed == true` — every candidate probe in the last ~5 minutes shows `max_prob ≥ 0.99` AND `legal_mass_sum < 0.01`. This is the equivalent of the UI Forward Pass demo showing 100% of mass on illegal moves.
  H5. `value_head_saturated == true` — `final_value_abs_mean ≥ 0.95`. The tanh has saturated, value gradients vanish.
  H6. `grad_norm_ever_exceeded_200 == true` AND `final_grad_global_norm > 1.5 × baseline.final_grad_global_norm` — sustained gradient explosion, not a transient spike. Compare against `previous_summary.collapse_signals.final_grad_global_norm`.
  H7. `policy_loss_extreme_negative == true` (`final_policy_loss < -10`) AND `arenas.promoted == 0` AND baseline had `arenas.promoted ≥ 1` — large negative pLoss with no actual chess-strength gain means the network is exploiting the entropy bonus / advantage scaling rather than learning legal play.

**SOFT-REJECT CRITERIA (≥2 must trip ⇒ classify `regressed`):**

  S1. `pEnt_ever_below_5 == true` (touched the floor mid-run, even if recovered) AND baseline had `pEnt_ever_below_5 == false`.
  S2. `policy_logit_blowup == true` (`max_policy_logit_abs_max > 30`) AND baseline had it `false` OR baseline `max_policy_logit_abs_max < 0.7 × new max`.
  S3. `legal_mass_below_pct1_at_end == true` AND baseline had `final_legal_mass > 0.05` — running away from legal moves rather than toward them.
  S4. `arenas.promoted == 0` AND baseline had `arenas.promoted ≥ 1` AND `training_elapsed_seconds ≥ 0.9 × baseline.total_training_seconds` — backsliding on the only "actual chess strength" signal at equal or longer elapsed time.
  S5. `grad_norm_ever_exceeded_100 == true` AND baseline `max_grad_global_norm < 100` — first time gradient norm crossed 100 on this baseline family.

**POSITIVE-HEALTH BANDS (the shape of a good run):**

A healthy run isn't defined by a single metric — it's a coherent multi-axis trajectory where four feedback loops (distribution shape, optimization stability, value sanity, self-play quality) are all moving the same way. Use these bands when reading `new_summary` to assess whether a clean run is meaningfully better than baseline, not just non-collapsing.

  *A. Distribution-shape axis (policy head learning the legal-move manifold)*
  - `min_policy_entropy` 5.0–7.5 (final 30% of run); monotone-decreasing trajectory without cliffs.
  - `final_legal_mass` ≥ 0.10 (strong: ≥ 0.30); rising vs `first` value in trajectory.
  - `final_top1_legal_fraction` ≥ 0.05 (strong: ≥ 0.10).
  - `max_policy_logit_abs_max` in 5–25 (decisive but not blown out).
  - `late_probe_collapsed == false` AND last candidate probe `max_prob` in 0.05–0.50 with `legal_mass_sum > 0.10`.

  *B. Optimization-stability axis*
  - `max_grad_global_norm` < 100 (strong: < 60); `final_grad_global_norm < first_grad_global_norm` (trajectory trends down as loss landscape smooths).
  - `final_policy_loss` in [-3, +1] band (large negatives without arena promotions = reward hacking, not learning).
  - `final_value_loss` ≤ baseline AND in 0.10–0.45 band.

  *C. Value-head sanity axis*
  - `final_value_abs_mean` 0.15–0.50 (saturated > 0.85; not learning < 0.05).
  - `final_value_mean` near 0 (chess is roughly drawn at the average position).

  *D. Self-play quality axis*
  - `diversity_unique_percent == 100` AND `diversity_avg_divergence_ply ≥ 1.5`.
  - `avg_game_length_final` 200–500 ply.
  - Replay ratio `ratio_current.final` within 0.10 of target.

  *E. Learning-validation axis (the only "is it stronger?" signal)*
  - `arenas.promoted ≥ 1` per ~30–45 min of training.
  - Arena scores trending in 0.50–0.55 band (a promotion is imminent) is acceptable; stuck at exactly 0.50 with > 90% draws across many rounds means learning has plateaued.

**IMPROVEMENT REQUIRES (all):**
  - `termination_reason == "timer_expired"`.
  - All hard-reject criteria clear.
  - At most 1 soft-reject criterion tripped.
  - **At least 2 positive signals fire from different axes** (e.g. one from A *and* one from E). Each signal:
    - A1. `final_legal_mass > baseline.final_legal_mass × 1.2`
    - A2. `final_top1_legal_fraction > baseline.final_top1_legal_fraction + 0.02`
    - A3. `min_policy_entropy` is in the 5.5–7.5 in-band AND ≥ 0.95 × baseline (didn't lose entropy headroom)
    - B1. `max_grad_global_norm < 0.7 × baseline.max_grad_global_norm` (clearly more stable training)
    - B2. `policy_logit_blowup == false` AND baseline had it `true` (recovered from blowup regime)
    - E1. `arenas.promoted > baseline.arenas.promoted`
    - E2. `arenas.mean_score > baseline.arenas.mean_score + 0.01` AND `arenas.count ≥ baseline.arenas.count`
  - `min_policy_entropy >= 0.85 × baseline.min_policy_entropy` (entropy-headroom guardrail; this is a separate floor, not one of the positive signals).
  - Trajectory metrics — pEnt min, gNorm max, pLogitAbsMax max — within 2× of baseline (no axis silently dragged out-of-band even if a positive signal fires elsewhere).

If only 0–1 positive signals fire, classify `neutral`, not `improved` — even a clean run that adds no progress shouldn't ratchet the baseline. Coincidental single-axis wins (e.g. one accidental arena promotion while gNorm exploded) are noise; requiring two-axis alignment guards against that.

When citing metrics in `analysis_commentary`, prefer the explicit collapse-signal / arena field names (e.g. "policy_logit_severe_blowup tripped: max=65", "arenas.promoted=2 vs baseline.promoted=1") over reconstructed numbers, so the analyzer's reasoning is auditable.
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

#### Code-change mode (step 5b iterations only)

The folder is `<timestamp>-codechange`. The working tree currently contains the proposer's Swift change (apply_code_proposal.py applied it before training). What happens at step 8 depends on classification, but with one critical extra step:

- **`improved` (ACCEPTED)**: same as normal accept — copy `result.json` → root (params didn't change so root params.json stays as-is, but copy it through anyway as a no-op for symmetry). The Swift files were modified in place during step 5b, so they show up as ` M` in `git status`. Stage them ALONGSIDE the experiment folder, alongside `parameters.json`, `results.json`, `experiment_results.js`, and any uncommitted prior folders. Commit message: `autotrain: accept <timestamp>-codechange — <change_details first line>`. Push.
- **`neutral` (NEUTRAL)**: **REVERT the Swift changes** (`git checkout HEAD -- <each allowlist file>`) so the working tree returns to the baseline build. Don't commit. The folder remains as evidence (with its `code_proposal.json` and `code_patch/` directory preserving the proposal). Log a brief line; the cadence counter resets next iteration's count to 0 since this row's `mode=codechange`.
- **`regressed` (REJECTED)**: **REVERT the Swift changes** the same way. Don't commit. The folder remains as evidence. Counts against the failure streak.
- **stub-reject from step 5b** (build failed, schema violation, or proposer JSON failure): the apply script already reverted the tree on build_failed; on schema_error nothing was written. Either way the tree is clean. Don't commit; folder remains.

The revert is **non-negotiable** for non-accept code-change iterations — leaving a partially-broken Swift change in the tree contaminates every subsequent normal-mode iteration's training run.

### 9. End iteration

Print a one-line summary: `autotrain <timestamp>: ACCEPTED|REJECTED — <change_details first line>`. That's it — `/loop` will re-invoke.

## Default parameters

The canonical defaults live in code, not in this file. To produce a fresh `parameters.json` with current defaults, run:

```sh
"$ROOT/run_latest.sh" --show-default-parameters > "$ROOT/parameters.json"
```

`--show-default-parameters` is sub-second and never opens the GUI. Per-parameter descriptions go to stderr; redirect with `2>` if you want them. The companion flag `--create-parameters-file [path]` writes both `parameters.json` (defaults) and `parameters.md` (descriptions, grouped by category) at `path` (default `./parameters.json`); refuses to overwrite an existing `parameters.json` unless `--force` is also passed.

The source-of-truth for the parameter schema is `DrewsChessMachine/DrewsChessMachine/TrainingParameters.swift` (the registry + per-key `@TrainingParameter` declarations).

## Analysis discipline (long-session-aware)

Long sessions (10 h default) produce ~50–200 MB session logs and large `result.json` files. **Prefer Python over eyeballing.**

- For `[STATS]` trajectories: `python3 -c "..."` to parse the session log and compute per-tick deltas, rolling means, monotonicity, and inflection points. Do this on every 5-minute cron tick rather than scrolling tails.
- For `result.json` deep-dives beyond what `summarize_results.py` produces: write a one-off Python script in `/tmp/` (e.g. `/tmp/analyze_<timestamp>.py`), run it, paste its (small) summary back. Never `Read` the full ~1 MB result.json.
- For `.dcmsession` autosaves (under `~/Library/Application Support/DrewsChessMachine/Sessions/`): each save bundles weights + config + checkpoint metadata. The autotrain skill does NOT need to load the weights to make decisions, but if the user asks for cross-session comparison, parse `metadata.json` inside each `.dcmsession` directory with Python — it carries the ModelID lineage, arena history, and the `[STATS]`-equivalent counters at save time. Sessions are saved every 4 h while Play-and-Train is active, plus after every arena promotion. Multiple saves from the same run let you reconstruct trajectory across the entire session even if the run later crashes.
- For arena promotion progression: grep `[ARENA]` and `[CHECKPOINT]` lines from the session log and pipe through Python — count promotions per hour, score progression, kept-vs-promoted candidate IDs.

Use these patterns **proactively** on each cron tick, not just at end-of-iteration. The user has explicitly authorized "TAKE ALL THE TIME you need to ANALYZE" — favor a deeper Python-driven look over a quick eyeball.

## Invariants

- Never modify tests.
- Never run `git reset`, `git stash`, or `git rebase`.
- Never force-push.
- Never skip commit hooks.
- **Standing default training time is 36000 seconds (10 hours).** There is no upper cap. Shorter windows are exceptions, not the rule.
- Early-kill of an in-flight run uses `kill -SIGUSR1 <pid>` ONLY. Never bare `kill`. SIGUSR1 routes through `EarlyStopCoordinator` and writes a complete `result.json` before exit.
- Only push to the branch the user confirmed in step 0 of the current session. If branch changed mid-loop, re-confirm.
- If `git status` shows unexpected staged changes at iteration start, stop and surface them — don't sweep them into an autotrain commit.
