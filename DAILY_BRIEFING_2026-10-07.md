# Daily briefing — 2026-10-07

Work from 2026-10-06 ~09:00 to 2026-10-07 ~06:55 (the 06:00 target slipped ~55 min: the relative cap was approved at ~04:00 and its review, fixes and final full suite needed the time). Everything below is merged on `main` and pushed unless it says otherwise. Nothing here needs a decision from you; the last section lists optional follow-ups.

## Headlines

- **Everything implemented is on `main` and the full test suite passes** on the final tree: **3,234 passed / 0 failed / 1 skipped** at b4845088 (scheme test plan, slow forensic suites on; no new compiler warnings); Python dashboard tests 233 OK; `bn_liveness.py --selftest` OK. The previous full run, on all three final-review fix merges before the relative cap, was 3,184 passed / 0 failed / 1 skipped (08937418).
- **Relative gradient-norm cap is implemented** (your "ok approved keep it going"): `cap = min(grad_clip_max_norm, max(floor, k × median of the last N pre-clip norms))`, defaults k = 3, N = 1000, warm-up W = 100 steps (hard max only before that), floor 0.5. It ships **log only** (computes and logs, changes nothing) until validation runs V-1 and V-3 pass — the plan's owner-decided gate. V-1 and V-3 are running now (below).
- **The step-20,600 B-silu blow-up was one hidden step.** Fixed caps 2.0 and 5.0, like 1.0, prevent it, and all three split from B-silu between trainer steps 19,751 and 19,799 — a step the 50-step log lines never showed, whose pre-clip norm was above 5 (≥ 15× the ≈ 0.33 logged median). A 3× trailing-median cap (≈ 1.0 there) would have acted on it. Write-up: E-0020 (caps 1.0 and the 15 control) and E-0023 (caps 2.0 and 5.0).
- **Three independent final reviews** (training, Lichess, Python) found 0 critical, 8 major, 27 minor issues across today's code; all fixed with regression tests (red before the fix, green after, unchanged). A fourth review of the relative cap found 3 major + 4 minor; all fixed.
- **All 17 real compiler warnings fixed.** 31 type-check-timing notes remain (decision D-7 below).

## What landed on `main`

| Area | What | Tests |
|---|---|---|
| Per-site activations (format v9/v10) | review fixes (shared activation list, null site keys named, version parity test) | full suite 2583/0/1 |
| Lichess bot: follow-lineage (plan P0–P6 + build-before-going-online) | the bot can follow a lineage's newest checkpoint; games in progress move to new generations; model is built before going online for every source | 2758/0/1 |
| Lichess bot: challenge log (P1–P7) | durable challenge JSONL (flock-protected append), game origin at start, back-fill from the protocol log (206 past games classified: 12 incoming / 132 matchmaking / 62 operator), UI, outcome fold | 2890/0/1 |
| Lichess bot: decline-reason key fix | Lichess sends lowercase keys (`nobot`, `timecontrol`); they were stored as `unrecognized` (39 + 19 yesterday). Old records are normalized on read, never rewritten | in the above |
| Lichess bot: record statistics (P0–P8, L1–L6, D1) | period buckets, time controls, models, self-assessment, endings, opponent strength; the cross-relaunch generation-credit bug (P0) | integration suite 3161/0/1 |
| Step-line cadence (format v11) | `training_step` is the overall trainer step from v11 on; step lines every 50 steps to 1k, then every `step_line_interval_sec` (180 s) plus each 1k mark; checkpoint names `<stem>-<tag>-step<trainerStep>`; older files read by their writer's meaning | in the integration suite |
| Hyperparameter recording (P1–P6, lineage schema 3) | build identity, every settings change journaled with its trainer step, configuration / run seeds / ancestry in every file's lineage record | in the integration suite |
| Training-health alarms (P1–P5) | 13 rules shared by GUI / corpus replay / train-vs-UCI; log-only by default; stop action → exit 35 (CLI) or training suspension (GUI); `--replay-health-log` replays saved logs | in the integration suite |
| Test log isolation | test runs no longer write into `~/Library/Logs/DrewsChessMachine` | 2616/0/1 |
| Final-review fixes: training (13), Lichess (13), Python (9) | see "Final review" below | 3173/0/1, 887 targeted, 233 Python |
| Compiler warnings | 17 fixed | 3161/0/1 |
| Relative gradient cap (P1–P4 + review fixes) | above; `[GRAD-CLIP]` lines, `gNormMax=` / `clips=` / `gCap=` on step lines, five parameters (Optimizer tab), per-step norm history saved with trainer state and restored by exact resumes | 3234/0/1 |
| Experiments | E-0019 (B-leaky), E-0020 (gradient cap 1.0 + cap-15 control), E-0021 (B-silu), E-0022 (B-leakyall), E-0023 (caps 2.0 / 5.0) | — |

## Final review (independent reviewers, read-only, then fixers)

- **Training (0 critical, 3 major, 10 minor):** Stop now quiets the ended run's health alarms and clears the suspended header (it kept beeping); the parameter-change journal now records an edit away from a held out-of-range value (it dropped it); `--help` text matched to the cadence (saves at trainer-step multiples of 1000, alarms every 50 trainer steps, enumerated names). Minor: a save during a health suspension no longer reopens the training segment; a new or escalated critical alarm ends a silence; `results.json` lineage at the cut's clock; parameter-change values keep their declared type; the GUI step line follows the trainer clock; train-vs-UCI executable hashing moved out of the cooperative thread pool; γ/β length check.
- **Lichess (0 critical, 2 major, 11 minor):** on a first run the challenge-log cutoff was nil, so the next launch would double-count outcomes and credits; the record-stats progression chart mixed different segments' step numbers into one series. Minor: parent folders now flushed on create; a game never switches to an older generation; a re-readable model file is re-read; one recompute per filed game; file-queue ordering (`nonisolated(nonsending)`); three slow SwiftUI views split into child views.
- **Python (0 critical, 3 major, 6 minor):** the replay tracker could misfile a resumed v11 segment's rows (probe back-fill, track, import-probes ignored the step basis); build identity ignored user-wide git excludes; input checks.
- **Relative cap (0 critical, 3 major, 4 minor):** a settings file with W > N could crash the app at start / popover Save and crash-loop the auto-resume — now refused everywhere it can enter (file load, session resume, GUI start, popover, CLI exit 2); a data race on the cap settings (now `SyncBox`); a test that proves the cap reaches the graph (mutation-checked: feeding the old placeholder makes it fail); step-line window after a promotion rewind; GUI first line re-reporting history; integration tests (CLI exact resume with history, refusal without `grad_norm_history`, promotion through the production capture/rewind path, matched-sampler resume, log-only bit-identical to off).

## Experiments

| trainer step | B (ReLU) | B-silu | cap 15 control | clip 1.0 | clip 2.0 | clip 5.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 19,000 | 1558.5 | 1571.9 | 1571.9 | 1571.9 | 1571.9 | 1571.9 |
| 20,000 | 1471.3 | 1390.2 | 1390.2 | 1442.5 | 1462.1 | 1436.9 |
| 21,000 | 1332.5 | 457.4 | 457.4 | 1386.6 | 1373.2 | 1353.6 |
| 22,000 | 1372.2 | 833.8 | 833.8 | 1412.8 | 1457.9 | 1407.1 |
| 23,000 | 1573.4 | 927.2 |  | 1567.8 | 1560.6 | 1567.3 |

- pElo; B-silu and the cap-15 control are identical through 22k (the control reproduced the blow-up exactly). Policy pre-BN parked channels at 21k–23k: B-silu 20–21, every capped arm 0.
- Written up in `experiments/summaries/E-0023` (new; E-0020 covers clip 1.0 and the cap-15 control) and the experiment README (d23671fa). Caveat recorded there: each capped arm left B-silu's path at 19,800, so at 20,600 they met different weights — the data shows the caps avoided the blow-up, not how a 2 or 5 cap would have handled step 20,600 itself.
- Largest logged gNorm: clip 1.0 0.462, clip 2.0 0.453, clip 5.0 0.475; B-silu 2.566 at 20,600.
- **Running now (relative-cap validation, plan Part V):**
  - **V-1** — exact resume of B-silu from 18k, log only, k = 1, floor 0.01, to 21k: measures the per-step distribution of norm / trailing median (every step above the median is logged) — the gate for k. Training math is B-silu's (log only feeds the hard max). Started 06:07 (pid 7506, log `dcm_log_20261007-060735.txt`, `[RESUME] EXACT`; build changed 2330 → 2390 with a matching behavior fingerprint), about 1.5 h.
  - **V-3** — fresh start on B's recipe and start net, cap on (k = 3, W = 100), 3,000 steps: checks the warm-up and the early relative cap leave early training intact (compare B's first 3,000 steps). Started 06:08 (pid 7689, log `dcm_log_20261007-060832.txt`; step 1 pre-clip norm 31.2, clipped by the hard max 15 during warm-up, as in B), about 1.5 h.
  - **Early checks passed:** V-1's probe at trainer step 19,000 is 1571.9 pElo — exactly B-silu's, so log-only mode leaves training unchanged on a real run. V-3's probe at 1,000 is 1018.2 — exactly B's: its only clips were steps 1–3 (pre-clip norms 31.2 / 30.3 / 26.5 > the hard max 15, as in B), so the warm-up and the early relative cap changed nothing early on. V-1 logged a would-clip on 69% of its first 380 post-warm-up steps at k = 1 (not ~50%), consistent with norms rising with the learning rate toward its peak; the write-up will check this.
  - When both pass, P5 flips the default to clip with V-1's k (plan V-7). Both run the frozen build `FrozenBuilds/DCM-2390-b4845088-relcap.app` (main with the cap); launch scripts `scratchpad/relcapV1_launch.sh` / `relcapV3_launch.sh`, parameter files `parameters-B-relcap-v1.json` / `-v3.json`; probes and `[GRAD-CLIP]` lines recorded as for the other arms. I'll write them up when they finish.

## Update 07:05 — V-1 found the hidden step

- V-1 (log only, so B-silu's exact path: its 20k probe is 1390.2, identical) logs every step above the trailing median. Through trainer step 20,000 (1,900 steps after warm-up, 1,650 above the median): the median ratio of a logged step is 1.07×, the 99th percentile 1.35×, and only **two steps exceed 2×: step 19,785 (pre-clip norm 8.45 = 28.1× the median 0.301) and step 19,795 (5.58 = 18.5×)**. These are the hidden steps that made clip 1.0 / 2.0 / 5.0 part from B-silu between 19,751 and 19,799 (8.45 and 5.58 both exceed 5; the 15 cap clipped neither).
- So a k = 3 relative cap would have clipped exactly those two steps up to 20k and nothing else: no false positives on this run. V-1 continues to 21,000 (through B-silu's 20,600 spike).

## Decisions I made (and why)

Decision IDs are spelled out each time.

- **D-1 Cadence plan OD-1 (what `training_step` means in old files):** no rewrite of files on disk; architecture format v11 marks the new meaning, older files are read by their writer (`creator`). Why: production-data rule; a version bump fails loudly in stale readers, a new key would be silently ignored.
- **D-2 Cadence OD-2..OD-15, OD-18 and the challenge-log / record-stats / hparam / relative-cap open decisions:** taken as the plans recommended after independent review. Why: you told me to decide and stop asking; each recommendation was reviewed.
- **D-3 Record-stats confidence interval (review R-1):** Wilson-style score interval instead of 1.96·sd/√n. Why: the latter shows ±0 for small, uniform samples.
- **D-4 Gradient-spike alarm threshold (5× the median of the prior 1,000 steps) kept** although it raises 247 / 173 times on the bf16-era v5 / qeu8 logs. Why: fp32 runs are silent; the bf16 spikes were real; a sustain requirement would hide B-silu's precursor.
- **D-5 Alarms validation runs V-2..V-5 and V-7 not run.** The permission check refused the command-line training launches as interfering with live training; I did not route around it. In-process tests cover their logic (`CorpusReplayHealthStopTests`: log-only raise, stop with `health-stop` + exit 35; `AutoTrainTerminationTests` for V-7's GUI stop path). The run script is ready if you want the real-corpus numbers: `scratchpad/alarms-validation/run_v.sh` (7 CLI runs, ~4,100 steps; needs a fresh frozen build).
- **D-6 GUI-only checks covered by in-process tests, no app launch** (alarms V-6/V-7, relative-cap V-5). Why: the focus rule (no window launches without permission).
- **D-7 31 type-check-timing warnings left.** They are not slow code from today: they move between builds and follow the per-file cost of expanding the `@TrainingParameter` macro. Removing them means restructuring `TrainingParameters` / `LichessBotController` (an architecture change, yours to make); raising the 100 ms thresholds would only hide them.
- **D-8 Overrode plan sequencing for time** (21:10): record-stats and challenge-log phases ran in parallel worktrees, merging main per phase, instead of waiting for follow-lineage; integration order main → cadence → hparam → alarms → challenge-log → record-stats, then one full suite.
- **D-9 Relative cap V-1 floor 0.01 (lowest allowed), not 0.5.** With k = 1 and a ≈ 0.33 median, a 0.5 floor would hide every step between the median and 0.5 — the very distribution V-1 measures.
- **D-10 Test edits** (all forced by approved design changes, under your "edit your tests and quit asking" rule; none deleted, no assertion weakened): cadence OD-17 set; `TrainingStepBasisTests` (pre-v11 unknown writer has no segment step); `TrainVsUciRefusalTests` (pre-flight digests); two `@MainActor` annotations; three Lichess call-list expectations gaining the new parent-folder flush; registry counts 86→98 (alarms) and 102→107 (relative cap); `TrainingHealthTestSupport` / `TrainingLiveStatsGatingTests` pass the new gradient-cap decision.
- **D-11 Training-fix agent's own choices accepted:** the journal recognizes a revert by key (no stored value, NaN-safe) instead of the review's stored-value fix; ended-run alarms stay listed but silent; train-vs-UCI executable hashing now runs in the synchronous pre-flight, so a missing engine exits 33 before an unknown `--preset` exits 2 (order reversed).
- **D-12 Lichess-fix agent's choices accepted:** a failed challenge-log load sets the cutoff to when the load began (nothing double-counted, history before it still shown); checkpoints without a cumulative step get their own "<run> segment <n>" series.
- **D-13 Relative-cap agent's choices accepted:** a restored history must end exactly at the clock (the plan's stricter rule); the arena's capture/rewind moved into `ChessTrainer` so the promotion test drives production code; history-less test overloads kept for old-form fixtures.
- **D-14 Clean-up:** removed every merged worktree (only the main checkout remains; branches kept) and every build folder; deleted only regenerable build output; thinned Time Machine local snapshots at the end (78 GB free); never touched `Sessions/`, `Models/` or `Logs/`. One new frozen build: `FrozenBuilds/DCM-2390-b4845088-relcap.app` (the V-runs').

## Blockers hit, and how they were resolved

- **drews-xcode-mcp built and tested the wrong project** (it resolves by workspace name; three same-named projects were open). Its `run_project_tests` ran another worktree's suite. Resolved: xcodebuild only, per-agent derived-data folders. The bot agent stalled ~5 h on it and was restarted with explicit instructions.
- **Disk full at ~22:05 (253 MB free).** Freed ~25 GB of regenerable build output (find -delete, not rm -rf), thinned the Time Machine local snapshot (your disk-cleanup playbook; df does not move without it), later `tmutil deletelocalsnapshots /` at your request.
- **Main-checkout xcodebuild hung** (Xcode project-load dialog, NSFileCoordinator wait). Resolved by `killall Xcode` (you allowed it) and killing one hung `xcodebuild -list`.
- **Integration failure:** cadence v11's lineage requirement refused the model-folder-cache test fixtures (8 failures); fixed (0240d049) before landing.
- **Usage limit at ~00:05 and again at ~05:50.** The interim briefing was written at the first; at the second the relative-cap agent stopped after committing its integration tests, which I ran and merged.
- **Permission classifier refused the alarms validation launches** — see D-5.

## Rule slips (disclosed by the agents)

- The alarms agent ran `git reset -q` once (index-only unstage; no data lost).
- The integrator ran `git checkout --ours` on the two generated build files, and `rm -rf` on a scratch path that did not exist (nothing deleted).
- The Lichess reviewer ran `rm -rf` on an empty test-result bundle it had just created in the scratchpad.
- My own: I first blamed clip 1.0's split from B-silu on GPU nondeterminism; the cap-15 control disproved it and the README was corrected.

## Worth knowing

- **Log lines hide single-step gradient spikes.** Step lines are sparse by design; the new `[GRAD-CLIP]` lines log every clip (and every would-be clip in log-only mode), and `gNormMax=` on each step line reports the largest pre-clip norm since the previous line.
- The `@TrainingParameter` macro cost is the main build-time hotspot (D-7).
- Lichess live checks (follow-lineage plan §6 1–12, challenge-log §6.5–6.7) need the bot online; they were not run (outward-facing). The next time you run the bot, `[LICHESS-BOT]` lines show the followed lineage and the challenge log fills `Challenges/` — no action needed beyond normal use.

## Optional follow-ups (no decision needed)

1. Real-corpus alarms validation V-2..V-5 (D-5), if you want measured numbers: say "run the alarm V-runs".
2. The type-check-timing restructuring (D-7), if build time matters.
3. When V-1 / V-3 finish: flip the relative-cap default to clip (plan P5) — the plan already decides this on a pass; I'll report the numbers.
