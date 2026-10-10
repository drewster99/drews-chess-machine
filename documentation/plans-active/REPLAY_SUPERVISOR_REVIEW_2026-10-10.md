# Review: scripts/replay_supervisor.py (2026-10-10; not yet adopted — owner decisions at the end)

Reviewed against `CLI/CorpusReplayRunner.swift`, `App/DrewsChessMachineApp.swift` (replay pre-flight), `App/CommandLineVersion.swift`, `Training/GPUFaultWatch.swift`, `Training/BatchHashChain.swift`, `CLI/CliTrainingRecorder.swift`, `scripts/compare_batch_hashes.py`, `scripts/dcm_lineage.py`.

## Verified correct
- Exit statuses 0 / 2 / 33 / 35 / 36; `termination_reason` `step_limit_reached` (also used for epoch limit and corpus end, so the target-file check in `verify_finished` is needed) and `gpu_fault`.
- Save order: rolling write, then the step file (same bytes); the step file is skipped when the rolling write fails. The final save at a step the autosave already wrote replaces this run's own step file. On a GPU fault there is no save after it.
- `--training-step-limit` counts segment steps; last reachable step = start + limit, so `limit = target - start` is right. A branch (no `--resume-exact`) starts at trainer step 0. The command-line limit overrides the parameters file.
- TrainerOutputFileGuard: step files in (start, start + limit] refuse the launch; a rolling file is accepted only when it holds the start model's `model_id` at its step; the out-model can't be the same file as the start model. The supervisor's quarantine, plus its copy-then-hard-link recreation (a new inode, not a link to `-latest`), meets all three rules.
- `--output` refuses an existing file: the supervisor uses a new results path for each attempt.
- `--version` exists, is the only argument allowed, and its format matches the regex.
- Naming port matches `EnumeratedCheckpointNaming` (first-occurrence marker, replace-all, rebuild check).
- Batch hashes: the hash covers CPU-side batch bytes, not weights. An exact replay resume restores the sampler stream. Saves land on 1,000-step window boundaries, so the resumed attempt's lines must equal the faulted attempt's at every common step, including steps after the fault. Expecting equality is right. A start that isn't on a boundary shows `partial` in both runs and is compared by hash only.
- `log show` ndjson: entries, then `{"count":N,"finished":1}`; timestamps carry their offset.
- Python 3.9.6 (/usr/bin/python3) and 3.14 both compile and pass the plan and classify tests.

## Changes
1. `log show --start/--end` in UTC with `+0000` (local wall time is ambiguous in the autumn DST hour). Entries are filtered by their exact timestamp inside the window, and the window starts at the child's start, not 1 s before (pid reuse). An unreadable timestamp makes the query count as failed.
2. `results.json` fault times that can't be parsed stop the supervisor. Before, the ValueError was uncaught.
3. A catch-all in `main` journals the traceback and notifies. Before, any OSError (EXDEV rename, EEXIST link, disk) killed the supervisor without notice.
4. Quarantine: checks that every source is on the quarantine folder's volume before moving anything (no partial EXDEV), writes the manifest before the moves, and uses `lexists` for the destination.
5. A live-trainer guard (`ps -axww`): refuses to launch, move or create files while any `--replay-corpus` / `--train-vs-uci` process names this out-model. The supervisor lock covers only supervisors. Readers (probe loops) don't count. Verified: it finds the live B-siluall trainer (pid 63702) on the fixture stem.
6. Binary check adds `git merge-base --is-ancestor b1e7ef35 <hash>`. The build counter is per machine and doesn't prove which source a build compiled. A stub at 4376aa0c is refused.
7. `--attach-pid`: checks that the process runs `--bin` with `--replay-corpus`, `--enumerate-checkpoints`, this out-model, the given results file and `--training-step-limit = target - start`. Before, a wrong pid or a pre-fault build (e.g. the running 2440) would have been planned against.
8. The start checkpoint header is read only for `--resume-exact`. A branch from a pre-v11 file was refused for nothing.
9. The rolling file written during an attempt must be in the start model's lineage run (the same check the step files already had).
10. The staging name is upper-case UUID (FileSafety's spelling, so the app's sweep recognizes crash debris); EEXIST on publish becomes a SupervisorError.
11. `--bin`, `--out-model` and `--run-dir` are made absolute. `ps` runs with `LC_ALL=C` for `lstart` parsing. The dry run reports live writers.

## Remaining risks
- Trust cutoff = file mtime < earliest fault − 10 s. If macOS logs a reset more than 10 s after the corruption, a bad save could be kept. Observed messages were ms apart. The app's barrier already guarantees saves precede every fault it saw.
- Exit 0 with faults for the pid stops for a human (conservative).
- A deterministic self-caused hang (our pid owns `…ErrorHang`) restarts until 3 attempts in a row keep no checkpoint (bounded, but up to ~3 × back-off of GPU time).
- Restarts from a checkpoint pass only `--start-model X --resume-exact`. A line that needed `--accept-inexact` for a gap that persists into its own checkpoints would get exit 2 and stop (not loop).
- A signal death (e.g. a manual `kill -9`) with GPU faults logged is restarted.
- `log show` over a very long attempt may be slow (900 s timeout → treated as unreadable → newest save distrusted).
- Simultaneous trainers fault together (21:37:09 was a hang by DCM pid 20985); each supervisor restarts on its own back-off.

## Owner decisions
1. Where quarantine lives (now `~/Library/Application Support/DrewsChessMachine/ReplaySupervisor/quarantine/`; refused when Models is on another volume), and whether quarantined files are ever cleaned (never automatically).
2. Probe loop / tracker: each restart changes the trainer PID (`probe_loop.sh` TRAINER_PID exits 3) and starts a new segment. Probes already taken of quarantined files, and the faulted attempt's log rows past the resume step, must be excluded from the CSV/registry. Today nothing does this.
3. `--notify-cmd` target (iMessage/push script) for stops.
4. Release-only binaries (a Debug build only warns now); `--allow-dirty-build` policy.
5. Whether exit 0 with logged faults should auto-accept a final save that predates the faults by the margin.
6. Margin (10 s) and caps (24 total, 3 without progress, 120 s back-off doubling to 30 min).
