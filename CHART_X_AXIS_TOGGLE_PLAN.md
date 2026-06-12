# Chart X-Axis Toggle Plan (time / steps / positions)

The main charts screen renders every series against wall-clock time. Training
dynamics are easier to reason about against trainer steps — and once batch
size is ever varied mid-run, *positions trained* becomes the only
batch-invariant axis. Plan: a three-way axis-mode toggle.

## Why three modes, not two

- **Time** — the right axis for rate/throughput diagnosis (e.g. the
  spDelay=3000ms discovery read directly off wall-clock gaps). Keep it.
- **Steps** — the natural axis for optimizer dynamics (LR changes, gNorm,
  loss curves) while batch size is constant.
- **Positions trained** — `Σ (batch size at each step)`. Identical to
  steps × batch while batch is fixed, but remains comparable across eras if
  batch is ever ramped (see 2026-06-12 discussion of batch-size schedules:
  a "step" is not a fixed unit of data; a position is). This is the axis to
  prefer for cross-run comparisons in ARCH_EXPERIMENTS.md going forward.

## Pre-requisite fix (latent bug, fix regardless of the toggle)

`UpperContentView.cumulativeStatusBar` computes
`totalPositions = totalSteps * trainingParams.trainingBatchSize` — i.e.
ALL historical steps multiplied by the CURRENT batch size. Correct while
batch has never changed mid-session; silently wrong the moment it does.
Fix: accumulate positions at the trainer (`completedTrainPositions +=
batchSize` per step), persist it in the session checkpoint alongside
`trainingSteps`, and read the accumulator everywhere `steps × batch` is
currently assumed (status bar, `lichessProbeHistory` tick capture,
`CliTrainingRecorder`). Old sessions decode with a backfill of
`trainingSteps × batchSize` (exact for every existing run, since none has
ever varied batch).

## Data model

Chart sample points must carry all three x-candidates. Most already carry
time + step (e.g. `OverallTickSample.timestamp` / `trainingStep`); add
`positionsTrained: Int?` to the per-sample structs that feed the main
charts and the probe trend charts, captured from the trainer accumulator at
record time. Optional so restored pre-feature history still decodes —
samples without it fall back to steps (exact backfill multiplication where
the era's batch size is known, else hidden in positions mode).

## UI

- One segmented control (Time | Steps | Positions) in the charts screen
  header, owned by `ChartCoordinator` so every tile on the screen switches
  together — per-tile axis modes would break cross-chart visual alignment.
- Persisted in `UserDefaults` (a display preference, not a training
  parameter — deliberately NOT a `@TrainingParameter`).
- `FastLineChart` x-label formatter switches with the mode (HH:mm for time,
  compact integer for steps/positions).
- Zoom windows (`ChartZoom`) are per-mode: a steps-domain window is
  meaningless after switching to time, so switching modes resets the zoom
  to full-fit rather than attempting a domain conversion.

## Batch-change annotations (only meaningful once batch varies)

If batch size ever changes mid-run, per-step metric levels shift
mechanically (gNorm ∝ 1/√batch among others) and would read as events.
When the recorded per-sample batch size changes between adjacent samples,
draw a thin vertical marker with the new batch value — same treatment as a
parameter-change annotation. (BATCH-STATS already logs batch per sample;
the chart rings need to carry it.)

## Out of scope

- The hover-step fixes for the Training-vs-Eval window and the Lichess
  Probe Detail OVERALL charts (notes A/B) — implemented separately on
  2026-06-12, ahead of this plan.
- Re-axising the Arena history / Elo charts (they are per-arena, not
  per-step; arena index remains their natural axis).

## Validation

1. Toggle each mode on a live session: all tiles switch together, x-labels
   reformat, zoom resets to full-fit.
2. Resume an old (pre-feature) session: positions mode renders via the
   steps×batch backfill with no decode failure.
3. The status-bar "Positions trained" cell matches the trainer accumulator
   after a simulated batch change (unit test on the accumulator + backfill
   logic).
4. All existing tests pass; new accumulator tests pass unmodified.
