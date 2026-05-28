# Stratified Replay-Buffer Sampling by Material Bucket

Status: **plan / not yet implemented**. Pending user approval before any code is written.

## Motivation

The 2026-05-28 champion (`20260525-1-sMe9-31`) buffer analysis showed the
4 active material buckets are not uniformly represented:

| Bucket | Non-pawn pieces | Positions | Share |
|---|---|---:|---:|
| 0–4  | deep endgame      | 307,866 | 31% |
| 5–8  | late endgame      | 340,933 | 34% |
| 9–14 | middlegame        | 191,559 | 19% |
| 15–22| full piece set    | 159,642 | 16% |

The trainer samples uniformly from the buffer, so its gradient is
**~2× weighted toward endgame positions vs full-piece-set positions**.

In the same analysis the policy entropy gap (vs uniform-over-legal) is
near-zero on the underweighted buckets:

| Bucket | meanEnt (nats) | uniformEnt | top-1 mass | value-scalar spread |
|---|---:|---:|---:|---|
| 0–4   | 2.19 | 2.42 | 27% | wide |
| 5–8   | 2.38 | 2.76 | 30% | wide |
| 9–14  | 2.95 | 3.20 | 18% | wide |
| 15–22 | 3.26 | 3.34 |  8% | collapsed near 0 |

The hypothesis is that part of the network's weakness on 9–14 and (especially)
15–22 is sample starvation: the loss function never gets enough gradient
from those phases to differentiate them from the prior.

The lighter-touch counter to "make self-play more exploratory" (raising
self-play tau toward 1.0) — which we just rejected because it would
worsen mate conversion and collapse the value head further — is to leave
self-play untouched and **rebalance the trainer's sampling distribution
across material phases**.

## Goal

Add an optional **per-bucket target distribution** to `ReplayBuffer.sample(…)`.
When enabled with target weights, each training minibatch contains
positions drawn approximately according to those weights across the 4
active material buckets, regardless of the buffer's natural skew.

Default behavior unchanged. Feature is opt-in via a new training parameter.

## Replay buffer on-disk format — UNCHANGED

This plan does **not** alter the `.dcmsession` / replay-buffer dump
layout. The `materialCount: UInt8` column per slot already exists in
the v6+ format (`ReplayBuffer.swift:1716`) and has been populated by
`BatchedSelfPlayDriver` since long before this plan. The new bucket
index is an **in-memory derived structure**, rebuilt by walking the
existing `materialCountStorage` column once during `restore(…)` (the
same place `resetCompositionAggregates()` runs).

Consequences:
- All existing `.dcmsession` files load identically. Recent saves
  resume without any migration shim.
- `ReplayBufferAnalyzer` continues to work exactly as today — it
  reads the same `materialCount` column.
- Rolling the feature back is purely a code deletion; no schema
  forward/backward concerns.

## Non-goals (V1)

- Cross-product stratification of (bucket × W/D/L). V1 ignores `maxDrawPercent`
  when bucket stratification is active. Document this clearly. Future work
  could combine them.
- Per-bucket length-tilt β. Same reason.
- Dynamic target weights (e.g. inversely proportional to network confidence
  per bucket). V1 uses static user-specified weights.
- Bucket stratification during arena, CLI, or `materialCount`-disabled paths.
  Trainer minibatch sampling only.

## Design

### Bucket boundary source of truth

Reuse `ReplayBufferAnalyzer.materialBuckets` (already declared as
`(low, high, label)` tuples) — do NOT duplicate the boundaries. Add an
internal `bucketIndex(forMaterialCount:)` helper on `ReplayBuffer` that
mirrors `ReplayBufferAnalyzer.materialBucketIndex(for:)` (clamps
out-of-range to the last bucket).

The 5th bucket (`23–30`) is unreachable in standard chess but must remain
defined to keep the indexing array-aligned with the analyzer.

### Slot-bucket index — maintained on insert/evict

Add to `ReplayBuffer`:

```swift
/// Per-bucket lists of currently-resident ring slot indices.
/// Outer index follows `ReplayBufferAnalyzer.materialBuckets`.
/// Maintained in lock-step with the per-slot `materialCountStorage`:
///   - On bulk append, every newly-resident slot is added to the
///     bucket determined by its incoming materialCount.
///   - On eviction (slot being overwritten while ring is full),
///     the slot is removed from whatever bucket holds it.
/// All operations are under `lock`. O(1) per slot via the
/// "swap-with-last + index map" idiom.
private var materialBucketSlots: [IndexedSlotSet]
```

`IndexedSlotSet` is a tiny internal struct:

```swift
/// Set-like container with O(1) insert/remove-by-value AND
/// O(1) random sampling. Backed by a contiguous `slots: [Int]`
/// array plus a `positionOf: [Int: Int]` dict mapping each
/// resident slot ID to its index in `slots`.
/// Random-pick: `slots[Int.random(in: 0..<slots.count)]`.
```

Memory cost: at full 1M-position buffer, 4 buckets × 8 bytes per Int per
slot × ~2× for the dict ≈ ~16 MB. Acceptable.

### Hooks

Two existing methods are the right hook points (both already run under
`lock.withLockUnchecked`):

- **Insertion**: extend the per-slot loop inside `append(...)` that
  already calls `incrementHashStat`/`incrementCompositionAggregates`.
  Add `materialBucketSlots[bucket].insert(slot)` after computing the
  bucket from the freshly written `materialCountStorage[slot]`.

- **Eviction**: extend `decrementCompositionAggregatesForSlot(_:)` to
  also call `materialBucketSlots[oldBucket].remove(slot)` BEFORE the
  slot's `materialCountStorage[slot]` is overwritten. (The existing
  `decrementCompositionAggregates…` already reads `outcome` and
  `gameLength` before the overwrite, so the ordering is established;
  bucket follows the same pattern.)

- **Restore from session**: `restore(...)` (called when reloading a
  `.dcmsession`) currently calls `resetCompositionAggregates()` then
  rebuilds. Add an analogous `resetAndRebuildMaterialBucketSlots()`.

### Sampling integration

`sample(count:…)` already has two paths:

1. **Fast path** (`constraints.isNoOp(forBatchSize:)`): bit-for-bit
   uniform with replacement.
2. **Constrained path**: rejection-sample with W/D/L tilt + K-cap +
   length-tilt.

Add a **third path**, taken when `constraints.materialBucketWeights != nil`:

```
// Per-bucket target counts (rounded, then clamped to resident count).
let targets: [Int] = computeBucketTargets(
    sampleCount: sampleCount,
    weights: constraints.materialBucketWeights!,
    residentPerBucket: materialBucketSlots.map { $0.count }
)
// Slack reallocation: if any bucket has residentCount < target,
// redistribute the deficit to the other (non-empty) buckets in
// proportion to their original weights. Same pattern the W/D/L
// stratifier uses for deficit reallocation.

// Per-bucket uniform-with-replacement draw.
var i = 0
for (bucket, target) in targets.enumerated() {
    let slots = materialBucketSlots[bucket].slots  // [Int]
    guard !slots.isEmpty else { continue }
    for _ in 0..<target {
        let srcIndex = slots[Int.random(in: 0..<slots.count)]
        emit(i, srcIndex)
        // …existing composition-tally bookkeeping…
        i += 1
    }
}
// Underfill guard (rare: all buckets but one empty AND total target
// after clamping < sampleCount) falls back to uniform fill for the
// remainder — same fallback the constrained path uses today.
```

Where the bucket-stratified path **diverges from the constrained path**
for V1: it does NOT enforce W/D/L tilt or K-cap. The
`SamplingResult.constraints` field still records what was configured so
the trainer's `[SAMPLER]` log line can note "bucket-stratified, draw% cap
ignored." If both are configured, log a one-line `[ALARM]` at session
start.

### Training parameter

Per the CLAUDE.md "@TrainingParameter checklist":

1. **Declare**:
   ```swift
   @TrainingParameter(
       defaultValue: false,
       category: .trainer,
       liveTunable: true,
       description: """
       Stratify training minibatches by game phase. \
       When on, each batch is drawn with roughly equal weight from \
       four game-phase buckets defined by NON-PAWN piece count: \
       0–4 (deep endgame), 5–8 (late endgame), 9–14 (middlegame), \
       15–22 (full piece set). This compensates for the replay \
       buffer's natural skew toward late-endgame positions. \
       The per-batch draw-percent cap and per-game K-cap do NOT \
       apply while this is on (V1 limitation).
       """
   )
   public enum ReplayBufferStratifyByMaterial: TrainingParameterKey {}
   ```
   The "by NON-PAWN piece count" qualifier appears verbatim in
   every user-facing surface (UserDefaults description, parameters.md
   doc, UI tooltip) to disambiguate "material" (which in chess parlance
   could also mean point value or material differential).
2. **Wire singleton**: add `replayBufferStratifyByMaterial: Bool` stored
   property + `collectValues` / `applyOne` / snapshot entry.
3. **Target weights**: rather than a 4-vector (annoying to expose in UI
   and persist), V1 uses a **single fixed balanced target [25, 25, 25, 25]**
   when the toggle is on. Document this and call out that custom weights
   are future work. Rationale: the 9–14 / 15–22 underweighting is the
   problem we're trying to fix; balanced is the natural first step.
4. **parameters.json save/load**: macro-generated; verify via
   `--show-default-parameters` round-trip.
5. **Session save/load**: add `replayBufferStratifyByMaterial: Bool?`
   to `SessionCheckpointState`, propagate through
   `buildCurrentSessionState`, add a `[RESUME-PARAM]` log block in
   `SessionController+Training.swift` matching the `batchStatsInterval`
   pattern.
6. **CliTrainingRecorder**: add the boolean to
   `BatchStatsSnapshot.samplingConstraints` so `results.json` records it.
7. **Runtime log**: add a `bucketMix=A/B/C/D` field to the `[BATCH-STATS]`
   line — the post-clamp achieved per-bucket count for the last batch.
   So a healthy run with stratification ON would show
   `bucketMix=32/32/32/32` instead of `32/32/16/14`.
8. **UI**: a checkbox in the trainer section of `TrainingSettingsPopover.swift`,
   tooltip pointing to the same one-line description.
9. **Live tunability**: per the checklist, since this is `liveTunable: true`
   the trainer needs to re-read `TrainingParameters.shared` on its
   periodic reconcile loop, NOT cache it in the per-session snapshot.
   Verify in `ChessTrainer.swift` that this path exists (it does for the
   other `liveTunable` params).

### UI — Replay tab of `TrainingSettingsPopover`

The Replay tab already has a **Sampling** section with a two-column
"buffer composition vs last batch composition" readout driven by
`bufferComposition: CompositionSnapshot?` and
`lastSamplingResult: SamplingResult?`. The bucket controls slot in
alongside that section, not as a new top-level section, so the visual
hierarchy stays consistent.

#### 1. Toggle row

Above the existing draw-cap / K-cap controls, add a single labelled
checkbox:

```
☐  Stratify training batches by game phase (non-pawn piece count)
    ⓘ  When on, each minibatch is drawn ~equally from 4 phase buckets
       (0–4 / 5–8 / 9–14 / 15–22 non-pawn pieces). Compensates for
       the buffer's natural skew toward endgame positions. Disables
       the draw cap and per-game cap below while active.
```

The `ⓘ` is a hover-over (`.help(_:)`) carrying the same text the
parameter description declares, so the source of truth stays in
`TrainingParameters.swift`.

Bound to `$trainingParams.replayBufferStratifyByMaterial`, so it
reflects/persists via the standard pipeline.

#### 2. Visual — bucket-distribution mini-chart

A small horizontal stacked bar (or 4 side-by-side bars — TBD during
implementation, mock both) immediately below the toggle, showing the
**most recent batch's bucket distribution**:

```
        0–4    5–8    9–14   15–22
buffer  [██████][███████][████][███]    31  34  19  16  %
batch   [█████ ][█████  ][████][████]   25  25  25  25  %       ← stratification ON
```

- **"buffer" row** = the buffer's current resident-set bucket
  distribution (pre-constraint, the natural skew). Sourced from the
  new per-bucket counts on `CompositionSnapshot`.
- **"batch" row** = the most recent `SamplingResult.achievedBucketCounts`,
  rendered as percentages.
- When stratification is OFF, the two rows should look ~identical
  (the batch is uniform from the buffer). This is itself a useful
  diagnostic — confirms the bucket-tracking is wired correctly.
- When stratification is ON with balanced target, dashed reference
  lines at 25% on each bucket column show "where the batch is aiming."
- A `bucketMix` numeric tuple (e.g. `32 / 31 / 33 / 32`) under each
  bar gives the precise per-batch count for users who prefer numbers.

Same hard-coded color per bucket across the UI (matches the existing
`[STATS]` and analyzer color usage if any — check during implementation).

#### 3. Grayed-out controls + explanation

When `replayBufferStratifyByMaterial == true`, the following existing
controls in the Sampling section become **visually disabled** (gray
text, no interactive editing, but still showing their last-saved value
so the user can see what would re-enable when they turn the toggle off):

- "Max draw % per batch" text field (`maxDrawPercentPerBatchText`).
- "Max positions per game" stepper (`maxPerGameInBatch` — confirm the
  exact binding name during implementation).
- Any length-tilt β control if it's also gated by V1's "stratification
  bypasses everything else" decision (it is — clarify in code).

An inline message appears in the same section, in `.secondary`
foreground style with an info icon, EXACTLY ONCE per section, NOT once
per disabled control:

> ⓘ  **Draw cap and per-game cap are disabled while phase
>     stratification is on.** V1 limitation — sampling is balanced
>     across phase buckets instead. Turn off phase stratification
>     above to re-enable these caps.

Implementation notes:
- Use `.disabled(trainingParams.replayBufferStratifyByMaterial)` on
  each control plus an `.opacity(…)` reduction to make the gray-out
  visually obvious (per the project's SwiftUI conventions, prefer
  manipulating opacity over conditional rendering — keeps view-type
  stability per `feedback_swiftui_view_stability`).
- The explanation banner is a normal `View` always present in the
  tree, with `opacity(stratifyOn ? 1 : 0)` and a fixed `frame(height:)`
  toggling between content-height when shown and 0 when hidden
  (project's view-stability rule: no `if`-gated visible content).
- A11y label on the disabled controls should announce "Disabled,
  phase stratification is on" so screen readers don't say a bare
  field label with no context.

#### 4. Behavioral edge cases for the UI

- **Toggle flipped during an active session**: the visual change
  (gray-out + banner) is instant; the actual sampling change takes
  effect on the next `[BATCH-STATS]` tick (live-tunable contract).
  The mini-chart's "batch" row will lag by 1 batch — acceptable; no
  explicit "applying…" indicator needed.
- **Empty buffer at session start**: `lastSamplingResult` is `nil`,
  so the "batch" row collapses to a placeholder ("no batches yet").
  Same pattern the existing W/D/L composition row uses today.
- **Toggle is on, but all 4 buckets aren't yet populated** (e.g.
  buffer has only burned-down endgame positions early in a fresh
  run): the mini-chart shows 0-height bars for empty buckets and a
  small text below: "Bucket 15–22 empty — slack reallocated."

### Persistence note for the UI controls

The toggle, like every other `@TrainingParameter`, persists to
`UserDefaults` via the macro-generated `didSet` (no `@AppStorage`
needed). The grayed-out controls' values are preserved across the
toggle, so flipping it on and off again restores the user's prior
draw cap / K-cap settings — `.disabled` blocks editing, it does not
clear values.

### Concurrency

`materialBucketSlots` is only mutated under `lock.withLockUnchecked`.
The trainer's `sample(…)` call already takes that lock. No new locks,
no actor changes.

## Test plan

Unit tests (XCTest, in `DrewsChessMachineTests`):

- `testMaterialBucketIndexBoundaries` — every boundary (0, 4, 5, 8, 9,
  14, 15, 22, 23) maps to the expected bucket.
- `testBucketIndexInsertionTracksMaterialCount` — push positions with
  controlled materialCounts, assert per-bucket sizes match.
- `testBucketIndexEvictionAfterRingWrap` — fill ring past capacity with
  positions of varying materialCount; assert evicted slots leave their
  buckets and incoming ones join the right bucket.
- `testBucketIndexRebuildOnRestore` — round-trip a buffer through
  `dump`/`restore` and assert `materialBucketSlots` matches what a
  fresh-from-scratch insertion path would produce.
- `testStratifiedSampleAchievesTargetDistribution` — fill buffer with a
  controlled bucket skew (e.g. 90/5/5/0), request a 256-position batch
  with balanced weights, assert achieved bucket counts are within 1 of
  64/64/64/0 (the 4th bucket is empty so its 64 gets reallocated).
- `testStratifiedSampleSkewedBufferUnderfillRedistributes` — buffer
  has only buckets 0 and 1 resident; request balanced 4-way batch;
  assert all 4 targets collapse onto the two resident buckets.
- `testFastPathStillUsedWhenStratificationOff` — toggle is off, the
  sampler returns `wasConstrainedPath = false` and bit-for-bit matches
  the existing fast-path implementation (regression guard).

Behavioral tests (manual, via the running app):

- Enable stratification mid-session; observe `[BATCH-STATS]` bucketMix
  swing from the natural buffer distribution to the target distribution
  within a few batches.
- Disable, observe it swing back.
- Save and resume the session with stratification on; `[RESUME-PARAM]`
  block reports the value.

## Validation (the "did it work" criterion)

Compare two 2-hour Play-and-Train runs from the **same starting checkpoint**:

- Run A: stratification OFF (control).
- Run B: stratification ON, balanced target.

For each run measure:

1. **Trainer-side policy entropy by material bucket** at the 2h boundary
   — via the existing ReplayBufferAnalyzer cron tick or a fresh
   `Analyze` run.
2. **Value-scalar percentile spread in the 15–22 bucket** — current is
   ±0.09 (collapsed). Hypothesis: it widens to at least ±0.15 in Run B.
3. **Arena promotion rate** — Run B's promotion rate should not be lower
   than Run A's by more than ~10%. (Stratification is a re-weighting,
   not a regression; if it craters promotion, abort and reconsider.)
4. **Policy CE loss curves split by bucket** — Run B's loss on
   15–22 minibatches should decrease faster than Run A's. (Requires a
   one-time addition to `[BATCH-STATS]` of per-bucket loss; defer to
   "nice-to-have" — V1 can validate from the analyzer cron alone.)

**Success criterion**: in Run B, the 15–22 bucket's `meanEntropyNats`
falls at least 0.10 nats below the run-A control AND `valueScalarPercentiles`
spread at least doubles. **Failure criterion**: arena promotion rate
falls by >25% OR value-head pD softmax rises above 0.80.

## Risks

- **Value-loss spike during transition.** The first minibatches after
  stratification turns on may show a temporary loss spike as the
  network sees an unfamiliar mix of 15–22 positions. Should be
  transient (≤ few minutes). If it isn't, the controller's
  `gNorm` would catch it.
- **Replay-ratio controller miscalibration.** Consumption rate stays
  the same, but per-batch information density changes. The controller
  observes cons/prod ratio, not per-batch loss, so it should be
  unaffected. Per the "controller is stable" memory, this is the
  expected place to NOT look first if a regression appears.
- **Underrepresented bucket = noisier gradient.** A balanced minibatch
  with 64 positions per bucket means only 64 examples for 15–22 per
  step. Combined with the value-target weakness on those positions,
  the per-bucket gradient may be noisier. If problematic, fall back
  to a less-aggressive weight (e.g. [20, 25, 30, 25]).

## Rollback

A single `replayBufferStratifyByMaterial = false` toggle disables the
feature. Per the live-tunable contract this takes effect within the
next reconcile tick (<1 s). No data migration needed; the bucket
index continues being maintained in the background so re-enabling is
instantaneous.

## Touchpoint checklist (for the implementing commit)

- [ ] `ReplayBuffer.swift` — add `IndexedSlotSet`, `materialBucketSlots`,
      insert/evict hooks, restore-rebuild, third sampling path.
- [ ] `ReplayBuffer.swift` — extend `SamplingConstraints` with
      `materialBucketWeights: [Float]?` (nil = off, [a,b,c,d,e] = target).
- [ ] `ReplayBuffer.swift` — extend `SamplingResult` with
      `achievedBucketCounts: [Int]` (5 entries; 5th always 0).
- [ ] `ReplayBuffer.swift` — extend `CompositionSnapshot` with
      `residentPerBucket: [Int]` so the UI's "buffer" row of the
      mini-chart has a source.
- [ ] `TrainingParameters.swift` — declare `ReplayBufferStratifyByMaterial`,
      add to `allKeys`, stored property, `collectValues`, `applyOne`,
      snapshot.
- [ ] `Persistence/SessionCheckpointFile.swift` — Optional
      `replayBufferStratifyByMaterial`.
- [ ] `SessionController+Checkpoint.swift` — propagate in
      `buildCurrentSessionState`.
- [ ] `SessionController+Training.swift` — `[RESUME-PARAM]` block.
- [ ] `ChessTrainer.swift` — re-read on reconcile loop (live-tunable
      contract).
- [ ] `ChessTrainer.swift` — `[BATCH-STATS]` `bucketMix=` field.
- [ ] `CliTrainingRecorder` — surface in
      `BatchStatsSnapshot.samplingConstraints`.
- [ ] `TrainingSettingsPopover.swift` + `TrainingSettingsPopoverModel.swift`
      — Replay tab additions: toggle row, bucket mini-chart, grayed-out
      draw-cap / K-cap with single explanation banner, `.help(_:)`
      tooltips sourced from the parameter description.
- [ ] New Replay-tab subview: `ReplaySamplingBucketChartView` (own
      file per project's "one View per file" convention), rendering
      the buffer-vs-batch bucket distribution mini-chart from the
      `SamplingResult.achievedBucketCounts` + `CompositionSnapshot.residentPerBucket`
      pair. Use opacity/frame-height swapping (not conditional
      rendering) for the toggle-on/off visibility, per the project
      view-stability rule.
- [ ] `ROADMAP.md` — mark this plan as **completed** after merge (per
      CLAUDE.md: completed items stay with full detail).
- [ ] Tests above, all passing.
- [ ] Single build at the end (per project convention).

## Decisions

1. **Balanced target [25,25,25,25] vs custom weights?**
   → V1: balanced (decided 2026-05-28). Custom weights are future work
   in the same plan if balanced proves over- or under-corrective.
2. **Toggle scope — trainer-only or also Analyze/CLI?**
   → V1: trainer-minibatch only. Analyzer continues surveying the
   buffer as-is so its output remains comparable across this change.
3. **Parameter key name (`replayBufferStratifyByMaterial`)**
   → Approved (decided 2026-05-28). User-visible surfaces must spell
   out "by NON-PAWN piece count" since "material" is ambiguous.
4. **ROADMAP entry**
   → Approved (decided 2026-05-28). Add as an in-flight item under a
   suitable existing section (likely "Training" or "Sampling"), with
   a one-paragraph summary pointing to this plan file. After merge,
   move it to **Completed** with full detail per the CLAUDE.md rule.
