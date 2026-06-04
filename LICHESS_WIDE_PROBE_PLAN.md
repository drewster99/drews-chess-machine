# Lichess wide-probe parallel set — as built

**Status: implemented** (feature commit `9818b66`, probe-timing follow-up same day,
branch `bf16-trainer`). This doc describes what shipped; it diverged from the
original plan in two ways, both noted inline below.

**Goal.** Run a large, fixed **WIDE** Lichess probe battery — ~4,435 puzzles
spanning rating 400–3200 (flat per-100 density 550–2800, mate-weighted) — as a
long-term longitudinal yardstick, *alongside* the existing 200-puzzle set. The
200-set is left byte-for-byte untouched (same data, chart, persistence); the
wide set is purely additive.

## What shipped

### Single watcher, one batched forward, split two ways
**(Divergence #1 from the original plan, which proposed a second watcher + a
second inference network.)** Both batteries are driven by the *same*
`LichessProbeWatcher` tick:

1. One trainer weight snapshot into the existing `lichessProbeInferenceNetwork`
   (candidate target) — no second inference network.
2. The 200 + wide probes are concatenated (~4,635 positions) and evaluated in a
   **single** `evaluateBatched(count: 4635)` via the new
   `TacticalProbeRunner.runBatch` — one batched forward, not a per-probe serial
   loop and not two separate batches.
3. The per-position result rows are split back by the primary count: first 200 →
   `lichessProbeHistory`, remainder → `lichessProbeWideHistory`.

Benefits of folding both into one batch: both sets are probed against
**identical weights** (directly comparable), only one weight snapshot per tick,
no second inference network, and a **fixed batch size** so the network's
per-batch-size graph/buffers compile + allocate **once** and are reused every
tick (no per-tick reallocation). The combined board tensor is **pre-encoded
once and cached** (probe boards are static — only weights change), so there's no
per-tick input allocation either.

Numerically equivalent to the old serial path: inference-mode batch-norm makes
each position in a batch independent, so a batched forward yields the same
per-position logits as N serial forwards (within float noise). The 200-set's
trajectory therefore stays continuous across the switch.

### Cadence
**(Divergence #2: the original plan leaned toward a sparse wide cadence to bound
cost.)** Because batching made the combined tick cheap, the wide set runs at the
**same 25-step cadence** as the 200-set — both batteries share the one tick, so
they're inherently step-aligned.

### Themes / categories
The wide set carries 13 themes (the 200-set's 8 plus `mateIn2`,
`discoveredAttack`, `deflection`, `sacrifice`, `promotion`). Five new
`ProbeCategory` cases were added and `themeToCategory` maps all 13, so the loader
never hits its unknown-theme `preconditionFailure`. Aggregation, the OVERALL
fold, and the charts are all category-agnostic (keyed dynamically), so no other
per-category code was needed.

### Start-of-training auto-export
~2 minutes after the trainer's step count starts advancing, both sets
auto-export a snapshot to `…/Performance/LichessProbes/` tagged
`training-start-set200` / `training-start-wide`, with the success window
suppressed (`announce: false`) so nothing pops while unattended. Driven by
`SessionController.scheduleStartOfTrainingProbeExport()` (fires once per launch);
`LichessProbeExporter.exportLatest` gained `tag:` + `announce:` (manual button
unchanged via defaults). The exports are valid "Compare…" snapshots.

### Persistence + UI
- New Optional `lichessProbeWideHistory` on `SessionCheckpointState`
  (back-compatible — older `.dcmsession` files decode it nil and start the wide
  monitor empty), saved via `withProbeHistories` and restored under
  `[RESUME-PROBE]`.
- A second OVERALL trend chart for the wide set shows beneath the 200-set chart
  in the Lichess Probe Detail window (`LichessProbeDetailView` gained an optional
  `wideHistory`). The per-puzzle *table* remains 200-set-only for now.

### Probe-cost telemetry
Each tick logs `[TACTICAL-LICHESS] timing n=<count> encodeMs=… snapshotMs=…
gpuMs=… postMs=… recordMs=… totalMs=…`:
- **encode** — one-time combined board encode (≈0 after the first tick, cached).
- **snapshot** — trainer weight export + load (candidate target only).
- **gpu** — the single batched forward + readback copy.
- **post** — CPU fold: softmax + legal-mask + verdicts over all positions.
- **record** — aggregate + history append + per-set summary log.
- **total** — the whole tick.

## "Don't lose the 200" guarantees (all held)
- 200 watcher/history/chart/persistence and `largeSet` unchanged; the watcher
  gained an optional wide battery, defaulted off.
- Metadata is a union (additive); the 200's entries never overwritten.
- The new session field is Optional → old saves load fine.
- Batched eval is numerically equivalent → no chart discontinuity.

## Files touched
`LichessProbeData` (wideSet + merged metadata + parameterized loader + 5 theme
mappings), `SessionController+TacticalProbe` (`ProbeCategory` cases + `runBatch` +
timing), `LichessProbeWatcher` (combined-batch tick, split, timing log),
`SessionController` (wide history + watcher wiring + auto-export driver),
`LichessProbeExporter` (`tag:` / `announce:`), `SessionCheckpointFile` +
`SessionController+Checkpoint` + `SessionController+Training` (persist/restore),
`LichessProbeDetailView` + `LichessProbeDetailWindow` (wide chart), and the
bundled `Resources/lichess_probes_wide.json`.

## Validation
Build compile-only (no app launch during training). Verified live on build 1642:
wide set ticks every 25 steps at ~15% argmax / pElo ~817 across all 13 themes,
both training-start auto-exports landed, the 200-set restored its 5,286-tick
history intact, and training continued without throughput regression.
