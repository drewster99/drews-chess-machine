# Pending: replay-buffer batch-stats instrumentation

User-approved design (2026-05-02). Implement after current 5400s run (`20260502-041819`) finishes — do not interrupt the in-flight run.

## New per-entry fields in ReplayBuffer.swift

Add 5 parallel `UnsafeMutablePointer<T>` arrays mirroring the existing pattern (`boardStorage`, `moveStorage`, `outcomeStorage`, `vBaselineStorage`). Do NOT introduce a struct-of-structs — the parallel-arrays layout is load-bearing for bulk-memcpy append.

| Field | Type | Bytes | Notes |
|---|---|---|---|
| `ply_index` | UInt16 | 2 | 0–65535 covers any chess game |
| `game_length_plies` | UInt16 | 2 | Lets `ply/length` be computed without a join |
| `worker_id` | UInt8 | 1 | Detects per-worker over-representation |
| `intra_worker_game_index` | UInt32 (used as 24 bits) | 4 | Or pack with worker_id into UInt32 if cleaner |
| `sampling_temperature` | Float | 4 | Tau actually used for this move |
| `state_hash` | UInt64 | 8 | Hashed AFTER encoding from a 32-byte canonical form derived from the encoded planes — agreement with what the GPU sees is non-negotiable |

Total: ~21 B/entry → +21 MB at capacity 1M. Negligible.

## Append-time bookkeeping (#1, #2)

A `Dictionary<UInt64, BufferedPositionStats>` keyed on `state_hash`, updated atomically on every append AND on every eviction (decrement when the ring overwrites that index). Value type:

```swift
struct BufferedPositionStats {
    var count: UInt32           // how many entries with this hash exist in the buffer right now
    var winSum: Int32           // Σ outcome > 0
    var drawSum: Int32          // Σ outcome ≈ 0
    var lossSum: Int32          // Σ outcome < 0
}
```

This gives us, separately from the sampler:
- Global per-position duplication count
- Whether duplicates carry the SAME outcome (pure dup) or DIFFERENT outcomes (legitimate diverse rollouts)

Cost: one dict update per appended position, ~400 µs per worker per game burst — fine.

## Append signature (breaking change)

`append()` becomes:

```swift
func append(
    boards: UnsafePointer<Float>,
    policyIndices: UnsafePointer<Int32>,
    vBaselines: UnsafePointer<Float>,
    plyIndices: UnsafePointer<UInt16>,
    gameLength: UInt16,                     // broadcast — same for all positions in this game
    workerId: UInt8,                        // broadcast — same for all positions in this game
    intraWorkerGameIndex: UInt32,           // broadcast
    samplingTemperatures: UnsafePointer<Float>,
    stateHashes: UnsafePointer<UInt64>,
    outcome: Float,                         // broadcast as before
    count positionCount: Int
)
```

Update `BatchedSelfPlayDriver` and any other callers to plumb these through. The driver already knows ply (move counter) and worker id; needs to track per-game intra-worker index.

## Per-batch summary computed in trainer (#6, #7)

Every 10th batch (configurable `batch_stats_interval`, default 10), after sampling but before the GPU step, compute:

- `unique_count` and `unique_pct` (= `unique / batch_size`)
- `dup_max` (largest count of any single hash in the batch)
- `dup_distribution` — histogram of `[1 occurrence, 2, 3, 4, 5+]`
- `ply_histogram` — buckets `[op:1–10, early:11–25, mid:26–40, late:41–60, end:61+]`
- `phase_by_material_histogram` — buckets `[op:≥14 non-pawn, early:12–13, mid:8–11, late:4–7, end:≤3]`
- `game_length_histogram` — `[short:≤80, med:81–250, long:251+]` plies
- `temperature_histogram` — bucketed by tau values actually emitted by SamplingSchedule
- `outcome_counts` — `wins`, `draws`, `losses`
- `phase_by_ply_x_outcome` cross-product — all 5×3 cells (early-game-wins, mid-game-draws, etc.)
- `worker_id_histogram` — counts per worker_id seen in this batch (sanity check for over-representation)

Output as a single JSON line (NOT key=value), prefix `[BATCH-STATS]`:

```
[BATCH-STATS] {"step":N,"uniq":4082,"uniq_pct":0.997,"dup_max":3,"dup_dist":{"1":4072,"2":7,"3":1},"ply_hist":{"op":412,"early":1093,"mid":1812,"late":577,"end":202},"phase_mat":{"op":2103,"early":982,"mid":655,"late":233,"end":123},"len":{"short":301,"med":2891,"long":904},"tau":{"2.00":104,"1.78":83,"...":0},"wld":{"W":1234,"D":1801,"L":1061},"phase_ply_x_wld":{"op_W":84,"op_D":290,"...":0},"worker":{"0":89,"1":94,"...":0}}
```

Single-line JSON keeps it greppable, lets the dashboard ingest pipeline parse it without a schema change, and survives field additions.

## Surface unique-state ratio in [STATS] too

Even with `[BATCH-STATS]` shipped, also surface `uniq_pct` (last computed value) inside the regular `[STATS]` line so it's visible in the same human eye-scan as pEnt and gNorm. One extra key=value pair: `bufferUniqPct=0.997`.

## Implementation order (independent compiles)

1. Add the parallel arrays + plumb append signature through `BatchedSelfPlayDriver` (driver populates per-position fields; trainer ignores them initially). Test: builds, app runs, no behavior change vs current.
2. Add the global hash-count dict with append/evict updates. Test: dict size = `storedCount`, sum of counts = `storedCount`.
3. Wire the per-batch summarizer + the new `[BATCH-STATS]` line + the `bufferUniqPct=...` field in `[STATS]`. Test: log line appears every 10 batches, JSON parses cleanly.

## Cautions (already accepted)

- `ReplayBuffer.append()`'s current bulk-memcpy of one outcome across `count` rows + a single vBaseline array stays — only NEW per-position fields (ply, hash, temperature) get parallel-array inputs.
- Per-batch sampling currently uses random indexing. **This change does NOT modify the sampler.** It adds observation only. Once we *see* the duplicate rate, the next conversation is whether to deduplicate / prioritize / re-weight; that's a separate decision.
- Don't ship until the in-flight 5400s run finishes — that's the first clean signal we have post-promotion-overlay. Build only after iteration `20260502-041819` lands.

## Cadence parameter

Add to `TrainingParameters.swift`:

```swift
@TrainingParameter(
    range: .integer(min: 1, max: 1000),
    category: .observability,
    liveTunable: false,
    description: "Compute and emit [BATCH-STATS] every N training batches. Cost is ~1ms per batch evaluated; default 10 keeps log volume reasonable."
)
public enum BatchStatsInterval: TrainingParameterKey {}
```

Default: 10.
