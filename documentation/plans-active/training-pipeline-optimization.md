# Training Pipeline Optimization Plan

> **Status: PENDING / TODO** (2026-06-13 audit). The pipelined-trainer overlap
> (P1/P2(N+1) with P3(N)) is NOT built; trainer phases remain serial. Note the
> precompiled-executable lever DID ship separately (see GPU_UTILIZATION_PLAN.md Phase 2).

## Context

We instrumented the trainer step with `[LEGAL-COST]` and observed at batch=4096:

| Phase | What | P50 | % of step |
|---|---|---:|---:|
| P1 | replay-buffer sample + setup (trainer queue, lock-held) | ~5 ms | 0.5% |
| P2 | fresh-baseline forward pass on `trainer.network` (network queue, GPU) | ~157 ms | 17% |
| P3 | training step: forward + backward + update (network queue, GPU) | ~750 ms | 82% |
| **interStep** | end-to-end wall-clock between consecutive step completions | ~915 ms | — |
| **gapMs** | `interStep − (P1+P2+P3)` (dispatch/idle dark time) | ~0–15 ms | — |

Conclusions from the data:
- The trainer pipeline is tight — phases account for ~99% of wall time. No hidden idle.
- GPU work dominates (P2+P3 = ~99% of step time).
- The phases run **strictly serial** today: `await` in `trainStep` between P1→P2 and P2→P3, no overlap.
- Outliers exist but are bounded: post-warmup p3 p99 max ~860 ms; p1 jitter to 21–27 ms in ~4 of 480 steps (mixed lock-contention and constrained-path rejection-loop variance — see Item 22 reassessment below).

Apple's MPS Tuning Tips (canonical guidance) bullet 1 explicitly identifies this serial pattern as the anti-pattern:

> *"Don't wait for results to complete before enqueuing more work. There can be a significant delay (up to 2.5 ms) just to get an empty command buffer through the pipeline to where the `waitUntilCompleted()` method returns. Instead, start encoding the next command buffer(s) while you wait for the first one to complete… throughput can be enhanced by up to a factor of ten."*

Our regime won't see 10× (P3 is dense GPU compute, not plumbing-dominated), but the principle applies: **the trainer's current synchronous `await` between phases is leaving framework-level pipelining on the floor.**

## Context — parallel work in flight

The arena code path is being ported from `MPSChessPlayer` to the new tick-based driver (`TickSelfPlayDriver` + `ActiveGame` + `MoveSampler`). When the port lands, `MPSChessPlayer`'s remaining users will be only the cold paths (human-vs-network play, Forward Pass / Play Game demo) and the deprecated `BatchedSelfPlayDriver`. This affects Item 24's value — see below.

## Headline opportunity — Pipelined trainer

Overlap **P1(N+1) and P2(N+1)** with **P3(N)** so the critical path collapses from `P1 + P2 + P3` to roughly `max(P1+P2, P3) = P3 alone`.

### Theoretical envelope

```
serial:  P1 ─→ P2 ─→ P3 ─→ next P1 ...
         5 ms  157 ms 750 ms

pipelined:                ┌─ P1(N+1) ─→ P2(N+1) ─┐
         P3(N) ───────────┤                       ├─→ P3(N+1) ──→
         750 ms           └─ overlap with P3(N) ─┘
```

Upper bound: `(P1 + P2) / total = 162 / 912 ≈ 18%` step-time reduction, i.e. **~22% more steps per second**.

Realistic bound depends on how well the GPU itself can pipeline P2's forward pass against P3's forward+backward+update:
- **CPU-side overlap only** (P1 + Swift plumbing + command-buffer encode hidden under P3's wait): floor ≈ **5–7%**.
- **Full GPU pipelining** (P2 forward runs concurrent with P3 on a second `MTLCommandQueue`): ceiling ≈ **15–18%**.

### Staleness of the value baseline — negligible

Today's `vBaseline` is computed from weights produced by step N−1. Pipelined: it'd be 1 step "older" — same staleness order. At lr ≈ 1e-4, weights drift by <0.1% per step; the resulting `vBaseline` shift is ~3 orders of magnitude below the policy-gradient noise floor. For comparison the pre-WDL implementation used *play-time* baselines (many promotions old) and that was bad enough to be bias-prone; "1 step stale" is fine.

### Implementation sketch

Approach **A — Pre-fetch inside trainStep** (smallest delta):

Inside `trainStep`, after P2 finishes and before P3 starts, kick off P1+P2 for step N+1 as a detached `Task`. Hold the in-flight future on the trainer; the next `trainStep` call awaits it instead of computing P1+P2 inline.

```swift
// (conceptual)
private var nextPhaseTwoInFlight: Task<(Phase1, [Float]), Error>?

func trainStep(...) async throws -> TrainStepTiming? {
    // 1. Take whatever P1+P2 the previous call pre-fetched, or do it now.
    let (phase1, freshValues) = try await (nextPhaseTwoInFlight?.value ?? computeP1P2Now())
    nextPhaseTwoInFlight = nil

    // 2. Pre-fetch P1+P2 of the NEXT step on a detached task.
    nextPhaseTwoInFlight = Task.detached { try await self.computeP1P2Now() }

    // 3. Run P3 now (the heavy GPU step). It runs *concurrent* with the pre-fetched task above.
    return try await runPhase3(phase1: phase1, freshValues: freshValues)
}
```

Approach **B — Three-stage explicit pipeline** (more general, more invasive):
Each of P1, P2, P3 runs on its own task with a bounded inter-stage channel (depth 1). The current `trainStep` becomes a thin facade that pulls a finished TrainStepTiming from P3's output channel.

**Recommend A**: minimal surface change, easy to disable behind a feature flag, easy to validate incrementally.

### Critical complications

These need to be handled regardless of approach:

1. **BN running-stats double writer.** P2 and P3 both go through `trainer.network`'s training-mode BN, which has assign ops that update `running_mean` / `running_variance`. Two concurrent forward passes through the same network would race on those assigns. **Fix:** P2 must run on a *sidecar* network that shares weights with `trainer.network` but has its own BN running-stats lineage. Plumbing involves:
   - A second `ChessNetwork` instance dedicated to P2.
   - Weight-copy from trainer → sidecar at session start and at every promotion (~rare).
   - Decide whether the sidecar uses training-mode BN with its own (drifting) running stats, or inference-mode BN with frozen stats. Either works as long as the BN math matches what P3's forward leg does. *Training-mode is the safer match for the advantage-baseline correctness story.*

2. **Separate `MTLCommandQueue` for P2.** The sidecar network needs its own command queue so its commands can submit concurrently with P3's. Two queues, both feeding the same Metal device, scheduled in parallel by the SoC.

3. **Replay buffer access ordering.** P1 holds the buffer lock; if P3(N) doesn't touch the buffer (it doesn't), P1(N+1) running concurrent with P3(N) is safe. Self-play appender contention with P1 is unchanged from today.

4. **Cancellation on Stop / Arena pause / session end.** Pre-fetched work must be cancellable without corrupting `_completedTrainSteps`, accumulators, or the replay buffer. The pre-fetched Task should check for cancellation at safe points; trainStep should drop a pre-fetched future on cancel. **Note on replay-buffer-sample accounting**: a pre-fetched P1 that has already returned has *consumed* its `ReplayBuffer.sample(count:)` rows — the sampling-constraints state machine (per-game caps, draw-keep quota, per-hash bucket) has advanced. Cancellation discards those rows without re-queueing them. That's the right behavior at low cancellation frequency (Stop / arena boundary; not in steady state), but the spec should be explicit so a future implementer doesn't try to "undo" the sample on cancel.

5. **Trainer-internal counter ordering.** Today `_completedTrainSteps` is incremented at the end of P3. Pre-fetched P1 reads it (for `isStatsStep` decisions). Two in-flight reads/writes need defined ordering — easiest is to decide isStatsStep at the moment the pre-fetch is *launched* (using the counter at that point) rather than re-reading later.

### Estimated cost

**1–3 days of focused work** including the sidecar-network plumbing, the pre-fetch state machine, cancellation paths, and validation. Highest leverage / highest risk item in the backlog.

---

## Recommended sequencing — REVISED

**Go directly to pipelining. Items 22 / 24 / 25 deferred indefinitely.**

This revises the earlier "22 → 24 → pipelining" sequence after a closer read of the affected code paths and the live `[LEGAL-COST]` data.

### Why 22 is deferred (was: "half a day, decent win"; revised: "~150-line refactor, ~0.5% under current load")

After reading `ReplayBuffer.sample(...)` in detail:

- The **constrained path** is the active path under our current defaults (`maxDrawPercentPerBatch=75` triggers it). It's ~250 lines that **interleave index selection with `emit()` calls** — rejection sampling, K-cap tracking, length-tilt β solver, and the per-game-count dictionary all live in the same loop as the emit. Moving emit outside the lock means restructuring the loop to defer emits to a second pass — workable but it's a delicate refactor on a delicate code path.
- The **fast path** is simpler but also interleaves emit with W/D/L tallying and per-game-count updates.
- **Tear-safety** is the real constraint: if we emit outside the main lock, an `append` can clobber a sampled slot mid-read of its 1920-float board → torn data across cache lines → corrupted training sample. To prevent that requires either per-position lock-acquire/release (4096× per batch with minor overhead and no actual lock-hold-time reduction in the trainer direction) or per-slot version counters (4 MB extra storage + memory-barrier discipline).
- **Realistic gain:** trainer's total lock-hold time stays ~5 ms either way (the memcpy dominates and has to happen under *some* lock for tear-safety). Self-play appenders' wait per collision drops from ~5 ms to ~100 µs — but appender collisions with the trainer's sample are rare (~1% of appends at current load). Net self-play throughput gain: **~0.5% under current load**.
- Outlier mitigation: the 21–27 ms p1 spikes we observed are mostly the *constrained-path rejection-loop variance*, not lock contention. The lock-shrink wouldn't fix them — the rejection loop still runs under the lock.

**Verdict:** Item 22 is a defensive engineering change for future heavier load (much higher worker counts where appender contention rises). It's not a today-win at ~0.5%. Skip in favor of pipelining; revisit if pipelining changes the contention dynamics or if we go to >5000-worker scale.

### Why 24 is deferred (arena port absorbs it)

Item 24 targets `MPSChessPlayer.swift:480`'s `Array(rowConst)` per-ply Sendable-bridge copy. The remaining users of that path once the arena port lands are:
- Human-vs-network play (human-paced; per-ply cost invisible)
- Forward Pass / Play Game demo (cold)
- Deprecated `BatchedSelfPlayDriver` (slated for removal)

The arena port itself eliminates the only hot user of `MPSChessPlayer.sampleMove` — arena games now go through `MoveSampler` with pointer-flavored eval, no per-ply Array allocation. Item 24's remaining beneficiaries are all paths where the optimization is invisible.

**Verdict:** Drop from the list once the arena port lands. If anything pulls `MPSChessPlayer` back onto a hot path unexpectedly, re-add.

### Why 25 stays deferred

The `[LEGAL-COST]` data shows the in-trainer legal-mask loop is ~32 ms / step = ~3.5% of total step time today, and ~4% of pipelined step time. The cost is +60 MB replay-buffer footprint plus extra per-ply self-play work and a per-position storage-layout change with `.dcmsession` serialization implications. The pipelining refactor doesn't depend on it and we can re-measure after pipelining lands.

**Verdict:** Defer until after pipelining; may skip entirely depending on post-pipelined `[LEGAL-COST]` numbers.

### Revised sequence

1. **Pipelining refactor** (Approach A) — 1–3 days. The main event.
2. **Reassess 22 / 24 / 25** with fresh `[LEGAL-COST]` data from the pipelined regime.
3. Items most likely to survive reassessment: only 25, and only if its relative cost rises significantly post-pipelining.

---

## Verification (whole plan)

End-to-end after pipelining:
1. **Build:** `mcp__drews-xcode-mcp__build_project` clean.
2. **Tests:** All `DrewsChessMachineTests` pass without modification.
3. **`[LEGAL-COST]` comparison:** Run a 15-minute training session; expect `interStepMs` p50 to drop from ~915 ms toward ~770 ms (P3 alone) ± GPU-pipelining efficiency. `gapMs` should remain near zero.
4. **Training-quality regression check:** Run 1+ hour of training and compare `[STATS]` line outputs against a recent baseline session: `pEnt`, `pLoss`, `vLoss`, `pW/pD/pL`, `gNorm` should evolve in the same ranges. The pipelined version produces a slightly different per-step `vBaseline` (1 step staler) — this should be statistically invisible at lr=1e-4, but verify.
5. **BN running-stats sanity:** Snapshot `running_mean` / `running_variance` of a few BN layers before/after pipelining lands; the running-stats trajectory should be the same (sidecar network shouldn't pollute trainer's stats; this is the BN-double-writer concern).
6. **Promotion smoke test:** Run an arena tournament; confirm candidate weights load correctly and W/L/D results behave sanely.
