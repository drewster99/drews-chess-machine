# Changelog

All notable changes to Drew's Chess Machine are recorded here, newest first.
Each entry is timestamped with the date and time the change was committed.

---

## 2026-04-17 17:23 CDT — Planned stability + learning-speed upgrade (DESIGN, not yet implemented)

Reasoning captured in `chess-engine-design.md` → "Stability Enhancements and Learning-Rate Upgrades". Summary grid:

| # | Change | Value | Helps with | Phase |
|---|--------|-------|------------|-------|
| 1 | Gradient clipping (global L2 norm) | `max_norm = 5.0` | Caps per-step parameter change. Prevents 2026-04-15-style single-step blowup. Silent on healthy batches. | Safety |
| 2 | Weight decay (L2 on all params) | `c = 1e-4` | Persistent pressure against slow weight growth. Generalization. Prevents the conditions that prime runaway logits. AlphaZero / ResNet standard. | Safety |
| 3 | Advantage baseline (`z − v.detached()`) | replace raw z in policy loss | 5–20× reduction in policy-gradient variance. Moves in obvious wins/losses get near-zero gradient; surprise outcomes get strong gradient. Biggest single learning-speed lever. | Speed |
| 4 | Batch size | `1024 → 4096` | 2× additional gradient-variance reduction at zero throughput cost (self-play is the bottleneck). Supports higher lr. Peak RAM 8.7 GB → 17.4 GB, within 37 GB budget. | Speed |
| 5 | Learning rate | `5e-4 → 1e-3` | Square-root scaling with 4× batch growth; conservative vs Lc0's linear rule which would say 2e-3. | Speed |
| 6 | K (policy-loss coefficient) | keep `50` | Already proven to produce signal post-fix `1ec8a13`; clipping removes need for K-warmup. | — |
| — | Logit L2 regularization | **skipped** | Redundant with weight decay; weight decay has more side benefits. | — |
| — | Advantage clamp | **skipped** | `v ∈ [−1, +1]` via tanh, so advantage is already bounded in [−2, +2]. Clamping would suppress most informative surprise cases. | — |
| — | Buffer pre-fill | **keep 20%** | Only affects time-to-first-step, not steady state. | — |
| — | K warmup | **skipped** | Gradient clipping handles the same "initial gradient too big" problem. | — |

**Implementation order:** #1 → #2 → #3 → (#4 + #5 together). Each its own commit and CHANGELOG entry with observed effect.

---

## 2026-04-17 15:23 CDT — Fix policy-loss scaling: drop `(w+1)` normalizer, set K=50

**File:** `DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift`
**Commit:** `1ec8a13`

**Before:**
```
total = (1000·policyLoss + valueLoss) / 1001
```
Effective coefficients: policy ≈ 0.999, value ≈ 0.001. The `/1001`
normalizer cancelled the ×1000 boost intended for the policy term and
simultaneously crushed the value term by ~1000×. Value head still
learned (its raw gradient is large enough to dominate even at 1/1001),
but the policy head saw no amplification — entropy stayed pinned near
`log(4096)` for 10k+ steps in multi-hour runs.

**After:**
```
total = 50·policyLoss + valueLoss
```
Effective coefficients: policy = 50, value = 1. Real 50× boost on the
policy path, value returns to its natural gradient magnitude.
Recommended pairing: drop `lr` from 1e-2 to 5e-4 so the shared trunk
doesn't diverge under the stronger combined gradient.

**Observed effect (1 h post-change, lr=5e-4, batch=1024):**
- `vLoss` dropping much faster than prior runs: 0.83 → 0.11 in 1 h
  (prior run's floor was ~0.29 after 19 h).
- `pEnt` shows first measurable directional movement:
  8.3046 → 8.3034 → 8.3024 → 8.2986 → 8.2973 over 45 min.
- Arena #1 score 0.507, arena #2 score 0.495 — candidate now tracking
  near parity, monitoring for regression.
