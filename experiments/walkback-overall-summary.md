# Walkback overall — no tier reproducible under strict all-3 criterion

| Tier | Config | Pass rate | Best run |
|---|---|---|---|
| T1 | tt=0.6, w=48, wd=2e-4, ent=0.008 | 0/1 (fail-fast) | — |
| T2 | tt=0.7, w=48, wd=2e-4, ent=0.008 | 1/2 | R1 (im=0.45, au=3) |
| T3 | tt=0.8, w=48, wd=2e-4, ent=0.008 | **2/3** | R1 (im=0.17, au=4) |
| T4 | tt=0.8, w=32, wd=2e-4, ent=0.008 | 0/1 (fail-fast) | — |
| T5 | tt=0.8, w=32, wd=1e-4, ent=0.008 | 1/2 | R1 marginal (im=0.80, au=1) |

**Pass criterion** (set up-front): all 3 replicates must have `max_prob_max < 0.85` AND `above_uniform_count_max ≥ 1`.

**Empirical pass rates (under-sampled but indicative):** ~0% to ~67%. T3 is the strongest at 2/3. The whole regime sits on a basin boundary; no tier within the accept stack achieves the strict all-3 criterion at n=3.

## Implications

1. The strict all-3 criterion may be too strict for this app's seed-noise regime. A 2/3 criterion would let T3 pass (and possibly T2/T5 if extended).
2. T3 (workers=48 @ tt=0.8) is the empirically best tier we have evidence for — it passed 2/3 and produced the lowest-noise results when it did pass. It also covers the wd=2e-4 + entropy=0.008 changes from T4 and T5, which on their own didn't reproduce.
3. Walking further back means dropping wd=2e-4 (returns wd to 1e-4) or entropy (returns to the seed-unstable 0.0025 regime). Neither is promising.

## Decision points (for user)

A. Loosen pass criterion to "2/3 replicates pass" → T3 is provisional baseline; document caveat.
B. Treat T3 as best-known and run more replicates (e.g. 5 total) to estimate its true pass-rate distribution.
C. Revisit app-level changes (controller tuning further, smaller learning rate, etc.) — the underlying instability is upstream of any single hyperparameter we can move.
D. Accept the streaming reality: at this app/build, n=1 acceptance is meaningless and we need a fundamentally different evaluation protocol (longer runs that average over the early-step variance, or stricter regularization that crushes the collapse basin entirely).

Total walk-back cost: 8 runs × 15 min ≈ 2 h.
