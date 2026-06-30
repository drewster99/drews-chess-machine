# Walkback T2 — target_tau=0.7 (accept dir 20260423-131113)

**Result: FAILED** (R2 collapsed, tier cannot satisfy all-3)

| Run | im_min | im_fin | max | au_max | au_fin | Verdict |
|-----|--------|--------|-----|--------|--------|---------|
| R1  | 0.4458 | 0.5192 | 0.248 | 3 | 3 | PASS |
| R2  | 0.9995 | 1.0000 | 1.000 | 0 | 0 | COLLAPSE |
| R3  | — | — | — | — | — | skipped (fail-fast) |

At T2 we saw 1 pass + 1 collapse. Seed variance still too wide. Moving to T3 (workers=32→48 accept @ target_tau=0.8).
