# Walkback T1 — target_tau=0.6 (accept dir 20260423-132740)

**Result: FAILED** (tier falsified on first replicate, fail-fast)

| Run | im_min | im_fin | max | au_max | au_fin | Verdict |
|-----|--------|--------|-----|--------|--------|---------|
| R1  | 0.9999 | 1.0000 | 1.0000 | 0 | 0 | COLLAPSE |
| R2  | — | — | — | — | — | skipped (fail-fast) |
| R3  | — | — | — | — | — | skipped (fail-fast) |

Original single-run accept claimed: im=0.14, max=0.24, au=9/9. Replicate failed to reproduce any of those.

Combined with the prior accept-time replicate (also full-collapsed, see `20260423-143233`), we now have **2/2 replicates at T1 full-collapsed**. Conclusion: target_tau=0.6 @ workers=48 / wd=2e-4 / ent=0.008 is not a seed-stable regime.

Next: walk back to T2 (target_tau=0.7, accept dir `20260423-131113`).
