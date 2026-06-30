# Walkback T0 — lr=5e-6, warmup=300, ent=0.0025, wd=3e-4, tt=0.65 (user-specified)

**Result: FAILED** (R1 collapse-bailed, fail-fast)

| Run | termination | secs | steps | im | max | au | Verdict |
|-----|---|---|---|---|---|---|---|
| R1  | legal_mass_collapse | 171 | 82 | 1.000 | 0.163 | 0/0 | COLLAPSE (early-bail) |
| R2  | — | — | — | — | — | — | skipped (fail-fast) |
| R3  | — | — | — | — | — | — | skipped (fail-fast) |

**Diagnostic:** lr=5e-6 is 6× lower than the prior best 3e-5 — combined with warmup=300, by the bail point (step 82) the effective lr was ~5e-6 × (82/300) ≈ 1.4e-6. That's effectively zero learning signal; policy never had a chance to sharpen. The collapse-detection caught it correctly.

**Time saved by collapse-bail:** ran 171 s vs requested 1200 s = ~85% of the run was avoided. New mechanism working as designed.

User's call on next direction.
