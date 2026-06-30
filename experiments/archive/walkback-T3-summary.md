# Walkback T3 — workers=48 @ target_tau=0.8 (accept dir 20260423-080402)

**Result: FAILED** (2/3 pass, R3 collapsed)

| Run | im_min | im_fin | max | au_max | au_fin | Verdict |
|-----|--------|--------|-----|--------|--------|---------|
| R1  | 0.171  | 0.204  | 0.413 | 4 | 4 | PASS |
| R2  | 0.322  | 0.334  | 0.339 | 4 | 4 | PASS |
| R3  | 0.908  | 0.994  | 0.997 | 0 | 0 | COLLAPSE |

T3 is the closest tier to passing so far — 2/3 runs achieved real learning (im~0.17-0.32, au=4) with no collapse. But R3 collapsed, so all-3 criterion fails.

The T3 regime (workers=48, target_tau=0.8) appears to have ~67% basin-of-attraction for learning. Worth keeping in mind if T4/T5 also fail.

Next: T4 (wd=2e-4 @ workers=32, target_tau=0.8).
