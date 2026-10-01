# Phase 2 validation: Ejp0 resumed on the fixed build (2026-10-01)

Head numerics plan, Phase 2: "Resuming Ejp0: the value offset is gone at load (logged), and no drift over 20k steps; the Phase 0 audit on the resulting checkpoints is fine."

- **Run:** current build (`e1d4323` source, Release), `--replay-corpus 20260624-192615-w3aA5b --start-model Models/20260702-Qeu8-resume3-replay-step681000.safetensors` (Ejp0 @681k, trained before the fix) `--parameters experiments/20260929-se-style-ab/parameters.json --training-step-limit 20000 --enumerate-checkpoints`. The LR cycle peaks at 0.1 (step 1000) and 0.0784 at step 20000, a deliberately hard setting for drift. Log `dcm_log_20261001-015706.txt`, 01:57–05:51, on AC power.
- **Load:** `[NUMERICS] value head recentered on load (…step681000.safetensors): meanRowNorm=28.9486 biasMean=+13.7292 velocity=none`.
- **Drift** (`head_mean_logits.csv`, every 50 steps, pre-centering means): value mean logit stayed within −0.0215 … +0.0049 for all 20k steps. Policy mean logit went from −57.58 (step 50) to −43.74 (step 20000), i.e. its magnitude only shrank. The policy offset can't be removed on load (it varies by square) and is now frozen against growth.
- **Audit** (`numerics_audit_*.json`, `audit_summary.txt`; checkpoints at 1k / 5k / 10k / 15k / 20k, weights as stored):

| step | verdict | value offset vs init | value bias mean | policy offset vs init | bf16 value ties | bf16 value ΔCE | bf16 policy KL | bf16 top-2 ties | bf16 top-1 changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | BAD | 0.009× | 0.0002 | 12.01× (BAD) | 0 | +1.33e-3 | 4.54e-4 | 0 | 1.89% |
| 5000 | BAD | 0.010× | 0.0000 | 10.24× (BAD) | 0 | −1.09e-3 | 2.65e-4 | 0 | 1.87% |
| 10000 | BAD | 0.010× | 0.0001 | 10.17× (BAD) | 0 | −8.10e-4 | 3.23e-4 | 0 | 2.39% |
| 15000 | BAD | 0.009× | 0.0001 | 10.15× (BAD) | 0 | −4.30e-5 | 3.79e-4 | 0 | 2.51% |
| 20000 | degraded | 0.008× | 0.0004 | 9.77× (degraded) | 0 | −5.52e-4 | 2.85e-4 | 0 | 1.93% |

**Verdict:**
- **Value head: pass.** The offset is removed at load, stays at ~0.01× init, and its mean logit never leaves ±0.022.
- **Policy head: no drift (pass), but not "fine".** The stored policy offset inherited from Ejp0 (12× init) is static-flagged BAD; it shrinks steadily (→ 9.77×) but cannot be removed on load. It does no damage: bf16 policy has 0 top-2 ties and KL ≤ 4.5e-4 at every checkpoint (the top-1 changes of ~2% are the bf16 tower cost seen on every model, fp32-trained ones included).
- The 1k checkpoint's bf16 value ΔCE (+1.33e-3) is just over Phase 1's 0.001 target; every later checkpoint is under it.
