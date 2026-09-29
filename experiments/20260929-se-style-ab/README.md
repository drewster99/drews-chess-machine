# 2026-09-29 — SE style A/B/C: scale+bias vs attenuate-only vs none (corpus replay)

Does the squeeze-and-excitation variant matter on an otherwise identical net?
Every long-running strong line (v5, qeu8/Ejp0, nT8Y) used `scale_and_bias` SE,
but no run has ever compared SE styles on the same architecture — past
comparisons mixed SE style with depth, width, kernel and stem changes.

## Design

- **Only variable:** `se_style` — `scale_and_bias` (`sigmoid(γ)·x + β`, FC2 emits
  2C) vs `attenuate_only` (`sigmoid(z)·x`, FC2 emits C) vs `none` (no SE; answers
  whether SE helps at all). The three preset files in `presets/` differ in nothing
  else (verified by diff; only `se_style` and `label`).
- **Architecture:** v5-style — basic30 input, 7×7 stem → 3×[7×7+7×7 @128, SE /4,
  ReLU pre-act, ReZero (α init 0.447), clean_add skip, LayerNorm out] · policy
  intermediate_conv (128) · value WDL (16ch → FC128) · bf16 compute (see `presets/`).
- **Training mode:** offline corpus replay, not self-play. All arms train on the
  identical recorded game stream (corpus `20260624-192615-w3aA5b`, 12 epochs), so
  architecture is the only difference. Self-play was rejected for this test: a
  single self-play run per arm is dominated by promotion/arena chaos (n=1 noise).
- **Parameters:** `parameters.json` here (copy of the repo-root file at launch).
  Replay-relevant values: weight decay 3e-4; decaying LR cycle — peak 1e-1→1e-4,
  trough 1e-3→1e-6, period 20k steps, decay horizon 1M, starting at the peak;
  momentum follows the LR cycle (0.85→0.90 low, 0.95 high); warmup 1000; grad clip
  15; batch 4096; replay ratio 0.48; replay buffer 500k positions, training starts at 250k.
- **Initialization:** each arm is a separately minted fresh net, so the random
  init differs. Past replay seed-to-seed spread (nt8y seed study) was roughly
  7–25 pElo — treat smaller gaps as noise.
- **Concurrency:** all arms run at the same time on the same Mac, sharing the GPU
  evenly. Compare on **step** and **games_fed**; the time axis is shared-GPU time
  and is not comparable to other dashboard runs.

## Runs

| arm | fresh model | out model (rolling) | enumerated checkpoints |
|---|---|---|---|
| scale+bias | `20260929-test_SE_scale+bias-fresh.safetensors` | `20260929-test_SE_scale+bias-replay-latest.safetensors` | `20260929-test_SE_scale+bias-replay-step<N>.safetensors` |
| attenuate-only | `20260929-test_SE_attenuate-only-fresh.safetensors` | `20260929-test_SE_attenuate-only-replay-latest.safetensors` | `20260929-test_SE_attenuate-only-replay-step<N>.safetensors` |
| none | `20260929-test_SE_none-fresh.safetensors` | `20260929-test_SE_none-replay-latest.safetensors` | `20260929-test_SE_none-replay-step<N>.safetensors` |

All in `~/Library/Application Support/DrewsChessMachine/Models/`. Launch command
per arm:

```
DrewsChessMachine --replay-corpus 20260624-192615-w3aA5b \
  --start-model <fresh model> --out-model <out model> \
  --parameters parameters.json --epochs 12 --enumerate-checkpoints
```

Launch time, build, git hash, ModelIDs and log files: see **Launch record** below.

## Launch record

Launched 2026-09-29 15:07:27–15:07:43 CDT, all three at once, Release binary built
14:36:02 from git `7a434ea` (includes the shared trainer-config fix `cbc1894`).
`[REPLAY-CYCLE]` in each startup log confirms the cycle is live; KL probes every 100 steps.

| arm | ModelID | pid | log |
|---|---|---|---|
| scale+bias | `20260929-12-JZOe` | 77368 | `dcm_log_20260929-150727.txt` |
| attenuate-only | `20260929-13-06yp` | 77398 | `dcm_log_20260929-150735.txt` |
| none | `20260929-18-D9is` | 77413 | `dcm_log_20260929-150743.txt` |

**Not exactly resumable.** This binary predates exact resume (`d15f706`): its checkpoints
lack the optimizer momentum tensors, the fp32 master weights (the bf16 working copy is
saved) and the `trainer_*` schedule keys, so `--resume-exact` refuses them. Decision
(2026-09-29): keep these runs going rather than restart; if one is interrupted it can
only continue as a new branch (fresh momentum, restarted cycle), which must be recorded
as a new segment.

**Aborted second launch (14:37, 3 arms, 1M buffer):** stopped at ~800 steps. Each
process's physical footprint was ~24 GB (7.18 GB of it the 1M-position buffer) — ~72 GB
total on a 64 GB Mac with swap nearly full. Relaunched with a 500k buffer (250k prefill).
Checkpoints and all of the day's logs discarded.

**Aborted first launch (2026-09-29 14:15, 2 arms):** stopped after ~250 steps when
it was found that the corpus-replay runner never applied the LR/momentum cycle
(only the GUI session set it), so both arms were training at the static LR 1e-3.
Partial checkpoints discarded. Relaunch follows the fix that routes GUI, replay and
train-vs-UCI trainer configuration through one shared path. Also found: the owner's
`parameters.json` had `self_play_target_tau` / `arena_target_tau` = 0.02, which the
CLI loader rejects (declared range 0.05…5.0) though the GUI accepted it; this
experiment's copy uses 0.05 for both (neither affects replay).

## Results

_(pending)_
