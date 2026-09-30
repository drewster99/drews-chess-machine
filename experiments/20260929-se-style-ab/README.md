# 2026-09-29 — SE style A/B/C: scale+bias vs attenuate-only vs none (corpus replay)

**Status:** in progress — all three arms are live (pids 77368 / 77398 / 77413); results below run through step 10000.

## Question

Does the squeeze-and-excitation variant matter on an otherwise identical net?
Every long-running strong line (v5, qeu8/Ejp0, nT8Y) used `scale_and_bias` SE,
but no run has ever compared SE styles on the same architecture — past
comparisons mixed SE style with depth, width, kernel and stem changes.

## Setup

### Design

- **Only variable:** `se_style` — `scale_and_bias` (`sigmoid(γ)·x + β`, FC2 emits
  2C) vs `attenuate_only` (`sigmoid(z)·x`, FC2 emits C) vs `none` (no SE; answers
  whether SE helps at all). The three preset files in `presets/` differ in nothing
  else (verified by diff; only `se_style` and `label`).
- **Architecture:** v5-style — basic30 input, 7×7 stem → 3×[7×7+7×7 @128, SE /4,
  ReLU pre-act, ReZero (α init 0.447), clean_add skip, LayerNorm out] · policy
  intermediate_conv (128) · value WDL (16ch → FC128) · bf16 compute (see `presets/`).
  Exact `[REPLAY-ARCH]` lines from the startup logs:
  - scale+bias: `v5 . in basic30(30) -> stem 128 (7x7) . 3x[7x7+7x7 @128, SE+/4, relu/pre, clean_add, ReZero(0.447·tanh≤0.447), out:layer_norm, drop*1] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 5,208,050 params`
  - attenuate-only: same with `SE/4` · `5,195,378 params`
  - none: same with `no-SE` · `5,170,322 params`
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

### Launch record

Launched 2026-09-29 15:07:27–15:07:43 CDT, all three at once, Release binary built
14:36:02 from git `7a434ea` (includes the shared trainer-config fix `cbc1894`).
`[REPLAY-CYCLE]` in each startup log confirms the cycle is live; KL probes every 100 steps.
[Audit: `7a434ea` differs from `cbc1894` only in `CHANGELOG.md`; both commits are timestamped 14:37:04, after the stated
14:36:02 build, so the binary was built from the working tree that became `cbc1894`. The build time itself is unverified —
see Audit notes.]

| arm | ModelID | pid | log |
|---|---|---|---|
| scale+bias | `20260929-12-JZOe` | 77368 | `dcm_log_20260929-150727.txt` |
| attenuate-only | `20260929-13-06yp` | 77398 | `dcm_log_20260929-150735.txt` |
| none | `20260929-18-D9is` | 77413 | `dcm_log_20260929-150743.txt` |

## Results

Probed by the dashboard tracker (`documentation/dashboards/data/se_{sb,att,none}.csv`, registry keys
`se_sb` / `se_att` / `se_none`) on each enumerated 1k-step checkpoint. Full per-metric history lives in
those CSVs; this table is pElo / nll.

| step | scale+bias | attenuate-only | none |
|---|---|---|---|
| 1000 | 879.2 / 3.1516 | 916.7 / 3.0519 | 916.1 / 3.1137 |
| 2000 | 1073.9 / 2.7400 | 1092.8 / 2.7291 | 1085.5 / 2.6770 |
| 3000 | 1160.8 / 2.5973 | 1149.3 / 2.6292 | 1172.2 / 2.5701 |
| 4000 | 1184.7 / 2.5454 | 1201.9 / 2.5424 | 1222.6 / 2.5411 |
| 5000 | 1240.7 / 2.4899 | 1256.2 / 2.4921 | 1246.9 / 2.5029 |
| 6000 | 1267.6 / 2.4710 | 1259.3 / 2.4888 | 1265.0 / 2.4813 |
| 7000 | 1268.1 / 2.4642 | 1265.0 / 2.4707 | 1297.5 / 2.4513 |
| 8000 | 1263.5 / 2.4757 | 1276.9 / 2.4701 | 1305.2 / 2.4458 |
| 9000 | 1269.1 / 2.4621 | 1294.4 / 2.4401 | 1313.5 / 2.4233 |
| 10000 | 1275.3 / 2.4619 | 1284.1 / 2.4556 | 1300.1 / 2.4457 |
| 11000 | 1278.4 / 2.4571 | 1303.7 / 2.4445 | 1327.4 / 2.4201 |
| 12000 | 1294.9 / 2.4294 | 1289.2 / 2.4551 | 1316.6 / 2.4237 |
| 13000 | 1285.1 / 2.4348 | 1291.8 / 2.4401 | 1311.9 / 2.4280 |
| 14000 | 1276.9 / 2.4595 | 1285.1 / 2.4554 | 1307.8 / 2.4207 |
| 15000 | 1283.6 / 2.4535 | 1293.4 / 2.4516 | 1315.5 / 2.4359 |
| 16000 | 1286.2 / 2.4363 | 1308.3 / 2.4185 | 1327.4 / 2.3972 |
| 17000 | 1252.6 / 2.4638 | 1287.7 / 2.4586 | 1299.6 / 2.4524 |
| 18000 | 1268.1 / 2.4748 | 1310.9 / 2.4062 | 1307.8 / 2.3787 |
| 19000 | 1222.1 / 2.4972 | 1327.9 / 2.4017 | 1316.1 / 2.3876 |
| 20000 | 1287.2 / 2.4410 | 1226.7 / 2.5372 | 1298.5 / 2.4057 |
| 21000 | 1308.8 / 2.4205 | 1316.1 / 2.3952 | 1313.0 / 2.3733 |
| 22000 | 1314.5 / 2.4109 | 1329.5 / 2.3725 | 1353.7 / 2.3575 |
| 23000 | 1314.0 / 2.4274 | 1288.7 / 2.4490 | 1386.6 / 2.3341 |

## Conclusion

**Provisional (runs live).**

Running read (updated as marks land): the arms were within seed noise through 6k; from 7k the no-SE arm
leads on both pElo and nll, and scale+bias has been flat since 6k while the LR descends toward the first
cycle trough (~11k).

## Caveats

- Seed noise: each arm has a different random init (n = 1 per arm). The past replay seed-to-seed spread was about 7–25 pElo
  (see Setup). The no-SE lead at 10000 (+24.8 over scale+bias, +16.0 over attenuate-only) is still inside or at the top of
  that band.
- Shared GPU: the time axis is not comparable to other runs (see Setup ▸ Concurrency).

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

## Follow-ups

- Continue adding 1000-step rows as the tracker probes new checkpoints. Settle the conclusion after at least the first LR
  cycle trough and ideally a full cycle (20k).
- If the no-SE lead holds beyond seed noise, run a second seed per arm before drawing an architecture conclusion.

## Audit notes

Audited 2026-09-29 (~21:20 CDT) without touching the running processes.

- **Presets:** `diff` of `presets/test_SE_scale+bias.json` against the other two shows only `se_style`
  (`scale_and_bias` / `attenuate_only` / `none`) and `label`. Confirmed.
- **Results table:** every cell matches `se_sb.csv` / `se_att.csv` / `se_none.csv` (cum_step 1000–10000, rounded to
  one decimal place for pElo; `frozen_file` = the matching `…-replay-step<N>.safetensors`). The CSVs have no 11000+ rows yet
  (logs were at steps 10,520 / 10,530 / 10,140 at audit time), so no rows were added.
- **ModelIDs:** the safetensors `__metadata__.model_id` of the three fresh models is `20260929-12-JZOe` (scale+bias),
  `20260929-13-06yp` (attenuate-only) and `20260929-18-D9is` (none). These match the table and each log's
  `[REPLAY] start-model: … modelID=` line.
- **Param counts:** the summed tensor element counts in the fresh-model headers are 5,208,050 / 5,195,378 / 5,170,322.
  These match the `[REPLAY-ARCH]` lines.
- **pids / launch times / logs:** `ps` start times for 77368 / 77398 / 77413 are 15:07:27 / 15:07:35 / 15:07:43. They match
  the first log lines of `dcm_log_20260929-150727/150735/150743.txt` and each command line (`--replay-corpus
  20260624-192615-w3aA5b`, Release binary).
- **Hyperparameters:** `[REPLAY-HPARAMS]` / `[REPLAY-CYCLE]` confirm wd 0.0003, batch 4096, gradClip 15, lrWarmup 1000,
  bufCap 500000, minPrefill 250000, replayRatio 0.48, dropout 0, and LR cycle trough 1e-3 / peak 1e-1 over 20000 steps
  decaying to peak 1e-4 / trough 1e-6 over 1,000,000 steps, momFollow low 0.850→0.900 high 0.950. The corpus has 46 sealed
  shards.
- **LR trough "~11k":** the logged LR is 0.1 at step 1000, 0.00105 at 10000 and 0.000963 at 10500, so it was still
  descending past 10k. This is consistent with the claim; the exact trough step is unverified.
- **Unverified:** the "built 14:36:02" time (the Release binary on disk was rebuilt at 15:26:51, after launch, and the
  startup logs carry no `[APP]` build/git line); the aborted launches' ~24 GB / 7.18 GB / ~72 GB footprints and the step
  counts (their logs were discarded); the nt8y seed-spread figure (not re-derived here).
- No corrections to the original text.
