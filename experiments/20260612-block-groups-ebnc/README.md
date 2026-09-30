# 2026-06-12 — First heterogeneous block-groups tower (eBNC): 3×[3×3 @128] → 3×[5×5 @256]

**Status:** abandoned. The run trained cleanly to step 49,924 (10 promotions). Every attempt to resume from its only surviving checkpoint (manual save, step 49,836) went non-finite or blew up within the first 1–16 training steps. Seven resume attempts across builds 1845–1890, one of them in forced float32, failed between 2026-06-13 19:56 and 2026-06-14 19:53 CDT, after which the machine moved to a fresh fp32 run (`wTp3`).

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 7 (marked "IN PROGRESS", full write-up deferred) and audited 2026-09-29. The original stub text is kept verbatim under "Original stub", with corrections marked there. Everything else is new, built from the logs and the surviving session.

## Question

This was the first production run of the block-groups feature (ARCHITECTURE_EXPANSION_PLAN.md Feature 2). Does a heterogeneous, multi-width tower with skip projections train, promote, and survive save/resume end to end? It is **not** a controlled architecture comparison; it is a shakedown of the new capability.

## Setup

- **Architecture** (`[ARCH] built champion 20260612-21-eBNC`; confirmed against the embedded `block_groups` JSON in the surviving `gFlw-manual` `champion.safetensors`):
  `v4 . in basic30(30) -> stem 128 (3x3) . 3x[3x3+3x3 @128, SE+/2, relu/pre, clean_add, ReZero(0.408), drop*1] -> 3x[5x5+5x5 @256, SE+/2, relu/pre, clean_add, ReZero(0.408), drop*1] . act relu . policy intermediate_conv(4864) . value WDL(32->FC128) . bfloat16 . 10,659,093 params`
  - Group 1: 3 blocks, 128 ch, 3×3 + 3×3, scale-and-bias SE /2, pre-activation ReLU, clean_add, ReZero α 0.408248 (= 1/√6), dropout multiplier 1.
  - Group 2: 3 blocks, 256 ch, 5×5 + 5×5, same recipe.
  - The 128→256 transition carries a bias-free 1×1 skip projection: tensor `blocks.3.skip_proj.weight` [256,128,1,1], with `blocks.3.conv1.weight` [256,128,5,5].
  - Stem `stem.conv.weight` [128,30,3,3]. Policy `intermediate_conv` (pre-conv 128). Value WDL with 32 conv channels → FC 128.
  - Param count: the header tensor-shape sum is exactly 10,659,093.
  - Note: the session `manifest.json` `architectureSummary` reads the misleading legacy string `v4 6x[256ch] in 30 planes (legacy)`. The embedded `block_groups` config is authoritative.
- **Training config** (`[STATS]` `reg=`/`lr=`; `[PARAM]`/`[RESUME-PARAM]`): constant LR **1e-2** (500-step warmup, `·√b` no-op at 4096), **weight_decay 5e-4**, grad_clip 30, μ 0.90, entropy 0, draw_penalty 0, value label smoothing ε 0.013. **Channel dropout 0.70 from step 0 to ~7.7k** (`[PARAM] dropoutRate applied to training graph: 0.7000` at 18:54, then `0.700 -> 0.300` at 20:27:43 CDT), **0.30 thereafter**. Batch 4096, **170** self-play workers, `spDelay=0ms`, replay-ratio target 0.48 with auto off, promote ≥ 0.53, 400-game arenas every 900 s.
- **Builds / git:** 1835 (`c23277b*`, fresh) → 1839 (`ed84386*`) → 1841 (`cf0e0d9*`, last healthy segment). Failed resumes ran on 1845 (`e08f390*`), 1872, 1874, 1881, 1883, 1890 (`e8204c8*`), and a second 1845 (`70157f2*`).
- **Machine:** local (logs on this Mac). The hardware is not logged.

## Runs

- **Live champion lineage:** `20260612-21-eBNC` → `eBNC-10`. The final trainer is `eBNC-11`. **Saved-session lineage:** `20260612-22-gFlw`. The original stub said "lineage `eBNC` (saved/live)"; the saved lineage is actually `gFlw`.

| Log | Build | Steps | What happened |
|---|--:|---|---|
| `dcm_log_20260612-185142.txt` | 1835 | 0 → 29,363 (`[ENCODE-COST]` to 29,380) | Fresh build 18:54 CDT. 20 arenas, 6 promotions. The log **ends abruptly at 01:09:11** on 06-13, 0.4 s after the post-promotion save at 29,363, with no error line. |
| `dcm_log_20260613-114410.txt` | 1839 | 29,363 → 34,443 | Auto-resume from the 29,363 promote save. Step time rose to ~1.8–3.4 s (vs ~0.72 s before and after), so arenas landed only ~200 steps apart. 16 arenas, 1 promotion (33,360). The process was relaunched at 16:01 on a new build. **The 1,083 steps after 33,360, and arenas at 33,821 and 34,289, were discarded** by the rewind. |
| `dcm_log_20260613-160117.txt` | 1841 | 33,360 → 49,924 | Resume from the 33,360 promote save. 17 arenas, 3 promotions. Manual save at **49,836** (19:20:22 CDT). Training continued to 49,924, and the process ended at 19:21:48. |
| `dcm_log_20260613-195604.txt` | 1845 | 49,836 → 49,837 | `[ALARM] loss non-finite … grad=nan` at the first step. `[STATS]` froze at steps=49,837, gNorm 757,760, legalMass ~0.03 until the process ended at 21:32. |
| `dcm_log_20260614-010858.txt` | 1872 | → 49,842 | `loss non-finite: value=1.29e9, grad=inf` at 16 s. Stopped. |
| `dcm_log_20260614-012630.txt` | 1874 | 49,836 → 50,070 | `forceFloat32` override (bf16 → fp32 for all nets). Trained 234 steps without NaN, but **already blown up**: pLoss 4.59 → 2.97 (pre-save ~1.80), gNorm 113 → 31 (pre-save ≤ 3.3), legalMass 0.000–0.023 (pre-save ≥ 0.977). Stopped at 01:39. |
| (`dcm_log_20260614-014435.txt`) | 1876 | — | **Not eBNC:** fresh `wTp3` (4×[3×3+3×3 @128], float32, 1,430,035 params). The machine moved on. |
| `dcm_log_20260614-103503.txt` | 1881 | 49,836 | Load via picker, then `loss non-finite: total 8.9e12, policy 3.5e8` at 4 s. Stopped. |
| `dcm_log_20260614-104947.txt` | 1883 | 49,836 | Same (`value=1.03e13`). First build with `[DIVERGE] training suspended` (commit `e8204c8`). |
| `dcm_log_20260614-173901.txt` | 1890 | 49,836 → 49,944 | Trained ~108 steps with **gNorm 1.1e10 → 1.7e10** (pre-clip; the "eBNC divergence moment gNorm ~1e10" cited in `documentation/dashboards/selfplay.py`), pLoss ~5.0, legalMass 0.000. Process ended at 17:42. |
| `dcm_log_20260614-195324.txt` | 1845 (`70157f2`) | 49,836 | `loss non-finite … grad=nan` at 9 s, then `[DIVERGE]` suspend. Last eBNC launch. |

**Arenas / promotions on the surviving lineage:** 51 arenas (matching `manifest.json` `arenaCount: 51`) and **10 promotions**. The two discarded arenas on the rewound branch are excluded.

| # | Promotion step | Score | New champion |
|--:|--:|--:|---|
| 1 | 1,765 | 0.5350 | eBNC-1 |
| 2 | 8,098 | 0.5325 | eBNC-2 |
| 3 | 15,639 | 0.5300 | eBNC-3 |
| 4 | 16,806 | 0.5387 | eBNC-4 |
| 5 | 17,960 | 0.5325 | eBNC-5 |
| 6 | 29,363 | 0.5513 | eBNC-6 |
| 7 | 33,360 | 0.5350 | eBNC-7 |
| 8 | 37,103 | 0.5300 | eBNC-8 |
| 9 | 38,353 | 0.5300 | eBNC-9 |
| 10 | 43,469 | 0.5475 | eBNC-10 |

The five arenas after #10 (44,735 → 49,798) scored 0.4750–0.5100.

**Saved sessions** (from `[CHECKPOINT] Saved session` lines, all `…-20260612-22-gFlw-…`): promote at 1,765 / 8,098 / 15,639 / 16,806 / 17,960 / 29,363 (06-12 build 1835); 33,360 (build 1839); 37,103 / 38,353 / 43,469 (build 1841); and **manual 49,836** (`20260614-002022-20260612-22-gFlw-manual`, 7.3 GB). **Only the manual save survives.** Its `champion.safetensors` is `20260612-21-eBNC-10`, `training_step` 49,836, `content_sha256` 460fc5d6…. Its `trainer.safetensors` is `eBNC-11` (parent `eBNC-10`), 21,312,746 stored values (weights plus velocity), `content_sha256` d9a876e4…. `session.json` shows `trainingSteps` 49,836, `elapsedTrainingSec` 47,653.5 (13.2 h), dropoutRate 0.3, build 1841.

## Results

All pElo is the **June 2026 recording-build in-app scale** (`[TACTICAL-LICHESS] tick`; wide = 4,435 puzzles, 200 = 200-puzzle set). It is not comparable to the selfplay_registry `endpoint_pElo` 606.1 / NLL 3.5653 (a later-binary re-probe of trainer `eBNC-11`) or to replay-era pElo. Buckets are 4,000-step means over the three healthy logs.

| Steps | Wide pElo | Wide NLL | 200 pElo | 200 NLL |
|--|--:|--:|--:|--:|
| 0–4k | 523.3 | 4.052 | 598.0 | 4.299 |
| 4k–8k | 531.8 | 3.787 | 632.1 | 3.973 |
| 8k–12k | 559.0 | 3.705 | 613.9 | 3.903 |
| 12k–16k | 565.9 | 3.670 | 625.3 | 3.873 |
| 16k–20k | 586.7 | 3.665 | 653.4 | 3.825 |
| 20k–24k | 587.7 | 3.671 | 664.5 | 3.812 |
| 24k–28k | 588.6 | 3.670 | 656.7 | 3.821 |
| 28k–32k | 602.7 | 3.630 | 674.9 | 3.767 |
| 32k–36k | 622.3 | 3.588 | 701.0 | 3.739 |
| 36k–40k | 635.5 | 3.580 | 715.2 | 3.721 |
| 40k–44k | 662.1 | 3.559 | 739.7 | 3.667 |
| 44k–48k | 677.6 | 3.526 | 748.2 | 3.645 |
| 48k–49.9k | 677.4 | 3.531 | 752.5 | 3.655 |

- **Peak wide pElo:** 734.2 pointwise (step 46,452, NLL 3.458). The 4k-bucket peak is 677.6 at 44–48k, i.e. **at the end of the run, not mid-run**. Wide NLL fell monotonically, 4.05 → 3.53.

`[STATS]` snapshots (pD/vAbs are trainer-batch statistics; draws and length are champion self-play `comp`):

| Step | pD | vAbs | Self-play draws | Mean game length (plies) | pEnt | gNorm |
|--:|--:|--:|--:|--:|--:|--:|
| 9,992 | 0.779 | 0.084 | 87.4% | 341 | 2.67 | 1.36 |
| 20,036 | 0.808 | 0.085 | 90.4% | 325 | 2.63 | 1.40 |
| 34,975 | 0.777 | 0.102 | 87.0% | 314 | 2.65 | 1.15 |
| 39,984 | 0.776 | 0.104 | 86.3% | 263 | 2.67 | 2.01 |
| 45,036 | 0.772 | 0.116 | 85.4% | 247 | 2.67 | 2.08 |
| 49,924 | 0.770 | 0.121 | 84.5% | 247 | 2.67 | 2.19 |

- **Health through 49,924:** `[STATS]` gNorm max 4.77 / 2.32 / 3.25 across the three logs, legalMass min 0.906 / 0.960 / 0.977, pEnt 2.59–2.71, pIllM 0.011 at the end. No non-finite alarm.
- **Value head:** pD stayed ~0.77–0.81 with self-play draws 84–90% for the whole run. vAbs rose slowly (0.08 → 0.12) and game length fell (341 → 247). This is a flat, slow-to-differentiate value head with a high draw rate. pD never approached 1.0, so it is not a value-head collapse.
- **Throughput:** step time ~0.70–0.80 s in the build-1835 and build-1841 segments, but ~1.8–3.4 s during the build-1839 segment (29.4k–34.4k).

## Conclusion

- **The block-groups machinery works end to end for training:** a heterogeneous 128→256 WRN-style tower with a 1×1 skip projection built, trained 49.9k steps in bf16 at lr 1e-2, promoted 10×, and survived two save/resume cycles (29,363 on build 1839, 33,360 on build 1841) with its tactical probe still improving at the end.
- **The shakedown's "survives save/resume end to end" goal ultimately failed.** Every resume from the 49,836 manual save diverged within its first 1–16 steps: NaN/inf losses, pre-clip gNorm 7.6e5 up to 1.7e10, and legal mass collapsing from ≥ 0.977 to ~0. That happened on six different builds, and also in forced float32. Because the original process was healthy for 88 steps *past* that save, the failure is tied to the **saved/restored state** (the trainer weights plus optimizer velocity in `trainer.safetensors`, or how builds ≥ 1845 restore them), not to the training dynamics of the run itself. The exact cause was never isolated and is unverified (see Caveats). The run was abandoned, and training moved to a fresh fp32 run (`wTp3`).
- **Strength (June in-app scale):** wide pElo climbed slowly from ~523 to ~678 (4k-bucket), 200-set to ~750. For context only (confounded: 170 vs 800 workers, wd 5e-4 vs 1e-4, dropout, different tower), the same basic30 input on the 5-block 7×7 tower (bzw3) had 13 promotions by 40k (eBNC 9) and a 200-set of ~808 at 40–42k (eBNC 740).

## Caveats
- **Divergence cause unverified.** What the evidence supports:
  1. The original process trained past the save point with healthy metrics.
  2. Resumes on builds 1845–1890 all failed at once, including fp32.
  3. Earlier resumes of the same run on builds 1839 and 1841 worked.
  4. The resumed `[RESUME-PARAM]` values (dropout 0.3, lr 0.01, μ 0.9, wd 5e-4, label smoothing 0.013) match the healthy 1841 resume exactly, apart from float formatting.
  5. Commits between 1841 (`cf0e0d9`) and 1845 (`e08f390`) include `54e3ca3` (review fixes including the `dropout_rate` resume fallback) and `0a3768c` (CLI `--parameters` made transient).

  Still to rule out, without running the app (forbidden during the current replay runs): a corrupted or inconsistent trainer/velocity snapshot in the 49,836 manual save, a restore-path regression in builds ≥ 1845, or both. A bf16 head-offset survey later read the *champion* `eBNC-10` as "fine" (`documentation/research/bf16-head-offset/README.md`), which points at the trainer side.
- **Disk-full exit at ~29.4k unverified.** The log ends abruptly 0.4 s after the 29,363 save with no error line, which is consistent with the stub's account but not proven by it.
- **The build-1839 slowdown is unexplained** in the log. Worker count, delays and ratio were unchanged; the machine was also being used for same-day dropout A/B and test work, which is unverified as the cause.
- **Dashboard data:** `documentation/dashboards/data/eBNC.csv`'s final row (cum_step 49,990, segment 5, pLoss 3.25, gNorm 35.2, legalMass 0.016) comes from the **diverged forced-fp32 resume**, not the healthy run. Its pElo 697.957 / NLL 3.45975 duplicates the step-49,828 probe. The selfplay_registry label "10.66M (10 promo, 9h)" understates training time (`elapsedTrainingSec` 47,653.5 s = 13.2 h at 49,836). These files were not edited here.
- Single seed. Not a controlled comparison. The dropout schedule changed mid-run (0.70 → 0.30 at ~7.7k).

## Follow-ups
- Diagnose the resume divergence offline once the machine is free: load `20260614-002022-20260612-22-gFlw-manual`, run one trainer step with velocity zeroed vs restored, and with weights from the champion vs the trainer, on the current build. That separates "bad snapshot" from "restore regression."
- If the snapshot is the cause, add a save-time check that a trainer round-trip reproduces the same loss and gradient norm on a fixed batch, as the existing bit-exact forward-pass check does for inference.
- Any real architecture comparison with block groups should hold workers / wd / dropout equal to the baseline.

## Original stub (verbatim from ARCH_EXPERIMENTS.md, with audit marks)

**arch** `v4 . in basic30(30) -> stem 128 (3x3) . 3x[3x3+3x3 @128, SE+/2, relu/pre, clean_add, ReZero(0.408), drop*1] -> 5x5 group ...`. Concretely a two-group WRN-style staircase: stem 128 (3×3) → **3× [3×3 @128, SE scale+bias /2]** → **3× [5×5 @256, SE scale+bias /2]**, ReZero, channel dropout rate 0.30 [Audit: 0.70 until ~step 7.7k, then 0.30], ~**10.66M params** [Audit ✓ 10,659,093]. The 128→256 transition carries a 1×1 skip projection [Audit ✓ `blocks.3.skip_proj.weight` [256,128,1,1]]. · **lineage** `eBNC` (saved/live) [Audit: live `eBNC`, saved `gFlw`] · **build** 1835→1841 [Audit ✓ for the healthy run; failed resumes on 1845–1890] · **dates** 2026-06-12 → [Audit: healthy to 2026-06-13 19:21 CDT; last resume attempt 2026-06-14 19:53]

**Why:** first production run of the block-groups feature (ARCHITECTURE_EXPANSION_PLAN.md Feature 2). It validates that a heterogeneous, multi-width tower with skip projections trains, promotes, and survives save/resume end to end. Not a controlled architecture comparison; a shakedown of the new capability.

**Status (2026-06-13, still running):** healthy throughout. ~step 46k, champion `eBNC-10` (10 promotions). Loss/entropy/gNorm all in-band; `drop=0.30` applied. Survived a disk-full process exit at step ~29.4k (Trash holding the earlier cleanup pinned the volume at 100%) and auto-resumed cleanly from the post-promotion checkpoint, with no weights lost. ~~Wide-probe pElo peaked ~643 mid-run.~~ → [Audit: wide pElo peaked **734.2 pointwise at step 46,452**, and the 4k-bucket peak of 677.6 was at 44–48k, i.e. still rising at the end. ~643 does not correspond to a peak.] [Audit: the "auto-resumed" resume was a relaunch at 11:44 on 06-13, not an automatic restart, ~10.6 h after the 01:09 exit; the resume sheet was auto-accepted.]

*(in progress: full write-up deferred until the run concludes or is promoted to a controlled comparison)* [Audit: superseded by this document. Status: abandoned.]

## Audit notes

Verified against the ten registry logs plus `dcm_log_20260614-014435.txt` (to identify it as `wTp3`), streaming `[APP]`, `[BUTTON]`, `[ARCH]`, `[SEGMENT]`, `[ARENA] #N kv`, `[CHECKPOINT]`, `[STATS]`, `[PARAM]`/`[RESUME-PARAM]`, `[ALARM] loss non-finite`, `[DIVERGE]`, `[TACTICAL-LICHESS] tick`. Also checked: the surviving `Sessions/20260614-002022-20260612-22-gFlw-manual.dcmsession` (both safetensors headers, param sums, `session.json`, `manifest.json`), `documentation/dashboards/data/eBNC.csv`, `selfplay_probe/eBNC.csv` (wide; last healthy tick 49,828), `selfplay_registry.json`, `git log cf0e0d9..e8204c8`, the CHANGELOG `54e3ca3` entry, and `documentation/dashboards/selfplay.py` (gNorm ~1e10 comment).

- **Resolved status:** IN PROGRESS → **abandoned**. Final healthy step 49,924, last checkpoint 49,836, 10 promotions, 51 arenas, peak wide pElo 734.2 (pointwise) / 677.6 (4k bucket), all June in-app scale.
- **Correction:** channel dropout "0.30" → 0.70 for steps 0–~7.7k, then 0.30 (evidence: `[PARAM] dropoutRate applied to training graph: 0.7000` at 18:54:17; `dropoutRate: 0.700 -> 0.300` at 20:27:43; 586 `[STATS]` lines carry `drop=0.70`).
- **Correction:** wide pElo "peaked ~643 mid-run" → 734.2 at 46,452 (pointwise). The bucket mean was still rising at the end (evidence: `selfplay_probe/eBNC.csv`; 4k buckets from the logs).
- **Correction:** "lineage `eBNC` (saved/live)" → saved lineage **`gFlw`** (evidence: every `[CHECKPOINT] Saved session` name).
- **Correction:** weight_decay is **5e-4**, not the 1e-4 used by Exp 1–4 (the stub did not state it; recorded to prevent assumption) (evidence: `reg=(… decay=5e-04 …)` in every `[STATS]` line).
- **Verified:** architecture string, 10,659,093 params, 1×1 skip projection, 10 promotions (steps above), champion `eBNC-10` by ~46k, the 29,363 exit/resume, the gNorm ~1e10 divergence (build 1890 resume: 1.1e10 / 1.7e10).
- **Unverified:** the disk-full cause of the 29.4k exit (no log line); the cause of the build-1839 slowdown; the root cause of the resume divergence (would require running the app); hardware.
- **Scale note:** the registry endpoint 606.1 / 3.5653 is a later re-probe of trainer `eBNC-11` on a different scale. Every pElo figure in this file is the June in-app scale.
