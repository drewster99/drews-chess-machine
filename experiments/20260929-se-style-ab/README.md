# 2026-09-29 — SE style A/B/C: scale+bias vs attenuate-only vs none (corpus replay)

**Status:** done (2026-09-30). All runs are stopped: seed 1 at 33,014 / 33,012 / 32,036 steps, seed 2 at 7,282 / 7,289 / 7,019, zero-β at 5,030 / 5,004. **Final write-up: [REPORT-final.md](REPORT-final.md).** **Follow-ups (leaky FC1, no ReZero) and current conclusions: [FOLLOW-UPS.md](FOLLOW-UPS.md).** The sections below were written while the runs were live; figures described as current are as of when each section was written (the first seed-1 sections through step 10000; pids 77368 / 77398 / 77413).

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

Full report at step 30,000: [REPORT-30k.md](REPORT-30k.md).


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
| 24000 | 1347.5 / 2.3891 | 1414.8 / 2.3260 | 1402.5 / 2.3281 |
| 25000 | 1439.5 / 2.2813 | 1422.0 / 2.2782 | 1437.9 / 2.2754 |
| 26000 | 1440.0 / 2.2569 | 1443.1 / 2.2489 | 1440.0 / 2.2730 |
| 27000 | 1441.5 / 2.2618 | 1460.0 / 2.2769 | 1463.6 / 2.2543 |
| 28000 | 1446.7 / 2.2618 | 1464.6 / 2.2726 | 1482.1 / 2.2232 |
| 29000 | 1451.3 / 2.2532 | 1455.4 / 2.2553 | 1474.9 / 2.2249 |
| 30000 | 1446.7 / 2.2533 | 1478.0 / 2.2342 | 1483.6 / 2.2227 |
| 31000 | 1438.4 / 2.2702 | 1463.6 / 2.2539 | 1469.8 / 2.2394 |

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

## Reproduce

**Status: full** (statistical, not bit-exact). The commit, the exact binary build, the corpus manifest, the presets, the parameters file, the fresh starting checkpoints and the exact commands are all recorded.

### 1. Code and build

- **Commit:** `7a434ea`. The binary that ran is **build 2255**. The replay logs have no `[APP]` banner, and the `[REPLAY]` lines carry no build, so the number comes from every enumerated checkpoint's `__metadata__`: `built_by_build = 2255`, `built_by_git = 5826e1c`.
  - `5826e1c` was HEAD when the binary was built, but the working tree was dirty: it already held the trainer-config change committed minutes later as `cbc1894` (14:37:04).
  - `7a434ea` is `cbc1894` plus a `CHANGELOG.md` line, so `7a434ea` is the source that ran.
  - `BuildInfo.swift` at `7a434ea` shows the next build, 2256 (14:36:20, `gitDirty = true`).
  - The Release binary now in DerivedData was rebuilt later (2026-09-29 21:34), so it is **not** build 2255.
- **Prerequisites:**
  - An Apple Silicon Mac. The runs used a 64 GB machine with all three arms at once. The aborted 1M-buffer launch reportedly used about 24 GB per process (unverified, see Caveats), which is why the 500k buffer was used.
  - The Xcode that builds this project (on the original machine, `~/Downloads/Xcode_27_beta_2.app`).
  - Git LFS.
  - About 3 GB for the corpus, plus the source download described in the corpus manifest.
- **Build:**
  ```sh
  git clone <repo> drews-chess-machine && cd drews-chess-machine
  git checkout 7a434ea
  git lfs pull                          # fetches experiments/20260929-se-style-ab/models/*.safetensors
  ```
  Open `DrewsChessMachine/DrewsChessMachine.xcodeproj` and build the `DrewsChessMachine` scheme with the **Release** configuration (e.g. Edit Scheme ▸ Run ▸ Build Configuration = Release). The build number will differ from 2255 because `build_counter.txt` bumps on every build. That is expected; the source is what matters.
  ```sh
  BIN=$(ls ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-*/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine)
  M=~/Library/Application\ Support/DrewsChessMachine/Models
  ```

### 2. Corpus

`20260624-192615-w3aA5b`, the first 20,935,171 games of the lichess 2026-05 standard dump. Rebuild and verify it with **[../corpora/20260624-192615-w3aA5b.md](../corpora/20260624-192615-w3aA5b.md)**. It is a truncated import: rebuild it with `--max-games 20935171`, then check the shard body hashes. Replay reads the games in corpus order: the prefill logged `gamesFed=3778` at `bufCount=250012`, and the 30k checkpoints all carry `replay_next_game_index = 3867363`. A corpus with identical bodies therefore feeds the identical game stream.

### 3. Inputs in this folder

- `presets/test_SE_{scale+bias,attenuate-only,none}.json`: the architecture of each arm. They differ only in `se_style` and `label`.
- `parameters.json`: the exact file passed with `--parameters`. It has `replay_buffer_capacity: 500000` and `replay_buffer_min_positions_before_training: 250000`, and the startup log confirms them (`bufCap=500000 minPrefill=250000`). The full resolved set is the `[REPLAY-HPARAMS]` / `[REPLAY-CYCLE]` line of each log, quoted under Audit notes.
- `models/*-fresh.safetensors` (Git LFS): the untrained starting nets actually used. Their ModelIDs are `20260929-12-JZOe` (scale+bias), `20260929-13-06yp` (attenuate-only) and `20260929-18-D9is` (none). `models/*-replay-step30000.safetensors` are the step-30000 results, and [MODELS.md](MODELS.md) has their SHA-256s.

Copy the fresh nets to where the commands expect them:

```sh
mkdir -p "$M" && cp experiments/20260929-se-style-ab/models/*-fresh.safetensors "$M/"
```

### 4. Launch (one process per arm, all three concurrently on one GPU)

These are the exact commands used, with `<arm>` = `scale+bias`, `attenuate-only` and `none`. Run them from the repo root:

```sh
"$BIN" --replay-corpus 20260624-192615-w3aA5b \
  --start-model "$M/20260929-test_SE_<arm>-fresh.safetensors" \
  --out-model "$M/20260929-test_SE_<arm>-replay-latest.safetensors" \
  --parameters experiments/20260929-se-style-ab/parameters.json \
  --epochs 12 --enumerate-checkpoints
```

- The original three started 8 s apart (15:07:27 / :35 / :43).
- `--enumerate-checkpoints` writes `20260929-test_SE_<arm>-replay-step<N>.safetensors` every 1000 steps, next to the rolling `-replay-latest` file.
- The three arms shared one GPU, so steps per hour are not comparable to a solo run. Compare on step or `games_fed`, never on time.
- Check each startup log (`~/Library/Logs/DrewsChessMachine/dcm_log_*.txt`) against Audit notes. The `[REPLAY-ARCH]` param counts must be 5,208,050 / 5,195,378 / 5,170,322, `[REPLAY-HPARAMS]` must match, and `[REPLAY-CYCLE]` must be present.

**Minting fresh nets instead of using the stored ones:**

```sh
"$BIN" --new-model --architecture experiments/20260929-se-style-ab/presets/test_SE_<arm>.json \
  --out-model "$M/20260929-test_SE_<arm>-fresh.safetensors"
```

This makes a new random init with a new ModelID. There is no seed option, so no two mints are the same. Prefer the stored fresh checkpoints: the measured seed-to-seed spread for replay runs is **6.4–43.7 pElo** (nt8y seed study), which is as large as the gaps between arms here. Reusing the stored inits removes that source of variance from a rerun.

### 5. Probe and report

- **Probing:** `documentation/dashboards/registry.json` has the runs as `se_sb` / `se_att` / `se_none`, each with `enum_stem`.
  - After the checkpoints exist, run the following from `documentation/dashboards/`:
    ```sh
    python3 -c "import replay; [replay.probe_backfill(r) for r in ('se_sb','se_att','se_none')]"
    ```
    or run `python3 replay.py track <run>` periodically while training.
  - Each checkpoint is probed with `"$BIN" --probe-model <ckpt> --probe-set wide` (the 4,435-puzzle set) and the result goes to `data/<run>.csv`.
  - `replay.py` picks the newest Release binary under DerivedData; set `DCM_BIN` to pin one.
  - To rerun into fresh CSVs rather than the committed ones, point `DCM_DASH_ROOT` at a copy of `documentation/dashboards/`.
- **Report:** `python3 experiments/20260929-se-style-ab/make_report.py` regenerates `REPORT-30k.md`, `report-30k.html` and the two SVG charts from those CSVs. It has two hardcoded paths, `ROOT` (the repo) and the scale+bias log path (`~/Library/Logs/DrewsChessMachine/dcm_log_20260929-150727.txt`, used for LR per step). Edit both for a rerun.

### 6. Expected exactness

- **Not bit-exact. The trainer is not deterministic,** for three reasons, all verified in source:
  - Minibatch sampling in `ReplayBuffer.sample` draws indices with unseeded `Int.random`.
  - The dropout RNG seed is `Int.random` (dropout is 0 here, so this one doesn't apply).
  - bf16 MPSGraph reductions on the GPU are not guaranteed order-stable.
- What *is* identical given the manifest-verified corpus is the fresh weights (stored) and the game stream fed into the buffer (corpus order).
- Expect curves that track closely but not exactly. Treat per-mark differences within the SE arms' own ±60–130 swings around the LR peak, and within the 6.4–43.7 seed band, as noise. Compare the trough marks (11k, 30k).
- Checkpoints from this build can't be `--resume-exact`ed (see Caveats), so an interrupted rerun must restart from the fresh net.

## Seed 1: end of run

All three seed-1 arms were stopped cleanly on 2026-09-30 (SIGINT right after an enumerated checkpoint, so each also wrote a stop save):

| arm | last 1k mark | pElo / nll | stop save | pElo / nll |
|---|---|---|---|---|
| scale+bias | 33000 | 1461.5 / 2.2611 | 33014 | 1452.8 / 2.2669 |
| attenuate-only | 33000 | 1484.1 / 2.2338 | 33012 | 1481.0 / 2.2369 |
| none | 32000 | 1477.5 / 2.2373 | 32036 | 1482.1 / 2.2376 |

Artifacts: gzipped logs in [`logs/`](logs/), final and stop-save checkpoints in [`models/`](models/) (Git LFS; see [MODELS.md](MODELS.md)). Full analysis at 30k: [REPORT-30k.md](REPORT-30k.md).

Windowed view (mean gap per window, positive = no SE better):

| steps | LR phase | none − s+b pElo | none − att pElo | s+b − none nll | att − none nll |
|---|---|---|---|---|---|
| 1–6k | first peak, descending | +16.9 | +5.4 | +0.018 | +0.008 |
| 7–12k | trough 1 | +35.2 | +24.5 | +0.023 | +0.021 |
| 13–18k | rising | +36.3 | +15.5 | +0.035 | +0.020 |
| 19–24k | peak 2 | +46.0 | +27.8 | +0.067 | +0.049 |
| 25–31k | descending to trough 2 | +21.1 | +9.3 | +0.018 | +0.015 |

The gap is largest around the LR peak and shrinks at low LR; at trough 2 attenuate-only is within probe noise of no SE. SE's β term barely trains (β bias mean is structurally pinned at 0 by the block's LayerNorm; per-channel |β| ≈ 0.01; β weight row norms decay 0.465 → 0.37 under weight decay). **Correction (final report):** the "barely trains" reading was wrong. Measured by direction rather than norm, β's weights move off their random init about as much as γ's do, and about as much as a zero-initialized β grows; the random init simply stays on top of what is learned. See [REPORT-final.md › β learning dynamics](REPORT-final.md#beta-learning-dynamics).

## Seed 2

Question: does the seed-1 ordering (none ≥ attenuate-only > scale+bias, and SE instability near the LR peak) repeat with fresh random initializations? Target: run to ≥ 31k (covers the second peak and trough).

**Launch record:** 2026-09-30 10:41:01–10:41:17 CDT, all three at once, same corpus, same `parameters.json`, same flags as seed 1.

| arm | fresh ModelID | pid | log |
|---|---|---|---|
| scale+bias | `20260930-1-H1Oq` | 94122 | `dcm_log_20260930-104101.txt` |
| attenuate-only | `20260930-2-Gf9P` | 94143 | `dcm_log_20260930-104109.txt` |
| none | `20260930-3-V9zk` | 94158 | `dcm_log_20260930-104117.txt` |

Fresh nets: `models/20260929-test_SE_<arm>-seed2-fresh.safetensors`. Dashboard registry keys `se_sb2` / `se_att2` / `se_none2`.

**Engine differences from seed 1.** Seed 1 ran build 2255 (git `5826e1c` with uncommitted changes = the code committed as `cbc1894`/`7a434ea`). Seed 2 runs the Release binary built 2026-09-29 21:34 from `5f46da0` plus an uncommitted doc-comment edit (= the code in `badae6c`). Engine commits in between (`git log 7a434ea..badae6c -- DrewsChessMachine/DrewsChessMachine`):

| commit | change | can it affect a fresh corpus replay? |
|---|---|---|
| `d15f706` | Exact resume for GUI, corpus replay and train-vs-UCI: checkpoints now carry optimizer velocity, fp32 master weights and schedule keys; replay start logs `origin=…`; tau minimum 0.01; resumed sessions keep their own values | **Checkpoint contents and size only** (≈2× larger enumerated files). The training step math, LR/momentum cycle and warmup are unchanged for a fresh start (`origin=trainerStep 0 cycleStep 0`, confirmed in the seed-2 logs). Tau floors are self-play/arena only. |
| `badae6c` | Doc comment in `NetworkArchitecture.swift` | No. |

Seed-2 startup lines match seed 1: `wd=0.0003 bufCap=500000 minPrefill=250000`, cycle starts at LR 1e-1 / momentum 0.85.

## Seed 2: end of run

Stopped early by decision on 2026-09-30 at 15:05 CDT, right after every arm's 7000-step checkpoint (SIGINT, so each also wrote a stop save). Two seeds were judged enough to establish the direction; the zero-β arm below was the more informative next test.

| arm | 7000 pElo / nll | stop save | pElo / nll |
|---|---|---|---|
| scale+bias | 1261.4 / 2.4751 | 7282 | 1262.9 / 2.4772 |
| attenuate-only | 1264.0 / 2.4732 | 7289 | 1267.6 / 2.4684 |
| none | 1299.6 / 2.4501 | 7019 | 1284.6 / 2.4671 |

Artifacts: gzipped logs in [`logs/`](logs/), the 7000 checkpoints and stop saves in [`models/`](models/) (Git LFS; see [MODELS.md](MODELS.md)). Per-mark data: `documentation/dashboards/data/se_{sb,att,none}2.csv`.

### Seed 1 vs seed 2, marks 1k–7k

Paired pElo gaps within each seed (row arm minus column arm; mean with sd in parentheses, n = 7 marks per seed):

| comparison | seed 1 | seed 2 | pooled | range, both seeds | seed 1, full 1k–31k | nll, seed 1 / seed 2 |
|---|---|---|---|---|---|---|
| none − s+b | +18.7 (16.0) | +45.2 (36.3) | +31.9 (30.2) | −2.6 … +104.9 | +30.8 (21.2) | −0.0174 / −0.0628 |
| none − att | +9.2 (16.3) | +14.7 (50.7) | +11.9 (36.3) | −90.4 … +69.0 | +16.2 (22.6) | −0.0095 / +0.0090 |
| att − s+b | +9.5 (17.7) | +30.5 (32.2) | +20.0 (27.3) | −11.5 … +95.0 | +14.5 (28.4) | −0.0079 / −0.0718 |

Seed-to-seed spread (seed 2 − seed 1, same arm, same step, 21 pairs):
- Mean |gap| 21.6 pElo, median 19.6, RMS 27.9, max 66.1, so **one run's seed noise has an sd of about 20 pElo** (RMS ÷ √2). This matches the 6.4–43.7 band from the nt8y seed study.
- Per arm, seed 2 − seed 1: scale+bias −26.1 (sd 13.3), attenuate-only −5.1 (19.6), none +0.3 (36.5). Seed 2's scale+bias init sits consistently lower, which inflates seed 2's gaps against scale+bias.
- A difference between two single runs therefore carries about ±28 pElo (1 sd) of seed noise, about the size of the effects measured. Resolving an arm difference to about ±10 would take roughly 8–9 seeds per arm.

### Conclusions

- **Scale+bias SE is worse than both attenuate-only and no SE.** Both seeds agree on pElo and on nll, and seed 1's full run agrees.
- **No SE is probably at least as good as attenuate-only, and at worst similar.** The mean gap is positive in both seeds, but seed 2's sd is large and nll disagrees between seeds.
- **Limits:** the independent sample is two seeds. Marks within one run are strongly correlated, so they are not independent evidence. With two seeds, the defensible claim is the direction both agree on.
- **The early window understates the effect:** seed 1's 1k–7k none − s+b gap (+18.7) is about 60% of its full-run value (+30.8).

## Zero-β scale+bias (4th arm)

Question: is scale+bias worse because of the β path itself, or because β starts from random weights? In seed 1, β barely trained (see above), so its random init persisted as a fixed, input-dependent offset of about the size of the signal. Prediction: zero-β behaves like attenuate-only.

**Design:** each zero-β net is derived from that seed's scale+bias fresh net and is bit-identical to it except the β half of each SE fc2 weight (rows 128–255 of the [256, 32] weight, set to 0). The β biases were already 0 at init and stay 0; the γ half is unchanged (verified byte-for-byte on all three blocks of both nets). Each zero-β run is compared **paired** with its Glorot-β parent run at every mark 1k–7k, which removes most of the seed noise measured above. Target: 7000 steps, matching seed 2.

**Derivation** (from the repo root, with `BIN` and `M` as in Reproduce §1):

```sh
for s in "" "-seed2"; do
  "$BIN" --derive-model \
    --from "experiments/20260929-se-style-ab/models/20260929-test_SE_scale+bias${s}-fresh.safetensors" \
    --set-se-beta-init zero \
    --out "$M/20260929-test_SE_zerobeta${s:-"-seed1"}-fresh.safetensors"
done
```

The outputs record the parent ModelID and source SHA-256 in `derivation_history`. Stored copies are in `models/` (see [MODELS.md](MODELS.md)).

**Launch record:** 2026-09-30 15:05:44 and 15:05:52 CDT, both at once, same corpus, same `parameters.json`, same flags as the other arms:

```sh
"$BIN" --replay-corpus 20260624-192615-w3aA5b \
  --start-model "$M/20260929-test_SE_zerobeta-<seed>-fresh.safetensors" \
  --out-model "$M/20260929-test_SE_zerobeta-<seed>-replay-latest.safetensors" \
  --parameters experiments/20260929-se-style-ab/parameters.json \
  --epochs 12 --enumerate-checkpoints
```

| arm | fresh ModelID | parent | pid | log |
|---|---|---|---|---|
| zero-β seed 1 | `20260930-7-crxN` | `20260929-12-JZOe` | 39607 | `dcm_log_20260930-150544.txt` |
| zero-β seed 2 | `20260930-8-8qyR` | `20260930-1-H1Oq` | 39629 | `dcm_log_20260930-150552.txt` |

Dashboard registry keys `se_zb1` / `se_zb2`. Startup lines match the other arms: `wd=0.0003 bufCap=500000 minPrefill=250000`, `origin=trainerStep 0 cycleStep 0`, and the architecture line shows `SE+/4 β0`.

**Engine differences from seed 2.** The zero-β runs use the Release binary built 2026-09-30 11:18, build 2261 (git `31253d5` with uncommitted changes = the code of `8926221`, per `derivation_history`). Engine change since seed 2's `badae6c`: `8926221` (#7) adds the `se_beta_init` option, architecture format v4 and `--derive-model`. For a `glorot` β the graph and training math are unchanged. The only effect here is the intended one: β starts at 0.

**Throughput:** two arms share the GPU instead of three, so steps per hour are higher than seed 2's. Compare on step, never on time.

## Zero-β: end of run

Stopped by decision on 2026-09-30 at 17:15 CDT, right after both runs' 5000-step checkpoints (SIGINT, so each also wrote a stop save). All training for this experiment is finished.

| run | 5000 pElo / nll | stop save | pElo / nll |
|---|---|---|---|
| zero-β seed 1 (`20260930-9-RrGx`) | 1228.8 / 2.5233 | 5030 | 1240.2 / 2.5022 |
| zero-β seed 2 (`20260930-10-H51a`) | 1270.7 / 2.4707 | 5004 | 1250.5 / 2.4969 |

- Paired with its Glorot-β parent over 1k–5k, zero-β averaged +2.4 pElo (seed 1) and +26.3 (seed 2), with per-mark sd 39.2 and 45.9: no consistent effect of β init. At 5k it was the worst seed-1 arm and the best seed-2 arm.
- β learns from zero and is still growing at 5k; the Glorot β's learned (orthogonal-to-init) part is about the same size.

Artifacts: gzipped logs in [`logs/`](logs/), the 5000 checkpoints and stop saves in [`models/`](models/) (Git LFS; see [MODELS.md](MODELS.md)).

**Final write-up for the whole experiment: [REPORT-final.md](REPORT-final.md)** (styled: [report-final.html](report-final.html)). It has every run's per-mark table, the paired and seed-noise analysis, the β dynamics, charts in [`charts/`](charts/), and the data files in [`data/`](data/) with column descriptions in [data/README.md](data/README.md). Regenerate all of it with `python3 experiments/20260929-se-style-ab/make_final_report.py`. Per-tensor statistics of every checkpoint (mean / min / max and more, with findings): [TENSOR-STATS.md](TENSOR-STATS.md), regenerated by `tensor_stats.py` then `tensor_report.py`.
