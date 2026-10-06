# 2026-10-05 — LR schedule A/B: constant 0.01 vs a 1.0 ↔ 0.001 cycle (R7 shape, basic24)

**Status:** A/B/C complete; arm B-leaky complete (2026-10-05 20:41 → 2026-10-06 16:24, E-0019); arms B-leakyall and B-silu (since 23:44) running; C-leaky stopped 23:43. A/B/C ran 2026-10-05 01:32–18:51 CDT: both arms to their 36,000-step limit (~17:14), then, at the
owner's request, to trainer step 40,000 by an exact resume (see "Continuation to 40,000"). Arm C was stopped earlier.
Summary: [E-0017](../summaries/E-0017_2026-10-05_lr-schedule-ab.html).

## Results

Probes (`--probe-set wide`, policy only) every 1,000 trainer steps; B's LR at each probe from its `[REPLAY]` lines.
Full per-probe data: `probes-A.jsonl`, `probes-B.jsonl` (to 36k) and `probes-*-seg1.jsonl` (37k–40k, step + 36,000).

| step | B phase | A pElo | B pElo | B − A | Avg(R7,R8) | B − Avg |
|---:|---|---:|---:|---:|---:|---:|
| 6,000 | trough | 1096.5 | 1405.6 | +309.1 | 1284.6 | +121.0 |
| 16,000 | trough | 1314.0 | 1571.4 | +257.4 | 1335.6 | +235.7 |
| 26,000 | trough | 1349.0 | 1609.4 | +260.3 | 1455.9 | +153.5 |
| 31,000 | peak | 1395.8 | 1372.7 | −23.1 | 1485.9 | −113.2 |
| 33,000 | falling | 1362.4 | 1578.5 | +216.2 | 1504.6 | +73.9 |
| 36,000 | trough | 1453.8 | 1620.7 | +166.8 | | |
| 38,000 | rising | 1427.1 | 1632.0 | +204.8 | | |
| 40,000 | rising (LR 0.5) | 1396.8 | 1507.7 | +110.9 | | |

- Best probe: B 1632.0 (NLL 2.0728) at 38k; A 1453.8 (NLL 2.2620) at 36k. B is ahead at 39 of 40 probes (the
  exception, 31k, is at an LR peak).
- B's best probe per low-LR stretch: 1409.2 (8k), 1581.6 (17k), 1609.4 (26k), 1632.0 (38k) — gains +172, +28, +23.
- A's probe-to-probe noise is ±50–100 (e.g. 1402.0 at 24k, 1289.8 at 28k).
- Avg(R7,R8) differs in input encoding (basic30), seeds and cycle (0.1 ↔ 0.001, 20k period, troughs 11k / 31k), so
  B − Avg mixes those effects; it shows the 1.0 peak is at least not worse than R7/R8's schedule.
- C: diverged at LR ≈ 3 (step 300), never recovered (see "Arm C").
- B lost 6 of 16 value-head BN channels at its first LR peak (5 at steps 250–300, a sixth at 1,250–1,300); they
  stayed dead. The policy probes do not measure the value head.
- Summary page: E-0017.

## BN health across activations (`bn_liveness.py`)

`[LAYER-HEALTH]`'s dead / mostly-off / always-on counts come from β/|γ| and are defined only for ReLU and leaky ReLU
(SiLU and GELU sites print n/a). `bn_liveness.py` reads the enumerated checkpoints and reports, per BN site that feeds
an activation, the expected gradient pass-through under a Gaussian input and the "parked" (below Φ(−3) above the
activation's floor) and "mostly off" counts, per activation and never summed across activations. For ReLU and leaky
ReLU these equal `[LAYER-HEALTH]`'s dead and mostly-off counts exactly; for SiLU they are the same line on the
activation's own derivative. Each checkpoint's activations come from its own architecture (`scripts/dcm_arch.py`)
and its trainer step from its lineage record (`scripts/dcm_lineage.py`).

```
python3 experiments/20261005-lr-schedule-ab/bn_liveness.py                 # every checkpoint of every arm
python3 experiments/20261005-lr-schedule-ab/bn_liveness.py --steps 1000,6000 --site value.bn
python3 experiments/20261005-lr-schedule-ab/bn_liveness.py --selftest
```

## Question

The 33k tests used an LR cycle (0.1 ↔ 0.001, 20k period). Our best long runs (v5 1770.5, qeu8
1742.1) used a constant 0.01 (E-0012). Within a cycle pElo stalls at high LR and gains on the way
down (E-0008). Does cycling beat a constant 0.01, and can a much higher peak (1.0) give a faster
start? Owner: run B to 36k unless it blows up badly within the first 10k.

## Design

- **One start net for both arms:** `r7_basic24.json` (R7's shape with the 24-plane input; format v8,
  standard init), minted on build 2320 with `--init-seed 20261005` →
  `20261005-r7b24-fresh.safetensors`, ModelID `20261005-22-yRzB`, 5,132,687 parameters
  (`mint.txt`).
- **Arm A** (`parameters-A.json`): constant LR 0.01 after the 1,000-step warmup; momentum 0.9;
  LR and momentum cycles off.
- **Arm B** (`parameters-B.json`): LR cycle peak 1.0, trough 0.001, period 10,000 (inverted: starts
  at the peak after warmup, so peaks at 1k, 11k, 21k, 31k and troughs at 6k, 16k, 26k, 36k);
  momentum follows the cycle inversely (0.85 at the peak → 0.95 at the trough); decay horizon
  10,000,000 steps, peak end 1e-4, trough end 1e-5 (by step 36,000 the peak bound is 3.2% and the trough bound 1.6% below
  their start values).
- **Shared:** both files are `../20261004-fatconv-1x15x15-98/parameters-continue.json` (R7/R8's
  parameters with the batch sampler at its nearest uniform equivalent, since build 2320 applies the
  constraints build 2275 ignored) with only the LR/momentum keys above changed. Build 2320
  (`DCM-2320-1ab52554`), corpus `20260624-192615-w3aA5b`, `--seed 20261005` for both (same
  sampler stream, so the arms see identical batches), 12 epochs, 36,000-step limit,
  `--enumerate-checkpoints`, `--policy-tail-precision fp32_from_pre_bn`. Both arms share the GPU,
  so compare on step, not time.
- **Comparator:** Avg(R7,R8) — basic30, other init seeds, LR cycle 0.1 ↔ 0.001 (20k), build 2275.
  basic24 drops only planes that can never be 1, so it is not a training variable.
- **Probes:** `experiments/probe_loop.sh` with build 2320 into `probes-A.jsonl` / `probes-B.jsonl`; `probes-*.errors/`
  holds each probe's stderr as kept by the loop (a startup banner only; no failed probes).

## Launch record

- Launched 2026-10-05 01:32:20 (A, pid 35080, log `dcm_log_20261005-013220-2.txt`) and 01:32:35
  (B, pid 35145, log `dcm_log_20261005-013235.txt`) by a local chain script after the timing
  benchmark (E-0011). Both `[RUN]` lines: `seed=20261005 mode=seeded(--seed)`.
- `[REPLAY-HPARAMS]` matches R7/R8's apart from the LR/momentum keys; `[REPLAY-CYCLE]`: A `lr=off
  mom=off`; B `lr=[trough 1.00e-03,peak 1.00e+00]^10000st … momFollow=[low 0.850->0.900 high
  0.950->0.950]`. Step 1 loss is identical in both arms (11.8138), as expected for one net and one batch.

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
E=experiments/20261005-lr-schedule-ab
"$BIN" --new-model --architecture $E/r7_basic24.json --init-seed 20261005 --out-model "$M/20261005-r7b24-fresh.safetensors"
for arm in A B; do stem=$([ $arm = A ] && echo 20261005-lrA-const01 || echo 20261005-lrB-cyc1)
  "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-r7b24-fresh.safetensors" \
    --out-model "$M/$stem-replay-latest.safetensors" --parameters $E/parameters-$arm.json --epochs 12 \
    --training-step-limit 36000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 &
done
PROBE_BIN="$BIN" experiments/probe_loop.sh 20261005-lrA-const01 $E/probes-A.jsonl &
PROBE_BIN="$BIN" experiments/probe_loop.sh 20261005-lrB-cyc1 $E/probes-B.jsonl &
```

## Continuation to 40,000 (owner, 2026-10-05)

- Owner: "allow them to go to 40k, then stop there". The step limit is a launch flag, so each arm ran to its
  36,000-step final save and was then continued with `--resume-exact` from that save
  (`<stem>-replay-latest.safetensors`, header `training_step` 36000 checked before resuming), same build 2320
  binary, same parameters file and `--seed`, `--training-step-limit 4000` (the limit counts the segment's own
  steps), new out stems `20261005-lrA-const01-r1` / `20261005-lrB-cyc1-r1`.
- A: log `dcm_log_20261005-171451.txt`, started 17:14:51. B: log `dcm_log_20261005-171541.txt`, started 17:15:41.
  Both `[RESUME] EXACT`, `[RUN] … seg=1 (exact resume of …)`.
- Enumerated files are `<stem>-r1-replay-seg1-step<N>`; probes in `probes-A-seg1.jsonl` / `probes-B-seg1.jsonl`,
  whose `training_step` is the segment step: trainer step = segment step + 36,000.
- Known gap: this build logs the `[REPLAY]` step lines on segment steps, so after the resume none of them lands on a
  diagnostics step and their `pEnt` / `pW` / `pD` / `pL` / `vAbs` / `pLogitMean` / `vLogitMean` fields print `--`
  (`documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md`). Loss, illegal mass, gNorm, LR, momentum,
  probes and `[LAYER-HEALTH]` are unaffected.
- At 40,000 B's cycle is near LR 0.5 (rising toward its 41,000 peak), so B's 40k value is not a trough-comparable
  point; its troughs are 6k, 16k, 26k, 36k.

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
E=experiments/20261005-lr-schedule-ab
for arm in A B; do stem=$([ $arm = A ] && echo 20261005-lrA-const01 || echo 20261005-lrB-cyc1)
  "$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/$stem-replay-latest.safetensors" --resume-exact \
    --out-model "$M/$stem-r1-replay-latest.safetensors" --parameters $E/parameters-$arm.json --epochs 12 \
    --training-step-limit 4000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 &
done
# then, per arm, with the trainer's pid:
PROBE_BIN="$BIN" PROBE_SEGMENT=1 TRAINER_PID=<pid> experiments/probe_loop.sh <stem>-r1 $E/probes-<arm>-seg1.jsonl &
```

## Arm B-leaky (added 2026-10-05 20:41, owner)

- Owner: "re-run B but with leaky relu where it counts ASAP". B's damage was confined to the value head (at 40k:
  `value.bn` 6 of 16 channels dead, `value.fc1` 27 of 128 hidden units at zero velocity; tower, tower end and policy
  head clean), so B-leaky is B with `value_head_conv_activation` and `value_head_fc1_hidden_activation` set to
  `leaky_relu` (slope 0.01). Everything else is B's: `parameters-B.json`, `--seed 20261005`, same flags, but a
  40,000-step limit in one segment.
- Architecture `r7_basic24_leakyvalue.json` (format v9, the per-site activations of HEAD_ACTIVATIONS_PLAN phase 1).
  Start net `20261005-r7b24-leakyvalue-fresh.safetensors`, ModelID `20261006-5-AUSd`, `--init-seed 20261005`: its 61
  tensors are byte-identical to B's start net `20261005-22-yRzB` (fresh trainables do not depend on activation), so
  only the two value-head activations differ.
- Build: phase-1 commit `4e70c615` plus in-progress phase-2 derive-setter edits (not on the training or mint path),
  so `dirty=true`. Frozen as `FrozenBuilds/DCM-2331-4e70c615-p1headact.app` (binary sha256 prefix `1ae960792717`);
  **the folder name says 2331, but the binary reports build 2330** in its `[RUN]` line — the counter had already
  advanced when it was copied. Identify it by the sha, not the name.
- Launched 20:41:08, log `dcm_log_20261005-204108.txt`, stem `20261005-lrBleaky-cyc1`, probes `probes-Bleaky.jsonl`.
  Shares the GPU with test runs of the per-site-activation implementation, which slows it but does not change its math.
  Step-1 loss 11.8118 (B: 11.8138; the value head's activation changes the value loss from step 1).

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2331-4e70c615-p1headact.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
E=experiments/20261005-lr-schedule-ab
"$BIN" --new-model --architecture $E/r7_basic24_leakyvalue.json --init-seed 20261005 --out-model "$M/20261005-r7b24-leakyvalue-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-r7b24-leakyvalue-fresh.safetensors" \
  --out-model "$M/20261005-lrBleaky-cyc1-replay-latest.safetensors" --parameters $E/parameters-B.json --epochs 12 \
  --training-step-limit 40000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005 &
PROBE_BIN="$BIN" TRAINER_PID=<pid> experiments/probe_loop.sh 20261005-lrBleaky-cyc1 $E/probes-Bleaky.jsonl &
```

### Results (finished 2026-10-06 16:24)

- Ran to trainer step 40,000 in one segment and exited cleanly (`[REPLAY] done: steps=40000 positionsFed=341561598
  gamesFed=5152480`, exit code 0, 16:24:09); the final save (`20261005-lrBleaky-cyc1-replay-latest`, step 40,000,
  ModelID `20261006-6-Xxvl`) succeeded, and all 40 enumerated checkpoints and probes exist. Wall time 19 h 43 min,
  sharing the GPU throughout (with the per-site-activation test runs, then with B-leakyall and B-silu), so its step
  time is not comparable with B's.
- Probes (`probes-Bleaky.jsonl`), with A and B from `probes-A*.jsonl` / `probes-B*.jsonl` (trainer step); LR is
  B-leaky's own `[REPLAY]` value at that step (the same schedule as B's):

| step | LR | A pElo | B pElo | B-leaky pElo | B-leaky − B | B NLL | B-leaky NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1 | 824.4 | 1018.2 | 1004.8 | −13.4 | 2.8753 | 2.9418 |
| 2,000 | 0.517 | 932.1 | 1251.6 | 1234.0 | −17.6 | 2.5588 | 2.5557 |
| 3,000 | 0.0918 | 967.1 | 1321.7 | 1331.0 | +9.3 | 2.4466 | 2.4359 |
| 4,000 | 0.0109 | 1026.2 | 1391.2 | 1383.0 | −8.2 | 2.3744 | 2.3787 |
| 5,000 | 0.00193 | 1049.6 | 1402.5 | 1388.1 | −14.4 | 2.3516 | 2.3573 |
| 6,000 | 0.000998 | 1096.5 | 1405.6 | 1388.1 | −17.5 | 2.3491 | 2.3577 |
| 7,000 | 0.00193 | 1121.7 | 1395.8 | 1391.7 | −4.1 | 2.3491 | 2.3547 |
| 8,000 | 0.0108 | 1149.9 | 1409.2 | 1395.8 | −13.4 | 2.3314 | 2.3446 |
| 9,000 | 0.0914 | 1154.0 | 1372.2 | 1366.5 | −5.7 | 2.3896 | 2.3981 |
| 10,000 | 0.513 | 1148.8 | 1384.0 | 1349.5 | −34.5 | 2.3439 | 2.4340 |
| 11,000 | 0.991 | 1205.5 | 1320.7 | 1240.2 | −80.5 | 2.4775 | 2.5424 |
| 12,000 | 0.512 | 1224.1 | 1430.7 | 1395.8 | −34.9 | 2.3321 | 2.3933 |
| 13,000 | 0.0911 | 1267.1 | 1554.9 | 1531.3 | −23.6 | 2.1679 | 2.1987 |
| 14,000 | 0.0108 | 1240.2 | 1557.5 | 1554.9 | −2.6 | 2.1639 | 2.1616 |
| 15,000 | 0.00192 | 1280.5 | 1566.2 | 1557.0 | −9.2 | 2.1507 | 2.1702 |
| 16,000 | 0.000993 | 1314.0 | 1571.4 | 1567.8 | −3.6 | 2.1442 | 2.1557 |
| 17,000 | 0.00192 | 1305.2 | 1581.6 | 1574.4 | −7.2 | 2.1427 | 2.1503 |
| 18,000 | 0.0108 | 1289.3 | 1573.9 | 1571.9 | −2.1 | 2.1437 | 2.1493 |
| 19,000 | 0.0907 | 1274.3 | 1558.5 | 1532.9 | −25.7 | 2.1444 | 2.1827 |
| 20,000 | 0.508 | 1295.4 | 1471.3 | 1471.8 | +0.5 | 2.2335 | 2.2656 |
| 21,000 | 0.982 | 1324.3 | 1332.5 | 1365.5 | +32.9 | 2.4590 | 2.3874 |
| 22,000 | 0.508 | 1305.2 | 1372.2 | 1396.8 | +24.7 | 2.3624 | 2.3339 |
| 23,000 | 0.0904 | 1329.5 | 1573.4 | 1589.8 | +16.4 | 2.1207 | 2.1131 |
| 24,000 | 0.0107 | 1402.0 | 1597.5 | 1611.4 | +13.9 | 2.1257 | 2.1139 |
| 25,000 | 0.00191 | 1338.7 | 1604.7 | 1600.1 | −4.6 | 2.1138 | 2.1159 |
| 26,000 | 0.000989 | 1349.0 | 1609.4 | 1606.8 | −2.6 | 2.1084 | 2.1099 |
| 27,000 | 0.00191 | 1394.3 | 1603.7 | 1607.3 | +3.6 | 2.1134 | 2.1080 |
| 28,000 | 0.0107 | 1289.8 | 1598.6 | 1618.1 | +19.5 | 2.1100 | 2.1021 |
| 29,000 | 0.09 | 1380.9 | 1596.0 | 1609.9 | +13.9 | 2.1360 | 2.1074 |
| 30,000 | 0.504 | 1347.0 | 1504.6 | 1376.3 | −128.4 | 2.2333 | 2.3738 |
| 31,000 | 0.973 | 1395.8 | 1372.7 | 1379.4 | +6.7 | 2.3466 | 2.3890 |
| 32,000 | 0.503 | 1323.3 | 1431.3 | 1394.3 | −37.0 | 2.3024 | 2.3829 |
| 33,000 | 0.0897 | 1362.4 | 1578.5 | 1550.8 | −27.7 | 2.1375 | 2.1463 |
| 34,000 | 0.0107 | 1329.5 | 1614.0 | 1613.5 | −0.5 | 2.0942 | 2.0962 |
| 35,000 | 0.0019 | 1396.3 | 1628.4 | 1628.9 | +0.5 | 2.0774 | 2.0825 |
| 36,000 | 0.000984 | 1453.8 | 1620.7 | 1626.3 | +5.7 | 2.0869 | 2.0830 |
| 37,000 | 0.0019 | 1423.0 | 1627.8 | 1631.4 | +3.6 | 2.0793 | 2.0759 |
| 38,000 | 0.0106 | 1427.1 | 1632.0 | 1641.7 | +9.8 | 2.0728 | 2.0664 |
| 39,000 | 0.0893 | 1381.4 | 1583.7 | 1582.1 | −1.5 | 2.1230 | 2.1348 |
| 40,000 | 0.5 | 1396.8 | 1507.7 | 1469.2 | −38.5 | 2.2211 | 2.2123 |

- Best probe: B-leaky 1641.7 (NLL 2.0664) at 38k; B 1632.0 (NLL 2.0728) at 38k, +9.8 pElo and −0.0064 NLL. Both
  arms' best NLL is at the same probe.
- Best probe per low-LR stretch (steps 4k–8k, 14k–18k, 24k–28k, 34k–38k): B-leaky 1395.8 / 1574.4 / 1618.1 /
  1641.7; B 1409.2 / 1581.6 / 1609.4 / 1632.0. At the troughs: 6k −17.5, 16k −3.6, 26k −2.6, 36k +5.7.
- Over the 20 low-LR probes the difference B-leaky − B averages −1.7 pElo (standard deviation 9.3, range −17.5 …
  +19.5; NLL +0.0026). Over the 9 high-LR probes (10k–12k, 20k–22k, 30k–32k) it averages −27.8 (standard deviation
  51.8, range −128.4 … +32.9). The two arms use the same seed and so the same batches; what differs is the value-head
  activation (and GPU nondeterminism).
- Value loss is the same in both arms: mean `vLoss` over steps 37,001–40,000 0.8032 (B-leaky) vs 0.8031 (B), A 0.8093;
  over 5k–7k 0.8136 vs 0.8135; over 25k–27k 0.8015 vs 0.8013. `pD` 0.07 in both at 40k, `vAbs` 0.250 vs 0.252.

#### Layer health at 40,000 (B-leaky vs B vs A)

From the `[LAYER-HEALTH] checkpoint replay-final` blocks (B-leaky `dcm_log_20261005-204108.txt`, B
`dcm_log_20261005-171541.txt`, A `dcm_log_20261005-171451.txt`) and `bn_liveness.py --steps 40000 --site <site>`, which
agree. Dead = β/|γ| < −3, mostly off = −3 … −2, always on > +3; pass-through = expected |f′| under a Gaussian input.
The stem BN feeds no activation here (n/a).

| site | act (B-leaky) | dead / off B-leaky | dead / off B | dead / off A | min β/|γ| B-leaky | min β/|γ| B | min β/|γ| A | median pass-through B-leaky / B / A | rv max/median B-leaky / B / A |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| blocks.0.bn1 | relu | 0 / 0 | 0 / 0 | 0 / 0 | −0.98 | −1.18 | −0.13 | 0.304 / 0.295 / 0.516 | 9.3 / 9.4 / 2.3 |
| blocks.0.bn2 | relu | 0 / 0 | 0 / 0 | 0 / 0 | −1.61 | −1.61 | −0.21 | 0.210 / 0.218 / 0.493 | 3.5 / 4.5 / 2.1 |
| blocks.1.bn1 | relu | 0 / 0 | 0 / 0 | 0 / 0 | −1.75 | −1.60 | −0.38 | 0.185 / 0.199 / 0.497 | 95.6 / 37.1 / 6.2 |
| blocks.1.bn2 | relu | 0 / 1 | 0 / 0 | 0 / 0 | −2.19 | −1.92 | −0.38 | 0.166 / 0.159 / 0.496 | 8.9 / 7.5 / 1.7 |
| blocks.2.bn1 | relu | 0 / 1 | 0 / 5 | 0 / 0 | −2.51 | −2.53 | −0.41 | 0.136 / 0.139 / 0.494 | 36.0 / 32.6 / 3.8 |
| blocks.2.bn2 | relu | 0 / 0 | 0 / 0 | 0 / 0 | −1.59 | −1.99 | −0.36 | 0.244 / 0.247 / 0.526 | 3.5 / 3.7 / 1.5 |
| tower_final_bn | relu | 0 / 0 | 0 / 0 | 0 / 0 | −0.86 | −0.73 | −0.07 | 0.558 / 0.523 / 0.535 | 5.7 / 5.8 / 6.0 |
| policy.pre_bn | relu | 0 / 0 | 0 / 0 | 0 / 0 | −1.55 | −1.22 | −0.05 | 0.642 / 0.635 / 0.523 | 4.7 / 4.3 / 2.7 |
| value.bn | leaky_relu (B, A: relu) | 0 / 1 | 6 / 0 | 0 / 0 | −2.30 | −10.63 | −0.14 | 0.096 / 0.053 / 0.470 | 2.4 / 3.1 / 1.6 |

- Value FC1 hidden units (128): zero velocity 0 / 27 / 0 (B-leaky / B / A); low velocity (nonzero, < 5% of the layer's
  p90 unit norm) 15 / 10 / 37. No always-on channels and no non-finite values in any arm.
- Over the whole run (801 `[LAYER-HEALTH] live` lines) B-leaky never had a channel past β/|γ| = −3; its largest
  mostly-off count was 5 (first at step 39,350; worst site `blocks.2.bn1`, 3 of them). B had 5 dead `value.bn` channels
  by step 300 and 6 from step 1,300 to the end.

#### Interpretation

- Leaky ReLU in the value head did what it was meant to do there: `value.bn` 0 of 16 channels past −3 (B: 6), value
  FC1 0 of 128 units at zero velocity (B: 27). Its most negative channel is at −2.30 and its median pass-through 0.096,
  so the head is still pushed negative, but every channel keeps a gradient.
- That changed neither the value loss nor policy strength measurably. B's dead value channels cost no value loss
  (0.8031 vs 0.8032 late), and at low LR the policy probes differ by −1.7 ± 9.3 pElo on average. The +9.8 at 38k is
  inside that spread; the high-LR probes differ more (±52) because a probe taken mid-peak depends on where in an
  unstable stretch it lands. One seed per arm.
- The tower (ReLU in both arms) drifts the same way under the cycle in both: by 40k the six block BNs' median β is −0.45 to
  −0.78 and `blocks.2.bn1`'s most negative channel is about −2.5 in both, against A's −0.41.
  `blocks.1.bn1`'s running-variance max/median is 95.6 in B-leaky (B 37.1, A 6.2), one channel (14) in both cycled
  arms. The LR cycle, not the value-head activation, drives this; B-leakyall and B-silu test the tower activation.
- The policy probes do not measure the value head, so a value-only change was not expected to move them; the
  value-head result above is the measured one.

Reproduce the analysis (read-only; the training and probe commands are above):

```
python3 experiments/20261005-lr-schedule-ab/bn_liveness.py --steps 40000                    # per-run summary, A / B / B-leaky
python3 experiments/20261005-lr-schedule-ab/bn_liveness.py --steps 40000 --site value.bn    # one site; repeat per site
grep -A16 'LAYER-HEALTH\] checkpoint replay-final' ~/Library/Logs/DrewsChessMachine/dcm_log_20261005-204108.txt
```

## Arms B-leakyall and B-silu (added 2026-10-05 23:44, owner)

- Owner: "start B leaky everywhere and B leaky+silu", after B's `blocks.2.bn1` was seen drifting toward the dead line
  across LR cycles (worst β/|γ| −1.17 → −2.55, 5 channels mostly off by 36k; the whole layer's median β −0.18 → −0.75)
  while A (constant 0.01) barely moved (−0.41 at 40k), including at matched NLL.
- **B-leakyall**: `r7_basic24_leakyall.json` — leaky ReLU at every activation (blocks, tower end, policy head, value conv,
  value FC1); start net `20261005-r7b24-leakyall-fresh.safetensors` (`20261006-40-7URa`, shared read-only with C-leaky's
  start). Stem `20261005-lrBleakyall-cyc1`, log `dcm_log_20261005-234434.txt`, probes `probes-Bleakyall.jsonl`.
- **B-silu**: `r7_basic24_silublocks_leakyheads.json` — SiLU in the block group and at the tower end (the tower), leaky ReLU
  in the policy head, value conv and value FC1 (the heads). The tower-end choice is mine: it is the tower's last
  activation, so it follows the blocks. Start net `20261005-r7b24-silublocks-leakyheads-fresh.safetensors`
  (`20261006-42-Lzj4`). Stem `20261005-lrBsilu-cyc1`, log `dcm_log_20261005-234437.txt`, probes `probes-Bsilu.jsonl`.
- Both: `parameters-B.json`, `--seed 20261005`, 40,000 steps, same flags and frozen build as B-leaky. Every trainable
  tensor is byte-identical to B's start net; only the 16 BN running-statistics tensors differ (mint-time calibration
  through the different activations). `[LAYER-HEALTH]` does not classify SiLU sites (dead/off/on = n/a), so B-silu's
  tower health is read from β/|γ| ranges and running-variance ratios instead; `bn_liveness.py` reports SiLU sites'
  pass-through.
- Three runs (B-leaky, B-leakyall, B-silu) and the implementation's test runs share the GPU.

## Arm C-leaky (added 2026-10-05 23:13, owner)

- Owner: "let's do a leaky version of C with the crazy high LR schedule". C's damage was not confined to the value head
  (339 of 1,040 BN-fed channels dead by step 513, policy pre-BN 91 of 128), so C-leaky uses `leaky_relu` (slope 0.01)
  at every activation: the block group (main path; its unused SE activation set to match, as the build's v9 rule
  requires), tower end, policy head, value-head conv and value FC1. Stem and feature skip have no activation here.
  Everything else is C's: `parameters-C.json` (cycle 10 ↔ 0.01), `--seed 20261005`, same flags; 40,000-step limit.
- Architecture `r7_basic24_leakyall.json` (format v9). Start net `20261005-r7b24-leakyall-fresh.safetensors`, ModelID
  `20261006-40-7URa`, `--init-seed 20261005`: every trainable tensor is byte-identical to C's start net
  `20261005-22-yRzB`; only the BN running means/variances differ (by about 1%), because the mint's calibration
  forward pass runs through the leaky activations.
- Build: the same frozen binary as B-leaky (`FrozenBuilds/DCM-2331-4e70c615-p1headact.app`, binary build 2330, sha256
  prefix `1ae960792717`). Launched 23:13:35, log `dcm_log_20261005-231335.txt`, stem `20261005-lrCleaky-cyc10`,
  probes `probes-Cleaky.jsonl`. Shares the GPU with B-leaky and with test runs.
- Result: it trained like C to LR 2 (step 200: loss 5.15, illegal-move mass 0.34), then broke at LR 3 differently from C.
  C went quiet (loss 8.6, gNorm 0.02); C-leaky exploded through the value head: value loss 0.93 → 6,622 (step 350) →
  32,846 (step 400); total loss 33,011 / 76,059 / 414,420 / 4,494,490 / 2,243,050 at steps 400 / 500 / 600 / 900 / 1,000;
  gNorm up to 393,032; illegal-move mass 0.95–0.97. Probe at step 1,000: pElo 497.2, NLL 4.2782 (B: 1018.2, 2.8753).
  Weights stayed finite throughout (0 non-finite values in 102 tensors at steps 1,000 and 1,175). BN-fed channels classified
  dead: 0 at step 50, 84 at 300, 367 at 400, 532 at 1,000, 543 at 1,175 (worst site policy pre-BN, 93 of 128).
- Conclusion: leaky ReLU everywhere does not make a 10-peak cycle survivable on this net and batch; it changes the
  failure from frozen (C) to runaway (C-leaky). The usable peak lies between 1 (B) and 3.
- **Stopped by the owner 2026-10-05 23:43** at step 1,175 (SIGINT, abort save `20261005-lrCleaky-cyc10-replay-latest`).

## Arm C (added 2026-10-05 09:04, owner)

- **Arm C** (`parameters-C.json`): identical to B except the cycle's peak 1.0 → **10** and trough
  0.001 → **0.01** (`diff parameters-B.json parameters-C.json` shows only those two keys). Same start
  net, same `--seed 20261005`, same 36,000-step limit. Stem `20261005-lrC-cyc10`; probes in
  `probes-C.jsonl`.
- **Build 2323** (`58e9f952` plus one uncommitted change: `LRCycleMax`'s declared range raised from
  `1.0e-7...1.0` to `1.0e-7...10.0` in `Training/TrainingParameters.swift`, so a peak of 10 loads).
  Frozen as `FrozenBuilds/DCM-2323-58e9f952-lrmax10.app` (binary sha256 prefix `621dfd1a33e9`). No
  training code differs from build 2320 beyond that range.
- Launched 2026-10-05 09:04:16 (log `dcm_log_20261005-090417.txt`) while A and B were at ~step 18,000,
  so all three now share the GPU. `[REPLAY-CYCLE]` reads `lr=[trough 1.00e-02,peak 1.00e+01]^10000st`;
  `[RUN]` `build=2323 seed=20261005 mode=seeded(--seed)`; step-1 loss 11.8138, identical to A and B.
  A local script stops C only on a non-finite loss.
- **Stopped 2026-10-05 09:22 at step 513** (SIGINT, clean abort save `20261005-lrC-cyc10-replay-latest`),
  applying the owner's rule for B ("run on unless it blows up badly"). It trained normally to LR 2 (step 200:
  loss 5.14, illegal-move mass 0.29), then diverged as the LR passed ~3: at step 300 loss 36.4, policy loss 34.1,
  illegal mass 0.997, gNorm 14.3. It did not recover: steps 400–500 had illegal mass 0.945–0.948, policy loss
  ~6.8 and gNorm 0.02–0.06. The abort save's layer health found 339 of 1,040 BN-fed channels dead and 101
  mostly off (worst: the policy pre-BN, 91 dead). No loss was non-finite, so the watchdog never fired.
  Conclusion: at this net and batch, a cycle peak of 10 (effective step LR/(1−μ) ≈ 20 already at LR 3)
  destroys the network; 1.0 does not.
- **Continued 2026-10-05 12:18 (owner: "see where it goes, or if we get NaNs").** `--resume-exact` from the
  step-513 abort save on build 2323 (`[RESUME] EXACT`, random streams restored), out stem
  `20261005-lrC-cyc10-r1` (enumerated files `…-r1-replay-seg1-step<N>`, probes in `probes-C-seg1.jsonl`; trainer step
  = segment step + 513), log `dcm_log_20261005-121841.txt`, no auto-stop. It passed its LR 10 peak and fell to the
  0.01 trough without recovering and without any non-finite value: probes at trainer steps 1,513–5,513 stayed at
  pElo 582–585 / NLL 4.15, illegal-move mass ~0.967, gNorm ~0.01–0.02. Layer health at trainer step 5,513: 350 of
  1,040 BN-fed channels dead, policy pre-BN 91 of 128 dead, value BN 14 of 16 dead, all 128 value-FC1 units at
  exactly zero velocity; 0 non-finite values.
- **Stopped by the owner 2026-10-05 15:34** at trainer step 6,116 (SIGINT, abort save
  `20261005-lrC-cyc10-r1-replay-latest`).
