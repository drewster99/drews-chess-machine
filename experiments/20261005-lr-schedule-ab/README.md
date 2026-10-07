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

## Arm B-silu-clip1 (added 2026-10-06 17:00, owner)

- Owner: "Do a resume of silu from 18000 checkpoint with cap at 1.0". The question is whether a gradient-norm cap of 1.0
  stops B-silu's step-20,600 blowup: gNorm 2.57 at LR 0.88, illegal mass 0.80, 20 of 128 policy pre-BN channels
  parked, and pElo 1571.9 at 19k down to 457.4 at 21k. B-silu itself ran with cap 15, which never bound after step 1.
- `--resume-exact` from `20261005-lrBsilu-cyc1-replay-step18000.safetensors` (trainer step 18,000, before the blowup),
  `parameters-Bsilu-clip1.json` (`parameters-B.json` with `grad_clip_max_norm` 1.0, nothing else changed),
  `--accept-inexact params`, `--training-step-limit 22000` (the limit counts this segment's steps, so it ends at trainer
  step 40,000), `--seed 20261005`, same frozen build 2330 (`DCM-2331-4e70c615-p1headact.app`) and flags as B-silu.
  Stem `20261006-lrBsilu-clip1` (segment-1 names `-replay-seg1-step<N>`, N = trainer step − 18,000), log
  `dcm_log_20261006-170000.txt`, probes `probes-Bsilu-clip1-seg1.jsonl` (`PROBE_SEGMENT=1`).
- The resume logged `[RESUME] EXACT` even though `grad_clip_max_norm` changed: in this build the `params` gap only covers
  a checkpoint with no parameter snapshot, a changed per-step feed, or a changed buffer capacity. It does not compare the
  parameter values themselves. The run's `[REPLAY-HPARAMS]` line shows `gradClip=1`, and step 1 ran at lr 0.0108, B's
  LR at trainer step 18,001. The RNG streams (sampler, dropout) were restored, so up to the cap this run follows B-silu's
  feed and draws. Until gNorm first exceeds 1.0 it should track B-silu step for step (weights bit for bit only where
  MPSGraph steps are deterministic).

## Arm B-silu-ctl15: control for B-silu-clip1 (added 2026-10-06 18:15, my decision)

- B-silu-clip1 stopped tracking B-silu at trainer step 19,800, before its cap ever acted: its gNorm never exceeded 0.8.
  The logged loss/gNorm agree with B-silu at every 50-step line through 19,750, then differ at 19,800 (loss 3.5357 vs
  3.5356, gNorm 0.379 vs 0.332). That is run-to-run GPU nondeterminism with the GPU shared by three runs (CLAUDE.md:
  weights match bit for bit only where MPSGraph steps are deterministic, not on a shared GPU). The 20k probes already
  differ (B-silu 1390.2, clip1 1442.5), so a difference after 20,600 cannot be credited to the cap from these two runs
  alone. The step-20,600 blowup could itself depend on the exact trajectory.
- Control: the same `--resume-exact` from `20261005-lrBsilu-cyc1-replay-step18000.safetensors`, with the original
  `parameters-B.json` (cap 15, so no `--accept-inexact`), `--training-step-limit 5000` (to trainer step 23,000, past
  the blowup window), same build, flags and seed. Stem `20261006-lrBsilu-ctl15`, log `dcm_log_20261006-181510.txt`,
  probes `probes-Bsilu-ctl15-seg1.jsonl`. Reading the two: if the control blows up near 20,600 and clip1 does not, the
  cap is credited; if neither blows up, the original blowup depended on its trajectory; if both do, the cap did not
  prevent it.

### B-silu-ctl15 result at 19,800 (2026-10-06 ~19:55)

- The control (cap 15) logs exactly what B-silu logged at trainer step 19,800 (loss 3.5357, gNorm 0.379), and
  matches it at every 50-step line from 18,050. B-silu-clip1 differs there (loss 3.5356, gNorm 0.332). So the
  clip1/B-silu split is caused by the cap, not by GPU nondeterminism. The earlier note above ("shared-GPU
  nondeterminism") is superseded.
- gNorm is logged only every 50 steps, and every logged clip1 value is below 1.0. The cap must therefore have
  acted on an unlogged step between 19,751 and 19,799, where the pre-clip norm exceeded 1.0. B-silu's gradients
  are identical through 19,750, so B-silu very likely had the same unlogged spike, unclipped, about 800 steps
  before its logged blowup (gNorm 2.57 at 20,600).
- clip1 so far: 21k 1386.6 and 22k 1412.8 pElo, 0 policy pre-BN channels parked. B-silu at the same steps:
  457.4 / 833.8, 20 / 21 parked.

### B-silu-ctl15 reproduces the blowup; the cap prevented it (2026-10-06 ~20:55)

| trainer step | B-silu (cap 15) | ctl15 (cap 15, exact resume) | clip1 (cap 1.0, exact resume) |
|---:|---|---|---|
| 20,550 | loss 3.5416 · illegal 0.0078 · gNorm 0.363 | identical | loss 3.5256 · illegal 0.0072 · gNorm 0.368 |
| 20,600 | loss 4.1303 · illegal 0.0095 · gNorm 2.566 | identical | loss 3.5584 · illegal 0.0065 · gNorm 0.364 |
| 20,650 | loss 7.7890 · illegal 0.7988 · gNorm 0.264 | identical | loss 3.5245 · illegal 0.0075 · gNorm 0.364 |

- The control reproduces B-silu bit for bit through the blowup: the 19k and 20k probes are equal to the last digit,
  and every logged line matches. So the blowup is deterministic given the state at 18,000 and this feed. It is not
  GPU noise.
- The only difference between ctl15 and clip1 is `grad_clip_max_norm` (15 vs 1.0). clip1 diverged at an unlogged
  step between 19,751 and 19,799, the first step where the pre-clip norm exceeded 1.0. It never blew up: 21k 1386.6,
  22k 1412.8 pElo, 0 policy pre-BN channels parked. **A global-norm cap of 1.0 prevented the blowup.** Whether a
  looser cap would also have prevented it is untested.
- The logged gNorm (every 50 steps) never exceeded 0.44 in clip1 and showed nothing unusual before 20,600 in B-silu.
  The precursor spike in 19,751–19,799 is invisible at this logging cadence (see the alarms plan's per-window max).

### Gradient-cap experiment: results (B-silu-ctl15 finished 2026-10-06 21:57; B-silu-clip1 finished 2026-10-07 02:45; E-0020)

- **B-silu-ctl15 ended** by owner decision: run to its 22,000 probe and stop if it still matched B-silu. It matched, and the
  watcher sent SIGINT at ~21:57. Clean abort save at trainer step 22,030 (`20261006-lrBsilu-ctl15-replay-latest`, enumerated copy
  `-seg1-step4030`), `[REPLAY] done: steps=4030`, rc=0. ModelID `20261006-61-xecu`.
- **ctl15 is B-silu, bit for bit.**
  - Its `[REPLAY]` lines (loss, gNorm, pIllM) equal B-silu's on all 80 logged steps from 18,050 to 22,000.
  - The probe pElo is equal to the last digit at 19k, 20k, 21k and 22k (1571.876115055085, 1390.1540696698366, 457.37418729915294, 833.7791475626248).
  - NLL is equal at 19k and 22k and differs only in the 16th significant digit at 20k and 21k (summation order in the probe).
  - At 22,000 its checkpoint `[LAYER-HEALTH]` line is identical to B-silu's: 22 BN-fed channels dead, policy.pre_bn 21 dead / 8 off, min β/|γ| −95.53@policy.pre_bn[75], running-variance max/median 16,194.1@blocks.1.bn1[76].
  - So the step-20,600 blowup is deterministic given the step-18,000 state and the feed.
- **clip1 (cap 1.0) is the same run with one parameter changed** (`grad_clip_max_norm` 15 → 1.0).
  - Its lines equal B-silu's through 19,750 and first differ at 19,800, so the cap first acted on an unlogged step in 19,751–19,799.
  - gNorm is logged every 50 steps, and no logged clip1 value exceeds 1.0: the largest is 0.462, through trainer step 24,400.
  - It never blew up, and its 22,000 checkpoint has 0 dead channels (min β/|γ| −1.67@value.bn[10], running-variance max/median 95.4).

| trainer step | B-silu = ctl15: loss · gNorm · illegal mass | clip1: loss · gNorm · illegal mass |
|---:|---|---|
| 19,700 | 3.5525 · 0.353 · 0.0036 | 3.5525 · 0.353 · 0.0036 |
| 19,750 | 3.5110 · 0.327 · 0.0036 | 3.5110 · 0.327 · 0.0036 |
| 19,800 | 3.5357 · 0.379 · 0.0046 | 3.5356 · 0.332 · 0.0042 |
| 19,850 | 3.5672 · 0.335 · 0.0042 | 3.5520 · 0.341 · 0.0040 |
| 20,000 | 3.5435 · 0.349 · 0.0045 | 3.5360 · 0.353 · 0.0044 |
| 20,250 | 3.5836 · 0.408 · 0.0057 | 3.5802 · 0.392 · 0.0053 |
| 20,500 | 3.5427 · 0.409 · 0.0071 | 3.5407 · 0.425 · 0.0074 |
| 20,550 | 3.5416 · 0.363 · 0.0078 | 3.5256 · 0.368 · 0.0072 |
| 20,600 | 4.1303 · **2.566** · 0.0095 | 3.5584 · 0.364 · 0.0065 |
| 20,650 | 7.7890 · 0.264 · **0.7988** | 3.5245 · 0.364 · 0.0075 |

| trainer step | B (ReLU, cap 15) pElo | B-silu (= ctl15) pElo | clip1 pElo | clip1 NLL |
|---:|---:|---:|---:|---:|
| 19,000 | 1558.5 | 1571.9 | 1571.9 | 2.0859 |
| 20,000 | 1471.3 | 1390.2 | 1442.5 | 2.2475 |
| 21,000 | 1332.5 | 457.4 | 1386.6 | 2.3419 |
| 22,000 | 1372.2 | 833.8 | 1412.8 | 2.3677 |
| 23,000 | 1573.4 | 927.2 | 1567.8 | 2.1558 |
| 24,000 | 1597.5 | 951.9 | 1597.5 | 2.1158 |

`bn_liveness.py` per site at trainer step 22,000. Each cell is parked / mostly off / lowest pass-through. ctl15 is identical to
B-silu at 22,000; at its 22,030 abort save, value.bn has 1 more channel mostly off.

| site (act) | B-silu = ctl15 | clip1 |
|---|---|---|
| blocks.0.bn1 (silu) | 0 / 1 / 0.0215 | 0 / 0 / 0.3148 |
| blocks.0.bn2 (silu) | 0 / 0 / 0.0440 | 0 / 0 / 0.1900 |
| blocks.1.bn1 (silu) | 0 / 0 / 0.0349 | 0 / 0 / 0.1655 |
| blocks.1.bn2 (silu) | 0 / 0 / 0.0239 | 0 / 0 / 0.1398 |
| blocks.2.bn1 (silu) | 0 / 0 / 0.0318 | 0 / 0 / 0.0589 |
| blocks.2.bn2 (silu) | 0 / 0 / 0.0674 | 0 / 0 / 0.1159 |
| tower_final_bn (silu) | 0 / 0 / 0.2195 | 0 / 0 / 0.3327 |
| policy.pre_bn (leaky_relu) | 21 / 8 / 0.0100 | 0 / 0 / 0.1906 |
| value.bn (leaky_relu) | 1 / 4 / 0.0100 | 0 / 0 / 0.0566 |

- **What it shows:**
  - A global-norm cap of 1.0 prevented this blowup.
  - The cap cost no strength over 19k–24k: clip1 tracks B (ReLU, cap 15) within −28.8 … +54.1 pElo, and is equal at 24k.
  - The precursor is invisible in the logs: one or more steps in 19,751–19,799 had a pre-clip norm above 1.0, at least 2.8× the logged 0.33–0.38 around it, and the 50-step log shows nothing until 20,600.
- **The new training-health alarms** (P2, on branch `worktree-agent-a13d06e3f01f924f5` at the time of writing) raise `gradient_spike` at 20,600 when replaying B-silu's log. In the app they evaluate every trainer step's norm, not just the logged ones. Whether the unlogged precursor would have crossed their 5× threshold is not known: only "> 1.0" is known.
- **Not tested:**
  - whether a looser cap (2 or 5) also prevents it;
  - whether the cap changes the long-run result;
  - more than one seed or one start state.
- **clip1 finished** at trainer step 40,000 on its own at 02:45 on 2026-10-07: `[REPLAY] saved trainer model (final) step=22000
  trainerStep=40000`, `[REPLAY] done: steps=22000`, rc=0 (chain.log). Largest logged gNorm over the whole run: 0.462 at
  trainer step 21,150; no logged value exceeds the 1.0 cap from 18,050 to 40,000.

#### clip1 to 40,000 (added 2026-10-07)

| trainer step | B's LR | B (ReLU, cap 15) | B-silu (cap 15) | ctl15 (cap 15) | clip1 (cap 1.0) | clip1 NLL | clip1 − B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19,000 | 0.0907 | 1558.5 | 1571.9 | 1571.9 | 1571.9 | 2.0859 | +13.3 |
| 20,000 | 0.508 | 1471.3 | 1390.2 | 1390.2 | 1442.5 | 2.2475 | −28.7 |
| 21,000 | 0.982 | 1332.5 | 457.4 | 457.4 | 1386.6 | 2.3419 | +54.0 |
| 22,000 | 0.508 | 1372.2 | 833.8 | 833.8 | 1412.8 | 2.3677 | +40.6 |
| 23,000 | 0.0904 | 1573.4 | 927.2 | | 1567.8 | 2.1558 | −5.6 |
| 24,000 | 0.0107 | 1597.5 | 951.9 | | 1597.5 | 2.1158 | +0.0 |
| 25,000 | 0.00191 | 1604.7 | 953.5 | | 1596.5 | 2.1193 | −8.2 |
| 26,000 | 0.000989 | 1609.4 | 963.3 | | 1597.5 | 2.1141 | −11.8 |
| 27,000 | 0.00191 | 1603.7 | 968.7 | | 1595.0 | 2.1119 | −8.7 |
| 28,000 | 0.0107 | 1598.6 | 960.6 | | 1600.6 | 2.1072 | +2.1 |
| 29,000 | 0.09 | 1596.0 | 977.9 | | 1589.8 | 2.1669 | −6.2 |
| 30,000 | 0.504 | 1504.6 | 773.9 | | 1499.0 | 2.2147 | −5.6 |
| 31,000 | 0.973 | 1372.7 | 1013.4 | | 1336.1 | 2.4305 | −36.5 |
| 32,000 | 0.503 | 1431.3 | 1048.5 | | 1442.5 | 2.2687 | +11.3 |
| 33,000 | 0.0897 | 1578.5 | 1216.4 | | 1572.9 | 2.1500 | −5.6 |
| 34,000 | 0.0107 | 1614.0 | 1230.4 | | 1601.7 | 2.0968 | −12.3 |
| 35,000 | 0.0019 | 1628.4 | 1251.6 | | 1620.1 | 2.0785 | −8.2 |
| 36,000 | 0.000984 | 1620.7 | 1250.0 | | 1618.6 | 2.0855 | −2.1 |
| 37,000 | 0.0019 | 1627.8 | | | 1617.1 | 2.0796 | −10.8 |
| 38,000 | 0.0106 | 1632.0 | | | 1639.2 | 2.0746 | +7.2 |
| 39,000 | 0.0893 | 1583.7 | | | 1600.1 | 2.1212 | +16.4 |
| 40,000 | 0.5 | 1507.7 | | | 1475.4 | 2.2313 | −32.3 |

(B-silu was stopped at 36,066 by the owner, E-0021; its abort-checkpoint probe was 1247.4. ctl15 stopped at 22,030.)

- **Did clip1 end up matching B?** Yes, within noise at low LR. Over the ten low-LR probes (24–28k and 34–38k) clip1 − B
  averages −5.3 pElo (SD 6.7); its best probe is 1639.2 (NLL 2.0746) at 38,000 against B's best 1632.0 at 38,000. It passed
  the second LR peak (31k, LR 0.973) with no blowup: 1336.1 against B's 1372.7. At 40,000 (LR 0.5) it is 1475.4 against B's
  1507.7, the usual high-LR dip.
- **End-of-run layer health at 40,000** (`bn_liveness.py` per site; checkpoint `20261006-lrBsilu-clip1-replay-seg1-step22000`
  and B's `20261005-lrB-cyc1-r1-replay-seg1-step4000`). Cells: parked / mostly off / lowest pass-through.

| site | clip1 (act) | B (relu) |
|---|---|---|
| blocks.0.bn1 | 0 / 0 / 0.2621 (silu) | 0 / 0 / 0.1190 |
| blocks.0.bn2 | 0 / 0 / 0.1667 (silu) | 0 / 0 / 0.0542 |
| blocks.1.bn1 | 0 / 0 / 0.1422 (silu) | 0 / 0 / 0.0551 |
| blocks.1.bn2 | 0 / 0 / 0.1180 (silu) | 0 / 0 / 0.0275 |
| blocks.2.bn1 | 0 / 0 / 0.0474 (silu) | 0 / 5 / 0.0056 |
| blocks.2.bn2 | 0 / 0 / 0.1113 (silu) | 0 / 0 / 0.0231 |
| tower_final_bn | 0 / 0 / 0.2858 (silu) | 0 / 0 / 0.2338 |
| policy.pre_bn | 0 / 0 / 0.1537 (leaky_relu) | 0 / 0 / 0.1117 |
| value.bn | 0 / 0 / 0.0447 (leaky_relu) | 6 / 0 / 0.0000 |

  The app's own checkpoint line at 40,000: clip1 `dead=0 off=0`, running-variance max/median 153.6@blocks.2.bn1[31]; B
  `dead=6`, 37.1@blocks.1.bn1[14]. clip1 has no parked or mostly-off channel anywhere; its largest running-variance spread is
  larger than B's (no verdict: the alarm rule's threshold is 1,000).
- **Conclusion:** with a 1.0 cap the SiLU-tower arm survives both LR peaks, ends level with ReLU B in strength and with
  healthier BN statistics (no dead channels; B's value head has 6). A fixed 1.0 cap is not a general setting (it would bind
  hard early in training, where gNorm is ~30 at step 1 and 1–3 for the first ~500 steps); a relative cap (k × running median)
  is the proposed follow-up, not yet planned.
- **Reproduce the analysis:** `python3 -c` comparisons of the three logs' `[REPLAY]` lines (`dcm_log_20261005-234437.txt`, `dcm_log_20261006-170000.txt`, `dcm_log_20261006-181510.txt`); `bn_liveness.analyze()` on `20261005-lrBsilu-cyc1-replay-step22000`, `20261006-lrBsilu-ctl15-replay-seg1-step4000` / `-step4030` and `20261006-lrBsilu-clip1-replay-seg1-step4000`.

## Arm B-leakyall: result (finished at 40,000; E-0022)

- **Finished cleanly:** `[REPLAY] done: steps=40000 positionsFed=341561598 gamesFed=5152480` at 2026-10-07 00:12:40, final
  save at step 40,000 (`20261005-lrBleakyall-cyc1-replay-latest.safetensors`, ModelID `20261006-43-a89C`, run
  `A41BEB12…`), rc 0 (chain.log `LRAB Bleakyall ended rc=0`). Log `dcm_log_20261005-234434.txt`; 40 probes in
  `probes-Bleakyall.jsonl`. Wall time 24 h 28 min on a GPU shared with up to three other runs and the implementation's
  test suites.
- **Policy probes vs B (ReLU)**, Δ = B-leakyall − B pElo per 1,000-step probe:
  - low-LR probes (LR < 0.1, n = 28): mean −12.5, SD 13.0; high-LR probes (n = 12): mean −11.8, SD 59.2.
  - by LR trough: 6k −6.2, 16k −20.5, 26k −3.6, 36k −18.5.
  - by stretch: 13k–19k mean −24.2 (every probe −18.0 … −30.8); 24k–29k mean −0.6; 33k–39k mean −17.8.
  - first probe 1104.9 vs B 1018.2 (+86.7); best 1629.4 at 38k vs B 1632.0 and B-leaky 1641.7 (all three peak at 38k);
    40k (LR 0.5) 1446.1 vs B 1507.7 (−61.6) and B-leaky 1469.2.
- **vs B-leaky (leaky value head only):** low-LR mean −9.7, SD 14.7.
- **Value head:** same as B-leaky. `value.bn` 0 of 16 parked at 40k (B 6), value FC1 0 of 128 units at zero velocity
  (B 27). Every one of the 801 live `[LAYER-HEALTH]` lines reports `dead=0`. Value loss over the last 60 step lines
  (37,050–40,000) averages 0.8031, B's 37k–40k value 0.8031.
- **Tower at 40k** (`bn_liveness.py`; parked / mostly off / lowest pass-through; leaky sites use the leaky pass-through):

  | site | B-leakyall | B-leaky | B |
  |---|---|---|---|
  | blocks.0.bn1 | leaky 0 / 0 / 0.1631 | relu 0 / 0 / 0.1632 | relu 0 / 0 / 0.1190 |
  | blocks.0.bn2 | leaky 0 / 0 / 0.0752 | relu 0 / 0 / 0.0532 | relu 0 / 0 / 0.0542 |
  | blocks.1.bn1 | leaky 0 / 0 / 0.0536 | relu 0 / 0 / 0.0399 | relu 0 / 0 / 0.0551 |
  | blocks.1.bn2 | leaky 0 / 1 / 0.0217 | relu 0 / 1 / 0.0144 | relu 0 / 0 / 0.0275 |
  | blocks.2.bn1 | leaky 0 / 4 / 0.0134 | relu 0 / 1 / 0.0061 | relu 0 / 5 / 0.0056 |
  | blocks.2.bn2 | leaky 0 / 0 / 0.0342 | relu 0 / 0 / 0.0554 | relu 0 / 0 / 0.0231 |
  | tower_final_bn | leaky 0 / 0 / 0.2131 | relu 0 / 0 / 0.1952 | relu 0 / 0 / 0.2338 |
  | policy.pre_bn | leaky 0 / 0 / 0.0587 | relu 0 / 0 / 0.0606 | relu 0 / 0 / 0.1117 |
  | value.bn | leaky 0 / 1 / 0.0226 | leaky 0 / 1 / 0.0205 | relu 6 / 0 / ≈0 |

  Worst β/|γ| at `blocks.2.bn1`: B-leakyall −2.70, B-leaky −2.51, B −2.53. Running-variance max/median (`[LAYER-HEALTH]`
  checkpoint line at 40k, `blocks.1.bn1[14]` in all three): B-leakyall 101.8, B-leaky 95.6, B 37.1.
- **Reading:** leaky ReLU everywhere keeps the value-head benefit B-leaky already showed (no parked `value.bn` channel,
  no zero-velocity value FC1 unit), but the tower drifts the same way as B's: `blocks.2.bn1` ends with 4 channels mostly
  off and the most negative β/|γ| of the three arms, so the leaky slope does not stop that layer's β from walking toward
  the dead line across LR cycles. Policy strength is not better than ReLU B at any trough (mean −12.5 at low LR, about one
  SD); the gap is largest through the second cycle's trough (13k–19k) and gone at the third (24k–29k). One seed, one start.

## Arm B-silu: result (stopped at 36,066; E-0021)

- **Stopped by the owner** at 2026-10-06 23:10 ("terminate b silu. we haven't recovered any channels"): SIGINT, clean abort
  save at trainer step 36,066 (`[REPLAY] done: steps=36066 … gamesFed=4647312`, rc 0), log `dcm_log_20261005-234437.txt`.
  The probe loop probed the abort checkpoint (`…-replay-step36066`): pElo 1247.4, NLL 2.5185.
- **Before the blowup (1k–19k)** B-silu tracked B: mean −3.7 pElo over the 19 probes, ahead by 5.7–45.1 through 9k and at
  19k (1571.9 vs 1558.5, its best), behind by 41.6–54.5 at 10k–13k around the first LR peak, within −23.1 … −3.1 at
  14k–18k. NLL was 0.019–0.112 lower than B's through 9k.
- **After the blowup** (20,600, see the gradient-cap sections above) policy pre-BN parked channels (β/|γ| < −3) stayed at
  20 of 128 at every 1k checkpoint from 21k to 36,066 (21 at 22k); value.bn went 1 → 2 parked from 32k. pElo climbed
  457.4 (21k) → 951.9 (24k), then stalled at 951.9–977.9 through 29k, then 1013–1048 at 31–32k and 1216–1252 at 33–36k,
  ending 1250.0 at 36k vs B's 1620.7 (−370.7) and 1247.4 at the abort save.
- **Batch loss recovered but probe strength did not.** Logged loss at 36,050 was 3.5684 (pLoss 2.7551, vLoss 0.8075,
  illegal mass 0.0059, gNorm 0.291) against 3.5319 (pLoss 2.7390, illegal 0.0049) at 19,950 before the blowup, while
  pElo stayed ~320 below its 19k value.
- **Final per-site health** (`bn_liveness.py`, step 36,066 vs B at 36,000; parked / mostly off / lowest pass-through):

  | site | B-silu (36,066) | B (ReLU, 36,000) |
  |---|---:|---:|
  | blocks.0.bn1 | silu 0 / 1 / 0.0199 | relu 0 / 0 / 0.1229 |
  | blocks.0.bn2 | silu 0 / 0 / 0.0447 | relu 0 / 0 / 0.0483 |
  | blocks.1.bn1 | silu 0 / 0 / 0.0373 | relu 0 / 0 / 0.0638 |
  | blocks.1.bn2 | silu 0 / 0 / 0.0230 | relu 0 / 0 / 0.0246 |
  | blocks.2.bn1 | silu 0 / 0 / 0.0313 | relu 0 / 5 / 0.0053 |
  | blocks.2.bn2 | silu 0 / 0 / 0.0520 | relu 0 / 0 / 0.0235 |
  | tower_final_bn | silu 0 / 0 / 0.2249 | relu 0 / 0 / 0.2387 |
  | policy.pre_bn | leaky 20 / 5 / 0.0100 | relu 0 / 0 / 0.1184 |
  | value.bn | leaky 2 / 2 / 0.0100 | relu 6 / 1 / ≈0 |

  The SiLU tower itself has no parked channel, though several of its β/|γ| are extreme (min −1,562.7 at blocks.1.bn1,
  −422.6 at blocks.0.bn1); SiLU still passes at least 0.0199 of the signal at every one of those channels. The damage is in the policy head's pre-BN
  layer. Checkpoint `[LAYER-HEALTH]` at 36,066: dead=22 off=7 over the 144 relu/leaky channels, worst policy.pre_bn
  (dead 20 off 5), running-variance max/median 9,097 at blocks.1.bn1[76], no non-finite values.
- **Reading.** The SiLU tower trained as well as ReLU B up to 19k, but the run did not survive the second LR peak at
  cap 15, and 16k steps later the 20 parked policy pre-BN channels had not come back at any LR, including two
  troughs. E-0020 shows the same start with a 1.0 gradient cap avoids the blowup entirely. A SiLU-tower comparison with
  B therefore needs the cap; this arm answers only "SiLU tower at cap 15", and the answer is no.

## Arms B-silu-clip2 and B-silu-clip5: how loose can a fixed cap be? (added 2026-10-07 03:50)

- Owner question after E-0020: a fixed cap of 1.0 prevented B-silu's step-20,600 blowup, but it cannot be used everywhere: early-training gNorm is ~30 at step 1 and 1–3 for hundreds of steps, so 1.0 would clip nearly every early step.
  These two arms bracket the blowup's precursor (a pre-clip norm above 1.0 somewhere in 19,751–19,799, against a median near 0.35, and 2.566 logged at 20,600).
- Same recipe as B-silu-ctl15 and B-silu-clip1: `--resume-exact` from `20261005-lrBsilu-cyc1-replay-step18000.safetensors`, `--accept-inexact params`, `--training-step-limit 5000` (to trainer step 23,000), same frozen build 2330, flags and seed;
  `parameters-Bsilu-clip2.json` / `parameters-Bsilu-clip5.json` (`parameters-B.json` with only `grad_clip_max_norm` changed). Both resumes logged `[RESUME] EXACT`.
- Stems `20261006-lrBsilu-clip2` / `20261006-lrBsilu-clip5`; logs `dcm_log_20261007-035017-2.txt` (clip2), `dcm_log_20261007-035017.txt` (clip5); probes `probes-Bsilu-clip2-seg1.jsonl` / `probes-Bsilu-clip5-seg1.jsonl`.
- Reading: a cap that is never exceeded leaves the run bit-identical to B-silu (the ctl15 control), so blowup or not is decided by whether the precursor spike exceeds the cap. A relative cap (k × running median of recent pre-clip norms) is the general form; it is being planned separately.

### B-silu-clip2 and B-silu-clip5: results (both finished 2026-10-07 05:47; E-0023)

- **Both finished cleanly** at trainer step 23,000 (segment limit 5,000): clip2 `[REPLAY] done: steps=5000 positionsFed=43422410
  gamesFed=649460` at 05:46:51, final save `20261006-lrBsilu-clip2-replay-latest` (ModelID `20261007-38-EHwm`); clip5 the same
  counts at 05:47:04, `20261006-lrBsilu-clip5-replay-latest` (ModelID `20261007-38-hdjU`). Both are segment 1 of B-silu's lineage
  run `085EF48A…` (parent ModelID `20261006-44-8ipq`, sha `4697674c64a7`), `[RESUME] EXACT`, rc 0 (chain.log). Wall time 1 h 56 min
  each, sharing the GPU with each other and with test suites.
- **Neither blew up.** No logged line in either run shows a gNorm above 0.475 or an illegal-move mass above 0.0087 (B-silu at
  20,650: 0.7988), and neither has a parked or mostly-off policy pre-BN channel at 21k, 22k or 23k (B-silu: 20, 21, 20 parked).

| trainer step | B (ReLU, cap 15) | B-silu (cap 15) | ctl15 (cap 15) | clip1 (cap 1.0) | clip2 (cap 2.0) | clip5 (cap 5.0) |
|---:|---:|---:|---:|---:|---:|---:|
| 19,000 | 1558.5 | 1571.9 | 1571.9 | 1571.9 | 1571.9 | 1571.9 |
| 20,000 | 1471.3 | 1390.2 | 1390.2 | 1442.5 | 1462.1 | 1436.9 |
| 21,000 | 1332.5 | 457.4 | 457.4 | 1386.6 | 1373.2 | 1353.6 |
| 22,000 | 1372.2 | 833.8 | 833.8 | 1412.8 | 1457.9 | 1407.1 |
| 23,000 | 1573.4 | 927.2 | | 1567.8 | 1560.6 | 1567.3 |

  pElo on the wide probe set (4,435 puzzles); ctl15 stopped at 22,030. NLL at 23,000: B 2.1207, clip1 2.1558, clip2 2.1605,
  clip5 2.1574 (B-silu 3.0834).

- **Logged lines** (loss · gNorm before clipping · illegal-move mass):

| trainer step | B-silu (cap 15) | clip1 (cap 1.0) | clip2 (cap 2.0) | clip5 (cap 5.0) |
|---:|---|---|---|---|
| 19,750 | 3.5110 · 0.327 · 0.0036 | 3.5110 · 0.327 · 0.0036 | 3.5110 · 0.327 · 0.0036 | 3.5110 · 0.327 · 0.0036 |
| 19,800 | 3.5357 · 0.379 · 0.0046 | 3.5356 · 0.332 · 0.0042 | 3.5374 · 0.330 · 0.0040 | 3.5337 · 0.367 · 0.0042 |
| 20,550 | 3.5416 · 0.363 · 0.0078 | 3.5256 · 0.368 · 0.0072 | 3.5525 · 0.407 · 0.0059 | 3.5358 · 0.475 · 0.0063 |
| 20,600 | 4.1303 · **2.566** · 0.0095 | 3.5584 · 0.364 · 0.0065 | 3.5899 · 0.403 · 0.0068 | 3.5682 · 0.354 · 0.0064 |
| 20,650 | 7.7890 · 0.264 · **0.7988** | 3.5245 · 0.364 · 0.0075 | 3.5141 · 0.372 · 0.0080 | 3.5341 · 0.417 · 0.0068 |
| 23,000 | 3.8331 · 0.338 · 0.0462 | 3.5242 · 0.246 · 0.0020 | 3.5254 · 0.241 · 0.0021 | 3.5216 · 0.246 · 0.0021 |

- **The precursor was above 5.** clip2 and clip5 match B-silu on every 50-step line from 18,050 to 19,750 and first differ at
  19,800, the same place clip1 did; the three capped runs also differ from each other there. The ctl15 control showed that a cap
  that never binds leaves the run identical to B-silu bit for bit, so clip5 differing means some step in 19,751–19,799 had a
  pre-clip norm above 5 (the 19,750 line's own norm is 0.327, and the 19,800 line's loss already differs, so the update that
  changed the weights came between them). That is more than 13× the 0.33–0.38 logged around it, and no 50-step line showed it.
  B-silu's largest logged gNorm before 20,600 (18k–20,550) was 0.424.
- **What the comparison does and does not show.** Each capped run leaves B-silu's trajectory at 19,800, so their batches at
  20,600 meet different weights: none of them logs a spike there (0.364, 0.403, 0.354). The caps avoided the blowup; this does not
  show that a cap of 2 or 5 would have been enough at 20,600 itself had the 19,751–19,799 step gone through unclipped.
- **Largest logged gNorm, 18k–23k:** clip2 0.453 (21,350), clip5 0.475 (20,900), clip1 0.462 (21,150, whole run), B-silu 2.566
  (20,600).
- **Layer health** (`bn_liveness.py`, parked / mostly off / lowest pass-through at 23,000):

| site | clip2 | clip5 | clip1 |
|---|---|---|---|
| blocks.0.bn1 | silu 0 / 0 / 0.3067 | silu 0 / 0 / 0.3049 | silu 0 / 0 / 0.3015 |
| blocks.0.bn2 | silu 0 / 0 / 0.1901 | silu 0 / 0 / 0.1878 | silu 0 / 0 / 0.1893 |
| blocks.1.bn1 | silu 0 / 0 / 0.1619 | silu 0 / 0 / 0.1622 | silu 0 / 0 / 0.1611 |
| blocks.1.bn2 | silu 0 / 0 / 0.1383 | silu 0 / 0 / 0.1422 | silu 0 / 0 / 0.1362 |
| blocks.2.bn1 | silu 0 / 0 / 0.0596 | silu 0 / 0 / 0.0578 | silu 0 / 0 / 0.0593 |
| blocks.2.bn2 | silu 0 / 0 / 0.1208 | silu 0 / 0 / 0.1197 | silu 0 / 0 / 0.1152 |
| tower_final_bn | silu 0 / 0 / 0.3243 | silu 0 / 0 / 0.3463 | silu 0 / 0 / 0.3236 |
| policy.pre_bn | leaky_relu 0 / 0 / 0.1944 | leaky_relu 0 / 0 / 0.2020 | leaky_relu 0 / 0 / 0.2024 |
| value.bn | leaky_relu 0 / 0 / 0.0409 | leaky_relu 0 / 2 / 0.0271 | leaky_relu 0 / 0 / 0.0416 |

  At 21k and 22k all three have 0 parked and 0 mostly off at every site.
- **Strength.** Across the three caps the spread per probe is 25.2 (20k), 33.0 (21k), 50.8 (22k) and 7.2 (23k) pElo, with no
  consistent order (clip2 best at 20k and 22k, clip1 at 21k and 23k); at 23k all three are within 12.8 of ReLU B (1560.6–1567.8
  vs 1573.4).
  One start, one seed, four probes per arm.
- **Reading.** A fixed cap anywhere from 1 to 5 was enough here, so the blowup came from one step far outside the run's normal
  range, not from a gradual rise; the cap only has to cut that one step. A cap set at a multiple of the recent median catches
  that step without binding in normal training: the relative gradient cap (`documentation/plans-active/RELATIVE_GRADIENT_CAP_PLAN.md`,
  cap = min(hard max, max(floor, k × median of the last N pre-clip norms)), k = 3, about 1.0 here) is now implemented and merged
  on main, defaulting to log only. Its validation runs V-1 (log only from B-silu's 18k checkpoint, measuring the per-step ratio
  distribution) and V-3 (fresh start on B's recipe with the cap on) come next.
- **Reproduce:** binary `~/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2331-4e70c615-p1headact.app` (build
  2330, git `4e70c615`, dirty; sha256 prefix `1ae960792717`); corpus `20260624-192615-w3aA5b`; start checkpoint
  `20261005-lrBsilu-cyc1-replay-step18000.safetensors` (ModelID `20261006-44-8ipq`). clip2 (clip5: replace `clip2` with `clip5`):

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2331-4e70c615-p1headact.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"; E=experiments/20261005-lr-schedule-ab
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-lrBsilu-cyc1-replay-step18000.safetensors" \
  --resume-exact --accept-inexact params \
  --out-model "$M/20261006-lrBsilu-clip2-replay-latest.safetensors" --parameters $E/parameters-Bsilu-clip2.json --epochs 12 \
  --training-step-limit 5000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005
PROBE_BIN="$BIN" PROBE_SEGMENT=1 TRAINER_PID=<trainer pid> experiments/probe_loop.sh 20261006-lrBsilu-clip2 $E/probes-Bsilu-clip2-seg1.jsonl
```

  `parameters-Bsilu-clip2.json` / `parameters-Bsilu-clip5.json` are `parameters-B.json` with only `grad_clip_max_norm` changed
  (2.0 / 5.0). Analysis: the `[REPLAY]` lines of `dcm_log_20261005-234437.txt` (B-silu), `dcm_log_20261006-170000.txt` (clip1),
  `dcm_log_20261007-035017-2.txt` (clip2) and `dcm_log_20261007-035017.txt` (clip5), compared by `trainerStep=`;
  `bn_liveness.analyze()` on `20261006-lrBsilu-clip{1,2,5}-replay-seg1-step{3000,4000,5000}` (trainer steps 21k–23k). A rerun on
  this GPU should match these lines exactly up to 19,750 (ctl15 matched B-silu bit for bit); after the split, MPSGraph's
  numerics on a shared GPU are not guaranteed to repeat.

## Relative gradient cap validation V-1 and V-3 (added 2026-10-07 06:07; E-0024)

- Plan `documentation/plans-active/RELATIVE_GRADIENT_CAP_PLAN.md`, Part V: cap = min(`grad_clip_max_norm`, max(floor, k × median of
  the last N pre-clip gradient norms)), applied once the window holds W entries. V-1 is the gate for k; V-3 (owner-required) checks
  that the cap leaves healthy early training alone. Both pass; the plan's P5 (default mode → clip with k = 3) is ready but not
  applied: the edit was refused by the session's permission check and waits on the owner's go.
- Both run frozen build 2390 (`FrozenBuilds/DCM-2390-b4845088-relcap.app`, git `b4845088`, clean; sha256 prefix `f9ac7a8ca47f`),
  corpus `20260624-192615-w3aA5b`, `--seed 20261005`, `--policy-tail-precision fp32_from_pre_bn`.
- **V-1** — exact resume of B-silu from `20261005-lrBsilu-cyc1-replay-step18000` (ModelID `20261006-44-8ipq`), `--accept-inexact params`,
  `parameters-B-relcap-v1.json` (`parameters-B.json` + mode 1 = log only, k = 1, N = 1000, W = 100, floor 0.01 — the lowest allowed,
  so the floor hides nothing between the median and 0.5), to trainer step 21,000. Log only feeds the hard max (15), so the training is
  B-silu's own; every step above 1 × its trailing median writes `[GRAD-CLIP] … applied=false` with its ratio. `[RESUME] EXACT` (build
  changed 2330 → 2390, behavior fingerprint matches). Stem `20261007-lrBsilu-relcapV1`, ModelID `20261007-47-QMBc`, log
  `dcm_log_20261007-060735.txt`, 06:07:35 → 07:28:28, rc 0, probes `probes-Bsilu-relcapV1.jsonl`.
- **V-3** — fresh start from B's own start net `20261005-r7b24-fresh` (ReLU tower, ModelID `20261005-22-yRzB`),
  `parameters-B-relcap-v3.json` (`parameters-B.json` + mode 2 = clip, k = 3, N = 1000, W = 100, floor 0.5), 3,000 steps. Stem
  `20261007-lrB-relcapV3`, ModelID `20261007-49-NZSp`, log `dcm_log_20261007-060832.txt`, 06:08:32 → 07:28:54, rc 0, probes
  `probes-B-relcapV3.jsonl`.

### V-1 and V-3: results (both finished 2026-10-07 07:28)

- **V-1 is B-silu, bit for bit.** Its `[REPLAY]` lines equal B-silu's at all six trainer steps both logged (18,600, 19,000, 19,600,
  20,000, 20,600, 21,000); its probes equal B-silu's (19k 1571.9, 20k 1390.2, 21k 457.4; NLL equal to the fourth decimal); and its
  21,000 checkpoint is byte-identical to B-silu's in every tensor (`scripts/safetensors_tensor_compare.py`, exit 0, max relative
  difference 0). Log-only mode changes nothing, and V-1 measured exactly the run that blew up.
- **Healthy span 18,101–19,750** (1,650 steps, from the first step with W = 100 entries to the last line before the precursor):
  1,381 steps above their trailing median, largest ratio **1.39×**; whole-distribution p99 ≈ 1.22× and p99.9 ≈ 1.31× (from the
  logged upper half). 84% of steps sit above the trailing median because the learning rate is rising (0.013 at 18,102 → 0.357 at 19,750),
  so each norm tends to exceed the median of the 1,000 before it. 19,800–20,598: largest 2.27× (one step, 20,368).
- **Every step above 2× its trailing median:**

| trainer step | pre-clip norm | trailing median | ratio | LR | |
|---:|---:|---:|---:|---:|---|
| 19,785 | 8.4497 | 0.3011 | 28.06 | 0.377 | precursor |
| 19,795 | 5.5758 | 0.3019 | 18.47 | 0.382 | precursor |
| 20,368 | 0.7823 | 0.3444 | 2.27 | 0.750 | isolated |
| 20,599 | 4.2119 | 0.3623 | 11.62 | 0.880 | burst |
| 20,600 | 2.5656 | 0.3624 | 7.08 | 0.881 | burst (the only one a 50-step line showed) |
| 20,601 | 3.0004 | 0.3624 | 8.28 | 0.881 | burst |
| 20,602 | 4.0610 | 0.3625 | 11.20 | 0.882 | burst |
| 20,603 | 17.5647 | 0.3625 | 48.45 | 0.882 | burst; above the hard max 15, clipped in B-silu too |
| 20,604 | 13.8180 | 0.3625 | 38.12 | 0.883 | burst |
| 20,605 | 17.4873 | 0.3627 | 48.21 | 0.883 | burst; above the hard max 15 |
| 20,606 | 3.6740 | 0.3629 | 10.12 | 0.884 | burst |
| 20,607 | 2.6632 | 0.3629 | 7.34 | 0.884 | burst |
| 20,608 | 1.6255 | 0.3630 | 4.48 | 0.885 | burst |
| 20,609 | 2.5394 | 0.3631 | 6.99 | 0.885 | burst |
| 20,610 | 0.7413 | 0.3633 | 2.04 | 0.885 | burst tail |
| 20,773–20,990 | 0.76–1.71 | 0.367–0.381 | 2.04–4.62 | 0.95–0.98 | 41 steps after the blowup, network damaged (20 parked policy pre-BN channels) |

  Above 3× after the blowup: 20,818 (3.44), 20,872 (4.62), 20,874 (3.35), 20,876 (3.52), 20,916 (3.36), 20,957 (3.45). These are
  steps of a network the burst had already broken; a run capped at 3× would have clipped the precursor and the burst and not
  reached that state, so they are not false positives on a healthy run, but a k = 3 cap would clip them too.
- **The plan's rule for k** ("the smallest of 3, 4, 5 with no healthy per-step ratio above it except isolated single steps") gives
  **k = 3**: the healthy maximum is 1.39×, and 3× clips both precursor steps and the whole burst from 20,599 to 20,609.
- **The blowup was an 11-step burst, not one step.** Steps 20,599–20,609 all exceed 4× their trailing median, three of them 38–48×;
  the hard max 15 clipped 20,603 and 20,605 in B-silu and it was not enough. The precursor (19,785 and 19,795) explains why caps of
  1.0, 2.0 and 5.0 all left B-silu's path between 19,750 and 19,800 (E-0020, E-0023).
- **V-3 is B, bit for bit.** Its only `[GRAD-CLIP]` events are the hard-max clips at steps 1–3 (pre-clip 31.17, 30.34, 26.50 → 15),
  which B made too; the relative term never bound in 3,000 steps. Its `[REPLAY]` lines equal B's at all 25 steps both logged; probes
  equal B's (1k 1018.2, 2k 1251.6, 3k 1321.7); its 3,000 checkpoint is byte-identical to B's in every tensor; `bn_liveness` per site
  at 3,000 is identical (value.bn 6 parked in both, nothing else).
- **V-3's cap trajectory** (`gCap=` on the step lines): 15 through step 100 (warm-up, fewer than W = 100 entries), then k × median:
  7.82 (150), 5.39 (450), 3.75 (700), 2.77 (1,000), 1.57 (1,480), 1.15 (2,000), 0.99 (2,600), 0.94 (3,000). The closest approach of a
  line interval's largest pre-clip norm (`gNormMax=`) to the line's cap was 5.01 vs 6.48 (interval ending 300); after 1,000 the
  largest was 0.92 vs 2.38 (1,120). Each line's cap is the cap at that line's step, slightly lower than earlier in its interval.
- **Reading.** On the incident run the gap between healthy steps (≤ 1.39×) and the steps that broke it (18–48×) is wide; 3× sits
  far from both. On a healthy fresh start the cap never bound. Both pass the plan's criteria; P5 waits on the owner. V-2 (k = 3 in
  clip mode from B-silu's 18k, to 23k) was not run.
- **Reproduce:** binary as above; launch scripts in `experiments/20261005-lr-schedule-ab/launch/` (`relcapV1_launch.sh <binary> params`, `relcapV3_launch.sh <binary>`). Commands:

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2390-b4845088-relcap.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"; R=/Users/andrew/cursor/drews-chess-machine; E=$R/experiments/20261005-lr-schedule-ab
# V-1
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-lrBsilu-cyc1-replay-step18000.safetensors" --resume-exact --accept-inexact params \
  --out-model "$M/20261007-lrBsilu-relcapV1-replay-latest.safetensors" --parameters $E/parameters-B-relcap-v1.json --epochs 12 \
  --training-step-limit 3000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005
PROBE_BIN="$BIN" PROBE_ABOVE_STEP=18000 TRAINER_PID=<V-1 pid> $R/experiments/probe_loop.sh 20261007-lrBsilu-relcapV1 $E/probes-Bsilu-relcapV1.jsonl
# V-3
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261005-r7b24-fresh.safetensors" \
  --out-model "$M/20261007-lrB-relcapV3-replay-latest.safetensors" --parameters $E/parameters-B-relcap-v3.json --epochs 12 \
  --training-step-limit 3000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn --seed 20261005
PROBE_BIN="$BIN" TRAINER_PID=<V-3 pid> $R/experiments/probe_loop.sh 20261007-lrB-relcapV3 $E/probes-B-relcapV3.jsonl
```

  Analysis: V-1's `[GRAD-CLIP] trainerStep=` lines (`preNorm=`, `median=`, `ratio=`); the `[REPLAY]` lines of
  `dcm_log_20261005-234437.txt` (B-silu) and `dcm_log_20261005-013235.txt` (B) against V-1's and V-3's, by `trainerStep=`;
  `python3 scripts/safetensors_tensor_compare.py "$M/20261005-lrBsilu-cyc1-replay-step21000.safetensors" "$M/20261007-lrBsilu-relcapV1-replay-step21000.safetensors"`
  and the same for `20261005-lrB-cyc1-replay-step3000` against `20261007-lrB-relcapV3-replay-step3000`.

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


## Policy tail precision is now an architecture field (format v12)

The `--policy-tail-precision` launch flag used above was removed when the tail
became an architecture field (`documentation/plans-active/POLICY_TAIL_ARCHITECTURE_PLAN.md`);
current builds refuse it as an unknown argument, and the launch lines above are kept
as the record of what ran. To reproduce a run under `fp32_from_pre_bn`, derive its
start model with the tail set and launch without the flag:
`DrewsChessMachine --derive-model --from <start.safetensors> --set-policy-tail-precision fp32_from_pre_bn --out <start-fp32tail.safetensors>`.
Every network is now built at its file's own tail, inference included: a checkpoint
recording `fp32_from_pre_bn` (its `trainer_policy_tail_precision` key or lineage
configuration) loads, plays and probes under it, while `experiments/probe_loop.sh`
never passed the flag, so probes made with a build from `de0f22be` (2026-10-01
15:18) on ran `mixed_final_projection`; re-probing such a checkpoint gives different
numbers from those probes. A checkpoint recording no tail loads as
`mixed_final_projection` until the owner-reviewed PT-D3 audit and header edit give it
the tail it ran.
