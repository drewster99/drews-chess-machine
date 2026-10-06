# 2026-10-05 — LR schedule A/B: constant 0.01 vs a 1.0 ↔ 0.001 cycle (R7 shape, basic24)

**Status:** A/B/C complete; arm B-leaky running since 2026-10-05 20:41. A/B/C ran 2026-10-05 01:32–18:51 CDT: both arms to their 36,000-step limit (~17:14), then, at the
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
