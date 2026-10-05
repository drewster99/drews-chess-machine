# 2026-10-05 — LR schedule A/B: constant 0.01 vs a 1.0 ↔ 0.001 cycle (R7 shape, basic24)

**Status:** running since 2026-10-05 01:32 CDT; both arms to 36,000 steps (est. ~16:00–17:00 CDT at the
measured ~1.5 s/step while they share the GPU).
Summary: [E-0017](../summaries/E-0017_2026-10-05_lr-schedule-ab.html).

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
