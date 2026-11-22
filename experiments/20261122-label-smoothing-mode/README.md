# 2026-11-22 — Policy label smoothing: mode 0 (fixed total) vs mode 1 (per move), plus no smoothing

**Status:** running (launched 2026-11-22 12:28 CST, all three arms at once). Owner request 2026-11-22:
per-move smoothing (mode 1, built 2026-10-02 as arm B of
`documentation/plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md`) had never been run.
R-elitecyc-fp32fix was stopped for it (abort save at trainer step 456,328).

## Question

Does giving each alternative move a fixed smoothing share δ (mode 1) instead of a fixed total ε spread over
the legal moves (mode 0) change what the policy learns? With ε = 0.1 and δ = 0.0033 the two targets are
the same at 30 legal moves (0.1/30 ≈ 0.0033). They differ only away from 30:

| legal moves n | mode 0: played-move target | mode 1: played-move target |
|---|---|---|
| 3 | 0.933 | 0.993 |
| 10 | 0.910 | 0.970 |
| 30 | 0.903 | 0.904 |
| 60 | 0.902 | 0.805 |
| 100 | 0.901 | 0.673 |

Arm C (no policy smoothing) shows how large a smoothing effect exists at all on this setup.

## Design

- **Arms** (only the policy label smoothing differs):
  - A: `policy_label_smoothing_mode` 0, ε 0.1 (the main line's setting)
  - B: mode 1, δ `policy_label_smoothing_per_move` 0.0033, cap 0.5
  - C: mode 0, ε 0
- **Paired:** every arm starts from the same fresh net (`Models/20261122-lsmode-fresh.safetensors`,
  model ID `20261122-115-Jdx1`, init seed 20261122) with `--seed 20261122` on the same corpus from game 0,
  so all three train on the same batches. Check: the `[BATCH-HASH]` lines must match across arms.
- **Model:** `architecture-6x3x3-64.json` — the main line's v6 recipe (SiLU, no SE, no ReZero, LayerNorm
  out, intermediate_conv policy head, W/D/L value head, bf16 with the fp32_from_pre_bn policy tail) at
  6 blocks of 3×3 at 64 channels, policy pre-conv 64, input `basic25`. 694,767 parameters.
- **Corpus:** elite `20260704-215145-op24Gp` (owner: straight to elite, no legal-move-count survey first).
  40,000 steps is under one epoch, so no position repeats.
- **Training:** the main line's parameters (`../20261008-zlra-selfplay-lr/parameters-zlra-elitecyc-decay.json`)
  carried key by key, except a fixed LR 0.005 (one keyframe, peak = trough) after the 1,000-step warmup, so
  no LR cycle interacts with the comparison. Complement CE on, value label smoothing 0.013, batch 4096,
  momentum 0.9, weight decay 3e-4.
- **Build:** frozen DCM-2687 (`d8afa524`). Launcher: `launch.sh <arm>`.
- **Length:** step limit 40,000; compare at 10k / 20k / 40k, stop early if the arms are clearly apart or
  clearly identical.

## Measurements

- lichess-wide pElo / NLL embedded in every 1,000-step checkpoint; lichess-large probe (90,378 puzzles,
  pElo SD ≈ 3.5) every 1,000 steps — the most sensitive of the two.
- Step-line metrics (pLoss, pEnt, playedP, vLoss).
- Not available yet: NLL by legal-move count, which measures the mechanism directly.

## Files

- `parameters-{A,B,C}.json`, `architecture-6x3x3-64.json`, `launch.sh`
- Out-models `Models/20261122-lsmode-{A,B,C}-latest.safetensors` and their `-step<N>` checkpoints
- `train-{A,B,C}.stdout`, `results-{A,B,C}.json`, probes `large_probe_{A,B,C}.jsonl`

## Results

Pending.
