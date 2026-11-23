# 2026-11-22 — Policy label smoothing: mode 0 (fixed total) vs mode 1 (per move), plus no smoothing

**Status:** complete (2026-11-22 11:59 → 21:24 CST; all three arms at once). Summary: [E-0030](../summaries/E-0030_2026-11-22_label-smoothing-mode.html). Owner request 2026-11-22:
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

## Segments

- Segment 0 (steps 0–9,000+): build 2687, replay buffer 1,000,000. Stopped by the owner at 9,733 / 9,875 / 9,833 (A / B / C)
  for the memory work (`../20261122-memory-footprint/`).
- Segment 1 (9,000 → 40,000): every arm resumed exactly from its step-9,000 file (`launch_r200k.sh`) on build 2695
  (`DCM-2695-8e889ffd`: one training executable, streamed shards), replay buffer 200,000 (min fill 100,000), stem
  `20261122-lsmode-<arm>-r200k-latest`. `[RESUME] NOT EXACT: params` only. `[BATCH-HASH]` matched across the arms at every
  100 steps in both segments. Logs: A `dcm_log_20261122-115904-2` / `-153853`, B `-115904-3` / `-153923-2`, C `-115904` /
  `-153923`.

## Results

Per-1000-step table: `final_table.md` (lichess-wide and lichess-large pElo/NLL, pLoss, pEnt, playedP per arm).

Paired differences on lichess-large, all 40 checkpoints:

| comparison | mean Δ pElo | SD | positive | mean Δ NLL | lower NLL |
|---|---|---|---|---|---|
| B − A (mode 1 vs mode 0) | +0.3 | 3.5 | 24/40 | +0.0016 | 19/40 |
| C − A (none vs ε 0.1) | +17.1 | 5.1 | 40/40 | −0.0422 | 40/40 |
| C − B (none vs mode 1) | +16.8 | 5.3 | 40/40 | −0.0438 | 40/40 |

By range: 1k–9k (1M buffer) C − A +20.1, B − A +1.2; 10k–40k (200k buffer) C − A +16.3, B − A +0.1; 30k–40k C − A
+14.7, B − A −1.6. Best lichess-large: A 1105.3, B 1101.7, C 1115.4 (all at 35k); mean of 36k–40k: 1088.8 / 1085.0 / 1102.8.

- **Mode 1 vs mode 0: no measurable difference.** δ = ε/30 makes the targets equal at 30 legal moves; on this corpus and
  model the positions far from 30 did not move the result.
- **No policy smoothing beat ε 0.1 at every checkpoint**, by about 17 pElo and 0.042 NLL, in both segments — consistent
  with E-0002 (ε 0.03 +20 over ε 0.1, one seed).
- Limits: one seed; correlated checkpoints; small model on non-repeating data (little overfitting for smoothing to
  prevent); puzzle sets score the network's own probabilities, which smoothing caps. Worth re-testing on a larger model
  over several epochs, and with ε ≤ 0.03.

## Reproduce

```
B="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2687-d8afa524.app/Contents/MacOS/DrewsChessMachine"
"$B" --new-model --architecture architecture-6x3x3-64.json --name "lsmode 6x3x3@64" --init-seed 20261122 \
  --out-model "$HOME/Library/Application Support/DrewsChessMachine/Models/20261122-lsmode-fresh.safetensors"
./launch.sh A; ./launch.sh B; ./launch.sh C          # segment 0 (stopped at ~9.8k; any stop past 9,000 works)
./launch_r200k.sh A; ./launch_r200k.sh B; ./launch_r200k.sh C   # segment 1 from step 9,000 on build 2695
python3 report.py 1000 40000                          # probes lichess-large, prints the table
```
A single-segment rerun (one build, one buffer size) needs only `launch.sh` with the buffer settings edited.
