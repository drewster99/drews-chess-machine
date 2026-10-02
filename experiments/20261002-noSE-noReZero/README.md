# 2026-10-02 — No SE, with vs without ReZero

**Status:** running (launched 2026-10-02 01:11 CDT), alongside the leaky-FC1 run.

## Question

Does ReZero help on the SE-experiment architecture without SE? The SE style
experiment found no-SE matching or beating every SE variant; ReZero has never been
tested on its own here.

## Design

- **Only variable:** `use_rezero` — on (existing runs) vs off (this run). With
  ReZero off, each block is `out = LayerNorm(x + F(x))` instead of
  `LayerNorm(x + α·F(x))` with α = 0.447·tanh(·).
- **Baselines (not re-run):** `se_none` seed 1 (32,036 steps) and seed 2
  (7,019), from `experiments/20260929-se-style-ab/`.
- **Init caveat:** removing ReZero removes tensors (the per-block α), so this run
  cannot start from a bit-identical copy of a baseline's fresh net. It is a fresh
  mint (`20261002-1-bh2u`, preset `bench_v5s3_noSE_noReZero`), so the comparison
  carries seed-to-seed noise (7–25 pElo on this setup). Two ReZero-on seeds bound
  that noise; add a second no-ReZero seed if the result is close.
- **Architecture:** basic30 → stem 128 (7×7) → 3×[7×7+7×7 @128, no SE, ReLU
  pre-act, clean_add, **no ReZero**, LayerNorm out] → policy intermediate_conv
  (128) · value WDL (16 → FC128) · bf16 · 5,170,319 params.
- **Corpus / parameters / numerics:** identical to the SE experiment (corpus
  `20260624-192615-w3aA5b`, its pinned `parameters.json`, 12 epochs, step limit
  33,000, `--policy-tail-precision fp32_from_pre_bn` to match the baselines' build).
- **Concurrency:** shares the GPU with the leaky-FC1 run; compare on step only.

## Launch record

| field | value |
|---|---|
| launched | 2026-10-02 01:11:24 CDT |
| build | Release build 2275 (frozen copy, same as the leaky-FC1 run), stamped `f6fdd88` — the working tree committed as `de0f22b` |
| fresh net | `20261002-bench_v5s3_noSE_noReZero-fresh.safetensors`, ModelID `20261002-1-bh2u` |
| out model | `20261002-bench_v5s3_noSE_noReZero-replay-latest.safetensors` (+ enumerated `…-replay-step<N>`) |
| log | `~/Library/Logs/DrewsChessMachine/dcm_log_20261002-011124.txt` |

`[REPLAY-HPARAMS]` matches the SE experiment's runs. `probe_loop.sh` probes every
1,000-step checkpoint (`--probe-set wide`) into `probes.jsonl`; `table.py` renders
the comparison.

## Seed 2 (launched 2026-10-02 03:55)

At 6,000 steps seed 1 was level with both ReZero seeds (1269.1 vs 1265.0 / 1263.5),
inside the 7–25 pElo seed spread, so a second no-ReZero seed runs now rather than
after seed 1 ends. It shares the GPU with leaky-FC1 and seed 1 (three runs).

| field | value |
|---|---|
| launched | 2026-10-02 03:55:13 CDT |
| build | same frozen build 2275 (`f6fdd88` stamp, `de0f22b` code) |
| fresh net | `20261002-bench_v5s3_noSE_noReZero-seed2-fresh.safetensors`, ModelID `20261002-3-x4gI` (fresh mint, different init from seed 1) |
| out model | `20261002-bench_v5s3_noSE_noReZero-seed2-replay-latest.safetensors` (+ enumerated `…-replay-step<N>`) |
| log | `~/Library/Logs/DrewsChessMachine/dcm_log_20261002-035513.txt` |
| probes | `probes-seed2.jsonl`, via `experiments/probe_loop.sh 20261002-bench_v5s3_noSE_noReZero-seed2 probes-seed2.jsonl` |

`[REPLAY-HPARAMS]` is identical to seed 1's. Reproduce: the commands below with
`-seed2` added to every model file name.

## Review at 10,000 steps

`review.py 10000` (training means over the `[REPLAY]` lines in the 500 steps up to
the step; branch scale from the enumerated checkpoint, identified by metadata).

| | ReZero s1 (`se_none`) | no ReZero s1 |
|---|---:|---:|
| pElo / NLL | 1300.1 / 2.4457 | 1296.0 / 2.4279 |
| loss / policy / value | 3.6026 / 2.7801 / 0.8137 | 3.6027 / 2.7806 / 0.8145 |
| policy entropy / gNorm | 2.879 / 0.679 | 2.883 / 0.688 |
| ‖conv2‖ per block | 15.57 / 15.80 / 16.68 | 15.55 / 15.90 / 17.19 |
| branch scale (eff α × ‖conv2‖) | 6.43 / 6.30 / 6.75 | 15.55 / 15.90 / 17.19 |

- **Level after an early lag.** pElo difference (no ReZero − ReZero, seed 1s) by
  1k: +19.9, −51.8, −45.9, −4.1, +27.4, +4.1, −7.2, −1.0, −7.7, −4.1. Both
  no-ReZero seeds trail at 2k (1033.7 / 1070.2 vs 1085.5 / 1140.5, NLL 2.84 / 2.78
  vs 2.68 / 2.64); seed 1 catches up by 4k–5k, seed 2 by 3k (1170.2 vs 1172.2 /
  1163.4). From 4k to 10k the mean difference is +1.1 pElo (ahead at 2 of 7), and
  NLL is lower at 10k.
- Training losses are identical to the fourth decimal on loss and policy loss.
- **Branch scale.** Without ReZero the residual branch runs at ~2.5× the ReZero
  nets' scale, and the network keeps it there (block 2 grows 16.0 → 17.2). With
  ReZero, α saturates at its tanh cap and the scale stops at ~6.5 (see
  `rezero-scale/`). The loss is indifferent between the two after the first few
  thousand steps.
- Tally 1k–10k: behind at 7 of 10 checkpoints (sign test p 0.34, mean −7.1),
  driven by the 2k–3k lag.

## Reproduce

```
"$BIN" --new-model --architecture experiments/20261002-noSE-noReZero/bench_v5s3_noSE_noReZero.json \
  --out-model "$M/20261002-bench_v5s3_noSE_noReZero-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b \
  --start-model "$M/20261002-bench_v5s3_noSE_noReZero-fresh.safetensors" \
  --out-model "$M/20261002-bench_v5s3_noSE_noReZero-replay-latest.safetensors" \
  --parameters experiments/20261002-noSE-noReZero/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints \
  --policy-tail-precision fp32_from_pre_bn
```
