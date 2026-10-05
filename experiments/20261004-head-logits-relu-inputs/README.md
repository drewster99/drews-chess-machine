# 2026-10-04 — ReLU inputs and pre-softmax logits: fatconv vs R7/R8

Summary: [E-0009](../summaries/E-0009_2026-10-04_relu-inputs-and-logits.html). Full report with
architecture tables and head diagrams: `report.html`.

## What was measured

For the 33,000-step checkpoints of fatconv (`20261004-fatconv98-b2275-replay-step33000`), R7
(`20261002-bench_v5s3_noSE_noReZero-replay-step33000`) and R8 (`…-seed2-replay-step33000`):
per-channel statistics of every tensor entering a ReLU (mean, std, min, max, fraction
negative / positive), and the policy and value logits just before their softmaxes (legal,
illegal, per position, illegal softmax mass, value slots, value CE against game results).

## Files

| file | content |
|---|---|
| `relu_inputs.py` | fp32 numpy forward pass mirroring `ChessNetwork.swift` (pre-activation tower, clean_add, LayerNorm out, no SE, no ReZero), `BoardEncoder.swift` (basic30) and `PolicyEncoding.swift`; collects the statistics |
| `relu_inputs_bot.json` | results on the app audit's positions: the start position plus every ply of DCM's own Lichess-bot games, capped at 4,096 (63 games) |
| `relu_inputs_corpus.json` | results on 8,172 training-corpus positions (shards 20, 26, 32, 38 of `20260624-192615-w3aA5b`, never reached by these runs; 1,024 games per shard, two random plies per game, seed 20261004). Its `start_wdl` fields are the first sampled corpus position, not the start position (written before `relu_inputs.py` was corrected to leave the field null in corpus mode) |
| `build_relu_report.py` | renders `report.html` from the corpus results (start-position row from the bot results) |
| `numerics-audits/` | the app's `--analyze-numerics` JSON and stdout for the three checkpoints (build 2320), used to validate the forward pass: start-position W/D/L, legal-logit quartiles, spreads, all-move mean, value offsets and largest \|logit\| match to every printed digit |

## Reproduce

Needs python-chess and numpy (`python3 -m venv v && v/bin/pip install chess numpy`; `v/bin/python` below is that venv).

```
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
F="$M/20261004-fatconv98-b2275-replay-step33000.safetensors"
R7="$M/20261002-bench_v5s3_noSE_noReZero-replay-step33000.safetensors"
R8="$M/20261002-bench_v5s3_noSE_noReZero-seed2-replay-step33000.safetensors"
cd experiments/20261004-head-logits-relu-inputs
v/bin/python relu_inputs.py relu_inputs_bot.json fatconv="$F" R7="$R7" R8="$R8"
POSITIONS=corpus v/bin/python relu_inputs.py relu_inputs_corpus.json fatconv="$F" R7="$R7" R8="$R8"
v/bin/python build_relu_report.py
# the app audits (output JSON into numerics-audits/):
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
for m in "$F" "$R7" "$R8"; do "$BIN" --analyze-numerics "$m" --numerics-out numerics-audits; done
```

Paths of the original scratch folder inside the `.stdout` files were rewritten to `./`.

The bot-position set reads `~/Library/Application Support/DrewsChessMachine/LichessBot/Games`,
which grows as the bot plays, so a later rerun of that set sees more games than the 63 used here;
the corpus set is fixed by its seed.
