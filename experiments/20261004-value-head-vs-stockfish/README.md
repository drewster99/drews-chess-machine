# 2026-10-04 — Value head vs Stockfish at predicting game results

Summary: [E-0013](../summaries/E-0013_2026-10-04_value-head-vs-stockfish.html).

`value_stage.py` takes the same 8,172 corpus positions as
`../20261004-head-logits-relu-inputs/` (same shards, counts, seed and draw order, plus each
position's ply, plies left and side-to-move material balance), runs R7's 33k checkpoint
(`20261002-bench_v5s3_noSE_noReZero-replay-step33000`) through that folder's fp32 forward pass, and
evaluates every position with Stockfish (`Stockfish dev-20260529-b1053e60`, 50,000 nodes, 1
thread per engine, 10 engines in parallel, `UCI_ShowWDL`). It reports the value cross-entropy
against the game result per bucket for: the overall base rate, the bucket's own base rate, our
value head, Stockfish's raw W/D/L, and Stockfish's eval mapped to these games' results by a
3-class logistic fit on tanh(cp/400), fitted on one half of the positions and scored on the other
(two folds, split by position index parity, so a game's two positions usually land in different folds). Results:
`value_stage.json`.

Reproduce (a venv with python-chess and numpy, e.g. `python3 -m venv v && v/bin/pip install chess numpy`;
Stockfish at `~/bin/stockfish`):

```
cd experiments/20261004-value-head-vs-stockfish
SF_NODES=50000 SF_WORKERS=10 <venv>/bin/python value_stage.py
```

It ran 2026-10-04 23:56–23:59 CDT while a GPU timing benchmark was running (CPU only, about 3 minutes); see
[E-0011](../summaries/E-0011_2026-10-04_policy-tail-timing.html).
