# 2026-10-02 — Zero-init ReZero (α₀ = 0, cap 1.0), no SE

**Status:** running since 2026-10-02 20:24 CDT.

## Question

Our "ReZero" has never started at zero: α₀ = 1/√5 = 0.447 and the tanh cap was tied to the
init (cap = α₀), so the effective α started at 0.76 of the cap and sat at the cap within
~1k steps — a fixed residual scale near 0.44, not ReZero as published
(`20261002-noSE-noReZero/rezero-scale/`). Removing it entirely made no difference after ~3k
steps (`20261002-noSE-noReZero/README.md`). This run tests the published form: every block
starts as the identity (α = 0) and learns how much branch to add, with a cap of 1.0 so α can
reach the branch scale the no-ReZero nets chose.

## Design

- **Only variable vs ReZero seed 1 (`se_none`):** ReZero α₀ 0.447 → **0** and cap 0.447 →
  **1.0** (architecture format v6, `rezero_alpha_cap`). Starting net derived bit-exactly
  from `se_none`'s fresh net:
  `--derive-model --from 20260929-test_SE_none-fresh.safetensors --set-rezero-alpha-init 0
  --set-rezero-alpha-cap 1` → `20260929-test_SE_none-rz0cap1-fresh.safetensors`, ModelID
  `20261002-6-SGuE`. Verified: only the three `blocks.<i>.rezero_alpha` tensors differ (now
  exactly 0.0); every other tensor is bit-identical to the source.
- **Architecture:** basic30 → stem 128 (7×7) → 3×[7×7+7×7 @128, no SE, ReLU pre-act,
  clean_add, ReZero(0·tanh≤1), LayerNorm out] → policy intermediate_conv · value WDL ·
  bf16 · 5,170,322 params.
- **Comparators (not re-run):** ReZero seed 1 (`se_none`, α₀ 0.447 capped, same weights
  otherwise) and no ReZero seeds 1 and 2 (`20261002-noSE-noReZero/`).
- **Corpus / parameters / numerics:** corpus `20260624-192615-w3aA5b`, the SE experiment's
  pinned `parameters.json` (copied here, identical), 12 epochs, step limit 33,000,
  `--policy-tail-precision fp32_from_pre_bn`.
- **Build:** needs format v6, so it runs on a newer build than the comparators (2275); the
  launch record names it. Engine changes between the two builds are listed there.
- **Measurements:** pElo / NLL every 1,000 steps (`--probe-set wide`, with the same newer
  binary); effective α per block over time (from the enumerated checkpoints); branch scale
  (α_eff × ‖conv2‖) as in `20261002-noSE-noReZero/rezero-scale/`.

## What would count as an answer

- If zero-init tracks no-ReZero (including its 2k–3k lag) and α climbs past 0.447, the
  capped-init version was just a fixed scale and the init does not matter here.
- If it beats both early (no lag) and stays level later, zero-init is the better default
  for this family.
- One seed: differences under ~25 pElo are inside the seed spread.

## Charts

Zero-init ReZero is charted with the no-SE ReZero / no-ReZero runs in
[that experiment's charts](../20261002-noSE-noReZero/README.md#charts):

![pElo by step](../20261002-noSE-noReZero/charts/rezero-pelo.svg)

![Each run minus no SE + ReZero (seed 1)](../20261002-noSE-noReZero/charts/rezero-vs-baseline.svg)

## Launch record

- **Launched** 2026-10-02 20:24:30 CDT (session log `dcm_log_20261002-202430.txt`), in the slot
  freed by no-ReZero seed 1, by the experiment queue.
- **Build** 2290, git `9a36f9f`, frozen as `FrozenBuilds/DCM-2290-9a36f9f.app` (binary sha256
  prefix `59228bed24fb`). It is the first build that reads architecture format v6. The comparators
  ran on build 2275 (`de0f22b`); the engine changes between the two are the zero-init ReZero
  support itself (explicit `rezero_alpha_cap`), per-move label smoothing (off here: mode
  `fixed_total`), and Lichess-bot work. Corpus replay's adjudication fix lands after this build,
  so this run's fed stream matches its comparators.
- **Run model ID** `20261003-1-NKTv` (parent `20261002-6-SGuE`, the derived starting net).
- **Command**

```
"$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2290-9a36f9f.app/Contents/MacOS/DrewsChessMachine" \
  --replay-corpus 20260624-192615-w3aA5b \
  --start-model "$HOME/Library/Application Support/DrewsChessMachine/Models/20260929-test_SE_none-rz0cap1-fresh.safetensors" \
  --out-model "$HOME/Library/Application Support/DrewsChessMachine/Models/20261002-rezero-zero-init-replay-latest.safetensors" \
  --parameters experiments/20261002-rezero-zero-init/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
```

- **Probes** `experiments/probe_loop.sh 20261002-rezero-zero-init experiments/20261002-rezero-zero-init/probes.jsonl`
  with `PROBE_BIN` set to the same build (build 2275 cannot read format v6).
