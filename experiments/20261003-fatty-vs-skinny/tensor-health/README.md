# Tensor health of fatty and slim-neck fatty at 33,000 steps (2026-10-04)

Summary: [E-0007](../../summaries/E-0007_2026-10-04_fatty-tensor-health.html).

- `numerics_audit_*_20261004-10-YpxP.json` / `…-13-K4Iu.json` and the two `.stdout` files: the
  app's `--analyze-numerics` (build 2320) on `20261003-fatty216-b2275-replay-step33000` and
  `20261003-fatty224s3-b2275-replay-step33000` — 4,097 positions (start position plus 63 games of
  DCM's Lichess bot), layer health, bf16/fp16 vs fp32 policy KL and top-1 changes, value offsets.
- `fatty_tensors.py`: per-tensor weight statistics, BN active-fraction estimates (Φ(β/|γ|)) and
  optimizer-velocity patterns (zero-velocity stem input planes, value FC1 and policy conv). Output:
  `fatty_tensors.out`. Fully-connected velocity is read in the trainer's native [in, out] layout
  (corrected 2026-10-05; the first version reshaped it as [out, in] and stopped with an error at value FC1).

Reproduce:

```
cd experiments/20261003-fatty-vs-skinny/tensor-health
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --analyze-numerics "$M/20261003-fatty216-b2275-replay-step33000.safetensors" --numerics-out .
"$BIN" --analyze-numerics "$M/20261003-fatty224s3-b2275-replay-step33000.safetensors" --numerics-out .
python3 fatty_tensors.py > fatty_tensors.out   # needs numpy
```

Paths of the original scratch folder inside the `.stdout` files were rewritten to `./`.
