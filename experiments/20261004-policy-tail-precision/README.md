# 2026-10-04 — fp32 vs mixed policy tail: accuracy and training speed

Summaries: [E-0010](../summaries/E-0010_2026-10-04_policy-tail-precision-accuracy.html) (accuracy and
logit growth), [E-0011](../summaries/E-0011_2026-10-04_policy-tail-timing.html) (training speed).

The policy head (`intermediate_conv`) is 1×1 pre-conv → BN → ReLU → 1×1 final conv (+ bias) →
4,864 logits. `--policy-tail-precision fp32_from_pre_bn` widens to fp32 before the BN;
`mixed_final_projection` (the app default) runs BN, ReLU and the final conv in the compute dtype
(bf16) and widens only the final conv's output (`ChessNetwork.swift`, the two `switch
tailPrecision` blocks in the intermediate_conv policy head).

## Accuracy (`probe-accuracy.txt`)

The 33,000-step checkpoints of fatconv, R7 and R8 probed on build 2320 (`--probe-set wide`) once
with each tail:

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
for f in 20261004-fatconv98-b2275-replay-step33000 20261002-bench_v5s3_noSE_noReZero-replay-step33000 20261002-bench_v5s3_noSE_noReZero-seed2-replay-step33000; do
  for p in fp32_from_pre_bn mixed_final_projection; do
    "$BIN" --probe-model "$HOME/Library/Application Support/DrewsChessMachine/Models/$f.safetensors" --probe-set wide --policy-tail-precision $p
  done
done
```

`probe-accuracy.txt` prints each model name cut to 40 characters: `20261004-fatconv98-b2275-replay-step3300` is fatconv's
step-33000 file, `…noSE_noReZero-replay` is R7's and `…noSE_noReZero-seed2-` is R8's (both step 33000); all on build 2320.

## Training speed (`timing/`)

Results in the next section.

### Timing results (2026-10-04 23:34 → 2026-10-05 01:32)

`timing/tail_abba.sh` ran, on build 2320 alone on the GPU (M5 Max, AC power), 600 corpus-replay
steps per run in A-B-B-A order (fp32, mixed, mixed, fp32) for: R7 at 33k
(`20261002-bench_v5s3_noSE_noReZero-replay-step33000`), the SE scale+bias seed-2 net at 7,282 steps
(`20260929-test_SE_scale+bias-seed2-replay-step7282`), and a fresh `v4_5block_7x7`
(`timing/v4fresh.mint.txt`). `timing/nt8y_after.sh` then timed 600 steps of nt8y's last checkpoint
(`20260701-nT8Y-resume4-replay-latest`, fp32 tail). Wall ms/step is the time from step 200 to step
550; "median step" is the median of the logged per-step `ms=` at those steps (every 50th step).
`timing/runs.txt` names every run's session log; `timing/analysis.txt` is `analyze.py`'s output.

| net | fp32 wall (A, B) | mixed wall (A, B) | mixed vs fp32 (wall) | median step fp32 → mixed |
|---|---|---|---:|---|
| R7 33k | 816.1, 808.1 | 801.4, (832.3 excluded) | −1.3% | 756.4 → 744.7 (−1.5%) |
| SE scale+bias | 818.5, 824.1 | 813.2, 813.5 | −1.0% | 756.9 → 752.5 (−0.6%) |
| fresh v4_5block_7x7 | 1175.3, 1173.9 | 1171.2, 1180.2 | +0.1% | 1116.7 → 1120.0 (+0.3%) |
| nt8y (fp32 only) | 443.1 | — | — | 413.1 |

R7's −1.3% excludes `r7-mixedB` by hand, (801.4 − 812.1) / 812.1; `timing/analysis.txt`'s R7 line (+0.6%)
includes it. `r7-mixedB`'s wall time is excluded: a CPU-only analysis (Stockfish, `../20261004-value-head-vs-stockfish/`)
ran during its timed window; its median step is in line with `r7-mixedA`. `timing/runs-contended-with-chain4.txt`
lists an earlier attempt that shared the GPU with a second benchmark started by a stale queue script; none
of its numbers are used.

Reproduce, from `experiments/20261004-policy-tail-precision/`: move the recorded `timing/runs.txt` aside (both scripts
refuse to append to it), then `zsh timing/tail_abba.sh` and `zsh timing/nt8y_after.sh` (they write into `timing/`;
before each run they refuse to start while a `--replay-corpus`, `--train` or `--probe-model` process is running — a GUI
session is not detected), then `python3 timing/analyze.py timing/runs.txt`. `tail_abba.sh` re-mints the v4 net with the
recorded seed (`--init-seed 2150607837842323118`) into `v4fresh.mint.rerun.txt`. Paths of the original scratch folder in
`timing/v4fresh.mint.txt` were rewritten to `./`.


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
