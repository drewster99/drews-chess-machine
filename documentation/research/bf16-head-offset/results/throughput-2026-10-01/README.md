# Throughput before and after the head numerics fix (2026-10-01)

Phase 1 validation item "throughput regression under 2%" (`documentation/plans-active/HEAD_NUMERICS_PLAN.md`).

- **Builds:** parent `3f02ae4` (build 2224) vs head fix `da15920` — adjacent commits, so the head fix is the only difference. Each built Release from its own git worktree.
- **Machine:** M5 Max laptop, macOS 27.2, **on AC power, sleep blocked** (`caffeinate -i -s`). Two earlier runs made on battery (old1/new1 against HEAD) ran at about half speed and are excluded.
- **Order:** A-B-B-A for both benchmarks.
- `throughput.csv` holds the per-run numbers (session logs named per row, in `~/Library/Logs/DrewsChessMachine/`).

## Training (corpus replay)

`--replay-corpus 20260624-192615-w3aA5b --start-model Models/20260702-Qeu8-resume3-replay-step681000.safetensors --parameters bench_params.json --training-step-limit 600` (Ejp0 @681k architecture, batch 4096). Measured as wall time from step 200 to 550.

| | parent A | fix A | fix B | parent B | parent mean | fix mean | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall ms/step | 634.0 | 682.2 | 679.5 | 644.1 | 639.1 | 680.9 | **+6.5%** |
| GPU wait ms/step (p50) | 484.1 | 522.1 | 522.1 | 497.7 | 490.9 | 522.1 | +6.4% |
| encode ms/step (p50) | 51.6 | 58.3 | 58.7 | 51.2 | 51.4 | 58.5 | +13.8% |

## Self-play (GUI `--train`, 480 s each, default v4 preset, fresh net)

Measured from the `[STATS]` lines between elapsed 120 s and the end (359 s window).

| | spold1 | spnew1 | spnew2 | spold2 | parent mean | fix mean | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| self-play plies/hour | 19,793,873 | 19,908,923 | 20,078,263 | 18,921,750 | 19,357,812 | 19,993,593 | +3.3% (noise) |
| concurrent training steps/s | 0.794 | 0.766 | 0.760 | 0.755 | 0.775 | 0.763 | −1.5% |

**Verdict:** self-play (inference) has no regression. Standalone training is 6.5% slower, which misses the < 2% target; the extra time is GPU work and encoding in the training graph (fp32 policy tail from the pre-BN and the fp32 loss path are the likely sources; not profiled per op).

Reproduce: set `BENCH_DIR` to a scratch folder holding `bench_params.json`, point the scripts at the two Release builds, run `fix_abba.sh` and `selfplay_bench.sh`, then `python3 analyze.py <runs file>`.
