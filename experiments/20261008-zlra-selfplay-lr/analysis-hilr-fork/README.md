# R-elitecyc-hilr vs R-elitecyc: does the original high-LR cycle still pay after 384,964?

Question (owner 2026-11-09): at trainer step 384,964 the main line (R-elitecyc) left the original cycle
(geometric cosine 0.002 ↔ 0.02, period 10,000) for the decaying keyframe schedule. The held-out study
(`../../20261108-elitecyc-heldout/README.md`, conclusion 3) suggested the high-LR phase was still gaining when it
was cut. The fork continues the original cycle from the same saved state to answer that directly.

## The two lines

| | R-elitecyc-hilr (fork) | R-elitecyc (main) |
|---|---|---|
| start | exact resume of `20261011-zlra-elitecyc-latest-step384964.safetensors` | same file (segment 14) |
| parameters | `parameters-zlra-elite-gentlecycle.json` | `parameters-zlra-elitecyc-decay.json` |
| LR peak / trough, 386k → 486k | 0.02 / 0.002 | 0.02 → 0.001 / 0.002 → 0.000447 (geometric) |
| LR peak / trough, 486k → 586k | 0.02 / 0.002 | 0.001 / 0.000447 → 0.0001 |
| cycle | cosine, period 10,000 | same |
| momentum | 0.90 | 0.90 |
| corpus, batch, seed, feed | elite corpus `20260704-215145-op24Gp`, same feed (both exact resumes of one state) | same |
| build | 2663 (`DCM-2663-b1ed8b52`, adds only the `[DYNAMICS]` log line) | 2620 |
| model ID | `20261110-1-lG5l` | `20261102-2-eBRM` (at 721,584) |

Fork: started 2026-11-09 23:45 EST, stopped by SIGINT (owner) 2026-11-12 20:11 EST, abort save at trainer step
528,527 (143,563 steps). No GPU faults, no non-finite values; the only alarm was the standing `policy_offset_drift`
warning (median |pLogitMean| ≈ 9.2–9.3), active since 385,050.

Every 10k checkpoint lands at the same phase of the 10,000-step cycle on both lines, so the 10k rows compare like
with like. The fork's abort save at 528,527 is mid-cycle (fed LR 0.0065 at the last step line), so its numbers are
not comparable to the 10k rows and are listed only for completeness.

## Large puzzle set (`lichess-large`, 90,378 puzzles)

Fork: `large_probe_step*.jsonl` here (build 2663). Main: `../../20261108-elitecyc-heldout/large_probe.jsonl` (build
2658). Same puzzle file (fingerprint `7483846931880bf9…`).

| trainer step | main pElo | fork pElo | Δ pElo | main NLL | fork NLL | Δ NLL | main top-1 | fork top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 390,000 | 1765.1 | 1764.2 | −0.9 | 1.8256 | 1.8277 | +0.0021 | 0.5792 | 0.5788 |
| 400,000 | 1755.0 | 1748.9 | −6.1 | 1.8347 | 1.8390 | +0.0043 | 0.5745 | 0.5716 |
| 410,000 | 1767.6 | 1759.6 | −8.0 | 1.8310 | 1.8408 | +0.0099 | 0.5804 | 0.5766 |
| 420,000 | 1780.0 | 1766.2 | −13.8 | 1.8113 | 1.8277 | +0.0164 | 0.5862 | 0.5797 |
| 430,000 | 1777.2 | 1767.1 | −10.1 | 1.8033 | 1.8152 | +0.0119 | 0.5849 | 0.5802 |
| 440,000 | 1788.8 | 1772.8 | −16.0 | 1.7872 | 1.8063 | +0.0191 | 0.5904 | 0.5828 |
| 450,000 | 1797.3 | 1780.4 | −16.9 | 1.7841 | 1.8034 | +0.0193 | 0.5943 | 0.5864 |
| 460,000 | 1794.3 | 1777.7 | −16.7 | 1.7893 | 1.8093 | +0.0200 | 0.5929 | 0.5851 |
| 470,000 | 1788.9 | 1763.0 | −25.9 | 1.7978 | 1.8247 | +0.0270 | 0.5904 | 0.5782 |
| 480,000 | 1795.0 | 1772.8 | −22.2 | 1.7866 | 1.8129 | +0.0262 | 0.5932 | 0.5828 |
| 490,000 | 1793.8 | 1760.7 | −33.1 | 1.7939 | 1.8299 | +0.0361 | 0.5927 | 0.5772 |
| 500,000 | 1793.0 | 1767.5 | −25.5 | 1.7910 | 1.8253 | +0.0343 | 0.5923 | 0.5804 |
| 510,000 | 1793.6 | 1767.2 | −26.4 | 1.7933 | 1.8259 | +0.0326 | 0.5926 | 0.5802 |
| 520,000 | 1797.0 | 1775.4 | −21.6 | 1.7861 | 1.8057 | +0.0196 | 0.5942 | 0.5841 |
| 528,527 (fork, mid-cycle) | — | 1756.3 | | — | 1.8286 | | — | 0.5751 |

Noise on this set: sd ≈ 3.5 pElo between neighbouring checkpoints (held-out study, after 600k).

## Held-out games (October 2026 elite games, corpus `20261108-100629-lahw9i`, first 20,000 games, 1,943,073 positions)

Fork and main 520k: `eval_heldout.jsonl` here (build 2663, one `--eval-corpus` run). Main 380k / 450k / 500k:
`../../20261108-elitecyc-heldout/eval_heldout.jsonl`.

| trainer step | main policy NLL | fork policy NLL | Δ | main top-1 | fork top-1 | main value CE | fork value CE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 380,000 (before the fork) | 1.9598 | — | | 0.5014 | — | 0.9239 | — |
| 450,000 | 1.9415 | 1.9551 | +0.0136 | 0.5073 | 0.5028 | 0.9545 | 0.9674 |
| 500,000 | 1.9373 | 1.9565 | +0.0192 | 0.5098 | 0.5023 | 0.8853 | 0.8971 |
| 520,000 | 1.9354 | 1.9525 | +0.0171 | 0.5102 | 0.5034 | 0.8964 | 0.9170 |
| 528,527 (fork, mid-cycle) | — | 1.9549 | | — | 0.5005 | — | 0.9312 |

Value CE alternates by ±0.03 between neighbouring checkpoints on both lines (held-out study), so the value column
shows no difference beyond that noise.

## Result

- The fork trails the main line at every 10k checkpoint from 400k on: by 10–17 pElo through 460k, 21–33 pElo from
  470k, and 0.014–0.019 nats of held-out policy NLL.
- The fork itself barely moved after the first 65k steps: large-set pElo 1764 → 1780 by 450k, then 1761–1778;
  held-out NLL 1.9598 (380k, shared) → 1.9551 (450k) → 1.9525 (520k).
- The main line gained most of its edge while its LR was decaying (386k → 486k): +32 pElo by 450k (from 390k) against the
  fork's +16.

## Interpretation

At 384,964 the original cycle had about run its course: continued for 143k more steps it added ~16 pElo and
0.005–0.007 nats of held-out NLL (380k → 450k / 520k), nearly all in the first ~65k steps, while the decaying
schedule over the same span added ~32 pElo and 0.024 nats. The held-out study's reading ("the high-LR phase was still paying ~10 pElo per 50k when
it was cut") is right only for the next ~65k steps; after that the high-LR line is flat and the anneal is worth
more. One seed per line; the gap (≥ 10 pElo from 420k, ≥ 20 from 470k) is several times the ~3.5 pElo
checkpoint-to-checkpoint noise, and the held-out NLL agrees in sign at every pair.

Not shown: whether a *longer* high-LR phase started earlier, or a higher peak, would end higher after its own
anneal. This fork never annealed. Nor does it say anything about the model's capacity.

## Reproduce

Build `FrozenBuilds/DCM-2663-b1ed8b52.app`; models under `~/Library/Application Support/DrewsChessMachine/Models/`.

```sh
gunzip -k ../../20261108-elitecyc-heldout/lichess_probes_large.json.gz
for s in 390000 400000 … 520000 528527; do
  DrewsChessMachine --probe-model "$M/20261109-zlra-elitecyc-hilr-latest-step$s.safetensors" \
    --probe-set-file lichess_probes_large.json --probe-out large_probe_step$s.jsonl
done
DrewsChessMachine --eval-corpus 20261108-100629-lahw9i --eval-games 20000 \
  --eval-model "$M/20261109-zlra-elitecyc-hilr-latest-step450000.safetensors" \
  --eval-model "$M/20261109-zlra-elitecyc-hilr-latest-step500000.safetensors" \
  --eval-model "$M/20261109-zlra-elitecyc-hilr-latest-step520000.safetensors" \
  --eval-model "$M/20261109-zlra-elitecyc-hilr-latest-step528527.safetensors" \
  --eval-model "$M/20261011-zlra-elitecyc-latest-step520000.safetensors" \
  --eval-out eval_heldout.jsonl
```

The training run: `../launch/elitecyc_hilr_fork_launch.sh` (`DCM_BIN` = the frozen build), stdout
`../train-elitecyc-hilr-seg0.stdout`, results `../results-elitecyc-hilr-seg0.json`, session log
`~/Library/Logs/DrewsChessMachine/dcm_log_20261109-234535.txt`.
