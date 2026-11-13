# R-elitecyc: held-out loss, sharpness and large-puzzle strength (2026-11-08)

Plan: `documentation/plans-active/HELDOUT_EVAL_AND_SHARPNESS_PLAN.md`. Question: is the plateau after 500k a capacity limit, a data (memorization) limit, or neither; why did the gradient norm rise 0.40 → 0.92 over the run.

## Data

- **Held-out:** Lichess standard rated games, October 2026, filtered to the Lichess Elite Database rule (one player 2500+, the other 2300+, no bullet/ultrabullet, decided result): 342,670 of 95,395,628 games (`fetch_heldout.sh`, `filter_elite.py`; only the filtered PGN touched disk). Imported as corpus `20261108-100629-lahw9i` (342,546 games, 124 skipped by the importer). The training corpus ends November 2025, so no overlap is possible.
- **Seen:** the training corpus `20260704-215145-op24Gp`, its last 20,000 games (November 2025), which the run has passed over ~5 times.
- **Puzzles:** `lichess_probes_large.json.gz`, 90,378 puzzles (curation preset `large`: the wide set's shape at ~22× density).

## 1. Held-out vs seen loss (`--eval-corpus`, 20,000 games each side, every position)

See `eval_table.md` (held-out 1,943,073 positions, seen 1,781,574).

| step | NLL seen | NLL held-out | gap | top-1 seen | top-1 held-out |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.0578 | 2.0934 | +0.036 | 0.475 | 0.455 |
| 150,000 | 1.9530 | 1.9948 | +0.042 | 0.512 | 0.490 |
| 380,000 | 1.9186 | 1.9598 | +0.041 | 0.524 | 0.501 |
| 500,000 | 1.8922 | 1.9373 | +0.045 | 0.534 | 0.510 |
| 800,000 | 1.8856 | 1.9309 | +0.045 | 0.536 | 0.512 |

- The gap is +0.036 nats already at 10k, before the model could have memorized anything: that is the shift between the two months/populations, not overfitting. Over 790k more steps and ~5 corpus passes it grew only to +0.045. **Memorization is ~0.01 nats: the model is not data-limited by repetition.**
- Held-out and seen loss move together. Held-out NLL kept falling through the high-LR phase (1.995 at 150k → 1.960 at 380k), dropped with the LR cut (→ 1.937 at 500k), and still falls slowly (1.931 at 800k).
- Training loss ≈ held-out loss means the model **underfits**: it cannot fit the data it has. That is the regime where a larger model or more effective optimization helps; more data alone would not.
- Value CE: seen 0.86–0.92, held-out 0.88–0.97, noisy from checkpoint to checkpoint (±0.03).

## 2. Puzzle strength (`--probe-set-file`, all 80 checkpoints 10k…800k)

`large_probe.md`. Noise after 600k: sd 3.5 pElo.

- 10k → 150k: 1633 → 1716; 150k → 380k: 1716 → 1762 (+46 while training losses looked flat); LR drop 380k → 450k: 1762 → 1797; 500k → 800k: 1795 → 1800.
- The high-LR phase was still gaining ~10 pElo per 50k steps when the LR was cut.

## 3. Sharpness (`--sharpness`, fp32, 16 fixed batches of 256 = 4,096 positions, 30 iterations)

`sharpness.jsonl`. Batches of 256 instead of the run's 4096 (fp32 at 4096 needs ~30 GiB); BN normalizes over the smaller batch, so values compare across checkpoints, not with the training run's own curvature.

| step | λmax | converged | ε vs 2ε | asymmetry | mean gradient norm | single-batch gradient norm |
|---:|---:|---|---:|---:|---:|---:|
| 10,000 | 9.4 | yes (13 it) | 1.2% | 0.9% | 0.48 | 1.3–1.6 |
| 380,000 | 22.0 | yes (15 it) | 2.1% | 0.8% | 0.82 | 2.4–3.1 |
| 500,000 | ~22–24 | no, alternates | 12% | 9% | 1.22 | 3.1–3.8 |
| 800,000 | ~26–30 | no, alternates | 16% | 32% | 1.31 | 3.4–4.1 |

- Sharpness rose 2.3× under the high LR (10k → 380k) and only ~10–35% after the LR drop. The alternating late estimates suggest two eigenvalues of similar size and opposite sign (|λ| ≈ 25); the finite-difference checks degrade there, so read the late values as approximate.
- The gradient is mostly noise: single-batch norms are ~3× the mean gradient's. After the LR drop the noise grew ~37% and sharpness ~10–35%: consistent with "sharper region → noisier gradient", not proof.
- **Stability margin:** SGD with momentum μ is stable while λmax < 2(1+μ)/LR. At the 0.02 peak that bound is ~190; λ ≈ 22 is ~12% of it. The run's LR was never near the stability limit.

## Conclusions

1. **Not a data limit.** ~0.01 nats of memorization after ~5 passes.
2. **Not shown to be a capacity limit either.** The model underfits, and held-out loss and puzzle pElo were still improving at high LR when the LR was cut; after the anneal both creep up slowly. What is exhausted is the low-LR phase, not necessarily the model.
3. **The LR schedule is the most direct lever:** the high-LR phase was still paying, sharpness sits far below the stability bound, and the anneal's +35 pElo is a one-time gain. A longer (or higher-peak) high-LR phase before a later anneal is the cheapest test; a larger model is the second.
   - **Tested 2026-11-12 (E-0029, `../20261008-zlra-selfplay-lr/analysis-hilr-fork/README.md`):** continuing the original cycle from 384,964 gained ~16 pElo in the next ~65k steps and nothing after; the main line's anneal over the same 143k steps gained ~32. The high-LR phase was nearly spent when it was cut.
4. **gNorm rise:** weight-decay shrink of scale-invariant weights explains ~1.86× of the conv gradients' 3.1× (`../20261008-zlra-selfplay-lr/analysis-elitecyc-793k/weight_norms.md`); rising sharpness and gradient noise are consistent with the rest.

## Unexplained

- The late sharpness estimates do not converge and their symmetry check degrades (32% at 800k): the value-baseline forward pass makes the probed operator the Jacobian of the update field rather than a pure Hessian; why that gap grows late is unknown.
- Value CE alternates by ±0.03 between neighboring checkpoints on both sides.

## Reproduce

Build 2658+ (`--eval-corpus`, `--probe-set-file`), `--sharpness-batch-size` from commit 5e1be61f. Models under `~/Library/Application Support/DrewsChessMachine/Models/`.

```
./fetch_heldout.sh                     # MONTH=2026-10
DrewsChessMachine --import-pgn ~/Dropbox/corpora_sources/elite_heldout_2026-10/lichess_elite_2026-10.pgn --corpus-name elite-heldout-2026-10
DrewsChessMachine --eval-corpus 20261108-100629-lahw9i --eval-games 20000 --eval-model <ckpt> ... --eval-out eval_heldout.jsonl
DrewsChessMachine --eval-corpus 20260704-215145-op24Gp --eval-games 20000 --eval-from-end --eval-model <ckpt> ... --eval-out eval_seen.jsonl
gunzip -k lichess_probes_large.json.gz
DrewsChessMachine --probe-model <ckpt> --probe-set-file lichess_probes_large.json      # per checkpoint, 10k..800k
DrewsChessMachine --sharpness --sharpness-model <ckpt> ... --sharpness-corpus 20260704-215145-op24Gp \
  --sharpness-batch-size 256 --sharpness-batches 16 --sharpness-max-footprint-gb 7 --sharpness-out sharpness.jsonl
```

## 4. Per-layer momentum-buffer norms over the run (`velocity_norms/`)

Every elitecyc trainer checkpoint stores each tensor's momentum buffer (`opt.<tensor>.velocity`; v ← μv + clipped g, so ‖v‖ ≈ a ~10-step gradient sum, before the LR). `vnorm.py` reads ‖w‖ and ‖v‖ per tensor from all 152 checkpoints (10k…846k, read-only); `group.py` groups them (`group_tables.txt`).

- **The gNorm rise is proportional across the tower, not a shift between layers and not one layer running away.** The tower convs carry ~75–80% of ‖v‖² at every checkpoint (block 0 24→28%, block 1 22→29%, block 2 21→22%); each conv group's ‖v‖ rose 2.5–3× (0.40–0.43 → 1.07–1.22).
- **Weight shrink accounts for most of it.** Conv ‖w‖ fell 38–53% (block 1 37.9 → 19.8), mostly before 380k; for a conv followed by BN the gradient scales with 1/‖w‖ (≈1.6–2.1×). Since 500k: ‖w‖ −3%, ‖v‖ +5–10%.
- **The value head is the jumpy part.** Across 57 checkpoints 1k apart (790k–846k) the checkpoint-to-checkpoint coefficient of variation of ‖v‖ is 5–6% for tower convs, 10% policy conv, 20% value conv, 24% value fc1, 37% value fc2 (0.05–0.31), 53% value BN; at 700k and 790k all value groups were 2–3× up together. It is ~2% of ‖v‖², invisible in the global gNorm.
- Combined with section 1's held-out mean pD swinging 0.138–0.194 between checkpoints on fixed positions: the value head follows the corpus region being fed. Corpus replay feeds the corpus in fixed (chronological) shard order every epoch; the 1M-position buffer turns over every ~117 steps (~12k consecutive games).

Reproduce: `/usr/bin/python3 vnorm.py vnorm.csv && /usr/bin/python3 group.py` (numpy).
