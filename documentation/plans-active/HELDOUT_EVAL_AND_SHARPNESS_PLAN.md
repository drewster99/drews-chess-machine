# Held-out evaluation, sharpness and large-puzzle probes

Owner request 2026-11-08 ("do it all"): test whether R-elitecyc's plateau is a capacity limit, a data limit, or neither, and explain its gradient-norm rise (`experiments/20261008-zlra-selfplay-lr/analysis-elitecyc-793k/weight_norms.md`). Three read-only measurements over saved checkpoints; none touches a live run.

## Questions

1. **Generalization gap.** Is loss on elite games the model never saw equal to loss on games it trained on (capacity or label-noise limit), higher (memorizing on its 5th corpus pass: a data/regularization limit), or still falling (no plateau)?
2. **Sharpness.** Did the loss surface around the weights get sharper (larger top Hessian eigenvalue λmax) after the LR fell at 386k? Rising λmax explains the part of the gNorm rise that weight shrinkage does not (‖W‖·‖g‖ rose 1.65× on the tower convs).
3. **Strength resolution.** pElo from 200 + 4,435 puzzles is noisy; a ~100k-puzzle stratified set resolves smaller differences and per-rating curves.

## Phase 0 — held-out data (no app change)

- `experiments/20261108-elitecyc-heldout/fetch_heldout.sh` streams Lichess's `lichess_db_standard_rated_2026-10.pgn.zst` (29 GB) through `filter_elite.py` without writing the download to disk: keep games where one player is 2500+ and the other 2300+, Event not Bullet/UltraBullet, result 1-0 / 0-1 / ½-½ — the Lichess Elite Database rule from December 2021 on (database.nikonoel.fr, which stops at November 2025, the training corpus's last month). Output: `~/Dropbox/corpora_sources/elite_heldout_2026-10/lichess_elite_2026-10.pgn`.
- Import with the existing `--import-pgn` into a new corpus (one PGN parser; expected tens of MB).
- Why October 2026, not May 2026: the May games went into corpus `w3aA5b`; a later month cannot overlap any corpus DCM has trained on.
- Disk: only the filtered PGN and its corpus land on disk; check free space before the import.

Validation: filter log `pipe status: 0 0 0`, counts plausible (hundreds of thousands kept of ~90M), `--validate-corpus` clean on the new corpus.

## Phase 1 — `--eval-corpus` (held-out loss)

`--eval-corpus <corpus id or dir> --eval-model <file|dir> [--eval-model …] [--eval-games N] [--eval-from-end] [--eval-out <jsonl>]`

- Reads the corpus read-only through the existing corpus reader (`GameCorpus` / `GameCorpusReadOnlyView`), takes the first (or, with `--eval-from-end`, the last) N games in corpus order, replays every position of each game, encodes with each model's own input encoding (`BoardEncoder`), and runs `evaluateBatched` in inference mode (BN running statistics), the same load path as `--probe-model`.
- Per model, over the identical position list:
  - `policy_nll`: −log p(played move) under the softmax over legal moves (the same legal masking as `MoveSampler`, temperature 1).
  - `top1`, `top5`: the played move is the argmax / in the top 5 legal moves.
  - `value_ce`: cross-entropy of the W/D/L head against the game result from the side to move (no label smoothing).
  - `value_acc`: argmax W/D/L equals the result; mean `pW`/`pD`/`pL`.
  - The same split by game phase (ply 0–19, 20–59, 60+).
  - `positions`, `games`, positions skipped and why (none expected).
- "Seen" side: the training corpus `20260704-215145-op24Gp` with `--eval-from-end` (its newest games, November 2025), the same N as the held-out side, so the time gap to the held-out month is as small as the data allows.
- Output: one JSON line per model per corpus to stdout or `--eval-out` (written with `FileSafety`, never overwriting). Non-finite outputs are an error for that model, not numbers.
- Pure helpers (legal-masked NLL / top-k, WDL CE, phase bucketing) are pure functions with unit tests.

Validation: build clean; unit tests written for the pure helpers (owner 2026-11-05: test suite not run unless asked — reported unrun); a smoke run on 200 games from each corpus with one checkpoint; NLL on the seen side is in the range of the trainer's logged policy figures.

## Phase 2 — `--sharpness`

`--sharpness --sharpness-model <file> --sharpness-corpus <corpus> [--sharpness-batches K] [--sharpness-iterations N] [--sharpness-seed S] [--sharpness-out <jsonl>]`

- Builds a `ChessTrainer` from the trainer-state file (separate process; never the live trainer), fills a replay buffer from the corpus the way corpus replay does, and computes the training loss's gradient g(w) with the trainer's own graph (same loss, same training-mode BN, same advantage baseline): velocity reset to zero, momentum 0 and clipping disabled for the probe, one step on a **fixed batch** (the sampler stream re-seeded identically before every step; dropout stream likewise, or dropout off if the architecture has none), then the velocity equals g(w). Weights are reloaded before every step, so the LR and weight decay applied by the step never matter.
- λmax by power iteration on Hessian-vector products, central differences: Hv ≈ (g(w + εv) − g(w − εv)) / 2ε with ‖v‖ = 1, ε relative to ‖w‖; N iterations (default 30), K fixed batches averaged (default 4). Also reports the Rayleigh quotient sequence (convergence), and per-group λmax (tower convs, BN affine, policy head, value head) from block-restricted power iterations.
- Reports the stability threshold 2(1 + μ)/LR at the checkpoint's own trainer step (from `LRMomentumCycleReadout`), so λmax can be read against it.
- Pure helpers (vector ops, Rayleigh quotient, convergence test) unit-tested.

Validation: build clean; unit tests for the helpers (unrun, as above); on one checkpoint, the finite-difference check g(w + εd) − g(w) ≈ ε·Hd for two ε values agrees; repeated runs with the same seed give the same λmax.

Run on 10k, 380k, 500k, 800k.

## Phase 3 — large puzzle set

- `scripts/curate_lichess_probes.py` gains a mode writing a ~100k-puzzle stratified set (rating × theme), same JSON format as the bundled sets, to a file outside the bundle (`lichess_db_puzzle.csv.zst` downloaded to scratch, deleted after).
- `--probe-model … --probe-set-file <json>`: probe with a set file instead of the bundled sets (same scoring, same summary fields, set identity from the file's `metadata`, fingerprint checked). Model files' recorded test sets are unchanged.

Validation: build clean; probe of one checkpoint on the bundled wide set via `--probe-set-file` equals `--probe-set wide`.

## Phase 4 — run and write up

Run all three over elitecyc checkpoints (250k, 385k, 500k, 600k, 700k, 800k plus the newest), write results into `experiments/20261108-elitecyc-heldout/README.md` with a Reproduce section, commit, push.

## Decisions taken without the owner (2026-11-08)

- October 2026 instead of May 2026 (no possible overlap; Dropbox May file left unmaterialized — it would need far more than the free disk).
- Corpus import + `--eval-corpus` instead of reading PGN directly (one PGN parser, one corpus reader).
- λmax by finite-difference Hessian-vector products through the existing training graph instead of a new double-backprop graph (no second loss definition).
- ~100k puzzles rather than all ~5.6M (resolution well past the noise, minutes per checkpoint).
