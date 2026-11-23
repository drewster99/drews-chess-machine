# 2026-11-23 — Unlikelihood negative-advantage loss: fork of R-elitecyc at 384,964

**Status:** running (launched 2026-11-22 23:58 CST), to trainer step 1,000,000 or until the owner cancels.

## Question

On negative-advantage positions the policy loss today (`complement_ce`) moves the played move's probability to a
target spread uniformly over the other legal moves, which also pushes strong alternatives down and blunders up. The
unlikelihood form, −log(1 − p_played) over the legal moves, pushes the played move down and raises the other legal
moves in proportion to their current probability (illegal moves pushed down). Does that train a stronger policy?

## Design

- **One change:** `negative_advantage_policy_loss` = `unlikelihood` (2) instead of `complement_ce` (1). Parameter added
  in commit 8087f714 (one live enum replacing the Bool `signed_advantage_complement_ce`; off / complement_ce /
  unlikelihood), fixed in e5e79ab2 (branch blend instead of `select`, which MPSGraph cannot differentiate; lock box for
  the live setting).
- **Start:** exact resume of R-elitecyc's `20261011-zlra-elitecyc-latest-step384964.safetensors` (model 20261025-1-qyUf),
  bf16, with the main line's decay schedule `../20261008-zlra-selfplay-lr/parameters-zlra-elitecyc-decay.json` plus the
  new key (`parameters-unlikelihood.json`). Accepted gaps: `schedule` (as for the main line's segment 14) — the loss
  change is logged `[RESUME-DIFF] negative_advantage_policy_loss: parent=1 this_run=2`. Behavior fingerprint matched
  across build 2523 → 2701.
- **Baseline, no control arm** (owner: GPU time is scarce): R-elitecyc's own record from 384,964 — lichess-wide every
  1,000 steps (`../20261008-zlra-selfplay-lr/tables/wide_pelo_record.json`, key `elitecyc`), lichess-large every 10k
  (`../20261108-elitecyc-heldout/large_probe.jsonl`). Batches must match R-elitecyc's at the same trainer steps
  (`[BATCH-HASH]`); a mismatch would break the pairing.
- **Build:** frozen DCM-2701-e5e79ab2. The always-built unlikelihood branch costs ~15 ms/step (+4%) on the small
  benchmark model (`../20261122-memory-footprint/train-steptype-bench-6x3x3-64-negadv.txt`).
- Out-model stem `20261123-zlra-elitecyc-unlik-latest`; launcher `launch.sh`.

## Results

Running.
