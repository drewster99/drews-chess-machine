# 2026-11-23 — Unlikelihood negative-advantage loss: fork of R-elitecyc at 384,964

**Status:** stopped by the owner 2026-11-24 14:15 CST at trainer step 504,514 (clean SIGINT; final save `20261123-zlra-elitecyc-unlik-latest-step504514.safetensors`, results `results-seg0.json`). Launched 2026-11-22 23:58 CST.

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

## Results (to 504,514)

Paired with R-elitecyc (identical batches) from 384,964:

| step | Δ lichess-large pElo (unlik − R-elitecyc) |
|---:|---:|
| 390k | +10.8 |
| 400k | +19.3 |
| 410k | +20.3 |
| 420k | +10.1 |
| 430k | +21.7 |
| 440k | +17.9 |
| 450k | +15.5 |
| 460k | +18.3 |
| 470k | +16.6 |
| 480k | +22.4 |
| 490k | +13.6 |
| 500k | +25.5 |

- Ahead on lichess-large at all 12 checks (mean +17.7); fork best large 1820.7 at 504k.
- lichess-wide pElo: early dip (−41 at 385k, loss switched at LR 0.016), level by 387k, then mostly ahead (+20 mean over 495k–504k).
- lichess-wide NLL ~0.29 lower throughout; mostly calibration (complement CE's uniform target spreads mass onto junk moves), not ranking. Large-set top-1 at 440k: 54,111 vs 53,355 of 90,378 (+1.4%).
- Policy entropy ~2.2 nats vs ~2.8; gNorm steady ~0.75–0.85; no instability.
- No control arm: a single run against R-elitecyc's record.

Per-1000 table: `table.py`; large-set probes: `large_probe.jsonl`.
