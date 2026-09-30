# 2026-06-12 — Channel dropout A/B (v1): rates 0 / 0.30 / 0.70 from a random-init champion

**Status:** done — no arm distinguishable from control at 600 steps; the dropout machinery was verified working.

## Question

Does channel (spatial) dropout in the residual blocks change what a net learns in a short self-play Play-and-Train run?
Secondary goal: prove the new dropout path works end to end (the rate reaches the training graph, telemetry shows it,
and rate 0.70 trains stably).

## Setup

- **Mode:** headless self-play Play-and-Train (`--train`), not corpus replay. The champion plays self-play games (170 workers)
  into a replay buffer (1,000,000-position capacity) and the trainer trains on it. The arena was effectively disabled
  (`arena_auto_interval_sec` 36000 against a run of about 22 min), so there were no promotions and the champion that
  generated games stayed at the fork weights for the whole of every arm.
- **Only variable:** `dropout_rate` (drop probability) — 0.00 / 0.30 / 0.70. The three `params_drop_*.json` files are
  otherwise identical: `weight_decay` 0.0005, `arena_auto_interval_sec` 36000,
  `replay_buffer_min_positions_before_training` 200000. Every other parameter was the app default.
- **Architecture:** `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu .
  policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params` (the `[ARCH]` line in each arm's log).
- **Start model:** random init. Stage 1 of `run_experiment.sh` launched a `--train` run, waited for Play-and-Train to
  start, waited 20 s more and sent SIGUSR2 (save and exit). No training step ran before the save (the fork log has zero
  `[STATS]` lines), so the fork champion `20260612-3-elVE` is untrained weights.
- **Arms (stage 2):** `--train --start-model <fork>/champion.safetensors --parameters params_drop_<R>.json
  --training-step-limit 600 --training-time-limit 2700 --output result_drop_<R>.json`, run one after another.
- **Stage 3:** restored the `LastSessionPointer` in `UserDefaults` to the live 5K7Z session and relaunched it via
  `run_latest.sh`, so that session auto-resumed afterwards.
- **Build:** 1824, git `6e4e233*` (dirty tree), branch `safetensors-storage`, Debug binary (`[APP]` line). The dropout
  feature was committed afterwards as `eacced3` / `600c016` (CHANGELOG 2026-06-12).
- **Machine:** this Mac (DerivedData hash `-eyigcdvyyrcsakaqcybzcfgsurbr` in the script).
- `default.profraw` is an empty (0-byte) coverage-profile file left behind by the Debug binary; it carries no data.

## Runs

| arm | log | results.json session_id | trainer | build | trainer steps (log) | steps in results.json | training time | self-play games |
|---|---|---|---|---|---|---|---|---|
| fork (stage 1) | `dcm_log_20260612-125609.txt` | — (saved `20260612-175644-20260612-4-gV3q-sigusr2.dcmsession`) | — | 1824 | 0 | — | — | — |
| 0.00 | `dcm_log_20260612-125702.txt` | `20260612-6-Zjbd` | `20260612-3-elVE-1` | 1824 | 600 (step-limit exit) | 588 | 1340 s | 40,100 |
| 0.30 | `dcm_log_20260612-131932.txt` | `20260612-8-lciw` | `20260612-3-elVE-1` | 1824 | 600 | 588 | 1336 s | 40,066 |
| 0.70 | `dcm_log_20260612-134159.txt` | `20260612-10-aXNv` | `20260612-3-elVE-1` | 1824 | 600 | 588 | 1340 s | 39,801 |
| resume (stage 3) | `dcm_log_20260612-140430.txt` | resumed session `20260601-12-5K7Z` | — | 1824 | — | — | — | — |

Every arm log shows `--train: loading start model …20260612-175644-20260612-4-gV3q-sigusr2.dcmsession/champion.safetensors`
and `loaded model champion.safetensors → 20260612-3-elVE`, the `--parameters overrides:` line with its own
`dropout_rate`, and `[PARAM] dropoutRate applied to training graph` / `drop=` in `[STATS]`. No checkpoint of the arms was
kept; the only artifacts are the three results JSONs here.

## Results

Wide puzzle set (`[TACTICAL-LICHESS] tick set=wide`, 24 ticks per arm, trainer model), from the arm logs. pElo here is on
the June 2026 in-app probe scale — not comparable to replay-era (July+) pElo.

| arm | first tick (step): NLL / pElo | last tick (step): NLL / pElo | mean NLL | mean pElo | NLL sd over ticks |
|---|---|---|---|---|---|
| 0.00 | (17) 3.641 / 513 | (578) 4.548 / 511 | 4.9618 | 475.9 | 0.6045 |
| 0.30 | (18) 3.644 / 514 | (578) 4.609 / 494 | 4.9634 | 479.6 | 0.5963 |
| 0.70 | (18) 3.645 / 514 | (578) 4.588 / 513 | 4.9607 | 479.5 | 0.6022 |

Final `stats` row of each results JSON (training step 588):

| arm | pLoss | vLoss | pEnt | legal mass | gNorm | pD | vAbs | final candidate probe: illegal mass |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 2.6474 | 0.6875 | 2.2953 | 0.8558 | 1.3561 | 0.7435 | 0.0616 | 0.0208 |
| 0.30 | 2.6563 | 0.6882 | 2.3013 | 0.8556 | 1.3324 | 0.7413 | 0.0629 | 0.0464 |
| 0.70 | 2.6646 | 0.6874 | 2.3023 | 0.8545 | 1.3503 | 0.7439 | 0.0635 | 0.0260 |

The results JSONs also hold 76 candidate-probe snapshots and ~500 per-step stats rows per arm (full batch stats, sampling
settings, ratio controller) for anyone who wants the trajectories.

## Conclusion

- No arm is distinguishable from control. Mean wide NLL differs by at most 0.003 across the arms, against a within-run
  tick-to-tick sd of about 0.6; the final losses differ in the second or third decimal.
- The machinery works: each arm's log shows its rate applied to the training graph and reported as `drop=`; the first
  probe ticks match (3.641 / 3.644 / 3.645 NLL, pElo 513 / 514 / 514), consistent with identical starting weights.
- Rate 0.70 trains stably (gNorm about 1.35–1.45, no entropy collapse), even though only about 38 of 128 channels
  survive each draw.
- Likely reason for the null result: an untrained net has no channel structure for dropout to disrupt, and with fresh
  self-play data there is little overfitting for always-on dropout to fight. This led to the v2 repeat from a trained
  champion (`experiments/dropout-ab-trained-20260612/`).

## Caveats

- n = 1 per arm and no replicate control, so there is no measured noise floor here. v2 added one (0.00A vs 0.00B).
- Self-play is stochastic, so each arm trained on different games (40,100 / 40,066 / 39,801 games generated) from the same
  fixed champion — a data confound on top of the dropout rate.
- 600 steps is a very short horizon. Any benefit from regularization would show over tens of thousands of steps.
- The wide NLL rises over the run in every arm (3.64 → about 4.55–4.61). This happens during early training from random
  init with an untrained champion's games, so it is not an effect of dropout.
- The logs report the step limit as reached at steps=600, but `training_steps` in each results JSON is 588 (the last stats
  snapshot the recorder captured). Treat the results JSON's step count as the snapshot step, not the exact stop step.
- Debug build with a dirty tree (`6e4e233*`). The exact source is only approximately `eacced3`.
- The fork session (`20260612-175644-…-4-gV3q-sigusr2.dcmsession`) no longer exists on disk, so its weights could not be
  checked through a safetensors header. Identity is taken from the log lines.

## Follow-ups

- Done the same day: the v2 four-arm repeat from the trained 5K7Z champion with a replicate control
  (`experiments/dropout-ab-trained-20260612/`). It also gave a null result.
- Open: a long-horizon dropout test (tens of thousands of steps). Under corpus replay the data is identical across arms,
  which removes the self-play data confound.

## Audit notes

- Mode, arms, limits and start model were derived from `run_experiment.sh` and confirmed in the four 2026-06-12 logs
  listed above (`[APP] --parameters overrides`, `--train: loading start model`, `[ARCH] loaded model`, `[PARAM]
  dropoutRate applied`, the step-limit exit lines).
- The fork being untrained: `dcm_log_20260612-125609.txt` has 0 `[STATS]` lines. SIGUSR2 arrived at 12:56:35 and the
  checkpoint was written at 12:57:00.
- The wide-probe numbers in the Results table were computed from every `tick set=wide` line in each arm log.
- The final-stats and candidate-probe numbers were read from `result_drop_*.json` (`stats[-1]`, `candidate_tests[-1]`).
- The CHANGELOG 2026-06-12 FINDING entry (gNorm ~1.47 at rate 0.70, ~38/128 channels surviving, matching first probe
  ticks, no arm distinguishable) is consistent with the data. gNorm at rate 0.70: mean 1.44 over the last 100 steps,
  1.35 at the final row.
- Session filenames use UTC time (`175644` = 12:56:44 CDT); log times are CDT.
- Unverified: the fork weights' `__metadata__`, because the session directory has since been deleted.
- No corrections: this is a new write-up.

## Reproduce

**Status: partial** — the script and parameter files are preserved, but the fork weights are deleted and the build was dirty.

- **Commit / build:** build 1824, git `6e4e233*` (dirty), Debug binary (`[APP]`). The committed harness and dropout code are at `eacced3` / `600c016` (see Caveats).
- **Corpus:** none (self-play).
- **Starting point:** the stage-1 fork `20260612-175644-20260612-4-gV3q-sigusr2.dcmsession/champion.safetensors` (loads as `20260612-3-elVE`, untrained). It **no longer exists**. Because it was a random init, a rerun gets a different one.
- **Parameters:** `params_drop_0.00.json`, `params_drop_0.30.json` and `params_drop_0.70.json` in this folder. Everything else was app defaults, including any `UserDefaults` in effect.
- **Commands:** `run_experiment.sh` in this folder. Per arm, it runs `--train --start-model <fork>/champion.safetensors --parameters params_drop_<R>.json --training-step-limit 600 --training-time-limit 2700 --output result_drop_<R>.json`.
- **Probe / analysis:** `result_drop_*.json` (`stats[-1]`) plus the arm logs listed in Runs.
- **Expected exactness:** statistical only. Self-play move sampling uses unseeded `Float.random` (`MoveSampler`), minibatch sampling uses unseeded `Int.random` (`ReplayBuffer.sample`), fresh nets use a random init with no seed flag, dropout draws a random seed, and GPU execution adds its own nondeterminism. A rerun can match the curves' shape and level, never bit-for-bit.
- **Missing:**
  - the fork weights and their `__metadata__`
  - the exact dirty-tree source
