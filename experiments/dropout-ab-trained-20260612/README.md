# 2026-06-12 — Channel dropout A/B (v2): rates 0 / 0 / 0.30 / 0.70 from the trained 5K7Z champion

**Status:** done — no effect distinguishable from the replicate noise floor at 600 steps.

## Question

v1 (`experiments/dropout-ab-20260612/`) found no effect from random init. Does channel dropout change learning when it is
applied to a trained net, which has real channel structure? A second 0.00 arm was added to measure run-to-run noise.

## Setup

- **Mode:** headless self-play Play-and-Train (`--train`), with the same harness as v1. The champion generates self-play
  games (170 workers) into a replay buffer (1,000,000 capacity) and the trainer trains on it. The arena was effectively
  off (`arena_auto_interval_sec` 36000), so the game-generating champion stayed fixed in every arm.
- **Only variable:** `dropout_rate` — 0.00 (arm A), 0.00 (arm B, replicate), 0.30, 0.70. The `params_drop_*.json` files are
  otherwise identical: `weight_decay` 0.0005, `arena_auto_interval_sec` 36000,
  `replay_buffer_min_positions_before_training` 200000. All other parameters were app defaults.
- **Start model:** `Sessions/20260612-191329-20260601-12-5K7Z-sigusr2.dcmsession/champion.safetensors`, the SIGUSR2 save
  of the live 5K7Z self-play session, which loads as champion `20260601-11-bzw3-32`. That session was at trainer step
  498,397 (first `[STATS]` of the resume log `dcm_log_20260612-140430.txt`). No fork stage.
- **Architecture:** `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu .
  policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params` (the `[ARCH] loaded model` line).
- **Arms:** `--train --start-model <above> --parameters <params> --training-step-limit 600 --training-time-limit 2700
  --output result_<ARM>.json`, run one after another. Afterwards the script restored the `LastSessionPointer` and
  relaunched the 5K7Z session.
- **Build:** varied by arm; see Runs and Caveats. Debug binary, branch `safetensors-storage`.
- **Machine:** this Mac (DerivedData hash `-eyigcdvyyrcsakaqcybzcfgsurbr`).

## Runs

| arm | log | results.json session_id | champion / trainer | build / git | steps (log) | steps in results.json | training time | self-play games |
|---|---|---|---|---|---|---|---|---|
| 0.00A | `dcm_log_20260612-141454.txt` | `20260612-13-0WGb` | `20260601-11-bzw3-32` / `-33` | 1824 / `6e4e233*` | 600 (step-limit exit) | 569 | 1159 s | 60,508 |
| 0.00B | `dcm_log_20260612-143424.txt` | `20260612-15-sCab` | same | 1825 / `6e4e233*` | 600 | 569 | 1147 s | 60,196 |
| 0.30 | `dcm_log_20260612-145341.txt` | `20260612-17-gGV4` | same | 1825 / `6e4e233*` | 601 | 568 | 1161 s | 60,328 |
| 0.70 | `dcm_log_20260612-151312.txt` | `20260612-19-Rihf` | same | 1827 / `0cb7ad7*` | 600 | 567 | 1143 s | 58,970 |
| resume | `dcm_log_20260612-153226.txt` | resumed `20260601-12-5K7Z` | champion `bzw3-32` | 1827 | — | — | — | — |

Each arm log shows its own `dropout_rate` in `--parameters overrides` and `[PARAM] dropoutRate applied to training graph`
(for example 0.7000 at 15:13:22 in the 0.70 arm). No arm checkpoints were kept; the results JSONs here are the only
artifacts.

## Results

Wide puzzle set (`[TACTICAL-LICHESS] tick set=wide`, trainer model `bzw3-33`), from the arm logs. pElo is on the June 2026
in-app probe scale, not the replay-era scale.

| arm | ticks | first tick (step): NLL / pElo | last tick (step): NLL / pElo | mean NLL | Δ mean NLL vs control mean (first 23 ticks) | mean pElo | NLL sd over ticks |
|---|---|---|---|---|---|---|---|
| 0.00A | 24 | (24) 3.179 / 888 | (579) 3.158 / 881 | 3.1583 | +0.0028 | 886.5 | 0.0170 |
| 0.00B | 23 | (24) 3.179 / 889 | (579) 3.116 / 893 | 3.1528 | −0.0028 | 889.3 | 0.0205 |
| 0.30 | 24 | (23) 3.181 / 885 | (579) 3.172 / 893 | 3.1558 | −0.0005 | 891.8 | 0.0125 |
| 0.70 | 24 | (24) 3.178 / 885 | (580) 3.138 / 902 | 3.1498 | −0.0053 | 891.5 | 0.0203 |

**Noise floor:** the two identical 0.00 arms differ by 0.0055 in mean NLL (±0.0028 around their mean), and a single run's
NLL moves 0.013–0.02 (sd) from tick to tick. Final pElo spans 881–893 between the two controls alone.

Final `stats` row of each results JSON:

| arm | step | pLoss | vLoss | pEnt | legal mass | gNorm | pD | vAbs |
|---|---|---|---|---|---|---|---|---|
| 0.00A | 569 | 2.1790 | 1.0001 | 2.5539 | 0.9985 | 1.0204 | 0.4003 | 0.3232 |
| 0.00B | 569 | 2.1840 | 1.0015 | 2.5575 | 0.9986 | 1.1053 | 0.3948 | 0.3286 |
| 0.30 | 568 | 2.1840 | 1.0027 | 2.5650 | 0.9984 | 1.0830 | 0.3969 | 0.3255 |
| 0.70 | 567 | 2.1806 | 1.0029 | 2.5614 | 0.9982 | 1.0639 | 0.3971 | 0.3271 |

## Conclusion

- There is no distinguishable effect. The 0.30 arm (−0.0005) sits inside the ±0.0028 replicate split. The 0.70 arm (−0.0053)
  is only about 2× the replicate half-spread and is well inside the per-tick sd, so it cannot be told apart from noise with
  n = 1 per arm. Final losses agree to the third decimal.
- Rate 0.70 stays stable on a trained net (gNorm about 1.06, pEnt 2.56, legal mass 0.998).
- The dropout machinery is validated end to end. Whether dropout helps is a long-horizon question (tens of thousands of
  steps), and a 600-step probe cannot answer it.

## Caveats

- **Build drift across arms:** A ran on build 1824, B and 0.30 on 1825 (all on git `6e4e233*`, dirty), and 0.70 on
  build 1827, git `0cb7ad7*`. `0cb7ad7` is the CHANGELOG commit on top of `eacced3` (dropout) / `600c016` (harness). The
  binary was rebuilt while the experiment was running. The per-build differences in the trained graph are unverified
  (dirty trees). The 0.70 arm is therefore confounded by build as well as rate.
- n = 1 per treated arm, with one replicate pair for the control.
- Self-play data differs per arm (58,970–60,508 games from the same fixed champion). This is a data confound that corpus
  replay would remove.
- The logs report the step limit reached at 600 (601 for 0.30), but `training_steps` in the results JSONs is 567–569. The
  JSON records the last captured stats snapshot, not the stop step.
- The start session (`20260612-191329-…-5K7Z-sigusr2.dcmsession`) no longer exists on disk, so its champion's
  `__metadata__` could not be read. Identity comes from the log's `[ARCH] loaded model champion.safetensors →
  20260601-11-bzw3-32` line.
- Session-folder timestamps are UTC (`191329` = 14:13:29 CDT). The script names a session that sorts "after" the 14:14
  launch only because of the time zone, not because of a later edit.

## Follow-ups

- Open: a long-horizon dropout comparison, preferably by corpus replay (same data in every arm) with at least two seeds per
  arm.
- The later "dropout=0.7 contamination" (ROADMAP: `--parameters` leaked into `UserDefaults`, fixed 2026-06-12) followed from
  these arm launches. The resume log `dcm_log_20260612-140430.txt` shows `[RESUME-PARAM] dropout_rate: saved=nil
  applied=0.7 (defaulted)`, meaning the v1 0.70 arm's rate leaked into the resumed 5K7Z session.
  `dcm_log_20260612-153226.txt` shows `dropout_rate: 0.7 -> 0.7 (from session)` after v2. This is recorded here only as
  provenance.

## Audit notes

- Script → logs: all four arm launches and their order (0.00A, 0.00B, 0.30, 0.70) match the `--parameters overrides`
  lines at 14:14:54, 14:34:24, 14:53:41 and 15:13:12. Each arm loads the same start file and champion `bzw3-32`.
- Start-model step: 498,397 at the first `[STATS]` of the 14:04 resume, which confirms CHANGELOG's "step-~498k".
- CHANGELOG 2026-06-12 v2 FINDING checked: NLL deltas −0.0005 (0.30) / −0.0053 (0.70) and the control split ±0.0028
  reproduce exactly from the first 23 wide ticks of each log (0.00B has only 23). Over all 24 ticks for the other arms the
  deltas would be +0.0003 / −0.0058; the CHANGELOG figures use matched ticks. "Mean pElo 886–892 across all four arms" is
  confirmed (886.5 / 889.3 / 891.8 / 891.5).
- Final-stats numbers were read from `result_*.json` `stats[-1]`. Builds were read from both `stats[-1].build_number` and
  the `[APP]` lines (agree).
- One correction relative to the script's header comment: "four arms from the TRAINED 5K7Z champion" is accurate at the
  session level (5K7Z is the session ModelID), but the champion weights' ModelID is `20260601-11-bzw3-32`. Evidence:
  `[ARCH] loaded model champion.safetensors → 20260601-11-bzw3-32` in each arm log. The script was not edited.
- Unverified: the start champion's safetensors `__metadata__` (the session directory is deleted), and the source
  differences between builds 1824/1825/1827 (dirty trees).
