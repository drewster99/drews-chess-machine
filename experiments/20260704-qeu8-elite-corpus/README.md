# 2026-07-04 — qeu8 architecture on the elite corpus (Qeu8e … Qeu8e5)

**Status:** done. Five chained segments, cum 220,837 steps / 42.02 h / ~10.4 epochs. Stopped 2026-07-09 00:04.

## Question

If the qeu8 architecture is trained from the same untrained seed on a small, high-quality corpus (lichess elite, 2.02M games) instead of the large std corpus (20.9M games), does it learn more per hour, per step or per game? Does repeated-epoch training on a small corpus keep improving?

## Setup

- **Architecture:** identical to qeu8. basic30 → stem 7×7→64 · 2×[15×15+15×15 @64, SE scale+bias/4, ReLU/pre, ReZero 0.5·tanh, LN-out] · policy intermediate_conv (pre-conv 512) · value WDL(16→FC64) · bf16. **3,929,393 params**, taken from the tensor sum and architecture JSON of the `20260704-Qeu8e*` headers and matching `engines_models.tsv`.
- **Seed:** `20260702-7-Qeu8` (the GUI-built qeu8 seed, file `20260702-164826-20260702-7-Qeu8-manual.safetensors`). This is the same starting point as the std-corpus qeu8 run. Header of Qeu8e X79T: `parent_model_id = 20260702-7-Qeu8`.
- **Corpus:** `20260704-001142-lLOrLj` = `lichess_elite_2025-05_to_11`.
  - 2,023,146 games, 181,908,093 plies (89.9 plies/game), 6 shards.
  - Imported by build 2007 `f67621f`. corpus.json has `complete: true`.
  - It is **not** the full elite corpus `20260704-215145-op24Gp`, which was never trained on.
- **Hyperparameters:**
  - e, e2, e3 used lr=0.01, batch 4096, **wd=0.00025, momentum=0.93**, gradClip 30, pLabelSmooth 0.1, vLabelSmooth 0.013, replayRatio 0.48. These are identical to std qeu8 seg0 (`[REPLAY-HPARAMS]`).
  - **e4 and e5 changed to wd=0.0005, momentum=0.9.** This confounds the later segments.
- **Builds:** 2013 / `48fd638` (e, e2, e3); 2028 / `31cddae` (e4); 2029 / `c8ea6e1` (e5).
- **Tracking:** separate elite dashboard `documentation/dashboards/elite/` (registry key `qeu8e`, `elite/data/qeu8e.csv`, 224 rows). It is not in the main `registry.json`.

## Runs

Each segment restarts the corpus at game 0: prefill `gamesFed=5382`.

| seg | out stem | log | start → ModelID | steps | games fed (`done:`) | epochs | cum base |
|---|---|---|---|---:|---:|---|---:|
| e | `20260704-Qeu8e` | `dcm_log_20260703-175402.txt` | 7-Qeu8 → **X79T** | 21,224 | 2,023,146 | 1 | 0 |
| e2 | `20260704-Qeu8e2` | `dcm_log_20260704-015814.txt` | X79T → **jSjr** | 42,507 | 4,046,292 | 2 | 21,224 |
| e3 | `20260704-Qeu8e3` | `dcm_log_20260704-110015.txt` | jSjr → **h7Pp** | 42,507 | 4,046,292 | 2 | 63,731 |
| e4 | `20260704-Qeu8e4` | `dcm_log_20260708-023905.txt` | h7Pp → **0YQL** | 26,492 | 2,514,986 | 1 + 491,840 games | 106,238 |
| e5 | `20260704-Qeu8e5` | `dcm_log_20260708-081952.txt` | 0YQL → **sFzi** | 88,107 | 8,375,169 | 4 + 282,585 games | 132,730 |

- **Totals:** 220,837 steps, **21,005,885 games** (10.38 passes over the corpus), 42.02 h `elapsed_train_sec`.
- All logs are present.
- Surviving checkpoints: 229 files matching `20260704-Qeu8e*` (e 23, e2 44, e3 44, e4 28, e5 90).

## Results

**Probe curve** (`elite/data/qeu8e.csv`; pElo / nll on the replay-era probe scale), compared with std-corpus qeu8 (`data/qeu8.csv`) at matched training time:

| T (h) | qeu8e (elite) pElo / nll (cum) | qeu8 (std) pElo / nll (cum) |
|---:|---|---|
| 3.0 | 1165 / 2.642 (15,000) | 1207 / 2.522 (16,000) |
| 5.5 | 1230 / 2.514 (28,224) | 1352 / 2.360 (30,000) |
| 8.9 | 1378 / 2.350 (46,224) | 1440 / 2.249 (49,407) |
| 17.0 | 1397 / 2.344 (86,731) | 1522 / 2.126 (97,407) |
| 25.9 | 1459 / 2.274 (130,238) | 1608 / 2.067 (146,915) |
| 27.8 | 1377 / 2.361 (140,730) | 1568 / 2.127 (156,915) |
| 31.3 | 1410 / 2.322 (160,730) | 1591 / 2.075 (174,915) |

Peaks and ends:

| | qeu8e (elite) | qeu8 (std), ≤ 31.3 h |
|---|---|---|
| peak pElo | **1492.9** (cum 126,238, 24.98 h, during e4) | 1630.9 (cum 169,915, 30.20 h) |
| best nll | **2.2014** (cum 132,238, 26.19 h, end of e4) | 2.0119 (cum 171,915) |
| end | 1441.0 / 2.271 (cum 220,837, 42.02 h) | |

- During e5 (new hyperparameters, epochs 6–10) pElo fell from ~1460–1480 to 1367–1405 between cum ~175k and ~215k, then recovered to 1441 at the end.
- The by-epoch trend flattens after epoch 3: 1397 at the end of e2, then 1425–1441 at e3's marks, and no further gain after e4.

**Arena** (2026-07-08, 38 engines, Ordo, Stockfish = 1320):

| engine (model_id) | position | rating ± err |
|---|---|---:|
| Qeu8e (X79T) | cum 21,224, 1 epoch | 382.0 ± 10.5 |
| Qeu8e2 (jSjr) | cum 63,731, 3 epochs | 499.5 ± 9.9 |
| Qeu8e3 (h7Pp) | cum 106,238, 5 epochs | **535.5** ± 9.4 |
| Qeu8e4 (0YQL) | cum 132,730 | 518.4 ± 10.8 |
| DCM - Qeu8e5 (frozen step 14,000, sFzi) | cum 146,730 | 528.1 ± 9.8 |
| Qeu8e5 (latest), a moving target during the arena | e5 still training | 499.0 ± 10.9 |
| *std* Qeu8 (Lnji) | cum 108,915 | 567.1 ± 11.9 |
| *std* Qeu8-resume2 (PVZp) | cum 175,915 (31.33 h) | 588.4 ± 10.8 |

Head-to-head (`h2h.txt`, 100 games each):
- std Qeu8 (cum 108,915) vs Qeu8e3 (cum 106,238), near-matched steps: **53.5%** for std (44-19-37).
- Qeu8e3 vs the frozen Qeu8e5 step 14,000: 49.0%.
- Qeu8e3 vs Qeu8e4: 54.0%.
- Qeu8e3 vs Qeu8e5 (latest): 53.0%.

## Conclusion

- **The small elite corpus lost to the large std corpus at every matched time from 3 h on.** It was ~40–190 pElo and ~0.10–0.25 nll behind. At near-matched steps (~107k) the std checkpoint also won the head-to-head, 53.5% (Ordo +31.6).
- **Repeated epochs on 2M games saturate around epoch 3–5.** Arena strength peaked at Qeu8e3 (5 epochs, 535.5). e4 and e5 did not beat it: 518.4, 528.1, 499.0. The probe peak (1492.9) and best nll (2.2014) both fall in e4, and e5 went backwards before a partial recovery.
- **Per game** the elite run saw 21.0M game-passes over 42 h, but only 2.02M distinct games. At parity std qeu8 had fed 22.6M games (see `20260629-replay-arch-parity`), almost all distinct, and was clearly stronger. For this net, data diversity matters more than game quality.

## Caveats

- e4 and e5 changed wd (0.00025 → 0.0005) and momentum (0.93 → 0.9) as well as continuing the epochs. The e5 decline cannot be attributed to over-repetition alone.
- Each segment re-read the corpus from game 0 and restarted optimizer velocity, so epoch boundaries and resume boundaries coincide.
- Single seed. The arena "latest" Qeu8e5 entry was overwritten mid-tournament, so its rating blurs several states. Use the frozen step-14,000 entry.
- The arena used the old `.arena` sampling schedule (noise through ~ply 45) and no opening book.

## Follow-ups

- Mix elite games into the std corpus instead of training on elite alone. Not done.
- The full elite corpus `20260704-215145-op24Gp` (14.6M games) was built but never trained on. It would separate "quality" from "quantity".

## Audit notes

- Verified from the logs: segment chain, steps and games (`start-model:` / `done:` lines), and the resulting cum bases (21,224 / 63,731 / 106,238 / 132,730 / 220,837). These match the elite registry `cumstep_base` values exactly.
- Verified from `corpus.json`: corpus identity, `2,023,146 games`, and that the Qeu8e runs used lLOrLj and not op24Gp, as memory `dcm-parity-goal.md` states. Every `20260704-Qeu8e*` header also has `replay_corpus_id = 20260704-001142-lLOrLj`.
- Verified: X79T's parent is `20260702-7-Qeu8`, so this is a fresh run from the qeu8 seed, not a fine-tune of std-trained qeu8.
- **New finding, not in any note:** the hyperparameters changed in e4 and e5 (`[REPLAY-HPARAMS]` wd=0.0005 momentum=0.9 in `dcm_log_20260708-023905.txt` and `dcm_log_20260708-081952.txt`).
- Commit `1bf45ff` states "stop point (cum 220,837, 42.0h)". This is confirmed: last CSV row 151,260.2 s = 42.02 h.
- **Unverified:**
  - Why e4 and e5 used different wd/momentum: no note or commit explains it. It may reflect changed `parameters.json` defaults between builds 2013 and 2028.
  - The e4 stop reason: the header says "abort @ step 26492", and the log is present but was not inspected for intent.
