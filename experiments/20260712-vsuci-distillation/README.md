# 2026-07-12 — Train-vs-UCI distillation from the Qeu8 init (sf10 / sf100 / sf200 / sloppy20 / sf100sl100)

**Status:** done (sf100sl100 terminated by the user 2026-07-27 09:40 after 1,536,000 cum steps; the four short configs were exploratory and superseded within the same day)

## Question

Starting from the same untrained init as the qeu8 corpus-replay run (`20260702-7-Qeu8`, 3,929,393 params), does training on games played against external UCI engines (both sides' moves distilled into the replay buffer) produce a stronger policy than corpus replay?

Short answer: **no.** At every matched step the corpus-replay curve is far ahead, and the gap never closes (table below).

## Setup

- **Architecture** (from every log's `[VS-UCI-ARCH]` line, identical in all runs, and identical to the qeu8 replay run's `[REPLAY-ARCH]`):
  `v5 . in basic30(30) -> stem 64 (7x7) . 2x[15x15+15x15 @64, SE+/4, relu/pre, clean_add, ReZero(0.5·tanh≤0.5), out:layer_norm, drop*1] . act relu . policy intermediate_conv(4864) . value WDL(16->FC64) . bfloat16 . 3,929,393 params`
- **Start model:** `20260702-164826-20260702-7-Qeu8-manual.safetensors` (`__metadata__` model_id `20260702-7-Qeu8`, no training_step, content_sha256 `9b11cbec…`). The qeu8 corpus-replay run (`dcm_log_20260702-095124.txt`) loaded the same file, so the comparison is from an identical init.
- **Mode:** `--train-vs-uci` (see `documentation/UCI.md`, Part 2). DCM plays every game against the engine pool; the terminal outcome is signed by the mover's colour, so even a ~100%-loss run fills the buffer about half with win-labelled engine positions whose policy target is the engine's move. The `W-L-D` in `[VS-UCI-STATS]` is the trainer-side match score, not the buffer's balance.
- **Trainer config** (every log): `batchSize=4096 minPrefill=500000 evalSyncEvery=10 stepLimit=none timeLimit=none`; buffer fills to 1,000,000 positions; lr 2e-05 at step 1 warming to 0.01.
- **Opponents:** `~/bin/stockfish` and `~/bin/sloppy`, every entry `go=movetime 10`. No strength options are set in the pool spec (Stockfish runs full strength, time-limited only).
- **Build/git:** unverified — the `--train-vs-uci` CLI path writes no `[APP] launched` line. The resume recipe in `~/dcm-sf100sl100-backup/RESUME.md` names the Release binary under DerivedData hash `-eyigcdvyyrcsakaqcybzcfgsurbr`.
- **Machine:** this Mac (logs are local). sf100sl100 seg2 ran partly on battery / Low Power Mode (see Caveats).
- **Registry / data:** `documentation/dashboards/vsuci_registry.json` (run `sf100sl100` only), `documentation/dashboards/data/sf100sl100.csv`, probe trajectory `~/dcm-sf100sl100-backup/sf100sl100_pelo.jsonl` (1,452 rows, cum 6,000 → 1,536,000), sf200 probes `~/dcm-sf100sl100-backup/vsuci_probes.jsonl` (13 rows).

## Runs

All runs start from `20260702-7-Qeu8`. Opponent pool and go timing are copied from each log's `[VS-UCI] opponent pool:` line. Checkpoint identities are from safetensors `__metadata__` (model_id / training_step), not filenames.

| Run | Log | Opponent pool (log) | Steps | Wall duration | End | Trainer ModelID | Surviving checkpoints (metadata step) |
|---|---|---|---|---|---|---|---|
| (aborted pre-attempt) | `dcm_log_20260712-121730.txt` | `stockfish×6 [go=movetime 10], sloppy×6 [go=movetime 10]` | 0 | 12:17 → 12:19 | all 6 Sloppy instances "failed to start: UCI engine did not respond within the timeout"; no training step logged; 4,426 games vs Stockfish (0 wins) | — | none (out-model was `20260712-qeu8init-vsuci-latest`) |
| sf10 | `dcm_log_20260712-123811.txt` | `stockfish×10 [go=movetime 10]` | 1,250 | 12:38 → 12:57 (~19 min) | log stops mid-run; next config launched 6 s later (superseded; no exit line) | `20260712-2-utcG` | `sf10-vsuci-step1000` = `-latest` (step 1000, identical sha) |
| sf100 | `dcm_log_20260712-125723.txt` | `stockfish×100 [go=movetime 10]` | 3,150 | 12:57 → 13:35 (~38 min) | superseded (next launch 11 s later) | `20260712-3-7tGs` | step1000, step2000, step3000 (= `-latest`) |
| sf200 | `dcm_log_20260712-133524.txt` | `stockfish×200 [go=movetime 10]` | 15,250 | 13:35 → 16:49 (~3.2 h) | superseded (next launch 15 s later) | `20260712-4-s3bg` | step1000 … step15000 (15 files; `-latest` = step 15000) |
| sloppy20 | `dcm_log_20260712-164958.txt` | `sloppy×20 [go=movetime 10]` | 4,500 | 16:50 → 17:39 (~50 min) | superseded (sf100sl100 launched 25 min later) | `20260712-5-iOfG` | step1000 … step4000 (= `-latest`) |
| **sf100sl100** seg0 | `dcm_log_20260712-180423.txt` | `sloppy×100 [go=movetime 10], stockfish×100 [go=movetime 10]` | 220,650 logged (weights kept to 220,000) | 2026-07-12 18:04 → 07-14 12:13 | paused for a macOS update (RESUME.md) | `20260712-6-lTiK` | `…-step220000-STOP` (lTiK, step 220000) |
| sf100sl100 seg1 | `dcm_log_20260714-123243.txt` | same | 758,050 logged (kept 758,000) | 07-14 12:32 → 07-20 18:13 | **GPU fault:** `failed: GPU command buffer failed during working-weight sync: status=MTLCommandBufferStatus(rawValue: 5), error=Internal Error … training halted.` | `20260714-1-NYAZ` | `…-step978000-STOP` (NYAZ, metadata step **758000** = segment-local; cum 978,000) |
| sf100sl100 seg2 | `dcm_log_20260722-102524.txt` | same | 558,500 logged (kept 558,000) | 07-22 10:25 → 07-27 09:40 | user terminated ("terminate this run and continue qeu8 indefinitely") | `20260722-1-syxR` | `…-step558000-STOP` = `-latest` (syxR, step 558000; cum 1,536,000) |

Each sf100sl100 segment is a warm restart: weights only; optimizer, replay buffer and step counter reset. cum_step = cumstep_base (0 / 220,000 / 978,000) + segment step.

Games played (cumulative per process, from the last `[VS-UCI-STATS]` aggregate lines; trainer-perspective W-L-D):

| Run | vs Stockfish: games (W-L-D) | vs Sloppy: games (W-L-D) |
|---|---|---|
| sf10 | 33,110 (0-33,110-0) | — |
| sf100 | 168,533 (0-168,531-2) | — |
| sf200 | 1,068,670 (1,903-1,066,346-421) | — |
| sloppy20 | — | 83,310 (0-82,847-463) |
| sf100sl100 seg0 | 6,849,217 (134,408-6,659,634-55,175) | 9,158,076 (9,878-5,124,721-4,023,477) |
| sf100sl100 seg1 | 22,972,440 (619,353-21,840,127-512,960) | 31,458,351 (132,484-12,215,922-19,109,945) |
| sf100sl100 seg2 | 17,124,611 (471,398-15,982,769-670,444) | 23,919,667 (179,159-8,031,813-15,708,695) |

sf100sl100 total: 111,482,362 games (seg0 16,007,293; seg1 54,430,791; seg2 41,044,278). Seg0 plies/game: Sloppy games 40.5, Stockfish games 50.0 (plies ÷ games from the same lines).

## Results

Puzzle pElo / nll on the `wide` probe set (current probe scale, same as the replay-era qeu8 numbers). qeu8 corpus-replay values from `documentation/dashboards/data/qeu8.csv`; its enumerated checkpoints land on cum steps ending in …915 after segment 1, so the nearest row is used and shown.

| cum step | qeu8 corpus replay pElo / nll (row) | sf100sl100 pElo / nll | sf200 pElo / nll | gap (qeu8 − sf100sl100) |
|---|---|---|---|---|
| 1,000 | 903.3 / 3.1243 | — (no probe) | 531.9 / 5.7772 | — |
| 5,000 | 1097.0 / 2.7030 | — | 803.0 / 3.6747 | — |
| 10,000 | 1175.4 / 2.5694 | 926.6 / 3.3437 | 868.9 / 3.3732 | +248.8 |
| 13,000 | 1166.5 / 2.5728 | 920.0 / 3.3112 | 879.2 / 3.3931 | +246.5 |
| 15,000 | 1238.6 / 2.4996 | 943.1 / 3.3568 | — | +295.5 |
| 20,000 | 1330.0 / 2.3910 | 902.2 / 3.3679 | — | +427.8 |
| ~220,000 | 1635.0 / 2.0335 (219,915) | 1179.5 / 2.7135 | — | +455.5 |
| ~300,000 | 1601.1 / 2.1207 (299,915) | 1173.3 / 2.7316 | — | +427.8 |
| ~978,000 | 1653.0 / 1.9915 (977,915) | 1304.2 / 2.5708 | — | +348.8 |
| ~1,302,000 | 1625.3 / 2.0705 (1,301,915) | **1434.3** / 2.3518 (all-time peak) | — | +191.0 |
| ~1,536,000 | 1585.7 / 2.1187 (1,535,915) | 1375.8 / 2.4397 | — | +209.9 |
| best ≤ 1,536,000 | **1742.1** @ 856,915; nll **1.8734** @ 1,509,915 | **1434.3** @ 1,302,000; nll **2.3134** @ 1,446,000 | 879.2 @ 13,000 | +307.8 (peak vs peak) |

Corrected training time at matched steps (CSV `elapsed_train_sec`): qeu8 39.4 h at 219,915 vs sf100sl100 42.0 h at 220,000; sf100sl100 reached 1,536,000 at 301.4 h vs qeu8 276.8 h at 1,535,915. Games-fed comparison: unverified (qeu8.csv has no `games_fed` column; vs-UCI "games" are generated, not corpus games fed).

sf100sl100 pElo by phase (jsonl):

| cum range | marks | mean | min | max |
|---|---|---|---|---|
| 6k–20k | 15 | 920.1 | 794.0 | 1008.0 |
| 20k–120k | 99 | 1058.0 | 902.2 | 1148.3 |
| 120k–220k | 101 | 1147.4 | 1076.0 | 1206.5 |
| 221k–300k (seg1 start) | 80 | 1181.9 | 1125.3 | 1238.6 |
| 300k–978k | 604 | 1257.1 | 1128.5 | 1344.9 |
| 979k–1,266k (seg2) | 286 | 1325.9 | 1223.1 | 1423.0 |
| 1,266k–1,536k | 271 | 1349.1 | 1256.2 | 1434.3 |

Short configs: sf200 13 probe points 531.9 (1k) → 879.2 (13k), nll 5.777 → 3.393. sf10, sf100 and sloppy20: no probe record found (unverified). Their last logged training lines: sf10 step 1250 pLoss 1.24 vLoss 0.170; sf100 step 3150 pLoss 1.58 vLoss 0.252; sloppy20 step 4500 pLoss 2.14 vLoss 0.594.

## Conclusion

- Distilling Stockfish/Sloppy play from the Qeu8 init **does not beat corpus replay** from the same init at any matched step: the gap is ~+250 pElo at 10k, widens to ~+455 at 220k, and is still ~+210 at 1.536M. The vs-UCI peak (1434.3) is 308 below the corpus-replay peak (1742.1), which came at fewer steps (856,915).
- The vs-UCI net keeps improving slowly over 1.5M steps and three warm restarts (band ~1150 → ~1350) while corpus replay plateaus near ~1600–1650 from ~220k onward — so the gap narrows, but from far behind and never near parity.
- Early on the vs-UCI data is also slower per step (first probes 532 @1k for sf200 vs 903 @1k for replay), i.e. the sampled positions against engines at 10 ms/move are a worse teacher per batch than the human corpus for this net, at least for the puzzle metric.

## Caveats

- Puzzle pElo measures argmax move-matching on tactics; it does not directly measure playing strength against engines. No arena or engine-match comparison between the vs-UCI net and the qeu8 replay net was found.
- One run per configuration; no seed replication. The short configs were stopped after 19 min – 3.2 h, so they say nothing about long-run behaviour.
- Warm restarts reset optimizer, buffer and LR warmup; each segment re-prefills ~500k positions before stepping.
- Time axis: seg2 ran inside three declared throttle windows (battery / Low Power Mode, ~1.8–1.9 s/step vs ~0.86 s baseline); `vsuci_registry.json` rescales those windows and strips system sleep, giving 301.4 h corrected vs ~311 h of wall-clock across the three processes.
- 79 of the 1,531 thousand-step marks between 6k and 1,536k have no probe row (1,452 present).
- The step reported in seg0/1/2 logs overshoots the saved STOP weights by 650 / 50 / 500 steps; those final steps were discarded.
- Build/git of the binary is unverified (not logged).

## Follow-ups

- Head-to-head arena (or cutechess match) between `…-step558000-STOP` (syxR) and a qeu8 replay checkpoint of similar pElo, to check whether the puzzle gap reflects a playing-strength gap.
- A mixed buffer (corpus + vs-UCI games) was not tried.
- `--train-vs-uci` still does not checkpoint optimizer / buffer / step (noted as "address later" at the time).

## Audit notes

Verified against primary data:
- Opponent pools, go timing, start models and architecture: each log's `[VS-UCI] opponent pool:`, `start-model:` and `[VS-UCI-ARCH]` lines (quoted above).
- Checkpoint identities: safetensors `__metadata__` of every `20260712-qeu8init-*` file (model_id, training_step, content_sha256). `-latest` files are byte-identical (same sha) to the final step file of each run.
- Memory note figures (`project_sf100sl100_paused_reboot.md`), against `sf100sl100_pelo.jsonl` and `data/sf100sl100.csv`:
  - cum 1,536,000 — confirmed (jsonl max step 1,536,000; CSV last row 1,536,000; STOP file syxR step 558,000 + base 978,000).
  - records pElo 1434.3 @ cum 1,302,000 — confirmed (1434.33).
  - nll 2.3134 @ cum 1,446,000 — confirmed (2.31343); top-5 3308 @ 1,446,000 — confirmed.
  - ~301 h corrected — confirmed (CSV `elapsed_train_sec` 1,085,059 s = 301.4 h).
  - 220k pause pElo ~1180 — confirmed (1179.5 @ 220,000; recent-10 mean 1173.9, RESUME.md "~1174").
  - pre-pause peak 1207 @ 203k — confirmed (1206.5); nll 2.6277 @ 203k — confirmed.
  - peak 1238.6 @ 290k — confirmed (1238.63; max over ≤300k). best nll 2.6147 @ 272k — confirmed.
  - "Final plateau ~1352 held ~270k steps" — approximately: mean 1349.1 over cum 1,266k–1,536k (range 1256–1434).
  - "~1452 marks" — confirmed (1,452 rows). RESUME.md "213 pts" at the pause — confirmed (213 rows ≤ 220,000).
- Correction: memory note "Combined run = 339 marks, cum 1000→339000" -> probe marks start at cum **6,000** (332 rows ≤ 339,000); cum 1,000–5,000 exist only as training-loss rows in the CSV (evidence: jsonl min step 6000).
- Correction (registry wording, not edited here): `vsuci_registry.json` labels seg2 "warm restart after macOS update reboot", but seg1 ended with a **GPU command-buffer failure** (`dcm_log_20260714-123243.txt`, 18:13:36 on 2026-07-20, file mtime), not a planned stop; the restart came ~40 h later. Whether a macOS update also happened in that gap is unverified.
- Correction: the frozen `…-step978000-STOP.safetensors` carries `training_step` **758000** (segment-local), not 978000; the name encodes the cum step. Identify it by model_id `20260714-1-NYAZ`.
- Unverified: RESUME.md "Arc 776 (untrained)" — no probe row for the init was found; the earliest vs-UCI probe is 794.0 @ cum 6,000. Binary build/git. pElo for sf10 / sf100 / sloppy20 (no probe files). Why the short configs were stopped beyond "superseded by the next launch" (logs end without an exit line). games_fed for either run.
