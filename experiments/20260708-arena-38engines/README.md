# 2026-07-08 Arena — 38-engine round-robin

Round-robin ranking the tracked DCM replay-model lines against each other and
against a deliberately-weakened Stockfish and an un-weakened Sloppy baseline.

- **Dates:** started 2026-07-08 evening, finished 2026-07-09 (overnight).
- **Games analyzed:** 70,538 (a 38-engine round-robin, 37 rounds, ~100 games/pair).
- **One-line result:** even at its weakest setting Stockfish is ~700–800 Elo
  above every DCM (best DCM scored 1/100 vs it); Sloppy swept the field; the
  strongest DCM lines are **v5** and **Qeu8-resume2**; Qeu8e5 lands mid-pack.

## Files in this archive

| File | What it is |
|---|---|
| `README.md` | this manifest |
| `ratings.txt` | **raw Ordo output** — the rating list |
| `h2h.txt` | **raw Ordo output** — the full head-to-head crosstable |
| `engines_models.tsv` | every engine → its Model file → resolved model id → parameter count |
| `engines.json` | exact cutechess engine config used (snapshot) |
| `cutechess.ini` | cutechess settings snapshot (time control etc.) |
| `games_full.pgn.gz` | **all 70,638 games** (gzip), including the 100 warmup tests |
| `slice_tests.py` | strips the 100 leading warmup games |
| `reproduce.sh` | decompress → slice → re-run the exact Ordo analysis |

The `.txt` files are **direct Ordo output — not post-processed**. The only
pre-processing was on the PGN: stripping the 100 stray warmup games (below).

## Setup & tool versions

- **Host:** Apple M5 Max, macOS 26/27 beta (26A5378j).
- **cutechess:** v1.5.1 (`v1.5.1-2-ge471973a`), launched with **`-style fusion`**
  (native QMacStyle crashes on this macOS beta — see Caveats).
- **Ordo:** 1.2.6 — rating analysis (https://github.com/michiguel/Ordo).
- **Stockfish:** recent build, `EvalFile = nn-71d6d32cb962.nnue`. **Weakened**,
  and the settings were verified as actually applied (in `sf_debug.log`):
  - `UCI_LimitStrength = true`
  - `UCI_Elo = 1320`  ← the *floor*; the weakest Stockfish's limiter allows
  - `Skill Level = 20` (ignored while LimitStrength is on)
- **Sloppy:** bundled xboard engine — **not** weakened.
- **DrewsChessMachine:** build **2033**, git `b5adf14` (dirty — includes the
  bare-name `Model` resolver and the space-preserving `setoption` parser landed
  in the session that produced this run). Each DCM engine is a **single
  forward-pass policy net** (no search, no time management); `Temperature = 0`
  (the default `.arena` sampling schedule). Models resolve against
  `~/Library/Application Support/DrewsChessMachine/Models/`.

  > **NOTE (behavior changed after this run):** at the time of this tournament,
  > `Temperature = 0` meant the decaying `.arena` exploration schedule
  > (`startTau 2.0 → 0.2`), so every DCM engine here played with real sampling
  > noise through ~ply 45 — it did **not** play argmax. The UCI default was
  > later changed so `Temperature = 0` floors to tau 0.01 (≈ argmax,
  > deterministic best move). A rerun on the current build will therefore play
  > stronger; get game variety from an opening book, not from temperature.
  > These ratings understate the nets' best play accordingly.

## Time control

- **40 moves / 5 s**, no increment.
- **No ply limit, no node limit** — games ran to a natural end (mate / stalemate
  / 3-fold / 50-move). Concurrency ≈ 2 (observed; not persisted in the ini).
- Consequence: weak nets that shuffle pieces dragged games out, so the run
  ballooned to ~70k games and took many hours. A future run should set a ply cap.

## Engines & models

38 engines total: **36 DCM** (31 `-replay-latest` line-heads + 5 earlier ad-hoc
configs) + **Stockfish** + **Sloppy**. Full mapping — engine, model filename,
resolved model id, parameter count — is in **`engines_models.tsv`**.

## Games

- **`games_full.pgn.gz` holds 70,638 games.**
- The **first 100** are stray warmup tests: `DCM - Qeu8e5` vs `Stockfish`, a
  2-engine match run right before the tournament (also tagged `Round 1`, so they
  can't be told apart by round number). `slice_tests.py` removes them.
- The **real tournament is games 101–70,638 = 70,538 games**, 38 engines,
  37 rounds.

## Results & how to read them

Point ratings are in `ratings.txt`; pairwise records in `h2h.txt`.

**`ratings.txt` columns:** `#` rank · `PLAYER` · `RATING` (Elo, Stockfish anchored
to 1320) · `ERROR` (± at 95%; with `-V`, relative to the **pool average**) ·
`POINTS` (win=1, draw=½) · `PLAYED` · `(%)` score · `CFS(%)` = **Confidence For
Superiority**, Ordo's probability this engine is truly stronger than the one
**ranked directly below it**.

**`h2h.txt` columns** (per opponent, from this engine's view): `games ( +, =, - )`
= total, wins, draws, losses · `(%)` score vs that opponent · `Diff` Elo
difference (this − opponent) · `SD` std-dev of that difference · `CFS (%)`
confidence this engine beats **that specific** opponent.

**Key findings:**
1. **Stockfish (1320) still dominates.** Best DCM (`v5_5block_7x7_lnout-wd2.5e4-m93`)
   went **1-0-99** vs it; `Qeu8e5 (latest)` **0-1-99**; frozen `DCM - Qeu8e5`
   **1-0-99**. DCMs sit ~700–800 Elo below crippled Stockfish → the "DCM beats
   Stockfish sometimes" goal is unreachable without also capping Stockfish's
   search (node/time limit — not done here).
2. **Strongest DCM lines:** the **v5** family and **Qeu8-resume2**; Qeu8e5 mid-pack.
3. **Sloppy swept all opponents** → Ordo purges it (100% ⇒ rating = +∞), so it
   appears only in `h2h.txt`, not `ratings.txt`.

## Caveats / non-obvious things

1. **Live `-replay-latest` weights are a MOVING TARGET.** Runs still training
   (Qeu8e5 among them) had their `-replay-latest` file overwritten by the trainer
   *during* the tournament, so those engines' ratings blur across shifting
   strength. Tell-tale: frozen `DCM - Qeu8e5` (step14000, #11) rated *higher*
   than `DCM Qeu8e5 (latest)` (#18). **For reproducible ladders, use frozen
   `-stepNNNN` files.** The games are therefore NOT bit-reproducible; only the
   analysis is.
2. **Sloppy purge** — see above; 100% score has no finite Elo.
3. **Anchor vs pool-relative:** Stockfish=1320 is just a scale label; `-V` reports
   `ERROR` relative to the pool average. To compare two engines rigorously, use
   the **h2h CFS**, not the overlap of the pool-relative bars.
4. **cutechess hung at the very end** (100% CPU, no engine subprocesses) — a Qt
   `QMacStyle` × macOS-beta rendering bug hit during finalize/redraw. All games
   were already played; the hang did not affect the data. Launch with
   `-style fusion` to avoid it.

## Reproduce the analysis

```sh
./reproduce.sh      # decompress → strip 100 warmup games → run Ordo
```
Needs `ordo` (1.2.6), `gzip`, `python3`. See the script for the exact flags
(`ordo -a 1320 -A Stockfish -V -s 100 -J ...`).

## Audit notes

Re-checked 2026-09-29 against `ratings.txt`, `h2h.txt`, `engines_models.tsv`, `engines.json`, `cutechess.ini`, a streamed parse of `games_full.pgn.gz`, and the safetensors `__metadata__` of every model file that still exists. The original text above is left as written; corrections are listed here.

**Confirmed**
- `games_full.pgn.gz` holds 70,638 games; the leading contiguous block of `DCM - Qeu8e5` vs `Stockfish` games is exactly 100; the remainder is 70,538 games, 38 distinct engines, 37 rounds, 703 pairings (= C(38,2)). Results: 26,145 `1-0`, 26,214 `0-1`, 18,179 draws. Every game is tagged `TimeControl "40/5"`; `cutechess.ini` has `moves_per_tc=40`, `time_per_tc=5000`, `increment=0`, `ply_limit=0`, `node_limit=0`. PGN dates span 2026.07.08–2026.07.09.
- Stockfish options in `engines.json`: `UCI_LimitStrength=true`, `UCI_Elo=1320`, `Skill Level=20`, `EvalFile=nn-71d6d32cb962.nnue`, `Threads=1`, `Hash=16`. Sloppy runs over `xboard` with no strength options.
- Sloppy won all 3,800 of its games (1,900 as White, 1,900 as Black) and is "Removed from calculation" in `h2h.txt`.
- Best-rated DCM `v5_5block_7x7_lnout-wd2.5e4-m93 (latest)` went 1-0-99 vs Stockfish; `Qeu8e5 (latest)` 0-1-99; `DCM - Qeu8e5` 1-0-99 (`h2h.txt`, Stockfish block).
- Frozen `DCM - Qeu8e5` is #11 (528.1), `DCM Qeu8e5 (latest)` #18 (499.0).
- `engines_models.tsv`: all 31 Models/ files named there still exist and their `__metadata__` model_id matches the tsv; parameter counts (sum of non-optimizer tensors) match the tsv for all of them. `Exp2` (`…oItC-manual.dcmsession/trainer.safetensors`, model_id `20260607-4-2Gd1-12`, 9,511,988) and `Exp1` (`…5K7Z-manual.dcmsession/trainer.safetensors`, model_id `20260601-11-bzw3-32`, 8,445,748) also match.

**Corrections**
- "~700–800 Elo above **every** DCM" -> Stockfish's h2h Diff ranges **+714.1 to +1253.7**; only the top 20 DCMs are within 700–875 of it (`h2h.txt`, Stockfish block; `ratings.txt` 1320.0 vs 605.9 … 66.3).
- "best DCM scored 1/100 vs it" -> true for the best-*rated* DCM only. DCMs won 19 games and drew 14 against Stockfish in total; the best *score* vs Stockfish was `nT8Y-resume2 (latest)` 3.5/100 (3-1-96), then `DCM v5` 3.0/100 (3-0-97) and `nT8Y-resume3 (latest)` 2.5/100 (1-3-96).
- "~100 games/pair" -> 700 pairs have exactly 100 games; three do not: `20260626-2-q2Bb-manual (latest)` vs `Sloppy` 200, `9blk16se-9x9stem (latest)` vs `v5_5block_7x7_lnout-wd5e4 (latest)` 200, `Early 2` vs `v5_5block_7x7_lnout-wd2.5e4-m93 (latest)` 138. This is why `ratings.txt` shows PLAYED 3,738 / 3,800 for those engines.
- "Strongest DCM lines: the **v5** family" -> ranks #2 and #3 are **the same weights**: `DCM v5` loaded `~/Downloads/20260628-v5_5block_7x7_lnout-wd2.5e4-m93-replay-latest.safetensors` and `…wd2.5e4-m93 (latest)` loaded the Models/ copy; both are model_id `20260629-1-Uf4p`, training_step 39419, identical content_sha256 `d9b06e11…` (the tsv lists the Models/ path for both). Their 605.9 vs 603.8 split and 58.0% head-to-head (48-20-32) are sampling noise between identical nets — a useful yardstick for how much noise a 100-game pairing carries at this temperature. Other v5-architecture lines are #12 (wd5e4, 525.6) and #19 (lnout, 472.9), so "v5 family" really means one net (Uf4p). #4 `Qeu8-resume2 (latest)` (588.4) is next, with #5 `nT8Y-resume3 (latest)` at 576.9 (CFS 94% between them).
- "Runs still training (Qeu8e5 among them) had their `-replay-latest` file overwritten during the tournament" -> by file mtime, `20260704-Qeu8e5-replay-latest.safetensors` (last written 2026-07-09 00:04, now step 88107) is the **only** listed file modified after 2026-07-08 08:16 (`Qeu8e4`); every other `-latest` file's last write predates the tournament's start date, so their current contents are the weights that played, unless they were rewritten and later restored (not indicated). Tournament start time itself is not recorded in the archive, so this is bounded by date, not hour.

**Unverified**
- `sf_debug.log` (cited for "settings verified as applied") is not in the archive.
- Host (Apple M5 Max, macOS 26A5378j), cutechess version / `-style fusion`, Ordo 1.2.6, concurrency ≈ 2, and the end-of-run hang: not recorded in the archived files.
- DCM build 2033 / git `b5adf14` dirty: commit `b5adf14` exists ("UCI: select model via setoption, deferred load, engine/model info"), but the build number and dirty state are not in the archive.
- The Temperature-0 schedule note (`startTau 2.0 → 0.2` at the time): not checked against the build-2033 source.
- `Early 2` (`…IWkd-manual.dcmsession/trainer.dcmmodel`, legacy format): model_id `20260525-1-sMe9-33` and 2,483,667 params not re-read.
- The pool-relative Elo scale here (Stockfish anchored at 1320) is unrelated to the puzzle pElo scale used in training dashboards; do not compare them.

## Reproduce

**Status: partial** — the rating analysis is fully reproducible (`reproduce.sh`); the games are not.

- **Commit / build:** DCM build 2033, git `b5adf14`, dirty (per Setup; the build number and dirty state are not in the archived files). cutechess v1.5.1 (`-style fusion`), Ordo 1.2.6, Stockfish with `nn-71d6d32cb962.nnue`.
- **Corpus:** none (engine tournament).
- **Starting point:** the 36 DCM model files in `engines_models.tsv`; 31 are `-replay-latest` files that were being overwritten during the run and later, so the exact weights played are not recoverable for those.
- **Parameters:** `engines.json` (engine configs) and `cutechess.ini` (40 moves / 5 s, no ply limit). The ini's `games_per_encounter=10` does not match the 100 games per pairing in the PGN; concurrency (~2) was not persisted.
- **Commands:** the tournament was set up in the cutechess GUI (`cutechess_tournament_window.txt`); no command line was recorded. Analysis: `./reproduce.sh` (Ordo `-a 1320 -A Stockfish -V -s 100 -J`).
- **Probe / analysis:** `reproduce.sh` → `ratings.txt`, `h2h.txt` from `games_full.pgn.gz` (first 100 warmup games stripped by `slice_tests.py`).
- **Expected exactness:** analysis bit-exact from the archived PGN. Games: not reproducible — DCM sampled with the old `Temperature = 0` = decaying arena schedule (real sampling noise), Stockfish ran under a 5 s clock, and current builds make `Temperature = 0` near-argmax, so a rerun plays differently.
- **Missing:** exact weights of the 31 live `-replay-latest` engines; cutechess command / tournament settings file; host load; the build's dirty diff.
