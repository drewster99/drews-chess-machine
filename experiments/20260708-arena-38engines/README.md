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
