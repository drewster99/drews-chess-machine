# Lichess bot: record statistics on the Overview

Status (2026-10-06): **PLAN ONLY.** Nothing here is implemented. Implementation starts after `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` lands (it edits the controller and the generation info this plan reads; §9 P-gate).
- Every `file:line` was checked against `main` at `c1a36294`. The working tree held uncommitted follow-lineage edits to `LichessBotModelSlots.swift`, `LichessBotModelSwitchStatusView.swift` and `CheckpointManager.swift`; nothing below cites those files' uncommitted lines.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/` or `scripts/`.
- Numbers about the bot's data were measured on this Mac on 2026-10-06 from `~/Library/Application Support/DrewsChessMachine/LichessBot/` itself (sizes base-2).
- This realizes part of `LICHESS_BOT_PLAN.md` §11 ("Statistics"), which has not shipped beyond today's Record card. When this ships, §11 gains a pointer here.

**Terms.** OD-n = owner decision n (§12). DCM = DrewsChessMachine. ML = maximum likelihood (§3.4). ECO = Encyclopaedia of Chess Openings. DST = daylight saving time. CLI = command line.

**The request (owner, 2026-10-06).** "Right now we have a large amount of UI space with just a few lines of output. We could add a few buckets... last hour, today, this week, this month, this year, all time. But also, think of what other kinds of stats or evals we can show."

**Proposed first pass, in order (not yet approved; OD-21):** items 1, 2, 4, 6, then 5 and 3 (numbering from the proposal the owner saw). Items 7–12 and the origin breakdown are later phases (§11).
1. Record table: last hour / today / this week / this month / this year / all time, plus performance rating, average opponent rating and net rating change on rated games.
2. Per time control: current rating, change today and this week, W–D–L, score, performance rating, rating sparkline.
4. Per model / checkpoint: games, score, performance rating, with checkpoint progression under follow-lineage.
6. The network judging itself from its own per-move W/D/L: calibration at moves 10/20/40 with a Brier score, blown wins and saves, the ply at which its eval turns decisive.
5. How games end.
3. Score by opponent-rating band against the Elo-expected score, with an estimated 50% point.

**What this plan does.**
- Defines every statistic exactly (§3), including the edge cases: no games, all wins, all losses, missing ratings, unrated games, aborted games, missing move decisions.
- Puts all computation in pure, tested functions under `LichessBot/Stats/` (§4.1). Views only format and lay out.
- Reduces each game record to a small set of per-game facts once, when its index row is built, and stores them in the existing games index (schema 2 → 3). Statistics never open a record file (§4.2).
- Computes the statistics off the main actor on a dedicated serial work queue, after every index change and when a period boundary passes (§4.3).
- Fills the Record card's empty left column with the expanded period table and a tabbed panel (Time controls, Models, Self-assessment, Endings, Opponent strength), keeping the recent-games list (§5).
- Fixes one latent record-builder bug that would misattribute moves to models (§1.4, P0), regression test first.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth: the game record is authoritative; the index row and its facts are a derived cache rebuilt from records; each statistic has one definition in one function;
- no silent defaults: a missing rating, rating change or decision is counted and shown as missing, never read as zero;
- no `try?`, no force unwraps;
- no long synchronous work on the main actor: computation runs on a `DispatchQueue` behind a continuation;
- SwiftUI: one `View` per new file, no `some View` helper properties, no `AnyView`, `.shown(_:)` instead of `if`-gated content, `onChange` 0/2-arg only, digits monospaced and aligned, light and dark mode, colors and fonts centralized;
- old files stay readable: records and index rows written before this plan decode;
- tests are not modified or deleted; new tests go in new files; the bug fix gets a failing regression test first.

---

## 1. How things work today (verified)

### 1.1 The Record card

- `LichessBotRecordCard` is a `GroupBox("Record")` with a height the operator drags, stored under `@AppStorage("lichessBot.overview.recordCardHeight")`, default 240, range 160…1600 (`LichessBot/UI/LichessBotRecordCard.swift:9`, `:13`, `:21-22`).
- Its content is an `HStack`: the record grid (left, `.fixedSize()`), a divider, and `LichessBotRecentGamesList` (right, up to 200 rows, scrolling) (`LichessBotRecordCard.swift:42-50`, `:139-141`).
- The grid has rows Today / This week / All time and columns Games, W–D–L, Score, vs bots, vs humans, as White, as Black (`:57-116`). `Games` counts aborted games too (`LichessBotResultTally.games`, `LichessBot/Stats/LichessBotRecordSummary.swift:12`).
- The tally is computed **in the view body**, on the main actor, once a minute, from `TimelineView(.everyMinute)` (`LichessBotRecordCard.swift:41-46`).
- The left column is about four rows tall; the right column fills the card. The empty space the owner sees is the left column under the grid.
- The Overview stacks Controls, Record, Outgoing challenges, then Account / Model / Requests, then Alarms in a `ScrollView` (`LichessBot/UI/LichessBotOverviewView.swift:11-27`).
- The window opens at 1180×820 with a minimum of 900×600 (`LichessBot/UI/LichessBotWindow.swift:11-12`); the sidebar takes 160–220 points (`LichessBot/UI/LichessBotRootView.swift:35`), so the detail pane can be as narrow as 680 points (the render tests use 680×570, `DrewsChessMachineTests/LichessBotSettingsViewRenderTests.swift:18`).

### 1.2 The games index

- `LichessBotGameSummary` is "the per-game fields Stats needs", derived from a record in one initializer (`LichessBot/Data/LichessBotIndex.swift:4-56`). It holds `gameID`, `createdAt`, `speed`, `perf`, `rated`, `ourColor`, opponent id / name / kind / title / rating, `ourRatingBefore`, `ourRatingDiff`, `status`, `winner`, `ourScore`, `plies`, `modelIDs`, `sourceKinds`, `builds`, `reconciliation`, `anomalyCount`.
- `index.json` is a derived cache with a signature of every record file (path, size, modification time) (`LichessBotIndex.swift:58-92`). `schemaVersion = 2` (`:66`).
- `load` returns the stored index when its count and signature match, else rebuilds from every record and rewrites it (`:168-179`). `readStored` decodes the file first and then rejects any other schema version (`:211-226`), so a schema bump rebuilds automatically and an older index with fewer row fields still decodes before being rejected.
- `upsert` adds one row incrementally after a game is filed, or rebuilds when the stored index does not exactly cover the other records (`:184-207`). It runs on the controller's general file queue (`LichessBot/Data/LichessBotRecordStore.swift:162-185`).
- After a game is filed the controller calls `refreshIndex()` (`LichessBot/App/LichessBotController.swift:3373-3384`), which loads the index on the file queue (`:3065-3073`); the `index` property's `didSet` rebuilds `recordsByOpponent` and `pastOpponents` on the main actor (`:208-218`).
- The window's root `.task` calls `refreshIndex()` when the window opens (`LichessBotRootView.swift:48-49`).

### 1.3 What a game record holds that the index does not

- Setup clock (`clockInitialMilliseconds`, `clockIncrementMilliseconds`), opening ECO and name, every model generation that chose a move, and per move: SAN, both clocks after the move, `ours`, `decision`, `generationID`, `postMilliseconds`, `offeredDraw` (`LichessBot/Data/LichessBotGameRecord.swift:24-32`, `:64-88`, `:146-150`).
- `LichessBotMoveDecision`: `chosenProbability` (after temperature), `topMoves` (the network's own policy at temperature 1, best first, top 5), `win` / `draw` / `loss` (value head, from our side, which is the side to move), `expectedScore = win + ½·draw`, `temperature`, `legalMoveCount`, `randomish`, and encode / inference / sample times (`LichessBot/Play/LichessBotMoveChooser.swift:12-39`).
- Outcome: Lichess `status`, `winner`, `pgnResult`, `ourScore` (nil for a game that never counted), `plies`, `localDrawCondition` (the draw rule DCM's own engine saw in the final position) (`LichessBotGameRecord.swift:47-62`). `ourScore` is nil exactly when the PGN result is `*`: aborted, noStart without a winner, created, started, unknownFinish (`:354-374`).
- `LichessBotGenerationInfo`: generation ID, source kind, model ID, training step, snapshot time, architecture summary, file path, file SHA-256, value-head recentering (`LichessBot/Play/LichessBotGameInterfaces.swift:21-46`). Follow-lineage adds `lineage: LichessBotGenerationLineage?` with `lineageRunID`, `segmentID`, `segmentIndex`, `cumTrainerStep`, `contentSHA256` (`documentation/plans-active/LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` §3.6).
- Draw statuses: Lichess's `draw` covers agreement and claims; `stalemate`, `insufficientMaterialClaim` and `outoftime` without a winner are separate statuses (`LichessBot/API/LichessBotAPIModels.swift:53-68`). `ChessDrawCondition` has `fiftyMoveRule`, `threefoldRepetition`, `insufficientMaterial` (`Chess/ChessGameEngine.swift:41-53`).

### 1.4 Finding: generation IDs restart at 1 on every going-online (latent misattribution)

- `LichessBotModelSlots.prepare` numbers the first generation 1 every time the bot goes online (committed `main`: `LichessBot/Play/LichessBotModelSlots.swift:238`).
- The record builder keeps a generation only if no earlier one has the same ID: `if !generations.contains(where: { $0.generationID == generation.generationID })` (`LichessBotGameRecord.swift:519-523`), and attaches only the ID to each move (`:497`, `:771-772`).
- A game whose journal spans two going-online sessions (a relaunch that resumes a game in progress) can therefore hold two different models under generation 1. The record keeps the first one's info and attributes every later move to it.
- Not seen in today's data: of 206 records, none spans two builds, and every record holds one generation. It becomes likely with follow-lineage (the model changes between sessions by design). Per-model statistics (item 4) would be wrong for such games, so P0 proves it with a regression test and fixes it (OD-17).

---

## 2. Measured (2026-10-06, this Mac)

- **Data:** 206 game records, all from 2026-10-01 onward; 11.7 MB of record JSON (average 58.1 KB, largest 143.7 KB); `index.json` 91.4 KB (about 454 B per row). At this rate (~200 games a day on 2026-10-06) 10,000 games is about 50 days of play.
- **Mix:** blitz 74, rapid 62, bullet 44, classical 26; rated 181, casual 25; bots 199, humans 7; reconciliation `matched` 206.
- **Statuses:** mate 162; draw with local threefold 26; outoftime with a winner 5; resign 4; stalemate 4; timeout with a winner 2; draw with local insufficient material 1; `insufficientMaterialClaim` 1; outoftime without a winner 1. No aborted record yet.
- **Decisions:** 7,712 of our moves, 7,711 with a decision and clocks (one without).
- **Finding: rated games without a rating change.** 4 of 181 rated records have no `us.ratingDiff` (`PqMafCHn`, `srKbqynk`, `Ucqt6xGS`, `lJd0q63p`, all 2026-10-06, all `matched`). The cause is not known (the export may have been fetched before Lichess set the diff); a summed rating change would silently miss them. §3.3 counts and shows them; the cause is a separate investigation (OD-8).
- **Rebuild cost (proxy).** Reading and parsing every record with `JSONSerialization` took 0.06–0.10 s for 206 records (two runs). `Codable` decoding into `LichessBotGameRecord` is slower; scaled linearly, 10,000 records would take several seconds to a few tens of seconds on the utility queue. P2 measures the real figure and logs it on every rebuild (§4.2).
- **Preview of item 6 on today's 206 records** (independent Python over the records, using §3.6's definitions; for the owner's orientation, not a result):

  | DCM's move | Games | Mean predicted E | Mean actual score | Brier (3-class) |
  |---|---:|---:|---:|---:|
  | 10 | 206 | 0.630 | 0.323 | 0.786 |
  | 20 | 192 | 0.567 | 0.310 | 0.636 |
  | 40 | 78 | 0.406 | 0.314 | 0.476 |

  - Reached a held win (p_win ≥ 0.80 on two consecutive moves of ours) in 114 games and did not win 68 of them; reached a held loss in 126 and did not lose 23.
  - Decisive ply (§3.6): won games mean 37.7 (n 43, 7 never); lost games mean 56.5 (n 103, 20 never).
  - DCM played its own top policy move on 97.1% of its moves (item 7, later).
  - The value head is far more optimistic than the results at move 10. Blown wins mostly measure that optimism, which is why the self-assessment pane puts the calibration table above the blown-win counts.

---

## 3. Statistic definitions

### 3.1 Which games count

- **Filed games only.** Statistics read the games index, so a game in progress counts once it is filed (§4.3).
- **Scored game:** `ourScore` is 1, ½ or 0. A game with `ourScore` nil (aborted, never started, unfinished) is **not counted** anywhere except the "not counted" footnote and the Endings pane's "Not counted" row.
- **W–D–L, score, splits, endings, self-assessment:** every scored game, rated and casual, against any opponent kind, unless the Rated / Casual filter (OD-3) narrows them.
- **Performance rating and average opponent rating:** scored games whose opponent has a rating (`opponentRating` non-nil). Lichess AI games (no rating) are left out of these two numbers only; the tooltip shows how many games the number covers.
- **Rating change:** rated games with a recorded `ourRatingDiff` (§3.3).
- **"Games" column:** scored games (today it includes aborted games; OD-2). The footnote under the table reads "N games not counted (aborted or never started)" when N > 0.

### 3.2 Periods (item 1)

All anchored on the game's **start** (`createdAt`, as today), in the **system calendar and time zone** (`Calendar.current`, captured when the statistics are computed), with no upper bound (a `createdAt` slightly in the future from clock skew counts in every current period).

| Period | Contains games with `createdAt` ≥ |
|---|---|
| Last hour | `now − 3,600 s` (rolling, not the clock hour) |
| Today | `calendar.startOfDay(for: now)` |
| This week | `calendar.dateInterval(of: .weekOfYear, for: now).start` (week start from the system locale, as today) |
| This month | `calendar.dateInterval(of: .month, for: now).start` |
| This year | `calendar.dateInterval(of: .year, for: now).start` |
| All time | (every game) |

- A calendar that returns no interval throws, as today (`LichessBotRecordSummary.ComputeError.noWeekInterval`, `LichessBotRecordSummary.swift:64-73`), extended to month and year.
- Last hour can contain games that Today does not (just after midnight). The row's tooltip says so.
- **When the numbers change without a new game:** the pure function `LichessBotStatsPeriods.nextChange(after: now, rows:, calendar:)` returns the earliest of: the oldest last-hour game's `createdAt + 3,600 s`, the next start of day, week, month and year. §4.3 recomputes when it passes.

### 3.3 Columns of the period table (item 1)

| Column | Definition | Display |
|---|---|---|
| Games | scored games | count |
| W–D–L | wins, draws, losses | padded counts (today's `LichessBotTallyText`) |
| Score | `(W + ½D) / (W + D + L)` | `%.1f%%`; "–" with no scored game |
| Perf | performance rating, §3.4, over scored games with an opponent rating | integer; "≥1890" / "≤1100" for a perfect or zero score; "–" with none |
| Opp avg | mean `opponentRating` over the same games | integer; "–" with none |
| Rating ± | Σ `ourRatingDiff` over rated games with a recorded diff | signed integer with a true minus sign; "–" with no rated game; a "*" marker when any rated game in the period lacks a diff, and the tooltip says "k of m rated games have a rating change" |
| vs bots, vs humans, as White, as Black | as today (W–D–L of that split) | as today |

- Rating change is **never inferred** from consecutive `ourRatingBefore` values: concurrent games make a later game's starting rating predate an earlier game's result (the bot plays several games at once).
- The all-speeds rows mix Lichess's per-speed rating pools in Perf and Opp avg (each game's ratings are that speed's). The tooltip says so; the Time controls pane gives the per-pool numbers (OD-5).

### 3.4 Performance rating (Elo maximum likelihood)

- Expected score of a game: `E(R, Rᵢ) = 1 / (1 + 10^((Rᵢ − R) / 400))`.
- Over n games with opponent ratings Rᵢ and total score S, the performance rating is the R that solves `Σᵢ E(R, Rᵢ) = S`, the maximum-likelihood (ML) estimate under the Elo logistic model. The sum is strictly increasing in R, so the solution is unique.
- Solved by bisection on `[min Rᵢ − 2000, max Rᵢ + 2000]` until the bracket is narrower than 0.01, in `Double`; displayed rounded to an integer.
- Cases:
  - n = 0 → `.none` ("–").
  - 0 < S < n → `.estimate(R)`.
  - S = n (every game won) → `.atLeast(R′)` where R′ solves the equation with S′ = n − ½ (the half-point convention). Displayed "≥R′".
  - S = 0 → `.atMost(R′)` with S′ = ½. Displayed "≤R′".
- Closed-form check used by the tests: against one opponent rating R₀ with score fraction p, `R = R₀ + 400·log₁₀(p / (1 − p))`.
- Chosen over the linear "algorithm of 400" (`avg + 400·(W − L)/n`), which can fall after a win against a much weaker opponent and does not match the expected scores of §3.7 (OD-4).

### 3.5 Time controls (item 2)

- Rows: `ultraBullet`, `bullet`, `blitz`, `rapid`, `classical`, `correspondence`, in that order, then any other `speed` string a record holds, verbatim. A row is shown when the account has a rating in it or a record has a game in it.
- Columns:
  - **Rating:** the account's current rating from `controller.account.perfs[speed]` (Lichess's own number, refreshed after every filed game), "?" when provisional. Read by the view directly; not part of the computed statistics.
  - **Today ±, Week ±:** §3.3's rating change restricted to the speed, for Today and This week.
  - **Games, W–D–L, Score, Perf:** for the speed, over the panel's selected period (OD-7).
  - **Trend:** a sparkline of `ourRatingBefore` over the speed's last 100 rated games with a rating, oldest to newest, then the current account rating as the final point. Every point is a rating Lichess reported; nothing is interpolated (OD-9).

### 3.6 The network judging itself (item 6)

Every number here comes from `decision.win / draw / loss` on DCM's own moves (the value head's view of the position with DCM to move). Opponent moves carry no evaluation.

- **DCM's Nth move:** full-move number N of DCM's color: ply `2(N − 1)` as White, `2(N − 1) + 1` as Black.
- **Calibration checkpoints at moves 10, 20, 40** (OD-13). For each checkpoint, over scored games where DCM made its Nth move:
  - n; games that reached the move but have no decision there are counted as "missing" and left out;
  - mean predicted expected score `E = win + ½·draw`, and mean actual score;
  - mean predicted W / D / L against actual W / D / L frequencies;
  - **Brier score (3-class):** mean over games of `(p_w − o_w)² + (p_d − o_d)² + (p_l − o_l)²`, where `o` is the one-hot actual result. Range 0 (perfect) to 2;
  - **Skill:** `1 − Brier / Brier_ref`, where `Brier_ref = 1 − Σₖ fₖ²` is the Brier score of always predicting the same games' own W / D / L frequencies `fₖ`. Positive = better than knowing only the base rates. "–" when `Brier_ref = 0` (every game the same result).
  - Games that ended before move N are not in that row (survivors only); the row's tooltip says so.
- **Reliability over every move** (all DCM moves with a decision): ten buckets of E, `[0, 0.1)`, …, `[0.9, 1.0]`; per bucket, the number of positions, mean predicted E and mean actual score of those positions' games. Positions in one game share its result, so long games weigh more; the tooltip says so.
- **Held win / held loss:** a game "reached a held win" when `win ≥ 0.80` on two consecutive DCM moves that both have decisions (a move without a decision breaks the run); likewise "held loss" with `loss ≥ 0.80` (OD-14).
  - **Blown win:** reached a held win, and the result is a draw or a loss.
  - **Save:** reached a held loss, and the result is a draw or a win.
  - Shown as "x blown of y held wins (z%)" and "x saved of y held losses (z%)", with the 20 most recent of each listed (opponent, result, ply where the hold began, a button that opens the game on Lichess through `LichessBotLinks.openGame`).
- **Decisive ply** (won and lost games only): the earliest ply of a DCM move from which the result's own probability (`win` for a win, `loss` for a loss) stays ≥ 0.80 on **every** later DCM move with a decision, through DCM's last decision. "Never" when the last decision is below 0.80. Reported for won and lost games separately: mean and median decisive ply, mean lead (`plies − decisive ply`), and the never count (OD-15).
- **Threshold constants** (0.80, two moves, checkpoints 10/20/40, ten buckets) live in one declaration, `LichessBotSelfAssessmentDefinition`. The per-game facts are reduced with them, so changing one bumps the index schema (§4.2).

### 3.7 Opponent strength (item 3)

- Over scored games with both `ourRatingBefore` and `opponentRating`:
  - `dᵢ = opponentRating − ourRatingBefore` (the gap at game start, same speed);
  - expected score `Eᵢ = 1 / (1 + 10^(dᵢ / 400))`.
- **Bands by gap** (OD-10), 100 points wide: `< −400`, `−400…−301`, `−300…−201`, `−200…−101`, `−100…−1`, `0…99`, `100…199`, `200…299`, `300…399`, `≥ 400` (band index `floor(d / 100)` clamped to −5…4). Gaps pool across speeds, because each is measured within its own pool.
- Per band: n, actual score, mean expected score, and the difference with a 95% interval `± 1.96 · sd(sᵢ − Eᵢ) / √n` (shown for n ≥ 2).
- **50% point:** the gap Δ solving `Σᵢ 1 / (1 + 10^((dᵢ − Δ) / 400)) = S` (§3.4's solver and edge cases). DCM scores 50% against opponents rated Δ above its own rating; Elo predicts Δ = 0 when the rating is accurate. Displayed "Scores 50% at +37 (n games)".

### 3.8 Models and checkpoints (item 4)

- **Model key** (OD-11):
  - the file's SHA-256 (`fileSHA256`, the whole file's bytes) when the generation was loaded from a file. The header's `content_sha256` (`lineage.contentSHA256`) is a different hash and is not mixed in, so one file never becomes two keys;
  - otherwise (source kind, model ID, training step).
  - Two generations of one key are one model (a relaunch that reloads the same file).
- **Attribution:** a game belongs to the generation that chose the most of DCM's moves in it; on a tie, the later one. Games played by more than one generation are also counted in a "mixed" column on the row they were attributed to.
- **Rows:** grouped by model ID (one CLI process = one model ID = one lineage segment), newest last game first. Each group expands into one row per checkpoint (key), ordered by training step. Group and checkpoint rows show: model ID, training step (and lineage cumulative step when recorded), games, W–D–L, score with 95% interval `± 1.96 · sd(sᵢ) / √n` (n ≥ 2), Perf, Opp avg, mixed, first and last game.
- **Progression chart** (follow-lineage): x = lineage cumulative trainer step, else training step; y = score with its interval, or Perf (picker). Consecutive checkpoints of one lineage run are merged into one point until it holds ≥ 30 scored games (OD-12); a run with fewer than two points shows the table only. At ~200 games a day and a checkpoint every ~33 minutes (follow-lineage plan §1.5), one checkpoint sees about 4–5 games, too few to plot alone.

### 3.9 How games end (item 5)

One classification, `LichessBotGameEnding`, from `(status, winner, ourScore, localDrawCondition)`:

| Ending | Rule |
|---|---|
| Checkmate | `mate` |
| Resignation | `resign` |
| Time forfeit | `outoftime` with a winner |
| Left the game | `timeout` with a winner (Lichess: a player left and the other claimed) |
| Stalemate | `stalemate` |
| Threefold repetition | `draw` and local `threefoldRepetition` |
| Fifty-move rule | `draw` and local `fiftyMoveRule` |
| Insufficient material | `draw` and local `insufficientMaterial` |
| Insufficient-material claim | `insufficientMaterialClaim` |
| Timeout vs insufficient material | `outoftime` without a winner |
| Agreed / other draw | `draw` with no local draw condition (agreement, or a claim DCM's engine did not see) (OD-18) |
| Other: `<status>` | any other scored status, labeled with Lichess's status name verbatim |
| Not counted | `ourScore` nil |

- The pane shows a matrix: endings as rows, DCM won / drew / lost as columns, counts and the share of each column. Rows with no games are left out of the data (not hidden in the view). Statuses this build does not know are never folded into a known row.

---

## 4. Design

### 4.1 Where the computation lives (`LichessBot/Stats/`, all pure)

| File | Contents |
|---|---|
| `LichessBotEloMath.swift` | expected score; `performanceRating(opponents:score:)` and `ratingOffset(gaps:score:)` returning `LichessBotRatingEstimate` (`none` / `estimate` / `atLeast` / `atMost`); one bisection solver |
| `LichessBotStatsPeriods.swift` | the six periods, their starts, `nextChange` |
| `LichessBotGameFacts.swift` | the per-game facts (§4.2) and their reduction from a `LichessBotGameRecord`; `LichessBotSelfAssessmentDefinition` |
| `LichessBotGameEnding.swift` | §3.9's classification |
| `LichessBotRecordStatistics.swift` | the snapshot (§4.4) and `compute(rows:now:calendar:filter:)` |
| `LichessBotStatsFormat.swift` | number formatting shared by every pane (signed rating with a true minus, "≥" / "≤" estimates, percentages, padded counts) |

- `LichessBotRecordSummary` stays. `compute(rows:now:calendar:)` and `Records.today / thisWeek / allTime` keep their signatures (`LichessBotRecordSummaryTests` uses them unchanged). There is one period enum, `LichessBotStatsPeriod` (in `LichessBotStatsPeriods.swift`); `LichessBotRecordSummary.Period` becomes a `typealias` of it, and `Records` gains `lastHour`, `thisMonth`, `thisYear`, all filled from `LichessBotStatsPeriods`, so the periods and their boundaries have one definition.
- Every aggregation is a single pass over the rows (dictionaries for grouping); nothing is quadratic in games. Each ML solve costs about 20 bisection steps × the games in its group.

### 4.2 Per-move data reaches the statistics through the index (schema 3)

- `LichessBotGameSummary` gains one field, `let facts: LichessBotGameFacts?`, filled by `init(record:)`. Optional so an index row without it decodes: nil means "no per-game facts" (only possible in a test fixture written by hand, since a schema-2 index is rejected and rebuilt). Statistics count such rows as "no move data".
- `LichessBotGameFacts` (Codable, Sendable, Equatable):
  - `localDrawCondition`, `ourMoveCount`, `ourMovesWithDecision`;
  - `checkpoints: [Checkpoint]` — `moveNumber`, `win`, `draw`, `loss` for each checkpoint DCM reached with a decision;
  - `expectedScoreBuckets: [Bucket]` — non-empty buckets only: `index`, `positions`, `sumExpected`;
  - `heldWinStartPly: Int?`, `heldLossStartPly: Int?` — the ply of the first DCM move of the first held run;
  - `decisivePly: Int?` — §3.6, nil for draws, unscored games and "never";
  - `generations: [GenerationFacts]` — model key parts (`sourceKind`, `modelID`, `trainingStep`, `fileSHA256`, lineage run ID, segment index, cumulative step) and `ourMoves` decided by it.
- Why the index and not a second cache: the index is already the derived per-game cache that Stats reads, rebuilt from records whenever it is stale (`LichessBotIndex.swift:4-5`, `:58-65`). A second cache would need its own staleness signature and could disagree with the first (OD-16).
- **Schema bump 2 → 3** (`LichessBotIndex.swift:66`). The first load after the upgrade rejects the schema-2 file and rebuilds from every record (`:168-179`, `:225`). One added log line, `[LICHESS-BOT] index rebuilt: <n> records in <s> s (<reason>)`, measures it on every rebuild.
- **Cost.** Each row grows by about 1 KB (three checkpoints, up to ten buckets, one generation with a 64-character hash), to about 1.5 KB: `index.json` ≈ 0.3 MB at 206 games and ≈ 15 MB at 10,000. The rebuild decodes every record once (§2: a fraction of a second today; seconds to tens of seconds at 10,000). Until the first rebuild finishes, the card shows "Loading the game records…" (today's behavior). After that, each filed game costs today's incremental upsert plus re-encoding the larger file on the utility queue.
- **Two builds alternating** (an older build reading a schema-3 index rejects it and writes schema 2, and back) rebuild on each switch. Accepted: the index is a cache (Risks).
- Records do not change (except P0's generation fix, §4.5). Old records decode as they do today, and their facts are computed from whatever they hold: a record without decisions yields facts with zero `ourMovesWithDecision`.

### 4.3 Computing off the main actor, and incremental updates

- The controller gains:
  - `private let statisticsQueue = LichessBotFileQueue(label: "drewschess.lichessbot.statistics", qos: .utility)`. It is a work executor for CPU work, separate from the file queue so an index rebuild never delays a recompute and the reverse. It is closed at bot shutdown beside the file queue (`LichessBotController.swift:1267`);
  - `private(set) var recordStatistics: LichessBotRecordStatisticsState` — `.loading`, `.ready(LichessBotRecordStatistics)`, `.failed(String)`;
  - `scheduleRecordStatistics(reason:)`: increments a request counter, captures `index.rows` (a value), `Date()`, `Calendar.current` and the filter, runs `LichessBotRecordStatistics.compute` on `statisticsQueue` through `run` (which resumes a continuation), and on the main actor applies the result only if its request is still the latest. An older result that finishes late is dropped.
- **Triggers:**
  - the `index` `didSet` (launch, every filed game, a rebuild): reason `index`;
  - the Rated / Casual filter changing: reason `filter`;
  - a clock loop started by the root view's `.task` (so it runs while the window is open and stops when it closes): every 60 s it recomputes when `now ≥ snapshot.validUntil` (from `nextChange`) or `TimeZone.current` differs from the snapshot's. Reason `clock`. Waking from sleep or a time-zone change is caught within a minute.
- **When a game finishes:** reconciler files it → `upsert` writes the index incrementally (unchanged) → `refreshIndex` → `didSet` → one full recompute over the in-memory rows. Recomputing from rows (rather than updating aggregates in place) keeps one code path and one definition; the P2 test bounds it at 10,000 synthetic rows (§7).
- **Log:** one `[LICHESS-BOT] record stats (<reason>): games=… W-D-L=… score=… perf=… rating=±… brier@20=… ms=…` line per `index` or `filter` recompute (not per clock tick), so a wrong number is visible in the session log and the validation script can compare against it.
- The view reads `controller.recordStatistics` only. The card stops calling `LichessBotRecordSummary.compute` in its body (`LichessBotRecordCard.swift:41-46`); the minute `TimelineView` stays for the recent-games list's relative times.

### 4.4 The snapshot (`LichessBotRecordStatistics`)

- `periodRows: [LichessBotStatsPeriod: PeriodRow]` (item 1).
- `byPeriod: [LichessBotStatsPeriod: PeriodBreakdowns]` with `timeControls`, `models`, `selfAssessment`, `endings`, `opponentStrength` — one per period, so the panel's period picker (OD-7) is a lookup, not a recompute. Cost: six passes per recompute.
- `ratingTrends: [String: [RatingPoint]]` (per speed; period-independent).
- `notCounted: Int`, `ratedWithoutRatingChange: Int`, `rowsWithoutFacts: Int`.
- `computedAt`, `validUntil`, `timeZoneIdentifier`, `filter`.

### 4.5 P0: generation attribution across sessions

- `Replay.apply(.moveDecided)` keeps a generation when no **equal** `LichessBotGenerationInfo` is already listed (it is `Equatable`; generations from two sessions differ at least in `snapshotAt`).
- `LichessBotGameRecord.Move` gains `generationIndex: Int?`, the index into `generations` of the generation that made the decision. Optional, so records written before it decode; nil there means "not recorded" and the facts reducer then resolves the move by `generationID`, which is exact in those records because they only ever held one generation per ID (they may hold the wrong one, which cannot be repaired without rebuilding the record from its kept journal; out of scope).
- The record schema version stays 1: the change is additive.

---

## 5. Layout

### 5.1 The card

- Wide (the left column's ideal width plus 380 points for recent games fits): `HStack` — left column, divider, recent games (`minWidth` 380, scrolling, as today).
- Narrow (the 680-point minimum): `VStack` — left column, then recent games. Chosen with `ViewThatFits(in: .horizontal)`; both variants are their own `View` structs, and the selected tab and period live in `@AppStorage`, so the switch on resize loses nothing.
- Left column, top to bottom:
  1. **Period table** (§3.3): six rows; header in `LichessBotStatsStyle.headerFont`, numbers in `LichessBotStatsStyle.numberFont` (monospaced), counts padded to one width per column as today. Inside a horizontal `ScrollView` so it never clips at 680 points.
  2. **Footnote line** (secondary): not-counted games, rated games without a rating change, rows without move data; each part only when non-zero (empty string otherwise; the line keeps its place).
  3. **Panel header:** a segmented `Picker` — Time controls · Models · Self-assessment · Endings · Opponent strength — then a menu `Picker` for the period (default All time) and the Rated / Casual / All filter (OD-3).
  4. **Panel content:** a `ZStack` holding every pane, each `.shown(selected == pane)`.
- The card's default height rises from 240 to 560 (OD-19); the `@AppStorage` key is unchanged, so an operator who already dragged it keeps their height.

### 5.2 Panes (one `View` per file under `LichessBot/UI/`)

| File | Pane / piece |
|---|---|
| `LichessBotRecordCardWideLayout.swift`, `LichessBotRecordCardNarrowLayout.swift` | the two arrangements |
| `LichessBotRecordPeriodTable.swift` | item 1 (replaces `LichessBotRecordGrid`, `LichessBotRecordCard.swift:57-116`) |
| `LichessBotRecordFootnote.swift` | footnote line |
| `LichessBotRecordPanel.swift`, `LichessBotRecordPanelHeader.swift` | tab and period pickers, pane `ZStack` |
| `LichessBotTimeControlTable.swift`, `LichessBotRatingSparkline.swift` | item 2 (sparkline: Swift Charts `LineMark`, 120×22, no axes, tooltip with first/last date and rating range) |
| `LichessBotModelRecordTable.swift`, `LichessBotModelRecordGroupRow.swift`, `LichessBotModelProgressChart.swift` | item 4 (grid with a disclosure `Button` per group; chart `PointMark` + `RuleMark` intervals, one color per lineage run) |
| `LichessBotSelfAssessmentPane.swift`, `LichessBotCalibrationTable.swift`, `LichessBotReliabilityChart.swift`, `LichessBotHeldGamesList.swift` | item 6 (calibration table on top, reliability chart beside the held-win / held-loss / decisive-ply figures, recent blown wins and saves below) |
| `LichessBotEndingsTable.swift` | item 5 |
| `LichessBotOpponentStrengthPane.swift`, `LichessBotRatingBandChart.swift` | item 3 (`BarMark` actual score per band, `PointMark` expected, 50% line in the header) |

- **Styles:** a new `LichessBotStatsStyle` enum holds the colors (win, draw, loss, expected, actual, neutral, interval) as semantic system colors that adapt to light and dark, and the fonts (header, number, row label). `LichessBotResultChip`'s inline `.green` / `.red` / `.gray` (`LichessBotRecordCard.swift:224-231`) move to it.
- Charts carry accessibility labels per mark; every number in a table is text.
- The Account card's rating grid (`LichessBotOverviewView.swift:171-221`) stays as it is (OD-20).

---

## 6. Edge cases

| Situation | Handling |
|---|---|
| No games at all | Every row shows 0 games and "–" for every ratio; panes show "No games in this period"; the snapshot is still `.ready`. |
| Period with games but none scored (all aborted) | Games 0, footnote counts them, ratios "–". |
| All wins / all losses | Score 100% / 0%; Perf "≥R′" / "≤R′" (§3.4); 50% point the same way. |
| One game | Perf from the half-point rule when won or lost (≥ / ≤ the opponent's rating), exact when drawn (= the opponent's rating); intervals "–" (n < 2). |
| Opponent without a rating (Lichess AI) | Counted in W–D–L and every split; left out of Perf, Opp avg, bands; tooltips state the covered count. |
| Our rating missing at game start | Left out of bands and the sparkline only. |
| Rated game without `ourRatingDiff` | Not summed; "*" marker and coverage tooltip; counted in the footnote. |
| Casual games | Never in Rating ±; in everything else unless filtered out. |
| Aborted / never started | Not counted (§3.1); Endings "Not counted" row. |
| Game without any decision (export-only moves, an old record) | Counted in results; no checkpoint, bucket or held-run contribution; "missing" counts where it reached a checkpoint. |
| One move without a decision mid-game | Breaks a held run; excluded from buckets; a checkpoint on it counts as missing. |
| Game shorter than move N | Not in checkpoint N (survivors only; tooltip). |
| Game played by two generations | Attributed by majority, counted as mixed (§3.8). |
| Same model reloaded (two generations, one file hash) | One key. |
| Generation ID reused across sessions in an old record | Resolved by ID (exact in old records); P0 fixes new records. |
| Unknown speed string | Its own Time controls row, labeled verbatim, after the known speeds. |
| Unknown status | "Other: <status>" ending row, never folded into a known row. |
| `createdAt` in the future | Counted in every current period (no upper bound). |
| Time zone or clock change | Clock loop recomputes within 60 s. |
| Midnight / week / month / year rollover | `nextChange` passes → recompute. |
| DST change day | `Calendar` gives the day start; tested with `America/Chicago` on 2026-11-01. |
| Index rebuild in progress | Previous snapshot stays on screen; "Loading…" only before the first index. |
| Stale compute finishing after a newer one | Dropped by the request counter. |
| 10,000+ games | Single-pass aggregation; sparkline capped at 100 points per speed; recent list capped at 200 rows as today. |
| Calendar returns no interval | `.failed(text)`, shown in red in the card (as today) and logged. |

---

## 7. Tests (new files only; existing tests unmodified)

**P0 — `DrewsChessMachineTests/LichessBotGenerationAttributionTests.swift`** (written first, must fail on today's code):
- `testTwoSessionsWithTheSameGenerationIDKeepBothModels` — a journal with session 1's generation 1 (model A) deciding plies 0–10 and, after a second header, session 2's generation 1 (model B) deciding plies 12–20: the record lists both generations and each move's `generationIndex` names its own.
- `testOldRecordWithoutGenerationIndexStillDecodes`.

**P1 — pure core:**
- `LichessBotEloMathTests`: expected score symmetry and the 400-point 10:1 odds; closed-form single-opponent cases (p = 0.25, 0.5, 0.75); mixed opponents against a hand-computed root; all wins / all losses / one win / one loss / one draw; n = 0; extreme ratings (bracket never fails); monotonicity; `ratingOffset` = 0 when every result equals its expectation.
- `LichessBotStatsPeriodsTests`: each boundary in `Europe/Paris`, `America/Chicago` (DST day), `Pacific/Auckland`; week start with `firstWeekday` 1 and 2; last hour across midnight; `nextChange` picks the earliest of the five candidates; future `createdAt`.
- `LichessBotGameEndingTests`: every row of §3.9, an unknown status, an unknown status with a winner, `outoftime` with and without a winner.
- `LichessBotGameFactsTests` (records built with `LichessBotRecordBuilder.build` from journals, as the data-layer tests do): checkpoints at the right plies for White and Black; a game ending before move 40; a missing decision at a checkpoint; bucket edges (E = 0.1 exactly, E = 1.0); held run of exactly two, a run broken by a missing decision, a single-move spike (no hold); decisive ply for a win, a loss, a "never", a draw (nil); per-generation move counts.
- `LichessBotRecordStatisticsTests`: every §3.3 column for each period from hand-built rows; Games excludes unscored; Rating ± skips missing diffs and reports coverage; Lichess AI excluded from Perf only; Rated / Casual filter; time-control rows including an unknown speed; sparkline cap and order; model key by hash vs (source, model, step); majority attribution and tie; mixed count; progression binning at 30 games; calibration means, Brier, skill (and "–" when every result is the same); reliability per bucket; blown wins and saves; bands at every edge (−401, −400, −1, 0, 399, 400); 50% point; no games; all aborted; rows without facts.
- `LichessBotStatsFormatTests`: true minus sign, "≥" / "≤", "–", padding.
- Existing suites that guard P1 (run, unmodified): `LichessBotRecordSummaryTests`, `LichessBotHistoryTests`, `LichessBotBotListTests`.

**P2 — index and pipeline:**
- `LichessBotIndexFactsTests`: a schema-2 `index.json` is rebuilt to schema 3 with facts equal to `LichessBotGameFacts(record:)` of each record; incremental `upsert` equals `rebuild` with facts; a row JSON without `facts` decodes (nil).
- `LichessBotRecordStatisticsPipelineTests` (controller with temporary defaults and data folder): a filed game updates the snapshot; a late older result is dropped; the clock recompute fires when `validUntil` passes (manual time); `.failed` on a calendar without intervals.
- `LichessBotRecordStatisticsScaleTests`: 10,000 synthetic rows compute in under 1 s (a guard against quadratic code, far above the expected cost).
- Existing suites that guard P2 (run, unmodified): `LichessBotDataLayerTests` (incremental index equals rebuild, stale index rebuilt), `LichessBotPostGameChatFilingTests`.

**P3–P8 — UI:** `LichessBotRecordCardRenderTests` in the style of `LichessBotSettingsViewRenderTests`: the card at 680 and 1,180 points wide, every pane, light and dark, with no games, with the 206-game fixture shape (synthetic), and with all-wins data; each must render a non-empty image.

---

## 8. Validation

1. **Independent numbers.** P1 adds `scripts/lichess_bot_record_stats.py`, a separate Python implementation of §3 over `LichessBot/Games/**/*.json` (the preview in §2 is its prototype). After P2, compare its all-time and today output with the `[LICHESS-BOT] record stats (index)` line of a fresh launch: games, W–D–L, score, Perf (±1), Rating ±, Brier at move 20 must agree exactly (Perf within rounding).
2. **Schema bump.** Launch once with the schema-2 index: one `[LICHESS-BOT] index rebuilt: 206 records in … s` line; `index.json` has `"schemaVersion":3` and a `facts` object per row; a second launch does not rebuild.
3. **A game finishing.** With the bot online, the `record stats (index)` line follows the game's `finalized` protocol entry within seconds, and the card's Last hour and Today rows include it.
4. **Rollover.** Last hour drops a game 60 minutes after its start (within the 60-s loop); a time-zone change in System Settings moves Today's boundary within a minute.
5. **P0.** The regression test fails before the fix and passes after, unmodified.
6. **Visual.** Render-test PNGs at 680 and 1,180 points, light and dark, checked for alignment of every numeric column and no clipping; then the live window at its minimum size.
7. **Tests.** Targeted suites per phase; the full suite once after P2 (persistence changed), per `CLAUDE.md`.

---

## 9. Phases (each: implement → build with drews-xcode-mcp → targeted tests → commit)

- **P-gate.** `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` merged on `main`, its tests green. Re-check this plan's `file:line` references against that commit.
- **P0.** Generation attribution fix (§4.5), regression test first (OD-17).
- **P1.** Pure core (§4.1) and its tests; `scripts/lichess_bot_record_stats.py`.
- **P2.** Index schema 3 with facts, rebuild timing log, controller pipeline (queue, state, triggers, clock loop, log line); tests; full suite.
Phases P3–P8 follow the recommended item order of OD-21; if the owner orders the items differently, P3–P8 are reordered to match (P0–P2 come first either way, since every pane needs them).

- **P3.** Item 1: card layouts, period table, footnote, panel header with the period and Rated / Casual pickers, `LichessBotStatsStyle`, `LichessBotStatsFormat` in the views; render tests.
- **P4.** Item 2: Time controls pane and sparkline.
- **P5.** Item 4: Models pane and progression chart.
- **P6.** Item 6: Self-assessment pane.
- **P7.** Item 5: Endings pane.
- **P8.** Item 3: Opponent strength pane. Final recheck of every definition against §3, validation §8.1–§8.6.

---

## 10. Risks

- **Rebuild time at scale.** The first launch after each schema bump decodes every record. Mitigated by the timing log and by the card keeping its previous snapshot; a later phase can rebuild in the background while showing the old index if the measured cost warrants it.
- **Overconfident value head reads as "blown wins".** §2's preview shows predicted E far above results. The pane leads with calibration so the blown-win count is read in its light.
- **Small samples.** Per-checkpoint and per-band numbers are noisy; every number shows its n, intervals appear from n ≥ 2, and the progression chart bins to 30 games.
- **Survivorship in checkpoints.** Move-40 rows describe long games only; stated in tooltips.
- **Alternating builds** rebuild the index on every switch (§4.2). Cache only; no data at risk.
- **Rating-pool mixing** in all-speed rows (OD-5).
- **Lichess rating refunds** (`LICHESS_BOT_PLAN.md` E49) make per-game diffs "at the time" values; the label says so.
- **The 4 rated games without a diff** may be a reconciliation-timing bug; until explained, Rating ± under-counts and says by how many games.

---

## 11. Later phases (not in the first pass)

- **L1 — item 7, move choice:** share of DCM moves equal to its own top policy move, mean chosen probability, randomish count, by period and model (facts: three counters; schema bump).
- **L2 — item 8, clock:** think time per move from consecutive own clocks plus increment, time left at the end, flags per time control (facts: clock summaries).
- **L3 — item 9, game length:** plies by result, short losses (< 30 plies) listed.
- **L4 — item 10, openings:** by ECO (Encyclopaedia of Chess Openings) family, as White and as Black, from `openingECO` / `openingName`.
- **L5 — item 11, opponents:** most played, highest-rated win, streaks (from existing index fields).
- **L6 — item 12, bot health:** anomalies, rejected moves, stream reconnects, reconciliation mismatches by period (facts: counts).
- **D1 — origin breakdown (depends on `LICHESS_BOT_CHALLENGE_LOG_PLAN.md`):** once records carry a game's origin (matchmaking / manual challenge / incoming accepted), the summary gains it (schema bump) and the panel gains an "Origin" split (W–D–L, score, Perf per origin). Records without it show as "not recorded", never folded into a known origin.
- Per-model self-assessment; a two-parameter logistic fit for the 50% point (free slope); the Lichess rating-history API for the sparkline (E49); `LICHESS_BOT_PLAN.md` §11's filters and CSV export.

---

## 12. Owner decisions (OD = owner decision)

- **OD-1 Layout:** tabs in the Record card's left column under the period table, recent games at the right, stacked when narrow. Alternative: a new "Stats" sidebar section. *Recommend the card* — it uses the space the owner pointed at; a sidebar section suits the later filters.
- **OD-2 Counting:** W–D–L, score and splits over scored rated and casual games; aborted and never-started games excluded and counted in a footnote; the "Games" column becomes scored games (today it includes aborted ones). *Recommend.*
- **OD-3 Rated / Casual / All filter** on the table and panes, default All, remembered. *Recommend* (cheap; lets the owner see rated-only Perf).
- **OD-4 Performance rating:** Elo maximum likelihood; a perfect or zero score shown as a "≥" / "≤" bound by the half-point convention. Alternative: linear "algorithm of 400". *Recommend ML.*
- **OD-5 All-speed rows mix rating pools** in Perf and Opp avg, with a tooltip; per-pool numbers in Time controls. Alternative: show Perf only per speed. *Recommend accept.*
- **OD-6 Periods:** anchored on game start, system calendar and time zone, last hour rolling 60 minutes, week start from the system locale (as today). *Recommend.*
- **OD-7 One period picker** for every pane, default All time, remembered. *Recommend.*
- **OD-8 Rating change:** sum of Lichess's per-game diffs on rated games that have one, with coverage shown, never inferred; investigate the 4 rated records without a diff separately (a GitHub issue). *Recommend.*
- **OD-9 Sparkline:** last 100 rated games of the speed from records, plus the current account rating. Lichess's rating-history API later. *Recommend.*
- **OD-10 Opponent bands by rating gap** (100 wide, open beyond ±400), pooled across speeds; 50% point as the Elo-scale offset Δ. Alternative: bands by absolute opponent rating, per speed. *Recommend gap.*
- **OD-11 Model key and attribution:** file hash when loaded from a file, else (source, model ID, step); a game goes to the generation that chose most of DCM's moves (tie → later), with a mixed count. *Recommend.*
- **OD-12 Progression chart bins:** merge consecutive checkpoints of one run until ≥ 30 scored games. *Recommend 30.*
- **OD-13 Calibration:** DCM's 10th / 20th / 40th move, 3-class Brier with skill against base rates, plus a ten-bucket reliability chart over every move. *Recommend.*
- **OD-14 Blown wins and saves:** `win` (`loss`) ≥ 0.80 held over two consecutive DCM moves; "not won" = draw or loss. Alternatives: a single move; expected score ≥ 0.80. *Recommend two moves* (one-move spikes before a recapture would otherwise count).
- **OD-15 Decisive ply:** the earliest DCM move from which the result's own probability stays ≥ 0.80 to DCM's last decision, won and lost games separately, with lead and never counts. *Recommend.*
- **OD-16 Per-move data path:** facts in the index summary (schema 3, one full rebuild per bump) rather than a separate cache. *Recommend the index.*
- **OD-17 Fold the generation-ID fix (P0) into this plan**, regression test first. Alternative: a separate fix before this plan. *Recommend fold* — per-model stats depend on it.
- **OD-18 Draws with no rule seen locally** labeled "Agreed / other draw". *Recommend.*
- **OD-19 Card default height** 240 → 560 for operators who never dragged it (key unchanged). *Recommend.*
- **OD-20 Account card ratings grid** stays as it is for now, though Time controls repeats its ratings. *Recommend keep*; revisit after P4.
- **OD-21 First-pass scope and order:** which of the twelve proposed items ship first, and in what order. *Recommend items 1, 2, 4, 6, then 5 and 3* (the record table, then per-time-control, per-model and self-assessment, which most directly show how strong the bot is and whether its own evaluations can be trusted; endings and opponent strength last, as they refine the same picture); items 7–12 and the origin breakdown as later phases (§11). Alternative: any other subset or order; the phase list (§9) is reordered to match.
