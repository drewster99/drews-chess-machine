# Lichess bot: record statistics on the Overview

Status (2026-10-06): **IMPLEMENTING.** P0 merged (`01fe2f62`); P1 onward in progress on a branch, started before `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` landed by the team lead's decision (see "Implementation notes" at the end).
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
- Reduces each game record to a small set of per-game facts once, when its index row is built, and stores them in the existing games index (next index schema, coordinated with the challenge-log plan, §4.2). Statistics never open a record file (§4.2).
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

- **Data:** 206 game records, the first created 2026-09-28 (by local date, America/Chicago: 64 on 09-28, 23 on 09-29, 4 on 09-30, 31 on 10-02, 84 on 10-06); 11.7 MB of record JSON (average 58.1 KB, largest 143.7 KB); `index.json` 91.4 KB (about 454 B per row). The busiest day, 2026-10-06, had 84 games between 11:34 and 17:55, about 13 games per hour online. At that rate 10,000 games is about 770 hours online.
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
| Rating ± | Σ `ourRatingDiff` over rated games with a recorded diff | signed integer with a true minus sign; "–" with no rated game, and "–*" with rated games of which none has a diff (never "0"); a "*" marker when any rated game in the period lacks a diff, and the tooltip says "k of m rated games have a rating change" |
| vs bots, vs humans, as White, as Black | as today (W–D–L of that split) | as today |

- Rating change is **never inferred** from consecutive `ourRatingBefore` values: concurrent games make a later game's starting rating predate an earlier game's result (the bot plays several games at once).
- The all-speeds rows mix Lichess's per-speed rating pools in Perf and Opp avg (each game's ratings are that speed's). The tooltip says so; the Time controls pane gives the per-pool numbers (OD-5).

### 3.4 Performance rating (Elo maximum likelihood)

- Expected score of a game: `E(R, Rᵢ) = 1 / (1 + 10^((Rᵢ − R) / 400))`.
- Over n games with opponent ratings Rᵢ and total score S, the performance rating is the R that solves `Σᵢ E(R, Rᵢ) = S`, the maximum-likelihood (ML) estimate under the Elo logistic model. The sum is strictly increasing in R, so the solution is unique.
- Solved by bisection in `Double` until the bracket is narrower than 0.01; displayed rounded to an integer. The bracket is derived from the data, never a fixed margin: with `p = S / n` (0 < p < 1 after the half-point rule below) and `L = 400·log₁₀(p / (1 − p))`, the root lies in `[min Rᵢ + L, max Rᵢ + L]`, because `n·E(R, max Rᵢ) ≤ S ≤ n·E(R, min Rᵢ)`. When every Rᵢ is equal the bracket has zero width and the root is `R₀ + L` exactly. (A fixed `[min Rᵢ − 2000, max Rᵢ + 2000]` misses the root once n exceeds about 50,000 games at a half-point score: `n / (1 + 10⁵) > ½`.)
- Cases:
  - n = 0 → `.none` ("–").
  - 0 < S < n → `.estimate(R)`.
  - S = n (every game won) → `.atLeast(R′)` where R′ solves the equation with S′ = n − ½ (the half-point convention). Displayed "≥R′".
  - S = 0 → `.atMost(R′)` with S′ = ½. Displayed "≤R′".
- Closed-form check used by the tests: against one opponent rating R₀ with score fraction p, `R = R₀ + 400·log₁₀(p / (1 − p))`.
- Chosen over the linear "algorithm of 400" (`avg + 400·(W − L)/n`), which can fall after a win against a much weaker opponent and does not match the expected scores of §3.7 (OD-4).

### 3.5 Time controls (item 2)

- Rows: `ultraBullet`, `bullet`, `blitz`, `rapid`, `classical`, `correspondence`, in that order, then any other `speed` string a record holds, verbatim. One of the six is shown when the account has a rating in it or a record has a game in it; any other row only when a record has a game in it. `account.perfs` also holds puzzle and variant pools ("Puzzle modes decode here too", `LichessBot/API/LichessBotAPIModels.swift:538-540`), which never become rows.
- Columns:
  - **Rating:** the account's current rating from `controller.account?.perfs?[speed]` (Lichess's own number, refreshed after every filed game, `LichessBotController.swift:3362-3369`), "?" appended when `prov` is true. "–" with the tooltip "account not loaded" while `account` is nil (no token, or before the first fetch). Read by the view directly; not part of the computed statistics.
  - **Today ±, Week ±:** §3.3's rating change restricted to the speed, for Today and This week.
  - **Games, W–D–L, Score, Perf:** for the speed, over the panel's selected period (OD-7).
  - **Trend:** a sparkline of `ourRatingBefore` over the speed's last 100 rated games with a rating, oldest to newest, then the current account rating as the final point (left off while `account` is nil). Every point is a rating Lichess reported; nothing is interpolated (OD-9).

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
  - The bucket index is `min(9, max(0, Int((Double(E) * 10).rounded(.down))))`, one function used by the reducer and the tests. `win`, `draw`, `loss` are `Float` (`LichessBotMoveChooser.swift:22-24`), so E can land a rounding step above 1.0 (clamped into bucket 9) or just below a tenth (it lands in the lower bucket; the tests pin that rather than assume decimal edges). A record never holds a non-finite probability: the record encoder uses `JSONEncoder`'s default `nonConformingFloatEncodingStrategy` (throw), so such a game is never filed.
- **Held win / held loss:** a game "reached a held win" when `win ≥ 0.80` on two consecutive DCM moves that both have decisions (a move without a decision breaks the run); likewise "held loss" with `loss ≥ 0.80` (OD-14).
  - **Blown win:** reached a held win, and the result is a draw or a loss.
  - **Save:** reached a held loss, and the result is a draw or a win.
  - Shown as "x blown of y held wins (z%)" and "x saved of y held losses (z%)", with the 20 most recent of each listed (opponent, result, ply where the hold began, a button that opens the game on Lichess through `LichessBotLinks.openGame`).
- **Decisive ply** (won and lost games only): the earliest ply of a DCM move **with a decision** whose result probability (`win` for a win, `loss` for a loss) is ≥ 0.80 and stays ≥ 0.80 on **every** later DCM move with a decision, through DCM's last decision. Moves without a decision are skipped here (unlike a held run, which they break). "Never" when the last decision is below 0.80. A won or lost game with no decision at all is "no data", not "never". Reported for won and lost games separately: mean and median decisive ply, mean lead (`plies − decisive ply`), the never count and the no-data count (OD-15).
- **Threshold constants** (0.80, two moves, checkpoints 10/20/40, ten buckets) live in one declaration, `LichessBotSelfAssessmentDefinition`. The per-game facts are reduced with them, so changing one bumps the index schema (§4.2).

### 3.7 Opponent strength (item 3)

- Over scored games with both `ourRatingBefore` and `opponentRating`:
  - `dᵢ = opponentRating − ourRatingBefore` (the gap at game start, same speed);
  - expected score `Eᵢ = 1 / (1 + 10^(dᵢ / 400))`.
- **Bands by gap** (OD-10), 100 points wide: `< −400`, `−400…−301`, `−300…−201`, `−200…−101`, `−100…−1`, `0…99`, `100…199`, `200…299`, `300…399`, `≥ 400` (band index `floor(d / 100)` clamped to −5…4). The floor is a floored division (`Int((Double(d) / 100).rounded(.down))`), not Swift's `/`, which truncates toward zero and would put −1…−99 in band 0. Gaps pool across speeds, because each is measured within its own pool.
- Per band: n, actual score, mean expected score, and the difference with a 95% interval `± 1.96 · sd(sᵢ − Eᵢ) / √n` (sample standard deviation, divisor n − 1; shown for n ≥ 2).
- **50% point:** the gap Δ solving `Σᵢ 1 / (1 + 10^((dᵢ − Δ) / 400)) = S` (§3.4's solver and edge cases). DCM scores 50% against opponents rated Δ above its own rating; Elo predicts Δ = 0 when the rating is accurate. Displayed "Scores 50% at +37 (n games)".

### 3.8 Models and checkpoints (item 4)

- **Model key** (OD-11):
  - the file's SHA-256 (`fileSHA256`, the whole file's bytes) when the generation was loaded from a file. The header's `content_sha256` (`lineage.contentSHA256`) is a different hash and is not mixed in, so one file never becomes two keys;
  - otherwise (source kind, model ID, training step).
  - Two generations of one key are one model (a relaunch that reloads the same file).
- **Attribution:** a game belongs to the generation that chose the most of DCM's moves in it; on a tie, the later one in the record's `generations` order (order of first use). Games played by more than one generation are also counted in a "mixed" column on the row they were attributed to. A scored game in which no DCM move has a generation (DCM never moved — the opponent resigned or left first — or every DCM move is known only from the export) goes to its own **"No model recorded"** row, never dropped and never given to a neighbor.
- **Rows:** grouped by model ID (one CLI process = one model ID = one lineage segment), newest last game first. Each group expands into one row per checkpoint (key), ordered by training step. Group and checkpoint rows show: model ID, training step (and lineage cumulative step when recorded), games, W–D–L, score with 95% interval `± 1.96 · sd(sᵢ) / √n` (sample standard deviation, divisor n − 1; n ≥ 2), Perf, Opp avg, mixed, first and last game.
- **Progression chart** (follow-lineage): x = lineage cumulative trainer step, else training step; y = score with its interval, or Perf (picker). Consecutive checkpoints of one lineage run are merged into one point until it holds ≥ 30 scored games (OD-12); a run with fewer than two points shows the table only. At about 13 games per hour online (§2) and a checkpoint every ~33 minutes (follow-lineage plan §1.5), one checkpoint sees about 7 games, too few to plot alone; a point merges about four checkpoints.

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
- Scored statuses the table does not name land in "Other: <status>" with Lichess's spelling, including `timeout` without a winner, `noStart` with a winner, `cheat` and `variantEnd` (`LichessBotGameStatusName`, `LichessBotAPIModels.swift:53-68`). Each has a test.

---

## 4. Design

### 4.1 Where the computation lives (`LichessBot/Stats/`, all pure)

| File | Contents |
|---|---|
| `LichessBotEloMath.swift` | expected score; `performanceRating(opponents:score:)` and `ratingOffset(gaps:score:)` returning `LichessBotRatingEstimate` (`none` / `estimate` / `atLeast` / `atMost`); one bisection solver |
| `LichessBotStatsPeriods.swift` | the six periods, their starts, `nextChange` |
| `LichessBotGameFacts.swift` | the per-game facts (§4.2) and their reduction from a `LichessBotGameRecord`; `LichessBotSelfAssessmentDefinition` |
| `LichessBotGameEnding.swift` | §3.9's classification |
| `LichessBotRecordStatistics.swift` | the snapshot (§4.4) and `compute(rows:now:calendar:)` (every filter, every period) |
| `LichessBotStatsFormat.swift` | number formatting shared by every pane (signed rating with a true minus, "≥" / "≤" estimates, percentages, padded counts) |

- `LichessBotRecordSummary` stays. `compute(rows:now:calendar:)` and `Records.today / thisWeek / allTime` keep their signatures (`LichessBotRecordSummaryTests` uses them unchanged). There is one period enum, `LichessBotStatsPeriod` (in `LichessBotStatsPeriods.swift`); `LichessBotRecordSummary.Period` becomes a `typealias` of it, and `Records` gains `lastHour`, `thisMonth`, `thisYear`, all filled from `LichessBotStatsPeriods`, so the periods and their boundaries have one definition.
  - `LichessBotStatsPeriod`'s raw values are stable identifiers (`lastHour`, `today`, …), because the selected period is persisted (§5.1); the display text is a separate `label`. Today's grid reads `period.rawValue` as its label (`LichessBotRecordCard.swift:86`), so P1 switches it to `label`; it shows all six rows until P3 replaces it.
  - `LichessBotRecordStatistics.compute` builds its W–D–L splits with `LichessBotPeriodRecord.add` (`LichessBotRecordSummary.swift:141-153`) through `LichessBotRecordSummary.compute`, so the split rules have one definition and `compute` keeps a production caller after the card stops calling it.
  - `LichessBotResultTally.games` keeps its meaning (scored + unscored; `LichessBotRecordSummaryTests.swift:34` and the History tab's sort key, `LichessBotRecordSummary.swift:39`, rely on it). OD-2's "Games" column reads `scored`.
- Every aggregation is a single pass over the rows (dictionaries for grouping); nothing is quadratic in games. Each ML solve costs about 20 bisection steps × the games in its group.
- The Rated / Casual / All filter (OD-3) is computed like the period: `compute` produces every filter × every period in one call, so changing the filter is a lookup, not a recompute, and the controller holds no filter state. Cost: three times the passes of a single filter (§7's scale test covers it).

### 4.2 Per-move data reaches the statistics through the index (next schema)

- `LichessBotGameSummary` gains one field, `let facts: LichessBotGameFacts?`, filled by `init(record:)`. Optional so an index row without it decodes: nil means "no per-game facts" (only possible in a test fixture written by hand, since a schema-2 index is rejected and rebuilt). Statistics count such rows as "no move data".
- `LichessBotGameFacts` (Codable, Sendable, Equatable):
  - `localDrawCondition`, `ourMoveCount`, `ourMovesWithDecision`;
  - `checkpoints: [Checkpoint]` — `moveNumber`, `win`, `draw`, `loss` for each checkpoint DCM reached with a decision;
  - `expectedScoreBuckets: [Bucket]` — non-empty buckets only: `index`, `positions`, `sumExpected`;
  - `heldWinStartPly: Int?`, `heldLossStartPly: Int?` — the ply of the first DCM move of the first held run; nil means "no held run" (a game with no decisions has none either, and `ourMovesWithDecision` tells the two apart);
  - `decisive: LichessBotDecisivePly` — §3.6, an enum, not an optional, so one nil never carries several meanings: `.notApplicable` (a draw or an unscored game), `.noDecisions` (won or lost, no DCM move with a decision), `.never`, `.atPly(Int)`;
  - nothing ply-based for a game that did not start from the standard position (`setup.variant` / `setup.initialFen`): the bot declines those (`LichessBot/Play/LichessBotChallengePolicy.swift:53-57`) and the record builder colors moves by ply parity (`LichessBotGameRecord.swift:751`), so such a record, if one ever exists, counts as "no move data";
  - `generations: [GenerationFacts]` — model key parts (`sourceKind`, `modelID`, `trainingStep`, `fileSHA256`, lineage run ID, segment index, cumulative step) and `ourMoves` decided by it.
- Why the index and not a second cache: the index is already the derived per-game cache that Stats reads, rebuilt from records whenever it is stale (`LichessBotIndex.swift:4-5`, `:58-65`). A second cache would need its own staleness signature and could disagree with the first (OD-16).
- **Schema bump** (`LichessBotIndex.swift:66`): to the **next** index schema version on `main` when P2 starts — 3 if `LICHESS_BOT_CHALLENGE_LOG_PLAN.md`'s index change (its `origin` row field, its §3.5, also written as "2 → 3") has not landed, 4 if it has. The two plans never ship different row shapes under one number: a build with only one plan's field would otherwise accept the other build's index and read the missing field as nil for every game. If both index changes are implemented together, one bump covers both. The first load after the upgrade rejects the older file and rebuilds from every record (`:168-179`, `:224`). One added log line, `[LICHESS-BOT] index rebuilt: <n> records in <s> s (<reason>)`, measures it on every rebuild.
- **Cost.** Each row grows by about 1 KB (three checkpoints, up to ten buckets, one generation with a 64-character hash), to about 1.5 KB: `index.json` ≈ 0.3 MB at 206 games and ≈ 15 MB at 10,000. The rebuild decodes every record once (§2: a fraction of a second today; seconds to tens of seconds at 10,000). Until the first rebuild finishes, the card shows "Loading the game records…" (today's behavior).
  - The rebuild runs on the controller's general file queue, which also carries the protocol log, player notes, the Keychain and the instance lock (`LichessBotController.swift:358-360`). At 10,000 records the one rebuild per bump delays those, going online included, by its duration. Today (206 records) it is well under a second.
  - Each filed game already decodes the whole index twice and encodes it once on that queue: `upsert` reads the stored index and writes the new one (`LichessBotIndex.swift:184-206`), then `refreshIndex` loads it again (`LichessBotController.swift:3065-3073`, `:3381-3382`). Facts triple the bytes those three passes handle. The P2 timing log covers `upsert` and `load` as well as rebuilds so the real cost is visible; if it matters, `finalize` can hand the `File` that `upsert` returns to the controller instead of the reload (today it is discarded, `LichessBotRecordStore.swift:178`).
- **Two builds alternating** (an older build reading the newer index rejects it and writes its own schema, and back) rebuild on each switch. Accepted: the index is a cache (Risks).
- Records do not change (except P0's generation fix, §4.5). Old records decode as they do today, and their facts are computed from whatever they hold: a record without decisions yields facts with zero `ourMovesWithDecision`.

### 4.3 Computing off the main actor, and incremental updates

- The controller gains:
  - `private let statisticsQueue = LichessBotFileQueue(label: "drewschess.lichessbot.statistics", qos: .utility)`. It is a work executor for CPU work, separate from the file queue so an index rebuild never delays a recompute and the reverse. It is closed at bot shutdown beside the file queue (`LichessBotController.swift:1267`);
  - `private(set) var recordStatistics: LichessBotRecordStatisticsState` — `.loading`, `.ready(LichessBotRecordStatistics)`, `.failed(String)`;
  - `scheduleRecordStatistics(reason:)`: does nothing once the controller has shut down (`isShutDown`, `LichessBotController.swift:1216-1218`); otherwise increments a request counter, captures `index.rows` (a value), `Date()` and `Calendar.current`, runs `LichessBotRecordStatistics.compute` on `statisticsQueue` through `run` (which resumes a continuation), and on the main actor applies the outcome only if its request is still the latest. An older outcome that finishes late — a result or an error — is dropped. The `Task` that awaits `run` catches every error inside its body: the latest request's error becomes `.failed(text)` and is logged (`[LICHESS-BOT] record stats failed (<reason>): …`); nothing is left to an unobserved `Task`. `compute` is synchronous CPU work and never runs on the main actor or the cooperative pool.
  - `remembered…` selections for the panel (tab, period, Rated / Casual / All filter), persisted in the controller's `defaults` exactly like `rememberedSettingsTab` (`LichessBotController.swift:346-351`, `:449-455`): a stored value that no longer exists is logged and ignored, and tests use a temporary defaults suite. Not `@AppStorage`, which would read the test host's real defaults in render tests. The card height stays in its existing `@AppStorage` key.
- **Triggers:**
  - the `index` `didSet` (launch, every filed game, a rebuild): reason `index`;
  - the clock: `recordStatisticsClockTick(now:)`, a controller method, recomputes (reason `clock`) when the state is `.ready` and `now ≥ snapshot.validUntil` (from `nextChange`) or `TimeZone.current.identifier` differs from the snapshot's. A `.failed` state is not retried by the clock (it would log every minute); the next index change or time-zone change retries it. Tests call the method with a manual `now`.
  - A loop in its own `.task` modifier on the root view calls the tick every 60 s while the window is open (not appended to the existing `.task`, whose awaits run in sequence, `LichessBotRootView.swift:49-56`). The loop follows `refreshGridClock`'s pattern (`LichessBotController.swift:605-613`): `Task.sleep` inside `do`, and a thrown error (cancellation when the window closes; `Task.sleep` throws nothing else) ends the loop with a comment saying so. `Task.sleep` runs on the continuous clock, which keeps counting while the Mac sleeps, so waking from sleep is caught at the next tick; a wall-clock change is caught by comparing against `validUntil`.
  - The controller also observes `.NSSystemTimeZoneDidChange` and `.NSSystemClockDidChange` and recomputes at once, unconditionally (reason `clock`): a clock set backwards leaves `now` before `validUntil`, so the tick's condition would never fire for it. Whether a running process's `TimeZone.current` follows a System Settings change without help is not established here; P2 checks it live (§8.4), and if it does not, the observer calls `NSTimeZone.resetSystemTimeZone()` before the tick.
- **When a game finishes:** reconciler files it → `upsert` writes the index incrementally (unchanged) → `refreshIndex` → `didSet` → one full recompute over the in-memory rows. Recomputing from rows (rather than updating aggregates in place) keeps one code path and one definition; the P2 test bounds it at 10,000 synthetic rows (§7).
- **Log:** one `[LICHESS-BOT] record stats (<reason>): games=… W-D-L=… score=… perf=… rating=±… brier@20=… ms=…` line (all time, filter All) per `index` recompute (not per clock tick), so a wrong number is visible in the session log and the validation script can compare against it.
- The view reads `controller.recordStatistics` only. The card stops calling `LichessBotRecordSummary.compute` in its body (`LichessBotRecordCard.swift:41-46`); the minute `TimelineView` stays for the recent-games list's relative times. The index rows are already newest first (`LichessBotIndex.swift:234-238`), so the recent list takes a prefix instead of re-sorting every row on the main actor each minute (`LichessBotRecordCard.swift:48`).

### 4.4 The snapshot (`LichessBotRecordStatistics`)

Held per filter (`LichessBotStatsFilter`: all, rated, casual), so the filter is a lookup like the period:
- `periodRows: [LichessBotStatsPeriod: PeriodRow]` (item 1).
- `byPeriod: [LichessBotStatsPeriod: PeriodBreakdowns]` with `timeControls`, `models`, `selfAssessment`, `endings`, `opponentStrength` — one per period, so the panel's period picker (OD-7) is a lookup, not a recompute. Cost: six passes per filter, eighteen per recompute.
- `notCounted: Int`, `ratedWithoutRatingChange: Int`, `rowsWithoutFacts: Int`.

Once per snapshot:
- `ratingTrends: [String: [RatingPoint]]` (per speed; from rated games, so independent of period and filter).
- `computedAt`, `validUntil`, `timeZoneIdentifier`.

### 4.5 P0: generation attribution across sessions

- `Replay.apply(.moveDecided)` keeps a generation when no **equal** `LichessBotGenerationInfo` is already listed (it is `Equatable`; generations from two sessions differ at least in `snapshotAt`).
- `LichessBotGameRecord.Move` gains `generationIndex: Int?`, the index into `generations` of the generation that made the decision. Optional, so records written before it decode; nil there means "not recorded" and the facts reducer then resolves the move by `generationID`, which is exact in those records because they only ever held one generation per ID (they may hold the wrong one, which cannot be repaired without rebuilding the record from its kept journal; out of scope).
- The record schema version stays 1: the change is additive.
- The PGN's per-move `gen=<id>` comment (`LichessBot/Data/LichessBotPGNWriter.swift:90-91`) keeps the per-session ID; across sessions it is ambiguous, and the `DCMModelIDs` tag still names every model. Unchanged here.
- `LichessBotGenerationInfo.generationID`'s doc comment says "Distinct per snapshot within one run of the app" (`LichessBotGameInterfaces.swift:22`); it is distinct per going-online. P0 corrects it, after follow-lineage's edit of that file lands (below).
- **Sequencing with follow-lineage.** `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` keeps the numbering (a first generation is 1 per `prepare`; "a failed build uses up no generation number", its Review) and states that records and the index need no change (its §3.6); its working-tree edits to `LichessBotModelSlots.swift` change build joining and file loading, not IDs. It adds `lineage` to `LichessBotGenerationInfo` with a nil default, which P0's equality covers and P0's test constructions compile with either way. P0 touches only `LichessBotGameRecord.swift` and a new test file, so it is **exempt from the P-gate** and should land as soon as possible: follow-lineage makes a model change between sessions routine, and every game resumed across a relaunch after that is misattributed until P0 is in. The doc-comment fix waits for follow-lineage's P3.

---

## 5. Layout

### 5.1 The card

- One arrangement view, `LichessBotRecordCardLayout`, places the same three children — left column, separator, recent games — with `AnyLayout(HStackLayout(…))` when the card is at least `LichessBotStatsStyle.wideCardWidth` points wide (a declared constant: the period table's measured width at the 680-point detail pane plus 380 for recent games, fixed in P3 from the render test) and `AnyLayout(VStackLayout(…))` below it. The width comes from `onGeometryChange(for: Bool.self)` on the card content (not `GeometryReader`).
  - Not `ViewThatFits`: it chooses by each child's ideal width, which here depends on the data (the longest opponent name in the recent list, the digit counts in the tables, the widest pane in the panel `ZStack`), so the arrangement would flip as games arrive, not only as the window resizes; and its two variants are different view types, so every flip would discard the panes' `@State` (the Models table's expanded groups) and scroll positions. `AnyLayout` keeps the children's identity across the switch.
  - Wide: recent games `minWidth` 380, scrolling, as today. Narrow: recent games below, at a fixed `LichessBotStatsStyle.narrowRecentGamesHeight` (200).
  - The separator is a `Divider`; the render tests at both widths confirm it is vertical in the horizontal arrangement and horizontal in the vertical one, and if not, it becomes a 1-point `Rectangle` in `LichessBotStatsStyle`'s separator color, sized per arrangement.
- The card's content has a fixed height (`LichessBotRecordCard.swift:21`), and a `.frame(height:)` does not clip. The period table, footnote and panel header take their ideal heights; the panel content takes the rest inside a vertical `ScrollView`, so a tall pane scrolls instead of overflowing the card.
- The card shows `controller.recordStatistics`'s three states as three children each `.shown(_:)` for its state (loading text, the content, the red failure text), not a `switch` in `body` (today's `LichessBotRecordGrid` switches, `LichessBotRecordCard.swift:62-66`).
- Left column, top to bottom:
  1. **Period table** (§3.3): six rows; header in `LichessBotStatsStyle.headerFont`, numbers in `LichessBotStatsStyle.numberFont` (monospaced), counts padded to one width per column as today. Inside a horizontal `ScrollView` so it never clips at 680 points.
  2. **Footnote line** (secondary): not-counted games, rated games without a rating change, rows without move data; each part only when non-zero (empty string otherwise; the line keeps its place).
  3. **Panel header:** a segmented `Picker` — Time controls · Models · Self-assessment · Endings · Opponent strength — then a menu `Picker` for the period (default All time) and the Rated / Casual / All filter (OD-3). The three selections bind to the controller's `remembered…` properties (§4.3). The Rated / Casual / All filter applies to the period table as well as the panes.
  4. **Panel content:** a `ZStack` holding every pane, each `.shown(selected == pane)`.
- The card's default height rises from 240 to 560 (OD-19); the `@AppStorage` key is unchanged, so an operator who already dragged it keeps their height. Sound as specified: `@AppStorage`'s declared default applies only while the key holds no value, and the resize handle writes the key only on a drag or an accessibility adjustment (`LichessBotHeightResizeHandle.swift:39`, `:52-54`), never on appear.

### 5.2 Panes (one `View` per file under `LichessBot/UI/`)

| File | Pane / piece |
|---|---|
| `LichessBotRecordCardLayout.swift` | the one arrangement view (`AnyLayout`, wide or stacked) |
| `LichessBotRecordPeriodTable.swift` | item 1 (replaces `LichessBotRecordGrid`, `LichessBotRecordCard.swift:57-116`) |
| `LichessBotRecordFootnote.swift` | footnote line |
| `LichessBotRecordPanel.swift`, `LichessBotRecordPanelHeader.swift` | tab, period and filter pickers, pane `ZStack` |
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
| Game played by two generations (a mid-game model switch, `midGameRefresh`) | Attributed by majority, counted as mixed (§3.8). Its self-assessment facts use every decision, whichever generation made it (per-model self-assessment is a later phase, §11). |
| Game resumed after a relaunch | One record from every kept journal, with every session's generations once P0 is in (§4.5); attributed and counted like any other game. Its periods use `createdAt`, not the resume time. |
| Scored game with no generation (DCM never moved, or only export-known moves) | Its own "No model recorded" row in Models (§3.8); counted everywhere else. |
| Won or lost game with no decision | Decisive ply "no data", not "never" (§3.6). |
| Same model reloaded (two generations, one file hash) | One key. |
| Generation ID reused across sessions in an old record | Resolved by ID (exact in old records); P0 fixes new records. |
| Unknown speed string | Its own Time controls row, labeled verbatim, after the known speeds. |
| Unknown status | "Other: <status>" ending row, never folded into a known row. |
| `createdAt` in the future | Counted in every current period (no upper bound). |
| Time zone or clock change | Recomputed at once from the system notifications, and by the 60-s tick at the latest (§4.3). |
| Account not loaded (no token, before the first fetch) | Time controls' Rating column "–" with a tooltip; no final sparkline point (§3.5). |
| Remembered tab / period / filter that no longer exists | Logged and ignored; the default is used (§4.3). |
| Midnight / week / month / year rollover | `nextChange` passes → recompute. |
| DST change day | `Calendar` gives the day start; tested with `America/Chicago` on 2026-11-01. |
| Index rebuild in progress | Previous snapshot stays on screen; "Loading…" only before the first index. |
| Stale compute finishing after a newer one | Dropped by the request counter. |
| 10,000+ games | Single-pass aggregation; sparkline capped at 100 points per speed; recent list capped at 200 rows as today. |
| Calendar returns no interval | `.failed(text)`, shown in red in the card (as today) and logged. |

---

## 7. Tests (new files only; existing tests unmodified)

**P0 — `DrewsChessMachineTests/LichessBotGenerationAttributionTests.swift`.** The first test is written first and must **compile and fail** on today's code, so it uses only existing API; a test that names `generationIndex` would not compile before the fix, which proves nothing.
- `testTwoSessionsWithTheSameGenerationIDKeepBothModels` (written first) — a journal with session 1's generation 1 (model A) deciding plies 0–10 and, after a second header (`resumed: true`), session 2's generation 1 (model B) deciding plies 12–20: `record.generations.map(\.modelID) == [A, B]` and `LichessBotGameSummary(record:).modelIDs == [A, B]`. Today it fails (one generation, model A).
- `testEachMoveNamesTheGenerationThatDecidedIt` (with the fix) — the same journal: every one of DCM's moves at plies 0–10 has `generationIndex` 0, at 12–20 index 1.
- `testOneGenerationJournaledManyTimesIsListedOnce` — equal infos collapse to one entry.
- `testOldRecordWithoutGenerationIndexStillDecodes` — a record JSON without the key decodes, and the facts reducer attributes its moves by `generationID`.

**P1 — pure core:**
- `LichessBotEloMathTests`: expected score symmetry and the 400-point 10:1 odds; closed-form single-opponent cases (p = 0.25, 0.5, 0.75); mixed opponents against a hand-computed root; all wins / all losses / one win / one loss / one draw; n = 0; extreme ratings and n = 100,000 at a half-point score (the data-derived bracket of §3.4 always holds the root); every opponent at one rating (zero-width bracket, exact `R₀ + L`); monotonicity; `ratingOffset` = 0 when every result equals its expectation.
- `LichessBotStatsPeriodsTests`: each boundary in `Europe/Paris`, `America/Chicago` (DST day), `Pacific/Auckland`; week start with `firstWeekday` 1 and 2; last hour across midnight; `nextChange` picks the earliest of the five candidates; future `createdAt`.
- `LichessBotGameEndingTests`: every row of §3.9, an unknown status, an unknown status with a winner, `outoftime` with and without a winner, `timeout` without a winner, `noStart` with a winner, `cheat`.
- `LichessBotGameFactsTests` (records built with `LichessBotRecordBuilder.build` from journals, as the data-layer tests do): checkpoints at the right plies for White and Black; a game ending before move 40; a missing decision at a checkpoint; bucket edges through the one index function (`Float` E of 0.1, 0.3, 0.9, 1.0 and one rounding step above 1.0); held run of exactly two, a run broken by a missing decision, a single-move spike (no hold); decisive ply for a win, a loss, a "never", a won game with no decisions (`.noDecisions`), a draw (`.notApplicable`), a decision-less move inside the decisive stretch (skipped); per-generation move counts with `generationIndex` and, for an old record, by `generationID`; a game in which DCM never moved (no generation).
- `LichessBotRecordStatisticsTests`: every §3.3 column for each period from hand-built rows; Games excludes unscored; Rating ± skips missing diffs and reports coverage, and is "–*" when no rated game has a diff; Lichess AI excluded from Perf only; every filter's numbers equal `compute` over the rows pre-filtered to it; time-control rows including an unknown speed; sparkline cap and order; model key by hash vs (source, model, step); majority attribution and tie (later in `generations` order); mixed count; the "No model recorded" row; progression binning at 30 games; calibration means, Brier, skill (and "–" when every result is the same); reliability per bucket; blown wins and saves; bands at every edge (−401, −400, −1, 0, 399, 400); 50% point; no games; all aborted; rows without facts.
- `LichessBotStatsFormatTests`: true minus sign, "≥" / "≤", "–", padding.
- Existing suites that guard P1 (run, unmodified): `LichessBotRecordSummaryTests`, `LichessBotHistoryTests`, `LichessBotBotListTests`. They decode hand-written row JSON without `facts` (`LichessBotRecordSummaryTests.swift:9-14`), which still decodes because synthesized `Codable` treats an absent `let facts: LichessBotGameFacts?` as nil; `compute`, `Records` and `LichessBotResultTally.games` keep their meaning (§4.1).

**P2 — index and pipeline:**
- `LichessBotIndexFactsTests`: an `index.json` written with `schemaVersion` = `LichessBotIndex.schemaVersion - 1` is rebuilt to `LichessBotIndex.schemaVersion` with facts equal to `LichessBotGameFacts(record:)` of each record; incremental `upsert` equals `rebuild` with facts; a row JSON without `facts` decodes (nil).
- `LichessBotRecordStatisticsPipelineTests` (controller with temporary defaults and data folder): a filed game updates the snapshot; a late older result is dropped, and so is a late older error; `recordStatisticsClockTick(now:)` recomputes once `validUntil` passes and not before; a `.failed` state is not retried by the tick; `.failed` on a calendar without intervals; a shut-down controller schedules nothing; the remembered tab, period and filter persist across controllers on one temporary suite, and a stored value that no longer exists is ignored (as `LichessBotSettingsViewRenderTests.swift:121-136` does for the Settings tab).
- `LichessBotRecordStatisticsScaleTests`: a guard against quadratic code that does not depend on the build configuration's speed (tests run unoptimized): best-of-three times for 2,500 and 10,000 synthetic rows, asserting the ratio is below 8 (linear ≈ 4, quadratic ≈ 16), plus a loose absolute bound of 10 s at 10,000 rows.
- Existing suites that guard P2 (run, unmodified): `LichessBotDataLayerTests` (incremental index equals rebuild, stale index rebuilt), `LichessBotPostGameChatFilingTests`.

**P3–P8 — UI:** `LichessBotRecordCardRenderTests` in the style of `LichessBotSettingsViewRenderTests` (controller on a temporary defaults suite): the card at 680 and 1,180 points wide (both arrangements), every pane, light and dark, in each state (loading, ready, failed), with no games, with the 206-game fixture shape (synthetic), and with all-wins data; each must render a non-empty image.

---

## 8. Validation

1. **Independent numbers.** P1 adds `scripts/lichess_bot_record_stats.py`, a separate Python implementation of §3 over `LichessBot/Games/**/*.json` (the preview in §2 is its prototype). After P2, compare its all-time and today output with the `[LICHESS-BOT] record stats (index)` line of a fresh launch: games, W–D–L, score, Perf (±1), Rating ± must agree exactly; Perf within ±1; Brier at move 20 to the logged three decimals (the records hold `Float` probabilities, which Python reads back as the shortest decimal, so the last digits of a `Double` sum can differ).
2. **Schema bump.** Launch once with the previous schema's index: one `[LICHESS-BOT] index rebuilt: <n> records in … s` line; `index.json` has the new `schemaVersion` and a `facts` object per row; a second launch does not rebuild.
3. **A game finishing.** With the bot online, the `record stats (index)` line follows the game's `finalized` protocol entry within seconds, and the card's Last hour and Today rows include it.
4. **Rollover.** Last hour drops a game 60 minutes after its start (within the 60-s loop); a time-zone change in System Settings moves Today's boundary at once (notification) or within a minute (tick). This step also establishes whether `TimeZone.current` follows the change unaided (§4.3).
5. **P0.** The regression test fails before the fix and passes after, unmodified.
6. **Visual.** Render-test PNGs at 680 and 1,180 points, light and dark, checked for alignment of every numeric column and no clipping; then the live window at its minimum size.
7. **Tests.** Targeted suites per phase; the full suite once after P2 (persistence changed), per `CLAUDE.md`.

---

## 9. Phases (each: implement → build with drews-xcode-mcp → targeted tests → commit)

- **P0.** Generation attribution fix (§4.5), regression test first (OD-17). Not behind the P-gate: it touches only `LichessBotGameRecord.swift` and a new test file, none of which follow-lineage edits; land it now (§4.5, sequencing). The generation-ID doc-comment fix in `LichessBotGameInterfaces.swift` waits for the P-gate.
- **P-gate.** `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` merged on `main`, its tests green. Re-check this plan's `file:line` references against that commit.
- **P1.** Pure core (§4.1) and its tests; `scripts/lichess_bot_record_stats.py`.
- **P2.** Next index schema with facts (number per §4.2's coordination with the challenge-log plan), rebuild timing log, controller pipeline (queue, state, triggers, clock loop, log line); tests; full suite.
Phases P3–P8 follow the recommended item order of OD-21; if the owner orders the items differently, P3–P8 are reordered to match (P0–P2 come first either way, since every pane needs them).

- **P3.** Item 1: card layouts, period table, footnote, panel header with the tab, period and Rated / Casual / All pickers on the controller's remembered selections, `LichessBotStatsStyle`, `LichessBotStatsFormat` in the views; render tests.
- **P4.** Item 2: Time controls pane and sparkline.
- **P5.** Item 4: Models pane and progression chart.
- **P6.** Item 6: Self-assessment pane.
- **P7.** Item 5: Endings pane.
- **P8.** Item 3: Opponent strength pane. Final recheck of every definition against §3, validation §8.1–§8.6.

---

## 10. Risks

- **Rebuild time at scale.** The first launch after each schema bump decodes every record, on the general file queue that also serves the Keychain, the instance lock and the protocol log (§4.2), so at 10,000 records going online waits for it. Mitigated by the timing log and by the card keeping its previous snapshot; a later phase can rebuild in the background while showing the old index if the measured cost warrants it.
- **Two plans bump the index schema.** This plan and the challenge-log plan each change the row; §4.2's rule (next number on `main`, never a shared number for two shapes) keeps them apart, and the challenge-log plan's §7 states the same rule. Each row field stays optional, so neither change depends on the other. Both plans' index tests name the version only through `LichessBotIndex.schemaVersion` (never a literal 2, 3 or 4), so whichever lands second does not break the first's tests, which would be a test edit needing owner approval. Each bump costs one rebuild.
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
- **D1 — origin breakdown (depends on `LICHESS_BOT_CHALLENGE_LOG_PLAN.md`):** that plan already adds `origin` to the record and to `LichessBotGameSummary` with its own index bump (its §3.5), so D1 needs **no further schema change**. D1 reads each game's origin from that plan's one resolver, the controller's `originsByGameID` (`LichessBotGameOriginDisplay`, its §3.6), not from the raw row field: the resolver also covers games played before origins were recorded (the challenge-log join), and it is "the one place that decides what origin a game shows". `scheduleRecordStatistics` captures that map beside the rows, recomputes when it changes, and the panel gains an "Origin" split by `LichessBotGameOriginCategory` (W–D–L, score, Perf per category). Its `unknown` category is its own row, never folded into a known origin.
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
- **OD-16 Per-move data path:** facts in the index summary (next index schema, one full rebuild per bump) rather than a separate cache. *Recommend the index.*
- **OD-17 Fold the generation-ID fix (P0) into this plan**, regression test first. Alternative: a separate fix before this plan. *Recommend fold* — per-model stats depend on it.
- **OD-18 Draws with no rule seen locally** labeled "Agreed / other draw". *Recommend.*
- **OD-19 Card default height** 240 → 560 for operators who never dragged it (key unchanged). *Recommend.*
- **OD-20 Account card ratings grid** stays as it is for now, though Time controls repeats its ratings. *Recommend keep*; revisit after P4.
- **OD-21 First-pass scope and order:** which of the twelve proposed items ship first, and in what order. *Recommend items 1, 2, 4, 6, then 5 and 3* (the record table, then per-time-control, per-model and self-assessment, which most directly show how strong the bot is and whether its own evaluations can be trusted; endings and opponent strength last, as they refine the same picture); items 7–12 and the origin breakdown as later phases (§11). Alternative: any other subset or order; the phase list (§9) is reordered to match.

**Decided (team lead, 2026-10-06, owner delegation).**
- **OD-17: fold.** P0 is part of this plan, regression test first (§4.5, §7 P0).
- **R-5 (Review): unchanged.** The PGN's per-move `gen=` comments stay per-session generation IDs; the `DCMModelIDs` tag still names every model.

**Decided (team lead, 2026-10-06, owner delegation).** The owner delegated the remaining decisions ("solve the problem yourself").
- **OD-1 … OD-20: as recommended.** Tabs in the Record card; scored games only with a not-counted footnote; Rated / Casual / All filter; ML performance rating with ≥ / ≤ bounds; pool-mixing tooltip; periods anchored on game start in the system calendar; one remembered period picker; summed per-game rating changes with coverage; record-based sparkline; gap bands and the 50% offset; file-hash model key with majority attribution; 30-game progression bins; calibration at 10 / 20 / 40 with Brier skill and reliability buckets; two-move holds; decisive ply; facts in the index; P0 folded in; "Agreed / other draw"; default card height 560; Account card unchanged.
- **OD-21: as recommended.** First pass in the order 1, 2, 4, 6, then 5 and 3 (P3 … P8 as listed in §9). Items 7–12 and the origin breakdown follow as later phases if time allows; D1 only once the challenge-log plan's resolver is on `main`.
- **R-1: Wilson.** Every score interval is the Wilson score interval at 95% (z = 1.96) on `p̂ = (W + ½D) / n`, `n = W + D + L` (`LichessBotEloMath.scoreInterval`): center `(p̂ + z²/2n) / (1 + z²/n)`, half-width `z / (1 + z²/n) · √(p̂(1 − p̂)/n + z²/4n²)`, clamped to 0…1. Draws are half points in `p̂`; the binomial variance `p(1 − p)` bounds a game score's variance (a score in 0…1 has `E[s²] ≤ E[s]`), so with draws the interval is slightly conservative, never too narrow. It replaces `± 1.96 · sd / √n` everywhere: the Models table's score interval, the progression chart's intervals, and the opponent bands (each band shows the Wilson interval of its actual score next to its mean expected score, so "actual − expected" reads against that interval).
- **R-2: `AnyLayout` with a width threshold** (§5.1, `LichessBotStatsStyle.wideCardWidth`).
- **R-3: remembered selections in the controller's defaults** (§4.3), as `rememberedSettingsTab`.
- **R-4: the index schema takes `main`'s current version + 1 when this plan's index change lands.** `main` held 2 when P1 landed here, so this plan uses 3; tests name the version only as `LichessBotIndex.schemaVersion` / `LichessBotIndex.schemaVersion - 1`. If the challenge-log plan's index change reaches `main` first, the merge moves this plan to the next number.

---

## Review (2026-10-06, against `main` at `dad00b85`)

Every claim below was checked in the code; the data figures were re-measured from the records with an independent script.

**Verified, no change needed.**
- Generation IDs restart at 1 per `prepare` (`LichessBotModelSlots.swift:238`, committed), and the record builder keeps a generation only if no earlier one has its ID, attaching only the ID to each move (`LichessBotGameRecord.swift:519-523`, `:497`, `:771-772`). The finding (§1.4) is real. `snapshotAt` is `Date()` at generation creation (`LichessBotModelSlots.swift:81-90`), so two sessions' generations are never equal; equality-based dedup (§4.5) is sound.
- The follow-lineage plan and its working-tree edits do not change generation numbering or the record builder (§4.5, sequencing).
- Old rows and the hand-written test JSON still decode with `facts` optional; `compute(rows:now:calendar:)` and `Records.today / thisWeek / allTime` keep their signatures. No existing test needs editing: no test touches the Record card, `LichessBotRecordSummary.Period`'s raw values, the index schema number or `LichessBotGameRecord.Move`'s initializer; `LichessBotDataLayerTests.swift:184` (`generations == [generation]`) still holds with equality dedup.
- The §2 self-assessment preview (calibration at 10/20/40, Brier, held wins and losses) reproduces exactly from the records.
- The §3.4 ML equation, the Brier score and its base-rate reference (`1 − Σ fₖ²`), the 50%-point equation and its sign, the band edges, and the DST date (2026-11-01 is the Sunday Chicago leaves DST) are correct.
- OD-19's height change is sound (§5.1).

**Must-fixes applied.**
1. **§2 data:** the records start 2026-09-28, not 2026-10-01; the bot played 84 games on 2026-10-06 (about 13 per hour online), not ~200 a day. The 10,000-game horizon and §3.8's games-per-checkpoint (about 7, not 4–5) follow.
2. **§3.4 bisection bracket:** the fixed `[min − 2000, max + 2000]` misses the root past about 50,000 games at a half-point score. Replaced by the exact data-derived bracket `[min Rᵢ + L, max Rᵢ + L]`, zero width when every rating is equal; tests added.
3. **§3.6 / §4.2 decisive ply:** `decisivePly: Int?` gave nil three meanings (draw, never, no data). Now an enum with `.noDecisions` separate from `.never`; decision-less moves are skipped explicitly; bucket indexing is one clamped function over `Float` E.
4. **§3.7 bands:** the band index needs floored division; Swift's `/` truncates toward zero and puts −1…−99 in band 0. Sample standard deviation (n − 1) stated for both intervals.
5. **§3.8 models:** scored games with no generation (DCM never moved) had no home; they get a "No model recorded" row. Tie order defined.
6. **§3.5 time controls:** `account.perfs` holds puzzle and variant pools too; only the six speeds come from the account. `account` nil is handled.
7. **Filter (OD-3):** the filter was captured by the controller but its single home was unspecified. It is now precomputed like the period (no controller filter state, no `filter` trigger), and the tab, period and filter selections are remembered in the controller's defaults like `rememberedSettingsTab` (testable on a temporary suite; unknown stored values logged), not `@AppStorage`. Period raw values are stable identifiers with a separate label.
8. **§4.1:** `LichessBotResultTally.games` keeps its meaning (a test and the History sort rely on it); the statistics reuse `LichessBotPeriodRecord.add` so `LichessBotRecordSummary.compute` keeps a production caller.
9. **§4.3 concurrency and the clock:** the awaiting `Task` handles every error; late errors are dropped like late results; nothing is scheduled after shutdown. The clock loop gets its own `.task` (the existing one runs its awaits in sequence), follows `refreshGridClock`'s cancellation pattern, and calls a controller tick method that tests drive with a manual `now`; time-zone and clock-change notifications recompute at once (a clock set backwards included), and P2 checks whether `TimeZone.current` updates unaided. `.failed` is not retried every minute.
10. **§4.2 schema and cost:** the challenge-log plan also bumps the index 2 → 3. Each plan now takes the next number on `main` (never two row shapes under one number). The cost now counts the two decodes and one encode per filed game and the first rebuild's hold on the general file queue (Keychain, instance lock, protocol log).
11. **§4.5 / §7 P0:** a regression test that names `generationIndex` cannot compile before the fix, so it cannot "fail first". The first test now uses only existing API (`generations`, the summary's `modelIDs`); `generationIndex` gets its own test. P0 is taken out of the P-gate: it touches no follow-lineage file and every game resumed across a relaunch under follow-lineage is misattributed until it lands.
12. **§5.1 layout:** `ViewThatFits` picks by data-dependent ideal widths (opponent names, digit counts, the widest pane), so the arrangement would flip as games arrive and lose the panes' `@State` on each flip. Replaced by one view with `AnyLayout` and a declared width threshold from `onGeometryChange`. Added: the panel content scrolls vertically inside the fixed card height (a `.frame(height:)` does not clip), and the three states render through `.shown`, not a `switch`.
13. **§7 / §8:** scale test made independent of the unoptimized test build (a 2,500 / 10,000-row time ratio plus a loose bound); Brier validation to three decimals (`Float` storage); the schema steps no longer assume the number 3; new edge-case tests (endings `timeout` without a winner, `noStart` with a winner, `cheat`; "–*" rating change; no-model row; `.noDecisions`).
14. **§9 / §11 D1:** P2 and D1 updated for the schema coordination; D1 reads origins from the challenge-log plan's resolver (`originsByGameID`) and needs no schema bump of its own.

**Open points for the owner.**
- **R-1 Score interval.** `± 1.96 · sd / √n` collapses to ±0 when every result in a group is the same (common at 2–5 games). Keep it and show "±0" with a tooltip, or switch to an interval that stays honest at small n (for example, a bootstrap or a Wilson-type interval on the score). The plan keeps the normal approximation.
- **R-2 Layout mechanism.** The review replaced `ViewThatFits` with `AnyLayout` + a width threshold (must-fix 12). If the owner prefers a different breakpoint rule (for example, always stacked below a fixed window width), only `LichessBotStatsStyle.wideCardWidth` changes.
- **R-3 Remembered selections** move from `@AppStorage` (as proposed) to the controller's defaults (must-fix 7), matching the Settings tab. The card height stays in `@AppStorage`.
- **R-4 Index schema order** (rule agreed with the challenge-log review: +1 from the value in the tree, symbolic versions in tests). Which of this plan's P2 and the challenge-log plan's index phase lands first decides who takes 3. One combined bump is possible only if both land together.
- **R-5 PGN `gen=` comments** stay per-session IDs, ambiguous across a relaunch (§4.5). Changing them would be a PGN format change; not proposed.

**Second pass (2026-10-06).** Rechecked for claims of owner approval: the first-pass order was already reworded as a recommendation with OD-21 in `f622a327`; every OD is a recommendation, and no other line claims approval. Index tests now name the schema only as `LichessBotIndex.schemaVersion` and `LichessBotIndex.schemaVersion - 1`, matching the challenge-log plan's §7.

## Implementation notes

Decisions taken while implementing, where the plan left a choice open or the code needed something it did not spell out.

**Sequencing.** The team lead started P1 before the follow-lineage plan landed (time). This plan's changes stay in `LichessBot/Stats/`, `LichessBot/Data/LichessBotIndex.swift`, the Record card's UI files, the controller's statistics pipeline and new files.

**P1**
- **The `facts` row field and the schema bump land in P1, not P2,** because `LichessBotRecordStatistics.compute` reads `row.facts`; a commit with the field but without the bump would accept an older index and read every row's facts as nil. P2 adds the index tests, the rebuild timing log and the pipeline.
- **`LichessBotGameFacts` nests its ply-based part** as `moves: LichessBotGameMoveFacts?` instead of one flat struct with zeros: nil for a game that did not start from the standard position, so "no move data" is never a set of meaningless zeros. `localDrawCondition` stays at the top level (the Endings pane needs it for every game).
- **A `draw` row without facts is "Draw (rule not recorded)",** not "Agreed / other draw": without facts the local draw condition is unknown, and calling it an agreement would be a silent default. Such rows exist only in hand-written fixtures.
- **A decision whose generation reference names no listed generation is counted** (`decisionsWithoutGeneration`, shown in the Models pane when non-zero), never given to a neighbor. An old record whose `generationID` matches several generations (impossible from the builder) counts the same way.
- **Rating ± counts scored rated games only.** An aborted rated game changes no rating, so it is not a game "missing" its change; counting it would inflate the "*" coverage gap.
- **The progression chart's trailing bin** (fewer than 30 games after the last full bin) stays its own point, labeled with its game count; a checkpoint with neither a lineage cumulative step nor a training step (a champion snapshot) cannot be placed on the x axis and appears in the table only.
- **The ML solver sums over distinct ratings weighted by their counts** (sorted, so the result is bit-identical for the same games in any order): the same root, at the cost of the distinct ratings per bisection step, which keeps the 10,000-row compute well inside the scale bound.
- **`nextChange` is inclusive at the rolling hour's edge:** a game counts through `createdAt + 3,600 s` inclusive (the plan's `≥`), so a game exactly at the edge yields `now` and the next tick drops it.
- **Ending and band labels, the time-control row order and the score interval are pure functions in `LichessBot/Stats/`** (`LichessBotGameEnding.label`, `LichessBotRatingBand.label`, `LichessBotTimeControlOrder.speeds`, `LichessBotEloMath.scoreInterval`) so the views only lay out text.
- **`scripts/lichess_bot_record_stats.py`** reads the records read-only and prints, per period, the numbers of the app's `record stats` log line (ASCII signs), so §8.1 compares line by line. First run (2026-10-06 21:24 CDT, 222 records): all time 222 games, 53–37–132, 32.2%, Perf 1412, rating −6330* (5 rated games without a change), Brier@20 0.653 (n 206).

**P2**
- **The pipeline is its own `@MainActor @Observable` object, `LichessBotRecordStatisticsPipeline`** (`LichessBot/App/`), held by the controller as `recordStatistics`, rather than stored properties on the controller: the controller is being edited heavily by the follow-lineage and challenge-log work, and the pipeline's state, request counter, queue, observers and remembered selections form one unit with one test seam. The controller's part is three lines: create it, feed it each new index from the `index` `didSet`, stop it at shutdown. Views read `controller.recordStatistics.state`; the plan's `scheduleRecordStatistics(reason:)` / `recordStatisticsClockTick(now:)` are its `schedule(reason:now:)` / `clockTick(now:)`.
- **Shutdown is split:** `stopScheduling()` runs synchronously at the start of `performShutdown`, so nothing is awaited before the runtime stops (the shutdown's existing ordering is unchanged) and an index change during the rest of the shutdown schedules nothing; the statistics queue is closed at the end, after the file queue.
- **The compute function and the calendar are injected** (defaults: `LichessBotRecordStatistics.compute`, `Calendar.current`), so the tests drive a late result, a late error and a time-zone change deterministically.
- **The system-time observers start with the window's clock task** (`LichessBotRecordStatisticsClock`, a `ViewModifier` in its own file), not in the controller's initializer, so a controller made for an unrelated test registers nothing. The observer calls `NSTimeZone.resetSystemTimeZone()` before recomputing, so the result does not depend on whether a running process's `TimeZone.current` follows a System Settings change by itself (§4.3's open question; not checked live, since changing the Mac's time zone would disturb the training runs and the operator's session).
- **The index logs every load path:** `index rebuilt: <n> records in <s> s (<reason>)` with the reason (no index, stale, unreadable, `schema <old>, now <new>`, filing a game the stored index does not cover, rebuild requested), `index loaded: <n> rows in <s> s`, and `index updated: <game> added, <n> rows in <s> s` for the incremental path.
- **Validation §8.1, run before any launch** (a launch would rewrite the operator's `index.json` and open windows): a scratch test, not committed, read the 222 real records read-only and computed the snapshot; its log line and every period's games, W–D–L, Perf, Rating ± and Brier@20 equal `scripts/lichess_bot_record_stats.py`'s output exactly (2026-10-06 21:34 CDT): all time `games=222 W-D-L=53-37-132 score=32.2% perf=1412 rating=-6330* brier@20=0.653`, today `100, 20-20-60, 1422, -93*, 0.759`. Calibration on the same data: Brier 0.796 / 0.653 / 0.507 at moves 10 / 20 / 40 (skill −0.42 / −0.16 / +0.15); 125 held wins, 76 not won; 132 held losses, 24 not lost; 50% point −175.
- **Scale:** 10,000 synthetic rows compute in 1.98 s unoptimized (2,500 in 0.47 s, ratio 4.24).

**P3**
- **The Record card file was split one view per file** (`LichessBotRecordCard`, `…CardContent`, `…CardLayout`, `LichessBotRecentGamesSection`, `…RecentGamesList`, `…RecentGameRow`, `LichessBotResultChip`, `LichessBotTallyText`), as the plan asks of new views; the moved views keep their behavior, with colors and fonts from `LichessBotStatsStyle`.
- **The separator is a 1-point `Rectangle`** in `LichessBotStatsStyle.separator`, sized per arrangement, not a `Divider`: `ImageRenderer` cannot show which way a `Divider` turns inside `AnyLayout` (§5.1's fallback, taken up front).
- **`wideCardWidth` is 1,300 points:** the render test measures the period grid's ideal width (842 points with three-digit counts) and asserts `grid + 380 + separator ≤ wideCardWidth`; the margin covers four-digit counts. At the default window (1,180 points) the card is therefore stacked; side by side needs a window of about 1,550 points.
- **The statistics' three states share one `ZStack`;** the ready snapshot is shown through a `ForEach` over zero or one item with a constant ID (`LichessBotReadyStatistics`), so a recompute never changes the content's identity and the panes keep their state.
- **The panel header has two rows** (the segmented pane picker, then the period and filter menus): five tabs plus two menus do not fit one row at 680 points.
- **One "No games in this period" line for every pane** lives in the panel (keyed on the period row's game count, unscored games included), so panes do not each repeat it.
- **`ImageRenderer` draws neither a `ScrollView`'s content nor AppKit controls** (the segmented picker renders as its "unsupported" placeholder), so the card renders are smoke tests only; the period grid and every pane are rendered on their own for inspection (`testTablesAndPanesForInspection`). The plan's live-window check (§8.6) needs the app running, which this work does not do (no window may be opened); it is left for the owner.
- **Two per-file type-check timing warnings remain** (`LichessBotRecordCardContent.body`, `LichessBotRecentGamesSection.body`, ~650 ms each): each is the first member access on `LichessBotController` in its compile job, the cost of resolving the 3,700-line controller, not of the expression; rewriting the expressions did not change it. (P4 threads the account into the panel as a value instead of reading `controller.account` there, which removed one of them; `LichessBotRecordCardContent` remains, as the one view that reads the controller.)

**P4**
- **The Time controls pane shows even in an empty period** (it lists the speeds the account is rated in, with their current ratings); the panel's "No games in this period" line still appears above it.
- **A speed with no game in the selected period shows 0 games and "–"** for score and Perf; a speed with no rated game today or this week shows "–" for that rating change (§3.3's rule).
- **The sparkline's x axis is the game sequence,** not time: points are evenly spaced, so a burst of games and a quiet week look alike. Its tooltip gives the dates and range; a series of fewer than two points is drawn invisible (the cell keeps its size, so the column stays aligned).
- **The account rating, its "?" and the sparkline's points and tooltip are pure functions** (`LichessBotStatsFormat.accountRating`, `LichessBotRatingSparklineSeries`) with their own tests.

**P5**
- **The Models table is a list of one row shape** (`LichessBotModelTableRow`: a group, a checkpoint or "No model recorded"), built by a pure function from the expanded set, so one row view draws every row with no `switch` in a body, and opening a group inserts rows rather than showing hidden ones. The expanded groups are the table's own `@State`; the panel keeps every pane mounted, so they survive tab switches and recomputes.
- **Group order:** newest last game first (§3.8); checkpoints by training step, a step-less checkpoint (a champion snapshot) after those with one, by first game. The "No model recorded" row is last and only when it has games; it has no Perf, Opp avg, Mixed or dates ("–"), but its score carries a Wilson interval.
- **A checkpoint's label** reads "step 5,000 · cum 120,000 · Model file · ab12cd34" (training step, the lineage cumulative step when recorded, the source's display name, a file hash's first eight characters).
- **The progression chart plots Perf only where it is an estimate:** a bin with a perfect or zero score has a bound, not a value, and is left off the Perf plot (it stays on the Score plot and in the table). Each point carries a vertical rule for its score interval; one color per run.
- **Lineage fields are not filled yet:** `LichessBotGenerationFacts.lineageRunID` / `segmentIndex` / `cumTrainerStep` stay nil until `LichessBotGenerationInfo.lineage` exists (follow-lineage plan P3, not on `main` when P5 landed). Until then runs are grouped by model ID and the chart's x axis is the training step. Wiring them is one assignment each in `LichessBotGameMoveFacts.init(record:)` plus an index schema bump (rows reduced before the wiring would hold nil).
