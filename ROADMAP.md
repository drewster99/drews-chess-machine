# Roadmap

Long-term goals, deferred work, and notes on decisions.

**Validated against implementation on 2026-05-05.** This roadmap was reconciled
against the actual Swift code under `DrewsChessMachine/DrewsChessMachine/` rather
than against comments or old roadmap assumptions. Items that were implemented
have been moved out of Future Improvements. Items that were rejected or made
obsolete are kept with context in Decisions Not Pursued / Historical Notes so the
original rationale is not lost.

## Future improvements (validated open)

- **Train-vs-UCI — continuous live training by playing external UCI engines (drafted 2026-07-09; IMPLEMENTED 2026-07-10 — all four components landed on main; end-to-end smoke vs real Stockfish still pending).**

  **Implementation status (2026-07-10).** All four components below are implemented, unit-tested, and committed: `UCIArbiter` (App/UCI/), `TrainVsUciDriver` + `ActiveGame.flushTrainerSide` (Training/), `TrainVsUciRunner` + `handleTrainVsUciIfPresent` (CLI/ + App/), and `TrainVsUciStatsFormatter` with the `[VS-UCI-STATS]` emit. Deviations from the original plan, with reasons:
  - **Separate `TrainVsUciDriver`, not a modification of `BatchedSelfPlayDriver`.** Follows the existing per-topology-driver precedent (`TickTournamentDriver` for arena); the self-play driver stays untouched.
  - **Whole-game recording — BOTH sides' plies are recorded/flushed** (standard two-sided `ActiveGame.flush`), matching `CorpusReplayFeeder` exactly: the opponent's (Stockfish's) moves become advantage-weighted imitation targets, evaluated by the trainer's forward pass at training time (no play-time forward pass on opponent moves, same as corpus rows). Decided 2026-07-10: against stronger opponents the imitation gradient dominates the own-move RL gradient, and that is the *desired* behavior — it is the same mechanism corpus replay uses with human games. This also keeps the buffer's adjacent-ply history reconstruction valid, so **history input encodings (full10ply*) work normally** — no pre-flight rejection. *(History note: the first implementation that morning recorded only the trainer's plies via a one-sided `flushTrainerSide`, on an on-policy-REINFORCE-purity argument; reversed the same day because corpus replay already trains the same loss on non-sampled human moves both-sided, and one-sided storage forced a history-encoding ban plus gameLength special-casing. `flushTrainerSide` and its tests were removed the same day once whole-game recording landed.)*
  - **"Live weights" is near-live:** the driver plays a separate eval `ChessMPSNetwork` whose weights are re-synced from the live trainer every `--eval-sync-steps` trainer steps (default 10), rather than evaluating the trainer's own graph. Avoids concurrent eval/train access to one MPSGraph and the `ChessNetwork`/`ChessMPSNetwork` type split (`trainer.network` is the inner `ChessNetwork`).
  - **Cap-hit games are dropped without a flush** (outcome unknown — no fake-draw label), tallied as `capDropped` in stats.
  - **Duplicate opponent kinds are auto-disambiguated** (`stockfish`, `stockfish-2`) when two specs name the same executable (e.g. two Elo pools), so per-kind stats and instance labels never collide.
  - **Validation still open:** the plan's end-to-end smoke (real Stockfish `nodes 1` + Sloppy; buffer fills, steps advance, checkpoints at the 1000-step cadence, `[VS-UCI-STATS]` cadence, spread sanity) has not yet run — deferred while a long `--replay-corpus` training run owns the GPU.

  Original plan follows, unchanged:

  **Why.** Corpus replay trains the trainer network on a *static* PGN corpus. This mode replaces the static corpus with a **live stream of games the current trainer generates by playing real external opponents** (Stockfish, Sloppy, …) over UCI. It is on-policy: the games cover the positions the trainer actually reaches and misplays, which is more useful for improving *this* network than a fixed corpus of games it may never visit. It is the live-play analog of corpus replay — same live-trainer training loop, just a live rather than pre-recorded data source.

  **Framing (locked in the design discussion; do not drift from these).**
  - **Mirrors corpus replay, NOT Play-and-Train.** Play-and-Train self-plays on the *champion* network and runs an Arena to decide promotion (`BatchedSelfPlayDriver.swift` docstring: both slot sides are the same champion; `SessionController+Training.swift` header). Train-vs-UCI does **none** of that: no champion, no Arena, no promotion. The network that plays the UCI games **is the live trainer**, updated continuously — exactly as corpus replay trains the trainer live.
  - **Outcome-only training signal (terminal WDL).** No per-position engine-eval labels; NNUE-style eval distillation is explicitly out of scope here.
  - **The spread problem is controlled by opponent mix, never by throttling.** Outcome-only policy-gradient needs a *spread* of wins and losses: full-strength opponents → ~100% losses = as information-free as all-draws. The operator tunes the win/loss spread purely by **how many instances of each opponent are provided and their per-opponent strength options** (`go` limit, `UCI_LimitStrength`/`UCI_Elo`). The replay buffer is **never throttled or quota'd** — the mix in the buffer is exactly the mix of instances configured.
  - **No search on our side** (consistent with the whole engine): the trainer picks moves by a single forward pass; each external opponent searches per its own `go` limit.
  - **Live weights, no frozen batches.** Same mildly-off-policy property self-play and corpus replay already accept.

  **Components.**
  1. **`UCIArbiter` (new — the only genuinely new subsystem).** The arbiter/driver side of UCI (correct term: we are the *arbiter/GUI*, the external engine is the *engine*; DCM's existing `App/UCI/UCIEngine.swift` is the opposite role — DCM-*as*-engine for `--uci`). Spawns each external engine via `Process`, speaks UCI over its stdin/stdout: `uci`→`uciok`, `setoption`, `isready`→`readyok`, `ucinewgame`, `position [startpos|fen …] moves …`, `go <limit>`, and awaits `bestmove` as an async result. **One process = one concurrent game slot** (UCI is serial per process — a single engine cannot multiplex two games). Launch each engine **once** and loop games with `ucinewgame`/`isready` — never respawn per game.
  2. **Adapt `BatchedSelfPlayDriver` (reuse, do not reinvent).** The existing tick-based driver already advances an arbitrary number of concurrent games that **end and reset raggedly and independently** (`handleGameEnds`/`resetForNewGame`, per-slot skip of terminated/`drawWatchTerminationRequested` slots) — this is the out-of-sync handling we need; it is already solved for self-play. Changes:
     - A slot's two sides become configurable: side A = the **live trainer** network, side B = an external `UCIArbiter` engine (vs today: both sides the same champion).
     - Each tick's batched `evaluateBatched(...)` covers the subset of slots where **it is the trainer's turn and the slot is ready**. Slots awaiting an external `bestmove` are skipped that tick — a natural extension of the existing per-slot skip logic. Dispatch each opponent `go` **off the tick barrier** (async) so a slow engine never stalls the batch; the slot rejoins the next trainer-eval batch when its `bestmove` arrives.
     - **Variable batch size, exactly as now — no padding.** Each tick batches whatever trainer-to-move slots are ready and no more, identical to the current driver's behavior (which already runs a tick with however many slots are ready). Do **not** add self-play slots or otherwise pad K to "keep the GPU full": a batch of however many vs-UCI slots happen to be ready is fine and is the existing contract.
     - Alternate the trainer's color per game per opponent for unbiased WDL labels.
  3. **Flush to `ReplayBuffer`.** Completed vs-UCI games flush via the existing self-play flush path with the terminal WDL outcome from the **trainer's** point of view. No new buffer machinery; no throttling.
  4. **Headless CLI runner (sibling to `CorpusReplayRunner`).** New `--train-vs-uci` mode, **headless-first** (runs with no window, like `--train`/corpus replay; long unattended/server runs). Surfaced **read-only** in the console (per-opponent W/L/D, games/sec, buffer fill; reuse the driver's existing `K==1 → live display` gate to watch one game). **No orchestration logic in views** (single-source-of-truth rule). Model I/O mirrors corpus replay exactly: `--start-model` (starting trainer), `--out-model` (rolling latest, overwritten by the periodic autosave — `autosaveEvery = 1000`), `--enumerate-checkpoints` (step-enumerated copy on each save = the **output model at 1000-step slots**), `--parameters`, `--preset`, `--output` (time/step-budget gating).

  **CLI shape (approved).** Repeatable per opponent kind:
  `--train-vs-uci "cmd=/path/to/stockfish;n=3;go=nodes 1;UCI_Elo=1400"`
  where each occurrence carries `cmd` (executable path), `n` (instance count = concurrent slots for that opponent), `go` (the opponent's per-move limit, e.g. `nodes 1` / `depth 4` / `movetime 50`), and arbitrary `setoption` pairs (e.g. `UCI_LimitStrength=true;UCI_Elo=1400`). Vocabulary intentionally echoes cutechess's `-engine cmd=… option.X=…`. Example: `--train-vs-uci "cmd=…/stockfish;n=3;go=nodes 1" --train-vs-uci "cmd=…/sloppy;n=4;go=depth 4"` → 7 concurrent opponent slots (+ optional self-play backfill).

  **Stats output (required) — per-kind summary then per-instance breakdown.** Every 10–15 s emit a session-log block (new tag, e.g. `[VS-UCI-STATS]`), reusing the existing periodic-emit cadence. First a **per-opponent-kind summary line** (one per kind, aggregating its `n` instances), then a **per-instance breakdown** (one line per individual engine process). Each line reports: games completed, plies played, games/sec, plies/sec, and W/L/D from the trainer's perspective. The per-kind summary is the operator's headline tuning instrument; the per-instance lines expose stragglers/imbalance between instances of the same kind (e.g. one Stockfish process wedged or slow). The spread is adjusted between runs by changing instance counts / `go` / strength options, never by throttling.

  **Validation / success criteria.**
  - **`UCIArbiter` unit tests** (pure-logic, per the CLAUDE.md testing guidance) driven against a **fake engine subprocess** (a tiny script emitting canned UCI, so no real Stockfish needed): handshake state machine (`uci`/`uciok`, `isready`/`readyok`), `position … moves …` construction from a game's move list, `bestmove` parsing incl. `bestmove (none)` and a trailing `ponder`, `setoption` formatting, and LAN round-trip through existing `ChessMove+UCI`.
  - **Driver-adaptation test** with a scripted deterministic "engine": a mixed batch (self-play + vs-UCI slots) advances; slots awaiting an external `bestmove` are excluded from the trainer-eval batch that tick; completed vs-UCI games land in `ReplayBuffer` with the correct terminal WDL from the trainer's POV for **both** colors.
  - **End-to-end smoke** — a short `--train-vs-uci` run against real Stockfish at `nodes 1` and Sloppy: assert (a) buffer fills, (b) trainer steps advance, (c) `--out-model` + enumerated checkpoints written at the 1000-step cadence, (d) the `[VS-UCI-STATS]` line appears every 10–15 s with nonzero games/plies and a W/L/D split, (e) changing instance counts shifts the observed mix.
  - **Spread sanity (operator check, not an automated gate):** a weak-opponent config yields a *mix* of W/L/D (not ~100% losses), confirming the strength-matching lever works.
  - All existing tests continue to pass unmodified.

- **Corpus shard format v2 — rich provenance + per-game/per-ply metadata (PLAN ONLY, drafted 2026-06-27; no code yet).**

  **Why.** The v1 `.dcmgames` format (see the shipped corpus entry below) is deliberately minimal. Per game it stores only `flags / outcome / terminationReason / moveCount / packed-moves / optional startFEN`; per shard only `corpusID / sourceID / shardSeq / createdAtUnix`. During PGN import everything else the PGN carried — White/Black Elo, `WhiteRatingDiff`/`BlackRatingDiff`, `TimeControl`, titles (incl. `BOT`), `ECO`/`Opening`, `UTCDate`/`UTCTime`, `Event`, `Site` (the unique lichess game id/URL), and per-move `[%eval]`/`[%clk]` annotations — is parsed *only* to apply the `minRating`/`timeControlClasses` filters and then **thrown away** (`PGNImporter.buildGame`). Consequences we hit on 2026-06-27 while debugging the ReZero runs: (a) no per-game unique id — a corpus game is addressable only positionally (shard seq + ordinal), so no dedup and no trace back to source; (b) cannot re-filter a built corpus (e.g. "≥2000 only", "classical only", "after date X") without a full re-import; (c) no record of *which* app build or *what* filter produced a corpus (`corpus.json` omits the filter spec); (d) no engine-eval signal, which is the single biggest missed opportunity (see below); (e) a latent enum bug (also below). Storage is cheap relative to the analyses this unlocks — **bias toward capturing more, not less.**

  **Design principle: extensible, self-describing, lossless-enough.** Bump `frontMagic`/`version` to v2 but make per-record bodies **TLV-structured** (a required core followed by a sequence of `(fieldTag: u16, len: u32, bytes)` optional fields) so new fields can be added forever without a version break and old readers skip unknown tags. Reserve a version bump only for changes to the required core. Keep the good v1 bones: append-only, per-record CRC32, front header + sealed trailer, little-endian.

  **Shard front header v2 (per-shard provenance).**
  - `shardUID` (16-byte UUID) — globally unique shard identity (v1 has only `shardSeq`).
  - `corpusID`, `sourceID`, `shardSeq`, `createdAtUnix` (carried over).
  - **Writer identity:** app name, app version string, build number, git hash (we already stamp these into `BuildInfo`/`corpus.json` sources — put them in the shard too so a detached shard is self-describing).
  - **Source descriptor block:**
    - `sourceKind` enum (`pgnImport` / `selfPlay` / `arena` / `external`).
    - `sourceText` free string — human description, e.g. `"Lichess standard rated games, lichess_db_standard_rated_2026-05, no filter"` or `"DCM self-play, champion 20260626-2-q2Bb"`.
    - `sourceURL` (e.g. the lichess database URL), `inputFilePath` (for conversions), and `inputFileSHA256` (provenance / dedup of source files).
    - **`filterSpec`** — the *exact* filter applied: `minRating`, `timeControlClasses`, date range, variant filter, "FEN/SetUp skipped" flag, etc. (Today this is unrecoverable — a corpus's filtering is invisible after the fact.)
    - `dateRangeCovered` (min/max game date in this shard/source).
  - **`featureFlags` bitset** — which optional per-game / per-ply fields this shard actually contains (`hasElo`, `hasRatingDiff`, `hasTimeControl`, `hasTitles`, `hasECO`, `hasOpeningName`, `hasNames`, `hasDate`, `hasEval`, `hasClock`, `hasRawTags`). Lets a reader/query know without scanning records.

  **Sealed trailer v2 (cheap query without a full scan).** Carry over `gameCount`/`plyCount`/CRC and add an **aggregate stats block** computed at seal: W/D/L counts, rating histogram (coarse buckets per side), time-control-class mix, ECO/opening histogram, game-length histogram, termination-reason mix, eval-coverage %. Most "what's in this corpus?" questions then answer instantly from trailers; consider also a corpus-level `stats.json` aggregating across shards.

  **Per-game record v2.**
  - *Required core:* `gameUID` (our 16-byte id, content-or-random), `outcome`, `terminationReason` (fixed enum, see below), `moveCount`, packed moves, optional `startFEN`.
  - *TLV optional fields:* `sourceGameID` (string — e.g. lichess 8-char id from `Site`), `sourceURL`, `whiteElo`/`blackElo` (u16), `whiteRatingDiff`/`blackRatingDiff` (i16), `ratingType`/`timeControlClass` enum, raw `TimeControl` (base seconds u16 + increment u8, or string), `whiteTitle`/`blackTitle` enum (incl. `BOT`), `whiteName`/`blackName` (strings — lichess data is public), `ecoCode` (pack `A00`–`E99` into u16), `openingName` (string), `utcDateTime` (i64 unix), `event`, `contentHash` (hash of `startFEN`+packed moves, for source-independent dedup), NAGs/comments if useful, and a **`rawTagsBlob`** (optional, compressed) holding the original PGN tag set verbatim as a full-fidelity escape hatch.

  **Per-ply optional channels (the big one): `[%eval]` and `[%clk]`.** Lichess "analysed" games carry a Stockfish eval per move and clock per move. Capturing eval per ply (i16 centipawns, with a mate-in-N sentinel encoding) would let us **train the value head against engine eval** — a vastly stronger, denser signal than the single terminal WDL label we use today — and enable policy-distillation experiments. Clock per ply (u16 seconds) enables time-management modelling and quality filtering (e.g. down-weight moves played in time-scramble as noisy). Store as optional parallel arrays gated by `featureFlags`; they roughly double a game's bytes when present, which is acceptable per the "keep data" stance — but make them opt-in per shard so self-play (no eval) doesn't pay.

  **Enum fixes (carry into v2).**
  - **`terminationReason` nil→checkmate collision (latent bug).** v1 `encodeRecordPayload` writes `terminationReason?.rawValue ?? 0`, and `checkmate == 0`, so a *nil* (unknown) reason is indistinguishable from checkmate in the byte — PGN imports leave it nil, so every imported game looks like "checkmate" if a reader ignores the `0x02` "present" flag (this confused our 2026-06-27 corpus analysis). v2: make `0 = unknown/unspecified` an explicit first enum case (shift the real reasons up), so there is no collision and no reliance on a sidecar flag.
  - **Expand `GameTerminationReason`** to cover PGN `[Termination]` reality: `unknown, checkmate, stalemate, fiftyMoveRule, insufficientMaterial, threefoldRepetition, resignation, timeForfeit, drawAgreement, abandoned, rulesInfraction/flagged, adjudication, other`. Termination quality matters for label trust (a timeout in a winning position is a noisy label).
  - Add `RatingType`/`TimeControlClass` (`bullet/blitz/rapid/classical/correspondence/unknown`) and `PlayerTitle` (`none/GM/IM/FM/CM/NM/WGM/.../BOT`) enums.

  **Reconsider loading PGNs directly (vs. the binary corpus).** Worth revisiting now that we want full PGN fidelity:
  - *Direct-PGN pros:* zero conversion loss, single source of truth, re-filter anytime, human-readable.
  - *Direct-PGN cons (why v1 went binary, still valid):* SAN parsing needs full legal-move generation per ply (slow); lichess monthly dumps are huge (tens of GB); no random access; no integrity check; and the parse cost repeats **every epoch** over millions of games.
  - *Recommendation:* keep the binary corpus as the **canonical training fast-path** (throughput over millions of plies/epoch rules out re-parsing PGN each pass), but (1) capture ~all PGN metadata in v2 so we rarely need the original, and (2) optionally retain the compressed `rawTagsBlob` per game for full fidelity / re-derivation. A streaming PGN *reader* for ad-hoc one-pass jobs is a reasonable separate utility, but not the training path.

  **Adjacent wins to consider while here:** a per-shard **offset index** (game→byte offset sidecar) to enable true shuffled/random sampling across the corpus instead of sequential replay (better training mix; also listed as a v1 "Deferred" item below); and content-hash **dedup** across multiple imported months/sources.

  **Questions v2 should make answerable** (the litmus test for the field set): rating/time-control/ECO/length/result distributions over any slice; "train only on ≥N-rated / classical / titled / non-bot games" as a post-hoc replay filter; trace any training game back to its lichess source; dedup across sources/months; value-head training from engine eval; time-pressure analysis; provenance audit ("which build + filter produced this corpus, from which input file"); and detecting meta/distribution shift across date ranges.

  **Migration / back-compat.** No in-place rewrite of v1 shards (append-only, and they're large); the reader keeps decoding v1 (`version == 1`) with the minimal field set, and new corpora are written v2. The TLV body means subsequent field additions are *not* version bumps. Note in the eventual commit that v1 corpora simply lack the richer fields (queries over them return "unknown" for absent fields).

- **Self-play corpus recording + replay — ✅ SHIPPED 2026-06-20** (`f0c0012`
  corpus format/store, `0ad27b3` recording tee, `c4243b5` offline replay
  `--replay-corpus`/`--epochs`, `461c95c` provenance, `049e94f` PGN import,
  `5a93c54` usage). The full design below is preserved as the as-built record;
  only the **Deferred** sub-list (seeded shuffle, batch-level replay,
  multi-writer ingestion, shard-index cache) and CLI-resume issues #1/#3/#4
  remain open — this entry belongs in Completed. Record completed self-play games to a reusable,
  architecture-independent **game corpus** so multiple architectures /
  hyperparameter settings can be trained on *identical* inputs, and so external
  PGN (Lichess `.pgn.zst`) can be imported as training data. PGN is an
  import/export converter only; the replay path consumes only the native
  `GameRecord`. Current scope: **game-level replay**; batch-level replay and
  seeded shuffle are deferred (see end of entry).

  **Why move-lists — not encoded tensors, not seeds.** A self-play game is just
  a move list + result, which re-encodes into *any* architecture's input planes
  at train time, so it is the only architecture-independent record. It is also
  tiny: a move packs into 2 bytes (from 6b + to 6b + promo 3b; castling/EP
  inferred on replay; per-ply tau is reconstructable from the deterministic
  `SamplingSchedule`, so not stored), i.e. ~160 MB/h at 80M plies/h — versus
  ~600 GB/h (F32) if we stored encoded `basic30` tensors (1920 floats/pos). We
  do **not** chase seed-level reproducibility of self-play: MPSGraph/Metal
  forward passes are not bit-reproducible run-to-run, so a categorical sample
  can flip on a ULP — record the games that actually happened instead.

  **`GameRecord`.** Start position (standard assumed; optional `startFEN` only
  for imported setups), the 2-byte move list, result (W/D/L), and a `sourceID`
  reference. The replay loader replays the moves through `ChessGameEngine` and
  the target arch's `BoardEncoder`, appending each game as one contiguous
  reverse-ply block via the **same append path self-play uses** — required so
  the buffer's history-plane reconstruction (which reads neighboring ring slots)
  stays correct.

  **Corpus store + shard files.** A standalone `Corpora/<corpusID>/` store
  (alongside `Models/`, `Sessions/`), **never embedded in a session folder** — a
  corpus is unbounded and append-only, and sessions re-save to a fresh folder on
  every trigger (today `replay_buffer.bin` + `training_chart.json` are
  re-serialized in full each save, `CheckpointManager.swift:656`), so embedding
  would be O(saves × corpus) copying. `session.json` instead carries a
  `recordingCorpusID` *reference* + a high-water mark (games/shards/plies at
  save) via the Optional-field + `[RESUME-PARAM]` pattern (mirror
  `batchStatsInterval`) — a NEW pattern; nothing currently references data
  outside its own session folder. Shards are **self-describing** and sized by a
  soft byte target (`--shard-soft-limit-mb`, default 64) cut at a whole-game
  boundary, so a shard always holds an integer number of complete games (the
  game is the atomic replay unit). Layout: a fixed 256-byte **front header**
  written once at create (magic, formatVersion, `corpusID`, shardSeq,
  `sourceID`, createdAt — records start at offset 256); a **body** of
  length-prefixed game records each with a CRC-32; and a fixed 64-byte
  **trailer** appended at seal (trailerMagic, gameCount, plyCount, sealUnix,
  SHA-256 over `[0, EOF−64)`). Seal-time facts (count, SHA) live in the trailer,
  **not** backfilled into the front — the file is pure append, matching the
  trailing-SHA pattern of `.dcmmodel`/`replay_buffer.bin`. Two checksums, two
  jobs: per-record CRC for open-log recovery, whole-shard SHA for sealed
  integrity. Open→seal: write the front header (fsync); append games with a
  streaming SHA, fsync on a cadence (per game / ~1 s); seal = finalize SHA,
  append trailer, fsync, atomic rename `.open`→final, fsync dir, then record the
  source. Crash-safe at every step — the trailer's presence is the commit; an
  `.open` file with no valid trailer is recovered by CRC-scanning to the last
  complete game (loses only the un-fsync'd tail).

  **Metadata — one `corpus.json`.** A single provenance-only file:
  `{ corpusID, name, comment, state, createdAt, sources: [...] }`, where
  `sources` is an **append-only** list with one record per ingestion (self-play
  session or PGN import): `sourceID`, kind, `input{filename,url}`, options (incl.
  shard size + import filters), `appBuild` (from `BuildInfo.swift`), timestamp,
  counts. Written **only on ingestion events** (rare) — shard seals never touch
  it. It holds only the **non-reconstructable** bits (name, comment, provenance);
  the shard list and aggregate counts are **derived** by `readdir` + reading each
  64-byte trailer, so a stale/lost `corpus.json` is never data loss (only
  name/comment/provenance would be lost — hence atomic-write + fsync it). No
  persisted shard index by default; add a throwaway `manifest.json` cache later
  only if listing is slow. Per-shard sidecars were considered and rejected
  (redundant with the trailer; two-file atomicity for no gain); per-source files
  only become worthwhile under concurrent multi-writer ingestion into one corpus
  — not a current need.

  **Recording (post-filter, both train paths).** Tee the corpus writer at the
  point games flush to the `ReplayBuffer` — i.e. **after** the random draw-keep
  filter — so the corpus *is* the exact training set, frozen and identical for
  every replay (recording raw would force re-applying a random filter at replay,
  giving different inputs per run). `moveHistory` is already in hand there (the
  same point `GameDiversityTracker` uses). Writes go through an async append
  queue so recording never stalls the self-play hot path. Gated by a new
  `recordSelfPlayGames` bool `@TrainingParameter`. Must be wired into **both** the
  GUI and the `--train` paths — the headless path has no game-capture hook today.

  **Corpus identity & lifecycle.** A recording run mints a **new corpus**
  (auto-id) by default, with opt-in append-to-named for deliberate continuation.
  State goes recording → sealed/frozen (replay uses a frozen corpus). Replay
  accepts **multiple** `--replay-corpus` so self-play + Lichess can be mixed; the
  feeder interleaves them.

  **PGN import.** Stream `zstd -dc` via subprocess (libzstd fallback; Apple's
  `Compression` framework lacks zstd) — never inflate a ~200 GB dump to disk.
  SAN→`ChessMove` by generating legal moves per ply and matching. Standard-start
  games only (skip FEN-setup / Chess960 / variants); filters `--min-rating`,
  `--time-control`, `--max-games`, min-plies; skip malformed. Output =
  `GameRecord` shards in a corpus; the import is one `sources[]` record with its
  file/url/options/appBuild.

  **Replay — offline, step-locked feeder.** Fixed corpus → `ReplayBuffer` →
  trainer, with **no self-play and no replay-ratio controller**. Pre-fill the
  ring, then append `K = batchSize / R` positions per training step (`R` = reuse;
  reuse the existing `replayRatioTarget`, reinterpreted offline). Ring capacity
  vs corpus selects the regime: fits → load once (fixed dataset); bigger →
  streaming moving window. Stream shards in stored order (deterministic; shuffle
  deferred). The random batch sampler is unchanged and **unseeded**, so for
  game-replay the A/B protocol is "same frozen corpus + same step-locked feed (so
  buffer contents are identical across runs), run each architecture N times and
  average out sampling noise." Stop on a `--steps` or `--epochs` budget (not
  corpus exhaustion); `--training-time-limit` stays opt-in for
  hardware-vs-hardware comparison only.

  **Replay run lifecycle (train-only).** Self-play workers off; the
  arena/promotion/candidate/`arenaChampionNetwork` apparatus stays dormant; the
  trainer trains `trainer.network` to the budget. Start fresh (build a new net of
  the chosen arch) or from a built net (`--start-model`). Checkpoint triggers =
  periodic + manual only (nothing promotes), and identity is stable (no mid-run
  ModelID forks). Strength is measured read-only via the periodic Lichess probe
  (eval-loss — deterministic, architecture-agnostic, the comparable metric); an
  arena vs a *fixed reference* net is allowed as pure measurement (never copies
  weights).

  **Reproducibility & the linchpin invariant.** The A/B harness already mostly
  exists: freeze params (`--create-parameters-file` → `--parameters
  frozen.json`), freeze init (`--start-model` for same-arch, or fresh×N for
  cross-arch), freeze budget (`--training-step-limit`/`--epochs`); the only new
  axis is the frozen corpus. **Required invariant:** re-encoding a recorded
  self-play game must reproduce *bit-for-bit* the input the trainer originally
  saw — `BoardEncoder` is a pure function of (start + move sequence), with the
  repetition/history planes determined by replaying the moves. This is testable
  (record → re-encode → compare to the live-encoded frames) and must hold before
  any replay result is trusted.

  **CLI surface + usage.** New flags: `--record-games`, `--import-pgn <path>`
  (+ `--min-rating`/`--time-control`/`--max-games`), `--corpus-name`,
  `--shard-soft-limit-mb` (default 64), `--replay-corpus <path>` (repeatable),
  `--epochs <n>`. Overhaul the usage banner to document **every** flag (new +
  existing) with worked examples for the common scenarios: a cross-architecture
  A/B, a same-init hyperparameter A/B, a PGN import, and a replay run from a built
  model vs fresh. Record the corpus id(s), reuse `R`, budget, and import filters
  in both `session.json` and the `--output` results.json for traceability.

  **Build order.** (1) `GameRecord` + corpus store + writer (`corpus.json`,
  self-describing shards, seal/recover); (2) recording tee'd off the self-play
  flush, GUI + `--train`; (3) replay loader + step-locked feeder + offline mode +
  `--replay-corpus`/`--epochs`; (4) train-only lifecycle wiring; (5) manifest
  fields in session.json/results.json; (6) PGN import; (7) usage overhaul. The
  same-init/same-params/same-budget pieces (`--start-model`, `--parameters`,
  `--training-step-limit`) already exist and are verified.

  **Deferred:** seeded game-level shard-shuffle (shard-order permutation +
  in-shard shuffle + optional reservoir), batch-level replay (seeded sampler /
  literal batches), concurrent multi-writer ingestion (would reintroduce
  per-source files), and a persisted shard-index cache.

  **Resume-from-non-latest provenance (must warn, must not silently destroy).**
  On resume, compare the session's saved watermark against the corpus manifest's
  current head. watermark == head → continue appending, no issue. watermark <
  head (resuming *behind* the corpus's current state, e.g. loading an older
  autosave) means the corpus already holds games "ahead" of the resume point.
  Policy: **never truncate or overwrite previously-sealed shards** (violates the
  project's "nothing is ever overwritten" + "never delete the user's data"
  rules). Detect the condition, surface it explicitly in the resume UI/log, and
  offer (a) **fork** — mint a new corpus id, original preserved sealed, new games
  recorded into the fork (default; clean provenance), or (b) **continue same
  corpus** — accept a superset containing both continuations past the watermark
  (fine as training data, muddy provenance). Headless/CLI resume has no
  interactive prompt, so it must default to fork + a loud log line stating
  exactly what happened. Also handle corpus-id-points-at-missing-data
  (deleted/moved externally) → warn + offer create-fresh or disable-recording,
  mirroring the `LastSessionPointer` target-deleted handling.

  **CLI resume open issues (documented 2026-06-20).**
  1. **`results.json` is not cumulative across resume.** The `--output` snapshot
     is an atomic overwrite written by a *per-run* `CliTrainingRecorder`
     (`SessionController+Training.swift:1015`; `CliTrainingRecorder.writeJSON`
     uses `.atomic`, no append). A stopped-then-resumed headless run writes a
     results.json covering only the post-resume segment. GUI session state
     (charts, counters, `TrainingSegment`s) *is* continuous on resume; the CLI
     snapshot is not. Needs a continuation mode (load prior recorder state /
     accumulate segments) if cumulative headless results are wanted.
  2. ~~Self-play recording is not wired into the CLI train path.~~ **RESOLVED
     2026-06-20.** Recording is wired into both the GUI and `--train` paths —
     `TrainingParameters.shared.recordSelfPlayGames` is read at
     `SessionController+Training.swift:1040` and tees games into the corpus.
  3. **`results.json` flush is signal-gated and misses SIGINT.** Written only on
     a budget firing (`--training-step-limit`/`--training-time-limit`) or
     SIGUSR1/SIGHUP/AppKit-terminate — *not* SIGINT/Ctrl-C, and never on crash.
     A Ctrl-C'd run intended for resume may have flushed nothing.
  4. **Step-limit granularity.** The step watcher polls every 2 s
     (`SessionController+Training.swift:2512`), so it stops a few steps past the
     limit — fine for budgets, not bit-exact for strict A/B; gate exactly at N
     if exactness is needed.

- **History dropout (training-time input masking) — deferred future feature.**
  For history-stacking input encodings (`full10ply200`), a
  `historyDropoutProbability` ∈ [0,1] training augmentation: with probability *p*,
  a sampled position has its history frames (planes 20–199) zeroed so the net
  trains on frame N only — regularizing against over-reliance on history (a known
  failure mode for history-input nets that AZ/Lc0 did not deliberately address).
  **Binary** (not random-depth), applied at *sample* time in `ChessTrainer` (one
  `vDSP_vclr` per masked position, before GPU upload) so the real full history
  stays in the replay buffer and the same position can be seen with/without
  history across epochs. `liveTunable`, default 0, **no-op for single-frame
  encodings** (basic20/basic30). Self-play / arena / Play Game always feed real
  full history (train-with-dropout, play-clean). Full spec + the
  `@TrainingParameter`-checklist touchpoints live in `FULL10PLY200_PLAN.md`
  (Phase 5). **Not in the current full10ply200 scope (Phases 1–4); revisit after
  the encoding is training cleanly and we can measure whether history
  over-reliance actually appears.**

- **Long-run UI hang + throughput decline (investigated 2026-06-05).** Symptom:
  after ~12 h a session develops a periodic ~1 s main-thread hang on the
  heartbeat cadence, and self-play/training throughput declines ~25 % over a
  day. **Root cause (proven by Instruments + the per-stage heartbeat traces):**
  every 5 s the `snapshotTimer` re-evaluates the whole 9.6k-line
  `UpperContentView.body`, which rebuilds a large `@Observable` tracking set —
  the ~1 s is pure `ObservationRegistrar.cancel`/`registerTracking` +
  `AnyKeyPath.hash` + `Set`/`Dictionary` churn, inside a `ViewThatFits`/
  `GeometryReader` layout (SwiftUI `Charts` internals). The tracking set grows
  with accumulated run state, so the hang only appears after many hours. The
  **throughput** half was the *same event*: `BatchedSelfPlayDriver.runOneTick`
  did `await MainActor.run { TrainingParameters.shared.X }` 3–5×/tick, so during
  each UI hang self-play blocked on the hung MainActor. Ruled out (with data):
  memory leak (rss/gpuMem flat), thermal (`thermalState` never above "fair"),
  the lichess probe (its charts are already `FastLineChart`), and the replay
  controller. Battery/clamshell-sleep episodes are a real but intermittent,
  non-progressive contributor to throughput only.

  **Phase 1 — DONE (2026-06-05, builds clean, not yet long-run-verified):**
  decoupled training from the MainActor.
  `BatchedSelfPlayDriver` now reads its 5 live-tunable params from a `SyncBox`
  (`SelfPlayLiveParams`) refreshed by an off-hot-path task — zero per-tick
  MainActor hops, so a UI hang can never stall game production again. Heartbeat
  `asyncCompletedTrainSteps()`/`asyncEffectiveLearningRate()` (which dispatch to
  the *starved* global pool, costing ~0.8 s each) swapped for the sync
  `completedTrainSteps`/`effectiveLearningRate` getters. Net: self-play's hot
  path does ~0 MainActor hops (was ~39/s), which also un-starves the global
  dispatch pool and shortens the heartbeat tick. **This fixes the throughput
  decline and shortens the tick; it does NOT remove the ~1 s observation
  teardown.**

  **Phase 2 — NOT done (the UI-hang cure), deliberately not shipped blind.** The
  fix is to stop the heartbeat from re-evaluating the whole body: move every
  read of the hot heartbeat-written `@Observable` props (`trainingStats`,
  `parallelStats`, `gameSnapshot`, `trainerWarmupSnap`, `replayRatioSnapshot`)
  out of `UpperContentView.body` into small dedicated observing `View` structs,
  and migrate the 4 always-visible native-`Charts` tiles (`ArenaActivityChart`,
  `ArenaWinChart`, `DrawWatchHistogramChart`, `DiversityHistogramChart`) to the
  path-based `FastLineChart` (no `ViewThatFits`). Two blockers to doing it as a
  one-shot: (a) it's **all-or-nothing per property** — partial moves yield no
  measurable win because the body still re-evaluates; (b) those reads are spread
  across the whole view, so it's a deep refactor, and there's **no fast feedback
  loop** (the hang takes ~12 h to reproduce; a fresh launch shows nothing).
  **Decisive next data point:** on the next long run, when the hang appears,
  expand the Instruments call tree just *above* `ObservationRegistrar.cancel` to
  the `DrewsChessMachine` view frame + the collection it iterates — that pins
  the exact growing structure in one profile, after which the migration/
  decomposition is targeted and confident rather than speculative. (Candidate
  but unconfirmed: a native-`Charts` tile's `ForEach` over growing
  `tournamentHistory`.) Also consider gating the heartbeat to a single
  end-of-tick batch of `@State`/`@Observable` writes (reduces re-eval frequency
  ~10×→1×) — contained to `SessionController+Heartbeat`, but a high-risk
  restructure of the UI-state path.

- **Safetensors-native storage + runtime-configurable architecture.** In
  progress on branch `safetensors-storage`; full design + phase plan in
  `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md`. **Done (tested):** model/session weight
  files are now safetensors (`.safetensors`, PyTorch-drop-in layout, Python-
  loadable, no exporter); legacy `.dcmmodel` still reads; `ChessNetwork` builds
  from a `NetworkArchitecture` value type; build any architecture via
  `architecture.json`. **Update 2026-06-23 — effectively all shipped:**
  non-default architectures are trainable end to end (heterogeneous block-groups
  towers, 2026-06-12) and saved with their embedded config; per-model compute
  precision (f32/bf16/fp16) is selectable (fp16 added 2026-06-17, inference-only —
  training NaNs by design); legacy `.dcmmodel` old-arch loading via the
  `archHash→config` fallback is in place; `--uci` (`28cf394`) and `--playchess`
  (`26c14e9`) both ship. The headless build flag now ships too: a single
  **`--new-model --architecture <name|preset.json|path>`** flag mints a fresh
  untrained net from a built-in preset, a user-saved preset, or an arch JSON file
  (`ArchitecturePresetStore.resolve(nameOrPath:)` + `NewModelCLI`) — see
  `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` for the authoritative phase list.

- **Standalone "Training vs Eval Loss" window.** Planned (added 2026-06-05).
  A separate, freely-resizable `NSWindow` (following the established
  `LichessProbeMonitorWindow` pattern: `NSWindowController` + single-instance
  registry + `Launcher.openWindow(sessionController:)` + a `Performance` menu
  button wired through `AppCommandHub`) that overlays two trajectories on **one
  plot with two independent auto-scaling Y axes**, both indexed on the shared
  X = trainer step (`ChessTrainer.completedTrainSteps`):
  - **Training total loss** (leading axis): `rollingPolicyLoss + rollingValueLoss`
    (`TrainingChartSample.rollingTotalLoss`), from `chartCoordinator.trainingRing`.
  - **Eval loss** (trailing axis): wide-set (~4,435-puzzle) bookmove cross-entropy
    `meanNegLogProb`, from `lichessProbeWideHistory.overallSeries`
    (`OverallTickSample`, already step-indexed and persisted).

  Interpretation caveat to keep in the docstring: the eval line is *pure* policy
  cross-entropy, while the training line is *outcome-weighted* policy CE + value
  CE (`pLoss` can go negative) — related but not the same metric, hence the dual
  auto-scaled axes; the *trends* are the signal (both falling = healthy; train
  falling while eval flattens = overfitting/plateau).

  **Enabling schema change:** add `trainingStep: Int?` to `TrainingChartSample`
  (Optional → additive-safe per that struct's own back-compat rule; persists
  through `ChartFileFormat` automatically), populated at the existing heartbeat
  construction site (`SessionController+Heartbeat.swift:516`) from the step the
  heartbeat already fetches. Pre-existing samples carry `nil` step.

  **Rendering:** mirror the lichess overall-trend look — reuse
  `SwiftUIFastCharts.FastLineChart` + the existing EMA helper
  (`LichessProbeOverallTrendChart.ema`) with the raw series faded to a faint
  "noise cloud" (opacity ~0.28) behind a bold EMA line, per-series toggle/span.
  Note `FastLineChart` exposes only a single `yDomain` with **leading-only** axis
  labels, so the dual-axis overlay requires (a) linearly remapping the eval
  series into the training-loss domain and (b) a hand-rolled trailing-axis label
  column whose labels inverse-map the same pixel positions back to eval units.
  (The lichess precedent stacks two single-axis charts sharing X instead — the
  native-grain alternative if the dual-overlay custom work isn't wanted.)

  **Decisions (locked 2026-06-05):**
  1. *Back-fill:* interpolate a step for each pre-existing stepless
     `TrainingChartSample` via piecewise-linear interpolation of its `elapsedSec`
     against exact `(step ↔ elapsedSec)` anchors — the regular lichess probe's
     `overallSeries` ticks every ~25 steps and stores `(trainingStep, timestamp)`
     (timestamp → elapsedSec via `chartElapsedAnchor`), plus a terminal anchor at
     `(current elapsedSec, current completedTrainSteps)`. Back-fill runs once
     in-memory (guarded) when the window first opens and **persists on the next
     save** (user OK'd saving estimated steps). Falls back to a global
     linear-in-time map if no anchors exist.
  2. *Layout:* single dual-axis overlay (leading = training total loss, trailing
     = eval NLL), implemented by adding **real secondary/trailing-axis support to
     `FastLineChart`** (per-series `yAxis: .primary/.secondary` + `secondaryYDomain`
     + trailing label column; backward-compatible, defaults preserve current
     behavior) so future charts reuse it — not a one-off overlay.

  No `TrainingParameter` involved (the parameter checklist doesn't apply).
  Pure-logic XCTests in scope: the step-interpolation function and the
  `ChartAxisLayout` right-column geometry.

- **`BatchFeedsInput` struct for `ChessTrainer.buildFeeds`.** ✅ **SHIPPED
  2026-05-11** (`49878fa`). `buildFeeds(_ input: BatchFeedsInput)` now takes a
  single named-field struct (`struct BatchFeedsInput` at `ChessTrainer.swift:5659`),
  so the compiler binds inputs by name rather than position — the original
  same-typed-`UnsafePointer<Float>`-swap safety concern is closed.

  Planned shape remains unchanged: wrap the inputs in a small `BatchFeedsInput`
  struct with named fields so the compiler binds by name rather than position.
  No behavioral change; pure call-site safety. Re-check `runPreparedStep` at the
  same time if it grows beyond its current `feeds`, `prepMs`, `queueWaitMs`, and
  `totalStart` argument list.

- **Autosave retention pruning.** Still open, with corrected current-state
  details. The old roadmap text was directionally right that autosaves are kept
  indefinitely, but it understated the current persistence payload: modern
  `.dcmsession` directories can include `replay_buffer.bin`, and the current
  replay-buffer file format is v7, so disk growth can be larger than the older
  "model + session.json only" session plan implied.

  Current implementation evidence:
  - `UpperContentView.periodicSaveIntervalSec` is `4 * 60 * 60`, and
    `PeriodicSaveController` schedules 4-hour saves while Play-and-Train is
    armed. The controller defers during arenas and resets its deadline after any
    successful manual, post-promotion, or periodic save.
  - Post-promotion autosave is enabled by `UpperContentView.autosaveSessionsOnPromote = true`.
  - `CheckpointPaths.makeSessionDirectoryName` generates unique names of the
    form `YYYYMMDD-HHMMSS-<sessionID>-<trigger>.dcmsession`; `CheckpointManager`
    refuses to overwrite existing targets.
  - No code path or menu item named "Manage Autosaves", "Trim to last N", or
    equivalent pruning UI was found. The File menu currently exposes "Resume
    Training from Autosave" and "Open Data Folder in Finder", not autosave
    retention management.

  Desired policy remains: manual saves are always preserved; post-promotion and
  periodic autosaves may be pruned beyond the last N (configurable, default on
  the order of 20); pruning should run lazily after successful saves so there is
  no dedicated sweep racing save/load; optional UI can show total disk footprint,
  counts per trigger, and a "Trim to last N" action. Deferred until disk
  footprint is a demonstrated problem; the "never overwrite" invariant remains
  in force until retention is explicitly implemented.

- **Human-vs-model play.** ✅ **SHIPPED 2026-05-14** (`15613c9`, "Add Chess menu
  with human-vs-network Play…"). `App/UpperContentView/PlayController.swift` is a
  full human-vs-network controller: a `HumanPlayOpponentChoice` model-slot picker
  (champion snapshot / trainer snapshot / live trainer / loaded file), a
  `humanColor` side picker, and tap-based human move entry. The distinct
  network-vs-network **Play Game** demo (`playSingleGame()`) still exists
  alongside it. The original "still open" analysis below is kept as design history.

  The original design goal remains valid: let a human play against either the
  champion or a trainer/candidate snapshot from the UI, for sanity-checking play
  quality and comparing a mid-training trainer against its parent champion.
  Implementation sketch, corrected against current code:
  - Engine side can reuse `ChessMachine`, `ChessGameEngine`, `ChessPlayer`,
    `MPSChessPlayer`, and `DirectMoveEvaluationSource`; a new human-controlled
    `ChessPlayer` implementation or a UI-driven move bridge is still needed.
  - Extend the current Play Game command/path with a model slot picker
    (champion / candidate inference network / frozen trainer snapshot) and a
    side-to-play picker.
  - For trainer snapshots, copy trainer weights into an inference network using
    the existing `exportWeights` → `loadWeights` path. The arena already uses
    this pattern for candidate/champion snapshots, and `ChessNetwork.loadWeights`
    validates tensor count and shape before assignment.
  - Do not block Play-and-Train: human-vs-model should use
    `candidateInferenceNetwork` or a dedicated inference slot, never call
    `trainer.network` directly while SGD continues.
  - If the user wants to preserve a mid-training opponent, expose a named
    snapshot/freeze action rather than tying the game to continually-mutating
    trainer weights.

- **Adaptive learning-rate schedule.** Still open, with important corrections.
  Current implementation is not a full schedule. `TrainingParameters.LearningRate`
  defaults to `1e-3` (raised for the bf16 weight path — see CHANGELOG) and is
  live-tunable/persisted. `ChessTrainer.buildFeeds`
  feeds the effective LR each step after applying two local multipliers: optional
  `sqrtBatchScalingForLR` and linear warmup over `lrWarmupSteps`. The UI lets the
  user edit learning rate and warmup; `[PARAM] learningRate` and
  `[PARAM] lrWarmupSteps` logs are emitted on manual changes. Session restore
  persists/restores `learningRate`, `lrWarmupSteps`, and `sqrtBatchScalingForLR`.

  Not implemented: no `lr_init`, positions-per-decay `τ`, exponential decay,
  promotion multiplier, LR floor, auto/off schedule toggle, schedule read-only
  slider mode, or schedule persistence fields were found. `trainingPositionsSeen`
  exists in `SessionCheckpointState`/logs but is not used to compute LR.

  ### Candidate trigger families (surveyed)

  - **Step-based decay.** LR multiplied by γ every N training steps
    (e.g., γ=0.5 every 100K steps). Predictable, tunable, but blind
    to actual training health.
  - **Plateau detection.** Watch a smoothed loss; when it stops
    decreasing for N consecutive measurements, multiply LR by 0.5.
    Standard "ReduceLROnPlateau" pattern.
  - **Promotion-driven.** Drop LR by a factor on every successful
    arena promotion. The intuition: promotion proves the current
    policy has shifted meaningfully, so subsequent updates should be
    gentler to lock in that progress before the next promotion
    window. Rejected as an upward-bump mechanism: arena failure more
    likely means the candidate diverged in a worse direction (LR too
    hot, or replay overfit) than "stuck in a flat region that a
    bigger step would escape." Raising LR on failure would make those
    cases worse. Promotion-driven *downward* nudge still useful as a
    secondary overlay.
  - **Cosine annealing with restarts (SGDR).** Smoothly decays LR
    across an "epoch" then restarts to the high value. Often
    empirically strong in supervised vision but adds another tunable
    (epoch length) and has no natural "epoch" concept in self-play.
  - **Replay-ratio aware.** Tie LR to the cons/prod ratio so that
    undertraining (cons < prod) doesn't get amplified by a too-hot
    LR.

  ### What AZ-family engines actually use (survey, 2026-04)

  Researched the five canonical systems' published LR schedules,
  normalized to positions-seen so batch-size differences don't
  confuse the comparison:

  - **AlphaZero** (Silver et al., 1712.01815 v1 + Science 2018). SGD
    + momentum 0.9. Start 0.2, step decay 10× per drop at 100K /
    300K / 500K training steps → 0.0002 floor. Batch 4,096 positions.
    Normalized: first drop at ~410M positions, final at ~2.05B, total
    ~2.87B positions across 700K steps. (Go variant starts at 0.02
    rather than 0.2.)
  - **AlphaGo Zero** (Silver et al., Nature 2017). SGD + momentum
    0.9, L2 = 1e-4. Start 1e-2, step decay 10× → 1e-3 → 1e-4.
    Extended Data Table 3 milestones not confirmable from primary
    source in this pass; ELF OpenGo (1902.04522), which explicitly
    reproduces AGZ faithfully, used 500K / 1M / 1.5M minibatches.
    Batch 2,048 positions. Normalized (via ELF): ~1.0B / ~2.0B /
    ~3.0B positions.
  - **Leela Chess Zero** (`lczero-training/tf/configs/example.yaml`).
    `lr_values=[0.02, 0.002, 0.0005]`, `lr_boundaries=[100000,
    130000]` steps, 250-step warmup, 140K total. Batch 2,048.
    Normalized: drops at 205M / 266M positions, total 287M. Caveat:
    that's the documented *example* — T60/T70/T78/T80/BT production
    runs don't publish per-run YAML values anywhere I could find.
    The blog/wiki describe "LR starts high and is occasionally
    reduced" without numbers.
  - **KataGo** (Wu, 1902.10565, and the live `python/train.py`).
    Parameterized in *samples* (= positions), not gradient steps.
    Paper "g170" run: base per-sample 6e-5, batch 256 →
    effective per-batch 0.01536. Warmup: first 5M samples at ⅓ LR.
    For the b20c256 run, one 10× drop late (after ~17.5 days). Not
    really step-decay in the AZ sense — effectively constant LR with
    warmup + one late drop, combined with SWA / EMA of snapshots
    (every ~250K samples snapshot; every ~1M samples a new EMA
    candidate with decay 0.75) doing work the others get from
    discrete drops. Current `train.py` has drifted to base per-sample
    3e-5 with a 9-stage warmup over first 2M samples and a piecewise
    `lr_scale_auto2` multiplier ramping 12× down to 0.05× over ~600M
    samples — much more elaborate than the paper.
  - **MuZero** (Schrittwieser et al., Nature 2020, board-games
    config in the paper's pseudocode). SGD + momentum 0.9, WD 1e-4.
    Start 0.1 (chess/shogi) or 0.01 (Go). **Exponential** decay, not
    step decay:
    `lr = lr_init · 0.1^(training_step / 400_000)`. Batch 2,048; 1M
    total steps. Normalized: LR falls 10× every ~819M positions. At
    1M steps (~2.05B positions) LR has decayed ~316× (chess final
    ≈ 3.2e-4).

  **Synthesis.** Common pattern is step decay at 10× per drop, ~3
  drops total, first drop somewhere between ~200M and ~1B positions
  seen. Nobody in this family uses cosine, cyclic, or warm restarts.
  Batches cluster at 2,048 (AGZ, lc0, MuZero), with AZ doubling to
  4,096 and KataGo going small at 256. Outliers: KataGo in *shape*
  (mostly flat + late drop + EMA), MuZero in *mechanism* (continuous
  exponential vs. discrete drops). The default `learningRate` is now `1e-3`
  (raised from `5e-5`/`5e-4` for the bf16 weight path), above KataGo's paper
  per-sample scale before any
  future schedule overlay.

  ### Chosen design (not yet implemented)

  MuZero-style continuous exponential decay keyed to
  `trainingPositionsSeen` (an invariant under live batch-size
  changes), with a promotion-driven secondary overlay:

  - **Primary schedule**: `lr = lr_init · γ_e^(positions / τ)` with
    `γ_e = 0.1` (10× per τ, matching the family) and default
    `τ = 500M` positions per 10× — puts us in the AZ / MuZero zone.
    Both `lr_init` and `τ` live-tunable in the UI.
  - **Promotion overlay**: on every successful arena promotion,
    additionally multiply LR by ~0.9 (consolidation nudge). Monotonic
    non-increasing — no upward bumps on arena failure. Per the
    rejection above, failure-upward is more likely to hurt than help.
  - **Floor**: 1e-7 so long runs don't collapse to zero.
  - **Warmup interaction**: linear warmup already exists today via
    `lrWarmupSteps` and should compose multiplicatively with the scheduled
    LR if/when the schedule is added.
  - **Manual UI override wins**: "Schedule: auto / off" toggle; with
    off, the slider is authoritative and the schedule is paused
    (preserves the current auto value so re-enabling doesn't snap).
    With on, the slider shows the current scheduled value read-only.
  - **Logging**: every schedule-driven change logged as
    `[PARAM] learningRate <old>→<new> reason=<decay|promotion|warmup>`
    so change history lives alongside `[STATS]` in the session log.
  - **Persistence**: `lr_init`, `τ`, `γ_promotion`, schedule on/off,
    and the last-computed LR should live in `session.json` so a reload
    resumes at the same scheduled value rather than jumping.

  Deliberately *not* doing: step-milestone decay (operates on
  training steps, which live-tunable batch size makes non-invariant),
  plateau-on-loss (`pLoss` is outcome-weighted and unbounded — unreliable
  plateau signal), replay-ratio aware (the `ReplayRatioController` already
  handles cons/prod; doubling up couples two control loops with no clear
  benefit), cosine/SGDR (no natural epoch in self-play; adds a tunable
  without a principled way to set it).

- **Compiled `MPSGraphExecutable`.** Largely shipped. **Batched inference**
  migrated to a compiled executable 2026-06-02 (`3f02ac4`,
  `ChessNetwork.inferenceExecutables`, `.level1`), and the **training step** now
  encodes through `ChessTrainer.trainingExecutables` (`graph.compile` +
  `executable.encode`; see GPU_UTILIZATION_PLAN.md Phase 2). Still on the direct
  `graph.run(...)` path: single-position `ChessNetwork.evaluate`, the export /
  BN-stats helpers. **Still open:** `ChessMPSNetwork.NetworkInitMode.package(URL)`
  throws `ChessMPSNetworkError.packageLoadingNotImplemented` — serialized-package
  loading is not implemented.

- **Replace `Engine ▸ Promote Trainee` (arena-only override) with a standalone
  "Promote Trainee Now" action.** **IMPLEMENTED 2026-05-11** (see CHANGELOG entry
  of the same date; pending the usual build + manual-validation pass below). Planned
  2026-05-11 after a user report that `Engine ▸ Promote Trainee` "didn't seem to do
  anything." Decision: **remove** the arena-override Promote entirely (not rename it)
  and add a new no-arena Promote-the-current-trainer action. Rationale: the
  arena-override semantics
  ("force-promote the in-flight candidate, ignore the score") are confusing and
  rarely what the user wants — if an arena is running and you want that candidate
  in, just let the arena finish or abort it; if you want the *current trainer*
  promoted right now, you want the new action. Keeping both is two near-identical
  buttons with subtly different targets.

  Why the old button felt broken, for the record (it was working as designed):
  `Engine ▸ Promote Trainee` → `commandHub.promoteCandidate` →
  `session.arenaOverrideBox?.promote()` (`UpperContentView.swift` ~1922,
  `ArenaOverrideBox.swift`) set a one-shot `.promote` decision that
  `runArenaParallel` consumed only when the tournament driver returned
  (`SessionController+Arena.swift` ~488–514, `promotionKind = .manual`); it was
  `.disabled(!realTraining || !isArenaRunning)` (`DrewsChessMachineApp.swift`
  ~417); and `TournamentDriver` only checks the override flag *between games*
  (`group.next()` loop ~298–299) and does **not** cancel in-flight arena games —
  they run to completion, replacements just stop spawning — so with several
  concurrent arena slots and slow games there is a long "nothing happened yet"
  window before `[ARENA] … promoted … kind=manual` and the `-promote.dcmsession`
  autosave appear.

  ### Part 1 — remove the arena-override Promote

  - `DrewsChessMachineApp.swift`: delete the `Button("Promote Trainee") { commandHub.promoteCandidate() }`
    line (and its `.disabled(...)`).
  - `AppCommandHub.swift`: delete `var promoteCandidate: () -> Void = {}`.
  - `UpperContentView.swift` `wireMenuCommandHub()`: delete the
    `commandHub.promoteCandidate = { … session.arenaOverrideBox?.promote() }` block.
  - `ArenaOverrideBox.swift`: the box now has exactly one user action — abort.
    Drop the `Decision` enum and the `promote()` method; make it abort-only
    (`requestAbort()` / `isActive` / `consumeAbort() -> Bool`), or rename to
    `ArenaAbortBox`. Update the class doc.
  - `SessionController+Arena.swift`: replace the
    `switch overrideDecision { case .abort / .promote / .none }` with a plain
    `let aborted = overrideBox.consumeAbort()` → `shouldPromote = !aborted && playedGames >= totalGames && score >= threshold`. The arena path's `promotionKind`
    is now always `.automatic` when `shouldPromote`.
  - `PromotionKind.swift`: keep `.manual` — it's now produced only by Part 2.
    Update its doc comment ("user clicked the Promote button" → "user invoked
    Promote Trainee Now").
  - Tests: no XCTest references `ArenaOverrideBox.Decision.promote` /
    `commandHub.promoteCandidate` (checked 2026-05-11); `PromotionKind` is still
    referenced by `ArenaLogFormatterTests` / `TournamentRecordTests` and stays
    valid. Re-grep at implementation time in case that drifts.

  ### Part 2 — add `Engine ▸ Promote Trainee Now`

  - `AppCommandHub.swift`: add `var promoteTrainerNow: () -> Void = {}`. Gating
    reuses the existing published `realTraining` / `isArenaRunning`.
  - `DrewsChessMachineApp.swift`: `Button("Promote Trainee Now") { commandHub.promoteTrainerNow() }`
    `.disabled(!commandHub.realTraining || commandHub.isArenaRunning)` in the
    Engine menu (in the old button's slot).
  - `UpperContentView.swift`: wire `commandHub.promoteTrainerNow = { promoteTrainerNowFromMenu() }`;
    `promoteTrainerNowFromMenu()` flips a `@State` bool that drives a
    `.confirmationDialog` (the view already uses `.confirmationDialog` — follow
    that pattern; do not add a `.alert` ahead of existing `.onChange`/`.onReceive`).
    Confirmation copy must say the weights are **not arena-validated**, e.g.
    title "Promote trainee to champion now?", message "The current trainee has
    not been validated by an arena. Promoting replaces the champion with the
    trainee's current weights.", buttons "Promote" (destructive role) / "Cancel".
    On confirm: `SessionLogger.shared.log("[BUTTON] Promote Trainee Now")` then
    `session.promoteTrainerNow()`.
  - New `SessionController` method `promoteTrainerNow()` — put it in a new file
    `App/SessionController+ManualPromote.swift` (one-File-per-View/feature house
    style), `@MainActor`:
    - Guard `realTraining && !isArenaRunning`; otherwise
      `checkpoint?.setCheckpointStatus("Cannot promote: …", kind: .error)` and
      return (so a stale menu state can't fire it mid-arena).
    - `await activeSelfPlayGate?.pauseAndWait()`; `await activeTrainingGate?.pauseAndWait()`.
    - In a `Task.detached(priority: .userInitiated)` capturing only the needed
      `Sendable` references: `let weights = try await trainer.network.exportWeights(); try await network.loadWeights(weights); return weights`.
      Note vs. the arena path: arena exports from `candidateInferenceNetwork`
      (the arena-*start* snapshot); here we export the **live** trainer weights,
      and we do **not** re-load them into `trainer.network` (it already has them).
      All `try` surfaced via `.value` inside a `do/catch` that records the error
      and still resumes both gates.
    - ModelID: `champion.identifier = trainer.identifier`;
      `trainer.identifier = ModelIDMinter.mintTrainerGeneration(from: champion.identifier ?? trainer.identifier ?? ModelIDMinter.mint())`
      — same rule as arena promotion. **Re-read `sampling-parameters.md` first.**
    - Do **NOT** touch trainer velocity, `trainer.completedTrainSteps`,
      `trainingBox` stats, alarms, or rolling windows: champion is taking the
      trainer's *current* state verbatim, so unlike the arena path there is no
      earlier weight surface to rewind the optimizer/step-count to. Add a comment
      stating this asymmetry explicitly.
    - `activeTrainingGate?.resume(); activeSelfPlayGate?.resume()`.
    - History/chart: append a `TournamentRecord` with `gamesPlayed: 0`,
      `candidateWins/championWins/draws = 0`, `score` = whatever sentinel the
      record/Elo code already tolerates for 0-game records, `promoted: true`,
      `promotionKind: .manual`, `promotedID: champion.identifier`,
      `durationSec: 0`. **Open decision** — confirm `ArenaEloStats` ignores
      0-game records before doing this (it has XCTest coverage; add a case if
      not). If it can't be made safe cheaply, fall back to: no `TournamentRecord`,
      just the log line + a point `ArenaChartEvent` (`recordArenaCompleted` with
      `startElapsedSec == endElapsedSec`, `promoted: true`).
    - `parallelWorkerStatsBox?.resetGameStats()` and a
      `[STATS] post-promote(manual) steps=… champion=… trainer=…` line, mirroring
      the arena path's post-promote logging.
    - Post-promotion autosave: if `Self.autosaveSessionsOnPromote`, reuse the
      arena path's detached `-promote.dcmsession` tail. Extract that tail
      (currently inline at ~685–~720 in `SessionController+Arena.swift`) into a
      shared helper `schedulePostPromotionAutosave(championWeights: [[Float]], championID: String, trainerID: String, creator: String)` so both call sites
      stay in lockstep; pass the just-exported `weights` so it never re-reads a
      live network. (This extraction is the only refactor of existing arena code
      this plan does — the rest of the arena promotion block stays as-is, since
      its velocity/step-rewind semantics genuinely differ.)
  - `sampling-parameters.md`: add a sentence noting that "Promote Trainee Now"
    follows the same champion-inherits-trainer-ID / mint-fresh-trainer-generation
    rule as arena promotion.

  ### Validation

  - Build via drews-xcode-mcp (only when no live training session is running).
  - XCTest suite green — including any `ArenaEloStats` 0-game-record case added
    for the history decision above; no existing test modified.
  - Manual, with Play-and-Train running and the trainer advanced a few hundred
    steps:
    - Invoke `Promote Trainee Now`, confirm. Session log shows
      `[BUTTON] Promote Trainee Now`, the promotion/`[STATS] post-promote(manual)`
      lines, and (if `autosaveSessionsOnPromote`) `[CHECKPOINT] Saved session (post-promotion): …`.
    - Champion `ModelID` == the trainer's `ModelID` from immediately before;
      trainer now has a fresh forked-generation ID; `trainer.completedTrainSteps`
      unchanged across the action.
    - Self-play and training resume normally afterward (no parked workers, ratio
      controller continues).
    - Menu item is disabled when not in Play-and-Train and while an arena is
      running; if forced (stale state) it surfaces an error and does not promote.
    - Load the autosaved `.dcmsession` back; the existing bit-exact forward-pass
      verification on load passes.

- **Composition-aware replay-buffer batch sampler (added 2026-05-12).**
  **✅ COMPLETED 2026-05-12.** Implemented as designed below (recommended v1:
  stratified-rejection + exponential length tilt + per-game tally; no alias
  tables). Deviations from the design text: (1) the constraints live *on*
  `ReplayBuffer` (`SamplingConstraints` struct + `setSamplingConstraints(_:)`),
  pushed from the main actor every UI heartbeat via
  `ReplayBuffer.SamplingConstraints.fromCurrentParameters()` plus once at session
  start — so `ChessTrainer`'s `sample(...)` call sites are unchanged and the
  off-main trainer never reads `TrainingParameters.shared`; (2) no `[SAMPLER]`
  degradation log line for v1 (the draw cap is documented as a ceiling and the
  per-batch `[BATCH-STATS]` line already shows the post-constraint distribution);
  (3) the popover "Replay sampling" section uses heartbeat-driven persistence on
  Save (no per-keystroke live callback) — the buffer picks up the change on the
  next ≤5 s tick. Files: `Training/ReplayBuffer.swift` (composition aggregates +
  `CompositionSnapshot` + `SamplingConstraints` + `sample` rewrite + `solveLengthTiltBeta`),
  `Training/TrainingParameters.swift` (3 new `Replay Buffer`-category `liveTunable`
  Int params, registry 36→39), `App/SessionController.swift` (`bufferComposition`
  mirrored prop), `App/SessionController+Heartbeat.swift` (push + mirror),
  `App/SessionController+Training.swift` (seed at session start; `comp=(…)` on
  `[STATS]`), `App/UpperContentView/UpperContentView.swift` (`Composition:` line),
  `App/UpperContentView/TrainingSettingsPopover.swift` + `…Model.swift` ("Replay
  sampling" section: 3 control rows + composition readout). Tests:
  `DrewsChessMachineTests/ReplayBufferSamplingConstraintsTests.swift` (no-op
  validity + aggregates, exact draw-cap ceiling, exact per-game cap, length-tilt
  tracking + monotonicity, write/restore round-trip of aggregates, under-fill).
  Original plan text retained below for the record.

  Today `ReplayBuffer.sample(count:…)` did `count` independent uniform
  draws over the resident ring — no awareness of which game a position came
  from, the game's length, or its outcome. On the current draw-saturated buffer
  (~85% of resident positions are from drawn games; mean game length ~370 plies;
  the `[STATS]` `avgLen` runs higher because long shuffle-marathons contribute
  one position per ply) that means the minibatch the trainer sees is ~85% z=0
  positions from long aimless games — the value head's W/D/L target then carries
  almost no position information (its honest fit is `p_draw → 1`), the
  policy-gradient baseline is ~0 everywhere, and `advantage = z − vBaseline ≈ z`
  degenerates to a spike at 0. Composition-aware sampling is the most direct
  lever on that degeneracy that does not require the champion to improve first
  (it changes what the *trainer* sees, not what the *champion* generates — the
  buffer's underlying draw rate is still champion-strength × sampling
  temperature, and is not affected by `draw_penalty`, which only re-labels `z`).

  **All three controls are per-training-batch** (enforced inside `sample()`,
  scoped to that one batch — not retention caps on the ring):
  1. **Max positions from any single game per batch** (`K`). Decorrelates the
     minibatch; the standard AlphaZero-ish knob.
  2. **Target mean game length of the sampled positions** (`T`). When populating
     a batch, bias selection so the *expected game length of a sampled position*
     ≈ `T`. Soft target (an exponential tilt, not an exact per-length stratum).
     `T ≥` the buffer's natural mean ⇒ no-op. Intended use is `T` *below* the
     natural mean, to de-weight shuffle-marathons.
  3. **Max % of sampled positions from drawn games per batch** (`D`). A ceiling:
     if the buffer holds fewer than `D%` drawn positions you take what is there
     (no padding); the freed batch slots go to decisive positions. Note (1)/(3)
     overlap heavily — the long games *are* the draws, so the draw cap already
     does most of the length-shortening; `T` is mostly fine-tuning on top.

  **Display (current buffer composition — pre-constraint, distinct from the
  existing per-sampled-batch stats summary, which is post-constraint and must be
  relabelled so the two are not confused):**
  - Mean resident game length, game-weighted = `storedCount / distinctResidentGames`
    ("average positions per game currently in the buffer" — the figure the `K`
    knob should be read against; ≈ true mean game length, modulo front-truncation
    of games straddling the FIFO write head — the un-truncated value is `avgLen`
    on `[STATS]`).
  - Mean game length, position-weighted = `Σ_resident-positions gameLength /
    storedCount` (= `E[L²]/E[L]`; ≥ the game-weighted mean whenever lengths
    vary — the gap between the two *is* the game-length dispersion, so show both).
  - % of resident positions from drawn games (z==0) vs decisive (z≠0), with the
    z=+1 / z=−1 split as a secondary line (a decisive game contributes ~50/50
    of each since `outcome` is relative to side-to-move; a >~55/45 skew is itself
    a red flag — sign-assignment bug or self-play colour imbalance — so it is
    worth surfacing even though it is not a composition lever).

  **Computing the display values.** Read on every UI refresh ⇒ must be O(1) at
  read time ⇒ maintain running aggregates under the buffer lock, updated
  incrementally in `append()` and in the eviction loop — the same hooks
  `hashStats` already uses. Add, alongside `hashStats`: `drawPositions` /
  `winPositions` / `lossPositions` (`+=` per class on append, read
  `outcomeStorage[evictedSlot]` and `-= 1` on eviction); `sumGameLengthOverResidentPositions`
  (Int/UInt64 accumulator; `+= gameLength * count` on append, `-= gameLengthStorage[evictedSlot]`
  on eviction); `residentGames: [UInt32: Int]` refcount map keyed on
  `workerGameId` (bump on append, decrement on eviction, drop entries at 0 —
  `distinctResidentGames = residentGames.count`). The `workerGameId` key has a
  known minor wart: across a *resumed* session the per-worker game index resets,
  so an old resident game can collide with a new one and be merged for
  counting/capping — harmless, not worth fixing now; note it in the code.

  **Populating a batch (recommended v1 — stratified-rejection + length tilt +
  per-game tally; no new index structures, no alias tables; keeps `sample()` at
  O(B + distinctLengths) plus the unchanged O(B · floatsPerBoard) memcpy that
  already dominates).** Constraint priority — (1) draw cap and (2) per-game cap
  are *hard*; (3) length target is *soft* and yields first; never silently
  violate a hard constraint, and log + surface "achieved" numbers (achieved
  draw %, achieved mean length, max per-game count) vs requested.
  - Stratify on outcome for the draw cap (exact): `B_draw = min(round(D% · B),
    drawPositions)`, `B_dec = B − B_draw`. Sample within each stratum by
    rejection on the existing uniform-from-ring draw — accept iff the slot's
    `outcome` matches the stratum, else redraw (~1.18× overdraw for the draw
    stratum at 85% draws; ~6.7× for the decisive stratum, but `B_dec` is small
    and a rejected draw is two array reads + an RNG call).
  - Length target via exponential tilt (soft): accept a slot with probability
    `exp(−β · max(0, gameLength − T))`, else redraw. One global `β` per `sample()`
    call, solved by a 1-D root find on the resident-length histogram so the
    tilted position-weighted mean equals `T` (monotone in β; sub-µs over ~hundreds
    of distinct lengths; `β = 0` when `T ≥` natural mean ⇒ this step is a no-op;
    cache β and re-solve only when the length histogram drifts past a threshold).
  - Per-game cap (exact): a `[UInt32: Int]` scratch tally (or a reused
    open-addressing scratch table) cleared each `sample()` call; after a slot
    passes the stratum + tilt checks, look up `workerGameId`, redraw if that game
    already has `K` picks in this batch. Expected redraws ≈ 0 for `K ≳ 3·B/G`.
  - Accept ⇒ memcpy board/move/outcome (and the optional gameLength/workerGameId
    out-params) into the trainer staging buffers exactly as today. Thread the
    existing RNG so the batch stays deterministic.
  - Degradation: if `K`, `D`, `T` are jointly infeasible for the current buffer,
    relax in order — keep (1) and (2), let (3) drift, log a `[SAMPLER]` line.

  **Upgrade path (only if rejection variance/exactness ever bites — hold in
  reserve):** maintain a per-game index `{ringStart, residentCount, outcomeClass,
  gameLength}` (a game's resident slots are always a contiguous wrapped run since
  both append and eviction are FIFO and a game is one append chunk), plus
  per-stratum Walker alias tables over games with weight `w_g = residentCount_g ·
  e^{−β L_g}`; sample game ∝ `w_g`, uniform offset within it, then the per-game
  tally + redraw only for (1). Makes (3) and the draw cap exact and removes the
  rejection loop, at the cost of maintaining the index + rebuilding/patching the
  alias tables. Much more code; the rejection version is almost certainly fast
  and accurate enough.

  **Open semantics / decisions resolved 2026-05-12:** all three controls are
  per-batch (confirmed — not ring-retention caps). `T` is "adjust the mean game
  length across the sampled positions when populating the batch" (confirmed).
  Still to decide at implementation time: exact parameter names/ranges and which
  `@TrainingParameter` category they land in (likely a new "Sampling" category);
  whether the per-batch stats summary's histograms get a
  "(post-sampling-constraints)" caption vs a separate pre-constraint panel;
  whether to expose `β`/the tilt strength directly as an alternative to
  specifying `T` (probably not — `T` is the intuitive knob, β is derived). These
  only take effect at the next `sample()` call, so mark them `liveTunable` and
  have the trainer re-read from `TrainingParameters.shared` (consistent with the
  existing live params) rather than snapshotting.

  **Edge cases that must be handled:** empty buffer / `storedCount < B`
  (under-fill — preserve today's behaviour, whatever it is); a stratum with zero
  resident positions (e.g. a fresh buffer with no decisive games yet ⇒ `B_dec`
  clamps to 0, log it); `distinctResidentGames == 0` guard on the display
  divisions; `B_draw` rounding so the two strata always sum to exactly `B`.

  **Interactions / validation:** the replay-ratio controller is unaffected (it
  counts positions consumed/produced — composition does not change the count).
  Validation plan: a unit test that builds a synthetic buffer with known
  game/outcome/length structure, draws many batches, and asserts the achieved
  draw % equals the cap (within rounding), no game exceeds `K` in any batch, the
  achieved mean length tracks `T` within tolerance over a range of `T`, and
  behaviour is bit-identical to the current sampler when all three knobs are at
  their no-op settings (`K ≥ B`, `D = 100`, `T` disabled/∞).

- **Stratified replay-buffer sampling by game phase (non-pawn piece count).**
  Added 2026-05-28. The 2026-05-28 buffer analysis of champion
  `20260525-1-sMe9-31` showed the 4 active material-phase buckets are
  unevenly represented (31 / 34 / 19 / 16% across 0–4 / 5–8 / 9–14 / 15–22
  non-pawn pieces), and the network's policy/value heads are weakest exactly
  on the underweighted 15–22 bucket (policy entropy 3.26 nats vs uniform 3.34,
  value-scalar spread collapsed to ±0.09). The proposed fix is a trainer-side
  opt-in toggle that draws each minibatch with balanced weight from the four
  phase buckets, instead of uniformly from the buffer, compensating for the
  natural skew without changing self-play dynamics. See
  `documentation/plans-completed/STRATIFIED_REPLAY_SAMPLING_PLAN.md` for the full plan,
  including the no-op-on-existing-`.dcmsession`-files design (the on-disk
  buffer layout is unchanged; only an in-memory bucket index is added and
  rebuilt from the existing `materialCount` column on restore). V1 ignores
  the `maxDrawPercent` and per-game K-cap while the toggle is on; the UI
  grays those out with a one-line explanation banner. Validation is an A/B
  Play-and-Train pair from the same checkpoint, with success defined as
  15–22 bucket entropy falling ≥0.10 nats and value-scalar spread at least
  doubling, without arena promotion rate falling >25%.

### Backlog migrated from standalone docs (2026-06-13)

Consolidated here when `TODO_NEXT.md` / `NEW_PARAMETERS.md` were retired (their
done/obsolete items live in CHANGELOG; only the still-open ones survive below).

- **Per-tensor gradient-norm readback** (from TODO_NEXT). Today only the global
  `gNorm` and a single `policyHeadWeightNorm` are read back; a per-tensor norm
  breakdown would localize which layers drive clip events. Not built.
- **Weight EMA / SWA for inference** (from TODO_NEXT). Maintain a slow-moving
  average of weights and evaluate the arena/probe champion from it. No
  SWA/EMA-inference code exists.
- **Horizontal-mirror data augmentation** (from TODO_NEXT). Chess is left-right
  symmetric; mirror board+policy at sample time to ~2× effective data. Detailed
  design preserved in git history (TODO_NEXT @ pre-retirement); not built.
- **Cosine (monotone) LR decay post-warmup** (from TODO_NEXT). Distinct from the
  now-shipped *cyclical* LR/momentum (`LRMomentumCycle.swift`); folds into the
  existing "Adaptive learning-rate schedule" item — reconcile against the cyclical
  schedule before pursuing.
- **Candidate training-parameter knobs** (from NEW_PARAMETERS). A parameter-design
  backlog, almost none shipped as proposed: expose the existing hardcoded Dirichlet
  config as `self_play_dirichlet_*` parameters; entropy-bonus schedule/controller;
  replay `sample_mode` (recency / TD-priority / without-replacement) and explicit
  per-outcome weights (NOTE: the composition-aware sampler — per-game cap, draw cap,
  length tilt — already shipped 2026-05-12, so only these remain; TD-priority needs
  real code beyond params because `vBaseline` goes stale); sample-time numeric knobs
  (softmax logit clip, policy top-k cap, min-legal-prob floor); a `legalMassCollapse`
  metric + min-batches guard. Full proposals in git history (NEW_PARAMETERS @
  pre-retirement).

## Code-review remediation roadmap (added 2026-05-11)

Result of an in-depth review of the codebase against Swift / SwiftUI / macOS /
ML best practices. The algorithmic core is in good shape (consistent
`OSAllocatedUnfairLock`/`SyncBox` discipline, MPSGraph `.run` correctly bridged
through `CheckedContinuation`, pure-logic XCTest coverage, no dead search code).
The findings below are ordered by impact. **Part A is the dominant item and is
being executed first** (see the staged plan); Parts B–D are recorded here for
later.

### Part A — decompose `App/UpperContentView/UpperContentView.swift` (10,978 lines)

One `View` struct so large the team had to invent hidden zero-sized "probe"
views (`ControlSideEffectsProbe`, `MenuHubSyncProbe`, `MenuHubSignature`) and
`AnyView` chains *purely to keep the Swift type-checker under its inference
timeout*. It holds a ~2,100-line `startRealTraining(mode:)`, a ~700-line
`runArenaParallel(...)`, ~75 `@State` scratch vars for two popovers, a ~388-line
`body`, and the alarm/checkpoint/auto-resume subsystems inline. This forces
violations of the project's own SwiftUI rules (no `AnyView`; no non-trivial
`var X: some View` outside `body`; one View struct per file) and makes the file
untestable and slow to build.

Target: `UpperContentView.swift` → ~1,500 lines composed of small
`@MainActor @Observable` controllers (`@State`-owned by `UpperContentView`;
`ContentView`/`DrewsChessMachineApp` need no change — verified `LowerContentView`
reads only `chartCoordinator`) + small `View` structs; probe scaffolding and
`AnyView` chains deleted; **zero behavior change** to the self-play→train→arena
loop. Each stage: move state + the methods that mutate it together into a
controller, leave then inline a one-line forwarding shim, build (compile check
via drews-xcode-mcp), commit + push. Stage ordering is load-bearing (Stage 4's
`startRealTraining` move needs Stages 2–3's state already on controllers).

- **Stage 0 — pure formatters out** (≈700 lines, near-zero risk; *first PR*):
  `TrainingStatsTextFormatter.swift` (`trainingStatsText`/`playAndTrainStatsText`/
  `sweepStatsText` + `rjust`/`pct`/`advFmt`), `ArenaTelemetryFormatter.swift`
  (string-building parts of `logArenaResult`/`emitArena*`), `PanelColorizer.swift`
  (`colorizedPanelBody`), `CliConfigOverrideApplier.swift`
  (`applyCliConfigOverrides`/`formatParameterValue`). `UpperContentView` keeps
  one-line forwards.
- **Stage 1 — popover `@Observable` models** (kills 6 `AnyView`, ~165 lines, ~50
  `@State`): `TrainingSettingsPopoverModel.swift` (~17 `*EditText`, ~21 `*Error`,
  4 `original*`, `isPresented`, `seedFromParams`/`cancel`/`save`, live-apply
  closures) + `ArenaSettingsPopoverModel.swift`. `TrainingSettingsPopover.swift`/
  `ArenaSettingsPopover.swift` take `@Bindable model`. Delete the
  `trainingSettingsChip*` `AnyView` chain.
- **Stage 2 — `TrainingAlarmController`** (~150 lines + ~10 `@State`):
  `@MainActor @Observable` owning `active`/`silenced`/streak counters/sound
  `Task`/thresholds; `evaluate(from:)`/`raise`/`clear`/`silence`/`dismiss`. Also
  **adds the missing value-head tanh-saturation alarm** (≈0.97–0.99 on `vAbs`,
  same path as `policyEntropyAlarmThreshold`).
- **Stage 3 — `CheckpointController` + `AutoResumeController`** (kills the
  auto-resume `AnyView`): the checkpoint/segments/save bucket + slow-save
  watchdog + `PeriodicSaveController` ref + the save/load handlers move to
  `CheckpointController`; the auto-resume bucket + countdown + `maybePresentSheet`
  (keeps the headless `XCTestConfigurationFilePath` guard) move to
  `AutoResumeController`; `autoResumeSheetContentView() -> AnyView` becomes a real
  `AutoResumeSheetView` struct. `.fileImporter`/`.fileExporter` stay in `body`.
- **Stage 4 — `SessionController` + `CandidateProbeController` + `SessionHeartbeat`**
  (the big lift; sub-stages 4a–4e). `SessionController` (`App/SessionController.swift`,
  `@MainActor @Observable`) owns the networks / parallel-diversity / arena /
  trainer-run / worker-count buckets + gates and the giant methods moved
  **verbatim** (preserve statement order — accidental reordering could change
  ModelID minting or BN warmup): `buildNetwork`/`ensure*`, `playSingleGame`/
  continuous-play, `trainOnce`/continuous-training, `startTrainingFromMenu`,
  `startRealTraining(mode:)`, `stopRealTraining`, `runArenaParallel` + `emitArena*`
  + `cleanupArenaState`, `buildSelfPlaySchedule`/`buildArenaSchedule`,
  `updateReplayRatioCompensator`, the diagnostics/recovery runners, sweep.
  `CandidateProbeController` (`App/UpperContentView/CandidateProbeController.swift`)
  owns the candidate-probe / on-board-display state shared by the driver and the
  board UI (`playAndTrainBoardMode`, `probeNetworkTarget`, `candidateProbe*`, the
  live `gameWatcher`/`gameSnapshot`, the probe `inferenceResult`) + `fireIfNeeded`
  (takes board state as a param) + the "force candidate-test when workers>1" nudge.
  `SessionHeartbeat` (`App/SessionHeartbeat.swift`, separate `@MainActor` class)
  owns `processSnapshotTimerTick` + the four `refresh*` helpers + in-flight flags +
  memory/usage caches. The interactive forward-pass / board-editing demo stays on
  `UpperContentView`. `commandHub` closures capture the controller *class refs*,
  never the View struct.
- **Stage 5 — delete scaffolding, shrink `body`, extract helper views**:
  dismantle + delete `ControlSideEffectsProbe.swift` (handlers re-homed in Stages
  1/4); delete `MenuHubSyncProbe.swift` if `body` now tolerates a direct
  `.onChange` (keep `MenuHubSignature.swift` — cheap Equatable, recomposed from
  the controllers' fields); extract remaining inline `body` chunks into structs in
  their own files (`UpperTitleBarView`, `InputTensorChannelStrip`, …); move the
  `fileprivate struct`s `UpperCumulativeStatusBar`/`UpperTrainingStatsColumn` to
  their own files; replace `cumulativeStatusBar: UpperCumulativeStatusBar<some View>`
  with a `@ViewBuilder`/struct; audit → zero `AnyView` under `App/UpperContentView/`.
  End state: `body` ~100–150 lines, file ~1,500 lines, no probe views, build never
  hits "expression too complex" (if a `body` change reintroduces it, the subview
  goes into a struct, not a probe).

What stays in `UpperContentView` permanently: `body`; layout `@ViewBuilder`
helpers (allowed in a same-file extension); the AppKit menu bridge
(`wireMenuCommandHub`/`syncMenuCommandHubState`); window-lifecycle hooks; the
view-local interaction state + the forward-pass/board-editing demo; the
controller references.

Risks: the verbatim `startRealTraining` (~2,100 lines) and `runArenaParallel`
(~700 lines) moves are the dangerous ones — guard with line-level diff +
behavioral baseline (`[STATS]`/`[ARENA]` ModelID lines, BN-warmup `[BATCHER]`
probe, loss trajectory per the monitoring rubric, headless `--train --output`
JSON). Don't touch `body` until Stages 1/3/5 or the type-checker may regress.

### Part B — split `Training/ChessTrainer.swift` (4,991 lines)

→ `ChessTrainerGraph.swift` (loss/optimizer graph construction — the advantage/CE/
label-smoothing/entropy ops, the decoupled-weight-decay + grad-clip + Polyak-
momentum update ops) + `ChessTrainerFeeds.swift` (`buildFeeds`/`runPreparedStep`
+ the long-wanted `BatchFeedsInput` named-field struct that closes the
three-same-typed-`UnsafePointer<Float>` call-site hazard already noted under
"Future improvements"). `ChessTrainer` keeps the step driver, `TrainingLiveStatsBox`,
and the public surface. No behavior change. Lower priority than Part A.

### Part C — small Swift / ML hygiene fixes (each a tiny standalone change)

- `Training/TrainingParameters.swift:527,833` — `return try! K.decode(...)` →
  explicit `do { return try ... } catch { preconditionFailure("default for \(K.id)
  does not round-trip: \(error)") }`. (Per the "never `try!` without explicit
  justification" rule.)
- Force unwraps → `guard`/iteration that names the invariant:
  `Network/ChessMPSNetwork.swift:134` `legal.randomElement()!`,
  `Encoding/BoardEncoder.swift:325,336` `PieceType(rawValue:)!`,
  `Chess/MoveGenerator.swift:73` `board[fromIndex]!`,
  `Training/ReplayBuffer.swift:578,584,589,595` `d[$0]!` in `jsonLine()`.
- `Network/ChessNetwork.swift:123` `policyHeadFinalWeights: MPSGraphTensor!` IUO →
  return-and-store from `policyHead(...)`, or add a `// SAFETY:` justification.
- Stale optimizer docs: `Training/ChessTrainer.swift:990` "plain SGD (no momentum,
  no Adam state)" is false since Polyak-momentum velocity buffers were added —
  reword to "SGD with Polyak momentum + decoupled (AdamW-style) weight decay; μ=0
  reduces bit-exact to plain SGD". `Training/TrainingParameters.swift:240,260`
  "Adam optimizer learning rate" / "Standard practice for Adam" — reword to
  SGD-with-momentum (the LR/batch-scaling rule is shared, but it isn't Adam).
- Module-wide `AnyView` audit → zero (most die in Part A Stages 1 and 3).
- `try? await Task.sleep(...)` (≈27 sites) swallows `CancellationError` — behavior
  is fine where a `while !Task.isCancelled` check follows, but standardize on a
  tiny helper or add a one-line per-site justification.
- Value-head tanh-saturation alarm (also done in Part A Stage 2).
- Investigate the wasted ~80 MB/step GPU→CPU policy readback in the trainer's
  fresh-baseline pass (it appears to use only the value output) → add
  `needsPolicy: Bool = true` to `evaluate(batchBoards:count:)` that skips the
  policy head + readback when false. (Do *not* touch the dense per-step legal-mask
  allocation `[batch, 4864]` without a rework — it's required by the masked-CE loss
  design; only flag if it shows up as a profiler hot spot.)
- `App/UpperContentView/UpperContentView.swift` (≈lines 1340–1404) — two `.sheet`
  modifiers (`autoResume.sheetShowing`, `showArenaHistorySheet`) sit *above* the
  `.onReceive(NSWindow.willCloseNotification)` / `.onChange(menuHubSignature)` /
  `.onReceive(snapshotTimer)` / `.onChange(realTraining)` chain, which violates the
  repo's ".sheet/.alert after .onReceive/.onChange unless intentionally scoped to
  the sheet content" rule. Pre-existing (the same ordering predates the Part A
  decomposition — it was moved verbatim) and harmless today because both sheets are
  `isPresented:`-driven and the onChange/onReceive handlers are app-wide signals
  unrelated to sheet content; reorder for hygiene the next time that modifier chain
  is touched.

### Part D — repo hygiene (non-code, light-touch)

Fold the live scratch markdown (`CHECK_NEXT.md`, `TODO_NEXT.md`, `ML_REVIEW_NOTES.md`,
`ROADMAP_NOTES.md`, `NEW_PARAMETERS.md`, `CONCURRENCY_CONCERNS.MD`, `CAPTURE_MOVE_MASK.md`)
into `ROADMAP.md`/`CHANGELOG.md`; remove the 0-byte `default.profraw`; move bulky
experiment artifacts (`results.json` ~6.8 MB, `experiment_results.js` ~921 KB, the
508-entry `experiments/` tree) out of the published tree. **Never edit `.gitignore`.**

## Completed / corrected from older Future entries

- **Model and session save/load — implemented, with scope expanded beyond the
  original future plan.** The old Future entry said "Today nothing persists
  across app launches — quit mid-training and you lose the champion, the trainer,
  every accumulated counter, and the replay buffer." That statement is now
  historical, not current. The original design context is preserved below, with
  corrections against current code.

  **Single model — `.dcmmodel` (flat binary file), implemented.** The original
  plan was: wrap one network's weights plus identity and metadata in a fixed
  binary header, then the tensors from `ChessMPSNetwork.exportWeights()` /
  `ChessNetwork.exportWeights()` in declared order, then a trailing 32-byte
  SHA-256 over all preceding bytes for integrity. Header carries magic
  `"DCMMODEL"`, format version, `archHash` (hash of filters / blocks / input
  channels / policy dim — hard-refuses to load on mismatch, no migration),
  `numTensors` sanity-check, creation wall-clock time, `ModelID`, parent
  `ModelID` at time of save, and a JSON metadata blob (arena stats at mint,
  training-step count, creator tag). Loadable into any training- or
  inference-mode `ChessNetwork` via the existing `loadWeights` path — this is
  the unit for "take any model at any point and use for inference."

  Current code evidence: `ModelCheckpointFile.swift` implements `.dcmmodel`
  encode/decode with magic/version/arch hash, tensor count, metadata, weights,
  and SHA-256 validation. `ChessNetwork.exportWeights()` returns the current
  trainable variables plus BN running stats in declared order; current tensor
  count is validated dynamically by `loadWeights` rather than being hard-coded in
  ROADMAP. `ChessNetwork.loadWeights(_:)` checks tensor count and each tensor's
  element count before assigning through prebuilt load placeholders.

  **Training session — `.dcmsession` (directory), implemented and expanded.**
  The original plan was a directory holding `champion.dcmmodel`,
  `trainer.dcmmodel`, and `session.json`, rather than a custom bundle, so (a)
  extraction is free — Finder-copy any model out of a session — and (b) only one
  binary model format needs debugging. `session.json` was planned as a Codable
  blob with stable `sessionID`, format version, save and session-start
  wall-clock timestamps, accumulated training time, STATS-line counters
  (`trainingSteps`, `selfPlayGames`, `selfPlayMoves`, `trainingPositionsSeen`),
  hyperparameters appearing in the arena footer (batch, lr, promote threshold,
  arena games, self-play/arena tau configs, self-play worker count), both
  network IDs duplicated from `.dcmmodel` headers for fast index reads, and light
  arena history (W/L/D + kept/promoted + step-at-run for each arena so far).

  Current code evidence: `SessionCheckpointState` now contains all of the above
  and more: game-result breakdown, replay-ratio settings, step delay / last
  auto-computed delay, LR warmup, sqrt-batch LR scaling, replay-buffer min
  positions, arena auto interval, candidate probe interval, legal-mass collapse
  thresholds, build metadata, replay-buffer presence/counters, training segments,
  arena concurrency, and expanded arena side-breakdown fields. `CheckpointManager`
  writes `champion.dcmmodel`, `trainer.dcmmodel`, `session.json`, and, when
  requested, `replay_buffer.bin`.

  **Important correction to the original v1 exclusions.** The original plan
  excluded the 500k-position replay buffer (~2.3 GB / later noted as 4.6 GB)
  because resume warmup/refill was considered acceptable, excluded the candidate
  network because it only exists mid-arena, and excluded in-flight self-play
  games because workers abandon on save like Stop. Current code no longer
  excludes replay-buffer contents when `state.hasReplayBuffer == true` and a
  `ReplayBuffer` is passed: `CheckpointManager.saveSession` writes
  `replay_buffer.bin`, updates `session.json` replay counters from the exact
  `ReplayBuffer.write(to:)` snapshot, and verifies by restoring into a scratch
  buffer. Candidate network and in-flight games remain excluded from session
  state.

  **Save triggers, implemented with naming updates.** Original plan: Menu items
  Save Session, Save Champion as Model, Load Session, Load Model; autosave on
  arena promotion defaults on; Save Session disabled mid-arena; Load Session and
  Load Model require Play-and-Train to be stopped. Current File menu implements
  Save Session, Save Champion, Load Session, Load Model, Load Parameters, Save
  Parameters, Resume Training from Autosave, and Open Data Folder in Finder.
  Save Session is disabled unless real training is active and no arena/save is
  running. Load Session/Model are disabled during real training, continuous play,
  continuous training, sweep, game play, build, or save-in-flight. Post-promotion
  autosave is enabled by `autosaveSessionsOnPromote = true`, and periodic
  4-hour autosave is also implemented.

  **File locations, implemented with corrected session naming.** Original plan:
  all saves — manual and auto — land under fixed Library paths:
  `~/Library/Application Support/DrewsChessMachine/Sessions/` for sessions and
  `~/Library/Application Support/DrewsChessMachine/Models/` for single models.
  Every save keeps the old file; nothing is overwritten; users prune manually.
  The planned naming scheme was `<YYYYMMDD-HHMMSS>-<modelID>-<trigger>.<ext>`
  where trigger is `manual` or `promote`; wall-clock prefix gives natural Finder
  sort order; a reveal/open button makes the hidden `Application Support`
  location discoverable; load uses a file importer so files can be loaded from
  anywhere.

  Current code evidence: `CheckpointPaths.rootURL`, `sessionsDir`, and
  `modelsDir` implement the Library paths. `CheckpointPaths.makeFilename` uses
  `<timestamp>-<modelID>-<trigger>.<ext>` for standalone models.
  `CheckpointPaths.makeSessionDirectoryName` uses
  `<timestamp>-<sessionID>-<trigger>.dcmsession` for sessions, so multiple
  autosaves for the same run cluster by stable session ID rather than by a fresh
  model ID. `CheckpointManager` refuses overwrites with target-exists guards.
  The UI command is currently named "Open Data Folder in Finder" rather than the
  originally proposed "Reveal Saves in Finder".

  **Every save is self-verified before it is marked successful — implemented and
  hardened.** Original plan: after atomic writing (tmp + fsync + rename), re-read
  the file(s), bit-compare tensors against exported `[[Float]]`, load weights
  into a throwaway `ChessMPSNetwork`, run forward pass on canonical test
  positions (starting position + one fixed mid-game FEN), and compare policy and
  value outputs bit-exactly to the source network. Any mismatch deletes the fresh
  `.tmp`, leaves prior saves untouched, and surfaces a user-visible error.

  Current code evidence: `CheckpointManager.saveModel` and `saveSession` perform
  model verification via `verifyModelFile`, session JSON decode round-trip,
  replay-buffer scratch restore/counter comparison when a replay buffer is
  present, `F_FULLFSYNC` on files/directories, tmp staging, atomic final rename,
  parent-directory sync, and launch-time orphan cleanup of interrupted `.tmp`
  artifacts. See the existing Completed entry "Session durability hardening —
  saved means golden" for the full durability pipeline.

  **Original validation checklist, status after implementation.**
  (1) Build succeeds — covered by current project/test workflow, not rerun by
  this documentation-only roadmap edit. (2) Round-trip a single model: Save
  Champion → quit → relaunch → Load Model → run Forward Pass on a fixed FEN →
  policy/value bit-exact to pre-save — supported by model encode/decode,
  `loadWeights`, and save verification. (3) Round-trip a session:
  Play-and-Train → Save Session → quit → relaunch → Load Session → counters and
  ModelIDs match → champion/trainer forward outputs bit-exact → Play-and-Train
  resumes and later arena can promote — supported by session state restore and
  model verification; replay buffer now can restore rather than always refill.
  (4) Arch-mismatch file refuses to load with a clear error — implemented via
  `.dcmmodel` arch hash. (5) Truncated `.dcmmodel` refuses to load. (6) SHA
  mismatch in `.dcmmodel` refuses to load. (7) Save-mid-arena is disabled or
  errors clearly — Save Session menu is disabled while `isArenaRunning`. (8)
  Save atomicity under `SIGKILL` while writing `.tmp` leaves prior saves intact
  and launch cleanup removes orphans — implemented via tmp staging, no-overwrite
  final rename, fsyncs, and `CheckpointPaths.cleanupOrphans()`. (9) Existing
  tests should still pass — not rerun for this roadmap-only task.

  **Session restore coverage — original table corrected against current code.**

  | Field | Save | Restore / current status |
  |---|---|---|
  | Champion + trainer weights | `.dcmmodel` files | loaded into networks |
  | Champion + trainer model IDs | `session.json` | restored to identifiers |
  | Session ID | `session.json` | inherited for continuity |
  | Elapsed training time | `session.json` | back-dated `sessionStart` anchor / training segments now add more context |
  | Training step count | `session.json` | seeded into stats/trainer state |
  | Self-play games / moves | `session.json` | seeded into `ParallelWorkerStatsBox` / display state |
  | Game results (W/B checkmates, stalemate, 50-move, 3-fold, insuff. material) | `session.json` Optional fields | restored when present, back-compatible when absent |
  | Learning rate | `session.json` | restored to `TrainingParameters` + trainer |
  | LR warmup + sqrt-batch LR scaling | `session.json` Optional fields | restored when present; this is newer than original table |
  | Replay ratio target + auto-adjust toggle | `session.json` Optional fields | restored to live parameters/controller state when present |
  | Step delay + last auto-computed delay | `session.json` Optional fields | restored when present |
  | Self-play worker count | `session.json` | restored/clamped to runtime worker bounds |
  | Arena concurrency | `session.json` Optional field | restored/clamped; newer than original table |
  | Arena/candidate/legal-mass tuning fields | `session.json` Optional fields | restored when present; newer than original table |
  | Build metadata | `session.json` Optional fields | displayed/used for forensic context; newer than original table |
  | Training segments | `session.json` Optional array | restored/summed for active-training-time history; newer than original table |
  | Arena history (W/L/D, score, promoted flag per arena) | `session.json` | rebuilt into `tournamentHistory`; side breakdown fields are optional/back-compatible |
  | Replay buffer contents | `replay_buffer.bin` when `hasReplayBuffer == true` | restored via `ReplayBuffer.restore(from:)` and cross-checked against `session.json`; older sessions without a buffer still resume by refilling |
  | Progress rate chart samples | not saved as the original table said | rebuilds from new data |
  | Rolling loss windows | not saved as the original table said | rebuilds from new steps |

- **Legal-move masking in the training policy loss — implemented for training,
  not for inference.** The old Future item "Fuse legal-move masking into the
  policy head" said the graph emitted a full policy and the CPU masked illegal
  moves. That is no longer fully accurate. `ChessTrainer.buildTrainingOps` now
  creates a `legal_move_mask` placeholder, builds `masked_logits =
  network.policyOutput + (1 - legalMask) * -1e9`, and feeds `maskedLogits` to
  `graph.softMaxCrossEntropy(...)`. `ChessTrainer.buildFeeds` writes the
  `legalMasks` pointer into a cached `legalMaskND` and includes
  `legalMaskPlaceholder` in the feeds dictionary. Thus the training loss's
  softmax is graph-masked.

  Inference remains intentionally unmasked at the network boundary:
  `ChessNetwork.evaluate` returns raw 4864 logits, `ChessRunner` softmaxes for
  the Forward Pass demo, and `MPSChessPlayer` samples over legal moves using the
  move list it is given. Keeping raw logits visible preserves diagnostics such
  as illegal-mass/top-cell collapse; do not describe current training as using a
  CPU-renormalized policy loss.

## Decisions not pursued / historical notes

- **Inference-side graph legal-mask softmax is not currently being pursued.**
  Because training now masks logits in-graph and inference diagnostics benefit
  from seeing illegal raw logits, the remaining version of this idea is only an
  inference-path optimization/design change: add a legal-mask feed to inference
  and return already-normalized legal probabilities. That would hide illegal-mass
  telemetry unless a raw-logit path were retained. Keep the current raw-logit
  inference contract unless a measured hot path needs graph-side inference
  masking.

- **Partial heap / quickselect for top-k policy moves is not worth changing now.**
  The original text cited a 4096-entry policy vector. Current architecture v2
  has `ChessNetwork.policySize = 76 * 8 * 8 = 4864`, and
  `ChessRunner.extractTopMoves` full-sorts the policy indices. That full sort is
  intentional after the catastrophic-collapse fix: sorting the whole vector
  guarantees enough on-board decoded moves even when the top cells are off-board.
  The path is the Forward Pass / Candidate Test UI path, not the self-play hot
  path. Revisit a heap/quickselect only if top-k extraction moves into a per-ply
  search/hot loop, and preserve the full-vector/off-board robustness.

- **Old per-worker-network self-play topology is historical.** The existing
  Completed section preserves the original N-worker design in detail, but current
  runtime uses a shared `BatchedMoveEvaluationSource` rather than
  `secondarySelfPlayNetworks`. Treat that section as context, not current
  architecture.

## Tech debt / migrations to remove

- **UNTESTED: the `--parameters` round-trip has never been empirically
  verified** *(added 2026-08-12, from the 2026-08-07 CLI audit)*. The chain
  `--create-parameters-file` → hand-edit the JSON → `--parameters <file>` is
  the documented way to configure a headless run, and every part of it is
  believed correct — `--show-default-parameters` emits all 62 keys by
  iterating `allKeys`/`definition` (macro-driven, so it cannot silently miss
  one), and unknown keys `throw unknownParameter` at apply time, so typos are
  loud. But **no test or manual run has ever confirmed that a value written by
  the generator, edited, and fed back is the value the trainer actually
  trains under.** Everything about it is inference from reading the code.
  - Known gap it would expose: `CliTrainingConfig.load` bridges only
    bool/int/double, so a string-valued parameter would throw `wrongType`.
    Latent — no parameter is string-typed today, but adding one silently
    breaks `--parameters` for it.
  - Worth a single integration test: generate → mutate a couple of numeric
    values → apply → assert `TrainingParameters.shared.snapshot()` reflects
    exactly those values and nothing else moved. Cheap, and it pins the one
    interface every headless experiment depends on.

- **KNOWN DEFICIENCY: weight I/O cannot detect a same-shape position swap**
  *(added 2026-08-12, commit 217aa4d)*. All weight transfer — in-memory
  (`exportWeights()` → `loadWeights()`) and on disk — pairs values to meaning
  purely by **index** against `NetworkArchitecture.weightTensorPlan()`. The
  plan is the sole authority for what the tensor at each position is *called*
  and which layout transform it gets: `SafetensorsModelIO.encode` labels
  `weights[i]` with `plan[i].name` and reshapes per `plan[i].kind`, and
  `loadWeights` assigns positionally.
  - **Hardened (done):** `ChessNetwork.init` validates its weight variables
    against the plan and throws `weightPlanMismatch`; `SafetensorsModelIO.decode`
    checks each stored tensor's dimensions against the plan's torch-layout
    shape (it previously discarded the shape and checked only element count,
    so a `.linear` written `[in, out]` instead of `[out, in]` passed and was
    silently scrambled by `fromTorchLayout`). Both compare **squeezed** shapes:
    the plan records logical shapes (`[C]`) while the builder declares
    broadcast-ready ones (`[1, C, 1, 1]`) — 132 positions differ that way in
    the current preset with identical element counts, so raw comparison would
    reject every correct network, and element count alone cannot tell
    `[in, out]` from `[out, in]`.
  - **Still deficient:** two tensors of the **same squeezed shape** swapping
    positions is undetectable. A BN `weight`/`bias` pair is `[C]` either way,
    as is `running_mean`/`running_var` — and they are adjacent in both the plan
    and the builder's append order, so a refactor that reorders them inside
    `batchNorm()` would mislabel every model written from that point and
    mis-assign every load, with no error raised anywhere.
  - **Why it is not closed:** the two sides use different naming schemes —
    builder `block0_bn1_gamma` vs plan `blocks.0.bn1.weight` — so names cannot
    be compared directly. Closing it means giving the builder the plan's
    canonical names, which breaks `ValueHeadAnalyzer` (matches
    `hasPrefix("value_")`, exact-matches `value_fc2_bias`) and
    `NetworkWeightAnalyzer` (`section(forVariableNamed:)`). That analyzer
    migration is the actual cost, not the rename.
  - **Mitigating (not a reason to skip it):** a real gamma/beta or mean/var
    swap produces a grossly broken network rather than a subtle regression, so
    it would show up in behavior immediately. The value of closing it is
    turning "mysteriously broken model, days of debugging" into an explicit
    error — which is precisely what the 2026-08-07 −280 Elo arena hunt cost.
  - Guards and their limits are documented in
    `ChessNetwork.validateAgainstPlan` and at the `decode` dims check.
  - Provenance: came out of the 2026-08-07 self-play/dropout audit. Two of
    that audit's claims about this area were wrong and are worth not
    re-deriving: the safetensors disk path was described as "loads by NAME and
    is safe" (it resolves *by* name but pairs values to the plan **by index**
    at both ends, which is why the contract matters for every saved model),
    and `--parameters` was reported as leaking into `UserDefaults` (it does
    not — `TrainingParameters.persist` returns early on `suppressPersistence`,
    which every apply site sets; that leak was real once and was fixed on
    2026-06-12 after the dropout=0.7 contamination).

- **Fully decouple training from the main thread / UI run loop** *(added
  2026-05-31)*. The training pipeline is supposed to be almost completely
  separated from the UI — emitting telemetry the user can read or ignore
  — but several main-thread couplings let UI events silently stall it.
  The concrete trigger: every analysis result dialog used a blocking
  `NSAlert.runModal()`, which owns the main thread until dismissed. Both
  the trainer step loop (`SessionController+Training.swift` —
  `await fireCandidateProbeIfNeeded()` and
  `await TrainingParameters.shared.snapshot()` per step) and the
  self-play driver (`BatchedSelfPlayDriver` — per-cycle
  `await MainActor.run { TrainingParameters.shared… }`) hop to the
  MainActor every iteration, so a left-open modal starved the MainActor
  job queue and halted ALL training. Observed in the field: a "Run All
  Analyses" result dialog left open ~4 hours advanced the trainer by 11
  steps total (`dcm_log_20260531-020125.txt`, steps 49632→49643 over
  09:25→13:39), resuming the instant it was dismissed.
  - **Done (point fix):** all analysis/result `NSAlert.runModal()` sites
    now route through `Utils/NonBlockingAlert.swift`, which presents a
    window-attached sheet via `beginSheetModal(for:)` and returns
    immediately. File-picker `NSOpenPanel.runModal()` sites
    (`LichessProbeComparisonLoader`) were deliberately left blocking —
    they only block while the user is actively interacting, never
    walked-away.
  - **Remaining refactor (deferred):** the deeper issue is that the
    training loops *read live-tunable parameters by hopping to the
    MainActor every iteration*. Any future main-thread stall (a modal we
    forget to make non-blocking, a long synchronous UI redraw, a
    spinning event-tracking loop) can therefore throttle training. The
    durable fix is to remove the per-iteration MainActor dependency:
    have the live-tunable parameters pushed into a `Sendable`,
    lock-protected snapshot box (`SyncBox`-style) that the loops read
    without touching the MainActor, updated on the existing reconcile
    cadence. Then training progress is structurally independent of the
    main run loop. Also worth a lightweight guard: a check (test or
    lint) that no new `NSAlert.runModal()` / blocking modal creeps into
    the training-adjacent code paths.

- **Drop v1 trainer.dcmmodel zero-pad migration** *(added 2026-05-04;
  remove after 2026-06-04)*. Trainer state persistence (Polyak momentum
  velocity) was added with `ModelCheckpointFile` format version 2,
  bumping from v1 (trainables + bn) to v2 (trainables + bn + velocity).
  The decoder accepts both versions; the trainer's
  `loadTrainerWeights(_:)` count-detects v1 files and leaves velocity
  at zero-init. After 2026-06-04, any in-flight v1 trainer.dcmmodel
  files should have been re-saved as v2 (a single Save Session
  re-emits with the new format), so the v1 acceptance branch can be
  removed:
  - Tighten `ModelCheckpointFile.supportedReadVersions` to `[2]` only.
  - Remove the `weights.count == v1Count` branch in
    `ChessTrainer.loadTrainerWeights(_:)`.
  - Remove the `// TODO(persist-velocity, after 2026-06-04)` marker
    comments in both files.

## Findings

- **fp16 (float16) is inference-only on the macOS-27-beta stack (2026-06-17).**
  fp16 forward + safetensors round-trip are sound and tested, but a real fp16
  `trainStep` diverges to a NaN gradient on the first step at every batch (worse
  than bf16, whose batch-1 is finite): forward CE components stay finite while
  the aggregate loss and gradient norm go NaN, so the overflow is in the fp16
  backward / auxiliary loss terms, not the fp32-accumulated CE. fp16's exponent
  range is far narrower than bf16/fp32. Viable fp16 training would need loss
  scaling and/or fp32 computation of the loss + aux terms — deferred. Captured by
  the two known-failing `FP16ComputePathTests` trainStep cells; full MPSGraph
  beta-stack writeup in `documentation/macos27-beta1-mpsgraph-findings.md`.

- **Batch-size sweep is reliable at 1 s per batch size.** The Batch Size
  Sweep panel runs a training-mode copy of the network through real SGD
  steps at each batch size and reports steady-state throughput. We tried
  longer per-size windows (15 s, 5 s, 3 s, 1.5 s) and found 1 s gives
  essentially the same shape and the same winner — the fast-warming MPSGraph
  caches mean each row converges within a handful of steps and the tail just
  accumulates redundant samples. Keeping it at 1 s makes the whole sweep
  cheap enough to run any time on a new machine to pick the most efficient
  batch size for *that* hardware, rather than baking a single number in.

- **Sweep memory guard is empirical, not architectural.** The sweep refuses
  to run a batch size whose predicted resident footprint exceeds 75 % of
  `min(recommendedMaxWorkingSetSize, maxBufferLength)`, or whose largest
  single buffer would exceed `maxBufferLength`. The prediction comes from
  a least-squares linear fit over the (batch, peak `phys_footprint`) pairs
  already observed during the same sweep — no per-architecture fudge
  factor. Peak `phys_footprint` is sampled by the UI heartbeat (~10 Hz)
  plus once at the start and end of each row, so we catch transient spikes
  during a step rather than relying on `MTLDevice.currentAllocatedSize`,
  which is post-step and undercounts. Skipped rows still appear in the
  table with the prediction and the reason they were skipped, so the
  sweep walks the full ladder and makes its limits visible.

- **First decisive arena promotion under the autotrain loop
  (2026-04-30, `experiments/20260430-170725/`, accepted as commit
  `42c35c9`).** A 2400 s Play-and-Train run produced one promotion at
  arena #3 of 4. Worth preserving in detail because it's the first
  arena result during automated parameter tuning where the candidate
  was clearly stronger than the champion rather than a coin-flip
  hovering around 0.50. Build 403, champion `20260430-53-gNPD`,
  candidate `20260430-53-gNPD-1` (promoted), trainer
  `20260430-53-gNPD-2`.

  Training/arena parameters in effect for this run:

  | Parameter                                          | Value     |
  |----------------------------------------------------|-----------|
  | `learning_rate`                                    | 5e-05     |
  | `lr_warmup_steps`                                  | 30        |
  | `K` (policy loss scale)                            | 5         |
  | `entropy_bonus`                                    | 0.016     |
  | `weight_decay`                                     | 2e-04     |
  | `grad_clip_max_norm`                               | 25        |
  | `draw_penalty`                                     | 0.1       |
  | `training_batch_size`                              | 4096      |
  | `self_play_workers`                                | 48        |
  | `replay_ratio_target` (auto-adjust on)             | 1.1       |
  | `replay_buffer_capacity`                           | 500 000   |
  | `replay_buffer_min_positions_before_training`      | 75 000    |
  | `self_play_start_tau` → `target_tau` / decay/ply   | 2.0 → 0.8 / 0.03 |
  | `arena_start_tau` → `target_tau` / decay/ply       | 2.0 → 0.5 / 0.01 |
  | `arena_promote_threshold`                          | 0.55      |
  | `arena_games_per_tournament`                       | 100       |
  | `arena_auto_interval_sec`                          | 300       |
  | `candidate_probe_interval_sec`                     | 15        |
  | `legal_mass_collapse_threshold` / grace / probes   | 0.999 / 600 s / 8 |
  | `training_time_limit` (this run window)            | 2400 s, 1427 trainer steps |

  Per-arena results (each tournament = 100 games, 50 as White +
  50 as Black; "W-D-L" is candidate-relative):

  | # | Finished @ step | W-D-L (cand) | White (W-D-L) | Black (W-D-L) | Score | Score CI95     | Elo | Elo CI95     | Promoted |
  |---|-----------------|--------------|---------------|---------------|-------|----------------|-----|--------------|----------|
  | 1 | 179             | 7-85-8       | 5-39-6        | 2-46-2        | 0.495 | [0.457, 0.533] | −3  | [−30, +23]   |          |
  | 2 | 528             | 10-83-7      | 6-38-6        | 4-45-1        | 0.515 | [0.475, 0.555] | +10 | [−18, +39]   |          |
  | 3 | 866             | 19-76-5      | 8-40-2        | 11-36-3       | 0.570 | [0.524, 0.616] | +49 | [+17, +82]   | ✅       |
  | 4 | 1175            | 6-85-9       | 3-43-4        | 3-42-5        | 0.485 | [0.447, 0.523] | −10 | [−37, +16]   |          |

  Score / Elo confidence intervals are the Wald 95% CI computed in
  `ArenaEloStats.summary` from per-game outcomes in {1, 0.5, 0}; Elo
  CI is the score CI mapped through `400·log10(p/(1−p))`. Promotion is
  gated on the point estimate vs `arena_promote_threshold`, not on the
  CI.

  Why arena #3 is decisive rather than borderline:

  - 19 wins vs 5 losses (24 decisive games; candidate took 79 % of them).
  - Score 0.570 with CI95 [0.524, 0.616] — the entire CI sits above
    0.50; the lower bound dips just under the 0.55 promote line but the
    point estimate clears it cleanly.
  - Elo +49 with CI95 [+17, +82] — even the lower bound is +17 Elo, so
    "candidate is genuinely stronger" is well-supported, not noise.
  - Balanced across colors (8 wins as White, 11 as Black) rather than
    one-sided color luck.

  The surrounding arenas (#1, #2, #4) all sit inside [0.485, 0.515] with
  CIs straddling 0.50 by a wide margin — typical noise-floor draws when
  two near-equivalent networks face off (draw rates 76–85 %). Arena #3
  is cleanly separated from that floor. Useful as a reference point for
  what a real training-driven promotion looks like in this engine, vs
  the borderline 0.50–0.53 promotions seen earlier in the project's
  history (e.g. the 5-arena run at scores `[0.51, 0.525, 0.515, 0.52,
  0.5075]` from the 2026-04-21 BN-warmup CHANGELOG entry, which the
  team correctly diagnosed as a stuck network rather than real
  progress).

  Followup pure-window-extension run (2700 s,
  `experiments/20260430-184042/`, accepted as commit `be9d2d3`)
  produced 0 promotions across 5 arenas (scores 0.51 / 0.535 / 0.53 /
  0.47 / 0.485) but dramatically healthier end-of-run policy state
  (max prob 0.150 vs baseline 0.998, illegal_mass 0.678 vs 1.000,
  pEnt 6.44 well above the 5.0 alarm threshold) — the autotrain goal
  axis ("longer training without full collapse") favored the longer
  window despite no promotion, on the principle that a healthy
  policy-head distribution is a prerequisite for future promotions.

## Completed

- **Fast legal-move generation: make/unmake + pin-based (2026-06-24, `81ac884`,
  `d4becf6`).** Move-generation legality was a copy-make filter — for each
  pseudo-legal move it allocated a whole new `GameState` (board-array COW) just
  to test whether the mover's king was attacked, ~30 per position. Profiling an
  import put 93% of CPU in `MoveGenerator.legalMoves`, roughly half of it in that
  per-candidate allocation. Two faster generators were added, each required to
  return the **identical** move set: `legalMovesMakeUnmake` (B) applies each move
  in place on one board buffer, tests the king square, and unmakes — no
  per-candidate allocation (~3× on perft); `legalMovesPinBased` (C) computes the
  check status and a `UInt64` absolute-pin bitboard once from the king's rays,
  then fast-paths any move that is not-in-check ∧ not-a-king-move ∧
  not-en-passant ∧ not-from-a-pinned-square with no apply at all, verifying only
  the residual via the shared make/unmake primitive (~6× on perft).
  `isSquareAttacked` gained an `UnsafeBufferPointer` overload so the legality
  test needs no `GameState` wrapper. Production `legalMoves` now calls
  `legalMovesPinBased`; `legalMovesCopyMake` is retained as the perft and
  cross-check reference. All three generators are pure (no shared mutable state),
  so they run unchanged across concurrent self-play workers. The pin fast-path is
  sound because the only ways a non-king move can newly expose its own king are
  vacating a king ray (absolute pin) or an en-passant rank discovery, both
  excluded. Validation: a `PerftTests` suite checks all three against published
  node counts for six standard positions (incl. Kiwipete) and runs a per-node
  differential of B and C against copy-make (~200K positions, zero divergence);
  the `--crosscheck-movegen` flag makes `legalMoves` additionally run all three
  per call and log any divergence (FEN + move diff) to stderr/SessionLogger,
  soak-tested over ~8K real self-play positions (zero divergence); plus an
  independent code review (no Critical findings).

- **Parallel PGN importer; replay-by-legality; fsync removal (2026-06-24,
  `81ac884`, `d4becf6`).** The PGN→corpus importer (`--import-pgn`) was a single
  serial loop bottlenecked on per-ply move generation; it is now a worker pool
  (`activeProcessorCount − 2`) that parses, replays, and encodes framed records
  off-thread, feeding a single serial writer that drains a per-sequence reorder
  buffer so the corpus preserves original file order, with `DispatchSemaphore`
  backpressure and a `DispatchGroup` barrier. Output is deterministic and
  independent of worker count. Each SAN token is resolved against the
  *pseudo-legal* moves with legality checked only on the match (avoiding a full
  legal-move generation per ply), and resolution rejects an ambiguous SAN
  (more than one legal match) rather than guessing. Replay is by legality only,
  so a game legally played through a claimable-but-unclaimed threefold-repetition
  or fifty-move draw imports correctly — the engine's self-play auto-termination
  is not applied to recorded games; the outcome comes from the `Result` tag.
  Hard-fail by default on the first unparseable game (`--lenient` counts and
  skips instead) and on a nonzero reader/decompressor exit (missing file, missing
  `zstd`, corrupt `.zst`) so a bad input fails loudly instead of producing a
  silent empty corpus. The shard writer's per-record fsync cadence (described in
  the corpus entry above) was removed as unnecessary for a re-runnable import —
  one `synchronize()` at seal remains; crash recovery still relies on the
  per-record CRC-32 and the sealed-shard SHA-256. New flags: `--max-storage
  <size>`, `--import-threads <n>`, `--lenient`. `FENParser` gained `GameState →
  FEN` encoding (inverse of `parse`) for reproducible divergence logging. Tests
  in `PGNImporterTests` (order-preservation, thread-count determinism, hard-fail,
  lenient, exact max-games cap, threefold replay, ambiguous-SAN rejection,
  missing-file hard-fail), `PerftTests`, and `FENParserTests`.

- **fp16 (float16) selectable compute precision — inference (2026-06-17,
  `b25f37e`).** `ComputeDataType` gains `.float16` beside `.float32` / `.bFloat16`,
  selectable in Build-New-Model and embedded in safetensors metadata. Closed the
  four remaining touchpoints over the pre-existing float16 conversion branches:
  exhaustive `mpsDataType(for:)` switch, the `readFloats(into:)` hot-path readback
  (vImage), and the trainer's two host-side feed narrows (vImage bulk / native
  `Float16` scalar). In-graph casts + the fp32-master mixed-precision path are
  dtype-generic so fp16 reuses them; config-D stays bf16-only. fp16 inference is
  tested and sound; fp16 **training** NaNs immediately on this beta stack (see
  Findings + `documentation/macos27-beta1-mpsgraph-findings.md`), so fp16 is an
  inference precision for now. Tests: `FP16ConversionTests` (all pass),
  `FP16ComputePathTests` (forward + checkpoint pass; trainStep cells fail by
  design as tripwires).

- **Block groups: heterogeneous towers with per-group widths (2026-06-12 →
  2026-06-13).** `NetworkArchitecture`'s uniform block fields became
  `blockGroups: [BlockGroup]` — an ordered list of (count + full per-group
  recipe: channels, both conv kernels, SE style/ratio, ReZero α, activation
  function/style, skip merge, dropout multiplier). The engine consumes only the
  flattened `expandedBlocks`; width transitions insert a bias-free 1×1 skip
  projection (WRN staircase). Encode writes `block_groups` only; legacy uniform
  keys decode forever, and uniform towers build byte-identically to the prior
  code (proven by the embedded bit-exact forward-pass save check). Phase B adds
  a per-group Build-New-Model editor + a live `ArchitectureDiagramView` (shared
  with the About popover). Per-conv stride was considered and dropped; two-level
  group entries simplified to one recipe + count. CHANGELOG `73a1bdd` (Phase A)
  / `120f46b` (Phase B) / `ed84386`, `54e3ca3` (recheck + review fixes). First
  production heterogeneous tower (eBNC, ~10.66M params) running since 2026-06-12;
  see ARCH_EXPERIMENTS Experiment 7.

- **Channel dropout (live-tunable) + headless A/B harness (2026-06-12).**
  Every training-mode residual block carries a channel (spatial) dropout node at
  the WRN slot; live-tunable `DropoutRate` parameter, inference graphs untouched.
  Headless `--start-model` + `--training-step-limit` CLI added for the A/B arms.
  Finding: ≈ zero distinguishable effect at 600 steps from both a random-init
  fork and the trained 5K7Z champion (machinery validated; any benefit is a
  long-horizon question). CHANGELOG `eacced3` / `600c016` + two A/B FINDING
  entries.

- **Rich Load Session picker (2026-06-12).** File > Load Session opens a
  lineage-grouped picker showing architecture / run-progress / performance /
  hyperparameters per save, instead of the bare `.fileImporter` (kept reachable
  via Browse…). `CheckpointManager` writes a small `manifest.json` at save;
  legacy sessions are indexed once into an out-of-folder cache. CHANGELOG
  `9939563`; edge-case fixes `54e3ca3`.

- **Tabbed Training Settings popover + main-screen control sweep +
  save-verify v2 fix + chart hover bug fix + latent arena-tau push
  bug fix (2026-05-05).** Cohesive UI consolidation covered in
  detail in CHANGELOG.md under the 2026-05-05 22:00 CDT entry. Net
  result: every editable training-side parameter now lives in a
  single tabbed popover (Optimizer / Self Play / Replay) anchored
  to the existing top-bar Training chip; the main screen keeps a
  read-only "Replay Ratio: X target: Y (auto)" status row; four
  replay-ratio control fields propagate live to `trainingParams`
  while the popover is open with a Cancel-reverts-to-stash
  mechanic; the trainer-file save-verify path now correctly
  handles v2 (trainables + bn + velocity) layouts; the two pLoss
  charts are now hover-aware; `arenaPopoverSave` finally pushes
  schedule changes into the live `samplingScheduleBox`.

- **Decoupled weight decay + arena velocity snapshot + momentum
  session save/load + `μ`/`vNorm` in STATS + chart additions
  (2026-05-05).** Five interlocking changes covered in detail in
  CHANGELOG.md under the 2026-05-05 17:30 CDT entry. Net result:
  the optimizer is now AdamW-style decoupled-decay (μ and weight
  decay tune independently); velocity buffers persist through arena
  promotion via snapshot/restore instead of being zeroed; the
  momentum coefficient itself is now in the session schema and
  no longer silently picks up the user's current slider on resume;
  the [STATS] line and chart grid surface velocity-norm and policy-
  head weight norm so collapse precursors are visible without log-
  diving; and the gNorm chart shows the active grad-clip threshold
  as a reference line so "is the clip permanently active?" is a
  glance rather than a calculation.

  **What landed (summary; see CHANGELOG for file-level detail):**
  - Optimizer update split: `v_new = μ·v_old + clipped_grad`,
    `θ_new = θ − lr·v_new − lr·decayC·θ` (decay term skipped for
    biases / BN affine via `network.trainableShouldDecay`). Bit-
    exact equivalent to the prior coupled-L2 form at μ=0.
    Verified by `testDecoupledDecayMatchesCoupledAtZeroMomentum`.
  - `momentumCoeff: Float?` added to `SessionCheckpointState`.
    `[RESUME-PARAM] momentum_coeff` log on both branches.
    Verified by `testMomentumCoeffRoundTripsThroughSessionState`.
  - `exportVelocitySnapshot()` / `loadVelocitySnapshot(_:)` API
    on `ChessTrainer`; arena start captures `trainerSnapshotVelocity`
    alongside the existing `trainerSnapshotWeights`; promotion
    restores the snapshot instead of zeroing.
    `resetVelocitiesToZero()` retained as an explicit-discard
    escape hatch (still used by `testResetVelocitiesToZero`).
    Verified by `testVelocitySnapshotRoundTrip`.
  - New `velocityGlobalNormTensor` in the training graph,
    accumulated alongside `gradGlobalNorm` from per-tensor `||v||²`
    sums; readback into a new `velocityNorm: Float` field on
    `TrainStepTiming`; rolling window in `TrainingLiveStatsBox`
    surfaces `rollingVelocityNorm: Double?` on the snapshot.
    `[STATS]` line gains `vNorm=` and `μ=` next to `gNorm=`.
  - Chart grid: standalone CPU and GPU MiniLineCharts merged into
    a new `CpuGpuChart` dual-line tile (CPU blue, GPU indigo,
    combined header `CPU N% / GPU M%`); freed slot now hosts a
    `pwNorm` MiniLineChart driven by `policyHeadWeightNorm`; gNorm
    MiniLineChart gains a dashed clip-threshold reference line via
    new optional `referenceLine` / `referenceLineLabel` /
    `referenceLineColor` parameters. `gradClipMaxNorm` plumbed
    through `ContentView → LowerContentView → TrainingChartGridView`.
  - Off-main `trainer.momentumCoeff` race fixed during recheck —
    [STATS] line now uses the existing main-actor-captured
    `momentum` local from the same `MainActor.run` block that
    captures every other live trainer scalar.
  - Three new tests covering the round-trip semantics, plus
    fixture updates to `ChartDecimatorTests` for the new
    `rollingPolicyHeadWeightNorm` field.

  **Compatibility:** Legacy `.dcmsession` files without
  `momentumCoeff` decode unchanged. v1 trainer.dcmmodel files (no
  velocity payload) continue to load via the existing v1-acceptance
  branch. v2 trainer.dcmmodel files written under the *old* coupled-
  decay formula and loaded under decoupled-decay carry baked-in
  decay terms in the saved velocity that wash out over ~`ln(0.01)
  /ln(μ)` steps — a transient, not a correctness break, and only
  visible if μ was high at save time (the default was 0 so most
  saves have zero-velocity).

- **Full parameter coverage in session save + Load/Save Parameters
  menu items + slow-save watchdog (2026-04-30).** Three coupled
  changes that close the parameter-reproducibility gap and add a
  save observability backstop. Original design captured in this
  ROADMAP under Future improvements, then implemented in the same
  session.

  **What landed:**
  - Eight new Optional fields on `SessionCheckpointState`
    (`SessionCheckpointFile.swift`): `lrWarmupSteps`,
    `sqrtBatchScalingForLR`, `replayBufferMinPositionsBeforeTraining`,
    `arenaAutoIntervalSec`, `candidateProbeIntervalSec`,
    `legalMassCollapseThreshold`, `legalMassCollapseGraceSeconds`,
    `legalMassCollapseNoImprovementProbes`. All Optional →
    older `.dcmsession` files decode unchanged with new fields nil.
    `buildCurrentSessionState` populates them; `startRealTraining`
    resume code reads them with `if let v = rs.foo { … = v } else { … = currentAppStorageValue }`
    fallback. Each restored field also writes back to its
    `@AppStorage` mirror so the UI shows what the session was
    actually running with, not what the user's current global
    preference happens to be.
  - `[RESUME-PARAM]` log lines added for every restored field
    (both the eight new ones and the existing pre-expansion ones
    `learning_rate`, `entropy_bonus`, `draw_penalty`,
    `weight_decay`, `grad_clip_max_norm`, `K`). Lines fire only
    when the saved value is present and valid — older sessions
    falling through to `@AppStorage` stay silent. Format:
    `[RESUME-PARAM] <field>: <before> -> <after> (from session)`.
  - `CliTrainingConfig` promoted from `Decodable` to `Codable`
    (`CliTrainingConfig.swift`) with a new `encodeJSON()` helper
    using `.prettyPrinted, .sortedKeys` for stable, diffable output.
    Optional fields with nil values omit cleanly via Swift's
    synthesized `encodeIfPresent`.
  - Two new File menu items wired through `AppCommandHub`:
    `Load Parameters…` (file picker → decode `CliTrainingConfig`
    → call `applyCliConfigOverridesFromMenu(cfg:)` which routes
    through the same `applyCliConfigOverrides(cfg:)` the launch
    `--parameters` flag uses) and `Save Parameters…` (file
    exporter → build a fully-populated `CliTrainingConfig` via
    `currentParametersConfig()` → encode JSON via `CliParametersDocument`
    `FileDocument` adapter). Load Parameters is disabled during
    realTraining / continuousPlay / continuousTraining / sweep /
    game-in-progress / building / save-in-flight, matching Load
    Session / Load Model. Save Parameters is always enabled (no
    destructive effects).
  - `applyCliConfigOverrides` refactored: no-arg overload reads
    `cliConfig` (launch path); new `applyCliConfigOverrides(cfg:)`
    parameterized variant takes a config directly; new
    `applyCliConfigOverridesFromMenu` is the menu's named entry
    point. All three return `[ParameterOverrideChange]` — a
    typealias for `(label: String, before: String, after: String)`
    — used by the menu handler to surface count and field labels
    in the status row: `Loaded <file>: N parameters changed
    (label1, label2, …)`. Per-field `[APP] --parameters override:`
    log lines were already there.
  - New `.slowProgress` case on `CheckpointStatusKind` (orange
    text + `clock.badge.exclamationmark.fill` icon, 120-second
    auto-clear). `slowSaveWatchdogSeconds = 10` constant.
    `startSlowSaveWatchdog(label:)` and `cancelSlowSaveWatchdog()`
    helpers. Wired into all four save sites: manual + periodic
    (`saveSessionInternal` via the shared `clearInFlight` helper),
    post-promotion (inline arena-coordinator task), and
    `Save Champion as Model` (`handleSaveChampionAsModel`). Each
    save's completion path (success, failure, timeout, export
    error) cancels the watchdog so a fast save's body never runs.
    A slow save logs `[CHECKPOINT-WARN] <label> still running
    after 10s — disk busy or replay buffer large?` and updates
    the status row to amber with a `(still running, 10s+)` suffix.
    Fires exactly once per save — no progressive warnings —
    because completion will eventually flip the row to
    success/error and restore normal styling.
  - Watchdog deadline tuned from 5 s (initial spec) to 10 s
    (final shipped value) after considering that the
    post-promotion save runs at `.utility` priority and could be
    delayed under load. 10 s leaves headroom against
    false-positive warnings while still surfacing genuinely stuck
    saves promptly.

  **Tests added:**
  - `CliTrainingConfigTests.testEncodeDecodeRoundTripPreservesEveryField`:
    every field round-trips through `encodeJSON()` → decode.
  - `CliTrainingConfigTests.testEncodeJSONUsesSortedKeys`: pins
    the sorted-keys output so the UI-saved file diffs cleanly
    against an autotrain-saved file with the same values.
  - `CliTrainingConfigTests.testEncodeJSONOmitsNilFields`: pins
    `encodeIfPresent` semantics — partial configs produce
    partial files.
  - `SessionCheckpointSchemaExpansionTests.testRoundTripPreservesAllExpansionFields`:
    8 new schema fields encode → decode cleanly.
  - `SessionCheckpointSchemaExpansionTests.testLegacySessionWithoutExpansionFieldsDecodes`:
    older `.dcmsession` files without the new keys still decode,
    with new fields nil — back-compat pin.
  - `SessionCheckpointSchemaExpansionTests.testCrossFormatKeysAreIndependent`:
    snake_case in `parameters.json` and camelCase in
    `session.json` decode independently.
  - Existing `testAllFieldsDecode` and
    `testPartialJsonLeavesMissingFieldsNil` extended to cover the
    new fields.

  **Cross-format invariant achieved:** an autotrain `parameters.json`
  is directly loadable in the UI; a UI-saved parameters file is
  directly usable as `--parameters` input to the CLI. Same Codable
  shape, same field names, same units.

  **Deviations from the original plan:** none of substance. The
  watchdog deadline went from 5 → 10 s during implementation per
  user direction. The `Save Parameters…` menu item is always
  enabled (no `networkReady` gate) — minor concession noted in
  review, since a defaults-dump can be useful as a starting
  template even before any model is built.

- **Engine-level legal-move validation (2026-04-20).** Previously
  `ChessGameEngine.applyMoveAndAdvance` trusted the caller to supply a
  legal move, and `MoveGenerator.applyMove` would trap on a force-unwrap
  for moves whose from-square was empty. The argument for the
  performance shortcut was that the game loop in `ChessMachine` already
  generates the legal-move list per ply for player choice and
  end-detection, so re-deriving it inside apply would be wasted work.
  That trust held in practice — the only caller was `ChessMachine`,
  which always sampled from `MoveGenerator.legalMoves(for:)` — but it
  left the engine unsafe for any future caller (UI drag-drop, loaded
  PGN, network input, a buggy player) to invoke directly.

  The fix preserves the one-`legalMoves`-call-per-ply invariant by
  having the engine *own* the legal-move list rather than duplicating
  it on the caller side. `ChessGameEngine` gained a
  `private(set) var currentLegalMoves: [ChessMove]` that is computed
  once at init (seeded from the starting state) and refreshed inside
  `applyMoveAndAdvance` after each successful move — using the same
  `nextMoves = MoveGenerator.legalMoves(for: state)` call that already
  powered end-of-game detection. `applyMoveAndAdvance` now guards
  `currentLegalMoves.contains(move)` before apply and throws the new
  `ChessGameError.illegalMove(ChessMove)` if the guard fails
  (`ChessMove` is already `Equatable`). No extra `legalMoves` calls on
  the hot path.

  `ChessMachine.runGameLoop` dropped its local `var currentLegalMoves`
  and now reads `engine.currentLegalMoves` both when calling
  `player.onChooseNextMove(...)` and implicitly through the engine's
  self-refresh. Illegal moves from a buggy player flow through the
  existing `playerErrored` + break path in the game loop, identical to
  any other thrown error — partial-game positions up to the failure
  point still flush to the replay buffer as before. `applyMoveAndAdvance`
  retains `@discardableResult` on its `[ChessMove]` return (kept for
  callers who prefer the inline value, e.g. tests), so the change is
  API-additive rather than breaking. Callers that used `try?` to
  discard errors (the ContentView sanity-check knight-shuffle probe,
  `RepetitionTrackingTests` paths) continue to work: their moves are
  legal so validation passes, and any genuine illegal move is now
  surfaced through the same error-swallowing behavior that already
  existed for `gameAlreadyOver`.

  Build green. This closes the roadmap item of the same name.

- **Session durability hardening — "saved means golden" (2026-04-20).**
  Closes TODO_NEXT.md #3. The save pipeline now guarantees that either
  a fully-verified, fsync'd `.dcmsession` bundle appears on disk under
  its final name, or nothing appears with that name. Restored
  sessions are bit-identical to what was saved, or loading fails with
  a specific error describing which check tripped.

  **Principle.** If any piece of a session save cannot be fully
  verified end-to-end, the whole save fails, all partials are removed,
  and no final-named artifact appears on disk. Built as a coordinated
  change across `ReplayBuffer.swift`, `CheckpointManager.swift`,
  `DrewsChessMachineApp.swift`, and `ContentView.swift`, plus docs in
  `replay_buffer_file_format.md` (new) and `dcmmodel_file_format.md`
  (new), and five new XCTest cases in `ReplayBufferTests.swift`.

  ### ReplayBuffer format v3 → v4

  `ReplayBuffer.fileVersion` bumped 3 → 4. Readers reject v1/v2/v3
  cleanly with `PersistenceError.unsupportedVersion`. No migration
  path — matches the project's delete-and-retrain stance (the user's
  existing v3 `.replay_buffer.bin` files will not load; the replay
  buffer is recovered by resumed self-play).

  - **SHA-256 trailer.** 32-byte digest appended over header + all
    four body sections, verified before any header field is trusted
    at load time. Mirrors the `.dcmmodel` integrity-trailer
    convention. Computed streaming during write (CryptoKit
    `SHA256.update(data:)` per chunk), so no extra hashing pass — the
    bytes fed to `handle.write(...)` are also fed to the hasher.
  - **Strict file-size equality check.** Restore computes
    `expectedBytes = headerSize(56) + storedCount × (floatsPerBoard ×
    4 + 12) + trailerSize(32)` and requires `actualFileSize ==
    expectedBytes`. Uses `==`, not `>=`, because the format is fully
    deterministic — any deviation is corruption. New
    `PersistenceError.sizeMismatch(expected, got)`.
  - **Upper-bound sanity caps.** Applied before any allocation or
    seek arithmetic so a corrupted header can't drive a
    multi-terabyte allocation or overflow the size computation.
    Caps: `floatsPerBoard ≤ 8_192`, `capacity ≤ 10_000_000`,
    `storedCount ≤ 10_000_000`. New
    `PersistenceError.upperBoundExceeded(field, value, max)`.
  - **`handle.synchronize()` before close.** In `_writeLocked`,
    forces APFS to flush dirty pages to the device before the file
    handle closes. On top of this, `CheckpointManager.saveSession`
    adds `fcntl(F_FULLFSYNC)` via `fullSyncPath` for drive-cache-bypass
    durability.
  - **Atomic write-and-snapshot.** `ReplayBuffer.write(to:)` now
    returns the `StateSnapshot` reflecting exactly the state
    serialized into the file, captured under the same `queue.sync`
    lock that serializes the write. Post-save verification compares
    against this value — a subsequent `stateSnapshot()` call would
    diverge because concurrent self-play workers resume appending
    after the save-gate releases the trainer pause. Annotated
    `@discardableResult` so existing callers (tests, etc.) compile
    unchanged. **This was a recheck-catch** — the first pass
    compared against a freshly-called `live.stateSnapshot()` and
    would have spuriously failed the counter comparison every time
    saves happened during active training.

  ### `CheckpointManager` durability pipeline

  `saveSession` now runs a full fsync pipeline on top of the existing
  tmp-dir-then-rename atomicity:

  1. Write all four files (two `.dcmmodel`, `session.json`,
     `.replay_buffer.bin`) into a `.tmp` staging directory.
  2. `fullSyncPath` each of the four files — issues `fcntl(fd,
     F_FULLFSYNC)` on Apple filesystems (falls back to `fsync` if
     unsupported). Bypasses the drive's write cache, not just the VFS
     page cache.
  3. Verify:
     - Both `.dcmmodel`: existing bit-exact + forward-pass
       round-trip (unchanged).
     - `session.json`: existing decode round-trip (unchanged).
     - **NEW** — `.replay_buffer.bin`: re-load into a scratch
       `ReplayBuffer` via `restore(from:)`. The scratch restore runs
       the full v4 verification stack (SHA, size, caps). Scratch is
       allocated at `max(1, writtenSnap.storedCount)` — sized to the
       actual data, not to the live ring's full capacity — so a
       half-full 1 M-slot ring does not double peak memory during
       verify. Then compare the restored `storedCount` and
       `totalPositionsAdded` against the `writtenSnap` captured
       atomically from the write. Drift here implies a write-path
       regression that produced a valid-SHA file with wrong bytes
       (the SHA alone cannot catch this class of bug if the write is
       internally consistent but semantically wrong).
  4. `fullSyncPath(tmpDir)` — flush directory-entry metadata.
  5. `fm.moveItem(tmpDir, finalDir)` — atomic rename.
     `FileManager.moveItem` aborts if the destination already
     exists (unlike POSIX `mv`); plus the existing
     `fileExists(finalDirURL)` guard. Two independent guards.
  6. `fullSyncPath(CheckpointPaths.sessionsDir)` — flush the parent
     so the rename itself is durable. A failure here leaves the
     session visible (rename already committed) and logs a warning
     that the parent-directory flush wasn't guaranteed — we do not
     remove the session, as it's already the "best we've got" on
     disk.

  Any failure in steps 1–4 triggers `cleanupTmp()` and throws. No
  final-named artifact appears on disk.

  `saveModel` gets the same treatment: `fullSyncPath` on the tmp file
  before verify-and-rename, and `fullSyncPath` on
  `CheckpointPaths.modelsDir` after rename.

  **New `CheckpointManagerError` cases:** `fsyncFailed(URL, Error)`,
  `replayVerificationFailed(String)`, `sessionReplayMismatch(detail:
  String)`.

  ### Load-time cross-check

  New helper `CheckpointManager.verifyReplayBufferMatchesSession(buffer:
  state:)` runs after `ReplayBuffer.restore(from:)` at session load
  time. Compares `buffer.stateSnapshot().totalPositionsAdded` against
  `state.replayBufferTotalPositionsAdded` from `session.json`.
  Mismatch throws `CheckpointManagerError.sessionReplayMismatch` and
  surfaces in the load UI.

  Only the lifetime counter is cross-checked, not `storedCount` or
  `capacity` — those two intentionally diverge when loading a larger
  saved ring into a smaller live one (existing restore
  `skip = fileStored - target` logic). `totalPositionsAdded` survives
  that logic verbatim and is effectively unique across sessions, so a
  mismatch strongly implies a file-pairing error (wrong replay paired
  with wrong session.json) or SHA-collision-scale corruption.
  `replayBufferTotalPositionsAdded` is Optional in
  `SessionCheckpointState` (back-compat) — missing → check is skipped
  rather than forced to mismatch.

  Wired into `ContentView.loadSessionFrom`'s post-restore path.

  ### Launch-time orphan sweep

  New `CheckpointPaths.cleanupOrphans()` runs from
  `DrewsChessMachineApp.init` after `SessionLogger.start`. Removes:

  - `Sessions/<name>.tmp/` directories — `saveSession`'s staging
    directory (matches the `.tmp` suffix appended to the target
    session dir name).
  - `Models/<name>.dcmmodel.tmp` files — `saveModel`'s staging file
    (matches the `.tmp` extension appended to the final `.dcmmodel`
    filename).

  Each removal is logged `[CLEANUP] Removed orphan <name>`; failures
  log `[CLEANUP-ERR]` and do not abort the sweep (a stuck orphan
  should not prevent the app from starting). Runs once at launch,
  before any save/load UI activates.

  ### Documentation

  - **NEW** `dcmmodel_file_format.md` — full byte-level spec for the
    `.dcmmodel` format, with expanded FNV-1a documentation
    (constants `0x811C9DC5` offset basis and `0x01000193` prime,
    algorithm pseudocode, Swift reference implementation, byte-order
    rationale, worked example, comparison vs. CRC32/xxHash/SHA-256),
    SHA-256 trailer spec, decode protocol, error taxonomy, explicit
    non-goals.
  - **NEW** `replay_buffer_file_format.md` — v3 (historical) + v4
    (current) sections. v4 section includes full layout, decode
    protocol ordering, write protocol, durability pipeline in
    session saves, cross-check semantics, launch-time orphan sweep,
    and full error taxonomy.
  - `CHANGELOG.md` entry at top (short form — points here for full
    detail).
  - `TODO_NEXT.md` §3 removed (was detailed there during planning;
    now done).

  ### Tests

  `DrewsChessMachineTests/ReplayBufferTests.swift` — 5 new tests, all
  green:

  - `testV3FileRejectedWithUnsupportedVersion` — synthesized v3
    header (no SHA trailer) rejects with `unsupportedVersion(3)`.
  - `testV4SHAMismatchRejected` — valid v4 file with one byte flipped
    at offset 56 rejects with `hashMismatch`.
  - `testV4SizeMismatchOnTruncation` — valid v4 file truncated by
    one byte rejects with `sizeMismatch`.
  - `testV4SizeMismatchOnTrailingGarbage` — valid v4 file with an
    extra byte appended rejects with `sizeMismatch`.
  - `testV4UpperBoundRejectedOnCapacity` — header with
    `capacity = Int64.max` rejects with `upperBoundExceeded(field:
    "capacity", ...)`, not an allocation crash.

  Existing tests (`testEmptyBufferWriteRead`,
  `testSinglePositionWriteRead`, `testV2FileRejectedWithUnsupportedVersion`,
  `testBadMagicRejected`, `testTruncatedHeaderRejected`) pass
  unchanged against v4 via the public API. Full test suite: 55/55
  green.

  ### Scope limits explicitly not taken

  - No per-record hashing (file-level SHA-256 is sufficient).
  - No compression (writes stay raw-float).
  - No cross-architecture or cross-version migration.
  - No `session.json` schema change (cross-check reads existing
    Optional fields).

  ### Parameter reference

  New constants (private statics on `ReplayBuffer`):
  - `fileVersion: UInt32 = 4` (was 3)
  - `trailerSize: Int = 32`
  - `maxReasonableCapacity: Int64 = 10_000_000`
  - `maxReasonableStoredCount: Int64 = 10_000_000`
  - `maxReasonableFloatsPerBoard: Int64 = 8_192`

  No existing hyperparameter (LR, batch size, clip, tau, etc.) was
  changed.

- **N-worker concurrent self-play in Play and Train.**

  **Superseded, 2026-04 onwards:** the per-worker-network topology
  described below was replaced by a single shared `BatchedMoveEvaluationSource`
  on the champion network — see the "Batched self-play evaluator"
  entry in the Completed section. The original design is preserved
  verbatim here (per the ROADMAP convention) for historical context
  and rationale. **Do not use this section as a description of
  current runtime behavior.** In particular:
    - `secondarySelfPlayNetworks` no longer exists (ContentView.swift
      has an in-code note to that effect where the field used to
      live).
    - On promotion, candidate weights are now copied into **both**
      the champion (`network`) and the trainer (`trainer.network`);
      there is no per-worker network to mirror.
    - The "topology is asymmetric — worker 0 reuses champion,
      workers 1..N-1 use secondaries" split is obsolete; all
      workers share the champion through the batcher.

  **Original design (as shipped, now historical):** Play and Train
  previously ran a single self-play worker, which at ~357 moves/sec against
  a 3,012 moves/sec training consumer meant every replay-buffer position
  was sampled ~8.4× on average before eviction — far above the 2–4×
  replay ratio common for off-policy RL, and the buffer also covered only
  ~625 games of play diversity. The fix is to spawn `N` concurrent
  self-play workers at session start, each with its own dedicated
  `ChessMPSNetwork` instance so no two concurrent `evaluate` calls share
  MPSGraph state. `ContentView.initialSelfPlayWorkerCount` (currently
  `6`) sets the default active count when a session begins;
  `ContentView.absoluteMaxSelfPlayWorkers` (currently `16`) is the hard
  ceiling — we pre-build that many inference networks and spawn that
  many worker tasks so the user can live-tune N inside
  `[1, absoluteMaxSelfPlayWorkers]` via a Stepper next to Run Arena
  without restarting the session. Topology is asymmetric: worker 0
  reuses the existing `network` (the champion, also the arena snapshot
  source), and workers `1..N-1` use new `secondarySelfPlayNetworks`
  mirrored from the champion at session start and at every arena
  promotion. Each worker owns its own `WorkerPauseGate`, so the
  arena-champion snapshot path (which only reads `network`) still
  pauses only worker 0, and only the promotion branch pauses every
  worker to `loadWeights` into every self-play network. Players
  (`MPSChessPlayer` white/black) are now allocated once per worker and
  reused across games — `ChessMachine.beginNewGame` already calls
  `onNewGame` on each, which resets per-game scratches while keeping
  backing storage alive. Under N=1 (checked live per game via
  `countBox.count == 1`, not captured at spawn), worker 0 wires
  `GameWatcher` as its `ChessMachine` delegate for the animated board;
  under N>1 no worker does, and a placeholder overlay "N = X concurrent
  games" hides the static board slot so the Candidate test picker
  remains usable. Aggregate self-play rates accumulate through the
  thread-safe `ParallelWorkerStatsBox`, which every worker calls
  identically via `recordCompletedGame(moves:durationMs:result:)` —
  no worker-0 specialness in the stats path. Setting N to 1
  reproduces the pre-change behavior (modulo the per-game player
  reuse cleanup). Memory cost is ~12 MB per additional inference
  network, trivial on unified memory.

  **Idle workers stay allocated deliberately.** When the user drops N
  from 6 to 3 via the Stepper, workers 3–5 finish their current game,
  then on their next iteration evaluate `countBox.count > workerIndex`,
  see false, and enter `WorkerPauseGate.markWaiting()` — a 50 ms
  sleep-poll loop that costs near-zero CPU. Their `ChessMPSNetwork`
  instances, `MPSChessPlayer` scratches, `WorkerPauseGate` state, and
  Swift tasks **all stay alive for the life of the session.** Only GPU
  cycles, CPU cycles for move generation / encoding / sampling, and
  replay-buffer lock contention are freed. Networks are only actually
  deallocated when Play and Train stops — and even then
  `secondarySelfPlayNetworks` persists in `@State` across sessions so
  re-entering Play and Train doesn't re-pay the MPSGraph build cost
  (~100 ms + per-network kernel JIT).

  This is a deliberate memory-vs-latency trade. The alternative design
  would cancel tasks and release networks on Stepper-down, then rebuild
  on Stepper-up — saving ~12 MB per idled worker but costing ~100–300 ms
  per + click for MPSGraph construction, first-run kernel JIT, and
  weight sync from the champion. Keeping everything pre-spawned means +
  and − clicks are effectively instant (≤50 ms, bounded by the idle
  poll interval) with no visible latency on the UI. At
  `absoluteMaxSelfPlayWorkers = 16` the steady-state memory cost is
  ~180 MB of idle network state plus ~74 MB of `MPSChessPlayer` scratch
  buffers, which is fine on Apple Silicon unified-memory systems. If
  that footprint ever becomes a problem on tighter hardware, the
  release-on-shrink design is the fallback; for now the latency win on
  live tuning is worth the static allocation.

- **Bundled architecture refresh (v2), 2026-04-19/20.** A coordinated
  redesign of the policy encoding, input planes, policy head, and
  value-baseline semantics, delivered as one bundle because the moves
  are coupled (the policy-encoding bijection and the fully-conv head
  both change the meaning of the 4864 logits; a staged rollout would
  have required throwaway migration code). Full design and phase
  breakdown live in `dcm_architecture_v2.md`. Summary of what shipped:

    - **AlphaZero-shape policy encoding (Phase 2).** The old flat 4096
      = 64×64 from-to encoding was replaced with 4864 = 76 × 64
      channel-square logits: 56 queen-style directions × distances, 8
      knight, 9 underpromotion (N/R/B × 3 directions, channels 64–72),
      3 queen-promotion (channel 73–75). Dedicated underpromotion
      channels fix the prior silent-collapse where all four promotion
      pieces mapped to the same index. The bijection lives in
      `PolicyEncoding.policyIndex(_:currentPlayer:)` — `ChessMove.policyIndex`
      was deleted so callers must think about the side-to-move frame flip.
    - **Fully-convolutional 1×1 policy head (Phase 4.3).** The old FC
      head (`128 × 64 → 4096`, ~528K params) was replaced with a 1×1
      conv `128 → 76` (~9.8K params, ~50× smaller). Translation
      equivariance is preserved end-to-end, which matches modern lc0
      practice and was the motivation cited in the ML review.
    - **20-plane input (Phase 3).** Planes 18 (≥1× before) and 19 (≥2×
      before) feed the network threefold-repetition context.
      Implementation reuses the engine's existing `positionCounts`
      table — no Zobrist machinery was needed (see `dcm_architecture_v2.md`
      Phase 1 for why we deviated from the originally-planned Zobrist
      path).
    - **Fresh-baseline value targets (Post-impl Addendum A).** The old
      `vBaseline` was a frozen self-play-time value; the trainer now
      runs an extra forward-only pass on its *current* network before
      each training step to recompute per-position `v(s)`, then
      overwrites the play-time staging before feeding the training
      graph. `MPSGraphGradientSemanticsTests` verified the
      placeholder-boundary stop-gradient semantics (MPSGraph has no
      `stop_gradient` op, and `with`-array exclusion does not prune
      backward-pass paths — the placeholder feed is the only correct
      way). Cost is ~33% extra forward FLOPs per training step;
      diagnostic `vBaselineDelta` now appears in `[STATS]`.
    - **`maxTensorElementCount` now computed from live arch
      (Phase 7.2).** `ModelCheckpointFile.maxTensorElementCount` is
      derived from `ChessNetwork.channels`, `inputPlanes`,
      `policyChannels`, and `seReductionRatio` (covers stem conv,
      residual conv, policy conv, SE FC). Acts as a defense-in-depth
      sanity cap on per-tensor element counts during `.dcmmodel`
      load — rejects implausible sizes before allocation, even when
      the SHA-256 trailer happens to match. Auto-tracks any future
      architecture change, no manual bump needed.

  The v2 bundle also introduced a `ReplayBuffer` format v3 (old
  buffers rejected cleanly on load), arch-hash bump on `.dcmmodel`
  (old checkpoints rejected with `.archMismatch`), and the first
  XCTest target. `CHANGELOG.md` has the commit-level breakdown;
  `dcm_architecture_v2.md` has the phase-by-phase design and a
  consolidated "Current state (as-built)" section that includes the
  post-impl follow-ups (sampling tau bump to 2.0, Candidate Test RAW-
  cell top-K display, entropy-alarm threshold) *and* the four post-v2
  commits listed below, plus a full parameter-defaults table.

  **Post-v2 follow-ups shipped during the first v2 run** (commits
  `9298273` → `068f805` → `7757418` → `cf1cc24`, all 2026-04-20):

    - **Advantage standardization + K dropped 50 → 5.** The policy-
      gradient weight is now `A_norm = (A − mean(A)) / sqrt(var(A) + 1e-6)`
      computed per batch inside the graph, autograd-safe because `A`
      depends only on the `z` and `vBaseline` placeholders. Removes the
      systematic bias when the value head has a global offset (e.g.
      `E[v] ≈ 0.45` once draws dominated self-play). With `A_norm`
      already at unit stdev, the pre-standardization `policyScaleK = 50`
      was pinning `gradClipMaxNorm` almost every step; dropped to `5.0`.
    - **Live-editable hyperparameters.** `weightDecayC`, `gradClipMaxNorm`,
      `policyScaleK`, `learnRate`, `entropyRegCoeff`, `drawPenalty`, and
      both sampling schedules are now fed to the training graph per step
      via scalar placeholders, so UI edits commit immediately without
      rebuilding the graph. Values persist in `@AppStorage` where
      applicable and are restored on session load. Every commit writes
      a `[PARAM] name: old -> new` line. `SessionCheckpointState` gained
      `policyScaleK: Float?` (Optional for back-compat) and the full
      `wd / clip / K / sp+ar tau` set on load.
    - **Diagnostics expansion.** New `TrainStepTiming` fields:
      `playedMoveProb`, `policyLogitAbsMax`, `policyHeadWeightNorm`,
      advantage distribution (`mean / std / min / max / fracPos / fracSmall`),
      plus `p05 / p50 / p95` from a rolling raw-advantage ring.
      Separate `legalMassSnapshot` probe (legal-mass + top1-legal) via
      `BoardEncoder.decodeSynthetic`, refreshed every 25 steps during
      bootstrap. `ParallelWorkerStatsBox` gained a 512-entry game-length
      ring (`p50 / p95 / avgLen`). `MPSChessPlayer` now counts
      "randomish" plies where post-temperature max probability is below
      `1.5 / N_legal` (policy-collapse signal independent of tau).
      `[STATS]` emitter restructured into bootstrap (per-step, first 500
      steps) + steady-state (60 s) phases.
    - **Full-sort top-K for catastrophic collapse.**
      `ChessRunner.extractTopMoves` now sorts the full 4864-cell policy
      vector rather than capping at `count × 4`, so a collapsed policy
      whose top cells are all off-board still produces `count` legal
      visualizations instead of an empty Candidate Test panel.
    - **MPSGraph reshape layout + sign-consistency tests.** New
      `MPSGraphReshapeLayoutTests` empirically verifies the policy head's
      `[B, 76, 8, 8] → [B, 4864]` reshape is NCHW row-major under
      `c·64 + r·8 + col` (plus end-to-end through `oneHot` +
      `softMaxCrossEntropy`). New `SignConsistencyTests` covers encoder
      symmetry, policy-index symmetry for mirrored moves, outcome-sign
      truth table, advantage-formula sign convention, geometric-decode
      round-trip, and bit-identical network output for bit-identical
      inputs.
    - **Advantage raw ring capped at 32K.** The `_advRawRing`
      in `TrainingLiveStatsBox` was originally sized `rollingWindow ×
      batchSize = 512 × 4096 ≈ 2 M Float`. `snapshot()` sorts the full
      filled portion for percentile extraction, and the 10 Hz UI
      heartbeat's `Task { @MainActor }` calls `snapshot()` via
      `queue.sync` — once the ring filled (~step 500) each sort cost
      ~150 ms on main, saturating the main actor. Because
      `fireCandidateProbeIfNeeded` is `@MainActor` and awaited after
      every training step, training throughput collapsed from
      ~2300 moves/sec to ~300 moves/sec and the UI went non-responsive.
      `advRawRingMaxCapacity = 32_768` drops sort cost from ~150 ms to
      ~1 ms while keeping percentile error below 0.5 % for a
      log-eyeballed diagnostic.

  **Default-parameter drift to record** (relative to this ROADMAP's
  earlier "N-worker concurrent self-play" entry):
  `initialSelfPlayWorkerCount` is now `24` (was `6`),
  `absoluteMaxSelfPlayWorkers` is now `64` (was `16`),
  `trainingBatchSize` is `4096`, `replayBufferCapacity` is `1_000_000`,
  `tournamentGames` is `200`, `tournamentPromoteThreshold` is `0.55`.
  The memory-vs-latency analysis in that older entry still applies —
  the numbers scaled up, the trade-off didn't change. The full current
  parameter-default table lives in `dcm_architecture_v2.md` under
  "Current parameter defaults".

  **Still open after v2.** `TODO_NEXT.md` #3 (ReplayBuffer durability —
  `fsync` + length invariant + reordered atomic save) remains
  unaddressed; adaptive LR schedule remains a design-only entry.
  One runtime regression observed but not yet root-caused: an
  `Unsupported MPS operation mps.placeholder` assertion during
  training after the live-hyperparameters change in `7757418`.
