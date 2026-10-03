# Determinism, Exact Resume, Ablation Init and Lineage — Design Plan

> **Note (2026-10-02):** this plan's format bump is now **v7** — v5 was taken by `se_activation` (2026-10-01) and v6 by `rezero_alpha_cap` (2026-10-02); see the note under §D2.

Status: **PLAN ONLY — nothing here is implemented, except phase P14 (autosave retention
pool), implemented 2026-10-01 (see P14) — and, by owner decision 2026-10-01, gated behind
the `automatic_save_pruning_enabled` setting (default off) and forced off in the current
build by `CheckpointPaths.automaticSavePruningForcedOff`.** Audited against `main` at
`47984b0` (on top of d15f706 "exact resume", 8926221 "#7 zero-β / format v4 /
`--derive-model`", 248e122). Owner decisions and review findings of 2026-09-30
folded in: **all owner decisions D-1…D-10 are decided** (end of file) and the
plan text below is written to match them. Every `file:line` was verified against
`main` at `bd32c15`; no app code has changed since (re-checked at `2cfe5a3`).

**Addition 2026-10-01 (owner decision):** `AUTOSAVE_RETENTION_PLAN.md` is folded into
D-8 — a combined autosave retention pool over `-periodic` and `-promote` saves, with
`-manual` and SIGUSR2 saves exempt. See the **Retention** sub-bullet of D-8 (Owner
decisions), the **Autosave retention** bullet of §D8, and phase **P14**. One
sub-question there (how "Promote Trainee Now" saves are treated) was flagged **OPEN**;
**RESOLVED 2026-10-01 (owner): treat them like automatic promotions** — they stay in the
pool under the shared `promote` tag. Everything in that addition is now decided, and P14
is **implemented (2026-10-01)**. The superseded plan is kept, marked, in
`documentation/plans-completed/AUTOSAVE_RETENTION_PLAN.md`.

This plan is the single design for four things that have to be designed together
because they share storage, format versioning and code paths:

- **A** Seeded randomness (one generator type, one master seed, named sub-streams).
- **B** Ablation-friendly, per-tensor-derived initialization and a `--derive-model` graft.
- **C** Exact resume for every training path, with a resume-equivalence harness.
- **D** Lineage, step and run-time tracking across segments, carried in the files.

It **supersedes** GitHub issues **#4** (lineage metadata — now Part D), **#5**
(resume defaults — now Phase 2 of Part C) and **#6** (exact-resume checklist —
now Part C). It **builds on** **#7** (zero-β, format v4, `--derive-model`),
which is implemented; Part B extends it. When this plan's umbrella issue is
opened, #4/#5/#6 are closed as "superseded by" it, each linking the section that
replaces it, so there is one open tracker, not four.

Rules this plan is held to (from CLAUDE.md files): single source of truth; the GUI
self-play, corpus-replay and train-vs-UCI paths share code (`TrainerHyperparameters`,
`TrainerResumeState.swift`); no silent defaults; no fallback for things that must
exist; no `try?`; the full parameter checklist for every new parameter; the
**format-version rule** from #7 (legacy files take old behavior; files at the new
version must carry the new fields, and a missing field there is a load error);
every older model must still load, train and infer. No migration code except the
owner-approved D-9 in-memory hash recompute for legacy buffers.

---

## 0. Honest scope statement (read first)

What "deterministic" can and cannot mean in this app:

| Path | Trajectory-exact after this plan? | Why |
|---|---|---|
| Corpus replay (`--train-from-corpus`, sequential loop) | **Yes, within GPU-numeric limits** (§A5). Same seed + same build + same device class + same OS → the sampled minibatch indices, dropout masks and schedule are identical; weights match bit-exact *if* MPSGraph is run-to-run deterministic on that device (measured, not assumed — §C6). | The loop is single-threaded and **step-locked** (`CLI/CorpusReplayRunner.swift:769-790`, target `:778`, `trainStep` `:786`): each iteration feeds whole games until `positionsFed ≥ prefillPositions + step·perStepFeed`, then awaits one `trainStep`; feeding and training never overlap. The step boundary is therefore a clean, lock-free snapshot point — no pause/drain protocol is needed for replay. Every random draw can come from a named stream. |
| GUI Play-and-Train | **No — only the random *streams* are exact, not the trajectory.** | Self-play and training run concurrently; which positions are in the buffer when the trainer samples depends on wall-clock interleaving, the `ReplayRatioController` delay loop (wall-clock driven), worker count changes, arena pauses. Seeding makes each *game* reproducible given its inputs and makes minibatch draws reproducible given the buffer, but the buffer contents at step N are a race. |
| Train-vs-UCI | **No.** | External engines (multi-threaded Stockfish, time-based `go`) are nondeterministic, plus the same concurrency as GUI. Seeding still makes our side's sampling and the trainer's minibatches reproducible *given* the games. |

**Decided (D-1, owner, 2026-09-30):** GUI self-play and train-vs-UCI resumes are
**state-exact at most** — they never reproduce the rest of the run — and corpus
replay is the only trajectory-exact path. Under D-8 the replay buffer is not
saved by default on those two paths, so a default GUI or train-vs-UCI resume is
not even buffer-exact: it refills the buffer from new games and is labeled
`NOT EXACT: buffer` (C1 #9, C3). Everything else it carries (weights, optimizer
state, schedule clock, RNG streams, serials, counters, lineage) is restored
exactly.

So the harness in §C6 asserts **bit-exact equivalence for corpus replay** and
**state-exact save/restore round-trips** (every piece of state restored equals what
was saved) for the GUI and train-vs-UCI, including the buffer only when the save
included it. That is the correct, provable target; promising trajectory-exact GUI
resume would be false.

---

# PART A — Seeded randomness

## A1. Inventory of every randomness source

Method: `grep -rnE '\.random\(|shuffled|shuffle\(|randomElement|SystemRandomNumberGenerator|arc4random|drand48|GKRandom|randomUniform|randomPhilox|RandomNumberGenerator|UUID\(\)'`
over `DrewsChessMachine/DrewsChessMachine/**/*.swift` (tests excluded), plus a
manual read of the init helpers, the sampler, the replay-buffer sample paths and
the BN calibration path. `drand48`, `GKRandom`, MPSGraph `randomUniformTensor`
and MPSGraph `dropout(...)`: **no hits** (dropout is hand-built from
`randomTensor(withShape:descriptor:stateTensor:)`, see A4).

**57 call sites** total: **19 training-relevant**, **2 evaluation/probe**, **36
irrelevant** (identity, UI, bot, benchmarks).

Classification key: **T** = needed for reproducible training; **E** = needed for
reproducible evaluation/probing; **I** = irrelevant (identity, UI, bot, benchmark).

### A1.1 Training-relevant (T) — 19 sites

| # | Site (`file:line`, under `DrewsChessMachine/DrewsChessMachine/`) | What | Treatment |
|---|---|---|---|
| 1 | `Network/ChessNetwork.swift:3424` (`fillUniform01`, called by `heInitFloats` `:3363`) | `arc4random_buf` → uniforms → Box–Muller for **every** He/Glorot weight. Consumers: stem `:693`, feature-skip fusion `:910`, block convs `:2738` (incl. `_skip_proj` via `makeConvWeight` `:2855`), SE fc1 `:2913`, SE fc2 `:2934`/`:2939`, policy head `:3032`/`:3057`/`:3071`/`:3091`/`:3107`, value head `:3181`/`:3210`/`:3234` | Replace with per-tensor streams (Part B1). The graph builder takes a `WeightInitialization` argument; no global RNG. |
| 2 | `Network/ChessNetwork.swift:781` | Dropout Philox seed `Int.random(in: 0..<Int.max)` **baked into the graph as a constant** at build time | Seed becomes a graph *input* (placeholder assign), value from the `dropout` stream; state capturable/restorable (A4). |
| 3 | `Network/ChessMPSNetwork.swift:153` | `legalMoves.randomElement()` — random game walked to build the BN-calibration warmup batch; the resulting running stats are written into a fresh network and **are saved** with the champion | `init.bn_calibration` stream (part of the init family). |
| 4 | `Training/ReplayBuffer.swift:290` | `MaterialBucketSlots.randomSlot()` (stratified sampling) | Buffer-owned `sampler` stream; stable `nextBounded`. |
| 5–11 | `Training/ReplayBuffer.swift:1783, 1924, 1973, 2189, 2198, 2235` | Uniform `Int.random(in: 0..<held)` in the degenerate, stratified, fast, tilted and budget-fallback paths | Same `sampler` stream, stable `nextBounded`. |
| 12 | `Training/ReplayBuffer.swift:2154` | `Double.random` length-weighted acceptance `tiltAccepts` | Same stream, stable `nextUnitDouble`. |
| 13 | `Training/BatchedSelfPlayDriver.swift:700` | `Double.random` draw-keep fraction | Per-game stream (A3.3), stable `nextUnitDouble`. |
| 14 | `Network/MoveSampler.swift:169` | Inverse-CDF move draw (self-play, arena, train-vs-UCI, UCI, human play) | `sampleMove(... rng: inout DCMRandom)` — caller owns the stream. |
| 15 | `Network/MoveSampler.swift:243` | Gamma boost `U^(1/α)` (Dirichlet, α<1) | Same `rng` parameter. |
| 16 | `Network/MoveSampler.swift:265` | Marsaglia–Tsang acceptance `u` | Same. |
| 17–18 | `Network/MoveSampler.swift:287, 288` | Box–Muller normals for Gamma | Same. |
| 19 | `Persistence/ModelDerivation.swift:521` | `--derive-model` re-init of the SE β half via `glorotInitFloatsFCInOut` (unseeded) | Per-tensor init stream keyed by the recorded derivation seed (B3). |

### A1.2 Evaluation / probe (E) — 2 sites (+ the shared sampler)

| Site | What | Treatment |
|---|---|---|
| `Training/ReplayBufferAnalyzer.swift:610` | `indices.shuffled()` subsample for per-material-bucket policy-entropy probe | `probe` stream seeded per probe invocation from `(masterSeed, "probe.entropy_by_bucket", trainerStep)`; `shuffled(using:)` is acceptable here (A2.4) but the stable Fisher–Yates is used anyway for cross-version comparability of probe series. |
| Arena move sampling (`Arena/TickTournamentDriver.swift:658` → `MoveSampler`) | Candidate-vs-champion tournament sampling | Per-game stream `arena.<arenaIndex>.game.<gameIndex>`. Pairing/colors are **deterministic already** (`candIsWhite = i % 2 == 0`, `:230`, `:359`) — no RNG. |

Train-vs-UCI colors are deterministic too (`Training/TrainVsUciDriver.swift:589`, `gameIndex % 2`); opponent pool order has no RNG. Its move sampling uses site 14–18 via `TrainVsUciDriver.swift:333`.

### A1.3 Irrelevant (I) — 36 sites, keep system randomness

| Group | Sites | Why keep |
|---|---|---|
| **ModelID / corpus IDs** | `Persistence/ModelID.swift:118`, `Persistence/GameCorpus.swift:48` | Identity must be **unique across reruns**. A seeded ID would collide on every rerun with the same seed — precisely the collision problem #4 documents (`bzw3-31` re-minted). Keep `SystemRandomNumberGenerator`. |
| **UUID identity** | `App/LichessProbeExporter.swift:68`; `App/UpperContentView/BuildNewModelModel.swift:73,103,148`; `App/UpperContentView/TrainingAlarmController.swift:324`; `App/UpperContentView/HumanPlayBoardView.swift:408,487`; `App/UpperContentView/ArenaBreakdownsView.swift:307,407,465,524,582,641,695,744,795,849`; `Network/MPSChessPlayer.swift:308`; `Chess/ChessPlayer.swift:59,88`; `Chess/HumanChessPlayer.swift:22`; `Arena/TournamentRecord.swift:8`; `LichessBot/App/LichessBotController.swift:444`; `LichessBot/Data/LichessBotRecordStore.swift:245` | SwiftUI identity / record keys. Not randomness in the math. |
| **Lichess bot** | `LichessBot/App/LichessBotController.swift:1735` (matchmaking generator; `LichessBotMatchmaking.swift:254,274` already take `using:`), backoff jitter `LichessBot/Play/LichessBotSessionManager.swift:480`, `LichessBot/Data/LichessBotReconciler.swift:297`, `LichessBot/Play/LichessBotGameSession.swift:270,299,872` | Network behavior. Jitter must stay random. |
| **Demo player** | `Chess/ChessPlayer.swift:77` | Random-mover demo opponent. |
| **Benchmarks** | `Training/ChessTrainer.swift:4415, 4420` (synthetic moves/outcomes), `:7430` (LCG seed for `fillRandomFloats`) | Throughput sweeps; values don't matter. Optionally take a fixed constant so sweeps are repeatable; not required. |

Not randomness (false positives checked): `Network/NetworkWeightAnalyzer.swift:162-167` (`randomElementCount` is an analytic expected-norm count), `LichessBot/Play/LichessBotChallengeQueue.swift:111` (`makeID` injection).

## A2. The generator

### A2.1 Type

New file `Utils/DCMRandom.swift`:

```swift
/// xoshiro256** (Blackman & Vigna 2018), seeded by SplitMix64.
struct DCMRandom: RandomNumberGenerator, Codable, Sendable, Equatable {
    /// The four 64-bit state words. Named, not a tuple, so `Codable` and
    /// `Equatable` are synthesized and the JSON form is self-describing.
    private(set) var s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64
    init(seed: UInt64)                      // SplitMix64 expands 1 word → 4; rejects the all-zero state (cannot occur from SplitMix64, asserted)
    init(state: DCMRandomState) throws      // restore; throws on all-zero
    mutating func next() -> UInt64          // RandomNumberGenerator requirement
    // Stable draws — OUR algorithms, pinned by golden tests:
    mutating func nextBounded(_ upperBound: UInt64) -> UInt64   // Lemire 2019 nearly-divisionless, unbiased
    mutating func nextBounded(_ upperBound: Int) -> Int          // precondition(upperBound > 0)
    mutating func nextUnitDouble() -> Double  // (next() >> 11) * 0x1p-53  ∈ [0, 1)
    mutating func nextUnitFloat() -> Float    // (next() >> 40) * 0x1p-24  ∈ [0, 1)
    mutating func nextStandardNormalPair() -> (Float, Float)   // Box–Muller with our own restricted-domain log/cos (A5, D-3)
    mutating func jump()                      // xoshiro256** 2^128 jump, for completeness/tests
}
```

- `Codable` form: `{"s0":"…","s1":"…","s2":"…","s3":"…"}` with the words written as
  **decimal strings** (JSON numbers are doubles in many readers; a UInt64 above
  2^53 would silently lose bits in Python tooling). Decoding validates: four
  parseable UInt64s, not all zero; otherwise a named error.
- Conformance to `RandomNumberGenerator` means every stdlib `using:` API works
  (`shuffled(using:)`, `randomElement(using:)`, `Int.random(in:using:)`).

### A2.2 Why our own bounded/unit draws

The stdlib's `using:` algorithms (`Int.random(in:using:)`, `Float.random(in:using:)`,
`shuffled(using:)`) are **not documented as stable across Swift versions** — the
implementation is free to change how many `next()` calls it makes and how it maps
them. A toolchain upgrade (this project builds with a beta Xcode) could therefore
change a "seeded" trajectory. For any draw that decides **training math** or must
match a **saved RNG state** on resume, we call our own `nextBounded`,
`nextUnitDouble`, `nextUnitFloat`, and our own Fisher–Yates.

### A2.3 Where each is used

| Use | Draw |
|---|---|
| Replay-buffer index picks, stratified slot picks | `nextBounded(held)` / `nextBounded(slots.count)` |
| Length-tilt acceptance, draw-keep | `nextUnitDouble()` |
| Move sampling (inverse CDF), Gamma/Dirichlet | `nextUnitFloat()`, `nextStandardNormalPair()` (Box–Muller — the Gamma sampler currently discards one normal per call; the pair form keeps the stream count explicit and the cached second value is part of the per-game state only within one call, so no extra state to persist) |
| Weight init | `nextStandardNormalPair()` per tensor stream (B1) |
| Corpus shuffle (if ever added), probe subsampling | our `stableShuffle(_:using:)` (Fisher–Yates with `nextBounded`) |
| Lichess matchmaking (already generic over `RandomNumberGenerator`) | stdlib `using:` acceptable — it stays on `SystemRandomNumberGenerator` anyway |

Rule written into `DCMRandom`'s doc comment: *stdlib `using:` APIs are acceptable
only where no saved state or golden result depends on the draw sequence.*

**Measured cost** (2026-09-30, `swiftc -O`, M4 Pro, 8.2M bounded draws into a
500k range): today's `Int.random(in:)` (system generator) **27.0 ns/draw**
(110.6 µs per 4096-sample batch); `DCMRandom`-style xoshiro256** + Lemire
**0.77 ns/draw** (3.2 µs per batch); xoshiro through stdlib `using:` 0.78 ns.
The seeded path is ~35× cheaper than today's; either is negligible next to a
training step (~2.2 s), where copying the sampled positions dominates. The
training and self-play loops must stay at least this fast: the P3 validation
re-measures `sample()` before/after.

### A2.4 Golden-sequence tests (`DrewsChessMachineTests/DCMRandomTests.swift`)

- SplitMix64 from seed 0 and from `0x0123456789ABCDEF`: first 8 outputs pinned to
  the reference C implementation's values (computed once from the reference
  `splitmix64.c`, checked in as literals).
- xoshiro256** from a fixed 4-word state: first 16 `next()` pinned to reference
  `xoshiro256starstar.c` output; after `jump()`, next 4 pinned.
- `nextBounded` for bounds {1, 2, 3, 7, 4864, 2^32+1, UInt64.max}: first 16 values
  pinned; plus a chi-square sanity on bound 3 over 3·10^6 draws (loose bound, not
  flaky).
- `nextUnitDouble`/`nextUnitFloat`: pinned values; assert range `[0,1)`.
- `nextStandardNormalPair`: pinned bit patterns (`Float.bitPattern`) for 8 pairs.
- `DCMNormalMath` (A5): **exhaustive** test of the private `log` over all 2²⁴
  values of `u1 = (k+1)/2²⁴` and of `cos` over all 2²⁴ angles `2π·k/2²⁴`, each
  against a `Double` reference, asserting the stated max-ULP bound (runs in
  seconds; not gated).
- `Codable` round trip, including a word > 2^53, and rejection of all-zero and of
  a numeric (non-string) word.
- Stream derivation (A3.1): pinned child seeds for names `"sampler"`, `"init"`,
  `"init/block3_conv1_weights"`, `"selfplay.game.17"`; and a test that **adding** a
  new name does not change any existing child seed (by construction, but pinned).

## A3. Seed plumbing

### A3.1 Master seed and named derivation

- One **master seed** (`UInt64`) per *lineage run* (Part D: `lineage_run_id`).
- `stableHash64(s) = UInt64(bigEndian: first 8 bytes of SHA-256(UTF-8(s)))`
  (CryptoKit, already used by `ModelDerivation`; trivially reproducible in Python
  with `hashlib`). **Never** Swift `Hasher` / `hashValue` / `String.hashValue`:
  they are randomly seeded per process, so they would silently make every
  "seeded" run different.
- Child seed for a stream named `name`:
  `childSeed = splitmix64(parentSeed ^ stableHash64(name))`, implemented once in
  `DCMRandomStreams.childSeed(parent:name:)` and used for **every** derivation in
  the app — run-level streams, per-game streams, per-tensor init seeds (B1) and
  per-chunk fill seeds (B1). One function, one golden-test file.
- Hashing the *name* (not an ordinal or a position in a list) is what makes streams
  **independent of declaration order and of membership**: adding `"arena"` later
  never shifts `"sampler"`; removing a stream shifts nothing.
- Rejected alternative: draw child seeds sequentially from one master generator
  over a sorted list of stream (or tensor) names. It is deterministic for a fixed
  list, but inserting or removing any entry shifts every later seed. For tensors
  this destroys cross-architecture sharing (adding `block5_*` would re-seed every
  tensor sorting after it, e.g. `policy_*`, `value_*`), and for run streams it
  makes adding a stream a silent trajectory change for existing ones.
- The derivation is part of the versioned `rng_derivation` / `init_scheme`
  identifiers (A3.2 / B1.1); changing it means a new identifier, never an edit.
- Hierarchical names with `/` or `.`: `init/<tensorName>`, `selfplay.game.<serial>`,
  `arena.<arenaIndex>.game.<gameIndex>`, `probe.<name>.<trainerStep>`.

Stream catalog (single source: `enum DCMStream` with a static `name` per case and a
`func name(...)` for parameterized ones):

| Stream | Owner | Persisted in checkpoints? |
|---|---|---|
| `init/<tensorName>` | graph builder, `--derive-model` | No — pure function of seed + name; the init seed is recorded. |
| `init.bn_calibration` | `ChessMPSNetwork.calibrateBNRunningStats` | No — same. |
| `sampler` | `ReplayBuffer` (one stream, under its existing `OSAllocatedUnfairLock`) | **Yes** — state after the last sampled batch. |
| `dropout` | trainer; seeds the Philox state variable | **Yes** — the 7×Int32 Philox state read back from the GPU (A4). |
| `selfplay.game.<serial>` | `ActiveGame` (one generator per game, created at game start) | No per-game state (in-flight games are dropped on save, C2); the **next game serial** is persisted. |
| `arena.<arenaIndex>.game.<gameIndex>` | `TickTournamentDriver` per game | No; `arenaIndex` (count of arenas so far) is persisted. |
| `vsuci.game.<serial>` | `TrainVsUciDriver` per game | No; next serial persisted. |
| `probe.<name>.<trainerStep>` | probes | No — a function of step. |
| `corpus.order.<epoch>` | reserved for a future corpus shuffle (today order is sequential — `CorpusReplayRunner`) | Would be a function of epoch; no state. |

Per-game streams (rather than one shared self-play stream) are essential: the tick
driver samples all K games in parallel inside a `withTaskGroup`
(`BatchedSelfPlayDriver.swift:501-550`, `TickTournamentDriver.swift:622-658`); a
shared stream would make draws depend on task scheduling order. Per-game streams
make each game's draws a function of `(seed, serial, its own history)` and need no
lock. Game serials are assigned in the serial game-start pass, so they are
deterministic given the tick order.

### A3.2 Parameters and flags (full CLAUDE.md checklist)

Two new `@TrainingParameter`s — **not** live-tunable:

| id | Type | Meaning |
|---|---|---|
| `random_seed_mode` | enum `seeded` \| `unseeded` | **Required choice, no default that hides it.** Default value `unseeded` to keep today's behavior for existing users, but see "no silent defaults" below. |
| `random_seed` | UInt64 (as decimal string in JSON) | Used when mode = `seeded`. Ignored (and logged as ignored) when `unseeded`. |

Plus CLI override `--seed <UInt64>` (sets mode = seeded for that process; logged as
`[PARAM] random_seed from --seed`) on `--train-from-corpus`, `--train-vs-uci`,
`--train`, and `--derive-model` (for graft init, B3).

**No silent default:** in `unseeded` mode the app still *draws* a master seed from
`SystemRandomNumberGenerator` at run start and uses the seeded machinery with it —
there is exactly **one code path**. The drawn seed is logged, stored in every
checkpoint, and reusable with `--seed` to replay the run. "Unseeded" means "the
seed was chosen for you", never "no seed".

Checklist walk:

1. **Declare** both with `@TrainingParameter`, add to `allKeys` (`Training/TrainingParameters.swift`). Range for `random_seed`: full UInt64 — the macro's numeric definition needs a UInt64 (or string-encoded) kind; if the macro only supports Int/Double today, add a `UInt64` value kind to `Packages/TrainingParametersMacro` (Phase 1 task) rather than squeezing into Int64.
2. **Singleton**: stored properties + `collectValues` / `applyOne` + snapshot accessors.
3. **`parameters.json`**: verify in `--show-default-parameters` and `--create-parameters-file` round trip.
4. **Session** — *not* a `[RESUME-PARAM]` block like other parameters: on resume the seed is **lineage state**, restored from the checkpoint's `dcm_lineage.rng.master_seed` (D2), never from current settings. The `[RESUME-PARAM]` block logs `random_seed: <saved> (lineage; current setting <x> ignored)`; a new-format session missing it is a load error (format rule); a legacy session gets `NOT EXACT: rng_sampler, dropout_state, …` (C3) and a freshly drawn seed, logged.
5. **`results.json`**: add `random_seed`, `random_seed_mode`, `rng_stream_derivation: "v1"`.
6. **Runtime log**: `[RUN] seed=<u64> mode=seeded|unseeded(drawn) derivation=v1` at run start (D4), and on resume `[RESUME] rng: sampler=restored dropout=restored`.
7. **UI** (D-2): Settings popover ▸ Training tab ▸ new "Reproducibility" row: mode segmented control (default `unseeded`) + seed text field (monospaced, validated) + "Copy seed" of the *current run's* seed; disabled while a session is running (not live-tunable). `TrainingSettingsPopoverModel` binding + validation. **Build New Model** gets its own optional **Init seed** field (monospaced, validated UInt64; empty ⇒ a seed is drawn and shown after the mint): it is the model's `init_seed` (B1.1), recorded in the file, and is separate from the run's master seed. The drawn or entered init seed is logged with the `[BUTTON]` build line.
8. **Live tunability**: `liveTunable: false` for both; the seed is fixed per lineage.
9. **Renames**: n/a (new).

Plus from #5 (C5): each declares `absentValue` — `random_seed_mode` absent ⇒ `unseeded` (the pre-feature behavior); `random_seed` absent ⇒ `refuseExact` (there is no pre-feature seed; exact resume is impossible and says so).

### A3.3 Threading the streams (code shape)

- `DCMRandomStreams` (value type, `Sendable`): holds `masterSeed`; `func generator(_ stream: DCMStream) -> DCMRandom`.
- `ReplayBuffer`: gains `private var samplerRNG: DCMRandom` guarded by the existing `lock`; `func samplerState() -> DCMRandom` / `func restoreSamplerState(_:)` under the lock. `MaterialBucketSlots.randomSlot(using:)` takes `inout DCMRandom`.
- `MoveSampler.sampleMove(..., rng: inout DCMRandom)` — all static helpers take `inout DCMRandom`; no hidden global.
- `ActiveGame` owns `var rng: DCMRandom`; the tick's parallel sample uses the game's own generator (games are already disjoint per task, so no lock).
- `ChessNetwork.init(arch:bnMode:initialization:)` where `initialization` is `WeightInitialization` (B1).

## A4. MPSGraph dropout determinism

How it works today (`Network/ChessNetwork.swift:747-867, 2684-2729`):

- The generator is **already stateful**: a graph **variable** `dropout_rng_state` of shape `[7]` Int32 (Philox state: key + counter words, as MPSGraph lays it out; `ChessNetwork.swift:775-784`). Each step continues the sequence where the previous one ended. The **only** nondeterministic input is the initial seed `Int.random(in: 0..<Int.max)` at `:781`; making the whole dropout sequence reproducible means replacing that one seed and saving/restoring the 7 words.
- `randomPhiloxStateTensor(withSeed: Int.random(in: 0..<Int.max))` creates a **constant** in the graph; `dropout_rng_seed_assign` copies it into the variable. The trainer runs that assign once per built graph (`ChessTrainer.swift:2198`, `:2410` on rebuild, `runDropoutSeedOnQueue` `:5712`).
- Each block's `randomTensor(withShape:descriptor:stateTensor:)` returns `[values, nextState]`; the state is threaded block to block, and `dropout_rng_advance` assigns the final state back to the variable after each step. The KL-probe path also runs the advance op (`ChessTrainer.swift:6900-6910`); whether that is the step's own single advance or an extra one is settled in P4 under the probe-isolation rule (C2), and the probe schedule is made step-derived either way (C1 #3).

Can it be seeded? **Yes.** Replace the constant with an input: add placeholder
`dropout_rng_seed_state` `[7]` Int32 and an assign from it; the seed op is fed a
`MPSGraphTensorData` built from our state words. To derive the initial state:
build it with `randomPhiloxStateTensor(withSeed:)` in a tiny one-off graph at
trainer setup (seed = `dropout` stream's first `next()` as `Int`), read the 7
words back, and feed them — or equivalently keep the seed-constant graph but
build it per seed (rejected: rebuilding/recompiling the executable per seed).

Can the state be captured and restored? **Yes.** Capture: run the graph with
`targetTensors: [dropout_rng_state]` (the same pattern the seed op already uses,
feeding `inputPlaceholder` the dummy batch) and read 7 Int32s. Restore: feed the
saved 7 words through the new placeholder assign. Persisted as
`dcm_lineage.rng.dropout_philox_state` = JSON array of 7 Int32 (D2). Test: capture →
run 3 steps (record masks via a tap) → restore → 3 steps → identical masks.

Caveat stated honestly: the **meaning** of the 7 words is MPSGraph's; we treat
them as an opaque blob. An OS update could in principle change the Philox layout
or the mapping from state to uniforms. The golden test "same Philox state →
same first mask" (§E P4) is the canary; a failure after an OS update means
"dropout streams are not comparable across that OS boundary", logged, not hidden.
The OS build is recorded in every lineage record (`device.os_version`, D2) for
exactly this reason, and an OS change on resume is flagged (C1 #33).

Inference graphs have no dropout scaffolding, so nothing changes for them.

## A5. GPU numeric nondeterminism — what cannot be bit-exact

- **MPSGraph does not document run-to-run determinism.** Reductions (`reductionSum`, `mean`, BN batch stats, gradient accumulation for shared conv weights) may use parallel trees or atomics whose order depends on dispatch; float addition is not associative. On the same device/OS/batch it is *often* deterministic, but that is an observation to measure (§C6 "determinism probe"), not a guarantee.
- **Kernel selection varies by shape, device and OS.** Different batch sizes pick different conv algorithms (the Winograd investigation in `ConvKernelExecutionPathNumericsTests` shows paths differ numerically). A resumed run must use the same batch size and precision to have any hope of bit-equality.
- **bf16 rounding** amplifies any upstream difference: a 1-ulp fp32 difference can flip a bf16 rounding, then propagate.
- **Across OS versions / chips** (M4 Pro vs M5 VM, macOS 26 vs 27 beta): not bit-exact, full stop. The lineage metadata records device + OS (Part D) so comparisons can be scoped.
- **CPU side — solved by design (D-3, decided).** Today's init uses vForce `vvlogf`/`vvsqrtf`/`vvcosf` and `MoveSampler` uses libm `log`/`cos`; both are system libraries whose results are only ~1-ULP accurate and can change with an OS update or per chip. Decision: **keep He-normal / Glorot-normal** (continuity with every existing run; no distribution change across eras) and compute them with **our own restricted-domain Box–Muller** (`Utils/DCMNormalMath.swift`, private to `DCMRandom`): polynomial `log`/`cos` built only from IEEE-exact `+ − × ÷` and `sqrt`, so a seed gives bit-identical values on every OS build and chip.
  - **Domain guaranteed by construction:** `u1 = (k+1)/2²⁴ ∈ (0,1]` (never 0, never denormal, NaN or infinite) and angle `2π·k/2²⁴ ∈ [0,2π)`, with `k` from `next() >> 40`. No other caller can pass other inputs, so no general-purpose range reduction or special-case handling is needed. (This replaces today's clamp of `u1` to `leastNormalMagnitude`.)
  - **FMA:** Swift does not contract `a * b + c` into a fused multiply-add on its own, so the explicit operation order is stable across builds; the golden bits catch any change.
  - **Validation:** (1) exhaustive test over all 2²⁴ inputs of each function against a `Double` reference, asserting a stated max-ULP bound (A2.4); (2) golden output bits for fixed seeds (A2.4, B1.1); (3) **re-benchmark on completion** against vForce and Swift libm and report the numbers. Prototype (2026-09-30, 8.45M values, `swiftc -O`, M4 Pro): vForce 10.4–13.9 ms, Swift libm 31.5 ms, draft polynomial 10.6 ms. Init runs once per mint, so speed is not a deciding factor.
  - **Re-benchmark of the finished transform (P5, 2026-10-02):** same harness shape, 8,445,748 values including the xoshiro draws, `swiftc -O`, best of several runs, on the M4 Pro while other builds and test runs were loading it (load average ≈ 4.4), so the absolute numbers are inflated against the idle prototype; the ratios are the useful part. `DCMRandom.nextStandardNormalPair`: 45.6 ms (earlier run 55.6); vForce Box–Muller: 33.6 ms (39.0); Swift libm (`Double`): 60.7 ms (74.8). Ours is ≈ 1.35× vForce and ≈ 0.75× libm. A re-run on an idle machine is still worth doing if the absolute figure ever matters.
  - **Not a concern — bf16 narrowing:** bf16 training keeps fp32 master weights (fp32 master + bf16 working copy; config D, issue #9, also stored fp32 and is removed anyway under D-10), so init always produces fp32 values, and the one CPU narrowing (`float32ToBFloat16Bits`, `ChessNetwork.swift:3572`) is a fixed round-to-nearest-even rule pinned by B1.1's goldens.
- **Self-play/GUI concurrency** (§0) — not numeric, but the dominant source of trajectory divergence.

Conclusion: the harness asserts **exact equality of every CPU-side state and of
all RNG draws**, and **bit-exact weights only where a same-process determinism
probe first proves MPSGraph is deterministic for that config**; otherwise it
asserts a stated tolerance (§C6).

## A6. Order dependence

A seeded stream only helps if the **order of draws** is fixed. Every site where
draw order, consumer identity or iteration order could vary:

| # | Site | Order hazard | Treatment |
|---|---|---|---|
| O1 | `Training/BatchedSelfPlayDriver.swift:501-550` (parallel per-game `MoveSampler.sampleMove` in `withTaskGroup`) | Task completion order is scheduler-dependent; a shared stream would interleave draws arbitrarily. | Per-game stream `selfplay.game.<serial>` owned by `ActiveGame` (A3.1). Each game's draws happen only inside its own task, in ply order. No shared generator. |
| O2 | `Arena/TickTournamentDriver.swift:622-658` (same pattern) | Same | Per-game `arena.<arenaIndex>.game.<gameIndex>`. |
| O3 | `Training/TrainVsUciDriver.swift:333` | Games run concurrently against external engines | Per-game `vsuci.game.<serial>`. |
| O4 | Game-serial assignment — new games are appended in `BatchedSelfPlayDriver` grow path and after the serial game-end pass (`:658` loop `for i in 0..<K`); shrink drops the tail (`:289 games.removeLast`) | Serial numbers must be assigned in a fixed order: the serial pass iterates slots in index order, so assign serials there (never inside the task group). Live worker-count changes (GUI) make the *set* of games time-dependent — accepted, §0. | Assign `serial = nextSerial; nextSerial += 1` only in the serial pass, in slot-index order; persist `nextSerial`. |
| O5 | `Training/BatchedSelfPlayDriver.swift:700` draw-keep | Runs in the serial game-end pass | Draw from the finishing game's own stream (not a driver-level stream), so it's independent of how many other games finished in the same tick. |
| O6 | `Training/ReplayBuffer.swift` `sample()` paths `:1783, 1904, 1924, 1973, 2154, 2189, 2198, 2235` | Multiple consumers could sample the buffer (trainer; `ReplayBufferAnalyzer` probes; batch-stats) | The `sampler` stream is used **only** by the trainer's minibatch draw, under the buffer `lock`, in batch-slot order (`i = 0..<sampleCount`). Every other consumer (analyzer/probes) gets its own named stream (`probe.<name>.<step>`), never `sampler`. |
| O7 | Stratified path `:1900-1915` iterates buckets `for b in 0..<bucketCount` then `randomSlot()` | Deterministic order, **but** `MaterialBucketSlots.slots` order is itself history-dependent (swap-remove `:280-286`, insertion order `:1223-1231`) — same RNG, different slot order ⇒ different picks after a rebuild. | **Decided (D-5):** the pick becomes "the k-th oldest slot in the bucket" (age rank, not stored array position), and the arrays are rebuilt in the buffer's **age order** on refill (C1 #4, #5). Nothing extra is persisted. |
| O7b | All `sample()` paths draw a **physical** slot index (`Int.random(in: 0..<held)` then `outcomeStorage[srcIndex]`, e.g. `:1783-1790`) | Physical slot numbering depends on where the ring's write pointer happened to be, so the same RNG state picks different positions after a refill that starts at another slot. | Draw a **logical** (age-ordered) index `i` (0 = oldest) and map it once: `physical = (oldest + i) % capacity`, with `oldest = writeIndex` when full and `0` otherwise. Batches then depend only on the buffer's contents in age order, never on the ring's physical offset (C1 #4). |
| O8 | Length-tilt rejection loop `:2150-2235` | Number of draws per emitted sample varies with acceptance; fine as long as the loop order is fixed (it is: sequential) | Keep sequential; document that the draw count is data-dependent (so the saved state is the only valid restore point, never "N draws since step X"). |
| O9 | Dictionary iteration in `sample()` — `for (_, c) in degPerGame/stratPerGame/fastPerGameScratch/perGameCount` `:1794, 1937, 1984, 2243` | `Dictionary` iteration order is randomized per process in Swift | **Checked: max-reductions only, no draws inside** ⇒ order-independent. Rule added to `DCMRandom` docs and a code-review checklist item: *no random draw inside iteration over `Dictionary`/`Set`*; if ever needed, iterate `keys.sorted()`. Other dictionaries in the buffer (`residentGames :199`, `hashStats :176`, `residentLengthHistogram :210`, `slotPosition :260`) are lookup/stat only. |
| O10 | `ReplayBuffer` insert order (GUI) — games flushed in the serial pass `for i in 0..<K` | Deterministic per tick given which games finished — concurrency with the trainer makes buffer *timing* nondeterministic (§0) | Accepted; noted. |
| O11 | Per-tensor init (B1) | Parallel fills | Per tensor sequential in `dcm-init-1`; any intra-tensor parallelism must use fixed-size chunk seeds (B1). Tensor iteration order is irrelevant by construction (name-derived seeds). |
| O12 | BN calibration random game (`Network/ChessMPSNetwork.swift:146-160`) | Single sequential walk | `init.bn_calibration` stream; `legalMoves` order from `MoveGenerator` must be deterministic — **verify** it's array-ordered by board scan (it is generated into an array; confirm no `Set` in its path), since `randomElement` maps an index to a move. |
| O13 | `MoveSampler` inverse-CDF over `legalMoves` order | Same dependence on move-generation order for a given index | Same verification as O12; add a test that `legalMoves(for:)` order is stable for 20 fixed FENs (pinned). |
| O14 | Dropout Philox state threaded block→block in graph build order (`ChessNetwork.swift:2684-2716, 858-859`) | Order is the graph's data dependency chain — fixed for a given architecture; KL probe adds extra advances (`ChessTrainer.swift:6906`) | Deterministic given step-derived probe schedule (C1 #3). Changing block order changes masks — inherent, arch-specific. |
| O15 | Run-level sub-streams | Adding a stream must not shift others | Name-hash derivation (A3.1) for all of them. |
| O16 | **Swift `Hasher` values used as data** — `ReplayBuffer.hashBoard` (`Training/ReplayBuffer.swift:649-657`) hashes each board with `Hasher()`, which is randomly keyed **per process**; the per-slot hashes are **persisted and restored** (`stateHashStorage`, written `:2937`, read `:3377`, re-counted into `hashStats` `:3405-3415`) | After a resume, restored slots carry the old process's hashes while new inserts of the same position get the new process's hash ⇒ duplicate counting splits one position into two keys. Its doc comment (`:643-647`, "the hash dict is rebuilt fresh … cross-process stability isn't required") is wrong: the dict is rebuilt, but from persisted hashes. **Checked 2026-09-30: observability only** — the hashes feed `uniquePositionCount`, `bufferedPositionStats(forHash:)` and `computeBatchStats` (`ChessTrainer.swift:4555-4584`), never a `sample()` path or a loss, so no training-math impact. | Replace with a fixed-key hash (SplitMix-style mix over the board bytes, or SipHash with a constant key), one function, documented as persistence-stable. Bug-fix rule: the regression test (hash of a fixed board equals a pinned constant, and a save → restore → re-insert of the same position yields count 2 under one key) is written first and must fail before the fix. Legacy sessions' persisted hashes come from random keys: **decided (D-9)** — when loading a buffer written before the fixed-key hash — `ReplayBuffer.fileVersion` goes 7 → 8 (`Training/ReplayBuffer.swift:2697`; v8 = fixed-key hashes + the slot-source column, C1 #4) and the strict `version == fileVersion` checks (`:586`, `:625`, `:3112`) widen to accept 7 as legacy — recompute every slot's hash from its stored board and rebuild `hashStats`, logging `[RESUME] recomputed N position hashes (legacy buffer)`. Owner-approved migration; the boards stay the source of truth. Test: a legacy fixture buffer loads with hashes equal to the fixed-key hash of each board. |
| O17 | **Whole-codebase sweep of `Set`/`Dictionary` iteration and hash-values-as-data** (O9 checked `sample()` only) | Any draw, emitted order, or persisted value that depends on per-process hashing is irreproducible. | P3 task: grep every `for … in <Dictionary/Set>`, `.keys`/`.values` iteration, `Set` → `Array` conversion and `hashValue`/`Hasher` use on the training, replay, self-play, arena, vs-UCI and persistence paths; each is classified (order-independent reduction / sorted / fixed) in a table appended here. Known safe: `TrainingParameters.swift:1508` and `CliTrainingConfig.swift:103` iterate `keys.sorted()`. |
| O18 | **Sort ties** — Swift's `sort` is stable in practice but not documented as stable | A sort with equal keys that decides data order (shard lists, bucket rebuilds, emitted records) could reorder between toolchains. | Rule: every sort that decides data order uses a total order (explicit tie-break on a unique key). Audited 2026-09-30 on data paths: corpus shard lists sort by unique `lastPathComponent` (`GameCorpus.swift:274, 360`) — no ties; `highestShardSeq` (`:374-384`) is a max, order-free; `SafetensorsFile.swift:192` sorts by unique byte offset. The rest found (`ReplayBufferAnalyzer`, `NumericsAudit`, `ModelFileCatalog`, `ModelLineageTree`, `CheckpointManager.swift:208`) are statistics or UI listings. P3 re-runs the sweep and records it. |
| O19 | **Float sums accumulated in thread-completion order** that feed training | Float addition isn't associative, so a thread-order sum varies by ULPs between runs; if it feeds training math, bit-exact replay breaks with no visible cause. Matters only for corpus replay (the only trajectory-exact path). | Preliminary check 2026-09-30: no `concurrentPerform`/`TaskGroup` in `CLI/CorpusReplayRunner.swift`, `Training/ReplayBuffer.swift`, `Training/ChessTrainer.swift` or `Persistence/GameCorpus.swift`; parallel regions exist only in self-play, arena, vs-UCI, UCI, the recorder and the Lichess bot. P3 confirms by reading the replay feed path (`CorpusReplayFeeder`) end to end and records the result; any such sum found on the replay path gets a fixed-order reduction. |

---

# PART B — Ablation- and derivation-friendly initialization

## B1. Per-tensor-derived initialization

Today every He/Glorot tensor draws from one unseeded `arc4random_buf` stream
(`ChessNetwork.swift:3363-3431`) in graph-build order, so (a) nothing is
reproducible and (b) two architectures that share a tensor get different values
for it even if seeded, because build order shifts the stream.

Design:

- `enum WeightInitialization: Sendable { case seeded(initSeed: UInt64); case overwrittenByLoad }`.
  - `.seeded` — every random tensor `name` (the **stable name from `weightTensorPlan()`**, e.g. `block3_conv1_weights`, `block3_se_fc2_weights`, `value_wdl_fc2_weights`) draws from its own `DCMRandom(seed: tensorSeed)` (B1.1). Values = `std(role, fanIn, fanOut) × N(0,1)` filled **row-major in the stored (on-disk) layout** (B1.1 item 5), then converted to the native layout `makeWeightData` receives.
  - `.overwrittenByLoad` — for networks whose variables are immediately replaced by `loadWeights` (session resume, `--start-model`, inference mirrors, arena networks, BN-calibration sibling `ChessMPSNetwork.swift:185`). Variables are built zero-filled; **no RNG work**. This removes today's wasted init of every loaded network. It is not a stub: it is a real mode whose contract ("every variable is overwritten before first use") is enforced by the existing tensor-count/shape checks in `loadWeights` plus a new debug assertion that a `.overwrittenByLoad` network refuses `evaluate`/`trainStep` until a load has happened.
- Deterministic constants stay constants: BN γ=1/β=0, running mean 0/var 1, biases 0, ReZero α = `rezeroAlphaInit`, WDL bias prior `[0, ln 6, 0]`, zero-β halves.
- **bf16/fp32 storage**: draws are always generated in **fp32**, then `makeWeightData(_, dataType:)` narrows exactly as today. The same seed therefore yields the same fp32 values in both precisions; bf16 files hold the round-to-nearest of them. The fp32 masters under mixed precision are seeded from the bf16 working weights (`runSyncMastersOnQueue`), unchanged.
- **Sharing**: two architectures with the same `(initSeed, tensor name, shape)` get bit-identical values. With the same name but a different shape (e.g. wider conv), the tensors share the leading run of normals but a different std/fan-in; documented as "not shared", and `graft` (B3) never treats them as matching.
- **Name stability is load-bearing.** `weightTensorPlan()` names are already the safetensors tensor names and the contract for file loading, so they are stable. But block names are **indexed** (`block<i>`): inserting a block before others renames every later block. That is correct for "architectures sharing tensors" (block 3 is block 3), and the graft op (B3) supports an explicit name map for the insert case.
- **Existing models**: loading never initializes randomly (it uses `.overwrittenByLoad` after this change; before it, the random init was overwritten anyway), so **every existing file loads bit-identically**. Init affects only new mints. The init seed of a new mint is recorded (`init_seed`, Part D); legacy files simply have none.
- **Performance** (~8.45M params for `v4_5block_7x7`; larger presets ~30M): measured prototype of the transform alone (A5): our restricted-domain polynomial 10.6 ms for 8.45M values vs vForce 10.4–13.9 ms and Swift libm 31.5 ms; xoshiro draws add ~0.8 ns/value (A2.2). Expected total well under 100 ms at 8.45M, single thread — re-measured on completion (A5). Build-once, so acceptable either way; if ever needed, tensors are independent streams, so `DispatchQueue.concurrentPerform` over tensors parallelizes **without changing values**. Parallelism *within* one tensor is allowed only with per-chunk seeds `splitmix64(tensorSeed ^ stableHash64("chunk/<index>"))` over a **fixed** chunk size that is itself part of the `init_scheme` — never "split among however many threads are available". `dcm-init-1` fills each tensor sequentially (simplest; parallel across tensors only). vForce is **not** used: its values could change with the OS (A5, D-3).

`NetworkWeightAnalyzer` / `NumericsAudit` expectations (analytic std per role) are unchanged because the distribution is unchanged.

### B1.1 Per-tensor seed and the versioned `init_scheme`

- `tensorSeed = splitmix64(initSeed ^ stableHash64(tensorName))` (A3.1's single
  derivation function), where `tensorName` is the UTF-8 safetensors tensor name
  from `weightTensorPlan()`. Independent of tensor list order and of which other
  tensors exist, so shared tensors match across architectures.
- **`init_scheme` identifier** — e.g. `"dcm-init-1"` — recorded in every fresh
  mint's and graft's metadata (`dcm_lineage.rng.init_scheme`, and per graft in
  `derivation_history`). A scheme ID pins **all** of:
  1. the name hash (`stableHash64` = SHA-256 first 8 bytes BE) and the derivation (`splitmix64(seed ^ hash)`);
  2. the generator (xoshiro256** seeded by SplitMix64, A2);
  3. the per-role distributions (He-normal std `sqrt(2/fanIn)` with each role's fan-in definition; Glorot std; the exact constants for γ/β/biases/α/WDL prior; the zero-β column ranges);
  4. the normal transform: Box–Muller with `u1 = (k+1)/2²⁴`, angle `2π·k/2²⁴`, and our own `DCMNormalMath` polynomial `log`/`cos` (exact coefficients and operation order — A5, D-3);
  5. the **in-tensor fill order: row-major in the stored (on-disk safetensors / PyTorch) layout** — for FC weights that is `[out, in]`, for convs OIHW — so a value's position is defined by the file, not by the in-memory native layout; the builder fills in stored order and converts with the existing `toTorchLayout` inverse;
  6. the fp32→bf16 rounding (round-to-nearest-even, as `makeWeightData` does today — pinned by test) and the rule that draws are always fp32.
- **A published scheme never changes.** Any change to any item above gets a new
  scheme ID (`dcm-init-2`), and the builder keeps the old scheme available so a
  graft/derive can reproduce a recorded scheme. Unknown scheme ID on a request to
  reproduce ⇒ error, not fallback.
- **Golden tests per scheme** (`DrewsChessMachineTests/InitSchemeGoldenTests.swift`):
  for `dcm-init-1`, seed `42`, pinned `tensorSeed` for 6 names; pinned first 8
  fp32 bit patterns and the SHA-256 of the full tensor for a conv, an FC, an SE
  fc2 with zero-β, and the WDL fc2 bias; the same tensors in bf16. Any change to
  the transform (A5) or the fill order breaks these pins, which is the point.
- **Cross-machine init test** (owner-run script, `scripts/init_reproducibility.sh`):
  for a set of seeds (e.g. 8) × the default preset and one heterogeneous preset,
  mint fresh models on **each** machine (M4 Pro native, M5 VM) and compare the
  SHA-256 of the **tensor data only** (every mint gets a fresh ModelID and
  timestamp, so whole-file hashes always differ); all must match. This is the direct proof of the
  "bit-identical on every OS and chip" claim; the result (machines, OS builds,
  seeds, hashes) is recorded in the experiment/plan notes.
- Fresh checkpoints committed via Git LFS (the experiments convention) remain the
  ultimate record of what a seed produced; the scheme makes them reproducible,
  the file makes them authoritative.

## B2. Init-neutral / ablation options audit

"Init-neutral" = the last layer of a path that is **added** into a residual stream
or an output, initialized so the path contributes nothing (or a known prior) at
step 0 while still receiving gradient.

| Path (`ChessNetwork.swift`) | Today | Zero kills learning? | Recommendation |
|---|---|---|---|
| **SE β half** (fc2 cols `C..<2C` + bias) `:2927-2951` | `se_beta_init: glorot\|zero` per group (#7) | No — input (fc1 act) is nonzero, so β gets gradient | **Done.** Keep. |
| **SE γ bias** (fc2 bias `0..<C`) `:2949` | 0 ⇒ `σ(0)=0.5` gate at init: every SE block halves its branch | Zero is today's value. The ablation is the *level*, not zero: e.g. bias `+2.2` ⇒ gate ≈ 0.9 ("near-identity SE") | Offer `se_gamma_bias_init: <float>` per group, default `0` (legacy value). Both scale_and_bias and attenuate_only. Not zero-vs-random; a constant. |
| **SE γ weights** (fc2 cols `0..<C`) | Glorot | Zeroing makes gate input constant ⇒ gate still gets gradient through the bias only; γ weights get gradient ∝ fc1 act, so learning survives, but SE degenerates to a learned per-channel constant at init | **Don't offer** as a separate option; `se_gamma_bias_init` covers the useful ablation. |
| **ReZero α** `:2807-2840` | `rezeroAlphaInit` per group (already an option) | **Zero is a known dead-start trap here**: with the tanh-ceiling reparam (`tanh(α/c)·c`, `c = α₀·multiple`) an α₀ of 0 makes `c = 0` ⇒ division by zero; and plain ReZero at α=0 gives the branch weights zero gradient at step 0 (gradient flows only to α). | Keep existing float option; **validate `> 0`** when the tanh ceiling is on (verify `validate()` does this; add if not). Do not add a "zero" choice. |
| **Block conv2 / last conv of branch** `:2738` | He | Zero on the last conv with BN after it: BN of a zero tensor ⇒ 0 ⇒ fine, gradient to conv2 is nonzero (input nonzero). Standard "zero-init last BN γ" (Goyal et al.) is the cleaner version. | Offer `branch_output_init: standard \| zero_last_bn_gamma` per group — zeroes the γ of the branch's **last** BN (post-act style) or equivalent. Gradient reaches γ; all conv weights still random. Not offered on pre-act blocks where no BN follows the last conv (there, zeroing the conv would zero the gradient of every earlier conv at step 0 — they'd only start after conv2 moves; allowed but slower; recommend not offering). |
| **Skip projection** `_skip_proj` (1×1, width transitions) `:2850-2876` | He | **Must not be zero**: it *is* the identity path at a width change; zero ⇒ block output = branch only ⇒ no identity signal. | Offer `skip_projection_init: he \| identity_like` (identity on the first `min(inC,outC)` channels, zeros elsewhere — "partial identity", the WRN/ResNet-D analog). Per group transition. |
| **`res_ln`** (post-merge LayerNorm) `:2881-2890` | γ=1, β=0 | Zero γ ⇒ block output 0 ⇒ **the residual stream is destroyed** (res_ln is on the main path, not a branch). | **Must not be offered.** |
| **Policy head final** (`policy` conv/fc) `:3032-3111` | He weights, zero bias | Zero weights ⇒ uniform logits (entropy = log 4864 exactly, legal-masked uniform); gradient to the final layer is nonzero (input nonzero), earlier head layers get zero gradient for step 0 only. Safe. | Offer `policy_head_final_init: he \| zero` (arch-level head option). Uniform-start policy is a legit ablation (lc0 does small-std). |
| **Value WDL fc2** `:3233-3249` | He weights + bias `[0, ln 6, 0]` | Zero weights ⇒ output = bias prior exactly; fc2 gets gradient. Safe. | Offer `value_head_final_init: he \| zero`. |
| **Value draw prior** (fc2 bias) `:3242` | hardcoded `ln 6` ⇒ p(draw)=0.75 | Prior of 0 ⇒ uniform 1/3. Not a kill. | Offer `value_head_draw_prior: <prob in (0,1)>` with legacy value `0.75`; bias = `[0, ln(2p/(1-p)), 0]`. Single source: a function `wdlBiasPrior(drawProbability:)`, replacing the literal. |
| **Feature-skip fusion conv** `:909-923` | He, then BN + act | The fused path *replaces* the head input (not an additive branch) ⇒ zero conv ⇒ BN of zeros ⇒ constant head input ⇒ **kills the heads' input signal**. | **Must not be offered.** |
| **Stem** `:693` | He | Main path. | Not offered. |

Expression: every per-block option lives on **`BlockGroup`** (the per-group recipe
already carries `seBetaInit`, `rezeroAlphaInit`, `dropoutMultiplier`), so groups
can differ. Head options live on `NetworkArchitecture`'s head sections. Every new
field is **format-gated** exactly like `se_beta_init` (#7): required from the new
format version (B/D share one bump, §D6), legacy ⇒ the old value, logged once per
load via `ArchitectureFormat.LegacyResolutionLog`. Legacy arch hashes/summaries stay
byte-identical (the #7 rule: a field at its legacy value is omitted from summary
text and hash inputs... **verify** how #7 kept hashes identical and use the same
mechanism).

**Decided (D-4, 2026-09-30): ship the recommended set** — `se_gamma_bias_init`,
`branch_output_init` (`standard | zero_last_bn_gamma`), `skip_projection_init`
(`he | identity_like`), `policy_head_final_init` (`he | zero`),
`value_head_final_init` (`he | zero`), `value_head_draw_prior`. The "must not be
offered" rows stay forbidden (validation rejects them with a named error), and
ReZero α gets the `> 0` check under the tanh ceiling.

### B2.1 Build New Model UI and `--derive-model` support (D-4)

- **One field per option** on `BlockGroup` (per-block options) or the head
  sections of `NetworkArchitecture` (head options), required from format v5,
  shown in presets, `architectureSummary` and `ArchitectureDiagramView` — the
  same treatment `se_beta_init` got in #7.
- **"Neutral init" button** in Build New Model: sets every option in every group
  and head to its neutral value (`se_gamma_bias_init` = the near-identity level,
  `zero_last_bn_gamma` where a BN follows the branch's last conv,
  `identity_like` at width transitions, both head finals `zero`). **The draw
  prior is deliberately excluded (owner-confirmed 2026-09-30)** and keeps its
  current value. Rationale: every neutral option makes a branch start as a no-op
  ("adds nothing") so gradient shapes it from zero; the draw prior is instead a
  claim about the data (the initial W/D/L prediction), and even uniform 1/3 is a
  claim — there is no neutral value. With `value_head_final_init = zero` the
  value head's initial output *is* the prior exactly, so it must stay an explicit,
  visible per-experiment choice rather than something a button changes silently.
  The right value is usually the training data's own draw rate (the elite corpus
  and the lichess corpus differ widely from each other and from the legacy
  `ln 6` ⇒ 0.75 default).
  **"Standard init" button**: resets every option to its standard (legacy) value.
  One function each on `BuildNewModelModel`, so the two buttons are the single
  source of what "neutral" and "standard" mean; `[BUTTON]` logs which was pressed.
- **Highlight vs standard:** every field whose value differs from the standard
  init is highlighted in the per-group editor and in the diagram — an accent
  tint plus a marker glyph (not color alone, for accessibility), in the theme's
  semantic accent in light and dark mode. The comparison is always against the
  standard value, never the last edit, so reopening a model or preset still shows
  what is non-standard.
- **Tooltips:** each highlighted field states what changes at step 0 (e.g.
  "final policy layer starts at zero → uniform policy at step 0"; "last BN γ = 0 →
  the branch adds nothing at step 0; γ still learns").
- **Views:** new UI pieces are their own `View` structs in their own files under
  `App/UpperContentView/` (project rule: one View per file, no helper
  `some View` properties).
- **`--derive-model`:** each option gets a shape-preserving derive operation
  (`--set-<option> <value> [--group <index>]`), re-initializing **only** the
  tensors that option owns under the `init/<name>` stream and a recorded seed,
  copying every other tensor bit-exact, and appending a `derivation_history`
  record — exactly as `--set-se-beta-init` does (#7). A convenience
  `--set-neutral-init` applies the same neutral set as the button, through the
  same function.

## B3. `--derive-model` graft operation

Extend `Persistence/ModelDerivation.swift`'s catalog with a new
`DeriveOperationKind` `graft`:

```
DrewsChessMachine --derive-model --from <ref.safetensors> --graft-to <arch preset name | arch.json>
    [--graft-map <old=new,...>] [--seed <u64>] [--init-overrides <field=value,...>] --out <file>
```

Semantics:

1. Target architecture = the named preset/JSON (validated; format version current).
2. For each tensor in the **target** plan: if the reference has a tensor with the
   same name (after `--graft-map` renames) **and identical shape**, copy bit-exact
   (in on-disk layout, as `derive` already does). Otherwise initialize it by its
   **role** (B1/B2 rules, including any per-group init options in the target
   arch) from `init/<targetName>` under `--seed` (drawn and logged if absent —
   no silent default; it is recorded either way).
3. BN running stats of copied BN layers: copied. Of new BN layers: identity
   (mean 0, var 1) — *not* re-calibrated, recorded as such (optional
   `--recalibrate-bn` runs the existing BN calibration with `init.bn_calibration`,
   GPU — only when not training, per CLAUDE.md run rules).
4. Reference tensors with no target counterpart are listed as **dropped**.
5. Guardrails (existing, extended): source must be a plain model file (no
   velocity/`trainer_*`); target validates; the output re-decodes; copied
   tensors verified bit-identical post-encode.
6. The graft is the **only** shape-changing derive; the existing
   "`shapeChangingRequest` refused" guard stays for all other ops.

Recorded in `derivation_history` (existing `DerivationRecord`) — the
`OperationRecord` gains optional fields (absent on #7 records, so old histories
decode):

```json
{"operation":"graft","arguments":{"target":"v5_…","graft_map":"…","seed":"1234"},
 "changed_architecture_fields":["*"],
 "rewritten_tensors":[…initialized…],
 "copied_tensors":[…], "dropped_tensors":[…],
 "init_seed":"1234", "init_rule_version":"v1",
 "per_tensor_init":{"block5_conv1_weights":"he_normal","value_wdl_fc2_bias":"wdl_prior(0.75)", …}}
```

Also retrofit: the existing `se_beta_init` op's re-init of the β half
(`ModelDerivation.swift:521`) uses the `init/<name>` stream under a recorded
seed, so a `zero→glorot` derive is reproducible.

## B4. Gap: `derivation_history` not carried forward by later saves

Today only `--derive-model` writes `derivation_history`
(`ModelDerivation.swift:~335`, key `derivationHistoryKey`). Every later save —
corpus replay rolling/enumerated (`CLI/CorpusReplayRunner.swift:579-605`),
train-vs-UCI (`CLI/TrainVsUciRunner.swift:~300`), GUI session trainer/champion
files and `Models/` saves (`Persistence/CheckpointManager.swift`) — builds
`__metadata__` from scratch in `SafetensorsModelIO.encode`
(`Persistence/SafetensorsModelIO.swift:114-145`), so the history is **lost at the
first training save** of a derived model.

Fix: `derivation_history` becomes a field of the new `LineageRecord` (Part D),
which every writer receives from the one place that loaded the parent and passes
to `SafetensorsModelIO.encode` as a typed argument (not via the stringly
`resumeMetadata` bag). `encode` refuses to write a file at the new format version
without a `LineageRecord`. The history is copied verbatim; training saves do not
append to it (they append a *segment*, D2). Test: derive → train 1 step in replay
→ save → header's `derivation_history` equals the derived file's.

---

# PART C — Exact resume, full audit

Baseline (d15f706): `TrainerResumeSnapshot` restores fp32 masters, velocity,
`completedTrainSteps`, warmup length, LR/momentum cycle + envelope, through one
function `ChessTrainer.restoreExactly(from:)` used by GUI resume
(`App/SessionController+Training.swift:1366-1387`), corpus replay and
train-vs-UCI `--resume-exact`. Velocity is fp32 (`ChessTrainer.swift:~4102`).

## C1. Everything NOT restored exactly

Impact: **M** = changes training math (weights/trajectory); **O** = changes
observability/cosmetics only; **S** = safety/operational.

| # | State | Where (today) | Paths | Impact | Fix |
|---|---|---|---|---|---|
| 1 | **Replay sampler RNG** | global RNG, `ReplayBuffer.swift:290,1783,1924,1973,2154,2189,2198,2235` | all | **M** — different minibatches from step +1 | A3 `sampler` stream; persist state (D2 `rng.sampler_state`). |
| 2 | **Dropout Philox state** | graph constant from `Int.random`, `ChessNetwork.swift:781`; reseeded per built graph | all with dropout>0 | **M** — different masks | A4 capture/restore; persist `rng.dropout_philox_state` (D2). |
| 3 | **KL-probe step counter** | `klProbeStepCounter` `ChessTrainer.swift:6263`, starts at 0 each process, never saved | all | **M when dropout>0** (probe advances the dropout state `:6906`, so probe phase shifts the dropout stream); **O** otherwise (probe fires at resume step 0 and phase shifts) | Derive probe steps from `completedTrainSteps` (like `batchStatsInterval` already does `:4539`) — deletes the counter: single source of truth. |
| 4 | **Replay buffer ring order / write index (corpus replay)** | reconstructed by refeeding `[until−1.5·cap/avgPly, until)` (`CorpusReplayRunner.swift:406-432`); `writeIndex` starts at 0 in a fresh buffer | replay | **M** — same *set* of positions but different slot indices ⇒ with a seeded sampler, slot `i` holds a different position ⇒ different batches | **Decided approach (owner, 2026-09-30): refill from slot 0 and reset the write pointer; only the *read* order must match.** (1) The sampler draws logical, age-ordered indices (A6 O7b), so the ring's physical offset is irrelevant. (2) On resume, refill slot 0…n−1 **oldest to newest**, starting at the oldest resident ply's recorded source (the slot-source record below), then set `writeIndex = n mod cap`. (3) Rebuild the stratified bucket arrays in that same age order (C1 #5). (4) Restore the sampler state. **Slot-source record:** a new per-slot column inside `ReplayBuffer`, a separate parallel array like the existing SoA columns (`plyIndexStorage`, `gameLengthStorage`, `stateHashStorage`, …) — ~16 B/slot (corpus shard, game offset within shard, ply; ~8 MB at 500k) — written under the same lock in the same insert. It keeps the board storage the trainer memcpys to the GPU untouched. For corpus replay **no buffer file is written**: the oldest slot's source + the corpus + the sampler state reproduce the buffer exactly. Test: slot-by-slot equality in age order vs the uninterrupted run, plus identical sampled positions. |
| 5 | **Material-bucket slot arrays order** | `MaterialBucketSlots` swap-remove (`ReplayBuffer.swift:280-291`), rebuilt in ring order on insert/`restore` (`:1223-1231`, `restore(from:)` `:3079`) | all (stratified on) | **M** — `randomSlot()` picks `slots[k]`; after evictions the live array order ≠ rebuilt order | **Decided (D-5): rebuild in age order and pick by age rank.** `randomSlot` draws `k = nextBounded(bucketCount)` and returns the bucket's **k-th oldest** resident slot, so the stored array order (which swap-remove scrambles) never affects the pick. The arrays are rebuilt in age order on every refill/restore (C1 #4). Implementation: the ring only ever evicts the globally oldest slot, which is necessarily the oldest member of its bucket, so each bucket can be a FIFO in age order — inserts append at the back, evictions pop the front — held as a per-bucket circular array; the k-th oldest is then an O(1) index, and swap-remove (with its `slotPosition` map, `:260`) goes away. **Verify in P9** that no other path removes slots mid-bucket (e.g. a restore or a dedup path); if one does, fall back to an order-statistic structure. P9 measures `sample()` µs against today's (A2.2) and must not regress. Nothing extra is persisted. Test: after a mixed insert/evict history, a rebuild-from-age-order buffer and the live buffer return the same slot for every k. |
| 6 | **Corpus replay: cross-epoch reconstruction** | refused (`CorpusReplayRunner.swift:423-426`) | replay | **S** — epoch-boundary checkpoints can't be exact-resumed | With #4's exact reconstruction, wrap backward across the epoch boundary (games `[N−k, N)` of the previous epoch then `[0, until)`), which is well-defined because corpus order is sequential. |
| 7 | **Corpus replay build drift** | warns only (`:399-401`) | replay | **M** if encoder/feeder changed | Keep warning; make it a refusal in `--resume-exact` when `encoder_version`/`feeder_version` (new header keys) differ; git hash alone stays a warning. |
| 8 | **Counters `gamesFed`/`positionsFed`** | restart at 0 (`CorpusReplayRunner.swift:728-729`) | replay | **O** + **M**: verified 2026-09-30 — the feed cadence `targetFed = prefillPositions + step * perStepFeed` (`:778`) uses the segment-local `step` and `positionsFed`. Whole games overshoot the target, so the feed **phase** (how far ahead of the target the last game left the counter) is state that a restart resets ⇒ which games are in the buffer before a given step shifts. | Persist the cumulative counters (D2) **and** the feed carry (`positionsFed − targetFed` at the save step); resume computes the target from the cumulative step so the feed phase is identical. Test: the games fed before each of the M post-resume steps equal the uninterrupted run's. |
| 9 | **Replay buffer on GUI and train-vs-UCI saves** | GUI: every session save writes `replay_buffer.bin` (7.2–11 GB observed), gated by `hasReplayBuffer`/`wantsReplayBuffer` (`Persistence/CheckpointManager.swift:666`, write `:716`; load honors absence `:1100`); vs-UCI: fresh `ReplayBuffer` (`TrainVsUciRunner.swift:214`), not saved | GUI, vs-UCI | **M** — a resume without the buffer trains its first steps on a small buffer drawn from the current champion only | **Decided (D-8).** (a) Train-vs-UCI checkpoints use the same folder format and writer as a GUI `.dcmsession` — `trainer.safetensors` + `session.json` + `manifest.json`, plus `replay_buffer.bin` **only when included** — through the same staging directory and rename (C1 #29). (b) **The buffer is not saved by default** on either path. It is included only when: the persisted parameter `session_save_include_replay_buffer` is on (autosave: periodic + post-promotion; D8 below), or the manual Save Session asks for it, or train-vs-UCI is launched with `--save-replay-buffer`. The existing `hasReplayBuffer` flag records which. (c) A resume from a save without a buffer refills from new games; training waits for `replay_buffer_min_positions_before_training` as on a fresh start; the resume is labeled `NOT EXACT: buffer` (C3). A resume from a save with a buffer restores it exactly (v8 format with the slot-source column; legacy v7 per D-9). |
| 10 | **Train-vs-UCI pool/opponent counters, next game serial** | not saved | vs-UCI | **M** (which opponent/color next) | Persist `vsuci_next_game_serial`, per-opponent game counts. |
| 11 | **Self-play in-flight games** | dropped on save | GUI, vs-UCI | **M** (small) | Accept + log the count (`[CHECKPOINT] dropped N in-flight games`). Persist `selfplay_next_game_serial` so per-game streams continue, not repeat. |
| 12 | **ReplayRatioController windows** | only `lastAutoComputedDelayMs` restored (`SessionController+Training.swift:899-903, 1007-1015`) | GUI | **M** (step delay ⇒ cons/prod ratio ⇒ which positions exist when sampled) — inherently wall-clock, so not trajectory-exact anyway | Persist prod/cons EMAs / window samples; seed controller. Best-effort by nature (§0). |
| 13 | **Arena trigger clock** | `ArenaTriggerBox()` fresh (`SessionController+Training.swift:1026`) | GUI | **M** (when arenas/promotions happen) | Persist `seconds_since_last_arena` measured in **training time** (C1 #19), seed the box. |
| 14 | **Periodic-save timer** | `PeriodicSaveController(interval:)` fresh (`:1045`) | GUI | **S** | Persist `seconds_since_last_save`; seed. |
| 15 | **Save during arena** | refused/deferred (`SessionController+Checkpoint.swift:31,144,178,226`) | GUI | **S** — kill during arena loses it | **Decided (D-6): discard and re-run.** No partial `TournamentRecord` is persisted. The last save predates the arena, so on resume the arena trigger clock (#13) is restored from that save and the arena runs again when it comes due. The save cannot know an arena later started, so nothing about it is logged beyond the restored clock value (`[RESUME] arena clock=<sec>`). Test: a session saved mid-arena-deferral resumes with the pre-arena clock and runs the arena. |
| 16 | **Promotion resume** | post-promotion save reuses arena-pause snapshots (`SessionController+Arena.swift:115-138, 376-381`) | GUI | **M** if the trainer file's clock/velocity don't match the rewound trainer | Test: after promotion save, trainer file `trainer_completed_steps` and velocity equal the live trainer's post-promotion values. |
| 17 | **Arena index** | not persisted | GUI | **M** via arena streams | Persist `arena_count` (already derivable from `arenaHistory.count` in `session.json` — use that as the single source). |
| 18 | **GameDiversityTracker window** | fresh `GameDiversityTracker(windowSize: 200)` (`SessionController+Training.swift:942`) | GUI | **O** | Persist the 200 hashes/sequences (small) or accept + log. Recommend persist (cheap). |
| 19 | **Elapsed-time accounting** | GUI: `elapsedTrainingSec` = wall since `currentSessionStart` (`SessionController+Checkpoint.swift:761-762`, resume sets `currentSessionStart = now − elapsed` `SessionController+Training.swift:1071`) — includes paused/arena/idle time; CLI: `runStart` wall per process (`CorpusReplayRunner.swift:815`, `TrainVsUciRunner.swift:390-391, 464`), resets per segment | all | **O** (and the dashboard time axis) | Two cumulative clocks carried across segments (D2): `cum_train_step_sec` = Σ measured step durations (the trainer already times steps), and `cum_wall_sec` = Σ segment wall. Replaces the dashboard's sleep-clamp inference for new files. |
| 20 | **Stats/EMA windows, alarms** | `TrainingAlarmController`, `DrawWatchTracker()` (`:950`), chart rolling windows, legal-mass-collapse grace timers — fresh each process | GUI (alarms), CLI (EMA in `[STATS]`) | **O**; **S** for legal-mass-collapse (its grace/no-improvement counters decide an auto-action) | Persist alarm controller state (armed flags, streaks, grace start in training time); EMAs: accept + log (they converge in seconds). |
| 21 | **Parameter defaults on resume (#5)** | 45 `saved=nil applied=<current>` branches (`SessionController+Training.swift`, e.g. `:129,141,158,641-661`) | GUI; CLI analog | **M** — Exp 6 dropout 0.7 incident | `absentValue:` on the macro; `[RESUME-DIFF]`; full parameter snapshot in every checkpoint (D2) — C5. |
| 22 | **CLI `--resume-exact` parameters other than schedule** | from `--parameters`/UserDefaults, not diffed | replay, vs-UCI | **M** | Checkpoint carries full `training_parameters` snapshot; `--resume-exact` restores it and diffs against `--parameters` (`[RESUME-DIFF]`), refusing live-untunable differences unless `--allow-param-change` (logged per key). |
| 23 | **Old-format checkpoints** | CLI refuses (correct, `TrainerResumeState.swift:~223`); GUI session without trainer file forks from champion with zero velocity (`SessionController+Training.swift:~1389-1396`) | all | **M** | Single line `[RESUME] NOT EXACT: <list>`; record `exact_resume=false` + the list in the segment record (D2). |
| 24 | **Step counters used by schedules** | tau schedule: per-game ply (no global counter — fine); LR/momentum/warmup/cycle: `completedTrainSteps` (restored ✓); `batchStatsInterval`: `completedTrainSteps` (✓ `:4539`); diagnostics cadence: same (✓); KL probe: separate counter (✗ #3); `ReplayRatioController`: time-based (#12); candidate-probe interval: seconds (`candidateProbeIntervalSec`) — wall clock, resets (**O**) | all | as listed | Only #3 needs a code change; probe interval ⇒ measure against `cum_train_step_sec`. |
| 25 | **`LastSessionPointer` ordering vs rename** | pointer written by `CheckpointController.recordLastSessionPointer` (`App/UpperContentView/CheckpointController.swift:444`, called from `SessionController+Checkpoint.swift:446` and `SessionController+Arena.swift:639`) → `LastSessionPointer.write` (`Persistence/LastSessionPointer.swift:90`); orphan `.tmp` sweep `Persistence/CheckpointManager.swift:120-164` | GUI | **S** | Test: kill between rename and pointer write leaves pointer at the prior complete save. |
| 26 | **Enumerated-name overwrite across segments** | same segment-local step name (`CorpusReplayRunner.swift:614-620`). *Since then:* enumerated writes never overwrite a file the run did not write, and a run whose stem already holds reachable step files refuses to start (`TrainerOutputFileGuard`); the data loss is closed, the segment index in the name is still open | replay | **S** (data loss) | Include `lineage_segment_index` in the enumerated name (`…-seg3-step41000`), and refuse to overwrite an existing enumerated file whose `lineage_segment_id` differs. |
| 27 | **`--start-model` (branch) vs `--resume-exact`** | branch: fresh clock, zero velocity (by design, `TrainerLaunchKind.newBranch`) | CLI | by design | Branch mints a **new `lineage_run_id`** and new master seed (unless `--seed`), recording parent (D2). Exact resume inherits both. |
| 28 | **Epoch boundaries (replay)** | epoch-completion checkpoint normalized to `(nextGame=0, epoch=N)` | replay | **S** (can't resume there, #6) | Fixed by #4/#6. |
| 29 | **Crash mid-save** | GUI: whole session folder staged as `Sessions/<name>.tmp`, then renamed; orphan `.tmp` debris swept at launch (`Persistence/CheckpointManager.swift:120-164`) ✓; CLI: single `.safetensors` written to a temp file and renamed ✓ | all | OK | **Keep the existing temp-then-rename pattern; add no new file.** Rule: every new piece of resume state (RNG states, feed carry, serials, lineage) lives **in the same file or folder as the weights it belongs to** — corpus replay: the one `.safetensors`'s `__metadata__`; GUI and train-vs-UCI (D-8): the session folder (`session.json`, plus `replay_buffer.bin` when included), covered by its `manifest.json` and the single rename. There is therefore no separate file that could be torn from its weights. |
| 30 | **Momentum/velocity dtype** | fp32 | all | presumed OK | Test: saved dtype fp32; bit-exact round trip under a bf16 network (from #6). |
| 31 | **Replay-buffer position hashes are per-process** | A6 O16 (`ReplayBuffer.swift:649-657`, persisted `:2937`/`:3377`) | GUI and vs-UCI, when a buffer is included (D-8) | **O** — duplicate/unique-position stats wrong after every resume; no training-math impact (checked) | Fixed-key hash; regression test first (O16); legacy v7 buffers recompute hashes on load (D-9, decided). |
| 32 | **Corpus changed under a resume** | `--resume-exact` compares `corpus_id` only (`CorpusReplayRunner.swift:397`) | replay | **M** — a re-imported or edited shard with the same ID feeds different games | Record each shard's SHA-256 in the lineage (`fed.corpus.shard_sha256`, D2; same values as `experiments/corpora/*.md` manifests); on resume re-hash the shards the refill and the run will read and **refuse on any mismatch**, naming the shard. Cost: one sequential read of those shards at startup. |
| 33 | **Build or OS changed under a resume** | CLI warns on git-hash change only (`:401`); GUI records build in `session.json` but doesn't compare | all | **M** — different code or MPSGraph/Philox can change the trajectory (A4, A5) | Compare `build.git_hash`, `build.git_dirty`, `build_number` and `device.os_version` with the running process: any difference ⇒ `[RESUME] WARNING …` on every path, and `--resume-exact` **refuses** unless `--accept-inexact build` / `os` names it (C3). Recorded in the segment record. |
| 34 | **Config D (`--bf16-cast-in-forward`, issue #9)** | GUI-only flag; not recorded in any checkpoint | GUI | **M** — a second optimizer/storage path (fp32 variables, no master/working pair) | **Decided (D-10): removed before P9** (phase P13), so no lineage field, resume check or harness variant is needed for it. |

**Top gaps by training-math impact:** #1 sampler RNG, #2 dropout state, #4/#5
buffer slot order (makes #1 useless even once seeded), #21/#22 parameter
defaulting/diffing, #8 feed phase, #32 corpus content, #33 build/OS, #9 buffer
save policy, #3 KL-probe counter coupling into dropout, #13 arena clock.

## C2. Save-point consistency rules

- **Corpus replay needs no pause at all:** its loop is step-locked (§0), so the
  code between two `trainStep` awaits is the save point, with no concurrent
  feeder or trainer to stop. The save runs there, reading the trainer snapshot,
  sampler state, Philox state and feed carry in one consistent cut.
- GUI and train-vs-UCI: all resume state is captured **under the same training pause** that already
  guards `exportResumeSnapshot()`; the replay-buffer sampler state and the
  dropout state are read inside that pause, after the last completed step, so
  "state after step N" is one consistent cut.
- GUI: self-play keeps running during a save today (only training is paused for
  the trainer snapshot; the buffer snapshot is taken under the buffer lock). When
  the buffer is included (D-8), the buffer file and the sampler state are
  consistent with each other (same lock);
  they are *not* consistent with an exact self-play cut, which §0 already rules
  out of scope.
- **Probe-isolation rule (memorialized):** *no probe, diagnostic or observer may
  change trainer, optimizer, replay-buffer or RNG state.* A probe reads a snapshot
  or runs on its own copy and its own named stream (`probe.<name>.<step>`). Case
  to settle in P4: the KL-probe path runs the dropout advance op
  (`ChessTrainer.swift:6900-6910`; its comment says the advance is there so the
  *next training step* doesn't reuse this step's mask). If that is the step's own
  single advance, it's correct and the only fix is C1 #3's step-derived schedule;
  if the probe adds an advance beyond the step's, it violates this rule and gets
  its own state or a save-and-restore around its run. Either way the P4 test
  asserts: dropout masks for steps N+1… are identical with the KL probe on and off. This project has had probe side-effect bugs
  before (the probe staging-buffer clobber). The rule goes into the project
  `CLAUDE.md` ("Concurrency invariants"), into `DCMRandom`'s doc comment, and is
  enforced by the harness: C6 runs with **every probe enabled**, so any probe that
  mutates state shows up as a trajectory mismatch.

## C3. Resume classification (single function)

New `ResumeExactness` (in `TrainerResumeState.swift`, shared by all three paths):
`exact` or `notExact(missing: [ResumeGap])`, computed from what the checkpoint
carries. Every path logs exactly one line
`[RESUME] EXACT` or `[RESUME] NOT EXACT: rng_sampler, dropout_state, buffer, …`
and records it in the new segment record (D2).

`ResumeGap` is a closed enum, one case per item, each with its log token:
`rng_sampler`, `dropout_state`, `feed_carry` (replay), `buffer` (GUI/vs-UCI save
without a buffer, D-8), `serials`, `clocks` (arena/save/ratio controller),
`params` (no full snapshot — legacy), `lineage` (no `dcm_lineage` — legacy),
`build`, `os` (C1 #33), `policy_tail` (the trainer file's
`trainer_policy_tail_precision` differs from the process's
`--policy-tail-precision`, or the file predates recording it).

*Interim, until `ResumeGap` lands (review fixes 2026-10-02):* trainer files
record `trainer_policy_tail_precision`; corpus replay and train-vs-UCI
`--resume-exact` refuse a recorded mismatch and log a loud warning for an
unrecorded value (`PolicyTailPrecisionResume`); a GUI resume logs
`[RESUME] NOT EXACT: policy_tail …` and never refuses. When `ResumeGap` lands,
`policy_tail` joins it and the unrecorded case follows `--accept-inexact`.

**Per path:**
- **Corpus replay:** `--resume-exact` refuses on `notExact`. **Decided (D-7):**
  `--accept-inexact <comma list>` allows it when the list names every missing
  item: each missing RNG stream is seeded freshly from the run's master seed via
  its named stream (logged `[RESUME] <stream>: fresh (not in checkpoint)`), other
  missing items take their fresh-start value (logged), and the segment is recorded
  `exact_resume=false, not_exact_items=[…]`. A corpus-content mismatch (C1 #32) is
  never acceptable. This is what lets any pre-v5 checkpoint be resumed at all.
- **Train-vs-UCI:** same flags. `buffer` is expected there under D-8 unless the
  save was made with `--save-replay-buffer`; `--accept-inexact buffer` names it.
- **GUI:** there is no refusal (a GUI resume is state-exact at most, D-1); the
  one `[RESUME] … NOT EXACT: …` line is logged and recorded, and the status bar's
  resume message says "resumed (not exact: buffer, …)".

No silent downgrade on any path.

## C4. `--resume-exact` vs `--start-model` (unchanged semantics, made explicit)

| | lineage_run_id | master seed | segment index | clock/velocity | RNG streams | parent recorded |
|---|---|---|---|---|---|---|
| `--resume-exact` | inherited | inherited | +1 | restored | restored | yes (parent sha) |
| `--start-model` (branch) | **new** | new (or `--seed`) | 0 | fresh/zero | fresh | yes (parent sha, `branch=true`) |
| fresh | new | new (or `--seed`) | 0 | fresh | fresh | no |

## C5. Parameter defaults on resume (supersedes #5)

Adopt #5's requirements verbatim, integrated:

1. `@TrainingParameter(..., absentValue: <pre-feature value> | .refuseExact)` in the macro; `allKeys` test enforces every key declares it.
2. Resume resolution is one function `TrainingParameterResolution.resolve(saved:current:)` used by GUI and CLI, replacing the 45 hand branches (`SessionController+Training.swift`), writing nothing to `UserDefaults` for defaulted keys.
3. `[RESUME-DIFF]` block (saved vs applied vs current setting) on every resume.
4. New-format checkpoints carry the full snapshot (`training_parameters`, D2), so `saved=nil` can only occur on legacy files.
5. CLAUDE.md checklist gains step "declare `absentValue`".
6. Regression test: Exp 6 case (legacy session, no dropout key, live 0.7) ⇒ 0.0 + `[RESUME-DIFF]`.

## C6. Verification design — resume-equivalence harness

`DrewsChessMachineTests/ResumeEquivalenceTests.swift` (GPU; gated behind
`DCM_RUN_SLOW_TESTS` only if it's slow — it asserts correctness, so preferably
kept small and ungated, per CLAUDE.md) plus a CLI script
`scripts/resume_equivalence.sh` for the real runners.

**Setup:** tiny architecture preset `test_tiny` (1 block, 16 ch — or smallest
valid), tiny synthetic corpus fixture (checked in, ~200 games), capacity 2,000,
batch 32, dropout 0.1, stratification on, length tilt on, KL probe every 3 steps,
seed `0xD5C0FFEE`.

1. **Determinism probe** (prerequisite, same process): run step 1 twice from the
   same state (weights + velocity + Philox + batch) and compare output tensors
   bit-for-bit. Result decides assertion mode: `bitExact` or
   `tolerance(rel: 1e-6 fp32 / 1e-2 bf16 on weights, abs on loss)`.
2. **Uninterrupted:** fresh → N+M steps (N=20, M=20); record per step: sampled
   indices (tap in `ReplayBuffer.sample`), dropout mask hash (graph tap), loss,
   gNorm, weights SHA-256 at the end.
3. **Interrupted:** fresh → N steps → save (real `SafetensorsModelIO`/session
   writer) → **new process-equivalent**: new trainer, new buffer, load, restore →
   M steps.
4. Assert: (a) sampled indices identical for all M steps (**exact, always**);
   (b) dropout mask hashes identical (**exact**); (c) LR/momentum/probe schedule
   identical (**exact**); (d) weights/loss per step equal under the probe's mode.
5. Variants (all with **every probe enabled** — C2 probe-isolation rule): resume at an epoch boundary; probes on vs off must give identical sampled indices and dropout masks; resume from a checkpoint written at a
   KL-probe step; resume after a parameter change (expect `[RESUME-DIFF]` +
   refusal without `--allow-param-change`); resume a legacy (v4) fixture file
   (expect the `NOT EXACT` list; refusal without `--accept-inexact`, and with it a
   continuation whose segment records `exact_resume=false` and the list — D-7).
6. **GUI/vs-UCI state round-trip tests** (no trajectory claim, D-1): for each state
   in C1 (sampler, Philox, ratio controller, arena clock, periodic-save clock,
   diversity tracker, alarm state, serials): save → restore → `==`. Buffer
   (D-8): a save **with** the buffer round-trips it slot-for-slot in age order
   with identical bucket k-th-oldest picks (D-5); a save **without** it resumes
   with an empty buffer, waits for the minimum fill, and logs
   `NOT EXACT: buffer`. Arena (D-6): a save taken while an arena is deferred
   resumes with the pre-arena clock and re-runs the arena. Legacy v7 buffer
   (D-9): hashes recomputed and equal to the fixed-key hash of each board.
7. **CLI end-to-end** (script, run by the owner when the GPU is free): real
   `--train-from-corpus` for 2N vs N + `--resume-exact` + N; compare final
   `content_sha256` (or tolerance per the probe).

---

# PART D — Lineage, step numbers and run time

## D1. What files hold today

- **Base `__metadata__`** (`Persistence/SafetensorsModelIO.swift:56-65, 113-145`): `dcm_format_version` (**4**, `Network/ArchitectureFormat.swift:41`), `model_id`, `created_at_unix`, `creator`, `training_step` (segment-local for CLI), `parent_model_id`, `notes`, `architecture`, value-head-centered marker, plus `content_sha256` (file-level).
- **`trainer_*`** (d15f706): `trainer_completed_steps` (cumulative across exact resumes), `trainer_lr_warmup_steps`, `trainer_lr_momentum_cycle`, `trainer_lr_momentum_cycle_envelope`.
- **Replay only** (`CLI/CorpusReplayRunner.swift:584-593`): `replay_corpus_id`, `replay_corpus_path`, `replay_next_game_index`, `replay_epoch`, `replay_populated_plies`, `replay_capacity`, `built_by_build`, `built_by_git`.
- **Derived only**: `derivation_history` (lost on next save, B4).
- **GUI `session.json`** (`Persistence/SessionCheckpointFile.swift`, `formatVersion` **1**, checked with `==` at `:753` — a bump must widen that check): hand-listed parameter subset (~60 optionals), counters (`trainingSteps`, `selfPlayGames`, `selfPlayMoves`, `trainingPositionsSeen`), wall `elapsedTrainingSec`, build info incl. `buildGitDirty`, `completedTrainingSegments` (`TrainingSegment` `:723-745`: start/end unix, duration, start/end steps/positions/games, build) — the GUI **already has a segment concept**; D2 generalizes it to all paths and files.
- **Missing everywhere**: run ID, segment index/ID, parent content hash, cumulative games/positions carried across CLI segments, full parameter snapshot in safetensors, training-time clock, device/OS, command line, seeds/RNG state, dirty flag on CLI files.

Pain points (from #4, `experiments/*/README.md`, `documentation/v5-lineage.md`,
`documentation/dashboards/registry.json`, `replay.py discover-stems`,
`selfplay.py`): hand-entered `cumstep_base` wrong by 1,779 (nt8y) and 550 (coxw);
enumerated names repeat/overwrite; `enum_stem` inferred by heuristic; ModelID
collisions (`bzw3-31`); hyperparameters unrecoverable when logs are lost;
`games_fed` blank when logs are lost; time axis restarts without pinned
`elapsed_base_sec`; device only in registry.

## D2. Schema (format version 5)

> **Renumbered 2026-10-01, and again 2026-10-02:** format version 5 was taken by the
> per-group `se_activation` field (issue #2), and format version 6 by the per-group
> `rezero_alpha_cap` field (the explicit ReZero soft-bound cap that makes a zero α
> init possible; `ArchitectureFormat.currentVersion = 6`). Each was a self-contained
> architecture-field bump that shipped ahead of this plan, so this plan's format bump
> is now **version 7**. Throughout this plan, "format v5", "`dcm_format_version` 5"
> and "v5 files/writers/readers" in the format-version sense mean this plan's bump —
> v7 — and "pre-v5" means "before this plan's bump" (files of v6 and older). ("v5" as
> an architecture / lineage name, e.g. the v5 line, is unrelated and unchanged.)

One typed struct `LineageRecord` (`Persistence/LineageRecord.swift`), `Codable`,
written into `__metadata__` as **one JSON value under `dcm_lineage`** plus a few
flat mirror keys for grep/tooling convenience (mirrors are *derived* from the
struct at write time and ignored on read — single source of truth is the JSON).
The same struct is embedded in `session.json` (GUI) so both carriers share code.

```jsonc
"dcm_lineage": {
  "schema": 1,
  "run": {
    "lineage_run_id": "UUID",            // minted on fresh/branch, inherited on exact resume
    "segment_index": 3,                   // 0 on fresh/branch, +1 per exact-resume process
    "segment_id": "UUID",                 // this process
    "segment_started_unix": 1790000000,
    "exact_resume": true,
    "not_exact_items": [],                // C3 list when false
    "branch": false
  },
  "parent": {                             // absent on fresh
    "model_id": "20260929-12-JZOe",
    "content_sha256": "…",                // parent FILE's content_sha256 — collision-proof
    "trainer_completed_steps": 41000,
    "lineage_run_id": "…", "segment_id": "…"
  },
  "steps": {
    "cum_trainer_step": 42000,            // == trainer_completed_steps (asserted equal at write)
    "segment_start_trainer_step": 41000,  // replaces hand-entered cumstep_base
    "segment_local_step": 1000            // == training_step for CLI files
  },
  "fed": {
    "cum_games": 123456, "cum_positions": 8123456, "cum_plies_generated": 8200000,
    "segment_games": 1000, "segment_positions": 64000,
    "corpus": {"corpus_id":"…","epoch":0,"next_game_index":2345,"shard":3,"offset":17,
               "total_positions_added": 8123456,
               "oldest_resident": {"epoch":0,"shard":2,"offset":9120,"ply":14},   // refill start (C1 #4)
               "feed_carry_positions": 37,                                        // C1 #8
               "shard_sha256": {"shard-000001": "…", "…": "…"}}                   // C1 #32; replay only; generalizes replay_*
  },
  "time": {
    "cum_train_step_sec": 51234.5,        // Σ measured step durations (GPU step wall incl. sampling), carried across segments
    "cum_wall_sec": 60210.0,              // Σ segment wall from process start to save
    "segment_train_step_sec": 812.3, "segment_wall_sec": 950.0
  },
  "parameters": {
    "snapshot": { … flat snake_case exactly as --show-default-parameters … },
    "sha256": "…",                        // of canonical sorted-keys JSON of snapshot
    "trainer_hyperparameters": { … resolved TrainerHyperparameters … }
  },
  "build": {"build_number": 2141, "git_hash": "47984b0", "git_branch":"main", "git_dirty": false},
  "invocation": {"argv": ["…"], "path_kind": "gui|replay|vsuci|derive"},
  "device": {"hw_model":"Mac16,8","chip":"Apple M4 Pro","is_vm":false,"os_version":"26.1 (25B…)","gpu_name":"…"},
  "rng": {
    "derivation": "v1", "master_seed": "1234567890", "seed_mode": "seeded|unseeded",
    "init_seed": "…", "init_scheme": "dcm-init-1",   // fresh mints/grafts (B1.1)
    "sampler_state": {"s0":"…","s1":"…","s2":"…","s3":"…"},
    "dropout_philox_state": [0,0,0,0,0,0,0],
    "selfplay_next_game_serial": 9876, "arena_count": 12, "vsuci_next_game_serial": 0
  },
  "segments": [ /* compact history of prior segments of this run: index, id, start/end steps, games, time, build, device, exact */ ],
  "derivation_history": [ … ]            // B4, carried verbatim
}
```

Flat mirrors (derived, read-ignored): `lineage_run_id`, `lineage_segment_index`,
`cum_trainer_step`, `cum_games_fed`, `cum_train_step_sec`, `git_dirty`.

Notes:
- `segments[]` makes a single newest file sufficient to rebuild the registry even
  when earlier files and logs are gone (the v5-lineage failure mode).
- `argv` is recorded verbatim; tokens that look like secrets (Lichess token
  flags) are redacted by an explicit deny-list.
- `device.is_vm` from `kern.hv_vmm_present`; `hw_model` from `hw.model`; chip
  from `machdep.cpu.brand_string`.
- Size: parameters snapshot ~6 KB, segments ~200 B each; negligible next to weights.

`session.json`: `formatVersion` → 2, embeds `lineage` (same struct) and
`parametersSnapshot` (full), keeps the existing explicit fields for legacy
decode only (v1 files). The `==` check at `SessionCheckpointFile.swift:753`
becomes `1...current` with per-version requirements.

## D3. Writers and readers

- **Single writer path:** `SafetensorsModelIO.encode(..., lineage: LineageRecord?)`.
  At format ≥ 5 a trainer-state file **must** have `lineage` (throws otherwise).
  Plain model exports (e.g. `Models/` champion) carry it too.
- **Single builder:** `LineageTracker` (one per run, `@unchecked Sendable` + `SyncBox`)
  owned by the trainer host in each path; accumulates fed/time counters; produces
  the record at save. GUI, replay and vs-UCI all construct it through
  `LineageTracker.start(kind: .fresh | .branch(parent) | .exactResume(parent))`.
- `replay_*` keys: still **written** for one version? No — per the "no backward
  compat unless requested" rule, v5 writers stop writing `replay_*`; v5 readers
  read `dcm_lineage.fed.corpus`; v4 files keep being read via the existing
  `readResumeMetadata` path (legacy branch).

## D4. Startup provenance line (all paths)

Every run path (GUI session start/resume, `--train`, `--train-from-corpus`,
`--train-vs-uci`, `--derive-model`, `--uci`) logs at start:

```
[RUN] path=replay run=<uuid> seg=3 (exact resume of <model_id> sha=<12>) build=2141 git=47984b0 dirty=false
      device="Apple M4 Pro" vm=false os="26.1" seed=1234567890 mode=seeded derivation=v1
      params_sha=<12> cum_step=41000 cum_games=123456 cum_train_sec=51234.5 argv="…"
```

The GUI's `[APP]` banner stays; `[RUN]` is added at each session start. The CLI
runners today log `[REPLAY-HPARAMS]`/`[VS-UCI-HPARAMS]` but no build/device/seed
banner — `[RUN]` fixes that via one shared formatter.

## D5. `results.json`

`CliTrainingRecorder` gains a top-level `lineage` object = the `LineageRecord` at
run end (minus `segments`/`derivation_history` to keep it small; plus
`checkpoint_sha256` of the final file). Per-sample rows gain `cum_trainer_step`,
`cum_train_step_sec`, `cum_games`.

## D6. Dashboard tracker derives the registry

- New `replay.py derive-registry [--models-dir ~/…/Models] [--write]`: scans
  safetensors headers (header-only read, like `ckpt_inventory.py`), groups by
  `lineage_run_id`, orders by `segment_index`, and emits registry segments with
  `cumstep_base = segment_start_trainer_step − run_origin`, `games_base`,
  `elapsed_base_sec` (= prior `cum_train_step_sec`), `wall_base_sec`, `device`,
  `model_id`, `enum_stem` (unneeded: enumeration keys on `segment_id`). It
  **diffs** against `registry.json` and writes only with `--write`.
- v5 files: `discover-stems` is bypassed; `games_fed` comes from
  `cum_games` (measured by the writer — never modeled). v4 files: unchanged
  heuristics + hand registry (the three-axis rules in CLAUDE.md stay intact).
- Time axis: new files provide `cum_train_step_sec`, which is *measured* step
  time; the sleep-clamp stays for v4 rows and stays auditable. Plot both where
  both exist.
- `selfplay.py`/`vsuci.py`: same header-derived segment table.
- Test: 3 synthetic v5 headers → expected registry JSON (pytest in
  `documentation/dashboards/tests/`).

## D7. Versioning and back-compat (the #7 rule, applied)

- **`dcm_format_version` 5** (one bump for B + D + A persisted state).
  - v ≤ 4 (legacy): no `dcm_lineage` ⇒ loads, trains, infers exactly as today;
    new B2 arch fields resolve to legacy values (logged once); exact resume only
    per d15f706 rules and flagged `NOT EXACT: rng_sampler, dropout_state, params, lineage, …` (C3).
  - v 5: `dcm_lineage` required on every file; new B2 arch fields required in
    every group/head; missing ⇒ load error naming field + file.
  - v > 5: `unsupportedFutureVersion` (existing).
- `session.json` formatVersion 2 as above.
- Presets / `architecture.json`: `format_version` 5 required for the new fields.
- `replay_buffer.bin`: `ReplayBuffer.fileVersion` 7 → 8 (fixed-key hashes +
  slot-source column); v7 files load as legacy with hashes recomputed (D-9) and
  no slot sources (so a v7 buffer resume is `NOT EXACT` for replay refill, which
  never reads a buffer file anyway).
- Nothing rewrites old files. The only migration code is the owner-approved D-9
  hash recompute on load (derived data, in memory; the file is not rewritten).

## D8. Replay-buffer save option (D-8)

The buffer is **not** saved by default on GUI or train-vs-UCI saves. Three
controls, one source of truth for the automatic case:

- **New `@TrainingParameter` `session_save_include_replay_buffer`** (Bool, default
  `false`, live-tunable). Governs every *automatic* save: periodic and
  post-promotion in the GUI, and every checkpoint train-vs-UCI writes. Full
  CLAUDE.md checklist:
  1. Declare with `@TrainingParameter`, add to `allKeys`.
  2. Singleton stored property + `collectValues` / `applyOne` + snapshot accessor.
  3. `parameters.json`: appears in `--show-default-parameters`; round trip via
     `--create-parameters-file`.
  4. Session: Optional field in `SessionCheckpointState`, passed through
     `buildCurrentSessionState`, and a `[RESUME-PARAM]` block (both "from
     session" and "saved=nil (defaulted)" branches log). Under C5 it declares
     `absentValue: false`: a pre-feature session always saved the buffer, but
     reproducing that on resume would re-enable multi-GB saves against D-8, so
     the owner's default applies and the `[RESUME-DIFF]` line shows it.
  5. `results.json`: recorded in the recorder's parameter snapshot.
  6. Runtime log: every save line reports it —
     `[CHECKPOINT] Saved session (<trigger>): … buffer=included|omitted`.
  7. UI: Settings popover ▸ **Sessions** tab, next to the existing Autosave
     controls (`App/UpperContentView/TrainingSettingsPopover.swift:1110-1169`):
     a toggle "Include replay buffer" with its size estimate
     (capacity × bytes per position, base-2) shown beside it.
     `TrainingSettingsPopoverModel` binding.
  8. Live tunability: read from `TrainingParameters.shared` at each save, never
     cached at session start (it is a save-time switch, not a loop knob).
  9. Rename: n/a.
- **Manual save:** File ▸ Save Session (`App/DrewsChessMachineApp.swift:618`)
  today saves immediately. It gains (owner-confirmed 2026-09-30) a small confirmation sheet with an
  "Include replay buffer" checkbox (initial state = the parameter's current
  value, per save only, not persisted) and the size estimate. `[BUTTON]` logs the
  choice.
- **Train-vs-UCI CLI flag `--save-replay-buffer`:** sets the parameter for that
  process (logged `[PARAM] session_save_include_replay_buffer from
  --save-replay-buffer`), like other CLI overrides; a repeated flag is rejected
  with a named error.
- **Writer:** the existing `wantsReplayBuffer` gate
  (`Persistence/CheckpointManager.swift:666`) and `hasReplayBuffer` flag
  (`Persistence/SessionCheckpointFile.swift:637`) already support omitting the
  file; the change is only who sets the flag. Loading already honors absence
  (`CheckpointManager.swift:1100`).
- **Resume without a buffer:** the session starts with an empty buffer; training
  waits for `replay_buffer_min_positions_before_training` exactly as a fresh
  start does; `[RESUME] … NOT EXACT: buffer` (C3).
- **Corpus replay:** never writes a buffer; unaffected.
- **Autosave retention — ADDITION 2026-10-01** (folded in from
  `AUTOSAVE_RETENTION_PLAN.md`, owner decision 2026-10-01; implemented in **P14**).
  *Before P14* (as written when this addition was made, kept for the record):
  `CheckpointPaths.prunePeriodicAutosaves` (`Persistence/CheckpointManager.swift:189` at
  the time; called only after a `periodic` save, `App/SessionController+Checkpoint.swift:460-466`)
  deletes only `-periodic.dcmsession` folders beyond `max_periodic_autosaves_kept`;
  `-promote` folders (arena post-promotion and "Promote Trainee Now", which share the
  `promote` disk tag — `Persistence/SessionSaveTrigger.swift`) are never pruned, nor are
  `-manual` and `-sigusr2`. (By the time P14 was implemented that function had already been
  narrowed further, in the uncommitted working tree, to prune **per session**: only the
  saving session's own `-periodic` folders, verified by exact name and by the `sessionID`
  in `session.json`, refusing placeholder session IDs, protecting the just-written save
  and the `LastSessionPointer` target. P14 replaced it.) Once D-8 lands, saves no longer
  carry a buffer by default, but the unpruned `-promote` pool still grows without bound on
  a long run. *Since P14 (2026-10-01):* `CheckpointPaths.pruneAutomaticSaves` implements the
  rule below. *Gated (owner decision 2026-10-01):* the rule applies only when the
  `automatic_save_pruning_enabled` setting (default off) is on and the build's kill switch
  `CheckpointPaths.automaticSavePruningForcedOff` is not set — and it is set in the current
  build, so nothing is pruned (see P14, "Deviation — gated and forced off"). The rule:
  - **One combined pool** of automatic saves: `-periodic` **and** `-promote` (arena
    post-promotion), sorted newest-first by folder name (already chronological).
    Folders beyond the newest `max_periodic_autosaves_kept` are deleted whole.
    **Global, not per session (owner, 2026-10-01):** the pool is every such folder in
    `Sessions/`, from any run or session, ranked by the leading UTC timestamp alone.
  - **Parameter:** reuse `max_periodic_autosaves_kept` with its scope widened to the
    combined pool. Its `id` and property name are unchanged (no UserDefaults reset);
    its description and the Sessions-tab help text change to say "periodic and
    post-promotion". **`0` keeps its current meaning — unlimited, prune nothing** —
    and the sweep short-circuits on it exactly as `prunePeriodicAutosaves` does today.
    Full CLAUDE.md checklist: 1–3 and 9 unchanged (no new key); 4 the existing
    `SessionCheckpointState` field and `[RESUME-PARAM]` block already cover it; 5 n/a
    (no training-math effect); 6 every sweep logs the resolved cap and pool sizes
    (`[PRUNE] retention: cap=N periodic=a promote=b`); 7 Sessions tab, existing field;
    8 read from `TrainingParameters.shared` at sweep time.
  - **When it runs:** after every successful `periodic` **or** `promote`-tagged save
    (both the arena's inline post-promotion save and the shared `saveSessionInternal`
    path), detached at utility priority as today.
  - **Never pruned:** `-manual` saves (File ▸ Save Session) and **SIGUSR2 saves
    (`-sigusr2`) — exempt by owner decision 2026-10-01**: both are deliberate "keep
    this" saves. Also never pruned: the just-written save (`protecting:`, as today) and
    the current `LastSessionPointer` target, even if a different save wrote it.
  - **"Promote Trainee Now" saves — RESOLVED 2026-10-01 (owner): treat like automatic
    promotions.** They keep the shared `promote` disk tag (now declared once as
    `SessionSaveTrigger.promotionDiskTag`, which the arena's inline save also writes) and
    are pool members exactly like arena post-promotion saves; no new tag, no filename-parser
    changes. *Superseded text, kept for the record — was OPEN (owner to confirm).* Recorded
    rule: treat them like manual saves — **never pruned** — unless the owner says otherwise. They
    are user-initiated, like a manual save. Consequence: they need their own disk
    tag (e.g. `-manualpromote`), because today they share `promote` with arena
    promotions and `session.json` does not record the trigger. That means checking every
    filename parser that keys off `-promote.dcmsession` and updating
    `documentation/disk-cleanup.md`. (The superseded plan had recommended the
    opposite — sweep them into the pool — before this decision.)
  - **Dropped from the superseded plan:** the separate **"Save Session (Weights Only)"**
    menu item — redundant under D-8, where every save is weights-only by default and
    the manual Save Session sheet has the include-buffer checkbox; and the
    **write-then-strip** approach (write every autosave's buffer, then delete
    `replay_buffer.bin` from older saves and rewrite `session.json`) — under D-8 the
    buffer is decided before the write, so there is nothing to strip. Its separate
    time-based window (`autosave_weights_retention_hours`, default 72 h) is **not**
    adopted either; with the count cap alone the pool's span is cap × save interval.
    If the owner wants a time window as well, that is a new decision.
  - Per-folder failures log `[PRUNE-ERR]` and do not abort the sweep (as today).
  - **Safety checks carried over and generalized (P14, 2026-10-01):** a folder is a pool
    member only if its name is exactly `YYYYMMDD-HHMMSS-<sessionID>-(periodic|promote).dcmsession`;
    it is counted (and so can ever be deleted) only if that session ID is a minted one
    (`yyyymmdd-N-XXXX`, `isMintedSessionID`), the entry is a real directory (`lstat`, not a
    symbolic link), and it holds a regular `session.json` whose `sessionID` equals the ID in
    the name. Anything else — placeholder IDs included — is kept, logged with the reason,
    and not counted. A protected folder keeps its rank (beyond the cap it is kept without
    pulling an older one back under). Deletion is `FileSafety.removeOwnedItem` with the
    identity recorded at inspection, so a folder swapped in under the same name between
    inspection and deletion is refused, not removed.

---

# PART E — Implementation plan

## E1. Phases and dependencies

```
P1 RNG core ──┬─> P3 Wire streams (sampler, sampler-order, MoveSampler, draw-keep, BN-cal, probes)
              ├─> P4 Dropout state seed/capture/restore
              └─> P5 Per-tensor init + WeightInitialization ──> P7 Init-neutral options (format v5 arch fields)
                                                            └─> P8 Graft derive op
P2 Resume parameter resolution (#5)   (independent; can land first)
P6 Format v5 + LineageRecord + writers (needs P1 for rng fields; P2 for params snapshot)
      ├─> P9  Exact-resume completion (buffer order, vs-UCI buffer, clocks, KL counter, C1 list)
      ├─> P10 [RUN] line, results.json, derivation_history carry-forward (B4)
      └─> P11 Dashboard derive-registry
P12 Resume-equivalence harness (needs P3,P4,P6,P9)  — its unit parts grow with each phase
P13 Config D removal (D-10)   (independent; must land before P9)
P14 Autosave retention pool (D-8 addition, 2026-10-01)   (independent; IMPLEMENTED 2026-10-01; gated, forced off)
```

Foundational: **P1** (generator) and **P6** (format v5). Independent: **P2**,
**P13**, **P11** (tracker, once P6 defines the schema), **P7/P8** after P5.

Per memory rule "multi-phase plans: build + commit per phase", each phase ends
with one build and one commit; targeted tests only while trainings are live; full
suite before merge.

## E2. Phase details

**P1 — `DCMRandom` + streams.** Files: `Utils/DCMRandom.swift`,
`Utils/DCMRandomStreams.swift`, `DrewsChessMachineTests/DCMRandomTests.swift`.
Tests: A2.4 goldens. Validation: goldens match reference C output; no call sites
changed yet (zero behavior change).

**Done (`c8c3e3c2`, 2026-10-02).** As built:
- `DCMSplitMix64` (reference `next()` plus `mix(_:)` = first output from a state,
  the `splitmix64(x)` of the child-seed derivation); `DCMRandom` (xoshiro256**,
  `init(seed:)` via SplitMix64, throwing `init(s0:s1:s2:s3:)` refusing all-zero,
  `next`, `jump`, Lemire `nextBounded` for `UInt64` and `Int`, `nextUnitDouble`,
  `nextUnitFloat`, `Codable` as four decimal-string words — a numeric, signed,
  hex, empty or out-of-range word is refused); `MutableCollection.stableShuffle(using:)`
  (Fisher–Yates from the last position down with `nextBounded(i + 1)`);
  `DCMRandomStreams` (`stableHash64` = SHA-256 first 8 bytes big-endian,
  `childSeed(parent:name:)`, `generator(_:)`) and `DCMStream` (sampler, dropout,
  `selfplay.game.<serial>`, `arena.<a>.game.<g>`, `vsuci.game.<serial>`,
  `probe.<name>.<step>`, `corpus.order.<epoch>`).
- Goldens: SplitMix64 from 0 and `0x0123456789ABCDEF`, xoshiro256** from state
  (1, 2, 3, 4) and after `jump()`, and the SplitMix64-seeded state from seed 42
  were generated by compiling the reference C algorithms; the bounded draws (bounds
  1, 2, 3, 7, 4864, 2^32+1, `UInt64.max`), unit draws (bit patterns), shuffle and
  child seeds by an independent Python implementation whose raw xoshiro output
  matched the C reference. 24 tests; written first (red: the new API did not
  exist), green after.
- Deviations: (1) `nextStandardNormalPair` is **not** in P1 — it depends on
  `DCMNormalMath`, which E2 assigns to P5, so it lands there with its exhaustive
  accuracy tests and normal goldens. (2) No separate `DCMRandomState` type:
  `DCMRandom` itself is the `Codable` state (one source of truth). (3) The `init/…`
  family is not a `DCMStream` case: per-tensor init seeds derive from a model's
  init seed (B1.1), not the run's master seed; P5 builds them with the same
  `childSeed`. (4) Stream serials and indices are `Int`.

**P2 — Resume parameter resolution (#5).** Files: `Packages/TrainingParametersMacro/*`
(`absentValue`, UInt64 kind for P3's seed), `Training/TrainingParameters.swift`,
`App/SessionController+Training.swift` (45 branches → one resolver),
`CLI/CorpusReplayRunner.swift`, `CLI/TrainVsUciRunner.swift`, CLAUDE.md checklist.
Tests: C5 list (Exp 6 regression first — must fail before the fix, pass after,
unmodified). Validation: `[RESUME-DIFF]` on a legacy fixture session; no
`UserDefaults` writes for defaulted keys.

**P3 — Wire streams.** Files: `Training/ReplayBuffer.swift` (sampler RNG under
existing lock; all 9 sites), `Network/MoveSampler.swift` (inout rng),
`Training/ActiveGame.swift`, `Training/BatchedSelfPlayDriver.swift`,
`Arena/TickTournamentDriver.swift`, `Training/TrainVsUciDriver.swift`,
`App/UCI/UCIEngine.swift`, `Network/MPSChessPlayer.swift`,
`Network/ChessMPSNetwork.swift` (BN-cal), `Training/ReplayBufferAnalyzer.swift`,
`Training/TrainingParameters.swift` (+ seed params, full checklist A3.2),
`App/UpperContentView/TrainingSettingsPopover*.swift`, `CLI/CliTrainingConfig.swift`
(`--seed`), `CLI/CliTrainingRecorder.swift`.
Tests: same seed ⇒ identical `sample()` index sequence on a fixture buffer;
per-game stream independence from worker count (K=1 vs K=8 produce the same game
for serial s given identical network outputs — use a stub `evaluateBatched`
returning fixed logits, which is a *test double*, not a production stub);
Dirichlet/Gamma distribution sanity (mean/var) with the new draws.
Also in P3 (A6): logical age-ordered index mapping in every `sample()` path (O7b);
the fixed-key board hash (O16 — **bug fix: its regression test is written and
committed first and must fail on the current code**, then pass unmodified), with
`ReplayBuffer.fileVersion` 7 → 8 and the D-9 legacy recompute on load (its test:
a checked-in v7 fixture buffer loads with every hash equal to the fixed-key hash
of its board, and logs the recompute count); the
recorded `Set`/`Dictionary`, sort-tie and float-sum sweeps (O17–O19), each
appended to A6 as a classified table.
Validation: `[RUN] seed=` logged on all paths; unseeded mode logs a drawn seed;
`sample()` µs per batch re-measured and not slower than today's (A2.2 baseline).
Risk: sampler lock — RNG work moves inside the existing lock; xoshiro is faster
than `SystemRandomNumberGenerator` (which calls `arc4random_buf`), so hold time
drops. Measure `sample()` µs before/after with the existing timing taps.

**P4 — Dropout state.** Files: `Network/ChessNetwork.swift` (placeholder assign,
no baked constant), `Training/ChessTrainer.swift` (`captureDropoutState()`,
`restoreDropoutState(_:)`, seed from stream; KL counter removed → step-derived),
`Training/TrainerResumeState.swift` (snapshot includes Philox state + sampler state).
Tests: capture/restore mask equality (A4); KL-probe-derived schedule equals old
cadence on a fresh run; masks for steps N+1… identical with the KL probe on and
off (C2 probe-isolation rule). Also: add the probe-isolation rule to the project
`CLAUDE.md` "Concurrency invariants". Validation: `[RESUME] rng: dropout=restored`.

**P5 — Per-tensor init.** Files: `Network/ChessNetwork.swift` (all init sites;
`WeightInitialization`), `Network/ChessMPSNetwork.swift`,
`Network/InferenceNetworkFactory.swift`, every `ChessNetwork(` construction site
(pass `.overwrittenByLoad` for load paths), `Persistence/ModelDerivation.swift:521`,
`Utils/DCMNormalMath.swift` (D-3 restricted-domain `log`/`cos`, private to
`DCMRandom`), `scripts/init_reproducibility.sh` (B1.1 cross-machine test),
`App/UpperContentView/BuildNewModel*.swift` (D-2 optional **Init seed** field,
A3.2 item 7).
Tests: `DCMNormalMath` exhaustive 2²⁴-input accuracy tests (A2.4); same seed ⇒
bit-identical weights across two builds; two archs sharing
`block0_*` ⇒ identical `block0_*`; fp32 vs bf16 builds: bf16 == round(fp32);
per-role std within tolerance (existing `NetworkWeightAnalyzer` expectations);
`.overwrittenByLoad` network refuses use before load. Validation: build time for
the default preset logged; ≤ 2× today's; **re-benchmark of the finished
transform** against vForce and Swift libm on 8.45M values (same harness as the
A5 prototype), numbers reported to the owner and recorded in A5; cross-machine
script run by the owner on both machines, all tensor-data hashes equal;
Build New Model with an entered init seed mints tensor data identical to
`--derive-model`/CLI mints with the same seed, and an empty field logs the drawn
seed.

**Done (`3b601c50`, 2026-10-02).** As built:
- `Utils/DCMNormalMath.swift`: `logOfGridUniform(k)` = `ln((k+1)/2²⁴)` by a
  power-of-two split, mantissa folded into (√½, √2], and an odd `atanh` series in
  Horner form; `cosSinOfGridAngle(k)` = exact quadrant and octant reduction on the
  grid, then Taylor `sin`/`cos` on [0, π/4]; `standardNormalPair(radiusIndex:angleIndex:)`
  rounds `r·cos`, `r·sin` to `Float`. `DCMRandom.nextStandardNormalPair()` takes two
  `next() >> 40` draws, always exactly two. No vForce, no libm anywhere on the path.
- `Network/WeightInitialization.swift`: `WeightInitialization` (`.seeded(initSeed:)`,
  `.overwrittenByLoad`, `drawnSeed()`); `WeightInitScheme` (`current = "dcm-init-1"`,
  `tensorSeed` = `DCMRandomStreams.childSeed(parent: initSeed, name: "init/" + name)`,
  `bnCalibrationSeed` = the `init.bn_calibration` child, `standardNormals`, fans from the
  tensor's own stored shape, He / Glorot std in fp32, `nativeValues` = draw row-major in
  the stored layout, scale, then `SafetensorsModelIO.fromTorchLayout`, and
  `seFC2NativeValues` = the whole Glorot draw with the zero-β native columns zeroed);
  `TensorInitializer` checks every builder request against `weightTensorPlan()` (name
  present, same native shape, at most once) and, after the build, that every conv /
  linear plan tensor was drawn.
- `ChessNetwork`: every random weight site goes through the initializer by plan name;
  `.overwrittenByLoad` fills zeros and the network refuses `evaluate`, batched
  evaluate, value distribution, value baseline, analysis taps, `exportWeights`,
  `computeBatchStats` and `trainStep` with `ChessNetworkError.weightsNotLoaded` until
  `loadWeights` runs. The old vForce helpers are gone; the two static helpers
  `PolicyHeadCorrectnessTests` calls (`heInitData`, `glorotInitDataFCInOut`) draw
  through the same transform from a system-drawn seed.
- `ChessMPSNetwork`: `NetworkInitMode` gains `.seededRandomWeights(initSeed:)` and
  `.weightsToBeLoaded`; the BN-calibration warmup game is walked with
  `nextBounded(legalMoves.count)` from the mint's `init.bn_calibration` stream, and its
  sibling network is built `.overwrittenByLoad`.
- Load paths pass `.weightsToBeLoaded` / `.overwrittenByLoad`: the session's
  persistent networks, model/session load, probe, Run All Analyses, numerics audit,
  UCI loader, checkpoint verification scratch, the train-vs-UCI eval net, the
  live-trainer mirror (`InferenceNetworkFactory.buildAwaitingLoad`), and corpus-replay /
  train-vs-UCI trainers started from a model. Fresh corpus-replay and train-vs-UCI
  trainers draw a seed and log it (`[REPLAY] fresh nets init_seed=`,
  `[VS-UCI] fresh trainer init_seed=`).
- Mints: `--new-model [--init-seed <u64>]` and Build New Model's optional
  **Init seed** field (empty = drawn; the Build button is disabled on an invalid
  entry) both build `.seededRandomWeights(initSeed:)` — the same code path, so the
  same seed gives the same tensors. The seed and scheme are logged (`[NEW-MODEL] …
  init_seed=… (entered|drawn) init_scheme=dcm-init-1`, `[BUTTON] Build Network …
  init_seed=…`) together with the build time (`[NEW-MODEL] built … in … ms`,
  `[BUILD] champion … built in … ms`).
- `--derive-model --set-se-beta-init glorot [--init-seed <u64>]` re-draws β with the
  tensor's own `init/<name>` stream (the β rows equal a fresh mint's under that seed)
  and records `init_seed` / `init_scheme` in the operation's `derivation_history`
  arguments; `--init-seed` with no weight-drawing operation is refused.
- `scripts/init_reproducibility.sh <binary> <scratch>` mints seeds 1, 2, 42, 1234,
  99999, 2⁶⁴−1, 7777777, 31337 × `v4_5block_7x7` and `nt8y_3x3stem` and prints, per
  mint, the SHA-256 of the trainable tensors and of the BN running statistics
  (`scripts/safetensors_tensor_hash.py`, tensor data only).
- Tests (all new; written before the implementation, red only because the API did
  not exist yet — compile-red): `DCMNormalMathTests` (7: exhaustive `log` and
  `cos`/`sin` over all 2²⁴ inputs — measured worst ≤ 4 ULP relative and ≤ 8 ULP
  absolute respectively; quadrant boundaries exact; each normal within 1 `Float` ULP of
  Foundation's; golden pairs; two draws per pair; moments), `InitSchemeGoldenTests` (8:
  tensor seeds, BN-calibration seed, conv / FC / zero-β SE fc2 goldens in fp32 and bf16
  with full-tensor SHA-256, odd count, non-random kinds refused, per-role std),
  `WeightInitializationTests` (12: same seed bit-identical, builder equals the scheme by
  plan name, shared tensors identical across architectures, bf16 = round(fp32), every
  preset builds seeded, the load gate, seeded mints reproducible including BN
  calibration, plan-check errors), `InitSeedRecordingTests` (7: metadata round trip and
  malformed records, legacy writer refuses a seed, seeded β re-draw equals the mint,
  Build-screen seed entry). Every golden came from an independent Python
  implementation of the scheme, not from the Swift code. Related existing classes (51
  classes, 466 tests) pass; the only failures in that run were
  `ChessNetworkValueDistributionTests`' two tests, which evaluate an
  `InferenceNetworkFactory.build(arch:)` network without loading — that entry point
  stays `.randomWeights` and the mirror got its own `buildAwaitingLoad`; re-run green.
- Build time (Debug build, loaded machine, `--new-model` via the script):
  `v4_5block_7x7` 4.2–5.0 s, `nt8y_3x3stem` 0.7–1.2 s after the first (3.5 s). The
  draw itself is ≈ 12 ms slower than vForce over 8.45M values (A5 re-benchmark),
  negligible against graph construction, so the ≤ 2× bound holds; no pre-P5 timing
  of the same binary was taken.
- Same-machine run of the script on the M4 Pro (macOS 27.2 beta): all 16 mints
  produced their hashes; the cross-machine comparison (M5 VM) is still owner-run.
- Deviations: (1) The init seed lives in flat `init_seed` / `init_scheme`
  safetensors metadata keys (`ModelInitRecord`), written by `--new-model` only, since
  `dcm_lineage` does not exist until P6 — P6 should fold them into
  `dcm_lineage.rng`. A GUI-built champion's seed is logged and shown, not yet stored in
  its saves. The legacy `.dcmmodel` writer refuses a record. (2) The B1.1 golden "WDL
  fc2 bias" is not pinned here: biases are constants, not draws; the pinned tensors are
  a conv, an FC, a zero-β SE fc2 and an odd-count tensor. (3) `DCMNormalMath` is an
  internal type rather than private to `DCMRandom`, so the exhaustive tests can call
  it. (4) `InferenceNetworkFactory.build(arch:)` keeps drawing random weights (an
  existing test uses it that way); the live-trainer mirror uses the new
  `buildAwaitingLoad(arch:)`. (5) `ChessNetwork`, `ChessTrainer` and the
  `TrainerHyperparameters` convenience init still default `initialization` to
  `.drawnSeed()`, because removing the default changes ~90 existing test call sites
  that P4 also edits; production sites all pass it explicitly. The owner's rule (no
  default; tests pass explicit values) is to be applied after this branch is merged
  with P4. (6) `SetSEBetaInitDeriveOperation.init(value:groupIndices:)` draws a seed
  (existing tests use that form); the CLI path always records the seed it used.

**P6 — Format v5 + `LineageRecord`.** Files: `Persistence/LineageRecord.swift`,
`Persistence/LineageTracker.swift`, `Network/ArchitectureFormat.swift` (v5),
`Persistence/SafetensorsModelIO.swift`, `Persistence/SessionCheckpointFile.swift`
(v2), `Persistence/CheckpointManager.swift`, `App/SessionController+Checkpoint.swift`,
both CLI runners, `Persistence/ModelDerivation.swift`.
Tests: v5 round trip; v4/v3/`.dcmmodel` fixtures load/train/infer unchanged
(E4); v5 file missing `dcm_lineage` rejected; exact-resume chain of 3 synthetic
segments: run id stable, index increments, `parent.content_sha256` equals parent
file's, cum counters monotone; branch mints new run id.
Validation: a real 200-step replay run and a GUI session each write v5 files
whose `dcm_lineage` decodes and whose flat mirrors equal the struct; E4 legacy
fixtures all pass.

**P7 — Init-neutral options** (B2, B2.1; D-4): `se_gamma_bias_init`,
`branch_output_init`, `skip_projection_init`, `policy_head_final_init`,
`value_head_final_init`, `value_head_draw_prior`; ReZero `> 0` validation.
Files: `Network/NetworkArchitecture.swift`, `Network/ChessNetwork.swift`,
`App/UpperContentView/BuildNewModel*.swift` (fields, Neutral/Standard init
buttons, highlight-vs-standard, tooltips; new View structs in their own files),
`ArchitectureDiagramView.swift` (highlight), `Presets/*.json`,
`Persistence/ModelDerivation.swift` + `App/DeriveModelCLI.swift` (one
shape-preserving `--set-<option>` derive op per option, plus
`--set-neutral-init`).
Tests per option: exact step-0 behavior (e.g. uniform policy logits; value
softmax equals prior; SE gate = σ(bias); zero last-BN γ ⇒ branch output 0);
one step makes the zeroed tensor nonzero; legacy hashes/summaries unchanged;
forbidden options rejected by validation. Model tests: Neutral then Standard
restores the standard architecture exactly; the "differs from standard" set is
exactly the fields changed. Derive tests: each `--set-<option>` rewrites only
its tensors (all others bit-identical) and records history.
Validation: build a neutral-init model in the GUI and screenshot the editor and
diagram in light and dark mode showing every changed field highlighted; its
summary line and preset JSON list the options; a `--set-neutral-init` derive of a
standard fresh net matches the GUI neutral mint's option values.

**P8 — Graft** (B3). Files: `Persistence/ModelDerivation.swift`, `App/DeriveModelCLI.swift`.
Tests: graft 5-block→6-block copies blocks 0–4 bit-exact, initializes block 5 from
`init/block5_*` under the recorded seed (reproducible), records copied/dropped/initialized;
graft with `--graft-map` for an inserted block; refuses velocity-carrying sources.
Validation: a graft of a real v5 5-block model to a 6-block preset loads,
infers and trains one step; its `derivation_history` lists copied/initialized/
dropped tensors matching the target plan.

**P9 — Exact-resume completion** (C1 #3–#20, #23–#33; D-5…D-8). Files:
`Training/ReplayBuffer.swift` (slot-source SoA column; age-order refill API that
starts at slot 0 and sets `writeIndex`; D-5 per-bucket age-ordered FIFO with
k-th-oldest pick replacing swap-remove), `CLI/CorpusReplayRunner.swift`
(age-order + cross-epoch refill from the oldest resident source; feed carry;
shard SHA-256 check; build/OS check; segment-indexed enum names; D-7
`--accept-inexact` semantics via the shared `ResumeExactness`),
`CLI/TrainVsUciRunner.swift` (D-8: session-folder checkpoints via the shared
session writer, buffer only with `--save-replay-buffer`; serials),
`App/SessionController+Training.swift` / `+Checkpoint.swift` (arena clock with
D-6 discard-and-rerun, save clock, ratio controller, diversity, alarms, serials,
build/OS check, D-8 buffer gate from `session_save_include_replay_buffer`, resume
without buffer), `App/DrewsChessMachineApp.swift` + a new manual-save sheet View
(D-8 checkbox), `Training/TrainingParameters.swift` +
`App/UpperContentView/TrainingSettingsPopover*.swift` (D8 parameter, full
checklist), `Arena/ArenaTriggerBox.swift`, `Training/ReplayRatioController.swift`,
`Training/GameDiversityTracker.swift`, `App/UpperContentView/TrainingAlarmController.swift`.
Requires P13 (config D removed).
Tests: C6 step 6 round trips (incl. with/without buffer, D-6 arena, D-9 legacy
buffer); refill slot-by-slot equality in age order vs the uninterrupted run,
incl. epoch wrap; k-th-oldest bucket picks identical after a mixed insert/evict
history vs an age-order rebuild; games fed before each post-resume step equal
the uninterrupted run's (feed carry); a changed shard byte ⇒ refusal naming the
shard, even with `--accept-inexact`; a changed git hash / OS string ⇒ warning,
and `--resume-exact` refusal without `--accept-inexact build`/`os`; a v4
checkpoint resumes only with `--accept-inexact` naming every gap, and the segment
records them.
Validation: C6 step 7 end to end on the real runner; a GUI session saved with
the default settings has no `replay_buffer.bin` and its save log line says
`buffer=omitted`, resumes, refills and trains; the same with the toggle on
writes and restores the buffer; `sample()` µs per batch not slower than the A2.2
baseline.

**P10 — Provenance + carry-forward.** `[RUN]` formatter (`Logging/`), recorder
fields, B4 fix. Tests: derive → train → save keeps `derivation_history`; `[RUN]`
contains all fields (string test on the formatter).
Validation: each run path (GUI, `--train`, `--train-from-corpus`,
`--train-vs-uci`, `--derive-model`, `--uci`) prints one `[RUN]` line at start in
a real launch; a `results.json` from a short replay run carries `lineage`.

**P11 — Tracker.** `documentation/dashboards/replay.py`, `ckpt_inventory.py`,
`selfplay.py`, `vsuci.py`; pytest fixtures. Validation: on existing v4 data,
output identical to today (no regressions); on synthetic v5 headers, expected registry.

**P12 — Harness** (C6). `DrewsChessMachineTests/ResumeEquivalenceTests.swift`,
`scripts/resume_equivalence.sh`, `test_tiny` preset + corpus fixture.
Validation criterion for the whole plan: corpus replay N+M vs N|resume|M gives
identical sampled indices and dropout masks for all M steps, and identical final
weights when the determinism probe reports `bitExact`.

**P13 — Config D removal** (D-10, issue #9; before P9). Delete the
`--bf16-cast-in-forward` flag parsing (`App/DrewsChessMachineApp.swift:264-273`),
the `SessionController` wiring (`App/SessionController.swift:944-956`), the
`TrainerHyperparameters` field (`:137, :155`), the `ChessTrainer` property and
branches (`:1560-1565, :2023, :2044, :2047, :2123, :2265, :2340, :3951-3957`),
and in `ChessNetwork`: `bf16CastInForward`, `bf16CastActive`,
`weightStorageDataType`, `castWeightForForward` and their use sites. **Approved
test deletions (owner, D-10):** the config-D case in `HeadNumericsTailTests`
(`:66-67`) and the config-D sweep in `MacOS27NaNIsolationTests` (`:657-713`); no
other test is modified. Docs: mark it removed (with history kept) in
`documentation/macos27-beta1-mpsgraph-findings.md`, `HEAD_NUMERICS_PLAN.md`,
CHANGELOG; ROADMAP note only with owner permission.
Tests: the default bf16 path's graph is byte-identical before/after (architecture
hash and a pinned forward-pass output hash on a bf16 fixture); head-numerics,
fp16 and `CheckpointManagerSafetensorsTests` pass.
Validation: build succeeds; `grep` finds no remaining `bf16CastInForward` /
`castWeightForForward` / `--bf16-cast-in-forward`; passing the old flag is now an
unknown-argument error; issue #9 closed with the commit.

**P14 — Autosave retention pool** (ADDITION 2026-10-01; D-8 Retention sub-bullet, §D8
"Autosave retention"). **IMPLEMENTED 2026-10-01** (uncommitted until the owner commits;
build, tests and the validation run below are the owner's). Independent of every other
phase. Files:
`Persistence/CheckpointManager.swift` (replace `prunePeriodicAutosaves` with one sweep
over the combined `-periodic` + `-promote` pool, built on a pure-logic decision function
that partitions folder names into keep/delete, mirroring `PeriodicSaveController`'s
testable-scheduler pattern), `App/SessionController+Checkpoint.swift` (run the sweep after
`periodic` and `manualPromote` saves — or only after `periodic` if "Promote Trainee Now"
gets its own tag), `App/SessionController+Arena.swift` (run it after the inline
post-promotion save), `Training/TrainingParameters.swift` +
`App/UpperContentView/TrainingSettingsPopover.swift` (description and help text for
`max_periodic_autosaves_kept`), `Persistence/LastSessionPointer.swift` (read the protected
target), `documentation/disk-cleanup.md`. If the owner confirms the open
"Promote Trainee Now" rule: `Persistence/SessionSaveTrigger.swift` (new disk tag) and every
parser of `-promote.dcmsession`.
Tests (new; no existing test modified): the decision function, given folder names of
every trigger, keeps exactly the newest `cap` of `-periodic` + `-promote` combined and
never selects `-manual`, `-sigusr2`, the protected URL or the `LastSessionPointer`
target; boundaries — exactly `cap` automatic folders present, `cap == 0` selects nothing,
a protected folder older than the cap is kept without shifting the count of others;
mixed periodic/promote ordering is by timestamp, not by trigger.
Validation: a GUI run (or `--train --parameters <file>`) with
`max_periodic_autosaves_kept: 2`, a short `periodic_autosave_interval_sec` and a short
arena interval, long enough to write several periodic and promotion saves plus one
manual save and, last (it shuts the process down), one SIGUSR2 save; `ls Sessions/` and the `[PRUNE]` lines show only the
newest two automatic folders remaining, every `-manual` and `-sigusr2` folder untouched,
and the `LastSessionPointer` target present. With the cap at `0`, nothing is deleted
(today's behavior). Resume from the newest surviving autosave works. Full test suite
passes.

*As implemented (2026-10-01), with deviations from the text above:*
- `Persistence/CheckpointManager.swift`: `prunePeriodicAutosaves` /
  `planPeriodicAutosavePrune` / `isPeriodicAutosaveFolderName` /
  `inspectPeriodicAutosaveOwnership` / `periodicSessionSuffix` replaced by
  `AutomaticSaveKind` (pool membership by disk tag), `parseAutomaticSaveFolderName`,
  `inspectAutomaticSaveFolder` (returns `.verified(identity:)` or a kept reason),
  the pure `planAutomaticSavePrune`, and the sweep `pruneAutomaticSaves(keeping:protecting:in:lastSessionPointerDefaults:)`.
  The pool is **global** (every session), per the owner's 2026-10-01 clarification; the
  previous per-session rule and its refusal to run for a non-minted saving session are
  gone — a non-minted ID is now judged per folder (kept and logged).
- Deletion uses `FileSafety.removeOwnedItem` with the inspected identity instead of a
  by-name `removeItem` (not in the original text).
- "Promote Trainee Now" resolved as an automatic promotion, so **no new disk tag** and
  no `-promote.dcmsession` parser changes; `Persistence/SessionSaveTrigger.swift` instead
  gained `promotionDiskTag` (single source for the arena's literal `"promote"`) and
  `CaseIterable` (for the trigger-selection test).
- Trigger: `SessionController.scheduleAutomaticSaveRetentionSweep(afterSaving:diskTag:)`
  (`App/SessionController+Checkpoint.swift`) decides from the disk tag via
  `AutomaticSaveKind(diskTag:)`, reads the cap live, and detaches the sweep at utility
  priority. Called exactly once per successful save from `saveSessionInternal` (periodic,
  Promote Trainee Now; manual and SIGUSR2 filtered out) and from the arena's inline
  post-promotion save (`App/SessionController+Arena.swift`).
- `cap == 0` still short-circuits (no listing, nothing deleted) but now logs one
  `[PRUNE] retention: cap=0 (unlimited)` line instead of being silent, so the cap is
  visible in the log (CLAUDE.md checklist item 6).
- `Persistence/LastSessionPointer.swift` needed no change (`LastSessionPointer.read` is
  used as is).
- Tests: the pruning tests live in `DrewsChessMachineTests/CheckpointHousekeepingTests.swift`,
  which was itself new and uncommitted; its earlier per-session pruning tests were
  rewritten to the global rule (no committed test was modified). Added cases: pool spans
  sessions; periodic/promote ordering by timestamp (sweep and planner); manual/SIGUSR2
  never pool members and not counted; resume-pointer target (from another session) and
  just-written save kept; unverified kinds kept and not counted (regular file, no
  `session.json`, `session.json` a directory, garbled JSON, other session's
  `session.json`, symbolic link, placeholder ID); cap boundaries (exactly cap, cap 0,
  protected beyond cap); folder-name parsing exactness; minted-ID shape; a folder swapped
  in after inspection is refused by the identity check; trigger selection over every
  `SessionSaveTrigger` case plus the arena's `promotionDiskTag`. The arena path's call
  itself needs a GPU session and is covered by the validation run above.
- Docs: `CLAUDE.md` ("Saved model state"), `documentation/disk-cleanup.md`, the parameter
  description (and the generated `documentation/parameters.md`), and the Sessions-tab help
  text.
- **Deviation — gated and forced off (owner decision 2026-10-01).** The pruning code is
  kept, but it no longer runs on the cap alone, and in the current build it does not run
  at all:
  - **New setting** `automatic_save_pruning_enabled` (`AutomaticSavePruningEnabled`,
    `Bool`, default `false`, live-tunable, category Sessions). It contradicts the text
    above ("checklist 1–3 and 9 unchanged (no new key)"): there is now a new key, with the
    full CLAUDE.md checklist walked — `allKeys`, singleton property / `collectValues` /
    `applyOne` / snapshot accessor; JSON defaults are registry-driven; an Optional
    `SessionCheckpointState.automaticSavePruningEnabled` written by
    `buildCurrentSessionState` and restored by a `[RESUME-PARAM]` block beside
    `max_periodic_autosaves_kept`'s (both branches log); `results.json` n/a (no training
    effect); Sessions-tab "Prune old autosaves" toggle; read live at sweep time.
    `max_periodic_autosaves_kept`'s description now says it applies only when pruning is on.
  - **Kill switch** `CheckpointPaths.automaticSavePruningForcedOff`, a `static let` set to
    `true`, pending D-8 (saves without the replay buffer by default) and more confidence in
    automatic deletion. Changing it to `false` is the only way to re-enable pruning; the
    setting alone cannot, since it can arrive from UserDefaults, a resumed `session.json`,
    or a `parameters.json`/CLI override. The Sessions-tab toggle is disabled with a
    "Disabled in this build" note while it is set.
  - **One pure decision**, `CheckpointPaths.automaticSavePruningDecision(forcedOff:settingEnabled:cap:)`
    → `.prune(keeping:)` only when `!forcedOff && settingEnabled && cap > 0`, otherwise
    `.forcedOff` / `.disabledBySetting` / `.unlimitedCap` (checked in that order).
    `scheduleAutomaticSaveRetentionSweep` still filters by disk tag first, then passes the
    constant to this function; when the answer is not `.prune`, it logs exactly one
    `[PRUNE] skipped after <folder>: off: <reason> (automatic_save_pruning_enabled=… max_periodic_autosaves_kept=…)`
    line instead of detaching the sweep (so a cap of 0 now logs that line rather than
    `pruneAutomaticSaves`' own `cap=0 (unlimited)` line, which remains for direct calls).
    Play-and-Train start logs the effective state once:
    `[PRUNE] automatic-save pruning at Play-and-Train start: …`.
  - **Tests** (new file `DrewsChessMachineTests/AutomaticSavePruningGateTests.swift`; no
    existing test modified): the shipped constant is `true`; forced off never prunes for any
    setting or cap; with the switch injected as lifted — setting off never prunes, on + cap
    0 does not, on + cap > 0 prunes to that cap; the declared defaults do not prune even with
    the switch lifted; log text; the parameter's id, default, category, live tunability,
    value round-trip and presence in the emitted defaults; the `session.json` field's
    round-trip and absence in older sessions. The existing `test_registry_size` count in
    `TrainingParametersTests` has to go up by one for the new key — an edit to a committed
    test, which needs the owner's approval.
  - **Validation (replaces the run above while the switch is set):** periodic and
    promotion saves each log one `[PRUNE] skipped … forced off in this build …` line and
    `Sessions/` keeps every folder. The validation run above applies once the switch is
    lifted and the setting is turned on.

## E3. Risks

- **CPU init time** (P5): scalar per-tensor streams; mitigated by per-tensor
  parallelism (values unchanged) and by `.overwrittenByLoad` removing init from
  every load path (net speedup for resume/arena setup).
- **Sampler lock** (P3): RNG draws already happen under the buffer lock; xoshiro
  is cheaper than the system RNG. Measure; no expected regression.
- **Self-play hot path** (P3): per-game `DCMRandom` is 32 bytes and lock-free; the
  current `Float.random` goes through `SystemRandomNumberGenerator`
  (`arc4random_buf` per call) — expected speedup.
- **Philox blob semantics across OS updates** (P4): opaque; canary test.
- **Own `log`/`cos` accuracy** (P5, D-3): removed as a drift risk by design; the
  remaining risk is a coefficient mistake, which the exhaustive 2²⁴-input tests
  catch. Distribution checks (`NetworkWeightAnalyzer` per-role std) confirm the
  normals are still He/Glorot-normal.
- **Config D removal** (P13, D-10): deleting a mixed-precision path could disturb
  the default bf16 graph; guarded by the byte-identical graph/output pins.
- **Buffer not saved by default** (D-8): a default resume trains its first steps
  on a small buffer from the current champion only (less varied) and waits for
  the minimum fill. Accepted by the owner; the toggle/checkbox/flag restore the
  old behavior per save.
- **Bucket FIFO assumption** (P9, D-5): relies on the ring only evicting the
  globally oldest slot; verified in P9 before swap-remove is replaced.
- **Shard re-hash cost** (P9): exact resume reads the needed shards once at
  startup to verify SHA-256 (C1 #32); sequential I/O, measured and logged.
- **Macro changes** (P2/P3): `TrainingParametersMacro` edits affect every
  parameter; covered by existing parameter tests + new `allKeys` coverage test.
- **Format v5 on live runs**: the three running SE seed-2 corpus-replay trainings
  are v4 writers on an old binary and will not be resumed (D-7). Any v4
  checkpoint resumed later on a v5 build is `NOT EXACT: rng_sampler,
  dropout_state, feed_carry, params, lineage` and needs `--accept-inexact`
  naming them (C3).
- **Scope creep in GUI**: C1 #12/#18/#20 are best-effort by nature (§0).

## E4. Backward-compatibility guarantees and tests

Guarantees:
1. Every existing `.safetensors` (v ≤ 4, with/without velocity, with/without
   `trainer_*`, derived or not), `.dcmmodel`, and `.dcmsession` (v1) **loads**,
   **infers** (bit-identical outputs to the pre-change build on a fixed input
   batch), and **trains** (continues as branch, or exact per d15f706 rules).
2. Existing arch hashes, summaries and preset files decode identically.
3. Random-init changes affect only **new mints**.
4. `--uci` behaves identically for a given file, Temperature 0 (deterministic
   already); Temperature > 0 now uses a per-game stream (drawn seed logged).

Tests (fixtures checked in under `DrewsChessMachineTests/Fixtures/legacy/`, tiny
arch so they're small): a v3 unversioned file, a v4 file with `se_beta_init`, a
v4 trainer file with velocity + `trainer_*`, a derived v4 file with
`derivation_history`, a `.dcmmodel`, a v1 `session.json` fixture. For each:
decode; forward pass equals a pinned output hash computed **before** the change
(generate the pins in P1 from the current build, commit them, then never modify
them); one training step succeeds; resume classification lists the expected gaps.
Plus `CheckpointManagerSafetensorsTests` must keep passing bit-exact.

---

# Owner decisions (all decided)

- **D-1 — DECIDED (2026-09-30):** GUI self-play and train-vs-UCI resumes are
  state-exact at most — and, under D-8, not buffer-exact unless the save included
  the buffer; they never reproduce the rest of the run. Corpus replay is the only
  trajectory-exact path (§0, C3, C6).
- **D-2 — DECIDED (2026-09-30):** GUI default is `unseeded` — a seed is drawn,
  logged and recorded in lineage. Build New Model also gets an optional seed
  field, so a GUI user can mint a model from a chosen seed.
- **D-3 — DECIDED (2026-09-30):** keep He-normal / Glorot-normal, computed by
  our own restricted-domain Box–Muller (polynomial `log`/`cos` from IEEE-exact
  operations), with exhaustive accuracy tests, golden bits, a cross-machine
  test and a re-benchmark on completion (A5, B1.1, P5). No He-uniform option
  exists or ever existed; none is added.
- **D-4 — DECIDED (2026-09-30):** ship the recommended B2 set (SE γ bias level,
  branch last-BN-γ zero, skip-projection identity-like, policy/value final zero,
  draw prior); the forbidden list stays forbidden. UI/format requirements:
  - every option is its own architecture field (per block group or per head, like
    `se_beta_init`), required from format v5, shown in presets, the summary line
    and `ArchitectureDiagramView`;
  - Build New Model gets a **Neutral init** button (sets every option to its
    neutral value) and a **Standard init** button (reverts);
  - any field that differs from the standard init is highlighted in the editor
    and the diagram (accent tint plus a marker, not color alone), compared
    against the standard init so the highlight survives reopening a model;
  - each highlighted field has a tooltip stating what changes at step 0;
  - `--derive-model` can apply every option, re-initializing only the affected
    tensors, as it does for `se_beta_init`.
- **D-5 — DECIDED (2026-09-30):** rebuild the bucket slot arrays in age order on
  refill, and pick "the k-th oldest slot in the bucket" so stored array order
  never matters. Nothing extra is persisted.
- **D-6 — DECIDED (2026-09-30):** an arena interrupted by a kill is discarded and
  re-run on resume; no partial arena records are persisted.
- **D-7 — DECIDED (2026-09-30):** keep `--accept-inexact`. `--resume-exact`
  refuses a checkpoint missing any exact-resume state; `--accept-inexact` allows it,
  seeds each missing stream freshly (logged), and marks the segment
  `NOT EXACT: <missing items>` in the log and lineage. Needed to resume any
  pre-v5 checkpoint. The current SE seed-2 runs will not be resumed.
- **D-8 — DECIDED (2026-09-30):** train-vs-UCI saves use the self-play session
  folder format and writer (C1 #9). **The replay buffer is NOT saved by default**
  for self-play or train-vs-UCI:
  - the autosave menu/settings gets a persisted "Include replay buffer" toggle
    (default off) covering periodic and post-promotion saves;
  - the manual Save Session dialog gets an "Include replay buffer" checkbox
    (default off);
  - train-vs-UCI gets a CLI flag to include it;
  - specified in D8: parameter `session_save_include_replay_buffer` (autosaves
    and train-vs-UCI), the manual-save sheet checkbox, CLI `--save-replay-buffer`;
  - a save without a buffer resumes by refilling from new games (training waits
    for the minimum fill) and is labeled `NOT EXACT: buffer`;
  - corpus replay never writes a buffer (rebuilt exactly from the corpus);
  - **Retention — ADDITION, DECIDED 2026-10-01** (folds in `AUTOSAVE_RETENTION_PLAN.md`;
    specified in §D8 "Autosave retention", phase P14): one combined retention pool for
    `-periodic` and arena `-promote` saves, capped by `max_periodic_autosaves_kept`
    (scope widened; `0` still means unlimited); `-manual` and **SIGUSR2 saves are
    exempt** (owner decision 2026-10-01); the separate "Save Session (Weights Only)"
    menu item and the write-then-strip approach are dropped as redundant under D-8.
    **RESOLVED 2026-10-01 (owner): "Promote Trainee Now" saves are treated like
    automatic promotions** — pool members under the shared `promote` tag, no new disk tag.
    *(Was OPEN: "Promote Trainee Now" saves were recorded as never pruned, like manual
    saves, unless the owner said otherwise — which would have required giving them their
    own disk tag.)* The pool is global across sessions (owner, 2026-10-01). Implemented in
    P14 (2026-10-01). **Gated and forced off (owner, 2026-10-01):** pruning also requires
    the new `automatic_save_pruning_enabled` setting (default off), and the build's kill
    switch `CheckpointPaths.automaticSavePruningForcedOff` holds it off regardless until D-8
    lands and there is more confidence in automatic deletion.
- **D-9 — DECIDED (2026-09-30):** recompute position hashes from the stored
  boards when loading a legacy buffer (owner-approved migration), logging the
  count recomputed.
- **D-10 — DECIDED (2026-09-30):** remove config D (issue #9) before P9. Owner
  approved deleting its two test cases (the config-D case in
  `HeadNumericsTailTests` and the config-D sweep in `MacOS27NaNIsolationTests`).
  C1 #34 then closes with the removal.
