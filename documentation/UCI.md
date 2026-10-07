# UCI: DrewsChessMachine and the UCI protocol

DCM touches UCI in two opposite directions:

1. **DCM *as* a UCI engine** (`--uci`) — DCM presents itself as a UCI engine so an
   external GUI or arbiter (cutechess-cli, a chess GUI, another driver) can play
   it. This is the usual way to *measure* DCM's strength. Covered first, below.
2. **DCM *driving* external UCI engines** (`--train-vs-uci`) — DCM plays its live
   trainer net against external engines (Stockfish, …) **and trains on the
   games**. A *training* regime, not a benchmark. Covered in the second half.

For a concrete opponent-vs-DCM match harness, see
[cutechess-setup.md](cutechess-setup.md).

---

# Part 1 — DCM as a UCI engine (`--uci`)

Source: `App/UCI/UCIEngine.swift` (protocol loop), `App/UCI/UCIModelLoader.swift`
(model resolution), dispatched from `DrewsChessMachineApp.handleUciIfPresent`.

## Launch

```
DrewsChessMachine --uci [--model <path-or-name>]
```

Only `--uci`, an optional `--model`, and the `--crosscheck-movegen` diagnostic
are accepted; any other `--`flag is a hard error (so a typo can't silently launch
the GUI). The binary is the Release build:

```
~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-<hash>/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
```

**Model loading is deferred** if `--model` is omitted: DCM loads weights when a
`setoption name Model` arrives, or lazily on the **first `go`** (which loads the
latest session's weights). So a GUI can launch it with zero args and it's still
playable.

## Handshake / identity

- `id name DrewsChessMachine <build> (<git>[·dirty]) model=<label>`
- `id author Andrew Benson`
- then the options (below), then `uciok`.
- On `isready` → `readyok`. Emits `info string` lines with the engine build and,
  once a model is loaded, its label / param count / arch summary.

## Options

| option | type | default | meaning |
|---|---|---|---|
| `Model` | string | current model path (**blank** when launched without `--model`, since loading is deferred) | Which network to play. Set via `setoption name Model value <path-or-name>`. Resolved by `UCIModelLoader`: a filesystem path, **or** a bare name/ID (exact filename → filename+extension → run-name → its `-latest`). Re-setting the same file is a no-op (mtime-idempotent). |
| `Temperature` | spin | `0` | Sampling temperature × 100 (UCI spin has no floats). `tau = value / 100`. **`0` or `1` → tau 0.01 ≈ argmax** (strongest, deterministic). `100` → tau 1.0. `1000` → tau 10.0 (near-flat/random). Range 0–1000; floored at tau 0.01. |

There is **no `UCI_Elo` / `UCI_LimitStrength`** on DCM's side — its strength knobs
are the **model** (which checkpoint) and, secondarily, **Temperature** (0 =
strongest). To weaken DCM, raise Temperature or load an earlier checkpoint.

## Supported commands

`uci`, `isready`, `ucinewgame`, `position`, `go`, `setoption`, `quit`, and the
no-ops `stop` / `ponderhit`.

- `position startpos [moves …]` and `position fen <6 fields> [moves …]`. The full
  move history is threaded so history/repetition input planes are populated (the
  net plays from the same representation it trained on).
  - **The GUI decides draws.** DCM applies the whole move list without its own
    draw rules, so a game continued past an unclaimed threefold repetition or the
    fifty-move mark (as Lichess and claim-based bridges allow) is followed exactly.
    The list stops only where no move exists (after checkmate or stalemate).
  - **A list that cannot be applied is rejected as a whole** (unknown token, bad
    FEN, illegal move, a move after mate). DCM answers `info string position
    rejected: …` at once, and every `go` until the next valid `position` or
    `ucinewgame` replies `bestmove 0000` with an `info string` naming the
    rejection — never a move for a position the GUI is not in.

## `go` — single forward pass, **ignores all limits**

This is the load-bearing particular. DCM does **no search** — a move is one
network forward pass. `handleGo` encodes the current position (with real ply
history), runs `DirectMoveEvaluationSource.evaluate`, masks illegal moves,
temperature-softmaxes over the legal logits, categorical-samples, and emits
`bestmove` **immediately**.

Consequently DCM **ignores every `go` limit** — `movetime`, `depth`, `nodes`,
and the clock params (`wtime`/`btime`/`movestogo`) are all disregarded. It
replies in a few ms to *any* `go`. `stop` is a no-op (nothing is searching);
there is no pondering; `bestmove 0000` is returned when there are no legal moves.

Two consequences for match setup:

- **Any time control works** — DCM answers instantly regardless. In a match you
  set the *opponent's* budget (its `go` limit / TC); DCM's think-time is fixed.
- **DCM is deterministic at Temperature 0.** With no opening book and no search,
  the same position always yields the same move (subject to sampling only when
  Temperature > 0). So a match harness **must** supply varied start positions (an
  opening book) or every game is identical. See cutechess-setup.md.

---

# Part 2 — DCM driving external engines (`--train-vs-uci`)

A headless mode that plays the live trainer network against one or more external
UCI engines **and trains on the resulting games** (SGD updates the net). Not a
benchmark — for pure strength measurement use cutechess-cli (Part 1 +
cutechess-setup.md). Plumbing: `UCIArbiter` (one actor per engine instance),
`TrainVsUciDriver` (game production + training), `TrainVsUciRunner`.

## What it does

`N` concurrent games per pool. Each ply alternates the **trainer net's** move
(batched GPU eval, sampled **argmax / tau 0.01**) and the **engine's** move (a
`go <limit>` round-trip). Finished games flush into a `ReplayBuffer`; a
**step-locked** SGD loop trains flat-out on minibatches, independent of
production rate.

**Whole-game, both-sides recording (distillation).** Every ply — the trainer's
*and* the engine's — is recorded, each with the mover's move as the policy target
and the terminal outcome signed by the mover's colour (`ActiveGame.flush`). So a
100 %-loss run still fills the buffer ~50 % with **win-labelled Stockfish
positions whose policy target is Stockfish's move** → the net **distills the
engine's strong play**. The `W-L-D=…` in `[VS-UCI-STATS]` is only the
trainer-side match scoreline, *not* the buffer's win/loss balance (~50/50 on
decisive games).

## CLI syntax

```
DrewsChessMachine \
  --train-vs-uci "cmd=<path>;n=<count>;go=<limit>;<Option>=<value>;..." \
  [--train-vs-uci "<second opponent pool>"] \
  [--start-model <model file | .dcmsession folder> [--resume-exact] | --preset <name>] \
  [--out-session-dir <folder>] [--save-replay-buffer] \
  [--enumerate-checkpoints [--checkpoint-stem <path stem>]] [--parameters <path>] \
  [--training-step-limit N] [--training-time-limit <seconds>] \
  [--max-plies 400] [--eval-sync-steps 10]
```

### Per-pool vs. global (both)

| scope | where | fields |
|---|---|---|
| **per opponent pool** | inside each `--train-vs-uci "…"` (`;`-delimited) | `cmd=` engine path · `n=` instance count · `go=` per-move limit · any other `KEY=VALUE` → `setoption name KEY value VALUE` (`UCI_Elo`, `Skill Level`, `Threads`, `Hash`, …) |
| **global (whole run)** | top-level flags | `--start-model` (+ `--resume-exact`) / `--preset`, `--out-session-dir`, `--save-replay-buffer`, `--parameters`, `--training-step-limit`, `--training-time-limit`, `--max-plies` (400; at least 1), `--eval-sync-steps` (10), `--enumerate-checkpoints` (+ `--checkpoint-stem`) |
| **hardcoded global** | `UCIArbiter.Configuration` (no flag) | `handshakeTimeout` 10 s, `moveTimeout` 30 s |

`--start-model` alone starts a **new branch**: the file's weights, with a fresh
trainer clock (warmup and the LR/momentum cycle start from their beginnings) and
zero optimizer velocity. Add `--resume-exact` to continue that checkpoint's
training exactly — fp32 master weights, optimizer velocity, the completed-step
clock, and the warmup length and cycle it trained under are all restored, so
warmup does not re-run. It needs a checkpoint written with exact trainer state
(`trainer_*` keys in `__metadata__` plus `opt.*.velocity` tensors); older
checkpoints are refused. `--start-model` takes a model file or a `.dcmsession`
folder (the session's `trainer.safetensors` is the start). An exact resume from a
session saved with `--save-replay-buffer` also restores its replay buffer; from a
model file, or a session saved without the buffer, the buffer refills from new
games before training resumes and the resume is `NOT EXACT: buffer` (pass
`--accept-inexact buffer`).

`--train-vs-uci` is **repeatable** — each is an independent pool with its own
engine, count, `go`, and options. Two pools may point at the *same* binary with
different settings (e.g. one Stockfish pool at `go=movetime 10`, a second at
`UCI_Elo=1400`). All `n` instances in a pool are identical.

**Output: session folders.** `--training-time-limit` takes **seconds** (a positive
number), e.g. `21600` for 6 h — not `6h`. **Both `--start-model` and `--preset`
are optional** — with neither, training starts from a fresh net at the current
default architecture (`NetworkArchitecture.current`).

A run saves `.dcmsession` folders, written by the same
`CheckpointManager.saveSession` the GUI uses: exclusive staging, bit-exact model
verification, a forward-pass round trip, a `session.json` round trip, a
replay-buffer round trip, `F_FULLFSYNC`, and a publish that never replaces an
existing folder. Every save is a new folder,
`<YYYYMMDD-HHMMSS>-<runModelID>-vsuci-<periodic|final|abort|health-stop>.dcmsession`, in
`--out-session-dir` (default: the app's `Sessions/` folder):

- `vsuci-periodic` — on the GUI's cadence, the `periodic_autosave_interval_sec`
  parameter (default 6 h; set it in a `--parameters` file for a shorter crash
  window); the clock starts at run start and restarts at each successful save.
- `vsuci-final` — the step or time limit was reached.
- `vsuci-abort` — Ctrl-C.
- `vsuci-health-stop` — a training-health alarm whose
  `training_health_action_<rule>` stops the run requested a stop; the process
  exits with status 35 after this save (see
  `documentation/training-health-alarms.md`).

Each folder holds `trainer.safetensors` (the complete trainer state — what the
rolling `--out-model` file held before session folders replaced it),
`champion.safetensors` (the play network, synced from the trainer at the save),
`session.json` (lineage `path_kind` `vsuci`) and, only with
`--save-replay-buffer`, `replay_buffer.bin` (several GB per save). The launch
line `[VS-UCI] session saves: …` states the folder, the cadence and whether the
buffer is included, so a run's crash exposure and disk cost are visible up front;
every save logs `[CHECKPOINT] Saved session (vsuci-…)`. A failed save is logged
and retried at the next step; two failures in a row stop the run, a full disk
stops it at once, and a failed final (or abort) save fails the run — nothing else
holds its end state.

Train-vs-UCI sessions are not GUI sessions: the GUI refuses to load one (naming
the resume command), they never count toward the GUI's automatic-save retention
pool (whose members are only `-periodic` / `-promote` folders), and the CLI never
moves the GUI's "Resume Training" pointer. Resume a run with
`--train-vs-uci … --start-model <folder> --resume-exact`.

`--out-model` and `--overwrite-out-model` belong to corpus replay; with
`--train-vs-uci` they are refused (the rolling model file was replaced by session
folders). A rolling `…-vsuci-latest.safetensors` written by an earlier build is
still an ordinary model file for `--start-model`.

**Step checkpoints.** With `--enumerate-checkpoints`, the trainer file is also
written at every trainer-step multiple of 1000 (and at the end, when the last
trainer step is not one, and the segment trained at least one step) as
`<stem>-vsuci-step<T>.safetensors`, `T` the trainer step — the same value as the
file's `training_step` (architecture format v11). The stem is `--checkpoint-stem`
(a path, which may contain dots but must not end in a model or session extension —
`.safetensors`, `.dcmmodel`, `.dcmsession` — nor itself be named like a step file);
otherwise the `--start-model` file's own stem, next to it (the names earlier runs
produced); otherwise — a fresh run or a session start — the run's model ID in
`Models/`. Step files are never overwritten. Names carry the trainer step and a
segment writes only above the trainer step it starts from, so an exact resume may
keep its run's `--checkpoint-stem` and continues the series (a resume from
`…-step1013` writes `…-step2000`, …). A run whose stem already holds step files it
could reach (above its start, up to its start plus `--training-step-limit`) refuses
to start — for example a second resume of an earlier file into a stem a later
resume already wrote into; give it its own `--checkpoint-stem`. Resumed segments
written before format v11 named their files `<stem>-vsuci-seg<k>-step<N>` (N their
own step); those names stay as they are.

## Timing model — fixed per-move only, no clock

The driver sends **`go <goLimit>` verbatim every move** (a fixed per-move
budget). There is **no game clock** — the arbiter never sends
`wtime`/`btime`/`winc`/`binc`/`movestogo`.

| `go=` | driver sends | meaning |
|---|---|---|
| `movetime 10` | `go movetime 10` | 10 ms/move wall-clock cap |
| `nodes 100000` | `go nodes 100000` | 100k nodes (hardware-independent) |
| `depth 6` | `go depth 6` | search to depth 6 |
| `depth 8 movetime 200` | `go depth 8 movetime 200` | depth 8 but ≤ 200 ms |
| *(omitted)* | `go depth 1` | **default** |

`depth 1` is **our** default, not a UCI convention: UCI has no default for `go`,
and a bare `go` is engine-dependent (most engines treat it as `go infinite`). We
default to `depth 1` so an omitted `go=` can't hang.

**Tournament time controls ("40 moves in 5 s") are not supported** — those need
the arbiter to keep each side's clock and send `go wtime … btime … movestogo …`
per move. `UCIArbiter` has no clock state. Fixed-per-move was chosen deliberately
for reproducibility.

## Known limitations / shortcomings

- **UCI-native engines only.** The arbiter speaks only UCI. xboard/CECP-only
  engines fail the handshake and their pool goes idle. Confirmed: **Sloppy 0.2.2
  is not UCI** (rejects `uci`; no `uciok` in the binary) → all instances time out
  at handshake. Verify UCI support before adding an engine.
- **No validation of the `go=` string.** `goCommand` just trims and appends after
  `go ` — no allow-list. A typo (`movetiem 10`) is sent verbatim; the engine
  ignores the unknown token and may search infinitely → every move stalls to the
  30 s `moveTimeout`.
- **Bare `go` is reachable.** `goCommand` emits a bare `go` for an empty limit
  (`trimmed.isEmpty ? "go" : "go \(trimmed)"`); the only guard is the `depth 1`
  *default*, not a hard block. Passing `go=` with an empty value overrides the
  default (`goLimit=""`) → bare `go` → infinite search → 30 s timeout/move.
- **No compliance checking.** `UCIArbiter.bestMove` waits for the first `bestmove`
  line, **discarding all `info` lines** and never wall-timing the move. It cannot
  tell whether the engine honoured the limit; the only backstop is the 30 s
  `moveTimeout`, which catches a total hang, not a soft over/under-shoot.
- **Two timeouts are hardcoded** (`handshakeTimeout` 10 s, `moveTimeout` 30 s),
  not flags. Keep any `movetime` well under 30 s.
- **Length-capped games are dropped.** A game hitting `--max-plies` (400) has an
  unknown outcome, so it is not flushed to the buffer (same as self-play).

## Performance particulars (measured, M5 Max, 18 cores)

- **Prefill (trainer idle):** production is bounded by **per-game round-trip
  latency** (engine think-time + UCI pipe I/O + per-slot tick orchestration), all
  wait, not compute. It **parallelises across games**, so more instances ≈ more
  fresh data — until the box saturates. Observed `n=10 → 100 → 200` ≈
  `~5M → ~25M → ~37M plies/hr` (clearly sub-linear past ~100: orchestration and
  the batched play-eval become the wall, with CPU still idle at 200).
- **Training (trainer running):** the step-locked trainer **saturates the single
  GPU** (~600–750 ms/step); the net's play-evals are cheap but **queue behind the
  training steps**, so production drops and engines idle (CPU falls). More
  instances still help (more evals queued to fill GPU spare time): training-phase
  production `n=10 ≈ 1.7M → n=100 ≈ 11M → n=200 ≈ 16M plies/hr`.
- **Buffer reuse** (consumption ÷ production) — the data-freshness metric — fell
  `~14× (n=10) → ~1.9× (n=100) → ~1.2× (n=200)`; ~1× is the floor. So ~100–200
  instances is the practical sweet spot on this box.

## Observability

- `[VS-UCI]` — lifecycle: start-model, `[VS-UCI-ARCH]`, opponent pool, the
  step-line cadence, step lines `step=… loss=… pLoss=… vLoss=… … trainerStep=…`,
  autosave/enumerate lines. Step lines follow the shared trainer-step cadence: the
  segment's first step, every 50 trainer steps through 1000, every trainer-step
  multiple of 1000, and the first diagnostics step `step_line_interval_sec`
  (default 180 s) after the previous line; every line after the first carries the
  diagnostics. `step=` is the segment's own step, `trainerStep=` the trainer step.
- `[VS-UCI-STATS]` — per-instance + aggregate: `games=`, `plies=`, `g/s=`, `p/s=`,
  `W-L-D=` (trainer-side scoreline only). Quote throughput as **plies/hour**.
- `[BATCH-STATS]` — written with each step line (session log only) —
  sampled-batch (buffer) composition: `game_length` (ply bins
  **short ≤50 / medium ≤150 / long ≤300 / very_long**), `phase_by_ply`
  (opening/early/mid/late/end), `bucket_mix` (**material** — non-pawn piece count,
  *not* plies), `outcome` (W/L/D balance of the batch), `buffer_stored` /
  `buffer_unique`.

Note: a `--train-vs-uci` process does **not** match `grep replay-corpus`, so the
run-agnostic corpus-replay monitoring cron does not track it — watch/register it
separately.
