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
  [--start-model <path> | --preset <name>] \
  --out-model <path> \
  [--enumerate-checkpoints] [--parameters <path>] \
  [--training-step-limit N] [--training-time-limit <seconds>] \
  [--max-plies 400] [--eval-sync-steps 10]
```

### Per-pool vs. global (both)

| scope | where | fields |
|---|---|---|
| **per opponent pool** | inside each `--train-vs-uci "…"` (`;`-delimited) | `cmd=` engine path · `n=` instance count · `go=` per-move limit · any other `KEY=VALUE` → `setoption name KEY value VALUE` (`UCI_Elo`, `Skill Level`, `Threads`, `Hash`, …) |
| **global (whole run)** | top-level flags | `--start-model` / `--preset`, `--out-model`, `--parameters`, `--training-step-limit`, `--training-time-limit`, `--max-plies` (400), `--eval-sync-steps` (10), `--enumerate-checkpoints` |
| **hardcoded global** | `UCIArbiter.Configuration` (no flag) | `handshakeTimeout` 10 s, `moveTimeout` 30 s |

`--train-vs-uci` is **repeatable** — each is an independent pool with its own
engine, count, `go`, and options. Two pools may point at the *same* binary with
different settings (e.g. one Stockfish pool at `go=movetime 10`, a second at
`UCI_Elo=1400`). All `n` instances in a pool are identical.

**Output files.** `--training-time-limit` takes **seconds** (a positive number),
e.g. `21600` for 6 h — not `6h`. The rolling trainer model is your `--out-model`
verbatim (`.safetensors` appended if missing); if `--out-model` is omitted it is
`<start-model-stem>-vsuci-latest.safetensors` (or `<runModelID>-vsuci-latest…`
with `--preset`). **Both `--start-model` and `--preset` are optional** — with
neither, training starts from a fresh net at the current default architecture
(`NetworkArchitecture.current`). With `--enumerate-checkpoints`, each step-N snapshot is derived
from the rolling file's stem: **if the stem contains `-vsuci-latest`**, that
substring is replaced with `-vsuci-step<N>`; **otherwise** it is
`<stem>-step<N>.safetensors`. (So `--out-model foo.safetensors` enumerates as
`foo-step<N>.safetensors`, while `foo-vsuci-latest.safetensors` enumerates as
`foo-vsuci-step<N>.safetensors`.)

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

- `[VS-UCI]` — lifecycle: start-model, `[VS-UCI-ARCH]`, opponent pool, per-step
  `step=… loss=… pLoss=… vLoss=…`, autosave/enumerate lines.
- `[VS-UCI-STATS]` — per-instance + aggregate: `games=`, `plies=`, `g/s=`, `p/s=`,
  `W-L-D=` (trainer-side scoreline only). Quote throughput as **plies/hour**.
- `[BATCH-STATS]` — sampled-batch (buffer) composition: `game_length` (ply bins
  **short ≤50 / medium ≤150 / long ≤300 / very_long**), `phase_by_ply`
  (opening/early/mid/late/end), `bucket_mix` (**material** — non-pawn piece count,
  *not* plies), `outcome` (W/L/D balance of the batch), `buffer_stored` /
  `buffer_unique`.

Note: a `--train-vs-uci` process does **not** match `grep replay-corpus`, so the
run-agnostic corpus-replay monitoring cron does not track it — watch/register it
separately.
