# cutechess-cli setup for DrewsChessMachine

How to play DCM against another UCI engine (usually Stockfish) with
`cutechess-cli`, to **measure** DCM's strength. This is the benchmark path —
unlike `--train-vs-uci`, it does not touch the weights. Prereq: DCM's `--uci`
engine mode — see [UCI.md](UCI.md) Part 1, especially the two particulars that
shape everything below:

- **DCM ignores all time controls** (single forward pass, replies in ms). So you
  set the *opponent's* budget; DCM's TC is irrelevant.
- **DCM is deterministic at Temperature 0** (no search, no opening book). So you
  **must** supply an opening book / varied start positions, or every game is
  identical.

## Install cutechess-cli

Not currently installed on this machine. Either:

- Build from source (reliable on Apple Silicon):
  ```
  git clone https://github.com/cutechess/cutechess
  cd cutechess && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  # binary: build/cutechess-cli
  ```
- Or grab a release / package if one is available for macOS, then confirm
  `cutechess-cli --version`.

## Engine paths

- **DCM** (Release build):
  ```
  ~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-<hash>/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
  ```
  Run as a UCI engine with the `--uci` argument.
- **Stockfish**: `~/bin/stockfish`.

## Registering the engines (inline)

DCM — pass `arg=--uci`, protocol `uci`, and pick the network + temperature via
`option.…`:

```
-engine name=DCM \
  cmd="$DCM_BIN" arg=--uci proto=uci \
  option.Model="$HOME/Library/Application Support/DrewsChessMachine/Models/<checkpoint>.safetensors" \
  option.Temperature=0
```

- `option.Model=` — which checkpoint to play (path, or a bare name/ID DCM can
  resolve). Omit to let DCM lazily load the latest session on first `go`.
- `option.Temperature=0` — strongest, deterministic play (default). Raise it to
  weaken / add variety.

Stockfish — set its **strength and budget** here (this is the dial that matters,
since DCM is fixed):

```
-engine name=SF1400 \
  cmd="$HOME/bin/stockfish" proto=uci \
  option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  option.Threads=1
```

Use `UCI_LimitStrength`+`UCI_Elo` (min 1320) for a *calibrated* opponent, and/or
cap its search with a per-engine `tc=`/`st=`/`nodes=` (below). `Threads=1` keeps
it deterministic and one core per game.

## Time control — set it on the opponent, not DCM

cutechess requires a TC, but DCM ignores it. Give DCM a nominal one and give the
opponent the budget that actually sets difficulty. Simplest: a fixed per-move
budget with `st=` (seconds/move) or a nodes limit via the engine's own option.

```
-each proto=uci                     # shared defaults
-engine name=DCM cmd="$DCM_BIN" arg=--uci option.Temperature=0 st=1
-engine name=SF1400 cmd="$HOME/bin/stockfish" option.UCI_Elo=1400 option.UCI_LimitStrength=true st=1
```

Here `st=1` (1 s/move) applies to both; DCM ignores it (instant), Stockfish
honours it. To make SF fast *and* weak, combine a small `st=` with `UCI_Elo`.

## Openings are mandatory (DCM is deterministic)

Without varied starts every DCM game at Temperature 0 is byte-identical. Supply a
book and randomise:

```
-openings file=<book.pgn or book.epd> format=<pgn|epd> order=random plies=8
```

(If you don't have one handy, any small PGN/EPD opening suite works; the point is
distinct start positions.) Alternatively raise `option.Temperature` on DCM to get
variety from sampling instead — but a book is the cleaner control.

## A full match

DCM vs a 1400-Elo Stockfish, 200 games, alternating colours, PGN out:

```
DCM_BIN=~/Library/Developer/Xcode/DerivedData/DrewsChessMachine-<hash>/Build/Products/Release/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine

cutechess-cli \
  -engine name=DCM cmd="$DCM_BIN" arg=--uci proto=uci \
     option.Model="$HOME/Library/Application Support/DrewsChessMachine/Models/<checkpoint>.safetensors" \
     option.Temperature=0 \
  -engine name=SF1400 cmd="$HOME/bin/stockfish" proto=uci \
     option.UCI_LimitStrength=true option.UCI_Elo=1400 option.Threads=1 \
  -each st=1 \
  -openings file=<book.pgn> format=pgn order=random plies=8 \
  -games 200 -repeat -concurrency 6 \
  -pgnout dcm_vs_sf1400.pgn \
  -ratinginterval 10
```

- `-repeat` plays each opening twice with colours reversed (fairer).
- `-concurrency 6` runs 6 games in parallel (each Stockfish is `Threads=1`).
- `-ratinginterval 10` prints a running score/Elo estimate.

## Getting an Elo number

DCM at Temperature 0 is weak (raw policy, no search). Full-strength Stockfish is
useless (100 % losses → no signal). Two good approaches:

1. **Elo ladder** — run DCM against several `UCI_Elo` rungs (e.g. 1320, 1500,
   1800) and find where it scores ~50 %. That brackets DCM's rating directly.
2. **Rate the PGN** — feed `-pgnout` results to `ordo` or `bayeselo` for a
   maximum-likelihood Elo, especially when running a pool of opponents.

## A/B comparing two DCM checkpoints

Because DCM is deterministic at Temperature 0, a fixed opening set makes the match
reproducible — ideal for comparing two checkpoints against the *same* opponent
under the *same* openings. Register both DCM builds as separate `-engine` blocks
(different `option.Model=`) and/or use a round-robin (`-tournament round-robin`)
including the opponent.

See [UCI.md](UCI.md) for the engine-mode details (options, the ignore-limits and
determinism particulars) and for the reverse direction (`--train-vs-uci`, where
DCM drives engines and trains on the games).
