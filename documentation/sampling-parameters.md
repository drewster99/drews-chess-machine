# Self-Play and Arena Sampling Parameters

Notes on how the engine samples moves during self-play (data generation) vs arena evaluation (candidate-vs-champion tournaments), the evolution of the temperature schedule, and how game diversity is tracked.

## Sampling method

After the network outputs 4,096 raw logits (one per from-square x to-square), moves are selected by:

1. **Filter** to legal moves only (gather their logits; illegal moves are excluded from the softmax entirely, equivalent to masking them to -infinity).
2. **Temperature-scale** the gathered logits: `scaled[i] = logit[i] / tau`.
3. **Softmax** (numerically stable: max-subtract, exp, normalize).
4. **Categorical sample** from the resulting distribution (single uniform random draw, walk the cumulative distribution).

No top-k, no top-p, no beam search, no MCTS. Temperature is the sole knob controlling exploration vs. exploitation.

## Temperature schedule: linear decay

Temperature follows a linear-decay schedule:

```
tau(ply) = max(floorTau, startTau - decayPerPly * ply)
```

where `ply` is the number of moves this player has made in the current game (0-indexed, per-player). The schedule is defined by three parameters:

| Parameter | Description |
|---|---|
| `startTau` | Temperature on the player's first move (ply 0) |
| `decayPerPly` | Temperature reduction per ply |
| `floorTau` | Minimum temperature (decay floor) |

The linear ramp replaces an earlier two-phase model (abrupt jump from opening temperature to main temperature at a fixed ply). The gradual decay is smoother — there's no single ply where behavior changes discontinuously — and gives each game a natural arc from exploratory opening to decisive endgame.

### Current presets

| Preset | `startTau` | `decayPerPly` | `floorTau` | Floor reached at ply | Used by |
|---|---|---|---|---|---|
| `.selfPlay` | 1.0 | 0.03 | 0.4 | 20 | Play-and-Train self-play workers |
| `.arena` | 0.7 | 0.04 | 0.2 | 13 | Arena tournament (candidate vs champion) |
| `.uniform` | 1.0 | 0.0 | 1.0 | never | Play Game / Forward Pass (legacy) |

### Self-play schedule rationale

The self-play schedule balances two competing needs:

- **Early diversity**: `startTau=1.0` ensures the first several moves explore broadly, filling the replay buffer with varied positions.
- **Decisive play**: By ply 20 (per player, roughly move 40 of the game), `floorTau=0.4` concentrates the distribution enough that the preferred move gets significantly more probability mass than alternatives. This produces more decisive games — fewer 50-move-rule draws — which means more non-zero `z` values for policy gradient training.

The floor of 0.4 is higher than the prior two-phase main temperature of 0.25. This trades some endgame sharpness for broader exploration throughout the game, which should improve replay buffer diversity.

### Arena schedule rationale

The arena exists to measure which network is stronger. Noise in the sampling reduces the signal-to-noise ratio of the evaluation, requiring more games for statistical significance.

- **Moderate opening diversity**: Without some stochastic opening play, color-alternating tournaments would collapse into a handful of deterministic lines. `startTau=0.7` provides enough opening variation to keep the 200-game tournament from repeating itself while staying closer to each network's actual preferences than the self-play `startTau=1.0`.
- **Faster decay** (`decayPerPly=0.04` vs 0.03): The arena tightens faster because evaluation accuracy matters more than exploration.
- **Lower floor** (`floorTau=0.2`): By ply 13 both networks are playing near their actual preferences, which is what we want for scoring.

## The one line of math

Temperature scaling applied to legal-move logits before softmax:

```
scaled[i] = logits[i] / tau
p[i]      = softmax(scaled)
```

- `tau = 1.0` -> unchanged logits -> current behavior.
- `tau < 1.0` -> logits scaled up -> distribution concentrates on the largest logit (approaches argmax as tau -> 0).
- `tau > 1.0` -> logits scaled down -> distribution flattens (approaches uniform as tau -> infinity).

This is applied inside the same numerically-stable max-subtract-then-exp path the sampler already uses; no separate softmax code path.

## Game diversity tracking

A `GameDiversityTracker` monitors whether the temperature schedule is producing enough game variety. Separate trackers run for self-play (rolling window of 200 games across all workers) and arena (one per tournament, window sized to the tournament game count).

### What it measures

For each completed game, the tracker:

1. **Hashes the full move sequence** (FNV-1a over policy indices) and checks for exact duplicates in the rolling window.
2. **Computes the divergence ply** — the longest prefix the new game shares with any game already in the window. This is the point at which the game "became unique."

### Displayed metrics

- **Unique game percentage**: `uniqueGames / gamesInWindow`. If this drops significantly below 100%, the temperature is too low (games are repeating).
- **Average divergence ply**: The mean ply at which games first differ from the most similar stored game. A low value means games branch early (high diversity). A high value means games follow the same lines deep into the middlegame.

Both metrics appear in the on-screen training stats panel and in periodic `[STATS]` and `[ARENA]` log lines.

### Implementation notes

- Move sequences are stored as `[Int16]` of policy indices (compact: 2 bytes per move).
- The FNV-1a hash provides good distribution for duplicate detection within a small window; cryptographic strength is unnecessary.
- For each new game, prefix comparison runs against all stored games: O(window_size * avg_game_length). With window=200 and ~80-ply games this is ~16,000 comparisons — negligible vs. neural network inference.
- Thread-safe via `NSLock` (self-play tracker is shared across concurrent workers).

## Model identity (ModelID): mint and inherit rules

Every network carries a `ModelID` (`yyyymmdd-N-XXXX`, with an optional
`-<generation>` suffix on a mutable trainer — e.g. `20260511-3-Ab9q-2`).
A *fresh* ID is minted (`ModelIDMinter.mint()`) only at a handful of
well-defined events; everywhere else an ID is **inherited verbatim** so
a snapshot can always be traced back to the weights it came from. Each
`[STATS]` and `[ARENA]` log line prints `trainer=…`, `champion=…`, and
(during an arena) `candidate=…` for exactly this reason.

- **Build Network** → the champion (`network`) gets a freshly minted ID.
- **Play-and-Train start (fresh)** → the trainer gets the *next trainer
  generation* of the champion's lineage (`mintTrainerGeneration(from: champion.id)`),
  e.g. champion `…-Ab9q` → trainer `…-Ab9q-1`.
- **Play-and-Train start (resume from a `.dcmsession`)** → champion and
  trainer keep the IDs stored in the loaded session file.
- **Arena start** → the candidate inference network inherits the
  *trainer's current ID* and the arena-champion network inherits the
  *live champion's ID*, both verbatim — the arena plays the trainer's
  arena-start snapshot against the champion's arena-start snapshot.
- **Arena promotion** (score ≥ `arenaPromoteThreshold`) → the live
  champion inherits the candidate's ID (i.e. the trainer's ID *as of
  arena start*), and the live trainer rolls forward to a fresh next
  generation forked from the new champion (`…-Ab9q` → `…-Ab9q-1` →
  `…-Ab9q-2`…). So both lineages converge on the arena-validated
  weights, then the trainer immediately forks off again to mutate.
- **`Engine ▸ Promote Trainee Now`** (manual, no arena) follows the
  *same rule* as arena promotion, just keyed on the trainer's *current*
  weights rather than an arena-start snapshot: the champion inherits the
  trainer's current ID and the trainer forks a fresh next generation
  from it. The resulting arena-history record has `gamesPlayed == 0` and
  `promotionKind == .manual`.
- **Save Champion as Model / Save Session** → the on-disk `.dcmmodel` /
  `.dcmsession` records whatever IDs the live networks have; no mint.

When in doubt, the invariant is: *only* Build, fresh Play-and-Train
start, and "roll the trainer to a new generation" mint or derive new
IDs; weight copies (snapshots, promotions, loads) inherit.

## History

### v1: Flat tau=1.0 everywhere

Original behavior. Both self-play and arena used temperature 1.0 for every move. Arena was a coin flip because sampling noise drowned the policy signal.

### v2: Two-phase schedule (abrupt transition)

Introduced `openingPliesPerPlayer` / `openingTau` / `mainTau`. Arena moved to tau=0.1 after 15 plies, self-play to tau=0.25 after 25 plies. Unblocked arena promotion. The abrupt temperature jump was functional but inelegant.

### v3: Linear decay (current)

Replaced the two-phase model with `startTau` / `decayPerPly` / `floorTau`. Temperature ramps down smoothly rather than jumping. Self-play floor raised to 0.4 for broader exploration; arena floor raised to 0.2 with faster decay. Added game diversity tracking.
