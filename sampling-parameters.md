# Self-Play and Arena Sampling Parameters

Notes on how the engine samples moves during self-play (data generation) vs arena evaluation (candidate-vs-champion tournaments), why the original flat-tau-1.0 sampling stalled arena promotion, and the temperature schedule we're moving to.

## Current state (before this change)

After ~50,000 SGD steps, training is mechanically healthy:

- **Value loss** descends smoothly from ~1.3 → ~0.11 at a constant 1e-4 learning rate. That's a classic, clean supervised-regression curve — the value head is learning, and the rate is not too high.
- **Policy loss** also descends (outcome-weighted CE, `mean(z · -log P[played_move])`).
- Training throughput is good: ~13 steps/sec, ~3,300 moves/sec in the combined self-play + train pipeline.

But the **arena is pinned at a coin flip**. The last 30 consecutive tournaments all landed between 0.482 and 0.540, with **every single one marked "kept"** (i.e. candidate failed to beat the 0.55 promotion threshold). The champion has never been updated past its initialization weights. A typical arena result looks like `17-14-169` — 200 games, 169 of them drawn, 17 candidate wins, 14 champion wins. The decisive-game sample size is ~30, and those 30 games are split almost evenly.

Self-play game outcomes tell the same story:

| Outcome | Share |
|---|---|
| Insufficient material | 53.3% |
| 50-move rule | 22.6% |
| Stalemate | 6.2% |
| Threefold repetition | 1.5% |
| **Draws (subtotal)** | **83.6%** |
| Checkmates (white + black) | 14.6% |

Two nearly-random networks shuffle pieces until the rules force a draw.

## Why flat tau=1.0 sampling stalls the arena

`MPSChessPlayer.sampleMove` takes the logits for the ~30 legal moves, runs a plain softmax, and draws one sample from the resulting distribution. That's temperature = 1.0 with no schedule. The exact same sampler runs in both self-play and the arena.

Temperature-1 sampling is the right thing to do during self-play — it's the exploration mechanism that fills the replay buffer with diverse positions. Without sampling, every self-play game from the same opening would be identical, and training would collapse into one thin groove of the game tree.

But in the arena the same sampler is a disaster for two reasons:

### Reason 1: sampling noise drowns the policy signal

Imagine the candidate has actually learned that move *A* is better than the other 29 legal moves, and its logits are:

```
move A:  logit 2.0 → exp = 7.39  → P(A) ≈ 8.6%
moves B–Z (29 each): logit 1.0 → exp = 2.72 → P(each) ≈ 3.1%
```

Even with a clearly-preferred move, the candidate plays its "best" move only **8.6% of the time**. The other 91.4% of the time it plays one of the 29 moves it thinks are worse. Under argmax the same network plays A 100% of the time.

Now compound that over a 40–80 ply game: every ply is an independent dice roll, and the per-move edge has to survive dozens of rolls to produce a win. Final game outcomes are dominated by sampling noise, not by the networks' actual preferences.

### Reason 2: the draw rate is so high there's no signal to measure anyway

With 169 of 200 arena games drawn, there are only ~30 decisive games per tournament. Detecting "0.55 vs 0.50" from 30 noisy Bernoulli trials is statistically impossible — the standard error on a 200-game score with this draw rate sits around ±3.5%, right where the signal is.

So: even if the candidate has improved meaningfully over the champion, the sampling-noise-plus-high-draw-rate combination hides that edge completely. The arena score sits at 0.50 ± noise, no matter how much the network has learned.

## What we're *not* changing (yet)

A bunch of other things are suspect — they are tracked separately and deliberately **out of scope** for this change so we can see the effect of the sampling fix in isolation:

- The policy loss is outcome-weighted (`mean(z · -log P[a*])`). For draws (83% of positions) `z = 0` → zero gradient on policy. For losses (`z = -1`) the loss is unbounded below — it can be minimized to −∞ by crushing `P[losing_move]` → 0. This is a real concern; we're leaving it alone until after we see whether fixing sampling alone unblocks arena promotion.
- No MCTS. Training targets are the moves the network actually played, not visit-count distributions from a tree search. This is the biggest deviation from AlphaZero, and likely contributes to the high draw rate. Scope-limited for now.
- Learn-rate schedule. A cosine or step-down decay is standard polish once loss plateaus, but value loss is still descending smoothly at flat 1e-4 — there's nothing to decay yet.

## The new sampling schedules

Both self-play and arena get a two-phase temperature schedule: a stochastic *opening* phase for game diversity, and a *main* phase that tightens up in arena (near-argmax to surface actual playing strength) and narrows-but-stays-stochastic in self-play (keep exploration alive while producing more decisive games).

### Self-play schedule

```
Phase     | Plies (this player) | Temperature | Effect
----------|---------------------|-------------|--------------------------------
Opening   | 0 to 7 (inclusive)  | tau = 1.0   | Raw softmax, maximum diversity
Main      | 8 onwards           | tau = 0.5   | Logits doubled before softmax
```

`openingPliesPerPlayer = 8` means each player makes its first 8 moves under tau=1.0, which is roughly the first 16 half-moves of the game — a solid opening window.

The tau=0.5 main phase keeps sampling (so exploration doesn't die and the replay buffer stays diverse), but the distribution is sharper: the preferred move might jump from ~8% to ~40%, and the tail of ~30 "why not" moves gets cut back hard. The result should be **more decisive games** — fewer 50-move-rule exits, more positions with non-zero `z`, which directly feeds the policy-loss gradient.

### Arena schedule

```
Phase     | Plies (this player) | Temperature | Effect
----------|---------------------|-------------|--------------------------------
Opening   | 0 to 3 (inclusive)  | tau = 1.0   | Raw softmax, 4+ opening plies
Main      | 4 onwards           | tau = 0.1   | Near-argmax, tiny jitter
```

**Why we can't just use pure argmax (tau = 0) in the arena.** `TournamentDriver` alternates colors on every game. If both sides played fully deterministic argmax, every "candidate as white, champion as black" game would be literally identical — same move sequence, same result. The 200-game arena would collapse into **two unique games played 100 times each**, which is the worst possible statistical setup: zero variance reduction from the large sample size. You need *some* stochasticity, otherwise color alternation just repeats the same two games.

**Why not just turn temperature way down (tau=0.1) from move 1?** With ~30 legal moves in the opening and tau=0.1, the preferred move already absorbs 90%+ of the probability mass, so the game is effectively deterministic from ply 0. Same collapse problem.

**Why a short 4-ply (per player) opening window is enough.** Four plies of tau=1.0 sampling per player gives ~8 stochastic half-moves at the start of the game. With the branching factor softmax gives us, that's more than enough to produce unique game trees for every game in a 200-game tournament — even if the same position were reached repeatedly, the opening sampling would diverge the games before deterministic play takes over.

After the opening, tau=0.1 means both networks play their current preferred move almost every time. That's exactly what we want for *evaluation*: measure how good the networks' preferences actually are, not how lucky their sampling was. Any improvement the candidate has over the champion should finally be visible in the score.

Both players use the same schedule, so fairness is preserved automatically.

## The one line of math

Temperature scaling is applied to the legal-move logits before softmax:

```
scaled[i] = logits[i] / tau
p[i]      = softmax(scaled)
```

- `tau = 1.0` → unchanged logits → current behavior.
- `tau < 1.0` → logits scaled up → distribution concentrates on the largest logit (approaches argmax as tau → 0).
- `tau > 1.0` → logits scaled down → distribution flattens (approaches uniform as tau → ∞).

This is applied inside the same numerically-stable max-subtract-then-exp path the sampler already uses; no separate softmax code path.

## Diagnostic plan

The whole point of starting with this change is that the two outcomes are **both informative**:

1. **Arena scores jump above 0.55 and promotion starts firing.**
   Confirms the candidate has already been outperforming the champion for thousands of steps, and sampling noise was masking it. Training-side code is fine as-is, and we can then focus on the draw rate / policy loss in future work.

2. **Arena scores stay pinned near 0.5.**
   Confirms the candidate isn't meaningfully better than the init-weight champion. We then know to look at the training signal itself — the outcome-weighted policy loss, the lack of MCTS, the 83% draw rate, etc. — rather than evaluation plumbing.

Either result points us directly at the next thing to fix, which is why this change was worth doing first.

## Reference: temperature presets at a glance

| Preset | `openingPliesPerPlayer` | `openingTau` | `mainTau` | Used by |
|---|---|---|---|---|
| `.selfPlay` | 8 | 1.0 | 0.5 | Play-and-Train self-play workers |
| `.arena` | 4 | 1.0 | 0.1 | Arena tournament (candidate vs champion) |
| `.uniform` | 0 | 1.0 | 1.0 | Play Game / Forward Pass (unchanged legacy behavior) |
