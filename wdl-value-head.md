# WDL value head — design & exact math

*Status: design, pre-implementation. Supersedes the scalar-`tanh` value head documented in `chess-engine-design.md`. The scalar head's measured behaviour (it works, but caps at ≈arena-parity because it goes silent on a draw-heavy buffer) is recorded in the 2026-05-11 `(FINDING)` entry of `CHANGELOG.md`.*

This document explains, for a reader who hasn't lived inside the trainer, exactly what changes when we switch the value head from a single scalar to a **win/draw/loss (WDL) probability distribution**, and why. It's written to be self-contained — terms like *logit*, *softmax*, *cross-entropy*, *one-hot*, *advantage*, *baseline* are defined inline as they come up.

---

## 1. Background: what the value head is, and what it's for

For a chess position, the network produces two things in one forward pass:

- a **policy** — 4864 numbers, one per (move-type × square) cell, that (after masking illegal moves, temperature-scaling, softmax-ing) is the probability distribution the engine actually samples its move from; and
- a **value** — the network's estimate of *how this position will turn out*, from the perspective of the side to move.

There is **no search** in this engine (no MCTS, no alpha-beta). The move that gets played comes entirely from the policy. **The value head is not used to pick moves.** It's used in *training*, in two places:

1. **As a training target.** The value head is trained to predict the actual game result. After a self-play game ends, every position in it gets labelled with that game's outcome (from that position's mover's view): `z = +1` if the side to move went on to win, `z = −1` if it lost, `z = 0` if it drew. Same label for every position in the game; no discounting. The value head is trained to predict `z`.

2. **As a baseline for the policy gradient.** The policy is trained with a quantity called the **advantage**:

   `advantage = z − value_baseline`

   "Did reaching this position (and playing the move that was played) turn out *better* than the value head expected (`advantage > 0` → nudge that move's probability up) or *worse* (`advantage < 0` → nudge it down)?" Subtracting the value estimate — the "baseline" — is a standard variance-reduction trick: it centres the signal so the policy update reacts to *surprise* rather than to the raw outcome. (In this codebase the positive branch is actually all that's kept — there's a `max(0, advantage)` gate after normalization — but the value head's role as the thing-subtracted-from-`z` is the same.)

So the value head matters because (1) it has to learn something true about positions, and (2) its estimate feeds the policy gradient as a baseline.

---

## 2. The old way (scalar `tanh`) and why it's not good enough

The value head ended in a fully-connected layer producing **one number**, squashed through `tanh` so it lives in `[−1, +1]`:

`v = tanh(...) ∈ [−1, +1]`,   trained against `z ∈ {−1, 0, +1}` with **mean-squared error**:   `value_loss = (z − v)²`.

Two problems, both fatal in practice:

- **A scalar can't tell "dead draw" from "wildly unclear".** A position that is a guaranteed draw and a position that's a knife-edge 50%-win/50%-loss both have the same *honest* expected value: `v ≈ 0`. The scalar has no way to express the difference, so it throws away information that the rest of the system (and especially a contempt/anti-draw mechanism) would want.

- **On a draw-heavy buffer it collapses to ≈0.** Self-play between weak (random-ish) networks produces *mostly draws* — empirically ~75–85% of games end drawn (3-fold repetition, 50-move, insufficient material). The MSE-minimizing constant prediction for a distribution that's mostly `z = 0` is ≈0. So the value head learns "predict ≈0 for everything," `value_baseline ≈ 0` everywhere, `advantage ≈ z − 0 = z` — i.e. there's effectively *no position-dependent baseline*, the policy gradient gets no help from the value head, and (observed in the build-893 run) `vAbs` — the mean magnitude of the value output — decays to ~0.017 and stays there. The value head is, functionally, silent.

The point of the AlphaZero-style loop is that the value head *amplifies* a weak per-game outcome signal into per-position guidance. A silent value head can't do that.

---

## 3. The new way: a WDL softmax head + categorical cross-entropy

### 3.1 The head

The value head now ends in a fully-connected layer producing **three raw numbers** — *logits* (unbounded real numbers; the term just means "the inputs to a softmax"). There is **no `tanh`** any more. The three logits are turned into a probability distribution by **softmax** — exponentiate each, divide by the sum, so the three results are all positive and sum to 1:

`(p_win, p_draw, p_loss) = softmax(logit_win, logit_draw, logit_loss)`

**Slot order is `[win, draw, loss]`** — the way people say it ("WDL"), which makes it harder to accidentally transpose somewhere. Concretely: slot 0 = win, slot 1 = draw, slot 2 = loss. Everything below assumes that order.

(Implementation note: route all access to the three slots through a named accessor — `enum WDL { case win = 0, draw = 1, loss = 2 }` or similar — never bare `[0]`/`[1]`/`[2]` scattered around. A stray index that should've been `.win` but reads `.loss` should be a compile error, not a silent sign flip.)

### 3.2 The training target — a "one-hot" of what actually happened

A **one-hot** vector is a vector that's all zeros except a single `1`. The training target for a position is the one-hot of *the result that game actually had*, in `[win, draw, loss]` order:

| game result for the side to move | `z` | target |
|---|---|---|
| won | `+1` | `[1, 0, 0]` |
| drew | `0`  | `[0, 1, 0]` |
| lost | `−1` | `[0, 0, 1]` |

The buffer still stores `z` as the same single scalar in `{−1, 0, +1}` it always has (nothing about that changes — see §6). The one-hot is built on the fly inside the training graph from `z`:

`target = oneHot(index = 1 − z, depth = 3)`

Check: `z = +1` → index `0` → `[1,0,0]` (win); `z = 0` → index `1` → `[0,1,0]` (draw); `z = −1` → index `2` → `[0,0,1]` (loss). ✓ (The `1 − z` is slightly less tidy than the `z + 1` you'd get with a `[loss, draw, win]` order, but `[win, draw, loss]` is the order that's hard to get wrong by eye, and the bookkeeping is one place either way.)

### 3.3 The loss — categorical cross-entropy

**Cross-entropy** between a target distribution `t` and a predicted distribution `p` is `−Σ_i t_i · log p_i` — sum over the three W/D/L slots. With a one-hot target `t = e_j` (a `1` in slot `j`), every term is zero except slot `j`, so it collapses to:

`value_loss = −log p(slot of the actual outcome)`   — averaged over the positions in the minibatch.

In words: "how much probability did the head put on the result that actually happened?" — small (→ 0) when the head was confidently right, large (→ ∞) when it put ≈0 there, `≈ log 3 ≈ 1.10` for a head that's maximally confused (uniform `⅓, ⅓, ⅓`). It's always ≥ 0. The gradient on each logit is `p_i − t_i` — bounded in `[−1, 1]` regardless of how concentrated the prediction is, so there's no logit-magnitude runaway from this term.

(Numerically you compute `logSoftmax(logits)` — `log(softmax(...))`, done in one stable step as `logits − logsumexp(logits)`, giving the three log-probabilities directly — and dot it with the (negated) target. The `Σ` is over the 3 W/D/L slots; the batch is averaged separately.)

### 3.4 Why WDL + CE is better

- **Richer signal.** The head fits a 3-number distribution instead of one number — strictly more for the gradient to grab onto.
- **It stays informative on a draw-heavy buffer.** Faced with mostly-draws, the CE-minimizing answer is `p_draw ≈ 1` — a *confident* prediction, not a "shrug" — and that confidence on draws does **not** crowd out confident win/loss probabilities on the ~15–25% of decisive games. Those decisive games are the only place a per-position skill signal exists; a head that handles them well (and *knows* it's confident, vs. the scalar's ambiguous `≈0`) is the difference between "value head provides gradient" and "value head silent."
- **It distinguishes dead-draw from sharp.** `(0.1, 0.8, 0.1)` vs `(0.45, 0.1, 0.45)` are very different positions; the scalar reported both as `≈0`. Knowing the difference matters for the head's own calibration and for any future risk-/contempt-aware mechanism.
- **Cleaner training target.** CE against a sharp one-hot is a better-behaved optimization landscape than MSE against a continuous target that's mostly sitting at 0 (which is precisely the "collapse to ≈0" attractor).

---

## 4. Label smoothing on the value target (knob present, default OFF)

A new tunable parameter, **`value_label_smoothing_epsilon`** (`ε`), is added — default **0.0** for the first run, range ~`[0, 0.5]`, live-tunable. When `ε > 0`, the hard one-hot target is "softened" by blending in a touch of the uniform distribution over the three slots:

`smoothed_target = (1 − ε) · oneHot + ε · (⅓, ⅓, ⅓)`

So the previously-`1` slot becomes `1 − ε + ε/3 = 1 − 2ε/3`, and each previously-`0` slot becomes `ε/3`. Worked examples for the win target `[1, 0, 0]`:

| ε | smoothed `[win, draw, loss]` target |
|---|---|
| 0.000 | `[1.0000, 0.0000, 0.0000]` (the hard target) |
| 0.025 | `[0.9833, 0.0083, 0.0083]` |
| 0.050 | `[0.9667, 0.0167, 0.0167]` |

(Same shape for the draw and loss targets, just with the `1−2ε/3` mass on the appropriate slot.)

**Why you might want it:** with a *hard* one-hot target, the loss is only truly minimized when `p = 1` in the right slot — and `softmax` only reaches `p = 1` when that logit is `+∞`. So a hard target gently pushes the logits to grow without bound. A *smoothed* target's minimum is at *finite* logits (you hit the target by being ~96.7% / 1.7% / 1.7% confident, not 100% / 0% / 0%), and the loss becomes **self-correcting** — if the head gets *more* confident than the smoothed target wants, the gradient on that logit flips sign and pulls it back. The smoothed head is also slightly better-calibrated.

**Why it's off for v1, but in the code:** the value head — unlike the policy head, which had an unbounded-logit divergence that label smoothing helped tame — has no logit-runaway failure mode of its own (its honest answer on a draw-heavy buffer is `p_draw ≈ 1`, which is *correct*, not pathological). So the first WDL run keeps `ε = 0` to make the result cleanly attributable to "WDL alone." If the value logits drift hot in practice, turn `ε` up to a small value (≈0.01–0.05) live; no rebuild.

---

## 5. How the value still feeds the policy gradient (and what we're *not* doing)

### 5.1 Reducing WDL → a scalar baseline

The policy gradient needs the advantage `A = z − value_baseline` to be a **single number per position**, on the same axis as `z` (which is `−1` / `0` / `+1`). So we project the WDL distribution onto that axis:

`value_baseline = p_win − p_loss`

That is exactly the **expected outcome** under the predicted distribution: `E[outcome] = (+1)·p_win + (0)·p_draw + (−1)·p_loss = p_win − p_loss`. So `A = z − E[outcome]` = "how far did the game beat (or fall short of) the value head's expectation," which is the textbook advantage.

(`p_draw` not appearing here is *not* "throwing it away" — a draw contributes `0` to the expected outcome either way, so it genuinely isn't part of `E[outcome]`. Two distributions `(0.3, 0.4, 0.3)` and `(0.1, 0.8, 0.1)` both have `E = 0` and *should* give the same baseline — in expectation the game is a wash for both. `p_draw` *is* used: it's a third of the value head's CE target, and the head learns it; it's just not part of the *expected-value* projection, because mathematically it isn't.)

Why the `[win, draw, loss]` slot order pairs with the weights `[+1, 0, −1]` for this reduction: slot 0 (win) gets `+1`, slot 1 (draw) gets `0`, slot 2 (loss) gets `−1`. Keep that pairing consistent with the one-hot mapping (`index = 1 − z`) everywhere.

### 5.2 Why not a "full 3-way advantage"

You could imagine comparing the realized one-hot outcome to the predicted distribution slot-by-slot — `A_vec = oneHot(actual) − (p_win, p_draw, p_loss)`, a 3-vector. But the policy gradient needs a scalar weight ("nudge this move how much, which way"), so you'd have to project `A_vec` onto *some* axis — and the only axis that means "did this move help" is the outcome axis `[+1, 0, −1]`, whose projection is `(oneHot(actual) − p) · [+1,0,−1] = z − (p_win − p_loss) = z − E[outcome]` — **exactly the scalar we already use.** The 3-vector carries no extra *directional* information; its only extra is a magnitude (≈ "how surprised was the value head"), which is a separate confidence-weighting idea, partly redundant with the advantage normalization we already do, and known to be noisy in RL. Meanwhile a genuine 3-way advantage would force re-deriving the whole policy-loss machinery (the RMS normalization, the `max(0,·)` gate, the outcome-weighted CE — all built around a scalar advantage), which is a big, risky change for zero gain. So: **WDL where it helps (the value head's representation and loss), scalar where the policy gradient needs it (the baseline), and the projection between them is exact for the directional content.**

### 5.3 No contempt knob in the WDL change

We considered adding a tunable weight `k` on `p_draw` in the scalar reduction (`p_win − k·p_draw − p_loss`) to make the engine prefer decisive games over draws. We're **not** adding it. In an *advantage-weighted policy gradient* (this engine), the direct way to discourage draws is to make the *outcome* of a drawn game look bad — turn `z = 0` into `z = −draw_penalty` for drawn positions, so `A = z − value_baseline` goes negative for draw-leading moves and the policy learns to avoid them. That's exactly what the **existing `draw_penalty` parameter** does (it's already in the code; currently 0). A baseline-side draw term has subtle sign- and `max(0,·)`-gate interactions and at best duplicates `draw_penalty`. So the WDL change introduces *no new contempt parameter*; if anti-draw pressure is ever wanted, the lever is `draw_penalty`. (A *future* WDL-aware refinement would be to scale the penalty by `p_draw` — penalize confidently-drawn games harder than sharp-but-drew ones — which the scalar head couldn't have done; but that is not part of this change.)

### 5.4 Which network's value is the baseline, and the dropped "drift" diagnostic

The `value_baseline` is computed by a **fresh, forward-only pass on the trainer's *current* network** over the training batch — a separate pass, *not* the value output of the main training-graph forward. Three reasons it has to be separate: (a) MPSGraph has no `stop_gradient` op, so using the training forward's value as *both* the CE target *and* the policy baseline leaks gradient back through the baseline path into the network body (verified by `MPSGraphGradientSemanticsTests`); (b) a separate forward with no backward is naturally gradient-detached; (c) it can run with inference-mode batch-norm, matching the value the network "plays with." (This forward pass already exists; the WDL change just makes it emit the 3-way distribution and reduce it to `p_win − p_loss`.)

Previously the replay buffer *also* stored, per position, the **champion's** value at the moment the move was played — used **only** to compute a `vBaselineDelta` "drift" diagnostic (how far the trainer's current value estimate has moved from what the champion thought). **This is being removed:** the buffer no longer stores any play-time value, and the `vBaselineDelta` / `freshBaselineMs` metrics go away. Nothing else is affected — the advantage baseline always used the fresh trainer-network pass, never the stored play-time value.

---

## 6. What does **not** change

- **The replay buffer's outcome encoding.** Still one scalar `z ∈ {−1, 0, +1}` per position, from that position's mover's perspective, equal to the final game result, no discounting. (The one-hot is built in-graph from `z` — see §3.2. The only buffer change is *removing* the play-time `vBaseline` field — see §5.4.)
- **The entire policy side.** Policy head topology; policy cross-entropy on *raw* (unmasked) logits with `policy_label_smoothing_epsilon = 0.1`; legal-move masking for the entropy bonus and for move selection; RMS advantage normalization; the `max(0, advantage)` gate; the entropy bonus; the illegal-mass penalty; gradient clipping (`gradClipMaxNorm`); weight decay; momentum; learning rate; the LR-warmup gate. All untouched.
- **Move selection.** Still policy-only: mask illegal moves → temperature-scale → softmax → categorical sample. The value head plays no part.
- **Self-play.** Still champion-vs-champion games into the replay buffer; the champion's value output is computed (it's part of the forward) but is no longer stored anywhere and never influenced move selection in the first place.
- **The arena.** Arena play is policy-only, unchanged. Only UI that *displays* a value scalar (dashboard, candidate probe) needs to derive it from WDL (`p_win − p_loss`, or show the W/D/L bar directly).
- **`value_loss_weight`.** Still multiplies the value loss; stays `1.0` for v1. (Note that categorical CE has a different magnitude than the old MSE — ~0 to ~1.5, vs ~0.1 to ~0.4 — so the *relative* weight of the value loss in the total loss shifts a bit; re-check the loss-component balance once it's running and re-tune if needed.)

---

## 7. Metrics changes (so the monitoring rubric stays accurate)

- New per-step metrics: `pW` / `pD` / `pL` — the mean across the batch of `p_win` / `p_draw` / `p_loss`.
- `vMean` is kept, redefined as the mean of `p_win − p_loss` (the scalar baseline) — preserves continuity of the value chart.
- New: `vConf` — the mean of `max(p_win, p_draw, p_loss)`. This is the WDL analog of the old "is the value head saturating?" watch (the old check was `vAbs → 1`). If `vConf` heads toward 1 for *every* position on a noisy buffer, the head has gone confidently-deterministic — a saturation-style failure worth flagging.
- `vAbs` (mean magnitude of the old scalar) is dropped — `vConf` replaces its diagnostic role.
- `vLoss` keeps its name but is now **categorical CE** (≈0–1.5), not MSE (≈0.1–0.4). Alarm thresholds and the monitoring rubric must note the scale change.
- `vBaselineDelta` and `freshBaselineMs` are removed (see §5.4).

These names need to be settled before the chart pipeline (`TrainStepTiming` → `TrainingLiveStatsBox` → `TrainingChartSample` → `ChartDecimation` → the value tile) is re-plumbed.

---

## 8. Compatibility

The value head's output shape changes (final FC layer `1 → 3`), so the network's architecture hash changes. **Old `.dcmsession` / `.dcmmodel` files become incompatible** and must be rejected on load with a clean error ("incompatible model — start fresh"), not a crash or a silent mis-load. This is a fresh-start change; you cannot resume a scalar-head session under the WDL build.

---

## 9. Initialization detail

The WDL head's final fully-connected layer: **weights** small / Xavier-ish (so the head starts ~position-independent); **bias** set so `softmax` starts near `(p_win, p_draw, p_loss) ≈ (0.125, 0.75, 0.125)` — the empirically-draw-heavy prior. That's a bias vector of roughly `[0, ln 6, 0] ≈ [0, 1.79, 0]` in `[win, draw, loss]` order (only the *differences* between the three matter; `[−0.9, 0.9, −0.9]` is the same thing). Low-stakes — the head learns the true draw rate within a few hundred steps regardless — but it's a free, principled head-start since the init code is being touched anyway.

---

## 10. Settled v1 scope (what's explicitly out)

- **No A/B toggle** — the WDL head outright replaces the scalar head; we do not maintain both code paths. (The scalar head's results are pinned in the `(FINDING)` CHANGELOG entry; it doesn't need to stay runnable.)
- **`value_label_smoothing_epsilon = 0` for v1** — the mechanism is in the code; the first run uses 0 so the result is attributable to WDL alone.
- **No resign / adjudication** — games still play to natural termination (mate / 50-move / 3-fold / insufficient material). An AlphaZero-style "resign when the value head is confident you've lost, but play out X% anyway to avoid bias" would speed up self-play and cut noise — and a *working* WDL head could drive it — but it has its own knobs and is a later, separate change.
- **No contempt knob** (§5.3) and **no dead-draw-aware anti-draw** — `draw_penalty` (already present, currently 0) is the anti-draw lever if/when wanted; the `p_draw`-scaled refinement is a future tweak.

---

## 11. The math, all in one place

Per position, with stored outcome `z ∈ {−1, 0, +1}` (from the mover's view) and value-head logits `ℓ = (ℓ_win, ℓ_draw, ℓ_loss)`:

```
p           = softmax(ℓ)                               # (p_win, p_draw, p_loss), sums to 1
target_hard = oneHot(index = 1 − z, depth = 3)         # win:[1,0,0] draw:[0,1,0] loss:[0,0,1]
target      = (1 − ε)·target_hard + ε·(⅓, ⅓, ⅓)        # ε = value_label_smoothing_epsilon (0 for v1)
value_loss  = − Σ_slots target[slot] · log p[slot]     # categorical cross-entropy; mean over batch
total_loss  = w_p·policy_loss  +  w_v·value_loss  −  ent·policy_entropy  +  illM·illegal_mass_penalty
                                  └ w_v = value_loss_weight (1.0)

# Used by the policy gradient (computed from a fresh, gradient-detached, inference-mode forward
# of the trainer's current network — NOT from the line above):
value_baseline = p_win − p_loss                        # = E[outcome] under p
advantage      = z − value_baseline                    # then RMS-normalized, then max(0, ·)-gated,
                                                       #   then multiplies the policy CE — all unchanged
```

Everything outside that box is exactly as it was before this change.
