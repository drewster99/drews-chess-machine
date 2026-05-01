# Candidate New Parameters for `parameters.json`

Brain-dump of training-relevant knobs that currently live as constants in
Swift code. Grouped by likely impact on training outcomes. Defaults are
chosen so a no-op iteration matches today's hardcoded behavior.

---

## High Impact

These are first-class RL/exploration levers with plausible large effects on
training-run metrics. Worth parameterizing before letting autotrain edit
the corresponding source files.

### Dirichlet exploration noise (self-play and arena)

Today: `DirichletNoiseConfig.alphaZero` is hardcoded into
`SamplingSchedule.selfPlay`; `SamplingSchedule.arena` has no noise. All CPU-side
(post-softmax mix), so nothing graph-related blocks exposing them.

Open design choice — two-schedule (recommended) vs. three-schedule:

**Two-schedule (recommended):** symmetric arena noise, preserves the
"arena measures strength symmetrically" invariant.

```
self_play_dirichlet_alpha          (default 0.3,  range [0.01, 5.0])
self_play_dirichlet_epsilon        (default 0.25, range [0.0,  1.0])
self_play_dirichlet_ply_limit      (default 30,   range [0,    200])

arena_dirichlet_alpha              (default 0.3,  range [0.01, 5.0])
arena_dirichlet_epsilon            (default 0.0,  range [0.0,  1.0])   # 0 = off
arena_dirichlet_ply_limit          (default 0,    range [0,    200])
```

Convention: `epsilon == 0` OR `ply_limit == 0` ⇒ noise disabled. No
separate boolean flag.

**Three-schedule (only if asymmetric arena noise is desired):** parallel
triples for `arena_champion_*` and `arena_candidate_*` instead of
`arena_*`. Breaks symmetry of the arena signal — only do this if there
is a specific reason to want asymmetric noise.

### Replay buffer sampling distribution

Today: uniform-with-replacement over all stored positions
(`Int.random(in: 0..<held)` per draw, independent draws per batch slot).
Real RL knobs orthogonal to anything currently in `parameters.json`.

**Sampling mode (mutually exclusive):**

```
replay_sample_mode                 (default "uniform",
                                    enum: "uniform" | "recency" | "td_priority")
```

**Without-replacement option (orthogonal to mode):**

```
replay_sample_without_replacement  (default false, bool)
```
When true, the `count` draws within a single minibatch are guaranteed
distinct (Fisher-Yates over the held range). Eliminates intra-batch
duplicates — most relevant when buffer is small relative to batch.

**Recency-biased sampling** (active when `replay_sample_mode = "recency"`):

```
replay_recency_decay_per_position  (default 0.0,    range [0.0, 1e-3])
```
Probability of position at age `a` (positions ago) ∝ `exp(-decay * a)`.
`0.0` reduces to uniform. `1e-5` over a 500K buffer gives the newest
position ~150× the weight of the oldest.

Alternative simpler form (could ship instead of decay):
```
replay_recency_window_fraction     (default 1.0,    range [0.05, 1.0])
```
Sample uniformly from the newest `floor(window_fraction * held)`
positions only. `1.0` reduces to uniform.

**TD-error / advantage-magnitude priority**
(active when `replay_sample_mode = "td_priority"`):

```
replay_priority_alpha              (default 0.6, range [0.0, 1.0])
replay_priority_beta               (default 0.4, range [0.0, 1.0])
replay_priority_epsilon            (default 1e-3, range [1e-6, 1e-1])
```
Weights position `i` ∝ `(|z_i - vBaseline_i| + epsilon)^alpha`. `alpha = 0`
reduces to uniform; `1.0` is fully proportional. `beta` controls
importance-sampling correction on the loss to compensate for the
non-uniform sampling (PER convention). `epsilon` keeps zero-advantage
positions sample-able.

Note: TD-priority requires recomputing or storing a freshness signal —
worth confirming `vBaseline` staleness is acceptable (it's the
inference-time `v` from when the position was played, not the current
network's `v`). If staleness matters, this knob may need more than just
parameter exposure — it's a code-and-params change.

**Outcome-class reweighting** (orthogonal to mode — multiplies the
mode's base weights):

```
replay_outcome_win_weight          (default 1.0, range [0.0, 10.0])
replay_outcome_loss_weight         (default 1.0, range [0.0, 10.0])
replay_outcome_draw_weight         (default 1.0, range [0.0, 10.0])
```
Skews sampling toward/away from outcome classes. `draw_weight = 0.3`
samples draw positions ~3× less often than W/L positions, useful when a
draw-heavy regime drowns the decisive signal. All `1.0` reduces to the
underlying mode unchanged. Implementation: per-position weight is
`mode_weight * outcome_weight`, then sampled accordingly.

---

## Low Impact

Numerical-stability hedges and metric-definition knobs. Probably not
worth parameterizing on their own — more useful as code changes that
autotrain proposes when something specific is misbehaving. Listed for
completeness.

### Sample-time numerical knobs (`MPSChessPlayer.chooseMove`)

```
softmax_logit_clip                 (default -50.0, range [-200.0, -10.0])
```
Clamp `(logit - max_logit) / tau` to this lower bound before `exp`.
Defends against denormal flush at extreme tau. Pure numerical hedge.

```
policy_top_k_cap                   (default 0,     range [0, 4864])
```
Truncate post-softmax legal-move distribution to top-K before
renormalize+sample. `0` disables. Sharpens self-play in a way Dirichlet
can't reach (cuts the long tail). Only mid-impact knob in this section
— could arguably go in High Impact if you have evidence the long tail
matters.

```
policy_min_legal_prob_floor        (default 0.0,  range [0.0, 1e-3])
```
Clamp legal-move probabilities to a minimum before final renormalize, so
the network can never assign exactly zero to a legal move during
sampling. `0.0` disables.

### `legalMassCollapse` metric definition

Today: threshold, grace seconds, no-improvement probes are all already
parameterized. What's left in code is the metric itself.

```
legal_mass_collapse_metric         (default "max_prob",
                                    enum: "max_prob" | "mean_entropy")
```
Switches between "max legal-move softmax probability < threshold"
(current) and "mean policy entropy < threshold". The latter responds to
diffuse-but-uniform collapse that `max_prob` would miss.

```
legal_mass_collapse_min_batches    (default 1, range [1, 100])
```
Require N consecutive batches over threshold before tripping the alarm.
Defaults to current behavior (1 = trip immediately).

Both of these are best left in code unless you've seen a specific case
where the metric definition itself was wrong.

---

## Caveats and judgment calls

- **TD-priority needs more than parameter exposure.** The `vBaseline`
  stored in the replay buffer is *stale* — it's the inference-time
  `v(position)` from the worker that played the position, not the current
  network's `v`. Prioritized Experience Replay assumes priority is
  recomputed (or at least refreshed periodically) against the current
  network. So `replay_sample_mode = "td_priority"` may need real code
  beyond parameter exposure to be effective. It's the riskiest of the
  high-impact knobs — listed because the surface exists, but don't ship
  the params without thinking through the staleness story (e.g. periodic
  re-scoring of the buffer, or accepting the bias of stale priorities).

- **`policy_top_k_cap` sits awkwardly between High/Low.** Placed in Low
  Impact because the rest of that block is numerical hedges, but it's the
  one knob there that could plausibly swing training metrics — it cuts
  the long tail of the policy in a way Dirichlet noise can't reach. Easy
  to promote to High Impact if there's evidence the long tail matters.

- **Three-schedule arena Dirichlet breaks symmetry.** The two-schedule
  design is strongly recommended over the three-schedule alternative.
  Asymmetric noise (different Dirichlet on champion vs. candidate)
  invalidates the "arena measures strength symmetrically" property — one
  side gets noise, the other doesn't, so the score is no longer a clean
  strength comparison. Only adopt three-schedule if there's a specific
  reason to want asymmetric noise.
