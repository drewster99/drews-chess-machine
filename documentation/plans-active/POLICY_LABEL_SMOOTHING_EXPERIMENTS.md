# Policy / value label smoothing — proposed experiments

**Status:** proposal (2026-10-01). Arm B's per-move mode **implemented 2026-10-02, not
yet run** (see "Implementation notes" below); arms C and D need no code. Queue after the
leaky-FC1 experiment (`experiments/20261001-se-fc1-leaky/`) frees the GPU.

## Background: what DCM does today

- **Policy** (`policy_label_smoothing_epsilon`, owner config 0.1; added in `419c07f`
  "Anti-collapse", 2026-05-10). Target per position, built in-graph
  (`Training/HeadLossGraph.swift`), legal moves only:

  `target = (1 − ε)·one_hot(played) + ε · legalMask / |legal|`, renormalized in fp32.

  Illegal cells are exactly 0. The complement target for negative-advantage
  positions mirrors it. Motivation at the time: a hard one-hot target is minimized
  only at logit(played) = +∞; before smoothing one run reached `pLogitAbsMax`
  45,710. Smoothing gives a finite optimum.
- **Value** (`value_label_smoothing_epsilon`, owner config 0.013): `(1 − ε)·one_hot(result)
  + ε·(⅓, ⅓, ⅓)`. Shipped at 0 with the W/D/L head (`wdl-value-head.md` §4: the value
  head showed no logit runaway); the owner's configuration later set 0.013.
- Since the head-numerics fix (`da15920`, Phase 2) the loss centers the logits, so
  the shared offset cannot drift whatever the targets; smoothing is no longer needed
  for that. It still bounds single-logit growth and shapes calibration.

## Why the current policy form is questionable

The total smoothing mass is fixed (ε), so the mass per alternative and the trained
logit gap depend on how many legal moves a position has:

| legal moves | each alternative | played-move target | equilibrium gap, played vs each alternative |
|---:|---:|---:|---:|
| 2 (e.g. in check) | 0.050 | 0.950 | ln(0.95/0.05) = 2.9 |
| 23 | 0.0043 | 0.904 | 5.3 |
| 30 | 0.0033 | 0.903 | 5.6 |
| 40 | 0.0025 | 0.903 | 5.9 |

So the net is trained to be *least* decisive in forced / narrow positions — often
where one move is clearly right — and the "≈ 0.3% floor" on every legal move moves
with the branching factor.

**Proposed alternative (owner's suggestion): constant mass per alternative.** Give
every non-played legal move the same δ, so the total grows with the number of
choices (additive / Lidstone smoothing — a per-move pseudo-count):

`target = (1 − δ·(|legal| − 1))·one_hot(played) + δ·(legalMask − one_hot(played))`

| legal moves | each alternative (δ = 0.0033) | played-move target | gap |
|---:|---:|---:|---:|
| 2 | 0.0033 | 0.997 | 5.7 |
| 23 | 0.0033 | 0.927 | 5.6 |
| 30 | 0.0033 | 0.903 | 5.6 |
| 40 | 0.0033 | 0.870 | 5.6 |

The equilibrium gap ≈ ln((1 − δ(n−1))/δ) is nearly independent of n, which fits a
softmax over per-move scores: "the played move is ~5.6 nats better than each
alternative" means the same thing in every position. A guard is needed for very wide
positions (max ~218 legal moves): cap the total, e.g. `min(δ·(n−1), ε_max)` with
ε_max such as 0.5, and say so in the log.

Expected impact on *play* is small: at the near-greedy sampling temperatures in use
(τ 0.2 → 0.02) both floors are raised to the power 1/τ and vanish, and argmax is
unchanged. Where it can show: calibration and NLL (probes score at τ = 1), forced
positions, and possibly sharper features from consistent per-position margins.

## Prior work and analogues

No published work found that applies a *constant per-move* label-smoothing mass over
a *variable legal-action set*. The closest analogues:

- **Standard label smoothing** (Szegedy et al. 2016, "Rethinking the Inception
  Architecture") and its analysis (Müller, Kornblith, Hinton 2019, "When Does Label
  Smoothing Help?", <https://arxiv.org/abs/1906.02629>): fixed total ε spread uniformly
  over a fixed class set K — the two forms coincide when K is fixed, which is why
  the question never arises in classification.
- **Masked label smoothing** (Chen, Xu, Chang, ACL 2022, "Focus on the Target's
  Vocabulary", <https://arxiv.org/abs/2203.02889>): smoothing restricted to the valid
  target-side vocabulary instead of all tokens — the analogue of DCM's legal-only
  smoothing; it keeps a fixed total and reports better translation quality and
  calibration than unmasked smoothing.
- **Additive / Lidstone / Dirichlet smoothing** of count estimates (classical NLP):
  a pseudo-count α per category, `(c_k + α)/(N + Kα)`, so the smoothing mass grows
  with the number of categories — the owner's per-move form in count language.
- **Root Dirichlet noise in AlphaZero vs KataGo** — the same fixed-total vs per-move
  question, for exploration noise rather than targets:
  - AlphaZero (Silver et al. 2017, <https://arxiv.org/abs/1712.01815>) uses a fixed α
    *per move* within each game (chess 0.3, shogi 0.15, Go 0.03), scaled per game "in
    inverse proportion to the approximate number of legal moves in a typical
    position" — so the total varies with each position's legal-move count.
  - KataGo (Wu 2019, <https://arxiv.org/abs/1902.10565>; KataGoMethods,
    <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>) instead
    uses a **fixed total** α of 10.83 (= 361 × 0.03) divided across the position's
    legal moves. Two strong engines made opposite choices; neither published an A/B.
- **Soft policy targets in search engines:** Lc0 and KataGo train the policy on MCTS
  visit distributions, not one-hot moves, so they rarely need label smoothing;
  KataGo additionally adds an auxiliary head on the policy target raised to 1/T.
  DCM has no search, so its policy targets are one-hot played moves and smoothing
  stands in for that softness.
- **Learned / instance-based smoothing** (e.g. Maher & Kull 2021, instance-based label
  smoothing, <https://arxiv.org/abs/2110.05355>; Label Smoothing++,
  <https://arxiv.org/abs/2509.05307>): distribute the smoothing mass non-uniformly by
  class similarity — a possible later step (e.g. weight alternatives by a teacher
  model) but out of scope here.

## Proposed experiments

All on the SE-experiment architecture (same fresh net, `20260929-12-JZOe`, derived
copies where needed), corpus `20260624-192615-w3aA5b`, the SE experiment's pinned
`parameters.json`, corpus replay to the same step count, run one at a time on an
otherwise idle GPU, enumerated checkpoints probed every 1,000 steps. Compare on step
(and `games_fed`), never time. Seed-to-seed spread on this setup is 7–25 pElo, so
treat smaller gaps as noise unless they persist across many checkpoints.

| # | arm | policy smoothing | value ε | question |
|---|---|---|---:|---|
| A | baseline | fixed total ε = 0.1 | 0.013 | (existing ReLU / leaky-FC1 runs) |
| B | per-move | per-move δ = 0.0033 (total capped at 0.5) | 0.013 | does a constant per-move mass help? |
| C | less policy smoothing | fixed total ε = 0.03 | 0.013 | is 0.1 more than needed now that the shared offset can't drift? |
| D | no value smoothing | fixed total ε = 0.1 | 0 | is value smoothing needed post-fix? |

Measurements: pElo / NLL (probe set `wide`); top-1 accuracy; NLL and top-1 **binned
by legal-move count** (≤ 5, 6–15, 16–30, 31–45, > 45) — the decisive diagnostic for B;
calibration (expected calibration error of p(played)); `pLogitAbsMax`, policy entropy;
value loss / W-D-L calibration for D; training loss for all.

## Implementation notes (for B; C and D need no code)

**Implemented 2026-10-02 — not yet run.** The plan as written, with
the deviations and decisions noted below.

- New parameter `policy_label_smoothing_mode` (`fixed_total` | `per_move`) and
  `policy_label_smoothing_per_move` (δ) — full parameter checklist (CLAUDE.md),
  session/resume fields, `[REPLAY-HPARAMS]` / `[STATS]` visibility, results.json.
  Old behaviour stays the default until the A/B decides.
  - *Done.* Three parameters, not two: the cap is its own parameter,
    `policy_label_smoothing_per_move_cap` (default 0.5, range 0…0.9 — the same ceiling
    as ε, since at 1.0 a wide position's played move would get no target mass).
    δ: default 0.0033, range 0…0.05.
  - The mode is stored as an `Int` (0 = `fixed_total`, 1 = `per_move`) because
    `ParameterType` has no enum case; it follows `arena_promotion_criterion`
    (`PolicyLabelSmoothingMode`, pinned to the declared range by test). In
    `parameters.json` write `"policy_label_smoothing_mode": 1` for arm B. Session files
    carry the token (`"fixed_total"` / `"per_move"`).
  - All three are `liveTunable` and **fed** to the graph every step, like ε: the mode is
    a 1/0 selector placeholder, both target forms are built, and `select` picks one. A
    mode switch reaches a running trainer on its next step without a graph rebuild.
  - They travel through `TrainerHyperparameters`, so GUI Play-and-Train, corpus replay
    and train-vs-UCI all get them from the one shared path.
  - Visible as `pLabelSmooth=… pLabelSmoothMode=… pLabelSmoothPerMove=…
    pLabelSmoothPerMoveCap=…` in `[REPLAY-HPARAMS]`, `[VS-UCI-HPARAMS]` and the GUI
    `[STATS]` line's `reg=(…)` group, and as `policy_label_smoothing_*` fields on every
    `results.json` stats entry. Resume logs one `[RESUME-PARAM]` line per field; a
    session saved before the feature resumes in `fixed_total`.
  - UI: Training settings ▸ Optimizer gets a mode picker with ε, δ and the cap beneath
    it (policy ε previously had no UI). The fields the current mode does not read stay
    visible but dimmed and disabled, as the Arena popover does for its
    criterion-dependent fields.
- `HeadLossGraph` target builder: branch on the mode for both the positive and the
  complement targets; keep fp32 and the renormalization; apply the total cap.
  - *Done* (`HeadLossGraph.policyTargets(…labelSmoothing:…)`). Positive target: each
    non-played legal move gets `per = min(δ, cap/(n − 1))`, the played move
    `1 − per·(n − 1)` — exactly δ below the cap, the cap shared equally above it, an
    exact one-hot at n = 1. Computing `per` as a `min` (rather than dividing the capped
    total back by n − 1) keeps the raw per-move mass exactly δ below the cap.
  - **Complement-target decision.** The fixed-total complement reuses the positive
    target's smoothing part (ε over all legal moves), which leaves the played move a
    floor of ε/|legal|. Reusing the per-move smoothing part literally would put δ only on
    the *other* legal moves, leave the played move at exactly 0, and make the complement
    plain uniform(other legal) for every δ — and a 0 target is the unbounded drive
    (p(played) → 0) smoothing exists to remove. Implemented instead as the role-swapped
    mirror: the played move (the one legal move outside the complement's target set)
    gets the per-move floor `min(δ, cap)`; the other legal moves share the rest equally.
    The negative branch's equilibrium is then p(played) → δ instead of ε/|legal|.
  - Rows with no legal moves (impossible in practice) are handled the way the
    fixed-total form handles them: |legal| clamped at 1, so the target is finite.
- Tests: target sums to exactly 1 for every legal count 1…218; per-move mass equals δ
  below the cap; cap engages above it; complement mirror; fixed-total mode
  bit-identical to today.
  - *Written* (`DrewsChessMachineTests/PolicyLabelSmoothingModeTests.swift`), plus: a
    single graph switches forms by feed; illegal cells exactly 0; the parameters'
    ids/defaults/ranges and `parameters.json` load + apply; the shared trainer path;
    session-state round trip and absent → nil → `fixed_total`.
