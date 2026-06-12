# Architecture Expansion Plan: Dropout + Heterogeneous Block Towers

Two new architecture capabilities for the Build-New-Model / runtime-config
system (`NetworkArchitecture`), motivated by the 2026-06-12 regularization
findings (Exp 6 dose-response) and the WRN/DenseNet design discussions:

1. **Dropout** — configurable regularization noise inside residual blocks.
2. **Heterogeneous blocks** — per-block specs (kernel sizes, SE, dropout)
   in a single tower, replacing the uniform-block assumption.

Both ride the embedded-config identity system (PLAN §6 of
RUNTIME_ARCHITECTURE_CONFIG_PLAN.md): no arch_hash involvement; identity is
the embedded JSON config + architectureSummary.

---

## Feature 1 — Dropout

### Design (revised 2026-06-12 after design discussion)

**Dose/structure split.** The dropout *rate* is a training hyperparameter
(like weight decay — no weights, no inference effect, and the Exp 6
dose-response discovery happened precisely because wd was tunable
mid-run). The dropout *structure* (which blocks, what granularity, what
relative strength) is architecture. Concretely:

- **`dropoutRate` — a new `@TrainingParameter`** (snake-case
  `dropout_rate`, range `[0.0, 0.95]`, default 0.0, popover field,
  `liveTunable` via the per-step scalar-feed mechanism below). Global
  dose knob for the whole tower.
- **`dropout_multiplier` — per-BlockSpec Float** (architecture, default
  1.0, validated `>= 0`). Effective rate for block i =
  `clamp(dropoutRate × multiplier_i, 0, 0.95)`. Multiplier 0 = this
  block has no dropout regardless of the global knob; a depth-ramped
  profile (e.g. 0.25/0.5/0.75/1.0/1.0) expresses the stochastic-depth-style
  "early blocks gentle, late blocks heavy" schedule as pure architecture.
  Per-block-group control falls out of Feature 2's grouped BlockSpecs.
- **`dropout_granularity` — per-BlockSpec**: `unit | channel`, default
  `channel` (spatial dropout — masks whole 8×8 feature maps; the form
  that bites in convnets). There is deliberately NO `none` style — rate 0
  / multiplier 0 is the off switch, and the node at rate 0 is an exact
  identity.
- **Placement (fixed, WRN slot):** after the conv2-side BN+ReLU,
  immediately before conv2, inside the residual branch (see the block
  diagram from the 2026-06-12 discussion). Rationale: the block's own BN
  sees clean activations; conv2 is the regularized consumer; the skip is
  never touched; a fully-masked branch degrades toward a no-op via the
  clean add. NOTE: SE sits downstream of the mask, so SE gains are
  computed from masked activations — a deliberate deviation from plain
  WRN (SE learns missing-channel robustness too); flag in experiment
  write-ups.
- **Training-mode graph ONLY.** Node inserted only in training-mode graph
  builds (same switch as BN batch-stats vs running-stats). Inference
  networks (champion, arena, candidate probe, CLI probe) never contain
  it. The bit-exact forward-pass save verification doubles as the proof.
- **Rate is FED, not baked**: a per-step scalar placeholder (the existing
  lr/momentum/weightDecay feed pattern). Per-block effective rate is
  computed in-graph as `constant(multiplier_i) × ratePlaceholder`, then
  threshold + mask + `× 1/(1 − effRate)` (inverted dropout). Rate 0 ⇒
  all-ones mask, scale 1.0 ⇒ exact identity with zero rebuild — "off"
  costs three trivial ops per block.
- **Randomness:** prefer MPSGraph's stateful random op (seeded state
  variable on the training graph, fresh draw per step). Verify against
  the installed SDK during implementation; fallback is a per-step
  CPU-generated mask placeholder ([batch, channels] bytes — negligible,
  and deterministic for tests). RNG state is NOT persisted across
  save/resume — dropout noise is not model identity.

### Identity / persistence

- `dropoutRate` rides the full TrainingParameter checklist (CLAUDE.md):
  macro declaration, singleton wiring, parameters.json, session
  save/load `[RESUME-PARAM]` block, results.json snapshot, visibility in
  `[STATS]` (e.g. `drop=0.30` in the reg=() group), popover UI +
  validation, live-reconcile consumer.
- BlockSpec keys (`dropout_multiplier`, `dropout_granularity`) decode
  with pre-feature fallbacks (absent ⇒ 1.0 / channel) — but since the
  node only exists in training graphs and rate defaults to 0, legacy
  configs behave identically.
- `architectureSummary` shows the structure only when it deviates from
  default: multiplier ≠ 1.0 or granularity ≠ channel appends e.g.
  `drop(unit ×0.5)`. The RATE never appears in the summary (it is not
  architecture). Existing summaries unchanged (golden-string test).
- Param count unchanged; `weightTensorPlan` untouched.

### Granularity decision record (2026-06-12, implemented + tested)

**Chosen: channel/spatial dropout** — one Bernoulli coin per
(sample, channel), broadcast across the whole 8×8 board, so a dropped
feature map goes dark as a unit ("knight-fork detector offline this
position"). Verified by `DropoutGraphSemanticsTests` (slab masking, exact
1/(1−r) scaling, expectation preservation, gradient semantics).

Rejected alternatives, in granularity order:

- **Unit ("pinhole") dropout** — independent coin per scalar element.
  Rejected: adjacent squares within a feature map are strongly correlated
  on a board, so pinholes are trivially interpolated away and the
  effective regularization is far weaker than the nominal rate. This is
  the SpatialDropout argument (Tompson et al. 2015). Caveat honestly
  noted: WRN itself used unit dropout between its convs and saw gains on
  32×32 CIFAR maps — unit dropout is weak in convs, not useless. Kept as
  the future `dropout_granularity = unit` axis on BlockSpec if a literal
  WRN replication is ever wanted.
- **DropBlock (contiguous patches)** — the unit/channel midpoint, best
  ImageNet results of the three for ResNets (Ghiasi et al. 2018).
  Rejected here because its niche — patches larger than the correlation
  length but smaller than the map — barely exists at 8×8: a 4×4 block is
  already a quarter of the board, so the granularity spectrum collapses
  to its endpoints at this resolution.
- **Stochastic depth / DropPath (whole residual branch)** — the
  transformer-era standard (essential in ViT/DeiT training; ConvNeXt uses
  it INSTEAD of dropout). Not rejected — deferred: it is a different
  regularizer (depth-dimension redundancy, not channel-dimension), our
  per-block ReZero α multiply is the natural gate point, and at 5–12
  blocks every block is load-bearing (measured α profile 0.88–4.25), so
  the lazy-block redundancy it exploits barely exists yet. Future axis,
  not part of this feature.

References:

- Srivastava, Hinton et al., "Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting," JMLR 2014 —
  https://jmlr.org/papers/v15/srivastava14a.html (NOTE: parameterizes by
  RETENTION p; modern frameworks and this codebase use DROP probability)
- Tompson et al., "Efficient Object Localization Using Convolutional
  Networks," 2015 (SpatialDropout) — https://arxiv.org/abs/1411.4280
- Zagoruyko & Komodakis, "Wide Residual Networks," 2016 (in-block
  dropout placement we adopt) — https://arxiv.org/abs/1605.07146
- Huang et al., "Deep Networks with Stochastic Depth," 2016 —
  https://arxiv.org/abs/1603.09382
- Ghiasi et al., "DropBlock: A regularization method for convolutional
  networks," 2018 — https://arxiv.org/abs/1810.12890
- Li et al., "Understanding the Disharmony between Dropout and Batch
  Normalization," 2018 (the variance-shift argument behind our
  after-BN placement) — https://arxiv.org/abs/1801.05134

Post-2018 developments (context for future experiments):

- Fan et al., "Reducing Transformer Depth on Demand with Structured
  Dropout" (LayerDrop), 2019 — https://arxiv.org/abs/1909.11556
- Touvron et al., "Training data-efficient image transformers" (DeiT),
  2020 — https://arxiv.org/abs/2012.12877 (stochastic depth as the
  load-bearing regularizer in modern vision stacks)
- Pham & Le, "AutoDropout: Learning Dropout Patterns to Regularize Deep
  Networks," 2021 — https://arxiv.org/abs/2101.01761
- Liang et al., "R-Drop: Regularized Dropout for Neural Networks," 2021 —
  https://arxiv.org/abs/2106.14448 (consistency loss between two masked
  forward passes — an interesting cheap variant for us since it needs no
  extra data, only a second forward)
- Liu et al., "Dropout Reduces Underfitting," ICML 2023 —
  https://arxiv.org/abs/2303.01500 (early-dropout/late-dropout
  scheduling; reframes dropout as a gradient-variance reducer EARLY in
  training — the most relevant modern result for our data-rich regime,
  where classic always-on dropout has little overfitting to fight; our
  live-tunable rate makes their early-dropout schedule manually
  reproducible with zero extra code)

### Out of scope / explicit non-goals

- No dropout on the skip path, heads, or stem (v1).
- No stochastic depth (whole-block drop) — natural follow-on as a
  separate per-block gate probability on the α multiply (see decision
  record above).
- No time-scheduled rates (live tunability covers manual schedules —
  including hand-driving the Liu et al. early-dropout schedule).

---

## Feature 2 — Heterogeneous blocks (multiple block sizes in one tower)

### Design

- Replace the uniform block axes with **`blocks: [BlockSpec]`**, ordered
  input → output. `BlockSpec` (Codable, Hashable, Sendable):
  - `conv1_kernel_size`, `conv2_kernel_size` (odd, validated)
  - `se_style`, `se_reduction_ratio`
  - `use_rezero`, `rezero_alpha_init`
  - `dropout_multiplier`, `dropout_granularity` (Feature 1's structural
    axes live here; the global rate is a TrainingParameter, not arch)
- `numBlocks` becomes derived (`blocks.count`); remove the stored field
  from the struct, keep it in summaries/exports as a derived value.
- **Channels stay uniform tower-wide in v1.** Per-block channel widths
  require projection shortcuts on every width change (1×1 conv on the
  skip), changing the clean-identity story and the weight plan
  substantially. Explicit non-goal now; sketched as v2 below so the
  BlockSpec shape doesn't preclude it.
- `blockActivationStyle` / `blockSkipMerge` stay tower-wide (mixing
  pre/post-activation styles within one tower has no experimental
  motivation and complicates the tower-end BN rule).

### Identity / persistence

- **Decode fallback for every existing config**: when the legacy uniform
  keys (`num_blocks`, `block_conv1_kernel_size`, …) are present and
  `blocks` is absent, expand to `numBlocks` identical BlockSpecs. All
  existing safetensors/sessions load unchanged. Encode always writes the
  new `blocks` array (and stops writing legacy block keys — the decoder
  keeps reading them forever).
- `architectureSummary` collapses runs of identical specs:
  `2x[7x7 conv, SE+/4, clean_add, ReZero] . 6x[3x3 conv, SE+/4, clean_add, ReZero]`
  — fully-uniform towers must produce **byte-identical summaries to
  today's** (golden-string test), so existing ARCH_EXPERIMENTS.md
  identities stay valid.
- `weightTensorPlan`: per-block tensor shapes derive from each BlockSpec
  (names already carry block indices — `block3_conv1_weights` etc., so
  loader/saver alignment is mechanical). Loader shape validation already
  compares against the embedded config; it inherits correctness.

### Consumers to walk (the silent-desync checklist)

1. `ChessNetwork` graph builders — block loop reads `arch.blocks[i]`
   instead of uniform fields (stem unchanged).
2. `NetworkWeightAnalyzer` — **fanIn must use each block's own kernel
   area** (this was already a real bug once, build ≤1566; per-block specs
   make the uniform assumption wrong again rather than just stale).
3. Build-New-Model UI — per-block editor: list of block rows with
   add/remove/duplicate-row, plus a "make all like this" convenience;
   validation per row (odd kernels, SE ratio divides channels, rate
   bounds). The ReZero α-init mismatch check (commit c3eb430) becomes
   per-block.
4. `--probe-model` / CLI paths — decode-only; covered by the fallback,
   verify with a legacy checkpoint fixture.
5. `parameterCount` — sum per-block.
6. ARCH_EXPERIMENTS.md convention note: heterogeneous arches list the
   collapsed summary as their identity line.

### v2 sketch (not in this plan's scope)

Per-block `channels` with automatic 1×1 projection on the skip at width
boundaries (WRN-style stage transitions, no spatial downsampling). Also
DenseNet-style stem-tap concat into the policy head — separate experiment,
separate plan.

---

## Phasing

- **Phase A (Feature 2 first):** BlockSpec array + decode fallback +
  summary collapsing + builders + analyzer + UI + tests. Feature 2 first
  because Feature 1's axes live ON BlockSpec — landing dropout first would
  mean migrating its keys twice.
- **Phase B (Feature 1):** dropout axes on BlockSpec, training-graph node,
  RNG approach verified against SDK, UI fields, tests.
- One build + commit per phase (multi-phase convention).

## Validation

1. **Round-trip identity:** every existing saved model/session decodes;
   a uniform tower encodes → decodes → re-encodes byte-stable; legacy
   uniform keys decode to the expansion. (XCTest with a captured legacy
   config JSON fixture.)
2. **Golden summaries:** uniform towers produce today's exact
   architectureSummary strings; a mixed tower produces the documented
   collapsed form. (XCTest.)
3. **Weight-plan alignment:** mixed-kernel tower's weightTensorPlan shapes
   match the built graph's variables 1:1 (existing verification machinery
   should catch this; add an explicit mixed-arch test).
4. **Dropout semantics:** training-mode graph with rate r produces masked
   activations with mean ≈ unmasked (inverted scaling, statistical test on
   a small graph); inference-mode graph for the same config is bit-exact
   with rate 0. Mask varies across steps. (XCTest on a tiny graph.)
5. **Forward-pass save verification** (the bit-exact save check) passes
   for a mixed-arch, dropout-enabled config — confirming inference graphs
   are dropout-free.
6. Build-New-Model creates a mixed tower (e.g. 2×7×7 + 6×3×3 + channel
   dropout 0.1) and Play-and-Train runs it; `[ARCH]` line shows the
   collapsed summary.
7. All existing tests pass unmodified.
