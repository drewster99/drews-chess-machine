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

## Feature 2 — Block groups: heterogeneous tower with per-group widths
## (REVISED 2026-06-12: widths now IN scope per user decision; UI must
## render a graphical view of the whole architecture)

### Design — config model (REVISED 2026-06-12: two-level groups — a
### group is a sequence of typed block runs, not one recipe × count.
### Stride was added and then DROPPED the same day — decision record
### in the BlockSpec bullet; spatial shape is immutably 8×8.)

- **`BlockGroup`** (Codable, Hashable, Sendable) — `count >= 1` + one
  fully-specified block recipe as flat fields (the project's
  flat-schema rule). EVERY block-configurable element lives here (user
  requirement 2026-06-12: with count = 1 groups, any block in the
  tower can differ from any other in any field):
  - `channels` (>= 8; the block's output width)
  - `conv1_kernel_size`, `conv2_kernel_size` (odd, validated)
  - **No stride fields — decision record (2026-06-12):** per-conv
    stride was added to this plan and dropped the same day. Stride's
    payoffs (compute on large inputs, faster receptive-field growth)
    barely apply to an 8×8 board that 12 stacked 3×3 blocks already
    cover several times over, while its cost lands exactly on the
    engine's core: the fully-convolutional policy head emits 76 logits
    PER SQUARE and structurally requires an 8×8 tower output, so any
    downsampling forces an FC policy head. Dropping it deletes the
    per-block (H, W) bookkeeping, the FC-head validation rule, the
    strided skip-projection variant, and the diagram's spatial
    staircase/saturation display. Spatial shape is therefore immutable
    (8×8 everywhere, all convs same-padded); channels are the only
    shape axis. Stride lives in the v2 sketch if a downsampling
    experiment is ever wanted.
  - `se_style`, `se_reduction_ratio` (ratio must divide channels)
  - `use_rezero`, `rezero_alpha_init`
  - `activation_function` (relu/mish/etc — whatever the enum offers)
  - `block_activation_style` (pre/post)
  - `block_skip_merge` (clean_add / activation_gated)
  - `dropout_multiplier` (>= 0; per-block scale on the global
    `DropoutRate` — effective rate = clamp(rate × multiplier); baked as
    a per-block graph constant composed with the rate variable)
- (SIMPLIFIED 2026-06-12 with user approval: the interim two-level
  shape — groups holding multiple typed entries plus a `repeatCount` —
  bought only authoring sugar that a "duplicate group" button
  replicates; it moved to the v2 sketch. "One wide-view block then
  seven local blocks" is simply two groups: `1×[7x7…] -> 7×[3x3…]`.)
- **Tower** = `blockGroups: [BlockGroup]`, ordered input → output.
- **Expansion is the engine's only view**: the config exposes
  `expandedBlocks: [BlockGroup]` — each group repeated `count` times,
  concatenated, every returned element normalized to `count = 1` so
  one element ≡ one block. Graph builders, weightTensorPlan,
  parameterCount, the analyzer, and the diagram's math consume ONLY
  the flat expanded list — groups are an authoring/persistence
  structure, never an engine concept. This kills all per-group
  special-casing: anything that previously keyed on "first block of
  group" now just reads adjacent expanded specs.
- `numBlocks` and `channels` become DERIVED (`expandedBlocks.count`;
  tower-output channels = last expanded block's). Stored legacy
  fields removed from the struct; decode keeps reading them forever
  (below).
- **Stem** outputs the FIRST expanded block's channels
  (`inputPlanes → expandedBlocks[0].channels`; tower-level stem
  kernel, explicit).
- **Transitions are per-block, not per-group**: block i's input width
  `inC` is the previous expanded block's channels (stem output for
  block 0); its output width `outC` is its own channels. A block gets
  the skip projection iff `inC != outC`:
  - branch conv1 maps `inC → outC`; conv2 and everything after run at
    `outC`; BN1 sized `inC`, BN2/SE/α sized `outC`.
  - the skip projection is a **1×1 conv** (`inC → outC`, bias-free,
    He-init, weight-decayed — a true weight matrix), named
    `blockN_skip_proj_weights` — a pure per-square linear remap of the
    feature vector, zero spatial mixing: the minimum repair that makes
    the add well-shaped. Every other block keeps the clean identity
    skip — the gradient highway survives wherever width is constant.
  - with one width this is exactly today's tower; with width steps it
    is the WRN staircase with the resolution axis removed (the board
    is always 8×8).
- **Heads** read the tower-output channels (last expanded block), not
  a global `channels` — policy pre-conv `towerOut → K`, value conv
  `towerOut → valueHeadConvChannels`. Tower-end BN sized `towerOut`.
  Spatial dims never change, so head flatten sizes keep their fixed
  8×8 forms exactly as today.
- **Dropout mask shapes** become per-width: `[N, block.channels, 1, 1]`,
  each built from the input placeholder's batch dim + a channel
  constant (same safe construction; one shape tensor per distinct
  width among the expanded blocks, shared by every block at that
  width).
- **Mixed activation styles compose cleanly** because each style is
  self-contained per block (pre: BN→act→conv→…→bare add; post:
  conv→BN→act→…→activated merge). The tower-end BN rule generalizes:
  `hasTowerEndBN = (LAST expanded block's style == .pre)` — a
  pre-activation tail ends un-normalized/un-activated and needs the
  conditioning BN; a post-activation tail is already conditioned. The
  stem keeps its tower-level kernel + the existing stem-activation
  rule evaluated against the FIRST expanded block's style (a pre-act
  first block defers the first nonlinearity to its own BN→act, exactly
  as today).

### Identity / persistence

- **Decode fallback**: legacy configs (uniform `num_blocks`,
  `channels`, `block_conv*_kernel_size`, …) decode to a single
  BlockGroup (repeatCount 1) with one entry: the legacy spec ×
  `num_blocks`. Every existing safetensors/session loads unchanged.
  Encode writes only `block_groups`; the decoder reads both forever.
  Encode/decode preserves the AUTHORED structure (group names,
  repeats, entry boundaries) — it never normalizes to the expansion.
- `architectureSummary` renders EVERY attribute of EVERY group
  explicitly — no silent defaults (user direction 2026-06-12: a reader
  should never need to know a default to read a summary). Per group:
  count, both conv kernels, channels, SE style/ratio, activation
  function + style, skip merge, ReZero α-init, dropout multiplier,
  with `->` between groups (skip projections are implied by any
  adjacent width change in the expansion, never written). e.g. the
  user's example tower —
  `1x[7x7+3x3 @128, SE+/4, relu/pre, clean_add, ReZero(0.20), drop*1.0]
  -> 7x[3x3+3x3 @128, SE+/4, relu/pre, clean_add, ReZero(0.20), drop*1.0]`.
  Uniform towers render in the SAME explicit form — this deliberately
  changes today's summary strings. The golden-string test pins the new
  explicit form for uniform, multi-entry, and repeated-group towers,
  and ARCH_EXPERIMENTS.md gets a one-time mapping note (old line → new
  explicit line for each known arch) in the same commit. Identity
  continuity is carried by the embedded config itself — safetensors
  files embed the full architecture and have no arch_hash; the legacy
  `.dcmmodel` hash is a read-path preset lookup only — so no stored
  artifact breaks.
- `weightTensorPlan` derives per-block shapes by walking
  `expandedBlocks` (names by flat block index, exactly today's
  `blockN_*` scheme) + the projection tensors where the per-block rule
  fires. Loader shape validation inherits correctness from the
  embedded config.

### Consumers to walk (silent-desync checklist)

1. `ChessNetwork` builders — walk `expandedBlocks`; per-block
   (inC, outC, spec) threading; projection wherever the per-block rule
   fires; per-width dropout mask shapes; tower-end BN + heads at
   towerOut.
2. `NetworkWeightAnalyzer` — fanIn from each block's own kernel AND
   channels (the uniform assumption becomes wrong, not just stale);
   new projection tensors get sections.
3. Build-New-Model UI — TWO-LEVEL editor: groups
   (add/remove/duplicate/reorder/rename, repeatCount) and entries
   within a group (spec fields + count; add/remove/duplicate/reorder);
   validation per spec (odd kernels, ratio divides channels,
   counts/repeats >= 1) + the **architecture diagram** (next section).
   ReZero α-init check flags against 1/√(expanded total blocks).
4. `--probe-model` / CLI / checkpoint loaders — decode-only; legacy
   fixture test.
5. `parameterCount` — sum over `expandedBlocks` + projections.
6. ARCH_EXPERIMENTS.md identity convention: the explicit grouped
   summary is the arch line.

### Architecture diagram (Build-New-Model UI)

A read-only, live-updating schematic of the ENTIRE network, rendered
from the draft config (updates as the user edits groups):

- Vertical flow: input planes (encoding name + plane count) → stem
  (kernel, in→out channels) → each block group as a bracketed segment
  (name + repeatCount badge) containing one cell stack per entry,
  every cell printing its FULL configuration (count badge, both
  kernels, channels, SE style/ratio, activation function + style,
  skip merge, ReZero α, dropout multiplier — no "differs from default"
  badges, nothing implied; same no-silent-defaults rule as the
  summary) → width-transition markers (1×1 projection) wherever the
  expansion fires the per-block projection rule, including inside a
  group → tower-end BN → policy head and value head side by side
  (their internal stages summarized).
- **Box width proportional to channel count** so the WRN staircase is
  visible at a glance; per-section parameter counts and the grand
  total displayed on each segment.
- Pure SwiftUI (no Canvas dependency needed beyond simple shapes),
  monospaced digits, light/dark. One View struct per file:
  `ArchitectureDiagramView.swift` + small cell subviews.
- Also shown read-only for the CURRENT champion (from its embedded
  config) — same component, two call sites — so "what am I running"
  and "what am I about to build" use one renderer.

### v2 sketch (still out of scope)

DenseNet-style stem-tap concat into the policy head;
`dropout_granularity = unit`; per-conv stride (added 2026-06-12,
dropped the same day — see the decision record in the BlockGroup
bullet; would reintroduce per-block (H, W) bookkeeping, the FC-policy
-head requirement, and the strided skip projection); two-level groups
(typed entries + repeatCount per group — authoring sugar dropped
2026-06-12 in favor of one recipe + count per group).

### Builder refactor contracts (Phase A, decided 2026-06-12)

The five contracts the `ChessNetwork.swift` rewrite must honor —
decided up front because each silently desyncs if improvised:

1. **Tensor order.** The trainables append order inside a block stays
   `(pre: bn1, conv1, bn2, conv2, SE, [rezero]; post: conv1, bn1,
   conv2, bn2, SE, [rezero])`, and `blockN_skip_proj_weights` (plan
   name `blocks.N.skip_proj.weight`) appends LAST within its block,
   after the rezero α — in both the builder and `weightTensorPlan`,
   which mirror each other tensor-for-tensor. Uniform towers produce
   no projections, so existing checkpoints keep their exact layout.
2. **What the projection consumes.** Pre-activation blocks: the skip
   projection reads the SHARED pre-activation (the BN1→act output the
   branch also reads — He et al. v2 convention), so both paths see the
   same normalized input at a transition. Post-activation blocks: the
   raw block input (v1 convention). Sizing: BN1 is `inC`; conv1 maps
   `inC → outC`; BN2/SE/α and conv2 run at `outC`.
3. **Projection hygiene.** Bias-free 1×1, He init with fan-in = inC,
   weight-decayed (a true weight matrix), built for training and
   inference graphs alike.
4. **Per-width dropout mask cache.** One mask shape tensor per
   distinct width among the expanded blocks (batch dim from the input
   placeholder + channel constant — the autodiff-safe construction
   from Feature 1), shared by every block at that width. Per-block
   `dropout_multiplier` composes as a baked constant with the live
   rate variable, clamped to [0, 0.95].
5. **Per-tensor fan-in.** Weight init and `NetworkWeightAnalyzer`
   derive fan-in from each tensor's OWN shape (kernel² × its real
   inC), never from tower-level fields.

---

## Phasing (revised 2026-06-12 — Feature 1 SHIPPED first as a global
## rate without per-block structure; commit eacced3)

- **Phase A (config + engine):** BlockSpec/BlockGroupEntry/BlockGroup
  model + expansion + decode fallbacks + explicit summary + ChessNetwork
  builders (flat expanded walk, per-block projections, per-width
  dropout shapes, heads at towerOut) + weightTensorPlan +
  parameterCount + analyzer + tests (round-trip with authored-structure
  preservation, golden summaries, mixed-arch weight-plan alignment,
  transition gradient flow).
- **Phase B (UI):** Build-New-Model group editor + ArchitectureDiagramView
  (draft + current-champion call sites).
- One build + commit per phase (multi-phase convention).

## Validation

1. **Round-trip identity:** every existing saved model/session decodes;
   any tower encodes → decodes → re-encodes byte-stable with its
   AUTHORED group structure (names, repeats, entry boundaries) intact —
   never normalized to the expansion; legacy uniform keys decode to the
   single-group/single-entry form. (XCTest with a captured legacy
   config JSON fixture.)
2. **Golden summaries (explicit form):** uniform and mixed towers both
   produce the documented fully-explicit strings (XCTest golden
   strings); the ARCH_EXPERIMENTS.md old→new mapping note lands in the
   same commit.
3. **Uniform-tower network identity:** a single-group, single-entry
   config carrying today's values builds a graph with the identical op
   sequence,
   identical weight tensor names/shapes/init scheme, and identical
   parameterCount; loading a pre-change checkpoint passes its embedded
   bit-exact forward-pass verification under the new builders. (XCTest
   + the existing save-verification machinery — this is the hard "same
   network as today" guarantee.)
4. **Weight-plan alignment:** mixed-kernel tower's weightTensorPlan shapes
   match the built graph's variables 1:1 (existing verification machinery
   should catch this; add an explicit mixed-arch test).
5. **Dropout semantics:** training-mode graph with rate r produces masked
   activations with mean ≈ unmasked (inverted scaling, statistical test on
   a small graph); inference-mode graph for the same config is bit-exact
   with rate 0. Mask varies across steps. (XCTest on a tiny graph.)
6. **Forward-pass save verification** (the bit-exact save check) passes
   for a mixed-arch, dropout-enabled config — confirming inference graphs
   are dropout-free.
7. Build-New-Model creates a mixed tower (e.g. one group of
   1×[7×7+3×3] + 7×[3×3+3×3] with a width step, channel dropout 0.1)
   and Play-and-Train runs it; `[ARCH]` line shows the explicit
   grouped summary.
8. All existing tests pass unmodified.
