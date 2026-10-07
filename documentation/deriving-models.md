# Deriving models (`--derive-model`)

`--derive-model` makes a new model file from an existing one. It applies one or more
*derive operations*: architecture edits that never change a tensor's name or shape.
Each operation re-initializes only the tensors it declares. Every other tensor is
copied bit-exact. The main use is a paired copy for an A/B: the same fresh net with
one init choice flipped, so the two arms differ in that choice and nothing else.

These operations are a pure file transform: no GPU, no network build, no window. The
one derive that changes the tensor layout, the [graft](#grafting-onto-another-architecture-graft-to)
(`--graft-to`), builds the target network once to read its fresh tensors (no forward
pass, no training). Both are safe to run next to a training job.

## Usage

```
DrewsChessMachine --derive-model --from <model.safetensors> <operation> <value> \
    [--group <index>]... [--init-seed <u64>] --out <new.safetensors>
DrewsChessMachine --derive-model --help      # lists every operation this build supports
```

- `--from`: the source model. It must be a model file (a fresh net or a champion), not a
  trainer-state file. Files with `opt.*.velocity` tensors or `trainer_*` schedule
  metadata are refused, because re-initialized weights would not match the saved
  optimizer state. An operation that rewrites tensors (`--set-se-beta-init`,
  `--set-rezero-alpha-init`, and every init-option operation) also needs an untrained source,
  because the rewrite would reset learned weights while the new file kept the source's
  training step and lineage. Any positive evidence of training refuses the source: a
  recorded `training_step` above zero; a lineage record whose `cum_trainer_step` is above
  zero; a lineage parent that stated a trainer step above zero (a graft or champion copy
  of a file written before lineage, whose own total is unrecorded); or a `graft` in the
  derivation history whose `source_training_step` is above zero (a graft writes no
  `training_step`, and an architecture-only derive after it keeps none). A
  `training_step` or `source_training_step` that is not a non-negative integer is refused
  as well. One case cannot be detected: a file written before lineage that was trained
  but states no `training_step` and no derivation history reads as untrained. Operations
  that change no tensor (`--set-activation`, `--set-se-activation`, the per-site
  activation setters such as `--set-value-head-fc1-hidden-activation`,
  `--set-rezero-alpha-cap`, `--set-policy-tail-precision`) work on a champion too.
- `--out`: the destination. It must end in `.safetensors`, must not exist, and must
  differ from `--from`. Nothing is ever overwritten.
- `--group <index>`: a 0-based block-group index (repeatable). It narrows operations that
  accept it. Without it, an operation applies to every group it can apply to.
- `--init-seed <u64>`: the init seed of every operation that draws weights
  (`--set-se-beta-init glorot`, and `he` for `--set-skip-projection-init`,
  `--set-policy-head-final-init` and `--set-value-head-final-init`). Without it a seed is drawn. Either way the seed and the
  init scheme (`dcm-init-1`) are written into that operation's `derivation_history`
  record, so the derive can be repeated exactly. Given with no weight-drawing operation,
  it is refused.
- The new file's path is printed on stdout. The rewritten tensors are listed on stderr
  and in the session log as `[DERIVE]` lines.

## Operations

| flag | values | changes | rewrites |
|---|---|---|---|
| `--set-se-beta-init` | `glorot` \| `zero` | `block_groups[].se_beta_init` on `scale_and_bias` groups | β half of each affected block's SE FC2: `blocks.<i>.se_scalebias.fc2.weight` rows C..2C−1 (on-disk `[2C, r]` layout) and `blocks.<i>.se_scalebias.fc2.bias` C..2C−1 |
| `--set-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | every architecture-level site activation the topology has (`stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_fc1_hidden_activation`; an absent site stays `does_not_apply`) and every `block_groups[].activation_function` (never `se_activation`) | none (activations have no parameters) |
| `--set-se-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `block_groups[].se_activation` on groups with an SE block | none (activations have no parameters) |
| `--set-stem-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `stem_activation` (a post-activation first block group only) | none (activations have no parameters) |
| `--set-tower-end-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `tower_end_activation` (a pre-activation last block group only) | none (activations have no parameters) |
| `--set-feature-skip-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `feature_skip_activation` (a compress fusion node routed to a head only) | none (activations have no parameters) |
| `--set-policy-head-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `policy_head_activation` (`intermediate_conv` / `fc_bottleneck` only) | none (activations have no parameters) |
| `--set-value-head-conv-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `value_head_conv_activation` | none (activations have no parameters) |
| `--set-value-head-fc1-hidden-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `value_head_fc1_hidden_activation` | none (activations have no parameters) |
| `--set-rezero-alpha-init` | a number `>= 0` | `block_groups[].rezero_alpha_init` on groups with ReZero | `blocks.<i>.rezero_alpha` (the one-element α tensor) of every block in an affected group, set to exactly the value |
| `--set-rezero-alpha-cap` | a number `> 0` | `block_groups[].rezero_alpha_cap` on groups with ReZero | none (the cap has no parameters) |
| `--set-neutral-init` | `all` | the Neutral init set (below) on every block group and head; not the draw prior | the tensors of every option the set changes |
| `--set-se-gamma-bias-init` | a finite number | `block_groups[].se_gamma_bias_init` on groups with an SE block | γ half of each affected block's SE FC2 bias (`blocks.<i>.se_attenuate.fc2.bias`, or `blocks.<i>.se_scalebias.fc2.bias` 0..C−1), set to the value |
| `--set-branch-output-init` | `standard` \| `zero_last_bn_gamma` | `block_groups[].branch_output_init` on post-activation groups | `blocks.<i>.bn2.weight` of every block in an affected group: 0 for `zero_last_bn_gamma`, 1 for `standard` |
| `--set-skip-projection-init` | `he` \| `identity_like` | `block_groups[].skip_projection_init` on groups that change width | each affected `blocks.<i>.skip_proj.weight`: the identity-like constant, or a He draw |
| `--set-policy-head-final-init` | `he` \| `zero` | `policy_head_final_init` | `policy.conv.weight` (`simple_conv`, `intermediate_conv`) or `policy.fc.weight` (`fc_bottleneck`): zeros, or a He draw |
| `--set-value-head-final-init` | `he` \| `zero` | `value_head_final_init` | `value.wdl_fc2.weight` (or `value.scalar_fc2.weight`): zeros, or a He draw; the bias is untouched |
| `--set-value-head-draw-prior` | a probability in (0, 1) | `value_head_draw_prior` (W/D/L head only) | `value.wdl_fc2.bias`, set to `[0, ln(2p/(1−p)), 0]` |
| `--set-policy-tail-precision` | `fp32_from_pre_bn` \| `mixed_final_projection` \| `does_not_apply` | `policy_tail_precision` (format v12): where the policy head leaves the compute dtype for fp32. A bf16 / fp16 model takes one of the first two; `does_not_apply` is the only value an fp32 model holds, and any other combination is refused by the architecture's validation. A value the model already holds is refused ("nothing to derive"). | none (no tensor depends on it) |

`zero` writes exact zeros to the β weights and bias. `glorot` re-draws the β weights
with the graph builder's own seeded per-tensor draw (the tensor's `init/<name>` stream
under the init seed, `WeightInitScheme`) and zeroes the β bias, so the β rows are exactly
those a fresh mint with that init seed has. The γ half is never touched.

`--set-activation` changes the main activation at every site at once: every
architecture-level site the model has (stem, tower end, feature-skip fusion node, policy
pre-block, value conv, value FC1 hidden layer) and each block group's main path and
`activation_gated` merge. A site the topology lacks (the stem of a pre-activation tower,
the tower end of a post-activation one, the policy pre-block of `simple_conv`, the fusion
node without compress fusion) holds `does_not_apply` and keeps it. `does_not_apply` is
not an activation function, so it is refused as a value. It doesn't accept `--group`.
It never changes `se_activation`: a group with an SE block keeps its FC1's activation,
and an SE-less group has no FC1, so its `se_activation` is `does_not_apply` and stays so.

`--set-se-activation` sets `se_activation`, the activation after the SE excitation FC1
(the pooled `C → C/r` bottleneck), on groups with an SE block: all of them, or those named
by `--group`. A group without an SE block named by `--group` is refused. It exists for the
issue #2 A/B: ReLU vs leaky ReLU at FC1 only, where dead units were measured, without
paying for leaky ReLU on every conv.

`--set-rezero-alpha-init` sets the ReZero α init on groups with ReZero (all of them, or
those named by `--group`) and writes that exact value into every block's α tensor in
those groups — the init is a tensor value, so changing only the field would describe a
net that was never built. `0` is the ReZero paper's init: every residual branch starts
off and α learns from step 1 (see `rezero-alpha-clamp.md`). The cap is left alone; a
legacy group's cap is its old init, so `--set-rezero-alpha-init 0` alone gives a zero
start under the old bound. A request whose value every selected group already states is
refused even when the file's α tensors differ from it; a trained source is refused before
that point (see `--from` above).

`--set-rezero-alpha-cap` sets `rezero_alpha_cap`, the asymptote `C` of the forward soft
bound `C·tanh(α/C)`, on groups with ReZero (all, or those named by `--group`). No tensor
changes. Both ReZero operations refuse a group named by `--group` that has no ReZero, and
the derived architecture must validate (cap finite and `> 0`, init finite and `>= 0`).

### Init-neutral options

The last seven operations set the init-neutral options (determinism plan B2.1, decision
D-4). Each starts a part of the network as a no-op, or as close to one as its layer
allows, so an ablation can ask what that part contributes when it starts from nothing.
Each writes exactly the values a fresh build of the target architecture has, from the
same functions the graph builder uses (`WeightInitScheme`, `ChessNetwork`), so a
derived tensor matches a fresh mint of the target.

- `se_gamma_bias_init` (standard `0`, gate σ(0) = 0.5 at step 0): the bias of the SE
  gate's pre-sigmoid half. The neutral value is `ln 9`, a gate of 0.9.
- `branch_output_init: zero_last_bn_gamma` (post-activation groups only): the BN after
  each block's last conv starts with γ = 0, so the residual branch adds nothing at step 0
  and the block passes its skip through. Pre-activation groups refuse it.
- `skip_projection_init: identity_like` (groups that change width only): the 1×1
  projection maps input channel i to output channel i with weight 1 and is 0 elsewhere,
  so the first `min(in, out)` channels pass through unchanged.
- `policy_head_final_init: zero`: the policy's final projection is zero, so every
  logit is its bias and the policy is uniform over legal moves at step 0.
- `value_head_final_init: zero`: the value head's final FC weight is zero, so its
  softmax is its bias prior at step 0.
- `value_head_draw_prior` (standard `0.75`, the bias `[0, ln 6, 0]` every model had
  before the option): the W/D/L head's draw probability at step 0. A scalar head has no
  draw class and requires the standard value.

A zero head final passes no gradient back into the trunk on the first step
(∂L/∂features = Wᵀ·∂L/∂logits = 0), so the first step moves only the head finals; a
zeroed last-BN γ starts moving on the second.

`--set-neutral-init all` applies the Build New Model screen's **Neutral init**
(`NetworkArchitecture.withNeutralInit`): each option above takes its neutral value
wherever its layer exists. The draw prior is not part of the set and is left as it is.
Every neutral value is a constant, so it draws nothing. A He re-draw (moving an option
back to `he`) uses the tensor's own `init/<name>` stream under the init seed.

### Order

Operations run in the order listed above, so `--set-activation X --set-se-activation Y`
gives activation X with an FC1 activation of Y, and passing the same value to both gives
that activation everywhere. The per-site setters run after `--set-activation`, so
`--set-activation X --set-value-head-fc1-hidden-activation Y` gives X everywhere except
the value FC1 hidden layer, which gets Y. `--set-neutral-init` runs before the per-option operations,
so `--set-neutral-init all --set-value-head-final-init he` is the neutral set with the
value head's final layer re-drawn. The SE gate (sigmoid) and the value output (softmax or tanh)
are structural and don't change.

## Examples

```
# Zero-β paired copy of a fresh scale+bias net (every scale_and_bias group):
DrewsChessMachine --derive-model --from fresh.safetensors --set-se-beta-init zero --out fresh-beta0.safetensors

# Only block group 1 (0-based):
DrewsChessMachine --derive-model --from fresh.safetensors --set-se-beta-init zero --group 1 --out fresh-g1-beta0.safetensors

# The same fresh net with leaky ReLU only after the SE FC1 (every group with an SE block):
DrewsChessMachine --derive-model --from fresh.safetensors --set-se-activation leaky_relu --out fresh-se-leaky.safetensors

# Leaky ReLU on the main activation, ReLU kept at the SE FC1:
DrewsChessMachine --derive-model --from fresh.safetensors --set-activation leaky_relu --out fresh-leaky-main.safetensors

# The same fresh net with leaky ReLU at every hidden activation:
DrewsChessMachine --derive-model --from fresh.safetensors --set-activation leaky_relu \
    --set-se-activation leaky_relu --out fresh-leaky.safetensors

# Leaky ReLU heads on a ReLU tower (policy pre-block, value conv, value FC1 hidden):
DrewsChessMachine --derive-model --from fresh.safetensors --set-policy-head-activation leaky_relu \
    --set-value-head-conv-activation leaky_relu --set-value-head-fc1-hidden-activation leaky_relu \
    --out fresh-leaky-heads.safetensors

# Leaky ReLU everywhere except a ReLU value FC1 hidden layer:
DrewsChessMachine --derive-model --from fresh.safetensors --set-activation leaky_relu \
    --set-value-head-fc1-hidden-activation relu --out fresh-leaky-relu-fc1.safetensors

# Zero-initialized ReZero (every α tensor exactly 0) with cap 1.0:
DrewsChessMachine --derive-model --from fresh.safetensors --set-rezero-alpha-init 0 \
    --set-rezero-alpha-cap 1 --out fresh-rz0-cap1.safetensors

# The neutral-init twin of a fresh net (constants only, no seed needed):
DrewsChessMachine --derive-model --from fresh.safetensors --set-neutral-init all --out fresh-neutral.safetensors

# Only the heads zeroed, with an initial draw probability of 0.5:
DrewsChessMachine --derive-model --from fresh.safetensors --set-policy-head-final-init zero \
    --set-value-head-final-init zero --set-value-head-draw-prior 0.5 --out fresh-heads0.safetensors

# Use the derived net as a fixed starting point:
DrewsChessMachine --replay-corpus <corpus> --start-model fresh-beta0.safetensors --parameters parameters.json ...
```

## What the derived file records

- A new `model_id`, `parent_model_id` = the source's `model_id`, and `creator` = `derive-model`.
- `notes`: a one-line summary (source ModelID, file, SHA-256, and the operations).
- `derivation_history`: a JSON array. It holds the source's own history, if the source
  was itself derived, plus one record for this derivation: `model_id`,
  `parent_model_id`, `source_file`, `source_sha256`, `source_format_version`,
  `created_at_unix`, `build`, and `operations`. Each operation records its name,
  arguments, the architecture fields it changes, and the tensors it rewrote. A chain of
  derivations can therefore be traced from the newest file alone.
- `dcm_format_version` = the current version (`ArchitectureFormat.currentVersion`), and the target architecture. A legacy
  source (format v9 or older) is read under the legacy rules (`se_beta_init` → `glorot`
  before v4, `se_activation` → the group's activation before v5 (`does_not_apply` on an
  SE-less group), `rezero_alpha_cap` → the group's `rezero_alpha_init` before v6, every
  init-neutral option → its standard value before v8, before v9 each architecture-level
  site activation → the file's top-level `activation_function` where the topology has the
  site and `does_not_apply` where it does not, and before v10 an SE-less group's
  `se_activation` → `does_not_apply`); the derived file states every field.
- Every other `__metadata__` key of the source is copied verbatim. This includes
  `training_step`, the value-head centering marker, and any `replay_*` provenance.

## Guardrails

- The target architecture must validate. It must also have exactly the source's tensor
  plan (names and shapes). A shape-changing request is refused before anything is
  written.
- After each rewrite, every element outside the operation's declared ranges is checked
  to be bit-identical to the source.
- A request that changes nothing is refused. So is a group out of range, a group whose SE
  style has no β half (`--set-se-beta-init`), a group without an SE block
  (`--set-se-activation`, `--set-se-gamma-bias-init`), a group without ReZero
  (`--set-rezero-alpha-init`, `--set-rezero-alpha-cap`), a group that is not
  post-activation (`--set-branch-output-init`), a group that does not change width
  (`--set-skip-projection-init`), a draw prior on a scalar value head, or
  `does_not_apply` as the value of `--set-activation`, `--set-se-activation` or a per-site
  setter (it marks a site the topology lacks; it is not an activation function), or a
  per-site setter on a site the model's topology lacks (the error names why it is absent,
  e.g. `--set-stem-activation` on a pre-activation tower).
- The output is decoded back through the normal loader before it is written.

## Grafting onto another architecture (`--graft-to`)

```
DrewsChessMachine --derive-model --from <model.safetensors> --graft-to <preset | arch.json> \
    [--graft-map <old=new,...>] [--init-seed <u64>] --out <new.safetensors>
```

A graft makes a model of a **target** architecture (a built-in or saved preset name, or an
architecture JSON file) from the source model. It is the only derive that may change the
tensor layout, and it is not combined with the operations above (graft first, then derive
the result).

- Every target tensor whose name (after `--graft-map`) and on-disk shape match a source
  tensor is **copied** bit-exact, BN running statistics included.
- Every other target tensor is **initialized** with the value a fresh mint of the target
  has under the init seed: drawn conv / FC weights from their `init/<name>` streams
  (`dcm-init-1`), and the graph builder's constants for biases, BN, ReZero α and the W/D/L
  prior. A new BN layer keeps the builder's identity running statistics (mean 0, variance 1);
  the graft does not recalibrate them, and the record says so. `--init-seed` sets the seed;
  without it one is drawn. Either way it is logged and recorded.
- Source tensors with no place in the target are **dropped** and listed.
- A target tensor that a source tensor would fill by name but with a different shape is
  refused rather than silently re-initialized; drop it explicitly with `name=` to
  initialize the target's instead.
- The target's size is checked like a new model's (`ModelSizeGuidance`): an
  `[ARCH] size guidance` line is logged before the build, and a target whose training state
  cannot fit in this Mac's physical memory is refused (exit 97). Every other size is
  allowed. The in-place operations never change a tensor's shape, so they only log the
  line.

`--graft-map` is a comma-separated list:

| entry | meaning |
|---|---|
| `old=new` | rename one source tensor |
| `old.=new.` | rename every source tensor starting with `old.` (both sides end in `.`) |
| `old=` | drop that source tensor |

Every entry must match a source tensor; a renamed tensor must land on a target tensor of
the same shape; two source tensors may not fill one target tensor; a source tensor matched
by two entries is refused.

**Trained sources.** Unlike the in-place operations, a graft accepts a trained source:
warm-starting a larger tower from a trained one is its purpose. Its output claims no
`training_step` of its own (the source's is recorded as `source_training_step` in the
derivation record), and it lists every tensor it initialized, so it claims nothing its
weights do not have. Its lineage continues the source's totals as a new derived run (a
source written before lineage leaves the totals unrecorded; its stated step is kept as the
record's parent step). Because of that record and lineage, a later tensor-rewriting derive
of the grafted file — or of any file derived from it — still sees the trained source and
refuses (see `--from` above).

**What the grafted file records.** A graft writes its own metadata rather than copying the
source's: `model_id`, `parent_model_id`, `creator` = `derive-model`, `notes`, the target
architecture, the lineage record, and the source's value-head centering marker when the
source has one. Its `derivation_history` record has operation `graft`, arguments `target`,
`graft_map`, `init_seed_origin`, `bn_running_stats` and `source_training_step`,
`changed_architecture_fields` = `["*"]`, `rewritten_tensors` = the initialized tensors, and
`copied_tensors` (renames shown as `old -> new`), `dropped_tensors`, `init_seed`,
`init_rule_version` and `per_tensor_init` (each initialized conv / FC weight's draw —
`he_normal`, `glorot_normal`, `glorot_normal_zero_beta` — or its init option's constant —
`identity_like`, `zero` — and `builder_constant` for every other tensor). A target with
non-standard init options gives the tensors the graft initializes those options' values. The records of the
in-place operations carry none of these graft fields.

```
# A fifth block on a trained four-block net, the new block drawn under seed 7:
DrewsChessMachine --derive-model --from trained.safetensors --graft-to my_5block_preset \
    --init-seed 7 --out trained-5block.safetensors

# Insert a new block at index 1 of a three-block net (old blocks 1 and 2 move up):
DrewsChessMachine --derive-model --from net.safetensors --graft-to my_4block_preset \
    --graft-map 'blocks.2.=blocks.3.,blocks.1.=blocks.2.' --out net-inserted.safetensors
```

## Adding an operation

1. Write a type conforming to `DeriveOperation` (in `Persistence/ModelDerivation.swift`)
   with a static `DeriveOperationKind`.
2. Add the kind to `ModelDerivation.operationKinds`.

The CLI flag, `--help`, and the derivation record all come from that list. Add a row to
the table above.
