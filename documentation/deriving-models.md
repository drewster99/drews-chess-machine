# Deriving models (`--derive-model`)

`--derive-model` makes a new model file from an existing one. It applies one or more
*derive operations*: architecture edits that never change a tensor's name or shape.
Each operation re-initializes only the tensors it declares. Every other tensor is
copied bit-exact. The main use is a paired copy for an A/B: the same fresh net with
one init choice flipped, so the two arms differ in that choice and nothing else.

It is a pure file transform: no GPU, no network build, no window. It is safe to run
next to a training job.

## Usage

```
DrewsChessMachine --derive-model --from <model.safetensors> <operation> <value> \
    [--group <index>]... --out <new.safetensors>
DrewsChessMachine --derive-model --help      # lists every operation this build supports
```

- `--from`: the source model. It must be a model file (a fresh net or a champion), not a
  trainer-state file. Files with `opt.*.velocity` tensors or `trainer_*` schedule
  metadata are refused, because re-initialized weights would not match the saved
  optimizer state.
- `--out`: the destination. It must end in `.safetensors`, must not exist, and must
  differ from `--from`. Nothing is ever overwritten.
- `--group <index>`: a 0-based block-group index (repeatable). It narrows operations that
  accept it. Without it, an operation applies to every group it can apply to.
- The new file's path is printed on stdout. The rewritten tensors are listed on stderr
  and in the session log as `[DERIVE]` lines.

## Operations

| flag | values | changes | rewrites |
|---|---|---|---|
| `--set-se-beta-init` | `glorot` \| `zero` | `block_groups[].se_beta_init` on `scale_and_bias` groups | β half of each affected block's SE FC2: `blocks.<i>.se_scalebias.fc2.weight` rows C..2C−1 (on-disk `[2C, r]` layout) and `blocks.<i>.se_scalebias.fc2.bias` C..2C−1 |
| `--set-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `activation_function` and every `block_groups[].activation_function`; `block_groups[].se_activation` on SE-less groups only | none (activations have no parameters) |
| `--set-se-activation` | `relu` \| `silu` \| `gelu` \| `leaky_relu` | `block_groups[].se_activation` on groups with an SE block | none (activations have no parameters) |

`zero` writes exact zeros to the β weights and bias. `glorot` re-draws the β weights
from the same Glorot-normal distribution the graph builder uses, and zeroes the β bias.
The γ half is never touched.

`--set-activation` changes the main activation at every site at once: stem, tower end,
both heads, and each block group's main path and `activation_gated` merge. It doesn't
accept `--group`. It leaves the SE FC1 activation (`se_activation`) of every group with an
SE block alone. On an SE-less group `se_activation` has no effect and must equal the
group's activation, so it changes along with it there.

`--set-se-activation` sets `se_activation`, the activation after the SE excitation FC1
(the pooled `C → C/r` bottleneck), on groups with an SE block: all of them, or those named
by `--group`. A group without an SE block named by `--group` is refused. It exists for the
issue #2 A/B: ReLU vs leaky ReLU at FC1 only, where dead units were measured, without
paying for leaky ReLU on every conv.

Operations run in the order listed above, so `--set-activation X --set-se-activation Y`
gives activation X with an FC1 activation of Y, and passing the same value to both gives
that activation everywhere. The SE gate (sigmoid) and the value output (softmax or tanh)
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
- `dcm_format_version` = the current version (5), and the target architecture. A legacy
  source (format v4 or older) is read under the legacy rules (`se_beta_init` → `glorot`,
  `se_activation` → the group's activation); the derived file states every field.
- Every other `__metadata__` key of the source is copied verbatim. This includes
  `training_step`, the value-head centering marker, and any `replay_*` provenance.

## Guardrails

- The target architecture must validate. It must also have exactly the source's tensor
  plan (names and shapes). A shape-changing request is refused before anything is
  written.
- After each rewrite, every element outside the operation's declared ranges is checked
  to be bit-identical to the source.
- A request that changes nothing is refused. So is a group out of range, a group whose SE
  style has no β half (`--set-se-beta-init`), or a group without an SE block
  (`--set-se-activation`).
- The output is decoded back through the normal loader before it is written.

## Adding an operation

1. Write a type conforming to `DeriveOperation` (in `Persistence/ModelDerivation.swift`)
   with a static `DeriveOperationKind`.
2. Add the kind to `ModelDerivation.operationKinds`.

The CLI flag, `--help`, and the derivation record all come from that list. Add a row to
the table above.
