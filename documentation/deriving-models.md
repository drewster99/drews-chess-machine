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

`zero` writes exact zeros to the β weights and bias. `glorot` re-draws the β weights
from the same Glorot-normal distribution the graph builder uses, and zeroes the β bias.
The γ half is never touched.

## Examples

```
# Zero-β paired copy of a fresh scale+bias net (every scale_and_bias group):
DrewsChessMachine --derive-model --from fresh.safetensors --set-se-beta-init zero --out fresh-beta0.safetensors

# Only block group 1 (0-based):
DrewsChessMachine --derive-model --from fresh.safetensors --set-se-beta-init zero --group 1 --out fresh-g1-beta0.safetensors

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
- `dcm_format_version` = the current version (4), and the target architecture.
- Every other `__metadata__` key of the source is copied verbatim. This includes
  `training_step`, the value-head centering marker, and any `replay_*` provenance.

## Guardrails

- The target architecture must validate. It must also have exactly the source's tensor
  plan (names and shapes). A shape-changing request is refused before anything is
  written.
- After each rewrite, every element outside the operation's declared ranges is checked
  to be bit-identical to the source.
- A request that changes nothing is refused. So is a group out of range, or a group
  whose SE style has no β half.
- The output is decoded back through the normal loader before it is written.

## Adding an operation

1. Write a type conforming to `DeriveOperation` (in `Persistence/ModelDerivation.swift`)
   with a static `DeriveOperationKind`.
2. Add the kind to `ModelDerivation.operationKinds`.

The CLI flag, `--help`, and the derivation record all come from that list. Add a row to
the table above.
