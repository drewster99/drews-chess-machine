# Per-site activations plan: the stem, tower end, feature-skip fusion, policy head and value head each choose their own activation

Status (2026-10-05): **PLAN ONLY.** Nothing here is implemented.
- Independent review (2026-10-05): **concurred after three passes.** First pass: the architecture direction was approved. Six must-fix items and nine clarifications were raised, and each is listed with what was done about it in **Review reconciliation** at the end.
- Every `file:line` was checked against `main` at `dc6a4785`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/` (project folder), `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/`, `experiments/` or `scripts/`.
- Real-file evidence comes from header-only reads of `~/Library/Application Support/DrewsChessMachine/Models/*.safetensors` and `Sessions/*/champion.safetensors`.
- Modelled end to end on how `se_activation` (format v5, issue #2, commit `de0f22be`) was added: `documentation/plans-active/RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §18, `SEActivationTests`.

**The request.** Make the activation function configurable per site. Today a block group's main path (`block_groups[].activation_function`) and its SE FC1 (`block_groups[].se_activation`) are already independent. One top-level field, `activation_function` (`NetworkArchitecture.activationFunction`, `Network/NetworkArchitecture.swift:1083-1086`), drives every other hidden activation at once.

**What this plan does.**
- Splits the top-level field into one field per site: stem, tower end, feature-skip fusion node, policy head, value-head conv and value-head hidden layer.
- Bumps the architecture format to v9. Every older file resolves each new field to its own `activation_function`, the value it trained with, so every existing model builds the identical graph.
- Adds per-site `--derive-model` setters, Build New Model pickers, per-site layer-health tagging and per-site Python mirrors.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth;
- no silent defaults, no fallbacks, no `try?`, no force unwraps;
- no migration code: older files are decoded under the format-version gate the way the project already decodes them, and are never rewritten;
- tests are never modified or deleted without the owner's approval (every needed edit is listed under **Owner decisions**, OD-6);
- one SwiftUI `View` per file; no helper `some View` properties (new UI is a child `View` struct in its own file);
- no app, test or CLI runs while a training run is live (validation waits for an idle machine).

---

## Summary

| # | Item | Phase |
|---|---|---|
| 1 | Six architecture-level site fields replace the one top-level `activation_function`: `stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_hidden_activation` | P1 |
| 2 | Architecture format v9. Older files resolve every new field to their own `activation_function`. A v9 file missing one is a load error. | P1 |
| 3 | `ChessNetwork` reads each site's own field. The tower-level `activation(_:_:arch:name:)` overload is deleted. | P1 |
| 4 | `LayerHealth` tags each BN site and the value hidden layer with that site's own activation | P1 |
| 5 | `--set-activation` keeps meaning "this activation at every hidden site" (all six fields plus every group) | P1 |
| 6 | Six per-site `--derive-model` setters (`--set-stem-activation`, …) | P2 |
| 7 | Build New Model: one picker per site, disabled where the site does not exist, plus "Use for every activation". The diagram shows each site's activation. | P3 |
| 8 | Python mirrors: `scripts/dcm_arch.py` gate + `site_activations`; site checks in the reusable forward-pass readers (`relu_inputs.py`, `fwd16.py`, `units.py`); frozen scripts documented as ReLU-only | P1 (P1 is the first commit that writes v9) |
| 9 | Docs: `deriving-models.md`, `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20, CLAUDE.md (OD-7), CHANGELOG | P4 (P1/P2 keep `deriving-models.md` in step) |

No tensor, parameter count, weight-init draw, model ID, content hash or default changes. Presets and new models stay ReLU everywhere.

---

## Today (verified)

### The architecture-level activation sites

Every one of these reads `arch.activationFunction` through the private overload at `Network/ChessNetwork.swift:2364-2372`.

| Site | Graph op | Code | Applied to | Exists when | BN site in `LayerHealth` |
|---|---|---|---|---|---|
| Stem | `stem_act` | `ChessNetwork.swift:731-733` | stem BN output | first group is post-activation (`hasStemActivation`, `NetworkArchitecture.swift:1577-1584`) | `stem.bn` (`Training/LayerHealth.swift:169-172`) |
| Tower end | `tower_final_act` | `ChessNetwork.swift:884-896` (act at `:895`) | tower-end BN output | last group is pre-activation (`hasTowerEndBN`, `:1571-1576`) | `tower_final_bn` (`:199-202`) |
| Feature-skip fusion | `feature_skip_act` | `ChessNetwork.swift:933` | `feature_skip_bn` output (compress node) | `featureSkipUsesCompressNode` (`NetworkArchitecture.swift:1460-1464`) | `feature_skip.bn` (`:203-206`) |
| Policy pre-block | `policy_pre_act` | `ChessNetwork.swift:3206` (`intermediate_conv`), `:3258` (`fc_bottleneck`) | `policy_pre_bn` output | `policy_head_style != simple_conv` | `policy.pre_bn` (`:207-213`) |
| Value conv | `value_act` | `ChessNetwork.swift:3467` | `value_bn` output | always | `value.bn` (`:214-215`) |
| Value hidden (FC1) | `value_fc1_act` | `ChessNetwork.swift:3495` | `value_fc1` + bias | always | hidden layer `value.fc1.weight` (`:242-252`) |

The brief listed five sites. The feature-skip fusion node (`ChessNetwork.swift:913-935`) is a sixth that also reads the top-level field. Leaving it there would give the renamed stem field two meanings, so it gets its own field (OD-2).

The other activations are already per-group or structural, and this plan does not touch them:
- the block main path and `activation_gated` merge (`ChessNetwork.swift:2856`, `:2865`, `:2878`, `:2991`: `spec.activationFunction`);
- the SE FC1 (`:3051`: `spec.seActivation`);
- the SE gate (sigmoid) and the value output (softmax / tanh).

### Who else reads the top-level field

| Reader | Where |
|---|---|
| Format decode / encode / uniform-tower legacy expansion | `NetworkArchitecture.swift:1313`, `:1403`, `:1364-1394` |
| Memberwise init / uniform convenience init | `:1135-1173` / `:1177-1253` (`:1214`, `:1226`, `:1235`) |
| Summary clause ` . act <fn>` | `:1924` |
| `--set-activation` | `Persistence/ModelDerivation.swift:738-806` (`:788-798`) |
| Build New Model "Tower-level activation" picker | `App/UpperContentView/BuildNewModelView.swift:93`; model field `BuildNewModelModel.swift:51`, `:103`, `:135`, `:158` |
| Diagram stem and tower-end lines | `App/UpperContentView/ArchitectureDiagramView.swift:53`, `:68` |
| Layer health | `Training/LayerHealth.swift:172`, `:201`, `:205`, `:212`, `:215`, `:251` |
| Tests | 13 lines in 4 files (OD-6) |

`ActivationFunction` (`NetworkArchitecture.swift:221-244`) has `relu`, `silu`, `gelu`, `leaky_relu`. Leaky ReLU has a fixed slope, `ActivationFunction.leakyReLUNegativeSlope = 0.01` (`:243`), applied by `graph.leakyReLU(with:alpha:)` (`ChessNetwork.swift:2385-2386`). The slope is a constant, not a field, and this plan keeps it that way.

### Census of saved models (2026-10-05; 4,377 headers)

| Format | Top-level / group activation | First / last group style | Policy head | Feature-skip fusion | Files |
|---|---|---|---|---|---|
| v3 | relu / relu | pre / pre | intermediate_conv | none | 3,605 |
| v3 | relu / relu | pre / pre | simple_conv | none | 271 |
| v3 | relu / relu | post / post | simple_conv | none | 27 |
| v3 | gelu / gelu | pre / pre | intermediate_conv | none | 4 |
| v4 | relu / relu | pre / pre | intermediate_conv | none | 37 |
| v4 | leaky_relu / leaky_relu | pre / pre | intermediate_conv | none | 1 |
| v5 | relu / relu | pre / pre | intermediate_conv | none | 315 |
| v6 | relu / relu | pre / pre | intermediate_conv | none | 35 |
| v8 | relu / relu | pre / pre | intermediate_conv | none | 82 |

What this means:
- In every file the top-level value equals every group's value.
- No file has a compress fusion node.
- 4,350 files have no stem activation (pre-activation first group).
- 27 files have no tower-end activation (all are post-activation v3, simple_conv).
- 298 files have no policy pre-block.
- The gelu and leaky files show why the legacy resolution must copy the file's own value, never `relu`.

Evidence files for validation (all in `Models/`):

| File | Format | Covers |
|---|---|---|
| `20261005-r7b24-fresh.safetensors` | v8 | current-format pre-act, intermediate_conv |
| `20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors` | v5 | se_activation era |
| `20261001-test_SE_scale+bias-leaky-fresh.safetensors` | v4 | leaky_relu at the top level |
| `20260630-002027-20260630-3-T97X-manual.safetensors` | v3 | gelu at the top level |
| `20260627-v3_8block_3x3-replay-latest.safetensors` | v3 | post-act: stem act present, no tower end, simple_conv |
| `20260630-000533-20260630-1-YkKk-manual.safetensors` | v3 | pre-act simple_conv |

---

# Part D — Design

## D1. Fields and names

| JSON key | Swift property (`NetworkArchitecture`) | Graph op it drives | Site exists when |
|---|---|---|---|
| `stem_activation` | `stemActivation` | `stem_act` | first group is `post` |
| `tower_end_activation` | `towerEndActivation` | `tower_final_act` | last group is `pre` |
| `feature_skip_activation` | `featureSkipActivation` | `feature_skip_act` | `featureSkipUsesCompressNode` |
| `policy_head_activation` | `policyHeadActivation` | `policy_pre_act` | `policy_head_style` is `intermediate_conv` or `fc_bottleneck` |
| `value_head_conv_activation` | `valueHeadConvActivation` | `value_act` | always |
| `value_head_hidden_activation` | `valueHeadHiddenActivation` | `value_fc1_act` | always |

Why these names:
- **`stem_activation`** replaces the top-level `activation_function` (OD-1). After the split the old key would mean "the stem activation, which every model on disk but 27 does not have". A key whose meaning narrows silently is exactly what the brief rules out. Renaming means:
  - an old reader (Python, a person reading JSON, an older build) can never mistake a v9 value for the old tower-wide one;
  - the Swift rename makes the compiler list every reader. Four test files would otherwise keep compiling with a different meaning: two tests silently test nothing and one fails at run time (OD-6).
- **`tower_end_activation`** matches the code's `hasTowerEndBN` and the UI's "tower-end BN". The graph op stays `tower_final_act`; graph op names never change in this plan.
- **`feature_skip_activation`** sits with the other `feature_skip_*` keys (OD-2).
- **`policy_head_activation`** sits with `policy_head_style` / `policy_head_final_init`. `policy_pre_activation` was rejected: "pre-activation" already means the block style.
- **`value_head_conv_activation`** pairs with `value_head_conv_channels`, and **`value_head_hidden_activation`** pairs with `value_head_hidden_units`, the width of the same FC1 layer (OD-3; the alternative is `value_head_fc1_activation`).

## D2. Types

- `ActivationFunction` is unchanged: same four cases, same raw values, and `leakyReLUNegativeSlope` stays a fixed constant shared by every site. Its doc comment (`NetworkArchitecture.swift:221-229`) is rewritten to list the per-site fields.
- Each new field is a non-optional `ActivationFunction` (see D3 for why not Optional).
- New `enum ArchitectureActivationSite: CaseIterable, Hashable, Sendable`, in graph build order: `stem`, `towerEnd`, `featureSkipFusion`, `policyHead`, `valueHeadConv`, `valueHeadHidden`. It follows the `InitOptionField` precedent (`NetworkArchitecture.swift:408-452`). Each case provides:
  - `jsonKey`, read from `NetworkArchitecture.CodingKeys` (the single source of every key);
  - `summaryLabel` (`stem`, `tower_end`, `fusion`, `policy`, `value_conv`, `value_hidden`);
  - `displayName` (the Build screen picker label);
  - `absentReason` (help text where the site does not exist).
- New `NetworkArchitecture` members. These are the single rules every consumer reads; none duplicates a site condition:
  - `func hasActivationSite(_:) -> Bool`, built from the existing `hasStemActivation`, `hasTowerEndBN`, `featureSkipUsesCompressNode` and `policyHeadStyle`. It never expands the tower, so the Build screen can call it on any draft.
  - `func activation(at:) -> ActivationFunction` and `mutating func setActivation(_:at:)`: a switch over the six stored properties, which stay the single source of truth.
  - `mutating func setActivationAtEveryArchitectureSite(_:)`: sets all six fields, including sites the topology lacks (D3).
  - `mutating func setMainActivationEverywhere(_:)`: `setActivationAtEveryArchitectureSite` plus every group's `BlockGroup.setActivationFunction` (`:640-653`). This is the `--set-activation` rule, shared by the derive operation and the Build screen's "Use for every activation", so an edit made either way gives the same architecture (the same guarantee `setActivationFunction` gives today).
    - `setActivationFunction` keeps a group's `se_activation` when the group has an SE block (`:648-653`), so `setMainActivationEverywhere` does too, exactly as `--set-activation` does today (`documentation/deriving-models.md:74-78`).
    - It therefore equals "the uniform convenience init with that activation" only on an SE-less tower. On a tower with SE blocks it equals that only together with `--set-se-activation` (X1 pins both halves).

## D3. Sites the topology lacks: value kept, never read (OD-4)

A pre-activation tower has no stem activation, a post-activation tower has no tower-end activation, `simple_conv` has no policy pre-block, and only the compress fusion builds `feature_skip_act`. Three designs were weighed.

1. **Recommended: non-optional field, any value accepted, never read where the site is absent.**
   - The project already does this for `policy_pre_conv_channels` under `simple_conv` ("Ignores `policyPreConvChannels`", `NetworkArchitecture.swift:457`).
   - It also does it for a group's ReZero init and cap with ReZero off ("Neither value is constrained on a group without ReZero … a group switched off keeps whatever the user had", `:1651-1655`).
   - Legacy decode fills an absent site from the file's own `activation_function`, which is the same thing the uniform convenience init does. So every legacy file and every preset still compares equal.
   - Existing fixtures that switch a style after construction stay valid:
     - `a.policyHeadStyle = …` in `HeadNumericsTailTests.swift:38`, `PolicyTailPrecisionTests.swift:32`, `StandardPathForwardPinTests.swift:61`;
     - `activationStyle` flips in `InitNeutralOptionsTests.swift:106`, `:207`;
     - `featureSkipFusion = .compressConvBNReLU` in `LayerHealthTests.swift:77`, `NetworkArchitectureTests.swift:356`, `:366`, `:384`.
   - Cost: two architectures that build the same graph but differ in an absent site's value compare unequal. The Build screen keeps such a value only when the user toggles a site away. Two places notice:
    - the Build screen's preset match (`BuildNewModelModel.swift:439`) shows "Custom" instead of the preset's name;
    - the behavior-fingerprint cache, keyed by `Settings` (which holds the architecture, `Training/BehaviorFingerprint.swift:76-86`, `:134-142`), computes a second fingerprint for such a twin. That costs one extra micro-computation per process and never a wrong answer: the fingerprint is of the built graph, so both entries hold the same bytes.
    - Nothing persists or compares architectures across files by hash (D7).
2. **Pinned, like `se_activation` on an SE-less group (`:1631-1638`).** Rejected: there is no anchor value. `se_activation` pins to the group's own activation. A head site has nothing comparable, and pinning to `relu` would write a value no file stated.
3. **Optional, `null` exactly when the site is absent, enforced by `validate()`.** Rejected: it fails validation in every fixture listed under (1), which would mean many test edits. It also forces every topology edit (style flips, the fusion toggle) to rewrite activations as a side effect.

Consequences of (1):
- `validate()` gains **no** activation check (every value is legal at every site).
- The graph builder and layer health read a field only where `hasActivationSite` is true.
- A per-site `--derive-model` setter refuses an absent site, which would otherwise be a silent no-op. This matches `RezeroDeriveSupport`'s refusal on a group without ReZero (`ModelDerivation.swift` "ReZero operations").
- The Build screen disables the picker of an absent site and keeps its value. This is the precedent of the always-present, disabled "SE activation" picker (`BuildNewModelView.swift:451-459`).
- The summary and the diagram show only sites that exist (D7).

## D4. JSON shape (v9)

Encoding writes all six keys on every architecture, even absent sites and values equal to their neighbours, and never writes a top-level `activation_function`. Group keys are unchanged. Example (sorted keys, other keys elided):

```json
{
  "block_groups": [{ "activation_function": "relu", "se_activation": "leaky_relu", "...": "..." }],
  "feature_skip_activation": "relu",
  "policy_head_activation": "leaky_relu",
  "stem_activation": "relu",
  "tower_end_activation": "relu",
  "value_head_conv_activation": "leaky_relu",
  "value_head_hidden_activation": "leaky_relu"
}
```

## D5. Format v9 and the decoding rules

- `ArchitectureFormat.currentVersion = 9` (`Network/ArchitectureFormat.swift:62`). New gate `siteActivationsRequiredFromVersion = 9`, and `DecodeFormat.allowsMissingSiteActivations` (beside `:180-192`).
- Version history comment (`:33-52`) gains: "v9: the architecture must state `stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_hidden_activation`. Older files resolve each to their own top-level `activation_function`, the activation every one of those sites used before the fields existed."
- `SafetensorsModelIO.formatVersion` follows automatically (`Persistence/SafetensorsModelIO.swift:20`). So do presets and `architecture.json` (`format_version`) and `--derive-model` output.
- `NetworkArchitecture.CodingKeys` (`:1269-1299`):
  - `activationFunction = "activation_function"` becomes the decode-only `legacyActivationFunction = "activation_function"`, listed with the other legacy keys;
  - six new keys are added.
- One decode helper, `ArchitectureFormat.decodeSiteActivation(key:in:decoder:format:legacyByConstruction:legacyTowerActivation:)`, modelled on `decodeInitOption` (`:250-273`). For each of the six keys:
  1. **Stated → used.** This holds at any version. Real pre-v9 files never state these keys, but the committed tests build "old" files by re-stamping a current-version encode (`SEActivationTests.swift:104-115`, `RezeroAlphaCapTests.swift:102-115`, `SEBetaInitTests.swift:283-292`), so a stated value has to win, exactly as `decodeInitOption` already allows.
  2. **Absent, and the file is older than v9 or in the uniform-tower form** (`isUniformTowerForm`, `NetworkArchitecture.swift:1322`, which no v9 writer emits): the value is the file's top-level `activation_function`. It is recorded on the format's legacy log as `<location.>stem_activation := relu (the file's activation_function)`. If the file lacks `activation_function` as well, that is `missingRequiredField(field: "activation_function", …)`: nothing is made up.
  3. **Absent in a v9+ block-groups file:** `missingRequiredField(field: <key>, location:, formatVersion:, source:)`, the same error and message shape as `se_activation` (`ArchitectureFormat.swift:134-137`).
- **Retired key (OD-5).** A v9+ block-groups architecture that still carries a top-level `activation_function` is refused with a new `FormatError.retiredField(field: "activation_function", location:, formatVersion:, source:, replacedBy: [six keys])`.
  - Without this, a preset hand-edited from an old one (`format_version` bumped, six keys added, the old key left in) decodes and silently ignores the key its author may think still sets the heads.
  - The uniform-tower form is exempt: it is legacy by construction and is decoded strict-current by committed tests (`BlockGroupArchitectureTests.swift:38-90`, `SEActivationTests.swift:186-204`, `RezeroAlphaCapTests.swift:194-210`).
- **Uniform-tower expansion** (`NetworkArchitecture.swift:1361-1395`):
  - the group's `activation_function` / `se_activation` read the decoded `legacyActivationFunction` (required in that form, as today);
  - the six site fields go through rule 1, then rule 2: a site key stated next to the uniform-tower keys (no writer emits that, but a hand-edited file can) is used, and only an absent one resolves from `activation_function`;
  - an unknown token in any of the six keys (e.g. `"swish"`) is the `DecodingError` that `ActivationFunction`'s `Decodable` already throws, naming the coding path. No new handling is needed and none is added.
  - the combined `legacy uniform-tower keys: …` log record (`:1388-1394`) lists them too.
- **One `[ARCH]` line per load, as today.** Example for the current-format file in the evidence list:
  `[ARCH] legacy file (format v8) 20261005-r7b24-fresh.safetensors: stem_activation := relu (the file's activation_function); tower_end_activation := relu (…); feature_skip_activation := relu (…); policy_head_activation := relu (…); value_head_conv_activation := relu (…); value_head_hidden_activation := relu (…)`.
  - Written by the loaders that already call `logLegacyResolutions()`: `Persistence/CheckpointManager.swift:1847`, `:1859`, `:1916-1917`; `App/DeriveModelCLI.swift:219`, `:353`.
  - Display-only readers (model catalog, preset scan) stay quiet.
- Encode (`:1398-1418`) writes the six keys in place of `activation_function`.

## D6. Initializers

- **Full memberwise init** (`NetworkArchitecture.swift:1135-1173`): `activationFunction:` is replaced by the six required parameters, with no defaults. Its only caller is `BuildNewModelModel.architecture` (`App/UpperContentView/BuildNewModelModel.swift:153-176`).
- **Uniform convenience init** (`:1177-1253`) keeps its `activationFunction:` label. Its meaning stays "the one activation a historical single-recipe tower used at every hidden site". It sets the group, the group's SE FC1 (as today) and all six site fields to that value. Every code preset (`:2387-2532`), `ArchSweepCLI.benchArch` (`App/ArchSweepCLI.swift:32-53`) and every test fixture built through it therefore keeps its exact meaning and identity, with no edits.
- `BlockGroup` and its inits are unchanged.

## D7. Identity and effects (each checked)

| Concern | Effect | Why |
|---|---|---|
| Equality / `Hashable` | New stored fields take part (synthesized). A legacy file decodes equal to its pre-change value and to its code preset. | D3 (1) + D5 rule 2 = what the convenience init sets. `hashValue` is never persisted: identity is the value itself (`NetworkArchitecture.swift:24-25`). |
| Legacy `.dcmmodel` `archHash` (FNV) | Unchanged | It mixes shape scalars and the version label only (`Persistence/ModelCheckpointFile.swift:199-233`). Its writer has no production caller (every save is safetensors, `CheckpointManager.swift:1291`); only tests reach `ModelCheckpointFile.encode()`. |
| `arch_hash` in logs / dashboards | Unchanged | It is that same legacy FNV value; nothing else hashes the architecture. |
| Model IDs | Unchanged | Minted at events, never derived from the architecture. |
| `content_sha256` | Unchanged | SHA-256 over the tensor data region only (`SafetensorsFile.swift:117`). The architecture JSON lives in `__metadata__`. |
| Lineage records | Unchanged schema | The record holds no architecture. A re-saved file's metadata has the v9 JSON. |
| Behavior fingerprint / resume exactness | Unchanged for every existing model | The fingerprint trains the checkpoint's architecture and hashes what it computes, not the architecture (`Training/BehaviorFingerprint.swift:26-46`, `:184`). Same graph gives the same bytes, so a v8 → v9 resume should add no `build` gap. A same-build test cannot show that across builds, so V3b checks it with an OLD-written checkpoint resumed by NEW. A real head-activation change does change the fingerprint (new test). |
| Parameter count / `weightTensorPlan` | Unchanged | Activations have no parameters; the leaky slope is a constant (`:240-243`). |
| Weight init (same `--init-seed`) | Trainable tensors bit-identical whatever the activations | Every draw depends only on `(init seed, tensor name)` and the tensor's shape (`Network/WeightInitialization.swift:53-121`). He-normal uses no activation gain; there is no activation branch. |
| BN running statistics of a fresh build | Identical when only head activations change; different downstream of a changed stem / tower-end / fusion activation | `.randomWeights` calibrates running stats with one GPU forward (`Network/ChessMPSNetwork.swift:21-31`, `:80-99`). A BN downstream of a changed activation sees different inputs. `policy_head_activation`, `value_head_conv_activation` and `value_head_hidden_activation` have no BN downstream; the tower-end activation feeds `policy.pre_bn`, `value.bn` and `feature_skip.bn`. |
| Bit-exact behavior of existing models | Unchanged | Each site reads a field equal to the old top-level value. Pinned by `StandardPathForwardPinTests.testStandardPathForwardIsBitIdenticalToThePinnedOutputs` (unmodified) and the new legacy forward test. |
| `architectureSummary` (`[REPLAY-ARCH]`, `[VS-UCI-ARCH]`, `[DERIVE]`, `[BUTTON] Build Network`, `[ARCH-CONFIG]`, About, session picker) | Byte-identical when every *existing* site has the same activation (every file on disk and every preset). Otherwise the clause lists the existing sites. | `:1924` becomes the clause below; golden tests `BlockGroupArchitectureTests.swift:136-158` pass unmodified. |
| `[ARCH] legacy file …` | Gains six resolutions per pre-v9 load (D5) | |
| Diagram | Shows each existing site's activation (P3) | |
| Presets | Built-in: no change (code, convenience init). User presets (10 in `Presets/`, 5 at v8, 5 unversioned) and committed experiment presets (`experiments/*/test_*.json`, `r7_basic24.json`, `experiments/presets/*.json`) load unchanged through D5 rule 2. Nothing must be re-generated (re-saving is optional, OD-10). | |
| `parameters.json` / `parameters.md` | Not affected | Architecture fields are not training parameters. |

**Summary clause rule** (replaces `" . act \(activationFunction.rawValue)"` at `:1924`):
- let `present` be the sites where `hasActivationSite` is true (never empty: both value sites always exist);
- if every present site has one activation `X`, the clause is ` . act X` (today's string);
- otherwise it is ` . act ` followed by `<summaryLabel> <fn>` for each present site, comma-separated, in build order. Example: ` . act tower_end relu, policy leaky_relu, value_conv leaky_relu, value_hidden leaky_relu`.

One shared function, `NetworkArchitecture.siteActivationClause`, is read by the summary. The diagram reads the per-site values.

---

# Part T — Touch points

### T1. `Network/ArchitectureFormat.swift`
- `currentVersion` 8 → 9 (`:62`); new constant `siteActivationsRequiredFromVersion` (after `:102`); `DecodeFormat.allowsMissingSiteActivations` (after `:192`).
- History comment (`:33-52`).
- `decodeSiteActivation` (beside `decodeInitOption`, `:240-273`).
- `FormatError.retiredField` with its description (`:124-146`), if OD-5 is approved.

### T2. `Network/NetworkArchitecture.swift`
- `ActivationFunction` doc (`:221-229`).
- `ArchitectureActivationSite` enum (new; beside `InitOptionField`, `:405-452`).
- The stored field `activationFunction` (`:1083-1086`) is replaced by the six fields, each with a doc comment saying where it applies and when the site exists.
- Full memberwise init (`:1135-1173`) and uniform convenience init (`:1177-1253`), per D6.
- `CodingKeys` (`:1269-1299`), decode (`:1309-1396`) and encode (`:1398-1418`), per D5.
- `hasActivationSite`, `activation(at:)`, `setActivation(_:at:)`, `setActivationAtEveryArchitectureSite`, `setMainActivationEverywhere` (new extension, beside the init-neutral extension at `:2152`).
- `architectureSummary` (`:1904-1930`) uses `siteActivationClause`.
- `validate()` (`:1608-1689`): no change (D3). The doc comment states that every activation is legal at every site.

### T3. `Network/ChessNetwork.swift`
- Delete the private tower-level overload (`:2364-2372`). A site can then only be built from its own field: the compiler refuses a call that passes `arch`.
- `:732` uses `Self.activation(g, x, arch.stemActivation, name: "stem_act")`.
- `:895` uses `arch.towerEndActivation`.
- `:933` uses `arch.featureSkipActivation`.
- `:3206` and `:3258` use `arch.policyHeadActivation`. Under the `float32FromPreBatchNorm` policy-tail precision this activation runs in fp32, under `mixedFinalProjection` in the compute dtype. Both paths are tested.
- `:3467` uses `arch.valueHeadConvActivation`.
- `:3495` uses `arch.valueHeadHiddenActivation`.
- Update the doc comment of `activation(_:_:_:name:)` (`:2374-2378`). Graph op names are unchanged, so `LayerHealthTests.graphNames(forSite:)` (`DrewsChessMachineTests/LayerHealthTests.swift:200-213`) still holds.

### T4. `Training/LayerHealth.swift`
- `batchNormSites` (`:159-217`) tags each site with its own field:
  - `stem.bn` → `hasStemActivation ? stemActivation : nil`;
  - `tower_final_bn` → `towerEndActivation`;
  - `feature_skip.bn` → `featureSkipActivation`;
  - `policy.pre_bn` → `policyHeadActivation`;
  - `value.bn` → `valueHeadConvActivation`.
- `valueFC1Layer` (`:242-252`) uses `valueHeadHiddenActivation`.
- Doc comments at `:159-166` and `:242-244` are rewritten.
- `classification(for:)` (`:274-282`) is unchanged: relu / leaky are classified, silu / gelu report `notApplicableSmoothActivation`. A smooth head site next to a ReLU tower end is therefore classified site by site.
- No change to `LayerHealthLog` or the table format: the checkpoint table already prints an `act` column per site (`:1110`, `:1129`), which is how the validation checks it.

### T5. Numerics audit: no code change
- The layer-health findings (`Network/NumericsAudit.swift:585-610`) and `--analyze-numerics`'s per-site table (`Network/NumericsAudit+Summary.swift:43-44`) come from `LayerHealth`, so they become per-site through T4.
- The dynamic audit compares formats tap by tap and makes no assumption about the activation.

### T6. Behavior fingerprint: no code change
- It is computed from the built graph (D7). The new test `testAHeadActivationChangeChangesTheBehaviorFingerprint` pins that a head-only change is detected.

### T7. `--derive-model`
- **P1:** `SetActivationDeriveOperation` (`Persistence/ModelDerivation.swift:738-806`):
  - `apply` uses `setMainActivationEverywhere`;
  - the no-op check covers all six fields and every group;
  - `summary` reads: "Set the main hidden activation everywhere: every architecture-level site (stem, tower end, feature-skip fusion, policy head, value conv, value hidden) and every block group's activation_function …";
  - `changedArchitectureFields` becomes the six keys plus `block_groups[].activation_function` and `block_groups[].se_activation`.
  - It sets absent sites too (D3). On an SE-less tower, `--set-activation leaky_relu` therefore equals the leaky convenience-built tower. On a tower with SE blocks it leaves every SE group's `se_activation` alone (D2, as today), and only `--set-activation X --set-se-activation X` equals the convenience-built tower.
- **P2:** new file `Persistence/SiteActivationDerive.swift` (the same per-family layout as `InitOptionDerive.swift`). The project uses folder-synchronized groups, so no `project.pbxproj` edit is needed.
  - One `SetSiteActivationDeriveOperation(site:value:)` type, with one `DeriveOperationKind` per site built from `ArchitectureActivationSite.allCases`: `--set-stem-activation`, `--set-tower-end-activation`, `--set-feature-skip-activation`, `--set-policy-head-activation`, `--set-value-head-conv-activation`, `--set-value-head-hidden-activation`.
  - Value syntax `relu|silu|gelu|leaky_relu`; `acceptsGroupSelection: false`; `rewrittenTensorsDescription: "none"`.
  - Refuses an unknown value, an absent site (naming `absentReason`) and a no-op. No tensor rewrites, so it is allowed on a trained source (`ModelDerivation.swift:28-37`).
- **Catalog order** (`ModelDerivation.operationKinds`, `:119-131`): the six follow `SetSEActivationDeriveOperation.kind`.
  - `--set-activation X --set-value-head-hidden-activation Y` therefore gives X everywhere except Y.
  - `--help` and the CLI parser are catalog-driven (`App/DeriveModelCLI.swift:57`, `:98`, `:150`), so they need no other edit.
- `documentation/deriving-models.md`:
  - `:37` (the list of no-tensor operations);
  - `:57` (the `--set-activation` row);
  - a new row per site setter after `:58`;
  - `:74-78` (`--set-activation` prose);
  - `:138-141` (operation order);
  - examples after `:155-162` (leaky heads on a ReLU tower);
  - `:188-193` (legacy rules: "every site activation → the file's `activation_function` before v9");
  - `:203-209` (refusals: a site the topology lacks).

### T8. Build New Model
- `App/UpperContentView/BuildNewModelModel.swift`:
  - the stored field `activationFunction` (`:51`) is replaced by six stored fields;
  - `init` (`:97-121`) and `load` (`:129-151`) copy them;
  - `architecture` (`:153-176`) passes them;
  - new `func applyMainActivationEverywhere(_:)` applies `NetworkArchitecture.setMainActivationEverywhere` to `architecture` and copies the six fields and each group back through its draft (the `applyInitOptions(of:)` pattern, `:289`);
  - new `func siteExists(_:) -> Bool`, read once per redraw by the screen like `nonStandardInitOptions` (`BuildNewModelView.swift:65-66`).
- New file `App/UpperContentView/ArchitectureSiteActivationPicker.swift`: a `View` struct taking the site, a `Binding<ActivationFunction>` and `siteExists`.
  - It is always present, so rows never shift (the SE-activation precedent, `BuildNewModelView.swift:451-459`).
  - It is disabled when the site does not exist, with `.help(site.absentReason)`, and otherwise `.help` with what the site does.
- `BuildNewModelView.swift`:
  - Tower section (`:91-99`): the "Tower-level activation" picker (`:93`) becomes the "Stem activation" picker (P1), plus "Tower-end activation" and a `Menu("Use for every activation")` listing the four functions, which calls `applyMainActivationEverywhere` (P3).
  - Feature skip section (`:165-180`): "Fusion activation", placed **outside** the existing `if model.featureSkipSource != .none` block (directly after the "Source" picker), so it is always present like every other site picker, and disabled through `siteExists(.featureSkipFusion)` (P3).
  - Policy head section (`:126-141`): "Pre-block activation" (P3).
  - Value head section (`:142-164`): "Conv activation" and "Hidden activation" (P3).
- `BlockGroupDraft.swift`: no change.
- **P1 keeps the screen buildable and lossless.** All six fields load and compose, so a loaded leaky-head model round-trips unchanged, but only "Stem activation" is editable until P3.

### T9. `App/UpperContentView/ArchitectureDiagramView.swift`
- P1 (needed to compile):
  - `:53` shows `arch.stemActivation` (still only when `hasStemActivation`);
  - `:68` shows `arch.towerEndActivation`.
- P3:
  - the policy cell's pre-block line (`:81`) becomes `"\(in) → K=\(K) · \(policyHeadActivation)"`;
  - the value cell line (`:91`) splits into `"\(in) → \(conv)ch · \(convAct)"` and `"→ FC\(h) · \(hiddenAct)"`;
  - `featureSkipMarker` (`:165-186`) appends ` · <featureSkipActivation>` when the compress node exists.
- No new `some View` properties.

### T10. CLI and logs
- `--new-model --architecture <preset | path>`: no code change. A v9 preset or `architecture.json` carries the six fields. `ArchitectureConfig.writeTemplate` (`Persistence/ArchitectureConfig.swift:62`) writes v9 with all six.
- `CommandLineHelp.usageText`: no change. The derive operations are listed by `--derive-model --help` from the catalog.
- Log strings: `architectureSummary` per D7; `[ARCH] legacy file …` per D5; `[LAYER-HEALTH]` checkpoint table per T4.

### T11. Python mirrors
- **`scripts/dcm_arch.py`:**
  - add `SITE_ACTIVATIONS_REQUIRED_FROM_VERSION = 9` and `SITE_ACTIVATION_KEYS` (after `:41`);
  - add `site_activations(arch, format_version)` and `site_activations_md(md)`, which return the six values under D5's rules: stated wins; older or uniform-tower resolves from `activation_function`; v9+ missing raises `ArchitectureError`; v9+ block-groups with a top-level `activation_function` raises (OD-5).
  - `norm_arch` (`:48-95`) is **not** changed. Its committed tests build synthetic v5 architectures with no top-level `activation_function` (`documentation/dashboards/tests/test_tooling.py:29-48`), and only scripts that model the heads need the site values.
Python scripts fall into two kinds, and they are handled differently.
- **Reusable readers** are run on new models (shared helpers, or tools with a documented reproduce/re-run use). They get the v9 rules and an explicit refusal of anything they do not model.
- **Frozen, model-specific scripts** are records of one analysis of named ReLU checkpoints. They are not edited (OD-9); their limits are documented instead.

**Reusable readers (changed in P1):**
- **`experiments/20261004-head-logits-relu-inputs/relu_inputs.py`:**
  - `forward(x, arch, t, record)` (`:204`) receives no metadata, and `load` (`:23-31`) returns `md` to `run` (`:247-258`). So `run` resolves everything once, right after `load`:
    - `arch = dcm_arch.norm_arch_md(md)` replaces the raw `json.loads`, so a pre-feature-skip file reads `feature_skip_source: none` by the app's own rule;
    - `sites = dcm_arch.site_activations_md(md)`.
  - It passes `sites` into `forward(x, arch, sites, t, record)`.
  - `forward` adds asserts next to its existing architecture assert (`:206-208`):
    - every group's `activation_function` is `relu`;
    - `sites['tower_end_activation']`, `sites['policy_head_activation']`, `sites['value_head_conv_activation']` and `sites['value_head_hidden_activation']` are `relu`;
    - `arch['feature_skip_source'] == 'none'`, because the forward never builds a feature skip; today a feature-skip model would be mis-modelled silently.
  - None of this changes its output on the three ReLU checkpoints it was run on (P1 gate: byte-identical `relu_inputs_bot.json`).
  - It gains the `sys.path` insert `dcm_arch.py`'s docstring prescribes (`scripts/dcm_arch.py:17-20`).
- **`documentation/research/fp16-feasibility/scripts/fwd16.py` and its callers** (one unit; `fwd16.py` already imports `dcm_arch`, `:22`). OD-9.
  - **The problem.** `forward(T, arch, x, R, q, capture)` (`:118`) and `forward_batched(T, arch, X, R, q, bs, keep)` (`:196-198`) receive an architecture that every caller has already normalized with `norm_arch(md['architecture'])`, **without** the format version. Normalization also expands the uniform-tower form, so by then neither the version nor the form needed for D5 is left. Site activations therefore cannot be resolved inside `forward`.
  - **Resolution from raw metadata, before normalization.** Each caller, right after `load` (`:33-41`, which returns `md`), computes `sites = dcm_arch.site_activations_md(md)`. That reads `dcm_format_version` and the raw architecture JSON, so it sees the real version and form. In the same line the caller replaces the version-less `norm_arch(md['architecture'])` with `norm_arch_md(md)`, the form `fwd16.py`'s own docstring prescribes (`:203-212`).
  - **Passing them through.** `forward` and `forward_batched` gain a required keyword-only parameter `sites` (`def forward(T, arch, x, R=frozenset(), q=ident, capture=False, *, sites)`). `forward_batched` passes it on. Keyword-only means a caller that is not updated fails with a `TypeError` at once, and no positional argument shifts.
  - **Affected callers, all updated in P1:**

    | Caller | Where `arch` is built | Calls |
    |---|---|---|
    | `sat16.py` | `:12` | `forward` `:23` |
    | `shared16.py` | `:12` | `forward_batched` `:15` |
    | `grad16.py` | `:33` | `forward_batched` `:41` |
    | `range16.py` | `:22` | `forward` `:63`, `:111`; `forward_batched` `:116`, `:132` |
    | `tails16.py` | `:16` | `forward_batched` `:17`, `:21` |
    | `tailstart16.py` | `:15` | `forward` `:16`; `forward_batched` `:18`, `:26` |

  - **Checks in `forward`**, next to the existing one-group assert (`:120`):
    - reject any feature skip: `assert arch['feature_skip_source'] == 'none'`, because this forward never builds one;
    - the group's main path is `relu` (kept from `:122`); the top-level half of `:122` is removed, since the key no longer exists in v9;
    - `sites['stem_activation'] == 'relu'` only when the group is post-activation (`:127` is the only place it is applied);
    - `sites['tower_end_activation'] == 'relu'` only when pre-activation (`:168`);
    - `sites['policy_head_activation'] == 'relu'` only when the policy pre-block exists (`policy_head_style != 'simple_conv'`; `:176`);
    - `sites['value_head_conv_activation'] == 'relu'` and `sites['value_head_hidden_activation'] == 'relu'` always (`:182`, `:184`).
    - A site the topology lacks is never checked, matching D3: its value is never read.
- **`experiments/20261001-se-fc1-leaky/full-model-analysis/scripts/units.py`** (built on the shared `fma_lib.py`, whose checkpoint object carries the raw `metadata` and the normalized `architecture`, `fma_lib.py:119-125`). OD-9.
  - `analyze` (`:188`) labels four sites as ReLU-fed:
    - `tower_final_bn` (`:249`);
    - `policy.pre_bn` (`:253`, `:257`);
    - `value.bn` (`:261`; its readers `:270-272`);
    - value FC1 and its downstream readers: `value.fc1` / `value.fc1.bias` `next_op="relu"` (`:262-263`) and `value.wdl_fc2.in` "reads relu(value.fc1)" (`:276`).
  - Today it checks only the group (`:198-207`).
  - New function `require_supported_architecture(ckpt)`, called first in `analyze` before any topology-dependent label is produced:
    1. reject an enabled feature skip (`ckpt.architecture['feature_skip_source'] != 'none'` raises `ValueError` naming the file), since the labels assume the heads read the tower output directly;
    2. resolve `sites = dcm_arch.site_activations_md(ckpt.metadata)` from the raw metadata (`L.dcm_arch`, already imported by `fma_lib.py:46`);
    3. require `relu` for `tower_end_activation`, `policy_head_activation`, `value_head_conv_activation` **and** `value_head_hidden_activation`, raising `ValueError` naming the file and the site.
    - The existing one-group / pre-activation / ReLU-main-path check (`:195-199`) stays where it is. The script already requires a pre-act tower, so the stem has no activation to check.
  - Rejection tests in `documentation/dashboards/tests/test_dcm_arch_site_activations.py`. `units.py` is importable (its `main` is guarded, `:349`). Each case uses a stand-in object with `file`, `metadata` and `architecture` built from a synthetic header:
    - accepted: a v9 all-ReLU file, and a v5 ReLU file resolved from `activation_function`;
    - rejected: `value_head_hidden_activation: leaky_relu` (the case the earlier draft missed), `tower_end_activation: gelu`, `policy_head_activation: silu`, `value_head_conv_activation: leaky_relu`, and `feature_skip_source: stem_output` (rejected before any activation check).
- **Not changed:**
  - `experiments/arch_flops.py` costs every activation as one op per element whatever its type (`:19`, `:33-74`) and reads no activation field.
  - **Frozen scripts** (OD-9). Not all of them fail on a v9 file. Counted by `grep -c activation_function`:

    | Script | Reads the top-level key? | On a v9 model |
    |---|---|---|
    | `documentation/research/bf16-head-offset/scripts/fwd3.py` | yes (4) | `KeyError`, loud |
    | `documentation/research/bf16-head-offset/scripts/fwd4.py` | yes (`:81`) | `KeyError`, loud |
    | `documentation/research/policy-head-2026-10-01/scripts/policy_head_lib.py` (3), `analyze_policy_head.py` (1) | yes | `KeyError`, loud |
    | `documentation/research/bf16-head-offset/scripts/fwd.py` (ReLU hard-coded, `:87-91`), `net681.py`; the other `policy-head-2026-10-01/scripts/*`; `experiments/20261003-fatty-vs-skinny/tensor-health/fatty_tensors.py` (labels BN sites ReLU-fed, `:25-29`); `experiments/20260929-se-style-ab/final_data.py`; `experiments/20261001-se-fc1-leaky/full-model-analysis/scripts/input_features.py` | no | **silently models every site as ReLU** |

  - Recommended: leave them unedited as records. P4 adds a "frozen scripts" paragraph to `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20 listing this table and stating that these scripts model ReLU at every site, are valid only for the checkpoints their own folders name, and may silently misinterpret a model with any non-ReLU site.
  - The alternative under OD-9 is a one-line `dcm_arch.site_activations_md` guard in each silent script.

### T12. Documentation
- `documentation/plans-active/RUNTIME_ARCHITECTURE_CONFIG_PLAN.md`: a new "§20. Architecture format v9: per-site activations", in the shape of §18 (`:845-884`), including the frozen-Python-scripts paragraph and table from T11.
- `CLAUDE.md:136` (OD-7). Proposed edits:
  - "current v8" → "current v9";
  - append "v9 replaces the top-level `activation_function` with one activation per architecture-level site (`stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_hidden_activation`), each resolved from an older file's own `activation_function`";
  - "→ tower-end BN (pre-act tail only)" → "→ tower-end BN + its activation (pre-act tail only)".
- `CHANGELOG.md`: one entry after the last phase, with each phase's commit hash.
- `ROADMAP.md`: only with the owner's permission (OD-11).

### T13. Verified untouched
- `Network/WeightInitialization.swift`;
- `Persistence/SafetensorsModelIO.swift` (beyond the derived `formatVersion`), `ModelGraft.swift` (targets carry their own activations), `SessionManifest.swift`, `ArchitecturePresetStore.swift`, `ModelCheckpointFile.swift`;
- `Training/BehaviorFingerprint.swift`;
- every caller of the convenience init;
- `BlockGroup`.

---

# Part X — Tests

### X1. New test files

**`DrewsChessMachineTests/ArchitectureActivationSiteTests.swift`** (pure; no Metal):

*Format gate:*
- `testCurrentVersionRequiresSiteActivations`: `siteActivationsRequiredFromVersion == 9`, `currentVersion >= 9`, `SafetensorsModelIO.formatVersion == String(currentVersion)`.
- `testNewFilesWriteEverySiteAndNoTopLevelActivationFunction`: an encoded model's `architecture` JSON has all six keys and no top-level `activation_function`.
- `testRoundTripPreservesEachSiteIndependently`: for every site × every `ActivationFunction`, set only that site; the safetensors, preset and `architecture.json` round trips are equal, with no legacy log line.
- `testPreV9FileResolvesEverySiteToItsOwnActivationFunction`:
  - versions `"8"`, `"5"`, `"3"` and none;
  - top-level `gelu`, group `relu`;
  - all six decode as `gelu`; the log names the file and each `:= gelu (the file's activation_function)`.
- `testPreV9PostActivationFileResolvesTheStemAndAnAbsentTowerEnd`: a v3 post-act `simple_conv` fixture; the stem is `relu` and is read by `LayerHealth`; the tower-end value resolves too but is never read.
- `testEveryPresetDecodesFromItsLegacyFormToItself`: for each `Preset`, strip the six keys, add `activation_function`, decode at v8 and v3. The result is equal, with the same hash, summary, parameter count, `weightTensorPlan` and legacy `archHash`.
- `testLegacyUniformTowerKeysResolveEverySiteAtAnyStatedVersion`: the uniform-tower JSON decoded strict-current and at v8.
- `testV9FileMissingASiteActivationIsRejectedNamingFieldAndFile`: for each key, `missingRequiredField(field:, location: "the top level", formatVersion: 9, source:)`. Also a preset at `format_version` 9 (field `location: "architecture"`), and an undecorated `JSONDecoder()` (strict current).
- `testPreV9FileWithNeitherSitesNorActivationFunctionIsRejected`: `missingRequiredField(field: "activation_function", …)`.
- `testV9BlockGroupsFileStatingActivationFunctionIsRefused` (OD-5), and `testUniformTowerFormIsNotTreatedAsRetired`.
- `testAStatedSiteValueWinsInAPreV9File` (D5 rule 1): a v5-stamped file with `activation_function: gelu` and only `policy_head_activation: leaky_relu` stated.
  - `policy_head_activation` decodes as `leaky_relu`; the other five decode as `gelu`.
  - The legacy log lists exactly the five resolved keys and not the stated one.
- `testAV9FileWithSomeSiteKeysNamesTheFirstMissingOne`: a v9 file stating three of the six keys fails with `missingRequiredField` naming an absent key, never a resolved value.
- `testAnUnknownSiteActivationTokenIsADecodeError`: `"swish"` in each key, at v9 and at v5, is a `DecodingError` whose coding path names the key. It is never resolved from `activation_function`.
- `testUniformTowerFormUsesStatedSiteKeysThenResolvesTheRest`: uniform-tower JSON (`activation_function: gelu`) that also states `value_head_hidden_activation: leaky_relu` decodes that site as `leaky_relu` and the other five as `gelu` (D5 uniform expansion: rule 1, then rule 2).

*Semantics:*
- `testSiteExistenceFollowsTheTopology`: the stem iff the first group is post; the tower end iff the last is pre; the fusion iff the compress node is built; policy iff not `simple_conv`; both value sites always. Covered on mixed pre→post and post→pre towers.
- `testAbsentSiteValuesValidate`: a pre-act tower with `stemActivation = .gelu` and a `simple_conv` head with `policyHeadActivation = .silu` validate (OD-4).
- `testUniformConvenienceInitSetsEverySiteToItsActivation`.
- `testSetMainActivationEverywhereMatchesTheConvenienceInitOnAnSELessTower`: for each function, on an SE-less ReLU tower (the convenience init with `blockSeStyle: .none`), `setMainActivationEverywhere(X)` equals the same tower built with `activationFunction: X`.
- `testSetMainActivationEverywhereKeepsEverySEGroupsFC1Activation`. Mixed fixture: three groups (an SE group with `se_activation` `relu`, an SE group with `leaky_relu`, an SE-less group), all main paths `relu`. After `setMainActivationEverywhere(.gelu)` it equals an **explicitly constructed expected architecture** with the same three groups, built field by field through the full `BlockGroup` and `NetworkArchitecture` memberwise inits:
  - all six sites `gelu`;
  - every group's `activation_function` `gelu`;
  - the two SE groups' `se_activation` still `relu` and `leaky_relu`;
  - the SE-less group's `se_activation` `gelu` (`BlockGroup.setActivationFunction`, `NetworkArchitecture.swift:648-653`);
  - every other field unchanged.
  The convenience init builds one uniform group (`NetworkArchitecture.swift:1199-1233`), so it can never equal this fixture and is not used for it.
- `testSetMainActivationEverywherePlusSEActivationEqualsTheConvenienceInitOnAnSETower`. Separate single-group SE fixture: the convenience init with `blockSeStyle: .scaleAndBias` and `activationFunction: .relu`. `setMainActivationEverywhere(.gelu)` followed by `blockGroups[0].seActivation = .gelu` equals the same convenience init with `activationFunction: .gelu`, every other argument identical. Without the second step it differs only in `blockGroups[0].seActivation` (`relu`).
- `testSummaryKeepsTheUniformClause`: golden ` . act relu` for `.current` and a gelu tower.
- `testSummaryListsExistingSitesWhenTheyDiffer`: golden ` . act tower_end relu, policy leaky_relu, value_conv leaky_relu, value_hidden leaky_relu`; absent sites never listed.

*Layer health:*
- `testEachBatchNormSiteTagFollowsOnlyItsOwnField`: a site × function matrix, changing **one site at a time**. `ActivationFunction` has four cases, so "five different functions at once" is impossible.
  - Base: a post-act first group → pre-act last group tower with a compress fusion routed to an `fc_bottleneck` policy, every site `relu`.
  - For each of the five BN sites (`stem.bn`, `tower_final_bn`, `feature_skip.bn`, `policy.pre_bn`, `value.bn`) and each function in `silu`, `gelu`, `leaky_relu`, set only that site's field. That site's tag equals the function, and every other site's tag (block BNs included) is unchanged from the base.
- `testValueHiddenLayerFollowsOnlyItsOwnField`: the same one-at-a-time matrix for `valueHeadHiddenActivation` and `LayerHealth.valueFC1Layer`. Every BN tag stays unchanged.
- `testSmoothHeadSitesAreNotClassifiedWhileAReLUTowerEndIs`: summary-level, through `summarizePlanAligned` with synthetic γ/β.

*Derive:*
- `testEachSiteSetterChangesOnlyItsFieldAndCopiesEveryTensorBitExact`.
- `testSiteSetterRefusesASiteTheTopologyLacks`: the stem on pre-act, the tower end on post-act, the fusion without compress, policy on `simple_conv`.
- `testSiteSetterRefusesANoOp`.
- `testSiteSetterRejectsAnUnknownValue`.
- `testSiteSettersAreAllowedOnATrainedSource`.
- `testSetActivationSetsEverySiteAndEveryGroup`.
- `testSetActivationThenASiteSetterGivesTheMixedArchitecture`: catalog order.
- `testLegacySourceDerivesToAV9FileStatingEverySite`: a v5 source; the output's `dcm_format_version` is the current one and it states all six keys.

*Build screen (`@MainActor`):*
- `testBuildScreenLoadsAndComposesEverySiteActivation`: load a leaky-head architecture; `model.architecture` equals it.
- `testBuildScreenSiteAvailabilityFollowsTheTopology`.
- `testASiteValueSurvivesTogglingTheSiteAway`: `simple_conv` → `intermediate_conv` keeps `policyHeadActivation`.
- `testUseForEveryActivationMatchesDeriveSetActivation`.

**`DrewsChessMachineTests/ArchitectureActivationSiteGraphTests.swift`** (GPU; `XCTSkip` without Metal; fp32 unless stated):
- `testLegacyDecodedArchitectureBuildsTheIdenticalForwardPass`: a fixture re-stamped `"8"` with the six keys stripped and `activation_function: gelu` added; same weights; policy and value outputs are bit-identical (`bitPattern`).
- `testEachHeadSiteChangesOnlyTheHeadItFeeds`: one weight set, ReLU everywhere vs one site at `leaky_relu`.
  - value conv or value hidden: policy logits bit-identical, value logits differ;
  - policy: value bit-identical, policy differs;
  - tower end: both differ.
- `testEachSiteBuildsItsSelectedFunction`. A numeric check, not op wiring.
  - Why not the wiring walk: copying `LayerHealthTests`' direct-consumer/name walk would show only that an op named `*_act` reads the BN, not which function was built. It also fails for GELU, whose final named multiply reads `<name>_halfx` and `<name>_1plus`, not the BN output (`ChessNetwork.swift:2390-2400`).
  - Instead, build `ChessNetwork(arch:bnMode: .inference, initialization: .seeded(initSeed:), analysisTaps: true)` (`ChessNetwork.swift:601`, `:649`). Read `evaluateAnalysisTaps(boards:count:)` (`:2125`) on a handful of positions, with the running variances set away from 1 and the β values spread across both signs so every function's negative side is exercised.
  - From the tapped site inputs, compute on the CPU in Double: BN in inference form (ε `1e-5`, `:2626`), then the selected function (relu, `x ≥ 0 ? x : 0.01x`, `x·σ(x)`, `0.5x(1+erf(x/√2))`). Compare with the tapped output:
    - stem: `stem_bn_input` → `stem_output` (post-act first group);
    - tower end: `tower_final_bn_input` → `tower_output`;
    - fusion: `feature_skip_bn_input` → BN → function → the routed policy pre-conv (1×1, from the exported weights) → `policy_pre_bn_input`;
    - policy: `policy_pre_bn_input` → `policy_pre_act`;
    - value conv and value hidden: `value_bn_input` → BN → conv function → flatten → `value.fc1` (exported weight and bias) → hidden function → `value_fc1_act`.
  - The value FC1 site, which the BN walk never covered, is checked here.
  - Run once per site × function, one site at a time, in fp32. Tolerance: 1e-5 relative to the site's max |value|. That absorbs MPSGraph's fused BN and its erf against Foundation's. It is far below the smallest difference between the four functions on these inputs (leaky vs relu differs by 0.01·|x| on every negative element), and the test asserts that margin too, so it can never pass with the wrong function.
- `testFreshTrainablesAreActivationIndependent`:
  - same `initSeed`, ReLU everywhere vs leaky at all six sites;
  - every trainable tensor is bit-identical;
  - with only the three head sites changed, every BN running statistic is bit-identical too (same machine and precision; D7).
- `testLeakyHeadsTrainOneFiniteStep`: bf16, under both `PolicyTailPrecision` values.
- `testAHeadActivationChangeChangesTheBehaviorFingerprint`: `BehaviorFingerprint.computeUncached` for ReLU vs a leaky value head; the SHA-256s differ.

**`DrewsChessMachineTests/BuildNewModelSiteActivationRenderTests.swift`** (hosted, like `BuildNewModelTowerShapeRenderTests.swift:20-60`): host the screen and, for each of pre/post first group, pre/post last group, `simple_conv`, and the compress fusion on/off, settle it and check:
- it draws without a trap;
- all six site pickers are present in every configuration, the "Fusion activation" picker included, even with `featureSkipSource == .none` (T8);
- each picker's enabled state equals `siteExists`.

**Python:** `documentation/dashboards/tests/test_dcm_arch_site_activations.py` (new; run with `python3 -m unittest discover -s documentation/dashboards/tests`):
- the legacy resolution for v8, v3, unversioned and uniform-tower inputs;
- the v9 missing-key `ArchitectureError`;
- the v9 retired key;
- stated-wins;
- an unknown token;
- the `units.py` acceptance and rejection cases (T11);
- `fwd16.forward` refusing an unchecked caller (a call without `sites=` raises `TypeError`), a feature skip, and a non-ReLU value-hidden site; and not checking an absent site (a pre-act fixture whose `stem_activation` is `gelu` passes).

If a GPU test that asserts bit-identity across two different graphs (`testEachHeadSiteChangesOnlyTheHeadItFeeds`) shows ulp differences in the unchanged head, that is MPSGraph compiling the shared tower differently for a different downstream graph. It is reported to the owner as a finding before any tolerance is introduced.

### X2. Existing tests that must change (OD-6)

Two causes:
- **The rename (OD-1)** makes every row but the last two a compile error, so none can keep compiling with a silently different meaning. Every such edit keeps the test's original intent: "the tower-level activation" becomes "every architecture-level site".
- **The version bump** fails `LineageRecordTests.swift:114` at run time, because it pins the literal `"8"`. The catalog pin is the P2 row.

Totals: 11 tests in 6 files; 14 lines in P1, plus the catalog pin in P2.

| File:line | Today | Change |
|---|---|---|
| `RuntimeArchReachTests.swift:292`, `:304` (`testSiluAndGeluChangeForwardPassVsReLU`) | `arch.activationFunction = x` | `arch.setActivationAtEveryArchitectureSite(x)`. Under OD-1's alternative, setting only the stem on the pre-act `.current` changes nothing, and the `XCTAssertNotEqual`s fail. |
| `RuntimeArchReachTests.swift:331` (`testGeluTrainerStepProducesFiniteLosses`) | same | same. Without it the test would train a model with no GELU anywhere and still pass. |
| `LeakyReLUTests.swift:121`, `:124` (`testLeakyReLUChangesTheForwardPassAndTrains`) | same | same |
| `LeakyReLUTests.swift:170`, `:178` (`testSetActivationChangesEverySiteAndCopiesEveryTensor`) | builds `source` / `expected` through `activationFunction` | through `setMainActivationEverywhere`, so `expected` matches the new `--set-activation` |
| `LeakyReLUTests.swift:175` | asserts `targetArchitecture.activationFunction == .leakyRelu` | asserts `activation(at:) == .leakyRelu` for every `ArchitectureActivationSite` |
| `LeakyReLUTests.swift:197` (`testSetActivationRefusesANoOp`) | `source.activationFunction = .leakyRelu` | `source.setActivationAtEveryArchitectureSite(.leakyRelu)` |
| `SEActivationTests.swift:370` (`testFC1UsesSEActivationIndependentlyOfTheMainPath`) | `leakyEverywhere.activationFunction = .leakyRelu` | `setActivationAtEveryArchitectureSite(.leakyRelu)` |
| `SEActivationTests.swift:543` (`testSetActivationLeavesSEGroupsFC1Alone`) | asserts `activationFunction == .leakyRelu` | every site `== .leakyRelu` |
| `SEActivationTests.swift:592` (`testBuildScreenAndDeriveApplyTheSameActivationRule`) | `model.activationFunction = .leakyRelu` | `model.applyMainActivationEverywhere(.leakyRelu)`. The draft loop after it stays (a no-op) and the assertion is unchanged. |
| `DeriveTrainedSourceTests.swift:87` (`testArchitectureOnlyOperationsStayAllowedOnATrainedSource`) | asserts `activationFunction == .leakyRelu` | every site `== .leakyRelu` |
| `LineageRecordTests.swift:114` (`testFileRoundTripCarriesTheRecordAndItsMirrors`) | `XCTAssertEqual(md[SafetensorsModelIO.Key.formatVersion], "8")` | `String(ArchitectureFormat.currentVersion)`, the form the version-gate suites already use (`SEBetaInitTests.swift:301`, `RezeroAlphaCapTests.swift:133`). The test is about lineage carriage, not about which version is current, so it stops failing at every future bump. (P1) |
| `SEBetaInitTests.swift:610` (`testOperationCatalogDrivesTheCLI`) | pins the derive catalog's flag list | the six site flags are inserted after `"--set-se-activation"` (P2). Precedent: owner-approved in `de0f22be`. |

Verified *not* to need edits:
- group-level reads such as `SEActivationTests.swift:148`, `:545`, `LeakyReLUTests.swift:122`, `:176`, `BuildNewModelDraftTests.swift:43`;
- the record-name string in `GraftDeriveTests.swift:236`;
- uniform-tower JSON fixtures (`BlockGroupArchitectureTests.swift:41-63`, `RezeroAlphaCapTests.swift:194-200`, `SEActivationTests.swift:188-198`), which stay legacy by construction;
- every test that re-stamps a current encode as an older version (D5 rule 1);
- `LayerHealthTests` (its fixtures use the convenience init, and `mixedArch`'s stem is the leaky value it asserts at `:187`).

### X3. Existing tests that guard the change

Each of these must pass. Each is **unchanged apart from the X2 edits listed for it** (`SEActivationTests` at `:370`, `:543`, `:592`; `SEBetaInitTests` at `:610`); every other class here is untouched.

- `StandardPathForwardPinTests` (pinned forward bits)
- `BehaviorFingerprintTests`
- `BlockGroupArchitectureTests` (golden summaries)
- `SEActivationTests` (X2 lines only), `RezeroAlphaCapTests`, `SEBetaInitTests` (X2 line only), `InitNeutralOptionsTests` (version-gate patterns)
- `LineageRecordTests` (X2 line only)
- `LayerHealthTests`
- `NetworkArchitectureTests`
- `PolicyTailPrecisionTests`, `HeadNumericsTailTests`
- `ArchitecturePresetStoreSaveTests`
- `GraftDeriveTests`
- `BuildNewModelDraftTests`, `BuildNewModelGroupRemovalRenderTests`, `BuildNewModelTowerShapeRenderTests`
- `documentation/dashboards/tests/test_tooling.py`

---

# Part P — Phasing

Each phase builds on its own and is committed after it builds and its targeted tests pass (the owner's standing rule for approved multi-phase plans; no push). One build at the end of each phase, through `drews-xcode-mcp` `build_project`. No test or CLI run while a training run is live.

### P1 — Format v9, graph, layer health, `--set-activation`, Python readers (T1–T4, T7-P1, T8-P1, T9-P1, T11)
P1 is the first commit whose builds write v9 files, so the Python readers move with it (T11). No commit leaves the tooling unable to read what the app writes.
- All of T1–T4.
- `SetActivationDeriveOperation`.
- The minimal Build screen and diagram edits that the rename requires.
- The `deriving-models.md` `--set-activation` row and prose (`:57`, `:74-78`).
- The X2 edits except `SEBetaInitTests.swift:610`, after OD-6 approval.
- The new pure and GPU test files (X1), except the derive-setter, render and Build-screen picker cases.
- All of T11: `dcm_arch.py`, the reusable readers the owner picks under OD-9, the new Python test, and the frozen-script note (written into `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20 in P4; until then this plan's T11 table is the record).
- Gate:
  - build;
  - targeted runs of X1 (P1 part) and every X2 and X3 class (`LineageRecordTests` included);
  - `python3 -m unittest discover -s documentation/dashboards/tests` (all pass, including the committed `test_tooling.py`);
  - `relu_inputs.py` re-run on the README's three checkpoints reproduces `relu_inputs_bot.json` byte for byte;
  - then the **full suite** (graph builder and persistence changed, per CLAUDE.md "Running the tests"), with the test plan's slow gate as configured.

### P2 — Per-site derive setters (T7-P2)
- `Persistence/SiteActivationDerive.swift`, the catalog entries, `deriving-models.md` (the rest of T7), the `SEBetaInitTests.swift:610` edit and the X1 derive cases.
- Gate: build; `-only-testing:` `ArchitectureActivationSiteTests`, `SEBetaInitTests`, `SEActivationTests`, `LeakyReLUTests`, `DeriveTrainedSourceTests`, `GraftDeriveTests`, `RezeroAlphaCapTests`, `InitNeutralOptionsTests`.

### P3 — Build New Model and diagram (T8, T9)
- `ArchitectureSiteActivationPicker.swift`, the pickers and the "Use for every activation" menu, the diagram lines, and the X1 Build-screen and render cases.
- Gate: build; `BuildNewModel*` tests and the X1 Build-screen cases; plus a screen check (V7).

### P4 — Documentation (T12)
- `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20, `CLAUDE.md` (after OD-7), `CHANGELOG.md`, and this plan's status line.
- Gate: the full suite once more if any code changed after P1's full run; the V-series below.

---

# Part V — Validation

Before P1 starts, freeze the build of P1's parent commit as `~/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-<build>-<hash>.app` (as the existing `DCM-2323-58e9f952-lrmax10.app` was). Call it `OLD`. `NEW` is the post-P1 (and later post-P4) build's binary. `S` is a scratch folder. `M="$HOME/Library/Application Support/DrewsChessMachine/Models"`. Run only when no training run is live.

**V1. Build.** Each phase's `build_project` reports `build_failed: false`.

**V2. Tests.** Per phase, as listed in Part P. At the end, the full suite passes (CLAUDE.md: about an hour cold). Every X3 class passes, unchanged apart from the X2 edits listed for it.

**V3. Existing models give bit-identical outputs (real files).** For each evidence file (the v8, v5, v4-leaky, v3-gelu, v3-post-act and v3-simple_conv rows above):
- run `"$OLD" --probe-model <file> --probe-set 200 --probe-out "$S/old-<n>.jsonl" --probe-positions-out "$S/old-pos-<n>.jsonl"`, and the same with `"$NEW"` into `new-*`;
- compare the two positions files line by line with a short `python3` `json` script, dropping only keys that name the build, the time or a path. Every remaining value is equal (exact float text);
- `NEW`'s session log has one `[ARCH] legacy file (format vN) <file>: …` line with the six site resolutions, each `:= <that file's activation_function>`: `gelu` for T97X, `leaky_relu` for the v4 leaky file.

**V3b. A resume across the change keeps its fingerprint (OLD vs NEW).**
1. With `OLD`, run a short corpus replay from a v8 model so that OLD records its own behavior fingerprint in a trainer file. For example: `"$OLD" --replay-corpus <the corpus named in 20261005-r7b24-fresh's runs' lineage records> --start-model "$M/20261005-r7b24-fresh.safetensors" --training-step-limit 2 --out-model "$S/old-trainer.safetensors"`. Its lineage record carries `rng.behavior_fingerprint` (recipe and SHA-256; `Persistence/LineageRecord.swift:517-528`).
2. With `NEW`, resume it exactly: `"$NEW" --replay-corpus <same> --start-model "$S/old-trainer.safetensors" --resume-exact --training-step-limit 4 --out-model "$S/new-trainer.safetensors"`.
3. Expected: `NEW`'s session log reports the build change as `[RESUME] build changed …` with `behavior fingerprint matches` (`Training/ResumeExactness.swift:113-121`), and its `[RESUME]` verdict carries no `build` gap.
4. `new-trainer.safetensors` is `dcm_format_version` `"9"`, and its record's fingerprint SHA-256 equals OLD's.

**V4. Layer health on an existing model is unchanged.**
- `"$OLD" --analyze-numerics "$M/20261005-r7b24-fresh.safetensors" --numerics-static-only --numerics-out "$S/old-num"`, and the same with `NEW` into `new-num`.
- The layer-health table (`act` column, counts, extremes) is identical.

**V5. Mint and derive a leaky-head model on a ReLU tower.**
1. `"$NEW" --new-model --architecture v4_5block_7x7 --init-seed 20261005 --out-model "$S/base.safetensors"`.
2. `"$NEW" --derive-model --from "$S/base.safetensors" --set-policy-head-activation leaky_relu --set-value-head-conv-activation leaky_relu --set-value-head-hidden-activation leaky_relu --out "$S/leaky-heads.safetensors"`.
3. Dump both headers (stdlib `json` + `struct` read of the safetensors header, as in `HPARAM_RECORDING_PLAN.md`'s dumper) and check:
   - `dcm_format_version` = `"9"`;
   - all six keys present; no top-level `activation_function`;
   - `stem_activation` / `tower_end_activation` / `feature_skip_activation` = `relu` in both;
   - the three head keys `leaky_relu` only in the derived file;
   - `derivation_history`'s last record (one record per derivation, one operation entry per operation, `Persistence/ModelDerivation.swift:369-373`) has exactly three operations, in catalog order: `set-policy-head-activation`, `set-value-head-conv-activation`, `set-value-head-hidden-activation`. Each lists only its own field in `changed_architecture_fields` and has an empty `rewritten_tensors`, with `arguments` `{"value": "leaky_relu"}`;
   - every tensor's bytes are identical between the two files.
4. Write the derived file's architecture into a v9 `NamedArchitecture` preset JSON in `$S` and run `"$NEW" --new-model --architecture "$S/leaky-heads.json" --init-seed 20261005 --out-model "$S/leaky-heads-minted.safetensors"`. Every trainable tensor and every BN running statistic is bit-identical to `base.safetensors` (D7: only head activations changed).
5. Refusals: `--set-stem-activation leaky_relu` on `base` (pre-act) exits non-zero naming the stem's absence. Repeating step 2's flags on `leaky-heads.safetensors` exits non-zero as a no-op.

**V6. Per-site layer health.**
- `"$NEW" --analyze-numerics "$S/leaky-heads.safetensors" --numerics-static-only`. The table lists `tower_final_bn relu`, `policy.pre_bn leaky_relu`, `value.bn leaky_relu`, and the value hidden layer as `leaky_relu`.
- A derive with `--set-value-head-conv-activation silu` shows `value.bn` as `silu` with `n/a` counts, while `tower_final_bn` stays classified.

**V7. Build New Model.** On an idle machine, open the screen. Check that:
- every site picker is present, "Fusion activation" included with the feature-skip source at `none`;
- "Stem activation" is disabled on the default (pre-act) tower and enabled after switching group 1 to `post`;
- "Tower-end activation" is disabled with a post last group;
- "Pre-block activation" is disabled on `simple_conv`;
- "Fusion activation" is enabled only for `compress_conv_bn_relu` with a routed head;
- "Use for every activation ▸ leaky_relu" makes the summary read ` . act leaky_relu` and every group `leaky_relu/pre`;
- setting only the value pickers to `leaky_relu` gives the listed-sites summary and the diagram's per-site lines;
- Save as Preset writes `format_version: 9` with all six keys, and reloading it restores the same picker values.

**V8. Python.**
- `python3 -m unittest discover -s documentation/dashboards/tests` passes.
- `dcm_arch.site_activations_md` on `leaky-heads.safetensors` returns the expected six values; on `T97X` it returns six `gelu`.

---

# Owner decisions needed

**Owner answers (2026-10-05):** OD-1 approved (rename to `stem_activation`); OD-2 approved (own `feature_skip_activation`); OD-3: name the value FC1 field `value_head_fc1_hidden_activation`; OD-4: an explicit `does_not_apply` value (owner: "none" reads ambiguously) — required exactly when the topology lacks the site, refused on a site that exists, never meaning an identity activation (design sections to be revised to this); OD-5 approved; OD-6 approved; OD-7 approved; OD-8 approved; OD-9: both — document the frozen scripts **and** add the guard to each silent one; OD-10: re-save the presets at v9 (approved, optional); OD-11: add to `ROADMAP.md` (approved).

| # | Decision | Recommendation |
|---|---|---|
| OD-1 | Rename the top-level `activation_function` → `stem_activation` (Swift `activationFunction` → `stemActivation`), or keep the key and the Swift name with the narrowed meaning "stem activation" | **Rename** (D1). Keeping it gives one key two meanings across format versions, and three committed tests would keep compiling while testing something else (X2). |
| OD-2 | Give the feature-skip compress node its own `feature_skip_activation`, or (b) let it follow the stem field, or (c) follow the tower-end field | **Own field.** (b) is the dual meaning the brief rules out. (c) fails for a post-act tower with a compress node, which has a fusion activation but no tower-end activation. |
| OD-3 | Name the value FC1 field `value_head_hidden_activation` (pairs with `value_head_hidden_units`) or `value_head_fc1_activation` | `value_head_hidden_activation` |
| OD-4 | Sites the topology lacks: keep any value, never read it (D3 option 1); pin; or Optional/`null` | **Keep, never read** |
| OD-5 | Refuse a v9+ block-groups architecture that still states a top-level `activation_function` (`FormatError.retiredField`) | **Refuse**; the uniform-tower form is exempt |
| OD-6 | Approve the test edits in X2: 11 tests in 6 files, 14 lines in P1 (13 forced by the rename, plus `LineageRecordTests.swift:114`'s format pin) and the catalog pin `SEBetaInitTests.swift:610` in P2 | Approve. Each is mechanical and keeps the test's intent. |
| OD-7 | Approve the `CLAUDE.md:136` text in T12 | Approve |
| OD-8 | Add the Build screen's "Use for every activation" menu | Yes. It is the `--set-activation` rule, from one shared function. |
| OD-9 | Python scope: update `dcm_arch.py` and `relu_inputs.py` (in the brief) and the other reusable readers `fwd16.py` and `units.py`. Frozen scripts: leave them unedited and document (T11 table; some fail loudly, others would silently model ReLU), or add a one-line guard to each silent one. | Update the reusable readers; leave frozen scripts unedited and document them |
| OD-10 | Re-save the user presets in `Presets/` at v9 | Not needed. They load unchanged; re-saving only removes their `[ARCH] legacy` line. Owner's choice. |
| OD-11 | Add this plan to `ROADMAP.md` | Owner's call (standing rule) |

---

# Risks

- **Older builds cannot read v9 files.** Any file the new build writes (`unsupportedFutureVersion`), including a resumed training run's next checkpoint, cannot be read by an older build. The live lrA/lrB runs are unaffected while they keep running on their own binary. Resuming one of them with the new build makes every later file v9-only. This is the same as every earlier format bump.
- **Copied BN statistics after a derive.** A `--derive-model` that changes the stem, tower-end or fusion activation of a *fresh* net copies BN running statistics calibrated under the old activation (D7). This already happens with `--set-activation`. Training replaces them with momentum updates within the first steps, and both arms of an A/B still start from identical trainables, which is what derive is for.
- **Stale enum name.** `FeatureSkipFusion.compressConvBNReLU` / `compress_conv_bn_relu` names ReLU, but its activation becomes `feature_skip_activation`. The token is persisted, so it is not renamed (non-goal). The Build screen caption already says "1×1-conv→BN→act" (`BuildNewModelView.swift:172-175`).
- **Unequal architectures with the same graph (OD-4).** Two architectures that differ only in an absent site's value compare unequal. The only visible effect is the Build screen's preset match.
- **Bit-identity across graphs (X1).** If MPSGraph compiles the shared tower differently when only one head's activation changes, the "only the head it feeds changes" test reports ulp differences. That is reported, not hidden behind a tolerance.
- **Smooth or leaky tower-end in bf16/fp16.** These are new numerics paths for training. `testLeakyHeadsTrainOneFiniteStep` covers bf16 under both tail precisions. fp16 inherits `LeakyReLUTests`' tolerance handling. Any experiment watches `[LAYER-HEALTH]` and `pLogitMean` / `vLogitMean` as usual.
- **Python scripts on v9 files.** Python that reads the top-level key on a v9 file raises `KeyError`. This is intended: loud, never silent (T11).
- **Test time.** The full suite takes about an hour; P1 and the end each need one full run on an idle machine.

---

# Non-goals

- Per-block activations within a group (a group remains one recipe; use count-1 groups).
- Learnable or per-site leaky slopes (PReLU), and making `leakyReLUNegativeSlope` a field.
- New activation functions.
- The SE gate (sigmoid) and the value output (softmax / tanh), which stay structural.
- Changing any default: presets, `newModelDefault` and `--new-model` builds stay ReLU everywhere. Whether a head should default to leaky is an experiment for later.
- Rewriting or migrating existing files, presets or experiment JSONs.
- Renaming persisted tokens (`compress_conv_bn_relu`) or graph op names (`tower_final_act`, `value_fc1_act`, …).
- Changing the legacy `.dcmmodel` reader or writer.
- Any search-based move selection (the project's standing non-goal).

---

# Review reconciliation

## Pass 1 (2026-10-05)

Every item was re-verified against the code before it was acted on.

**Must-fix items**

| # | Item | Verified | Outcome |
|---|---|---|---|
| 1 | `LineageRecordTests.swift:114` pins format `"8"` and is missing from X2 | Yes: `XCTAssertEqual(md[SafetensorsModelIO.Key.formatVersion], "8")`. No other test pins the number (searched for `"8"` next to format/version) | **Accepted.** Added to X2 (→ `String(ArchitectureFormat.currentVersion)`), OD-6 and P1's targeted list. Recounted: 11 tests in 6 files, 14 lines in P1 plus the P2 catalog pin. |
| 2 | `setMainActivationEverywhere` vs the convenience init contradicts `BlockGroup.setActivationFunction` keeping an SE group's `se_activation` | Yes, `NetworkArchitecture.swift:648-653` | **Accepted.** The equality test is restricted to SE-less towers, and `testSetMainActivationEverywhereKeepsEverySEGroupsFC1Activation` pins SE retention. D2 states the rule. T7's "equals a leaky convenience-built tower" now holds only for SE-less towers, or with `--set-se-activation`. The setter's semantics are unchanged. |
| 3a | Layer-health test asked for five different functions; there are four | Yes, `NetworkArchitecture.swift:230-238` | **Accepted.** Replaced by a site × function matrix, one site at a time, for the five BN sites and value FC1. |
| 3b | Copying the direct-consumer/name walk proves nothing about the built function and fails for GELU | Yes. The GELU final multiply reads `<name>_halfx` and `<name>_1plus` (`ChessNetwork.swift:2390-2400`), and the walk at `LayerHealthTests.swift:233-262` checks only names. | **Accepted, numeric variant.** `testEachSiteBuildsItsSelectedFunction` compares tapped site outputs with a CPU computation of the selected function from tapped inputs, for all six sites (value FC1 included), with an asserted margin between functions. |
| 4 | The fusion picker inside `featureSkipSource != .none` contradicts "always present" | Yes, `BuildNewModelView.swift:165-180` | **Accepted.** The picker goes outside the conditional, disabled through `siteExists(.featureSkipFusion)`. The render test and V7 check its presence with the source at `none`. |
| 5 | V5 expected one operation to list three fields | Yes: one `OperationRecord` per operation, each with its kind's `changedArchitectureFields` (`ModelDerivation.swift:369-373`) | **Accepted.** V5 checks the last record's three operations one by one. |
| 6 | The claim that excluded Python scripts all fail loudly is false | Yes. `fwd.py:87-91` and `fatty_tensors.py:25-29` never read the key (T11 table counts each). | **Accepted.** T11 separates reusable readers (guarded in P1) from frozen scripts (documented in §20, with a per-script loud/silent table). Guarding the silent ones too is the OD-9 alternative. |

**Missing items and clarifications**

| Item | Outcome |
|---|---|
| Tests for stated-wins, partial presence, invalid tokens, uniform-tower with stated site keys | **Accepted.** Four new X1 format cases and a Python unknown-token case. |
| D5 uniform expansion should say "rule 1, then rule 2" | **Accepted.** D5 now says so and covers unknown tokens. |
| How site activations reach `relu_inputs.forward`; reject unsupported feature skips | **Accepted.** Resolved once in `run` after `load`, passed as a `sites` argument. `forward` asserts `feature_skip_source == none` (from `dcm_arch.norm_arch_md`). |
| X3/V2 "unmodified" contradicts the X2 edits | **Accepted.** Both say "unchanged apart from the X2 edits listed for it". |
| P1 writes v9 while P4 repaired readers | **Accepted.** All Python work moved into P1 (the old P4 is gone; docs are now P4). No intermediate commit leaves the tooling behind. |
| D3's "only preset matching notices" is wrong: equality keys the fingerprint cache | **Accepted.** D3 names the cache (`BehaviorFingerprint.swift:76-86`, `:134-142`) and its cost: one redundant computation, never a wrong result. |
| OLD vs NEW fingerprint comparison for a legacy checkpoint | **Accepted.** V3b: an OLD-written v8 trainer file resumed exactly by NEW must report `behavior fingerprint matches` with no `build` gap. D7 now cites V3b instead of claiming it from same-build tests. |

**Rejected:** none.

## Pass 2 (2026-10-05)

The reviewer confirmed every pass-1 change except three, each re-verified against the code.

| # | Item | Verified | Outcome |
|---|---|---|---|
| 1 | `fwd16.py` cannot resolve sites: `forward` gets an already-normalized architecture with no format version, and normalization erases the uniform-tower form | Yes. `forward` (`fwd16.py:118`) and `forward_batched` (`:196-198`) take only `arch`. All six callers build it with the version-less `norm_arch(md['architecture'])` (`sat16.py:12`, `shared16.py:12`, `grad16.py:33`, `range16.py:22`, `tails16.py:16`, `tailstart16.py:15`). | **Accepted.** T11 now: sites resolved from raw `md` in each caller before normalization (plus `norm_arch_md`), passed as a required keyword-only `sites` through `forward_batched` and `forward`, every call site listed for P1. Topology-aware checks (stem only post, tower end only pre, policy only with a pre-block, value always), and feature skips rejected. `grad16.py` moved out of the frozen "silent" row, since it is now covered through `fwd16`. |
| 2 | The `units.py` guard omits value FC1 | Yes. `:262-263` and `:276` label FC1 and its reader ReLU. | **Accepted.** New `require_supported_architecture(ckpt)`: rejects feature skips first, then resolves `site_activations_md(ckpt.metadata)` and requires ReLU at tower end, policy pre-block, value conv and value hidden. Rejection tests cover a leaky value-hidden site and an enabled feature skip, among others. |
| 3 | The SE test asserted a mixed fixture equals the convenience-built `gelu` tower, which is impossible since the convenience init builds one uniform group | Yes, `NetworkArchitecture.swift:1199-1233` | **Accepted.** The mixed fixture is compared with an explicitly constructed expected architecture that keeps its three groups. The convenience-init equality moved to a separate single-group SE fixture (`testSetMainActivationEverywherePlusSEActivationEqualsTheConvenienceInitOnAnSETower`). |

**Rejected:** none.

