# Per-site activations plan: the stem, tower end, feature-skip fusion, policy head and value head each choose their own activation

Status (2026-10-05): **IMPLEMENTING.** Phase status:
- [x] **P1** — format v9, `does_not_apply`, graph, layer health, `--set-activation`, Build screen pickers, Python (`4e70c615`).
- [x] **P2** — per-site derive setters (commit recorded in **Implementation notes**).
- [ ] **P3** — "Use for every activation" and the diagram.
- [ ] **P4** — documentation, presets, ROADMAP completion.

The original planning status, kept for the record: **PLAN ONLY.** Nothing here was implemented when the review passes below ran.
- Independent review (2026-10-05): **concurred after three passes** on the first design. First pass: the architecture direction was approved. Six must-fix items and nine clarifications were raised, and each is listed with what was done about it in **Review reconciliation** at the end.
- Revised the same day for the owner's answers (see **Owner decisions**): the value FC1 field is `value_head_fc1_hidden_activation` (OD-3); a site the topology lacks holds the explicit value `does_not_apply`, checked in both directions (OD-4, replacing the earlier "keep any value, never read it" design); the frozen Python scripts that would silently model ReLU get a guard (OD-9); the user presets are re-saved at v9 (OD-10); `ROADMAP.md` gets an entry (OD-11). The revised design (with `does_not_apply`) has had its own review passes, recorded as passes 4 onward in **Review reconciliation** with the verdict of each: external review passes 4 and 5 (must-fix items, all applied), then, with the external reviewer out of quota, an independent agent review that **concurred**; its fixes are applied.
- Open owner decisions raised by the revision: OD-12, OD-13, OD-14 (see **Owner decisions**). **OD-12 and OD-14 block P1**; OD-13 does not. OD-15 records the interaction with the alarms plan and needs no decision here.
- Every `file:line` was checked against `main` at `c0130599` and still resolves at `a0bc4051` (no code, script or test changed since `dc6a4785`; later commits touch only plans and `experiments/20261005-lr-schedule-ab/`).
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/` (project folder), `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/`, `experiments/` or `scripts/`.
- Real-file evidence comes from header-only reads of `~/Library/Application Support/DrewsChessMachine/Models/*.safetensors` and `Sessions/*/champion.safetensors`.
- Modelled end to end on how `se_activation` (format v5, issue #2, commit `de0f22be`) was added: `documentation/plans-active/RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §18, `SEActivationTests`.

**The request.** Make the activation function configurable per site. Today a block group's main path (`block_groups[].activation_function`) and its SE FC1 (`block_groups[].se_activation`) are already independent. One top-level field, `activation_function` (`NetworkArchitecture.activationFunction`, `Network/NetworkArchitecture.swift:1083-1086`), drives every other hidden activation at once.

**What this plan does.**
- Splits the top-level field into one field per site: stem, tower end, feature-skip fusion node, policy head, value-head conv and value-head FC1 hidden layer.
- Adds the value `does_not_apply` to `ActivationFunction`. A site field holds it exactly when the model's topology lacks that site, and a real function exactly when the site exists. Both directions are checked on every load and every edit.
- Bumps the architecture format to v9. Every older file resolves each new field to its own `activation_function` where the site exists and to `does_not_apply` where it does not, so every existing model builds the identical graph and compares equal to its preset.
- Adds per-site `--derive-model` setters, Build New Model pickers, per-site layer-health tagging, per-site Python mirrors and a guard in each frozen Python script that would otherwise silently assume ReLU.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth;
- no silent defaults, no fallbacks, no `try?`, no force unwraps;
- no migration code: older files are decoded under the format-version gate the way the project already decodes them, and are never rewritten (the owner-approved re-save of the user presets, OD-10, goes through the app's own save path after a backup);
- tests are never modified or deleted without the owner's approval (every needed edit is listed in X2 under OD-6 and OD-12);
- one SwiftUI `View` per file; no helper `some View` properties (new UI is a child `View` struct in its own file);
- no app, test or CLI runs while a training run is live (validation waits for an idle machine).

---

## Summary

| # | Item | Phase |
|---|---|---|
| 1 | Six architecture-level site fields replace the one top-level `activation_function`: `stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_fc1_hidden_activation` | P1 |
| 2 | `ActivationFunction` gains `does_not_apply`. A site field is `does_not_apply` exactly when the site does not exist; block groups and SE FC1 never accept it. Checked on decode, in `validate()` and in every activation setter. | P1 |
| 3 | Architecture format v9. Older files resolve each new field from their own `activation_function` (existing sites) or to `does_not_apply` (absent sites). A v9 file missing one is a load error. | P1 |
| 4 | `ChessNetwork` reads each site's own field. The tower-level `activation(_:_:arch:name:)` overload is deleted; reaching `does_not_apply` in the graph builder is a programming error. | P1 |
| 5 | `LayerHealth` tags each BN site and the value FC1 layer with that site's own activation | P1 |
| 6 | `--set-activation` means "this activation at every existing hidden site" (every existing architecture-level site plus every group); `--set-activation` and `--set-se-activation` refuse `does_not_apply` | P1 |
| 7 | Build New Model: one picker per site (disabled where the site does not exist); a site that appears must be chosen before Build or Save | P1 |
| 8 | Python mirrors: `scripts/dcm_arch.py` gate + `site_activations`; site checks in the reusable forward-pass readers (`relu_inputs.py`, `fwd16.py`, `units.py`); a guard in each silent frozen script; frozen scripts documented | P1 (P1 is the first commit that writes v9) |
| 9 | Six per-site `--derive-model` setters (`--set-stem-activation`, …) | P2 |
| 10 | Build New Model "Use for every activation" menu (OD-8); the diagram shows each site's activation | P3 |
| 11 | `ROADMAP.md` entry (OD-11): added in P1, marked complete in P4 | P1, P4 |
| 12 | Docs: `deriving-models.md`, `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20, CLAUDE.md (OD-7), CHANGELOG; re-save the user presets at v9 (OD-10) | P4 (P1/P2 keep `deriving-models.md` in step) |

No tensor, parameter count, weight-init draw, model ID, content hash or default changes. Presets and new models stay ReLU at every existing site.

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
| Value FC1 hidden | `value_fc1_act` | `ChessNetwork.swift:3495` | `value_fc1` + bias | always | hidden layer `value.fc1.weight` (`:242-252`) |

The brief listed five sites. The feature-skip fusion node (`ChessNetwork.swift:913-935`) is a sixth that also reads the top-level field. Leaving it there would give the renamed stem field two meanings, so it gets its own field (OD-2).

The other activations are already per-group or structural, and this plan does not change how they are chosen:
- the block main path and `activation_gated` merge (`ChessNetwork.swift:2856`, `:2865`, `:2878`, `:2991`: `spec.activationFunction`);
- the SE FC1 (`:3051`: `spec.seActivation`);
- the SE gate (sigmoid) and the value output (softmax / tanh).

Both group fields are of type `ActivationFunction`, so the new `does_not_apply` value must be refused there (D3).

### Who else reads the top-level field or lists every `ActivationFunction`

| Reader | Where |
|---|---|
| Format decode / encode / uniform-tower legacy expansion | `NetworkArchitecture.swift:1313`, `:1403`, `:1364-1394` |
| Memberwise init / uniform convenience init | `:1135-1173` / `:1177-1253` (`:1214`, `:1226`, `:1235`) |
| Summary clause ` . act <fn>` | `:1924` |
| `--set-activation` | `Persistence/ModelDerivation.swift:738-806` (value list `:763`, `:775-778`; apply `:788-798`) |
| `--set-se-activation` value list | `Persistence/ModelDerivation.swift:824`, `:832-835` |
| Build New Model "Tower-level activation" picker | `App/UpperContentView/BuildNewModelView.swift:93`; model field `BuildNewModelModel.swift:51`, `:103`, `:135`, `:158` |
| Build New Model group "Activation" / "SE activation" pickers (`ActivationFunction.allCases`) | `BuildNewModelView.swift:451`, `:457`; `enumPicker`'s `CaseIterable` constraint `:566-572` |
| Diagram stem and tower-end lines | `App/UpperContentView/ArchitectureDiagramView.swift:53`, `:68` |
| Layer health | `Training/LayerHealth.swift:172`, `:201`, `:205`, `:212`, `:215`, `:251`; `classification(for:)` `:274-282` (exhaustive switch) |
| Graph builder | `ChessNetwork.swift:2379-2401` (exhaustive switch) |
| Tests | 13 lines in 4 files read the top-level field; 2 lines iterate `ActivationFunction.allCases` (X2) |

`ActivationFunction` (`NetworkArchitecture.swift:221-244`) has `relu`, `silu`, `gelu`, `leaky_relu`, and is `CaseIterable`. Leaky ReLU has a fixed slope, `ActivationFunction.leakyReLUNegativeSlope = 0.01` (`:243`), applied by `graph.leakyReLU(with:alpha:)` (`ChessNetwork.swift:2385-2386`). The slope is a constant, not a field, and this plan keeps it that way.

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
- No file has a compress fusion node, so every file resolves `feature_skip_activation` to `does_not_apply`.
- 4,350 files have no stem activation (pre-activation first group): `stem_activation` resolves to `does_not_apply`.
- 27 files have no tower-end activation (all are post-activation v3, simple_conv).
- 298 files have no policy pre-block.
- The gelu and leaky files show why the legacy resolution of an existing site must copy the file's own value, never `relu`.

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

| JSON key | Swift property (`NetworkArchitecture`) | `ArchitectureActivationSite` case | Graph op it drives | Site exists when |
|---|---|---|---|---|
| `stem_activation` | `stemActivation` | `stem` | `stem_act` | first group is `post` |
| `tower_end_activation` | `towerEndActivation` | `towerEnd` | `tower_final_act` | last group is `pre` |
| `feature_skip_activation` | `featureSkipActivation` | `featureSkipFusion` | `feature_skip_act` | `featureSkipUsesCompressNode` |
| `policy_head_activation` | `policyHeadActivation` | `policyHead` | `policy_pre_act` | `policy_head_style` is `intermediate_conv` or `fc_bottleneck` |
| `value_head_conv_activation` | `valueHeadConvActivation` | `valueHeadConv` | `value_act` | always |
| `value_head_fc1_hidden_activation` | `valueHeadFC1HiddenActivation` | `valueHeadFC1Hidden` | `value_fc1_act` | always |

Why these names:
- **`stem_activation`** replaces the top-level `activation_function` (OD-1). After the split the old key would mean "the stem activation, which every model on disk but 27 does not have". A key whose meaning narrows silently is exactly what the brief rules out. Renaming means:
  - an old reader (Python, a person reading JSON, an older build) can never mistake a v9 value for the old tower-wide one;
  - the Swift rename makes the compiler list every reader. Four test files would otherwise keep compiling with a different meaning: two tests silently test nothing and one fails at run time (OD-6).
- **`tower_end_activation`** matches the code's `hasTowerEndBN` and the UI's "tower-end BN". The graph op stays `tower_final_act`; graph op names never change in this plan.
- **`feature_skip_activation`** sits with the other `feature_skip_*` keys (OD-2).
- **`policy_head_activation`** sits with `policy_head_style` / `policy_head_final_init`. `policy_pre_activation` was rejected: "pre-activation" already means the block style.
- **`value_head_conv_activation`** pairs with `value_head_conv_channels`.
- **`value_head_fc1_hidden_activation`** is the owner's name (OD-3): it names the layer (FC1) and its role (the hidden layer whose width is `value_head_hidden_units`). The Swift property follows the project's acronym style (`valueFC1Layer`, `Training/LayerHealth.swift:245`): `valueHeadFC1HiddenActivation`.

## D2. Types

- **`ActivationFunction` gains one case** (OD-4): `case doesNotApply = "does_not_apply"` (the raw value matches the snake_case of `leaky_relu`). Its doc comment (`NetworkArchitecture.swift:221-229`) is rewritten to list the per-site fields and to state:
  - `does_not_apply` is not a function. It marks an architecture-level site the model's topology does not have. It never means identity or linear.
  - It is legal only in the six site fields, and there exactly when the site does not exist (D3).
- **`ActivationFunction` keeps `CaseIterable` (OD-14, owner's decision), and gains `static let functions: [ActivationFunction]` = `allCases` without `does_not_apply`**, documented as "every case that names a function: the list every picker, every derive value syntax and every per-function test iterates". Why the list is needed:
  - `allCases` now includes `does_not_apply`, so every list a person or a CLI chooses a function from must use `functions` instead: the group pickers (`BuildNewModelView.swift:451`, `:457`), the site pickers, and the `--set-activation` / `--set-se-activation` value syntax and parse (`ModelDerivation.swift:763`, `:775-778`, `:824`, `:832-835`), plus the two test loops (X2);
  - with the conformance kept, the compiler does not find a future `allCases` use. The guard is a test (`testEveryActivationChoiceListIsTheFunctionsList`, X1) pinning that `functions` excludes `does_not_apply` and that each picker list (`ArchitectureSiteActivationPicker.functionChoices`, `BuildNewModelView.groupActivationChoices`) and each derive value syntax equals `functions`, plus review.
  - `enumPicker` (`BuildNewModelView.swift:566-572`) keeps its `CaseIterable` constraint; `ActivationFunction` still satisfies it.
- Each new site field is a non-optional `ActivationFunction`.
- New `enum ArchitectureActivationSite: CaseIterable, Hashable, Sendable`, in graph build order: `stem`, `towerEnd`, `featureSkipFusion`, `policyHead`, `valueHeadConv`, `valueHeadFC1Hidden`. It follows the `InitOptionField` precedent (`NetworkArchitecture.swift:408-452`). Each case provides:
  - `jsonKey`, read from `NetworkArchitecture.CodingKeys` (the single source of every key);
  - `summaryLabel` (`stem`, `tower_end`, `fusion`, `policy`, `value_conv`, `value_fc1_hidden`);
  - `displayName` (the Build screen picker label: "Stem activation", "Tower-end activation", "Fusion activation", "Pre-block activation", "Conv activation", "FC1 hidden activation");
  - `absentReason` (why the site is missing, for errors and help text, e.g. "the first block group is pre-activation, so the stem has no activation").
- New `NetworkArchitecture` members (new extension, beside the init-neutral extension at `:2152`). These are the single rules every consumer reads; none duplicates a site condition:
  - `func hasActivationSite(_:) -> Bool`, built from the existing `hasStemActivation`, `hasTowerEndBN`, `featureSkipUsesCompressNode` and `policyHeadStyle`. It never expands the tower, so the Build screen can call it on any draft. (Decode never reaches it with zero groups: `init(from:)` already refuses an empty `block_groups`, `:1355-1360`.)
  - `func activation(at:) -> ActivationFunction`: a switch over the six stored properties, which stay the single source of truth.
  - `var activationSiteMismatch: ActivationSiteMismatch?`: the first site, in build order, whose value disagrees with its existence (`does_not_apply` at an existing site, or a function at an absent one). `ActivationSiteMismatch` (`site`, `value`, `siteExists`; `Equatable` and `Sendable`, since both `ArchitectureFormat.FormatError`, `ArchitectureFormat.swift:124`, and `NetworkArchitectureError`, `NetworkArchitecture.swift:977`, are `Equatable` and carry it) carries the one `description` every error uses, naming the JSON key, the site and the reason. Examples:
    - `policy_head_activation is 'does_not_apply', but the policy head has a pre-block (intermediate_conv): choose one of relu, silu, gelu, leaky_relu`;
    - `stem_activation is 'relu', but the first block group is pre-activation, so the stem has no activation: it must be 'does_not_apply'`.
  - `mutating func setActivation(_:at:) throws`: refuses a value that disagrees with the site's existence (`NetworkArchitectureError.activationSiteMismatch`), otherwise sets the field.
  - `mutating func setActivationAtEveryExistingSite(_:) throws`: refuses `does_not_apply` (`NetworkArchitectureError.notAnActivationFunction(context:)`) and sets every site where `hasActivationSite` is true. It never touches an absent site.
  - `mutating func setMainActivationEverywhere(_:) throws`: refuses `does_not_apply` before changing anything (the architecture is untouched on failure), then `setActivationAtEveryExistingSite` plus every group's `BlockGroup.setActivationFunction` (`:640-653`). This is the `--set-activation` rule, shared by the derive operation and the Build screen's "Use for every activation", so an edit made either way gives the same architecture (the same guarantee `setActivationFunction` gives today).
    - `setActivationFunction` keeps a group's `se_activation` when the group has an SE block (`:648-653`), so `setMainActivationEverywhere` does too, exactly as `--set-activation` does today (`documentation/deriving-models.md:74-78`).
    - It therefore equals "the uniform convenience init with that activation" only on an SE-less tower. On a tower with SE blocks it equals that only together with `--set-se-activation` (X1 pins both halves).
  - `mutating func clearActivationSitesTheTopologyLacks()`: sets every absent site to `does_not_apply` and leaves every existing site alone. It never fills a site that exists: a site that has just appeared keeps whatever it holds, which is `does_not_apply` if it was cleared when it last disappeared, and `validate()` then names it until a function is chosen. It is for code that has just changed the topology (the Build screen, D8; tests that flip a style, X2). Decode and `validate()` never call it: a stored mismatch is an error, never repaired.

## D3. Sites the topology lacks: `does_not_apply` (OD-4)

A pre-activation tower has no stem activation, a post-activation tower has no tower-end activation, `simple_conv` has no policy pre-block, and only the compress fusion builds `feature_skip_act`. The owner's answer: such a site's field holds `does_not_apply`, and only such a site's field may hold it.

**The rule** (one function, `activationSiteMismatch`, read by every check):
- site exists ⇔ field is a function (one of `ActivationFunction.functions`);
- site absent ⇔ field is `does_not_apply`.

**Where it is enforced** (both directions, every path; nothing is ever repaired silently):

| Path | Check | Error |
|---|---|---|
| Decode (`NetworkArchitecture.init(from:)`, every loader: safetensors, session, preset, `architecture.json`, catalog) | after the six fields are resolved (D5) | `ArchitectureFormat.FormatError.activationSiteMismatch(mismatch, location:, formatVersion:, source:)`: names the key, the site, the reason and the file |
| `validate()` (every build, `--new-model`, Build screen, preset save, derive, graft) | after `validateInitOptions()` (`:1667`) and the feature-skip checks, before the parameter count (`:1688`) | `NetworkArchitectureError.activationSiteMismatch(mismatch)` |
| `setActivation(_:at:)`, `setActivationAtEveryExistingSite`, `setMainActivationEverywhere` | on the value passed | `activationSiteMismatch` / `notAnActivationFunction` |
| `--derive-model` site setters, `--set-activation`, `--set-se-activation` | at parse and apply (T7) | `DeriveError` naming the flag and the reason |

The order inside `validate()` is deliberate: an architecture with an older, more specific error (for example `branch_output_init` on a pre-activation group, `InitNeutralOptionsTests.swift:207-209`) still reports that error first.

**Groups and SE never accept `does_not_apply`.** A group's main path and an SE FC1 always exist when their group exists. Checked on:
- decode: `BlockGroup.init(from:)` refuses the token in `activation_function` or `se_activation` with `FormatError.doesNotApplyAtAnAlwaysPresentSite(field:, location:, formatVersion:, source:)`. In the uniform-tower form the group's two fields come from the top-level `activation_function`, so the same refusal names that key.
- `validate()`: `NetworkArchitectureError.doesNotApplyAtAnAlwaysPresentSite(field: "blockGroups[i].activationFunction" | "blockGroups[i].seActivation")`, checked first in the group loop.
- `BlockGroup.setActivationFunction(_:)` (`:648-653`) gets `precondition(activation != .doesNotApply, …)` before it changes anything. It cannot throw without breaking its non-throwing caller, the computed `BlockGroupDraft.activationFunction` setter (`App/UpperContentView/BlockGroupDraft.swift:42-45`). Every caller passes a function: the group picker lists `ActivationFunction.functions`, `setMainActivationEverywhere` refuses `does_not_apply` first, and `--set-activation` refuses it at parse and at apply. A `does_not_apply` reaching it is a defect, the same class as the graph builder's case below.
- An SE-less group's `se_activation` keeps today's rule: it must equal the group's `activation_function` (`:1631-1638`). This plan does not change group-level fields (OD-13).

**The graph builder.** `ChessNetwork.activation(_:_:_:name:)` (`:2379-2401`) gets `case .doesNotApply: preconditionFailure("ChessNetwork: 'does_not_apply' reached the graph builder at '\(name)'. validate() refuses it at every site that is built, so this is a defect.")`.
- `ChessNetwork.init` runs `validate()` first (`:602`), and every site call is reached only where the site exists. Reaching this case means the validation or a call site is wrong.
- It follows the project's precedent for "validate() rejects this" (`hasTowerEndBN`, `:1571-1576`). It is never a fallback: no activation, identity or otherwise, is ever built for `does_not_apply`.
- `LayerHealth.classification(for:)` (`:274-282`) gets the same treatment (`preconditionFailure`). Layer health reads a site field only where the site exists (T4), so the case is unreachable.

**Why this is better than the earlier "keep any value" design** (kept here for the record):
- An absent site's activation field can no longer make two otherwise-equal architectures compare unequal: it has exactly one legal value. (This is about these six fields only. Other settings that a given topology does not read, such as `policy_pre_conv_channels` under `simple_conv`, are still stored and compared as before.) So the Build screen's preset match (`BuildNewModelModel.swift:439`) and the behavior-fingerprint cache (keyed by `Settings`, which holds the architecture, `Training/BehaviorFingerprint.swift:76-86`, `:134-142`) never see a twin that differs only in an absent site's activation.
- A reader never has to know which fields are dead: the JSON says so.

**What it costs:**
- Code that flips a topology style after building an architecture must clear or choose the affected sites. In the app that is the Build screen (D8). In the tests it is 7 helper lines (X2, OD-12).
- Every list of choosable activations must use `ActivationFunction.functions`, not `allCases` (D2, OD-14).

## D4. JSON shape (v9)

Encoding writes all six keys on every architecture, `does_not_apply` included, and never writes a top-level `activation_function`. Group keys are unchanged. Example for a pre-activation tower with no feature skip (sorted keys, other keys elided):

```json
{
  "block_groups": [{ "activation_function": "relu", "se_activation": "leaky_relu", "...": "..." }],
  "feature_skip_activation": "does_not_apply",
  "policy_head_activation": "leaky_relu",
  "stem_activation": "does_not_apply",
  "tower_end_activation": "relu",
  "value_head_conv_activation": "leaky_relu",
  "value_head_fc1_hidden_activation": "leaky_relu"
}
```

## D5. Format v9 and the decoding rules

- `ArchitectureFormat.currentVersion = 9` (`Network/ArchitectureFormat.swift:62`). New gate `siteActivationsRequiredFromVersion = 9`, and `DecodeFormat.allowsMissingSiteActivations` (beside `:180-192`).
- Version history comment (`:33-52`) gains: "v9: the architecture must state `stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_fc1_hidden_activation`; each is `does_not_apply` exactly when the topology lacks the site. Older files resolve an existing site to their own top-level `activation_function`, the activation every one of those sites used before the fields existed, and an absent site to `does_not_apply`."
- `SafetensorsModelIO.formatVersion` follows automatically (`Persistence/SafetensorsModelIO.swift:20`). So do presets and `architecture.json` (`format_version`) and `--derive-model` output.
- `NetworkArchitecture.CodingKeys` (`:1269-1299`):
  - `activationFunction = "activation_function"` becomes the decode-only `legacyActivationFunction = "activation_function"`, listed with the other legacy keys;
  - six new keys are added.
- **Two passes inside `init(from:)`.** The site rules need the topology (groups, policy style, feature skip), and Swift allows instance methods only once every stored property is set. So:
  1. **First pass**, after `blockGroups`, `policyHeadStyle` and the `featureSkip*` fields are decoded: each of the six fields is set by `ArchitectureFormat.decodeSiteActivation(key:in:decoder:format:legacyByConstruction:legacyTowerActivation:)`, modelled on `decodeInitOption` (`:250-273`). `legacyTowerActivation` is an `ActivationFunction?` read once with `decodeIfPresent` for the `legacyActivationFunction` key (an eager `decode` would break the re-stamped fixtures of rule 1, which state all six keys and no `activation_function`). In the uniform-tower form it stays required, as today, because the group's own fields come from it.
     - **Stated → used** (rule 1). This holds at any version. Real pre-v9 files never state these keys, but the committed tests build "old" files by re-stamping a current-version encode (`SEActivationTests.swift:104-115`, `RezeroAlphaCapTests.swift:102-115`, `SEBetaInitTests.swift:283-292`), so a stated value has to win, exactly as `decodeInitOption` already allows.
     - **Absent, and the file is older than v9 or in the uniform-tower form** (`isUniformTowerForm`, `NetworkArchitecture.swift:1322`, which no v9 writer emits) (rule 2): the value is provisionally the file's top-level `activation_function`, and the key is marked *resolved*. If the file lacks `activation_function` too, decoding fails with a dedicated `FormatError.legacyActivationFunctionMissing(unresolvedSites:, location:, formatVersion:, source:)`, thrown only when at least one key actually needs resolving. Its message says that a file older than v9 must state either the six site keys or the top-level `activation_function` they resolve from, names the unresolved keys and the file, and makes nothing up. (`missingRequiredField`'s text, "only files written before the field existed may omit it", `ArchitectureFormat.swift:134-137`, would be backwards here: it is v9+ files that omit `activation_function`.)
     - **Absent in a v9+ block-groups file** (rule 3): `missingRequiredField(field: <key>, location:, formatVersion:, source:)`, the same error and message shape as `se_activation` (`ArchitectureFormat.swift:134-137`).
  2. **Second pass**, once `self` is fully initialized:
     - every *resolved* key whose site does not exist (`hasActivationSite` false) becomes `does_not_apply`;
     - each resolution is recorded on the format's legacy log with its final value and reason: `<location.>stem_activation := does_not_apply (the first block group is pre-activation, so the stem has no activation)`, `<location.>tower_end_activation := relu (the file's activation_function)`;
     - then `activationSiteMismatch` is checked. A mismatch is the decode error of D3. It can come only from a stated key (rule 1) or from a legacy file whose `activation_function` is itself `does_not_apply`; rule 2 never produces one.
  - The provisional value in step 1 lives only inside the initializer and is never observable.
- **Retired key (OD-5).** A v9+ block-groups architecture that still carries a top-level `activation_function` is refused with a new `FormatError.retiredField(field: "activation_function", location:, formatVersion:, source:, replacedBy: [six keys])`.
  - Without this, a preset hand-edited from an old one (`format_version` bumped, six keys added, the old key left in) decodes and silently ignores the key its author may think still sets the heads.
  - The uniform-tower form is exempt: it is legacy by construction and is decoded strict-current by committed tests (`BlockGroupArchitectureTests.swift:38-90`, `SEActivationTests.swift:186-204`, `RezeroAlphaCapTests.swift:194-210`).
- **Uniform-tower expansion** (`NetworkArchitecture.swift:1361-1395`):
  - the group's `activation_function` / `se_activation` read the decoded `legacyActivationFunction` (required in that form, as today). A `does_not_apply` there is refused (D3);
  - the six site fields go through rule 1, then rule 2 and the second pass: a site key stated next to the uniform-tower keys (no writer emits that, but a hand-edited file can) is used and checked, and only an absent one resolves;
  - an unknown token in any of the six keys (e.g. `"swish"`) is the `DecodingError` that `ActivationFunction`'s `Decodable` already throws, naming the coding path. No new handling is needed and none is added.
  - the combined `legacy uniform-tower keys: …` log record (`:1388-1394`) lists them too.
- **One `[ARCH]` line per load, as today.** Example for the current-format file in the evidence list (pre-activation, `intermediate_conv`, no feature skip):
  `[ARCH] legacy file (format v8) 20261005-r7b24-fresh.safetensors: stem_activation := does_not_apply (the first block group is pre-activation, so the stem has no activation); tower_end_activation := relu (the file's activation_function); feature_skip_activation := does_not_apply (no compress fusion node); policy_head_activation := relu (…); value_head_conv_activation := relu (…); value_head_fc1_hidden_activation := relu (…)`.
  - Written by the loaders that already call `logLegacyResolutions()`: `Persistence/CheckpointManager.swift:1847`, `:1859`, `:1916-1917`; `App/DeriveModelCLI.swift:219`, `:353`.
  - The model catalog stays quiet. The preset scan (`ArchitecturePresetStore.allPresets`, `Persistence/ArchitecturePresetStore.swift:190-200`) goes through `loadFile`, which logs one `[ARCH]` line per legacy preset (`:155-163`), as it does today for the earlier gated fields. A file whose sites mismatch fails these readers' decode with the D3 error, as any other decode error does (the preset scan logs `[PRESET] Ignoring '<name>.json': …`, `:199-200`).
- Encode (`:1398-1418`) writes the six keys in place of `activation_function`.

## D6. Initializers

- **Full memberwise init** (`NetworkArchitecture.swift:1135-1173`): `activationFunction:` is replaced by the six required parameters, with no defaults. Like every other field it is not validated in the init (`validate()` is). Its only caller is `BuildNewModelModel.architecture` (`App/UpperContentView/BuildNewModelModel.swift:153-175`); no test calls it today.
- **Uniform convenience init** (`:1177-1253`) keeps its `activationFunction:` label. Its meaning stays "the one activation a historical single-recipe tower used at every hidden site". It sets the group, the group's SE FC1 (as today), every site that its topology has to that value, and every other site to `does_not_apply`. Its topology is fully known from its own arguments: the stem exists iff `blockActivationStyle == .post`, the tower end iff `.pre`, the policy pre-block iff `policyHeadStyle != .simpleConv`, and the fusion never (feature skip is off, `:1245-1251`). It is written as "build, then `clearActivationSitesTheTopologyLacks()`", so the existence rule is not restated.
  - Every code preset (`:2387-2532`; the fusion preset only routes `concat_direct`, and the `lnout` / `nt8y` presets only set `outputNorm`), `ArchSweepCLI.benchArch` (`App/ArchSweepCLI.swift:32-53`) and every test fixture built through it therefore keeps its exact meaning and identity, with no edits.
  - Passing `.doesNotApply` to it builds an architecture `validate()` refuses (the group field), like any other invalid argument.
- `BlockGroup` and its inits are unchanged.

## D7. Identity and effects (each checked)

| Concern | Effect | Why |
|---|---|---|
| Equality / `Hashable` | New stored fields take part (synthesized). A legacy file decodes equal to its pre-change value and to its code preset. An absent site's activation can no longer make two otherwise-equal architectures unequal (it has exactly one legal value); other unread settings are unchanged (D3). | D3 + D5 rule 2 = what the convenience init sets. `hashValue` is never persisted: identity is the value itself (`NetworkArchitecture.swift:24-25`). |
| Legacy `.dcmmodel` `archHash` (FNV) | Unchanged | It mixes shape scalars and the version label only (`Persistence/ModelCheckpointFile.swift:199-233`). Its writer has no production caller (every save is safetensors, `CheckpointManager.swift:1291`); only tests reach `ModelCheckpointFile.encode()`. |
| `arch_hash` in logs / dashboards | Unchanged | It is that same legacy FNV value; nothing else hashes the architecture. |
| Model IDs | Unchanged | Minted at events, never derived from the architecture. |
| `content_sha256` | Unchanged | SHA-256 over the tensor data region only (`SafetensorsFile.swift:117`). The architecture JSON lives in `__metadata__`. |
| Lineage records | No schema change from this plan | Today the record holds no architecture; a re-saved file's metadata has the v9 JSON. `HPARAM_RECORDING_PLAN.md` P4 adds `ancestry.runs[].architecture_at_departure`, the source file's `architecture` text copied byte for byte with its `dcm_format_version`. Once both plans land, a v9 file's record can carry a v8 (pre-split) architecture text. Anything that decodes that text must decode it under its own stated version through `ArchitectureFormat` (D5), which resolves the six sites; reading it as a v9 architecture would be refused (missing keys, retired `activation_function`). |
| Behavior fingerprint / resume exactness | Unchanged for every existing model | The fingerprint trains the checkpoint's architecture and hashes what it computes, never the architecture's serialized form (`Training/BehaviorFingerprint.swift:26-46`, `:184`). Same graph gives the same bytes, so a v8 → v9 resume should add no `build` gap. A same-build test cannot show that across builds, so V3b checks it with an OLD-written checkpoint resumed by NEW. A real head-activation change does change the fingerprint (new test). The in-process fingerprint cache is keyed by `Settings` (architecture + tail precision); with D3 it never holds two entries for architectures that differ only in an absent site's activation (other unread settings can still split an entry, as today). |
| Parameter count / `weightTensorPlan` | Unchanged | Activations have no parameters; the leaky slope is a constant (`:240-243`). |
| Weight init (same `--init-seed`) | Trainable tensors bit-identical whatever the activations | Every draw depends only on `(init seed, tensor name)` and the tensor's shape (`Network/WeightInitialization.swift:53-121`). He-normal uses no activation gain; there is no activation branch. |
| BN running statistics of a fresh build | Identical when only head activations change; different downstream of a changed stem / tower-end / fusion activation | `.randomWeights` calibrates running stats with one GPU forward (`Network/ChessMPSNetwork.swift:21-31`, `:80-99`). A BN downstream of a changed activation sees different inputs. `policy_head_activation`, `value_head_conv_activation` and `value_head_fc1_hidden_activation` have no BN downstream; the tower-end activation feeds `policy.pre_bn`, `value.bn` and `feature_skip.bn`. |
| Bit-exact behavior of existing models | Unchanged | Each existing site reads a field equal to the old top-level value; an absent site is never built. Pinned by `StandardPathForwardPinTests.testStandardPathForwardIsBitIdenticalToThePinnedOutputs` (unmodified; its fixtures are `intermediate_conv` and `fc_bottleneck` on `.current`, so no site appears or disappears) and the new legacy forward test. |
| `architectureSummary` (`[REPLAY-ARCH]`, `[VS-UCI-ARCH]`, `[DERIVE]`, `[BUTTON] Build Network`, `[ARCH-CONFIG]`, About, session picker) | Byte-identical when every *existing* site has the same activation (every file on disk and every preset). Otherwise the clause lists the existing sites. `does_not_apply` never appears in it. | `:1924` becomes the clause below; golden tests `BlockGroupArchitectureTests.swift:136-158` pass unmodified. |
| `[ARCH] legacy file …` | Gains six resolutions per pre-v9 load (D5) | |
| `[LAYER-HEALTH]` | Unchanged for existing models; `does_not_apply` never appears (an absent stem prints `-`, as today, `LayerHealth.swift:1129`) | T4 |
| Diagram | Shows each existing site's activation (P3) | |
| Presets | Built-in: no change (code, convenience init). User presets (10 in `Presets/`: 4 at v8, 6 unversioned) and committed experiment presets (`experiments/*/test_*.json`, `r7_basic24.json`, `experiments/presets/*.json`) load unchanged through D5 rule 2. The user presets are re-saved at v9 in P4 (OD-10); the committed experiment presets stay as written (they are reproduction records). | |
| `parameters.json` / `parameters.md` | Not affected | Architecture fields are not training parameters. |

**Summary clause rule** (replaces `" . act \(activationFunction.rawValue)"` at `:1924`):
- let `present` be the sites where `hasActivationSite` is true (never empty: both value sites always exist);
- if every present site has one activation `X`, the clause is ` . act X` (today's string);
- otherwise it is ` . act ` followed by `<summaryLabel> <fn>` for each present site, comma-separated, in build order. Example: ` . act tower_end relu, policy leaky_relu, value_conv leaky_relu, value_fc1_hidden leaky_relu`.

One shared function, `NetworkArchitecture.siteActivationClause`, is read by the summary. The diagram reads the per-site values.

## D8. Build New Model: how a site is chosen (no silent default)

The screen edits the topology in several places: each group's style (bound directly to `BlockGroupDraft`), group add / duplicate / remove / move (which change the first and last group), the policy style and the feature-skip fields. Any of them can make a site appear or disappear.

- **The model stores the six values** (`BuildNewModelModel.stemActivation`, …), loaded from and composed into the architecture like every other field.
- **Composition clears absent sites.** `architecture` builds the memberwise value from the stored fields and then calls `clearActivationSitesTheTopologyLacks()` (D2). So the composed architecture never holds a function at an absent site.
- **A disappeared site's stored choice is dropped at the edit itself, synchronously.** `syncSiteActivationsWithTopology()` copies the composed architecture's six values back into the stored fields. It runs inside every model mutation that can change which sites exist, before the mutation returns, so a disappear → reappear sequence can never skip it, however edits are coalesced or rendered:
  - `didSet` on the stored model fields `policyHeadStyle`, `featureSkipSource`, `featureSkipFusion`, `featureSkipToPolicyHead` and `featureSkipToValueHead` (`featureSkipToFinalBlock` does not change `featureSkipUsesCompressNode`);
  - the end of `duplicateGroup`, `appendCopyOfLastGroup`, `removeGroup` and `moveGroup` (`BuildNewModelModel.swift:234-263`), which can change the first or last group;
  - a group's style: `BlockGroupDraft` gains `var activationStyle` (get `group.activationStyle`; set it, then call `onTopologyChange()`), the same computed-property pattern as its `activationFunction` (`BlockGroupDraft.swift:42-45`). `onTopologyChange` is a non-optional `let` of type `@MainActor () -> Void` (a `let`, so `@Observable` does not track it), passed to `BlockGroupDraft.init` by the model, the only creator of drafts (`BuildNewModelModel.swift:101`, `:133`, `:236`, `:244`; no test creates one). The model passes `{ [weak self] in self?.syncSiteActivationsWithTopology() }`, with a comment saying why the optional call is not a silent fallback: SwiftUI can keep a row's bindings alive past the update that removes the row (`BlockGroupDraft.swift:13-27`), and those bindings hold the draft, not the model, so a write can reach a draft whose model is gone. Such a draft is detached exactly like a removed draft, whose late writes the design already treats as harmless: the write changes only the orphan draft, and there is no model state left to sync. (`unowned` would crash on that write.) `init` sets `blockGroupDrafts = []` first, then every other stored property, and only then builds the drafts, so phase-one initialization is complete before any closure captures `self`. Every field that has a `didSet` is assigned before phase one ends, and under `@Observable` those assignments go through the init accessor, so no `didSet` runs inside `init`. The two assignments after phase one (the drafts and `availablePresets`) have no `didSet`. Nothing reads the composed architecture during the empty-array phase, so `hasStemActivation` / `hasTowerEndBN`'s empty-tower preconditions cannot trip. The "Activation style" picker binds `$draft.activationStyle` instead of `$draft.group.activationStyle` (`BuildNewModelView.swift:461`), which is the screen's only writer of a group's style;
  - `load(_:)` replaces every field from a consistent architecture (a whole snapshot), but its assignments run one by one and trigger the `didSet` syncs above, each against a half-replaced topology. So `load` assigns the drafts and every topology field first and the six site activations **last**, in one block after them. Any intermediate sync can only rewrite the six stored fields, and the final block overwrites all six with the snapshot's own consistent values, so the loaded state equals the snapshot whatever the previous state was. (`init` needs no ordering: as above, no `didSet` runs during it.)
  - No view `.onChange` takes part, so correctness does not depend on render timing. The site's earlier choice is therefore never restored silently when it reappears.
- **A site that appears must be chosen.** Its stored value is `does_not_apply`, so:
  - its picker becomes enabled and shows "choose…" (the `does_not_apply` tag, labelled for an existing site), with the row's label in the screen's existing invalid colour (`.orange`, as "Total blocks" uses at `BuildNewModelView.swift:94-98`);
  - `validationError` is the D3 message naming the site (for example `policy_head_activation is 'does_not_apply', but the policy head has a pre-block (intermediate_conv): choose one of relu, silu, gelu, leaky_relu`), shown where validation errors already appear (`:240-268`);
  - Build (`:326`) and Save as Preset (`:313`) stay disabled until the user picks a function, or uses "Use for every activation" (P3), which fills every existing site at once.
- **Pickers** (new `ArchitectureSiteActivationPicker.swift`, D2's `displayName`):
  - always present, so rows never shift (the SE-activation precedent, `BuildNewModelView.swift:451-459`);
  - absent site: disabled, showing "does not apply", `.help(site.absentReason)`;
  - existing site: enabled, listing `ActivationFunction.functions`, plus the "choose…" entry only while the value is `does_not_apply`; `.help` says what the site does.
- Loading a preset or a model (`load(_:)`, `init`) copies its six values, which are consistent by D3, so no site needs choosing.

---

# Part T — Touch points

### T1. `Network/ArchitectureFormat.swift`
- `currentVersion` 8 → 9 (`:62`); new constant `siteActivationsRequiredFromVersion` (after `:102`); `DecodeFormat.allowsMissingSiteActivations` (after `:192`).
- History comment (`:33-52`).
- `decodeSiteActivation` (beside `decodeInitOption`, `:240-273`).
- `FormatError` (`:124-146`): `retiredField` (OD-5), `activationSiteMismatch`, `doesNotApplyAtAnAlwaysPresentSite`, `legacyActivationFunctionMissing`, each with its description.

### T2. `Network/NetworkArchitecture.swift`
- `ActivationFunction` (`:221-244`): the `doesNotApply` case, `functions` (`allCases` without it; `CaseIterable` kept, OD-14), rewritten doc (D2).
- `ArchitectureActivationSite` enum and `ActivationSiteMismatch` (new; beside `InitOptionField`, `:405-452`).
- `NetworkArchitectureError` (`:977-1066`): `activationSiteMismatch`, `notAnActivationFunction(context:)`, `doesNotApplyAtAnAlwaysPresentSite(field:)`, with descriptions.
- `BlockGroup.init(from:format:)` (`:833-901`): refuse `does_not_apply` in `activation_function` / `se_activation` (D3). `BlockGroup.setActivationFunction` (`:648-653`): the `does_not_apply` precondition (D3).
- The stored field `activationFunction` (`:1083-1086`) is replaced by the six fields, each with a doc comment saying where it applies, when the site exists, and that it is `does_not_apply` otherwise.
- Full memberwise init (`:1135-1173`) and uniform convenience init (`:1177-1253`), per D6.
- `CodingKeys` (`:1269-1299`), decode (`:1309-1396`, two passes) and encode (`:1398-1418`), per D5.
- `hasActivationSite`, `activation(at:)`, `activationSiteMismatch`, `setActivation(_:at:)`, `setActivationAtEveryExistingSite`, `setMainActivationEverywhere`, `clearActivationSitesTheTopologyLacks` (new extension, beside the init-neutral extension at `:2152`).
- `architectureSummary` (`:1904-1930`) uses `siteActivationClause`.
- `validate()` (`:1608-1689`): the group `does_not_apply` check first in the group loop; the site check after the feature-skip checks and before `checkedParameterCountBreakdown()` (`:1688`). Its doc comment states the rule.

### T3. `Network/ChessNetwork.swift`
- Delete the private tower-level overload (`:2364-2372`). A site can then only be built from its own field: the compiler refuses a call that passes `arch`.
- `:732` uses `Self.activation(g, x, arch.stemActivation, name: "stem_act")`.
- `:895` uses `arch.towerEndActivation`.
- `:933` uses `arch.featureSkipActivation`.
- `:3206` and `:3258` use `arch.policyHeadActivation`. Under the `float32FromPreBatchNorm` policy-tail precision this activation runs in fp32, under `mixedFinalProjection` in the compute dtype. Both paths are tested.
- `:3467` uses `arch.valueHeadConvActivation`.
- `:3495` uses `arch.valueHeadFC1HiddenActivation`.
- `activation(_:_:_:name:)` (`:2379-2401`) gets the `.doesNotApply` `preconditionFailure` (D3), and its doc comment (`:2374-2378`) says so. Graph op names are unchanged, so `LayerHealthTests.graphNames(forSite:)` (`DrewsChessMachineTests/LayerHealthTests.swift:200-213`) still holds.

### T4. `Training/LayerHealth.swift`
- `batchNormSites` (`:159-217`) tags each site with its own field:
  - `stem.bn` → `hasStemActivation ? stemActivation : nil` (an absent stem is `nil`, "the output feeds no activation", exactly as today; `does_not_apply` never becomes a tag);
  - `tower_final_bn` → `towerEndActivation` (appended only when the site exists, as today);
  - `feature_skip.bn` → `featureSkipActivation` (same);
  - `policy.pre_bn` → `policyHeadActivation` (same);
  - `value.bn` → `valueHeadConvActivation`.
- `valueFC1Layer` (`:242-252`) uses `valueHeadFC1HiddenActivation`.
- Doc comments at `:159-166` and `:242-244` are rewritten.
- `classification(for:)` (`:274-282`): relu / leaky are classified, silu / gelu report `notApplicableSmoothActivation`, `does_not_apply` is a `preconditionFailure` (D3). A smooth head site next to a ReLU tower end is therefore classified site by site.
- No change to `LayerHealthLog` or the table format: the checkpoint table already prints an `act` column per site (`:1110`, `:1129`), which is how the validation checks it. No Python reader parses the layer-health activation tags (searched: no `.py` file reads `layer_health`).
- **Zero-velocity counting is unchanged and ignores the activation.** `valueFC1Layer` (`:245-252`) passes the activation into `HiddenUnitLayer` only as a label for the velocity row (`:678`, `:1143`); `valueFC1ZeroVel` counts exactly-zero weight-velocity columns whatever the function. This plan keeps reporting it under every `value_head_fc1_hidden_activation`: it is a measurement, and the row names the activation next to it.
- **Interaction with `TRAINING_HEALTH_ALARMS_PLAN.md`** (its Risks name this plan; whichever lands second re-checks rules 2 and 3):
  - *Rule 2 (dead BN channels by β/|γ|)* reads `LayerHealth`'s classification, which stays as it is: `relu` and `leaky_relu` sites are classified, `silu` / `gelu` sites are not (`:274-282`), and an absent site has no BN entry or a `nil` tag. With per-site activations the set of classified sites can change per model (for example a `silu` value conv takes `value.bn`, the site rule 2's 20% per-site arm is most sensitive to, out of classification). Under `leaky_relu` "dead" still means β/|γ| < −3, though such a channel keeps a 0.01 slope; that is today's behavior for leaky towers and is not changed here.
  - *Rule 3 (value-FC1 zero velocity)* assumes a dead ReLU unit, whose gradient is exactly zero. Under `leaky_relu`, `silu` or `gelu` a unit's gradient is almost never exactly zero, so rule 3 would almost never fire and its silence would mean nothing. The alarms plan already handles this in its own design (its rule-3 note, `TRAINING_HEALTH_ALARMS_PLAN.md:276`): rule 3 evaluates only when `LayerHealth.valueFC1Layer(for:)` reports `relu`, and is otherwise `not_applicable(activation=<fn>)` with no dedicated velocity reads. That note says whichever plan lands second makes `valueFC1Layer(for:)` return the value-head hidden activation; this plan's T4 change above does exactly that (`valueHeadFC1HiddenActivation`), so rule 3 follows the field from one source. Nothing else is needed here (this plan's OD-15; the alarms plan's own OD-15 is a different decision, its rule-3 cadence).

### T5. Numerics audit: no code change
- The layer-health findings (`Network/NumericsAudit.swift:585-610`) and `--analyze-numerics`'s per-site table (`Network/NumericsAudit+Summary.swift:43-44`) come from `LayerHealth`, so they become per-site through T4.
- The dynamic audit compares formats tap by tap and makes no assumption about the activation.

### T6. Behavior fingerprint: no code change
- It is computed from the built graph (D7). The new test `testAHeadActivationChangeChangesTheBehaviorFingerprint` pins that a head-only change is detected.

### T7. `--derive-model`
- **P1:** `SetActivationDeriveOperation` (`Persistence/ModelDerivation.swift:738-806`):
  - value syntax and parse use `ActivationFunction.functions` (`:763`, `:775-778`). `does_not_apply` is refused at parse: "'does_not_apply' is not an activation function; it marks a site the topology lacks, and --set-activation sets only sites that exist";
  - `apply` uses `setMainActivationEverywhere`;
  - the no-op check covers every existing site and every group;
  - `summary` reads: "Set the main hidden activation everywhere: every architecture-level site the model has (stem, tower end, feature-skip fusion, policy head, value conv, value FC1 hidden) and every block group's activation_function …";
  - `changedArchitectureFields` (the fields the kind may change) becomes the six keys plus `block_groups[].activation_function` and `block_groups[].se_activation`.
  - Absent sites stay `does_not_apply`. On an SE-less tower, `--set-activation leaky_relu` therefore equals the leaky convenience-built tower. On a tower with SE blocks it leaves every SE group's `se_activation` alone (D2, as today), and only `--set-activation X --set-se-activation X` equals the convenience-built tower.
- **P1:** `SetSEActivationDeriveOperation` (`:816-892`): value syntax and parse use `ActivationFunction.functions` (`:824`, `:832-835`), so `--set-se-activation does_not_apply` is refused with the same message shape. Its `apply` (`:876`) also refuses `value == .doesNotApply` before touching the source, so a directly constructed operation (as tests build them) can never return an invalid architecture. `SetActivationDeriveOperation.apply` gets the same guarantee from `setMainActivationEverywhere`, which refuses before changing anything.
- **P2:** new file `Persistence/SiteActivationDerive.swift` (the same per-family layout as `InitOptionDerive.swift`). The project uses folder-synchronized groups, so no `project.pbxproj` edit is needed.
  - One `SetSiteActivationDeriveOperation(site:value:)` type, with one `DeriveOperationKind` per site built from `ArchitectureActivationSite.allCases`: `--set-stem-activation`, `--set-tower-end-activation`, `--set-feature-skip-activation`, `--set-policy-head-activation`, `--set-value-head-conv-activation`, `--set-value-head-fc1-hidden-activation`.
  - Value syntax `relu|silu|gelu|leaky_relu` (from `functions`); `acceptsGroupSelection: false`; `rewrittenTensorsDescription: "none"`.
  - Refuses `does_not_apply` (not a function) at parse and again at apply (a directly constructed operation), an unknown value, an absent site (naming `absentReason`) and a no-op. Applies through `setActivation(_:at:)`, which also refuses a mismatch; the source is untouched on every refusal. No tensor rewrites, so it is allowed on a trained source (`ModelDerivation.swift:28-37`).
- No derive operation changes the topology, so none needs to clear or choose a site.
- **Catalog order** (`ModelDerivation.operationKinds`, `:119-131`): the six follow `SetSEActivationDeriveOperation.kind`.
  - `--set-activation X --set-value-head-fc1-hidden-activation Y` therefore gives X everywhere except Y.
  - `--help` and the CLI parser are catalog-driven (`App/DeriveModelCLI.swift:57`, `:98`, `:150`), so they need no other edit.
- `documentation/deriving-models.md`:
  - `:37` (the list of no-tensor operations);
  - `:57` (the `--set-activation` row);
  - a new row per site setter after `:58`;
  - `:74-78` (`--set-activation` prose: existing sites only; `does_not_apply` refused);
  - `:138-141` (operation order);
  - examples after `:155-162` (leaky heads on a ReLU tower);
  - `:188-193` (legacy rules: "an existing site → the file's `activation_function`, an absent site → `does_not_apply`, before v9");
  - `:203-209` (refusals: `does_not_apply` as a value of `--set-activation` / `--set-se-activation` in P1; a site the topology lacks and `does_not_apply` for the site setters in P2).
- `ModelGraft` needs no change: a graft's target architecture comes from a preset or `architecture.json` and is validated (`ModelGraft.swift:251`).

### T8. Build New Model (P1, except the menu)
- `App/UpperContentView/BuildNewModelModel.swift`:
  - the stored field `activationFunction` (`:51`) is replaced by six stored fields;
  - `init` (`:97-121`) and `load` (`:129-151`) copy them;
  - `architecture` (`:153-175`) passes them and then clears absent sites (D8);
  - new `func syncSiteActivationsWithTopology()`, called synchronously by every topology mutation (D8): `didSet` on `policyHeadStyle` and the four `featureSkip*` fields that decide the compress node, and the end of the four group methods (`:234-263`);
  - drafts are created with the `onTopologyChange` closure (`:101`, `:133`, `:236`, `:244`); in `init`, only after `blockGroupDrafts = []` and every other stored property are set (D8);
  - `load(_:)` (`:129-151`) assigns the six site activations last, after the drafts and every topology field (D8);
  - new `func siteExists(_:) -> Bool`, read once per redraw by the screen like `nonStandardInitOptions` (`BuildNewModelView.swift:65-66`);
  - new `func applyMainActivationEverywhere(_:) throws` (P1; the menu that calls it is P3) applies `NetworkArchitecture.setMainActivationEverywhere` to `architecture` and copies the six fields and each group back through its draft (the `applyInitOptions(of:)` pattern, `:289`). Its only callers pass a member of `functions`; the menu shows a thrown error in the screen's status line, never drops it.
- `App/UpperContentView/BlockGroupDraft.swift`: `init(_:onTopologyChange:)` (non-optional closure) and the computed `activationStyle` property (D8). Nothing else changes.
- New file `App/UpperContentView/ArchitectureSiteActivationPicker.swift`: a `View` struct taking the site, a `Binding<ActivationFunction>` and `siteExists` (D8).
- `BuildNewModelView.swift`:
  - Tower section (`:91-99`): the "Tower-level activation" picker (`:93`) becomes "Stem activation" and "Tower-end activation"; in P3 a `Menu("Use for every activation")` listing `ActivationFunction.functions`, which calls `applyMainActivationEverywhere`, joins them.
  - Feature skip section (`:165-180`): "Fusion activation", placed **outside** the existing `if model.featureSkipSource != .none` block (directly after the "Source" picker), so it is always present like every other site picker, and disabled through `siteExists(.featureSkipFusion)`.
  - Policy head section (`:126-141`): "Pre-block activation".
  - Value head section (`:142-164`): "Conv activation" and "FC1 hidden activation".
  - Group pickers (`:451`, `:457`): `ActivationFunction.functions` (through `BuildNewModelView.groupActivationChoices`); `enumPicker` (`:566-572`) is unchanged (OD-14 keeps `CaseIterable`).
  - The group "Activation style" picker (`:461`) binds `$draft.activationStyle` (D8).

### T9. `App/UpperContentView/ArchitectureDiagramView.swift`
- P1 (needed to compile):
  - `:53` shows `arch.stemActivation` (still only when `hasStemActivation`);
  - `:68` shows `arch.towerEndActivation` (only when `hasTowerEndBN`, as today).
- P3:
  - the policy cell's pre-block line (`:81`) becomes `"\(in) → K=\(K) · \(policyHeadActivation)"`;
  - the value cell line (`:91`) splits into `"\(in) → \(conv)ch · \(convAct)"` and `"→ FC\(h) · \(fc1HiddenAct)"`;
  - `featureSkipMarker` (`:165-186`) appends ` · <featureSkipActivation>` when the compress node exists.
  - Each line is drawn only where its site exists, so `does_not_apply` is never drawn.
- No new `some View` properties.

### T10. CLI and logs
- `--new-model --architecture <preset | path>`: no code change. A v9 preset or `architecture.json` carries the six fields. `ArchitectureConfig.writeTemplate` (`Persistence/ArchitectureConfig.swift:62`) writes v9 with all six.
- `CommandLineHelp.usageText`: no change. The derive operations are listed by `--derive-model --help` from the catalog.
- Log strings: `architectureSummary` per D7; `[ARCH] legacy file …` per D5; `[LAYER-HEALTH]` checkpoint table per T4.

### T11. Python mirrors
- **`scripts/dcm_arch.py`:**
  - add `SITE_ACTIVATIONS_REQUIRED_FROM_VERSION = 9`, `SITE_ACTIVATION_KEYS` and `DOES_NOT_APPLY = 'does_not_apply'` (after `:41`);
  - add `site_exists(norm, key)`: the D1 existence rule on a normalized architecture (first / last group `activation_style`, `policy_head_style`, and the compress node: `feature_skip_source != 'none'`, `feature_skip_fusion == 'compress_conv_bn_relu'`, routed to the policy or value head);
  - add `site_activations(arch, format_version)` and `site_activations_md(md)`, which return the six values under D5's rules: stated wins; an older or uniform-tower file resolves an existing site from `activation_function` and an absent one to `does_not_apply`; v9+ missing raises `ArchitectureError`; v9+ block-groups with a top-level `activation_function` raises (OD-5). Then both directions of D3 are checked, and a `does_not_apply` in any group's `activation_function` / `se_activation` raises. Every `ArchitectureError` names the key and the site;
  - add `require_relu(md, source, sites, block_main_path=False)`: resolves `site_activations_md(md)` and raises `ArchitectureError` naming `source` and the site unless every named site is exactly `relu` (a site the file lacks is `does_not_apply`, so it raises too: the caller models it as a present ReLU). With `block_main_path=True` every group's `activation_function` must be `relu` too. This is the one-line guard of the frozen scripts below;
  - `norm_arch` (`:52-97`) gets one fix and nothing else. Its uniform-tower expansion (`:66-75`) builds a group without `rezero_alpha_cap`, so the strict check at `:92-95` rejects a uniform-tower input stamped v6 or later, while Swift decodes that form at any stated version (legacy by construction, D5). The expansion therefore also sets `rezero_alpha_cap = rezero_alpha_init * REZERO_TANH_CEILING_MULTIPLE`, the Swift expansion's `legacyRezeroAlphaCap` (`NetworkArchitecture.swift:1362-1363`). Block-groups inputs keep the strict missing-cap rejection. Output is unchanged for every real file (every uniform-tower file is v3, census) and for the committed tests, which build synthetic v5 block-groups architectures with no top-level `activation_function` (`documentation/dashboards/tests/test_tooling.py:29-48`). The site values live in `site_activations`, not in `norm_arch`: only scripts that model the heads need them.

Python scripts fall into three kinds, and they are handled differently.
- **Reusable readers** are run on new models (shared helpers, or tools with a documented reproduce/re-run use). They get the v9 rules and an explicit refusal of anything they do not model.
- **Frozen scripts that would silently model ReLU** are records of one analysis of named ReLU checkpoints. They are not otherwise edited, but each gets the one-line `dcm_arch.require_relu` guard (OD-9), so a v9 model with a non-ReLU site can never be analysed wrongly.
- **Frozen scripts that fail loudly or model no activation** are left exactly as they are and documented.

**Reusable readers (changed in P1):**
- **`experiments/20261004-head-logits-relu-inputs/relu_inputs.py`:**
  - `forward(x, arch, t, record)` (`:204`) receives no metadata. `load` (`:23-31`) returns `(md, json.loads(md["architecture"]), t)` to its two callers: `run` (`:247-258`) and `experiments/20261004-value-head-vs-stockfish/value_stage.py`'s `net_probs` (`:35-39`, which imports this file as `ri`, `:6-7`).
  - `load` returns `dcm_arch.norm_arch_md(md)` in place of the raw `json.loads`, so a pre-feature-skip file reads `feature_skip_source: none` by the app's own rule (normalization only adds fields `forward` does not read, so outputs are unchanged).
  - `forward` gains a required keyword-only `sites` (`def forward(x, arch, t, record, *, sites)`), so a caller that is not updated fails with a `TypeError` at once and no positional argument shifts.
  - Both callers resolve `sites = dcm_arch.site_activations_md(md)` (which also enforces D3) right after `load` and pass `sites=sites`. `value_stage.py` reaches the module as `ri.dcm_arch`. Both are updated in P1.
  - `forward` adds asserts next to its existing architecture assert (`:206-208`):
    - every group's `activation_function` is `relu`;
    - `sites['tower_end_activation']`, `sites['policy_head_activation']`, `sites['value_head_conv_activation']` and `sites['value_head_fc1_hidden_activation']` are `relu` (each site its forward applies);
    - `arch['feature_skip_source'] == 'none'`, because the forward never builds a feature skip; today a feature-skip model would be mis-modelled silently.
  - None of this changes its output on the three ReLU checkpoints it was run on, `F`, `R7` and `R8` (`experiments/20261004-head-logits-relu-inputs/README.md:30-32`). `value_stage.py` was run on R7 only (`experiments/20261004-value-head-vs-stockfish/README.md:5-8`); its gate uses all three because `relu_inputs.forward` supports all three.
  - **P1 gates** (both compare the pre-edit scripts, extracted with `git show <P1 parent>:…` into a scratch folder, against the edited ones, on identical inputs and the same Python and numpy):
    - `relu_inputs.run` on `F`, `R7`, `R8` with the same position list: identical results. (Its positions come from a glob over the live `LichessBot/Games` folder, `relu_inputs.py:105`, so a fresh run need not match the committed `relu_inputs_bot.json`; reproducing that file byte for byte is a secondary check, and a difference there is investigated, not taken as a regression by itself.)
    - `value_stage.net_probs` on `F`, `R7`, `R8`, on the same 256 encoded positions (from `value_stage.sample` with a fixed seed): bit-identical arrays. It needs `python-chess` (the README's venv), not Stockfish.
  - It gains the `sys.path` insert and `import dcm_arch` that `dcm_arch.py`'s docstring prescribes (`scripts/dcm_arch.py:17-20`).
- **`documentation/research/fp16-feasibility/scripts/fwd16.py` and its callers** (one unit; `fwd16.py` already imports `dcm_arch`, `:22`). OD-9.
  - **The problem.** `forward(T, arch, x, R, q, capture)` (`:118`) and `forward_batched(T, arch, X, R, q, bs, keep)` (`:196-198`) receive an architecture that every caller has already normalized with `norm_arch(md['architecture'])`, **without** the format version. Normalization also expands the uniform-tower form, so by then neither the version nor the form needed for D5 is left. Site activations therefore cannot be resolved inside `forward`.
  - **Resolution from raw metadata, before normalization.** Each caller, right after `load` (`:33-41`, which returns `md`), computes `sites = dcm_arch.site_activations_md(md)`. The callers import `fwd16` with `from fwd16 import *` (e.g. `sat16.py:6`), which does not export `fwd16`'s private `_dcm_arch` (`fwd16.py:22`); so each caller adds `import dcm_arch` right after that line (importing `fwd16` has already put `scripts/` on `sys.path`, `fwd16.py:21`). That reads `dcm_format_version` and the raw architecture JSON, so it sees the real version and form. In the same line the caller replaces the version-less `norm_arch(md['architecture'])` with `norm_arch_md(md)`, the form `fwd16.py`'s own docstring prescribes (`:203-212`).
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

  - **Checks in `forward`**, next to the existing one-group assert (`:120`). `sites` has already passed D3, so an absent site is known to be `does_not_apply`; each check below names the site the forward applies at that point:
    - reject any feature skip: `assert arch['feature_skip_source'] == 'none'`, because this forward never builds one;
    - the group's main path is `relu` (kept from `:122`); the top-level half of `:122` is removed, since the key no longer exists in v9;
    - when the group is post-activation, `sites['stem_activation'] == 'relu'` (`:127` is the only place it is applied);
    - when it is pre-activation, `sites['tower_end_activation'] == 'relu'` (`:168`);
    - when `policy_head_style != 'simple_conv'`, `sites['policy_head_activation'] == 'relu'` (`:176`);
    - always, `sites['value_head_conv_activation'] == 'relu'` and `sites['value_head_fc1_hidden_activation'] == 'relu'` (`:182`, `:184`).
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
    3. require `relu` for `tower_end_activation`, `policy_head_activation`, `value_head_conv_activation` **and** `value_head_fc1_hidden_activation`, raising `ValueError` naming the file and the site. A `simple_conv` model has `policy_head_activation: does_not_apply` and is refused here, which is right: `analyze` labels `policy.pre_bn`, which such a model lacks.
    - The existing one-group / pre-activation / ReLU-main-path check (`:195-199`) stays where it is. The script already requires a pre-act tower, so the stem has no activation to check.
  - Rejection tests in `documentation/dashboards/tests/test_dcm_arch_site_activations.py`. `units.py` is importable (its `main` is guarded, `:349`). Each case uses a stand-in object with `file`, `metadata` and `architecture` built from a synthetic header:
    - accepted: a v9 ReLU file (`stem_activation` and `feature_skip_activation` `does_not_apply`), and a v5 ReLU file resolved from `activation_function`;
    - rejected: `value_head_fc1_hidden_activation: leaky_relu` (the case the earlier draft missed), `tower_end_activation: gelu`, `policy_head_activation: silu`, `value_head_conv_activation: leaky_relu`, a `simple_conv` file, and `feature_skip_source: stem_output` (rejected before any activation check).

**Frozen scripts that would silently model ReLU (one-line guard, P1; OD-9).** Found by reading every script that loads a model and applies or labels an activation without reading an activation key. Each guard is placed right where the script reads the file's metadata, with the sites that script models, and adds the `sys.path` insert `dcm_arch.py` prescribes where the script lacks one.

| Script | Models as ReLU (without reading a key) | Guard, right after |
|---|---|---|
| `documentation/research/bf16-head-offset/scripts/fwd.py` | block main path (`:91-92`), tower end (`:102`), value conv (`:103`), value FC1 (`:105`), all inside `forward` (`:81`) | not in `load`: `recenter.py:2` imports `fwd.load` (its forward comes from another module, `recenter.py:3`), and `posset.py:4` / `posset_ejp0.py:16` import `fwd`'s encoder; none of them calls `fwd.forward`, so a guard in `load` would constrain them for a model `fwd.py` never applies. `forward` has no caller in the repo (searched), so it gains a required keyword-only `md` (`def forward(T, arch, x, precision='f64', return_all=False, *, md)`) and its first line is `dcm_arch.require_relu(md, 'fwd.forward', ('tower_end_activation', 'value_head_conv_activation', 'value_head_fc1_hidden_activation'), block_main_path=True)` |
| `experiments/20261003-fatty-vs-skinny/tensor-health/fatty_tensors.py` | labels `blocks.0.bn1/bn2`, `tower_final_bn`, `policy.pre_bn`, `value.bn` as ReLU-fed (`:25-29`) | `md=h.pop('__metadata__')` in `load` (`:5`): `('tower_end_activation', 'policy_head_activation', 'value_head_conv_activation')`, `block_main_path=True` |
| `experiments/20261001-se-fc1-leaky/full-model-analysis/scripts/input_features.py` | reads value-conv channels as `ReLU(value.bn)` (`:9-10`) | `ckpt = L.Checkpoint(path)` (`:47`): `L.dcm_arch.require_relu(ckpt.metadata, ckpt.file, ('value_head_conv_activation',))`. Only that site: the script is run on the leaky-SE-FC1 run, so it must not require ReLU anywhere it does not model. |

- Each guard must pass on the checkpoints the script was actually run on, where there are such (V8 names them and checks this by calling the guard on those files, without re-running the analyses). It is not required to pass on every checkpoint a folder mentions: for example `bf16-head-offset/results/numerics-audit-2026-09-30/README.md:17` also names the post-activation fp32 control mUF5, which `fwd.forward` never modelled and its guard correctly refuses (no tower-end activation).

**Frozen scripts left unedited (documented in P4).**

| Script | On a v9 model |
|---|---|
| `documentation/research/bf16-head-offset/scripts/fwd3.py` (reads the top-level key, `:69`, `:92-93`) | `KeyError`, loud |
| `documentation/research/bf16-head-offset/scripts/fwd4.py` (`:81`) and `net681.py`, which calls `fwd4.forward` (`net681.py:11`, `:19`) | `KeyError`, loud |
| the other `bf16-head-offset/scripts/*` that run a forward from `fwd3` or `fwd4` (both read the top-level key: `fwd3.py:69`, `:92-93`; `fwd4.py:81`, reached from `forward_batched` too): `analyze.py`, `trajce.py`, `validate2.py`, `variants.py` (`from fwd3 import *`); `calib.py`, `calib_scan.py`, `gradcheck.py`, `heads681.py`, `sfcompare.py`, `trend_fwd.py` (from `fwd4`) | `KeyError`, loud |
| `documentation/research/policy-head-2026-10-01/scripts/analyze_policy_head.py` (`:85`) | `KeyError`, loud. `policy_head_lib.py` synthesizes the key only for legacy `.dcmmodel` layouts (`:143-153`, `:226-229`). |
| `experiments/20260929-se-style-ab/final_data.py`; the other `policy-head-2026-10-01/scripts/*` (`findings.py`, `render_tables.py`, `run_all.py`, `scan_headers.py`, `assemble_report.py`); `bf16-head-offset/scripts/traj2.py` (imports only `bf16` from `fwd3`); `bf16-head-offset/scripts/lines_list.py` (reads a pre-built `scan2.json`; it assigns `af = a.get('activation_function')` at `:19` but never prints or stores it, `:21`, `:26`, so a v9 entry lists correctly) | model no activation of a loaded file (SE FC2 weight norms, header scans, or tables from earlier results) |

- **Not changed:** `experiments/arch_flops.py` costs every activation as one op per element whatever its type (`:19`, `:33-74`) and reads no activation field.
- P4 adds a "frozen scripts" paragraph to `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20 listing both tables. It states that these scripts model ReLU at every site they touch and are valid only for the checkpoints their own folders name; the three silent ones now refuse anything else by guard, and the rest fail loudly or model no activation.

### T12. Documentation
- `documentation/plans-active/RUNTIME_ARCHITECTURE_CONFIG_PLAN.md`: a new "§20. Architecture format v9: per-site activations", in the shape of §18 (`:845-884`), including `does_not_apply` and the frozen-Python-scripts paragraph and tables from T11 (P4).
- `CLAUDE.md:136` (OD-7). Proposed edits (P4):
  - "current v8" → "current v9";
  - append "v9 replaces the top-level `activation_function` with one activation per architecture-level site (`stem_activation`, `tower_end_activation`, `feature_skip_activation`, `policy_head_activation`, `value_head_conv_activation`, `value_head_fc1_hidden_activation`); a site the topology lacks holds `does_not_apply` (never an identity activation), and an older file resolves each existing site from its own `activation_function`";
  - "→ tower-end BN (pre-act tail only)" → "→ tower-end BN + its activation (pre-act tail only)".
- `ROADMAP.md` (OD-11, approved): **P1** adds an entry in the section where in-progress architecture work is listed: "Per-site activations (format v9)", one paragraph summarizing the request, the six fields, `does_not_apply`, and a link to this plan. **P4** marks it completed and adds the commit hashes, keeping every word of the entry (global ROADMAP rule).
- `CHANGELOG.md`: one entry after the last phase, with each phase's commit hash.

### T13. Verified untouched
- `Network/WeightInitialization.swift`;
- `Persistence/SafetensorsModelIO.swift` (beyond the derived `formatVersion`), `ModelGraft.swift` (targets carry their own activations and are validated), `SessionManifest.swift`, `ArchitecturePresetStore.swift`, `ModelCheckpointFile.swift`;
- `Training/BehaviorFingerprint.swift`;
- every caller of the convenience init;
- `BlockGroup`'s fields and inits (its decode gains only the `does_not_apply` refusal, T2).

---

# Part X — Tests

### X1. New test files

**`DrewsChessMachineTests/ArchitectureActivationSiteTests.swift`** (pure; no Metal). "Full-site fixture" below means a post-activation first group → pre-activation last group tower with a compress fusion routed to an `fc_bottleneck` policy, where all six sites exist.

*`does_not_apply` itself:*
- `testDoesNotApplyIsNotAFunction`: the raw value is `"does_not_apply"`; `ActivationFunction.functions` is exactly `[.relu, .silu, .gelu, .leakyRelu]`.
- `testEveryActivationChoiceListIsTheFunctionsList` (OD-14): `functions` excludes `does_not_apply` and equals `allCases` without it; `ArchitectureSiteActivationPicker.functionChoices`, `BuildNewModelView.groupActivationChoices` and the `--set-activation` / `--set-se-activation` value syntax all equal `functions`.

*Format gate:*
- `testCurrentVersionRequiresSiteActivations`: `siteActivationsRequiredFromVersion == 9`, `currentVersion >= 9`, `SafetensorsModelIO.formatVersion == String(currentVersion)`.
- `testNewFilesWriteEverySiteAndNoTopLevelActivationFunction`: an encoded `.current` model's `architecture` JSON has all six keys, `stem_activation` and `feature_skip_activation` are `"does_not_apply"`, and there is no top-level `activation_function`.
- `testRoundTripPreservesEachSiteIndependently`: on the full-site fixture, for every site × every member of `functions`, set only that site; the safetensors, preset and `architecture.json` round trips are equal, with no legacy log line. Also `.current` (two absent sites) round-trips equal.
- `testPreV9FileResolvesExistingSitesToItsActivationFunctionAndAbsentSitesToDoesNotApply`:
  - versions `"8"`, `"5"`, `"3"` and none;
  - a pre-activation `intermediate_conv` tower with no feature skip, top-level `gelu`, group `relu`;
  - tower end, policy, value conv and value FC1 hidden decode as `gelu`; stem and fusion as `does_not_apply`;
  - the log names the file, each `:= gelu (the file's activation_function)` and each `:= does_not_apply (<absentReason>)`.
- `testPreV9PostActivationFileResolvesTheStem`: a v3 post-act `simple_conv` fixture: stem and both value sites `relu`, read by `LayerHealth`; tower end, policy and fusion `does_not_apply`.
- `testEveryPresetDecodesFromItsLegacyFormToItself`: for each `Preset`, strip the six keys, add `activation_function: relu` (every preset is ReLU at every existing site), decode at v8 and v3. The result is equal, with the same hash, summary, parameter count, `weightTensorPlan` and legacy `archHash`.
- `testLegacyUniformTowerKeysResolveEverySiteAtAnyStatedVersion`: the uniform-tower JSON decoded strict-current and at v8; existing sites take its `activation_function`, absent ones `does_not_apply`.
- `testV9FileMissingASiteActivationIsRejectedNamingFieldAndFile`: for each key, `missingRequiredField(field:, location: "the top level", formatVersion: 9, source:)`. Also a preset at `format_version` 9 (field `location: "architecture"`), and an undecorated `JSONDecoder()` (strict current).
- `testPreV9FileWithNeitherSitesNorActivationFunctionIsRejected`: `legacyActivationFunctionMissing` naming every unresolved key and the file; a pre-v9 file stating all six keys and no `activation_function` decodes (rule 1, the re-stamped-fixture shape).
- `testV9BlockGroupsFileStatingActivationFunctionIsRefused` (OD-5), and `testUniformTowerFormIsNotTreatedAsRetired`.
- `testAStatedSiteValueWinsInAPreV9File` (D5 rule 1): a v5-stamped pre-activation `intermediate_conv` file with `activation_function: gelu` and only `policy_head_activation: leaky_relu` stated.
  - `policy_head_activation` decodes as `leaky_relu`; the other existing sites as `gelu`; stem and fusion as `does_not_apply`.
  - The legacy log lists exactly the five resolved keys and not the stated one.
- `testAV9FileWithSomeSiteKeysNamesTheFirstMissingOne`: a v9 file stating three of the six keys fails with `missingRequiredField` naming an absent key, never a resolved value.
- `testAnUnknownSiteActivationTokenIsADecodeError`: `"swish"` in each key, at v9 and at v5, is a `DecodingError` whose coding path names the key. It is never resolved from `activation_function`.
- `testUniformTowerFormUsesStatedSiteKeysThenResolvesTheRest`: uniform-tower JSON (`activation_function: gelu`, pre-activation) that also states `value_head_fc1_hidden_activation: leaky_relu` decodes that site as `leaky_relu`, the other existing sites as `gelu` and the absent ones as `does_not_apply` (D5 uniform expansion: rule 1, then rule 2).
- `testASiteMismatchIsALoadErrorNamingFieldSiteAndFile`, both directions, for each site, through `SafetensorsModelIO.decode`, `ArchitecturePresetStore.loadFile` and `ArchitectureConfig.load`:
  - a function at an absent site (`stem_activation: relu` on a pre-activation tower; `tower_end_activation: relu` on a post-activation last group; `feature_skip_activation: relu` with no compress node; `policy_head_activation: relu` on `simple_conv`);
  - `does_not_apply` at an existing site (each of the six on the full-site fixture);
  - the error is `FormatError.activationSiteMismatch` naming the key, the site and the file; at v9 and with the key stated in a v5 file.
- `testALegacyActivationFunctionOfDoesNotApplyIsRefused`: a v5 block-groups file whose top-level `activation_function` is `does_not_apply` (refused through the site check), and a uniform-tower file with it (refused as a group field).
- `testDoesNotApplyIsRefusedInEveryBlockGroupField`: `activation_function` or `se_activation` of `does_not_apply`, on an SE group and an SE-less group, at v9 and v5: `FormatError.doesNotApplyAtAnAlwaysPresentSite` on decode, and `NetworkArchitectureError.doesNotApplyAtAnAlwaysPresentSite` from `validate()` on an in-memory value.

*Semantics:*
- `testSiteExistenceFollowsTheTopology`: the stem iff the first group is post; the tower end iff the last is pre; the fusion iff the compress node is built; policy iff not `simple_conv`; both value sites always. Covered on mixed pre→post and post→pre towers.
- `testValidateRefusesASiteMismatchInBothDirections`: for each site, a function at an absent site and `does_not_apply` at an existing one; `NetworkArchitectureError.activationSiteMismatch` naming the site. An architecture that also has a `branch_output_init` error reports that error first (D3 order).
- `testSettersRefuseDoesNotApplyAndMismatches`: `setActivationAtEveryExistingSite(.doesNotApply)` and `setMainActivationEverywhere(.doesNotApply)` throw `notAnActivationFunction`; `setActivation(.relu, at: .stem)` on a pre-act tower and `setActivation(.doesNotApply, at: .valueHeadConv)` throw `activationSiteMismatch`; the architecture is unchanged after each throw.
- `testSetActivationAtEveryExistingSiteLeavesAbsentSitesAlone`: on `.current`, every existing site takes the value; stem and fusion stay `does_not_apply`.
- `testClearActivationSitesTheTopologyLacksClearsOnlyAbsentSites`: switching `.current` to `simple_conv` and clearing gives `policy_head_activation: does_not_apply` and leaves the other fields unchanged. Switching back to `intermediate_conv` leaves it `does_not_apply`, and `validate()` names it (it never fills a site).
- `testUniformConvenienceInitSetsExistingSitesAndDoesNotApplyElsewhere`: pre / post × each policy style.
- `testSetMainActivationEverywhereMatchesTheConvenienceInitOnAnSELessTower`: for each member of `functions`, on an SE-less ReLU tower (the convenience init with `blockSeStyle: .none`), `setMainActivationEverywhere(X)` equals the same tower built with `activationFunction: X`.
- `testSetMainActivationEverywhereKeepsEverySEGroupsFC1Activation`. Mixed fixture: three groups (an SE group with `se_activation` `relu`, an SE group with `leaky_relu`, an SE-less group), all main paths `relu`, all pre-activation. After `setMainActivationEverywhere(.gelu)` it equals an **explicitly constructed expected architecture** with the same three groups, built field by field through the full `BlockGroup` and `NetworkArchitecture` memberwise inits:
  - every existing site `gelu`; stem and fusion `does_not_apply`;
  - every group's `activation_function` `gelu`;
  - the two SE groups' `se_activation` still `relu` and `leaky_relu`;
  - the SE-less group's `se_activation` `gelu` (`BlockGroup.setActivationFunction`, `NetworkArchitecture.swift:648-653`);
  - every other field unchanged.
  The convenience init builds one uniform group (`NetworkArchitecture.swift:1199-1233`), so it can never equal this fixture and is not used for it.
- `testSetMainActivationEverywherePlusSEActivationEqualsTheConvenienceInitOnAnSETower`. Separate single-group SE fixture: the convenience init with `blockSeStyle: .scaleAndBias` and `activationFunction: .relu`. `setMainActivationEverywhere(.gelu)` followed by `blockGroups[0].seActivation = .gelu` equals the same convenience init with `activationFunction: .gelu`, every other argument identical. Without the second step it differs only in `blockGroups[0].seActivation` (`relu`).
- `testSummaryKeepsTheUniformClause`: golden ` . act relu` for `.current` and a gelu tower.
- `testSummaryListsExistingSitesWhenTheyDiffer`: golden ` . act tower_end relu, policy leaky_relu, value_conv leaky_relu, value_fc1_hidden leaky_relu`; absent sites never listed; `does_not_apply` never appears in any summary.

*Layer health:*
- `testEachBatchNormSiteTagFollowsOnlyItsOwnField`: a site × function matrix on the full-site fixture (every site `relu` in the base), changing **one site at a time**. `ActivationFunction.functions` has four members, so "five different functions at once" is impossible.
  - For each of the five BN sites (`stem.bn`, `tower_final_bn`, `feature_skip.bn`, `policy.pre_bn`, `value.bn`) and each of `silu`, `gelu`, `leaky_relu`, set only that site's field. That site's tag equals the function, and every other site's tag (block BNs included) is unchanged from the base.
  - On `.current`, `stem.bn`'s tag is `nil` and no `feature_skip.bn` site is listed.
- `testValueFC1LayerFollowsOnlyItsOwnField`: the same one-at-a-time matrix for `valueHeadFC1HiddenActivation` and `LayerHealth.valueFC1Layer`. Every BN tag stays unchanged.
- `testSmoothHeadSitesAreNotClassifiedWhileAReLUTowerEndIs`: summary-level, through `summarizePlanAligned` with synthetic γ/β.

*Derive:*
- `testEachSiteSetterChangesOnlyItsFieldAndCopiesEveryTensorBitExact` (P2).
- `testSiteSetterRefusesASiteTheTopologyLacks` (P2): the stem on pre-act, the tower end on post-act, the fusion without compress, policy on `simple_conv`.
- `testSiteSetterRefusesDoesNotApply` (P2), `testSiteSetterRefusesANoOp` (P2), `testSiteSetterRejectsAnUnknownValue` (P2), `testSiteSettersAreAllowedOnATrainedSource` (P2).
- `testSetActivationSetsEveryExistingSiteAndEveryGroup` (P1).
- `testSetActivationAndSetSEActivationRefuseDoesNotApply` (P1): both flags refuse the value at parse, naming the flag.
- `testDirectlyConstructedOperationsRefuseDoesNotApply` (P1 for `SetActivationDeriveOperation(value: .doesNotApply, …)` and `SetSEActivationDeriveOperation(value: .doesNotApply, …)`; P2 adds `SetSiteActivationDeriveOperation`): `apply` throws and the source architecture is unchanged.
- `testSetActivationThenASiteSetterGivesTheMixedArchitecture` (P2): catalog order.
- `testLegacySourceDerivesToAV9FileStatingEverySite` (P1): a v5 source; the output's `dcm_format_version` is the current one and it states all six keys, `does_not_apply` where the site is absent.

*Build screen (`@MainActor`, P1 unless stated):*
- `testBuildScreenLoadsAndComposesEverySiteActivation`: load a leaky-head architecture; `model.architecture` equals it.
- `testBuildScreenSiteAvailabilityFollowsTheTopology`.
- `testBuildScreenClearsASiteTheTopologyRemoves`: set `model.policyHeadStyle = .simpleConv`; with no other call, both `model.architecture.policyHeadActivation` and the stored `model.policyHeadActivation` are `does_not_apply`, and the architecture validates.
- `testASiteThatAppearsMustBeChosen`: from `simple_conv`, set `.intermediateConv`: `validationError` names `policy_head_activation`, `buildRequest` is nil and `isValid` is false (Save disabled); setting the stored field to `leaky_relu` makes it valid.
- `testLoadingASnapshotReproducesItAcrossTopologies`: from each of several starting states (pre / post first group, `simple_conv` / `intermediate_conv`, feature skip off / compress routed), `load` each of several snapshots of the same kinds (including a compress-fusion snapshot into a model with feature skip off, and a post-activation snapshot into a pre-activation model). After each load, the six stored fields and `model.architecture` equal the snapshot's. `BuildNewModelModel(snapshot)` (the `init` path) reproduces each snapshot the same way. After one load, a topology edit (`draft.activationStyle` on a loaded draft) clears the site it removes, proving the loaded drafts' `onTopologyChange` closures are wired.
- `testADisappearedChoiceIsNeverRestored`, with no settling and no explicit sync call between the two edits of each pair (the D8 guarantee):
  - policy: `leaky_relu` chosen, `simple_conv`, back to `intermediate_conv` → `does_not_apply`;
  - fusion: compress routed with `relu` chosen, then routing removed (`featureSkipToPolicyHead = false` with no value route), then restored → `does_not_apply`;
  - stem: `draft.activationStyle = .post` on group 0, `relu` chosen, `.pre`, `.post` → `does_not_apply`;
  - tower end through group operations: a two-group tower whose last group is post; `moveGroup` brings the pre group last (the tower end appears), `relu` chosen; `moveGroup` back and forth again → `does_not_apply`; the same with `removeGroup` / `appendCopyOfLastGroup`.
- `testUseForEveryActivationMatchesDeriveSetActivation` (P3).

**`DrewsChessMachineTests/ArchitectureActivationSiteGraphTests.swift`** (GPU; `XCTSkip` without Metal; fp32 unless stated):
- `testLegacyDecodedArchitectureBuildsTheIdenticalForwardPass`: a fixture re-stamped `"8"` with the six keys stripped and `activation_function: gelu` added; same weights; policy and value outputs are bit-identical (`bitPattern`).
- `testEachHeadSiteChangesOnlyTheHeadItFeeds`: one weight set, ReLU everywhere vs one site at `leaky_relu`.
  - value conv or value FC1 hidden: policy logits bit-identical, value logits differ;
  - policy: value bit-identical, policy differs;
  - tower end: both differ.
- `testEachSiteBuildsItsSelectedFunction`. A numeric check, not op wiring.
  - Why not the wiring walk: copying `LayerHealthTests`' direct-consumer/name walk would show only that an op named `*_act` reads the BN, not which function was built. It also fails for GELU, whose final named multiply reads `<name>_halfx` and `<name>_1plus`, not the BN output (`ChessNetwork.swift:2390-2400`).
  - Instead, build `ChessNetwork(arch:bnMode: .inference, initialization: .seeded(initSeed:), analysisTaps: true)` (`ChessNetwork.swift:601`, `:649`) on the full-site fixture. Read `evaluateAnalysisTaps(boards:count:)` (`:2125`) on a handful of positions, with the running variances set away from 1 and the β values spread across both signs so every function's negative side is exercised.
  - From the tapped site inputs, compute on the CPU in Double: BN in inference form (ε `1e-5`, `:2626`), then the selected function (relu, `x ≥ 0 ? x : 0.01x`, `x·σ(x)`, `0.5x(1+erf(x/√2))`). Compare with the tapped output:
    - stem: `stem_bn_input` → `stem_output` (post-act first group);
    - tower end: `tower_final_bn_input` → `tower_output`;
    - fusion: `feature_skip_bn_input` → BN → function → the routed policy pre-conv (1×1, from the exported weights) → `policy_pre_bn_input`;
    - policy: `policy_pre_bn_input` → `policy_pre_act`;
    - value conv and value FC1 hidden: `value_bn_input` → BN → conv function → flatten → `value.fc1` (exported weight and bias) → hidden function → `value_fc1_act`.
  - The value FC1 site, which the BN walk never covered, is checked here.
  - Run once per site × function, one site at a time, in fp32. Tolerance: 1e-5 relative to the site's max |value|. That absorbs MPSGraph's fused BN and its erf against Foundation's. It is far below the smallest difference between the four functions on these inputs (leaky vs relu differs by 0.01·|x| on every negative element), and the test asserts that margin too, so it can never pass with the wrong function.
- `testFreshTrainablesAreActivationIndependent`:
  - same `initSeed`, ReLU everywhere vs leaky at every site of the full-site fixture;
  - every trainable tensor is bit-identical;
  - with only the three head sites changed, every BN running statistic is bit-identical too (same machine and precision; D7). The calibration runs a training-mode graph that contains the changed head activations (`Network/ChessMPSNetwork.swift:80-99`), so MPSGraph may compile the shared tower differently; any ulp difference is reported to the owner as a finding, never absorbed by a tolerance (the rule of `testEachHeadSiteChangesOnlyTheHeadItFeeds`).
- `testLeakyHeadsTrainOneFiniteStep`: bf16, under both `PolicyTailPrecision` values.
- `testAHeadActivationChangeChangesTheBehaviorFingerprint`: `BehaviorFingerprint.computeUncached` for ReLU vs a leaky value head; the SHA-256s differ.

**`DrewsChessMachineTests/BuildNewModelSiteActivationRenderTests.swift`** (hosted, like `BuildNewModelTowerShapeRenderTests.swift:20-60`). *As implemented (P1):* the drawn pickers cannot be read back in-process — SwiftUI builds no accessibility tree for an in-process query (the hosting view's `accessibilityChildren()` is empty without an accessibility client; checked with a scratch probe) and draws a Form picker as a graphics view, not an `NSPopUpButton`. So the picker's whole presentation is a value, `ArchitectureSiteActivationPicker.presentation(site:activation:siteExists:)` (enabled, needs-choice, the menu entries, the help), which the body draws and the test checks for every site of every hosted configuration; the screen is hosted and settled to prove it draws without a trap; picker presence on screen moves to V7 (checked on the running app). The planned checks were: host the screen and, for each of pre/post first group, pre/post last group, `simple_conv`, and the compress fusion on/off, settle it and check:
- it draws without a trap;
- all six site pickers are present in every configuration, the "Fusion activation" picker included, even with `featureSkipSource == .none` (T8);
- each picker's enabled state equals `siteExists`;
- after a topology change made through the model mutators the controls bind to (`draft.activationStyle`, `model.policyHeadStyle`, `model.featureSkip*`), then settling, a site that appeared shows its picker in the "choose…" state, and a site that disappeared has stored value `does_not_apply`. (Like the existing render tests, `BuildNewModelTowerShapeRenderTests.swift:66`, this drives the model, not a SwiftUI control.)

**Python:** `documentation/dashboards/tests/test_dcm_arch_site_activations.py` (new; run with `python3 -m unittest discover -s documentation/dashboards/tests`):
- the legacy resolution for v8, v3, unversioned and uniform-tower inputs, existing and absent sites;
- `norm_arch` / `norm_arch_md` on a uniform-tower input stamped v8 and v9: accepted, with `rezero_alpha_cap = rezero_alpha_init * REZERO_TANH_CEILING_MULTIPLE`; a block-groups v6+ input missing the cap still raises;
- the v9 missing-key `ArchitectureError`;
- the v9 retired key;
- stated-wins;
- both mismatch directions, and `does_not_apply` in a group field;
- an unknown token;
- `require_relu`: passes on an all-ReLU v9 file, raises naming the site for a non-ReLU site and for a named site the file lacks, and checks group main paths only with `block_main_path=True`;
- the `units.py` acceptance and rejection cases (T11);
- `fwd.forward` refusing a caller that does not pass `md=` (`TypeError`), and its guard refusing a leaky value FC1 file before any computation (`relu_inputs.py` imports `python-chess` at load, `:16`, which this test environment does not require, so its two callers are covered by the P1 re-run gates instead);
- `fwd16.forward` refusing an unchecked caller (a call without `sites=` raises `TypeError`), a feature skip, and a non-ReLU value FC1 hidden site; a pre-act fixture with `stem_activation: does_not_apply` passes, and one stating `stem_activation: relu` is refused by `site_activations_md` before `forward` is reached.

If a GPU test that asserts bit-identity across two different graphs (`testEachHeadSiteChangesOnlyTheHeadItFeeds`) shows ulp differences in the unchanged head, that is MPSGraph compiling the shared tower differently for a different downstream graph. It is reported to the owner as a finding before any tolerance is introduced.

### X2. Existing tests that must change (OD-6, OD-12)

Four causes:
- **The rename (OD-1)** makes the first rows compile errors, so none can keep compiling with a silently different meaning. Every such edit keeps the test's original intent: "the tower-level activation" becomes "every existing architecture-level site".
- **The version bump** fails `LineageRecordTests.swift:114` at run time, because it pins the literal `"8"`. The catalog pin is a P2 row.
- **`ActivationFunction.allCases` now includes `does_not_apply`** (OD-4, OD-14, D2): two test loops over `allCases` would fail, since `validate()` refuses `does_not_apply` in a group field, so they iterate `ActivationFunction.functions`.
- **Topology flips after construction (OD-4, D3)**: seven fixture helpers change a style or the fusion after building an architecture and then validate or build it. A site that disappears now holds a value it must not; a site that appears holds `does_not_apply`. Each gets one line that clears or chooses the affected site. The rest of each test is unchanged.

Totals: OD-6 (approved): 11 tests in 6 files; 14 lines in P1, plus the catalog pin in P2. OD-12 (new): 9 lines in 6 files, every one in P1. Every line is listed below, with its owner decision.

| File:line | Today | Change | OD |
|---|---|---|---|
| `RuntimeArchReachTests.swift:292`, `:304` (`testSiluAndGeluChangeForwardPassVsReLU`) | `arch.activationFunction = x` | `try arch.setActivationAtEveryExistingSite(x)`. Under OD-1's alternative, setting only the stem on the pre-act `.current` changes nothing, and the `XCTAssertNotEqual`s fail. | 6 |
| `RuntimeArchReachTests.swift:331` (`testGeluTrainerStepProducesFiniteLosses`) | same | same. Without it the test would train a model with no GELU anywhere and still pass. | 6 |
| `LeakyReLUTests.swift:121`, `:124` (`testLeakyReLUChangesTheForwardPassAndTrains`) | same | same | 6 |
| `LeakyReLUTests.swift:170`, `:178` (`testSetActivationChangesEverySiteAndCopiesEveryTensor`) | builds `source` / `expected` through `activationFunction` | through `setMainActivationEverywhere`, so `expected` matches the new `--set-activation` | 6 |
| `LeakyReLUTests.swift:175` | asserts `targetArchitecture.activationFunction == .leakyRelu` | asserts, for every `ArchitectureActivationSite`, `activation(at:) == (hasActivationSite(site) ? .leakyRelu : .doesNotApply)` | 6 |
| `LeakyReLUTests.swift:197` (`testSetActivationRefusesANoOp`) | `source.activationFunction = .leakyRelu` | `try source.setActivationAtEveryExistingSite(.leakyRelu)` | 6 |
| `SEActivationTests.swift:370` (`testFC1UsesSEActivationIndependentlyOfTheMainPath`) | `leakyEverywhere.activationFunction = .leakyRelu` | `try leakyEverywhere.setActivationAtEveryExistingSite(.leakyRelu)` | 6 |
| `SEActivationTests.swift:543` (`testSetActivationLeavesSEGroupsFC1Alone`) | asserts `activationFunction == .leakyRelu` | the per-site assertion of the `LeakyReLUTests.swift:175` row | 6 |
| `SEActivationTests.swift:592` (`testBuildScreenAndDeriveApplyTheSameActivationRule`) | `model.activationFunction = .leakyRelu` | `try model.applyMainActivationEverywhere(.leakyRelu)`. The draft loop after it stays (a no-op) and the assertion is unchanged. `applyMainActivationEverywhere` lands on the model in P1 (only the menu that calls it waits for P3), so this is a P1 edit. | 6 |
| `DeriveTrainedSourceTests.swift:87` (`testArchitectureOnlyOperationsStayAllowedOnATrainedSource`) | asserts `activationFunction == .leakyRelu` | the per-site assertion of the `LeakyReLUTests.swift:175` row | 6 |
| `LineageRecordTests.swift:114` (`testFileRoundTripCarriesTheRecordAndItsMirrors`) | `XCTAssertEqual(md[SafetensorsModelIO.Key.formatVersion], "8")` | `String(ArchitectureFormat.currentVersion)`, the form the version-gate suites already use (`SEBetaInitTests.swift:301`, `RezeroAlphaCapTests.swift:133`). The test is about lineage carriage, not about which version is current, so it stops failing at every future bump. (P1) | 6 |
| `SEBetaInitTests.swift:610` (`testOperationCatalogDrivesTheCLI`) | pins the derive catalog's flag list | the six site flags are inserted after `"--set-se-activation"` (P2). Precedent: owner-approved in `de0f22be`. | 6 |
| `SEActivationTests.swift:249` (`testRoundTripPreservesEveryValue`) | `for value in ActivationFunction.allCases` | `for value in ActivationFunction.functions` | **12** |
| `SEActivationTests.swift:319` (`testSELessGroupMustMatchItsActivation`, "any value is valid on a group that has an SE block") | same | same | **12** |
| `HeadNumericsTailTests.swift:38` (helper `arch(_:policy:value:)`; `testHeadOutputsAreFP32ForEveryComputeDtypeAndPolicyStyle` loops over `PolicyHeadStyle.allCases` at `:57`, which includes `simple_conv`) | `if let policy { a.policyHeadStyle = policy }` | one line after it: `a.clearActivationSitesTheTopologyLacks()` (simple_conv removes the policy site; the other styles keep it, so nothing appears) | **12** |
| `PolicyTailPrecisionTests.swift:32` (helper `arch(_:policy:)`; loops over `PolicyHeadStyle.allCases` at `:60`, `:75`, `:108`) | same | same one line | **12** |
| `InitNeutralOptionsTests.swift:106` (`testNeutralLeavesOptionsWithoutTheirLayerStandard`) | `arch.blockGroups[0].activationStyle = .pre` on a post-activation fixture, then `XCTAssertNoThrow(try neutral.validate())` (`:110`) | one line after it: `arch.clearActivationSitesTheTopologyLacks()` (the stem disappears; the last group stays post, so no site appears) | **12** |
| `LayerHealthTests.swift:56-73` (`mixedArch`, used by `:111`, `:186-187`, `:238`, `:517`) | replaces a pre-activation tower's groups with a post → pre tower, so the stem appears holding `does_not_apply`; `:117` validates it and `:187` asserts the stem tag is `leaky_relu` | one line after the groups: `arch.stemActivation = .leakyRelu` (the value `:187` already asserts) | **12** |
| `LayerHealthTests.swift:75-83` (`compressSkipArch`) | sets `featureSkipFusion = .compressConvBNReLU` with a routed policy head, so the fusion site appears holding `does_not_apply` | one line: `arch.featureSkipActivation = .relu` (what the node computed when it read the tower-level ReLU) | **12** |
| `NetworkArchitectureTests.swift:356` (`testFeatureSkipValidationRules`, "Valid: compress to heads") | `compress.featureSkipFusion = .compressConvBNReLU`, then `XCTAssertNoThrow(try compress.validate())` | one line: `compress.featureSkipActivation = .relu` | **12** |
| `NetworkArchitectureTests.swift:384` (`testFeatureSkipCompressModeDualContract`) | `a.featureSkipFusion = .compressConvBNReLU`, then `try a.validate()` | one line: `a.featureSkipActivation = .relu` | **12** |

Verified *not* to need edits:
- group-level reads such as `SEActivationTests.swift:148`, `:545`, `LeakyReLUTests.swift:122`, `:176`, `BuildNewModelDraftTests.swift:43`;
- the record-name string in `GraftDeriveTests.swift:236`;
- uniform-tower JSON fixtures (`BlockGroupArchitectureTests.swift:41-63`, `RezeroAlphaCapTests.swift:194-200`, `SEActivationTests.swift:188-198`), which stay legacy by construction and resolve consistently;
- every test that re-stamps a current encode as an older version (D5 rule 1: the six keys are stated and consistent);
- every assertion that a current-format reload has no legacy log line (`SEActivationTests.swift:255`, `:472`, `RezeroAlphaCapTests.swift:266`, `:517`, `SEBetaInitTests.swift:344`, `:514`): a v9 file states all six keys;
- `InitNeutralOptionsTests.swift:207` (`preAct`): it asserts `validate()` names `branch_output_init`, which D3's order reports before the site mismatch;
- `NetworkArchitectureTests.swift:366` (`compressFinal`) and `:372-373` (`noDest`): they assert only that `validate()` throws (the feature-skip error comes first), and `noDest` uses `concat_direct`;
- every fixture that replaces or appends groups without changing the first or last group's style: `BlockGroupArchitectureTests.swift:16`, `:107`; `BuildNewModelDraftTests.swift:24`; `BuildNewModelGroupRemovalRenderTests.swift:28`; `BuildNewModelTowerShapeRenderTests.swift:28`; `DropoutRNGStateTests.swift:21`; `DropoutRunStreamTests.swift:23`; `DropoutGraphWiringTests.swift:24`; `InitNeutralOptionsTests.swift:53`; `NetworkArchitectureTowerShapeTests.swift:31`, `:111` (`concat_direct`); `MacOS27NaNIsolationTests.swift:988`, `:1153`, `:1253`; `RezeroAlphaCapTests.swift:58`; `SEActivationTests.swift:60`; `SEBetaInitTests.swift:40`; `WeightInitializationTests.swift:41`;
- `StandardPathForwardPinTests.swift:61`: its fixtures set `intermediate_conv` (the current style) or `fc_bottleneck`, both with a pre-block, so no site appears or disappears and the pinned bits test runs unmodified;
- `LayerHealthTests.concatToFinalBlockArch` (`:85-93`): `concat_direct` builds no fusion node, and `simple_conv` comes through the convenience init;
- `LayerHealthTests.swift:301` iterates an explicit `[.silu, .gelu]`, not `allCases`.

### X3. Existing tests that guard the change

Each of these must pass. Each is **unchanged apart from the X2 edits listed for it**; every other class here is untouched.

- `StandardPathForwardPinTests` (pinned forward bits; no edit)
- `BehaviorFingerprintTests`
- `BlockGroupArchitectureTests` (golden summaries)
- `SEActivationTests` (X2 lines only), `RezeroAlphaCapTests`, `SEBetaInitTests` (X2 line only), `InitNeutralOptionsTests` (X2 line only; version-gate patterns)
- `LineageRecordTests` (X2 line only)
- `LayerHealthTests` (X2 lines only)
- `NetworkArchitectureTests` (X2 lines only)
- `PolicyTailPrecisionTests`, `HeadNumericsTailTests` (X2 line each)
- `ArchitecturePresetStoreSaveTests`
- `GraftDeriveTests`
- `BuildNewModelDraftTests`, `BuildNewModelGroupRemovalRenderTests`, `BuildNewModelTowerShapeRenderTests`
- `documentation/dashboards/tests/test_tooling.py`

---

# Part P — Phasing

Each phase builds on its own and is committed after it builds and its targeted tests pass (the owner's standing rule for approved multi-phase plans; no push). One build at the end of each phase, through `drews-xcode-mcp` `build_project`. No test or CLI run while a training run is live.

### P1 — Format v9, `does_not_apply`, graph, layer health, `--set-activation`, Build screen pickers, Python (T1–T4, T7-P1, T8 except the menu, T9-P1, T11, ROADMAP entry)
**P1 starts only after OD-12 and OD-14 are answered** (they decide P1's test edits and whether `CaseIterable` stays; both answered 2026-10-05: OD-12 approved, OD-14 keep). P1 is the first commit whose builds write v9 files, so the Python readers and guards move with it (T11). No commit leaves the tooling unable to read what the app writes. The Build screen's six pickers land here too: with `does_not_apply` a style switch can make a site appear that must be chosen, so the screen needs every picker from the first v9 build.
- All of T1–T4.
- `SetActivationDeriveOperation` and the `SetSEActivationDeriveOperation` value list.
- The Build screen model, pickers and topology sync (T8 except the "Use for every activation" menu, which is P3; `applyMainActivationEverywhere` itself lands in P1 for the X2 `SEActivationTests.swift:592` edit).
- The minimal diagram edits the rename requires (T9-P1).
- The `deriving-models.md` `--set-activation` row and prose (`:57`, `:74-78`), the legacy-rule line (`:188-193`), and the refusal of `does_not_apply` by `--set-activation` / `--set-se-activation` (`:203-209`), which ships in P1.
- The X2 edits except `SEBetaInitTests.swift:610`.
- The new pure and GPU test files (X1), except the cases marked P2 / P3, and the render test file.
- All of T11: `dcm_arch.py`, the reusable readers, the three frozen-script guards, the new Python test, and the frozen-script note (written into `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20 in P4; until then this plan's T11 tables are the record).
- The `ROADMAP.md` entry (T12, OD-11).
- Gate:
  - build;
  - targeted runs of X1 (P1 part), the render test file, and every X2 and X3 class;
  - `python3 -m unittest discover -s documentation/dashboards/tests` (all pass, including the committed `test_tooling.py`);
  - `relu_inputs.run` and `value_stage.net_probs`, pre- vs post-edit on `F`, `R7`, `R8` and identical inputs, give identical results (T11); `relu_inputs_bot.json` reproduction is a secondary check;
  - then the **full suite** (graph builder and persistence changed, per CLAUDE.md "Running the tests"), with the test plan's slow gate as configured.

### P2 — Per-site derive setters (T7-P2)
- `Persistence/SiteActivationDerive.swift`, the catalog entries, `deriving-models.md` (the rest of T7), the `SEBetaInitTests.swift:610` edit and the X1 cases marked P2.
- Gate: build; `-only-testing:` `ArchitectureActivationSiteTests`, `SEBetaInitTests`, `SEActivationTests`, `LeakyReLUTests`, `DeriveTrainedSourceTests`, `GraftDeriveTests`, `RezeroAlphaCapTests`, `InitNeutralOptionsTests`.

### P3 — "Use for every activation" and the diagram (T8 menu, T9-P3)
- The menu (OD-8), the diagram's per-site lines, and the X1 cases marked P3.
- Gate: build; `BuildNewModel*` tests, the render tests and the X1 Build-screen cases; plus a screen check (V7).

### P4 — Documentation, presets, ROADMAP completion (T12, OD-10)
- `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §20, `CLAUDE.md` (OD-7), `CHANGELOG.md`, this plan's status line, and the `ROADMAP.md` entry marked complete (no detail removed).
- **Re-save the user presets at v9 (OD-10)**, after V1–V8 pass, with no training run live:
  1. Back up first. Copy every file in `~/Library/Application Support/DrewsChessMachine/Presets/` into a new folder `~/Library/Application Support/DrewsChessMachine/Backups/Presets-pre-v9-<YYYYMMDD>/`, created fresh (`mkdir` without `-p` on the leaf, so an existing folder stops the step) with `cp -p` (no `-f`). Compare SHA-256 of every original and copy; stop on any difference.
  2. Re-save each `.json` preset through the app's own save path: Build New Model ▸ load the preset ▸ type the preset's existing label into the label field (otherwise the saved label becomes the file name, `BuildNewModelView.swift:310`) ▸ Save as Preset with the same name ▸ confirm Replace (`:205-218`; `FileSafety.replaceRegularFile`, identity-checked). Non-preset files there (`default.profraw`) are left alone.
  3. Check each re-saved file against its backup:
     - **Shape** (stdlib `json`): `format_version` is `9`; all six site keys are present; there is no top-level `activation_function`; the `label` equals the backup's.
     - **Same architecture, as NEW decodes it.** A key-by-key JSON comparison would be wrong: the re-save drops `activation_function` and writes every field the older file left to a legacy resolution (head and group init options, `se_activation`, `rezero_alpha_cap`, …; `NetworkArchitecture.swift:1323-1335`). Instead mint both with NEW: `"$NEW" --new-model --architecture <backup.json> --init-seed 20261005 --out-model "$S/preset-old-<name>.safetensors"` and the same for the re-saved file into `preset-new-<name>`. The two files' `architecture` metadata, parsed as JSON, are equal (both were encoded by NEW from the decoded values, so equal decoded architectures give equal JSON), and every trainable tensor is bit-identical (same seed and architecture).
     - If any check fails, the backup is copied back over that preset (it is the verified original) and the failure is reported.
  - Committed experiment presets under `experiments/` are not re-saved: they are reproduction records and load unchanged (D7).
- Gate: the full suite once more if any code changed after P1's full run; the V-series below.

---

# Part V — Validation

Before P1 starts, freeze the build of P1's parent commit as `~/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-<build>-<hash>.app` (as the existing `DCM-2323-58e9f952-lrmax10.app` was). Call it `OLD`. `NEW` is the post-P1 (and later post-P4) build's binary. `S` is a scratch folder. `M="$HOME/Library/Application Support/DrewsChessMachine/Models"`. Run only when no training run is live.

**V1. Build.** Each phase's `build_project` reports `build_failed: false`.

**V2. Tests.** Per phase, as listed in Part P. At the end, the full suite passes (CLAUDE.md: about an hour cold). Every X3 class passes, unchanged apart from the X2 edits listed for it.

**V3. Existing models give bit-identical outputs (real files).** For each evidence file (the v8, v5, v4-leaky, v3-gelu, v3-post-act and v3-simple_conv rows above):
- run `"$OLD" --probe-model <file> --probe-set 200 --probe-out "$S/old-<n>.jsonl" --probe-positions-out "$S/old-pos-<n>.jsonl"`, and the same with `"$NEW"` into `new-*`;
- compare the two positions files line by line with a short `python3` `json` script, dropping only keys that name the build, the time or a path. Every remaining value is equal (exact float text);
- `NEW`'s session log has one `[ARCH] legacy file (format vN) <file>: …` line with the six site resolutions: each existing site `:= <that file's activation_function>` (`gelu` for T97X, `leaky_relu` for the v4 leaky file), each absent site `:= does_not_apply (<reason>)`. Expected absent sites: stem and fusion for the pre-act files; tower end, policy and fusion for the v3 post-act `simple_conv` file; stem, policy and fusion for the v3 pre-act `simple_conv` file.

**V3b. A resume across the change keeps its fingerprint (OLD vs NEW).**
1. With `OLD`, run a short corpus replay from a v8 model so that OLD records its own behavior fingerprint in a trainer file. For example: `"$OLD" --replay-corpus <the corpus named in 20261005-r7b24-fresh's runs' lineage records> --start-model "$M/20261005-r7b24-fresh.safetensors" --training-step-limit 2 --out-model "$S/old-trainer.safetensors"`. Its lineage record carries `rng.behavior_fingerprint` (recipe and SHA-256; `Persistence/LineageRecord.swift:517-528`).
2. With `NEW`, resume it exactly: `"$NEW" --replay-corpus <same> --start-model "$S/old-trainer.safetensors" --resume-exact --training-step-limit 4 --out-model "$S/new-trainer.safetensors"`.
3. Expected: `NEW`'s session log reports the build change as `[RESUME] build changed …` with `behavior fingerprint matches` (`Training/ResumeExactness.swift:113-121`), and its `[RESUME]` verdict carries no `build` gap.
4. `new-trainer.safetensors` is `dcm_format_version` `"9"`, states `stem_activation` and `feature_skip_activation` as `does_not_apply` and the other four as `relu`, and its record's fingerprint SHA-256 equals OLD's.

**V4. Layer health on an existing model is unchanged.**
- `"$OLD" --analyze-numerics "$M/20261005-r7b24-fresh.safetensors" --numerics-static-only --numerics-out "$S/old-num"`, and the same with `NEW` into `new-num`.
- The layer-health table (`act` column, counts, extremes) is identical.

**V5. Mint and derive a leaky-head model on a ReLU tower.**
1. `"$NEW" --new-model --architecture v4_5block_7x7 --init-seed 20261005 --out-model "$S/base.safetensors"`.
2. `"$NEW" --derive-model --from "$S/base.safetensors" --set-policy-head-activation leaky_relu --set-value-head-conv-activation leaky_relu --set-value-head-fc1-hidden-activation leaky_relu --out "$S/leaky-heads.safetensors"`.
3. Dump both headers (stdlib `json` + `struct` read of the safetensors header, as in `HPARAM_RECORDING_PLAN.md`'s dumper) and check:
   - `dcm_format_version` = `"9"`;
   - all six keys present; no top-level `activation_function`;
   - `stem_activation` and `feature_skip_activation` = `does_not_apply`, and `tower_end_activation` = `relu`, in both;
   - the three head keys `leaky_relu` only in the derived file;
   - `derivation_history`'s last record (one record per derivation, one operation entry per operation, `Persistence/ModelDerivation.swift:369-373`) has exactly three operations, in catalog order: `set-policy-head-activation`, `set-value-head-conv-activation`, `set-value-head-fc1-hidden-activation`. Each lists only its own field in `changed_architecture_fields` and has an empty `rewritten_tensors`, with `arguments` `{"value": "leaky_relu"}`;
   - every tensor's bytes are identical between the two files.
4. Write the derived file's architecture into a v9 `NamedArchitecture` preset JSON in `$S` and run `"$NEW" --new-model --architecture "$S/leaky-heads.json" --init-seed 20261005 --out-model "$S/leaky-heads-minted.safetensors"`. Every trainable tensor and every BN running statistic is bit-identical to `base.safetensors` (D7: only head activations changed). A BN-statistic ulp difference is reported to the owner, never absorbed by a tolerance (Risks).
5. Refusals, each exiting non-zero with a message naming the flag and the reason:
   - `--set-stem-activation leaky_relu` on `base` (pre-act: the stem does not exist);
   - `--set-value-head-conv-activation does_not_apply` on `base` (not an activation function);
   - `--set-activation does_not_apply` on `base`;
   - repeating step 2's flags on `leaky-heads.safetensors` (no-op).
6. Mismatch on load: copy `leaky-heads.json` to `$S/bad.json` with `stem_activation` set to `relu`; `"$NEW" --new-model --architecture "$S/bad.json" …` exits non-zero naming `stem_activation`, the stem site and `bad.json`. The same with `value_head_conv_activation` set to `does_not_apply`.

**V6. Per-site layer health.**
- `"$NEW" --analyze-numerics "$S/leaky-heads.safetensors" --numerics-static-only`. The batch-norm table lists `stem.bn -`, `tower_final_bn relu`, `policy.pre_bn leaky_relu` and `value.bn leaky_relu`, followed by `velocity checks: not included (…)`: a fresh `--new-model` file carries no optimizer velocity (`App/NewModelCLI.swift:145`, `includesVelocity: false`), a derive copies that, and the FC hidden-unit rows are printed only with velocity (`Training/LayerHealth.swift:1060-1068`).
- The value FC1 activation is checked on a trainer checkpoint of the same architecture: a 2-step corpus replay from `leaky-heads.safetensors` (the V3b corpus, `--training-step-limit 2 --out-model "$S/leaky-heads-trained.safetensors"`), then `--analyze-numerics` on that file. Its velocity rows list the value FC1 layer with `leaky_relu` (`LayerHealth.swift:1143`), and its BN rows match the line above. `leaky-heads.safetensors` itself is kept unchanged for V5's byte comparisons.
- A derive with `--set-value-head-conv-activation silu` shows `value.bn` as `silu` with `n/a` counts, while `tower_final_bn` stays classified.

**V7. Build New Model.** On an idle machine, open the screen. Check that:
- every site picker is present, "Fusion activation" included with the feature-skip source at `none`;
- "Stem activation" is disabled, showing "does not apply", on the default (pre-act) tower;
- switching group 1 to `post` enables "Stem activation" in the "choose…" state, shows the `stem_activation` validation message, and disables Build and Save as Preset; choosing `relu` clears the message;
- switching group 1 back to `pre` disables the picker again, and switching it to `post` once more shows "choose…" again (the earlier choice is not restored);
- "Tower-end activation" is disabled with a post last group;
- "Pre-block activation" is disabled on `simple_conv`, and switching back to `intermediate_conv` asks for a choice;
- "Fusion activation" is enabled only for `compress_conv_bn_relu` with a routed head;
- "Use for every activation ▸ leaky_relu" (P3) fills every existing site, makes the summary read ` . act leaky_relu` and every group `leaky_relu/pre`;
- setting only the value pickers to `leaky_relu` gives the listed-sites summary and the diagram's per-site lines;
- Save as Preset writes `format_version: 9` with all six keys (`does_not_apply` at the absent ones), and reloading it restores the same picker values.

**V8. Python.**
- `python3 -m unittest discover -s documentation/dashboards/tests` passes.
- `dcm_arch.site_activations_md` on `leaky-heads.safetensors` returns the expected six values; on `T97X` it returns `gelu` at the four existing sites and `does_not_apply` for the stem and fusion.
- Each frozen-script guard passes on the checkpoints that script was run on: a short script imports `dcm_arch` and calls `require_relu` with that script's site list on each file:
  - `fwd.py`: guard-only coverage. No script in the repo calls `fwd.forward`, and its own asserts (one pre-activation group, `se_style` `none`, `fwd.py:87`, `:94`) fit none of the folder's recorded checkpoints, so there is no checkpoint it is known to have run on. The check calls its guard (sites and main path) on `20260702-Qeu8-resume3-replay-step681000.safetensors` (Ejp0 @681k, a ReLU pre-activation `intermediate_conv` tower: passes) and on `leaky-heads.safetensors` (raises);
  - `fatty_tensors.py`: `20261003-fatty216-b2275-replay-step33000.safetensors` and `20261003-fatty224s3-b2275-replay-step33000.safetensors` (`fatty_tensors.py:12`, `:32`);
  - `input_features.py`: every file `fma_lib.discover("leaky")` and `fma_lib.discover("relu_s2")` return (`input_features.py:42-43`).
  - `fatty_tensors.py` and `input_features.py` also raise, naming the site, on `leaky-heads.safetensors` (V5) for the sites they model that that file makes leaky (`fatty_tensors.py`: policy and value conv; `input_features.py`: value conv).

---

# Implementation notes

Recorded as each phase lands; every deviation from the text above is listed with its reason.

**P1** (`4e70c615`):
- Built once (`build_failed: false`); full suite on the scheme's test plan (slow suites on): 2,380 passed, 0 failed, 1 skipped (`LegacyDcmmodelLoadTests`, gated on `DCM_RUN_LEGACY_LOAD` as before). `drews-xcode-mcp`'s `run_project_tests` has no `-only-testing`, so every "targeted run" of this plan is the full suite (about 23 minutes on this machine).
- Python: `python3 -m unittest discover -s documentation/dashboards/tests` 117 passed (including the new `test_dcm_arch_site_activations.py`). Gate: pre-edit (`git show 20f64f68:…`) vs post-edit `relu_inputs.run` on F, R7, R8 over the same 4,097 bot positions — identical JSON; `value_stage.net_probs` on 256 positions from `value_stage.sample([20], 128, 2, 20261004)` — bit-identical arrays for all three.
- `OLD` frozen before P1 as `FrozenBuilds/DCM-2324-20f64f68.app` (build 2324 of `20f64f68`).
- `ArchitectureFormat.decodeSiteActivation` takes one more argument than D5 names, `allSiteKeys`, and returns `DecodedSiteActivation` (value + whether it was resolved), so `legacyActivationFunctionMissing` names every unresolved key whichever key is decoded first.
- `ActivationSiteMismatch` carries a `reason` (from the new `NetworkArchitecture.activationSiteReason(_:)`: why the site exists or not in this architecture), so the message can name the policy style or the group style, as D2's examples do. `ArchitectureActivationSite` also has `codingKey` (the `CodingKeys` case `jsonKey` reads) and `siteDescription` (the picker's help for an existing site). `ActivationFunction.functionList` renders `functions` for messages.
- OD-14 (keep `CaseIterable`): `ActivationFunction.functions` is `allCases` without `does_not_apply`; the picker lists go through `ArchitectureSiteActivationPicker.functionChoices` and `BuildNewModelView.groupActivationChoices`, pinned by the new `testEveryActivationChoiceListIsTheFunctionsList`. `enumPicker` keeps its constraint.
- `BuildNewModelModel` gained `existingActivationSites` (one composition per redraw, beside `siteExists(_:)`) and `storedActivation(at:)`; `applyMainActivationEverywhere` logs `[BUTTON] Build New Model: Use for every activation <fn>`.
- The render test checks the picker's presentation value instead of the drawn controls (see its X1 entry).
- `ModelDerivation.parseActivationFunctionValue` / `doesNotApplyRefusal` are the one parser and refusal of every activation operation (P2's site setters use them too).

**P2** (commit: see `git log`; recorded here by the next phase's commit):
- `Persistence/SiteActivationDerive.swift`: one `SetSiteActivationDeriveOperation(site:value:)`, its six catalog kinds built from `ArchitectureActivationSite.allCases` (`SetSiteActivationDeriveOperation.kinds`, inserted after `--set-se-activation`). Operation names are `set-` plus the site's JSON key in kebab case.
- `SEBetaInitTests.swift:610` catalog pin updated (OD-6). The P2 cases of X1 added to `ArchitectureActivationSiteTests`; `testDirectlyConstructedOperationsRefuseDoesNotApply` now covers the site setter too.
- Built once (`build_failed: false`); full suite: 2,387 passed, 0 failed, 1 skipped (the env-gated legacy-load test).
- `deriving-models.md`: the six rows, the order note, two examples, the trained-source and refusal lines.

# Owner decisions

**Owner answers (2026-10-05):** OD-1 approved (rename to `stem_activation`); OD-2 approved (own `feature_skip_activation`); OD-3: name the value FC1 field `value_head_fc1_hidden_activation`; OD-4: an explicit `does_not_apply` value (owner: "none" reads ambiguously) — required exactly when the topology lacks the site, refused on a site that exists, never meaning an identity activation; OD-5 approved; OD-6 approved; OD-7 approved; OD-8 approved; OD-9: both — document the frozen scripts **and** add the guard to each silent one; OD-10: re-save the presets at v9 (approved, optional); OD-11: add to `ROADMAP.md` (approved).

| # | Decision | Recommendation | Status |
|---|---|---|---|
| OD-1 | Rename the top-level `activation_function` → `stem_activation` (Swift `activationFunction` → `stemActivation`), or keep the key and the Swift name with the narrowed meaning "stem activation" | **Rename** (D1). Keeping it gives one key two meanings across format versions, and three committed tests would keep compiling while testing something else (X2). | **Decided: rename** (2026-10-05). |
| OD-2 | Give the feature-skip compress node its own `feature_skip_activation`, or (b) let it follow the stem field, or (c) follow the tower-end field | **Own field.** (b) is the dual meaning the brief rules out. (c) fails for a post-act tower with a compress node, which has a fusion activation but no tower-end activation. | **Decided: own field** (2026-10-05). D1, D4, T2, T3, T4, T8 and X1 use `feature_skip_activation` throughout. |
| OD-3 | Name the value FC1 field `value_head_hidden_activation` (pairs with `value_head_hidden_units`) or `value_head_fc1_activation` | `value_head_hidden_activation` | **Decided: `value_head_fc1_hidden_activation`** (2026-10-05; Swift `valueHeadFC1HiddenActivation`, site `valueHeadFC1Hidden`, summary label `value_fc1_hidden`, flag `--set-value-head-fc1-hidden-activation`, picker "FC1 hidden activation"). Applied everywhere. |
| OD-4 | Sites the topology lacks: keep any value, never read it (D3 option 1); pin; or Optional/`null` | **Keep, never read** | **Decided: explicit `does_not_apply`** (2026-10-05), required exactly when the site is absent and refused on an existing site, never an identity activation. D2, D3, D5, D6, D8, T2-T4, T7, T8, T11, X1, X2 and V rewritten to it; the "keep, never read" design is removed. |
| OD-5 | Refuse a v9+ block-groups architecture that still states a top-level `activation_function` (`FormatError.retiredField`) | **Refuse**; the uniform-tower form is exempt | **Decided: refuse** (2026-10-05). |
| OD-6 | Approve the test edits in X2: 11 tests in 6 files, 14 lines in P1 (13 forced by the rename, plus `LineageRecordTests.swift:114`'s format pin) and the catalog pin `SEBetaInitTests.swift:610` in P2 | Approve. Each is mechanical and keeps the test's intent. | **Decided: approved** (2026-10-05). The rows' new text follows OD-3/OD-4 (`setActivationAtEveryExistingSite`, per-site assertions with `does_not_apply` at absent sites). |
| OD-7 | Approve the `CLAUDE.md:136` text in T12 | Approve | **Decided: approved** (2026-10-05). The text now mentions `does_not_apply` and the OD-3 name. |
| OD-8 | Add the Build screen's "Use for every activation" menu | Yes. It is the `--set-activation` rule, from one shared function. | **Decided: yes** (2026-10-05). P3. |
| OD-9 | Python scope: update `dcm_arch.py` and `relu_inputs.py` (in the brief) and the other reusable readers `fwd16.py` and `units.py`. Frozen scripts: leave them unedited and document (T11 table; some fail loudly, others would silently model ReLU), or add a one-line guard to each silent one. | Update the reusable readers; leave frozen scripts unedited and document them | **Decided: both** (2026-10-05): reusable readers updated, every frozen script documented, and a one-line `dcm_arch.require_relu` guard added to each silent one (`fwd.py`, `fatty_tensors.py`, `input_features.py`; T11). P1. |
| OD-10 | Re-save the user presets in `Presets/` at v9 | Not needed. They load unchanged; re-saving only removes their `[ARCH] legacy` line. Owner's choice. | **Decided: re-save** (2026-10-05). P4, after a verified backup, through the app's own save path (Part P). |
| OD-11 | Add this plan to `ROADMAP.md` | Owner's call (standing rule) | **Decided: add** (2026-10-05). Entry added in P1, marked complete in P4 (T12). |

**New decisions raised by the revision:**

| # | Decision | Recommendation | Status |
|---|---|---|---|
| OD-12 | Approve the 9 additional existing-test lines OD-4 causes (X2 rows marked **12**): `SEActivationTests.swift:249`, `:319` (`allCases` → `functions`); one clearing line after the style change in `HeadNumericsTailTests.swift:38`, `PolicyTailPrecisionTests.swift:32` and `InitNeutralOptionsTests.swift:106`; one explicit site value in `LayerHealthTests.swift` `mixedArch` (stem `leaky_relu`) and `compressSkipArch` (fusion `relu`), and in `NetworkArchitectureTests.swift:356`, `:384` (fusion `relu`). | Approve. Each keeps the test's intent and makes its fixture a legal architecture under OD-4. The alternative is a design in which absent sites hold any value, which OD-4 rules out. | **Decided (owner, 2026-10-05): approved** — edit the 9 test lines. |
| OD-13 | An SE-less group's `se_activation` keeps today's rule (it must equal the group's `activation_function`, `NetworkArchitecture.swift:1631-1638`) rather than becoming `does_not_apply`. The SE FC1 is a site the topology lacks there, so the OD-4 principle could apply. | **Keep today's rule.** The brief scopes `does_not_apply` to the architecture-level sites. Changing a group field would need format v9 to rewrite every group's `se_activation` resolution, touch `BlockGroup.setActivationFunction`, the group editor and `SEActivationTests.testSELessGroupMustMatchItsActivation`, and would change no graph. It can be a later, separate change. | **Not answered by the owner; implemented as recommended ("keep today's rule")** (2026-10-05, per the owner's instruction to implement the recommendation). |
| OD-14 | Remove `CaseIterable` from `ActivationFunction` (D2), so every list of activations is chosen explicitly (`ActivationFunction.functions`), or keep it and replace each known `allCases` use by hand | **Remove it.** The compiler then finds every use, including any added later; kept, `allCases` silently offers `does_not_apply` in pickers and derive value lists. | **Decided (owner, 2026-10-05): keep `CaseIterable`**; every picker, derive value list and test loop uses `ActivationFunction.functions` (= `allCases` without `does_not_apply`), and a new test pins that `functions` excludes `does_not_apply` and that each picker / derive list equals `functions`. (A future `allCases` use is not caught by the compiler; that test and review are the guard.) |
| OD-15 | `TRAINING_HEALTH_ALARMS_PLAN.md` rule 3 (value-FC1 zero velocity) under a non-`relu` `value_head_fc1_hidden_activation`. `LayerHealth` keeps reporting `valueFC1ZeroVel` either way (T4). | Rule 3 applies only to `relu`. | **No decision needed in this plan.** The alarms plan's own design already says this (its rule-3 note, `TRAINING_HEALTH_ALARMS_PLAN.md:276`); whether that text is owner-approved is tracked in that plan, not here. This plan only makes `valueFC1Layer(for:)` return the new field (T4), which that note relies on. Does not block any phase. |

---

# Risks

- **Older builds cannot read v9 files.** Any file the new build writes (`unsupportedFutureVersion`), including a resumed training run's next checkpoint and the re-saved user presets (OD-10), cannot be read by an older build. The live lrA/lrB runs are unaffected while they keep running on their own binary. Resuming one of them with the new build makes every later file v9-only. This is the same as every earlier format bump. The pre-v9 preset backup (Part P, P4) keeps the old presets for older builds.
- **A hand-edited file with a site mismatch now fails to load** (D3), where before the earlier draft would have accepted any value. That is the owner's rule; the error names the key, the site and the file.
- **Copied BN statistics after a derive.** A `--derive-model` that changes the tower-end activation of a *fresh* net copies BN running statistics calibrated under the old activation (D7). This already happens with `--set-activation`. Training replaces them with momentum updates within the first steps, and both arms of an A/B still start from identical trainables, which is what derive is for. (No derive operation changes which sites exist, so none can introduce or remove a stem or fusion activation.)
- **Stale enum name.** `FeatureSkipFusion.compressConvBNReLU` / `compress_conv_bn_relu` names ReLU, but its activation becomes `feature_skip_activation`. The token is persisted, so it is not renamed (non-goal). The Build screen caption already says "1×1-conv→BN→act" (`BuildNewModelView.swift:172-175`).
- **Build screen topology edits must go through the model.** The sync runs inside each model mutation that can change the sites (D8). A future control that wrote `draft.group.activationStyle` or a topology field around those paths would skip it; the composed architecture would still be cleared (never invalid), but a disappeared choice could come back. `testADisappearedChoiceIsNeverRestored` covers every path that exists today.
- **Alarms plan rules 2 and 3** (`TRAINING_HEALTH_ALARMS_PLAN.md`, T4): the set of classified BN sites now varies with each model's site activations, and rule 3 is meaningful only under a `relu` value FC1 hidden activation. The alarms plan's rule-3 note already limits rule 3 to `relu` through `LayerHealth.valueFC1Layer(for:)`, which T4 points at the new field. Whichever plan lands second re-checks both rules against the other.
- **Bit-identity across graphs (X1, V5).** If MPSGraph compiles the shared tower differently when only one head's activation changes, three checks can show ulp differences: the "only the head it feeds changes" test, the BN running statistics in `testFreshTrainablesAreActivationIndependent`, and V5 step 4 (the BN calibration forward runs a graph containing the changed heads, `Network/ChessMPSNetwork.swift:80-99`). Each is reported to the owner as a finding, never hidden behind a tolerance.
- **Extra `[ARCH]` lines until the presets are re-saved.** After the bump every user preset is pre-v9, so each Build-screen model `init` (re-run on every SwiftUI view init) logs one `[ARCH] legacy` line per preset: 10 instead of today's 6 (the 4 v8 presets join the 6 unversioned ones). OD-10's re-save in P4 removes them.
- **Smooth or leaky tower-end in bf16/fp16.** These are new numerics paths for training. `testLeakyHeadsTrainOneFiniteStep` covers bf16 under both tail precisions. fp16 inherits `LeakyReLUTests`' tolerance handling. Any experiment watches `[LAYER-HEALTH]` and `pLogitMean` / `vLogitMean` as usual.
- **Python scripts on v9 files.** Python that reads the top-level key on a v9 file raises `KeyError`, and the three silent frozen scripts now raise from their guard. This is intended: loud, never silent (T11).
- **Test time.** The full suite takes about an hour; P1 and the end each need one full run on an idle machine.

---

# Non-goals

- Per-block activations within a group (a group remains one recipe; use count-1 groups).
- Learnable or per-site leaky slopes (PReLU), and making `leakyReLUNegativeSlope` a field.
- New activation functions. (`does_not_apply` is a marker, not a function.)
- Applying `does_not_apply` to group-level fields (OD-13).
- The SE gate (sigmoid) and the value output (softmax / tanh), which stay structural.
- Changing any default: presets, `newModelDefault` and `--new-model` builds stay ReLU at every existing site. Whether a head should default to leaky is an experiment for later.
- Rewriting or migrating existing model files, sessions or experiment JSONs (the user presets are re-saved by the owner's choice, OD-10, through the normal save path).
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

## Pass 3 (2026-10-05)

The reviewer verified the three pass-2 fixes against the code and found no new errors: **concurred**.

## Owner-answer revision (2026-10-05)

Applied before the next review pass. Besides the design changes listed in the status line, the revision re-checked the frozen-script table against the scripts and corrected three rows of the earlier draft:
- `net681.py` was listed as silent; it calls `fwd4.forward` (`net681.py:11`, `:19`), which reads the top-level key (`fwd4.py:81`), so it fails loudly.
- `final_data.py` was listed as silent; it models no activation (it reads SE FC2 weight norms).
- "the other `policy-head-2026-10-01/scripts/*`" were listed as silent; none of them models an activation of a loaded file (`analyze_policy_head.py` reads the key and fails loudly).

## Pass 4 (2026-10-05): first review of the `does_not_apply` revision

Verdict: **MUST-FIX ISSUES REMAIN (6)**. Each item was re-verified against the code before it was acted on. The reviewer found no missed or wrong X2 row and confirmed the two-pass `init(from:)` order, the `validate()` ordering, the conditional layer-health tags, the graph-builder case and the `CaseIterable` removals.

| # | Item | Verified | Outcome |
|---|---|---|---|
| 1 | `BlockGroup.setActivationFunction` (`NetworkArchitecture.swift:648`) and a directly constructed `SetSEActivationDeriveOperation` (`ModelDerivation.swift:876`) accept `does_not_apply`; the plan changed only the parsers | Yes | **Accepted.** `setActivationFunction` gets a `does_not_apply` precondition (it cannot throw: its caller is the non-throwing `BlockGroupDraft.activationFunction` setter; every caller passes a function). `SetSEActivationDeriveOperation.apply` refuses before mutation; `SetActivationDeriveOperation.apply` refuses through `setMainActivationEverywhere`, which now refuses before changing anything; the per-site operation refuses at apply too. New test `testDirectlyConstructedOperationsRefuseDoesNotApply`. No existing test changes. |
| 2 | D8's view `.onChange` + `Task` sync could miss a disappear → reappear between renders | Yes | **Accepted.** The sync now runs synchronously inside every model mutation that can change the sites: `didSet` on the policy and feature-skip fields, the end of the four group methods, and a new `BlockGroupDraft.activationStyle` computed property calling a model-supplied `onTopologyChange` closure (the picker binds it). No view `.onChange` takes part. New test `testADisappearedChoiceIsNeverRestored` (no settling between edits). |
| 3 | `value_stage.py:36-38` calls `ri.load` / `ri.forward` and would break | Yes (`experiments/20261004-value-head-vs-stockfish/value_stage.py:35-39`) | **Accepted.** `relu_inputs.load` returns the normalized architecture; `forward` takes a required keyword-only `sites`; both callers pass it. P1 gate: `net_probs` bit-identical before and after on its README's three checkpoints. |
| 4 | `dcm_arch.norm_arch`'s uniform expansion omits `rezero_alpha_cap`, so a uniform-tower input stamped v6+ is refused, unlike Swift | Yes (`scripts/dcm_arch.py:66-75`, `:92-95`) | **Accepted.** The expansion sets the legacy cap; block-groups inputs keep the strict rule; Python tests at v8 and v9. |
| 5 | V6 expects a value-FC1 activation row on a fresh derived file, which has no velocity | Yes (`App/NewModelCLI.swift:145`, `Training/LayerHealth.swift:1060-1068`) | **Accepted.** V6 checks BN rows and "velocity checks: not included" on the fresh file, and the FC1 row on a 2-step trainer checkpoint of the same architecture. |
| 6 | P4's key-by-key preset comparison cannot hold: the re-save writes every legacy-resolved field | Yes (`NetworkArchitecture.swift:1323-1335`; the unversioned presets lack those fields) | **Accepted.** Equality is checked as NEW decodes it: both files minted by NEW with one seed, parsed `architecture` metadata equal and trainables bit-identical; the shape and label checked separately. |

**Should-fix items:** all accepted — the equality claims narrowed to the six fields (D3, D7); the preset scan's `[ARCH]` logging stated (D5); `applyMainActivationEverywhere` placed in P1 consistently (T8); `fwd16` callers add `import dcm_arch` (`from fwd16 import *` does not export `_dcm_arch`); V8 names the checkpoints each frozen script was run on; `norm_arch` cited at `:52-97`; preset inventory corrected to 4 at v8 and 6 unversioned.

**Found while fixing:** `fwd.py`'s `load` is also imported by `recenter.py` (and its encoder by `posset.py` / `posset_ejp0.py`), none of which calls `fwd.forward`, so its guard moved from `load` into `forward`, which has no caller and gains a required keyword-only `md`.

**Cross-plan recheck (same day), folded in:** the status line now matches this record; OD-12 to OD-15 carry an explicit status and say whether they block P1; T4 and Risks state the interaction with `TRAINING_HEALTH_ALARMS_PLAN.md` rules 2 and 3 (OD-15); D7's lineage row accounts for `HPARAM_RECORDING_PLAN.md`'s `architecture_at_departure`; the stale "(design sections to be revised to this)" was removed from the owner-answer line; `BuildNewModelModel.architecture` is cited `:153-175`.

## Pass 5 (2026-10-05): review of the pass-4 fixes

Verdict: **MUST-FIX ISSUES REMAIN (2)**. Items 1, 3, 4, 5 and 6 of pass 4 and the should-fix items were confirmed addressed; item 2 (the Build-screen sync) was partly addressed. Both new items were re-verified against the code and accepted.

| # | Item | Verified | Outcome |
|---|---|---|---|
| 1 | `load(_:)` assigns fields one by one (`BuildNewModelModel.swift:129-151`), so the new `didSet` syncs run against a half-replaced topology and can clear an incoming site (e.g. a compress-fusion preset loaded into a model with feature skip off) | Yes | **Accepted.** `load` assigns the drafts and every topology field first and the six site activations last; the final block overwrites whatever an intermediate sync did. New test `testLoadingASnapshotReproducesItAcrossTopologies`. |
| 2 | Building the drafts last does not make capturing `self` legal while `blockGroupDrafts` itself is uninitialized (`:49`, no initial value) | Yes | **Accepted.** `init` sets `blockGroupDrafts = []` first, every other stored property next, then builds the drafts; nothing reads the composed architecture in the empty phase. |

**Should-fix items (accepted):** the fingerprint-cache claim in D7 narrowed to absent-site activations; V8's `fwd.py` check described as guard-only coverage (no script calls `fwd.forward`, and its asserts fit none of the folder's recorded checkpoints); the `recenter.py` wording corrected (it uses `fwd.load`, not `fwd.forward`); OD-15 reconciled with the alarms plan, whose rule-3 note already limits rule 3 to `relu` (no owner approval is implied here).

**Pass 6 could not run:** the external reviewer's usage limit was reached (until 2026-10-12) right after pass 5. The two pass-5 fixes above and the should-fix edits are therefore **not yet reviewed**; an independent agent review takes their place.

## Independent review (2026-10-05, second reviewer): review of the pass-5 fixes and the whole plan

The external reviewer could not run (usage limit), so an independent agent reviewed the plan read-only against `b50e6285` (code unchanged since `c0130599`). Verdict: **concurs**, with 1 must-fix (a wrong citation) and 6 should-fix items. It confirmed both pass-5 fixes (`load` ordering, `init` sequencing) correct and found no design defect. It spot-checked about 45 citations and the full X2 / OD-12 test-edit list, and found no other existing test that needs an edit. Each item below was re-verified against the code before it was applied.

| # | Item | Verified | Outcome |
|---|---|---|---|
| Must | T11 cited `F` / `R7` / `R8` in the value-stage README, which names only R7 | Yes: they are at `experiments/20261004-head-logits-relu-inputs/README.md:30-32`; `value_stage.py` ran on R7 (`experiments/20261004-value-head-vs-stockfish/README.md:5-8`) | **Accepted.** Citation fixed; the gate keeps all three, since `relu_inputs.forward` supports them. |
| S1 | `[unowned self]` is unsafe: SwiftUI can keep a removed row's bindings, which hold the draft, alive past its model | Yes (`BlockGroupDraft.swift:13-27`) | **Accepted.** `[weak self]` with a comment: a write to a draft whose model is gone changes only that orphan draft, as for a removed draft. |
| S2 | The `relu_inputs_bot.json` byte-for-byte gate depends on the live `LichessBot/Games` folder and the numpy build | Yes (`relu_inputs.py:105`) | **Accepted.** The gate compares pre-edit and post-edit scripts on identical inputs; the committed JSON is a secondary check. |
| S3 | Render tests cannot drive SwiftUI controls | Yes (they mutate the model, `BuildNewModelTowerShapeRenderTests.swift:66`) | **Accepted.** Reworded to "through the model mutators the controls bind to". |
| S4 | Bit-identical BN running statistics across graphs carry the same compile risk as the head test | Yes (`Network/ChessMPSNetwork.swift:80-99`) | **Accepted.** `testFreshTrainablesAreActivationIndependent` and V5 step 4 follow the "report to owner, no tolerance" rule; Risks says so. |
| S5 | Frozen-script tables incomplete; `lines_list.py:19` "prints `None`" on v9 and should be guarded | Partly. The `fwd3` / `fwd4` users exist and fail loudly; four more were found beyond the review's list (`validate2.py`, `variants.py`, `calib.py`, `calib_scan.py`, `sfcompare.py`; `traj2.py` imports only `bf16`). But `lines_list.py` assigns `af` at `:19` and never uses it (absent from its output dict `:21` and its print `:26`) | **Tables completed. `lines_list.py`: not guarded (disagreement, verified).** It prints nothing about activations, so a v9 file lists correctly; it goes in the "models no activation" row, and OD-9's guarded set stays the three scripts that model ReLU. |
| S6 | `decodeSiteActivation`'s `legacyTowerActivation` must be optional, and `missingRequiredField`'s text is backwards for `activation_function` | Yes (`ArchitectureFormat.swift:134-137`; the re-stamped fixtures state all six keys and no `activation_function`) | **Accepted.** `ActivationFunction?` from `decodeIfPresent`, and a dedicated `FormatError.legacyActivationFunctionMissing`, thrown only when a key needs resolving. |
| Nits | `ActivationSiteMismatch` `Equatable` + `Sendable`; `onTopologyChange` a `let`; the precise `init` / `didSet` rationale; "OD-15" qualified against the alarms plan's own OD-15; citations re-anchored at `a0bc4051`; the `[ARCH]` line count until the presets are re-saved (Risks); the `deriving-models.md` refusal line in P1; the load test also covers `init(snapshot)` and a topology edit after a load | Yes | **All applied.** |
