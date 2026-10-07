# Policy tail precision as an architecture field

Status: **planned, not started** — waiting for the owner's go.

## Why

`ChessNetwork.PolicyTailPrecision` (`fp32_from_pre_bn` / `mixed_final_projection`) decides where the policy head leaves the compute dtype for fp32. It changes the graph a model is built as — same weights, different arithmetic, different policy outputs (the numerics audit measures the gap) — exactly the way `computeDataType` does, and `computeDataType` is an architecture field. Today the tail is a process-wide launch flag (`--policy-tail-precision`), so it does not travel with a model, every launch has to repeat it, and resume needs a special `policy_tail` gap to notice a mismatch.

## Owner decisions (2026-10-07)

- **PT-D1** The tail precision is a `NetworkArchitecture` field, chosen in Build New Model and set on an existing model with `--derive-model`. Recorded in `CLAUDE.md`.
- **PT-D2** Architecture format **v12** adds it. A file written before v12 resolves to the tail it records, else `mixed_final_projection` (PT-D3).
- **PT-D4** The `--policy-tail-precision` launch flag is removed everywhere; no process-wide or launch-time precision setting remains.
- **PT-D5** On an fp32 architecture the field holds `does_not_apply` (the rule the v9/v10 activation sites use for a setting a model cannot have).

## Older files (PT-D3, owner 2026-10-07)

Supersedes the earlier "copy yesterday's files" proposal.

1. **Recorded facts are read.** A file written before v12 resolves to the tail it records — its own `trainer_policy_tail_precision` key, else its lineage `configuration.policy_tail_precision` — and only a file recording neither resolves to `mixed_final_projection` (`does_not_apply` on fp32). Survey (headers, 2026-10-07): 278 files record it (273 `fp32_from_pre_bn`, 5 `mixed_final_projection`), including all 157 `fp32_from_pre_bn` files of 2026-10-06/07, so those need no copy or rewrite.
2. **Audit of unrecorded files.** A one-time read-only tool lists every file written while an fp32 tail existed (from `da159208`, 2026-09-28 21:55) that records no tail — about 478 at the survey — with its run, build and date; the evidence for the tail it ran (the session log's launch line naming it; the experiment README / launch script; else the build's default at the time: `fp32_from_pre_bn` was the only tail until `de0f22be`, 2026-10-01 15:18, made mixed the default); and every experiment record naming its whole-file hash. Files from before `da159208` ran no fp32 tail at all and stay unrecorded (rule 1 gives them `mixed_final_projection`).
3. **Approved rewrite.** After the owner reviews the audit, a one-time migration rewrites only the approved files' headers in place — key added, format v12, weight bytes and `content_sha256` unchanged, every tensor verified bit for bit against the original before the rewrite is published (via `FileSafety` replace-the-exact-file-written) — and updates the experiment records citing their old whole-file hashes in the same pass. A deliberate, owner-approved one-off exception to "nothing is ever overwritten". A file whose evidence is missing or conflicting is listed as such and left untouched.

## Implementation

### 1. Model

- `NetworkArchitecture`: add `policyTailPrecision: PolicyTailPrecisionSetting` beside `computeDataType` (stored property, memberwise init with no default, `CodingKeys` `policy_tail_precision`, encode always, description string). `PolicyTailPrecisionSetting` = `.fp32FromPreBatchNorm`, `.mixedFinalProjection`, `.doesNotApply` (raw values `fp32_from_pre_bn`, `mixed_final_projection`, `does_not_apply`).
- `validate()`: `does_not_apply` exactly when `computeDataType == .float32`.
- The uniform convenience init takes the historical value (`mixed_final_projection` on bf16/fp16, `does_not_apply` on fp32), as it does for `.he` / `.glorot`; built-in presets built with the memberwise init (nt8y) state it.
- **Changing `computeDataType` on a copy** (numerics audit `runDynamic`, `forceFloat32` in `SessionController+Checkpoint.swift:1028-1033`, Build New Model's dtype picker) goes through one function, `NetworkArchitecture.withComputeDataType(_:tail:)`, that sets the tail consistently: to fp32 → `does_not_apply`; from fp32 to bf16/fp16 → an explicit tail the caller passes (the audit passes the one under test; Build New Model the picker's value). No path assigns `computeDataType` directly.
- `ChessNetwork` / `ChessTrainer` / `ChessMPSNetwork` / `ModelGraft` / `NumericsAuditCLI` read the precision from `arch`; the `policyTailPrecision:` init parameters and `.process` go away. `policyHead` keeps its switch; `does_not_apply` builds the fp32 graph (bit-identical to both today, pinned by `testFP32BuildIsIdenticalUnderBothPrecisions`).

### 2. Format v12

- `ArchitectureFormat.currentVersion = 12`, `policyTailPrecisionRequiredFromVersion = 12`, `DecodeFormat.allowsMissingPolicyTailPrecision`, history comment entry.
- Decode: a stated value at any version; missing at v ≤ 11 or in the uniform-tower form → the file's recorded tail (PT-D3.1), else `mixed_final_projection` (`does_not_apply` on fp32), with one `[ARCH] legacy file` entry (`policy_tail_precision := …`); missing at v12 → `missingRequiredField`. Same `if let stated / else if allowsMissing / else throw` shape as `se_beta_init` (`NetworkArchitecture.swift:1060-1072`).
- Builds before v12 refuse v12 files (`requireSupported`), presets and `architecture.json` included — the existing rule.
- `scripts/dcm_arch.py` `CURRENT_FORMAT_VERSION = 12` and its pinning test; `documentation/dashboards/tests/test_lineage.py` header pins.
- Legacy `.dcmmodel`: the shape-only `archHash` cannot carry the tail, so the `.dcmmodel` writer refuses an architecture whose tail is not the preset's (the reader rebuilds the preset).

### 3. Persistence, lineage, resume

- The flat `trainer_policy_tail_precision` key: no longer written; read only to resolve a pre-v12 file (PT-D3.1), together with the lineage `configuration` value (a file whose two disagree is refused, as the writer already refused writing one).
- Lineage schema 3 `configuration.policy_tail_precision` stays (required in schema 3), written from the architecture.
- `PolicyTailPrecisionResume` and `ResumeGap.policyTail`: removed — a resume builds the trainer from the file's own architecture, so the tail cannot differ. `--accept-inexact policy_tail` is refused as an unknown token like any removed gap; recorded `not_exact_items` strings still decode (they are `[String]`).
- `BehaviorFingerprint`: keep the hashed header text `policy_tail=<raw value>`, filled from the architecture, so every existing fingerprint stays valid (`does_not_apply` only on fp32, where the header previously read the process value — fp32 fingerprints change once; recipe stays 3, noted in the changelog).
- `[RUN]` `policy_tail=`, `CliTrainingRecorder` `policy_tail_precision`, `ProbeModelCLI` JSON, `NumericsAudit.Result`, `[NUMERICS]` lines: read from the architecture.

### 4. Launch flag removal (PT-D4)

- Delete `PolicyTailPrecision.flag`, `Source`, `Resolution`, `ResolutionError`, `resolve`, `processResolution`, `process`, `processLogLine`; every `allowedFlags` entry and argument scan in `DrewsChessMachineApp.swift`; the `CommandLineHelp` entry and the `--analyze-numerics` usage mention. Passing the flag is then an unknown-argument error.
- Experiment launch scripts that pass it are historical records of what ran; they are left as they are, each README gets one line saying the flag was removed and how to reproduce (derive the start model with the tail set).

### 5. UI and derive

- Build New Model: a "Policy tail" `enumPicker` in the Precision section beside "Compute dtype" (`BuildNewModelView.swift:225-227`), disabled and showing "does not apply" on fp32; `BuildNewModelModel` field, seeded from a loaded architecture.
- `ArchitectureDiagramView`: show it beside the dtype.
- `--derive-model --set-policy-tail-precision <value>`: one `DeriveOperation` (no tensor rewrites, so allowed on a trained plain-model source; trainer files still refused by `requirePlainModelSource`); row in `documentation/deriving-models.md`.
- Training settings show the loaded model's tail read-only (it is the model's, not a setting).

### 6. Numerics audit

- `runDynamic` compares formats as today and, on bf16/fp16, runs both tails (`PolicyTailPrecisionSetting` bf16-applicable cases) through `withComputeDataType`, replacing "run the CLI twice with two flags".

## Validation

1. Build with no new warnings.
2. New tests: v12 requires the field; v ≤ 11 and uniform-form JSON resolve per PT-D2/PT-D3 with one `[ARCH] legacy file` entry; encode always writes it; a stated value is accepted at any version; `validate()` refuses fp32 + a tail and bf16 + `does_not_apply`; `withComputeDataType` keeps it consistent both ways; network and trainer take the precision from `arch`; the derive op rewrites no tensors and runs on a trained source; Build New Model round-trips it; the `.dcmmodel` writer refusal; a pre-v12 file recording `fp32_from_pre_bn` (flat key or lineage) loads as `fp32_from_pre_bn` and one recording nothing as `mixed_final_projection`, each with the legacy line; a rewritten file loads as its stated tail with weights bit-identical to before.
3. Changed tests (forced by PT-D4/removal, listed for the owner): `PolicyTailPrecisionTests` and `StandardPathForwardPinTests` move from the init parameter to the arch field with **unchanged expected hashes**; `PolicyTailPrecisionProvenanceTests` resolver/process tests deleted and the gap test replaced; `GuiResumeGapsTests`, `GuiResumeContinuationGapsTests`, `BehaviorFingerprintTests`, `ArchitectureActivationSiteGraphTests` and the signature-only callers (inventory in this session's notes) updated; `ResumeEquivalenceTests:772` format literal 11 → 12; `SEBetaInitTests` derive flag list.
4. Fingerprints: every existing bf16 fingerprint recomputes identically (pinned by a test over a stored v11 file).
5. Full suite passes; a v11 model and a v11 trainer session from `Models/` / `Sessions/` load in the GUI with the expected `[ARCH] legacy file` line, and a corpus-replay `--resume-exact` from a pre-v12 `fp32_from_pre_bn` trainer file reports EXACT.
6. PT-D3 audit reviewed by the owner before any header is rewritten.

## Phases (build + commit each)

1. Model field, format v12, decode/encode, validate, `withComputeDataType`, graph/trainer read from arch, presets, Python mirror — with tests.
2. Persistence/lineage/resume/fingerprint/logging reads; gap removal; flag removal; help text.
3. Build New Model picker, diagram, derive op, docs (`deriving-models.md`, CLAUDE.md wording, CHANGELOG), experiment README notes.
4. Numerics audit both-tails comparison.
5. (PT-D3) audit tool, owner review, verified in-place header rewrite of the approved files and update of the records that cite them.
