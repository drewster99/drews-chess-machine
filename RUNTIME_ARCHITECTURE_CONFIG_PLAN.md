# Safetensors-native storage + runtime-configurable architecture — Plan

Status: **APPROVED — in progress.** Branch `safetensors-storage` (from `bf16-trainer`).
Build + commit per phase; no stopping between phases.

## 1. Goal

Two-part feature, sequenced **storage-format-first**:

**Part I (do first): switch model storage to safetensors-native.** Replace the
custom flat-binary `.dcmmodel` weight format with **safetensors** as the *native*
on-disk format (the saved file is itself Python-loadable — no exporter/bridge).
Done for the *current* architecture first, so the persistence rewrite is verified
against a known-good net before any architecture variability exists.

**Part II: make the network architecture runtime-configurable** — block count,
block width (channels), conv width (kernel), SE shape, and compute precision —
saved with each model, loadable by reading the embedded config and building a
matching graph, and usable headless via `--uci` / `--playchess`.

### Locked decisions

- **safetensors is the NATIVE format**, not an export view. New weight files use
  the **`.safetensors`** extension. No separate exporter; the saved file is the
  interchange file.
- **From-scratch Swift safetensors reader/writer** (no SPM dependency). Must match
  the canonical spec so HuggingFace `safetensors` loads our files; validated by an
  actual Python round-trip in a throwaway venv.
- **Weights stay Float32 on disk.** bf16/f16 *on-disk* is a deferred future goal.
  `computeDataType` is a GPU-compute choice + informational metadata only.
- **Legacy `.dcmmodel` reader retained** (read-only) so existing models load.
  Conversion to `.safetensors` is **lazy** (only on re-save). NOTE: most historical
  models are *different architectures*, so loading/round-tripping them depends on
  Part II (flexible arch) — full old-model loading lands in the v3/v4 phase, not
  the storage phase.
- **PyTorch-ready tensor naming** (module-path `state_dict` keys). Standard modules
  keep standard names (auto-map); only materially-different ones carry flag tokens:
  `se_scalebias`, `rezero_alpha`, `value.wdl_fc2`. (Internal Swift `*.operation.name`s
  will eventually be normalized to match; for now names are produced at the writer
  boundary.)
- **Integrity** via `__metadata__["content_sha256"]` (hash over the tensor-data
  region) + `__metadata__["dcm_format_version"]`. No `DCMMODEL` magic needed.
- **The bit-exact forward-pass save verification stays** (`CheckpointManager
  .verifyModelFile`) — format-agnostic; it's the oracle proving any new format
  round-trips (byte-for-byte AND forward-pass-for-forward-pass).
- `.dcmsession` stays a **directory**; only the inner weight files become
  `champion.safetensors` / `trainer.safetensors`. `session.json`,
  `replay_buffer.bin`, chart JSON unchanged.

### Fixed-by-engine (NOT configurable)

Pinned by the encoders; validated, not user-editable: `inputPlanes = 30`
(`BoardEncoder`), `boardSize = 8`, `policyChannels = 76` / `policySize = 4864`
(`PolicyEncoding`), `valueHeadClasses = 3` (WDL).

## 2. Why storage-first

The format switch is independent of arch flexibility, so doing it first verifies
the highest-consequence layer (persistence feeding a live training pipeline) in
isolation against the current known-good net. With only one architecture in play,
`verifyModelFile` is a perfect oracle: if the safetensors round-trip reproduces
this net bit-for-bit and output-for-output, the format is correct. Adding arch
variability later then can't be confused with a format bug. Doing the manifest in
the *custom* format first and then switching would be throwaway work — hence
switch now.

## 3. Current-state facts (from code investigation)

- Arch constants are `static let` in `Network/ChessNetwork.swift:159–235`
  (+ `static let dataType: MPSDataType = .bFloat16` at `:157`), read in ~80 sites
  incl. `parameterCount` (`:250–286`), `architectureSummary` (`:294–300`), graph
  build, inference shape checks, and external callers. A few hardcode `128`
  (SE/policy, `:2005,2010–2011,2048–2054,2115–2116`) and value FC2 bias `[1,3]`
  (`:2252`).
- Only **v4** topology is buildable (`residualBlock` `:1918–2077`): pre-activation,
  clean skip + ReZero α (`1/√numBlocks`), scale-and-bias SE (FC1 `128→32`,
  FC2 `32→256` split `sigmoid(γ)·z + β`). v3 (post-activation, ReLU-gated skip,
  attenuate-only SE FC2 `32→128`, no ReZero) exists only in git history (`cb3b4cf^`).
- No global mutable state blocks multiple `ChessNetwork` instances of differing
  config (champion/trainer/arena nets are already separate instances).
- `.dcmmodel` (`Persistence/ModelCheckpointFile.swift:80–95`): magic `DCMMODEL`,
  `formatVersion 2` (read `{2}`), `archHash`, tensor count, createdAt, modelID,
  sorted-key metadata JSON (`creator, trainingStep, parentModelID, notes`), flat
  `Float32` tensor list (anonymous — names/shapes implicit in build order),
  trailing SHA-256. **Hard-refuses** on archHash mismatch (`:314–320`).
- `currentArchHash` (`:116–141`): FNV-1a over `channels, numBlocks, inputPlanes,
  boardSize, policySize, valueHeadClasses, architectureVersion` (7 scalars). Source
  of the README's documented hashes.
- Load validates weights against the live `trainableVariables +
  bnRunningStatsVariables` (count exact + per-tensor shape). Weight order = build
  order. Trainer file appends optimizer velocity after trainables+BN.
- `verifyModelFile` (`CheckpointManager.swift:998–1111`): re-read + byte-compare
  every float, then pre-save vs post-save forward pass on the starting position,
  bit-compare policy+value. Save writes temp → verify → rename; failure discards.
- `--uci [--model]` (`UCIModelLoader.swift`) builds `ChessMPSNetwork(.randomWeights)`
  at the default arch and loads weights — does not build to the file's config.
  `--playchess` doesn't exist; human-play `.loadedFile` opponent
  (`PlayController.swift:939`) already loads a `.dcmmodel` and builds a fresh net.
- Precision: `makeWeightData`/`readFloats` branch on `Self.dataType`
  (`:2633–2785`); disk always Float32. Trainer fp32-master + velocity path
  (`ChessTrainer.swift:1336–1467`).

## 4. safetensors format design (native)

Canonical safetensors on disk: `[u64 LE header length][JSON header][raw tensor
bytes]`. JSON header: per-tensor `{ "<name>": {dtype, shape, data_offsets:[b,e]} }`
plus `"__metadata__": { <string>: <string> }`. We implement a minimal reader/writer
matching this exactly.

- **Tensors:** every weight (trainables + BN running stats; trainer adds
  `opt.<name>.velocity`) as a **named** `F32` tensor, C-contiguous, our layout
  (conv OIHW; FC stored `[in,out]` — see §11 re: torch transpose at *export* time,
  not on disk).
- **`__metadata__` keys:** `dcm_format_version`, `content_sha256` (hash over the
  tensor-data region), `model_id`, `created_at_unix`, `creator`, `training_step`
  (omitted if nil), `parent_model_id`, `notes`, `arch_hash`, and `architecture`
  (the full `NetworkArchitecture` as a JSON string). String→string only.
- **Names:** PyTorch-ready module paths (§ table in 11). The ordered name+shape
  list comes from `NetworkArchitecture.weightTensorPlan` — single source of truth
  shared by builder, analyzer, writer, and loader.
- **Load:** parse header → match tensors **by name** (not position) → validate
  shapes against the built net → write into the graph. (Name-keyed load replaces
  the brittle positional exact-count contract.)
- **Legacy `.dcmmodel` reader:** retained read-only; decodes the old positional
  format and maps to the current net (and, post-Part-II, via the archHash→config
  fallback map for historical arches).
- **`verifyModelFile`** unchanged in intent; pointed at the safetensors path.

## 5. `NetworkArchitecture` config struct

`Sendable, Codable, Hashable` value type — single source of truth (sketch):

```swift
public struct NetworkArchitecture: Sendable, Codable, Hashable {
    var channels, numBlocks, towerConvKernelSize: Int
    var blockStyle: BlockStyle            // v3PostActivation / v4PreActivation
    var se: SEStyle                        // none / attenuateOnly / scaleAndBias
    var seReductionRatio: Int
    var valueHeadConvChannels, valueHeadHiddenUnits: Int
    var computeDataType: ComputeDataType   // float32 / bFloat16 (not on disk)
    // fixed-by-engine (validated): inputPlanes=30, boardSize=8,
    //                              policyChannels=76, valueHeadClasses=3
}
```

Computed (moved off `ChessNetwork` statics): `parameterCount`,
`architectureSummary`, `architectureVersion` (v3→3 / v4→4), `archHash` (§6),
`weightTensorPlan` (ordered `(name, shape)`), `validate()`.

**Presets:** `v3_8block_3x3` (`0x13ba0b55`), `v3_16block_3x3` (`0x5347c53d`),
`v4_12block_3x3` (`0xbad32ced`), `v4_5block_7x7` (current, `0xdf23a86c`),
`v4_8block_3x3`. `current` = `v4_5block_7x7`.

Test fixtures (parameterCount): 8-blk-v3 **2,483,667**; 16-blk-v3 **4,934,867**;
12-blk-v4 **3,898,139**; 5-blk-v4-7×7 **8,445,748**; predicted 8-blk-v4-3×3
**≈2,664,087**.

## 6. archHash strategy

Keep the existing 7-scalar FNV formula (preserves documented hashes) but compute
from a `NetworkArchitecture`. Demote from load-gate to a coarse identity tag;
precise identity = embedded config. Old files without embedded config →
`archHash → NetworkArchitecture` fallback map (the four historical hashes). This
is the only migration code.

## 7. static→instance refactor

Add `let arch: NetworkArchitecture` to `ChessNetwork`; replace every `Self.<const>`
/ hardcoded `128` / `[1,3]` with `self.arch.<field>`; convert `parameterCount` /
`architectureSummary` to instance members; update external callers and
`NetworkWeightAnalyzer`. **Parity gate:** default config rebuilds the current net
bit-exactly (forward-pass parity test + the existing save verification).

## 8. v3 / v4 block + SE builder

`residualBlock` branches on `arch.blockStyle` (v4 clean skip + ReZero; v3
post-act ReLU-gated skip, no ReZero). SE branches on `arch.se` (none /
attenuate-only FC2→C / scale-and-bias FC2→2C). Per-style `weightTensorPlan` is the
shared source of truth; verify against historical models via `verifyModelFile`.

## 9. Precision

`computeDataType` moves to `arch`; `makeWeightData`/`readFloats` branch on it;
disk stays F32; trainer fp32-master path active for bf16.

## 10. Build UX + CLI

- **Build-new-net:** architecture panel (preset + free-form: block count,
  channels, conv kernel, SE style+ratio, value-head dims, precision) with live
  `parameterCount`/summary + validation; `architecture.json` for CLI/autotrain
  (separate from `parameters.json`). Build constructs `ChessNetwork(arch:)`.
- **`--uci`:** build the network from the file's embedded config (shared resolver).
- **`--playchess`:** GUI-launch flag (like `--train`) routing into the human-play
  UI with the model resolved the same way as `--uci` (auto-default or `--model`),
  via the existing `.loadedFile` opponent path.

## 11. Python / framework interop

Native format = interoperable; no exporter. Conventions to honor + document in
`chess-engine-design.md`:

- **Layout:** conv OIHW, FC stored `[in,out]`, **C-contiguous**, LE F32. A torch
  consumer transposes FC to `[out,in]`; TF transposes conv to HWIO.
- **BN:** gamma→`weight`, beta→`bias`; `running_mean`/`running_var` match torch.
  No `num_batches_tracked` (fixed-EMA; positional-exact loader had no slot) →
  synthesize `0` on torch import-from-us; drop it on import-to-us.
- **Naming table** (DCM internal → safetensors/torch key):
  `stem_conv_weights`→`stem.conv.weight`; `*_bn_gamma/beta`→`*.bn.weight/bias`;
  `block<i>_conv{1,2}_weights`→`blocks.<i>.conv{1,2}.weight`;
  `block<i>_se_fc{1,2}_*`→`blocks.<i>.se_scalebias.fc{1,2}.{weight,bias}` (**flag:**
  FC2 is 2C); `block<i>_res_scale`→`blocks.<i>.rezero_alpha` (**flag**);
  `value_fc2_*`→`value.wdl_fc2.{weight,bias}` (**flag:** 3-logit WDL);
  `policy_conv`→`policy.conv.*` (standard weight; output convention flagged in
  spec, not the tensor); trainer velocity→`opt.<name>.velocity`.
- **Semantics naming can't encode (spec doc):** scale-and-bias SE
  `sigmoid(γ)·z + β`; ReZero α residual `input + α·F(input)`; value = 3 WDL logits,
  derived `v = p_win − p_loss`; policy 4864 index `channel*64 + row*8 + col` +
  Black-perspective vertical flip; 30-plane input. No bit-exact cross-framework
  promise — "same weights + equivalent topology."
- **Validation:** Python round-trip in a throwaway venv (`pip install safetensors
  numpy`) — load our file, check names/shapes/values. Inference-in-Python is a
  later, separate validation.

## 12. Phased implementation (build + commit per phase, no stopping)

1. **`NetworkArchitecture` foundation** — struct, enums, presets, `parameterCount`
   / `architectureSummary` / `archHash` / `weightTensorPlan` (matching the current
   v4 build order) + validation. XCTests for hashes, param counts, plan counts,
   validation. No `ChessNetwork` rewiring.
2. **Safetensors-native storage (current arch)** — from-scratch Swift
   reader/writer; `CheckpointManager` save/load via safetensors; `.safetensors`
   extension; name-keyed load into the current net; `__metadata__` schema (incl.
   `content_sha256`, `dcm_format_version`, embedded `architecture`); legacy
   `.dcmmodel` reader; update loaders/pointers/open-panel filters
   (`UCIModelLoader`, `PlayController`, `SessionCheckpointLayout`,
   `LastSessionPointer`); keep `verifyModelFile`. **Validate:** round-trip on the
   current net (byte + forward) + Python load.
3. **static→instance refactor** — thread `arch` through `ChessNetwork` &
   `NetworkWeightAnalyzer`; **bit-exact parity gate**.
4. **Build-new-net flexibility** — architecture UI + `architecture.json` +
   validation; Build constructs from config; persistence already config-driven.
5. **Per-model precision** — `computeDataType` in config; mixed-precision trainer.
6. **v3/v4 block-style + SE variants + archHash→config fallback map** — load
   historical v3 & v4 models, verified via `verifyModelFile`.
7. **CLI** — `--uci` builds from embedded config; implement `--playchess`.
8. **Capstone** — build 8-block-v4 from preset; load all four historical arches;
   full suite green; Python inference spot-check.

## 13. Risks & mitigations

- **Persistence rewrite feeds a live pipeline** (highest consequence): `verifyModelFile`
  gates every save; we never overwrite existing files; the live session is
  terminal-launched and independent of our xcode-mcp builds/runs.
- **Cross-era weight-order parity:** per-style `weightTensorPlan` as single source
  of truth + bit-exact forward verification against saved models.
- **Refactor regressions:** default-config bit-exact parity gate (Phase 3).
- **safetensors spec compliance:** validate with real Python load.
- **Disk:** XCTest/DerivedData can be large; user is disk-constrained — keep an eye out.

## 14. Out of scope (unless separately requested)

- Changing `inputPlanes`/`policyChannels`/`valueHeadClasses`/`boardSize`.
- bf16/f16 weight storage on disk (deferred future goal; design keeps it additive).
- Search of any kind (explicit project non-goal).
