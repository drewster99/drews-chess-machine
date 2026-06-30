# Safetensors-native storage + runtime-configurable architecture — Plan

Status: **SHIPPED (2026-06-23 audit) — one CLI item open.** Branch `safetensors-storage`
(from `bf16-trainer`). Build + commit per phase; no stopping between phases.

**Progress:** Phases 1–4 complete (NetworkArchitecture foundation, safetensors-native
storage, static→instance refactor, build-new-net flexibility). Phase 6a complete
(resume/load by embedded architecture). **Effectively all remaining phases have now
shipped:**
- **Phase 5 (per-model compute precision) — SHIPPED.** `ComputeDataType` (`.float32` /
  `.bFloat16` / `.float16`) is embedded per-model and honored by the graph builders
  (`ChessNetwork.swift` / `NetworkArchitecture.swift`); fp16 added 2026-06-17 (`b25f37e`,
  inference-ready). No bf16 gate — honored as-is, per §9.
- **Phase 6 (block-style / SE / heterogeneous towers) — SHIPPED** via the block-groups
  refactor (`73a1bdd` Phase A: heterogeneous tower config + per-block graph builders;
  `120f46b` Phase B: per-group Build-New-Model editor + diagram). Non-default-architecture
  training (mixed/WRN-staircase towers) is exercised by GPU tests. Legacy `archHash→config`
  fallback survives as a `.dcmmodel`-only lookup table.
- **Phase 7 (`--uci` / `--playchess`) — SHIPPED.** Both flags ship in the arg parser
  (`DrewsChessMachineApp.swift`): `--uci` (`28cf394`, cutechess / external opponents) and
  `--playchess` (`26c14e9`).
- **Safetensors-native storage — SHIPPED** (`Persistence/SafetensorsModelIO.swift`); identity
  = embedded config, integrity = `content_sha256`.

**The one genuinely-open item** is the headless **`--architecture-preset` /
`--architecture-file` CLI flags** (§10): the preset store (`ArchitecturePresetStore`) exists
and drives Build-New-Model, but it is GUI-only — those flags are *not* in the arg parser
(grep-confirmed), so a fresh tower cannot yet be built/trained from the CLI. The deferred
`architecture_version`-drop decision (free-text `label` instead) also remains open. See §15
for the two invariants that constrain this remaining work.

**Design revision in progress (§5a / §6 / §10):** a component-by-component design
walkthrough with the user has finalized the configurable-architecture surface, and it
**diverges from the as-built Phases 1–4** — so parts of them need rework: the
monolithic `block_style` is **decomposed** into orthogonal axes (§5a); `arch_hash` is
**removed from safetensors** (identity = embedded config; integrity = `content_sha256`)
and demoted to a legacy-`.dcmmodel`-only lookup table (§6); the well-known
`architecture.json` loader is **replaced** by `--architecture-preset` /
`--architecture-file` + a Presets folder (§10); `architecture_version` is **dropped**
in favor of a free-text `label`; `input_encoding` (`basic20`/`basic30`) and
`value_head_style` (`scalar_tanh`/`wdl_softmax`) become first-class buildable+trainable
axes. The §12 phase list will be re-sequenced once the spec walkthrough is complete.

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
  (`ChessTrainer.swift:1336–1467`). `dataType` is a hardcoded
  `static let .bFloat16` with **no hardware-availability check** today
  (addressed in §9). `arch.computeDataType` exists but is currently
  *informational only* (read in `architectureSummary`, not at the 52 real
  compute sites) — per-model precision is genuinely unwired (Phase 5).
- **Execution path (corrected):** the high-frequency forward passes run through
  compiled `MPSGraphExecutable` (`.level1`), NOT `graph.run`: batched self-play /
  arena inference (`evaluateBatched` → `inferenceExecutables`, `:1118/1130`) and the
  trainer's GPU→GPU value-baseline forward (`valueBaselineExecutables`, `:1266`),
  each cached per batch size, compiled once, reused for the net's life (introduced
  `3f02ac4`). Only the *low-frequency* paths still use `graph.run`: single-position
  `evaluate(board:)` (arena one-at-a-time + Play Game), `evaluateValueDistribution`
  (diagnostics probe), and weight-load / export / BN-warmup plumbing. The
  static→instance refactor (Phase 3) touched **zero** executable code.

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
  (omitted if nil), `parent_model_id`, `notes`, and `architecture`
  (the full `NetworkArchitecture` as a JSON string). String→string only. **No
  `arch_hash`** — identity is the embedded config; integrity is `content_sha256`
  (see §6).
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

### Ground truth from the actual saved files (verified, not guessed)

Decoding the archHash (byte offset 12) of the real `.dcmmodel` files on disk, and
brute-forcing the FNV-1a formula over the scalar set, **every historical arch is
30-plane, 3-class WDL** — settling the two scariest scope questions:

| arch | hash | numBlocks | inputPlanes | valueHeadClasses | version | champion / trainer tensors | where |
|------|------|-----------|-------------|------------------|---------|----------------------------|-------|
| 8-block-v3  | `0x13ba0b55` | 8  | **30** | **3 (WDL)** | none | 128 / 220 | Ko63, IWkd sessions; sMe9 model |
| 16-block-v3 | `0x5347c53d` | 16 | **30** | **3 (WDL)** | none | (README)  | README |
| 12-block-v4 | `0xbad32ced` | 12 | **30** | **3 (WDL)** | 4    | 205 / 354 | WcRm sessions |
| 5-block-v4-7×7 | `0xdf23a86c` | 5 | **30** | **3 (WDL)** | 4 | (current) | running session |

**Consequences (scope reduction):**
- **No 20-plane models exist on disk.** `inputPlanes = 30` stays fixed; no
  `BoardEncoder` rework, no un-fixing of the "fixed-by-engine" field. The
  scalar-tanh and 20-plane eras predate every file the user actually has.
- **No scalar-tanh value heads exist.** The WDL 3-logit value head is universal,
  so the value-head output contract is **identical** across all four arches — the
  v3 builder reuses the *current* `valueHead` verbatim.
- Therefore the **only** axes that differ v3↔v4 (for real files) are
  **`blockStyle`** (post-act ReLU-gated skip, no ReZero ↔ pre-act clean skip +
  ReZero α) and **`se`** (attenuate-only FC2→C ↔ scale-and-bias FC2→2C). Both are
  already enum axes on `NetworkArchitecture`. The 8-block-v3 is the user's
  most-trained lineage (weeks: Ko63 → IWkd → sMe9), so it is the priority load
  target and the primary `verifyModelFile` fixture.

## 5a. Configurable architecture — component spec (design walkthrough)

Built component-by-component with the user. **Settled** components below; remaining:
residual block + SE, policy head, value head (precision/bf16 gate is §9).

### Cross-cutting principles (apply to every component)

- **No silent defaults.** `NetworkArchitecture`'s memberwise init has **zero**
  defaulted fields — every configurable parameter is passed explicitly. The only
  place concrete values live is a **preset** (a named, fully-specified config;
  `.current` is one preset). Consequence: the `= .current` convenience defaults on
  `ChessNetwork.init` / `ChessMPSNetwork.init` / `ChessTrainer.init` (added in
  Phase 3) are **removed** — every construction site states its arch explicitly
  (touches those inits + tests doing `ChessNetwork()`).
- **Flat schema (decision A).** Style + its numeric param are *separate* flat fields
  (`se` + `se_reduction_ratio`), not associated-value enums. `validate()` enforces
  consistency. Keeps `Codable` trivial, `architecture.json` human-readable, and the
  selection enums `CaseIterable`.
- **Naming:** Swift **camelCase** property ↔ JSON **lower_snake_case** key, via an
  **explicit `CodingKeys`** enum (not a global `.convertToSnakeCase` strategy — no
  silent mis-conversion). Style enum rawValues are snake/lowercase tokens too:
  `block_style:"v4_pre_activation"`, `se:"scale_and_bias"`,
  `value_head_style:"wdl_softmax"`/`"scalar_tanh"`, `input_encoding:"basic30"`,
  `compute_data_type:"bfloat16"`.
- **Required stored fields (settled so far):** `input_encoding`, `channels`,
  `num_blocks`, `stem_conv_kernel_size`, **`activation_function`** (network-wide),
  `block_activation_style`, `block_skip_merge`, `block_use_rezero`,
  `rezero_alpha_init`, `block_conv1_kernel_size`, `block_conv2_kernel_size`,
  `block_se_style`, `block_se_reduction_ratio`, `policy_head_style`,
  `policy_pre_conv_channels`, `value_head_style`, `value_head_conv_channels`,
  `value_head_hidden_units`, `compute_data_type`. (`architecture_version`: orphaned by
  the archHash deprecation — likely dropped; see §6.)
- **Derived (computed, never stored — can't be silently wrong):** `input_planes`
  (←`input_encoding`), `value_head_classes` + value loss type (←`value_head_style`),
  stem-ReLU / tower-end-BN presence (←`block_activation_style`), same-padding
  (←kernel), `arch_hash`, `parameter_count`.
- **Fixed engine constants (not in the init at all):** `board_size = 8`,
  `policy_channels = 76` / `policy_size = 4864`.
- **Limits philosophy:** the goal is "don't crash," not arbitrary caps.
  `num_blocks` 1–5000; `tower_conv_kernel_size` / `stem_conv_kernel_size` 1–29
  **odd** (symmetric same-padding); `channels` floor ~1–2 (≥`se_reduction_ratio` and
  divisible by it when SE on), ceiling **memory-bound** — `validate()` rejects via an
  estimated-memory budget (`recommendedMaxWorkingSetSize`/`maxBufferLength`) rather
  than a magic max.

### Component: input tensor — `InputEncoding` enum

`enum InputEncoding: String, Codable, CaseIterable, Sendable, Hashable`. rawValue =
the storage key (treated as a permanent contract; renaming orphans old JSON).
Cases so far: **`basic20`**, **`basic30`**. Single source of truth shared by
`BoardEncoder` (writes the planes), `ChessNetwork` (stem input depth), and
`ReplayBuffer` (per-position stride = `plane_count × 64`).

- `planeCount` is **derived** from a structured `planeGroups: [PlaneGroup]`
  (`PlaneGroup{range, meaning}`); `description` *renders* from `planeGroups`, and a
  unit test asserts `BoardEncoder` fills exactly those ranges (no doc/impl drift).
- Wiring: `NetworkArchitecture.inputEncoding` replaces the fixed `inputPlanes`
  (which becomes computed); `BoardEncoder.encode(...)` takes the encoding and every
  call site passes the session's; `ReplayBuffer` stride becomes **instance-derived**
  (static→instance, mirroring `ChessNetwork`), and **resume validates** the saved
  buffer's stride equals `plane_count × 64` (clear error, never silent mis-stride).
- archHash nuance: `inputPlanes` enters the hash only as a *count*, so a future
  different-but-same-count encoding collides on the hash with `basic20`. Acceptable:
  hash = coarse tag; precise identity = the embedded `input_encoding` key. Legacy
  `.dcmmodel` loads via the canonical hash→preset map (no two historical files share
  a plane count with different encodings).

**Corrected current 30-plane layout (`basic30`)** — always from the **side-to-move's**
perspective; piece order **P, N, B, R, Q, K**:

| Planes | Contents |
|--------|----------|
| 0–5   | my pieces: pawn, knight, bishop, rook, queen, king |
| 6–11  | opponent's pieces: same order |
| 12–13 | my castling: kingside, queenside |
| 14–15 | opponent's castling: kingside, queenside |
| 16    | en passant target (one plane) |
| 17    | halfmove / 50-move clock, `min(clock,99)/99` |
| 18    | repetition: current position seen ≥1× before |
| 19    | repetition: seen ≥2× before (3-fold threshold) |
| 20–29 | temporal-repetition history: plane `20+i` = position `i+1` plies ago is a strict duplicate |

`basic20` = planes **0–19** exactly (drop 20–29). 18–19 are scalar repetition
*counts*; 20–29 are the temporal *pattern* — different signals, both kept.

### Component: stem

Single layer: conv (`inputPlanes → channels`) → BN → [ReLU]. Configurable: a
**separate, explicit `stem_conv_kernel_size`** (1–29 odd; not tied to the tower —
lets a wide stem feed a cheap deep trunk). Derived/fixed: stem output width
(=`channels`), stem ReLU presence (←`block_style`: on for v3, off for v4), stem BN
always present (γ=1, β=0, running_mean=0, running_var=1, ε=1e-5, EMA momentum 0.99),
conv bias-free (BN β subsumes), same-padding (←kernel). Current stem detail: conv
`[128,30,7,7]` OIHW, He-init (fan-in 1470), pad 3 → `[B,128,8,8]` → `stem_bn` → no
ReLU. Persistent tensors: 3 trainable + 2 running = 5 (188,672 params).

### Component: residual block + SE (fully decomposed)

The old monolithic "v3 vs v4" is **decomposed into orthogonal axes** (one code path
branches on each) so off-diagonal experiments are expressible (e.g. pre-activation
with attenuate-only SE). `channels` / `num_blocks` stay un-prefixed (tower-level:
trunk width + depth).

| JSON key | Swift | Values |
|----------|-------|--------|
| `block_activation_style` | `blockActivationStyle` | `pre` \| `post` |
| `block_skip_merge` | `blockSkipMerge` | `clean_add` \| `activation_gated` |
| `block_use_rezero` | `blockUseRezero` | bool |
| `rezero_alpha_init` | `rezeroAlphaInit` | float (consumed only when `use_rezero`) |
| `block_conv1_kernel_size` | `blockConv1KernelSize` | int, odd, 1–29 |
| `block_conv2_kernel_size` | `blockConv2KernelSize` | int, odd, 1–29 |
| `block_se_style` | `blockSeStyle` | `none` \| `attenuate_only` \| `scale_and_bias` |
| `block_se_reduction_ratio` | `blockSeReductionRatio` | int (divides `channels` when SE on) |

Settled semantics:
- Activation is a **single network-wide `activation_function`** (`relu`/`silu`/`gelu`)
  — see the dedicated note below. It governs every *hidden* nonlinearity (block
  main-path, SE FC1, tower-end, and both heads). The SE **gate stays `sigmoid`**
  regardless (bounded 0–1 gate, not a free activation).
- **`block_skip_merge`**: `clean_add` = `out = input + α·F` (v4 identity highway);
  `activation_gated` = `out = activation_function(input + F)` (v3 was the ReLU
  special case).
- **`block_use_rezero`** false ⇒ no `α` tensor in the block (and `rezero_alpha_init`
  is ignored — the agreed (A) flat-schema price: required but sometimes unused).

Current v4 block (reference): `pre` / `relu` / `clean_add` / rezero on (α init
`1/√num_blocks`) / conv1=conv2=7 / SE `scale_and_bias` / ratio 4. Per-block tensors
(this config) = 8 trainable + 4 running = 12. v3 block: `post` / `relu` /
`activation_gated` / rezero off / SE `attenuate_only` (FC2→C, no bias half) → drops
`α`, halves FC2.

### Network-wide `activation_function`

A **single** `activation_function` (`relu` \| `silu` \| `gelu`) applied at **every
hidden nonlinearity** in the network — block main-path, SE FC1, tower-end, and both
heads. Verified across all of git history: every architecture from the first commit
to today used **ReLU** at every hidden site, so `activation_function: relu`
reproduces 100% of historical nets. Not controlled by it (structural/output, owned by
their component styles): SE **gate** = `sigmoid`; value output = `tanh`/`softmax`
(←`value_head_style`); policy output = raw logits / softmax (←`policy_head_style`).

### Component: policy head

| JSON key | Swift | Values |
|----------|-------|--------|
| `policy_head_style` | `policyHeadStyle` | `simple_conv` \| `intermediate_conv` \| `fc_bottleneck` |
| `policy_pre_conv_channels` | `policyPreConvChannels` | int (the K below) |

All three styles emit **4864 raw logits** in today's `PolicyEncoding` (76×64) — masked
CPU-side downstream, so **no in-head softmax**. The final →76 projection and the
4864 reshape are **fixed** (the AlphaZero move encoding; changing it = a separate
`policy_encoding` axis, deferred — see note).

- **`simple_conv`** (8-block-v3): `1×1 conv channels→76 (+bias) → reshape`. 2 tensors.
  Ignores `policy_pre_conv_channels`.
- **`intermediate_conv`** (current/16-block-v3): `1×1 conv channels→K → BN → ReLU →
  1×1 conv K→76 (+bias) → reshape`. K = `policy_pre_conv_channels` (current uses
  K=channels=128). Renormalizes the tower output before projection.
- **`fc_bottleneck`** (adapted from the original bab8654 head, **option (a)**):
  `1×1 conv channels→K → BN → ReLU → flatten(K·64) → FC (K·64→4864) (+bias)`. K =
  `policy_pre_conv_channels` (original used K=2). The literal original also used a
  **4096** from-to policy space + in-head softmax; we deliberately build it in the
  **current 4864 space with raw logits** instead. The genuine 4096 `policy_encoding`
  is a separate future axis, only needed to load ancient files (none on disk).

### Component: value head

| JSON key | Swift | Values |
|----------|-------|--------|
| `value_head_style` | `valueHeadStyle` | `scalar_tanh` \| `wdl_softmax` |
| `value_head_conv_channels` | `valueHeadConvChannels` | int ≥ 1 (1×1 conv compression width) |
| `value_head_hidden_units` | `valueHeadHiddenUnits` | int ≥ 1 (FC1 width) |

- **`wdl_softmax`** (current): `1×1 conv channels→C_v → BN → ReLU → flatten(C_v·64) →
  FC→H → ReLU → FC→3 → softmax`; scalar = `p_win−p_loss`. Current: C_v=16, H=128.
- **`scalar_tanh`** (original bab8654): `…conv channels→1 → BN → ReLU → flatten(64) →
  FC 64→64 → ReLU → FC 64→1 → tanh`; scalar = `tanh(x)`. (C_v=1, H=64.)

**Derived from `value_head_style`** (not knobs): `value_head_classes` (1/3); output
activation (`tanh`/`softmax`); FC2 bias init (`0` / `[0, ln6, 0]`); derived inference
scalar (`tanh(x)` / `p_win−p_loss`, both ∈[−1,1] → **inference consumers unaffected**).
Hidden ReLUs follow the network-wide `activation_function`.

**Cross-component consequence — training loss.** `value_head_style` dictates the value
loss, so **`ChessTrainer.buildTrainingOps` branches**: `scalar_tanh` → **MSE** on the
scalar vs `z ∈ {−1,0,+1}`; `wdl_softmax` → **categorical CE** on 3 logits vs one-hot
`idx = 1−z`. The targeted value-head tensors differ (1 scalar vs 3 logits). Same
one-code-path, build-time branch — but in the trainer's loss graph, the only place the
value-head axis reaches outside `ChessNetwork`.

## 6. Architecture identity & the legacy archHash

**Identity = the embedded config (exact `Equatable`). There is no config hash.**
After inventorying every archHash use in the code, its only functional job was the
`.dcmmodel` load-gate — which the config-driven loader *replaces* (read config → build
that arch). So archHash is **deprecated to a legacy-`.dcmmodel`-only artifact**, and
all the content-address machinery (SHA-256, canonical-JSON, float-determinism,
scheme-version, 64-bit) is **dropped as unneeded**.

- **safetensors: no `arch_hash` field at all.** Integrity = `content_sha256` (tensor
  data). Precise identity/equality = the embedded canonical `architecture`. Load =
  build any *valid* config, reject anything that doesn't make sense. (Remove the
  `arch_hash` write + verify already added to `SafetensorsModelIO` in Phase 2.)
- **archHash survives only as a hardcoded *bidirectional* table** between the four
  documented `.dcmmodel` stored hashes and historical presets:
  `0x13ba0b55↔8-block-v3`, `0x5347c53d↔16-block-v3`, `0xbad32ced↔12-block-v4`,
  `0xdf23a86c↔5-block-v4`.
  - **Read `.dcmmodel`:** stored hash → preset → build that config (replaces the old
    hard-refuse).
  - **Write `.dcmmodel`** (near-never; round-trip tests only): allowed *only* if the
    current config matches a legacy preset → write its hash back; a novel config can't
    be represented as `.dcmmodel` → disallow with a clear error. Literal table, no
    formula, no recompute.
- **We read `.dcmmodel`, we write `.safetensors`.** `.dcmmodel` writing isn't a real
  path ⇒ **no float/ULP/canonicalization concerns anywhere.**
- **Traceability (logs/exports):** use the human-readable `architectureSummary` (+
  existing `ModelID`), NOT a hash ("a hash isn't meaningful on its face"). Replaces the
  `arch_hash=` token in the `[APP]`/`[BUILD]` lines and analysis-export metadata.
- **`architecture_version`:** orphaned (its only job was the old hash;
  `architectureSummary` already carries the era marker). **Lean: drop it** — keep only
  if a hand-set generation label is wanted. *(OPEN — last remaining identity decision.)*

## 7. static→instance refactor

Add `let arch: NetworkArchitecture` to `ChessNetwork`; replace every `Self.<const>`
/ hardcoded `128` / `[1,3]` with `self.arch.<field>`; convert `parameterCount` /
`architectureSummary` to instance members; update external callers and
`NetworkWeightAnalyzer`. **Parity gate:** default config rebuilds the current net
bit-exactly (forward-pass parity test + the existing save verification).

## 8. v3 / v4 block + SE builder — ONE code path

**No parallel `ChessNetworkV3` class and no duplicated `residualBlockV3`/`V4`
functions.** The existing single builders gain internal branches keyed off the
`NetworkArchitecture` fields; the struct is the only discriminator.

- `residualBlock` adds `switch arch.blockStyle`:
  - `.v4PreActivation` (today's arm, verbatim): `BN→ReLU→conv→BN→ReLU→conv→[SE]`,
    clean identity skip `out = input + α·F(input)`, per-block ReZero α (`1/√numBlocks`).
  - `.v3PostActivation` (reintroduced from `cb3b4cf^`): `conv→BN→ReLU→conv→BN→[SE]`,
    ReLU-gated skip `out = ReLU(input + F(input))`, **no** ReZero scalar.
- SE already branches on `arch.se` (`none` / `attenuateOnly` FC2→C / `scaleAndBias`
  FC2→2C). v3 uses `attenuateOnly`; v4 uses `scaleAndBias`.
- **Value head is shared verbatim** — all real files are 3-class WDL (see §5 ground
  truth), so no value-head branch is needed.
- `weightTensorPlan()` gains the matching `blockStyle`/`se` branches (drop the
  `rezero_alpha` slot for v3; `attenuateOnly` FC2 shape `[r, C]`; post-act BN
  ordering) so it stays the single source of truth shared by builder, analyzer,
  writer, and loader.

**The branches execute once, at graph-build time, on the CPU.** For a v4/current
arch every branch lands on today's exact code, so the constructed graph — and the
`MPSGraphExecutable` compiled from it — is bit-identical (same ops, names, shapes,
order). v3 arms are dead code for a v4 build. See the §15 invariants.

Verify the v3 reconstruction by *loading the real 8-block-v3 file* and running
`verifyModelFile` (byte + forward-pass) — git history tells us the topology, the
saved weights tell us we got it right.

## 9. Precision

`compute_data_type` (`float32` | `bfloat16`) is a **per-model config field** in
`NetworkArchitecture`, **honored as configured** — no hardware detection, no auto-
fallback, no warn. It moves off the static `ChessNetwork.dataType`; `makeWeightData` /
`readFloats` and the compute sites branch on the model's value; disk stays F32; the
trainer's fp32-master mixed-precision path stays active for bf16.

**No bf16 gate** (per the user). bf16 works on every supported Mac and doesn't crash —
it's simply only *faster* on apple10 (M5+); on M4/earlier it runs ≈ fp32. That's a known
property the user accounts for when choosing a model's `compute_data_type`; the app does
not detect, gate, warn, or override.

Background (for reference, not implemented): apple10 = M5 GPU "Neural Accelerators";
measured bf16 ≈ 7.6×/1.5× on M5 vs 1.08× on M4. `tools/bf16-probe.swift` is kept as a
standalone diagnostic to check a given machine's bf16 speedup — it is **not** wired into
the app.

## 10. Build UX + CLI

- **Build-new-net (GUI):** architecture panel — preset picker (built-ins + user-saved)
  + free-form fields (every required topology knob) with live `parameterCount` /
  `architectureSummary` + validation (incl. the memory-budget check). "Save as preset"
  writes a new file to the Presets folder. Build constructs `ChessNetwork(arch:)`.
- **Build-new-net (headless):** **no well-known `architecture.json`.** Arch comes from
  `--architecture-preset <name>` (a built-in or user-saved preset) **or**
  `--architecture-file <path>` (an explicit arch JSON at any path). Removes the Phase-4
  `ArchitectureConfig.loadDefaultIfPresent` well-known-file loader.
- **`--uci`:** build from the model file's embedded config (shared resolver) — never
  needs a preset/arch file.
- **`--playchess`:** GUI-launch flag (like `--train`) routing into the human-play
  UI with the model resolved the same way as `--uci` (auto-default or `--model`),
  via the existing `.loadedFile` opponent path.
- **Built-ins are NOT exported to files** (no `--create-architecture-presets`) — they
  stay compiled-in and immutable so they can't drift. The Presets folder is
  **user-saved presets only**.

### Presets store

- **Location:** `~/Library/Application Support/DrewsChessMachine/Presets/`, sibling to
  `Models/` / `Sessions/`. **One `.json` file per preset**, human-editable (snake_case
  keys per convention).
- **A preset file = the full topology + the free-text `label`.** `--architecture-file`
  reads the *same* format from any path (a preset file *is* an arch file, just in the
  well-known folder).
- **Identity:** the **filename stem is the preset name** (the `--architecture-preset`
  key + picker key); `label` is the prettier display string (may differ). No separate
  id field.
- **Built-ins** stay compiled-in (the `Preset` enum — also the legacy-load table's
  targets) and are **never written to the folder** (immutable, can't drift).
  `--architecture-preset` resolves built-ins **and** the folder, with **built-in names
  reserved** (a user file can't shadow a built-in).
- This also settles `label` placement: it lives in the preset/arch-file wrapper and is
  embedded into saved-model metadata; `NetworkArchitecture` stays **purely topological**
  (decision (b)).

### Build New Model screen (required)

A dedicated screen for creating a fresh network — opened from the Build action (and a
**File ▸ New Model…** menu item), replacing the old immediate "build default" path. One
SwiftUI `View` struct in its own file under `App/UpperContentView/`, backed by an
`@Observable` model (mirrors `TrainingSettingsPopoverModel`).

- **Preset picker** — built-in presets + user-saved (from the Presets folder); selecting
  one populates every field; switches to "Custom" once any field is edited.
- **Free-form controls — every required topology field (§5a), grouped:**
  - *Input:* `input_encoding` (picker; shows the `planeGroups` table for the selected case).
  - *Tower:* `channels`, `num_blocks`, `stem_conv_kernel_size`, `activation_function`.
  - *Block:* `block_activation_style`, `block_skip_merge`, `block_use_rezero`
    (+`rezero_alpha_init`, enabled only when on), `block_conv1_kernel_size`,
    `block_conv2_kernel_size`, `block_se_style` (+`block_se_reduction_ratio`, enabled only
    when SE ≠ none).
  - *Policy:* `policy_head_style` (+`policy_pre_conv_channels`).
  - *Value:* `value_head_style`, `value_head_conv_channels`, `value_head_hidden_units`.
  - *Precision:* `compute_data_type`.
  - *Name:* free-text `label`.
- **Live readouts** (recompute on every change): `parameterCount` (monospaced, whitespace-
  padded), `architectureSummary`, estimated memory vs the device budget, and inline
  validation — invalid fields flagged, **Build disabled** until `validate()` passes.
- **Actions:** **Build** (constructs `ChessNetwork(arch:)` + readies the session),
  **Save as Preset…** (writes a `.json` to the Presets folder), **Reset to preset**.
- **Conventions:** one `View` per file; `@Observable` model with per-field bindings +
  validation (mirrors `TrainingSettingsPopoverModel`); monospaced padded digits;
  light/dark; aligned columns; keyboard/accessibility per the house UI rules.

Built in **Phase H**.

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

## 12. Phased implementation (RE-SEQUENCED — build + commit per phase)

Dependency-ordered. "(rework)" = revises as-built Phase 1–4 code to the finalized design.
**Capability comes online incrementally:** after Phase B you can build+train any
*v4-family* arch (any blocks/channels/kernels/SE/policy/activation, WDL, basic30) — most
of the experimentation surface; C adds basic20, D adds scalar_tanh, G adds fp32/bf16
choice. The two safety gates are placed where they matter: the **parity gate** at B
(protects the current model) and **`verifyModelFile`** at E/F (proves persistence +
historical loads).

**A. `NetworkArchitecture` foundation (rework).** Rebuild the struct as the full
decomposed, all-required, purely-topological surface (§5a): every field, no defaults;
`InputEncoding` enum + `planeGroups`; canonical snake_case `Codable` (`sortedKeys`);
`validate()` (odd kernels, SE divisibility, memory budget); `parameterCount`,
`architectureSummary`, `weightTensorPlan()` branching on every axis; built-in `Preset`
enum (8-blk-v3, 16-blk-v3, 12-blk-v4, 5-blk-v4, 8-blk-v4). Drop the SHA/arch_hash
identity machinery; `label` lives outside the struct.
*Validate:* XCTests — param + plan counts for every preset, validation rejects, `Codable`
round-trip + canonical determinism (reorder fields → identical bytes). No `ChessNetwork`
wiring yet.

**B. Forward-graph builder = one code path + PARITY GATE (rework).** `ChessNetwork`'s
static builders branch on all decomposed axes (block_activation_style, skip_merge,
use_rezero, conv1/2 kernel, se_style/ratio, policy_head_style + pre_conv_channels,
value_head_style forward graph, network-wide activation_function incl. silu/gelu exact).
Remove the `= .current` init defaults — every call site passes its arch.
*Validate (INVARIANT CHECKPOINT, §15):* default/current config builds the **bit-identical**
graph (forward-pass parity test + `verifyModelFile`); build-and-forward smoke for every
preset.

**C. Input-encoding reach (`BoardEncoder` + `ReplayBuffer`).** `input_encoding`-driven
plane count: `BoardEncoder.encode(…, encoding:)` branches (basic20/basic30);
`tensorLength`/stem depth/`ReplayBuffer` stride become instance-derived; resume validates
buffer stride. *Validate:* encoder fills exactly `planeGroups` ranges (golden test per
encoding); a basic20 net builds + short self-play fills the buffer at the right stride.

**D. Trainer reach (value-loss branch).** `ChessTrainer.buildTrainingOps` branches on
`value_head_style`: wdl→categorical CE, scalar_tanh→MSE (resurrect the pre-WDL loss +
tanh baseline). *Validate:* a few steps on a wdl AND a scalar_tanh net; loss drops,
gNorm finite.

**E. Safetensors-native, no arch_hash (rework).** Embed full canonical `architecture` +
`label`; identity = embedded config; integrity = `content_sha256`; **remove the
`arch_hash` write+verify**. Name-keyed load that **builds the graph from the embedded
config** then loads (config-driven build replaces the hard-refuse). *Validate:*
save→load→resave bit-exact on current + a couple presets; `verifyModelFile`; Python load
(throwaway venv).

**F. Legacy `.dcmmodel` (read-mostly).** Bidirectional `[storedHash ↔ preset]` table;
read → build preset arch → load (replaces hard-refuse); optional write only when config
matches a legacy preset. *Validate:* load the **real 8-block-v3 and 12-block-v4** files
through `verifyModelFile` (bit-exact). [**10% "load old models" milestone.**] (16-blk-v3
unverifiable — no file on disk.)

**G. Per-model precision.** `compute_data_type` honored as-is (no gate, §9); thread off
the static `dataType`; fp32-master mixed-precision for bf16. *Validate:* build+train a
bf16 and an fp32 net; save/load preserves requested dtype; current bf16 behavior
unchanged. [**90% "build+train any arch" fully enabled.**]

**H. Build UX + presets + CLI (rework Phase 4).** The **Build New Model screen** (full
spec in §10: preset picker built-ins+user, every free-form field, live
paramCount/summary, memory validation, Save-as-Preset, `label`) opened from the Build
action + **File ▸ New Model…**; Presets folder (user-saved only); `--architecture-preset`
/ `--architecture-file` (remove the `architecture.json` loader); `--uci` from embedded
config; `--playchess`. *Validate:* the screen builds a free-form arch that trains; CLI
build from preset + file; `--uci`/`--playchess` resolve.

**I. Capstone.** Build a fresh 8-block-v4 from preset; load every available historical
file; full suite green; Python inference spot-check.

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

- Changing `board_size` (chess is 8×8) and the `policy_channels`=76 / 4864 move encoding
  (`PolicyEncoding`). *(NOTE: `input_planes` via `input_encoding` and `value_head_classes`
  via `value_head_style` are now IN scope — they're first-class configurable axes per the
  §5a walkthrough. The 4096 from-to `policy_encoding` is a deferred future axis, §5a.)*
- bf16/f16 weight storage on disk (deferred future goal; design keeps it additive).
- Search of any kind (explicit project non-goal).
- Direct Metal Tensor / Metal Performance Primitives matmul path for the apple10 matrix
  units (a future perf optimization beyond MPSGraph; §9).

## 15. Hard invariants (constrain all remaining work)

These two are non-negotiable and gate every remaining phase:

1. **One code path.** No parallel architecture classes, no duplicated builder
   functions. All architecture variability is expressed as branches inside the
   *existing* single builders, keyed off the `NetworkArchitecture` value type. The
   struct is the only discriminator.
2. **Zero runtime change to the current model.** For the v4/current architecture the
   constructed MPSGraph — and therefore the compiled `MPSGraphExecutable` — must be
   bit-identical to today: same ops, names, shapes, build order, same in-GPU
   operations, same performance. This works because all branching happens **once, at
   graph-build time, on the CPU**; the v3/precision arms are dead code for a v4
   build. The bit-exact forward-pass save gate (`verifyModelFile`) + the default-arch
   parity test are the standing proof; any change that perturbs them for the default
   arch is a regression, not a feature.

## 16. Resolved decisions (formerly open)

All four are now settled by the §5a/§9 walkthrough:

1. **bf16 → RESOLVED (simplified):** NO detection/gate — `compute_data_type` is honored
   as configured. bf16 works everywhere (no crash), only faster on M5+, which the user
   accounts for. Probe kept as a standalone diagnostic. (§9)
2. **Precision scope → RESOLVED:** per-model precision *selection* **and** the gate ship
   together in Phase 5 (user: "both in the same phase").
3. **Presets → RESOLVED:** built-ins compiled-in & immutable (never exported); user-saved
   presets are files in the Presets folder; build UI offers built-ins + user-saved +
   free-form; v3 presets are selectable for building. (§10)
4. **v3 depth → RESOLVED:** full **build + train** support for *every* configurable
   architecture (user: "we need to be able to train every single architecture we can
   create" — 90% experimentation). So v3 training-resume (trainer graph + velocity) is in
   scope, not just inference load.

**Remaining step before implementation:** re-sequence §12 into the actual build order
reflecting the §5a/§6/§9/§10 finalized design (and the Phase 1–4 rework it implies).
