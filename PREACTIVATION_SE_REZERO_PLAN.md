# Pre-activation (ResNet v2) tower + scale-and-bias SE + ReZero — implementation plan

Status: **IMPLEMENTED** (architecture v4 = `ChessNetwork.architectureVersion`). See
the 2026-05-31 CHANGELOG entries for the as-built summary.

> **Correction since drafting (updated 2026-06-23):** the default tower is now the
> **`v4_5block_7x7` preset — 5 blocks** (`NetworkArchitecture.Preset.current`), not the
> 16 of the original draft nor the 12 of the earlier correction. The depth-appropriate
> ReZero α init is therefore **`1/√5 ≈ 0.447`**, not the `1/√16 = 0.25` quoted
> throughout this doc, and the per-block variance-growth figures (e.g. "×1.28 over 16
> blocks") shift accordingly. α is still derived from `1/√numBlocks`, so it tracks the
> configured block count automatically — read the `0.25`/`16` references below as
> `1/√numBlocks`/`numBlocks`. The default learning rate was also raised to `1e-3` for
> the bf16 path (separate CHANGELOG entry).
>
> The `architectureVersion` / `currentArchHash` mechanics described in §3.B and §6 were
> superseded by the **runtime-configurable-architecture refactor**: architecture is no
> longer a compile-time `architectureVersion` constant. `architectureVersionLabel`
> (`NetworkArchitecture.swift`) is now a *computed* label, and `archHash` is reduced to
> a **legacy-`.dcmmodel`-only lookup** (`legacyDcmmodelArchHashes` in
> `NetworkArchitecture.swift`, resolved by `ModelCheckpointFile`) — the modern
> `.safetensors` format embeds the full `NetworkArchitecture` config instead of relying
> on a hash bump for compatibility. Read §3.B/§6's "bump `architectureVersion` to
> perturb `archHash`" prose as historical.

Branch context: `bf16-trainer`. This is a fresh-architecture change (architecture
"v4"). It is **not** weight-compatible with any existing `.dcmmodel` / `.dcmsession`;
old files will be rejected on load (see §6). No migration code (per project rules).

---

## 1. Goal / target architecture

Replace the current post-activation (ResNet **v1**) residual block with a
pre-activation (ResNet **v2**) block carrying a *scale-and-bias* Squeeze-Excitation
module and a per-block ReZero/SkipInit scalar on a **clean identity skip**. One
final `BN → ReLU` is added at the tower end (because pre-activation blocks end in a
bare conv, the tower output is otherwise un-normalized before the heads).

Target block (C = 128, SE reduction r = 4 → reduced = 32, convs 3×3 bias-free):

```
x ─────────────────────────────────────────────────────────────┐  CLEAN IDENTITY
[B,128,8,8]                                                      │  (untouched: no
   │                                                             │   BN / ReLU / scale)
   ▼  F(x):                                                      │
   BN1 (γ=1, β=0) → ReLU → conv1 (3×3, 128→128, no bias, He)     │
   BN2 (γ=1, β=0) → ReLU → conv2 (3×3, 128→128, no bias, He)     │
   z = conv2 out                                                 │
   ── SE (scale-and-bias) ───────────────────────────────────   │
     GAP(8×8) → [B,128]                                          │
     FC1 128→32   (He W, bias 0) → ReLU                          │
     FC2 32→256   (He W, bias 0)        # 2C = gammas ‖ betas    │
     split → gammas[B,128], betas[B,128]                         │
     SE_out = sigmoid(gammas) · z + betas        # [B,128,8,8]   │
   ──────────────────────────────────────────────────────────   │
   branch = α · SE_out          # α: trainable SCALAR, init 1/√16 = 0.25
   ▼                                                             │
out = x + branch   ◄──────────────────────────────────────────  ┘
[B,128,8,8]        ★ NO activation after the add (v2 identity highway)

... after the LAST block:  BN_final → ReLU  →  policy head / value head
```

### How the six original bullets map to this spec (the spec is authoritative)

| Original bullet | Resolution in this plan |
|---|---|
| 1. `bn2 γ = 1/√(numBlocks)` | Becomes a **per-block ReZero scalar α**, init `1/√16 = 0.25`, on the branch output before the add (not bn2's γ). In v2 the convs are last, so bn2's γ no longer governs branch magnitude — the scalar does. `bn2` now uses standard `γ=1`. (Confirmed via clarifying Q2 → "learnable scale".) |
| 2. "SE with bias" | The **scale-and-bias gate**: `SE_out = sigmoid(gammas)·z + betas`. FC2 now emits **2C=256** (a `gammas` half and a `betas` half); sigmoid is applied **only** to the `gammas` half, `betas` is added linearly. (Confirmed Q1.) |
| 3. "SE Glorot init" | **FC2 → Glorot (Xavier) normal; FC1 stays He-normal.** FC2 feeds the symmetric-ish sigmoid gate, where Glorot's `std = √(2/(fanIn+fanOut))` is the correct variance target; FC1 feeds ReLU, where He is correct. (Confirmed: "use gloriot for SE FC2".) |
| 4. "SE sigmoid instead of relu" | Resolves to the gate split above. **FC1 keeps ReLU** (spec diagram shows FC1 → ReLU). No standalone activation swap. |
| 5. "ResNet v2 (BN/ReLU first, conv last)" | Pre-activation block: `BN→ReLU→conv` ×2. |
| 6. "Skips come through unchanged" | Clean identity skip: `out = x + α·F(x)`, **no** post-add ReLU. |

### Notes / deliberate non-changes
- **No MLH head.** The spec diagram says "policy / value / MLH heads" as a generic
  template; this network has only policy + value. No moves-left head is added.
- **Stem → `conv → BN`, drop the ReLU.** Keep the stem BN (it bounds `x_0`, the skip
  highway's starting value, to unit per-channel scale — which keeps the accumulated
  `x_0 + Σ branches` well-conditioned for `BN_final` at the bottom), but drop the
  stem ReLU so no nonlinearity is applied before the highway begins — the first ReLU
  is correctly deferred to block0's pre-activation. block0's `BN1` re-normalizing the
  already-BN'd stem output is mildly redundant but harmless. The only "naked" (no
  trailing BN) convs in the net are the heads' final logit convs (output
  projections); everything else is `BN→ReLU→conv` or, for the stem, `conv→BN`.
- **Heads unchanged.** Each head still begins `conv1×1 → BN → ReLU`. With `BN_final
  → ReLU` ahead of them, the head input is normalized & non-negative — the same
  regime the heads were designed for (today the tower ends in a post-add ReLU; in v2
  that role moves to `BN_final → ReLU`).

---

## 2. ML rationale (the "why", for the record)

- **Pre-activation / clean skip (v2).** In v1 the identity path passes through a
  ReLU on every skip-add, so it is not a true identity — gradients are gated at each
  of the 16 blocks. v2 puts BN/ReLU *inside* the residual function and leaves the
  skip a bare add, creating an uninterrupted additive highway `x_L = x_0 +
  Σ α_i F_i(x_i)`. Gradient to `x_0` is `1 + Σ ∂(α_i F_i)/∂x_0` — the `1` never
  vanishes, which is the whole point at depth 16.
- **ReZero α (init 0.25 = 1/√L).** With L additive branches of ~unit variance, the
  tower output variance grows ~L. Scaling each branch by `1/√L` keeps total variance
  O(1) at init. Unlike the old zero-γ trick (branch dead at init, must be "woken up"
  purely by gradient), α=0.25 lets every block contribute signal **and** gradient
  from step 1, while still bounding depth variance. It's a single learnable scalar,
  so the network can grow/shrink each block's contribution freely.
- **Scale-and-bias SE.** Vanilla SE can only *attenuate* channels (multiply by
  (0,1)). Adding a learned `betas` lets the module also **inject** a per-channel,
  globally-pooled additive signal — a cheap conditional bias driven by the whole
  board. Sigmoid stays on the gain half so scaling remains bounded; the bias half is
  linear so it can push either direction.
- **`bn2` γ=1 (no more zero-γ).** The depth-variance job moves entirely to α, so
  bn2 reverts to a standard unit-gain BN.

---

## 3. File-by-file changes

### A. `Network/ChessNetwork.swift` — core graph

1. **Add an architecture-version constant** (e.g. `static let architectureVersion =
   4`) used only to perturb `archHash` (see §6). Keep `numBlocks = 16`.
2. **Stem → `conv → BN`, drop ReLU.** Keep the `stem_bn` `batchNorm(...)` call;
   remove only the `stem_relu`. `x` flows `stem_conv → stem_bn →` block loop.
   (No tensor-count change for the stem — BN gamma/beta + running stats stay.)
3. **`batchNorm(...)`**: remove the `zeroInitGamma` parameter and its zero-γ branch
   (bn2 no longer needs it). All BN layers init γ=1, β=0.
4. **Rewrite `residualBlock(...)`** to the pre-activation form:
   - `BN1 → ReLU → conv1` then `BN2 → ReLU → conv2` (both convs bias-free, He — keep
     `heInitDataConvOIHW`). BN layers append trainables/running-stats as today.
   - **SE FC2 widened to 2C + Glorot init:** weight `[reduced, 256]`, bias `[256]`
     (`glorotInitDataFCInOut(shape: [reduced, 256])`, `zerosData(count: 256)`).
     Glorot normal `std = √(2/(fanIn+fanOut)) = √(2/(32+256)) ≈ 0.0833`. FC1 stays
     `heInitDataFCInOut` (feeds ReLU).
   - **Add `glorotInitDataFCInOut(shape:)` helper** in the Data Helpers section,
     alongside `heInitDataFCInOut`: same `[in,out]` layout, but `std =
     sqrt(2/(shape[0]+shape[1]))`. Reuse the vectorized `heInitFloats(count:std:)`
     normal-sampler (rename it mentally as "gaussian with given std" — it already
     just takes a std), so only the std differs.
   - **Split FC2 output `[B,256]` → gammas `[B,128]`, betas `[B,128]`** via
     `graph.sliceTensor(_, dimension: 1, start: 0, length: 128)` and `start: 128`.
     (MPSGraph has no zero-cost reshape-split for this; two slices is the idiom.
     Add a short entry to `documentation/mpsgraph-primitives.md`.)
   - Reshape both to `[-1,128,1,1]`; `scale = sigmoid(gammas)`;
     `SE_out = scale * z + betas` (broadcast multiply then broadcast add).
   - **ReZero scalar α:** `graph.variable(with: makeWeightData([0.25]), shape: [1],
     dataType: .dataType, name: "\(prefix)_res_scale")`. Append to `trainables`
     with `shouldDecay = false`. `branch = graph.multiplication(SE_out, α)`.
   - `out = graph.addition(input, branch)`. **No ReLU.**
5. **Add `BN_final → ReLU` after the block loop**, before policy/value heads
   (`name: "tower_final_bn"`). Standard BN (appends trainables + running stats).
   Required: the last block ends in a bare conv-add, so the tower output is an
   un-normalized, never-activated linear accumulation — this is the canonical v2
   "post-activation at the very end" that normalizes + activates it before the heads'
   first conv.
6. **Update `parameterCount`:**
   - `seFC2 = (seReduced * 2*channels) + 2*channels`.
   - `+ 1` per block for α (`perBlock += 1`).
   - `+ (4 * channels)` for `tower_final_bn` (γ+β+mean+var; γ/β trainable,
     mean/var running) — add a `towerFinalBN = 4 * channels` term. (Stem term
     unchanged — stem keeps its BN.)
   - Re-verify the headline param count in the class doc comment + AboutPopover text.
7. **Update class doc comment** (lines ~57–88) and `residualBlock` / `batchNorm` doc
   comments to describe v2 + scale-and-bias SE + ReZero + `conv→BN` stem. Remove
   zero-γ prose.

### B. `Persistence/ModelCheckpointFile.swift`
- Mix `ChessNetwork.architectureVersion` into `currentArchHash` so old files reject
  with a clear "architecture mismatch" rather than a deep tensor-count error.
- Update `maxTensorElementCount` comment (SE FC2 is now `32×256 = 8,192`; still far
  under cap — no functional change).

### C. `Network/NetworkWeightAnalyzer.swift`
- Remove the `_bn2_gamma` zero-γ special case in the init-L2 reference (bn2 is now
  γ=1, covered by the general `_gamma` rule).
- `_se_fc2_weights` now uses a **Glorot** init-L2 reference, not He: expected L2 =
  `√n · √(2/(fanIn+fanOut))` with fanIn=32, fanOut=256, n=32×256. Add a Glorot case
  (the analyzer's `expectedInitL2`/`heInitL2` path currently assumes He for all
  `_se_*_weights`). FC1 stays He.
- Add references for the new variables: `*_res_scale` (α, scalar, init 0.25 — add a
  deterministic-init case so it isn't flagged against a He baseline) and
  `tower_final_bn_*` (γ=1/β=0 — general rule covers it; just ensure the section map
  routes `tower_final_*` somewhere sane, e.g. its own "tower_final" section or
  "stem").
- The per-SE gate distribution analysis must read the **first 128** of the 256-wide
  FC2 output as the `gammas` (the sigmoid'd gate); the last 128 are `betas` and are
  not a (0,1) gate. Update that aggregation accordingly.

### D. `App/UpperContentView/AboutPopoverContent.swift`
- No code change (reads `ChessNetwork.parameterCount`), but verify the displayed
  "~X.XM parameters" text reads sensibly after the param-count change.

### E. Docs
- `documentation/mpsgraph-primitives.md`: add a `sliceTensor` split entry (FC2 →
  gammas/betas). (The trainer already uses `graph.sliceTensor(dimension:start:length:)`
  at `ChessTrainer.swift:2600` for the W/D/L columns — same idiom, confirmed available.)
- `documentation/dcm_architecture_v2.md` + `documentation/chess-engine-design.md`:
  update the tower/block description (pre-activation v2, scale-and-bias SE, ReZero,
  `conv→BN` stem, `BN_final→ReLU`).
- `CHANGELOG.md`: new dated, git-hash-tagged entry (newest first) summarizing the
  architecture-v4 change and the intentional old-checkpoint incompatibility.

---

## 4. Things that are automatically handled (verified)

- **Gradients / optimizer.** `ChessTrainer.buildTrainingOps` iterates
  `network.trainableVariables` generically (per-variable `gradientMissing` guard) and
  allocates velocity buffers parallel to it. α and `tower_final_bn` γ/β get gradients,
  velocity, and weight-decay exclusion (`shouldDecay=false`) for free.
- **Weight transfer** (`exportWeights`/`loadWeights`) and per-variable load
  infrastructure are built from the live variable list — no hardcoded counts.
- **BN warmup** (`computeBatchStats` / `bnBatchMean|VarTensors`) auto-covers the new
  `tower_final_bn` because it appends to the same arrays.
- **bf16 path:** α/γ/β init values route through `makeWeightData` (Float32→bf16) like
  every other weight. No special handling.

---

## 5. Tests (per project rules — pure-logic invariants get tests)

The change is mostly Metal/MPSGraph (not unit-testable directly), but:
- **`MomentumOptimizerTests` — safe, no change needed** (audited): every count it
  checks is derived live from `trainer.network.trainableVariables.count` /
  `bnRunningStatsVariables.count`, so the added α + `tower_final_bn` tensors flow
  through automatically. (Re-run to confirm.)
- **`PolicyHeadCorrectnessTests.testInferenceNetworkPolicyAtInitHasReasonableLegalMass`
  — comments are now stale; assertion should still hold** (audited). The test builds
  a raw `ChessMPSNetwork(.randomWeights)` and asserts `meanRatio ≥ 0.3`. Its long
  comment block (lines ~210–258) attributes well-conditioned init to the **zero-γ
  identity init we are removing** and says "8-block tower" — both stale. The test
  actually passes because `.randomWeights` runs a **one-shot BN warmup** that primes
  running stats so inference BN truly normalizes; that warmup is generic over the
  BN-tensor arrays and will prime the new `tower_final_bn` too. **Action:** rewrite
  the stale comment to describe the ReZero + SE-damping + BN-warmup rationale (and
  16 blocks); **do NOT modify the `0.3` assertion**. Verify the printed
  `[init-legalmass] meanRatio` still sits comfortably above 0.3 (expected ~0.5–1.5).
- **Add** a small test asserting `ChessNetwork.parameterCount ==` the summed element
  count of `exportWeights()` on a freshly-built network (guards the manual
  `parameterCount` formula against drift — this is exactly the kind of pure-logic
  invariant the formula can silently desync on).
- **Add** a slice-correctness test for the FC2 gammas/betas split if feasible at the
  graph level (verify a known `[B,256]` splits into the intended halves), else cover
  via the diagnostics probe.
- All existing tests must pass unmodified (except deliberate count updates above).

---

## 6. Checkpoint compatibility (no migration)

- archHash bump (§3.B) → existing `.dcmmodel` / `.dcmsession` reject cleanly on load
  with "Architecture mismatch". Even without the bump they'd fail (FC2 element count
  + new α/BN tensors change the per-tensor counts and total tensor count), but the
  explicit bump gives the clean up-front error.
- Trainer session velocity buffers are parallel to `trainableVariables`; their count
  changes too, so old `.dcmsession` resume is intentionally impossible. **Call this
  out in the commit message** (per the parameter-rename convention).

---

## 7. Validation (success criteria)

1. **Build** via xcode-mcp-server (`build_project`) — clean compile. *(No app/test
   run while a training session is live — confirm idle first.)*
2. **Fresh-build sanity** (Engine Diagnostics / Run Forward Pass on a new network):
   - Policy entropy near uniform `log(4864) ≈ 8.49` nats at init (clean-skip + α
     scaling should keep init logits un-inflated; if `pEnt` is far below ~8, the
     branch is over-contributing → check α and BN_final).
   - Derived value scalar ≈ 0 at init (W/D/L bias init `[0, ln6, 0]` unchanged).
   - No NaN/Inf in policy or value readback.
3. **archHash:** attempt to load a pre-change checkpoint → expect a clean
   "Architecture mismatch" error (not a crash, not silent wrong output).
4. **`parameterCount` test passes** (matches exported tensor element sum).
5. **Short Play-and-Train smoke run** (watch the session log, not just the console):
   - `gNorm` finite and not exploding in the first ~200 steps.
   - `pEnt` falls smoothly from ~8.49 (no instant collapse; alarm threshold 5.0).
   - `pD` falls off its 0.75 init prior (value head learning, not draw-collapsing).
   - α values (via NetworkWeightAnalyzer) drift away from 0.25 over training — sign
     that blocks are individually tuning their contribution.
6. **`PolicyHeadCorrectnessTests` init-legalmass:** run it explicitly and read the
   printed `[init-legalmass] meanRatio` — confirm it sits well above 0.3 (the BN
   warmup should keep it ~0.5–1.5 even with ReZero replacing zero-γ). If it lands
   near 0.3, investigate before proceeding (do not lower the threshold).
7. **All XCTest targets green.**

---

## 9. Recheck findings (audit pass)

- **Gradient / optimizer / weight-transfer / BN-warmup paths are all generic** over
  the live variable arrays — α, `tower_final_bn`, and the widened FC2 flow through
  with no count hardcoding. `MomentumOptimizerTests` verified generic.
- **`sliceTensor` confirmed available** (used in `ChessTrainer` already).
- **Init-legalmass regression is covered by BN warmup**, not zero-γ — removing
  zero-γ is safe; only the test's explanatory comments need fixing (see §5).
- **Variance sanity at init:** SE gate ≈ 0.5 (Glorot→small gammas→sigmoid≈0.5) and
  ReZero α=0.25 make each branch contribute ≈ `0.125·z`, so per-block variance
  growth is ≈ ×1.016 → ≈ ×1.28 over 16 blocks. Well-conditioned; no runaway. (This
  is far tamer than the pre-zero-γ ×2/block blowup the old regression caught.)
- **No bad decisions found.** ReZero α=0.25 (vs classic 0) is deliberate and safe
  given the SE 0.5× pre-damping. 0.25 = 2⁻² is exact in bf16; Glorot std ≈ 0.083 fine.

### Coordination with upcoming value-head changes
The user may edit the **value head** nearby. This plan does **not** modify the value
head's internals, but several touchpoints are shared — coordinate on these to avoid
collisions:
- **`ChessNetwork.parameterCount`** has a `value = valueConvBN + valueFC1 + valueFC2`
  term. If the value head's shapes change, both edits touch this function — merge
  carefully (each side updates only its own term).
- **`tower_final_bn → ReLU`** now sits immediately *before* the value head's first
  conv (as well as the policy head's). A value-head redesign should assume its input
  is the normalized+ReLU'd tower output, not the raw tower sum.
- **`currentArchHash`** — only one version bump is needed; if both changes land
  together, bump `architectureVersion` once (don't double-count).
- **`NetworkWeightAnalyzer`** section map / fanIn table — value-head variable names
  (`value_*`) are independent of the block/SE/α names this plan touches, so no
  overlap there.
- **`valueHeadClasses`** feeds `archHash` and `parameterCount`; if the value-head
  change alters it, that alone changes `archHash` (old files reject) — note in
  whichever commit lands it.

---

## 8. Open items to confirm before/while implementing

- ~~Q (Glorot vs He for SE)~~ **RESOLVED:** FC2 → Glorot (whole [32,256]), FC1 → He.
- ~~Q (stem)~~ **RESOLVED:** stem = `conv → BN`, drop the ReLU.
- ~~Q (tower end)~~ **RESOLVED:** keep the closing `BN_final → ReLU` before the heads.
- **Q (α granularity):** spec says scalar per block (planned). Per-channel
  `[1,C,1,1]` is the LayerScale variant — not planned unless you want it.
```
