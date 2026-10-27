# Inference compute-dtype self-play speed A/B — fp32 vs bf16 vs fp16

Owner request 2026-10-19. Measure **self-play generation rate (plies/hour)** for the *same trained weights* under three compute dtypes on this M5 Max. The M5 Max is claimed to have extra fp16 hardware; the question is whether fp16 inference is **materially faster than bf16** (the incumbent, numerically safe reduced precision). Mathematical quality is explicitly a separate, later question — here we measure speed and verify fp16 does not blow up (non-finite).

## Decision rule (set before measuring)

- Head-to-head is **bf16 vs fp16**. fp32 is the upper-precision reference only.
- Adopt fp16 only if it is **≥ 25 % faster in plies/hour than bf16** at the self-play batch size (K = `self_play_concurrency`) in real use, **and** the fp16 forward pass stays finite (no `[ALARM] non_finite` / `divergence`, `[LAYER-HEALTH] nonFinite=0`).
- If fp16 ≈ bf16 (< ~10 %), not worth fp16's narrower exponent range.

## Why this is a clean A/B (facts confirmed in code)

- **Compute dtype is per-model-architecture only** (`arch.computeDataType`); there is **no process/env/launch override** except the GUI resume knob `forceFloat32` (→fp32 only, `SessionController+Checkpoint.swift:1054`). Enum `ComputeDataType` = `float32` / `bfloat16` / `float16` (`NetworkArchitecture.swift:671`).
- **Weights are always stored fp32** and cast to the compute dtype at load (`ChessNetwork.swift:1976`, design doc `:145-166`). So all three arms read the *identical* fp32 bytes; only the load-time cast differs. Weight-load/memory cost is identical across arms → any bf16→fp16 delta is pure GPU ALU throughput (the M5 Max fp16 path), not storage/bandwidth.
- **Head tails always run fp32** (`headTailDataType = .float32`, `ChessNetwork.swift:3303`); `reducedPrecisionFastMath` defaults to `.none`. So fp16 overflow risk is confined to **tower conv/matmul intermediates**, not the policy/value logits. There is no explicit fp32-accumulation forcing on the inference forward path (training has its own fp32 reduction path, not used here), so accumulation precision for tower ops is whatever MPSGraph does internally for the dtype.
- **The one blessed way to set the dtype** is `NetworkArchitecture.withComputeDataType(_:tail:)` (`:2522`): to fp32 → tail `does_not_apply`; to bf16/fp16 → caller states a tail in `reducedPrecisionCases` (`fp32_from_pre_bn` | `mixed_final_projection`).

## Build step (minimal, in-architecture)

There is **no CLI to re-dtype existing trained weights** (`--derive-model` has `--set-policy-tail-precision` but no compute-dtype op; `--new-model` mints a fresh untrained net at a preset's baked dtype). So add one derive operation, mirroring `SetPolicyTailPrecisionDeriveOperation` exactly:

- New file `Persistence/ComputeDataTypeDerive.swift`: `SetComputeDataTypeDeriveOperation` with flag **`--set-compute-dtype <float32|bfloat16|float16>`**.
  - `apply(to:)` calls `architecture.withComputeDataType(dtype, tail:)` where tail = `nil` for fp32; for bf16/fp16, preserve the source's tail if already a `reducedPrecisionCase`, else default **`fp32_from_pre_bn`** (keeps the policy tail in fp32 so only the **main-path** dtype differs across the two reduced arms — the cleanest isolation).
  - Refuse if the model already holds that dtype ("nothing to derive"), like every other op.
  - `tensorRewrites = []` — storage stays fp32, only header `compute_data_type` (+ `policy_tail_precision`) change; tensors copied bit-exact (same guarantee `--set-policy-tail-precision` relies on).
  - `changedArchitectureFields = [compute_data_type, policy_tail_precision]`.
- Register `SetComputeDataTypeDeriveOperation.kind` in `ModelDerivation.operationKinds` (`ModelDerivation.swift:124`).
- Build once via **drews-xcode-mcp** (scheme `DrewsChessMachine`, project `DrewsChessMachine/DrewsChessMachine.xcodeproj`); never `xcodebuild`.

## Variants (same weights, three dtypes)

Source: latest clean elitecyc weights `Models/20261011-zlra-elitecyc-latest-step300300.safetensors` (trained, realistic activation magnitudes → meaningful blow-up test). Derive three files with the new op:

| variant | computeDataType | policy tail |
|---|---|---|
| `zlra-speed-fp32` | float32 | does_not_apply |
| `zlra-speed-bf16` | bfloat16 | fp32_from_pre_bn |
| `zlra-speed-fp16` | float16 | fp32_from_pre_bn |

Identical topology (3 blocks ×128ch 9×9, 8.27M params) and identical fp32 weights; only the tower compute dtype differs.

## Harness — pure self-play window

No self-play-only CLI exists; self-play runs only inside Play-and-Train, launched headless with **`--train`** (`DrewsChessMachineApp.swift:234`; autoTrain fires from `onAttached`, so no window/reopen gotcha — `UpperContentView.swift:1742`). To get a **trainer-free** window, set `replay_buffer_min_positions_before_training == replay_buffer_capacity`: the trainer loops idle on the buffer gate (`SessionController+Training.swift:1337`, strict `<`) until the buffer is completely full, so the whole run is pure self-play generation with zero trainer GPU work.

- Parameters file: `min_positions_before_training = replay_buffer_capacity = 1_000_000`, fixed `self_play_concurrency = K`, default sampling schedule, fixed `--seed`.
- Launch the freshly built binary directly (as the corpus runs do):
  `DrewsChessMachine --train --start-model <variant> --parameters <cfg> --output <results.json> --seed <N>`
- Rate read from the `[STATS]` line token **`spRate=<n>/hr`** (= emitted positions/hr = plies/hr, `SessionController+Training.swift:1790`); cross-check with raw `spMoves` / wall-elapsed.

## Smoke test (this task)

3 minutes per arm (fp32, bf16, fp16), single K (self-play default, or 512 if default is tiny — record it), GPU otherwise idle (both trainers stopped). Per arm:
- Discard first ~60 s (Metal shader compile + ramp), then read steady-state `spRate` and `spMoves`/elapsed.
- fp16 blow-up gate: `[LAYER-HEALTH] nonFinite=0`, no `[ALARM] non_finite`/`divergence`, spRate stable.
Report: plies/hour per arm, fp16/bf16 and fp32/bf16 ratios, fp16 finiteness verdict.

## Full test (later, if smoke warrants)

Sweep K ∈ {64, 256, 1024, 4096}; interleave fp32/bf16/fp16 ×3 to cancel thermal drift; log `[MEM]`/thermal and discard throttled windows; optional `--probe-model` `gpuMs` per-batch forward micro-benchmark as a controlled secondary measure. Write up as an `E-NNNN` experiment page with a Reproduce section (commands, model SHAs, build number).

## Non-goals / safety

- Not evaluating move quality or final strength here.
- No change to stored precision (always fp32 on disk) and no change to training. The derive op only adds a header-level dtype setter; `--set-policy-tail-precision` is unaffected.
