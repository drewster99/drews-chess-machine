# macOS 27 beta 1 + MPSGraph — broken / changed behavior found in DCM

A single place to collect every MPSGraph numerical / correctness issue this
project has hit on the **Xcode 27 beta 1 + macOS 27 beta** toolchain (first seen
2026-06), on Apple Silicon. Companion to `mpsgraph-primitives.md` (which is the
"how to use the ops"; this is "how they broke"). Newest understanding wins —
update in place as the picture sharpens or a future toolchain fixes something.

Two facts frame everything below:

- **It is new with the toolchain bump.** Byte-identical source trained bf16 for
  hundreds of thousands of steps before this OS/Xcode upgrade. A stale Metal
  toolchain was ruled out (reinstalling the matching one did not help;
  `xcode-select` already pointed at the beta).
- **The umbrella symptom is "finite forward, NaN gradient."** `ChessTrainer.trainStep`
  returns a finite forward loss but a non-finite gradient
  (`nonFiniteLoss(total: nan, …, gradNorm: nan)`), which poisons the weights
  within a few steps. Everything below is either a confirmed cause, a candidate
  cause under an A/B probe, or a thing ruled out.

The live reproduction harness is `DrewsChessMachineTests/MacOS27NaNIsolationTests.swift`
(precision × batch × step-count sweep, plus the layout/precision A/B probes) and
`DrewsChessMachineTests/FP16ComputePathTests.swift` (the fp16 cells). These tests
are intentionally **pinned out of the default scheme run** and several stand as
known-failing tripwires; do not "fix" them by relaxing the assertion.

---

## 1. Fused dual-write buffer stomp — CONFIRMED, fixed by splitting the write

**What.** Under bf16 mixed precision, the per-trainable optimizer tail emits
**two** target writes into one compiled executable: the fp32 master `assign` and
the bf16 working `assign` through an (unnamed) `cast` temporary. On this beta
stack that fused dual-write **corrupts the trainable weight buffers**: the bf16
working weights read back NaN and the fp32 masters read back garbage (an exact
`1.0` sentinel) from step 2 onward.

**Fingerprint.**
- **bf16-only** — the fp32 path (single write, no master, no cast temp) is clean.
- **NOT the cast op** — a standalone `cast([128,32] fp32 → bf16)` through both
  `graph.run` and the compiled-executable path is bit-exact clean.
- **layout / liveness sensitive** — it favors the rank-2 SE `fc1` `[128,32]` and
  `fc2`-bias `[1,256]` tensors, and the corrupted-element count grows with tower
  size. That profile reads as a buffer-aliasing / liveness-planner fault around
  the fused dual-write, not a value bug in any single op.

**Fix (shipped, default on).** `ChessTrainer.splitWorkingWeightSync = true`
splits the working-weight sync `working = cast(master)` OUT of the fused training
executable and runs it as a separate `graph.run` after the master update's
command buffer has fully completed (different allocation/liveness scope, its own
command buffer). This takes the single-block repro from **~1,010,433 non-finite
working elements to 0 across 16 steps**, reproducibly. Only the *trainable*
working writes are split; the BN running-stat master+working write is left fused
(it never corrupted). Set false to restore the original fused single-executable
path (one `graph.run` per step, faster) once a future toolchain fixes the stomp,
or to reproduce the bug.

---

## 2. Automatic NCHW→NHWC layout conversion is now the default — CANDIDATE

**What changed.** macOS 27 / Xcode 27 b1 made automatic NCHW→NHWC layout
conversion for GPU convolutions the **default**.
`MPSGraphCompilationDescriptor.convertLayoutToNHWC` is now a deprecated no-op;
the opt-out is the new `MPSGraph.disableAutoLayoutConversion()` (macOS 27). The
divergence appeared at the same toolchain bump, so a layout-conversion numerics
change is a natural suspect.

**Probe (shipped A/B knob).** `ChessNetwork(disableAutoLayoutConversion:)` /
`ChessTrainer(disableAutoLayoutConversion:)` apply the opt-out at every
`graph.compile` site. DCM declares every conv explicitly `.NCHW` / `.OIHW`, so
opting out is *expected to be inert* on a healthy stack — this is a falsification
probe, not a fix. There is also a deterministic forward-divergence probe
(`MacOS27NaNIsolationTests.layoutForwardDivergence`): two identical-weight
inference nets (layout default vs disabled) run one fixed batch; the element-wise
max-abs output gap is the layout-induced numeric divergence. In fp32 a healthy
gap is last-bit (~1e-5); a large fp32 gap would indicate something structural
(transposed/mis-padded tensor), i.e. a real beta bug.

---

## 3. `reducedPrecisionFastMath` reduced-precision conv shortcuts — CANDIDATE

**What.** `MPSGraphCompilationDescriptor.reducedPrecisionFastMath` (macOS 26+)
lets MPSGraph take reduced-precision conv shortcuts: **FP16 winograd-transform
intermediates** and **FP32→FP19/TF32 operand narrowing**. The documented default
is `.none` (full precision), but the compiler "could use these paths … not
guaranteed", so the autotuner may pick a winograd-FP16 path for the 7×7 convs
unless explicitly forbidden — and **FP16 winograd intermediates overflow exactly
like the inf→nan signature** we see.

**Probe (shipped A/B knob).** `ChessNetwork(reducedPrecisionFastMathRaw:)` /
`ChessTrainer(reducedPrecisionFastMathRaw:)` force `.none` on every compile
descriptor. Because the documented default is already `.none`, this is a
force/verify lever, not a behavior change. The trainer logs
`[EXEC] reducedPrecisionFastMath default=… override=…` so a run records what
MPSGraph actually defaulted to. If forcing `.none` keeps the failing cells finite
where the canonical cells go NaN, a reduced-precision conv path is implicated.

---

## 4. ANE rejects fp32 intermediates — NOISE (most likely benign), not excluded

**What.** Throughout bf16 runs the console spews, many times per step:

```
Error: ANE cannot handle intermediate tensor type fp32
Failed to create unit plist.
```

The Apple Neural Engine is fp16-only, so it correctly refuses the fp32
intermediates the bf16 mixed-precision path uniquely carries (fp32 masters,
gradient-widening casts): MPSGraph attempts to partition onto the ANE, the ANE
rejects the fp32 work, and it should fall back to GPU. That makes this **most
likely benign fallback noise, not the cause**.

**Not fully excluded.** The message is about an *intermediate*, and the §1 stomp
is about the fp32 master / cast-temp intermediate, so a bad ANE↔GPU
partition/fallback handoff that the §1 split happens to move off the offending
boundary remains possible. The split fixes the stomp without disabling the ANE,
so the two are not distinguished. **An ANE-disable test would settle it.**

---

## 5. fp16 training diverges immediately — CONFIRMED (worse than bf16)

**What.** With `ComputeDataType.float16` (added 2026-06-17), a real fp16
`trainStep` diverges to a **NaN gradient on the very first step at every batch**
(batch 1 and batch 64 both), which is *worse* than bf16 — bf16's batch-1 cell is
finite and only the batch-64 multi-step accumulation goes NaN.

**Fingerprint.** The forward CE components come back finite (`policy` ≈ ln(4864)
for fresh random weights, `value` finite) while the **aggregate loss and gradient
norm are NaN**. So the overflow is in the fp16 backward and/or an auxiliary loss
term (entropy bonus / illegal-mass penalty), not the CE reductions (which the
trainer already accumulates in fp32). fp16's exponent range
(max ≈ 65504, min normal ≈ 6.1e-5) is far narrower than bf16/fp32, which share
fp32's exponent range — bf16 survives the same arithmetic that fp16 overflows.

**Status.** fp16 **inference** is sound and tested (finite, well-formed forward
pass; bit-exact safetensors round-trip incl. the forward-verify gate). fp16
**training** is not viable as-is; it would need **loss scaling** and/or **fp32
computation of the loss + auxiliary terms**. Captured by the two known-failing
`FP16ComputePathTests` trainStep cells.

---

## What is NOT broken (ruled out / clean)

- **fp32 end to end** — the `.float32` path (single weight write, no master, no
  cast temp) is clean across `graph.run` and both executable optimization levels.
- **The `cast` op itself** — a standalone fp32↔bf16 / fp32↔fp16 `cast` is
  bit-exact on GPU and matches the CPU helpers (see `BF16CastEquivalenceTests`,
  `FP16ConversionTests`); the half↔fp32 widen/narrow conversions round-trip
  exactly. The breakage is in *scheduling/liveness around fused multi-writes*,
  not in any single arithmetic op.
- **Stale Metal toolchain** — ruled out (reinstall did not help; `xcode-select`
  already on the beta).

---

## Workarounds shipped (and their toggles)

| Workaround | Where | Default | Purpose |
|---|---|---|---|
| Split trainable working-sync into its own pass | `ChessTrainer.splitWorkingWeightSync` | **on** | The §1 fix — eliminates the fused dual-write stomp |
| Config-D: store fp32, cast to bf16 in forward (no masters) | `ChessNetwork.bf16CastInForward` / `--bf16-cast-in-forward` | off (experimental) | Alternative to the master/working path; sidesteps the dual-write differently |
| Opt out of auto layout conversion | `disableAutoLayoutConversion()` | off | §2 A/B probe |
| Force `reducedPrecisionFastMath = .none` | `reducedPrecisionFastMathRaw` | off (leave default) | §3 A/B probe |
| Continue training in **fp32** on resume | Auto-resume sheet `loadAsFloat32` (→ `forceFloat32` load path) | offered | Escape hatch: bf16 is unstable on this beta; fp32 weights load losslessly |

App-level guardrails that interact with the symptom: on detected divergence the
session is **suspended (banner kept), arenas gated, and the poisoned NaN-weights
session is NOT persisted** (commit `e8204c8`) — a NaN net can't keep stepping and
must not overwrite a good checkpoint.

---

## If reporting to Apple / re-testing on a new toolchain

The minimal repro is §1: a bf16 mixed-precision optimizer tail with a fused
fp32-master + bf16-working dual-write into one compiled `MPSGraphExecutable`,
reading back NaN working weights / `1.0`-sentinel masters within two steps, on
the rank-2 SE tensors first. The decisive next experiment for the §4 question is
an ANE-disable run (the §1 split fixes the stomp without disabling the ANE, so
they're currently confounded). When a future toolchain ships, flip
`splitWorkingWeightSync` back to false and run the `MacOS27NaNIsolationTests`
bf16 × batch64 cells; if they stay finite, the underlying stomp is fixed.
