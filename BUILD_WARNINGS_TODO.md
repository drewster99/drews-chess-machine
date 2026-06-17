# Build Warnings — cleanup TODO

Captured 2026-06-14 during the bf16 split-fix session, deferred to next session.

All of these are **pre-existing** — none were introduced by the `splitWorkingWeightSync` fix (that built clean). The concurrency/module ones are likely amplified by the **Xcode 27 / macOS 27 beta** toolchain's stricter checks.

- [ ] **`App/UpperContentView/UpperContentView.swift:813`** — `Cannot use generic class 'Autoconnect' / enum 'Publishers' in a property declaration member of a type not marked '@_implementationOnly'; 'Combine' was not imported by this file.`
  Likely a `Timer.publish(…).autoconnect()` publisher used without importing Combine.
  **Fix:** add `import Combine` to that file.

- [ ] **`Training/ChessTrainer.swift:5395–5404`** — non-Sendable captures in a `@Sendable` closure: `nda` (`MPSNDArray`), `ph` (`MPSGraphTensor`), `td` (`MPSGraphTensorData`), `rateVar` (`MPSGraphTensor`), `assign` (`MPSGraphOperation`).
  **Fix:** the compiler suggests `@preconcurrency import MetalPerformanceShaders` / `MetalPerformanceShadersGraph` (lines 5–6) to downgrade these to warnings, or restructure the closure so the non-Sendable values aren't captured across the isolation boundary.

- [ ] **`Training/ChessTrainer.swift:5656`** — `Initialization of immutable value 'dtype' was never used` in `feedsForBatch`.
  `let dtype = ChessNetwork.mpsDataType(for: arch)` is computed but unused — all four feed ND arrays are hardcoded `.float32`.
  **Fix:** delete the unused `let dtype`.

- [ ] **`App/UpperContentView/SessionPickerModel.swift:108` and `:116`** — `'weak' ownership of capture 'self' differs from implicitly-captured strong reference in outer scope.`
  **Fix:** capture `self` consistently — make the outer capture `weak` too, or capture strongly throughout.

## Related, also pending before commit
- [ ] Remove diagnostic scaffolding from `DrewsChessMachineTests/MacOS27NaNIsolationTests.swift`: `DISABLED_bf16_fingerprintAliasingProbe` (GPU-hanging, disabled), and the file-writing pinpoint/single-block/standalone/split probes that dump to `~/Library/Logs/DrewsChessMachine/cast_probe_*.txt`. Keep the bf16/fp32 finiteness matrix cells and a slim split A/B as regression tests.
