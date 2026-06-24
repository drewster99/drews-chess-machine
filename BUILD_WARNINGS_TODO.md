# Build Warnings — cleanup TODO

Captured 2026-06-14 during the bf16 split-fix session, deferred to next session.

All of these are **pre-existing** — none were introduced by the `splitWorkingWeightSync` fix (that built clean). The concurrency/module ones are likely amplified by the **Xcode 27 / macOS 27 beta** toolchain's stricter checks.

> **Status (2026-06-23 audit):** items (1), (3), and (4) are RESOLVED in the source
> (verified by grep — see per-item notes). Item (2) (non-Sendable captures) and the
> test-scaffolding removal could NOT be confirmed from source alone — they need a build
> to verify and remain open.

- [x] **RESOLVED — `App/UpperContentView/UpperContentView.swift`** — `Cannot use generic class 'Autoconnect' / enum 'Publishers' in a property declaration member of a type not marked '@_implementationOnly'; 'Combine' was not imported by this file.`
  Likely a `Timer.publish(…).autoconnect()` publisher used without importing Combine.
  **Fixed:** `import Combine` is now present at the top of the file (line 2).

- [ ] **OPEN (needs build to verify) — `Training/ChessTrainer.swift`** — non-Sendable captures in a `@Sendable` closure: `nda` (`MPSNDArray`), `ph` (`MPSGraphTensor`), `td` (`MPSGraphTensorData`), `rateVar` (`MPSGraphTensor`), `assign` (`MPSGraphOperation`).
  **Fix:** the compiler suggests `@preconcurrency import MetalPerformanceShaders` / `MetalPerformanceShadersGraph` (lines 5–6) to downgrade these to warnings, or restructure the closure so the non-Sendable values aren't captured across the isolation boundary.
  *(2026-06-23: could not be confirmed fixed from source — requires a build.)*

- [x] **RESOLVED — `Training/ChessTrainer.swift`** — `Initialization of immutable value 'dtype' was never used` in `feedsForBatch`.
  `let dtype = ChessNetwork.mpsDataType(for: arch)` was computed but unused — all four feed ND arrays are hardcoded `.float32`.
  **Fixed:** `feedsForBatch` no longer contains an unused `let dtype` binding.

- [x] **RESOLVED — `App/UpperContentView/SessionPickerModel.swift`** — `'weak' ownership of capture 'self' differs from implicitly-captured strong reference in outer scope.`
  **Fixed:** the scan closure now uses a consistent `[weak self]` outer capture (`indexQueue.async { [weak self] in … }`) with `guard let self` reentry inside the inner main-actor `Task`s, so there is no strong/weak mismatch.

## Related, also pending before commit
- [ ] **OPEN (needs build to verify)** — Remove diagnostic scaffolding from `DrewsChessMachineTests/MacOS27NaNIsolationTests.swift`: `DISABLED_bf16_fingerprintAliasingProbe` (GPU-hanging, disabled), and the file-writing pinpoint/single-block/standalone/split probes that dump to `~/Library/Logs/DrewsChessMachine/cast_probe_*.txt`. Keep the bf16/fp32 finiteness matrix cells and a slim split A/B as regression tests.
  *(2026-06-23: could not be confirmed done from source alone — requires a build/test review.)*
