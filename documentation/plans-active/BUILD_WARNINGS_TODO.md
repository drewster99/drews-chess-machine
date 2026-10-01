# Build Warnings — cleanup TODO

Captured 2026-06-14 during the bf16 split-fix session, deferred to next session.

All of these are **pre-existing** — none were introduced by the `splitWorkingWeightSync` fix (that built clean). The concurrency/module ones are likely amplified by the **Xcode 27 / macOS 27 beta** toolchain's stricter checks.

> **Status (2026-06-23 audit):** items (1), (3), and (4) are RESOLVED in the source
> (verified by grep — see per-item notes). Item (2) (non-Sendable captures) and the
> test-scaffolding removal could NOT be confirmed from source alone — they need a build
> to verify and remain open.
>
> **Status (2026-10-01 re-audit):** all four original items are resolved. Item (2) is
> now confirmed resolved from Xcode build logs (no new build was run for this audit):
> `ChessTrainer.swift` was recompiled in the 2026-09-30 11:18 build (Xcode 27.1 beta)
> and the 2026-10-01 01:38 build (Xcode 27.2 beta), and neither reported any
> non-Sendable-capture warning — only type-check-time warnings. The
> `MacOS27NaNIsolationTests` scaffolding removal is **still open** (confirmed by
> source). A new set of warnings from those builds is listed under
> **New warnings (2026-10-01)** below.

- [x] **RESOLVED — `App/UpperContentView/UpperContentView.swift`** — `Cannot use generic class 'Autoconnect' / enum 'Publishers' in a property declaration member of a type not marked '@_implementationOnly'; 'Combine' was not imported by this file.`
  Likely a `Timer.publish(…).autoconnect()` publisher used without importing Combine.
  **Fixed:** `import Combine` is now present at the top of the file (line 2).

- [x] **RESOLVED (confirmed from build logs 2026-10-01) — `Training/ChessTrainer.swift`** — non-Sendable captures in a `@Sendable` closure: `nda` (`MPSNDArray`), `ph` (`MPSGraphTensor`), `td` (`MPSGraphTensorData`), `rateVar` (`MPSGraphTensor`), `assign` (`MPSGraphOperation`).
  **Fix:** the compiler suggests `@preconcurrency import MetalPerformanceShaders` / `MetalPerformanceShadersGraph` (lines 5–6) to downgrade these to warnings, or restructure the closure so the non-Sendable values aren't captured across the isolation boundary.
  *(2026-06-23: could not be confirmed fixed from source — requires a build.)*
  *(2026-10-01: confirmed — `ChessTrainer.swift` compiled in the 2026-09-30 and
  2026-10-01 builds with no Sendable-capture warnings. The imports are still plain, not
  `@preconcurrency`, so the fix came from restructuring or from the toolchain.)*

- [x] **RESOLVED — `Training/ChessTrainer.swift`** — `Initialization of immutable value 'dtype' was never used` in `feedsForBatch`.
  `let dtype = ChessNetwork.mpsDataType(for: arch)` was computed but unused — all four feed ND arrays are hardcoded `.float32`.
  **Fixed:** `feedsForBatch` no longer contains an unused `let dtype` binding.

- [x] **RESOLVED — `App/UpperContentView/SessionPickerModel.swift`** — `'weak' ownership of capture 'self' differs from implicitly-captured strong reference in outer scope.`
  **Fixed:** the scan closure now uses a consistent `[weak self]` outer capture (`indexQueue.async { [weak self] in … }`) with `guard let self` reentry inside the inner main-actor `Task`s, so there is no strong/weak mismatch.

## Related, also pending before commit
- [ ] **OPEN (confirmed still present 2026-10-01)** — Remove diagnostic scaffolding from `DrewsChessMachineTests/MacOS27NaNIsolationTests.swift`: `DISABLED_bf16_fingerprintAliasingProbe` (GPU-hanging, disabled), and the file-writing pinpoint/single-block/standalone/split probes that dump to `~/Library/Logs/DrewsChessMachine/cast_probe_*.txt`. Keep the bf16/fp32 finiteness matrix cells and a slim split A/B as regression tests.
  *(2026-06-23: could not be confirmed done from source alone — requires a build/test review.)*
  *(2026-10-01: not done — `DISABLED_bf16_fingerprintAliasingProbe` is still in the file,
  as are the probes writing `cast_probe_report.txt`, `cast_probe_single_block.txt`,
  `cast_probe_standalone.txt`, `cast_probe_fingerprint.txt` and `cast_probe_split.txt`.
  Note the suite is now gated behind `DCM_RUN_SLOW_TESTS` (`SlowTestGate`); removing test
  code needs the owner's approval per the test rules.)*

## New warnings (2026-10-01)

Found in the 2026-09-30 / 2026-10-01 build logs (Xcode 27.1 / 27.2 beta). Line numbers
are as of commit `852737e`.

- [ ] **`LichessBot/App/LichessBotController.swift:985`** —
  `'weak' ownership of capture 'self' differs from implicitly-captured strong reference in outer scope`.
  In `updateChallengeOutcomeLog`: the `fileQueue.enqueue { … }` closure captures `self`
  strongly (implicitly), and the inner `Task { @MainActor [weak self] in self?.raiseAlarm(text) }`
  in its `catch` declares it weak. Same shape as the resolved `SessionPickerModel` item:
  make the outer capture `[weak self]` too (or capture what's needed explicitly).
- [ ] **`DrewsChessMachineTests/LichessBotChallengeQueueControllerTests.swift:149`** —
  `main actor-isolated class property 'finishedGameHold' can not be referenced from a nonisolated context; this is an error in the Swift 6 language mode`.
  Will become a build error under Swift 6 mode. Fix by isolating the test (or the
  reference) to the main actor.
- [ ] **`DrewsChessMachineTests/LichessBotChallengeQueueTests.swift`** — 8 ×
  `result of call to 'add(_:request:pendingUserIDs:makeID:)' is unused` (lines 53, 63,
  75, 95, 104, 113, 122, 135). Either assign to `_` where the result is genuinely
  irrelevant, assert on it, or mark `add` `@discardableResult` if ignoring it is normal use.
- [ ] **`DrewsChessMachineTests/MacOS27NaNIsolationTests.swift:491, 493`** —
  `variable 'taus' / 'materials' was never mutated; consider changing to 'let'`.
- [ ] **Type-check-time warnings** — the project sets
  `-warn-long-function-bodies=100 -warn-long-expression-type-checking=100` in
  `OTHER_SWIFT_FLAGS`, so every function body or expression over 100 ms produces a
  warning. A full build of the app target reports several dozen. The worst offender is
  `SessionController+Training.swift` `startRealTraining(mode:)` (~0.6–1.6 s); others
  over ~250 ms include `DrewsChessMachineApp.init()`, `ContentView.swift:155`,
  `SessionController+Arena.runArenaParallel`, `SweepCLI.runAndExit`, and
  `SessionController+Heartbeat.__processSnapshotTimerTick`, plus a cluster of
  ~230–280 ms `body` getters in `LichessBot/UI/`. These are compile-time costs, not
  correctness issues; fix by splitting large bodies or adding type annotations, or
  accept them deliberately.
