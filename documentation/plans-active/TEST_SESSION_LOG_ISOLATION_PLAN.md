# Test session-log isolation plan

Status: implemented (see the commit that adds this file).

## Problem

The XCTest suite writes into the user's real session-log folder,
`~/Library/Logs/DrewsChessMachine/`. That folder is what the experiment
tooling, the dashboards and people read (49 GB across ~9,700 files on
2026-10-06), and test runs leave `dcm_log_*.txt` files in it full of fake
opponents (FitBot, cbob, ccarol) and synthetic training lines.

## Cause

- The test target is hosted by the app (`TEST_HOST` = `DrewsChessMachine.app`),
  so every test run launches the app's `init`, which calls
  `SessionLogger.shared.start()` and logs the `[APP] launched …` banner.
- `SessionLogger.shared` is hard-wired to `.userLibraryLogs`
  (`Logging/SessionLogger.swift`), so that banner, and every line any code
  under test logs through `SessionLogger.shared`, lands in a new
  `dcm_log_<stamp>.txt` in the real folder.
- The app already knows when it is an XCTest host (it skips strict CLI parsing,
  the orphan-staging sweep and the auto-resume sheet), but that knowledge is
  two inline `XCTestConfigurationFilePath` checks
  (`DrewsChessMachineApp.init`, `AutoResumeController.maybePresentSheet`) and
  never reached the logger.

Confirmed in the real folder: recent `dcm_log_*` files carry
`[RESUME] Skipping auto-resume sheet — running under XCTest`.

## Hypotheses for the right fix

1. **The logger's destination is chosen without knowing the process is a test
   host.** Fix it where the destination is chosen: `SessionLogger.shared`
   resolves its `Location` from one XCTest signal, and under XCTest the
   location is a per-test-run folder in the temporary directory.
2. **The app should not start the shared logger at all under XCTest.** Skip
   `SessionLogger.shared.start()` in `DrewsChessMachineApp.init` when hosted;
   `log` before `start` is already a no-op.

Assessment of 2: it covers only the one `start()` call in the app's `init`.
Any test that drives a CLI path (`CorpusReplayRunner`, `TrainVsUciRunner`, the
model CLIs, `UCIEngine`) or calls `start()` itself would still open a file in
the real folder, and the forensic suites that log diagnostics on purpose
(`MacOS27NaNIsolationTests`, `MPSGraphGradientSemanticsTests`) would silently
lose them. It patches a call site, not the decision.

**Chosen: 1.** The decision lives where the folder is chosen, so no path —
present or future — can reach the real folder from a test process.

## Design

- `Utils/XCTestHostDetection.swift`: `XCTestHostDetection.isRunningUnderXCTest`,
  the one place the process decides it is an XCTest host
  (`XCTestConfigurationFilePath` in the environment, which xctest sets in the
  hosting app). Evaluated once per process (`static let`). The two existing
  inline checks are replaced by it, so the CLI-parsing skip, the orphan-sweep
  skip, the auto-resume skip and the logger all read the same signal.
- `SessionLogger.Location` gains `.xcTestRunTemporaryDirectory` and
  `static func forProcess(runningUnderXCTest:)`, the one explicit mapping:
  `true` → `.xcTestRunTemporaryDirectory`, `false` → `.userLibraryLogs`.
  `SessionLogger.shared` is built from
  `forProcess(runningUnderXCTest: XCTestHostDetection.isRunningUnderXCTest)`.
  No fallback: if the temporary folder cannot be made, `start()` reports it on
  stderr like any other folder failure and the logger drops lines; it never
  falls back to the real folder.
- `.xcTestRunTemporaryDirectory` resolves to
  `$TMPDIR/DrewsChessMachine-XCTest-SessionLogs/run-<yyyyMMdd-HHmmss>-pid<pid>/`,
  computed once per process so a `start()` after `shutdown()` stays in the same
  folder. The parent is created with intermediate directories (shared by every
  test run); the per-run folder is created on first use.
- The folder is **left in the temporary directory**, not removed at teardown:
  - the app-hosted test process has no teardown hook that runs after the last
    line is written (the logger writes until the process exits);
  - the forensic suites log diagnostics there on purpose, and removing them
    would throw those away;
  - macOS clears unused `$TMPDIR` items on its own schedule.
  The app's existing `[APP] session log: <path>` stdout line names the file.
- The app and every CLI (`--replay-corpus`, `--train-vs-uci`, `--uci`, …) run
  without `XCTestConfigurationFilePath`, so they resolve `.userLibraryLogs` and
  log exactly as before. `.directory(URL)` (explicit folders for tests) is
  unchanged.
- Tests that read back log lines use injected sinks (`SessionParameterResume`,
  `BehaviorFingerprint` comparisons) or their own `SessionLogger(location:
  .directory(…))`; none reads `SessionLogger.shared`'s file, so none changes.

## Tests

- `SessionLoggerTestIsolationTests` (new):
  - **Regression (written first, red before the fix):** start
    `SessionLogger.shared`, log a line, read `activeLogPath`, and assert the
    file is not inside `~/Library/Logs/DrewsChessMachine` and is inside the
    process's temporary directory. Uses only API that existed before the fix,
    so it compiles and fails on the old code.
  - `XCTestHostDetection.isRunningUnderXCTest` is true in the test process.
  - `Location.forProcess(runningUnderXCTest: false) == .userLibraryLogs` (the
    app and CLI keep the real folder) and `forProcess(runningUnderXCTest: true)
    == .xcTestRunTemporaryDirectory`.
  - The shared logger's file is in `SessionLogger.xcTestRunLogsDirectory`.
- Existing `AutoResumeControllerTests.testMaybePresentSheetIsNoOpUnderXCTest`
  keeps covering the auto-resume skip through the shared detection.

## Validation

1. Red: run the regression test on the unfixed code; it fails, naming a path in
   `~/Library/Logs/DrewsChessMachine`.
2. Fix; the same test, unmodified, passes.
3. End to end: snapshot `~/Library/Logs/DrewsChessMachine` (name, size, mtime)
   before and after a targeted run of `LichessBotCasualFallbackTests`. Live
   training runs and other worktrees' test runs also write there, so every new
   or changed file is attributed by content: none may carry this worktree's
   branch in its `[APP] launched … branch=` line or the test's fake
   opponents. The training logs (`dcm_log_20261005-234434/234437`,
   `20261006-170000`, `20261006-181510`) are expected to grow and are only
   read. Nothing in `~/Library/Logs` is deleted or modified.
4. Build, targeted tests, full suite.
