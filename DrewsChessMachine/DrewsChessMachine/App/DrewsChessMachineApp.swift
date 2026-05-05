import AppKit
import SwiftUI

@main
struct DrewsChessMachineApp: App {
    /// AppDelegate adaptor wires SwiftUI's App lifecycle to AppKit's
    /// `NSApplicationDelegate` so we can install signal handlers,
    /// disable sudden termination, and route AppKit termination
    /// requests through the early-stop flush path.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    /// Single shared command hub that bridges the menu bar commands
    /// to `ContentView`'s state and action functions. Owned here at
    /// the `App` level so the `.commands` DSL below and the
    /// `ContentView` in `WindowGroup` see the same instance.
    @State private var commandHub = AppCommandHub()

    /// View > Show Training Graphs preference. Persisted across launches
    /// via UserDefaults. Independent of `chartCoordinator.isActive`
    /// (which only reflects whether chart data is being collected) so
    /// the user can hide the lower pane during training to reclaim
    /// vertical space without stopping data capture.
    @AppStorage("showTrainingGraphs") private var showTrainingGraphs: Bool = true

    /// View > Collect Chart Data preference. Persisted across launches
    /// via UserDefaults. When `false`, every chart-collection entry
    /// point on `ChartCoordinator` becomes a no-op AND the underlying
    /// ring buffers stay at zero element storage (lazy first-block
    /// allocation in `ChartSampleRing`). Intended for clean perf-
    /// isolation runs where chart bookkeeping must not perturb the
    /// training hot path.
    @AppStorage("chartCollectionEnabled") private var chartCollectionEnabled: Bool = true

    /// True iff the process was launched with `--train` on the
    /// command line. When set, `ContentView` skips the Resume-from-
    /// Autosave sheet on first appearance and instead chains
    /// Build Network → Play-and-Train → switch to Candidate Test
    /// as an automated sequence. Captured once in `init` from
    /// `CommandLine.arguments` so the value is stable for the life
    /// of the launch.
    private let autoTrainOnLaunch: Bool

    /// Parsed payload of `--parameters <file>`. Nil when the flag
    /// wasn't passed, or when the file was missing / malformed
    /// (in which case the load error is surfaced to the session
    /// log and the app continues with UI defaults rather than
    /// silently running with unknown values). Fields inside are
    /// individually optional — a partial file only overrides the
    /// keys it names.
    private let cliConfig: CliTrainingConfig?

    /// Destination URL for `--output <file>`. When set, the runtime
    /// spins up a `CliTrainingRecorder`, wires arena/stats/probe
    /// events into it, and writes a JSON snapshot at
    /// `training_time_limit` expiry before terminating the process.
    /// Nil = no snapshot.
    private let cliOutputURL: URL?

    init() {
        // Parse launch-time CLI flags before any logging so the
        // [APP] banner can record whether auto-train mode is on
        // for this launch. `CommandLine.arguments[0]` is the
        // executable path; we only care about the tail.
        //
        // All flags are positional-free: each known flag is
        // located by name, which lets the user pass them in any
        // order (e.g. `--train --output X --parameters Y` and
        // `--parameters Y --output X --train` are equivalent).
        // After consuming every recognized flag + its value, any
        // unrecognized leftover is a hard error — we print usage
        // and `_exit(2)` rather than silently running with the
        // stray arg ignored, because "it looked like it was
        // accepted" is the exact failure mode that masks typos
        // in scripted runs.
        //
        // Skip strict CLI parsing when running under XCTest:
        // the xctest runner injects its own arguments (e.g.
        // `-XCTest`, test-bundle paths) which the strict parser
        // would reject, tearing down the whole test target.
        // `XCTestConfigurationFilePath` is set by xctest and
        // is the canonical "we are in a test run" signal.
        let isRunningUnderXCTest = ProcessInfo.processInfo
            .environment["XCTestConfigurationFilePath"] != nil
        let rawArgs: [String] = isRunningUnderXCTest
            ? []
            : Array(CommandLine.arguments.dropFirst())

        // Pre-flight: handle the two defaults-emitter flags BEFORE any
        // SwiftUI / AppKit / Metal initialization. They're sub-second
        // exits and never touch the singleton; the only user-visible
        // effect is bytes on stdout (and stderr, for the descriptions
        // variant) followed by `_exit(0)`.
        Self.handleDefaultsFlagsIfPresent(rawArgs: rawArgs)

        // Known flags.
        let booleanFlags: Set<String> = ["--train"]
        let valueFlags: Set<String> = ["--parameters", "--output", "--training-time-limit"]

        // Indices of rawArgs that were consumed by a known flag.
        // Anything NOT in this set after parsing is unknown and
        // triggers the usage-error path below.
        var consumedIndices = Set<Int>()
        var errors: [String] = []

        // Extract the value that follows a value-flag. Validates
        // that (a) the flag appears only once, (b) a value exists
        // at idx+1, and (c) the value doesn't itself start with
        // `--` (which would indicate the user forgot to supply a
        // value before the next flag). Marks both the flag and
        // its value as consumed on success; on failure marks only
        // the flag so the value token falls through the
        // unknown-argument scan below if it's actually a stray.
        func takeValue(for flag: String) -> String? {
            let indices = rawArgs.indices.filter { rawArgs[$0] == flag }
            guard let idx = indices.first else { return nil }
            consumedIndices.insert(idx)
            if indices.count > 1 {
                errors.append("\(flag) specified \(indices.count) times; only one allowed")
                for extra in indices.dropFirst() { consumedIndices.insert(extra) }
            }
            let valueIdx = idx + 1
            guard valueIdx < rawArgs.count else {
                errors.append("\(flag) requires a value but none was given")
                return nil
            }
            let value = rawArgs[valueIdx]
            if value.hasPrefix("--") {
                errors.append("\(flag) requires a value but got flag '\(value)' instead")
                return nil
            }
            consumedIndices.insert(valueIdx)
            return value
        }

        // `--train` — boolean flag, no value. Also reject if it
        // appears more than once so a scripted invocation with a
        // duplicate flag fails loudly.
        let trainIndices = rawArgs.indices.filter { rawArgs[$0] == "--train" }
        self.autoTrainOnLaunch = !trainIndices.isEmpty
        for idx in trainIndices { consumedIndices.insert(idx) }
        if trainIndices.count > 1 {
            errors.append("--train specified \(trainIndices.count) times; only one allowed")
        }

        // `--parameters <path>` — optional hyperparameter override
        // file. Values that the JSON doesn't name fall back to
        // the normal UI defaults. File-not-found and malformed
        // JSON are hard errors — a scripted run with a typo in
        // the path or a mid-edit JSON file is exactly the case
        // where silently running with defaults would be worst.
        var parsedConfig: CliTrainingConfig? = nil
        if let path = takeValue(for: "--parameters") {
            let expanded = (path as NSString).expandingTildeInPath
            let url = URL(fileURLWithPath: expanded)
            do {
                parsedConfig = try CliTrainingConfig.load(from: url)
            } catch {
                errors.append("--parameters: failed to load \(url.path): \(error.localizedDescription)")
            }
        }

        // `--training-time-limit <seconds>` — standalone CLI flag
        // for the single most commonly-scripted knob. When both
        // `--parameters`' `training_time_limit` and this flag are
        // present, the CLI flag wins.
        var trainingTimeLimitCliOverride: Double? = nil
        if let raw = takeValue(for: "--training-time-limit") {
            if let parsed = Double(raw), parsed > 0, parsed.isFinite {
                trainingTimeLimitCliOverride = parsed
            } else {
                errors.append("--training-time-limit value '\(raw)' is not a positive finite number")
            }
        }
        if let override = trainingTimeLimitCliOverride {
            if parsedConfig == nil {
                parsedConfig = CliTrainingConfig(
                    trainingParameters: [:],
                    trainingTimeLimitSec: override
                )
            } else {
                parsedConfig?.trainingTimeLimitSec = override
            }
        }
        self.cliConfig = parsedConfig

        // `--output <path>` — destination for the final JSON
        // snapshot. Stored as a URL so later code doesn't have to
        // re-resolve the tilde or the current working directory.
        var parsedOutputURL: URL? = nil
        if let path = takeValue(for: "--output") {
            let expanded = (path as NSString).expandingTildeInPath
            parsedOutputURL = URL(fileURLWithPath: expanded)
        }
        self.cliOutputURL = parsedOutputURL

        // Unknown-argument scan. Anything that wasn't consumed
        // by a known flag above is rejected — including stray
        // positional args, typos like `--out` instead of
        // `--output`, and unsupported flags such as `--help`.
        // (A dedicated `--help` path could be added later; for
        // now the error surfaces the same usage banner anyway.)
        var unknown: [String] = []
        for (i, arg) in rawArgs.enumerated() where !consumedIndices.contains(i) {
            unknown.append("'\(arg)'")
        }
        if !unknown.isEmpty {
            errors.append("unrecognized argument(s): \(unknown.joined(separator: ", "))")
        }
        _ = booleanFlags; _ = valueFlags  // kept for documentation; helper already enforces

        // If anything was wrong, print the error(s) + usage to
        // stderr and terminate. Use `_exit(2)` rather than `exit(2)`
        // so the app bails before SwiftUI / AppKit / SessionLogger
        // have done any setup — the user is clearly not running a
        // valid session here, and a half-initialized window
        // appearing briefly would be confusing.
        if !errors.isEmpty {
            let usage = """
            Usage: DrewsChessMachine [--train] [--parameters <file>] [--output <file>] [--training-time-limit <seconds>]

            Options (any order):
              --train                         Headless mode: auto build fresh network, start Play-and-Train,
                                              switch to Candidate Test view.
              --parameters <file>             JSON file of hyperparameter overrides. Unknown keys are accepted
                                              by the JSON decoder only if they match a known field.
              --output <file>                 Write JSON snapshot to <file> on training_time_limit expiry.
                                              Without this flag, the snapshot goes to stdout.
              --training-time-limit <seconds> Seconds of Play-and-Train before the JSON snapshot is written
                                              and the process exits. Overrides any value in --parameters.
                                              Only honored under --train.
            """
            for err in errors {
                let line = "DrewsChessMachine: error: \(err)\n"
                FileHandle.standardError.write(Data(line.utf8))
            }
            FileHandle.standardError.write(Data("\(usage)\n".utf8))
            Darwin._exit(2)
        }

        // Start the session logger before any view work so every event
        // from this launch — button taps, arena results, periodic
        // stats — lands in a single `dcm_log_yyyymmdd-HHMMSS.txt`
        // file under the app's Library/Logs directory.
        SessionLogger.shared.start()
        let dirtyMarker = BuildInfo.gitDirty ? "*" : ""
        let archHashHex = String(format: "0x%08x", ModelCheckpointFile.currentArchHash)
        let autoTrainMarker = autoTrainOnLaunch ? " autoTrain=on" : ""
        SessionLogger.shared.log(
            "[APP] launched build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirtyMarker) branch=\(BuildInfo.gitBranch) date=\(BuildInfo.buildDate) timestamp=\(BuildInfo.buildTimestamp) arch_hash=\(archHashHex) inputPlanes=\(ChessNetwork.inputPlanes) policySize=\(ChessNetwork.policySize)\(autoTrainMarker)"
        )
        if let path = SessionLogger.shared.activeLogPath {
            SessionLogger.shared.log("[APP] session log: \(path)")
            print("[APP] session log: \(path)")
        } else {
            print("[APP] session log: (failed to open)")
        }
        if autoTrainOnLaunch {
            SessionLogger.shared.log("[APP] --train flag detected; will build fresh network and start Play-and-Train on first appear")
        }
        // Reflect the chart-collection gate at launch so a perf
        // isolation run is unambiguously identifiable in the session
        // log. Reads UserDefaults directly here (the @AppStorage on
        // `self` isn't usable from `init`).
        let chartsEnabledAtLaunch = UserDefaults.standard.object(forKey: "chartCollectionEnabled") as? Bool ?? true
        if !chartsEnabledAtLaunch {
            SessionLogger.shared.log("[APP] chart data collection: DISABLED (View > Collect Chart Data)")
        }
        if let override = trainingTimeLimitCliOverride {
            SessionLogger.shared.log("[APP] --training-time-limit=\(override)s (overrides any value in --parameters)")
        }
        if let cfg = cliConfig {
            SessionLogger.shared.log("[APP] --parameters overrides: \(cfg.summaryString())")
        }
        if let outURL = cliOutputURL {
            SessionLogger.shared.log("[APP] --output destination: \(outURL.path)")
        }

        // Sweep away `.tmp` staging debris from a save that was
        // interrupted mid-flight by a prior process kill, kernel
        // panic, or power loss. Runs once at launch, before any save
        // or load can race with the cleanup.
        CheckpointPaths.cleanupOrphans()
    }

    var body: some Scene {
        WindowGroup {
            ContentView(
                commandHub: commandHub,
                autoTrainOnLaunch: autoTrainOnLaunch,
                cliConfig: cliConfig,
                cliOutputURL: cliOutputURL,
                showTrainingGraphs: showTrainingGraphs,
                chartCollectionEnabled: chartCollectionEnabled
            )
        }
        .commands {
            // File menu additions — Save / Load / reveal-in-Finder.
            // Placed after the standard "New" slot so they appear at
            // the top of the File menu alongside the other
            // file-scope operations.
            CommandGroup(after: .newItem) {
                Divider()
                Button("Save Session") { commandHub.saveSession() }
                    .disabled(
                        !commandHub.realTraining
                        || commandHub.isArenaRunning
                        || commandHub.checkpointSaveInFlight
                    )
                Button("Save Champion") { commandHub.saveChampion() }
                    .disabled(
                        !commandHub.networkReady
                        || commandHub.checkpointSaveInFlight
                        || commandHub.isArenaRunning
                        || (commandHub.isBusy && !commandHub.realTraining)
                    )
                Divider()
                Button("Load Session…") { commandHub.loadSession() }
                    .disabled(
                        commandHub.realTraining
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.sweepRunning
                        || commandHub.gameIsPlaying
                        || commandHub.isBuilding
                        || commandHub.checkpointSaveInFlight
                    )
                Button("Load Model…") { commandHub.loadModel() }
                    .disabled(
                        commandHub.realTraining
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.sweepRunning
                        || commandHub.gameIsPlaying
                        || commandHub.isBuilding
                        || commandHub.checkpointSaveInFlight
                    )
                Divider()
                Button("Load Parameters…") { commandHub.loadParameters() }
                    .disabled(
                        commandHub.realTraining
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.sweepRunning
                        || commandHub.gameIsPlaying
                        || commandHub.isBuilding
                        || commandHub.checkpointSaveInFlight
                    )
                Button("Save Parameters…") { commandHub.saveParameters() }
                Divider()
                Button("Resume Training from Autosave") {
                    commandHub.resumeFromAutosave()
                }
                .disabled(!commandHub.canResumeFromAutosave)
                Divider()
                Button("Open Data Folder in Finder") { commandHub.revealSaves() }
            }

            // View menu additions — zoom in/out and auto-zoom for
            // the training chart grid. Merges into the system View
            // menu (after the Show Sidebar slot) rather than
            // declaring a new top-level menu.
            CommandGroup(after: .sidebar) {
                Divider()
                Toggle("Show Training Graphs", isOn: $showTrainingGraphs)
                Toggle("Collect Chart Data", isOn: $chartCollectionEnabled)
                Divider()
                Button("Zoom In Charts") { commandHub.chartZoomIn() }
                    .keyboardShortcut("=", modifiers: .command)
                    .disabled(!commandHub.chartZoomInAvailable)
                Button("Zoom Out Charts") { commandHub.chartZoomOut() }
                    .keyboardShortcut("-", modifiers: .command)
                    .disabled(!commandHub.chartZoomOutAvailable)
                Button("Auto Zoom Charts") { commandHub.chartZoomEnableAuto() }
                    .disabled(!commandHub.chartZoomAutoAvailable)
            }

            // Train menu — the primary training-session lifecycle
            // plus the arena-stage controls. SwiftUI places
            // `CommandMenu` entries before Window; on a standard
            // macOS menu bar we get: File Edit View Train Debug
            // Window Help. "Debug between Window and Help" isn't
            // reachable via `CommandMenu`, so Debug lands adjacent
            // to Train (before Window) as the closest SwiftUI
            // approximation.
            CommandMenu("Train") {
                Button("Build Network") { commandHub.buildNetwork() }
                    .disabled(commandHub.isBusy || commandHub.networkReady)
                Button(commandHub.pendingLoadedSessionExists ? "Continue Training" : "Play and Train") {
                    commandHub.startRealTraining()
                }
                .disabled(
                    commandHub.isBusy
                    || !commandHub.networkReady
                    || commandHub.realTraining
                    || commandHub.continuousPlay
                    || commandHub.continuousTraining
                    || commandHub.sweepRunning
                )
                Divider()
                Button("Stop") { commandHub.stopAnyContinuous() }
                    .keyboardShortcut(.escape, modifiers: [])
                    .disabled(
                        !(commandHub.continuousPlay
                          || commandHub.continuousTraining
                          || commandHub.sweepRunning
                          || commandHub.realTraining)
                    )
                Divider()
                Button("Run Arena") { commandHub.runArena() }
                    .disabled(!commandHub.realTraining || commandHub.isArenaRunning)
                Button("Abort Arena") { commandHub.abortArena() }
                    .disabled(!commandHub.realTraining || !commandHub.isArenaRunning)
                Button("Promote Trainee") { commandHub.promoteCandidate() }
                    .disabled(!commandHub.realTraining || !commandHub.isArenaRunning)
            }

            CommandMenu("Debug") {
                Button("Run Forward Pass") { commandHub.runForwardPass() }
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(commandHub.isBusy || !commandHub.networkReady)
                Divider()
                Button("Play Game") { commandHub.playSingleGame() }
                    .disabled(commandHub.isBusy || !commandHub.networkReady)
                Button("Play Continuous") { commandHub.startContinuousPlay() }
                    .disabled(
                        commandHub.isBusy
                        || !commandHub.networkReady
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.realTraining
                    )
                Divider()
                Button("Train Once") { commandHub.trainOnce() }
                    .disabled(commandHub.isBusy || !commandHub.networkReady)
                Button("Train Continuous") { commandHub.startContinuousTraining() }
                    .disabled(
                        commandHub.isBusy
                        || !commandHub.networkReady
                        || commandHub.continuousTraining
                        || commandHub.continuousPlay
                        || commandHub.sweepRunning
                        || commandHub.realTraining
                    )
                Divider()
                Button("Sweep Batch Sizes") { commandHub.startSweep() }
                    .disabled(
                        commandHub.isBusy
                        || !commandHub.networkReady
                        || commandHub.sweepRunning
                        || commandHub.continuousPlay
                        || commandHub.continuousTraining
                        || commandHub.realTraining
                    )
                Divider()
                Button("Run Engine Diagnostics") { commandHub.runEngineDiagnostics() }
                    .disabled(commandHub.isBusy)
                Button("Run Policy-Conditioning Probe") { commandHub.runPolicyConditioningDiagnostic() }
                    .disabled(commandHub.isBusy)
                Divider()
                Button("Open Session Log") {
                    if let path = SessionLogger.shared.activeLogPath {
                        NSWorkspace.shared.open(URL(fileURLWithPath: path))
                    }
                }
                .disabled(SessionLogger.shared.activeLogPath == nil)
                Button("Analyze Log") {
                    LogAnalysisLauncher.openWindow()
                }
                .disabled(SessionLogger.shared.activeLogPath == nil)
            }

            // Tools menu — Debug builds only. Houses one-shot
            // maintenance utilities like the Arena History
            // log-recovery scan that aren't part of the day-to-day
            // training loop.
            CommandMenu("Tools") {
                Button("Recover Arena History from Logs") {
                    commandHub.recoverArenaHistoryFromLogs()
                }
                .disabled(commandHub.arenaRecoveryInProgress)
            }
        }
    }

    // MARK: - Defaults-emitter pre-flight (--show-default-parameters / --create-parameters-file)

    /// Inspects `rawArgs` for the two defaults-emitter flags.
    /// If either is present, validates the allowed flag combinations,
    /// performs the action, and exits the process. Sub-second; never
    /// touches the singleton (the registry is `nonisolated` and walks
    /// definition defaults directly).
    ///
    /// Allowed combinations:
    /// - `--show-default-parameters` alone (no other flags, including no `--force`)
    /// - `--create-parameters-file` alone (default path is `./parameters.json`)
    /// - `--create-parameters-file --force`
    /// - `--create-parameters-file <path>` (positional path argument)
    /// - `--create-parameters-file <path> --force`
    ///
    /// Anything else with these flags is a usage error → exit 2.
    private static func handleDefaultsFlagsIfPresent(rawArgs: [String]) {
        let showFlag = "--show-default-parameters"
        let createFlag = "--create-parameters-file"
        let forceFlag = "--force"

        let hasShow = rawArgs.contains(showFlag)
        let hasCreate = rawArgs.contains(createFlag)

        if !hasShow && !hasCreate {
            return
        }

        // Mutual exclusion.
        if hasShow && hasCreate {
            FileHandle.standardError.write(Data("error: \(showFlag) and \(createFlag) are mutually exclusive\n".utf8))
            Darwin.exit(3)
        }

        if hasShow {
            // --show-default-parameters: must appear alone.
            let allowed: Set<String> = [showFlag]
            if let bad = rawArgs.first(where: { !allowed.contains($0) }) {
                FileHandle.standardError.write(Data("error: \(showFlag) does not accept '\(bad)' (must appear alone)\n".utf8))
                Darwin.exit(4)
            }
            runShowDefaultParametersAndExit()
        }

        // --create-parameters-file: --force allowed; one positional path allowed.
        // Any OTHER flag-shaped arg (anything starting with `--` that isn't
        // --create-parameters-file or --force) is a hard error.
        let force = rawArgs.contains(forceFlag)
        let allowed: Set<String> = [createFlag, forceFlag]
        if let badFlag = rawArgs.first(where: { $0.hasPrefix("--") && !allowed.contains($0) }) {
            FileHandle.standardError.write(Data("error: \(createFlag) does not accept '\(badFlag)' (only --force is allowed alongside)\n".utf8))
            Darwin.exit(5)
        }
        let positional = rawArgs.filter { !allowed.contains($0) }
        if positional.count > 1 {
            FileHandle.standardError.write(Data("error: \(createFlag) accepts at most one path argument; got \(positional.count)\n".utf8))
            Darwin.exit(6)
        }
        let path = positional.first ?? "./parameters.json"
        runCreateParametersFileAndExit(path: path, force: force)
    }

    private static func runShowDefaultParametersAndExit() -> Never {
        do {
            let json = try TrainingParameters.defaultsJSON()
            FileHandle.standardOutput.write(json)
            FileHandle.standardOutput.write(Data("\n".utf8))
            for line in TrainingParameters.defaultsDescriptionLines() {
                FileHandle.standardError.write(Data("\(line)\n".utf8))
            }
            Darwin.exit(0)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            Darwin.exit(1)
        }
    }

    private static func runCreateParametersFileAndExit(path: String, force: Bool) -> Never {
        let expanded = (path as NSString).expandingTildeInPath
        let jsonURL = URL(fileURLWithPath: expanded)
        let mdURL = jsonURL.deletingPathExtension().appendingPathExtension("md")

        let fm = FileManager.default
        if fm.fileExists(atPath: jsonURL.path) && !force {
            FileHandle.standardError.write(Data("error: \(jsonURL.path) already exists; pass --force to overwrite\n".utf8))
            Darwin.exit(7)
        }

        do {
            let jsonData = try TrainingParameters.defaultsJSON()
            let mdData = Data(TrainingParameters.defaultsMarkdown().utf8)

            // Atomic write via temp + rename.
            let jsonTmp = jsonURL.appendingPathExtension("tmp")
            let mdTmp = mdURL.appendingPathExtension("tmp")
            do {
                try jsonData.write(to: jsonTmp, options: [.atomic])
                try mdData.write(to: mdTmp, options: [.atomic])
            } catch {
                try? fm.removeItem(at: jsonTmp)
                try? fm.removeItem(at: mdTmp)
                throw error
            }
            // Both temp files written successfully; promote both. If
            // either rename fails, attempt to clean up the other so we
            // don't leave a half-applied state on disk.
            do {
                if fm.fileExists(atPath: jsonURL.path) {
                    try fm.removeItem(at: jsonURL)
                }
                try fm.moveItem(at: jsonTmp, to: jsonURL)
            } catch {
                try? fm.removeItem(at: jsonTmp)
                try? fm.removeItem(at: mdTmp)
                throw error
            }
            do {
                if fm.fileExists(atPath: mdURL.path) {
                    try fm.removeItem(at: mdURL)
                }
                try fm.moveItem(at: mdTmp, to: mdURL)
            } catch {
                try? fm.removeItem(at: mdTmp)
                // jsonURL is already in place; per the plan, parameters.md
                // is overwritten freely "only when parameters.json is also
                // being written". The json write succeeded; surfacing the
                // md failure as a non-zero exit is the conservative choice.
                throw error
            }

            FileHandle.standardOutput.write(Data("wrote: \(jsonURL.path)\n".utf8))
            FileHandle.standardOutput.write(Data("wrote: \(mdURL.path)\n".utf8))
            Darwin.exit(0)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            Darwin.exit(1)
        }
    }
}
