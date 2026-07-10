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
    @AppStorage("showPolicyChannelsPanel") private var showPolicyChannelsPanel: Bool = false
    @AppStorage("showEmitWindowStats") private var showEmitWindowStats: Bool = false

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

    /// True iff the process was launched with `--playchess`. A GUI-launch
    /// flag (like `--train`, not a pre-flight exit): the window opens and
    /// `UpperContentView` immediately starts a human-vs-network game
    /// against the model resolved from `playChessModelPath` (the same
    /// resolution `--uci` uses). Mutually exclusive with `--train`.
    private let autoPlayChessOnLaunch: Bool

    /// Value of `--model <path>` when `--playchess` is present. Nil ⇒ the
    /// most recently saved session's trainer file is used. Resolved via
    /// `UCIModelLoader.resolveModelURL` so it matches `--uci`'s behavior.
    private let playChessModelPath: String?

    /// Value of `--start-model <path>` when `--train` is present: the
    /// saved model loaded as the starting champion instead of a fresh
    /// random init. Nil ⇒ the classic fresh-build auto-train path.
    private let trainStartModelPath: String?

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

        // Pre-flight: handle the offline replay-buffer analyzer flag.
        // Same pattern as the defaults-emitter flags — exits the process
        // before any SwiftUI / Metal init. Reads the saved
        // `replay_buffer.bin` directly via `ReplayBuffer.restore` (no
        // network state needed), runs `ReplayBufferAnalyzer.run`, prints
        // JSON to stdout and a human-readable summary to stderr.
        Self.handleAnalyzeReplayBufferIfPresent(rawArgs: rawArgs)

        // Pre-flight: handle UCI mode. When `--uci` is present, hand
        // control to `UCIEngine.runAndExit` and never return — the
        // process behaves as a pure stdin/stdout chess engine for
        // cutechess and friends. Critically, this exits BEFORE the
        // SwiftUI `WindowGroup` is created, so no window appears and
        // the launch-time auto-resume countdown sheet (which would
        // otherwise re-start training automatically) never runs.
        Self.handleUciIfPresent(rawArgs: rawArgs)

        // Pre-flight: headless batch-size sweep (--sweep). Builds a fresh net,
        // runs the same ChessTrainer.runSweep the GUI button uses, prints the
        // throughput table + [SWEEP] logs, and exits — before SwiftUI / Metal
        // GUI init, so no window and no auto-resume sheet.
        Self.handleSweepIfPresent(rawArgs: rawArgs)

        // Pre-flight: headless depth (block-count) sweep (--arch-sweep). Builds a
        // fresh trainer at each requested block count, times build + per-step, and
        // streams JSONL — investigation tool for the deep-tower "hang". Exits
        // before SwiftUI / Metal GUI init.
        Self.handleArchSweepIfPresent(rawArgs: rawArgs)

        // Pre-flight: headless checkpoint probe (--probe-model). Loads saved
        // champions (a weight file, one .dcmsession, or a whole directory of
        // sessions) and runs the Lichess probe batteries against each,
        // emitting JSONL — retro-fills OOD measurements for runs that
        // predate the live probes. Exits before SwiftUI / Metal GUI init.
        Self.handleProbeModelIfPresent(rawArgs: rawArgs)

        // Pre-flight: headless fresh-net mint (--new-model). Builds an untrained
        // network from a --preset and writes it to a .safetensors for reuse as a
        // fixed --start-model across runs. Must run BEFORE the replay handler,
        // which also reads --preset. Exits before SwiftUI / Metal GUI init.
        Self.handleNewModelIfPresent(rawArgs: rawArgs)

        // Pre-flight: headless offline corpus-replay trainer (--replay-corpus).
        // Builds a fresh net + trainer, fills the replay buffer from recorded
        // games (no self-play, no arena, no promotion), runs a step-locked SGD
        // loop to a --training-step-limit / --epochs budget, and exits before
        // SwiftUI / Metal GUI init.
        Self.handleReplayCorpusIfPresent(rawArgs: rawArgs)

        // Pre-flight: headless train-vs-UCI trainer (--train-vs-uci). Plays
        // the live trainer network against a pool of external UCI engines and
        // trains on the resulting games (the live analog of --replay-corpus),
        // then exits before SwiftUI / Metal GUI init.
        Self.handleTrainVsUciIfPresent(rawArgs: rawArgs)

        // Pre-flight: PGN → game-corpus import (--import-pgn). Streams the
        // .pgn(.zst), converts games to the corpus format, and exits.
        Self.handleImportPGNIfPresent(rawArgs: rawArgs)

        // Pre-flight: corpus validation (--validate-corpus). Verifies shard
        // integrity + corpus.json consistency for one or more corpora, optionally
        // repairs stale metadata counts (--fix), prints a report, and exits.
        Self.handleValidateCorpusIfPresent(rawArgs: rawArgs)

        // Known flags.
        let booleanFlags: Set<String> = ["--train", "--playchess"]
        let valueFlags: Set<String> = ["--parameters", "--output", "--training-time-limit", "--training-step-limit", "--start-model", "--model"]

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

        // `--playchess` — boolean GUI-launch flag. Opens the window and
        // starts a human-vs-network game immediately (see `--model`).
        // Mutually exclusive with `--train` (one launch mode at a time).
        let playChessIndices = rawArgs.indices.filter { rawArgs[$0] == "--playchess" }
        self.autoPlayChessOnLaunch = !playChessIndices.isEmpty
        for idx in playChessIndices { consumedIndices.insert(idx) }
        if playChessIndices.count > 1 {
            errors.append("--playchess specified \(playChessIndices.count) times; only one allowed")
        }
        if !trainIndices.isEmpty && !playChessIndices.isEmpty {
            errors.append("--train and --playchess are mutually exclusive")
        }

        // `--bf16-cast-in-forward` — experimental: train bf16 models with config
        // D (weights stored fp32, cast to bf16 in the forward; no bf16 working
        // variable / fp32 master), the workaround for the macOS-27-beta bf16
        // training divergence. Read directly in `SessionController.ensureTrainer`
        // via `CommandLine.arguments`; here we only consume the flag so the
        // unknown-argument scan accepts it. A no-op for fp32 models.
        let bf16CastIndices = rawArgs.indices.filter { rawArgs[$0] == "--bf16-cast-in-forward" }
        for idx in bf16CastIndices { consumedIndices.insert(idx) }
        if bf16CastIndices.count > 1 {
            errors.append("--bf16-cast-in-forward specified \(bf16CastIndices.count) times; only one allowed")
        }

        // `--crosscheck-movegen` — diagnostic soak: every `MoveGenerator.legalMoves`
        // call also runs the make/unmake and pin-based generators and logs any
        // divergent position. Read in `MoveGenerator` via `CommandLine.arguments`;
        // consumed here so the unknown-argument scan accepts it.
        let crosscheckMovegenIndices = rawArgs.indices.filter { rawArgs[$0] == "--crosscheck-movegen" }
        for idx in crosscheckMovegenIndices { consumedIndices.insert(idx) }

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
                    trainingTimeLimitSec: override,
                    trainingStepLimit: nil
                )
            } else {
                parsedConfig?.trainingTimeLimitSec = override
            }
        }

        // `--training-step-limit <steps>` — stop after the trainer
        // completes this many SGD steps (snapshot + exit, same dance
        // as the time limit; whichever budget fires first wins). CLI
        // flag wins over a `training_step_limit` key in --parameters.
        var trainingStepLimitCliOverride: Int? = nil
        if let raw = takeValue(for: "--training-step-limit") {
            if let parsed = Int(raw), parsed > 0 {
                trainingStepLimitCliOverride = parsed
            } else {
                errors.append("--training-step-limit value '\(raw)' is not a positive integer")
            }
        }
        if let override = trainingStepLimitCliOverride {
            if parsedConfig == nil {
                parsedConfig = CliTrainingConfig(
                    trainingParameters: [:],
                    trainingTimeLimitSec: nil,
                    trainingStepLimit: override
                )
            } else {
                parsedConfig?.trainingStepLimit = override
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

        // `--model <path>` — opponent weights for `--playchess`. (Under
        // `--uci` this flag is consumed by the UCI pre-flight above and
        // never reaches here, so any `--model` seen now belongs to a
        // GUI launch — valid only alongside `--playchess`.) Nil ⇒ the
        // most recent saved session's trainer is used.
        let parsedPlayChessModelPath = takeValue(for: "--model")
        self.playChessModelPath = parsedPlayChessModelPath
        if parsedPlayChessModelPath != nil && playChessIndices.isEmpty {
            errors.append("--model is only valid alongside --playchess (UCI passes --model via --uci)")
        }

        // `--start-model <path>` — with --train, load this saved model's
        // weights into the champion instead of random initialization (the
        // trainer then forks from it as usual). The lever for controlled
        // A/B experiments: N headless runs from one identical starting
        // net, with per-arm hyperparameters from --parameters.
        let parsedTrainStartModelPath = takeValue(for: "--start-model")
        self.trainStartModelPath = parsedTrainStartModelPath
        if parsedTrainStartModelPath != nil && trainIndices.isEmpty {
            errors.append("--start-model is only valid alongside --train")
        }

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
            Usage: DrewsChessMachine [mode] [options]          (flags may be given in any order)

            Launch modes (pick at most one; default = open the normal training console GUI):
              --train                         Headless: auto-build a fresh network, start Play-and-Train,
                                              switch to the Candidate Test view.
              --playchess                     Open the GUI and immediately start a human-vs-network game.
                                              Opponent weights resolved like --uci (see --model below).

            Training options (with --train):
              --parameters <file>             JSON file of hyperparameter overrides (partial files allowed;
                                              only keys matching a known field are applied).
              --output <file>                 Write the JSON snapshot to <file> on training_time_limit expiry.
                                              Without this flag, the snapshot goes to stdout.
              --training-time-limit <seconds> Seconds of Play-and-Train before the JSON snapshot is written
                                              and the process exits. Overrides any value in --parameters.
                                              Only honored under --train.
              --training-step-limit <steps>   Stop after the trainer completes this many SGD steps (snapshot
                                              + exit, same as the time limit; first budget to fire wins).
                                              Overrides any training_step_limit in --parameters.
              --start-model <path>            Load this saved model (.safetensors / .dcmmodel) as the starting
                                              champion instead of a fresh random init; the trainer forks from
                                              it. For controlled A/B runs from one identical starting net.

            Opponent selection (with --playchess):
              --model <path>                  .safetensors or .dcmmodel weights to play against. Without it,
                                              the most recently saved session's trainer is used.

            Headless engine / tools (each runs without opening a window, then exits):
              --uci [--model <path>]          Run as a UCI engine on stdin/stdout (cutechess, etc.). --model
                                              selects weights (default: latest saved session's trainer).
              --sweep [--sweep-sizes <csv>] [--sweep-seconds <n>]
                                              Batch-size throughput sweep; print the table and exit.
              --analyze-replay-buffer <path>  Analyze a replay_buffer.bin (or a .dcmsession dir); print JSON,
                                              human summary to stderr, and exit.
              --probe-model <path> [--probe-set 200|wide|both] [--probe-out <file>]
                                              Run the Lichess probe batteries against saved checkpoints
                                              (a weight file, one .dcmsession, or a directory of sessions);
                                              one JSON line per checkpoint x set, then exit.
              --show-default-parameters       Print every default training parameter as JSON and exit.
              --create-parameters-file [<path>] [--force]
                                              Write parameters.json + parameters.md (default: ./) and exit.

            Offline corpus replay & PGN import (each runs headless, then exits):
              --replay-corpus <dir|id>        Train on a fixed recorded game corpus — a path, or a bare corpus
                                              ID resolved under Corpora/ (repeatable, to mix corpora):
                                              no self-play, no arena, no promotion. Fills the replay buffer from
                                              the games and runs a step-locked SGD loop (K = batch / replay-ratio
                                              positions per step). Pair with --training-step-limit <n> OR
                                              --epochs <n> (default: 1 pass) and --parameters <file> to pin the
                                              hyperparameters. Pass --start-model <file> to continue training from a
                                              saved model (its embedded architecture is used). Ctrl-C stops cleanly
                                              and saves; press again to force-quit. The trainer model is saved every
                                              1000 steps and on exit/abort to a single rolling file (overwritten):
                                              --out-model <path>, else next to --start-model, else the app Models dir
                                              named after the corpus (<corpusID>-replay-latest.safetensors).
              --out-model <path>              Destination for the rolling trainer-model file (overwrites in place);
                                              a .safetensors extension is appended if you don't supply one.
              --epochs <n>                    Replay budget: number of full passes over the corpus.
              --import-pgn <path>             Convert a .pgn / .pgn.zst (e.g. a Lichess monthly dump) into a
                                              corpus, then exit. .zst needs the `zstd` CLI on PATH; standard-start
                                              games only. Filters: --min-rating <elo> (both sides),
                                              --max-games <n>, --min-plies <n>,
                                              --time-control <bullet,blitz,rapid,classical>, --corpus-name <name>,
                                              --shard-soft-limit-mb <mb> (default 64),
                                              --max-storage <size> (e.g. 2GB; stops near that corpus body size),
                                              --import-threads <n> (default cores-2),
                                              --lenient (count parse failures instead of hard-failing on the first).
              --validate-corpus <dir|id>      Validate a corpus (repeatable): checks every sealed shard's integrity
                                              (front magic, corpus-ID stamp, trailer, whole-shard SHA-256, per-record
                                              CRC) and that corpus.json is consistent with the shards (per-source
                                              game/ply counts, sequence numbers, recording state), then prints a
                                              report and the true game/ply totals. Exit 0 if valid, 1 if any problem
                                              remains. Add --fix to repair fixable metadata (recompute stale per-source
                                              gamesAdded/pliesAdded from the shard trailers and rewrite corpus.json;
                                              shard bytes are never modified). Add --quick to skip the SHA/CRC body
                                              pass (header/trailer counts only — fast, no integrity check).

            Self-play recording: set the `record_self_play_games` parameter (e.g. in a --parameters file) to
            record every kept self-play game into a corpus under Corpora/ during a --train run.

            Examples:
              # Cross-architecture A/B on identical games (build is fresh each run; average over N runs):
              DrewsChessMachine --replay-corpus <CorpusDir> --parameters frozen.json --training-step-limit 50000 --output archA.json

              # Same-architecture hyperparameter A/B from one identical starting net:
              DrewsChessMachine --train --start-model champ.safetensors --parameters lrA.json --training-step-limit 50000 --output lrA.json

              # Import a Lichess dump (rated blitz/rapid, 1800+, first 1M games):
              DrewsChessMachine --import-pgn lichess_2026-05.pgn.zst --min-rating 1800 --time-control blitz,rapid --max-games 1000000 --corpus-name lichess-2026-05

              # Replay a corpus for 3 full passes:
              DrewsChessMachine --replay-corpus <CorpusDir> --epochs 3 --parameters frozen.json
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
        // (Channel display names are now derived per-encoding from
        // `InputEncoding.channelNames` — sized to `planeCount` by construction
        // — so the old "TensorChannelNames must match the default inputPlanes"
        // launch precondition is gone; a count test guards it instead.)
        let dirtyMarker = BuildInfo.gitDirty ? "*" : ""
        let autoTrainMarker = autoTrainOnLaunch ? " autoTrain=on" : ""
        let playChessMarker = autoPlayChessOnLaunch ? " playChess=on" : ""
        SessionLogger.shared.log(
            "[APP] launched build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirtyMarker) branch=\(BuildInfo.gitBranch) date=\(BuildInfo.buildDate) timestamp=\(BuildInfo.buildTimestamp)\(autoTrainMarker)\(playChessMarker)"
        )
        // The launch banner deliberately no longer prints arch fields: at
        // launch nothing is loaded, so the only arch knowable here is the
        // compile-time default — printing it as bare `inputPlanes=`/`arch_hash=`
        // invited mistaking it for the live (runtime-configured) architecture.
        // Log the default explicitly labelled instead; the real arch is logged
        // via `[ARCH]` the moment a model is built, loaded, or resumed.
        SessionLogger.shared.logArchitecture(
            event: "default preset (no model loaded yet)",
            arch: .current
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
        if autoPlayChessOnLaunch {
            let modelDesc = playChessModelPath ?? "latest saved session trainer"
            SessionLogger.shared.log("[APP] --playchess flag detected; will start a human-vs-network game on first appear (opponent: \(modelDesc))")
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
                autoPlayChessOnLaunch: autoPlayChessOnLaunch,
                playChessModelPath: playChessModelPath,
                trainStartModelPath: trainStartModelPath,
                cliConfig: cliConfig,
                cliOutputURL: cliOutputURL,
                showTrainingGraphs: showTrainingGraphs,
                chartCollectionEnabled: chartCollectionEnabled,
                showPolicyChannelsPanel: showPolicyChannelsPanel
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
                    .keyboardShortcut("s", modifiers: .command)
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
                Toggle("Show Policy Channels Panel", isOn: $showPolicyChannelsPanel)
                Toggle("Show Emit Window Stats", isOn: $showEmitWindowStats)
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
                Button("New Network…") { commandHub.presentBuildNewModel() }
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
                Divider()
                Button("Promote Trainee Now") { commandHub.promoteTrainerNow() }
                    .disabled(!commandHub.realTraining || commandHub.isArenaRunning)
            }

            // Chess menu — human-vs-network play. The user picks the
            // opponent (champion / trainer / a saved model file) and
            // which side they want in the setup popover that the
            // Play… item opens.
            //
            // Play is intentionally available concurrently with real
            // training, arenas, sweeps, and the debug single-game
            // path: the human game's AI side runs on a snapshotted
            // inference network owned solely by the human game, so it
            // doesn't compete with self-play workers, the arena
            // candidate, or the live champion for graph state. The
            // only gate is "another human game is already running in
            // this window" — multi-window support for two-or-more
            // simultaneous human games comes later.
            CommandMenu("Chess") {
                Button("Play…") { commandHub.openHumanPlaySetup() }
                    .disabled(commandHub.humanGameInFlight)
                Button("Reset Game") { commandHub.resetHumanGame() }
                    .disabled(!commandHub.humanGameCanReset)
                Button("Stop Game") { commandHub.stopHumanGame() }
                    .disabled(!commandHub.humanGameInFlight)
            }

            // Performance metrics for the training network: the two
            // always-on probe monitors. "Probe now" / "Open detail" /
            // "Export…" all live inside the monitor windows themselves,
            // so the menu only needs the entry points.
            CommandMenu("Performance") {
                Button("Open Tactical Probe Monitor…") { commandHub.openTacticalProbeMonitor() }
                    .disabled(!commandHub.networkReady)
                Button("Open Lichess Probe Monitor…") { commandHub.openLichessProbeMonitor() }
                    .disabled(!commandHub.networkReady)
                Button("Open Lichess Probe Detail…") { commandHub.openLichessProbeDetail() }
                    .disabled(!commandHub.networkReady)
                Divider()
                Button("Open Training vs Eval Loss…") { commandHub.openCombinedLossWindow() }
                    .disabled(!commandHub.networkReady)
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
                Button("Analyze Replay Buffer…") { commandHub.analyzeReplayBuffer() }
                Button("Analyze Value Head Weights…") { commandHub.analyzeValueHead() }
                Button("Analyze Network Weights (Champion)…") { commandHub.analyzeNetworkWeights() }
                Button("Analyze Network Weights (Trainer)…") { commandHub.analyzeNetworkWeightsTrainer() }
                Button("Run All Analyses…") { commandHub.runAllAnalyses() }
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

    // MARK: - UCI mode pre-flight (--uci [--model <path.dcmmodel>])

    /// Inspects `rawArgs` for `--uci`. If present, validates the
    /// allowed companion flag set (`--model <path>` is the only one),
    /// hands control to `UCIEngine.runAndExit`, and never returns.
    ///
    /// `--uci` deliberately runs before the strict-CLI parser below
    /// (which would reject it as unknown) AND before the SwiftUI
    /// `WindowGroup` is created — so the cutechess-launched process
    /// behaves as a pure stdin/stdout engine. No window, no menu bar,
    /// no `AutoResumeController` countdown sheet auto-resuming a
    /// training session under us.
    ///
    /// Model resolution lives in `UCIModelLoader`:
    /// - `--model <path>` loads that `.dcmmodel` file directly.
    /// - no `--model` ⇒ most recently saved session's
    ///   `trainer.dcmmodel` via `LastSessionPointer`.
    private static func handleUciIfPresent(rawArgs: [String]) {
        let uciFlag = "--uci"
        let modelFlag = "--model"
        guard rawArgs.contains(uciFlag) else { return }

        // Any other `--`-prefixed flag besides `--uci` / `--model` is
        // a usage error — a typo in `--mode` would otherwise silently
        // launch the GUI with a confusing model-load failure.
        // `--crosscheck-movegen` is a global diagnostic (read in
        // `MoveGenerator` via `CommandLine.arguments`); permit it so a UCI
        // self-play run can exercise the move-generator cross-check.
        let allowedFlags: Set<String> = [uciFlag, modelFlag, "--crosscheck-movegen"]
        if let badFlag = rawArgs.first(where: {
            $0.hasPrefix("--") && !allowedFlags.contains($0)
        }) {
            FileHandle.standardError.write(Data(
                "error: \(uciFlag) does not accept '\(badFlag)' (only \(modelFlag) <path.dcmmodel> is allowed alongside)\n".utf8
            ))
            Darwin.exit(20)
        }

        // Extract `--model <path>` if present. Exactly zero or one
        // occurrence allowed; if present, the immediately following
        // token is the path.
        var modelPath: String? = nil
        let modelIndices = rawArgs.indices.filter { rawArgs[$0] == modelFlag }
        if modelIndices.count > 1 {
            FileHandle.standardError.write(Data(
                "error: \(modelFlag) specified \(modelIndices.count) times; only one allowed\n".utf8
            ))
            Darwin.exit(21)
        }
        if let idx = modelIndices.first {
            let valueIdx = idx + 1
            guard valueIdx < rawArgs.count, !rawArgs[valueIdx].hasPrefix("--") else {
                FileHandle.standardError.write(Data(
                    "error: \(modelFlag) requires a path value\n".utf8
                ))
                Darwin.exit(22)
            }
            modelPath = rawArgs[valueIdx]
        }

        // Anything in rawArgs that isn't `--uci`, `--model`, or
        // `--model`'s value is a stray positional we don't want to
        // silently accept.
        var consumed = Set<Int>()
        for (i, arg) in rawArgs.enumerated() where arg == uciFlag {
            consumed.insert(i)
        }
        if let idx = modelIndices.first {
            consumed.insert(idx)
            consumed.insert(idx + 1)
        }
        // `--crosscheck-movegen` is a global diagnostic read in `MoveGenerator`
        // via `CommandLine.arguments`; consume it so it isn't flagged as a stray.
        for (i, arg) in rawArgs.enumerated() where arg == "--crosscheck-movegen" {
            consumed.insert(i)
        }
        for (i, arg) in rawArgs.enumerated() where !consumed.contains(i) {
            FileHandle.standardError.write(Data(
                "error: \(uciFlag) does not accept positional argument '\(arg)'\n".utf8
            ))
            Darwin.exit(23)
        }

        UCIEngine.runAndExit(modelPath: modelPath)
    }

    // MARK: - Offline corpus-replay pre-flight (--replay-corpus)

    /// Inspects `rawArgs` for `--replay-corpus <dir>` (repeatable). If present,
    /// snapshots the training parameters on the main actor (applying a
    /// `--parameters` file first if given), builds a `CorpusReplayConfig`, hands
    /// control to `CorpusReplayRunner.runAndExit`, and never returns. Runs
    /// before the strict-CLI parser (which would reject `--replay-corpus` /
    /// `--epochs` as unknown) and before the SwiftUI WindowGroup.
    private static func handleReplayCorpusIfPresent(rawArgs: [String]) {
        guard rawArgs.contains("--replay-corpus") else { return }

        var corpusPaths: [String] = []
        var epochs: Int? = nil
        var stepLimit: Int? = nil
        var parametersPath: String? = nil
        var startModelPath: String? = nil
        var outModelPath: String? = nil
        var presetName: String? = nil
        var startShard: Int? = nil
        var startGameIndex: Int? = nil
        var resumeExact = false
        var enumerateCheckpoints = false

        // Strict validation: a recognized flag with a missing or unparseable
        // value is a HARD error, never a silent default. A mistyped
        // `--start-shard 9x` silently starting from shard 0 would waste a
        // multi-hour pass landing nowhere near the intended resume point.
        // Anything unexpected — an unknown `--flag` OR a stray/misplaced bare
        // token — is a hard error, not a silent skip (this handler exits before
        // the strict-CLI parser that would otherwise catch unknown flags, and
        // `rawArgs` is `CommandLine.arguments.dropFirst()` so argv[0] is already
        // gone and every flag consumes its own value — nothing legitimate lands
        // in `default`). `nextValue` is nil when the next token is absent OR is
        // itself a `--flag`, so "flag with no value" is caught even when another
        // flag immediately follows.
        func requireValue(_ flag: String, _ v: String?) -> String {
            guard let v else {
                FileHandle.standardError.write(Data("error: \(flag) requires a value\n".utf8))
                Darwin.exit(2)
            }
            return v
        }
        func requireInt(_ flag: String, _ v: String?) -> Int {
            let s = requireValue(flag, v)
            guard let n = Int(s) else {
                FileHandle.standardError.write(Data("error: \(flag) expects an integer value, got '\(s)'\n".utf8))
                Darwin.exit(2)
            }
            return n
        }

        var i = 0
        while i < rawArgs.count {
            let arg = rawArgs[i]
            let nextValue: String? = {
                guard i + 1 < rawArgs.count else { return nil }
                let v = rawArgs[i + 1]
                return v.hasPrefix("--") ? nil : v
            }()
            switch arg {
            case "--replay-corpus":
                corpusPaths.append(requireValue(arg, nextValue)); i += 2
            case "--epochs":
                epochs = requireInt(arg, nextValue); i += 2
            case "--training-step-limit":
                stepLimit = requireInt(arg, nextValue); i += 2
            case "--parameters":
                parametersPath = requireValue(arg, nextValue); i += 2
            case "--start-model":
                startModelPath = requireValue(arg, nextValue); i += 2
            case "--out-model":
                outModelPath = requireValue(arg, nextValue); i += 2
            case "--preset":
                presetName = requireValue(arg, nextValue); i += 2
            case "--start-shard":
                startShard = requireInt(arg, nextValue); i += 2
            case "--start-game-index":
                startGameIndex = requireInt(arg, nextValue); i += 2
            case "--resume-exact":
                resumeExact = true; i += 1   // boolean flag, no value
            case "--enumerate-checkpoints":
                enumerateCheckpoints = true; i += 1   // boolean flag, no value
            default:
                FileHandle.standardError.write(Data("error: unexpected argument '\(arg)' (with --replay-corpus)\n".utf8))
                Darwin.exit(2)
            }
        }

        guard !corpusPaths.isEmpty else {
            FileHandle.standardError.write(Data("error: --replay-corpus requires at least one corpus directory path\n".utf8))
            Darwin.exit(2)
        }

        // --resume-exact preconditions (parse-time, before any GPU work): it
        // reconstructs the buffer from the start-model's saved resume metadata,
        // so it needs a --start-model and can't combine with the approximate
        // cold-refill resume flags.
        if resumeExact {
            if startModelPath == nil {
                FileHandle.standardError.write(Data("error: --resume-exact requires --start-model (a checkpoint carrying replay_* metadata)\n".utf8))
                Darwin.exit(2)
            }
            if startShard != nil || startGameIndex != nil {
                FileHandle.standardError.write(Data("error: --resume-exact is mutually exclusive with --start-shard / --start-game-index\n".utf8))
                Darwin.exit(2)
            }
        }

        // Snapshot parameters on the main actor (init() runs on the main
        // thread), applying a --parameters file first. Everything below the
        // snapshot is plain Sendable data, so the off-actor replay task never
        // touches the @MainActor singleton (which would deadlock against the
        // syncWait semaphore held on this thread).
        let params: ReplayParams = MainActor.assumeIsolated {
            if let pp = parametersPath {
                do {
                    let url = URL(fileURLWithPath: (pp as NSString).expandingTildeInPath)
                    let cfg = try CliTrainingConfig.load(from: url)
                    // A --parameters apply is transient to this process. Without
                    // this guard each setter's didSet persists to UserDefaults,
                    // and because this headless process shares the GUI's bundle id
                    // (same UserDefaults domain) the overrides would silently
                    // become the GUI's next-launch defaults — the same
                    // cross-process contamination already guarded on the --train
                    // path. Safe to toggle here: this block runs synchronously on
                    // the main actor.
                    TrainingParameters.suppressPersistence = true
                    defer { TrainingParameters.suppressPersistence = false }
                    try TrainingParameters.shared.apply(cfg.trainingParameters)
                    if stepLimit == nil { stepLimit = cfg.trainingStepLimit }
                } catch {
                    FileHandle.standardError.write(Data("error: --parameters load/apply failed: \(error.localizedDescription)\n".utf8))
                    Darwin.exit(2)
                }
            }
            let tp = TrainingParameters.shared
            return ReplayParams(
                learningRate: tp.learningRate,
                entropyBonus: tp.entropyBonus,
                drawPenalty: tp.drawPenalty,
                weightDecay: tp.weightDecay,
                gradClipMaxNorm: tp.gradClipMaxNorm,
                policyLossWeight: tp.policyLossWeight,
                valueLossWeight: tp.valueLossWeight,
                illegalMassWeight: tp.illegalMassWeight,
                policyLabelSmoothingEpsilon: tp.policyLabelSmoothingEpsilon,
                valueLabelSmoothingEpsilon: tp.valueLabelSmoothingEpsilon,
                momentumCoeff: tp.momentumCoeff,
                signedAdvantageComplementCE: tp.signedAdvantageComplementCE,
                sqrtBatchScalingLR: tp.sqrtBatchScalingLR,
                lrWarmupSteps: tp.lrWarmupSteps,
                trainingBatchSize: tp.trainingBatchSize,
                replayBufferCapacity: tp.replayBufferCapacity,
                replayRatioTarget: tp.replayRatioTarget,
                replayBufferMinPositionsBeforeTraining: tp.replayBufferMinPositionsBeforeTraining
            )
        }

        // Mint the run's saved-model ModelID here on the main thread — the
        // minter is main-actor isolated and the replay loop runs off-actor.
        let runModelID = MainActor.assumeIsolated { ModelIDMinter.mint().value }

        let config = CorpusReplayConfig(
            corpusDirectories: corpusPaths.map { arg in
                guard let dir = resolveCorpusDirectory(arg) else {
                    let triedPath = URL(fileURLWithPath: (arg as NSString).expandingTildeInPath).path
                    let triedID = CorpusPaths.corporaDir.appendingPathComponent(arg).path
                    FileHandle.standardError.write(Data(
                        "error: corpus not found for '\(arg)' — no corpus.json at \(triedPath) or \(triedID)\n".utf8))
                    Darwin.exit(2)
                }
                return dir
            },
            stepLimit: stepLimit,
            epochs: epochs,
            startModelPath: startModelPath,
            presetName: presetName,
            startShard: startShard,
            startGameIndex: startGameIndex,
            resumeExact: resumeExact,
            outModelPath: outModelPath,
            enumerateCheckpoints: enumerateCheckpoints,
            runModelID: runModelID
        )
        CorpusReplayRunner.runAndExit(config: config, params: params)
    }

    // MARK: - Train-vs-UCI pre-flight (--train-vs-uci)

    /// Headless trainer that plays the live trainer network against external
    /// UCI engines and trains on the games. Repeatable `--train-vs-uci
    /// "cmd=/path/to/stockfish;n=3;go=nodes 1;UCI_Elo=1400"` declares one
    /// opponent kind (cmd = path, n = instance count, go = per-move limit,
    /// everything else = setoption pairs). Mirrors the --replay-corpus model
    /// I/O (--start-model, --out-model, --enumerate-checkpoints, --preset,
    /// --parameters, --training-step-limit, --training-time-limit).
    private static func handleTrainVsUciIfPresent(rawArgs: [String]) {
        guard rawArgs.contains("--train-vs-uci") else { return }

        var opponentSpecStrings: [String] = []
        var startModelPath: String? = nil
        var outModelPath: String? = nil
        var presetName: String? = nil
        var parametersPath: String? = nil
        var stepLimit: Int? = nil
        var timeLimitSec: Double? = nil
        var enumerateCheckpoints = false
        var maxPliesPerGame = 400
        var evalSyncEverySteps = 10

        func requireValue(_ flag: String, _ v: String?) -> String {
            guard let v else {
                FileHandle.standardError.write(Data("error: \(flag) requires a value\n".utf8))
                Darwin.exit(2)
            }
            return v
        }
        func requireInt(_ flag: String, _ v: String?) -> Int {
            let s = requireValue(flag, v)
            guard let n = Int(s) else {
                FileHandle.standardError.write(Data("error: \(flag) expects an integer value, got '\(s)'\n".utf8))
                Darwin.exit(2)
            }
            return n
        }

        var i = 0
        while i < rawArgs.count {
            let arg = rawArgs[i]
            let nextValue: String? = {
                guard i + 1 < rawArgs.count else { return nil }
                let v = rawArgs[i + 1]
                return v.hasPrefix("--") ? nil : v
            }()
            switch arg {
            case "--train-vs-uci":
                opponentSpecStrings.append(requireValue(arg, nextValue)); i += 2
            case "--start-model":
                startModelPath = requireValue(arg, nextValue); i += 2
            case "--out-model":
                outModelPath = requireValue(arg, nextValue); i += 2
            case "--preset":
                presetName = requireValue(arg, nextValue); i += 2
            case "--parameters":
                parametersPath = requireValue(arg, nextValue); i += 2
            case "--training-step-limit":
                stepLimit = requireInt(arg, nextValue); i += 2
            case "--training-time-limit":
                let s = requireValue(arg, nextValue)
                guard let t = Double(s), t > 0 else {
                    FileHandle.standardError.write(Data("error: --training-time-limit expects seconds > 0, got '\(s)'\n".utf8))
                    Darwin.exit(2)
                }
                timeLimitSec = t; i += 2
            case "--max-plies":
                maxPliesPerGame = requireInt(arg, nextValue); i += 2
            case "--eval-sync-steps":
                evalSyncEverySteps = requireInt(arg, nextValue); i += 2
            case "--enumerate-checkpoints":
                enumerateCheckpoints = true; i += 1
            default:
                FileHandle.standardError.write(Data("error: unexpected argument '\(arg)' (with --train-vs-uci)\n".utf8))
                Darwin.exit(2)
            }
        }

        // Parse each opponent spec: "cmd=/path;n=3;go=nodes 1;UCI_Elo=1400".
        func parseOpponent(_ s: String) -> TrainVsUciOpponentSpec {
            var command: String? = nil
            var count = 1
            var goLimit = "depth 1"
            var options: [UCIArbiter.Option] = []
            for rawField in s.split(separator: ";") {
                let field = rawField.trimmingCharacters(in: .whitespaces)
                if field.isEmpty { continue }
                guard let eq = field.firstIndex(of: "=") else {
                    FileHandle.standardError.write(Data("error: --train-vs-uci field '\(field)' is not key=value\n".utf8))
                    Darwin.exit(2)
                }
                let key = String(field[..<eq]).trimmingCharacters(in: .whitespaces)
                let value = String(field[field.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
                switch key.lowercased() {
                case "cmd": command = value
                case "n":
                    guard let n = Int(value), n >= 1 else {
                        FileHandle.standardError.write(Data("error: --train-vs-uci n= expects a positive integer, got '\(value)'\n".utf8))
                        Darwin.exit(2)
                    }
                    count = n
                case "go": goLimit = value
                default: options.append(UCIArbiter.Option(name: key, value: value))
                }
            }
            guard let command, !command.isEmpty else {
                FileHandle.standardError.write(Data("error: --train-vs-uci spec '\(s)' is missing required cmd=<path>\n".utf8))
                Darwin.exit(2)
            }
            let kind = URL(fileURLWithPath: (command as NSString).expandingTildeInPath)
                .deletingPathExtension().lastPathComponent
            return TrainVsUciOpponentSpec(command: command, count: count, goLimit: goLimit, options: options, kind: kind)
        }
        let opponents = opponentSpecStrings.map(parseOpponent)

        // Snapshot training parameters on the main actor (mirrors the
        // --replay-corpus handler), applying a --parameters file first.
        let params: ReplayParams = MainActor.assumeIsolated {
            if let pp = parametersPath {
                do {
                    let url = URL(fileURLWithPath: (pp as NSString).expandingTildeInPath)
                    let cfg = try CliTrainingConfig.load(from: url)
                    TrainingParameters.suppressPersistence = true
                    defer { TrainingParameters.suppressPersistence = false }
                    try TrainingParameters.shared.apply(cfg.trainingParameters)
                    if stepLimit == nil { stepLimit = cfg.trainingStepLimit }
                    if timeLimitSec == nil { timeLimitSec = cfg.trainingTimeLimitSec }
                } catch {
                    FileHandle.standardError.write(Data("error: --parameters load/apply failed: \(error.localizedDescription)\n".utf8))
                    Darwin.exit(2)
                }
            }
            let tp = TrainingParameters.shared
            return ReplayParams(
                learningRate: tp.learningRate,
                entropyBonus: tp.entropyBonus,
                drawPenalty: tp.drawPenalty,
                weightDecay: tp.weightDecay,
                gradClipMaxNorm: tp.gradClipMaxNorm,
                policyLossWeight: tp.policyLossWeight,
                valueLossWeight: tp.valueLossWeight,
                illegalMassWeight: tp.illegalMassWeight,
                policyLabelSmoothingEpsilon: tp.policyLabelSmoothingEpsilon,
                valueLabelSmoothingEpsilon: tp.valueLabelSmoothingEpsilon,
                momentumCoeff: tp.momentumCoeff,
                signedAdvantageComplementCE: tp.signedAdvantageComplementCE,
                sqrtBatchScalingLR: tp.sqrtBatchScalingLR,
                lrWarmupSteps: tp.lrWarmupSteps,
                trainingBatchSize: tp.trainingBatchSize,
                replayBufferCapacity: tp.replayBufferCapacity,
                replayRatioTarget: tp.replayRatioTarget,
                replayBufferMinPositionsBeforeTraining: tp.replayBufferMinPositionsBeforeTraining
            )
        }

        let runModelID = MainActor.assumeIsolated { ModelIDMinter.mint().value }

        let config = TrainVsUciConfig(
            opponents: opponents,
            stepLimit: stepLimit,
            timeLimitSec: timeLimitSec,
            startModelPath: startModelPath,
            presetName: presetName,
            outModelPath: outModelPath,
            enumerateCheckpoints: enumerateCheckpoints,
            maxPliesPerGame: maxPliesPerGame,
            evalSyncEverySteps: evalSyncEverySteps,
            runModelID: runModelID
        )
        TrainVsUciRunner.runAndExit(config: config, params: params)
    }

    // MARK: - Corpus validation pre-flight (--validate-corpus)

    /// Inspects `rawArgs` for `--validate-corpus <dir|id>` (repeatable). If
    /// present, validates each corpus (shard integrity + `corpus.json`
    /// consistency), optionally repairs the fixable metadata with `--fix`, prints
    /// a report, and exits: 0 if every corpus is valid, 1 if any problem remains
    /// (or a target can't be resolved), 2 on a usage error. `--quick` skips the
    /// shard-body SHA/CRC pass (header/trailer counts only). Runs before the
    /// strict-CLI parser and the SwiftUI WindowGroup.
    private static func handleValidateCorpusIfPresent(rawArgs: [String]) {
        guard rawArgs.contains("--validate-corpus") else { return }

        var targets: [String] = []
        var fix = false
        var quick = false

        var i = 0
        while i < rawArgs.count {
            let arg = rawArgs[i]
            let nextValue: String? = {
                guard i + 1 < rawArgs.count else { return nil }
                let v = rawArgs[i + 1]
                return v.hasPrefix("--") ? nil : v
            }()
            switch arg {
            case "--validate-corpus":
                guard let v = nextValue else {
                    FileHandle.standardError.write(Data("error: --validate-corpus requires a corpus directory path or ID\n".utf8))
                    Darwin.exit(2)
                }
                targets.append(v); i += 2
            case "--fix":
                fix = true; i += 1   // boolean flag, no value
            case "--quick":
                quick = true; i += 1  // boolean flag, no value
            default:
                FileHandle.standardError.write(Data("error: unexpected argument '\(arg)' (with --validate-corpus)\n".utf8))
                Darwin.exit(2)
            }
        }

        guard !targets.isEmpty else {
            FileHandle.standardError.write(Data("error: --validate-corpus requires at least one corpus directory path or ID\n".utf8))
            Darwin.exit(2)
        }

        func out(_ s: String) { FileHandle.standardOutput.write(Data((s + "\n").utf8)) }

        var worstExit: Int32 = 0
        for target in targets {
            guard let dir = resolveCorpusDirectory(target) else {
                out("Corpus '\(target)': NOT FOUND (no corpus.json at that path or under Corpora/)")
                worstExit = max(worstExit, 1)
                continue
            }
            let report: CorpusValidationReport
            do {
                report = try CorpusValidator.validate(directory: dir, verifyIntegrity: !quick, fix: fix)
            } catch {
                out("Corpus at \(dir.path): ERROR — \(error.localizedDescription)")
                worstExit = max(worstExit, 1)
                continue
            }
            out(formatCorpusValidationReport(report))
            if !report.isValid { worstExit = max(worstExit, 1) }
        }
        Darwin.exit(worstExit)
    }

    /// Render a `CorpusValidationReport` as a human-readable block for the
    /// `--validate-corpus` CLI.
    private static func formatCorpusValidationReport(_ r: CorpusValidationReport) -> String {
        var lines: [String] = []
        lines.append("Corpus \(r.corpusID)")
        lines.append("  path    : \(r.directory.path)")
        lines.append("  shards  : \(r.shardCount)   games: \(r.totalGames)   plies: \(r.totalPlies)   integrity: \(r.integrityVerified ? "verified (SHA+CRC)" : "counts-only")")
        if r.findings.isEmpty {
            lines.append("  result  : OK — no problems")
            return lines.joined(separator: "\n")
        }
        for f in r.findings {
            let tag: String
            if f.fixed {
                tag = "fixed"
            } else {
                switch f.severity {
                case .error:   tag = "error"
                case .warning: tag = f.fixable ? "warn*" : "warn "
                case .info:    tag = "info "
                }
            }
            lines.append("  [\(tag)] \(f.code): \(f.message)")
        }
        let remaining = r.unresolvedProblemCount
        if remaining == 0 {
            let fixedCount = r.findings.filter { $0.fixed }.count
            lines.append("  result  : OK — \(fixedCount) fixed, no problems remain")
        } else {
            let hasFixable = r.findings.contains { $0.fixable && !$0.fixed }
            let hint = hasFixable ? "; rerun with --fix to repair the fixable ones (marked warn*)" : ""
            lines.append("  result  : \(remaining) problem(s) remain — \(r.errorCount) error, \(r.warningCount) warning\(hint)")
        }
        return lines.joined(separator: "\n")
    }

    /// Resolve a `--replay-corpus` argument to a corpus directory: accepts
    /// either a filesystem path (absolute, or relative to the working
    /// directory) or a bare corpus ID resolved under the shared `Corpora/`
    /// store. Prefers an existing path; returns the directory that contains a
    /// `corpus.json`, or nil if neither location does. `corporaDir` is injected
    /// for testing.
    static func resolveCorpusDirectory(_ arg: String,
                                       corporaDir: URL = CorpusPaths.corporaDir) -> URL? {
        let fm = FileManager.default
        let asPath = URL(fileURLWithPath: (arg as NSString).expandingTildeInPath, isDirectory: true)
        if fm.fileExists(atPath: asPath.appendingPathComponent("corpus.json").path) { return asPath }
        let asID = corporaDir.appendingPathComponent(arg, isDirectory: true)
        if fm.fileExists(atPath: asID.appendingPathComponent("corpus.json").path) { return asID }
        return nil
    }

    // MARK: - PGN import pre-flight (--import-pgn)

    /// Inspects `rawArgs` for `--import-pgn <path>`. If present, parses the
    /// import filters, hands control to `PGNImporter.runImportAndExit`, and
    /// never returns. No GPU / no SwiftUI — pure parse + disk I/O.
    private static func handleImportPGNIfPresent(rawArgs: [String]) {
        guard rawArgs.contains("--import-pgn") else { return }

        var inputPath: String? = nil
        var corpusName: String? = nil
        var minRating: Int? = nil
        var maxGames: Int? = nil
        var minPlies = 1
        var timeControls: [String]? = nil
        var shardSoftLimitMB = 64
        var maxStorageBytes: Int? = nil
        var importThreads: Int? = nil
        var failOnError = true

        var i = 0
        while i < rawArgs.count {
            let arg = rawArgs[i]
            let nextValue: String? = {
                guard i + 1 < rawArgs.count else { return nil }
                let v = rawArgs[i + 1]
                return v.hasPrefix("--") ? nil : v
            }()
            switch arg {
            case "--import-pgn":
                if let v = nextValue { inputPath = v; i += 2 } else { i += 1 }
            case "--corpus-name":
                if let v = nextValue { corpusName = v; i += 2 } else { i += 1 }
            case "--min-rating":
                if let v = nextValue { minRating = Int(v); i += 2 } else { i += 1 }
            case "--max-games":
                if let v = nextValue { maxGames = Int(v); i += 2 } else { i += 1 }
            case "--min-plies":
                if let v = nextValue { minPlies = max(1, Int(v) ?? 1); i += 2 } else { i += 1 }
            case "--shard-soft-limit-mb":
                if let v = nextValue { shardSoftLimitMB = max(1, Int(v) ?? 64); i += 2 } else { i += 1 }
            case "--time-control":
                if let v = nextValue {
                    timeControls = v.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                    i += 2
                } else { i += 1 }
            case "--max-storage":
                guard let v = nextValue, let bytes = parseByteSize(v) else {
                    FileHandle.standardError.write(Data("error: --max-storage requires a valid size (e.g. 2GB, 500MB)\n".utf8))
                    Darwin.exit(2)
                }
                maxStorageBytes = bytes; i += 2
            case "--import-threads":
                guard let v = nextValue, let n = Int(v), n >= 1 else {
                    FileHandle.standardError.write(Data("error: --import-threads requires a positive integer\n".utf8))
                    Darwin.exit(2)
                }
                importThreads = n; i += 2
            case "--lenient":
                failOnError = false; i += 1
            default:
                i += 1
            }
        }

        guard let path = inputPath else {
            FileHandle.standardError.write(Data("error: --import-pgn requires a file path (.pgn or .pgn.zst)\n".utf8))
            Darwin.exit(2)
        }

        let config = PGNImportConfig(
            inputPath: path,
            corpusName: corpusName,
            minRating: minRating,
            maxGames: maxGames,
            minPlies: minPlies,
            timeControlClasses: timeControls,
            shardSoftLimitMB: shardSoftLimitMB,
            maxStorageBytes: maxStorageBytes,
            importThreads: importThreads,
            failOnError: failOnError
        )
        PGNImporter.runImportAndExit(config: config)
    }

    /// Parse a human byte size like `2GB`, `500MB`, `1.5G`, or a bare byte
    /// count (binary units: KiB/MiB/GiB/TiB). Returns nil if unparseable.
    private static func parseByteSize(_ s: String) -> Int? {
        let t = s.trimmingCharacters(in: .whitespaces).uppercased()
        let units: [(String, Double)] = [
            ("TB", 1099511627776), ("GB", 1073741824), ("MB", 1048576), ("KB", 1024),
            ("T", 1099511627776), ("G", 1073741824), ("M", 1048576), ("K", 1024), ("B", 1)
        ]
        for (suffix, mult) in units where t.hasSuffix(suffix) {
            let numPart = t.dropLast(suffix.count).trimmingCharacters(in: .whitespaces)
            guard let value = Double(numPart), value.isFinite, value >= 0 else { return nil }
            let bytes = value * mult
            guard bytes < Double(Int.max) else { return nil }
            return Int(bytes)
        }
        return Int(t)
    }

    // MARK: - Batch-size sweep pre-flight (--sweep)

    /// Inspects `rawArgs` for `--sweep`. If present, validates the optional
    /// companions (`--sweep-sizes <csv>`, `--sweep-seconds <n>`), hands control
    /// to `SweepCLI.runAndExit`, and never returns. Runs before the strict-CLI
    /// parser (which would reject `--sweep` as unknown) and before the SwiftUI
    /// `WindowGroup` — so it's a pure headless measurement, no window, no
    /// auto-resume sheet starting training under us.
    private static func handleSweepIfPresent(rawArgs: [String]) {
        let sweepFlag = "--sweep"
        let sizesFlag = "--sweep-sizes"
        let secondsFlag = "--sweep-seconds"
        guard rawArgs.contains(sweepFlag) else { return }

        // Only the two companion flags are allowed alongside.
        let allowedFlags: Set<String> = [sweepFlag, sizesFlag, secondsFlag]
        if let bad = rawArgs.first(where: { $0.hasPrefix("--") && !allowedFlags.contains($0) }) {
            FileHandle.standardError.write(Data(
                "error: \(sweepFlag) does not accept '\(bad)' (only \(sizesFlag) <csv> and \(secondsFlag) <n> allowed)\n".utf8
            ))
            Darwin.exit(40)
        }

        // --sweep-sizes <comma-separated positive ints>
        var sizes: [Int]? = nil
        if let idx = rawArgs.firstIndex(of: sizesFlag) {
            let valueIdx = idx + 1
            guard valueIdx < rawArgs.count, !rawArgs[valueIdx].hasPrefix("--") else {
                FileHandle.standardError.write(Data(
                    "error: \(sizesFlag) requires a comma-separated list, e.g. 256,512,1024\n".utf8
                ))
                Darwin.exit(41)
            }
            let parsed = rawArgs[valueIdx]
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard !parsed.isEmpty, parsed.allSatisfy({ $0 > 0 }) else {
                FileHandle.standardError.write(Data(
                    "error: \(sizesFlag) value '\(rawArgs[valueIdx])' is not a list of positive integers\n".utf8
                ))
                Darwin.exit(42)
            }
            sizes = parsed
        }

        // --sweep-seconds <positive double> (default: SessionController's)
        var secondsPerSize = SessionController.sweepSecondsPerSize
        if let idx = rawArgs.firstIndex(of: secondsFlag) {
            let valueIdx = idx + 1
            guard valueIdx < rawArgs.count, let v = Double(rawArgs[valueIdx]), v > 0 else {
                FileHandle.standardError.write(Data(
                    "error: \(secondsFlag) requires a positive number of seconds\n".utf8
                ))
                Darwin.exit(43)
            }
            secondsPerSize = v
        }

        SweepCLI.runAndExit(sizes: sizes, secondsPerSize: secondsPerSize)
    }

    // MARK: - Depth (block-count) sweep pre-flight (--arch-sweep)

    /// Inspects `rawArgs` for `--arch-sweep`. If present, parses the optional
    /// companions (`--arch-sweep-blocks <csv>`, `--arch-sweep-steps <n>`,
    /// `--arch-sweep-batch <n>`, `--arch-sweep-out <path>`) and hands control to
    /// `ArchSweepCLI.runAndExit`, which never returns. Investigation-only.
    private static func handleArchSweepIfPresent(rawArgs: [String]) {
        let flag = "--arch-sweep"
        guard rawArgs.contains(flag) else { return }
        let blocksFlag = "--arch-sweep-blocks"
        let stepsFlag = "--arch-sweep-steps"
        let batchFlag = "--arch-sweep-batch"
        let outFlag = "--arch-sweep-out"

        let allowedFlags: Set<String> = [flag, blocksFlag, stepsFlag, batchFlag, outFlag]
        if let bad = rawArgs.first(where: { $0.hasPrefix("--") && !allowedFlags.contains($0) }) {
            FileHandle.standardError.write(Data(
                "error: \(flag) does not accept '\(bad)'\n".utf8
            ))
            Darwin.exit(50)
        }

        func value(after f: String) -> String? {
            guard let idx = rawArgs.firstIndex(of: f) else { return nil }
            let vi = idx + 1
            guard vi < rawArgs.count, !rawArgs[vi].hasPrefix("--") else {
                FileHandle.standardError.write(Data("error: \(f) requires a value\n".utf8))
                Darwin.exit(51)
            }
            return rawArgs[vi]
        }

        var blocks = [20, 50, 80, 110, 140]
        if let raw = value(after: blocksFlag) {
            let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard !parsed.isEmpty, parsed.allSatisfy({ $0 > 0 }) else {
                FileHandle.standardError.write(Data("error: \(blocksFlag) '\(raw)' is not a list of positive ints\n".utf8))
                Darwin.exit(52)
            }
            blocks = parsed
        }
        var steps = 6
        if let raw = value(after: stepsFlag) {
            guard let v = Int(raw), v > 0 else {
                FileHandle.standardError.write(Data("error: \(stepsFlag) requires a positive int\n".utf8))
                Darwin.exit(53)
            }
            steps = v
        }
        var batch = 512
        if let raw = value(after: batchFlag) {
            guard let v = Int(raw), v > 0 else {
                FileHandle.standardError.write(Data("error: \(batchFlag) requires a positive int\n".utf8))
                Darwin.exit(54)
            }
            batch = v
        }
        let outPath = value(after: outFlag) ?? "/tmp/arch_bench.jsonl"

        ArchSweepCLI.runAndExit(blocks: blocks, steps: steps, batch: batch, outPath: outPath)
    }

    // MARK: - Checkpoint probe pre-flight (--probe-model)

    /// Inspects `rawArgs` for `--probe-model`. If present, parses the
    /// optional companions (`--probe-set <200|wide|both>`,
    /// `--probe-out <path>`) and hands control to
    /// `ProbeModelCLI.runAndExit`, which never returns.
    /// Investigation-only — retro-probes saved checkpoints with the
    /// Lichess tactical batteries.
    private static func handleProbeModelIfPresent(rawArgs: [String]) {
        let flag = "--probe-model"
        guard rawArgs.contains(flag) else { return }
        let setFlag = "--probe-set"
        let outFlag = "--probe-out"

        let allowedFlags: Set<String> = [flag, setFlag, outFlag]
        if let bad = rawArgs.first(where: { $0.hasPrefix("--") && !allowedFlags.contains($0) }) {
            FileHandle.standardError.write(Data(
                "error: \(flag) does not accept '\(bad)'\n".utf8
            ))
            Darwin.exit(60)
        }

        func value(after f: String) -> String? {
            guard let idx = rawArgs.firstIndex(of: f) else { return nil }
            let vi = idx + 1
            guard vi < rawArgs.count, !rawArgs[vi].hasPrefix("--") else {
                FileHandle.standardError.write(Data("error: \(f) requires a value\n".utf8))
                Darwin.exit(62)
            }
            return rawArgs[vi]
        }

        guard let modelPath = value(after: flag) else {
            FileHandle.standardError.write(Data(
                "error: \(flag) requires a path (weight file, .dcmsession dir, or a directory of sessions)\n".utf8
            ))
            Darwin.exit(63)
        }
        var set = ProbeModelCLI.ProbeSet.both
        if let raw = value(after: setFlag) {
            guard let parsed = ProbeModelCLI.ProbeSet(rawValue: raw) else {
                FileHandle.standardError.write(Data(
                    "error: \(setFlag) must be 200, wide, or both; got '\(raw)'\n".utf8
                ))
                Darwin.exit(64)
            }
            set = parsed
        }
        ProbeModelCLI.runAndExit(modelPath: modelPath, set: set, outPath: value(after: outFlag))
    }

    // MARK: - Fresh-net mint pre-flight (--new-model)

    /// `--new-model --preset <name> [--out-model <path>]`: build an untrained
    /// net from the preset and write it to safetensors, then exit. No training.
    /// Mints the ModelID here (main actor) and hands the GPU build off to
    /// `NewModelCLI.runAndExit`.
    private static func handleNewModelIfPresent(rawArgs: [String]) {
        let flag = "--new-model"
        guard rawArgs.contains(flag) else { return }
        let presetFlag = "--preset"
        let outFlag = "--out-model"

        let allowedFlags: Set<String> = [flag, presetFlag, outFlag]
        if let bad = rawArgs.first(where: { $0.hasPrefix("--") && !allowedFlags.contains($0) }) {
            FileHandle.standardError.write(Data(
                "error: \(flag) does not accept '\(bad)'\n".utf8
            ))
            Darwin.exit(75)
        }

        func value(after f: String) -> String? {
            guard let idx = rawArgs.firstIndex(of: f) else { return nil }
            let vi = idx + 1
            guard vi < rawArgs.count, !rawArgs[vi].hasPrefix("--") else {
                FileHandle.standardError.write(Data("error: \(f) requires a value\n".utf8))
                Darwin.exit(76)
            }
            return rawArgs[vi]
        }

        guard let presetName = value(after: presetFlag) else {
            let valid = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
            FileHandle.standardError.write(Data(
                "error: \(flag) requires --preset <name>. Valid: \(valid)\n".utf8
            ))
            Darwin.exit(77)
        }

        // Mint on the main actor (the minter is main-actor isolated); the build
        // runs off-actor inside runAndExit.
        let modelID = MainActor.assumeIsolated { ModelIDMinter.mint().value }
        NewModelCLI.runAndExit(presetName: presetName, outPath: value(after: outFlag), modelID: modelID)
    }

    // MARK: - Replay-buffer analyzer pre-flight (--analyze-replay-buffer)

    /// Inspects `rawArgs` for `--analyze-replay-buffer`. If present,
    /// validates that the only other argument is one positional path
    /// (either a session directory containing `replay_buffer.bin`, or
    /// the binary file itself), runs `ReplayBufferAnalyzer`, prints
    /// JSON to stdout + a human-readable summary to stderr, and exits.
    /// Anything else is a usage error → non-zero exit.
    ///
    /// Sub-second on a fresh allocation + restore of a 1 M-position
    /// buffer; never touches the `TrainingParameters` singleton or any
    /// network/Metal state.
    private static func handleAnalyzeReplayBufferIfPresent(rawArgs: [String]) {
        let analyzeFlag = "--analyze-replay-buffer"
        guard rawArgs.contains(analyzeFlag) else { return }

        // Only allowed companion args are positional (non-`--`-prefixed).
        // Any other `--`-prefixed flag is a usage error so a typo doesn't
        // get silently accepted.
        let allowedFlags: Set<String> = [analyzeFlag]
        if let badFlag = rawArgs.first(where: {
            $0.hasPrefix("--") && !allowedFlags.contains($0)
        }) {
            FileHandle.standardError.write(Data(
                "error: \(analyzeFlag) does not accept '\(badFlag)'\n".utf8
            ))
            Darwin.exit(8)
        }
        let positional = rawArgs.filter { !allowedFlags.contains($0) }
        guard positional.count == 1 else {
            FileHandle.standardError.write(Data(
                "error: \(analyzeFlag) requires exactly one positional argument (path to session dir or replay_buffer.bin); got \(positional.count)\n".utf8
            ))
            Darwin.exit(9)
        }
        runAnalyzeReplayBufferAndExit(inputPath: positional[0])
    }

    private static func runAnalyzeReplayBufferAndExit(inputPath: String) -> Never {
        let expanded = (inputPath as NSString).expandingTildeInPath
        let inputURL = URL(fileURLWithPath: expanded)
        let fm = FileManager.default

        // Resolve `inputPath` to the actual `replay_buffer.bin` file. The
        // caller may pass either the session-dir (containing both the
        // session JSON and `replay_buffer.bin`) or the bin file itself —
        // both are accepted so the flag composes naturally with shell
        // tab-completion either way.
        var bufferURL = inputURL
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: inputURL.path, isDirectory: &isDir) else {
            FileHandle.standardError.write(Data(
                "error: '\(inputURL.path)' does not exist\n".utf8
            ))
            Darwin.exit(10)
        }
        if isDir.boolValue {
            bufferURL = SessionCheckpointLayout.replayBufferURL(in: inputURL)
            guard fm.fileExists(atPath: bufferURL.path) else {
                FileHandle.standardError.write(Data(
                    "error: '\(inputURL.path)' is a directory but does not contain replay_buffer.bin\n".utf8
                ))
                Darwin.exit(11)
            }
        }

        // Allocate a ReplayBuffer of exactly the file's capacity so the
        // restore neither truncates nor over-allocates.
        let buffer: ReplayBuffer
        do {
            let cap = try ReplayBuffer.peekCapacity(at: bufferURL)
            let fpb = try ReplayBuffer.peekFloatsPerBoard(at: bufferURL)
            let b = ReplayBuffer(capacity: cap, floatsPerBoard: fpb)
            try b.restore(from: bufferURL)
            buffer = b
        } catch {
            FileHandle.standardError.write(Data(
                "error: failed to load replay buffer at '\(bufferURL.path)': \(error)\n".utf8
            ))
            Darwin.exit(12)
        }

        let modelLabel = "<cli-analysis:\(bufferURL.lastPathComponent)>"
        let result = ReplayBufferAnalyzer.run(buffer: buffer, modelLabel: modelLabel)

        // JSON to stdout, summary to stderr — matches the
        // `--show-default-parameters` convention so the same UNIX-pipe
        // idioms work (`... > out.json` keeps the JSON, the summary
        // still scrolls past on the terminal).
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            let data = try encoder.encode(result)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            FileHandle.standardError.write(Data(
                "error: JSON encode failed: \(error)\n".utf8
            ))
            Darwin.exit(13)
        }

        FileHandle.standardError.write(Data(result.textSummary().utf8))
        Darwin.exit(0)
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
