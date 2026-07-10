import Foundation
import Darwin
import os

/// Cross-thread one-shot "please stop" flag for the train-vs-UCI loop.
/// The SIGINT `DispatchSource` handler flips it; the training loop reads
/// it once per step. Mirrors `CorpusReplayRunner`'s abort flag.
private final class TrainVsUciAbortFlag: @unchecked Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)
    func request() { state.withLock { $0 = true } }
    var isRequested: Bool { state.withLock { $0 } }
}

/// One opponent kind parsed from a `--train-vs-uci` spec.
struct TrainVsUciOpponentSpec: Sendable {
    /// Engine executable path.
    let command: String
    /// Number of concurrent instances of this engine.
    let count: Int
    /// The `go` limit sent every move, e.g. `"nodes 1"`, `"depth 4"`.
    let goLimit: String
    /// `setoption` pairs applied at handshake (e.g. `UCI_Elo=1400`).
    let options: [UCIArbiter.Option]
    /// Aggregation label (executable basename, e.g. `"stockfish"`).
    let kind: String
}

/// Configuration for one `--train-vs-uci` run.
struct TrainVsUciConfig: Sendable {
    var opponents: [TrainVsUciOpponentSpec]
    var stepLimit: Int?
    var timeLimitSec: Double?
    var startModelPath: String?
    var presetName: String?
    var outModelPath: String?
    var enumerateCheckpoints: Bool
    /// Max total half-moves before a game is dropped without flush.
    var maxPliesPerGame: Int
    /// How often (in trainer steps) to refresh the play network's weights
    /// from the live trainer. Small = closer to truly-live play.
    var evalSyncEverySteps: Int
    var runModelID: String
}

enum TrainVsUciError: LocalizedError {
    case noOpponents
    case startModelTooSmall(have: Int, need: Int)
    case noGamesProduced

    var errorDescription: String? {
        switch self {
        case .noOpponents:
            return "--train-vs-uci requires at least one opponent engine"
        case let .startModelTooSmall(have, need):
            return "--start-model has \(have) weight tensors but the network needs at least \(need)"
        case .noGamesProduced:
            return "no games were produced (all opponent engines failed to start or produce moves)"
        }
    }
}

/// Headless trainer that plays the live trainer network against a pool of
/// external UCI engines and trains on the resulting games — the live
/// analog of `--replay-corpus`. Invoked from the `--train-vs-uci`
/// CLI pre-flight handler.
enum TrainVsUciRunner {

    struct Result: Sendable {
        var steps: Int
        var gamesCompleted: Int
    }

    private static func emit(_ message: String) {
        SessionLogger.shared.log(message)
        print(message)
    }

    /// Run to completion and exit the process. Never returns.
    static func runAndExit(config: TrainVsUciConfig, params: ReplayParams) -> Never {
        SessionLogger.shared.start()
        emit("[VS-UCI] starting train-vs-UCI over \(config.opponents.count) opponent kind(s)")

        // Ctrl-C: first press requests a clean abort (finish the step, save,
        // exit); second press force-quits. Same pattern as CorpusReplayRunner.
        let abort = TrainVsUciAbortFlag()
        signal(SIGINT, SIG_IGN)
        let sigSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        sigSource.setEventHandler {
            if abort.isRequested {
                signal(SIGINT, SIG_DFL)
                raise(SIGINT)
                return
            }
            abort.request()
            emit("[VS-UCI] SIGINT received — finishing current step, saving, then exiting (Ctrl-C again to force-quit)")
        }
        sigSource.resume()

        let result: Result
        do {
            result = try withExtendedLifetime(sigSource) {
                try syncWait {
                    try await runTraining(config: config, params: params, abort: abort)
                }
            }
        } catch {
            FileHandle.standardError.write(Data("train-vs-uci: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[VS-UCI] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }
        emit("[VS-UCI] done: steps=\(result.steps) gamesCompleted=\(result.gamesCompleted)")
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - The run

    private static func runTraining(config: TrainVsUciConfig, params p: ReplayParams, abort: TrainVsUciAbortFlag) async throws -> Result {
        guard !config.opponents.isEmpty else { throw TrainVsUciError.noOpponents }

        // Resolve architecture from --start-model (embeds its own arch) or a
        // fresh preset / the current default.
        let arch: NetworkArchitecture
        let startModelFile: ModelCheckpointFile?
        let parentModelID: String
        if let sm = config.startModelPath {
            let url = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
            let file = try CheckpointManager.loadModelFile(at: url)
            startModelFile = file
            parentModelID = file.modelID
            arch = file.architecture
            emit("[VS-UCI] start-model: \(url.lastPathComponent) modelID=\(file.modelID) encoding=\(arch.inputEncoding.rawValue)")
        } else {
            startModelFile = nil
            parentModelID = ""
            if let pn = config.presetName {
                guard let preset = NetworkArchitecture.Preset(rawValue: pn) else {
                    let names = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
                    FileHandle.standardError.write(Data("error: unknown --preset '\(pn)'. Available: \(names)\n".utf8))
                    Darwin.exit(2)
                }
                arch = NetworkArchitecture.preset(preset)
                emit("[VS-UCI] fresh net from preset: \(pn)")
            } else {
                arch = NetworkArchitecture.current
            }
        }

        emit("[VS-UCI-ARCH] (\(startModelFile == nil ? "default preset" : "start-model")) \(arch.architectureSummary)")

        emit("[VS-UCI] building play network + trainer (encoding=\(arch.inputEncoding.rawValue))")
        // `evalNet` is the network the driver plays with. It is kept ~live by
        // syncing its weights from the trainer every `evalSyncEverySteps`
        // steps (see the loop). A separate instance from the trainer's graph
        // avoids concurrent eval/train GPU access to one network and the
        // ChessNetwork/ChessMPSNetwork type mismatch (trainer.network is a
        // ChessNetwork; the driver + ActiveGame need a ChessMPSNetwork).
        let evalNet = try ChessMPSNetwork(.randomWeights, arch: arch)
        let trainer = try ChessTrainer(
            learningRate: Float(p.learningRate),
            entropyRegularizationCoeff: Float(p.entropyBonus),
            drawPenalty: Float(p.drawPenalty),
            weightDecayC: Float(p.weightDecay),
            gradClipMaxNorm: Float(p.gradClipMaxNorm),
            policyLossWeight: Float(p.policyLossWeight),
            valueLossWeight: Float(p.valueLossWeight),
            illegalMassPenaltyWeight: Float(p.illegalMassWeight),
            policyLabelSmoothingEpsilon: Float(p.policyLabelSmoothingEpsilon),
            valueLabelSmoothingEpsilon: Float(p.valueLabelSmoothingEpsilon),
            momentumCoeff: Float(p.momentumCoeff),
            useSignedAdvantageComplementCE: p.signedAdvantageComplementCE,
            sqrtBatchScalingForLR: p.sqrtBatchScalingLR,
            lrWarmupSteps: p.lrWarmupSteps,
            arch: arch
        )
        let buffer = ReplayBuffer(capacity: p.replayBufferCapacity, inputEncoding: evalNet.inputEncoding)

        // Number of base tensors (trainables + BN running stats) — the prefix
        // both the trainer's masters/working copy and evalNet's inference net
        // are seeded from and re-synced with.
        let baseCount = evalNet.network.trainableVariables.count + evalNet.network.bnRunningStatsVariables.count

        if let file = startModelFile {
            guard file.weights.count >= baseCount else {
                throw TrainVsUciError.startModelTooSmall(have: file.weights.count, need: baseCount)
            }
            let base = Array(file.weights.prefix(baseCount))
            try await evalNet.network.loadWeights(base)
            try await trainer.loadBaseWeightsResetVelocity(base)
            emit("[VS-UCI] start-model weights loaded into trainer + play net (base tensors=\(baseCount))")
        } else {
            // Fresh run: seed evalNet from the trainer so both start identical.
            let base = Array((try await trainer.network.exportWeights()).prefix(baseCount))
            try await evalNet.network.loadWeights(base)
        }

        // Build the opponent pool: one UCIArbiter per instance.
        var opponents: [TrainVsUciDriver.Opponent] = []
        for spec in config.opponents {
            for k in 1...max(1, spec.count) {
                let label = "\(spec.kind)#\(k)"
                let arbiterConfig = UCIArbiter.Configuration(
                    command: URL(fileURLWithPath: (spec.command as NSString).expandingTildeInPath),
                    options: spec.options,
                    goLimit: spec.goLimit,
                    label: label
                )
                opponents.append(TrainVsUciDriver.Opponent(
                    arbiter: UCIArbiter(configuration: arbiterConfig),
                    kind: spec.kind,
                    instanceLabel: label))
            }
        }
        emit("[VS-UCI] opponent pool: " + config.opponents.map { "\($0.kind)×\($0.count) [go=\($0.goLimit)]" }.joined(separator: ", "))

        let driver = TrainVsUciDriver(
            network: evalNet,
            buffer: buffer,
            opponents: opponents,
            schedule: .selfPlay,
            maxPliesPerGame: config.maxPliesPerGame)

        // Rolling trainer-model output file (mirrors CorpusReplayRunner).
        let outModelURL: URL = {
            if let explicit = config.outModelPath {
                let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
                return url.pathExtension.lowercased() == "safetensors" ? url : url.appendingPathExtension("safetensors")
            }
            if let sm = config.startModelPath {
                let smURL = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
                let stem = smURL.deletingPathExtension().lastPathComponent
                return smURL.deletingLastPathComponent().appendingPathComponent("\(stem)-vsuci-latest.safetensors")
            }
            return CheckpointPaths.modelsDir.appendingPathComponent("\(config.runModelID)-vsuci-latest.safetensors")
        }()
        emit("[VS-UCI] trainer-model output: \(outModelURL.path)")

        func saveTrainerModel(step: Int, reason: String) async throws {
            let encoded: Data
            do {
                let weights = try await trainer.network.exportWeights()
                let metadata = ModelCheckpointMetadata(
                    creator: "train-vs-uci",
                    trainingStep: step,
                    parentModelID: parentModelID,
                    notes: "train-vs-uci \(reason) @ step \(step)")
                encoded = try SafetensorsModelIO.encode(
                    modelID: config.runModelID,
                    createdAtUnix: Int64(Date().timeIntervalSince1970),
                    metadata: metadata,
                    weights: weights,
                    architecture: arch,
                    includesVelocity: false,
                    resumeMetadata: [:])
                try FileManager.default.createDirectory(
                    at: outModelURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try encoded.write(to: outModelURL, options: [.atomic])
                emit("[VS-UCI] saved trainer model (\(reason)) step=\(step) -> \(outModelURL.lastPathComponent)")
            } catch {
                try CorpusReplayRunner.reportSaveFailure(error, step: step, what: "trainer-model save (\(reason))")
                return
            }
            if config.enumerateCheckpoints {
                let stem = outModelURL.deletingPathExtension().lastPathComponent
                let enumName = stem.contains("-vsuci-latest")
                    ? stem.replacingOccurrences(of: "-vsuci-latest", with: "-vsuci-step\(step)")
                    : "\(stem)-step\(step)"
                let enumURL = outModelURL.deletingLastPathComponent()
                    .appendingPathComponent(enumName).appendingPathExtension("safetensors")
                do {
                    try encoded.write(to: enumURL, options: [.atomic])
                    emit("[VS-UCI] enumerated checkpoint -> \(enumURL.lastPathComponent)")
                } catch {
                    try CorpusReplayRunner.reportSaveFailure(error, step: step, what: "enumerated checkpoint")
                }
            }
        }

        /// Refresh the play network's weights from the live trainer.
        func syncEvalNet() async throws {
            let base = Array((try await trainer.network.exportWeights()).prefix(baseCount))
            try await evalNet.network.loadWeights(base)
        }

        let batchSize = max(1, p.trainingBatchSize)
        let minPrefill = max(batchSize, p.replayBufferMinPositionsBeforeTraining)
        let syncEvery = max(1, config.evalSyncEverySteps)
        emit("[VS-UCI] batchSize=\(batchSize) minPrefill=\(minPrefill) evalSyncEvery=\(syncEvery) stepLimit=\(config.stepLimit.map(String.init) ?? "none") timeLimit=\(config.timeLimitSec.map { String(format: "%.0fs", $0) } ?? "none")")

        // Start the game producer. `driverDone` flips when driver.run()
        // returns on its own — in practice only when every engine failed to
        // launch/handshake (a healthy producer runs until cancelled) — so the
        // prefill loop below can fail fast instead of sitting out its full
        // deadline against a producer that has already given up.
        let driverDone = SyncBox<Bool>(false)
        let driverTask = Task { await driver.run(); driverDone.value = true }

        // Periodic [VS-UCI-STATS] block: per-kind summary lines, then a
        // per-instance breakdown, with rates over the window since the
        // previous emit. Runs until teardown cancels it.
        let statsIntervalSec: Double = 12
        let statsTask = Task {
            var previous: [TrainVsUciDriver.SlotStats] = []
            var lastEmit = Date()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(statsIntervalSec))
                if Task.isCancelled { break }
                let now = Date()
                let snapshot = driver.statsSnapshot()
                for line in TrainVsUciStatsFormatter.lines(
                    current: snapshot,
                    previous: previous,
                    intervalSec: now.timeIntervalSince(lastEmit)
                ) {
                    emit(line)
                }
                previous = snapshot
                lastEmit = now
            }
        }

        func dg(_ v: Float, _ digits: Int) -> String { v.isFinite ? String(format: "%.\(digits)f", v) : "--" }

        let startWall = Date()
        func elapsed() -> Double { Date().timeIntervalSince(startWall) }
        func overTime() -> Bool { if let tl = config.timeLimitSec { return elapsed() >= tl }; return false }

        // Everything from here to teardown runs with the producer live. Any
        // throw must cancel + await the driver first — `runAndExit`'s
        // `Darwin.exit` would otherwise bypass the driver's engine shutdown
        // and orphan the external UCI engine subprocesses.
        var step = 0
        var aborted = false
        do {
            // Wait for the producer to prefill the buffer (bounded, so an
            // all-engines-dead run fails fast instead of hanging forever).
            let prefillDeadline = Date().addingTimeInterval(120)
            while buffer.count < minPrefill {
                if abort.isRequested || overTime() { break }
                if driverDone.value || Date() >= prefillDeadline {
                    throw TrainVsUciError.noGamesProduced
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            emit("[VS-UCI] prefilled: bufCount=\(buffer.count)")

            // Step-locked SGD loop. Games are produced concurrently by the driver.
            let logEvery = 50
            let autosaveEvery = 1000
            while true {
                if abort.isRequested { aborted = true; emit("[VS-UCI] abort requested — stopping at step \(step)"); break }
                if let sl = config.stepLimit, step >= sl { break }
                if overTime() { emit("[VS-UCI] time limit reached at step \(step)"); break }

                // The buffer is filled asynchronously; if the producer transiently
                // falls behind, wait rather than stopping.
                if buffer.count < batchSize {
                    try await Task.sleep(for: .milliseconds(50)); continue
                }
                guard let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize) else {
                    try await Task.sleep(for: .milliseconds(50)); continue
                }
                step += 1

                // Keep the play network ~live.
                if step % syncEvery == 0 { try await syncEvalNet() }

                if step == 1 || step % logEvery == 0 {
                    let liveLR = trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: nil)
                    let line = "[VS-UCI] step=\(step)"
                        + String(format: " loss=%.4f pLoss=%.4f vLoss=%.4f", timing.loss, timing.policyLoss, timing.valueLoss)
                        + " pEnt=\(dg(timing.policyEntropy, 3)) playedP=\(dg(timing.playedMoveProb, 3))"
                        + String(format: " gNorm=%.3f lr=%.3g ms=%.1f", timing.gradGlobalNorm, liveLR, timing.totalMs)
                        + " buf=\(buffer.count)"
                    emit(line)
                }
                if step % autosaveEvery == 0 {
                    try await saveTrainerModel(step: step, reason: "autosave")
                }
            }
        } catch {
            // Tear the producer down (shuts every engine down) before the
            // error propagates to runAndExit's Darwin.exit.
            statsTask.cancel()
            driverTask.cancel()
            _ = await driverTask.value
            throw error
        }

        // Stop the producer and wait for it to shut down its engines cleanly.
        statsTask.cancel()
        driverTask.cancel()
        _ = await driverTask.value

        // Sync one last time so the saved model reflects the final weights,
        // then final save.
        try await saveTrainerModel(step: step, reason: aborted ? "abort" : "final")

        let totalGames = driver.statsSnapshot().reduce(0) { $0 + $1.gamesCompleted }
        return Result(steps: step, gamesCompleted: totalGames)
    }

    // MARK: - async→sync bridge (mirrors CorpusReplayRunner.syncWait)

    private final class SyncBoxRef<T>: @unchecked Sendable {
        var success: T?
        var failure: Error?
    }

    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = SyncBoxRef<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("TrainVsUciRunner.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}
