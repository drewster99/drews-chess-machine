import Foundation

/// Plain, `Sendable` snapshot of the training parameters an offline replay run
/// needs. Captured on the main actor (from `TrainingParameters.shared`) before
/// the run starts, then passed into the off-actor GPU work — so the detached
/// replay task never touches the `@MainActor` singleton (which would deadlock
/// against the `syncWait` semaphore held on the main thread).
struct ReplayParams: Sendable {
    var learningRate: Double
    var entropyBonus: Double
    var drawPenalty: Double
    var weightDecay: Double
    var gradClipMaxNorm: Double
    var policyLossWeight: Double
    var valueLossWeight: Double
    var illegalMassWeight: Double
    var policyLabelSmoothingEpsilon: Double
    var valueLabelSmoothingEpsilon: Double
    var momentumCoeff: Double
    var signedAdvantageComplementCE: Bool
    var sqrtBatchScalingLR: Bool
    var lrWarmupSteps: Int
    var trainingBatchSize: Int
    var replayBufferCapacity: Int
    var replayRatioTarget: Double
    var replayBufferMinPositionsBeforeTraining: Int
}

/// Configuration for one offline corpus-replay run.
struct CorpusReplayConfig: Sendable {
    var corpusDirectories: [URL]
    var stepLimit: Int?
    var epochs: Int?
    var startModelPath: String?
}

enum CorpusReplayError: LocalizedError {
    case noGames
    var errorDescription: String? {
        switch self {
        case .noGames: return "No sealed shards found in the provided corpus path(s)"
        }
    }
}

/// Headless offline trainer: builds a network + trainer, fills the
/// `ReplayBuffer` from a fixed game corpus (no self-play, no arena, no
/// promotion), runs a step-locked SGD loop to a budget, and exits. Invoked
/// from the `--replay-corpus` CLI pre-flight handler.
enum CorpusReplayRunner {

    struct Result: Sendable {
        var steps: Int
        var positionsFed: Int
        var gamesFed: Int
        var epochs: Int
    }

    /// Run the replay to completion and exit the process. Never returns.
    static func runAndExit(config: CorpusReplayConfig, params: ReplayParams) -> Never {
        SessionLogger.shared.start()
        SessionLogger.shared.log(
            "[REPLAY] starting offline corpus replay over \(config.corpusDirectories.count) corpus path(s)"
        )
        let result: Result
        do {
            result = try syncWait {
                try await runReplay(config: config, params: params)
            }
        } catch {
            FileHandle.standardError.write(Data("replay: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[REPLAY] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }
        let summary = "[REPLAY] done: steps=\(result.steps) positionsFed=\(result.positionsFed) gamesFed=\(result.gamesFed) epochs=\(result.epochs)"
        print(summary)
        SessionLogger.shared.log(summary)
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - The run

    private static func runReplay(config: CorpusReplayConfig, params p: ReplayParams) async throws -> Result {
        let arch = NetworkArchitecture.current
        SessionLogger.shared.log("[REPLAY] building network + trainer (encoding=\(arch.inputEncoding.rawValue))")
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
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
        let buffer = ReplayBuffer(capacity: p.replayBufferCapacity, inputEncoding: net.inputEncoding)
        let feeder = CorpusReplayFeeder(network: net, buffer: buffer)

        if let sm = config.startModelPath {
            SessionLogger.shared.log("[REPLAY] note: --start-model (\(sm)) is not yet wired for replay mode; using a fresh random net")
        }

        // Gather sealed shard URLs across all corpora, in stable order.
        var shardURLs: [URL] = []
        for dir in config.corpusDirectories {
            let corpus = try GameCorpus.open(directory: dir)
            let urls = try corpus.sealedShardURLs()
            shardURLs.append(contentsOf: urls)
            SessionLogger.shared.log("[REPLAY] corpus \(corpus.corpusID): \(urls.count) sealed shard(s)")
        }
        guard !shardURLs.isEmpty else { throw CorpusReplayError.noGames }

        let batchSize = max(1, p.trainingBatchSize)
        let reuse = max(0.01, p.replayRatioTarget)
        // Positions to feed per step so each is sampled ~`reuse` times before
        // eviction: K = batchSize / R.
        let perStepFeed = max(1, Int((Double(batchSize) / reuse).rounded()))
        let minPrefill = max(batchSize, p.replayBufferMinPositionsBeforeTraining)

        // Budget: explicit step limit wins; otherwise bound by epochs (default
        // a single pass when neither is given).
        let stepLimit = config.stepLimit
        let epochLimit: Int? = config.epochs ?? (stepLimit == nil ? 1 : nil)

        SessionLogger.shared.log(
            "[REPLAY] batchSize=\(batchSize) reuse=\(String(format: "%.2f", reuse)) K=\(perStepFeed) minPrefill=\(minPrefill) stepLimit=\(stepLimit.map(String.init) ?? "none") epochLimit=\(epochLimit.map(String.init) ?? "none")"
        )

        // Streaming game source, cycling the shard list for epochs.
        var shardCursor = 0
        var currentGames: [GameRecord] = []
        var gameCursor = 0
        var epochsCompleted = 0

        func nextGame() -> GameRecord? {
            while gameCursor >= currentGames.count {
                if shardCursor >= shardURLs.count {
                    epochsCompleted += 1
                    if let el = epochLimit, epochsCompleted >= el { return nil }
                    shardCursor = 0
                }
                let url = shardURLs[shardCursor]
                shardCursor += 1
                do {
                    currentGames = try GameCorpusShardIO.readSealed(at: url).games
                } catch {
                    SessionLogger.shared.log("[REPLAY] skipping unreadable shard \(url.lastPathComponent): \(error.localizedDescription)")
                    currentGames = []
                }
                gameCursor = 0
            }
            let g = currentGames[gameCursor]
            gameCursor += 1
            return g
        }

        var positionsFed = 0
        var gamesFed = 0
        var corpusExhausted = false

        // Pre-fill.
        while buffer.count < minPrefill {
            guard let g = nextGame() else { corpusExhausted = true; break }
            positionsFed += feeder.feed(g)
            gamesFed += 1
        }
        let prefillPositions = positionsFed
        SessionLogger.shared.log("[REPLAY] pre-filled: bufCount=\(buffer.count) positionsFed=\(positionsFed) gamesFed=\(gamesFed)")

        // Step-locked SGD loop.
        var step = 0
        let logEvery = 50
        while true {
            if let sl = stepLimit, step >= sl { break }
            let targetFed = prefillPositions + step * perStepFeed
            while positionsFed < targetFed && !corpusExhausted {
                guard let g = nextGame() else { corpusExhausted = true; break }
                positionsFed += feeder.feed(g)
                gamesFed += 1
            }
            if corpusExhausted && epochLimit != nil { break }

            guard let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize) else {
                SessionLogger.shared.log("[REPLAY] trainStep returned nil (bufCount=\(buffer.count)); stopping")
                break
            }
            step += 1
            if step == 1 || step % logEvery == 0 {
                let line = String(
                    format: "[REPLAY] step=%d loss=%.4f pLoss=%.4f vLoss=%.4f pEnt=%.3f gNorm=%.3f buf=%d fed=%d games=%d epoch=%d",
                    step, timing.loss, timing.policyLoss, timing.valueLoss, timing.policyEntropy,
                    timing.gradGlobalNorm, buffer.count, positionsFed, gamesFed, epochsCompleted
                )
                SessionLogger.shared.log(line)
                print(line)
            }
        }

        return Result(steps: step, positionsFed: positionsFed, gamesFed: gamesFed, epochs: epochsCompleted)
    }

    // MARK: - async→sync bridge (mirrors SweepCLI.syncWait)

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
            preconditionFailure("CorpusReplayRunner.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}
