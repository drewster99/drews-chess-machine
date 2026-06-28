import Foundation
import Darwin
import os

/// Cross-thread one-shot "please stop" flag for the replay loop. The SIGINT
/// `DispatchSource` handler (running on a global queue) flips it; the training
/// loop (running on the detached replay task) reads it once per step. An
/// `OSAllocatedUnfairLock` — the project standard — guards the single Bool;
/// this is not on any hot path (one read per GPU step), so the lock cost is
/// irrelevant.
private final class ReplayAbortFlag: @unchecked Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)
    func request() { state.withLock { $0 = true } }
    var isRequested: Bool { state.withLock { $0 } }
}

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
    /// Optional built-in preset name (`NetworkArchitecture.Preset` rawValue, e.g.
    /// `v3_8block_3x3`) for a fresh-init run. Used only when `startModelPath` is
    /// nil; selects that architecture instead of `NetworkArchitecture.current`.
    var presetName: String?
    /// Explicit destination for the rolling trainer-model file. When nil the
    /// runner derives a path next to `--start-model` (or inside the first
    /// corpus directory). The same file is overwritten by the periodic
    /// autosave and by the final save on exit/abort.
    var outModelPath: String?
    /// Freshly-minted `ModelID` for this run's saved model, minted on the main
    /// actor in the pre-flight handler (the `ModelIDMinter` is main-actor
    /// isolated and the replay loop runs off-actor, so it can't mint there).
    var runModelID: String
}

enum CorpusReplayError: LocalizedError {
    case noGames
    case startModelTooSmall(have: Int, need: Int)
    var errorDescription: String? {
        switch self {
        case .noGames:
            return "No sealed shards found in the provided corpus path(s)"
        case let .startModelTooSmall(have, need):
            return "--start-model has \(have) weight tensors but the network needs at least \(need) (trainables + BN running stats)"
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

        // Ctrl-C handling. Install BEFORE the run so an early interrupt is
        // honored. We ignore the default SIGINT disposition (which would kill
        // the process immediately, losing the final save) and instead route
        // the signal to a DispatchSource handler on a background queue, where
        // it's safe to do real work (signal handlers proper are
        // async-signal-unsafe). First press → request a clean abort; the loop
        // breaks after the in-flight step and the final save runs. Second
        // press → restore the default disposition and re-raise, so an
        // impatient or wedged run can still be force-killed.
        let abort = ReplayAbortFlag()
        signal(SIGINT, SIG_IGN)
        let sigSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        sigSource.setEventHandler {
            if abort.isRequested {
                signal(SIGINT, SIG_DFL)
                raise(SIGINT)
                return
            }
            abort.request()
            let msg = "[REPLAY] SIGINT received — finishing current step, saving, then exiting (Ctrl-C again to force-quit)"
            print(msg)
            SessionLogger.shared.log(msg)
        }
        sigSource.resume()

        // Hold a strong reference to the dispatch source across the blocking
        // run. `sigSource` is otherwise unused after `resume()`, and in a
        // Release build ARC may shorten its lifetime to that last use and
        // deallocate it — a released signal source stops delivering, silently
        // breaking Ctrl-C while we're parked in `syncWait`.
        let result: Result
        do {
            result = try withExtendedLifetime(sigSource) {
                try syncWait {
                    try await runReplay(config: config, params: params, abort: abort)
                }
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

    private static func runReplay(config: CorpusReplayConfig, params p: ReplayParams, abort: ReplayAbortFlag) async throws -> Result {
        // --start-model: load a saved model and continue training from it. The
        // file embeds its own architecture, which then drives both the trainer
        // and the feeder net — a start model of a different shape than the
        // current default preset trains correctly. nil → fresh random net at
        // the current preset.
        let arch: NetworkArchitecture
        let startModelFile: ModelCheckpointFile?
        let parentModelID: String
        if let sm = config.startModelPath {
            let url = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
            let file = try CheckpointManager.loadModelFile(at: url)
            startModelFile = file
            parentModelID = file.modelID
            arch = file.architecture
            SessionLogger.shared.log("[REPLAY] start-model: \(url.lastPathComponent) modelID=\(file.modelID) encoding=\(arch.inputEncoding.rawValue)")
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
                SessionLogger.shared.log("[REPLAY] fresh net from preset: \(pn)")
            } else {
                arch = NetworkArchitecture.current
            }
        }

        // Startup banner: make the network type and the training hyperparameters
        // explicit in the log so a replay run is self-documenting (otherwise the
        // only clue was the input encoding). `architectureSummary` is the
        // fully-explicit form — version, encoding, block groups, heads, compute
        // dtype, and parameter count, with no silent defaults.
        let archSource = startModelFile == nil ? "default preset" : "start-model"
        SessionLogger.shared.log("[REPLAY-ARCH] (\(archSource)) \(arch.architectureSummary)")
        // Numeric knobs via String(format:) (%ld for Int, %g for Double); the
        // two on/off flags are interpolated rather than passed through %@ (Swift
        // String + %@ relies on NSString bridging — avoid it).
        let hparamsLine = String(
            format: "[REPLAY-HPARAMS] lr=%.6g batch=%ld wd=%.4g momentum=%.3g gradClip=%.3g entropyBonus=%.4g drawPenalty=%.4g policyW=%.3g valueW=%.3g illegalW=%.4g pLabelSmooth=%.4g vLabelSmooth=%.4g lrWarmup=%ld bufCap=%ld replayRatio=%.3g minPrefill=%ld",
            p.learningRate, p.trainingBatchSize, p.weightDecay, p.momentumCoeff, p.gradClipMaxNorm,
            p.entropyBonus, p.drawPenalty, p.policyLossWeight, p.valueLossWeight, p.illegalMassWeight,
            p.policyLabelSmoothingEpsilon, p.valueLabelSmoothingEpsilon,
            p.lrWarmupSteps, p.replayBufferCapacity, p.replayRatioTarget,
            p.replayBufferMinPositionsBeforeTraining
        )
            + " complementCE=\(p.signedAdvantageComplementCE ? "on" : "off")"
            + " sqrtBatchLR=\(p.sqrtBatchScalingLR ? "on" : "off")"
        SessionLogger.shared.log(hparamsLine)
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

        // Seed both the trainer (the network that actually learns) and the
        // feeder net (computes the value baseline while feeding) from the start
        // model's base weights — exactly trainables + BN running stats. A
        // trainer source file carries optimizer velocity after that block, so
        // take the leading base prefix (same rule as ProbeModelCLI /
        // UCIModelLoader).
        //
        // The trainer must be seeded via `loadBaseWeightsResetVelocity`, NOT a
        // bare `network.loadWeights`: under the canonical bf16 mixed-precision
        // path the optimizer steps the fp32 *master* weights and re-derives the
        // bf16 working copy from them each step. Writing only the working copy
        // would leave the masters at random init, and the first SGD step would
        // overwrite our loaded weights with that random surface. This call
        // writes the working copy, seeds the fp32 masters from the same values,
        // and zeros optimizer velocity (a fresh fork — momentum re-accumulates).
        // The feeder net is a plain inference network (no masters/velocity), so
        // a direct `loadWeights` is correct there.
        if let file = startModelFile {
            let baseCount = net.network.trainableVariables.count + net.network.bnRunningStatsVariables.count
            guard file.weights.count >= baseCount else {
                throw CorpusReplayError.startModelTooSmall(have: file.weights.count, need: baseCount)
            }
            let base = Array(file.weights.prefix(baseCount))
            try await net.network.loadWeights(base)
            try await trainer.loadBaseWeightsResetVelocity(base)
            SessionLogger.shared.log("[REPLAY] start-model weights loaded into trainer (working+masters, velocity zeroed) + feeder net (base tensors=\(baseCount))")
        }

        // Rolling trainer-model output file. The same file is overwritten by
        // the periodic autosave and by the final save on exit/abort, so it
        // always holds the latest weights. Destination precedence: explicit
        // --out-model; else next to --start-model; else the app's Models
        // directory named after the corpus. Overwrite is deliberate here (the
        // CheckpointManager never-overwrite history rule is for the curated
        // Models/Sessions store) — this is a single "latest" convenience file.
        let outModelURL: URL = {
            if let explicit = config.outModelPath {
                // Always land on a `.safetensors` extension. The file is
                // safetensors-encoded, and the loaders that consume it
                // (--probe-model, --start-model) key off the extension — a
                // bare name like `corp1model` would be written verbatim and
                // then rejected as "no .safetensors found". Append it when the
                // caller didn't supply it (a supplied `.safetensors` is kept).
                let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
                return url.pathExtension.lowercased() == "safetensors"
                    ? url
                    : url.appendingPathExtension("safetensors")
            }
            if let sm = config.startModelPath {
                let smURL = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
                let stem = smURL.deletingPathExtension().lastPathComponent
                return smURL.deletingLastPathComponent().appendingPathComponent("\(stem)-replay-latest.safetensors")
            }
            // No --start-model: default into the app's Models directory — always
            // writable (even when the corpus is a read-only mounted volume),
            // keeps the corpus data dir pristine, and lands where --probe-model
            // and the GUI already look. Named after the corpus so runs over
            // different corpora don't collide on one file.
            let corpusName = config.corpusDirectories[0].lastPathComponent
            return CheckpointPaths.modelsDir.appendingPathComponent("\(corpusName)-replay-latest.safetensors")
        }()
        SessionLogger.shared.log("[REPLAY] trainer-model output: \(outModelURL.path)")

        // Export the trainer's current base weights and overwrite the rolling
        // output file. Failures here are logged but non-fatal: a convenience
        // autosave that can't write (e.g. a read-only corpus volume) must not
        // tear down an otherwise-healthy training run. Pass --out-model to a
        // writable location if the derived path can't be written.
        func saveTrainerModel(step: Int, reason: String) async {
            do {
                let weights = try await trainer.network.exportWeights()
                let metadata = ModelCheckpointMetadata(
                    creator: "replay",
                    trainingStep: step,
                    parentModelID: parentModelID,
                    notes: "corpus replay \(reason) @ step \(step)"
                )
                let encoded = try SafetensorsModelIO.encode(
                    modelID: config.runModelID,
                    createdAtUnix: Int64(Date().timeIntervalSince1970),
                    metadata: metadata,
                    weights: weights,
                    architecture: arch,
                    includesVelocity: false
                )
                try FileManager.default.createDirectory(
                    at: outModelURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try encoded.write(to: outModelURL, options: [.atomic])
                let msg = "[REPLAY] saved trainer model (\(reason)) step=\(step) -> \(outModelURL.lastPathComponent)"
                print(msg)
                SessionLogger.shared.log(msg)
            } catch {
                let msg = "[REPLAY] WARNING: trainer-model save (\(reason)) failed at step \(step): \(error.localizedDescription)"
                FileHandle.standardError.write(Data((msg + "\n").utf8))
                SessionLogger.shared.log(msg)
            }
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

        // Format a possibly-not-measured diagnostic. The trainer only computes
        // the diagnostic bundle (entropy, value W/D/L, played-move prob, illegal
        // mass) on its diagnostic-cadence steps, leaving the field NaN otherwise
        // (e.g. the first logged step). Render those as "--" rather than "nan"
        // so the line stays readable.
        func dg(_ v: Float, _ digits: Int) -> String {
            v.isFinite ? String(format: "%.\(digits)f", v) : "--"
        }

        // Step-locked SGD loop.
        var step = 0
        let logEvery = 50
        let autosaveEvery = 1000
        var aborted = false
        while true {
            // Ctrl-C: stop cleanly before starting another step so the
            // post-loop save captures a complete, non-mid-step state.
            if abort.isRequested {
                aborted = true
                SessionLogger.shared.log("[REPLAY] abort requested — stopping at step \(step)")
                break
            }
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
                // Live, warmup-adjusted LR read from the trainer (single source
                // of truth — don't re-derive the warmup formula here).
                let liveLR = trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: nil)
                let line = "[REPLAY] step=\(step)"
                    + String(format: " loss=%.4f pLoss=%.4f vLoss=%.4f", timing.loss, timing.policyLoss, timing.valueLoss)
                    + " pEnt=\(dg(timing.policyEntropy, 3)) pIllM=\(dg(timing.illegalMassPenalty, 4))"
                    + " playedP=\(dg(timing.playedMoveProb, 3))"
                    + " pW=\(dg(timing.valueProbWin, 2)) pD=\(dg(timing.valueProbDraw, 2)) pL=\(dg(timing.valueProbLoss, 2)) vAbs=\(dg(timing.valueAbsMean, 3))"
                    + String(format: " gNorm=%.3f lr=%.3g ms=%.1f", timing.gradGlobalNorm, liveLR, timing.totalMs)
                    + " buf=\(buffer.count) plies=\(positionsFed) games=\(gamesFed) epoch=\(epochsCompleted)"
                SessionLogger.shared.log(line)
                print(line)
            }
            // Periodic autosave (overwrites the rolling output file).
            if step % autosaveEvery == 0 {
                await saveTrainerModel(step: step, reason: "autosave")
            }
        }

        // Final save on any clean exit path — step/epoch limit, corpus
        // exhaustion, or Ctrl-C abort. A thrown error skips this (it propagates
        // out of runReplay before we get here): the network state after a hard
        // failure isn't worth persisting over the last good autosave.
        await saveTrainerModel(step: step, reason: aborted ? "abort" : "final")

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
