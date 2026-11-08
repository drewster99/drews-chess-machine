import Darwin
import Foundation

/// `--sharpness --sharpness-model <file> [--sharpness-model <file> …]
/// --sharpness-corpus <corpus id or folder> [options]`: the top eigenvalue
/// λmax of the training loss's Hessian at saved trainer checkpoints, by power
/// iteration on finite-difference Hessian-vector products, one JSON line per
/// checkpoint (and per parameter group with `--sharpness-groups`), then exit
/// (no GUI). Read-only: nothing is written but the optional `--sharpness-out`
/// file and the session log.
///
/// Why it exists (`documentation/plans-active/HELDOUT_EVAL_AND_SHARPNESS_PLAN.md`,
/// Phase 2): R-elitecyc's pre-clip gradient norm rose from 0.40 to 0.92 after
/// its learning rate fell, more than its weights' shrinkage explains. A loss
/// surface that sharpened as the LR fell (the "edge of stability" picture:
/// λmax climbs toward the step size's stability threshold 2(1 + μ)/LR) would
/// explain the rest. λmax at checkpoints across the run, read against that
/// threshold at each checkpoint's own trainer step, tests it.
///
/// **The gradient is the trainer's own.** No second loss definition: each
/// gradient g(w) is one `ChessTrainer.trainStep` of a trainer built in this
/// process (never a live trainer) from the checkpoint's architecture and
/// configured from the checkpoint's own lineage parameter snapshot through
/// `ReplayParams` / `TrainerHyperparameters`, the path an exact corpus-replay
/// resume configures its trainer by — same loss terms and weights, label
/// smoothing, training-mode BatchNorm, value baseline, sampling constraints.
/// Before every step the weights are loaded with the optimizer velocity
/// zeroed (`loadBaseWeightsResetVelocity`); the step computes
/// v_new = μ·0 + clip·g and the probe reads v_new back
/// (`exportVelocitySnapshot`). Weight decay is decoupled and never enters the
/// velocity, and LR only scales the weight update the next load discards, so
/// v_new is exactly g whatever μ, LR or decay are — provided clip = 1. The
/// probe therefore disables clipping: the hard max is set to
/// `Float.greatestFiniteMagnitude` (the graph's `cap / max(norm, cap)` is then
/// exactly 1) and the relative cap's mode to `off`; every step's own
/// `GradientCapDecision` is checked and a clipped step is an error.
///
/// **A fixed batch.** Each of the K batches has its own named sampler stream
/// (`probe.sharpness.batch.<k>.0`); the replay buffer's sampler is restored to
/// it before every step on that batch, so every evaluation of batch k draws
/// the same positions. With dropout on, the graph's Philox state is captured
/// once and restored before every step (the same masks); with dropout off
/// there is nothing to fix. The buffer is filled once per checkpoint with the
/// corpus-replay feeder (`CorpusReplayFeeder`) from a fixed corpus position
/// (`--sharpness-corpus-start-game`, default 0, the corpus's first game), so
/// every checkpoint is probed on the same positions — the comparison across
/// checkpoints is the point. The streams use trainer step 0 in their names for
/// the same reason. Determinism is verified, not assumed: batch 0's gradient
/// is evaluated twice at w and the difference reported (`determinism`).
///
/// **Precision.** The checkpoints compute in bf16 (~3 significant digits); a
/// finite-difference perturbation that small would round away in the working
/// weights. The probe trainer is built in fp32 compute by default
/// (`NetworkArchitecture.withComputeDataType(.float32, tail: nil)`), loading
/// the file's fp32 master weights exactly; `--sharpness-compute bf16` builds
/// it at the checkpoint's own bf16 architecture for comparison. The compute
/// used is in every record.
///
/// **What is computed.** ‖v‖ = 1 over the trainable tensors (the ones the
/// velocity covers); ε = `--sharpness-epsilon` × ‖w‖ (‖w‖ over the probed
/// group's tensors). Hv = mean over the K batches of
/// (g_k(w + εv) − g_k(w − εv)) / 2ε, i.e. the Hessian of the mean of the K
/// batch losses (each batch's BatchNorm uses its own batch statistics, as in
/// training). Power iteration from a seeded Gaussian start
/// (`probe.sharpness.direction.<scope>.0`), v ← Hv/‖Hv‖, Rayleigh quotient
/// λ = vᵀHv each iteration, stopping after `--sharpness-iterations` or when two
/// consecutive λ agree to `--sharpness-tolerance`. Power iteration finds the
/// eigenvalue of largest magnitude; a negative `lambda_max` means negative
/// curvature dominates. With `--sharpness-groups` it is repeated with v
/// restricted to each `SharpnessParameterGroup` (the block of the Hessian on
/// that group's tensors).
///
/// **Checks in every record.** `epsilon_check`: λ on the final vector at ε
/// and at 2ε (central differences agree to O(ε²) when ε is in the linear
/// range). `symmetry_check`: uᵀJv against vᵀJu for the random start u and the
/// final v. The value baseline is computed by a forward pass of the current
/// weights and fed to the loss as a constant (a placeholder, no gradient), so
/// the finite-difference operator J is the Jacobian of the training update
/// field — what governs the stability of the training iteration — and equals
/// the loss Hessian plus the (non-symmetric) term from the baseline's
/// dependence on w. The symmetry check measures how much that term matters.
///
/// Also reported: ‖g‖ per batch (the graph's own pre-clip norm, comparable to
/// the run's logged `gNorm`, and the read-back velocity's norm), ‖mean g‖,
/// ‖w‖, the checkpoint's trainer step, the LR and momentum the checkpoint's
/// recorded schedule feeds at that step (`LRMomentumCycleReadout`), the
/// stability threshold 2(1 + μ)/LR, gradient evaluations and elapsed time.
///
/// GPU faults: the run watches this process's GPU faults
/// (`GPUFaultWatch`); a barrier after every checkpoint turns a fault into an
/// error for that checkpoint and ends the run, since the gradients may be
/// garbage.
///
/// Threading: argument parsing, the corpus read, the buffer fill and all
/// vector math run on the launching (main) thread; each GPU operation (a
/// gradient evaluation, a weight load) is one `runBlocking`, whose work runs
/// on the trainer's own serial queue behind its continuations, so no long
/// synchronous work runs inside a `Task`.
enum SharpnessCLI {

    static let modeFlag = "--sharpness"
    static let modelFlag = "--sharpness-model"
    static let corpusFlag = "--sharpness-corpus"
    static let corpusStartGameFlag = "--sharpness-corpus-start-game"
    static let bufferPositionsFlag = "--sharpness-buffer-positions"
    static let batchesFlag = "--sharpness-batches"
    static let iterationsFlag = "--sharpness-iterations"
    static let toleranceFlag = "--sharpness-tolerance"
    static let epsilonFlag = "--sharpness-epsilon"
    static let seedFlag = "--sharpness-seed"
    static let computeFlag = "--sharpness-compute"
    static let groupsFlag = "--sharpness-groups"
    static let footprintLimitFlag = "--sharpness-max-footprint-gb"
    static let outFlag = "--sharpness-out"
    /// `--sharpness-out` refuses an existing file unless this flag is also
    /// given (the `--probe-out-overwrite` rule).
    static let replaceOutFlag = "--sharpness-out-overwrite"

    static let defaultBatchCount = 4
    static let defaultIterationCount = 30
    static let defaultTolerance = 1e-3
    static let defaultRelativeEpsilon = 1e-3
    /// ~480 elite games (~0.25 GB of basic24 boards): the fewest that hold
    /// R-elitecyc's 4096-position batch under its 10-positions-per-game cap
    /// (≥ 410 games) with a margin. A smaller buffer cannot draw the batch.
    static let defaultBufferPositions = 40_000
    /// The process footprint (`phys_footprint`) above which the probe stops,
    /// in GiB: it shares the machine with a live trainer, and a first run
    /// grew to 38 GB (graphics memory) over ~70 gradient evaluations.
    static let defaultFootprintLimitGiB = 6.0
    static let defaultSeed: UInt64 = 0

    /// The probe's compute precision.
    enum Compute: String, Equatable {
        /// fp32 compute, the default (see the type doc: bf16 rounds tiny
        /// perturbations away).
        case float32 = "fp32"
        /// The checkpoint's own bf16 architecture, unchanged.
        case bFloat16 = "bf16"
    }

    private enum ExitCode: Int32 {
        case usage = 110
        case corpusUnusable = 111
        case modelPathUnusable = 112
        case outputUnusable = 113
        case outputWriteFailed = 114
        case modelsFailed = 115
        case gpuFault = 116
        case footprintLimit = 117
    }

    // MARK: - Options

    struct Options: Equatable {
        var modelPaths: [String]
        var corpusArgument: String
        var corpusStartGame: Int
        var bufferPositions: Int
        var batchCount: Int
        var iterationCount: Int
        var tolerance: Double
        var relativeEpsilon: Double
        var seed: UInt64
        var compute: Compute
        var perGroup: Bool
        /// Stop the run when the process footprint exceeds this.
        var footprintLimitBytes: UInt64
        var outPath: String?
        var replaceExistingOut: Bool
    }

    struct UsageError: LocalizedError, Equatable {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// Parse the launch arguments (without the executable path). Every `--`
    /// token must be one of this mode's flags; `--sharpness-model` may
    /// repeat, the others may appear once. A system-supplied `-Key value`
    /// pair (Xcode's `-NSDocumentRevisionsDebugMode YES`) is skipped.
    static func parseOptions(_ rawArgs: [String]) throws -> Options {
        let valueFlags: Set<String> = [modelFlag, corpusFlag, corpusStartGameFlag, bufferPositionsFlag, batchesFlag,
                                       iterationsFlag, toleranceFlag, epsilonFlag, seedFlag, computeFlag, footprintLimitFlag,
                                       outFlag]
        let booleanFlags: Set<String> = [modeFlag, groupsFlag, replaceOutFlag]
        var values: [String: [String]] = [:]
        var present: Set<String> = []
        var index = 0
        while index < rawArgs.count {
            let token = rawArgs[index]
            if valueFlags.contains(token) {
                let valueIndex = index + 1
                guard valueIndex < rawArgs.count, !rawArgs[valueIndex].hasPrefix("--") else {
                    throw UsageError("\(token) requires a value")
                }
                values[token, default: []].append(rawArgs[valueIndex])
                index += 2
            } else if booleanFlags.contains(token) {
                guard present.insert(token).inserted else { throw UsageError("\(token) given more than once") }
                index += 1
            } else if token.hasPrefix("--") {
                throw UsageError("\(modeFlag) does not accept '\(token)'")
            } else if token.hasPrefix("-") {
                index += (index + 1 < rawArgs.count && !rawArgs[index + 1].hasPrefix("-")) ? 2 : 1
            } else {
                throw UsageError("unexpected argument '\(token)' (each model needs its own \(modelFlag))")
            }
        }
        for flag in valueFlags where flag != modelFlag && (values[flag]?.count ?? 0) > 1 {
            throw UsageError("\(flag) given more than once")
        }
        guard let models = values[modelFlag], !models.isEmpty else {
            throw UsageError("\(modeFlag) needs at least one \(modelFlag) <trainer-state .safetensors file>")
        }
        guard let corpus = values[corpusFlag]?.first else {
            throw UsageError("\(modeFlag) needs \(corpusFlag) <corpus id or folder> (the positions the gradients are taken on)")
        }
        func positiveInteger(_ flag: String, default fallback: Int) throws -> Int {
            guard let raw = values[flag]?.first else { return fallback }
            guard let parsed = Int(raw), parsed > 0 else {
                throw UsageError("\(flag) must be a positive whole number; got '\(raw)'")
            }
            return parsed
        }
        func positiveReal(_ flag: String, default fallback: Double) throws -> Double {
            guard let raw = values[flag]?.first else { return fallback }
            guard let parsed = Double(raw), parsed.isFinite, parsed > 0 else {
                throw UsageError("\(flag) must be a positive number; got '\(raw)'")
            }
            return parsed
        }
        var startGame = 0
        if let raw = values[corpusStartGameFlag]?.first {
            guard let parsed = Int(raw), parsed >= 0 else {
                throw UsageError("\(corpusStartGameFlag) must be a whole number ≥ 0; got '\(raw)'")
            }
            startGame = parsed
        }
        var seed = defaultSeed
        if let raw = values[seedFlag]?.first {
            guard let parsed = UInt64(strictDecimal: raw) else {
                throw UsageError("\(seedFlag) must be a decimal whole number 0…\(UInt64.max); got '\(raw)'")
            }
            seed = parsed
        }
        var compute = Compute.float32
        if let raw = values[computeFlag]?.first {
            guard let parsed = Compute(rawValue: raw) else {
                throw UsageError("\(computeFlag) must be \(Compute.float32.rawValue) or \(Compute.bFloat16.rawValue); got '\(raw)'")
            }
            compute = parsed
        }
        let outPath = values[outFlag]?.first
        let replaceExistingOut = present.contains(replaceOutFlag)
        if replaceExistingOut && outPath == nil {
            throw UsageError("\(replaceOutFlag) needs \(outFlag) <file>")
        }
        let tolerance = try positiveReal(toleranceFlag, default: defaultTolerance)
        let relativeEpsilon = try positiveReal(epsilonFlag, default: defaultRelativeEpsilon)
        let footprintLimitGiB = try positiveReal(footprintLimitFlag, default: defaultFootprintLimitGiB)
        guard relativeEpsilon < 1 else {
            throw UsageError("\(epsilonFlag) is relative to ‖w‖ and must be below 1; got \(relativeEpsilon)")
        }
        return Options(
            modelPaths: models, corpusArgument: corpus, corpusStartGame: startGame,
            bufferPositions: try positiveInteger(bufferPositionsFlag, default: defaultBufferPositions),
            batchCount: try positiveInteger(batchesFlag, default: defaultBatchCount),
            iterationCount: try positiveInteger(iterationsFlag, default: defaultIterationCount),
            tolerance: tolerance, relativeEpsilon: relativeEpsilon, seed: seed, compute: compute,
            perGroup: present.contains(groupsFlag),
            footprintLimitBytes: UInt64(footprintLimitGiB * 1024 * 1024 * 1024),
            outPath: outPath, replaceExistingOut: replaceExistingOut)
    }

    /// Never returns when `--sharpness` is on the command line. Main actor:
    /// the corpus lookup it shares with `--replay-corpus`
    /// (`DrewsChessMachineApp.resolveCorpusDirectory`) is main-actor isolated.
    @MainActor
    static func handleIfRequested(rawArgs: [String]) {
        guard rawArgs.contains(modeFlag) else { return }
        let options: Options
        do {
            options = try parseOptions(rawArgs)
        } catch {
            fail("error: \(error.localizedDescription)", code: .usage, loggerStarted: false)
        }
        runAndExit(options)
    }

    // MARK: - Run

    @MainActor
    private static func runAndExit(_ options: Options) -> Never {
        SessionLogger.shared.start()

        let models: [URL]
        do {
            models = try options.modelPaths.map(resolveModel)
        } catch {
            fail("error: \(error.localizedDescription)", code: .modelPathUnusable)
        }
        guard let corpusDirectory = DrewsChessMachineApp.resolveCorpusDirectory(options.corpusArgument) else {
            fail("error: corpus not found for '\(options.corpusArgument)' — no corpus.json at that path or under "
                 + CorpusPaths.corporaDir.path, code: .corpusUnusable)
        }
        let corpus: GameCorpusReadOnlyView
        let games: CorpusGames
        do {
            corpus = try GameCorpus.openReadOnly(directory: corpusDirectory)
            games = try loadGames(of: corpus, startGame: options.corpusStartGame, positionBudget: options.bufferPositions)
        } catch {
            fail("error: corpus \(corpusDirectory.path): \(error.localizedDescription)", code: .corpusUnusable)
        }
        if !corpus.ignoredOpenShardURLs.isEmpty {
            progress("WARNING corpus \(corpus.corpusID): ignoring \(corpus.ignoredOpenShardURLs.count) unsealed .open shard(s)")
        }
        let output: FileSafety.OpenedForWriting?
        do {
            output = try openOutput(path: options.outPath, models: models, replaceExisting: options.replaceExistingOut)
        } catch {
            fail("error: \(error.localizedDescription)", code: .outputUnusable)
        }
        progress("corpus \(corpus.corpusID): \(games.records.count) game(s) from game \(options.corpusStartGame) "
                 + "(\(games.moveCount) recorded moves) for a \(options.bufferPositions)-position buffer; "
                 + "\(models.count) model(s); compute \(options.compute.rawValue); K=\(options.batchCount) "
                 + "N=\(options.iterationCount) tol=\(options.tolerance) eps=\(options.relativeEpsilon)·‖w‖ seed=\(options.seed)")

        func emit(_ line: Data) {
            guard let text = String(data: line, encoding: .utf8) else {
                fail("error: an output line is not UTF-8", code: .outputWriteFailed)
            }
            print(text)
            fflush(stdout)
            guard let output else { return }
            do {
                try ProbeModelCLI.appendLine(line, to: output.handle)
            } catch {
                fail("error: write to \(outFlag) failed: \(error.localizedDescription); the file may end in a partial line",
                     code: .outputWriteFailed)
            }
        }
        func emitFailure(model: URL, _ message: String) {
            do {
                emit(try ProbeModelCLI.encodeLine(["event": "error", "model": model.path, "error": message]))
            } catch {
                fail("error: could not encode the failure record for \(model.path): \(error.localizedDescription)",
                     code: .outputWriteFailed)
            }
        }

        let faultWatch = GPUFaultWatch.startForRun()
        var failedModels: [String] = []
        for model in models {
            let started = DispatchTime.now()
            var lines: [Data] = []
            var failure: String?
            do {
                let records = try probe(modelURL: model, options: options, corpus: corpus, games: games, started: started)
                lines = try records.map(ProbeModelCLI.encodeLine)
            } catch let limit as SharpnessError {
                if case .footprintLimitExceeded = limit {
                    emitFailure(model: model, limit.localizedDescription)
                    fail("error: \(limit.localizedDescription); stopping", code: .footprintLimit)
                }
                failure = limit.localizedDescription
            } catch {
                failure = error.localizedDescription
            }
            // A GPU fault in this process may have corrupted any gradient
            // since the run began: the checkpoint's records are withheld and
            // the run ends.
            let fault: GPUFaultLedger.Fault?
            do {
                fault = try runBlocking { await faultWatch.barrier() }
            } catch {
                fail("error: the GPU fault barrier failed: \(error.localizedDescription)", code: .gpuFault)
            }
            if let fault {
                emitFailure(model: model, "GPU fault during the probe (\(fault.summary)); results withheld")
                fail("GPU fault: \(fault.summary); stopping", code: .gpuFault)
            }
            if let failure {
                failedModels.append(model.path)
                emitFailure(model: model, failure)
                continue
            }
            lines.forEach(emit)
        }
        if let output, let outPath = options.outPath {
            do {
                try output.handle.close()
            } catch {
                fail("error: closing \(outFlag) \(outPath): \(error.localizedDescription)", code: .outputWriteFailed)
            }
        }
        guard failedModels.isEmpty else {
            fail("\(failedModels.count) of \(models.count) model(s) failed: " + failedModels.joined(separator: ", "),
                 code: .modelsFailed)
        }
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    private static func progress(_ message: String) {
        FileHandle.standardError.write(Data("[SHARPNESS] \(message)\n".utf8))
        SessionLogger.shared.log("[SHARPNESS] \(message)")
    }

    private static func fail(_ message: String, code: ExitCode, loggerStarted: Bool = true) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        if loggerStarted {
            SessionLogger.shared.log("[SHARPNESS] \(message)")
            SessionLogger.shared.shutdown()
        }
        Darwin.exit(code.rawValue)
    }

    // MARK: - Errors

    enum SharpnessError: LocalizedError {
        case modelPathMissing(String)
        case modelPathNotSafetensors(String)
        case outputIsAModel(output: String, model: String)
        case outputExists(String)
        case outputNotARegularFile(path: String, kind: FileSafety.ItemKind)
        case corpusStartBeyondEnd(start: Int, corpusGames: Int)
        case shardGameCountMismatch(shard: String, trailer: Int, read: Int)
        case notATrainerStateFile(String)
        case noLineageParameters(String)
        case checkpointNotBFloat16(ComputeDataType)
        case trainableCountMismatch(plan: Int, network: Int, velocity: Int)
        case weightCountTooSmall(have: Int, need: Int)
        case bufferTooSmall(positions: Int, batchSize: Int)
        case batchNotDrawn(batch: Int)
        case stepClipped(norm: Float, cap: Float)
        case nonFiniteGradient(batch: Int)
        case nonFiniteEstimate(String)
        case footprintLimitExceeded(footprintBytes: UInt64, limitBytes: UInt64, evaluations: Int)

        var errorDescription: String? {
            switch self {
            case .modelPathMissing(let path):
                return "\(modelFlag) \(path) does not exist"
            case .modelPathNotSafetensors(let path):
                return "\(modelFlag) \(path) is not a .safetensors file"
            case let .outputIsAModel(output, model):
                return "\(outFlag) \(output) is the model \(model) being probed; refusing to write over it"
            case .outputExists(let path):
                return "\(outFlag) \(path) already exists; pass \(replaceOutFlag) to replace it"
            case let .outputNotARegularFile(path, kind):
                return "\(outFlag) \(path) is a \(kind), not a regular file; refusing to write to it"
            case let .corpusStartBeyondEnd(start, corpusGames):
                return "\(corpusStartGameFlag) \(start) is not below the corpus's \(corpusGames) game(s)"
            case let .shardGameCountMismatch(shard, trailer, read):
                return "shard \(shard) trailer says \(trailer) game(s) but \(read) were read"
            case .notATrainerStateFile(let path):
                return "\(path) carries no trainer schedule (trainer_lr_schedule / trainer_completed_steps): "
                    + "not a trainer-state file, so its trainer step and fed LR are unknown"
            case .noLineageParameters(let path):
                return "\(path) has no lineage parameter snapshot to configure the trainer from"
            case .checkpointNotBFloat16(let type):
                return "\(computeFlag) \(Compute.bFloat16.rawValue) probes at the checkpoint's own bf16 architecture, "
                    + "but this checkpoint computes in \(type.rawValue)"
            case let .trainableCountMismatch(plan, network, velocity):
                return "trainable tensors disagree: plan \(plan), trainer network \(network), velocity \(velocity)"
            case let .weightCountTooSmall(have, need):
                return "weight file has \(have) tensors but the network needs at least \(need)"
            case let .bufferTooSmall(positions, batchSize):
                return "the replay buffer holds \(positions) position(s), fewer than one \(batchSize)-position batch"
            case .batchNotDrawn(let batch):
                return "the replay buffer could not draw batch \(batch)"
            case let .stepClipped(norm, cap):
                return "a probe step was clipped (pre-clip norm \(norm) > fed cap \(cap)); its velocity is not the gradient"
            case .nonFiniteGradient(let batch):
                return "the gradient on batch \(batch) is not finite"
            case .nonFiniteEstimate(let what):
                return "\(what) is not finite"
            case let .footprintLimitExceeded(footprint, limit, evaluations):
                return String(format: "process footprint %.2f GiB exceeds the %.2f GiB limit (%@) after %d gradient evaluation(s)",
                              Double(footprint) / 1_073_741_824, Double(limit) / 1_073_741_824, footprintLimitFlag, evaluations)
            }
        }
    }

    // MARK: - Models, corpus, output

    private static func resolveModel(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else { throw SharpnessError.modelPathMissing(url.path) }
        guard url.pathExtension.lowercased() == "safetensors" else { throw SharpnessError.modelPathNotSafetensors(url.path) }
        return url
    }

    /// The same checks, through the same `FileSafety` calls, as
    /// `EvalCorpusCLI`'s `--eval-out`.
    private static func openOutput(path: String?, models: [URL], replaceExisting: Bool) throws -> FileSafety.OpenedForWriting? {
        guard let path else { return nil }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        for model in models where try FileSafety.mayNameTheSameFile(url, model) {
            throw SharpnessError.outputIsAModel(output: url.path, model: model.path)
        }
        if let existing = try FileSafety.existingItem(at: url) {
            guard existing.kind == .regularFile else {
                throw SharpnessError.outputNotARegularFile(path: url.path, kind: existing.kind)
            }
            guard replaceExisting else { throw SharpnessError.outputExists(url.path) }
        }
        do {
            return try FileSafety.openForWritingReportingCreation(at: url,
                                                                  existingRegularFile: replaceExisting ? .truncate : .refuse)
        } catch FileSafetyError.notARegularFile(_, let kind) {
            throw SharpnessError.outputNotARegularFile(path: url.path, kind: kind)
        } catch FileSafetyError.alreadyExists(_, let kind) {
            guard kind == .regularFile else { throw SharpnessError.outputNotARegularFile(path: url.path, kind: kind) }
            throw SharpnessError.outputExists(url.path)
        }
    }

    /// The games the buffer is filled from, in corpus order from `startGame`.
    struct CorpusGames {
        let corpusID: String
        let startGame: Int
        let records: [GameRecord]
        var moveCount: Int { records.reduce(0) { $0 + $1.moves.count } }
    }

    /// Read games in corpus order from `startGame` until their recorded moves
    /// reach `positionBudget` (or the corpus ends). Only the shards that hold
    /// them are read, each fully verified by `GameCorpusShardIO.readSealed`.
    private static func loadGames(of corpus: GameCorpusReadOnlyView, startGame: Int,
                                  positionBudget: Int) throws -> CorpusGames {
        let shardGameCounts = try corpus.sealedShardURLs.map { try GameCorpusShardIO.readSealedTrailer(at: $0).gameCount }
        let corpusGames = shardGameCounts.reduce(0, +)
        guard startGame < corpusGames else {
            throw SharpnessError.corpusStartBeyondEnd(start: startGame, corpusGames: corpusGames)
        }
        var records: [GameRecord] = []
        var moves = 0
        var shardStart = 0
        for (shardURL, shardGames) in zip(corpus.sealedShardURLs, shardGameCounts) {
            defer { shardStart += shardGames }
            guard moves < positionBudget else { break }
            guard shardStart + shardGames > startGame else { continue }
            let shard = try GameCorpusShardIO.readSealed(at: shardURL)
            guard shard.games.count == shardGames else {
                throw SharpnessError.shardGameCountMismatch(shard: shardURL.lastPathComponent,
                                                            trailer: shardGames, read: shard.games.count)
            }
            for game in shard.games[max(0, startGame - shardStart)...] {
                guard moves < positionBudget else { break }
                records.append(game)
                moves += game.moves.count
            }
        }
        return CorpusGames(corpusID: corpus.corpusID, startGame: startGame, records: records)
    }

    /// The replay buffer, filled the way corpus replay fills it
    /// (`CorpusReplayFeeder`), from `games` in order, skipping any game that
    /// would overflow the ring. The feeder network only supplies the input
    /// encoding (`ActiveGame` never evaluates it); it is released on return,
    /// before any gradient work.
    private static func fillBuffer(architecture: NetworkArchitecture, games: CorpusGames, options: Options,
                                   constraints: ReplayBuffer.SamplingConstraints,
                                   sampler: DCMRandom) throws -> (ReplayBuffer, CorpusReplayFeedTally) {
        let feederNetwork = try ChessMPSNetwork(.overwrittenByLoad, arch: architecture)
        let buffer = ReplayBuffer(capacity: options.bufferPositions, inputEncoding: feederNetwork.inputEncoding,
                                  sampler: sampler)
        buffer.setSamplingConstraints(constraints)
        let feeder = CorpusReplayFeeder(network: feederNetwork, buffer: buffer)
        var feedTally = CorpusReplayFeedTally()
        for game in games.records where buffer.count + game.moves.count <= options.bufferPositions {
            if let rejection = feedTally.record(feeder.feed(game)) {
                progress("game rejected while filling the buffer: \(rejection)")
            }
        }
        return (buffer, feedTally)
    }

    // MARK: - Probe

    /// One checkpoint's records: the whole-network line, then one per group
    /// with `--sharpness-groups`.
    private static func probe(modelURL: URL, options: Options, corpus: GameCorpusReadOnlyView,
                              games: CorpusGames, started: DispatchTime) throws -> [[String: Any]] {
        progress("\(modelURL.lastPathComponent): loading")
        let file = try CheckpointManager.loadModelFile(at: modelURL)
        guard let schedule = file.metadata.trainerSchedule else {
            throw SharpnessError.notATrainerStateFile(modelURL.path)
        }
        guard let recordedParameters = file.safetensorsProvenance?.lineage.record?.parameters else {
            throw SharpnessError.noLineageParameters(modelURL.path)
        }
        // The run's parameters as the file records them, read by the one
        // parameter-JSON reader. A key the record lacks takes its declared
        // default inside `declaredDefaults`; those keys are reported, never
        // silent.
        let recordedValues = try ParameterValue.parametersObject(
            fromJSON: Data(recordedParameters.snapshotJSON.utf8), currentValue: { _ in nil })
        let parameters = try TrainingParametersSnapshot.declaredDefaults(overriding: recordedValues)
        let absentFromRecord = TrainingParameters.allKeys.map { $0.id }
            .filter { recordedValues[$0] == nil && !LineageRecord.Parameters.excludedParameterIDs.contains($0) }
            .sorted()
        let replayParams = try ReplayParams(parameters)
        var hyperparameters = replayParams.trainer
        // The probe's only departures from training (see the type doc): no
        // clipping, so the velocity is the gradient; no batch-stats or KL
        // probe work, which reads but never changes the gradient.
        hyperparameters.gradClipMaxNorm = Float.greatestFiniteMagnitude
        let cap = hyperparameters.relativeGradientCap
        hyperparameters.relativeGradientCap = RelativeGradientCapSettings(
            modeRawValue: RelativeGradientCapMode.off.rawValue, multiple: cap.multiple, windowSteps: cap.windowSteps,
            minimumHistorySteps: cap.minimumHistorySteps, floor: cap.floor)
        hyperparameters.batchStatsInterval = 0
        hyperparameters.klProbeInterval = 0

        let architecture: NetworkArchitecture
        switch options.compute {
        case .float32:
            architecture = try file.architecture.withComputeDataType(.float32, tail: nil)
        case .bFloat16:
            guard file.architecture.computeDataType == .bFloat16 else {
                throw SharpnessError.checkpointNotBFloat16(file.architecture.computeDataType)
            }
            architecture = file.architecture
        }
        let plan = architecture.trainableTensorPlan()
        let streams = DCMRandomStreams(masterSeed: options.seed)
        let trainer = try ChessTrainer(
            dropoutStream: streams.generator(.probe(name: "sharpness.dropout", trainerStep: 0)),
            hyperparameters: hyperparameters, arch: architecture, initialization: .overwrittenByLoad)

        let baseCount = plan.count + architecture.weightTensorPlan().filter { $0.kind == .bnRunningStat }.count
        guard file.weights.count >= baseCount else {
            throw SharpnessError.weightCountTooSmall(have: file.weights.count, need: baseCount)
        }
        let batchSamplers = (0..<options.batchCount).map {
            streams.generator(.probe(name: "sharpness.batch.\($0)", trainerStep: 0))
        }
        let (buffer, feedTally) = try fillBuffer(architecture: file.architecture, games: games, options: options,
                                                 constraints: replayParams.samplingConstraints, sampler: batchSamplers[0])
        let batchSize = replayParams.trainingBatchSize
        guard buffer.count >= batchSize else {
            throw SharpnessError.bufferTooSmall(positions: buffer.count, batchSize: batchSize)
        }

        let trainableCount = plan.count
        let networkTrainables = trainer.network.trainableVariables.count
        let weights = Array(file.weights.prefix(trainableCount))
        let runningStatistics = Array(file.weights[trainableCount..<baseCount])
        let oracle = GradientOracle(trainer: trainer, buffer: buffer, batchSize: batchSize,
                                    runningStatistics: runningStatistics, batchSamplers: batchSamplers,
                                    footprintLimitBytes: options.footprintLimitBytes)
        if hyperparameters.dropoutRate > 0 {
            try oracle.fixDropoutMasks(loading: weights)
        }

        // The gradient at w on each batch, then batch 0 again (determinism).
        progress("\(modelURL.lastPathComponent): gradients at w on \(options.batchCount) batch(es) of \(batchSize)")
        // Only batch 0's gradient is kept (for the determinism check); the
        // others go into the mean as they arrive, so at most two full
        // gradients are alive here.
        var atWeights: [(timing: TrainStepTiming, velocityNorm: Double)] = []
        var meanGradientSum = SharpnessVectorMath.zeroAccumulator(shapedLike: weights)
        var firstGradient: [[Float]] = []
        for batch in 0..<options.batchCount {
            let evaluation = try oracle.gradient(at: weights, batch: batch)
            if batch == 0 {
                guard evaluation.gradient.count == trainableCount, networkTrainables == trainableCount else {
                    throw SharpnessError.trainableCountMismatch(plan: trainableCount, network: networkTrainables,
                                                                velocity: evaluation.gradient.count)
                }
                try SharpnessVectorMath.requireSameLayout(evaluation.gradient, weights)
                firstGradient = evaluation.gradient
            }
            atWeights.append((evaluation.timing, SharpnessVectorMath.norm(evaluation.gradient)))
            try SharpnessVectorMath.accumulate(into: &meanGradientSum, evaluation.gradient, scale: 1 / Double(options.batchCount))
        }
        let determinism = try SharpnessVectorMath.difference(firstGradient, oracle.gradient(at: weights, batch: 0).gradient)
        firstGradient = []
        let meanGradient = SharpnessVectorMath.narrowed(meanGradientSum)
        meanGradientSum = []

        // What the checkpoint's own schedule feeds at its trainer step.
        let fed = LRMomentumCycleReadout.values(
            completedTrainSteps: schedule.completedTrainSteps, lrWarmupSteps: schedule.lrWarmupSteps,
            cycle: schedule.lrMomentumCycle, staticLearningRate: Float(parameters.learningRate),
            staticMomentum: Float(parameters.momentumCoeff), batchSize: batchSize,
            sqrtBatchScaling: parameters.sqrtBatchScalingLR,
            sqrtScaleBaseBatchSize: ChessTrainer.sqrtScaleBaseBatchSize)
        let stabilityThreshold = 2 * (1 + Double(fed.momentum)) / Double(fed.learningRate)

        let everyTensor = Set(weights.indices)
        var scopes: [(name: String, tensors: Set<Int>)] = [("all", everyTensor)]
        if options.perGroup {
            let groups = try SharpnessParameterGroup.tensorIndices(of: plan)
            for group in SharpnessParameterGroup.allCases {
                guard let tensors = groups[group], !tensors.isEmpty else { continue }
                scopes.append((group.rawValue, tensors))
            }
        }

        let common: [String: Any] = [
            "event": "sharpness",
            "model": modelURL.path,
            "model_id": file.modelID,
            "trainer_step": schedule.completedTrainSteps,
            "compute": architecture.computeDataType.rawValue,
            "checkpoint_compute": file.architecture.computeDataType.rawValue,
            "policy_tail": architecture.policyTailPrecision.rawValue,
            "corpus_id": games.corpusID,
            "corpus_path": corpus.directory.path,
            "corpus_first_game": games.startGame,
            "buffer": ["capacity": options.bufferPositions, "positions": buffer.count, "games_fed": feedTally.games,
                       "games_rejected": feedTally.rejected, "games_skipped": feedTally.skipped] as [String: Any],
            "batch_size": batchSize,
            "batches": options.batchCount,
            "seed": String(options.seed),
            "dropout_rate": Double(hyperparameters.dropoutRate),
            "parameters_absent_from_record": absentFromRecord,
            "gradient": [
                "graph_pre_clip_norm_per_batch": atWeights.map { Double($0.timing.gradGlobalNorm) },
                "velocity_norm_per_batch": atWeights.map(\.velocityNorm),
                "mean_gradient_norm": SharpnessVectorMath.norm(meanGradient),
                "loss_per_batch": atWeights.map { Double($0.timing.loss) },
                "policy_loss_per_batch": atWeights.map { Double($0.timing.policyLoss) },
                "value_loss_per_batch": atWeights.map { Double($0.timing.valueLoss) },
            ] as [String: Any],
            "determinism": ["bit_identical": determinism.bitIdentical,
                            "max_abs_difference": determinism.maxAbsolute] as [String: Any],
            "schedule": ["learning_rate": Double(fed.learningRate), "momentum": Double(fed.momentum),
                         "stability_threshold": stabilityThreshold] as [String: Any],
            "weight_norm_all": SharpnessVectorMath.norm(weights),
            "iterations_max": options.iterationCount,
            "tolerance": options.tolerance,
            "epsilon_relative": options.relativeEpsilon,
        ]

        var records: [[String: Any]] = []
        for scope in scopes {
            let scopeStarted = DispatchTime.now()
            let evaluationsBefore = oracle.evaluationCount
            let result = try powerIterate(oracle: oracle, weights: weights, tensors: scope.tensors, scopeName: scope.name,
                                          options: options, streams: streams, modelName: modelURL.lastPathComponent)
            var record = common
            record["scope"] = scope.name
            record["tensors"] = scope.tensors.count
            record["weight_norm"] = result.weightNorm
            record["epsilon"] = result.epsilon
            record["group_mean_gradient_norm"] = SharpnessVectorMath.norm(
                try SharpnessVectorMath.masked(meanGradient, keeping: scope.tensors))
            record["lambda_max"] = result.lambda
            record["rayleigh_sequence"] = result.sequence
            record["iterations"] = result.sequence.count
            record["converged"] = result.converged
            record["lambda_over_stability_threshold"] = result.lambda / stabilityThreshold
            record["epsilon_check"] = ["lambda_epsilon": result.lambda, "lambda_2epsilon": result.lambdaAtDoubleEpsilon,
                                       "relative_difference": result.epsilonRelativeDifference] as [String: Any]
            record["symmetry_check"] = ["u_J_v": result.startDotProductOfFinal, "v_J_u": result.finalDotProductOfStart,
                                        "relative_asymmetry": result.relativeAsymmetry] as [String: Any]
            record["gradient_evaluations"] = oracle.evaluationCount - evaluationsBefore
            record["peak_footprint_bytes"] = oracle.peakFootprintBytes
            record["scope_elapsed_ms"] = (Double(DispatchTime.now().uptimeNanoseconds - scopeStarted.uptimeNanoseconds) / 1e6).rounded()
            record["elapsed_ms"] = (Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1e6).rounded()
            records.append(record)
        }
        return records
    }

    // MARK: - Power iteration

    struct PowerIterationResult {
        let weightNorm: Double
        let epsilon: Double
        let lambda: Double
        let sequence: [Double]
        let converged: Bool
        let lambdaAtDoubleEpsilon: Double
        let epsilonRelativeDifference: Double
        let startDotProductOfFinal: Double
        let finalDotProductOfStart: Double
        let relativeAsymmetry: Double
    }

    private static func powerIterate(oracle: GradientOracle, weights: [[Float]], tensors: Set<Int>, scopeName: String,
                                     options: Options, streams: DCMRandomStreams, modelName: String) throws -> PowerIterationResult {
        let weightNorm = SharpnessVectorMath.norm(try SharpnessVectorMath.masked(weights, keeping: tensors))
        let epsilon = options.relativeEpsilon * weightNorm
        guard epsilon.isFinite, epsilon > 0 else { throw SharpnessError.nonFiniteEstimate("ε (from ‖w‖ = \(weightNorm))") }

        /// J·v restricted to the scope: the mean over batches of the
        /// central difference, projected back onto the scope's tensors.
        func product(_ direction: [[Float]], step: Double) throws -> [[Float]] {
            let plus = try SharpnessVectorMath.adding(direction, scaledBy: step, to: weights)
            let minus = try SharpnessVectorMath.adding(direction, scaledBy: -step, to: weights)
            var sum = SharpnessVectorMath.zeroAccumulator(shapedLike: weights)
            for batch in 0..<options.batchCount {
                let gradientPlus = try oracle.gradient(at: plus, batch: batch).gradient
                let gradientMinus = try oracle.gradient(at: minus, batch: batch).gradient
                try SharpnessVectorMath.accumulateDifference(into: &sum, plus: gradientPlus, minus: gradientMinus,
                                                             scale: 1 / (2 * step * Double(options.batchCount)))
            }
            return try SharpnessVectorMath.masked(SharpnessVectorMath.narrowed(sum), keeping: tensors)
        }

        var random = streams.generator(.probe(name: "sharpness.direction.\(scopeName)", trainerStep: 0))
        let start = try SharpnessVectorMath.normalized(
            SharpnessVectorMath.gaussianVector(shapedLike: weights, keeping: tensors, using: &random)).unit
        var vector = start
        var sequence: [Double] = []
        var converged = false
        var lastProduct: [[Float]] = []
        for iteration in 1...options.iterationCount {
            let applied = try product(vector, step: epsilon)
            let lambda = try SharpnessVectorMath.rayleighQuotient(vector: vector, product: applied)
            guard lambda.isFinite else { throw SharpnessError.nonFiniteEstimate("the Rayleigh quotient at iteration \(iteration)") }
            sequence.append(lambda)
            lastProduct = applied
            progress(String(format: "%@ [%@] iteration %d: lambda %.6g (footprint %.2f GiB after %d evaluations)",
                            modelName, scopeName, iteration, lambda,
                            Double(oracle.lastFootprintBytes) / 1_073_741_824, oracle.evaluationCount))
            if let previous = sequence.dropLast().last,
               SharpnessVectorMath.hasConverged(previous: previous, current: lambda, relativeTolerance: options.tolerance) {
                converged = true
                break
            }
            if iteration < options.iterationCount {
                vector = try SharpnessVectorMath.normalized(applied).unit
            }
        }
        guard let lambda = sequence.last else {
            preconditionFailure("power iteration ran no iteration (iterations ≥ 1 is parsed)")
        }
        // `vector` is the direction whose product `lastProduct` is and whose
        // Rayleigh quotient is `lambda`.
        let atDoubleEpsilon = try SharpnessVectorMath.rayleighQuotient(vector: vector, product: product(vector, step: 2 * epsilon))
        let productOfStart = try product(start, step: epsilon)
        let startDotProductOfFinal = try SharpnessVectorMath.dot(start, lastProduct)
        let finalDotProductOfStart = try SharpnessVectorMath.dot(vector, productOfStart)
        let asymmetryScale = abs(startDotProductOfFinal) + abs(finalDotProductOfStart)
        return PowerIterationResult(
            weightNorm: weightNorm, epsilon: epsilon, lambda: lambda, sequence: sequence, converged: converged,
            lambdaAtDoubleEpsilon: atDoubleEpsilon,
            epsilonRelativeDifference: abs(atDoubleEpsilon - lambda) / max(abs(lambda), Double.leastNormalMagnitude),
            startDotProductOfFinal: startDotProductOfFinal, finalDotProductOfStart: finalDotProductOfStart,
            relativeAsymmetry: asymmetryScale > 0 ? abs(startDotProductOfFinal - finalDotProductOfStart) / asymmetryScale : 0)
    }

    // MARK: - Gradient oracle

    /// g_k(w): the trainer's gradient at trainable weights w on fixed batch k
    /// (see the type doc). Main-thread object; each evaluation is one
    /// `runBlocking` whose work runs on the trainer's queue.
    private final class GradientOracle {
        struct Evaluation {
            let gradient: [[Float]]
            let timing: TrainStepTiming
        }

        private let trainer: ChessTrainer
        private let buffer: ReplayBuffer
        private let batchSize: Int
        private let runningStatistics: [[Float]]
        private let batchSamplers: [DCMRandom]
        private let footprintLimitBytes: UInt64
        private var dropoutState: DropoutPhiloxState?
        private(set) var evaluationCount = 0
        /// The process footprint after the latest evaluation, and the largest seen.
        private(set) var lastFootprintBytes: UInt64 = 0
        private(set) var peakFootprintBytes: UInt64 = 0

        init(trainer: ChessTrainer, buffer: ReplayBuffer, batchSize: Int, runningStatistics: [[Float]],
             batchSamplers: [DCMRandom], footprintLimitBytes: UInt64) {
            self.footprintLimitBytes = footprintLimitBytes
            self.trainer = trainer
            self.buffer = buffer
            self.batchSize = batchSize
            self.runningStatistics = runningStatistics
            self.batchSamplers = batchSamplers
        }

        /// Capture the dropout Philox state after a load so every later step
        /// starts from it (the same masks on every evaluation).
        func fixDropoutMasks(loading weights: [[Float]]) throws {
            let base = weights + runningStatistics
            let trainer = self.trainer
            dropoutState = try runBlocking {
                try await trainer.loadBaseWeightsResetVelocity(base)
                return try await trainer.captureDropoutState()
            }
        }

        func gradient(at weights: [[Float]], batch: Int) throws -> Evaluation {
            let base = weights + runningStatistics
            let trainer = self.trainer
            let buffer = self.buffer
            let batchSize = self.batchSize
            let sampler = batchSamplers[batch]
            let dropoutState = self.dropoutState
            let (velocity, timing) = try runBlocking { () async throws -> ([[Float]], TrainStepTiming) in
                try await trainer.loadBaseWeightsResetVelocity(base)
                if let dropoutState {
                    try await trainer.restoreDropoutState(dropoutState)
                }
                buffer.restoreSamplerState(sampler)
                guard let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize) else {
                    throw SharpnessError.batchNotDrawn(batch: batch)
                }
                return (try await trainer.exportVelocitySnapshot(), timing)
            }
            evaluationCount += 1
            // The footprint guard: a failed read is an error, not a pass.
            switch ChessTrainer.readPhysFootprintBytes() {
            case .success(let bytes):
                lastFootprintBytes = bytes
                peakFootprintBytes = max(peakFootprintBytes, bytes)
                guard bytes <= footprintLimitBytes else {
                    throw SharpnessError.footprintLimitExceeded(footprintBytes: bytes, limitBytes: footprintLimitBytes,
                                                                evaluations: evaluationCount)
                }
            case .failure(let error):
                throw error
            }
            guard !timing.gradientCap.clipped(preClipNorm: timing.gradGlobalNorm) else {
                throw SharpnessError.stepClipped(norm: timing.gradGlobalNorm, cap: timing.gradientCap.fedCap)
            }
            guard SharpnessVectorMath.norm(velocity).isFinite else { throw SharpnessError.nonFiniteGradient(batch: batch) }
            return Evaluation(gradient: velocity, timing: timing)
        }
    }
}
