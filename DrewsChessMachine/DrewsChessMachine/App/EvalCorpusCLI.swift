import Darwin
import Foundation

/// `--eval-corpus <corpus id or folder> --eval-model <file|folder> [--eval-model …]
/// [--eval-games N] [--eval-from-end] [--eval-out <file>] [--eval-out-overwrite]`:
/// policy and value loss of saved models on the positions of a game corpus,
/// one JSON line per model, then exit (no GUI). Read-only: the corpus is
/// opened through `GameCorpus.openReadOnly` and the models through
/// `CheckpointManager.loadModelFile`; nothing is written but the optional
/// `--eval-out` file.
///
/// Why it exists (`documentation/plans-active/HELDOUT_EVAL_AND_SHARPNESS_PLAN.md`,
/// Phase 1): to tell a capacity limit from memorization, the loss on games a
/// model trained on (the newest games of its training corpus) is compared with
/// the loss on games it never saw (a held-out corpus from a later month), on
/// the same number of games, model by model.
///
/// What is measured, per model, over one fixed position list:
///
/// - **Positions.** The first (or, with `--eval-from-end`, the last) N games
///   in corpus order (sealed shards in sequence order, games in shard order),
///   replayed move by move through `ChessGameEngine` exactly as corpus replay
///   does (`CorpusReplayFeeder`: server-authoritative adjudication, so a game
///   is replayed to its recorded end; the engine's history window feeds the
///   repetition / history planes, so each encoded input is the one training
///   saw). Every position before a move is scored; the move played is the
///   policy target and the game's result, from the side to move, the value
///   target.
/// - **Skipped games.** The same games corpus replay skips or rejects — no
///   moves, a FEN start, or a move that cannot be made (illegal, or after
///   the game ended) — are skipped whole, as training never sees any of their
///   positions. They are counted by reason in `skipped` and the first few are
///   listed in `rejections`; nothing is dropped silently. The screening runs
///   once, before any model, so every model is scored on the identical list.
/// - **Metrics** (`CorpusEvaluationMetrics`): `policy_nll`, `top1`, `top5`
///   under the legal-only softmax, `value_ce`, `value_acc` and the mean
///   W/D/L probabilities, overall and per ply bucket (`by_ply`).
///
/// Each model is built as an inference network from its own embedded
/// architecture (BN running statistics, its own input encoding and policy
/// tail), the way `--probe-model` builds one; a trainer-state file's
/// optimizer velocity is ignored. A model whose forward pass returns any
/// non-finite output, or whose results do not encode, is an
/// `"event":"error"` line, never numbers; the run carries on with the next
/// model and exits non-zero naming every failed one.
///
/// Threading: the corpus read, the replay and the encoding run on the
/// launching (main) thread; only the forward passes cross into Swift
/// concurrency, one `runBlocking` per batch, and each batch is scored inside
/// the network's own serial queue (the `evaluateBatched` callback), so no
/// long synchronous work runs inside a `Task`.
enum EvalCorpusCLI {

    static let corpusFlag = "--eval-corpus"
    static let modelFlag = "--eval-model"
    static let gamesFlag = "--eval-games"
    static let fromEndFlag = "--eval-from-end"
    static let outFlag = "--eval-out"
    /// `--eval-out` refuses an existing file unless this flag is also given
    /// (the `--probe-out-overwrite` rule).
    static let replaceOutFlag = "--eval-out-overwrite"

    static let defaultGameCount = 50_000
    /// Positions per forward pass.
    static let positionsPerForwardPass = 4096
    /// How many skipped games are described one by one in `rejections`.
    static let describedRejectionLimit = 20

    private enum ExitCode: Int32 {
        case usage = 90
        case corpusUnusable = 91
        case modelPathUnusable = 92
        case outputUnusable = 93
        case outputWriteFailed = 94
        case modelsFailed = 95
    }

    // MARK: - Options

    struct Options: Equatable {
        var corpusArgument: String
        var modelPaths: [String]
        var gameCount: Int
        var origin: CorpusEvaluationGameSelection.Origin
        var outPath: String?
        var replaceExistingOut: Bool
    }

    struct UsageError: LocalizedError, Equatable {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// Parse the launch arguments (without the executable path). Every `--`
    /// token must be one of this mode's flags; `--eval-model` may repeat,
    /// the others may appear once. A token no flag consumed is an error,
    /// except a single-dash argument the system adds (`-NSDocumentRevisionsDebugMode
    /// YES` when launched from Xcode) and its value.
    static func parseOptions(_ rawArgs: [String]) throws -> Options {
        let valueFlags: Set<String> = [corpusFlag, modelFlag, gamesFlag, outFlag]
        let booleanFlags: Set<String> = [fromEndFlag, replaceOutFlag]
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
                throw UsageError("\(corpusFlag) does not accept '\(token)'")
            } else if token.hasPrefix("-") {
                // A system-supplied `-Key value` pair.
                index += (index + 1 < rawArgs.count && !rawArgs[index + 1].hasPrefix("-")) ? 2 : 1
            } else {
                throw UsageError("unexpected argument '\(token)' (each model needs its own \(modelFlag))")
            }
        }
        for flag in [corpusFlag, gamesFlag, outFlag] where (values[flag]?.count ?? 0) > 1 {
            throw UsageError("\(flag) given more than once")
        }
        guard let corpus = values[corpusFlag]?.first else {
            throw UsageError("\(corpusFlag) requires a corpus ID or a corpus folder")
        }
        guard let models = values[modelFlag], !models.isEmpty else {
            throw UsageError("\(corpusFlag) needs at least one \(modelFlag) <model file or folder>")
        }
        var gameCount = defaultGameCount
        if let raw = values[gamesFlag]?.first {
            guard let parsed = Int(raw), parsed > 0 else {
                throw UsageError("\(gamesFlag) must be a positive whole number; got '\(raw)'")
            }
            gameCount = parsed
        }
        let outPath = values[outFlag]?.first
        let replaceExistingOut = present.contains(replaceOutFlag)
        if replaceExistingOut && outPath == nil {
            throw UsageError("\(replaceOutFlag) needs \(outFlag) <file>")
        }
        return Options(corpusArgument: corpus, modelPaths: models, gameCount: gameCount,
                       origin: present.contains(fromEndFlag) ? .end : .start,
                       outPath: outPath, replaceExistingOut: replaceExistingOut)
    }

    /// Never returns when `--eval-corpus` is on the command line. Main
    /// actor: called from the app's launch, and the corpus lookup it shares
    /// with `--replay-corpus` (`DrewsChessMachineApp.resolveCorpusDirectory`)
    /// is main-actor isolated.
    @MainActor
    static func handleIfRequested(rawArgs: [String]) {
        guard rawArgs.contains(corpusFlag) else { return }
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

        guard let corpusDirectory = DrewsChessMachineApp.resolveCorpusDirectory(options.corpusArgument) else {
            fail("error: corpus not found for '\(options.corpusArgument)' — no corpus.json at that path or under "
                 + CorpusPaths.corporaDir.path, code: .corpusUnusable)
        }
        let corpus: GameCorpusReadOnlyView
        do {
            corpus = try GameCorpus.openReadOnly(directory: corpusDirectory)
        } catch {
            fail("error: cannot open corpus \(corpusDirectory.path): \(error.localizedDescription)", code: .corpusUnusable)
        }
        if !corpus.ignoredOpenShardURLs.isEmpty {
            let names = corpus.ignoredOpenShardURLs.map(\.lastPathComponent).joined(separator: ", ")
            progress("WARNING corpus \(corpus.corpusID): ignoring \(corpus.ignoredOpenShardURLs.count) unsealed .open "
                     + "shard(s) (\(names)); their games are not in corpus order here and are not read")
        }

        let models: [URL]
        do {
            models = try resolveModels(options.modelPaths)
        } catch {
            fail("error: \(error.localizedDescription)", code: .modelPathUnusable)
        }

        let output: FileSafety.OpenedForWriting?
        do {
            output = try openOutput(path: options.outPath, models: models, replaceExisting: options.replaceExistingOut)
        } catch {
            fail("error: \(error.localizedDescription)", code: .outputUnusable)
        }

        let selection: CorpusEvaluationGameSelection
        let screening: GameScreening
        do {
            let loaded = try loadSelectedGames(of: corpus, gameCount: options.gameCount, origin: options.origin)
            selection = loaded.selection
            screening = screen(loaded.games, firstGameIndex: selection.gameRange.lowerBound)
        } catch {
            fail("error: corpus \(corpus.corpusID): \(error.localizedDescription)", code: .corpusUnusable)
        }
        guard screening.positionCount > 0 else {
            fail("error: none of the \(selection.requestedGames) selected game(s) has a replayable position "
                 + "(\(screening.skips.summary))", code: .corpusUnusable)
        }
        progress("corpus \(corpus.corpusID): games \(selection.gameRange.lowerBound)..<\(selection.gameRange.upperBound) "
                 + "of \(selection.corpusGames) (\(selection.origin.rawValue)); \(screening.games.count) game(s), "
                 + "\(screening.positionCount) position(s) to score; skipped: \(screening.skips.summary); "
                 + "\(models.count) model(s)")

        /// Print one encoded line and append it to `--eval-out`; a failed
        /// write ends the run.
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

        var failedModels: [String] = []
        for model in models {
            let started = DispatchTime.now()
            let line: Data
            do {
                let evaluation = try evaluate(modelURL: model, games: screening.games,
                                              totalPositions: screening.positionCount)
                let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000
                line = try ProbeModelCLI.encodeLine(record(of: evaluation, modelURL: model, corpus: corpus,
                                                           selection: selection, screening: screening,
                                                           elapsedMs: elapsedMs))
            } catch {
                failedModels.append(model.path)
                let failure: Data
                do {
                    failure = try ProbeModelCLI.encodeLine(["event": "error", "model": model.path,
                                                            "corpus_id": corpus.corpusID,
                                                            "error": error.localizedDescription])
                } catch let encodeError {
                    fail("error: could not encode the failure record for \(model.path): \(encodeError.localizedDescription)",
                         code: .outputWriteFailed)
                }
                emit(failure)
                continue
            }
            emit(line)
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
        FileHandle.standardError.write(Data("[EVAL-CORPUS] \(message)\n".utf8))
    }

    private static func fail(_ message: String, code: ExitCode, loggerStarted: Bool = true) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        if loggerStarted { SessionLogger.shared.shutdown() }
        Darwin.exit(code.rawValue)
    }

    // MARK: - Errors

    enum EvalCorpusError: LocalizedError {
        case modelPathMissing(String)
        case modelPathNotAModel(String)
        case modelFolderUnreadable(path: String, reason: String)
        case modelFolderHoldsNoModels(String)
        case outputIsAModel(output: String, model: String)
        case outputExists(String)
        case outputNotARegularFile(path: String, kind: FileSafety.ItemKind)
        case shardGameCountMismatch(shard: String, trailer: Int, read: Int)
        case selectedGameCountMismatch(expected: Int, read: Int)
        case weightCountTooSmall(have: Int, need: Int)
        case screenedGameNotReplayable(gameIndex: Int, ply: Int, reason: String)
        case forwardPassReturnedNoResult
        case forwardPassOutputSizeMismatch(policy: Int, wdl: Int, positions: Int)
        case nonFiniteOutputs(positions: Int, of: Int)

        var errorDescription: String? {
            switch self {
            case .modelPathMissing(let path):
                return "\(modelFlag) \(path) does not exist"
            case .modelPathNotAModel(let path):
                return "\(modelFlag) \(path) is not a .safetensors / .dcmmodel file or a folder"
            case let .modelFolderUnreadable(path, reason):
                return "\(modelFlag) cannot list \(path): \(reason)"
            case .modelFolderHoldsNoModels(let path):
                return "\(modelFlag) folder \(path) holds no .safetensors / .dcmmodel file"
            case let .outputIsAModel(output, model):
                return "\(outFlag) \(output) is the model \(model) being evaluated; refusing to write over it"
            case .outputExists(let path):
                return "\(outFlag) \(path) already exists; pass \(replaceOutFlag) to replace it"
            case let .outputNotARegularFile(path, kind):
                return "\(outFlag) \(path) is a \(kind), not a regular file; refusing to write to it"
            case let .shardGameCountMismatch(shard, trailer, read):
                return "shard \(shard) trailer says \(trailer) game(s) but \(read) were read"
            case let .selectedGameCountMismatch(expected, read):
                return "selected \(expected) game(s) but read \(read)"
            case let .weightCountTooSmall(have, need):
                return "weight file has \(have) tensors but the network needs at least \(need)"
            case let .screenedGameNotReplayable(gameIndex, ply, reason):
                return "game \(gameIndex) replayed in screening but not for this model, at ply \(ply): \(reason)"
            case .forwardPassReturnedNoResult:
                return "a forward pass returned without scoring its batch"
            case let .forwardPassOutputSizeMismatch(policy, wdl, positions):
                return "a forward pass of \(positions) position(s) returned \(policy) policy and \(wdl) W/D/L values"
            case let .nonFiniteOutputs(positions, of):
                return "\(positions) of \(of) positions in a batch have non-finite policy logits or W/D/L probabilities"
            }
        }
    }

    // MARK: - Models and output

    /// Each `--eval-model` argument: a model file as is, or a folder's
    /// model files (directly inside it, sorted by name — the save-timestamp
    /// prefix makes that chronological).
    private static func resolveModels(_ paths: [String]) throws -> [URL] {
        let fileManager = FileManager.default
        func isModelFile(_ url: URL) -> Bool {
            let pathExtension = url.pathExtension.lowercased()
            return pathExtension == "safetensors" || pathExtension == "dcmmodel"
        }
        var models: [URL] = []
        for path in paths {
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
                throw EvalCorpusError.modelPathMissing(url.path)
            }
            guard isDirectory.boolValue else {
                guard isModelFile(url) else { throw EvalCorpusError.modelPathNotAModel(url.path) }
                models.append(url)
                continue
            }
            let children: [URL]
            do {
                children = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil,
                                                               options: [.skipsHiddenFiles])
            } catch {
                throw EvalCorpusError.modelFolderUnreadable(path: url.path, reason: error.localizedDescription)
            }
            let folderModels = children.filter(isModelFile).sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard !folderModels.isEmpty else { throw EvalCorpusError.modelFolderHoldsNoModels(url.path) }
            models.append(contentsOf: folderModels)
        }
        return models
    }

    /// Validate and open `--eval-out` before any model is loaded: it may not
    /// name an evaluated model (directly, through a link, by case or as a
    /// hard link), and an existing file must be a regular file and needs
    /// `--eval-out-overwrite`. Same checks, through the same `FileSafety`
    /// calls, as `ProbeModelCLI.openOutputs`.
    private static func openOutput(path: String?, models: [URL], replaceExisting: Bool) throws -> FileSafety.OpenedForWriting? {
        guard let path else { return nil }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        for model in models where try FileSafety.mayNameTheSameFile(url, model) {
            throw EvalCorpusError.outputIsAModel(output: url.path, model: model.path)
        }
        if let existing = try FileSafety.existingItem(at: url) {
            guard existing.kind == .regularFile else {
                throw EvalCorpusError.outputNotARegularFile(path: url.path, kind: existing.kind)
            }
            guard replaceExisting else { throw EvalCorpusError.outputExists(url.path) }
        }
        do {
            return try FileSafety.openForWritingReportingCreation(at: url,
                                                                  existingRegularFile: replaceExisting ? .truncate : .refuse)
        } catch FileSafetyError.notARegularFile(_, let kind) {
            throw EvalCorpusError.outputNotARegularFile(path: url.path, kind: kind)
        } catch FileSafetyError.alreadyExists(_, let kind) {
            guard kind == .regularFile else { throw EvalCorpusError.outputNotARegularFile(path: url.path, kind: kind) }
            throw EvalCorpusError.outputExists(url.path)
        }
    }

    // MARK: - Corpus

    /// Read the selected games. Shard game counts come from the trailers;
    /// only the shards overlapping the selection are read (and fully
    /// verified, by `GameCorpusShardIO.readSealed`).
    private static func loadSelectedGames(
        of corpus: GameCorpusReadOnlyView,
        gameCount: Int,
        origin: CorpusEvaluationGameSelection.Origin
    ) throws -> (selection: CorpusEvaluationGameSelection, games: [GameRecord]) {
        let shardGameCounts = try corpus.sealedShardURLs.map { try GameCorpusShardIO.readSealedTrailer(at: $0).gameCount }
        let selection = try CorpusEvaluationGameSelection(origin: origin, requestedGames: gameCount,
                                                          corpusGames: shardGameCounts.reduce(0, +))
        var games: [GameRecord] = []
        games.reserveCapacity(selection.requestedGames)
        var shardStart = 0
        for (shardURL, shardGames) in zip(corpus.sealedShardURLs, shardGameCounts) {
            let shardRange = shardStart..<(shardStart + shardGames)
            let wanted = shardRange.clamped(to: selection.gameRange)
            if !wanted.isEmpty {
                progress("reading \(shardURL.lastPathComponent) (games \(wanted.lowerBound)..<\(wanted.upperBound))")
                let shard = try GameCorpusShardIO.readSealed(at: shardURL)
                guard shard.games.count == shardGames else {
                    throw EvalCorpusError.shardGameCountMismatch(shard: shardURL.lastPathComponent,
                                                                 trailer: shardGames, read: shard.games.count)
                }
                games.append(contentsOf: shard.games[(wanted.lowerBound - shardStart)..<(wanted.upperBound - shardStart)])
            }
            shardStart = shardRange.upperBound
        }
        guard games.count == selection.requestedGames else {
            throw EvalCorpusError.selectedGameCountMismatch(expected: selection.requestedGames, read: games.count)
        }
        return (selection, games)
    }

    /// Skipped selected games, by reason — the same games corpus replay skips
    /// (`CorpusReplayFeeder.feed`).
    struct SkipCounts: Equatable {
        var emptyGames = 0
        var startFENGames = 0
        var illegalMoveGames = 0
        var moveAfterGameEndGames = 0
        /// Moves recorded in all skipped games: positions not scored.
        var positionsInSkippedGames = 0

        var skippedGames: Int { emptyGames + startFENGames + illegalMoveGames + moveAfterGameEndGames }

        var summary: String {
            "\(skippedGames) game(s) (empty \(emptyGames), FEN start \(startFENGames), illegal move \(illegalMoveGames), "
                + "move after game end \(moveAfterGameEndGames)), \(positionsInSkippedGames) position(s)"
        }

        var jsonObject: [String: Any] {
            ["games": skippedGames,
             "games_empty": emptyGames,
             "games_start_fen": startFENGames,
             "games_illegal_move": illegalMoveGames,
             "games_move_after_game_end": moveAfterGameEndGames,
             "positions": positionsInSkippedGames]
        }
    }

    /// The selected games that replay, and what was skipped.
    struct GameScreening {
        /// Replayable games with their global corpus index, in corpus order.
        let games: [(corpusIndex: Int, record: GameRecord)]
        let positionCount: Int
        let skips: SkipCounts
        /// The first `describedRejectionLimit` skipped games, described.
        let rejections: [[String: Any]]
    }

    /// Replay every selected game once, model-independently, so each model is
    /// scored on the same positions and the skip counts are stated once.
    private static func screen(_ games: [GameRecord], firstGameIndex: Int) -> GameScreening {
        var replayable: [(corpusIndex: Int, record: GameRecord)] = []
        replayable.reserveCapacity(games.count)
        var positions = 0
        var skips = SkipCounts()
        var rejections: [[String: Any]] = []
        func describe(_ corpusIndex: Int, _ reason: String, ply: Int?) {
            guard rejections.count < describedRejectionLimit else { return }
            var description: [String: Any] = ["game_index": corpusIndex, "reason": reason]
            if let ply { description["ply"] = ply }
            rejections.append(description)
        }
        for (offset, game) in games.enumerated() {
            let corpusIndex = firstGameIndex + offset
            guard !game.moves.isEmpty else {
                skips.emptyGames += 1
                describe(corpusIndex, "no moves", ply: nil)
                continue
            }
            guard game.startFEN == nil else {
                skips.startFENGames += 1
                skips.positionsInSkippedGames += game.moves.count
                describe(corpusIndex, "FEN start", ply: nil)
                continue
            }
            let engine = ChessGameEngine(adjudication: .serverAuthoritative)
            var failure: (ply: Int, error: ChessGameError)?
            for (ply, move) in game.moves.enumerated() {
                do {
                    try engine.applyMoveAndAdvance(move)
                } catch let error as ChessGameError {
                    failure = (ply, error)
                    break
                } catch {
                    preconditionFailure("ChessGameEngine.applyMoveAndAdvance threw a non-ChessGameError: \(error)")
                }
            }
            if let failure {
                switch failure.error {
                case .illegalMove: skips.illegalMoveGames += 1
                case .gameAlreadyOver: skips.moveAfterGameEndGames += 1
                }
                skips.positionsInSkippedGames += game.moves.count
                describe(corpusIndex, failure.error.localizedDescription, ply: failure.ply)
                continue
            }
            replayable.append((corpusIndex, game))
            positions += game.moves.count
        }
        return GameScreening(games: replayable, positionCount: positions, skips: skips, rejections: rejections)
    }

    // MARK: - Evaluation

    private struct ModelEvaluation {
        let file: ModelCheckpointFile
        let tally: CorpusEvaluationTally
        let gamesScored: Int
    }

    /// One batch's per-position facts, handed to the scoring callback.
    private struct BatchTargets: Sendable {
        var count = 0
        /// Legal moves' policy indices, all positions back to back.
        var legalPolicyIndices: [Int32] = []
        /// `legalStarts[i]..<legalStarts[i + 1]` is position i's slice of
        /// `legalPolicyIndices`.
        var legalStarts: [Int] = [0]
        var playedOffsets: [Int] = []
        var results: [WDLClass] = []
        var plies: [Int] = []
    }

    private enum BatchScoring: Sendable {
        case scored(CorpusEvaluationTally)
        case nonFinite(positions: Int)
        case outputSizeMismatch(policy: Int, wdl: Int)
    }

    private static func evaluate(
        modelURL: URL,
        games: [(corpusIndex: Int, record: GameRecord)],
        totalPositions: Int
    ) throws -> ModelEvaluation {
        let file = try CheckpointManager.loadModelFile(at: modelURL)
        let network = try ChessMPSNetwork(.overwrittenByLoad, arch: file.architecture)
        // Trainer files carry optimizer velocity after the base block;
        // inference needs only the leading trainables + BN running stats
        // (the `ProbeModelCLI.probeOne` / `UCIModelLoader.buildAndLoad` rule).
        let baseCount = network.network.trainableVariables.count + network.network.bnRunningStatsVariables.count
        guard file.weights.count >= baseCount else {
            throw EvalCorpusError.weightCountTooSmall(have: file.weights.count, need: baseCount)
        }
        let baseWeights = Array(file.weights.prefix(baseCount))
        try runBlocking { try await network.network.loadWeights(baseWeights) }

        let encoding = network.inputEncoding
        let tensorLength = BoardEncoder.tensorLength(for: encoding)
        let capacity = positionsPerForwardPass
        var boards = [Float](repeating: 0, count: capacity * tensorLength)
        var targets = BatchTargets()
        var tally = CorpusEvaluationTally()
        var scoredPositions = 0
        var lastReport = DispatchTime.now()
        let started = lastReport

        func scoreBatch() throws {
            let batchTargets = targets
            let batchBoards = batchTargets.count == capacity
                ? boards : Array(boards.prefix(batchTargets.count * tensorLength))
            let outcome = SyncBox<BatchScoring?>(nil)
            try runBlocking {
                try await network.evaluateBatched(batchBoards: batchBoards, count: batchTargets.count) { policy, _, wdl in
                    outcome.value = score(batchTargets, policy: policy, wdl: wdl)
                }
            }
            switch outcome.value {
            case .none:
                throw EvalCorpusError.forwardPassReturnedNoResult
            case .scored(let batchTally):
                tally.merge(batchTally)
            case .nonFinite(let positions):
                throw EvalCorpusError.nonFiniteOutputs(positions: positions, of: batchTargets.count)
            case let .outputSizeMismatch(policy, wdl):
                throw EvalCorpusError.forwardPassOutputSizeMismatch(policy: policy, wdl: wdl, positions: batchTargets.count)
            }
            scoredPositions += batchTargets.count
            targets = BatchTargets()
            let now = DispatchTime.now()
            if now.uptimeNanoseconds - lastReport.uptimeNanoseconds >= 15_000_000_000 || scoredPositions == totalPositions {
                lastReport = now
                let seconds = Double(now.uptimeNanoseconds - started.uptimeNanoseconds) / 1e9
                let percent = 100 * Double(scoredPositions) / Double(totalPositions)
                progress(String(format: "%@: %d / %d positions (%.1f%%), %.0f positions/s",
                                modelURL.lastPathComponent, scoredPositions, totalPositions, percent,
                                Double(scoredPositions) / max(seconds, 1e-9)))
            }
        }

        for game in games {
            let engine = ChessGameEngine(adjudication: .serverAuthoritative)
            for (ply, move) in game.record.moves.enumerated() {
                let state = engine.state
                let legalMoves = engine.currentLegalMoves
                guard let playedOffset = legalMoves.firstIndex(of: move) else {
                    throw EvalCorpusError.screenedGameNotReplayable(gameIndex: game.corpusIndex, ply: ply,
                                                                    reason: "the played move is not legal")
                }
                let slot = targets.count
                boards.withUnsafeMutableBufferPointer { all in
                    let region = UnsafeMutableBufferPointer(rebasing: all[(slot * tensorLength)..<((slot + 1) * tensorLength)])
                    BoardEncoder.encode(state, history: engine.recentStates, into: region, encoding: encoding)
                }
                for legalMove in legalMoves {
                    targets.legalPolicyIndices.append(
                        Int32(PolicyEncoding.policyIndex(legalMove, currentPlayer: state.currentPlayer)))
                }
                targets.legalStarts.append(targets.legalPolicyIndices.count)
                targets.playedOffsets.append(playedOffset)
                targets.results.append(WDLClass(outcome: game.record.outcome, sideToMove: state.currentPlayer))
                targets.plies.append(ply)
                targets.count += 1
                if targets.count == capacity {
                    try scoreBatch()
                }
                do {
                    try engine.applyMoveAndAdvance(move)
                } catch {
                    throw EvalCorpusError.screenedGameNotReplayable(gameIndex: game.corpusIndex, ply: ply,
                                                                    reason: error.localizedDescription)
                }
            }
        }
        if targets.count > 0 {
            try scoreBatch()
        }
        return ModelEvaluation(file: file, tally: tally, gamesScored: games.count)
    }

    /// Score one forward pass. Runs inside the network's serial queue (the
    /// `evaluateBatched` callback), so it only reads the buffers it is
    /// handed. Any non-finite logit or probability in the batch makes the
    /// whole batch `.nonFinite`.
    private static func score(_ targets: BatchTargets,
                              policy: UnsafeBufferPointer<Float>,
                              wdl: UnsafeBufferPointer<Float>) -> BatchScoring {
        let policySize = ChessNetwork.policySize
        guard policy.count == targets.count * policySize, wdl.count == targets.count * 3 else {
            return .outputSizeMismatch(policy: policy.count, wdl: wdl.count)
        }
        var nonFinite = 0
        var tally = CorpusEvaluationTally()
        var legalLogits: [Float] = []
        legalLogits.reserveCapacity(MoveSampler.scratchCapacity)
        for position in 0..<targets.count {
            let logitsStart = position * policySize
            var finite = true
            for index in logitsStart..<(logitsStart + policySize) where !policy[index].isFinite {
                finite = false
                break
            }
            let win = wdl[position * 3], draw = wdl[position * 3 + 1], loss = wdl[position * 3 + 2]
            guard finite, win.isFinite, draw.isFinite, loss.isFinite else {
                nonFinite += 1
                continue
            }
            legalLogits.removeAll(keepingCapacity: true)
            for slot in targets.legalStarts[position]..<targets.legalStarts[position + 1] {
                legalLogits.append(policy[logitsStart + Int(targets.legalPolicyIndices[slot])])
            }
            let policyScore = legalLogits.withUnsafeBufferPointer { legal in
                CorpusEvaluationMetrics.legalPolicyScore(legalLogits: legal, playedOffset: targets.playedOffsets[position])
            }
            let probabilities = WDLProbabilities(win: Double(win), draw: Double(draw), loss: Double(loss))
            let valueScore = CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: targets.results[position])
            tally.add(ply: targets.plies[position], policy: policyScore, value: valueScore, probabilities: probabilities)
        }
        return nonFinite > 0 ? .nonFinite(positions: nonFinite) : .scored(tally)
    }

    // MARK: - Output

    private static func metricsObject(_ accumulator: CorpusEvaluationAccumulator) -> [String: Any] {
        var object: [String: Any] = ["positions": accumulator.positions]
        guard let means = accumulator.means else { return object }
        object["policy_nll"] = means.policyNLL
        object["top1"] = means.top1
        object["top\(CorpusEvaluationMetrics.wideTopK)"] = means.wideTopK
        object["value_ce"] = means.valueCrossEntropy
        object["value_acc"] = means.valueAccuracy
        object["p_win"] = means.meanWinProbability
        object["p_draw"] = means.meanDrawProbability
        object["p_loss"] = means.meanLossProbability
        return object
    }

    private static func record(
        of evaluation: ModelEvaluation,
        modelURL: URL,
        corpus: GameCorpusReadOnlyView,
        selection: CorpusEvaluationGameSelection,
        screening: GameScreening,
        elapsedMs: Double
    ) -> [String: Any] {
        let file = evaluation.file
        let stepReading = file.trainingStepReading
        var object = metricsObject(evaluation.tally.overall)
        var byPly: [String: Any] = [:]
        for bucket in CorpusEvaluationPlyBucket.allCases {
            byPly[bucket.jsonKey] = metricsObject(evaluation.tally.accumulator(for: bucket))
        }
        object["event"] = "eval"
        object["model"] = modelURL.path
        object["model_id"] = file.modelID
        // The trainer step where the file records one, else the step it
        // states (`--analyze-numerics`' reading); null when it states none.
        object["training_step"] = stepReading.trainerStepOrStatedStep.map { $0 as Any } ?? NSNull()
        object["training_step_basis"] = stepReading.basis.rawValue
        object["encoding"] = file.architecture.inputEncoding.rawValue
        object["corpus_id"] = corpus.corpusID
        object["corpus_path"] = corpus.directory.path
        object["selection"] = [
            "origin": selection.origin.rawValue,
            "requested_games": selection.requestedGames,
            "corpus_games": selection.corpusGames,
            "first_game_index": selection.gameRange.lowerBound,
            "end_game_index": selection.gameRange.upperBound,
        ] as [String: Any]
        object["games"] = evaluation.gamesScored
        object["by_ply"] = byPly
        object["skipped"] = screening.skips.jsonObject
        object["rejections"] = screening.rejections
        object["elapsed_ms"] = elapsedMs.rounded()
        return object
    }
}
