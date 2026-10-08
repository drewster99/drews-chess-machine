import Foundation

/// Evaluates a model's weights on the puzzle test sets, for the results every
/// model file carries (`ModelTestSetResultsField`). A protocol so tests can
/// write files without spending a GPU evaluation per save.
protocol ModelTestSetEvaluating: Sendable {
    /// Evaluate `weights` (base tensors first; a trainer file's trailing
    /// velocity tensors are ignored) built as `architecture`. Never throws:
    /// a failure is the `.failed` result, so the caller's save goes ahead.
    func evaluate(weights: [[Float]], architecture: NetworkArchitecture) async -> ModelTestSetResultsField
}

extension ModelTestSetEvaluating {
    /// `evaluate`, for a file about to be written: logs one
    /// `[CHECKPOINT] test sets <file>` line with the time it took and the
    /// results or the failure. Every writer goes through here.
    func evaluateForSave(weights: [[Float]], architecture: NetworkArchitecture, file: String) async -> ModelTestSetResultsField {
        let start = DispatchTime.now().uptimeNanoseconds
        let field = await evaluate(weights: weights, architecture: architecture)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        let tag: String
        if case .failed = field {
            tag = "[CHECKPOINT-ERR]"
        } else {
            tag = "[CHECKPOINT]"
        }
        SessionLogger.shared.log("\(tag) test sets \(file) (\(String(format: "%.0f", ms)) ms): \(field.logSummary)")
        return field
    }
}

/// The production evaluator. Builds a fresh inference network from the
/// weights' own architecture, loads the weights, runs every set in one
/// batched forward (`TacticalProbeRunner.runBatch`, the probe watcher's and
/// `--probe-model`'s path) and folds each set with `ProbeBatterySummary`.
///
/// A network is built per evaluation rather than cached: the evaluator then
/// holds no state, so concurrent saves (a session's champion and trainer
/// files) can't load weights into one network under each other, and a save
/// costs one graph build (the save's own verification builds one too).
///
/// Probe isolation (CLAUDE.md): it reads only the weights handed to it, on
/// its own network; argmax and softmax only, no random draws.
struct ModelTestSetEvaluator: ModelTestSetEvaluating {
    let testSets: [ProbeTestSet]

    /// The evaluator every model-file writer uses: the bundled sets
    /// (`LichessProbeData.modelFileTestSets`).
    static let modelFiles = ModelTestSetEvaluator(testSets: LichessProbeData.modelFileTestSets)

    /// Graph builds are long synchronous work; they run here, never on the
    /// cooperative pool.
    private static let buildQueue = DispatchQueue(label: "ModelTestSetEvaluator.build", qos: .userInitiated)

    /// A synchronous `ModelTestSetEvaluation` for the command-line writers
    /// that assemble a file synchronously (`--derive-model`, graft). It
    /// blocks the calling thread until the evaluation ends, so it must be
    /// called from a plain thread (the CLI's main thread), never from a
    /// Swift-concurrency task.
    func blockingEvaluation(file: String) -> ModelTestSetEvaluation {
        { weights, architecture in
            let box = BlockingEvaluationBox()
            let done = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                box.field = await self.evaluateForSave(weights: weights, architecture: architecture, file: file)
                done.signal()
            }
            done.wait()
            guard let field = box.field else {
                preconditionFailure("ModelTestSetEvaluator.blockingEvaluation: the evaluation ended without a result")
            }
            return field
        }
    }

    func evaluate(weights: [[Float]], architecture: NetworkArchitecture) async -> ModelTestSetResultsField {
        do {
            return .evaluated(try await results(weights: weights, architecture: architecture))
        } catch {
            return .failed(reason: "\(error)")
        }
    }

    private func results(weights: [[Float]], architecture: NetworkArchitecture) async throws -> ModelTestSetResults {
        guard !testSets.isEmpty else { throw EvaluationError.noTestSets }
        let network: ChessMPSNetwork = try await withCheckedThrowingContinuation { continuation in
            Self.buildQueue.async {
                continuation.resume(with: Result { try ChessMPSNetwork(.overwrittenByLoad, arch: architecture) })
            }
        }
        network.network.commandQueue.label = "ModelTestSetEvaluator"
        let baseCount = network.network.trainableVariables.count + network.network.bnRunningStatsVariables.count
        guard weights.count >= baseCount else {
            throw EvaluationError.tooFewTensors(have: weights.count, need: baseCount)
        }
        try await network.network.loadWeights(Array(weights.prefix(baseCount)))

        let probes = testSets.flatMap(\.probes)
        let encoding = network.inputEncoding
        var input: [Float] = []
        input.reserveCapacity(probes.count * BoardEncoder.tensorLength(for: encoding))
        for probe in probes {
            input.append(contentsOf: BoardEncoder.encode(probe.state, encoding: encoding))
        }
        let batch = await TacticalProbeRunner.runBatch(probes, encodedInput: input, against: network)
        guard batch.results.count == probes.count else {
            throw EvaluationError.resultCountMismatch(have: batch.results.count, want: probes.count)
        }

        var sets: [ModelTestSetResults.SetResult] = []
        var start = 0
        for testSet in testSets {
            let range = start..<(start + testSet.probes.count)
            start = range.upperBound
            sets.append(try Self.setResult(testSet, results: Array(batch.results[range])))
        }
        return ModelTestSetResults(
            evaluatedAtUnix: Int64(Date().timeIntervalSince1970),
            build: BuildInfo.buildNumber,
            policyTailPrecision: architecture.policyTailPrecision.rawValue,
            sets: sets
        )
    }

    /// One set's record. Refuses an incomplete evaluation rather than
    /// recording numbers that would read as the network's own.
    static func setResult(_ testSet: ProbeTestSet, results: [ProbeResult]) throws -> ModelTestSetResults.SetResult {
        guard !results.isEmpty else { throw EvaluationError.emptySet(testSet.id) }
        let summary = ProbeBatterySummary(results: results)
        let overall = summary.overall
        guard overall.errored == 0 else {
            throw EvaluationError.positionsErrored(set: testSet.id, errored: overall.errored, of: overall.totalProbes)
        }
        guard overall.countWithRank == overall.totalProbes, let avgRank = overall.avgExpectedRank else {
            throw EvaluationError.positionsWithoutRank(set: testSet.id, missing: overall.totalProbes - overall.countWithRank)
        }
        let pElo: ModelTestSetResults.PuzzleElo
        switch summary.puzzleElo {
        case .infinity: pElo = .allCorrect
        case -.infinity: pElo = .allWrong
        case let value where value.isFinite: pElo = .estimate(value)
        default: throw EvaluationError.noPuzzleElo(set: testSet.id)
        }
        let byTheme = Dictionary(uniqueKeysWithValues: summary.aggregates.map { ($0.theme, $0) })
        var themes: [ModelTestSetResults.ThemeResult] = []
        for category in ProbeCategory.allCases {
            guard let aggregate = byTheme[category] else { continue }
            guard let themeID = category.lichessThemeID else {
                throw EvaluationError.notALichessTheme(set: testSet.id, theme: category.rawValue)
            }
            themes.append(.init(id: themeID, title: category.title, correct: aggregate.argmaxCorrect, total: aggregate.total))
        }
        return ModelTestSetResults.SetResult(
            id: testSet.id,
            title: testSet.title,
            description: testSet.description,
            fingerprintSHA256: testSet.fingerprintSHA256,
            positions: overall.totalProbes,
            top1Correct: overall.argmaxCorrect,
            top5Correct: overall.top5Correct,
            avgCorrectProbability: Double(overall.avgExpectedProb),
            avgCorrectRank: Double(avgRank),
            nll: overall.meanNegLogProb,
            pElo: pElo,
            themes: themes
        )
    }

    enum EvaluationError: Error, CustomStringConvertible {
        case noTestSets
        case tooFewTensors(have: Int, need: Int)
        case resultCountMismatch(have: Int, want: Int)
        case emptySet(String)
        case positionsErrored(set: String, errored: Int, of: Int)
        case positionsWithoutRank(set: String, missing: Int)
        case noPuzzleElo(set: String)
        case notALichessTheme(set: String, theme: String)

        var description: String {
            switch self {
            case .noTestSets:
                return "no test sets to evaluate"
            case .tooFewTensors(let have, let need):
                return "the weights have \(have) tensors but the network needs at least \(need)"
            case .resultCountMismatch(let have, let want):
                return "the batched evaluation returned \(have) results for \(want) positions"
            case .emptySet(let id):
                return "test set \(id) has no positions"
            case .positionsErrored(let set, let errored, let of):
                return "the forward pass failed for \(errored) of \(of) positions in \(set)"
            case .positionsWithoutRank(let set, let missing):
                return "\(missing) positions in \(set) have no legal right move"
            case .noPuzzleElo(let set):
                return "no puzzle Elo for \(set) (no rated puzzles)"
            case .notALichessTheme(let set, let theme):
                return "\(set) holds theme \(theme), which is not a Lichess theme"
            }
        }
    }
}

/// One battery's results folded into the numbers every reader reports:
/// per-theme aggregates, the overall fold and the puzzle Elo. The one fold
/// shared by model-file results (`ModelTestSetEvaluator`) and
/// `--probe-model` (`ProbeModelCLI`), so they can't drift apart.
struct ProbeBatterySummary {
    let aggregates: [LichessProbeHistory.Aggregate]
    let overall: LichessProbeOverallSummary
    /// `LichessProbeHistory.mlePuzzleElo` over the rated puzzles: ±inf when
    /// every answer is right or wrong, NaN with none.
    let puzzleElo: Double

    init(results: [ProbeResult]) {
        aggregates = LichessProbeHistory.aggregates(from: results)
        overall = LichessProbeOverallSummary(folding: aggregates)
        let pairs: [(rating: Int, correct: Bool)] = results.compactMap {
            guard let meta = LichessProbeData.metadata[$0.probe.name] else { return nil }
            let correct = $0.verdict == .correctAndConfident || $0.verdict == .correctButFlat
            return (rating: meta.rating, correct: correct)
        }
        puzzleElo = LichessProbeHistory.mlePuzzleElo(pairs: pairs)
    }
}

/// Hands a detached evaluation's result back to the blocked thread; the
/// semaphore orders the write before the read.
private final class BlockingEvaluationBox: @unchecked Sendable {
    var field: ModelTestSetResultsField?
}
