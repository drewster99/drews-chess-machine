import CryptoKit
import Foundation
import os

/// Evaluates a model's weights on the puzzle test sets, for the results every
/// model file carries (`ModelTestSetResultsField`). A protocol so tests can
/// write files without spending a GPU evaluation per save.
protocol ModelTestSetEvaluating: Sendable {
    /// Evaluate `weights` (base tensors first; a trainer file's trailing
    /// velocity tensors are ignored) built as `architecture`. Always a fresh
    /// evaluation. Never throws: a failure is the `.failed` result, so the
    /// caller's save goes ahead.
    func evaluate(weights: [[Float]], architecture: NetworkArchitecture) async -> ModelTestSetResultsField

    /// The results for a file about to be written: `evaluate`, or an earlier
    /// evaluation of bit-identical base weights under an equal architecture
    /// (`ModelTestSetEvaluator` remembers its last few). `file` names the
    /// file, for a later reuse's log line.
    func resultsForSave(weights: [[Float]], architecture: NetworkArchitecture, file: String) async
        -> (field: ModelTestSetResultsField, source: ModelTestSetResultsSource)
}

/// Where a file's results came from.
enum ModelTestSetResultsSource: Sendable, Equatable {
    case evaluatedForThisFile
    /// Bit-identical base weights under an equal architecture were evaluated
    /// for `file` earlier in this process (that save may not have finished).
    case reusedFrom(file: String)
}

extension ModelTestSetEvaluating {
    func resultsForSave(weights: [[Float]], architecture: NetworkArchitecture, file: String) async
        -> (field: ModelTestSetResultsField, source: ModelTestSetResultsSource) {
        (await evaluate(weights: weights, architecture: architecture), .evaluatedForThisFile)
    }

    /// `resultsForSave`, logging one `[CHECKPOINT] test sets <file>` line with
    /// the time it took (and the reused evaluation, if any) and the results
    /// or the failure. Every writer goes through here.
    func evaluateForSave(weights: [[Float]], architecture: NetworkArchitecture, file: String) async -> ModelTestSetResultsField {
        let start = DispatchTime.now().uptimeNanoseconds
        let (field, source) = await resultsForSave(weights: weights, architecture: architecture, file: file)
        let ms = String(format: "%.0f", Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        let tag: String
        if case .failed = field {
            tag = "[CHECKPOINT-ERR]"
        } else {
            tag = "[CHECKPOINT]"
        }
        let how: String
        switch source {
        case .evaluatedForThisFile:
            how = "\(ms) ms"
        case .reusedFrom(let origin):
            how = "\(ms) ms, reused the evaluation made for \(origin): identical base weights"
        }
        SessionLogger.shared.log("\(tag) test sets \(file) (\(how)): \(field.logSummary)")
        return field
    }
}

/// The production evaluator. Builds a fresh inference network from the
/// weights' own architecture, loads the weights, runs every set in one
/// batched forward (`TacticalProbeRunner.runBatch`, the probe watcher's and
/// `--probe-model`'s path) and folds each set with `ProbeBatterySummary`.
///
/// A network is built per evaluation rather than cached, so two saves
/// running at once (a GUI save and a post-promotion save, say) can't load
/// weights into one network under each other. The cost is one graph build
/// per evaluated file, beside the save's own verification build.
///
/// `resultsForSave` reuses an evaluation of bit-identical base weights
/// (`ModelTestSetMemo`): a train-vs-UCI or post-promotion session's champion
/// is its trainer's base weights, a train-vs-UCI final save writes the same
/// weights three times, and a GUI champion is unchanged across periodic
/// saves until a promotion.
///
/// Probe isolation (CLAUDE.md): it reads only the weights handed to it, on
/// its own network; argmax and softmax only, no random draws.
struct ModelTestSetEvaluator: ModelTestSetEvaluating {
    let testSets: [ProbeTestSet]
    /// This evaluator's own memo: its test sets are part of every key.
    private let memo = ModelTestSetMemo()

    init(testSets: [ProbeTestSet]) {
        self.testSets = testSets
    }

    /// The evaluator every model-file writer uses: the bundled sets
    /// (`LichessProbeData.modelFileTestSets`).
    static let modelFiles = ModelTestSetEvaluator(testSets: LichessProbeData.modelFileTestSets)

    /// Graph builds and board encoding are long synchronous work; they run
    /// here, never on the cooperative pool.
    private static let buildQueue = DispatchQueue(label: "ModelTestSetEvaluator.build", qos: .userInitiated)

    /// A synchronous `ModelTestSetEvaluation` for the command-line writers
    /// that assemble a file synchronously (`--derive-model`, graft); see
    /// `runBlocking` for where it may be called.
    func blockingEvaluation(file: String) -> ModelTestSetEvaluation {
        { weights, architecture in
            try runBlocking {
                await self.evaluateForSave(weights: weights, architecture: architecture, file: file)
            }
        }
    }

    func evaluate(weights: [[Float]], architecture: NetworkArchitecture) async -> ModelTestSetResultsField {
        do {
            return .evaluated(try await results(weights: weights, architecture: architecture))
        } catch {
            return .failed(reason: "\(error)")
        }
    }

    func resultsForSave(weights: [[Float]], architecture: NetworkArchitecture, file: String) async
        -> (field: ModelTestSetResultsField, source: ModelTestSetResultsSource) {
        let key = await ModelTestSetMemo.Key(weights: weights, architecture: architecture)
        if let key, let hit = memo.results(for: key) {
            return (.evaluated(hit.results), .reusedFrom(file: hit.file))
        }
        let field = await evaluate(weights: weights, architecture: architecture)
        // Only a success is remembered: a failure may be transient.
        if let key, case .evaluated(let results) = field {
            memo.remember(results, for: key, file: file)
        }
        return (field, .evaluatedForThisFile)
    }

    private func results(weights: [[Float]], architecture: NetworkArchitecture) async throws -> ModelTestSetResults {
        guard !testSets.isEmpty else { throw EvaluationError.noTestSets }
        let probes = testSets.flatMap(\.probes)
        let (network, input): (ChessMPSNetwork, [Float]) = try await withCheckedThrowingContinuation { continuation in
            Self.buildQueue.async {
                continuation.resume(with: Result {
                    let network = try ChessMPSNetwork(.overwrittenByLoad, arch: architecture)
                    return (network, TacticalProbeRunner.encodedBoards(probes, encoding: network.inputEncoding))
                })
            }
        }
        network.network.commandQueue.label = "ModelTestSetEvaluator"
        let baseCount = network.network.trainableVariables.count + network.network.bnRunningStatsVariables.count
        guard weights.count >= baseCount else {
            throw EvaluationError.tooFewTensors(have: weights.count, need: baseCount)
        }
        try await network.network.loadWeights(Array(weights.prefix(baseCount)))

        let batch = await TacticalProbeRunner.runBatch(probes, encodedInput: input, against: network)
        guard batch.results.count == probes.count else {
            throw EvaluationError.resultCountMismatch(have: batch.results.count, want: probes.count)
        }
        guard batch.nonFinitePositions == 0 else {
            throw EvaluationError.nonFiniteOutputs(positions: batch.nonFinitePositions, of: probes.count)
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
            policyTailPrecision: architecture.policyTailPrecision,
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
        case nonFiniteOutputs(positions: Int, of: Int)
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
            case .nonFiniteOutputs(let positions, let of):
                return "\(positions) of \(of) positions have non-finite policy logits or value outputs"
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

/// The last few evaluations an evaluator made, keyed by architecture and a
/// SHA-256 of the base weights (`ModelTestSetEvaluator.resultsForSave`). A
/// digest, not the weights: an entry costs a few KB, not a model's size.
/// Most recently used last; `capacity` covers a session's champion and
/// trainer plus a GUI champion unchanged across periodic saves. Two
/// concurrent misses on one key both evaluate (correct, occasionally
/// redundant).
final class ModelTestSetMemo: Sendable {
    static let capacity = 4

    struct Key: Equatable, Sendable {
        let architecture: NetworkArchitecture
        let baseWeightsSHA256: [UInt8]

        /// Nil when `weights` lacks the architecture's base tensors (the
        /// evaluation then fails, and nothing is remembered).
        init?(weights: [[Float]], architecture: NetworkArchitecture) async {
            let baseCount = architecture.weightTensorPlan().count
            guard weights.count >= baseCount else { return nil }
            self.architecture = architecture
            self.baseWeightsSHA256 = await withCheckedContinuation { continuation in
                Self.digestQueue.async {
                    continuation.resume(returning: Self.digest(weights.prefix(baseCount)))
                }
            }
        }

        /// SHA-256 hashing of a model's weights is long synchronous work.
        private static let digestQueue = DispatchQueue(label: "ModelTestSetMemo.digest", qos: .userInitiated)

        /// Each tensor's length and raw bytes, so bit patterns decide (+0 and
        /// -0 differ; identical NaNs match).
        private static func digest(_ tensors: ArraySlice<[Float]>) -> [UInt8] {
            var hasher = SHA256()
            for tensor in tensors {
                withUnsafeBytes(of: UInt64(tensor.count).littleEndian) { hasher.update(bufferPointer: $0) }
                tensor.withUnsafeBytes { hasher.update(bufferPointer: $0) }
            }
            return Array(hasher.finalize())
        }
    }

    struct Hit: Sendable {
        let results: ModelTestSetResults
        let file: String
    }

    private struct Entry: Sendable {
        let key: Key
        let hit: Hit
    }

    private let entries = OSAllocatedUnfairLock<[Entry]>(initialState: [])

    func results(for key: Key) -> Hit? {
        entries.withLock { list in
            guard let index = list.lastIndex(where: { $0.key == key }) else { return nil }
            let entry = list.remove(at: index)
            list.append(entry)
            return entry.hit
        }
    }

    func remember(_ results: ModelTestSetResults, for key: Key, file: String) {
        entries.withLock { list in
            list.removeAll { $0.key == key }
            list.append(Entry(key: key, hit: Hit(results: results, file: file)))
            if list.count > Self.capacity {
                list.removeFirst(list.count - Self.capacity)
            }
        }
    }
}
