import Foundation

/// The loss math of `--eval-corpus` (`EvalCorpusCLI`): how one recorded
/// position is scored against a network's outputs, how scores are bucketed by
/// game phase and how they add up. Pure — no GPU, no files — so every rule is
/// pinned by unit tests (`CorpusEvaluationMetricsTests`).
///
/// Why these definitions:
///
/// - **Policy.** The network emits 4864 raw logits; only the legal moves'
///   logits enter the softmax, at temperature 1 — the same masking
///   `MoveSampler` applies before sampling. The score is the played move's
///   negative log-probability under that legal-only softmax (the corpus's
///   human move is the target, as in corpus replay). The log-sum-exp runs in
///   `Double` after subtracting the largest legal logit, so logits of any
///   magnitude give a finite answer.
/// - **Rank / top-k.** Competition ranking: a move's rank is one more than
///   the number of legal moves with a strictly larger logit, so moves whose
///   logits tie share the best rank among them. A tie at the top therefore
///   counts as top-1 for every tied move; the alternative (breaking ties by
///   move order) would make the result depend on move-generation order.
/// - **Value.** The W/D/L head's probabilities are relative to the side to
///   move; the target is the game's recorded result from that side,
///   unsmoothed (training smooths its labels; an evaluation measures against
///   the plain result). Cross-entropy is −ln p(result) on the probabilities
///   the network reports, not renormalized. A probability of exactly 0 gives
///   an infinite cross-entropy, which the caller reports as a non-finite
///   result rather than a number. The prediction is correct when no other
///   class has a strictly larger probability (ties count as correct, matching
///   the policy's tie rule).
enum CorpusEvaluationMetrics {

    /// The `k` of the reported top-k accuracy besides top-1.
    static let wideTopK = 5

    /// Score the played move under the legal-only softmax.
    ///
    /// - Parameters:
    ///   - legalLogits: the raw logits of the position's legal moves, one per
    ///     legal move. Must be non-empty and finite (the caller refuses a
    ///     forward pass with non-finite outputs before scoring).
    ///   - playedOffset: the played move's index in `legalLogits`.
    static func legalPolicyScore(legalLogits: UnsafeBufferPointer<Float>, playedOffset: Int) -> LegalPolicyScore {
        precondition(!legalLogits.isEmpty, "CorpusEvaluationMetrics.legalPolicyScore: no legal moves")
        precondition(legalLogits.indices.contains(playedOffset),
                     "CorpusEvaluationMetrics.legalPolicyScore: played offset \(playedOffset) outside \(legalLogits.count) legal moves")
        let played = Double(legalLogits[playedOffset])
        var largest = -Double.infinity
        var strictlyLarger = 0
        for logit in legalLogits {
            let value = Double(logit)
            if value > largest { largest = value }
            if value > played { strictlyLarger += 1 }
        }
        var expSum = 0.0
        for logit in legalLogits {
            expSum += Foundation.exp(Double(logit) - largest)
        }
        // expSum ≥ 1 (the largest term is exp(0)), so the log is finite.
        let negativeLogProbability = (largest - played) + Foundation.log(expSum)
        return LegalPolicyScore(negativeLogProbability: negativeLogProbability, rank: strictlyLarger + 1)
    }

    /// Score the value head's W/D/L probabilities against the recorded result.
    static func valueScore(probabilities: WDLProbabilities, result: WDLClass) -> ValueScore {
        let target = probabilities.probability(of: result)
        let isCorrect = WDLClass.allCases.allSatisfy { probabilities.probability(of: $0) <= target }
        return ValueScore(crossEntropy: -Foundation.log(target), isPredictionCorrect: isCorrect)
    }
}

/// One position's policy score under the legal-only softmax.
struct LegalPolicyScore: Equatable, Sendable {
    /// −ln p(played move), nats.
    let negativeLogProbability: Double
    /// 1-based competition rank of the played move among the legal moves.
    let rank: Int

    /// The played move is in the top `k` legal moves (ties share the best rank).
    func isInTop(_ k: Int) -> Bool { rank <= k }
}

/// One position's value score.
struct ValueScore: Equatable, Sendable {
    /// −ln p(recorded result), nats.
    let crossEntropy: Double
    /// No class has a strictly larger probability than the recorded result.
    let isPredictionCorrect: Bool
}

/// A game result from the side to move's point of view — the value head's
/// class order.
enum WDLClass: Int, CaseIterable, Sendable {
    case win = 0
    case draw = 1
    case loss = 2

    /// The recorded outcome (White's point of view) as seen by `sideToMove`.
    init(outcome: GameOutcome, sideToMove: PieceColor) {
        switch (outcome, sideToMove) {
        case (.draw, _): self = .draw
        case (.whiteWin, .white), (.blackWin, .black): self = .win
        case (.whiteWin, .black), (.blackWin, .white): self = .loss
        }
    }
}

/// The value head's probabilities for one position, side to move.
struct WDLProbabilities: Equatable, Sendable {
    let win: Double
    let draw: Double
    let loss: Double

    func probability(of result: WDLClass) -> Double {
        switch result {
        case .win: return win
        case .draw: return draw
        case .loss: return loss
        }
    }
}

/// Game phase by the position's game ply (0 = the start position).
enum CorpusEvaluationPlyBucket: CaseIterable, Sendable {
    case plies0To19
    case plies20To59
    case plies60AndLater

    init(ply: Int) {
        precondition(ply >= 0, "CorpusEvaluationPlyBucket: negative ply \(ply)")
        switch ply {
        case 0..<20: self = .plies0To19
        case 20..<60: self = .plies20To59
        default: self = .plies60AndLater
        }
    }

    /// The bucket's key in the JSON output.
    var jsonKey: String {
        switch self {
        case .plies0To19: return "ply_0_19"
        case .plies20To59: return "ply_20_59"
        case .plies60AndLater: return "ply_60_plus"
        }
    }
}

/// Running sums over scored positions. Means are derived only when at least
/// one position was added (`means` is nil otherwise — never a 0 that would
/// read as a perfect or a failed score).
struct CorpusEvaluationAccumulator: Equatable, Sendable {
    private(set) var positions = 0
    private(set) var negativeLogProbabilitySum = 0.0
    private(set) var top1Count = 0
    private(set) var wideTopKCount = 0
    private(set) var valueCrossEntropySum = 0.0
    private(set) var valueCorrectCount = 0
    private(set) var winProbabilitySum = 0.0
    private(set) var drawProbabilitySum = 0.0
    private(set) var lossProbabilitySum = 0.0

    mutating func add(policy: LegalPolicyScore, value: ValueScore, probabilities: WDLProbabilities) {
        positions += 1
        negativeLogProbabilitySum += policy.negativeLogProbability
        if policy.isInTop(1) { top1Count += 1 }
        if policy.isInTop(CorpusEvaluationMetrics.wideTopK) { wideTopKCount += 1 }
        valueCrossEntropySum += value.crossEntropy
        if value.isPredictionCorrect { valueCorrectCount += 1 }
        winProbabilitySum += probabilities.win
        drawProbabilitySum += probabilities.draw
        lossProbabilitySum += probabilities.loss
    }

    mutating func merge(_ other: CorpusEvaluationAccumulator) {
        positions += other.positions
        negativeLogProbabilitySum += other.negativeLogProbabilitySum
        top1Count += other.top1Count
        wideTopKCount += other.wideTopKCount
        valueCrossEntropySum += other.valueCrossEntropySum
        valueCorrectCount += other.valueCorrectCount
        winProbabilitySum += other.winProbabilitySum
        drawProbabilitySum += other.drawProbabilitySum
        lossProbabilitySum += other.lossProbabilitySum
    }

    /// Per-position means; nil when no position was added.
    var means: Means? {
        guard positions > 0 else { return nil }
        let count = Double(positions)
        return Means(
            policyNLL: negativeLogProbabilitySum / count,
            top1: Double(top1Count) / count,
            wideTopK: Double(wideTopKCount) / count,
            valueCrossEntropy: valueCrossEntropySum / count,
            valueAccuracy: Double(valueCorrectCount) / count,
            meanWinProbability: winProbabilitySum / count,
            meanDrawProbability: drawProbabilitySum / count,
            meanLossProbability: lossProbabilitySum / count)
    }

    struct Means: Equatable, Sendable {
        let policyNLL: Double
        let top1: Double
        let wideTopK: Double
        let valueCrossEntropy: Double
        let valueAccuracy: Double
        let meanWinProbability: Double
        let meanDrawProbability: Double
        let meanLossProbability: Double
    }
}

/// Accumulators per ply bucket. The overall figures are the merge of the
/// buckets — one source of truth, so the split always adds up to the total.
struct CorpusEvaluationTally: Equatable, Sendable {
    private(set) var byBucket: [CorpusEvaluationPlyBucket: CorpusEvaluationAccumulator] = [:]

    mutating func add(ply: Int, policy: LegalPolicyScore, value: ValueScore, probabilities: WDLProbabilities) {
        byBucket[CorpusEvaluationPlyBucket(ply: ply), default: CorpusEvaluationAccumulator()]
            .add(policy: policy, value: value, probabilities: probabilities)
    }

    mutating func merge(_ other: CorpusEvaluationTally) {
        for (bucket, accumulator) in other.byBucket {
            byBucket[bucket, default: CorpusEvaluationAccumulator()].merge(accumulator)
        }
    }

    func accumulator(for bucket: CorpusEvaluationPlyBucket) -> CorpusEvaluationAccumulator {
        byBucket[bucket] ?? CorpusEvaluationAccumulator()
    }

    /// Every bucket merged, in bucket declaration order.
    var overall: CorpusEvaluationAccumulator {
        var total = CorpusEvaluationAccumulator()
        for bucket in CorpusEvaluationPlyBucket.allCases {
            total.merge(accumulator(for: bucket))
        }
        return total
    }
}

/// Which games of a corpus an evaluation reads: the first or the last
/// `requestedGames` in corpus order (sealed shards in sequence order, games
/// in shard order). Deterministic for a given corpus.
struct CorpusEvaluationGameSelection: Equatable, Sendable {
    enum Origin: String, Sendable {
        case start
        case end
    }

    let origin: Origin
    let requestedGames: Int
    let corpusGames: Int
    /// Global game indices read.
    let gameRange: Range<Int>

    enum SelectionError: LocalizedError, Equatable {
        case nonPositiveGameCount(Int)
        case corpusTooSmall(requested: Int, corpusGames: Int)

        var errorDescription: String? {
            switch self {
            case .nonPositiveGameCount(let count):
                return "the game count must be at least 1 (got \(count))"
            case let .corpusTooSmall(requested, corpusGames):
                return "the corpus holds \(corpusGames) game(s), fewer than the \(requested) requested; ask for at most \(corpusGames)"
            }
        }
    }

    /// Refuses a non-positive count and a count larger than the corpus —
    /// never quietly evaluates fewer games than asked for.
    init(origin: Origin, requestedGames: Int, corpusGames: Int) throws {
        guard requestedGames > 0 else { throw SelectionError.nonPositiveGameCount(requestedGames) }
        guard requestedGames <= corpusGames else {
            throw SelectionError.corpusTooSmall(requested: requestedGames, corpusGames: corpusGames)
        }
        self.origin = origin
        self.requestedGames = requestedGames
        self.corpusGames = corpusGames
        switch origin {
        case .start: gameRange = 0..<requestedGames
        case .end: gameRange = (corpusGames - requestedGames)..<corpusGames
        }
    }
}
