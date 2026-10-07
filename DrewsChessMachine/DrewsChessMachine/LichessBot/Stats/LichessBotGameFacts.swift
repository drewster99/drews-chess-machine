import Foundation

/// The thresholds of the self-assessment statistics
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.6), in one declaration. Per-game
/// facts are reduced with them when an index row is built, so changing any
/// of them means bumping `LichessBotIndex.schemaVersion`: otherwise rows
/// reduced under the old values would be mixed with rows reduced under the
/// new ones.
enum LichessBotSelfAssessmentDefinition {
    /// A result probability at or above this on consecutive DCM moves is a
    /// "held" win or loss (OD-14).
    static let heldProbability: Float = 0.80
    /// How many consecutive DCM moves (each with a decision) make a hold.
    /// Two, so a one-move spike before a recapture does not count.
    static let heldMoveCount = 2
    /// The result's own probability from which the result is "decided"
    /// (OD-15).
    static let decisiveProbability: Float = 0.80
    /// DCM's Nth moves at which the calibration table compares the value
    /// head's prediction with the result (OD-13).
    static let checkpointMoveNumbers = [10, 20, 40]
    /// Expected-score buckets of the reliability chart: `[0, 0.1)` …
    /// `[0.9, 1.0]`.
    static let bucketCount = 10

    /// The bucket an expected score falls in. One function for the reducer
    /// and the tests: `E` is a `Float` sum, so it can land a rounding step
    /// above 1.0 (clamped into the top bucket) or just below a tenth (it
    /// stays in the lower bucket — the tests pin that rather than assume
    /// decimal edges).
    static func bucketIndex(expectedScore: Float) -> Int {
        min(bucketCount - 1, max(0, Int((Double(expectedScore) * Double(bucketCount)).rounded(.down))))
    }

    /// The ply of DCM's Nth move (full-move number N of its color).
    static func ply(ofOurMove moveNumber: Int, ourColor: LichessBotColorName) -> Int {
        2 * (moveNumber - 1) + (ourColor == .white ? 0 : 1)
    }
}

/// When a won or lost game's result became settled in the network's own
/// eyes (§3.6). An enum rather than an optional ply, so "draw", "no data"
/// and "never" are never one nil.
enum LichessBotDecisivePly: Sendable, Codable, Equatable {
    /// A draw, or a game without a result: the statistic does not apply.
    case notApplicable
    /// Won or lost, but no DCM move carries a decision.
    case noDecisions
    /// Won or lost, and DCM's last decision put the result below the
    /// threshold.
    case never
    /// The ply of the earliest DCM move from which the result's probability
    /// stays at or above the threshold through DCM's last decision.
    case atPly(Int)
}

/// Identity of a model DCM played with, for the Models pane (§3.8, OD-11).
/// A file is named by its bytes' SHA-256, so reloading the same file after
/// a relaunch is one model; an in-memory snapshot by source, model ID and
/// training step. The header's `content_sha256` is a different hash and is
/// never mixed in, so one file never becomes two keys.
enum LichessBotModelKey: Sendable, Hashable, Codable {
    case file(sha256: String)
    case snapshot(sourceKind: LichessBotModelSourceKind, modelID: String, trainingStep: Int?)
}

/// The facts of one generation that played in a game: what identifies its
/// weights, and how many of DCM's moves it decided.
struct LichessBotGenerationFacts: Sendable, Codable, Equatable {
    let sourceKind: LichessBotModelSourceKind
    let modelID: String
    let trainingStep: Int?
    let fileSHA256: String?
    /// The followed lineage's run, segment and cumulative trainer step, when
    /// the generation was loaded from a file carrying a lineage record.
    /// Nil until generations record their lineage
    /// (`LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` §3.6).
    let lineageRunID: String?
    let segmentIndex: Int?
    let cumTrainerStep: Int?
    /// DCM moves this generation decided (moves in the final move list,
    /// so a decision whose move was taken back does not count).
    let ourMoves: Int

    var modelKey: LichessBotModelKey {
        if let fileSHA256 {
            return .file(sha256: fileSHA256)
        }
        return .snapshot(sourceKind: sourceKind, modelID: modelID, trainingStep: trainingStep)
    }
}

/// The per-move facts of one game, reduced once when its index row is built
/// so that statistics never open a record file (§4.2). Nil in a game that did
/// not start from the standard position: the record builder colors moves by
/// ply parity, and the bot declines such games, so ply-based numbers would
/// be meaningless; those games count as "no move data".
struct LichessBotGameMoveFacts: Sendable, Codable, Equatable {
    /// A calibration checkpoint DCM reached with a decision: the value
    /// head's W/D/L at its Nth move.
    struct Checkpoint: Sendable, Codable, Equatable {
        let moveNumber: Int
        let win: Float
        let draw: Float
        let loss: Float
    }

    /// One non-empty expected-score bucket: how many of DCM's decisions
    /// fell in it, and the sum of their expected scores.
    struct Bucket: Sendable, Codable, Equatable {
        let index: Int
        let positions: Int
        let sumExpected: Double
    }

    let ourMoveCount: Int
    let ourMovesWithDecision: Int
    /// Checkpoints DCM reached with a decision, in move-number order.
    let checkpoints: [Checkpoint]
    /// Checkpoints DCM reached without a decision there ("missing").
    let checkpointsWithoutDecision: [Int]
    /// Non-empty buckets only, by index.
    let expectedScoreBuckets: [Bucket]
    /// The ply of the first DCM move of the first held-win run; nil with
    /// none (`ourMovesWithDecision` tells "no decisions" apart).
    let heldWinStartPly: Int?
    let heldLossStartPly: Int?
    let decisive: LichessBotDecisivePly
    /// Every generation of the record, in the record's order (order of
    /// first use), with the DCM moves each decided.
    let generations: [LichessBotGenerationFacts]
    /// DCM decisions whose generation reference names no listed generation
    /// (a corrupt record); counted rather than given to a neighbor.
    let decisionsWithoutGeneration: Int
}

/// Everything about a game beyond its index row's plain fields that the
/// statistics need.
struct LichessBotGameFacts: Sendable, Codable, Equatable {
    /// The draw rule DCM's own engine saw in the final position.
    let localDrawCondition: ChessDrawCondition?
    /// Nil for a game that did not start from the standard position.
    let moves: LichessBotGameMoveFacts?

    init(localDrawCondition: ChessDrawCondition?, moves: LichessBotGameMoveFacts?) {
        self.localDrawCondition = localDrawCondition
        self.moves = moves
    }

    init(record: LichessBotGameRecord) {
        localDrawCondition = record.outcome.localDrawCondition
        let standard = record.setup.variant == "standard"
            && LichessBotPositionTracker.isStandardStart(record.setup.initialFen)
        moves = standard ? LichessBotGameMoveFacts(record: record) : nil
    }
}

extension LichessBotGameMoveFacts {
    init(record: LichessBotGameRecord) {
        typealias Definition = LichessBotSelfAssessmentDefinition
        let ourMoves = record.moves.filter(\.ours).sorted { $0.ply < $1.ply }
        let decided = ourMoves.compactMap { move in move.decision.map { (ply: move.ply, decision: $0) } }
        ourMoveCount = ourMoves.count
        ourMovesWithDecision = decided.count

        let ourMovesByPly = Dictionary(ourMoves.map { ($0.ply, $0) }, uniquingKeysWith: { first, _ in first })
        var checkpoints: [Checkpoint] = []
        var missing: [Int] = []
        for moveNumber in Definition.checkpointMoveNumbers {
            guard let move = ourMovesByPly[Definition.ply(ofOurMove: moveNumber, ourColor: record.ourColor)] else { continue }
            if let decision = move.decision {
                checkpoints.append(Checkpoint(moveNumber: moveNumber, win: decision.win, draw: decision.draw, loss: decision.loss))
            } else {
                missing.append(moveNumber)
            }
        }
        self.checkpoints = checkpoints
        checkpointsWithoutDecision = missing

        var buckets: [Int: (positions: Int, sum: Double)] = [:]
        for (_, decision) in decided {
            let expected = decision.expectedScore
            let index = Definition.bucketIndex(expectedScore: expected)
            var bucket = buckets[index] ?? (0, 0)
            bucket.positions += 1
            bucket.sum += Double(expected)
            buckets[index] = bucket
        }
        expectedScoreBuckets = buckets.keys.sorted().compactMap { index in
            buckets[index].map { Bucket(index: index, positions: $0.positions, sumExpected: $0.sum) }
        }

        heldWinStartPly = Self.firstHeldRunStart(ourMoves) { $0.win }
        heldLossStartPly = Self.firstHeldRunStart(ourMoves) { $0.loss }

        switch record.outcome.ourScore {
        case .some(1):
            decisive = Self.decisivePly(decided) { $0.win }
        case .some(0):
            decisive = Self.decisivePly(decided) { $0.loss }
        default:
            decisive = .notApplicable
        }

        var movesByGeneration = Array(repeating: 0, count: record.generations.count)
        var withoutGeneration = 0
        for move in ourMoves where move.decision != nil {
            if let index = Self.generationIndex(of: move, in: record.generations) {
                movesByGeneration[index] += 1
            } else {
                withoutGeneration += 1
            }
        }
        generations = zip(record.generations, movesByGeneration).map { generation, moves in
            LichessBotGenerationFacts(
                sourceKind: generation.sourceKind,
                modelID: generation.modelID,
                trainingStep: generation.trainingStep,
                fileSHA256: generation.fileSHA256,
                lineageRunID: nil,
                segmentIndex: nil,
                cumTrainerStep: nil,
                ourMoves: moves
            )
        }
        decisionsWithoutGeneration = withoutGeneration
    }

    /// The generation that decided `move`: by `generationIndex` when the
    /// record has it; in a record written before it existed, by
    /// `generationID`, which is exact there because such records list at
    /// most one generation per ID. Nil when the reference names no listed
    /// generation, or (impossible in a builder-made record) when an ID
    /// matches several.
    static func generationIndex(of move: LichessBotGameRecord.Move, in generations: [LichessBotGenerationInfo]) -> Int? {
        if let index = move.generationIndex {
            return generations.indices.contains(index) ? index : nil
        }
        guard let id = move.generationID else { return nil }
        let matches = generations.indices.filter { generations[$0].generationID == id }
        return matches.count == 1 ? matches[0] : nil
    }

    /// The ply of the first DCM move of the first run of
    /// `heldMoveCount` consecutive DCM moves, each with a decision, whose
    /// `probability` is at or above `heldProbability`. A DCM move without a
    /// decision breaks a run: nothing is known about that position.
    private static func firstHeldRunStart(_ ourMoves: [LichessBotGameRecord.Move], probability: (LichessBotMoveDecision) -> Float) -> Int? {
        typealias Definition = LichessBotSelfAssessmentDefinition
        var runStart: Int?
        var runLength = 0
        for move in ourMoves {
            guard let decision = move.decision, probability(decision) >= Definition.heldProbability else {
                runStart = nil
                runLength = 0
                continue
            }
            if runLength == 0 {
                runStart = move.ply
            }
            runLength += 1
            if runLength >= Definition.heldMoveCount {
                return runStart
            }
        }
        return nil
    }

    /// The earliest decided DCM move from which `probability` stays at or
    /// above `decisiveProbability` on every later decided DCM move. Moves
    /// without a decision are skipped (unlike a held run, which they
    /// break): the statistic asks when the network's view settled, and a
    /// position it never evaluated says nothing either way.
    private static func decisivePly(_ decided: [(ply: Int, decision: LichessBotMoveDecision)], probability: (LichessBotMoveDecision) -> Float) -> LichessBotDecisivePly {
        guard !decided.isEmpty else {
            return .noDecisions
        }
        var start: Int?
        for entry in decided.reversed() {
            guard probability(entry.decision) >= LichessBotSelfAssessmentDefinition.decisiveProbability else { break }
            start = entry.ply
        }
        return start.map { .atPly($0) } ?? .never
    }
}
