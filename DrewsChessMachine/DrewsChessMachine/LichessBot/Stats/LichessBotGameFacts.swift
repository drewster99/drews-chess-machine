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
    /// The run, segment and cumulative trainer step of the file the
    /// generation was loaded from, when the file carries a lineage record
    /// (`LichessBotGenerationInfo.lineage`, `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md`
    /// §3.6); nil for in-memory sources and files without one.
    let lineageRunID: String?
    let segmentIndex: Int?
    let cumTrainerStep: Int?
    /// How the weights were trained (`ModelTrainingHistory`): the record's
    /// own, or for a record written before generations kept it, the played
    /// file's when the index could read it (`LichessBotPlayedFileHistories`);
    /// nil when neither says.
    let trainingHistory: ModelTrainingHistory?
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
    /// How DCM picked among its own policy's moves (§11 L1).
    let choice: LichessBotMoveChoiceFacts
    /// DCM's think time and time left (§11 L2); nil for a game without a
    /// clock (correspondence, or a record without the clock setup).
    let clock: LichessBotClockFacts?
}

/// How DCM's moves related to its own policy (§11 L1): how often it played
/// the policy's top move, the probability its sampling gave the move it
/// chose, and how many decisions were close to a random pick.
struct LichessBotMoveChoiceFacts: Sendable, Codable, Equatable {
    /// Decisions that recorded the policy's top moves (the rest can't say
    /// whether the top move was played).
    let decisionsWithTopMoves: Int
    /// Of those, decisions that played the policy's top move.
    let topMoveChosen: Int
    /// Σ of the sampling probability of the chosen move, over every
    /// decision.
    let sumChosenProbability: Double
    /// Decisions whose post-temperature distribution was essentially
    /// uniform.
    let randomish: Int
}

/// DCM's clock use in one game (§11 L2), from the server clocks recorded
/// after each of its moves.
struct LichessBotClockFacts: Sendable, Codable, Equatable {
    /// DCM moves whose think time is known: the move and DCM's previous
    /// move both carry DCM's clock. DCM's first move is never among them
    /// (Lichess starts a side's clock after its first move).
    let thinkTimeMoves: Int
    /// Σ think time over those moves: previous clock + increment − clock.
    /// Time the opponent gave DCM counts against it, so a move can come out
    /// negative; it is kept, not clamped, so the sum stays the clocks' own.
    let sumThinkMilliseconds: Double
    /// DCM's clock after its last move that carries one; nil with none.
    let finalClockMilliseconds: Int?
}

/// Everything about a game beyond its index row's plain fields that the
/// statistics need.
struct LichessBotGameFacts: Sendable, Codable, Equatable {
    /// The draw rule DCM's own engine saw in the final position.
    let localDrawCondition: ChessDrawCondition?
    /// Nil for a game that did not start from the standard position.
    let moves: LichessBotGameMoveFacts?
    /// The opening Lichess's export named (§11 L4); nil without an export.
    let openingECO: String?
    let openingName: String?
    /// Moves Lichess refused while the game went on (not those that raced
    /// the game's end), and game-stream reconnections (§11 L6).
    let rejectedMoves: Int
    let streamReconnects: Int

    init(localDrawCondition: ChessDrawCondition?, moves: LichessBotGameMoveFacts?, openingECO: String?, openingName: String?, rejectedMoves: Int, streamReconnects: Int) {
        self.localDrawCondition = localDrawCondition
        self.moves = moves
        self.openingECO = openingECO
        self.openingName = openingName
        self.rejectedMoves = rejectedMoves
        self.streamReconnects = streamReconnects
    }

    /// The facts the record alone states: a generation's training history
    /// is the one it recorded.
    init(record: LichessBotGameRecord) {
        self.init(record: record, trainingHistory: \.trainingHistory)
    }

    /// The facts of `record`, each generation's training history from
    /// `trainingHistory` (the index's: the recorded one, else the played
    /// file's).
    init(record: LichessBotGameRecord, trainingHistory: (LichessBotGenerationInfo) -> ModelTrainingHistory?) {
        localDrawCondition = record.outcome.localDrawCondition
        let standard = record.setup.variant == "standard"
            && LichessBotPositionTracker.isStandardStart(record.setup.initialFen)
        moves = standard ? LichessBotGameMoveFacts(record: record, trainingHistory: trainingHistory) : nil
        openingECO = record.openingECO
        openingName = record.openingName
        rejectedMoves = record.rejectedMoves.count
        streamReconnects = record.streamReconnects
    }
}

extension LichessBotGameMoveFacts {
    /// The move facts the record alone states (see
    /// `LichessBotGameFacts.init(record:)`).
    init(record: LichessBotGameRecord) {
        self.init(record: record, trainingHistory: \.trainingHistory)
    }

    init(record: LichessBotGameRecord, trainingHistory: (LichessBotGenerationInfo) -> ModelTrainingHistory?) {
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

        choice = LichessBotMoveChoiceFacts(
            decisionsWithTopMoves: decided.filter { !$0.decision.topMoves.isEmpty }.count,
            topMoveChosen: decided.filter { $0.decision.topMoves.first?.uci == $0.decision.uci }.count,
            sumChosenProbability: decided.reduce(0.0) { $0 + Double($1.decision.chosenProbability) },
            randomish: decided.filter(\.decision.randomish).count
        )
        clock = Self.clockFacts(ourMoves, ourColor: record.ourColor, incrementMilliseconds: record.setup.clockIncrementMilliseconds, initialMilliseconds: record.setup.clockInitialMilliseconds)

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
                lineageRunID: generation.lineage?.lineageRunID,
                segmentIndex: generation.lineage?.segmentIndex,
                cumTrainerStep: generation.lineage?.cumTrainerStep,
                trainingHistory: trainingHistory(generation),
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

    /// DCM's clock use, or nil when the game had no clock (both the initial
    /// time and the increment are needed: think time is the previous clock
    /// plus the increment minus the clock after the move).
    private static func clockFacts(_ ourMoves: [LichessBotGameRecord.Move], ourColor: LichessBotColorName, incrementMilliseconds: Int?, initialMilliseconds: Int?) -> LichessBotClockFacts? {
        guard let increment = incrementMilliseconds, initialMilliseconds != nil else { return nil }
        let clocks = ourMoves.map { ourColor == .white ? $0.whiteClockMilliseconds : $0.blackClockMilliseconds }
        var thinkTimeMoves = 0
        var sumThink = 0.0
        for index in clocks.indices.dropFirst() {
            guard let previous = clocks[index - 1], let current = clocks[index] else { continue }
            thinkTimeMoves += 1
            sumThink += Double(previous + increment - current)
        }
        return LichessBotClockFacts(
            thinkTimeMoves: thinkTimeMoves,
            sumThinkMilliseconds: sumThink,
            finalClockMilliseconds: clocks.compactMap { $0 }.last
        )
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
