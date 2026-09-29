import Foundation

extension NumericsAudit {

    /// One audited position: its encoded board, the policy indices of its
    /// legal moves, and — when the game's result is known — the value
    /// target from the side to move (0 win, 1 draw, 2 loss).
    struct AuditPosition: Sendable {
        let board: [Float]
        let legalPolicyIndices: [Int]
        let valueTarget: Int?
    }

    /// The fixed position set the dynamic checks run on. Deterministic: the
    /// same inputs always give the same positions in the same order, so
    /// audits compare across checkpoints and over time.
    struct PositionSet: Sendable {
        let positions: [AuditPosition]
        let summary: PositionSetSummary
    }

    struct PositionSetSummary: Codable, Sendable {
        let total: Int
        let startPosition: Int
        let corpusPositions: Int
        let corpusSource: String?
        /// Why no corpus positions are included, when none are.
        let corpusNote: String?
        let lichessGames: Int
        let lichessPositions: Int
        /// Filed games left out because they couldn't be read or replayed,
        /// each with the reason.
        let lichessGamesSkipped: [String]
        /// Why no Lichess bot positions are included, when none are.
        let lichessNote: String?
        let withValueTarget: Int
    }

    /// Corpus sampling: games drawn from the shard, and positions from each.
    static let corpusGameSampleCount = 300
    static let corpusPositionsPerGame = 3
    /// Fixed seed so the corpus sample never changes between audits.
    static let corpusSampleSeed: UInt64 = 0x4E55_4D45_5249_4353
    /// Cap on Lichess bot positions (every ply of every filed game, oldest
    /// file first, up to this many).
    static let lichessPositionCap = 4096

    /// Build the position set: the start position, a seeded sample from
    /// `corpusShardURL` when one is given, and every ply of the Lichess
    /// bot's filed games under `lichessDirectory` when there are any.
    /// Synchronous file reads: call off the cooperative pool.
    static func buildPositionSet(
        encoding: InputEncoding,
        corpusShardURL: URL?,
        lichessDirectory: LichessBotDataDirectory?
    ) throws -> PositionSet {
        var positions: [AuditPosition] = []

        let startEngine = ChessGameEngine(state: .starting, adjudication: .serverAuthoritative)
        positions.append(position(from: startEngine, encoding: encoding, valueTarget: nil))

        var corpusCount = 0
        var corpusSource: String?
        var corpusNote: String?
        if let corpusShardURL {
            let shard = try GameCorpusShardIO.readSealed(at: corpusShardURL)
            corpusSource = corpusShardURL.path
            var rng = NumericsAuditSampleRNG(seed: corpusSampleSeed)
            let games = shard.games
            if games.isEmpty {
                corpusNote = "the corpus shard holds no games"
            } else {
                for _ in 0..<min(corpusGameSampleCount, games.count) {
                    let game = games[Int(rng.next() % UInt64(games.count))]
                    guard game.moves.count > 1 else { continue }
                    var plies = Set<Int>()
                    for _ in 0..<corpusPositionsPerGame {
                        plies.insert(Int(rng.next() % UInt64(game.moves.count)))
                    }
                    let sampled = try samplePositions(in: game, atPlies: plies.sorted(), encoding: encoding)
                    positions.append(contentsOf: sampled)
                    corpusCount += sampled.count
                }
            }
        } else {
            corpusNote = "no corpus shard was given"
        }

        var lichessGames = 0
        var lichessCount = 0
        var lichessSkipped: [String] = []
        var lichessNote: String?
        if let lichessDirectory {
            let files = try LichessBotIndex.recordFiles(in: lichessDirectory).sorted { $0.url.path < $1.url.path }
            if files.isEmpty {
                lichessNote = "no filed Lichess bot games"
            }
            for file in files where lichessCount < lichessPositionCap {
                // One unreadable or unreplayable record costs that game, not
                // the audit; each one is listed in the summary.
                do {
                    let record = try LichessBotIndex.readRecord(at: file.url)
                    let game = try lichessPositions(record: record, encoding: encoding, cap: lichessPositionCap - lichessCount)
                    if !game.isEmpty {
                        lichessGames += 1
                        lichessCount += game.count
                        positions.append(contentsOf: game)
                    }
                } catch {
                    lichessSkipped.append("\(file.url.lastPathComponent): \(error.localizedDescription)")
                }
            }
        } else {
            lichessNote = "no Lichess bot data directory was given"
        }

        let summary = PositionSetSummary(
            total: positions.count,
            startPosition: 1,
            corpusPositions: corpusCount,
            corpusSource: corpusSource,
            corpusNote: corpusCount == 0 ? (corpusNote ?? "the corpus sample produced no positions") : nil,
            lichessGames: lichessGames,
            lichessPositions: lichessCount,
            lichessGamesSkipped: lichessSkipped,
            lichessNote: lichessCount == 0 ? (lichessNote ?? "the Lichess bot games produced no positions") : nil,
            withValueTarget: positions.filter { $0.valueTarget != nil }.count
        )
        return PositionSet(positions: positions, summary: summary)
    }

    private static func position(from engine: ChessGameEngine, encoding: InputEncoding, valueTarget: Int?) -> AuditPosition {
        let state = engine.state
        let board = BoardEncoder.encode(state, history: engine.recentStates, encoding: encoding)
        let indices = engine.currentLegalMoves.map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
        return AuditPosition(board: board, legalPolicyIndices: indices, valueTarget: valueTarget)
    }

    /// The value target for the side to move, from a game result given from
    /// White's point of view.
    private static func valueTarget(whiteScore: GameOutcome, sideToMove: PieceColor) -> Int {
        switch (whiteScore, sideToMove) {
        case (.draw, _): return 1
        case (.whiteWin, .white), (.blackWin, .black): return 0
        case (.whiteWin, .black), (.blackWin, .white): return 2
        }
    }

    private static func samplePositions(in game: GameRecord, atPlies plies: [Int], encoding: InputEncoding) throws -> [AuditPosition] {
        let start = try game.startFEN.map { try FENParser.parse($0) } ?? .starting
        let engine = ChessGameEngine(state: start, adjudication: .serverAuthoritative)
        var out: [AuditPosition] = []
        var wanted = plies[...]
        for (ply, move) in game.moves.enumerated() {
            guard let next = wanted.first else { break }
            if ply == next {
                wanted = wanted.dropFirst()
                if !engine.currentLegalMoves.isEmpty {
                    let target = valueTarget(whiteScore: game.outcome, sideToMove: engine.state.currentPlayer)
                    out.append(position(from: engine, encoding: encoding, valueTarget: target))
                }
            }
            try engine.applyMoveAndAdvance(move)
        }
        return out
    }

    private static func lichessPositions(record: LichessBotGameRecord, encoding: InputEncoding, cap: Int) throws -> [AuditPosition] {
        guard record.setup.variant == "standard" else { return [] }
        let start: GameState = LichessBotPositionTracker.isStandardStart(record.setup.initialFen)
            ? .starting
            : try FENParser.parse(record.setup.initialFen)
        let outcome: GameOutcome?
        switch record.outcome.pgnResult {
        case "1-0": outcome = .whiteWin
        case "0-1": outcome = .blackWin
        case "1/2-1/2": outcome = .draw
        default: outcome = nil
        }
        let engine = ChessGameEngine(state: start, adjudication: .serverAuthoritative)
        var out: [AuditPosition] = []
        for move in record.moves where out.count < cap {
            guard !engine.currentLegalMoves.isEmpty else { break }
            let target = outcome.map { valueTarget(whiteScore: $0, sideToMove: engine.state.currentPlayer) }
            out.append(position(from: engine, encoding: encoding, valueTarget: target))
            guard let parsed = ChessMove.parseUCI(move.uciAsGiven, legal: engine.currentLegalMoves, state: engine.state) else {
                throw NumericsAuditError.recordReplayFailed("Lichess game \(record.gameID): move \(move.uciAsGiven) at ply \(move.ply) does not replay")
            }
            try engine.applyMoveAndAdvance(parsed)
        }
        return out
    }
}

/// A small deterministic generator for the corpus sample (the SplitMix64 mixer), so
/// the audit's positions never depend on the system random source.
struct NumericsAuditSampleRNG {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
