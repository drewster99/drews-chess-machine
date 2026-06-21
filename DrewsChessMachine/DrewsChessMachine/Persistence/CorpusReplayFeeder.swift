import Foundation

/// Replays recorded games from a corpus into a `ReplayBuffer`, producing the
/// exact same encoded positions self-play would. For each game it walks the
/// move list through a `ChessGameEngine`, encodes each position with the
/// target architecture's `BoardEncoder`, and stages it into an `ActiveGame` —
/// the same staging + reverse-ply flush path the live self-play driver uses,
/// so a replayed game is byte-identical to the original (the linchpin
/// invariant for replay validity).
///
/// Single-threaded by design: the offline runner feeds games sequentially, so
/// the one reused encode scratch needs no synchronization.
final class CorpusReplayFeeder {
    private let network: ChessMPSNetwork
    private let buffer: ReplayBuffer
    private let schedule: SamplingSchedule
    /// Full network-input length (history encodings need more than one frame);
    /// `recordPly` copies just the first frame out of this scratch.
    private let tensorLength: Int
    private let scratch: UnsafeMutablePointer<Float>
    /// Distinct id per fed game so the buffer's per-game caps see them as
    /// separate games (each fresh `ActiveGame` resets its intra-worker index).
    private var gameCounter: UInt16 = 0

    init(network: ChessMPSNetwork,
         buffer: ReplayBuffer,
         schedule: SamplingSchedule = .selfPlay) {
        self.network = network
        self.buffer = buffer
        self.schedule = schedule
        self.tensorLength = BoardEncoder.tensorLength(for: network.inputEncoding)
        self.scratch = UnsafeMutablePointer<Float>.allocate(capacity: tensorLength)
        self.scratch.initialize(repeating: 0, count: tensorLength)
    }

    deinit {
        scratch.deinitialize(count: tensorLength)
        scratch.deallocate()
    }

    /// Replay one recorded game into the buffer. Returns the number of
    /// positions appended (0 if the game was skipped — empty, a FEN-setup game,
    /// or an illegal/corrupt move encountered during replay).
    @discardableResult
    func feed(_ game: GameRecord) -> Int {
        guard !game.moves.isEmpty else { return 0 }
        // Standard-start games only for now; a FEN setup would need a FEN
        // parser and would truncate the repetition/history planes anyway.
        guard game.startFEN == nil else { return 0 }

        let engine = ChessGameEngine()
        let active = ActiveGame(
            workerId: gameCounter,
            whiteNetwork: network,
            blackNetwork: network,
            // +2 headroom so the per-side staging cap can never be exhausted
            // by an off-by-one (a `recordPly` overflow is a fatalError).
            capPlies: game.moves.count + 2,
            schedule: schedule
        )
        gameCounter = gameCounter &+ 1

        let encoding = network.inputEncoding
        var ply = 0
        for move in game.moves {
            let state = engine.state
            let sideToMove = state.currentPlayer
            BoardEncoder.encode(
                state,
                history: engine.recentStates,
                into: UnsafeMutableBufferPointer(start: scratch, count: tensorLength),
                encoding: encoding
            )
            let policyIndex = PolicyEncoding.policyIndex(move, currentPlayer: sideToMove)
            // Non-pawn piece count for the material-phase bucket — the same
            // per-ply calculation the self-play driver performs.
            var matCount = 0
            for sq in state.board where (sq != nil && sq?.type != .pawn) {
                matCount += 1
            }
            let materialCount = UInt8(min(matCount, Int(UInt8.max)))
            active.recordPly(
                side: sideToMove,
                encodedBoardSrc: UnsafePointer(scratch),
                policyIndex: policyIndex,
                samplingTau: schedule.tau(forPly: ply),
                materialCount: materialCount
            )
            do {
                try engine.applyMoveAndAdvance(move)
            } catch {
                // Corrupt / illegal move (e.g. an imported game that doesn't
                // replay cleanly): abandon this game; its staged plies are
                // dropped with the discarded ActiveGame.
                return 0
            }
            ply += 1
        }

        let result = Self.gameResult(for: game.outcome)
        let flushed = active.flush(buffer: buffer, result: result)
        return flushed?.positions ?? 0
    }

    /// The corpus stores only an objective W/D/L outcome; map it to a
    /// `GameResult` for `ActiveGame.flush`, which only needs winner-vs-draw to
    /// sign the per-position value targets (the exact draw/termination type is
    /// irrelevant to the training signal).
    private static func gameResult(for outcome: GameOutcome) -> GameResult {
        switch outcome {
        case .whiteWin: return .checkmate(winner: .white)
        case .blackWin: return .checkmate(winner: .black)
        case .draw:     return .stalemate
        }
    }
}
