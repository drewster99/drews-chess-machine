import Foundation

/// Replays recorded games from a corpus into a `ReplayBuffer`, producing the
/// exact same encoded positions self-play would. For each game it walks the
/// move list through a `ChessGameEngine`, encodes each position with the
/// target architecture's `BoardEncoder`, and stages it into an `ActiveGame` —
/// the same staging + reverse-ply flush path the live self-play driver uses,
/// so a replayed game is byte-identical to the original (the linchpin
/// invariant for replay validity).
///
/// A recorded game is replayed to its recorded end. Its own result is the
/// ground truth for the value targets, and draws the players did not claim
/// (an unclaimed threefold repetition, a position past the fifty-move mark,
/// insufficient material by DCM's definition) must not stop the replay —
/// Lichess players legally play on past them, and ending the replay there
/// once threw away every such game whole. So the engine here never
/// adjudicates draws (`.serverAuthoritative`); only a move that cannot be
/// made — an illegal move, or any move after checkmate or stalemate — marks
/// a game as corrupt, and that game is reported, never silently dropped.
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

    /// Replay one recorded game into the buffer and report what happened to
    /// it. A rejected game contributes no positions: its staged plies are
    /// discarded with its `ActiveGame`, so a half-replayed game never reaches
    /// the buffer.
    func feed(_ game: GameRecord) -> CorpusReplayFeedOutcome {
        guard !game.moves.isEmpty else { return .skippedEmpty }
        // Standard-start games only for now; a FEN setup would need a FEN
        // parser and would truncate the repetition/history planes anyway.
        guard game.startFEN == nil else { return .skippedStartFEN }

        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        let active = ActiveGame(
            workerId: gameCounter,
            whiteNetwork: network,
            blackNetwork: network,
            // +2 headroom so the per-side staging cap can never be exhausted
            // by an off-by-one (a `recordPly` overflow is a fatalError).
            capPlies: game.moves.count + 2,
            schedule: schedule,
            // A recorded game's moves come from the record; it never draws.
            random: DCMRandom.seededFromSystem()
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
                return .rejected(ply: ply, error: error)
            }
            ply += 1
        }

        let result = Self.gameResult(for: game.outcome)
        // `flush` returns nil only for a game with no recorded plies, and every
        // move above recorded one before it was applied — so nil here is a
        // defect in this function, not a property of the corpus.
        guard let flushed = active.flush(buffer: buffer, result: result) else {
            preconditionFailure("CorpusReplayFeeder: flush of a fully replayed \(game.moves.count)-move game appended nothing")
        }
        return .fed(positions: flushed.positions)
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

/// What `CorpusReplayFeeder.feed` did with one recorded game.
enum CorpusReplayFeedOutcome {
    /// Every position was appended to the replay buffer.
    case fed(positions: Int)
    /// The game has no moves.
    case skippedEmpty
    /// The game starts from a FEN setup, which replay does not support.
    case skippedStartFEN
    /// The move at `ply` (0-based) could not be made — illegal, or played
    /// after checkmate or stalemate. Nothing from the game was appended.
    case rejected(ply: Int, error: any Error)
}

/// Running totals of the games a corpus-replay run consumed, shared by every
/// place the run feeds games (pre-fill, exact-resume reconstruction, the
/// training loop) so the counts and the per-rejection report have one
/// definition.
struct CorpusReplayFeedTally {
    /// Positions appended to the replay buffer.
    private(set) var positions = 0
    /// Games consumed from the corpus, whatever happened to them — the
    /// `games=` / `games_fed` compute axis.
    private(set) var games = 0
    /// Games skipped as empty or FEN-setup.
    private(set) var skipped = 0
    /// Games whose move list could not be replayed.
    private(set) var rejected = 0

    /// Count one consumed game. Returns a description of a rejection for the
    /// caller to log, or nil when there is nothing to report.
    mutating func record(_ outcome: CorpusReplayFeedOutcome) -> String? {
        games += 1
        switch outcome {
        case .fed(let fedPositions):
            positions += fedPositions
            return nil
        case .skippedEmpty, .skippedStartFEN:
            skipped += 1
            return nil
        case .rejected(let ply, let error):
            rejected += 1
            return "rejected at ply \(ply): \(error)"
        }
    }

    /// ` rejected=N skipped=N`, appended after `games=` on the progress lines.
    var countsSuffix: String { " rejected=\(rejected) skipped=\(skipped)" }
}
