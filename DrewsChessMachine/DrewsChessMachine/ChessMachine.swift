import Foundation

// MARK: - Per-Game Stats

struct GameStats: Sendable {
    let totalMoves: Int
    let whiteMoves: Int
    let blackMoves: Int
    let whiteThinkingTimeMs: Double
    let blackThinkingTimeMs: Double
    let totalGameTimeMs: Double

    var avgWhiteMoveTimeMs: Double {
        whiteMoves > 0 ? whiteThinkingTimeMs / Double(whiteMoves) : 0
    }

    var avgBlackMoveTimeMs: Double {
        blackMoves > 0 ? blackThinkingTimeMs / Double(blackMoves) : 0
    }

    var avgMoveTimeMs: Double {
        totalMoves > 0 ? (whiteThinkingTimeMs + blackThinkingTimeMs) / Double(totalMoves) : 0
    }
}

// MARK: - Errors

enum ChessMachineError: LocalizedError {
    case alreadyPlaying

    var errorDescription: String? {
        switch self {
        case .alreadyPlaying:
            return "A game is already in progress"
        }
    }
}

// MARK: - Delegate

/// Delegate for observing game events — UI updates, logging, training data collection.
///
/// Delegate methods are invoked on `ChessMachine.delegateQueue`, a private
/// serial dispatch queue running at `.userInteractive` priority. The serial
/// queue guarantees ordering (didApplyMove for move N strictly before move
/// N+1, and gameEndedWith strictly after the last didApplyMove). The game
/// loop dispatches asynchronously, so delegate work never blocks game
/// progress. Implementations that touch `@MainActor` state should redispatch
/// to the main actor themselves.
protocol ChessMachineDelegate: AnyObject {
    func chessMachine(
        _ machine: ChessMachine,
        didApplyMove move: ChessMove,
        newState: GameState
    )
    func chessMachine(
        _ machine: ChessMachine,
        gameEndedWith result: GameResult,
        finalState: GameState,
        stats: GameStats
    )
    func chessMachine(
        _ machine: ChessMachine,
        playerErrored player: any ChessPlayer,
        error: any Error
    )
}

extension ChessMachineDelegate {
    func chessMachine(
        _ machine: ChessMachine,
        didApplyMove move: ChessMove,
        newState: GameState
    ) {}
    func chessMachine(
        _ machine: ChessMachine,
        gameEndedWith result: GameResult,
        finalState: GameState,
        stats: GameStats
    ) {}
    func chessMachine(
        _ machine: ChessMachine,
        playerErrored player: any ChessPlayer,
        error: any Error
    ) {}
}

// MARK: - Chess Machine

/// Orchestrates chess games between two ChessPlayer instances.
///
/// Uses ChessGameEngine internally for move application and game-end detection.
/// Runs the game loop asynchronously. Tracks per-move timing for both players.
///
/// Marked @unchecked Sendable — access serialized: one game task at a time,
/// and `beginNewGame` throws if a game is already in progress (it does not
/// silently cancel one).
final class ChessMachine: @unchecked Sendable {
    /// Serial queue used for all delegate callbacks. `.userInteractive` so
    /// that delegate work runs promptly without contending for the main
    /// queue. Serial so callbacks are delivered in the same order the game
    /// loop produced them.
    static let delegateQueue = DispatchQueue(label: "drewschess.delegate", qos: .userInteractive)

    private var engine: ChessGameEngine?
    private var whitePlayer: (any ChessPlayer)?
    private var blackPlayer: (any ChessPlayer)?
    private var gameTask: Task<GameResult, Never>?

    weak var delegate: (any ChessMachineDelegate)?

    /// Delay between moves for visualization. Zero = full speed.
    var moveDelay: Duration = .zero

    var gameState: GameState { engine?.state ?? .starting }
    var result: GameResult? { engine?.result }
    var moveHistory: [ChessMove] { engine?.moveHistory ?? [] }
    var isPlaying: Bool { gameTask != nil && result == nil }

    /// Start a new game. Throws if a game is already in progress.
    @discardableResult
    func beginNewGame(white: any ChessPlayer, black: any ChessPlayer) throws -> Task<GameResult, Never> {
        if isPlaying {
            throw ChessMachineError.alreadyPlaying
        }

        whitePlayer = white
        blackPlayer = black
        engine = ChessGameEngine()

        white.onNewGame(true)
        black.onNewGame(false)

        let task = Task(priority: .userInitiated) { [weak self] in
            await self?.runGameLoop() ?? .stalemate
        }
        gameTask = task
        return task
    }

    /// Request that the in-progress game stop after its current move.
    ///
    /// **Stop is cooperative, not immediate.** This sets the cancellation
    /// flag on the game task and clears the local handle, but the running
    /// game loop checks `Task.isCancelled` only between moves. The current
    /// move (and its delegate callbacks) will complete before the loop
    /// exits, so callers should not assume `isPlaying == false` immediately
    /// after calling this. If a fast cancellation is ever needed, the
    /// MPSGraph inference call would also need to be made cancellable.
    func cancelGame() {
        gameTask?.cancel()
        gameTask = nil
    }

    // MARK: - Game Loop

    private func runGameLoop() async -> GameResult {
        guard let engine else { return .stalemate }

        let gameStart = CFAbsoluteTimeGetCurrent()
        var whiteThinkMs: Double = 0
        var blackThinkMs: Double = 0
        var whiteMoveCount = 0
        var blackMoveCount = 0
        var lastMove: ChessMove?

        // Generate the initial legal moves once. From here on, every
        // applyMoveAndAdvance returns the next ply's legal moves, so we
        // never call legalMoves() more than once per ply.
        var currentLegalMoves = MoveGenerator.legalMoves(for: engine.state)

        while engine.result == nil, !Task.isCancelled {
            if moveDelay > .zero {
                do {
                    try await Task.sleep(for: moveDelay)
                } catch {
                    break
                }
            }

            let isWhiteTurn = engine.state.currentPlayer == .white
            let player: any ChessPlayer
            if isWhiteTurn {
                guard let p = whitePlayer else { break }
                player = p
            } else {
                guard let p = blackPlayer else { break }
                player = p
            }

            do {
                let moveStart = CFAbsoluteTimeGetCurrent()
                let move = try await player.onChooseNextMove(
                    opponentMove: lastMove,
                    newGameState: engine.state,
                    legalMoves: currentLegalMoves
                )
                let moveTimeMs = (CFAbsoluteTimeGetCurrent() - moveStart) * 1000

                if isWhiteTurn {
                    whiteThinkMs += moveTimeMs
                    whiteMoveCount += 1
                } else {
                    blackThinkMs += moveTimeMs
                    blackMoveCount += 1
                }

                // Apply the move; engine generates the next ply's legal
                // moves and uses them for end-detection. We reuse the same
                // list on the next iteration.
                currentLegalMoves = try engine.applyMoveAndAdvance(move)
                lastMove = move

                let snapshotState = engine.state
                let event = DelegateEvent.didApplyMove(move: move, newState: snapshotState)
                emit(event)
            } catch {
                let event = DelegateEvent.playerErrored(player: player, error: error)
                emit(event)
                break
            }
        }

        let totalGameMs = (CFAbsoluteTimeGetCurrent() - gameStart) * 1000
        let stats = GameStats(
            totalMoves: whiteMoveCount + blackMoveCount,
            whiteMoves: whiteMoveCount,
            blackMoves: blackMoveCount,
            whiteThinkingTimeMs: whiteThinkMs,
            blackThinkingTimeMs: blackThinkMs,
            totalGameTimeMs: totalGameMs
        )

        let finalResult = engine.result ?? .stalemate
        let finalState = engine.state
        whitePlayer?.onGameEnded(finalResult, finalState: finalState)
        blackPlayer?.onGameEnded(finalResult, finalState: finalState)

        emit(.gameEnded(result: finalResult, finalState: finalState, stats: stats))

        return finalResult
    }

    /// Dispatch a delegate event onto the serial delegate queue. Fires and
    /// forgets — the game loop never blocks on delegate work. The serial
    /// queue preserves event ordering even though we never wait. The box
    /// holds the machine strongly so the machine cannot be released between
    /// the async dispatch and the delegate-queue delivery, which would
    /// otherwise drop the final gameEnded event when the caller's local
    /// reference goes out of scope (e.g. on continuous-play Stop).
    private func emit(_ event: DelegateEvent) {
        let box = DelegateBox(machine: self, delegate: delegate)
        Self.delegateQueue.async {
            box.deliver(event)
        }
    }
}

// MARK: - Delegate Dispatch Internals

/// Discrete delegate-call payloads. Sendable so they can cross the
/// dispatch-queue boundary cleanly.
private enum DelegateEvent: @unchecked Sendable {
    case didApplyMove(move: ChessMove, newState: GameState)
    case gameEnded(result: GameResult, finalState: GameState, stats: GameStats)
    case playerErrored(player: any ChessPlayer, error: any Error)
}

/// Shim that carries the machine and (weak) delegate to the delegate queue.
/// Marked `@unchecked Sendable` because `ChessMachine` itself promises
/// serialized access — the delegate is only touched on the serial delegate
/// queue. The machine is held strongly so it survives until every queued
/// event has been delivered; the delegate is weak so events become a no-op
/// when the UI owning the delegate has gone away.
private final class DelegateBox: @unchecked Sendable {
    let machine: ChessMachine
    weak var delegate: AnyObject?

    init(machine: ChessMachine, delegate: (any ChessMachineDelegate)?) {
        self.machine = machine
        self.delegate = delegate
    }

    func deliver(_ event: DelegateEvent) {
        guard let delegate = delegate as? any ChessMachineDelegate else { return }
        switch event {
        case .didApplyMove(let move, let newState):
            delegate.chessMachine(machine, didApplyMove: move, newState: newState)
        case .gameEnded(let result, let finalState, let stats):
            delegate.chessMachine(machine, gameEndedWith: result, finalState: finalState, stats: stats)
        case .playerErrored(let player, let error):
            delegate.chessMachine(machine, playerErrored: player, error: error)
        }
    }
}
