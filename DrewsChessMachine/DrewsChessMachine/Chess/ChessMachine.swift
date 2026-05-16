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
    private var gameInProgress: Bool = false

    weak var delegate: (any ChessMachineDelegate)?

    /// Delay between moves for visualization. Zero = full speed.
    var moveDelay: Duration = .zero

    var gameState: GameState { engine?.state ?? .starting }
    var result: GameResult? { engine?.result }
    var moveHistory: [ChessMove] { engine?.moveHistory ?? [] }
    var isPlaying: Bool { gameInProgress }

    /// Play one game to completion against the supplied players.
    ///
    /// Structured concurrency: this is a single `async` call on the
    /// caller's Task rather than an unstructured inner `Task`. Cancelling
    /// the caller's Task propagates to the game loop at the next
    /// ply-boundary `Task.checkCancellation()` (or at any `await`
    /// downstream of `onChooseNextMove`), so `stopAll`-style pauses no
    /// longer have to wait out an entire 300-ply game before a slot can
    /// exit. That was the mechanism behind the 15-second save-session
    /// pause timeouts observed in the session log.
    ///
    /// On cancellation this throws `CancellationError` **before**
    /// calling `onGameEnded` on either player and **before** emitting the
    /// `gameEnded` delegate event. Partial-game positions that
    /// `MPSChessPlayer` recorded in per-game scratch during the cancelled
    /// game therefore never reach the replay buffer — they're silently
    /// discarded when `onNewGame` resets the scratch for the next game.
    /// This keeps fake-stalemate labels out of training data on every
    /// cancellation event.
    ///
    /// Non-cancellation player / engine errors propagate the same way:
    /// the machine emits `playerErrored`, then throws without calling
    /// `onGameEnded` or synthesizing a draw result.
    ///
    /// Throws `ChessMachineError.alreadyPlaying` if a game is already
    /// in progress (same contract as before).
    ///
    /// `maxPlies`, when non-nil, is a hard ceiling on the number of
    /// plies before the loop short-circuits and returns
    /// `.terminatedEarly`. Used by self-play to bound games that the
    /// chess engine's natural termination rules (50-move, 3-fold,
    /// checkmate, stalemate) failed to end. Arena, human play, and
    /// the forward-pass demo pass `nil` (default) and always receive
    /// `.terminatedNormally(...)`.
    @discardableResult
    func beginNewGame(
        white: any ChessPlayer,
        black: any ChessPlayer,
        maxPlies: Int? = nil
    ) async throws -> RawGameResult {
        if gameInProgress {
            throw ChessMachineError.alreadyPlaying
        }

        whitePlayer = white
        blackPlayer = black
        engine = ChessGameEngine()
        gameInProgress = true
        defer { gameInProgress = false }

        white.onNewGame(true)
        black.onNewGame(false)

        return try await runGameLoop(maxPlies: maxPlies)
    }

    // MARK: - Game Loop

    private func runGameLoop(maxPlies: Int?) async throws -> RawGameResult {
        guard let engine else { return .terminatedNormally(.stalemate) }

        let gameStart = CFAbsoluteTimeGetCurrent()
        var whiteThinkMs: Double = 0
        var blackThinkMs: Double = 0
        var whiteMoveCount = 0
        var blackMoveCount = 0
        var lastMove: ChessMove?
        /// Set when the loop bails because `engine.moveHistory.count`
        /// hit the `maxPlies` cap before the engine declared a
        /// natural termination. Drives the `.terminatedEarly` return
        /// path below; the player/delegate path is unchanged and
        /// sees this as a stalemate (the same `engine.result ??
        /// .stalemate` fallback the prior code used for "engine left
        /// result unset" cases).
        var terminatedByMaxPlies = false

        // The engine owns `currentLegalMoves` — computed once at init
        // and refreshed inside `applyMoveAndAdvance` after each move —
        // so `MoveGenerator.legalMoves(for:)` still runs exactly once
        // per ply and the engine validates every incoming move against
        // that same list.

        while engine.result == nil {
            // Cooperative cancellation point between plies. Throwing
            // here bails *before* the end-of-game emit block below —
            // which is intentional: we don't want to flush fake-
            // stalemate outcomes into training data on a real cancel
            // (save-session pause, concurrency step-down, session
            // stop). `Task.sleep` and `player.onChooseNextMove`
            // further down are also cancellation points whose
            // `CancellationError` propagates the same way.
            try Task.checkCancellation()

            if moveDelay > .zero {
                try await Task.sleep(for: moveDelay)
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
                    legalMoves: engine.currentLegalMoves
                )
                let moveTimeMs = (CFAbsoluteTimeGetCurrent() - moveStart) * 1000

                if isWhiteTurn {
                    whiteThinkMs += moveTimeMs
                    whiteMoveCount += 1
                } else {
                    blackThinkMs += moveTimeMs
                    blackMoveCount += 1
                }

                // Apply the move; the engine validates it against
                // `currentLegalMoves`, then generates the next ply's
                // legal moves and uses them for end-detection. We pick
                // up the refreshed list from `engine.currentLegalMoves`
                // on the next iteration.
                try engine.applyMoveAndAdvance(move)
                lastMove = move

                let snapshotState = engine.state
                let event = DelegateEvent.didApplyMove(move: move, newState: snapshotState)
                emit(event)

                // Max-plies guard. Check AFTER the move is applied so
                // we end the game on the exact ply that hits the cap
                // (rather than the ply after). Only trips when the
                // engine has not already declared a natural result on
                // this same move — natural termination takes priority.
                if let cap = maxPlies,
                   engine.result == nil,
                   engine.moveHistory.count >= cap {
                    terminatedByMaxPlies = true
                    break
                }
            } catch is CancellationError {
                // Propagate cancellation through runGameLoop so the
                // caller sees `CancellationError` and skips the
                // end-of-game bookkeeping path below.
                throw CancellationError()
            } catch {
                let event = DelegateEvent.playerErrored(player: player, error: error)
                emit(event)
                throw error
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

        // For the player + delegate path, max-plies-terminated games
        // are reported as a stalemate — the same fallback the prior
        // code used whenever `engine.result` was unset at end-of-loop.
        // The self-play driver consumes the `.terminatedEarly` return
        // value separately and drops the game (recorded plies on the
        // player's per-game scratch never get flushed, and the next
        // `onNewGame` zeroes them).
        let finalResult = engine.result ?? .stalemate
        let finalState = engine.state
        whitePlayer?.onGameEnded(finalResult, finalState: finalState)
        blackPlayer?.onGameEnded(finalResult, finalState: finalState)

        emit(.gameEnded(result: finalResult, finalState: finalState, stats: stats))

        if terminatedByMaxPlies {
            return .terminatedEarly
        }
        return .terminatedNormally(finalResult)
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
