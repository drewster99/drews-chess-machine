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

// MARK: - Delegate

/// Delegate for observing game events — UI updates, logging, training data collection.
protocol ChessMachineDelegate: AnyObject {
    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState)
    func chessMachine(_ machine: ChessMachine, gameEndedWith result: GameResult, finalState: GameState, stats: GameStats)
    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error)
}

extension ChessMachineDelegate {
    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState) {}
    func chessMachine(_ machine: ChessMachine, gameEndedWith result: GameResult, finalState: GameState, stats: GameStats) {}
    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error) {}
}

// MARK: - Chess Machine

/// Orchestrates chess games between two ChessPlayer instances.
///
/// Uses ChessGameEngine internally for move validation and game-end detection.
/// Runs the game loop asynchronously. Tracks per-move timing for both players.
///
/// Marked @unchecked Sendable — access serialized: one game task at a time,
/// beginNewGame cancels any prior game before starting a new one.
final class ChessMachine: @unchecked Sendable {
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

    /// Start a new game. Cancels any game in progress.
    @discardableResult
    func beginNewGame(white: any ChessPlayer, black: any ChessPlayer) -> Task<GameResult, Never> {
        cancelGame()

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

        while engine.result == nil, !Task.isCancelled {
            if moveDelay > .zero {
                do { try await Task.sleep(for: moveDelay) }
                catch { break }
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
                    newGameState: engine.state
                )
                let moveTimeMs = (CFAbsoluteTimeGetCurrent() - moveStart) * 1000

                if isWhiteTurn {
                    whiteThinkMs += moveTimeMs
                    whiteMoveCount += 1
                } else {
                    blackThinkMs += moveTimeMs
                    blackMoveCount += 1
                }

                try engine.applyMove(move)
                lastMove = move
                delegate?.chessMachine(self, didApplyMove: move, newState: engine.state)
            } catch {
                delegate?.chessMachine(self, playerErrored: player, error: error)
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
        whitePlayer?.onGameEnded(finalResult, finalState: engine.state)
        blackPlayer?.onGameEnded(finalResult, finalState: engine.state)
        delegate?.chessMachine(self, gameEndedWith: finalResult, finalState: engine.state, stats: stats)

        return finalResult
    }
}
