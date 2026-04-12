import SwiftUI

// MARK: - Result Types

struct EvaluationResult: Sendable {
    let topMoves: [MoveVisualization]
    let textOutput: String
    let inputTensor: [Float]
}

// MARK: - Game Watcher

/// Holds live game state mutated by the ChessMachine delegate queue.
///
/// **Not @Observable.** SwiftUI doesn't observe its mutations directly —
/// instead, ContentView polls `snapshot()` on a 100ms timer and copies the
/// values into local @State. That decouples UI redraw frequency from game
/// throughput: continuous self-play can run hundreds of moves per second
/// while the UI updates at most 10 times per second, and the game loop
/// never waits for SwiftUI invalidation.
///
/// Mutations come from two sources:
/// 1. The ChessMachine delegate queue (didApplyMove, gameEnded, playerErrored).
/// 2. Direct calls from ContentView actions (resetCurrentGame, markPlaying).
/// Both go through `lock`, so reads from any thread are safe.
final class GameWatcher: ChessMachineDelegate, @unchecked Sendable {
    /// Snapshot of all displayable values, taken atomically. Sendable so it
    /// can flow from any thread to the main actor.
    struct Snapshot: Sendable {
        var state: GameState = .starting
        var result: GameResult?
        var moveCount = 0
        var isPlaying = false
        var lastGameStats: GameStats?

        var totalGames = 0
        var totalMoves = 0
        var totalGameTimeMs: Double = 0
        var totalWhiteThinkMs: Double = 0
        var totalBlackThinkMs: Double = 0
        var sessionStartTime: CFAbsoluteTime?

        var whiteCheckmates = 0
        var blackCheckmates = 0
        var stalemates = 0
        var fiftyMoveDraws = 0
        var insufficientMaterialDraws = 0
    }

    private let lock = NSLock()
    private var s = Snapshot()

    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return s
    }

    func resetCurrentGame() {
        lock.lock()
        defer { lock.unlock() }
        s.state = .starting
        s.result = nil
        s.moveCount = 0
        // Keep lastGameStats — show previous game until the next one ends
    }

    func resetAll() {
        lock.lock()
        defer { lock.unlock() }
        s = Snapshot()
    }

    func markPlaying(_ playing: Bool) {
        lock.lock()
        defer { lock.unlock() }
        s.isPlaying = playing
    }

    func markSessionStart() {
        lock.lock()
        defer { lock.unlock() }
        if s.sessionStartTime == nil {
            s.sessionStartTime = CFAbsoluteTimeGetCurrent()
        }
    }

    // MARK: - Delegate (called on ChessMachine.delegateQueue, never main)

    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState) {
        lock.lock()
        defer { lock.unlock() }
        s.state = newState
        s.moveCount += 1
    }

    func chessMachine(
        _ machine: ChessMachine,
        gameEndedWith result: GameResult,
        finalState: GameState,
        stats: GameStats
    ) {
        lock.lock()
        defer { lock.unlock() }
        s.result = result
        s.state = finalState
        s.lastGameStats = stats
        s.isPlaying = false

        s.totalGames += 1
        s.totalMoves += stats.totalMoves
        s.totalGameTimeMs += stats.totalGameTimeMs
        s.totalWhiteThinkMs += stats.whiteThinkingTimeMs
        s.totalBlackThinkMs += stats.blackThinkingTimeMs
        // Move counting handed off to totalMoves; zero the per-game counter
        // atomically so display helpers using `totalMoves + moveCount` don't
        // double-count between gameEnded and the next resetCurrentGame call.
        s.moveCount = 0

        switch result {
        case .checkmate(let winner):
            if winner == .white {
                s.whiteCheckmates += 1
            } else {
                s.blackCheckmates += 1
            }
        case .stalemate:
            s.stalemates += 1
        case .drawByFiftyMoveRule:
            s.fiftyMoveDraws += 1
        case .drawByInsufficientMaterial:
            s.insufficientMaterialDraws += 1
        }
    }

    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error) {
        lock.lock()
        defer { lock.unlock() }
        s.isPlaying = false
    }
}

// MARK: - Snapshot Display Helpers

extension GameWatcher.Snapshot {
    var avgMoveTimeMs: Double {
        totalMoves > 0 ? (totalWhiteThinkMs + totalBlackThinkMs) / Double(totalMoves) : 0
    }

    var avgGameTimeMs: Double {
        totalGames > 0 ? totalGameTimeMs / Double(totalGames) : 0
    }

    func gamesPerHour(now: CFAbsoluteTime) -> Double {
        guard let start = sessionStartTime else { return 0 }
        let elapsed = now - start
        guard elapsed > 0, totalGames > 0 else { return 0 }
        return Double(totalGames) / (elapsed / 3600)
    }

    func movesPerHour(now: CFAbsoluteTime) -> Double {
        guard let start = sessionStartTime else { return 0 }
        let elapsed = now - start
        // Include moves from the in-progress game so the rate updates smoothly
        // instead of sagging during a game and jumping when it completes.
        let moves = totalMoves + moveCount
        guard elapsed > 0, moves > 0 else { return 0 }
        return Double(moves) / (elapsed / 3600)
    }

    func sessionElapsedSeconds(now: CFAbsoluteTime) -> Double {
        guard let start = sessionStartTime else { return 0 }
        return max(0, now - start)
    }

    /// Fixed-layout stats text. Every section is always present — values show
    /// "--" when no data exists. Structure never changes so layout stays stable.
    func statsText(continuousPlay: Bool) -> String {
        let dash = "--"
        let now = CFAbsoluteTimeGetCurrent()

        // Status line
        let status: String
        if isPlaying {
            let turn = state.currentPlayer == .white ? "White" : "Black"
            let check = MoveGenerator.isInCheck(state, color: state.currentPlayer) ? " CHECK" : ""
            status = "\(turn) to move (move \(moveCount + 1))\(check)"
        } else if let result {
            switch result {
            case .checkmate(let winner):
                status = "\(winner == .white ? "White" : "Black") wins by checkmate"
            case .stalemate:
                status = "Draw by stalemate"
            case .drawByFiftyMoveRule:
                status = "Draw by fifty-move rule"
            case .drawByInsufficientMaterial:
                status = "Draw by insufficient material"
            }
        } else {
            status = dash
        }

        // Session — totalMoves only counts completed games' moves; add the
        // in-progress game's moveCount so the displayed count and rate update
        // smoothly instead of jumping at game-end.
        let liveMoves = totalMoves + moveCount
        let sGames = totalGames > 0 ? totalGames.formatted(.number.grouping(.automatic)) : dash
        let sMoves = liveMoves > 0 ? liveMoves.formatted(.number.grouping(.automatic)) : dash
        let sTime = sessionStartTime != nil
            ? Self.formatHMS(seconds: sessionElapsedSeconds(now: now))
            : dash
        let sAvgMove = totalMoves > 0 ? String(format: "%.2f ms", avgMoveTimeMs) : dash
        let sAvgGame = totalGames > 0 ? String(format: "%.1f ms", avgGameTimeMs) : dash
        let sMovesHr = liveMoves > 0
            ? Int(movesPerHour(now: now).rounded()).formatted(.number.grouping(.automatic))
            : dash
        let sGamesHr = totalGames > 0
            ? Int(gamesPerHour(now: now).rounded()).formatted(.number.grouping(.automatic))
            : dash

        // Last Game — only shown when not in active continuous play. During
        // continuous play it has no value (it's already part of session totals
        // and changes too fast to read).
        let lastGameSection: String
        if continuousPlay {
            lastGameSection = ""
        } else if let stats = lastGameStats {
            let lgMoves = String(format: "%d (%dW + %dB)", stats.totalMoves, stats.whiteMoves, stats.blackMoves)
            let lgTime = String(format: "%.1f ms", stats.totalGameTimeMs)
            let lgAvg = String(format: "%.2f ms", stats.avgMoveTimeMs)
            let lgWAvg = String(format: "%.2f ms", stats.avgWhiteMoveTimeMs)
            let lgBAvg = String(format: "%.2f ms", stats.avgBlackMoveTimeMs)
            lastGameSection = """
                Last Game
                  Moves:     \(lgMoves)
                  Time:      \(lgTime)
                  Avg move:  \(lgAvg)
                  White avg: \(lgWAvg)
                  Black avg: \(lgBAvg)


                """
        } else {
            lastGameSection = ""
        }

        return """
            Status: \(status)

            \(lastGameSection)Session
              Games:     \(sGames)
              Moves:     \(sMoves)
              Time:      \(sTime)
              Avg move:  \(sAvgMove)
              Avg game:  \(sAvgGame)
              Moves/hr:  \(sMovesHr)
              Games/hr:  \(sGamesHr)

            Results
              Checkmate:      \(whiteCheckmates + blackCheckmates)\(pct(whiteCheckmates + blackCheckmates))
                White wins:     \(whiteCheckmates)\(pct(whiteCheckmates))
                Black wins:     \(blackCheckmates)\(pct(blackCheckmates))
              Stalemate:      \(stalemates)\(pct(stalemates))
              50-move draw:   \(fiftyMoveDraws)\(pct(fiftyMoveDraws))
              Insufficient:   \(insufficientMaterialDraws)\(pct(insufficientMaterialDraws))
            """
    }

    private func pct(_ count: Int) -> String {
        guard totalGames > 0 else { return "" }
        return String(format: "  (%.1f%%)", Double(count) / Double(totalGames) * 100)
    }

    static func formatHMS(seconds: Double) -> String {
        let totalTenths = Int((seconds * 10).rounded())
        let tenths = totalTenths % 10
        let totalSeconds = totalTenths / 10
        let h = totalSeconds / 3600
        let m = (totalSeconds % 3600) / 60
        let s = totalSeconds % 60
        return String(format: "%02d:%02d:%02d.%d", h, m, s, tenths)
    }
}

// MARK: - Content View

struct ContentView: View {
    // Network
    @State private var network: ChessMPSNetwork?
    @State private var runner: ChessRunner?
    @State private var networkStatus = ""
    @State private var isBuilding = false

    // Inference
    @State private var inferenceResult: EvaluationResult?
    @State private var isEvaluating = false
    @State private var selectedOverlay = 0

    // Game — gameWatcher is mutated by the delegate queue and is NOT
    // SwiftUI-observed. A 100ms timer copies its `snapshot()` into
    // `gameSnapshot`, which is what the body actually reads. This caps UI
    // refresh rate regardless of game throughput.
    //
    // `gameWatcher` MUST be `@State`, not `let`. SwiftUI may reconstruct
    // ContentView's struct across body invocations; a plain `let` initializer
    // would build a fresh `GameWatcher` each time and any in-flight machine
    // (which only holds the delegate via `weak`) would lose its delegate.
    @State private var gameWatcher = GameWatcher()
    @State private var gameSnapshot = GameWatcher.Snapshot()
    @State private var continuousPlay = false
    @State private var continuousTask: Task<Void, Never>?

    /// 100 ms heartbeat that pulls the latest snapshot from `gameWatcher`
    /// into `gameSnapshot`. Standard SwiftUI Combine timer pattern — the
    /// publisher is created when the view struct is initialized and SwiftUI
    /// manages the subscription lifecycle via `.onReceive` below.
    private let snapshotTimer = Timer.publish(
        every: 1.0/60.0, on: .main, in: .common
    ).autoconnect()

    private var networkReady: Bool { network != nil }
    private var isBusy: Bool { isBuilding || isEvaluating || gameSnapshot.isPlaying || continuousPlay }
    private var isGameMode: Bool { gameSnapshot.isPlaying || gameSnapshot.totalGames > 0 }

    private var displayedPieces: [Piece?] {
        if isGameMode { return gameSnapshot.state.board }
        return GameState.starting.board
    }

    private var overlayLabel: String {
        if selectedOverlay == 0 { return "Top Moves" }
        return "Channel \(selectedOverlay - 1): \(TensorChannelNames.names[selectedOverlay - 1])"
    }

    private var currentOverlay: BoardOverlay {
        if isGameMode { return .none }
        guard let result = inferenceResult else { return .none }
        if selectedOverlay == 0 {
            return .topMoves(result.topMoves)
        } else {
            let start = (selectedOverlay - 1) * 64
            return .channel(Array(result.inputTensor[start..<start + 64]))
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Chess Neural Network")
                .font(.title2)
                .fontWeight(.semibold)

            Text(
                "Forward pass through a ~2.9M parameter convolutional network using MPSGraph " +
                "on the GPU. Weights are randomly initialized (He initialization) — no training " +
                "has occurred."
            )
                .font(.callout)
                .foregroundStyle(.secondary)

            // Buttons
            HStack(spacing: 8) {
                Button("Build Network") { buildNetwork() }
                    .disabled(isBusy || networkReady)

                Button("Run Forward Pass") { runForwardPass() }
                    .disabled(isBusy || !networkReady)
                    .keyboardShortcut(.return)

                Divider().frame(height: 20)

                Button("Play Game") { playSingleGame() }
                    .disabled(isBusy || !networkReady)

                if continuousPlay {
                    Button("Stop") { stopContinuousPlay() }
                        .keyboardShortcut(.escape, modifiers: [])
                } else {
                    Button("Play Continuous") { startContinuousPlay() }
                        .disabled(isBusy || !networkReady)
                }

                if isBusy {
                    ProgressView().controlSize(.small)
                    Text(busyLabel).foregroundStyle(.secondary)
                }
            }

            // Board + text side by side
            HStack(alignment: .top, spacing: 24) {
                VStack(spacing: 6) {
                    if inferenceResult != nil, !isGameMode {
                        Text(overlayLabel)
                            .font(.system(.subheadline, design: .monospaced))
                    }

                    HStack(spacing: 8) {
                        let leftDisabled = inferenceResult == nil || selectedOverlay == 0 || isGameMode
                        Button(
                            action: { navigateOverlay(-1) },
                            label: {
                                Image(systemName: "chevron.left").font(.title3).frame(width: 24)
                            }
                        )
                        .buttonStyle(.plain)
                        .disabled(leftDisabled)
                        .opacity(leftDisabled ? 0.2 : 0.6)

                        ChessBoardView(pieces: displayedPieces, overlay: currentOverlay)

                        let rightDisabled = inferenceResult == nil || selectedOverlay == 18 || isGameMode
                        Button(
                            action: { navigateOverlay(1) },
                            label: {
                                Image(systemName: "chevron.right").font(.title3).frame(width: 24)
                            }
                        )
                        .buttonStyle(.plain)
                        .disabled(rightDisabled)
                        .opacity(rightDisabled ? 0.2 : 0.6)
                    }
                }
                .frame(minWidth: 320, maxWidth: 420)

                // Text panel — single Text view for stable layout
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        if !networkStatus.isEmpty {
                            Text(networkStatus)
                                .foregroundStyle(.secondary)
                        }

                        if isGameMode {
                            Text(gameSnapshot.statsText(continuousPlay: continuousPlay))
                        }

                        if let result = inferenceResult, !isGameMode {
                            Text(result.textOutput)
                        }
                    }
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .layoutPriority(1)

            // Input tensor channel strip
            if let result = inferenceResult, !isGameMode {
                Divider()
                HStack(spacing: 2) {
                    ForEach(0..<18, id: \.self) { channel in
                        let start = channel * 64
                        let isSelected = selectedOverlay == channel + 1
                        VStack(spacing: 1) {
                            ChannelBoardView(values: Array(result.inputTensor[start..<start + 64]))
                                .frame(width: 40, height: 40)
                                .clipShape(RoundedRectangle(cornerRadius: 2))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 2)
                                        .stroke(
                                            isSelected ? Color.accentColor : Color.gray.opacity(0.2),
                                            lineWidth: isSelected ? 2 : 0.5
                                        )
                                )
                            Text(TensorChannelNames.shortNames[channel])
                                .font(.system(size: 8))
                                .foregroundStyle(isSelected ? .primary : .tertiary)
                                .lineLimit(1)
                        }
                    }
                }
            }
        }
        .padding(24)
        .frame(minWidth: 900, minHeight: 600)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.leftArrow) { navigateOverlay(-1); return .handled }
        .onKeyPress(.rightArrow) { navigateOverlay(1); return .handled }
        .onReceive(snapshotTimer) { _ in
            // Pull the latest game state into @State at most every 100ms.
            // Cheap (single locked struct copy) and bounds UI work even
            // when the game loop is doing hundreds of moves per second.
            gameSnapshot = gameWatcher.snapshot()
        }
    }

    private var busyLabel: String {
        if isBuilding { return "Building network..." }
        if gameSnapshot.isPlaying { return "Game \(gameSnapshot.totalGames + 1), move \(gameSnapshot.moveCount)..." }
        return "Running inference..."
    }

    // MARK: - Navigation

    private func navigateOverlay(_ direction: Int) {
        guard inferenceResult != nil, !isGameMode else { return }
        let next = selectedOverlay + direction
        if next >= 0, next <= 18 { selectedOverlay = next }
    }

    // MARK: - Actions

    private func buildNetwork() {
        isBuilding = true
        networkStatus = ""

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Self.performBuild()
            }.value

            switch result {
            case .success(let net):
                network = net
                runner = ChessRunner(network: net)
                networkStatus = """
                    Network built in \(String(format: "%.1f", net.buildTimeMs)) ms
                    Parameters: ~2,917,383 (~2.9M)
                    Architecture: 18x8x8 -> stem(128)
                      -> 8 res blocks -> policy(4096) + value(1)
                    """
            case .failure(let error):
                network = nil
                runner = nil
                networkStatus = "Build failed: \(error.localizedDescription)"
            }
            isBuilding = false
        }
    }

    private func runForwardPass() {
        guard let runner else { return }
        isEvaluating = true
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()

        Task {
            let evalResult = await Task.detached(priority: .userInitiated) {
                Self.performInference(with: runner)
            }.value
            inferenceResult = evalResult
            selectedOverlay = 0
            isEvaluating = false
        }
    }

    private func playSingleGame() {
        inferenceResult = nil
        gameWatcher.resetCurrentGame()
        gameWatcher.markSessionStart()
        gameWatcher.markPlaying(true)
        // Synchronously refresh the snapshot so isBusy reflects the new
        // playing state immediately — the polling task only runs every
        // 100ms, which would otherwise leave a window where the Play
        // button stayed enabled and a fast double-click could spawn two
        // concurrent ChessMachine instances against the same gameWatcher.
        gameSnapshot = gameWatcher.snapshot()

        Task { [network] in
            guard let network else { return }
            let machine = ChessMachine()
            machine.delegate = gameWatcher
            let white = MPSChessPlayer(name: "White", network: network)
            let black = MPSChessPlayer(name: "Black", network: network)
            do {
                let task = try machine.beginNewGame(white: white, black: black)
                _ = await task.value
            } catch {
                gameWatcher.markPlaying(false)
            }
        }
    }

    private func startContinuousPlay() {
        inferenceResult = nil
        gameWatcher.resetAll()
        gameWatcher.markSessionStart()
        continuousPlay = true

        continuousTask = Task { [network] in
            guard let network else { return }

            while !Task.isCancelled {
                gameWatcher.resetCurrentGame()
                gameWatcher.markPlaying(true)

                let machine = ChessMachine()
                machine.delegate = gameWatcher
                let white = MPSChessPlayer(name: "White", network: network)
                let black = MPSChessPlayer(name: "Black", network: network)
                do {
                    let task = try machine.beginNewGame(white: white, black: black)
                    _ = await task.value
                } catch {
                    gameWatcher.markPlaying(false)
                    break
                }

                do {
                    try await Task.sleep(for: .milliseconds(1))
                } catch {
                    break
                }
            }

            await MainActor.run { continuousPlay = false }
        }
    }

    private func stopContinuousPlay() {
        continuousTask?.cancel()
        continuousTask = nil
    }

    // MARK: - Background Work

    nonisolated private static func performBuild() -> Result<ChessMPSNetwork, Error> {
        Result { try ChessMPSNetwork(.randomWeights) }
    }

    nonisolated private static func performInference(with runner: ChessRunner) -> EvaluationResult {
        var lines: [String] = []
        var topMoves: [MoveVisualization] = []
        let board = BoardEncoder.encodeStartingPosition()

        do {
            let inference = try runner.evaluate(board: board)
            topMoves = inference.topMoves

            lines.append(String(format: "Forward pass: %.2f ms", inference.inferenceTimeMs))
            lines.append("")
            lines.append("Value Head")
            lines.append(String(format: "  Output: %+.6f", inference.value))
            lines.append(String(format: "  %.1f%% win / %.1f%% loss",
                                (inference.value + 1) / 2 * 100, (1 - inference.value) / 2 * 100))
            lines.append("")
            lines.append("Policy Head (Top 4)")
            for (rank, move) in inference.topMoves.enumerated() {
                let fromName = BoardEncoder.squareName(move.fromRow * 8 + move.fromCol)
                let toName = BoardEncoder.squareName(move.toRow * 8 + move.toCol)
                let rankCol = String(rank + 1).padding(toLength: 4, withPad: " ", startingAt: 0)
                let moveCol = "\(fromName)-\(toName)".padding(toLength: 8, withPad: " ", startingAt: 0)
                lines.append("  \(rankCol)\(moveCol)\(String(format: "%.4f%%", move.probability * 100))")
            }
            lines.append("")
            lines.append("Policy Stats")
            lines.append(String(format: "  Sum: %.6f", inference.policy.reduce(0, +)))
            let nonZeroCount = inference.policy.filter { $0 > 1e-10 }.count
            lines.append(String(format: "  Non-negligible: %d / 4096", nonZeroCount))
            if let maxProb = inference.policy.max(), let minProb = inference.policy.min() {
                lines.append(String(format: "  Min: %.6f", minProb))
                lines.append(String(format: "  Max: %.6f", maxProb))
            }
        } catch {
            lines.append("Error: \(error.localizedDescription)")
        }

        return EvaluationResult(topMoves: topMoves, textOutput: lines.joined(separator: "\n"), inputTensor: board)
    }
}

#Preview {
    ContentView()
}
