import SwiftUI

// MARK: - Result Types

struct EvaluationResult: Sendable {
    let topMoves: [MoveVisualization]
    let textOutput: String
    let inputTensor: [Float]
}

// MARK: - Game Watcher

@Observable
final class GameWatcher: ChessMachineDelegate, @unchecked Sendable {
    // Current game
    var state: GameState = .starting
    var result: GameResult?
    var moveCount = 0
    var isPlaying = false
    var lastGameStats: GameStats?

    // Aggregate stats
    var totalGames = 0
    var totalMoves = 0
    var totalGameTimeMs: Double = 0
    var totalWhiteThinkMs: Double = 0
    var totalBlackThinkMs: Double = 0
    var sessionStartTime: CFAbsoluteTime?

    // Result breakdown
    var whiteCheckmates = 0
    var blackCheckmates = 0
    var stalemates = 0
    var fiftyMoveDraws = 0
    var insufficientMaterialDraws = 0

    // Computed rates
    var gamesPerHour: Double {
        guard let start = sessionStartTime else { return 0 }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        guard elapsed > 0, totalGames > 0 else { return 0 }
        return Double(totalGames) / (elapsed / 3600)
    }

    var movesPerHour: Double {
        guard let start = sessionStartTime else { return 0 }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        guard elapsed > 0, totalMoves > 0 else { return 0 }
        return Double(totalMoves) / (elapsed / 3600)
    }

    var avgMoveTimeMs: Double {
        totalMoves > 0 ? (totalWhiteThinkMs + totalBlackThinkMs) / Double(totalMoves) : 0
    }

    var avgGameTimeMs: Double {
        totalGames > 0 ? totalGameTimeMs / Double(totalGames) : 0
    }

    /// Fixed-layout stats text. Every section is always present — values show "--"
    /// when no data exists. Structure never changes so layout stays stable.
    var statsText: String {
        let dash = "--"

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

        // Last game
        let lg: (moves: String, time: String, avg: String, wAvg: String, bAvg: String)
        if let s = lastGameStats {
            lg = (
                String(format: "%d (%dW + %dB)", s.totalMoves, s.whiteMoves, s.blackMoves),
                String(format: "%.1f ms", s.totalGameTimeMs),
                String(format: "%.2f ms", s.avgMoveTimeMs),
                String(format: "%.2f ms", s.avgWhiteMoveTimeMs),
                String(format: "%.2f ms", s.avgBlackMoveTimeMs)
            )
        } else {
            lg = (dash, dash, dash, dash, dash)
        }

        // Session
        let sGames = totalGames > 0 ? "\(totalGames)" : dash
        let sMoves = totalMoves > 0 ? "\(totalMoves)" : dash
        let sTime = totalGames > 0 ? String(format: "%.0f ms", totalGameTimeMs) : dash
        let sAvgMove = totalMoves > 0 ? String(format: "%.2f ms", avgMoveTimeMs) : dash
        let sAvgGame = totalGames > 0 ? String(format: "%.1f ms", avgGameTimeMs) : dash
        let sMovesHr = totalMoves > 0 ? String(format: "%.0f", movesPerHour) : dash
        let sGamesHr = totalGames > 0 ? String(format: "%.0f", gamesPerHour) : dash

        return """
            Status: \(status)

            Last Game
              Moves:     \(lg.moves)
              Time:      \(lg.time)
              Avg move:  \(lg.avg)
              White avg: \(lg.wAvg)
              Black avg: \(lg.bAvg)

            Session
              Games:     \(sGames)
              Moves:     \(sMoves)
              Time:      \(sTime)
              Avg move:  \(sAvgMove)
              Avg game:  \(sAvgGame)
              Moves/hr:  \(sMovesHr)
              Games/hr:  \(sGamesHr)

            Results
              Checkmate:      \(whiteCheckmates + blackCheckmates)\(pct(whiteCheckmates + blackCheckmates))
                White wins:   \(whiteCheckmates)\(pct(whiteCheckmates))
                Black wins:   \(blackCheckmates)\(pct(blackCheckmates))
              Stalemate:      \(stalemates)\(pct(stalemates))
              50-move draw:   \(fiftyMoveDraws)\(pct(fiftyMoveDraws))
              Insufficient:   \(insufficientMaterialDraws)\(pct(insufficientMaterialDraws))
            """
    }

    private func pct(_ count: Int) -> String {
        guard totalGames > 0 else { return "" }
        return String(format: "  (%.1f%%)", Double(count) / Double(totalGames) * 100)
    }

    func resetCurrentGame() {
        state = .starting
        result = nil
        moveCount = 0
        // Keep lastGameStats — show previous game until the next one ends
    }

    func resetAll() {
        resetCurrentGame()
        lastGameStats = nil
        totalGames = 0
        totalMoves = 0
        totalGameTimeMs = 0
        totalWhiteThinkMs = 0
        totalBlackThinkMs = 0
        sessionStartTime = nil
        whiteCheckmates = 0
        blackCheckmates = 0
        stalemates = 0
        fiftyMoveDraws = 0
        insufficientMaterialDraws = 0
    }

    // MARK: - Delegate

    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState) {
        Task { @MainActor in
            self.state = newState
            self.moveCount += 1
        }
    }

    func chessMachine(_ machine: ChessMachine, gameEndedWith result: GameResult, finalState: GameState, stats: GameStats) {
        Task { @MainActor in
            self.result = result
            self.state = finalState
            self.lastGameStats = stats
            self.isPlaying = false

            self.totalGames += 1
            self.totalMoves += stats.totalMoves
            self.totalGameTimeMs += stats.totalGameTimeMs
            self.totalWhiteThinkMs += stats.whiteThinkingTimeMs
            self.totalBlackThinkMs += stats.blackThinkingTimeMs

            switch result {
            case .checkmate(let winner):
                if winner == .white { self.whiteCheckmates += 1 }
                else { self.blackCheckmates += 1 }
            case .stalemate:
                self.stalemates += 1
            case .drawByFiftyMoveRule:
                self.fiftyMoveDraws += 1
            case .drawByInsufficientMaterial:
                self.insufficientMaterialDraws += 1
            }
        }
    }

    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error) {
        Task { @MainActor in
            self.isPlaying = false
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    // Network
    @State private var network: ChessMPSNetwork?
    @State private var runner = ChessRunner()
    @State private var networkStatus = ""
    @State private var isBuilding = false

    // Inference
    @State private var inferenceResult: EvaluationResult?
    @State private var isEvaluating = false
    @State private var selectedOverlay = 0

    // Game
    @State private var gameWatcher = GameWatcher()
    @State private var continuousPlay = false
    @State private var continuousTask: Task<Void, Never>?

    private var networkReady: Bool { network != nil }
    private var isBusy: Bool { isBuilding || isEvaluating || gameWatcher.isPlaying || continuousPlay }
    private var isGameMode: Bool { gameWatcher.isPlaying || gameWatcher.totalGames > 0 }

    private var displayedPieces: [[Piece?]] {
        if isGameMode { return gameWatcher.state.board }
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

            Text("Forward pass through a ~2.9M parameter convolutional network using MPSGraph on the GPU. Weights are randomly initialized (He initialization) — no training has occurred.")
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
                        Button(action: { navigateOverlay(-1) }) {
                            Image(systemName: "chevron.left").font(.title3).frame(width: 24)
                        }
                        .buttonStyle(.plain)
                        .disabled(leftDisabled)
                        .opacity(leftDisabled ? 0.2 : 0.6)

                        ChessBoardView(pieces: displayedPieces, overlay: currentOverlay)

                        let rightDisabled = inferenceResult == nil || selectedOverlay == 18 || isGameMode
                        Button(action: { navigateOverlay(1) }) {
                            Image(systemName: "chevron.right").font(.title3).frame(width: 24)
                        }
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
                            Text(gameWatcher.statsText)
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
    }

    private var busyLabel: String {
        if isBuilding { return "Building network..." }
        if gameWatcher.isPlaying { return "Game \(gameWatcher.totalGames + 1), move \(gameWatcher.moveCount)..." }
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
            let (net, status) = await Task.detached(priority: .userInitiated) {
                Self.performBuild()
            }.value
            network = net
            if net != nil {
                runner = ChessRunner()
                do { try runner.buildNetwork() } catch {}
            }
            networkStatus = status
            isBuilding = false
        }
    }

    private func runForwardPass() {
        isEvaluating = true
        gameWatcher.resetAll()

        Task {
            let evalResult = await Task.detached(priority: .userInitiated) { [runner] in
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
        gameWatcher.isPlaying = true
        if gameWatcher.sessionStartTime == nil {
            gameWatcher.sessionStartTime = CFAbsoluteTimeGetCurrent()
        }

        Task { [network] in
            guard let network else { return }
            let machine = ChessMachine()
            machine.delegate = gameWatcher
            let white = MPSChessPlayer(name: "White", network: network)
            let black = MPSChessPlayer(name: "Black", network: network)
            _ = await machine.beginNewGame(white: white, black: black).value
        }
    }

    private func startContinuousPlay() {
        inferenceResult = nil
        gameWatcher.resetAll()
        gameWatcher.sessionStartTime = CFAbsoluteTimeGetCurrent()
        continuousPlay = true

        continuousTask = Task { [network] in
            guard let network else { return }

            while !Task.isCancelled {
                gameWatcher.resetCurrentGame()
                await MainActor.run { gameWatcher.isPlaying = true }

                let machine = ChessMachine()
                machine.delegate = gameWatcher
                let white = MPSChessPlayer(name: "White", network: network)
                let black = MPSChessPlayer(name: "Black", network: network)
                _ = await machine.beginNewGame(white: white, black: black).value

                do { try await Task.sleep(for: .milliseconds(1)) }
                catch { break }
            }

            await MainActor.run { continuousPlay = false }
        }
    }

    private func stopContinuousPlay() {
        continuousTask?.cancel()
        continuousTask = nil
    }

    // MARK: - Background Work

    nonisolated private static func performBuild() -> (ChessMPSNetwork?, String) {
        do {
            let net = try ChessMPSNetwork(.randomWeights)
            let status = """
                Network built in \(String(format: "%.1f", net.buildTimeMs)) ms
                Parameters: ~2,917,383 (~2.9M)
                Architecture: 18x8x8 -> stem(128)
                  -> 8 res blocks -> policy(4096) + value(1)
                """
            return (net, status)
        } catch {
            return (nil, "Build failed: \(error.localizedDescription)")
        }
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
