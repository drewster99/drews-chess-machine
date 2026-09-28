import SwiftUI

/// One Lichess game, large (plan §14.3a): the board at the browsed position
/// between both players' clocks, the value head's W/D/L, the move list with
/// browse controls, DCM's reasoning for the selected move, and the game's
/// protocol transcript and chat. Browsing only changes what is shown.
struct LichessBotGameDetailView: View {
    let game: LichessBotLiveGame
    let headToHead: (wins: Int, draws: Int, losses: Int)?
    /// Opens the game in its own window; nil inside that window.
    let onPopOut: (() -> Void)?
    let claimsKeyboardShortcuts: Bool

    @State private var cursor = GameBrowseCursor()
    @State private var panel: Panel = .transcript

    enum Panel: String, CaseIterable, Identifiable {
        case transcript = "Transcript"
        case chat = "Chat"
        var id: String { rawValue }
    }

    var body: some View {
        ZStack {
            Text("Waiting for the game's details from Lichess…")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .shown(game.ourColor == nil)
            // Rendered once DCM's color is known: the board's orientation
            // and which clock is whose depend on it.
            ForEach(game.ourColor.map { [$0] } ?? [], id: \.self) { ourColor in
                LichessBotGameDetailContent(
                    game: game,
                    ourColor: ourColor,
                    headToHead: headToHead,
                    onPopOut: onPopOut,
                    claimsKeyboardShortcuts: claimsKeyboardShortcuts,
                    cursor: $cursor,
                    panel: $panel
                )
            }
        }
        .onChange(of: game.id) {
            cursor.goLive()
        }
    }
}

/// The game view's content once DCM's color is known.
struct LichessBotGameDetailContent: View {
    let game: LichessBotLiveGame
    let ourColor: PieceColor
    let headToHead: (wins: Int, draws: Int, losses: Int)?
    let onPopOut: (() -> Void)?
    let claimsKeyboardShortcuts: Bool
    @Binding var cursor: GameBrowseCursor
    @Binding var panel: LichessBotGameDetailView.Panel

    var body: some View {
        let totalPlies = game.plies.count
        let displayed = cursor.displayedPlyCount(totalPlies: totalPlies)
        let clocksRun = !game.isFinished
        VStack(alignment: .leading, spacing: 8) {
            LichessBotGameHeaderView(game: game, onPopOut: onPopOut)
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    LichessBotPlayerLineView(
                        player: game.opponent,
                        isOurs: false,
                        headToHead: headToHead,
                        clockMilliseconds: ourColor == .white ? game.blackClockMilliseconds : game.whiteClockMilliseconds,
                        clockReceivedAt: game.clocksReceivedAt,
                        clockRunning: clocksRun && game.sideToMove != ourColor
                    )
                    LichessBotBoardView(game: game, plyCount: displayed)
                        .frame(minWidth: 280, minHeight: 280)
                    LichessBotPlayerLineView(
                        player: ourColor == .white ? game.white : game.black,
                        isOurs: true,
                        headToHead: nil,
                        clockMilliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                        clockReceivedAt: game.clocksReceivedAt,
                        clockRunning: clocksRun && game.sideToMove == ourColor
                    )
                    GameBrowseControlsView(cursor: $cursor, totalPlies: totalPlies, claimsKeyboardShortcuts: claimsKeyboardShortcuts)
                }
                .frame(minWidth: 300, maxWidth: 560)
                VStack(alignment: .leading, spacing: 6) {
                    LichessBotDecisionView(
                        decision: decision(atPlyCount: displayed) ?? (cursor.isLive ? game.latestDecision : nil),
                        plyLabel: cursor.isLive ? "latest" : "ply \(displayed)"
                    )
                    BrowsableMoveListView(
                        sanMoves: game.plies.map(\.san),
                        displayedPlyCount: displayed,
                        isLive: cursor.isLive,
                        onSelectPlyCount: { cursor.select(plyCount: $0, totalPlies: totalPlies) }
                    )
                    .frame(minHeight: 120)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.06)))
                    Picker("", selection: $panel) {
                        ForEach(LichessBotGameDetailView.Panel.allCases) { panel in
                            Text(panel.rawValue).tag(panel)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    ZStack {
                        LichessBotTranscriptView(entries: game.transcript)
                            .shown(panel == .transcript)
                        LichessBotChatView(messages: game.chat)
                            .shown(panel == .chat)
                    }
                    .frame(minHeight: 160, maxHeight: .infinity)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.06)))
                }
                .frame(minWidth: 320)
            }
        }
        .padding(10)
    }

    /// DCM's decision for the move that produced the position after
    /// `plyCount` plies, if DCM made it.
    private func decision(atPlyCount plyCount: Int) -> LichessBotMoveDecision? {
        plyCount > 0 ? game.decisions[plyCount - 1] : nil
    }
}

/// The game's title line: speed, time control, rated/casual, status, and
/// the actions for the game.
struct LichessBotGameHeaderView: View {
    let game: LichessBotLiveGame
    let onPopOut: (() -> Void)?

    var body: some View {
        HStack(spacing: 10) {
            Text(game.id)
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
            Text(timeControlText)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(game.rated == true ? "Rated" : "Casual")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(statusText)
                .font(.callout.weight(.semibold))
                .foregroundStyle(game.isFinished ? Color.primary : Color.green)
            Text(game.opponentOffersDraw ? "draw offered" : (game.opponentProposesTakeback ? "takeback proposed" : ""))
                .font(.callout)
                .foregroundStyle(.orange)
                .shown(game.opponentOffersDraw || game.opponentProposesTakeback)
            Text("\(game.anomalies.count) anomalies")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(!game.anomalies.isEmpty)
                .help(game.anomalies.joined(separator: "\n"))
            Spacer()
            Button {
                LichessBotLinks.openGame(game.id)
            } label: {
                Label("lichess.org", systemImage: "safari")
            }
            .font(.callout)
            Button {
                onPopOut?()
            } label: {
                Label("Pop Out", systemImage: "macwindow.on.rectangle")
            }
            .font(.callout)
            .shown(onPopOut != nil)
        }
    }

    private var timeControlText: String {
        guard let initial = game.clockInitialMilliseconds, let increment = game.clockIncrementMilliseconds else {
            return game.speed ?? ""
        }
        let minutes = Double(initial) / 60_000
        let minutesText = minutes == minutes.rounded() ? "\(Int(minutes))" : String(format: "%.1f", minutes)
        return "\(minutesText)+\(increment / 1000) \(game.speed ?? "")"
    }

    private var statusText: String {
        guard game.isFinished else { return "Playing" }
        let result: String
        switch game.winner {
        case "white": result = "1-0"
        case "black": result = "0-1"
        default: result = game.status == "aborted" ? "aborted" : "½-½"
        }
        return "\(result) · \(game.status)"
    }
}

/// DCM's reasoning for one of its moves: the value head's W/D/L, the
/// sampling probability, and the policy's top moves.
struct LichessBotDecisionView: View {
    let decision: LichessBotMoveDecision?
    let plyLabel: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("DCM (\(plyLabel))")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(summary)
                    .font(.system(.caption, design: .monospaced))
            }
            WLDBar(wins: Double(decision?.win ?? 0), draws: Double(decision?.draw ?? 0), losses: Double(decision?.loss ?? 0))
                .frame(height: 8)
            Text(topMovesText)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }

    private var summary: String {
        guard let decision else { return "no DCM move here" }
        return String(format: "W %.2f  D %.2f  L %.2f  E %.2f  p %.2f  τ %.2f", decision.win, decision.draw, decision.loss, decision.expectedScore, decision.chosenProbability, decision.temperature)
    }

    private var topMovesText: String {
        guard let decision else { return "" }
        return "top: " + decision.topMoves.map { "\($0.uci) " + String(format: "%.2f", $0.probability) }.joined(separator: "  ")
    }
}

/// The game's chat, both rooms.
struct LichessBotChatView: View {
    let messages: [LichessBotLiveGame.ChatMessage]

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 3) {
                ForEach(messages) { message in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(message.room == "spectator" ? "spec" : "player")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 44, alignment: .leading)
                        Text(message.username)
                            .font(.caption.weight(.semibold))
                        Text(message.text)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(6)
        }
    }
}
