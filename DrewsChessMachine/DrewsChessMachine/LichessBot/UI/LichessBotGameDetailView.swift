import SwiftUI

/// One Lichess game, large (plan §14.3a): the board at the browsed position
/// between both players' clocks, the value head's W/D/L, the move list with
/// browse controls, DCM's reasoning for the selected move, and the game's
/// protocol transcript and chat. Browsing only changes what is shown.
struct LichessBotGameDetailView: View {
    let controller: LichessBotController
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
        case opponent = "Opponent"
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
                    controller: controller,
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
    }
}

/// The game view's content once DCM's color is known.
struct LichessBotGameDetailContent: View {
    let controller: LichessBotController
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
        let material = MaterialCount(game.state(afterPlies: displayed))
        VStack(alignment: .leading, spacing: 8) {
            LichessBotGameHeaderView(game: game, onPopOut: onPopOut)
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    LichessBotPlayerLineView(
                        controller: controller,
                        player: game.opponent,
                        pieceColor: ourColor == .white ? .black : .white,
                        isOurs: false,
                        headToHead: headToHead,
                        clockMilliseconds: ourColor == .white ? game.blackClockMilliseconds : game.whiteClockMilliseconds,
                        clockReceivedAt: game.clocksReceivedAt,
                        clockRunning: clocksRun && game.sideToMove != ourColor,
                        material: material
                    )
                    LichessBotBoardView(game: game, plyCount: displayed)
                        .frame(minWidth: 280, minHeight: 280)
                    LichessBotPlayerLineView(
                        controller: controller,
                        player: ourColor == .white ? game.white : game.black,
                        pieceColor: ourColor,
                        isOurs: true,
                        headToHead: nil,
                        clockMilliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                        clockReceivedAt: game.clocksReceivedAt,
                        clockRunning: clocksRun && game.sideToMove == ourColor,
                        material: material
                    )
                    GameBrowseControlsView(cursor: $cursor, totalPlies: totalPlies, claimsKeyboardShortcuts: claimsKeyboardShortcuts)
                    LichessBotMovePacingControls(game: game)
                        .shown(!game.isFinished)
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
                    // Both panels keep their full size while hidden: a
                    // collapsed scroll view loses its scroll position.
                    ZStack {
                        LichessBotTranscriptView(game: game, isVisible: panel == .transcript)
                            .opacity(panel == .transcript ? 1 : 0)
                            .allowsHitTesting(panel == .transcript)
                            .accessibilityHidden(panel != .transcript)
                        LichessBotChatPanel(controller: controller, game: game)
                            .opacity(panel == .chat ? 1 : 0)
                            .allowsHitTesting(panel == .chat)
                            .accessibilityHidden(panel != .chat)
                        LichessBotOpponentPanel(controller: controller, opponent: game.opponent)
                            .padding(6)
                            .opacity(panel == .opponent ? 1 : 0)
                            .allowsHitTesting(panel == .opponent)
                            .accessibilityHidden(panel != .opponent)
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
            Button(
                action: { LichessBotLinks.openGame(game.id) },
                label: { Label("lichess.org", systemImage: "safari") }
            )
            .font(.callout)
            Button(
                action: { onPopOut?() },
                label: { Label("Pop Out", systemImage: "macwindow.on.rectangle") }
            )
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

    /// The result in words: who won, how, and what it means for DCM.
    private var statusText: String {
        guard game.isFinished else { return "Playing" }
        let how = Self.describe(status: game.status)
        guard let winner = game.winner, winner == "white" || winner == "black" else {
            switch game.status {
            case "aborted", "noStart": return "Aborted"
            case LichessBotLiveGame.leftUnfinishedStatus: return "Left unfinished"
            default:
                guard game.ourScore == 0.5 else { return "No result \(how)" }
                return "Draw \(game.status == "draw" ? Self.describe(drawRule: game.localDrawCondition) : how)"
            }
        }
        let winnerIsWhite = winner == "white"
        let winnerName = (winnerIsWhite ? game.white?.name : game.black?.name) ?? winner
        let dcmWon = (game.ourColor == .white) == winnerIsWhite
        return "\(winnerName) (\(winnerIsWhite ? "White" : "Black")) won \(how) · DCM \(dcmWon ? "won" : "lost")"
    }

    /// Lichess reports every agreed or rule draw as status "draw"; DCM's
    /// engine can say which rule held in the final position. No rule means
    /// the players agreed (or Lichess applied one DCM doesn't track).
    private static func describe(drawRule: ChessDrawCondition?) -> String {
        switch drawRule {
        case .threefoldRepetition: return "by threefold repetition"
        case .fiftyMoveRule: return "by the fifty-move rule"
        case .insufficientMaterial: return "by insufficient material"
        case .none: return "by agreement"
        }
    }

    /// Lichess's status as "by …".
    private static func describe(status: String) -> String {
        switch status {
        case "mate": return "by checkmate"
        case "resign": return "by resignation"
        case "outoftime": return "on time"
        case "timeout": return "(opponent left)"
        case "stalemate": return "by stalemate"
        case "draw": return "by agreement or rule"
        case "insufficientMaterialClaim": return "by insufficient material"
        case "cheat": return "(cheat detected)"
        case "variantEnd": return "(variant end)"
        default: return "(\(status))"
        }
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

/// The game's chat, both rooms, as bubbles (ours on the right), with the
/// operator's input bar while the game is live (plan §14.3c).
struct LichessBotChatPanel: View {
    let controller: LichessBotController
    let game: LichessBotLiveGame

    var body: some View {
        VStack(spacing: 0) {
            LichessBotChatView(game: game)
            Divider()
            LichessBotChatInputBar(controller: controller, gameID: game.id)
                .padding(6)
                .shown(!game.isFinished)
        }
    }
}

/// The chat messages. Each bubble takes at most 75% of the panel's width.
struct LichessBotChatView: View {
    let game: LichessBotLiveGame

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(game.chat) { message in
                        LichessBotChatBubble(message: message, isOurs: game.isFromUs(message))
                            .id(message.id)
                    }
                }
                .padding(6)
            }
            .onChange(of: game.chat.last?.id) {
                if let last = game.chat.last?.id {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
    }
}

/// One chat message: room, author and text, left for others and right for
/// our account.
struct LichessBotChatBubble: View {
    let message: LichessBotLiveGame.ChatMessage
    let isOurs: Bool

    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)
                .frame(maxWidth: isOurs ? .infinity : 0)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(message.at.formatted(.dateTime.hour(.twoDigits(amPM: .omitted)).minute(.twoDigits).second(.twoDigits)))
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(message.room == "spectator" ? "spectators" : "players")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(message.username)
                        .font(.caption.weight(.semibold))
                    Text(originText)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .shown(message.origin != nil)
                }
                Text(message.text)
                    .font(.callout)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isOurs ? Color.green.opacity(0.12) : Color.blue.opacity(0.10))
            )
            .containerRelativeFrame(.horizontal, alignment: isOurs ? .trailing : .leading) { length, _ in
                length * 0.75
            }
            Spacer(minLength: 0)
                .frame(maxWidth: isOurs ? 0 : .infinity)
        }
    }

    /// Why DCM's account said it, and whether Lichess echoed it back.
    private var originText: String {
        guard let origin = message.origin else { return "" }
        let kind: String
        switch origin {
        case .greeting: kind = "greeting"
        case .goodbye: kind = "goodbye"
        case .commandReply: kind = "command reply"
        case .operator: kind = "operator"
        }
        return message.echoed ? kind : "\(kind) · sent (not echoed by Lichess)"
    }
}

/// The operator's chat input: room, text with a live count against
/// Lichess's limit, a link warning, and Send (plan §14.3c).
struct LichessBotChatInputBar: View {
    let controller: LichessBotController
    let gameID: String

    @State private var room: LichessBotChatRoom = .player
    @State private var text = ""
    @State private var sending = false
    @State private var sendError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Picker("", selection: $room) {
                    Text("Players").tag(LichessBotChatRoom.player)
                    Text("Spectators").tag(LichessBotChatRoom.spectator)
                }
                .labelsHidden()
                .fixedSize()
                TextField("Message as DrewsChessMachine", text: $text)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { send() }
                Text(String(format: "%3d/%d", LichessBotOperatorChat.length(of: trimmed), LichessBotChat.maximumLength))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(LichessBotOperatorChat.length(of: trimmed) > LichessBotChat.maximumLength ? Color.red : Color.secondary)
                Button("Send") {
                    send()
                }
                .disabled(sending || LichessBotOperatorChat.problem(with: trimmed) != nil)
            }
            Text("Looks like a link: Lichess silently drops links in bot messages")
                .font(.caption)
                .foregroundStyle(.orange)
                .shown(LichessBotOperatorChat.looksLikeLink(trimmed))
            Text(sendError ?? "")
                .font(.caption)
                .foregroundStyle(.red)
                .shown(sendError != nil)
        }
    }

    private var trimmed: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func send() {
        guard !sending, LichessBotOperatorChat.problem(with: trimmed) == nil else { return }
        let message = trimmed
        let target = room
        sending = true
        sendError = nil
        Task {
            do {
                try await controller.sendOperatorChat(gameID: gameID, room: target, text: message)
                text = ""
            } catch {
                sendError = error.localizedDescription
            }
            sending = false
        }
    }
}

/// The operator's move pacing for this game (plan §14.3c): a delay before
/// each of DCM's moves after its first, or holding each move for Play move.
/// A move is played automatically once DCM's clock nears the safety floor.
struct LichessBotMovePacingControls: View {
    @Bindable var game: LichessBotLiveGame

    var body: some View {
        HStack(spacing: 12) {
            Stepper(value: $game.moveDelaySeconds, in: 0...30) {
                Text(String(format: "Delay %2d s", game.moveDelaySeconds))
                    .font(.system(.callout, design: .monospaced))
            }
            .fixedSize()
            .disabled(game.holdsMoves)
            Toggle("Hold moves", isOn: $game.holdsMoves)
            Spacer(minLength: 8)
            Text(game.heldMove.map { "Holding \($0.san)" } ?? "")
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .foregroundStyle(.orange)
                .shown(game.heldMove != nil)
            Button("Play move") {
                game.requestRelease()
            }
            .disabled(game.heldMove == nil || game.releaseRequested)
        }
        .help("Applies after DCM's first move. A held or delayed move is played automatically when DCM's clock gets low.")
    }
}

/// The opponent's card, or why there is none (Lichess's own AI has no
/// account).
struct LichessBotOpponentPanel: View {
    let controller: LichessBotController
    let opponent: LichessBotLiveGame.Player?

    var body: some View {
        ZStack(alignment: .topLeading) {
            Text(opponent == nil ? "Opponent not known yet" : "\(opponent?.name ?? "") has no Lichess account to show")
                .foregroundStyle(.secondary)
                .shown(opponent?.id == nil)
            ForEach(opponent?.id == nil ? [] : [opponent?.name].compactMap { $0 }, id: \.self) { username in
                LichessBotOpponentCard(controller: controller, username: username)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
