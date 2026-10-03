import SwiftUI

/// One game in progress in the finishing sheet, with what the operator needs
/// to decide whether to resign it or let the bot play it out: who the
/// opponent is, what the game is, where it stands (move, clocks, material,
/// the network's latest estimate) and how it got there (the last moves).
struct LichessBotFinishingGameDetails: View {
    let game: LichessBotLiveGame
    let ourColor: PieceColor
    @Binding var choice: LichessBotFinishingChoice?

    /// Plies of recent moves shown.
    static let recentPlyCount = 8

    var body: some View {
        let opponent = game.opponent
        let material = MaterialCount(game.state(afterPlies: game.plies.count))
        let materialLead = ourColor == .white ? material.white - material.black : material.black - material.white
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(opponent?.name ?? game.id)
                    .font(.body.weight(.semibold))
                    .lineLimit(1)
                Text(Self.ratingText(opponent?.rating))
                    .font(.system(.callout, design: .monospaced))
                Text(opponent?.title == "BOT" ? "bot" : "human")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Text(Self.gameKindText(game))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Text("DCM plays \(ourColor == .white ? "white" : "black")")
                    .font(.callout)
            }
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("move \(Self.padded(game.plies.count / 2 + 1, width: 3))")
                    .font(.system(.callout, design: .monospaced))
                Text(game.sideToMove == ourColor ? "DCM to move" : "opponent to move")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(width: 120, alignment: .leading)
                Text("DCM")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                LichessBotClockView(
                    milliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                    receivedAt: game.clocksReceivedAt,
                    isRunning: game.sideToMove == ourColor,
                    isOurs: true
                )
                Text("opponent")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                LichessBotClockView(
                    milliseconds: ourColor == .white ? game.blackClockMilliseconds : game.whiteClockMilliseconds,
                    receivedAt: game.clocksReceivedAt,
                    isRunning: game.sideToMove != ourColor,
                    isOurs: false
                )
                Spacer(minLength: 8)
                Text("material \(Self.signedPadded(materialLead))")
                    .font(.system(.callout, design: .monospaced))
                    .help("DCM's material minus the opponent's, in pawns")
            }
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(Self.estimateText(game.latestDecision))
                    .font(.system(.callout, design: .monospaced))
                    .help("The network's win / draw / loss estimate for DCM at its latest move")
                Spacer(minLength: 8)
                Picker("Decision", selection: $choice) {
                    ForEach(LichessBotFinishingChoice.allCases) { option in
                        Text(option.title).tag(Optional(option))
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 180)
                .accessibilityLabel("Decision for the game against \(opponent?.name ?? game.id)")
            }
            Text(Self.recentMovesText(game))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.head)
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.08)))
    }

    static func ratingText(_ rating: Int?) -> String {
        guard let rating else { return "rating not shown" }
        return padded(rating, width: 4)
    }

    static func gameKindText(_ game: LichessBotLiveGame) -> String {
        let ratedText: String
        switch game.rated {
        case true?: ratedText = "rated"
        case false?: ratedText = "casual"
        case nil: ratedText = "rated or casual not received"
        }
        guard let initial = game.clockInitialMilliseconds, let increment = game.clockIncrementMilliseconds else {
            return "\(ratedText) · no clock received"
        }
        let minutes = Double(initial) / 60_000
        let minutesText = minutes == minutes.rounded() ? String(Int(minutes)) : String(format: "%.1f", minutes)
        let speedText = game.speed.map { " \($0)" } ?? ""
        return "\(ratedText) · \(minutesText)+\(increment / 1000)\(speedText)"
    }

    static func estimateText(_ decision: LichessBotMoveDecision?) -> String {
        guard let decision else { return "no estimate yet (DCM has not moved)" }
        return "W \(percent(decision.win))  D \(percent(decision.draw))  L \(percent(decision.loss))"
    }

    static func recentMovesText(_ game: LichessBotLiveGame) -> String {
        let plies = game.plies.suffix(recentPlyCount)
        guard !plies.isEmpty else { return "no moves yet" }
        var parts: [String] = []
        for ply in plies {
            if ply.id.isMultiple(of: 2) {
                parts.append("\(ply.id / 2 + 1).\(ply.san)")
            } else if ply.id == plies.first?.id {
                parts.append("\(ply.id / 2 + 1)…\(ply.san)")
            } else {
                parts.append(ply.san)
            }
        }
        return "last moves: " + parts.joined(separator: " ")
    }

    /// U+2007 FIGURE SPACE: as wide as a digit, so padded numbers align.
    private static let figureSpace = "\u{2007}"

    static func padded(_ value: Int, width: Int) -> String {
        let text = String(value)
        return String(repeating: figureSpace, count: max(0, width - text.count)) + text
    }

    static func signedPadded(_ value: Int) -> String {
        let text = value == 0 ? "\u{00B1}0" : String(format: "%+d", value)
        return String(repeating: figureSpace, count: max(0, 3 - text.count)) + text
    }

    static func percent(_ probability: Float) -> String {
        padded(Int((probability * 100).rounded()), width: 3) + "%"
    }
}
