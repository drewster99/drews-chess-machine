import SwiftUI

/// Shown when going offline or quitting with games in progress (plan §13):
/// the bot is draining, each game's live status is listed, and it finishes
/// on its own when the last game ends. Abort resigns them all; Quit now
/// abandons them.
struct LichessBotFinishingGamesSheet: View {
    let controller: LichessBotController
    let purpose: LichessBotController.FinishingPurpose
    @State private var confirmingAbandon = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(purpose == .quit ? "Finishing games before quitting" : "Finishing games before going offline")
                .font(.title2.weight(.semibold))
            Text("No new games are being accepted. \(purpose == .quit ? "DrewsChessMachine quits" : "The bot goes offline") by itself when the last game ends.")
                .font(.callout)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(controller.gamesInProgress) { game in
                    LichessBotFinishingGameRow(game: game)
                }
            }
            .padding(8)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.08)))
            HStack {
                Button("Cancel") {
                    controller.cancelFinishing()
                }
                .keyboardShortcut(.cancelAction)
                .help(purpose == .quit ? "Don't quit; the bot keeps draining" : "Close this; the bot keeps draining")
                Spacer()
                Button(purpose == .quit ? "Quit Now…" : "Go Offline Now…", role: .destructive) {
                    confirmingAbandon = true
                }
                .confirmationDialog("Abandon \(controller.gamesInProgress.count) game(s)?", isPresented: $confirmingAbandon) {
                    Button("Abandon Games", role: .destructive) {
                        controller.abandonAndStop()
                    }
                } message: {
                    Text("The opponents can claim the games once the abandonment timer runs out. The games are reconciled at the next launch.")
                }
                Button("Abort (Resign All)") {
                    Task { await controller.resignAll() }
                }
            }
        }
        .padding(20)
        .frame(width: 560)
    }
}

/// One game's status in the finishing sheet.
struct LichessBotFinishingGameRow: View {
    let game: LichessBotLiveGame

    var body: some View {
        HStack(spacing: 10) {
            ForEach(game.ourColor.map { [$0] } ?? [], id: \.self) { ourColor in
                LichessBotFinishingGameStatus(game: game, ourColor: ourColor)
            }
            Text("waiting for game details")
                .font(.callout)
                .foregroundStyle(.secondary)
                .shown(game.ourColor == nil)
        }
    }
}

/// A finishing game's opponent, move and clocks, once DCM's color is known.
struct LichessBotFinishingGameStatus: View {
    let game: LichessBotLiveGame
    let ourColor: PieceColor

    var body: some View {
        HStack(spacing: 10) {
            Text(game.opponent?.name ?? game.id)
                .font(.body.weight(.medium))
                .frame(width: 160, alignment: .leading)
                .lineLimit(1)
            Text("move \(game.plies.count / 2 + 1)")
                .font(.system(.callout, design: .monospaced))
                .frame(width: 80, alignment: .leading)
            Text(game.sideToMove == ourColor ? "DCM to move" : "opponent to move")
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(width: 130, alignment: .leading)
            LichessBotClockView(
                milliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                receivedAt: game.clocksReceivedAt,
                isRunning: game.sideToMove == ourColor,
                isOurs: true
            )
            LichessBotClockView(
                milliseconds: ourColor == .white ? game.blackClockMilliseconds : game.whiteClockMilliseconds,
                receivedAt: game.clocksReceivedAt,
                isRunning: game.sideToMove != ourColor,
                isOurs: false
            )
        }
    }
}
