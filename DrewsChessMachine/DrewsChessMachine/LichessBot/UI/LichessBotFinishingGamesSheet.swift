import SwiftUI

/// What the operator chose for one game in the finishing sheet.
enum LichessBotFinishingChoice: String, CaseIterable, Identifiable {
    case playOn
    case resign

    var id: String { rawValue }

    var title: String {
        switch self {
        case .playOn: return "Play on"
        case .resign: return "Resign"
        }
    }
}

/// Shown when going offline or quitting with games in progress (plan §13).
/// The bot is draining: no new games start, and it goes offline (or the app
/// quits) by itself when the last game ends. Each game is listed with what
/// it takes to decide on it, and the operator chooses per game whether the
/// bot plays it out or resigns it; Resign Chosen applies the choices.
/// Abort resigns every game; Quit now abandons them.
struct LichessBotFinishingGamesSheet: View {
    let controller: LichessBotController
    let purpose: LichessBotController.FinishingPurpose
    @State private var confirmingAbandon = false
    /// The operator's choice per game id. Every game listed starts at Play
    /// on, which is what draining does with it anyway.
    @State private var choices: [String: LichessBotFinishingChoice] = [:]

    var body: some View {
        let games = controller.gamesInProgress
        let chosenForResignation = games.map(\.id).filter { choices[$0] == .resign }
        VStack(alignment: .leading, spacing: 12) {
            Text(purpose == .quit ? "Finishing games before quitting" : "Finishing games before going offline")
                .font(.title2.weight(.semibold))
            Text("No new games are being accepted. Choose for each game whether DCM plays it out or resigns it. \(purpose == .quit ? "DrewsChessMachine quits" : "The bot goes offline") by itself when the last game ends.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(games) { game in
                        LichessBotFinishingGameRow(game: game, choice: $choices[game.id])
                    }
                }
            }
            .frame(minHeight: 120, maxHeight: 460)
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
                .confirmationDialog(
                    "Abandon \(games.count) game(s)?",
                    isPresented: $confirmingAbandon,
                    actions: {
                        Button("Abandon Games", role: .destructive) {
                            controller.abandonAndStop()
                        }
                    },
                    message: {
                        Text("The opponents can claim the games once the abandonment timer runs out. The games are reconciled at the next launch.")
                    }
                )
                Button("Abort (Resign All)") {
                    Task { await controller.resignAll() }
                }
                Button("Resign Chosen (\(chosenForResignation.count))") {
                    Task { await controller.resign(gameIDs: chosenForResignation) }
                }
                .disabled(chosenForResignation.isEmpty)
                .help("Resign the games set to Resign; the others play on")
            }
        }
        .padding(20)
        .frame(width: 720)
        .onAppear {
            addPlayOnChoices(for: games)
        }
        .onChange(of: games.map(\.id)) {
            Task { @MainActor in
                addPlayOnChoices(for: controller.gamesInProgress)
            }
        }
    }

    private func addPlayOnChoices(for games: [LichessBotLiveGame]) {
        for game in games where choices[game.id] == nil {
            choices[game.id] = .playOn
        }
    }
}

/// One game in the finishing sheet: its details once DCM's color is known.
struct LichessBotFinishingGameRow: View {
    let game: LichessBotLiveGame
    @Binding var choice: LichessBotFinishingChoice?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(game.ourColor.map { [$0] } ?? [], id: \.self) { ourColor in
                LichessBotFinishingGameDetails(game: game, ourColor: ourColor, choice: $choice)
            }
            Text("\(game.id): waiting for game details")
                .font(.callout)
                .foregroundStyle(.secondary)
                .shown(game.ourColor == nil)
        }
    }
}
