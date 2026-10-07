import SwiftUI

/// The later panes (§11 L1–L6), each shown only when selected.
struct LichessBotLaterPanes: View {
    let later: LichessBotLaterBreakdowns
    let pane: LichessBotRecordPane

    var body: some View {
        ZStack(alignment: .topLeading) {
            LichessBotMoveChoicePane(moveChoice: later.moveChoice)
                .shown(pane == .moveChoice)
            LichessBotClockPane(rows: later.clock)
                .shown(pane == .clock)
            LichessBotGameLengthPane(gameLength: later.gameLength)
                .shown(pane == .gameLength)
            LichessBotOpeningsPane(openings: later.openings)
                .shown(pane == .openings)
            LichessBotOpponentsPane(opponents: later.opponents)
                .shown(pane == .opponents)
            LichessBotBotHealthPane(health: later.health)
                .shown(pane == .botHealth)
            LichessBotOriginsPane(origins: later.origins)
                .shown(pane == .origins)
        }
    }
}
