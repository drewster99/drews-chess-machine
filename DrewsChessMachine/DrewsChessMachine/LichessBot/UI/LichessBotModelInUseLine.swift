import SwiftUI

/// The Online panel's model line (issue #20): what the bot plays now, the
/// file's modified time and the last switch, then any older generations
/// games in progress still play (mid-game switching off). Shown only while
/// a generation exists, i.e. while the bot is online. Times are refreshed
/// every 30 s so their ages stay current.
struct LichessBotModelInUseLine: View {
    let controller: LichessBotController

    var body: some View {
        TimelineView(.periodic(from: .now, by: 30)) { context in
            LichessBotModelInUseLineText(modelInUse: controller.modelInUse,
                                         gamesInProgress: controller.gamesInProgress,
                                         now: context.date)
        }
        .shown(controller.modelInUse != nil)
    }
}

/// The model line's text at `now`.
struct LichessBotModelInUseLineText: View {
    let modelInUse: LichessBotModelSwitch?
    let gamesInProgress: [LichessBotLiveGame]
    let now: Date

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(modelInUse.map { LichessBotModelInUseText.onlineLine($0, now: now) } ?? "")
                .font(.system(.callout, design: .monospaced))
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            Text(olderLines.joined(separator: "\n"))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .shown(!olderLines.isEmpty)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var olderLines: [String] {
        guard let modelInUse else { return [] }
        return LichessBotModelInUseText.olderGenerationLines(games: gamesInProgress, current: modelInUse.to)
    }
}
