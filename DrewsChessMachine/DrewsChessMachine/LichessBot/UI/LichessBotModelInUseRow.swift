import SwiftUI

/// Settings ▸ Model's "In use" row (issue #20): the model the running bot
/// plays now — its file (or source), model ID and step, the file's modified
/// time and since when it plays — which can differ from the settings above
/// it, since those apply to the next game. Offline it says nothing plays.
/// Times are refreshed every 30 s so their ages stay current.
struct LichessBotModelInUseRow: View {
    let controller: LichessBotController

    var body: some View {
        LabeledContent("In use") {
            TimelineView(.periodic(from: .now, by: 30)) { context in
                LichessBotModelInUseDetails(modelInUse: controller.modelInUse, now: context.date)
            }
        }
    }
}

/// The In use row's text at `now`.
struct LichessBotModelInUseDetails: View {
    let modelInUse: LichessBotModelSwitch?
    let now: Date

    var body: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(modelInUse.map { LichessBotModelInUseText.name($0.to) } ?? LichessBotModelInUseText.notOnline)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            Text(detailText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .shown(modelInUse != nil)
        }
    }

    private var detailText: String {
        guard let modelInUse else { return "" }
        var lines = [LichessBotModelInUseText.identity(modelInUse.to)]
        if let modified = LichessBotModelInUseText.fileModified(modelInUse.to, now: now) {
            lines.append(modified)
        }
        lines.append(LichessBotModelInUseText.switchText(modelInUse, now: now))
        return lines.joined(separator: "\n")
    }
}
