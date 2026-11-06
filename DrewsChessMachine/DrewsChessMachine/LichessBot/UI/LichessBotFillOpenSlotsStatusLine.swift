import SwiftUI

/// Fill Open Slots' progress on the Overview: a spinner, the count sent and
/// the next send while it runs, with Cancel; how it ended once it is done.
/// Hidden before the first fill.
struct LichessBotFillOpenSlotsStatusLine: View {
    let controller: LichessBotController

    var body: some View {
        let run = controller.fillOpenSlotsRun
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
                .shown(run.isRunning)
            Text(Self.text(for: run))
                .font(.callout.weight(run.isRunning ? .semibold : .regular))
                .foregroundStyle(run.isRunning ? Color.primary : Color.secondary)
                .lineLimit(2)
                .textSelection(.enabled)
            Button("Cancel Fill") {
                controller.cancelFillingOpenSlots()
            }
            .controlSize(.small)
            .shown(run.isRunning)
        }
        .shown(run != .idle)
    }

    static func text(for run: LichessBotFillOpenSlotsRun) -> String {
        switch run {
        case .idle:
            return ""
        case .running(let request, let sent, let startedAt, let waitingUntil):
            var parts = ["Filling open slots: \(sent) of \(request.challengeCount) sent since \(startedAt.formatted(date: .omitted, time: .standard))"]
            if let waitingUntil {
                parts.append("next send at \(waitingUntil.formatted(date: .omitted, time: .standard))")
            }
            if request.overridesTimeLimits {
                parts.append("time limits overridden")
            }
            return parts.joined(separator: " · ")
        case .finished(let request, let sent, let endedAt, let reason):
            return "Fill Open Slots done at \(endedAt.formatted(date: .omitted, time: .standard)): sent \(sent) of \(request.challengeCount); \(reason)"
        }
    }
}
