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
        case .running(let request, let progress, let startedAt, let waiting):
            var parts = ["Filling open slots since \(startedAt.formatted(date: .omitted, time: .standard)): \(LichessBotController.fillOpenSlotsTally(request, progress: progress))"]
            if let waiting {
                parts.append(waitText(waiting))
            }
            if request.overridesTimeLimits {
                parts.append("time limits overridden")
            }
            return parts.joined(separator: " · ")
        case .finished(let request, let progress, let endedAt, let reason):
            return "Fill Open Slots done at \(endedAt.formatted(date: .omitted, time: .standard)): \(LichessBotController.fillOpenSlotsTally(request, progress: progress)); \(reason)"
        }
    }

    static func waitText(_ wait: LichessBotFillOpenSlotsWait) -> String {
        switch wait {
        case .sendLimit(let until):
            return "next send at \(until.formatted(date: .omitted, time: .standard))"
        case .freeSlot:
            return "waiting for a free slot"
        case .pendingAnswers:
            return "waiting for answers to its pending challenges"
        case .retry(let until, let after):
            return "trying again at \(until.formatted(date: .omitted, time: .standard)) (\(after))"
        }
    }
}
