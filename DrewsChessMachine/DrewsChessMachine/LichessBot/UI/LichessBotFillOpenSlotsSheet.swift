import SwiftUI

/// Fill Open Slots' confirmation: how many matchmaking challenges to send
/// (one per game it tries to start; the slots open now by default, at most
/// those) and whether to override matchmaking's time limits for this fill.
struct LichessBotFillOpenSlotsSheet: View {
    let controller: LichessBotController
    @Binding var isPresented: Bool

    /// The count as typed, parsed on every keystroke: a value-formatted
    /// field updates only on commit, so Start could send a stale count.
    /// Set from the open slots when the sheet appears (not from an
    /// initializer argument, which `@State` would keep stale).
    @State private var challengeCountText = ""
    @State private var overridesTimeLimits = false

    var body: some View {
        let range = LichessBotFillOpenSlotsRequest.challengeCountRange(openSlots: controller.openSlotsToFill)
        let challengeCount = Self.parsedCount(challengeCountText)
        VStack(alignment: .leading, spacing: 14) {
            Text("Fill Open Slots")
                .font(.title2.weight(.semibold))
            LichessBotFillOpenSlotsSlotsLine(controller: controller)
            LichessBotFillOpenSlotsCountRow(challengeCountText: $challengeCountText, range: range)
            LichessBotFillOpenSlotsOverrideToggle(overridesTimeLimits: $overridesTimeLimits)
            LichessBotFillOpenSlotsButtons(
                canStart: challengeCount.map { range?.contains($0) == true } == true && controller.connection == .online,
                onCancel: { isPresented = false },
                onStart: start
            )
        }
        .padding(20)
        .frame(width: 460)
        .onAppear {
            challengeCountText = "\(max(1, controller.openSlotsToFill))"
        }
    }

    /// Digits only (surrounding spaces ignored); nil for anything else.
    static func parsedCount(_ text: String) -> Int? {
        Int(text.trimmingCharacters(in: .whitespaces))
    }

    private func start() {
        guard let challengeCount = Self.parsedCount(challengeCountText) else { return }
        controller.startFillingOpenSlots(LichessBotFillOpenSlotsRequest(challengeCount: challengeCount, overridesTimeLimits: overridesTimeLimits))
        isPresented = false
    }
}

/// The slots open now, read live: games and pending challenges change while
/// the sheet is open.
struct LichessBotFillOpenSlotsSlotsLine: View {
    let controller: LichessBotController

    var body: some View {
        let open = controller.openSlotsToFill
        let challenge = controller.settings.challenge
        Text("\(open) slot(s) open now: \(challenge.maxConcurrentGames) concurrent games, \(challenge.gamesReservedForHumans) reserved for humans, \(controller.committedGameSlots) taken by games and pending challenges.")
            .font(.callout)
            .foregroundStyle(open > 0 ? Color.secondary : Color.orange)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// The editable challenge count, with its bounds.
struct LichessBotFillOpenSlotsCountRow: View {
    @Binding var challengeCountText: String
    let range: ClosedRange<Int>?

    var body: some View {
        let bounds = range ?? 0...0
        let count = LichessBotFillOpenSlotsSheet.parsedCount(challengeCountText)
        let isValid = count.map { range?.contains($0) == true } == true
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("Challenges to send")
                TextField("Challenges", text: $challengeCountText)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 60)
                    .multilineTextAlignment(.trailing)
                Stepper("Challenges to send") {
                    step(by: 1)
                } onDecrement: {
                    step(by: -1)
                }
                .labelsHidden()
                .disabled(range == nil)
            }
            Text(range == nil ? "No slot is open." : "From \(bounds.lowerBound) to \(bounds.upperBound). Each challenge is one game it tries to start; a declined challenge frees its slot but isn't replaced.")
                .font(.callout)
                .foregroundStyle(range != nil && !isValid ? Color.red : Color.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The stepper moves from the typed count (or the low bound when it
    /// isn't a number), kept inside the range.
    private func step(by delta: Int) {
        guard let range else { return }
        let current = LichessBotFillOpenSlotsSheet.parsedCount(challengeCountText) ?? range.lowerBound
        challengeCountText = "\(min(max(current + delta, range.lowerBound), range.upperBound))"
    }
}

/// "Override time limits": for this fill only.
struct LichessBotFillOpenSlotsOverrideToggle: View {
    @Binding var overridesTimeLimits: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle("Override time limits", isOn: $overridesTimeLimits)
            Text("Ignores matchmaking's hourly challenge cap and its off-peak schedule for this fill. The \(Int(LichessBotMatchmaking.minimumSendSpacing))-second spacing between sends, Lichess's own limits and the circuit breakers still apply.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct LichessBotFillOpenSlotsButtons: View {
    let canStart: Bool
    let onCancel: () -> Void
    let onStart: () -> Void

    var body: some View {
        HStack {
            Spacer()
            Button("Cancel", role: .cancel, action: onCancel)
                .keyboardShortcut(.cancelAction)
            Button("Start", action: onStart)
                .keyboardShortcut(.defaultAction)
                .disabled(!canStart)
        }
    }
}
