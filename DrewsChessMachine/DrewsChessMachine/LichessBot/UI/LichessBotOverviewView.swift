import SwiftUI

/// The bot's control room (plan §14.3): state and controls, account, the
/// model generation in use, the request gate, the pending challenge, and
/// alarms.
struct LichessBotOverviewView: View {
    let controller: LichessBotController
    @State private var showingChallengeSheet = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                LichessBotControlsCard(controller: controller, onChallenge: { showingChallengeSheet = true })
                HStack(alignment: .top, spacing: 16) {
                    LichessBotAccountCard(controller: controller)
                        .frame(maxHeight: .infinity, alignment: .top)
                    LichessBotModelCard(controller: controller)
                        .frame(maxHeight: .infinity, alignment: .top)
                    LichessBotGateCard(snapshot: controller.gateSnapshot)
                        .frame(maxHeight: .infinity, alignment: .top)
                }
                .fixedSize(horizontal: false, vertical: true)
                LichessBotAlarmsCard(controller: controller)
            }
            .padding(16)
        }
        .sheet(isPresented: $showingChallengeSheet) {
            LichessBotChallengeSheet(controller: controller, isPresented: $showingChallengeSheet)
        }
    }
}

/// State, the on/off controls, and the challenge controls.
struct LichessBotControlsCard: View {
    let controller: LichessBotController
    let onChallenge: () -> Void
    @State private var confirmingResignAll = false

    var body: some View {
        let connection = controller.connection
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Circle()
                        .fill(LichessBotStatusStyle.color(for: connection))
                        .frame(width: 12, height: 12)
                    Text(connection.label)
                        .font(.title2.weight(.semibold))
                    Text(controller.oneGameRequested ? "one game" : "")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .shown(controller.oneGameRequested)
                    Text("filing records before going offline…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .shown(controller.isFilingRecords)
                    Text(errorText)
                        .font(.callout)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .shown(!errorText.isEmpty)
                    Spacer()
                    Text("\(controller.activeGameIDs.count) game(s) in progress")
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 8) {
                    Button("Go Online") {
                        Task { await controller.goOnline() }
                    }
                    .disabled(!canGoOnline)
                    .help(hasToken ? "Accept challenges under the policy in Settings" : "Add a token in Settings ▸ Account first")
                    Button("Play One Game") {
                        Task { await controller.goOnline(oneGame: true) }
                    }
                    .disabled(!canGoOnline)
                    .help(hasToken ? "Take one game — from a challenge you accept or send — then go offline when it ends" : "Add a token in Settings ▸ Account first")
                    Button("Stop Accepting (Drain)") {
                        Task { await controller.drain() }
                    }
                    .disabled(connection != .online)
                    Button("Go Offline") {
                        Task { await controller.goOffline() }
                    }
                    .disabled(!controller.isRunning)
                    Button("Resign All…", role: .destructive) {
                        confirmingResignAll = true
                    }
                    .disabled(controller.activeGameIDs.isEmpty)
                    .confirmationDialog("Resign all \(controller.activeGameIDs.count) game(s) in progress?", isPresented: $confirmingResignAll) {
                        Button("Resign All", role: .destructive) {
                            Task { await controller.resignAll() }
                        }
                    }
                    Spacer()
                    Button("Challenge…", action: onChallenge)
                        .disabled(connection != .online || !controller.hasChallengeScope || controller.pendingChallenge != nil)
                        .help(controller.hasChallengeScope ? "Challenge an online bot or any player" : "The token lacks challenge:write")
                }
                HStack(spacing: 8) {
                    Text(pendingText)
                        .font(.callout)
                    Button("Cancel Challenge") {
                        Task { await controller.cancelChallenge() }
                    }
                    .shown(controller.pendingChallenge != nil)
                    Text(controller.lastChallengeOutcome ?? "")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .shown(controller.pendingChallenge == nil && controller.lastChallengeOutcome != nil)
                }
                .shown(controller.pendingChallenge != nil || controller.lastChallengeOutcome != nil)
            }
            .padding(6)
        }
    }

    private var canGoOnline: Bool {
        guard hasToken else { return false }
        switch controller.connection {
        case .offline, .error: return true
        case .connecting, .online, .draining: return false
        }
    }

    /// A token is stored (it may not have been checked yet, or the check
    /// may have failed for a network reason: going online checks it again).
    private var hasToken: Bool {
        switch controller.tokenState {
        case .none, .checking: return false
        case .unknown, .saved, .error: return true
        }
    }

    private var errorText: String {
        if case .error(let message) = controller.connection {
            return message
        }
        return ""
    }

    private var pendingText: String {
        guard let pending = controller.pendingChallenge else { return "" }
        return "Challenge to \(pending.username) waiting since \(pending.sentAt.formatted(date: .omitted, time: .standard))"
    }
}

/// The Lichess account the token belongs to.
struct LichessBotAccountCard: View {
    let controller: LichessBotController

    var body: some View {
        GroupBox("Account") {
            VStack(alignment: .leading, spacing: 4) {
                Text(controller.account?.username ?? controller.settings.connection.expectedAccountID)
                    .font(.headline)
                Text(accountStatus)
                    .font(.callout)
                    .foregroundStyle(controller.account?.isBot == true ? Color.green : Color.orange)
                ForEach(ratingRows, id: \.perf) { row in
                    HStack {
                        Text(row.perf)
                            .font(.callout)
                            .frame(width: 80, alignment: .leading)
                        Text(row.text)
                            .font(.system(.callout, design: .monospaced))
                    }
                }
                Text(tokenText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    private var ratingRows: [(perf: String, text: String)] {
        guard let perfs = controller.account?.perfs else { return [] }
        return ["bullet", "blitz", "rapid", "classical"].compactMap { perf in
            guard let rating = perfs[perf], let value = rating.rating else { return nil }
            let games = rating.games.map { "  \($0) games" } ?? ""
            return (perf, String(format: "%4d", value) + (rating.prov == true ? "?" : " ") + games)
        }
    }

    private var accountStatus: String {
        if let account = controller.account {
            return account.isBot ? "BOT account" : "Not a BOT account yet"
        }
        switch controller.tokenState {
        case .none: return "Add a token in Settings ▸ Account"
        case .checking: return "Checking…"
        default: return "Not checked"
        }
    }

    private var tokenText: String {
        switch controller.tokenState {
        case .unknown: return "Token not checked"
        case .none: return "No token saved"
        case .checking: return "Checking token…"
        case .saved(let info): return "Token: \(info.scopes)"
        case .error(let message): return "Token problem: \(message)"
        }
    }
}

/// The model generation new games use.
struct LichessBotModelCard: View {
    let controller: LichessBotController

    var body: some View {
        GroupBox("Model") {
            VStack(alignment: .leading, spacing: 4) {
                Text(controller.settings.model.source.rawValue)
                    .font(.headline)
                Text(controller.generation.map { "\($0.modelID) · generation \($0.generationID)" } ?? "No generation built yet")
                    .font(.system(.callout, design: .monospaced))
                Text(controller.generation?.trainingStep.map { "step \($0)" } ?? "")
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .shown(controller.generation?.trainingStep != nil)
                Text(controller.generation.map { "snapshot \($0.snapshotAt.formatted(date: .omitted, time: .standard))" } ?? "")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }
}

/// The account-wide request gate (plan §5).
struct LichessBotGateCard: View {
    let snapshot: LichessBotRequestGate.Snapshot?

    var body: some View {
        GroupBox("Requests") {
            VStack(alignment: .leading, spacing: 4) {
                Text(phaseText)
                    .font(.headline)
                    .foregroundStyle(phaseColor)
                Text(snapshot.map { "busy \($0.busy ? "yes" : "no") · waiting \($0.waiting) · awaiting move \($0.gamesAwaitingOurMove)" } ?? "")
                    .font(.system(.callout, design: .monospaced))
                    .shown(snapshot != nil)
                Text(recentText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    private var phaseText: String {
        switch snapshot?.phase {
        case .none: return "Offline"
        case .open: return snapshot?.lowClockUrgency == true ? "Open (low clock: urgent only)" : "Open"
        case .coolingDown(let remaining): return "Cooling down (\(remaining.formatted(.units(allowed: [.minutes, .seconds]))))"
        case .closed(let reason): return "Closed: \(reason)"
        }
    }

    private var phaseColor: Color {
        switch snapshot?.phase {
        case .open: return .green
        case .coolingDown: return .orange
        case .closed: return .red
        case .none: return .secondary
        }
    }

    private var recentText: String {
        guard let snapshot else { return "" }
        let counts = snapshot.recentRequestCounts
        guard !counts.isEmpty else { return "no requests in the last minute" }
        return "last minute: " + counts.sorted { $0.key.label < $1.key.label }.map { "\($0.key.label) \($0.value)" }.joined(separator: " · ")
    }
}

/// Alarms raised since launch, newest first.
struct LichessBotAlarmsCard: View {
    let controller: LichessBotController

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Alarms")
                        .font(.headline)
                    Text("\(controller.unreconciledGameIDs.count) unreconciled game(s)")
                        .font(.callout)
                        .foregroundStyle(.orange)
                        .shown(!controller.unreconciledGameIDs.isEmpty)
                    Spacer()
                    Button("Clear") {
                        controller.dismissAlarms()
                    }
                    .disabled(controller.alarms.isEmpty)
                }
                Text("None")
                    .foregroundStyle(.secondary)
                    .shown(controller.alarms.isEmpty)
                ForEach(controller.alarms.reversed()) { alarm in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(alarm.at.formatted(date: .omitted, time: .standard))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Text(alarm.text)
                            .font(.callout)
                            .textSelection(.enabled)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }
}

/// Colors for the bot's connection state, shared by the chip and window.
enum LichessBotStatusStyle {
    static func color(for state: LichessBotController.ConnectionState) -> Color {
        switch state {
        case .offline: return .gray
        case .connecting: return .yellow
        case .online: return .green
        case .draining: return .orange
        case .error: return .red
        }
    }
}
