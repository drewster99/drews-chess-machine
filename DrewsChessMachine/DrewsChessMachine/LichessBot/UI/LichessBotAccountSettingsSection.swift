import SwiftUI

/// The Lichess account and its token (plan §12.2): paste a token (checked
/// with Lichess, then stored only in the Keychain — never shown again), and
/// the guarded, irreversible BOT upgrade.
struct LichessBotAccountSettingsSection: View {
    let controller: LichessBotController
    @Binding var expectedAccountID: String
    @State private var tokenText = ""
    @State private var upgradeConfirmation = ""
    @State private var confirmingRemove = false
    @State private var confirmingUpgrade = false

    var body: some View {
        Section("Account") {
            LabeledContent("Lichess account") {
                TextField("Lichess account", text: $expectedAccountID, prompt: Text("account id"))
                    .labelsHidden()
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 220)
            }
            LabeledContent("Token") {
                Text(statusText)
                    .foregroundStyle(statusColor)
                    .textSelection(.enabled)
            }
            LabeledContent("New token") {
                HStack {
                    SecureField("New token", text: $tokenText, prompt: Text("paste the token here"))
                        .labelsHidden()
                        .frame(width: 280)
                        .onSubmit { submit() }
                    Button("Check & Save") {
                    submit()
                }
                .disabled(tokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || controller.tokenState == .checking)
                Button("Remove…", role: .destructive) {
                    confirmingRemove = true
                }
                .disabled(!hasSavedToken)
                .confirmationDialog("Remove the saved token from the Keychain?", isPresented: $confirmingRemove) {
                    Button("Remove Token", role: .destructive) {
                        Task { await controller.removeToken() }
                    }
                }
                }
            }
            Text("Mint the token at lichess.org ▸ Preferences ▸ API access tokens, with only “Play games with the bot API” and “Create, accept, decline challenges”. One bot token, one machine.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LichessBotUpgradeBox(
                controller: controller,
                confirmationText: $upgradeConfirmation,
                confirmingUpgrade: $confirmingUpgrade
            )
        }
        .task {
            if controller.tokenState == .unknown {
                await controller.refreshTokenState()
            }
        }
    }

    private var hasSavedToken: Bool {
        if case .saved = controller.tokenState { return true }
        return false
    }

    private var statusText: String {
        switch controller.tokenState {
        case .unknown: return "Not checked"
        case .none: return "No token saved"
        case .checking: return "Checking with Lichess…"
        case .saved(let info):
            let expiry = info.expires.map { Date(timeIntervalSince1970: Double($0) / 1000).formatted(date: .abbreviated, time: .omitted) } ?? "never"
            return "Saved for \(info.userId) · \(info.scopes) · expires \(expiry)"
        case .error(let message): return message
        }
    }

    private var statusColor: Color {
        switch controller.tokenState {
        case .saved: return .green
        case .error: return .red
        default: return .secondary
        }
    }

    private func submit() {
        let text = tokenText
        tokenText = ""
        Task {
            await controller.submitToken(text)
        }
    }
}

/// The irreversible BOT upgrade, behind two confirmations.
struct LichessBotUpgradeBox: View {
    let controller: LichessBotController
    @Binding var confirmationText: String
    @Binding var confirmingUpgrade: Bool

    var body: some View {
        let username = controller.account?.username ?? ""
        VStack(alignment: .leading, spacing: 6) {
            Text("Upgrade to a BOT account")
                .font(.headline)
            Text(stateText)
                .font(.callout.weight(.semibold))
                .foregroundStyle(controller.account?.isBot == true ? Color.green : Color.secondary)
            Text("This is permanent. A BOT account can never play as a human again, and the upgrade is refused once the account has played any game. Type the account name to enable it.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(gamesText)
                .font(.callout)
                .foregroundStyle(controller.account?.count.map { $0.all > 0 } == true ? Color.red : Color.secondary)
            HStack {
                TextField("Confirmation", text: $confirmationText, prompt: Text(username.isEmpty ? "account name" : "type \(username)"))
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 260)
                    .disabled(!controller.canUpgradeToBot)
                Button("Upgrade to BOT…", role: .destructive) {
                    confirmingUpgrade = true
                }
                .disabled(!controller.canUpgradeToBot || confirmationText != username || username.isEmpty)
                .confirmationDialog("Permanently upgrade \(username) to a BOT account?", isPresented: $confirmingUpgrade) {
                    Button("Upgrade Permanently", role: .destructive) {
                        confirmationText = ""
                        Task { await controller.upgradeToBot() }
                    }
                } message: {
                    Text("This cannot be undone.")
                }
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.red.opacity(0.4)))
    }

    /// Where the account stands with respect to the upgrade.
    private var stateText: String {
        guard let account = controller.account else {
            return "Available once a token is saved and checked."
        }
        return account.isBot ? "\(account.username) is a BOT account." : "\(account.username) is not a BOT account yet."
    }

    private var gamesText: String {
        guard let count = controller.account?.count else { return "Games played: shown once the account is checked" }
        return "Games played: \(count.all)" + (count.all == 0 ? "" : " — the upgrade is no longer possible")
    }
}
