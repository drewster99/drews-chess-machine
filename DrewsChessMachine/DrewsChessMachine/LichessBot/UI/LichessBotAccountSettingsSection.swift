import SwiftUI

/// The Lichess account and its token (plan §12.2): paste a token (checked
/// with Lichess, then stored only in the Keychain — never shown again), and
/// the guarded, irreversible BOT upgrade.
///
/// Compact once set up: all of it matters before the bot can go online, and
/// almost none of it afterwards. So a one-line status leads (account, BOT
/// or not, token state); the token's scopes, the new-token field and the
/// minting help sit under "Token details", closed unless the token needs
/// attention; and the upgrade box is shown only while the upgrade is still
/// possible — otherwise the status line says why it is gone.
struct LichessBotAccountSettingsSection: View {
    let controller: LichessBotController
    @Binding var expectedAccountID: String
    /// The account id being typed; applied only on Return or Apply, since
    /// the token, the records and a running bot are all tied to it.
    @State private var accountIDText = ""
    @State private var tokenText = ""
    @State private var upgradeConfirmation = ""
    @State private var confirmingRemove = false
    @State private var confirmingUpgrade = false
    /// Whether "Token details" is open. It opens by itself whenever the
    /// token needs attention (none saved, or a check failed) and otherwise
    /// stays as the operator left it — it never closes by itself, so a
    /// token just saved stays in view with its scopes.
    @State private var showsTokenDetails: Bool

    init(controller: LichessBotController, expectedAccountID: Binding<String>) {
        self.controller = controller
        _expectedAccountID = expectedAccountID
        _showsTokenDetails = State(initialValue: Self.tokenNeedsAttention(controller.tokenState))
    }

    var body: some View {
        Section("Account") {
            // The account id in force, not the draft's: the token state is
            // for that account, and a draft id that failed validation was
            // never applied.
            LabeledContent("Status") {
                LichessBotAccountStatusLine(status: LichessBotAccountStatus(
                    tokenState: controller.tokenState,
                    account: controller.account,
                    canUpgradeToBot: controller.canUpgradeToBot,
                    configuredAccountID: controller.settings.connection.expectedAccountID
                ))
            }
            LabeledContent("Lichess account") {
                HStack {
                    TextField("Lichess account", text: $accountIDText, prompt: Text("account id"))
                        .labelsHidden()
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 220)
                        .onSubmit { commitAccountID() }
                    Button("Apply") {
                        commitAccountID()
                    }
                    .disabled(accountIDText.trimmingCharacters(in: .whitespaces).lowercased() == expectedAccountID.lowercased())
                }
                // The running bot, its token and its records are tied to this
                // account; it changes only while offline.
                .disabled(controller.isRunning || controller.connection == .connecting)
                .help(controller.isRunning ? "Go offline to change the account" : "Press Return or Apply to use this account")
            }
            DisclosureGroup("Token details", isExpanded: $showsTokenDetails) {
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
            }
            // Kept in place while hidden so its confirmation dialog, which
            // it hosts, is never torn down mid-confirmation.
            LichessBotUpgradeBox(
                controller: controller,
                confirmationText: $upgradeConfirmation,
                confirmingUpgrade: $confirmingUpgrade
            )
            .shown(controller.canUpgradeToBot)
        }
        .task {
            if controller.tokenState == .unknown {
                await controller.refreshTokenState()
            }
        }
        .onAppear {
            accountIDText = expectedAccountID
        }
        .onChange(of: expectedAccountID) {
            Task { @MainActor in
                accountIDText = expectedAccountID
            }
        }
        .onChange(of: controller.tokenState) { _, tokenState in
            Task { @MainActor in
                if Self.tokenNeedsAttention(tokenState) {
                    showsTokenDetails = true
                }
            }
        }
    }

    /// No token saved, or the last check of one failed: the operator has to
    /// act in "Token details" before the bot can go online.
    private static func tokenNeedsAttention(_ tokenState: LichessBotController.TokenState) -> Bool {
        switch tokenState {
        case .none, .error: return true
        case .unknown, .checking, .saved: return false
        }
    }

    private func commitAccountID() {
        expectedAccountID = accountIDText.trimmingCharacters(in: .whitespaces).lowercased()
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
            return "Saved for \(info.userId) · \(info.scopes) · expires \(LichessBotAccountStatus.expiryText(info))"
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

/// The Account tab's status line: the account, the token, and the account's
/// standing with respect to the BOT upgrade.
struct LichessBotAccountStatusLine: View {
    let status: LichessBotAccountStatus

    var body: some View {
        Text("\(status.identity) · \(Text(status.token).foregroundStyle(tokenColor))")
            .monospacedDigit()
            .multilineTextAlignment(.trailing)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }

    private var tokenColor: Color {
        switch status.tokenTone {
        case .good: return .green
        case .needsSetup: return .orange
        case .problem: return .red
        case .neutral: return .secondary
        }
    }
}

/// What the Account tab's status line says, from what the controller knows
/// right now — never a guess. The account's BOT standing is stated only
/// while the token is verified (`.saved`): the account details are the
/// ones that verification fetched, and while a check is pending or after
/// one failed they may belong to an earlier token or account id, so the
/// line names only the configured account id.
struct LichessBotAccountStatus: Equatable {
    enum Tone: Equatable {
        case good
        /// Nothing is wrong, but setup isn't finished.
        case needsSetup
        case problem
        case neutral
    }

    /// The account, and — once verified — its BOT standing.
    let identity: String
    let token: String
    let tokenTone: Tone

    /// - Parameter canUpgradeToBot: `LichessBotController.canUpgradeToBot`,
    ///   the one rule for whether the upgrade box is shown, so the line and
    ///   the box always agree.
    init(
        tokenState: LichessBotController.TokenState,
        account: LichessBotAccount?,
        canUpgradeToBot: Bool,
        configuredAccountID: String
    ) {
        switch tokenState {
        case .saved(let info):
            if let account {
                identity = "\(account.username) · \(Self.botStanding(of: account, canUpgradeToBot: canUpgradeToBot))"
            } else {
                identity = "\(configuredAccountID) · account details not loaded"
            }
            token = "token OK · expires \(Self.expiryText(info))"
            tokenTone = .good
        case .unknown:
            identity = configuredAccountID
            token = "token not checked yet"
            tokenTone = .neutral
        case .checking:
            identity = configuredAccountID
            token = "checking…"
            tokenTone = .neutral
        case .none:
            identity = configuredAccountID
            token = "no token"
            tokenTone = .needsSetup
        case .error(let message):
            identity = configuredAccountID
            token = "token problem: \(message)"
            tokenTone = .problem
        }
    }

    /// When a token expires, as the operator reads it; "never" for a token
    /// without an expiry.
    static func expiryText(_ info: LichessBotTokenInfo) -> String {
        guard let expires = info.expires else { return "never" }
        return Date(timeIntervalSince1970: Double(expires) / 1000).formatted(date: .abbreviated, time: .omitted)
    }

    /// Whether the account is a BOT and, if not, why the upgrade box is or
    /// isn't shown.
    private static func botStanding(of account: LichessBotAccount, canUpgradeToBot: Bool) -> String {
        if account.isBot {
            return "BOT account"
        }
        if canUpgradeToBot {
            return "not a BOT account yet · upgrade below"
        }
        guard let count = account.count else {
            return "not a BOT account · games played not reported, so no upgrade"
        }
        return "not a BOT account · upgrade no longer possible: \(count.all) \(count.all == 1 ? "game" : "games") played"
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
                .confirmationDialog(
                    "Permanently upgrade \(username) to a BOT account?",
                    isPresented: $confirmingUpgrade,
                    actions: {
                        Button("Upgrade Permanently", role: .destructive) {
                            confirmationText = ""
                            Task { await controller.upgradeToBot() }
                        }
                    },
                    message: {
                        Text("This cannot be undone.")
                    }
                )
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
