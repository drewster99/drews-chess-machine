import SwiftUI

/// Challenge an online bot or any player (plan §7.1). The game starts
/// through the bot's event stream if they accept.
struct LichessBotChallengeSheet: View {
    let controller: LichessBotController
    @Binding var isPresented: Bool

    @State private var source: Source = .onlineBots
    @State private var selectedBotID: String?
    @State private var typedUsername = ""
    @State private var lookedUp: LichessBotUserSummary?
    @State private var lookupError: String?
    @State private var clock: ClockChoice = .blitz5plus3
    @State private var color: LichessBotChallengeColorName = .random
    @State private var rated = false
    @State private var sendError: String?
    @State private var sending = false

    enum Source: String, CaseIterable, Identifiable {
        case onlineBots = "Online Bots"
        case username = "Username"
        var id: String { rawValue }
    }

    /// Common time controls, as Lichess offers them.
    enum ClockChoice: String, CaseIterable, Identifiable {
        case blitz3plus0 = "3+0"
        case blitz3plus2 = "3+2"
        case blitz5plus0 = "5+0"
        case blitz5plus3 = "5+3"
        case rapid10plus0 = "10+0"
        case rapid10plus5 = "10+5"
        case rapid15plus10 = "15+10"

        var id: String { rawValue }

        var seconds: (limit: Int, increment: Int) {
            switch self {
            case .blitz3plus0: return (180, 0)
            case .blitz3plus2: return (180, 2)
            case .blitz5plus0: return (300, 0)
            case .blitz5plus3: return (300, 3)
            case .rapid10plus0: return (600, 0)
            case .rapid10plus5: return (600, 5)
            case .rapid15plus10: return (900, 10)
            }
        }

        /// Lichess's speed for this clock: estimated duration
        /// `limit + 40 × increment`.
        var speed: LichessBotSpeed {
            let estimate = seconds.limit + 40 * seconds.increment
            switch estimate {
            case ..<30: return .ultraBullet
            case ..<180: return .bullet
            case ..<480: return .blitz
            case ..<1500: return .rapid
            default: return .classical
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Challenge a Player")
                .font(.title2.weight(.semibold))
            Picker("", selection: $source) {
                ForEach(Source.allCases) { source in
                    Text(source.rawValue).tag(source)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            ZStack(alignment: .topLeading) {
                LichessBotOnlineBotList(controller: controller, selectedBotID: $selectedBotID)
                    .shown(source == .onlineBots)
                LichessBotUsernameLookup(
                    controller: controller,
                    typedUsername: $typedUsername,
                    lookedUp: $lookedUp,
                    lookupError: $lookupError
                )
                .shown(source == .username)
            }
            .frame(height: 260)
            Form {
                Picker("Time control", selection: $clock) {
                    ForEach(ClockChoice.allCases) { choice in
                        Text("\(choice.rawValue)  (\(choice.speed.rawValue))").tag(choice)
                    }
                }
                Picker("DCM plays", selection: $color) {
                    Text("Random").tag(LichessBotChallengeColorName.random)
                    Text("White").tag(LichessBotChallengeColorName.white)
                    Text("Black").tag(LichessBotChallengeColorName.black)
                }
                Toggle("Rated", isOn: $rated)
            }
            Text(policyWarning)
                .font(.callout)
                .foregroundStyle(.orange)
                .shown(!policyWarning.isEmpty)
            Text(sendError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .textSelection(.enabled)
                .shown(sendError != nil)
            HStack {
                Text(targetSummary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)
                Button("Send Challenge") {
                    Task { await send() }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(target == nil || sending)
            }
        }
        .padding(20)
        .frame(width: 560)
        .task {
            await controller.refreshOnlineBots()
        }
    }

    /// The chosen opponent's username.
    private var target: String? {
        switch source {
        case .onlineBots:
            return selectedBotID.flatMap { id in controller.onlineBots.first { $0.id == id }?.username }
        case .username:
            return lookedUp?.username
        }
    }

    private var targetID: String? {
        switch source {
        case .onlineBots: return selectedBotID
        case .username: return lookedUp?.id
        }
    }

    private var targetSummary: String {
        guard let target else { return "Choose an opponent" }
        let record = controller.headToHead(against: targetID)
        return "\(target) · DCM's record vs them: \(record.wins)–\(record.draws)–\(record.losses)"
    }

    /// Choices outside the acceptance policy are allowed, with a warning
    /// (plan §7.1).
    private var policyWarning: String {
        var notes: [String] = []
        if rated && !controller.settings.challenge.acceptRated {
            notes.append("rated, but the policy accepts casual games only")
        }
        if !controller.settings.challenge.allowedSpeeds.contains(clock.speed) {
            notes.append("\(clock.speed.rawValue) is outside the policy's speeds")
        }
        return notes.isEmpty ? "" : "Outside the usual policy: " + notes.joined(separator: "; ")
    }

    private func send() async {
        guard let target else { return }
        sending = true
        defer { sending = false }
        let seconds = clock.seconds
        do {
            try await controller.sendChallenge(
                to: target,
                request: LichessBotOutgoingChallenge(rated: rated, clockLimitSeconds: seconds.limit, clockIncrementSeconds: seconds.increment, color: color)
            )
            isPresented = false
        } catch {
            sendError = error.localizedDescription
        }
    }
}

/// Bots online now, with their blitz and rapid ratings.
struct LichessBotOnlineBotList: View {
    let controller: LichessBotController
    @Binding var selectedBotID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("\(controller.onlineBots.count) bots online")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Refresh") {
                    Task { await controller.refreshOnlineBots() }
                }
            }
            List(controller.onlineBots, selection: $selectedBotID) { bot in
                HStack {
                    Text(bot.username)
                    Spacer()
                    Text(LichessBotOnlineBotList.ratingText(bot, "blitz"))
                        .font(.system(.callout, design: .monospaced))
                        .frame(width: 90, alignment: .trailing)
                    Text(LichessBotOnlineBotList.ratingText(bot, "rapid"))
                        .font(.system(.callout, design: .monospaced))
                        .frame(width: 90, alignment: .trailing)
                }
                .tag(bot.id)
            }
        }
    }

    static func ratingText(_ user: LichessBotUserSummary, _ perf: String) -> String {
        guard let rating = user.rating(perf)?.rating else { return "\(perf) –" }
        return "\(perf.prefix(1).uppercased()) " + String(format: "%4d", rating)
    }
}

/// Look up any player by username.
struct LichessBotUsernameLookup: View {
    let controller: LichessBotController
    @Binding var typedUsername: String
    @Binding var lookedUp: LichessBotUserSummary?
    @Binding var lookupError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField("Lichess username", text: $typedUsername)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await lookUp() } }
                Button("Look Up") {
                    Task { await lookUp() }
                }
                .disabled(typedUsername.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            Text(lookedUpText)
                .font(.callout)
                .shown(lookedUp != nil)
            Text(lookupError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(lookupError != nil)
            Spacer()
        }
    }

    private var lookedUpText: String {
        guard let user = lookedUp else { return "" }
        let kind = user.isBot ? "BOT" : (user.title ?? "player")
        return "\(user.username) · \(kind) · \(LichessBotOnlineBotList.ratingText(user, "blitz")) · \(LichessBotOnlineBotList.ratingText(user, "rapid"))"
            + (user.disabled == true ? " · account closed" : "")
    }

    private func lookUp() async {
        lookupError = nil
        lookedUp = nil
        do {
            lookedUp = try await controller.lookUpUser(typedUsername.trimmingCharacters(in: .whitespaces))
        } catch {
            lookupError = error.localizedDescription
        }
    }
}
