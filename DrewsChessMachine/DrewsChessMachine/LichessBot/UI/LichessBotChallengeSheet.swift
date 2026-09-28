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
    @State private var search = ""
    /// The rating filter, remembered across sheets and launches (a viewing
    /// preference, not a bot setting). The bounds are plain text parsed on
    /// every keystroke: a value-formatted field updates only on commit and
    /// can keep a stale value when cleared. Empty means no bound.
    @AppStorage("lichessBot.challenge.ratingFilterEnabled") private var ratingFilterEnabled = false
    @AppStorage("lichessBot.challenge.minimumRatingText") private var minimumRatingText = ""
    @AppStorage("lichessBot.challenge.maximumRatingText") private var maximumRatingText = ""
    @AppStorage("lichessBot.challenge.hidesProvisional") private var hidesProvisional = false
    @State private var sortOrder = [KeyPathComparator(\LichessBotBotRow.usernameSortKey)]
    @State private var selectedLeaderboardID: String?
    @State private var onlinePlayersError: String?

    enum Source: String, CaseIterable, Identifiable {
        case onlineBots = "Online Bots"
        case favorites = "Favorites"
        case onlinePlayers = "Online Players"
        case leaderboard = "Leaderboard"
        case username = "Username"
        var id: String { rawValue }
    }

    /// How old the online list may get while the sheet is open.
    private static let listMaximumAge: TimeInterval = 300

    /// Common time controls, as Lichess offers them.
    enum ClockChoice: String, CaseIterable, Identifiable {
        case blitz3plus0 = "3+0"
        case blitz3plus2 = "3+2"
        case blitz5plus0 = "5+0"
        case blitz5plus3 = "5+3"
        case rapid10plus0 = "10+0"
        case rapid10plus5 = "10+5"
        case rapid15plus10 = "15+10"
        case classical30plus0 = "30+0"
        case classical30plus20 = "30+20"

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
            case .classical30plus0: return (1800, 0)
            case .classical30plus20: return (1800, 20)
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
            VStack(alignment: .leading, spacing: 8) {
                LichessBotBotListHeader(controller: controller, source: source)
                LichessBotBotFilterBar(
                    search: $search,
                    ratingFilterEnabled: $ratingFilterEnabled,
                    minimumRatingText: $minimumRatingText,
                    maximumRatingText: $maximumRatingText,
                    hidesProvisional: $hidesProvisional,
                    speed: clock.speed
                )
            }
            .shown(source == .onlineBots || source == .favorites || source == .onlinePlayers)
            ZStack(alignment: .topLeading) {
                LichessBotBotTable(
                    rows: LichessBotBotList.ordered(onlineRows, filter: filter, sortOrder: sortOrder),
                    selection: $selectedBotID,
                    sortOrder: $sortOrder,
                    onToggleFavorite: { controller.toggleFavorite($0) }
                )
                .shown(source == .onlineBots)
                LichessBotBotTable(
                    rows: LichessBotBotList.ordered(favoriteRows, filter: filter, sortOrder: sortOrder),
                    selection: $selectedBotID,
                    sortOrder: $sortOrder,
                    onToggleFavorite: { controller.toggleFavorite($0) }
                )
                .shown(source == .favorites)
                LichessBotBotTable(
                    nameTitle: "Player",
                    rows: LichessBotBotList.ordered(onlinePlayerRows, filter: filter, sortOrder: sortOrder),
                    selection: $selectedBotID,
                    sortOrder: $sortOrder,
                    onToggleFavorite: { controller.toggleFavorite($0) }
                )
                .shown(source == .onlinePlayers)
                .task(id: source) {
                    guard source == .onlinePlayers else { return }
                    do {
                        try await controller.refreshOnlinePlayers()
                        onlinePlayersError = nil
                    } catch {
                        onlinePlayersError = "Couldn't load online players (an undocumented Lichess route that may have changed): \(error.localizedDescription)"
                    }
                }
                LichessBotLeaderboardList(controller: controller, speed: clock.speed, selection: $selectedLeaderboardID)
                    .shown(source == .leaderboard)
                LichessBotUsernameLookup(
                    controller: controller,
                    typedUsername: $typedUsername,
                    lookedUp: $lookedUp,
                    lookupError: $lookupError
                )
                .shown(source == .username)
            }
            .frame(height: 300)
            ZStack(alignment: .topLeading) {
                Text("Select a bot to see its profile")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .shown(target == nil)
                ForEach(target.map { [$0] } ?? [], id: \.self) { username in
                    LichessBotOpponentCard(controller: controller, username: username, compact: true)
                }
            }
            .frame(height: 58, alignment: .topLeading)
            .shown(source != .username)
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
            Text(onlinePlayersError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(source == .onlinePlayers && onlinePlayersError != nil)
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
        .frame(width: 720)
        .task {
            await controller.refreshOnlineBots()
            await controller.refreshFavoriteStatuses()
            // Keep the list fresh while the sheet is open; the task is
            // cancelled when the sheet closes, which ends the loop.
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(60))
                } catch {
                    return
                }
                if let fetchedAt = controller.onlineBotsFetchedAt, Date().timeIntervalSince(fetchedAt) >= Self.listMaximumAge {
                    await controller.refreshOnlineBots()
                    await controller.refreshFavoriteStatuses()
                }
            }
        }
    }

    private var filter: LichessBotBotListFilter {
        LichessBotBotListFilter(
            search: search,
            minimumRating: ratingFilterEnabled ? LichessBotBotListFilter.bound(from: minimumRatingText) : nil,
            maximumRating: ratingFilterEnabled ? LichessBotBotListFilter.bound(from: maximumRatingText) : nil,
            speed: clock.speed,
            hidesProvisional: hidesProvisional
        )
    }

    private var onlineRows: [LichessBotBotRow] {
        LichessBotBotList.onlineRows(bots: controller.onlineBots, notes: controller.playerNotes, now: Date(), records: controller.recordsByOpponent)
    }

    private var onlinePlayerRows: [LichessBotBotRow] {
        LichessBotBotList.onlineRows(bots: controller.onlinePlayers?.users ?? [], notes: controller.playerNotes, now: Date(), records: controller.recordsByOpponent)
    }

    private var favoriteRows: [LichessBotBotRow] {
        LichessBotBotList.favoriteRows(notes: controller.playerNotes, bots: controller.onlineBots, statuses: controller.favoriteStatuses, now: Date(), records: controller.recordsByOpponent)
    }

    /// The selected row on the current tab.
    private var selectedRow: LichessBotBotRow? {
        guard let selectedBotID else { return nil }
        switch source {
        case .onlineBots: return onlineRows.first { $0.id == selectedBotID }
        case .favorites: return favoriteRows.first { $0.id == selectedBotID }
        case .onlinePlayers: return onlinePlayerRows.first { $0.id == selectedBotID }
        case .leaderboard, .username: return nil
        }
    }

    /// The chosen opponent's username.
    private var target: String? {
        switch source {
        case .onlineBots, .favorites, .onlinePlayers:
            // An offline bot can't accept; a favorite of unknown status may.
            guard let row = selectedRow, row.isOnline != false else { return nil }
            return row.username
        case .leaderboard:
            return leaderboardSelection?.username
        case .username:
            return lookedUp?.username
        }
    }

    private var leaderboardSelection: LichessBotLeaderboardUser? {
        guard let selectedLeaderboardID else { return nil }
        return controller.leaderboards[clock.speed]?.users.first { $0.id == selectedLeaderboardID }
    }

    private var targetID: String? {
        switch source {
        case .leaderboard: return leaderboardSelection?.id
        case .onlineBots, .favorites, .onlinePlayers: return selectedRow?.id
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
        var choices: [String] = []
        if rated && !controller.settings.challenge.acceptRated {
            choices.append("“rated”")
        }
        if !controller.settings.challenge.allowedSpeeds.contains(clock.speed) {
            choices.append("“\(clock.speed.rawValue)”")
        }
        switch choices.count {
        case 0: return ""
        case 1: return "\(choices[0]) is outside the acceptance settings"
        default: return "\(choices.joined(separator: " and ")) are outside the acceptance settings"
        }
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

/// The list's size and age, DCM's own bot-game count against Lichess's
/// daily limit, and Refresh.
struct LichessBotBotListHeader: View {
    let controller: LichessBotController
    let source: LichessBotChallengeSheet.Source

    var body: some View {
        TimelineView(.periodic(from: .now, by: 30)) { context in
            HStack(spacing: 12) {
                Text(countText)
                    .lineLimit(1)
                Text(ageText(now: context.date))
                Text(ownLimitText(now: context.date))
                    .help("Lichess allows a BOT account \(LichessBotLimits.botGamesPerDay) games against other bots per rolling day")
                Spacer()
                Button("Refresh") {
                    Task {
                        await controller.refreshOnlineBots()
                        await controller.refreshFavoriteStatuses()
                    }
                }
            }
            .font(.callout)
            .foregroundStyle(.secondary)
        }
    }

    private var countText: String {
        switch source {
        case .favorites:
            guard let notes = controller.playerNotes else { return "favorites not loaded" }
            return "\(notes.favoriteIDs.count) favorites"
        case .onlinePlayers:
            guard let players = controller.onlinePlayers else { return "Online players: loading" }
            return "Top \(players.users.count) online players by rating · many refuse bots"
        case .onlineBots, .leaderboard, .username:
            let count = controller.onlineBots.count
            let capped = count >= LichessBotLimits.onlineBotsMaximum ? " (Lichess lists at most \(LichessBotLimits.onlineBotsMaximum))" : ""
            return "\(count) bots online\(capped) · ? = provisional"
        }
    }

    private func ageText(now: Date) -> String {
        let fetched = source == .onlinePlayers ? controller.onlinePlayers?.fetchedAt : controller.onlineBotsFetchedAt
        guard let fetchedAt = fetched else { return "not loaded" }
        let minutes = Int(now.timeIntervalSince(fetchedAt) / 60)
        return minutes < 1 ? "updated just now" : "updated \(minutes) min ago"
    }

    private func ownLimitText(now: Date) -> String {
        guard let games = controller.botGamesInLastDay(now: now) else { return "DCM bot games: loading" }
        return "DCM: \(games)/\(LichessBotLimits.botGamesPerDay) bot games in 24 h"
    }
}

/// Search, and a rating range for the selected time control's speed.
struct LichessBotBotFilterBar: View {
    @Binding var search: String
    @Binding var ratingFilterEnabled: Bool
    @Binding var minimumRatingText: String
    @Binding var maximumRatingText: String
    @Binding var hidesProvisional: Bool
    let speed: LichessBotSpeed

    var body: some View {
        HStack(spacing: 10) {
            TextField("Search name, real name or bio", text: $search)
                .textFieldStyle(.roundedBorder)
            Toggle("\(speed.rawValue) rating", isOn: $ratingFilterEnabled)
                .help("Filter by rating for the selected time control's speed; the range is kept while off")
            LichessBotRatingBoundField(placeholder: "min", text: $minimumRatingText)
                .disabled(!ratingFilterEnabled)
                .opacity(ratingFilterEnabled ? 1 : 0.5)
            Text("–")
                .foregroundStyle(.secondary)
            LichessBotRatingBoundField(placeholder: "max", text: $maximumRatingText)
                .disabled(!ratingFilterEnabled)
                .opacity(ratingFilterEnabled ? 1 : 0.5)
            Toggle("Hide provisional", isOn: $hidesProvisional)
        }
    }
}

/// Bots in a sortable table: a favorite star on the left, then blitz and
/// rapid ratings and games. Favorites always sort first.
struct LichessBotBotTable: View {
    /// "Bot" or "Player", for the name column.
    var nameTitle = "Bot"
    let rows: [LichessBotBotRow]
    @Binding var selection: String?
    @Binding var sortOrder: [KeyPathComparator<LichessBotBotRow>]
    let onToggleFavorite: (String) -> Void

    var body: some View {
        Table(rows, selection: $selection, sortOrder: $sortOrder) {
            TableColumn("") { row in
                Button {
                    onToggleFavorite(row.id)
                } label: {
                    Image(systemName: row.isFavorite ? "star.fill" : "star")
                        .foregroundStyle(row.isFavorite ? Color.yellow : Color.secondary)
                }
                .buttonStyle(.plain)
                .help(row.isFavorite ? "Remove from favorites" : "Add to favorites")
            }
            .width(24)
            TableColumn(nameTitle, value: \.usernameSortKey) { row in
                LichessBotBotNameCell(row: row)
            }
            TableColumn("Blitz", value: \.blitzSortKey) { row in
                Text(row.summary?.ratingText("blitz") ?? "")
                    .font(.system(.body, design: .monospaced))
            }
            .width(min: 70, ideal: 80)
            TableColumn("Rapid", value: \.rapidSortKey) { row in
                Text(row.summary?.ratingText("rapid") ?? "")
                    .font(.system(.body, design: .monospaced))
            }
            .width(min: 70, ideal: 80)
            TableColumn("Best", value: \.bestSortKey) { row in
                Text(row.summary?.bestSpeed.map { "\($0.speed) \($0.rating)\($0.provisional ? "?" : "")" } ?? "")
                    .font(.system(.body, design: .monospaced))
                    .help("The player's highest-rated speed among those they have played")
            }
            .width(min: 110, ideal: 130)
            TableColumn("vs DCM", value: \.recordSortKey) { row in
                Text(row.record.map { "\($0.wins)–\($0.draws)–\($0.losses)" } ?? "")
                    .font(.system(.body, design: .monospaced))
                    .help("DCM's wins–draws–losses against them, from its own records")
            }
            .width(min: 70, ideal: 80)
            TableColumn("Games", value: \.gamesSortKey) { row in
                Text(row.summary.map { String(format: "%6d", $0.ratedSpeedGames) } ?? "")
                    .font(.system(.body, design: .monospaced))
                    .help("Rated games at every speed")
            }
            .width(min: 70, ideal: 80)
        }
    }
}

/// A bot's name, dimmed when offline, with its daily-limit time when
/// Lichess has refused it.
struct LichessBotBotNameCell: View {
    let row: LichessBotBotRow

    var body: some View {
        HStack(spacing: 6) {
            Text(row.username)
                .foregroundStyle(row.isOnline == false ? Color.secondary : Color.primary)
            Text(row.isOnline == false ? "offline" : (row.isOnline == nil ? "status unknown" : ""))
                .font(.caption)
                .foregroundStyle(.secondary)
                .shown(row.isOnline != true)
            Text(row.limitUntil.map { "limit until \($0.formatted(date: .omitted, time: .shortened))" } ?? "")
                .font(.caption)
                .foregroundStyle(.orange)
                .help("Lichess refused a challenge: this bot played its daily bot games")
                .shown(row.limitUntil != nil)
        }
    }
}

/// Look up any player by username.
struct LichessBotUsernameLookup: View {
    let controller: LichessBotController
    @Binding var typedUsername: String
    @Binding var lookedUp: LichessBotUserSummary?
    @Binding var lookupError: String?

    @State private var matches: [LichessBotLightUser] = []
    @State private var matchesError: String?
    @State private var selectedID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField("Lichess username (3+ letters to search)", text: $typedUsername)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await lookUp(typedUsername) } }
                Button("Look Up") {
                    Task { await lookUp(typedUsername) }
                }
                .disabled(query.isEmpty)
            }
            Text(listTitle)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            List(rows, selection: $selectedID) { user in
                LichessBotLightUserRow(controller: controller, user: user)
                    .tag(user.id)
            }
            .frame(minHeight: 120)
            Text(matchesError ?? "")
                .font(.caption)
                .foregroundStyle(.red)
                .shown(matchesError != nil)
            Text(lookedUpText)
                .font(.callout.weight(.semibold))
                .shown(lookedUp != nil)
            Text(lookupError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(lookupError != nil)
        }
        // Type-ahead: wait until typing pauses, then ask Lichess. A new
        // keystroke cancels the pending search.
        .task(id: query) {
            guard query.count >= LichessBotLimits.autocompleteMinimumCharacters else {
                matches = []
                matchesError = nil
                return
            }
            do {
                try await Task.sleep(for: .milliseconds(350))
            } catch {
                return
            }
            do {
                matches = try await controller.autocompleteUsers(query)
                matchesError = nil
            } catch {
                matchesError = error.localizedDescription
            }
        }
        .onChange(of: selectedID) {
            Task { @MainActor in
                guard let selectedID, let user = rows.first(where: { $0.id == selectedID }) else { return }
                await lookUp(user.name)
            }
        }
    }

    private var query: String {
        typedUsername.trimmingCharacters(in: .whitespaces)
    }

    /// Recent opponents until a search is typed; then Lichess's matches.
    private var rows: [LichessBotLightUser] {
        query.count >= LichessBotLimits.autocompleteMinimumCharacters ? matches : (controller.recentOpponents ?? [])
    }

    private var listTitle: String {
        if query.count >= LichessBotLimits.autocompleteMinimumCharacters {
            return "Players whose names start with “\(query)”"
        }
        return controller.recentOpponents == nil ? "Loading DCM's past opponents…" : "DCM's past opponents, most recent first"
    }

    private var lookedUpText: String {
        guard let user = lookedUp else { return "" }
        let kind = user.isBot ? "BOT" : (user.title ?? "player")
        return "Selected: \(user.username) · \(kind) · blitz \(user.ratingText("blitz")) · rapid \(user.ratingText("rapid"))"
            + (user.disabled == true ? " · account closed" : "")
    }

    private func lookUp(_ username: String) async {
        let name = username.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return }
        lookupError = nil
        lookedUp = nil
        do {
            lookedUp = try await controller.lookUpUser(name)
        } catch {
            lookupError = error.localizedDescription
        }
    }
}

/// A player in the Username and Leaderboard lists, with a favorite star.
struct LichessBotLightUserRow: View {
    let controller: LichessBotController
    let user: LichessBotLightUser

    var body: some View {
        HStack(spacing: 8) {
            LichessBotFavoriteStar(controller: controller, userID: user.id)
            Circle()
                .fill(user.online == true ? Color.green : Color.clear)
                .frame(width: 7, height: 7)
                .help(user.online == true ? "Online now" : "")
            Text(user.title ?? "")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.orange)
                .shown(user.title != nil)
            Text(user.name)
        }
    }
}

/// The leaderboard for the selected time control's speed (plan §7.2):
/// Lichess's top players with a stable rating who played that speed
/// recently (Lichess's own window), each flagged online or not.
struct LichessBotLeaderboardList: View {
    let controller: LichessBotController
    let speed: LichessBotSpeed
    @Binding var selection: String?

    @State private var onlineOnly = true
    @State private var loadError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 12) {
                Text(headerText)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Toggle("Online only", isOn: $onlineOnly)
                Spacer()
                Button("Refresh") {
                    Task { await load(force: true) }
                }
            }
            Text(loadError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(loadError != nil)
            Table(rows, selection: $selection) {
                TableColumn("#") { row in
                    Text("\(row.rank)")
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .width(36)
                TableColumn("Player") { row in
                    LichessBotLightUserRow(controller: controller, user: LichessBotLightUser(id: row.user.id, name: row.user.username, title: row.user.title, online: row.user.online))
                }
                TableColumn("Rating") { row in
                    Text(row.user.perf(speed.rawValue).map { "\($0.rating)" } ?? "")
                        .font(.system(.body, design: .monospaced))
                }
                .width(min: 60, ideal: 70)
                TableColumn("Recent") { row in
                    Text(row.user.perf(speed.rawValue).map { $0.progress > 0 ? "+\($0.progress)" : "\($0.progress)" } ?? "")
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .help("Rating change over roughly the last 12 rated games")
                }
                .width(min: 60, ideal: 70)
            }
        }
        .task(id: speed) {
            await load(force: false)
        }
    }

    private struct Row: Identifiable {
        var id: String { user.id }
        let rank: Int
        let user: LichessBotLeaderboardUser
    }

    private var rows: [Row] {
        let users = controller.leaderboards[speed]?.users ?? []
        return users.enumerated()
            .map { Row(rank: $0.offset + 1, user: $0.element) }
            .filter { !onlineOnly || $0.user.online == true }
    }

    private var headerText: String {
        guard let board = controller.leaderboards[speed] else { return "Top \(speed.rawValue) players: loading…" }
        let online = board.users.filter { $0.online == true }.count
        return "Top \(board.users.count) \(speed.rawValue) players · \(online) online now"
    }

    private func load(force: Bool) async {
        do {
            try await controller.refreshLeaderboard(speed, force: force)
            loadError = nil
        } catch {
            loadError = "Couldn't load the \(speed.rawValue) leaderboard: \(error.localizedDescription)"
        }
    }
}

/// One rating bound: digits only, empty for no bound; anything else is
/// outlined in red and not applied.
struct LichessBotRatingBoundField: View {
    let placeholder: String
    @Binding var text: String

    var body: some View {
        let invalid = !text.trimmingCharacters(in: .whitespaces).isEmpty && LichessBotBotListFilter.bound(from: text) == nil
        TextField(placeholder, text: $text)
            .textFieldStyle(.roundedBorder)
            .font(.system(.body, design: .monospaced))
            .frame(width: 64)
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(Color.red, lineWidth: invalid ? 1.5 : 0)
            )
            .help(invalid ? "Whole numbers only" : "Leave empty for no bound")
    }
}
