import SwiftUI

/// Challenge an online bot or any player (plan §7.1). The game starts
/// through the bot's event stream if they accept. Several players may be
/// selected (⌘/⇧-click); they go into the controller's challenge queue with
/// one clock, color and rated setting (plan §7.3 A).
struct LichessBotChallengeSheet: View {
    let controller: LichessBotController
    @Binding var isPresented: Bool

    @State private var source: Source = .onlineBots
    @State private var selectedBotIDs: Set<String> = []
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
    @State private var selectedLeaderboardIDs: Set<String> = []
    @State private var selectedHistoryIDs: Set<String> = []
    @State private var onlinePlayersError: String?
    @State private var tableStatusError: String?

    enum Source: String, CaseIterable, Identifiable {
        case onlineBots = "Online Bots"
        case favorites = "Favorites"
        case onlinePlayers = "Online Players"
        case leaderboard = "Leaderboard"
        case history = "History"
        case username = "Username"
        var id: String { rawValue }

        /// Shown in the shared player table (the other tabs have their own
        /// lists).
        var usesPlayerTable: Bool {
            switch self {
            case .onlineBots, .favorites, .onlinePlayers: return true
            case .leaderboard, .history, .username: return false
            }
        }
    }

    /// How old the online list may get while the sheet is open.
    private static let listMaximumAge: TimeInterval = 300

    /// Common time controls, as Lichess offers them; matchmaking picks from
    /// the same list.
    typealias ClockChoice = LichessBotClockChoice

    var body: some View {
        // Built once per render: the rows can be the whole online list.
        let rows = listRows
        let orderedRows = LichessBotBotList.ordered(rows, filter: filter, sortOrder: sortOrder)
        let chosen = chosenOpponents(in: orderedRows)
        let dailyWarning = dailyOpponentWarning(chosen)
        VStack(alignment: .leading, spacing: 12) {
            Text("Challenge a Player")
                .font(.title2.weight(.semibold))
            Picker("Players", selection: $source) {
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
                // One table for the three player-list tabs: only the visible
                // tab's rows are built, and sort order and selection carry
                // across them.
                LichessBotBotTable(
                    controller: controller,
                    nameTitle: source == .onlineBots ? "Bot" : "Player",
                    rows: orderedRows,
                    selection: $selectedBotIDs,
                    sortOrder: $sortOrder
                )
                .modifier(LichessBotPlayerStatusRefresh(
                    controller: controller,
                    ids: rows.map(\.id),
                    isActive: source.usesPlayerTable,
                    errorText: $tableStatusError
                ))
                .shown(source.usesPlayerTable)
                LichessBotLeaderboardList(controller: controller, speed: clock.speed, selection: $selectedLeaderboardIDs, isVisible: source == .leaderboard)
                    .shown(source == .leaderboard)
                LichessBotHistoryList(controller: controller, selection: $selectedHistoryIDs, isVisible: source == .history)
                    .shown(source == .history)
                LichessBotUsernameLookup(
                    controller: controller,
                    typedUsername: $typedUsername,
                    lookedUp: $lookedUp,
                    lookupError: $lookupError,
                    isVisible: source == .username
                )
                .shown(source == .username)
            }
            // The lists take whatever height the resizable sheet gives.
            .frame(minHeight: 300, maxHeight: .infinity)
            .task(id: source) {
                guard source == .onlinePlayers else { return }
                do {
                    try await controller.refreshOnlinePlayers()
                    onlinePlayersError = nil
                } catch {
                    // Cancelled by leaving the tab: not a failure to show.
                    guard !Task.isCancelled else { return }
                    onlinePlayersError = "Couldn't load online players (an undocumented Lichess route that may have changed): \(error.localizedDescription)"
                }
            }
            ZStack(alignment: .topLeading) {
                Text(selectionText(chosen))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .shown(chosen.count != 1)
                // The profile card for a single choice only.
                ForEach(chosen.count == 1 ? chosen.map(\.username) : [], id: \.self) { username in
                    LichessBotOpponentCard(controller: controller, username: username, compact: true)
                }
            }
            // Room for the card's two wrapped lines each, so the layout
            // doesn't jump as the selection changes.
            .frame(height: 84, alignment: .topLeading)
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
            Text(dailyWarning)
                .font(.callout)
                .foregroundStyle(.orange)
                .shown(!dailyWarning.isEmpty)
            Text(onlinePlayersError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(source == .onlinePlayers && onlinePlayersError != nil)
            Text(tableStatusError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(source.usesPlayerTable && tableStatusError != nil)
            Text(sendError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .textSelection(.enabled)
                .shown(sendError != nil)
            HStack {
                Text(targetSummary(chosen))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)
                Button(chosen.count > 1 ? "Send \(chosen.count) Challenges" : "Send Challenge") {
                    Task { await send(to: chosen) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(chosen.isEmpty || sending)
                .help(chosen.count > 1 ? "Queue them: each is sent when a slot is free, one at a time" : "Send the challenge now")
            }
        }
        .padding(20)
        // A flexible frame makes the sheet resizable; the lists grow with it.
        .frame(minWidth: 820, idealWidth: 980, maxWidth: .infinity, minHeight: 720, idealHeight: 800, maxHeight: .infinity)
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
                let isStale: Bool
                if let fetchedAt = controller.onlineBotsFetchedAt {
                    isStale = Date().timeIntervalSince(fetchedAt) >= Self.listMaximumAge
                } else {
                    // Never loaded (the first fetch failed): try again rather
                    // than leaving the list empty while the sheet is open.
                    isStale = true
                }
                if isStale {
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

    /// The player table's rows for the current tab, unfiltered; empty on
    /// the tabs with their own lists.
    private var listRows: [LichessBotBotRow] {
        switch source {
        case .onlineBots:
            return LichessBotBotList.onlineRows(bots: controller.onlineBots, notes: controller.playerNotesWithLiveBotLimits, now: Date(), records: controller.recordsByOpponent, statuses: controller.playerStatuses)
        case .favorites:
            return LichessBotBotList.favoriteRows(notes: controller.playerNotes, bots: controller.onlineBots, statuses: controller.playerStatuses, now: Date(), records: controller.recordsByOpponent)
        case .onlinePlayers:
            // Not loaded yet (or failed, with the error shown): no rows.
            return LichessBotBotList.onlineRows(bots: controller.onlinePlayers?.users ?? [], notes: controller.playerNotesWithLiveBotLimits, now: Date(), records: controller.recordsByOpponent, statuses: controller.playerStatuses)
        case .leaderboard, .history, .username:
            return []
        }
    }

    /// The chosen opponent: the name to challenge and the id for records.
    struct ChosenOpponent: Equatable {
        let username: String
        let id: String
    }

    /// The opponents chosen on the current tab, in the order they are sent:
    /// the player table's visible order, leaderboard rank, or most recently
    /// played first.
    private func chosenOpponents(in orderedRows: [LichessBotBotRow]) -> [ChosenOpponent] {
        switch source {
        case .onlineBots, .favorites, .onlinePlayers:
            // An offline bot can't accept; a favorite of unknown status may.
            return orderedRows
                .filter { selectedBotIDs.contains($0.id) && $0.isOnline != false }
                .map { ChosenOpponent(username: $0.username, id: $0.id) }
        case .leaderboard:
            return (controller.leaderboards[clock.speed]?.users ?? [])
                .filter { selectedLeaderboardIDs.contains($0.id) }
                .map { ChosenOpponent(username: $0.username, id: $0.id) }
        case .history:
            return controller.pastOpponents
                .filter { selectedHistoryIDs.contains($0.id) }
                .map { ChosenOpponent(username: $0.name, id: $0.id) }
        case .username:
            return lookedUp.map { [ChosenOpponent(username: $0.username, id: $0.id)] } ?? []
        }
    }

    /// Shown instead of the profile card unless exactly one player is chosen.
    private func selectionText(_ chosen: [ChosenOpponent]) -> String {
        guard !chosen.isEmpty else { return "Select a player to see their profile; ⌘- or ⇧-click to select several" }
        return "\(chosen.count) players selected: \(chosen.map(\.username).joined(separator: ", "))"
    }

    private func targetSummary(_ chosen: [ChosenOpponent]) -> String {
        guard let first = chosen.first else { return "Choose an opponent" }
        guard chosen.count == 1 else { return "\(chosen.count) players · queued and sent one at a time as slots free" }
        let record = controller.headToHead(against: first.id)
        return "\(first.username) · DCM's record vs them: \(record.wins)–\(record.draws)–\(record.losses)"
    }

    /// Chosen opponents DCM has already played today up to the daily
    /// per-opponent acceptance limit: a warning only (plan §7.1).
    private func dailyOpponentWarning(_ chosen: [ChosenOpponent]) -> String {
        let limit = controller.settings.challenge.maxGamesPerOpponentPerDay
        let over = chosen.filter { controller.gamesTodayByOpponent[$0.id.lowercased(), default: 0] >= limit }
        guard !over.isEmpty else { return "" }
        if over.count == 1, let only = over.first {
            let played = controller.gamesTodayByOpponent[only.id.lowercased(), default: 0]
            return "DCM has played \(only.username) \(played) times today; the acceptance settings allow \(limit) per opponent per day"
        }
        return "DCM has played \(over.map(\.username).joined(separator: ", ")) at least \(limit) times each today, the acceptance settings' per-opponent daily limit"
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

    /// One player is challenged at once, with any refusal shown here; several
    /// go into the challenge queue, whose progress the Overview shows.
    private func send(to chosen: [ChosenOpponent]) async {
        guard let first = chosen.first else { return }
        sending = true
        defer { sending = false }
        let request = clock.challenge(rated: rated, color: color)
        do {
            if chosen.count == 1 {
                try await controller.sendChallenge(to: first.username, request: request)
            } else {
                try controller.enqueueChallenges(
                    to: chosen.map { LichessBotChallengeQueue.Player(username: $0.username, userID: $0.id) },
                    request: request
                )
            }
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
                    .help("Lichess allows a BOT account \(LichessBotLimits.botGamesPerDay) games against other bots in a 24-hour window that opens with the first such game and then clears all at once; counted from DCM's records")
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
        case .onlineBots, .leaderboard, .history, .username:
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
        guard let window = controller.botGameWindow(now: now) else { return "DCM bot games: loading" }
        let clears = window.closesAt.map { " · clears \($0.formatted(date: .omitted, time: .shortened))" } ?? ""
        return "DCM: \(window.gamesCounted)/\(LichessBotLimits.botGamesPerDay) bot games\(clears)"
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

/// Players in a sortable table: the shared player cell (star, online dot,
/// title, name, playing or offline), then ratings, DCM's record and games.
/// Favorites always sort first.
struct LichessBotBotTable: View {
    let controller: LichessBotController
    /// "Bot" or "Player", for the name column.
    let nameTitle: String
    let rows: [LichessBotBotRow]
    @Binding var selection: Set<String>
    @Binding var sortOrder: [KeyPathComparator<LichessBotBotRow>]

    var body: some View {
        Table(rows, selection: $selection, sortOrder: $sortOrder) {
            TableColumn(nameTitle, value: \.usernameSortKey) { row in
                LichessBotPlayerNameCell(
                    controller: controller,
                    userID: row.id,
                    name: row.username,
                    title: row.title,
                    online: row.isOnline,
                    playing: row.isPlaying,
                    limitUntil: row.limitUntil
                )
            }
            .width(min: 200, ideal: 260)
            TableColumn("Bullet", value: \.bulletSortKey) { row in
                Text(row.summary?.ratingText("bullet") ?? "")
                    .font(.system(.body, design: .monospaced))
            }
            .width(min: 60, ideal: 70)
            .alignment(.trailing)
            TableColumn("Blitz", value: \.blitzSortKey) { row in
                Text(row.summary?.ratingText("blitz") ?? "")
                    .font(.system(.body, design: .monospaced))
            }
            .width(min: 60, ideal: 70)
            .alignment(.trailing)
            TableColumn("Rapid", value: \.rapidSortKey) { row in
                Text(row.summary?.ratingText("rapid") ?? "")
                    .font(.system(.body, design: .monospaced))
            }
            .width(min: 60, ideal: 70)
            .alignment(.trailing)
            TableColumn("Best", value: \.bestSortKey) { row in
                Text(row.summary?.bestSpeed.map { Self.bestText($0) } ?? "")
                    .font(.system(.body, design: .monospaced))
                    .help("The player's highest-rated speed among those they have played")
            }
            .width(min: 150, ideal: 180)
            TableColumn("vs DCM", value: \.recordSortKey) { row in
                Text(row.record.map { "\($0.wins)–\($0.draws)–\($0.losses)" } ?? "")
                    .font(.system(.body, design: .monospaced))
                    .help("DCM's wins–draws–losses against them, from its own records")
            }
            .width(min: 70, ideal: 80)
            .alignment(.trailing)
            TableColumn("Games", value: \.gamesSortKey) { row in
                Text(row.summary.map { "\($0.ratedSpeedGames)" } ?? "")
                    .font(.system(.body, design: .monospaced))
                    .help("Rated games at every speed")
            }
            .width(min: 60, ideal: 70)
            .alignment(.trailing)
        }
    }

    /// The speed and rating (with "?" when provisional): a plain string, so
    /// the rating isn't digit-grouped like a localized number.
    private static func bestText(_ best: (speed: String, rating: Int, provisional: Bool)) -> String {
        "\(best.speed) \(best.rating)\(best.provisional ? "?" : "")"
    }
}

/// Look up any player by username.
struct LichessBotUsernameLookup: View {
    let controller: LichessBotController
    @Binding var typedUsername: String
    @Binding var lookedUp: LichessBotUserSummary?
    @Binding var lookupError: String?
    let isVisible: Bool

    @State private var matches: [LichessBotLightUser] = []
    @State private var matchesError: String?
    @State private var statusError: String?
    @State private var selectedID: String?
    /// The name of the lookup whose result may still be shown. A slower,
    /// older lookup finishing last must not replace the newer choice, or
    /// Send would challenge the wrong player.
    @State private var currentLookup: String?

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
                LichessBotPlayerNameCell(
                    controller: controller,
                    userID: user.id,
                    name: user.name,
                    title: user.title,
                    online: user.online,
                    playing: user.playing
                )
                .tag(user.id)
            }
            .frame(minHeight: 120, maxHeight: .infinity)
            // Autocomplete and DCM's records carry no online status; it is
            // fetched for whatever is listed.
            .modifier(LichessBotPlayerStatusRefresh(controller: controller, ids: listed.map { $0.id.lowercased() }, isActive: isVisible, errorText: $statusError))
            Text(matchesError ?? "")
                .font(.caption)
                .foregroundStyle(.red)
                .shown(matchesError != nil)
            Text(statusError ?? "")
                .font(.caption)
                .foregroundStyle(.red)
                .shown(statusError != nil)
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
            // A new name is a new choice: drop the old result and any
            // lookup still in flight.
            if lookedUp?.username.lowercased() != query.lowercased() {
                lookedUp = nil
                lookupError = nil
                currentLookup = nil
                // So the same row can be chosen again after typing.
                selectedID = nil
            }
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
                guard !Task.isCancelled else { return }
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

    /// Recent opponents until a search is typed, then Lichess's matches.
    private var listed: [LichessBotLightUser] {
        query.count >= LichessBotLimits.autocompleteMinimumCharacters ? matches : (controller.recentOpponents ?? [])
    }

    /// `listed`, each with its fetched status when known.
    private var rows: [LichessBotLightUser] {
        listed.map { user in
            guard let status = controller.playerStatuses[user.id.lowercased()] else { return user }
            return LichessBotLightUser(id: user.id, name: user.name, title: user.title ?? status.title, online: status.online == true, playing: status.playing == true)
        }
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
        return "Selected: \(user.username) · \(kind) · bullet \(user.ratingText("bullet")) · blitz \(user.ratingText("blitz")) · rapid \(user.ratingText("rapid"))"
            + (user.disabled == true ? " · account closed" : "")
    }

    private func lookUp(_ username: String) async {
        let name = username.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return }
        currentLookup = name
        lookupError = nil
        lookedUp = nil
        do {
            let user = try await controller.lookUpUser(name)
            guard currentLookup == name else { return }
            lookedUp = user
        } catch {
            guard currentLookup == name else { return }
            lookupError = error.localizedDescription
        }
    }
}

/// A player's name as every Challenge-sheet list shows it: favorite star,
/// a green dot when online, title, name (dimmed when offline), "playing" or
/// "offline", and the bot-limit time when Lichess has refused them.
struct LichessBotPlayerNameCell: View {
    let controller: LichessBotController
    let userID: String
    let name: String
    let title: String?
    /// Nil when unknown.
    let online: Bool?
    /// Nil when unknown.
    let playing: Bool?
    var limitUntil: Date? = nil

    /// "playing", "offline", or nothing when online and idle or unknown.
    private var statusText: String {
        if playing == true { return "playing" }
        if online == false { return "offline" }
        return ""
    }

    var body: some View {
        HStack(spacing: 8) {
            LichessBotFavoriteStar(controller: controller, userID: userID)
            Circle()
                .fill(online == true ? Color.green : Color.clear)
                .frame(width: 7, height: 7)
                .help(online == true ? "Online now" : (online == false ? "Offline" : "Online status not known"))
                // Only "online" needs the dot spoken: "offline" is in the text.
                .accessibilityLabel("Online")
                .accessibilityHidden(online != true)
            Text(title ?? "")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.orange)
                .shown(title != nil)
            Text(name)
                .foregroundStyle(online == false ? Color.secondary : Color.primary)
                .lineLimit(1)
            Text(statusText)
                .font(.caption)
                .foregroundStyle(playing == true ? Color.orange : Color.secondary)
                .shown(!statusText.isEmpty)
            Text(limitUntil.map { "limit until \($0.formatted(date: .omitted, time: .shortened))" } ?? "")
                .font(.caption)
                .foregroundStyle(.orange)
                .help("Lichess refused a challenge: this bot played its daily bot games")
                .shown(limitUntil != nil)
        }
    }
}

/// Keeps `controller.playerStatuses` fresh for the ids a list shows while
/// it is visible: asks when the ids change, then again each time the
/// statuses age out. Ids fetched recently are skipped, so re-sorting or a
/// refreshed list costs no request for players already known.
struct LichessBotPlayerStatusRefresh: ViewModifier {
    let controller: LichessBotController
    let ids: [String]
    let isActive: Bool
    @Binding var errorText: String?

    private struct TaskKey: Equatable {
        let ids: [String]
        let isActive: Bool
    }

    func body(content: Content) -> some View {
        content.task(id: TaskKey(ids: ids, isActive: isActive)) {
            guard isActive, !ids.isEmpty else { return }
            while !Task.isCancelled {
                do {
                    try await controller.refreshPlayerStatuses(ids)
                    errorText = nil
                } catch {
                    // Cancelled because the list changed or was hidden: the
                    // next task reports its own outcome.
                    guard !Task.isCancelled else { return }
                    errorText = "Couldn't check who is online or playing: \(error.localizedDescription)"
                }
                do {
                    try await Task.sleep(for: .seconds(LichessBotController.playerStatusMaximumAge))
                } catch {
                    return
                }
            }
        }
    }
}

/// The leaderboard for the selected time control's speed (plan §7.2):
/// Lichess's top players with a stable rating who played that speed
/// recently (Lichess's own window), each flagged online or not.
struct LichessBotLeaderboardList: View {
    let controller: LichessBotController
    let speed: LichessBotSpeed
    @Binding var selection: Set<String>
    let isVisible: Bool

    @State private var onlineOnly = true
    @State private var loadError: String?
    @State private var statusError: String?
    @State private var sortOrder = [KeyPathComparator(\LichessBotLeaderboardRow.rank)]

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
            Text(statusError ?? "")
                .font(.callout)
                .foregroundStyle(.red)
                .shown(statusError != nil)
            Table(rows.sorted(using: sortOrder), selection: $selection, sortOrder: $sortOrder) {
                TableColumn("#", value: \.rank) { row in
                    Text("\(row.rank)")
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .width(36)
                .alignment(.trailing)
                TableColumn("Player", value: \.nameSortKey) { row in
                    LichessBotPlayerNameCell(
                        controller: controller,
                        userID: row.user.id,
                        name: row.user.username,
                        title: row.user.title,
                        online: row.online,
                        playing: row.playing
                    )
                }
                .width(min: 200, ideal: 260)
                TableColumn("\(speed.displayName) rating", value: \.rating) { row in
                    Text(row.rating == Int.min ? "" : "\(row.rating)")
                        .font(.system(.body, design: .monospaced))
                }
                .width(min: 90, ideal: 110)
                .alignment(.trailing)
                TableColumn("Recent", value: \.progress) { row in
                    Text(row.progress == Int.min ? "" : (row.progress > 0 ? "+\(row.progress)" : "\(row.progress)"))
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .help("Rating change over roughly the last dozen rated games")
                }
                .width(min: 60, ideal: 70)
                .alignment(.trailing)
            }
            .modifier(LichessBotPlayerStatusRefresh(
                controller: controller,
                ids: controller.leaderboards[speed]?.users.map { $0.id.lowercased() } ?? [],
                isActive: isVisible,
                errorText: $statusError
            ))
        }
        .task(id: LoadKey(speed: speed, isVisible: isVisible)) {
            guard isVisible else { return }
            await load(force: false)
        }
    }

    private struct LoadKey: Equatable {
        let speed: LichessBotSpeed
        let isVisible: Bool
    }

    private var rows: [LichessBotLeaderboardRow] {
        rows(onlineOnly: onlineOnly)
    }

    /// Not loaded yet (or failed, with the error shown): no rows.
    private func rows(onlineOnly: Bool) -> [LichessBotLeaderboardRow] {
        let users = controller.leaderboards[speed]?.users ?? []
        return users.enumerated()
            .map { LichessBotLeaderboardRow(rank: $0.offset + 1, user: $0.element, speed: speed, status: controller.playerStatuses[$0.element.id.lowercased()]) }
            .filter { !onlineOnly || $0.online == true }
    }

    private var headerText: String {
        guard let board = controller.leaderboards[speed] else { return "Top \(speed.rawValue) players: loading…" }
        let online = rows(onlineOnly: false).filter { $0.online == true }.count
        return "Top \(board.users.count) \(speed.rawValue) players · \(online) online now"
    }

    private func load(force: Bool) async {
        do {
            try await controller.refreshLeaderboard(speed, force: force)
            loadError = nil
        } catch {
            // Cancelled by a change of tab or time control: not a failure.
            guard !Task.isCancelled else { return }
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

/// Everyone DCM has played, from its own records (plan §7.2): favorite
/// star, online status (fetched for the most recent players), kind, games,
/// DCM's W–D–L, and when they last played.
struct LichessBotHistoryList: View {
    let controller: LichessBotController
    @Binding var selection: Set<String>
    let isVisible: Bool

    @State private var statusError: String?
    @State private var sortOrder = [KeyPathComparator(\LichessBotPastOpponent.lastPlayedAt, order: .reverse)]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(headerText)
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(statusError ?? "")
                .font(.caption)
                .foregroundStyle(.red)
                .shown(statusError != nil)
            Table(controller.pastOpponents.sorted(using: sortOrder), selection: $selection, sortOrder: $sortOrder) {
                TableColumn("Player", value: \.nameSortKey) { opponent in
                    let status = controller.playerStatuses[opponent.id]
                    LichessBotPlayerNameCell(
                        controller: controller,
                        userID: opponent.id,
                        name: opponent.name,
                        title: opponent.title,
                        online: status.map { $0.online == true },
                        playing: status.map { $0.playing == true }
                    )
                }
                .width(min: 200, ideal: 260)
                TableColumn("Kind") { opponent in
                    Text(Self.kindText(opponent.kind))
                        .foregroundStyle(.secondary)
                }
                .width(min: 60, ideal: 80)
                TableColumn("Games", value: \.gamesSortKey) { opponent in
                    Text("\(opponent.record.games)")
                        .font(.system(.body, design: .monospaced))
                }
                .width(min: 50, ideal: 60)
                .alignment(.trailing)
                TableColumn("DCM W–D–L") { opponent in
                    Text("\(opponent.record.wins)–\(opponent.record.draws)–\(opponent.record.losses)")
                        .font(.system(.body, design: .monospaced))
                }
                .width(min: 80, ideal: 100)
                .alignment(.trailing)
                TableColumn("Last played", value: \.lastPlayedAt) { opponent in
                    Text(opponent.lastPlayedAt.formatted(.relative(presentation: .named)))
                        .foregroundStyle(.secondary)
                }
                .width(min: 100, ideal: 120)
            }
        }
        // Statuses for the most recent opponents only: one request's worth,
        // however long the history grows.
        .modifier(LichessBotPlayerStatusRefresh(controller: controller, ids: statusIDs, isActive: isVisible, errorText: $statusError))
    }

    private var statusIDs: [String] {
        controller.pastOpponents.prefix(LichessBotLimits.userStatusMaximumIDs).map(\.id)
    }

    private var headerText: String {
        guard controller.index != nil else { return "Loading DCM's games…" }
        let online = statusIDs.filter { controller.playerStatuses[$0]?.online == true }.count
        return "\(controller.pastOpponents.count) players DCM has played · \(online) online now"
    }

    private static func kindText(_ kind: LichessBotOpponentKind) -> String {
        switch kind {
        case .bot: return "bot"
        case .human: return "human"
        case .lichessAI: return "Lichess AI"
        }
    }
}

/// One leaderboard entry with sort keys for its speed. A missing value
/// sorts below every real one.
struct LichessBotLeaderboardRow: Identifiable {
    var id: String { user.id }
    let rank: Int
    let user: LichessBotLeaderboardUser
    let rating: Int
    let progress: Int
    /// A fetched status is newer than the leaderboard's own online flag,
    /// which Lichess computes when it builds the list.
    let online: Bool?
    /// Nil until a status is fetched (the leaderboard doesn't say).
    let playing: Bool?
    var nameSortKey: String { user.username.lowercased() }

    init(rank: Int, user: LichessBotLeaderboardUser, speed: LichessBotSpeed, status: LichessBotUserStatus?) {
        self.rank = rank
        self.user = user
        let perf = user.perf(speed.rawValue)
        rating = perf?.rating ?? Int.min
        progress = perf?.progress ?? Int.min
        if let status {
            online = status.online == true
            playing = status.playing == true
        } else {
            online = user.online
            playing = nil
        }
    }
}
