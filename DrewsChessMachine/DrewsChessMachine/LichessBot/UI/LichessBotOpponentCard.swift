import SwiftUI

/// An opponent's public profile (plan §14.3b): identity, ratings per speed,
/// totals, and DCM's head-to-head from both Lichess and our own records.
/// Fetched once per app session in the background; this view never waits
/// on the fetch.
struct LichessBotOpponentCard: View {
    let controller: LichessBotController
    let username: String
    /// One summary line instead of the full card (the Challenge sheet).
    var compact = false

    var body: some View {
        let id = username.lowercased()
        ZStack(alignment: .topLeading) {
            switch controller.opponentProfiles[id] {
            case .none, .loading:
                Text("Loading \(username)'s profile…")
                    .foregroundStyle(.secondary)
            case .failed(let reason):
                HStack {
                    Text("Couldn't load \(username)'s profile: \(reason)")
                        .foregroundStyle(.red)
                        .lineLimit(2)
                    Button("Retry") {
                        controller.loadOpponentProfile(username, retry: true)
                    }
                }
            case .loaded(let user, let crosstable, let crosstableError):
                LichessBotOpponentCardContent(
                    controller: controller,
                    user: user,
                    lichessHeadToHead: LichessBotOpponentCardContent.headToHeadText(crosstable, us: controller.botAccountID, them: id, error: crosstableError),
                    ourRecord: controller.headToHead(against: id),
                    compact: compact
                )
            }
        }
        .font(.callout)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .onAppear {
            controller.loadOpponentProfile(username)
        }
        .onChange(of: username) {
            Task { @MainActor in
                controller.loadOpponentProfile(username)
            }
        }
    }
}

/// The loaded card.
struct LichessBotOpponentCardContent: View {
    let controller: LichessBotController
    let user: LichessBotUserSummary
    let lichessHeadToHead: String
    let ourRecord: (wins: Int, draws: Int, losses: Int)
    let compact: Bool

    private static let speeds = ["bullet", "blitz", "rapid", "classical", "correspondence"]

    var body: some View {
        if compact {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    LichessBotFavoriteStar(controller: controller, userID: user.id)
                    Text([identityText, totalsText, lichessHeadToHead].filter { !$0.isEmpty }.joined(separator: " · "))
                        .lineLimit(1)
                }
                Text(user.bioFirstLine ?? "")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        LichessBotFavoriteStar(controller: controller, userID: user.id)
                        Text(user.title ?? "")
                            .font(.headline)
                            .foregroundStyle(.orange)
                            .shown(user.title != nil)
                        Text(user.username)
                            .font(.headline)
                        Label("verified", systemImage: "checkmark.seal.fill")
                            .foregroundStyle(.blue)
                            .shown(user.verified == true)
                        Text("patron")
                            .foregroundStyle(.purple)
                            .shown(user.patronColor != nil)
                        Text("account closed")
                            .foregroundStyle(.red)
                            .shown(user.disabled == true)
                        Text("marked for a terms-of-service violation")
                            .foregroundStyle(.red)
                            .shown(user.tosViolation == true)
                    }
                    Text(identityText)
                        .foregroundStyle(.secondary)
                    Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 2) {
                        GridRow {
                            Text("")
                            Text("Rating")
                            Text("±")
                            Text("Recent")
                            Text("Games")
                            Text("Rank")
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        ForEach(Self.speeds.filter { user.rating($0)?.rating != nil }, id: \.self) { speed in
                            let perf = user.rating(speed)
                            GridRow {
                                Text(speed)
                                    .gridColumnAlignment(.leading)
                                Text(user.ratingText(speed))
                                Text(perf?.rd.map { "\($0)" } ?? "")
                                Text(perf?.prog.map { $0 > 0 ? "+\($0)" : "\($0)" } ?? "")
                                Text(perf?.games.map { "\($0)" } ?? "")
                                Text(perf?.rank.map { "#\($0)" } ?? "")
                            }
                            .font(.system(.callout, design: .monospaced))
                        }
                    }
                    Text(totalsText)
                    Text(lichessHeadToHead)
                    Text("DCM's records vs them: \(ourRecord.wins)–\(ourRecord.draws)–\(ourRecord.losses)")
                    Text(overTheBoardText)
                        .foregroundStyle(.secondary)
                        .shown(!overTheBoardText.isEmpty)
                    Text(user.profile?.bio ?? "")
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(6)
                        .shown(user.profile?.bio?.isEmpty == false)
                }
                .padding(6)
            }
        }
    }

    /// Real name, location, account age, last seen.
    private var identityText: String {
        var parts: [String] = []
        if let realName = user.profile?.realName, !realName.isEmpty {
            parts.append(realName)
        }
        if let location = user.profile?.location, !location.isEmpty {
            parts.append(location)
        }
        if let createdAt = user.createdAt {
            parts.append("since \(Date(timeIntervalSince1970: Double(createdAt) / 1000).formatted(.dateTime.year().month(.abbreviated)))")
        }
        if let seenAt = user.seenAt {
            parts.append("seen \(Date(timeIntervalSince1970: Double(seenAt) / 1000).formatted(.relative(presentation: .named)))")
        }
        return parts.joined(separator: " · ")
    }

    /// "12,345 games (8,000 rated) · 6,000–1,000–5,345 · vs humans 1–0–2".
    private var totalsText: String {
        guard let count = user.count else { return "" }
        var text = "\(count.all) games"
        if let rated = count.rated {
            text += " (\(rated) rated)"
        }
        if let win = count.win, let draw = count.draw, let loss = count.loss {
            text += " · \(win)–\(draw)–\(loss)"
        }
        if let win = count.winH, let draw = count.drawH, let loss = count.lossH {
            text += " · vs humans \(win)–\(draw)–\(loss)"
        }
        if let seconds = user.playTime?.total {
            text += " · \(seconds / 3600) h played"
        }
        return text
    }

    private var overTheBoardText: String {
        var parts: [String] = []
        if let fide = user.profile?.fideRating { parts.append("FIDE \(fide)") }
        if let uscf = user.profile?.uscfRating { parts.append("USCF \(uscf)") }
        if let ecf = user.profile?.ecfRating { parts.append("ECF \(ecf)") }
        return parts.joined(separator: " · ")
    }

    /// Lichess's all-time score between the two accounts.
    static func headToHeadText(_ crosstable: LichessBotCrosstable?, us: String, them: String, error: String?) -> String {
        if let error {
            return "Lichess head-to-head unavailable: \(error)"
        }
        guard let crosstable else { return "" }
        guard crosstable.nbGames > 0 else { return "No games against DCM on Lichess" }
        let ours = crosstable.users[us].map { String(format: "%g", $0) } ?? "?"
        let theirs = crosstable.users[them].map { String(format: "%g", $0) } ?? "?"
        return "Lichess head-to-head: DCM \(ours) – \(theirs) over \(crosstable.nbGames) games"
    }
}
