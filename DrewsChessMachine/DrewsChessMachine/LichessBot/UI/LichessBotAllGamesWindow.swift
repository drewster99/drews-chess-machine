import AppKit
import SwiftUI

/// A window listing every filed game, newest first, in a sortable table
/// (the Record card's "More…"). One at a time; reopening brings it forward.
@MainActor
final class LichessBotAllGamesWindowController: NSWindowController, NSWindowDelegate {
    private static var shared: LichessBotAllGamesWindowController?

    static func open(controller: LichessBotController) {
        SessionLogger.shared.log("[BUTTON] Open Lichess bot game list")
        if let shared {
            shared.showWindow(nil)
            shared.window?.makeKeyAndOrderFront(nil)
            return
        }
        let windowController = LichessBotAllGamesWindowController(controller: controller)
        shared = windowController
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)
    }

    private init(controller: LichessBotController) {
        let hosting = NSHostingController(rootView: LichessBotAllGamesView(controller: controller))
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1080, height: 640))
        window.minSize = NSSize(width: 820, height: 360)
        window.title = "Lichess Bot Games"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not supported")
    }

    func windowWillClose(_ notification: Notification) {
        if Self.shared === self {
            Self.shared = nil
        }
    }
}

/// Every filed game in a sortable table, with how each began (challenge-log
/// plan §3.9) and a filter by origin. The shown rows are filtered and sorted
/// into `shownRows` when the index, the origins, the filter or the sort
/// change, never in `body`.
struct LichessBotAllGamesView: View {
    let controller: LichessBotController
    @State private var sortOrder = [KeyPathComparator(\LichessBotAllGamesRow.createdAt, order: .reverse)]
    @State private var originFilter: LichessBotAllGamesOriginFilter = .all
    @State private var shownRows: [LichessBotAllGamesRow] = []
    @State private var summary = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Text(summary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Spacer()
                LichessBotAllGamesOriginFilterPicker(filter: $originFilter)
            }
            LichessBotAllGamesTable(controller: controller, rows: shownRows, sortOrder: $sortOrder)
        }
        .padding(12)
        .onChange(of: controller.index?.rows, initial: true) {
            Task { @MainActor in
                refreshShownRows()
            }
        }
        .onChange(of: controller.originsByGameID) {
            Task { @MainActor in
                refreshShownRows()
            }
        }
        .onChange(of: originFilter) {
            Task { @MainActor in
                refreshShownRows()
            }
        }
        .onChange(of: sortOrder) {
            Task { @MainActor in
                refreshShownRows()
            }
        }
    }

    private func refreshShownRows() {
        guard let index = controller.index else {
            shownRows = []
            summary = "Loading the games index…"
            return
        }
        let filtered = LichessBotAllGamesRow.rows(index.rows, origins: controller.originsByGameID, filter: originFilter)
        shownRows = filtered.sorted(using: sortOrder)
        summary = LichessBotAllGamesRow.summary(filedCount: index.rows.count, shown: filtered, filter: originFilter)
    }
}

/// The All Games window's table: one row per filed game, sortable by the
/// columns that have a sort key.
struct LichessBotAllGamesTable: View {
    let controller: LichessBotController
    let rows: [LichessBotAllGamesRow]
    @Binding var sortOrder: [KeyPathComparator<LichessBotAllGamesRow>]

    var body: some View {
        Table(rows, sortOrder: $sortOrder) {
            TableColumn("") { row in
                LichessBotResultChip(ourScore: row.summary.ourScore)
            }
            .width(26)
            TableColumn("When", value: \.createdAt) { row in
                Text(row.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.system(.callout, design: .monospaced))
            }
            .width(min: 130, ideal: 150)
            TableColumn("Color") { row in
                PieceColorDisc(color: row.summary.ourColor == .white ? .white : .black, diameter: 10)
            }
            .width(40)
            TableColumn("Origin", value: \.originSortKey) { row in
                LichessBotGameOriginLabel(display: row.origin)
            }
            .width(min: 90, ideal: 120)
            TableColumn("Opponent", value: \.opponentSortKey) { row in
                HStack(spacing: 4) {
                    LichessBotFavoriteStar(controller: controller, userID: row.summary.opponentID)
                    Text(row.summary.opponentTitle ?? "")
                        .foregroundStyle(.orange)
                        .shown(row.summary.opponentTitle != nil)
                    Text(row.summary.opponentName ?? "?")
                }
            }
            TableColumn("Rating", value: \.opponentRatingSortKey) { row in
                Text(row.summary.opponentRating.map { "\($0)" } ?? "")
                    .font(.system(.callout, design: .monospaced))
            }
            .width(min: 50, ideal: 60)
            TableColumn("Speed", value: \.speed) { row in
                Text(row.summary.speed + (row.summary.rated ? " · rated" : ""))
            }
            .width(min: 90, ideal: 110)
            TableColumn("End", value: \.status) { row in
                Text(row.summary.status)
                    .foregroundStyle(.secondary)
            }
            .width(min: 70, ideal: 90)
            TableColumn("Plies", value: \.plies) { row in
                Text("\(row.summary.plies)")
                    .font(.system(.callout, design: .monospaced))
            }
            .width(50)
            TableColumn("Game") { row in
                Button(row.summary.gameID) {
                    LichessBotLinks.openGame(row.summary.gameID)
                }
                .buttonStyle(.link)
                .help("Open on lichess.org")
            }
            .width(min: 80, ideal: 90)
        }
    }
}

/// Which games the All Games window shows, by how they began.
enum LichessBotAllGamesOriginFilter: Hashable {
    case all
    case only(LichessBotGameOriginCategory)
}

/// A filed game with its shown origin and sort keys for the table.
struct LichessBotAllGamesRow: Identifiable {
    let summary: LichessBotGameSummary
    /// From `LichessBotController.originsByGameID`, which holds every
    /// indexed game; nil would be a game the map doesn't hold yet, shown as
    /// not yet known.
    let origin: LichessBotGameOriginDisplay?
    var id: String { summary.gameID }
    var createdAt: Date { summary.createdAt }
    /// The category's place in `LichessBotGameOriginCategory.allCases`; an
    /// origin not yet known sorts after every category.
    var originSortKey: Int { origin.map { $0.category.displayOrder } ?? Int.max }
    var opponentSortKey: String { (summary.opponentName ?? "").lowercased() }
    var opponentRatingSortKey: Int { summary.opponentRating ?? Int.min }
    var speed: String { summary.speed }
    var status: String { summary.status }
    var plies: Int { summary.plies }

    /// The index rows `filter` shows, in index order. An origin not yet
    /// known matches only "All".
    static func rows(_ summaries: [LichessBotGameSummary], origins: [String: LichessBotGameOriginDisplay],
                     filter: LichessBotAllGamesOriginFilter) -> [LichessBotAllGamesRow] {
        summaries.compactMap { summary in
            let row = LichessBotAllGamesRow(summary: summary, origin: origins[summary.gameID])
            switch filter {
            case .all:
                return row
            case .only(let category):
                return row.origin?.category == category ? row : nil
            }
        }
    }

    /// The summary line: how many games are filed, how many are shown, and
    /// the shown games per origin category in category order (categories
    /// with none left out).
    static func summary(filedCount: Int, shown: [LichessBotAllGamesRow], filter: LichessBotAllGamesOriginFilter) -> String {
        var counts: [LichessBotGameOriginCategory: Int] = [:]
        var notYetKnown = 0
        for row in shown {
            if let category = row.origin?.category {
                counts[category, default: 0] += 1
            } else {
                notYetKnown += 1
            }
        }
        var parts = LichessBotGameOriginCategory.allCases.compactMap { category -> String? in
            guard let count = counts[category] else { return nil }
            return "\(LichessBotGameOriginStyle.shortLabel(for: category)) \(count)"
        }
        if notYetKnown > 0 {
            parts.append("\(LichessBotGameOriginStyle.notYetKnownLabel) \(notYetKnown)")
        }
        let head: String
        switch filter {
        case .all: head = "\(filedCount) filed games"
        case .only: head = "\(shown.count) of \(filedCount) filed games"
        }
        return parts.isEmpty ? head : "\(head): \(parts.joined(separator: " · "))"
    }
}

extension LichessBotGameOriginCategory {
    /// Where the category sorts: its place in `allCases` (pinned by a test).
    var displayOrder: Int {
        switch self {
        case .incoming: return 0
        case .challengeSheet: return 1
        case .casualResendOffer: return 2
        case .challengeQueue: return 3
        case .matchmaking: return 4
        case .matchmakingCasualResend: return 5
        case .outgoingSenderNotRecorded: return 6
        case .tournament: return 7
        case .unknown: return 8
        }
    }
}
