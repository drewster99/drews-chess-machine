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
        window.setContentSize(NSSize(width: 980, height: 640))
        window.minSize = NSSize(width: 720, height: 360)
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

/// Every filed game in a sortable table.
struct LichessBotAllGamesView: View {
    let controller: LichessBotController
    @State private var sortOrder = [KeyPathComparator(\LichessBotAllGamesRow.createdAt, order: .reverse)]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(summary)
                .font(.callout)
                .foregroundStyle(.secondary)
            Table(rows.sorted(using: sortOrder), sortOrder: $sortOrder) {
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
        .padding(12)
    }

    private var rows: [LichessBotAllGamesRow] {
        (controller.index?.rows ?? []).map(LichessBotAllGamesRow.init)
    }

    private var summary: String {
        guard let index = controller.index else { return "Loading the games index…" }
        return "\(index.rows.count) filed games"
    }
}

/// A filed game with sort keys for the table.
struct LichessBotAllGamesRow: Identifiable {
    let summary: LichessBotGameSummary
    var id: String { summary.gameID }
    var createdAt: Date { summary.createdAt }
    var opponentSortKey: String { (summary.opponentName ?? "").lowercased() }
    var opponentRatingSortKey: Int { summary.opponentRating ?? Int.min }
    var speed: String { summary.speed }
    var status: String { summary.status }
    var plies: Int { summary.plies }
}
