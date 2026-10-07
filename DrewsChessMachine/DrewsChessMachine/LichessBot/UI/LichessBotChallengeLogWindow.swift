import AppKit
import SwiftUI

/// A window listing every challenge DCM sent or received, from the live
/// challenge log and from the history rebuilt from the protocol log
/// (challenge-log plan §3.9; the outcomes card's "Challenge Log…"). One at a
/// time; reopening brings it forward.
@MainActor
final class LichessBotChallengeLogWindowController: NSWindowController, NSWindowDelegate {
    private static var shared: LichessBotChallengeLogWindowController?

    static func open(controller: LichessBotController) {
        SessionLogger.shared.log("[BUTTON] Open Lichess bot challenge log")
        if let shared {
            shared.showWindow(nil)
            shared.window?.makeKeyAndOrderFront(nil)
            return
        }
        let windowController = LichessBotChallengeLogWindowController(controller: controller)
        shared = windowController
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)
    }

    private init(controller: LichessBotController) {
        let hosting = NSHostingController(rootView: LichessBotChallengeLogView(controller: controller))
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1500, height: 680))
        window.minSize = NSSize(width: 1000, height: 400)
        window.title = "Lichess Bot Challenge Log"
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

/// The Challenge Log: filters, the table and a footer. The rows are built
/// (`LichessBotChallengeLogRow.rows`) when the ledger, the rebuilt history
/// or the pending challenges change, and filtered and sorted into
/// `shownRows` when those or the filter or sort change, never in `body`.
struct LichessBotChallengeLogView: View {
    let controller: LichessBotController
    @State private var filter = LichessBotChallengeLogFilter()
    @State private var sortOrder = [KeyPathComparator(\LichessBotChallengeLogRow.at, order: .reverse)]
    @State private var allRows: [LichessBotChallengeLogRow] = []
    @State private var shownRows: [LichessBotChallengeLogRow] = []
    @State private var counts = LichessBotChallengeLogCounts()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            LichessBotChallengeLogFilterBar(filter: $filter)
            LichessBotChallengeLogTable(controller: controller, rows: shownRows, sortOrder: $sortOrder)
            LichessBotChallengeLogFooter(controller: controller, counts: counts, totalRowCount: allRows.count)
        }
        .padding(12)
        .onChange(of: controller.challengeLedger, initial: true) {
            Task { @MainActor in
                rebuildRows()
            }
        }
        .onChange(of: controller.challengeHistory) {
            Task { @MainActor in
                rebuildRows()
            }
        }
        .onChange(of: controller.pendingChallenges) {
            Task { @MainActor in
                rebuildRows()
            }
        }
        .onChange(of: filter) {
            Task { @MainActor in
                refreshShownRows()
            }
        }
        .onChange(of: sortOrder) {
            Task { @MainActor in
                refreshShownRows()
            }
        }
        .task {
            // The bot window loads the ledger when it opens; this covers a
            // Challenge Log window that outlives it. Loads only while unloaded.
            await controller.loadChallengeLog()
        }
    }

    private func rebuildRows() {
        allRows = LichessBotChallengeLogRow.rows(
            ledger: controller.challengeLedger,
            reconstruction: controller.challengeHistory,
            pendingChallengeIDs: Set(controller.pendingChallenges.map(\.id))
        )
        refreshShownRows()
    }

    /// The date range is measured from now, at each refresh.
    private func refreshShownRows() {
        let filtered = filter.apply(to: allRows, now: Date())
        counts = LichessBotChallengeLogCounts(filtered)
        shownRows = filtered.sorted(using: sortOrder)
    }
}
