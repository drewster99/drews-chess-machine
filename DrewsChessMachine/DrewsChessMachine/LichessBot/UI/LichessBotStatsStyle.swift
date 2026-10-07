import AppKit
import SwiftUI

/// The Record card's colors, fonts and layout constants, in one place
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §5.2). Each color carries one
/// meaning everywhere — a win is always `win` — and every one is a system
/// semantic color, so light and dark mode both follow the system.
enum LichessBotStatsStyle {

    // MARK: Colors

    static let win = Color.green
    static let draw = Color.gray
    static let loss = Color.red
    /// No result (aborted, never started).
    static let unscored = Color.secondary.opacity(0.4)
    /// Text on a filled result chip.
    static let chipText = Color.white
    /// What Elo or the value head predicted.
    static let expected = Color.orange
    /// What actually happened.
    static let actual = Color.accentColor
    /// A 95% interval's whisker.
    static let interval = Color.secondary
    /// Secondary text: footnotes, "no data" lines.
    static let neutral = Color.secondary
    /// The failure text when the statistics can't be computed.
    static let failure = Color.red
    /// The line between the statistics and the recent games.
    static let separator = Color(nsColor: .separatorColor)

    // MARK: Fonts

    /// Column headers.
    static let headerFont = Font.caption.weight(.semibold)
    /// Every number in a table: monospaced, so padded columns align.
    static let numberFont = Font.system(.callout, design: .monospaced)
    /// Sentences that carry numbers (the self-assessment figures): the body
    /// font with monospaced digits, so the numbers align without the whole
    /// sentence in a code font.
    static let figureFont = Font.callout.monospacedDigit()
    /// The text of a list row (recent games, held games, short losses).
    static let rowFont = Font.callout
    /// A row's name (a period, a speed, a model).
    static let rowLabelFont = Font.callout.weight(.medium)
    /// A pane's section title.
    static let sectionFont = Font.callout.weight(.semibold)
    /// Footnotes and secondary lines.
    static let noteFont = Font.caption
    /// The letter on a result chip.
    static let chipFont = Font.system(.caption, design: .monospaced).weight(.bold)

    // MARK: Layout

    /// The card switches from stacked to side by side at this width: the
    /// period table's ideal width plus `recentGamesMinimumWidth` and the
    /// separator, so side by side never squeezes the table. The render test
    /// measures the table (842 points with three-digit counts, 2026-10-06)
    /// and checks this bound; 1,300 leaves room for four-digit counts.
    static let wideCardWidth: CGFloat = 1_300
    static let recentGamesMinimumWidth: CGFloat = 380
    /// Beside the statistics, the recent games grow no wider than this; the
    /// statistics take the rest.
    static let recentGamesMaximumWidth: CGFloat = 532
    /// The recent games' height when stacked below the statistics.
    static let narrowRecentGamesHeight: CGFloat = 200
    /// The selected pane's least height under the panel's pickers. Without
    /// it a short card left the pane no room, so changing Show or Period
    /// seemed to do nothing.
    static let paneMinimumHeight: CGFloat = 220
    static let columnSpacing: CGFloat = 18
    static let rowSpacing: CGFloat = 5
    static let sectionSpacing: CGFloat = 10
}
