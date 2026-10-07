import SwiftUI

/// The ratings grid itself, for one reading of the clock.
struct LichessBotAccountRatingsGridContent: View {
    let rows: [LichessBotAccountRatingRow]
    /// Nil until the games index has loaded.
    let counts: LichessBotAccountGameCounts?

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 2) {
            GridRow {
                Text("")
                Text("Rating")
                Text("Rated")
                Text("Unrated")
                    .help("From DCM's game records; Lichess doesn't report unrated games per speed")
                Text("Today")
                    .help("Games started today, rated or not, from DCM's game records")
                Text("Last 24 h")
                    .help("Games started in the 24 hours before now, rated or not, from DCM's game records")
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            ForEach(rows, id: \.speed) { row in
                GridRow {
                    Text(row.speed)
                        .font(.callout)
                        .gridColumnAlignment(.leading)
                    Text(row.rating)
                    Text(row.ratedGames)
                    Text(count(counts?.unrated, row.speed))
                    Text(count(counts?.today, row.speed))
                    Text(count(counts?.lastDay, row.speed))
                }
                .font(.system(.callout, design: .monospaced))
            }
        }
    }

    /// The speed's count, 0 when none; "…" while the index loads.
    private func count(_ counts: [String: Int]?, _ speed: String) -> String {
        counts.map { "\($0[speed] ?? 0)" } ?? "…"
    }
}

/// One speed's rating and Lichess' rated-game count, as the account reports
/// them.
struct LichessBotAccountRatingRow: Equatable {
    let speed: String
    let rating: String
    let ratedGames: String
}
