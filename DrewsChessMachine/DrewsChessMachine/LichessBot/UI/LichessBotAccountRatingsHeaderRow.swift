import SwiftUI

/// The ratings grid's column headings.
struct LichessBotAccountRatingsHeaderRow: View {
    var body: some View {
        GridRow {
            Text("")
            Text("Rating")
            Text("Rated")
            Text("Unrated")
                .help("From DCM's game records; Lichess doesn't report unrated games per speed")
            Text("Today")
                .help("Games started today in this Mac's calendar, rated or not, including games in progress")
            Text("Last 24 h")
                .help("Games started in the 24 hours before now, rated or not, including games in progress")
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.secondary)
    }
}
