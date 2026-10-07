import SwiftUI

/// The ratings grid itself, for one reading of the clock.
struct LichessBotAccountRatingsGridContent: View {
    let controller: LichessBotController
    let now: Date

    var body: some View {
        let rows = LichessBotAccountRatingRow.rows(perfs: controller.account?.perfs)
        let counts = controller.accountGameCounts(now: now, calendar: .current)
        VStack(alignment: .leading, spacing: 0) {
            Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 2) {
                LichessBotAccountRatingsHeaderRow()
                ForEach(rows, id: \.speed) { row in
                    LichessBotAccountRatingsSpeedRow(row: row, counts: counts)
                }
            }
            Text(counts.failureText)
                .font(.caption)
                .foregroundStyle(.red)
                .padding(.top, 2)
                .shown(counts.isFailed)
        }
    }
}

/// One speed's rating and Lichess' rated-game count, as the account reports
/// them.
struct LichessBotAccountRatingRow: Equatable {
    let speed: String
    let rating: String
    let ratedGames: String

    /// The speeds the grid lists, in speed order.
    static let speeds = ["ultraBullet", "bullet", "blitz", "rapid", "classical"]

    /// The speeds the account has a rating in, in speed order; none before
    /// the account is loaded.
    static func rows(perfs: [String: LichessBotPerfRating]?) -> [LichessBotAccountRatingRow] {
        guard let perfs else { return [] }
        return speeds.compactMap { speed in
            guard let rating = perfs[speed], let value = rating.rating else { return nil }
            let ratingText = "\(value)" + (rating.prov == true ? "?" : " ")
            return LichessBotAccountRatingRow(speed: speed, rating: ratingText, ratedGames: rating.games.map { "\($0)" } ?? "–")
        }
    }
}
