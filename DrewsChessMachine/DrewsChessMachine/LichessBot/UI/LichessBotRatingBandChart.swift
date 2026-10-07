import Charts
import SwiftUI

/// DCM's actual score per rating-gap band (bars, with the 95% interval as a
/// rule) against Elo's expected score (points), §3.7. Bands run from the
/// weakest opponents on the left to the strongest on the right.
struct LichessBotRatingBandChart: View {
    let bands: [LichessBotRatingBand]

    var body: some View {
        let order = bands.map { LichessBotRatingBand.label(index: $0.index) }
        Chart(bands) { band in
            let label = LichessBotRatingBand.label(index: band.index)
            BarMark(x: .value("Gap", label), y: .value("Score", 100 * band.actualScore), width: .ratio(0.6))
                .foregroundStyle(LichessBotStatsStyle.actual.opacity(0.6))
                .accessibilityLabel("Gap \(label), \(band.games) games")
                .accessibilityValue("Actual \(LichessBotStatsFormat.percent(band.actualScore)), expected \(LichessBotStatsFormat.percent(band.meanExpected))")
            RuleMark(x: .value("Gap", label), yStart: .value("Low", 100 * (band.interval?.lower ?? band.actualScore)), yEnd: .value("High", 100 * (band.interval?.upper ?? band.actualScore)))
                .foregroundStyle(LichessBotStatsStyle.interval)
            PointMark(x: .value("Gap", label), y: .value("Expected", 100 * band.meanExpected))
                .foregroundStyle(LichessBotStatsStyle.expected)
                .symbol(.diamond)
        }
        .chartXScale(domain: order)
        .chartYScale(domain: 0...100)
        .chartXAxisLabel("Opponent rating minus DCM's")
        .chartYAxisLabel("Score %")
        .frame(height: 200)
        .help("Bars: DCM's actual score, with its 95% interval. Diamonds: the score Elo expects from the rating gaps.")
    }
}
