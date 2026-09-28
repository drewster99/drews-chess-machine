import SwiftUI

/// Three-color proportional bar: wins, draws, losses — as counts (an arena
/// record) or as probabilities (a value head's W/D/L). Width-flex; the
/// caller fixes the height. With nothing to show it renders a neutral gray
/// track, so the layout slot doesn't collapse.
struct WLDBar: View {
    let wins: Double
    let draws: Double
    let losses: Double

    init(wins: Double, draws: Double, losses: Double) {
        self.wins = wins
        self.draws = draws
        self.losses = losses
    }

    init(wins: Int, draws: Int, losses: Int) {
        self.init(wins: Double(wins), draws: Double(draws), losses: Double(losses))
    }

    var body: some View {
        GeometryReader { geo in
            let total = wins + draws + losses
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.gray.opacity(0.15))
                HStack(spacing: 0) {
                    Color.green.opacity(0.85).frame(width: total > 0 ? geo.size.width * wins / total : 0)
                    Color.gray.opacity(0.55).frame(width: total > 0 ? geo.size.width * draws / total : 0)
                    Color.red.opacity(0.85).frame(width: total > 0 ? geo.size.width * losses / total : 0)
                }
                .clipShape(RoundedRectangle(cornerRadius: 2))
            }
        }
        .accessibilityElement()
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        let total = wins + draws + losses
        guard total > 0 else { return "No results" }
        func share(_ value: Double) -> Int { Int((100 * value / total).rounded()) }
        return "Wins \(share(wins))%, draws \(share(draws))%, losses \(share(losses))%"
    }
}
