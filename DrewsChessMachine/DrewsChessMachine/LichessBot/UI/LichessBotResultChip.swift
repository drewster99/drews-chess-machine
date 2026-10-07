import SwiftUI

/// W / D / L for DCM, colored; "–" for a game without a result.
struct LichessBotResultChip: View {
    let ourScore: Double?

    var body: some View {
        Text(letter)
            .font(LichessBotStatsStyle.chipFont)
            .foregroundStyle(LichessBotStatsStyle.chipText)
            .frame(width: 20, height: 18)
            .background(RoundedRectangle(cornerRadius: 4).fill(color))
            .help(helpText)
    }

    private var letter: String {
        switch ourScore {
        case .some(1): return "W"
        case .some(0): return "L"
        case .some: return "D"
        case .none: return "–"
        }
    }

    private var color: Color {
        switch ourScore {
        case .some(1): return LichessBotStatsStyle.win
        case .some(0): return LichessBotStatsStyle.loss
        case .some: return LichessBotStatsStyle.draw
        case .none: return LichessBotStatsStyle.unscored
        }
    }

    private var helpText: String {
        switch ourScore {
        case .some(1): return "DCM won"
        case .some(0): return "DCM lost"
        case .some: return "Draw"
        case .none: return "No result (aborted)"
        }
    }
}
