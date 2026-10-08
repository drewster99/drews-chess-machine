import Foundation

/// The calibration table's columns: each one's title and the tooltip that
/// explains it, in one place for the header and the cells.
enum LichessBotCalibrationColumn: CaseIterable {
    case move, games, missing, predicted, actual, predictedTriple, actualTriple, brier, skill

    var title: String {
        switch self {
        case .move: return "Move"
        case .games: return "Games"
        case .missing: return "Missing"
        case .predicted: return "Predicted"
        case .actual: return "Actual"
        case .predictedTriple: return "Predicted W / D / L"
        case .actualTriple: return "Actual W / D / L"
        case .brier: return "Brier"
        case .skill: return "Skill"
        }
    }

    var help: String {
        switch self {
        case .move:
            return "DCM's Nth move of the game (its 10th, 20th, 40th). Each row covers the games that reached that move; games that ended earlier are not in it."
        case .games:
            return "Games that reached this move with the value head's reading recorded there."
        case .missing:
            return "Games that reached this move with no reading recorded there. They are left out of the row."
        case .predicted:
            return "The score the value head expected at this move (win + ½ draw), averaged over these games. Compare with Actual: higher means the network was too optimistic."
        case .actual:
            return "The score DCM really got in these games (win 1, draw ½, loss 0), averaged."
        case .predictedTriple:
            return "The value head's win / draw / loss probabilities at this move, averaged over these games. Predicted is win + ½ draw of these."
        case .actualTriple:
            return "How often these games were really won / drawn / lost."
        case .brier:
            return "How far each game's predicted (W, D, L) was from its actual result: the sum of the three squared differences, averaged. 0 is perfect, 2 the worst; lower is better."
        case .skill:
            return "The Brier score against a baseline that predicts these games' own result frequencies for every game: 1 − Brier ÷ baseline Brier. Above 0 beats the baseline, 0 matches it, below 0 is worse; \"–\" when every game had the same result."
        }
    }
}
