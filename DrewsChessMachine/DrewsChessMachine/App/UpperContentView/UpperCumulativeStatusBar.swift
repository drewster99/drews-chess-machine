import SwiftUI

/// Top-of-window cumulative status bar wrapper. Owns the
/// rendering of the ~10 left-side history cells; right-side chips
/// are supplied by the caller as a `@ViewBuilder` closure so the
/// chips keep their concrete View types (no `AnyView` erasure).
/// The bar's identity boundary doesn't depend on the popovers'
/// bindings/error flags — SwiftUI can short-circuit re-evaluation
/// of the cell list when only chip state changes.
struct UpperCumulativeStatusBar<RightChips: View>: View {
    let hasHistory: Bool
    let canRunArena: Bool
    let activeTrainingTime: String
    /// `nil` outside the LR warm-up window — the cell is then
    /// omitted. (This still produces a `_ConditionalContent`
    /// flip; if it ever becomes a hot path, switch to an
    /// always-rendered cell with a `frame(width:)`-collapsed
    /// hidden state.)
    let warmupLREffective: String?
    let trainingSteps: String
    let positionsTrained: String
    let trainingRate: String
    let legalMass: String
    let runs: String
    let arenas: String
    let promotions: String
    let lastPromoteCell: StatusBarCell
    let scoreCell: StatusBarCell
    @ViewBuilder let rightChips: () -> RightChips

    var body: some View {
        CumulativeStatusBar(
            hasHistory: hasHistory,
            isVisible: hasHistory || canRunArena,
            historyCells: {
                StatusBarCell(label: "Active training time", value: activeTrainingTime)
                if let lr = warmupLREffective {
                    StatusBarCell(label: "LR effective", value: lr)
                }
                StatusBarCell(label: "Training steps", value: trainingSteps)
                StatusBarCell(label: "Positions trained", value: positionsTrained)
                StatusBarCell(label: "Training rate", value: trainingRate)
                StatusBarCell(label: "Legal mass", value: legalMass)
                StatusBarCell(label: "Runs", value: runs)
                StatusBarCell(label: "Arenas", value: arenas)
                StatusBarCell(label: "Promotions", value: promotions)
                lastPromoteCell
                scoreCell
            },
            rightChips: rightChips
        )
    }
}
