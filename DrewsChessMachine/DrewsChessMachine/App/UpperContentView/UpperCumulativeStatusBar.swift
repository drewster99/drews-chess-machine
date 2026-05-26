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
    /// Click action for the "Promotions" cell. When non-nil the cell
    /// becomes interactive (hover highlight + pointing-hand cursor)
    /// and invokes this on tap. Wired by the parent to open the
    /// promotions sheet — a filtered view of the arena history list.
    let onShowPromotions: (() -> Void)?
    let lastPromoteCell: StatusBarCell
    let scoreCell: StatusBarCell
    /// "Tactical" rolling probe score — sum of expected-move ranks
    /// across the latest entry of each tactical probe, minus the
    /// number of probes that contributed a valid rank. 0 = every
    /// probe got its expected move ranked #1 (target). Rendered as
    /// the right-most history cell so it sits next to the chip
    /// boundary, making "how is the champion doing on the manual
    /// tactical battery" an at-a-glance integer.
    let tacticalCell: StatusBarCell
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
                StatusBarCell(label: "Promotions", value: promotions, action: onShowPromotions)
                lastPromoteCell
                scoreCell
                tacticalCell
            },
            rightChips: rightChips
        )
    }
}
