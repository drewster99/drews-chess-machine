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
    /// Live actual learning rate the optimizer is being fed (cycle value,
    /// if active, composed with √batch + warmup). `nil` outside a training
    /// session — the cell is then omitted. Shown whenever a trainer exists,
    /// not only during warm-up. (The optional still produces a
    /// `_ConditionalContent` flip on session start/stop; not a hot path.)
    let learningRate: String?
    /// True while the LR is still ramping through warm-up, so the cell can
    /// label itself distinctly without hiding the actual value.
    let learningRateInWarmup: Bool
    /// Live actual Polyak momentum being fed (cycle value, if active, or the
    /// static coefficient). `nil` outside a training session.
    let momentum: String?
    let trainingSteps: String
    let positionsTrained: String
    let trainingRate: String
    let legalMass: String
    let runs: String
    let arenas: String
    /// Click action for the "Arenas" cell. When non-nil the cell
    /// becomes interactive (hover highlight + pointing-hand cursor)
    /// and invokes this on tap. Wired by the parent to open the full
    /// arena history sheet — the same view that the "more history"
    /// button on the arena config popover surfaces.
    let onShowArenaHistory: (() -> Void)?
    let promotions: String
    /// Click action for the "Promotions" cell. When non-nil the cell
    /// becomes interactive (hover highlight + pointing-hand cursor)
    /// and invokes this on tap. Wired by the parent to open the
    /// promotions sheet — a filtered view of the arena history list.
    let onShowPromotions: (() -> Void)?
    let lastPromoteCell: StatusBarCell
    let scoreCell: StatusBarCell
    /// "Tactical rank" rolling probe score — sum of expected-move
    /// ranks across the latest entry of each tactical probe, minus
    /// the number of probes that contributed a valid rank. 0 =
    /// every probe got its expected move ranked #1 (target). Click
    /// opens the Tactical Probe Monitor window.
    let tacticalRankCell: StatusBarCell
    /// "Tactical prob" companion to `tacticalRankCell` — mean of the
    /// legal-masked `expectedProb` across the latest entry of each
    /// probe, rendered as a percentage. 100.0000% = every probe puts
    /// all of its legal probability mass on the right move. Click
    /// opens the same Tactical Probe Monitor window.
    let tacticalProbCell: StatusBarCell
    @ViewBuilder let rightChips: () -> RightChips

    var body: some View {
        CumulativeStatusBar(
            hasHistory: hasHistory,
            isVisible: hasHistory || canRunArena,
            historyCells: {
                StatusBarCell(label: "Active training time", value: activeTrainingTime)
                if let lr = learningRate {
                    StatusBarCell(label: learningRateInWarmup ? "LR (warm-up)" : "LR", value: lr)
                }
                if let m = momentum {
                    StatusBarCell(label: "Momentum", value: m)
                }
                StatusBarCell(label: "Training steps", value: trainingSteps)
                StatusBarCell(label: "Positions trained", value: positionsTrained)
                StatusBarCell(label: "Training rate", value: trainingRate)
                StatusBarCell(label: "Legal mass", value: legalMass)
                StatusBarCell(label: "Runs", value: runs)
                StatusBarCell(label: "Arenas", value: arenas, action: onShowArenaHistory)
                StatusBarCell(label: "Promotions", value: promotions, action: onShowPromotions)
                lastPromoteCell
                scoreCell
                tacticalRankCell
                tacticalProbCell
            },
            rightChips: rightChips
        )
    }
}
