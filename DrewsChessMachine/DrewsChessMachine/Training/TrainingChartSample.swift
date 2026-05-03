import Foundation

/// Per-second sample of training metrics for the chart grid.
/// Populated by the heartbeat alongside `ProgressRateSample`.
struct TrainingChartSample: Identifiable, Sendable {
    let id: Int
    let elapsedSec: Double

    let rollingPolicyLoss: Double?
    let rollingValueLoss: Double?
    let rollingPolicyEntropy: Double?
    let rollingPolicyNonNegCount: Double?
    let rollingPolicyNonNegIllegalCount: Double?
    let rollingGradNorm: Double?
    let replayRatio: Double?
    /// Outcome-partitioned policy loss — mean over batch positions
    /// where outcome z > +0.5 (winning game) / z < -0.5 (losing game).
    /// Splitting the conventional `rollingPolicyLoss` by outcome makes
    /// the curve unambiguous; rendered together on the upper-left
    /// chart instead of total loss.
    let rollingPolicyLossWin: Double?
    let rollingPolicyLossLoss: Double?
    /// Legal-masked Shannon entropy (in nats) over the legal-only
    /// renormalized policy softmax. Distinct from `rollingPolicyEntropy`
    /// (which is over the full 4864-dim head): a high value means
    /// "diffuse across legal moves" while the full-head pEnt can be
    /// high while concentrating on illegals. Charted on the same
    /// tile as `rollingPolicyEntropy` so the two trajectories can
    /// be compared directly.
    let rollingLegalEntropy: Double?
    /// Sum of softmax probability mass that lands on legal moves at
    /// the probed position. In `[0, 1]` — the complement is mass on
    /// illegal moves. Pulled from the periodic `LegalMassSnapshot`
    /// probe (same source as `rollingLegalEntropy`).
    let rollingLegalMass: Double?

    // System metrics
    let cpuPercent: Double?
    let gpuBusyPercent: Double?
    let gpuMemoryMB: Double?
    let appMemoryMB: Double?

    /// Whether macOS Low Power Mode was on at sample time. Charted
    /// as 0 (off) / 1 (on) — a step trace that sits along the
    /// bottom and pops up to 1 only while the user (or the system
    /// automatically on battery) has enabled the mode.
    let lowPowerMode: Bool?
    /// `ProcessInfo.ThermalState` at sample time. Charted as the
    /// raw-value offset by +2 (so nominal=2, fair=3, serious=4,
    /// critical=5), keeping the line strictly above the low-power
    /// step trace at 0/1 so they never overlap. Hover resolves the
    /// offset back to the named thermal state.
    let thermalState: ProcessInfo.ThermalState?

    var rollingTotalLoss: Double? {
        guard let p = rollingPolicyLoss, let v = rollingValueLoss else { return nil }
        return p + v
    }

    var appMemoryGB: Double? {
        appMemoryMB.map { $0 / 1024.0 }
    }

    var gpuMemoryGB: Double? {
        gpuMemoryMB.map { $0 / 1024.0 }
    }
}
