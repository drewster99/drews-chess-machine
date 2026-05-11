import Foundation

/// Per-second sample of training metrics for the chart grid.
/// Populated by the heartbeat alongside `ProgressRateSample`.
///
/// On-disk schema note: this struct is persisted as JSON inside
/// `.dcmsession` bundles via `Charts/ChartFileFormat.swift`. New
/// fields must be added as Optional with no semantic break to
/// existing fields, so older saved files keep decoding (the
/// missing key just decodes as `nil`). Renaming a field, removing
/// one, or changing its units silently corrupts older saves —
/// bump `formatVersion` in the chart-file envelope and add an
/// explicit migration path for any such change.
struct TrainingChartSample: Identifiable, Sendable, Codable, Equatable {
    let id: Int
    let elapsedSec: Double

    let rollingPolicyLoss: Double?
    let rollingValueLoss: Double?
    let rollingPolicyEntropy: Double?
    let rollingPolicyNonNegCount: Double?
    let rollingPolicyNonNegIllegalCount: Double?
    let rollingGradNorm: Double?
    /// Rolling-window mean of the post-update SGD velocity-buffer L2
    /// norm (`||v||`). Sits next to `rollingGradNorm` (`||g||`) on the
    /// chart grid so velocity-vs-gradient magnitude can be compared
    /// at a glance — informative when raising the momentum coefficient
    /// μ. Source: `TrainingRunStats.rollingVelocityNorm`.
    let rollingVelocityNorm: Double?
    /// Rolling-window mean of the policy head's final 1×1 conv weight
    /// L2 norm (`pwNorm` on the [STATS] line). Tracks the magnitude of
    /// the layer that emits raw logits — monotonic growth with low
    /// entropy and saturating gradient clip = the classic "policy is
    /// sharpening into a few logits" trajectory weight decay is
    /// supposed to hold in check. Source:
    /// `TrainingRunStats.rollingPolicyHeadWeightNorm`.
    let rollingPolicyHeadWeightNorm: Double?
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
    /// Rolling-window mean of `|v|` over the value head's per-batch
    /// outputs (in `[0, 1]` since `v ∈ [-1, +1]` via `tanh`). The
    /// classic value-head saturation signal: when `vAbs → 1`, the
    /// `tanh` is in its flat tails and gradients through it have
    /// effectively vanished, which silently kills the value-loss
    /// learning signal. Source: `TrainingRunStats.rollingValueAbsMean`,
    /// the same number reported as `vAbs=` on the `[STATS]` line and
    /// already surfaced in the post-mortem dump. Fed to the
    /// value-head saturation alarm in `TrainingAlarmController` so
    /// the banner raises before the value head has been silent for
    /// a long time. Charted on the same tile as `rollingValueLoss`.
    let rollingValueAbsMean: Double?

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

/// Codable shim for Apple's `ProcessInfo.ThermalState`. The enum has
/// `Int` rawValue (0=nominal, 1=fair, 2=serious, 3=critical) but
/// does not get `Codable` conformance for free. We encode the raw
/// `Int` so JSON readers can map it back to a thermal state without
/// needing this enum type.
///
/// `@retroactive` opts in to declaring `Codable` on a type owned by
/// another module (Foundation). If Apple ever adds first-party
/// `Codable` to `ThermalState`, this declaration becomes a duplicate
/// and is the place to delete — Swift will surface a duplicate-
/// conformance error pointing here, not a silent semantic change.
extension ProcessInfo.ThermalState: @retroactive Codable {
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(Int.self)
        guard let value = ProcessInfo.ThermalState(rawValue: raw) else {
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(),
                debugDescription: "Unknown ProcessInfo.ThermalState rawValue: \(raw)"
            )
        }
        self = value
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(self.rawValue)
    }
}
