import Foundation

/// One of the three kinds of training step, all run by the one compiled
/// training executable `ChessTrainer` keeps per batch size
/// (`ChessTrainer.trainingExecutables`), which always computes the diagnostic
/// reductions and writes the KL probe's stash.
///
/// What differs is the work around the executable: a diagnostics step reads
/// the diagnostic values back (chosen by the step-line schedule,
/// `ChessTrainer.isDiagnosticsStep`), and a probe step also runs the KL
/// probe's forward pass before the dropout-RNG advance (the probe schedule,
/// `ChessTrainer.isKLProbeStep`; a production probe step is usually also a
/// diagnostics step). A training log mixes them and cannot say what each
/// costs; `--train-variant-benchmark` (`TrainVariantBenchmarkCLI`) runs each
/// one alone to measure that.
enum TrainingStepType: CaseIterable, Sendable, Hashable {
    /// No diagnostic readback, no probe.
    case plain
    /// The diagnostic readback the step lines use.
    case diagnostics
    /// Diagnostic readback plus the KL probe's forward pass.
    case probe

    var includeDiagnostics: Bool {
        switch self {
        case .plain: return false
        case .diagnostics, .probe: return true
        }
    }

    var runsKLProbe: Bool {
        switch self {
        case .plain, .diagnostics: return false
        case .probe: return true
        }
    }

    /// The name in the benchmark's table, `[BENCH]` lines and JSON.
    var label: String {
        switch self {
        case .plain: return "plain"
        case .diagnostics: return "diagnostics"
        case .probe: return "probe"
        }
    }

    /// The `klProbeInterval` that makes every random-data step of this type
    /// run the probe (1) or none (0) (`ChessTrainer.isKLProbeStep`).
    var klProbeInterval: Int { runsKLProbe ? 1 : 0 }
}

/// Order statistics of one step type's per-step times over its timed window.
///
/// Percentiles use linear interpolation between the two closest ranks
/// (Hyndman–Fan type 7, numpy's default): for sorted samples x₀…xₙ₋₁ the
/// p-quantile is at rank h = (n − 1)·p, x⌊h⌋ + (h − ⌊h⌋)·(x⌊h⌋₊₁ − x⌊h⌋).
/// One sample gives that sample for every percentile.
struct TimingDistribution: Equatable, Sendable {
    let count: Int
    let mean: Double
    let p10: Double
    let median: Double
    let p90: Double
    let min: Double
    let max: Double

    /// Nil when `samples` is empty (a window that completed no step has no
    /// distribution, and none is invented). Refuses non-finite samples: a
    /// wall-clock difference is always finite, so one would be a bug upstream.
    init?(samples: [Double]) {
        guard !samples.isEmpty else { return nil }
        precondition(samples.allSatisfy(\.isFinite), "TimingDistribution: non-finite sample")
        let sorted = samples.sorted()
        count = sorted.count
        mean = sorted.reduce(0, +) / Double(sorted.count)
        p10 = Self.quantile(sorted, 0.10)
        median = Self.quantile(sorted, 0.50)
        p90 = Self.quantile(sorted, 0.90)
        min = sorted[0]
        max = sorted[sorted.count - 1]
    }

    /// Type-7 quantile of already-sorted, non-empty samples.
    static func quantile(_ sorted: [Double], _ p: Double) -> Double {
        precondition(!sorted.isEmpty, "quantile of no samples")
        precondition((0...1).contains(p), "quantile p outside [0, 1]")
        let rank = Double(sorted.count - 1) * p
        let lower = Int(rank.rounded(.down))
        let upper = Swift.min(lower + 1, sorted.count - 1)
        let fraction = rank - Double(lower)
        return sorted[lower] + fraction * (sorted[upper] - sorted[lower])
    }
}

/// Signed difference `after − before` of two device byte counts. The counts
/// can shrink (MPSGraph releases transient buffers), so the delta is signed.
func signedByteDelta(from before: UInt64, to after: UInt64) -> Int64 {
    after >= before ? Int64(after - before) : -Int64(before - after)
}
