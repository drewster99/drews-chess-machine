import Foundation

/// One of the four compiled training-step executables `ChessTrainer` caches
/// per batch size (`TrainingExecutableKey`): with or without the diagnostic
/// readback targets, and with or without the KL probe's pre-update policy
/// stash.
///
/// Which one a production step runs is decided per step — diagnostics by the
/// step-line schedule (`ChessTrainer.isDiagnosticsStep`), the stash by the KL
/// probe schedule (`ChessTrainer.isKLProbeStep`) — so a training log mixes
/// them and cannot say what each costs. `--train-variant-benchmark`
/// (`TrainVariantBenchmarkCLI`) runs each one alone to measure that.
enum TrainingExecutableVariant: CaseIterable, Sendable, Hashable {
    /// Lean targets only: losses, gradient norm, parameter dynamics.
    case plain
    /// Lean targets plus the diagnostic reductions the step lines read.
    case diagnostics
    /// Lean targets plus the KL stash; every step also runs the KL probe's
    /// forward pass and the separate dropout-RNG advance.
    case klStash
    /// Both.
    case diagnosticsAndKLStash

    var includeDiagnostics: Bool {
        switch self {
        case .plain, .klStash: return false
        case .diagnostics, .diagnosticsAndKLStash: return true
        }
    }

    var includeKLStash: Bool {
        switch self {
        case .plain, .diagnostics: return false
        case .klStash, .diagnosticsAndKLStash: return true
        }
    }

    /// The name in the benchmark's table, `[BENCH]` lines and JSON.
    var label: String {
        switch self {
        case .plain: return "plain"
        case .diagnostics: return "diagnostics"
        case .klStash: return "kl-stash"
        case .diagnosticsAndKLStash: return "diagnostics+kl-stash"
        }
    }

    /// The `klProbeInterval` that makes every random-data step take this
    /// variant's KL-stash half: 1 probes every step, 0 never does
    /// (`ChessTrainer.isKLProbeStep`).
    var klProbeInterval: Int { includeKLStash ? 1 : 0 }
}

/// Order statistics of one variant's per-step times over its timed window.
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
