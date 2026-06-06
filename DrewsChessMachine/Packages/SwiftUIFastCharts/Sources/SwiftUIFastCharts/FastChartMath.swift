import Foundation

/// Small pure numeric transforms shared by chart callers. Lives in the
/// package so every chart that wants a smoothed overlay computes it the
/// same way (single source of truth) rather than re-deriving the
/// recurrence inline.
public enum FastChartMath {
    /// Exponential moving average over `ys` with the given span
    /// (alpha = 2/(span+1)). Inputs are assumed finite — callers pass
    /// already-plotted, finite y-values. Returns `ys` unchanged when
    /// it has 0 or 1 elements, or when `span < 1`.
    public static func ema(_ ys: [Double], span: Int) -> [Double] {
        guard ys.count > 1, span >= 1 else { return ys }
        let alpha = 2.0 / (Double(span) + 1.0)
        var out = [Double]()
        out.reserveCapacity(ys.count)
        var prev = ys[0]
        out.append(prev)
        for i in 1..<ys.count {
            prev = alpha * ys[i] + (1.0 - alpha) * prev
            out.append(prev)
        }
        return out
    }
}
