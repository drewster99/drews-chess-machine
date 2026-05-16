import Foundation

/// Default axis-label formatters. Callers can swap in their own
/// closures on `FastLineChart` for domain-specific formatting.
public enum FastChartFormatters {

    /// Compact short label suitable for chart axes — switches between
    /// `1.2K` / `3.4M` / `0.05` / `1.2e-5` based on magnitude.
    /// Mirrors DCM's existing `TrainingChartGridView.compactLabel`.
    public static let compact: @Sendable (Double) -> String = { value in
        let abs = Swift.abs(value)
        if abs >= 1_000_000 {
            return String(format: "%.1fM", value / 1_000_000)
        } else if abs >= 1_000 {
            return String(format: "%.1fK", value / 1_000)
        } else if abs >= 100 {
            return String(format: "%.0f", value)
        } else if abs >= 10 {
            return String(format: "%.1f", value)
        } else if abs >= 1 {
            return String(format: "%.2f", value)
        } else if abs >= 0.01 {
            return String(format: "%.3f", value)
        } else if abs == 0 {
            return "0"
        } else {
            return String(format: "%.1e", value)
        }
    }

    /// Elapsed-time formatter for the X axis: mm:ss under an hour,
    /// h:mm:ss otherwise. Mirrors DCM's existing
    /// `TrainingChartGridView.formatElapsedAxis`.
    public static let elapsedTime: @Sendable (Double) -> String = { seconds in
        let secs = max(0, Int(seconds.rounded()))
        let h = secs / 3600
        let m = (secs % 3600) / 60
        let s = secs % 60
        if secs < 60 {
            return String(format: "0:%02d", s)
        } else if secs < 3600 {
            return String(format: "%d:%02d", m, s)
        } else {
            return String(format: "%d:%02d:%02d", h, m, s)
        }
    }

    /// `0.0`-`1.0` → "73%". Convenient default for charts whose
    /// domain is a probability.
    public static let percent: @Sendable (Double) -> String = { value in
        String(format: "%.0f%%", value * 100)
    }
}
