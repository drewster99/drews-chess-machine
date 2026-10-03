import Foundation

/// A byte count as the app shows it: base-2 units (1 KB = 1024 B,
/// 1 MB = 1024², 1 GB = 1024³) labeled KB / MB / GB, the way `du -h` reports
/// disk use, so an on-screen size estimate matches what the files take.
enum BinaryByteCount {
    /// The units above bytes, smallest first, each 1024 times the last.
    private static let units = ["KB", "MB", "GB"]

    /// `bytes` in the largest unit that keeps the shown value at least 1,
    /// with one decimal below 100 of that unit and none from 100 up. The
    /// value is rounded before the unit and the decimals are chosen, so a
    /// count that rounds up to 1024 of a unit is shown as 1.0 of the next
    /// one, and one that rounds up to 100 is shown without a decimal.
    static func text(_ bytes: Int) -> String {
        var scaled = Double(bytes)
        guard scaled >= 1024 else { return "\(bytes) B" }
        var unitIndex = 0
        scaled /= 1024
        while true {
            let shown = displayed(scaled)
            let hasLargerUnit = unitIndex + 1 < units.count
            if shown < 1024 || !hasLargerUnit {
                return String(format: shown >= 100 ? "%.0f %@" : "%.1f %@", shown, units[unitIndex])
            }
            scaled /= 1024
            unitIndex += 1
        }
    }

    /// `value` rounded the way it is shown: to one decimal below 100, and to
    /// a whole number once that rounding reaches 100.
    private static func displayed(_ value: Double) -> Double {
        let oneDecimal = (value * 10).rounded() / 10
        return oneDecimal >= 100 ? value.rounded() : oneDecimal
    }
}
