import Foundation

/// A byte count as the app shows it: base-2 units (1 KB = 1024 B,
/// 1 MB = 1024², 1 GB = 1024³) labeled KB / MB / GB, the way `du -h` reports
/// disk use, so an on-screen size estimate matches what the files take.
enum BinaryByteCount {
    private static let kilobyte = 1024.0
    private static let megabyte = kilobyte * 1024
    private static let gigabyte = megabyte * 1024

    /// `bytes` in the largest unit that keeps the value at least 1, with one
    /// decimal below 100 of that unit and none from 100 up.
    static func text(_ bytes: Int) -> String {
        let value = Double(bytes)
        let (scaled, unit): (Double, String)
        if value >= gigabyte {
            (scaled, unit) = (value / gigabyte, "GB")
        } else if value >= megabyte {
            (scaled, unit) = (value / megabyte, "MB")
        } else if value >= kilobyte {
            (scaled, unit) = (value / kilobyte, "KB")
        } else {
            return "\(bytes) B"
        }
        return String(format: scaled >= 100 ? "%.0f %@" : "%.1f %@", scaled, unit)
    }
}
