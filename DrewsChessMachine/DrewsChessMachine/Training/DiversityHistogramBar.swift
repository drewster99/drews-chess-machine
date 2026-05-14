import Foundation

/// Bar slice in the diversity histogram. Identifiable so SwiftUI's
/// `Chart` can key BarMarks correctly as counts update; Equatable so
/// `[DiversityHistogramBar]` parameters on chart views compare cheaply
/// during SwiftUI's view-diff (lets `LowerContentView` skip body
/// re-eval when the histogram hasn't changed).
struct DiversityHistogramBar: Identifiable, Sendable, Equatable {
    let id: Int          // bucket index (stable across updates)
    let label: String    // bucket label from GameDiversityTracker.histogramLabels
    let count: Int
}
