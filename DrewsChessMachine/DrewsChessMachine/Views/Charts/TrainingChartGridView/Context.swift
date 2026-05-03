import Foundation

extension TrainingChartGridView {
    /// Shared chart-rendering inputs forwarded to every per-chart
    /// subview. Bundles the X-axis configuration so each chart
    /// subview's parameter list stays compact, and so a future
    /// change to (for example) the visible-domain semantics only
    /// has to touch one type definition.
    struct Context: Equatable {
        /// Full-data X domain used by `chartXScale(domain:)`.
        /// Computed as `0...max(lastElapsed, visibleDomainSec)`.
        let timeSeriesXDomain: ClosedRange<Double>
        /// Length in seconds of the visible scroll window. Passed
        /// directly to `chartXVisibleDomain(length:)`.
        let visibleDomainSec: Double
        /// Width of one decimation bucket in seconds. Used by hover-
        /// readout helpers to size the "is the cursor near a bucket"
        /// tolerance — at very wide zoom levels, the per-sample
        /// 1.5 s tolerance would cause every hover to read as
        /// no-data.
        let bucketWidthSec: Double
    }
}
