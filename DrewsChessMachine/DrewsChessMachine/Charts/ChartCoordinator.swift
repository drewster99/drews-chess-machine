import Foundation
import Observation

/// Shared chart state and decimation pipeline used by both
/// `UpperContentView` (which appends samples from the heartbeat) and
/// `LowerContentView` (which renders the chart grid).
///
/// Lifting this out of `UpperContentView` is what makes the
/// `ContentView` → `UpperContentView` + `LowerContentView` split
/// real: with state in a single coordinator, the two child views
/// can be true siblings under `ContentView` rather than having
/// `LowerContentView` nested inside `UpperContentView` to inherit
/// access to the chart `@State`.
///
/// `@Observable` so SwiftUI tracks per-property reads automatically.
/// `@MainActor` because every consumer (the heartbeat, the chart
/// grid, the menu-driven zoom commands) runs on the main actor; the
/// internal `ChartSampleRing`s are also `@MainActor`-isolated, so
/// matching isolation here keeps the call sites lock-free.
@Observable
@MainActor
final class ChartCoordinator {
    // MARK: - Underlying sample storage
    //
    // The rings are pre-allocated 24h buffers — see ChartSampleRing
    // for the chunking design. They are `let`-bound on the
    // coordinator (reference-typed), so SwiftUI's @Observable
    // tracking does NOT fire on every `append`; only the
    // downstream `decimatedFrame` mutation drives chart re-renders.

    let trainingRing = ChartSampleRing<TrainingChartSample>()
    let progressRateRing = ChartSampleRing<ProgressRateSample>()

    // MARK: - Rendered frame

    /// Output of the most recent decimation pass — read by every
    /// chart in `LowerContentView`. Recomputed on each sample
    /// append, on scroll change, on zoom change, and on session
    /// reset.
    var decimatedFrame: DecimatedChartFrame = .empty

    // MARK: - Chart navigation state

    /// Left edge of the visible scroll window in elapsed seconds.
    /// Two-way bound to `chartScrollPosition(x:)` inside the chart
    /// grid; auto-follow updates it after each sample append.
    var scrollX: Double = 0
    /// Hovered elapsed-second across every chart that participates
    /// in the synchronized crosshair; `nil` when no chart in the
    /// grid is currently under the cursor.
    var hoveredSec: Double?
    /// Active zoom-stop index into `ChartZoom.stops`.
    var chartZoomIdx: Int = ChartZoom.defaultIndex
    /// `true` when the chart-zoom is auto-snapping to the data span.
    /// Toggled `false` by manual ⌘= / ⌘- presses or by the explicit
    /// Auto checkbox; re-engaged after `chartZoomAutoReengageSec` of
    /// inactivity.
    var chartZoomAuto: Bool = true
    /// `true` when the chart should auto-advance `scrollX` to keep
    /// the latest sample at the right edge of the visible window.
    /// Flipped `false` when the user scrolls backward past a small
    /// tolerance, and back to `true` when they scroll forward to
    /// the right edge again.
    var followLatest: Bool = true
    /// Timestamp of the most recent manual zoom interaction. Drives
    /// the 1-hour auto-re-engage rule.
    var lastManualChartZoomAt: Date?

    // MARK: - Sampling cadence state

    /// Most recent wall-clock timestamp at which a progress-rate
    /// sample was appended. Used to throttle the heartbeat to 1 Hz.
    var progressRateLastFetch: Date = .distantPast
    /// Monotonic id counter for `ProgressRateSample.id`. Reset on
    /// session restart.
    var progressRateNextId: Int = 0
    /// Monotonic id counter for `TrainingChartSample.id`.
    var trainingChartNextId: Int = 0
    /// Trailing total-GPU-ms reading used to compute per-second GPU
    /// busy %. Reset to 0 on session restart so the first post-reset
    /// sample reports a sensible delta.
    var prevChartTotalGpuMs: Double = 0

    // MARK: - Auxiliary chart inputs

    /// Completed arena tournaments rendered as duration bands on
    /// the arena-activity tile. Appended to as each arena ends;
    /// cleared on session reset.
    var arenaChartEvents: [ArenaChartEvent] = []
    /// Elapsed-second mark when the in-progress arena began, or
    /// `nil` if no arena is running. Drives the live blue band on
    /// the arena chart.
    var activeArenaStartElapsed: Double?
    /// Diversity histogram bars latched off the diversity tracker
    /// once per heartbeat (with a dirty check so a stable reading
    /// doesn't re-render the chart every tick).
    var currentDiversityHistogramBars: [DiversityHistogramBar] = []

    /// Session-wide running maximum of the rolling-legal-mass series.
    /// Drives the tiered Y-axis on `LegalMassChart` so the scale is
    /// keyed off the highest legal-mass value the session has ever
    /// produced (not just the visible window). 0 until the first
    /// non-nil legal-mass sample arrives.
    var legalMassMaxAllTime: Double = 0

    // MARK: - Active flag

    /// Mirrors `UpperContentView.realTraining`. `ContentView`
    /// reads this to decide whether to render `LowerContentView`
    /// as the second child of its root `VStack`.
    var isActive: Bool = false

    // MARK: - Hard-disable switch

    /// When `false`, every collection entry point on this coordinator
    /// (`appendProgressRate`, `appendTrainingChart`, the arena hooks,
    /// the diversity-bar latch) becomes a no-op and the underlying
    /// ring buffers stay empty (and, thanks to lazy block allocation
    /// in `ChartSampleRing`, hold zero element storage). Backed by
    /// the `chartCollectionEnabled` UserDefaults key, written from
    /// the View > Collect Chart Data menu toggle. Bootstrapped from
    /// UserDefaults at init so the first `appendTrainingChart` call
    /// — which can fire before any SwiftUI `.onChange` from
    /// `@AppStorage` propagates — already sees the user's choice.
    /// Reactive thereafter: ContentView mirrors the @AppStorage
    /// value into this property via `.onChange`, so a mid-run flip
    /// stops/resumes data capture immediately.
    var collectionEnabled: Bool = (
        UserDefaults.standard.object(forKey: "chartCollectionEnabled") as? Bool ?? true
    )

    // MARK: - Constants

    nonisolated static let chartZoomAutoReengageSec: TimeInterval = 3600

    // MARK: - Sample appends

    /// Append a freshly-built `ProgressRateSample`. Updates the
    /// `progressRateLastFetch` timestamp (caller may also pass it
    /// in; we use the sample's own `timestamp` so the throttle and
    /// the sample stay aligned), bumps the id counter, and
    /// auto-advances `scrollX` to keep the latest sample on screen.
    /// Does NOT recompute `decimatedFrame` — `appendTrainingChart`
    /// is called immediately after on the same heartbeat tick and
    /// triggers the recompute.
    func appendProgressRate(_ sample: ProgressRateSample) {
        guard collectionEnabled else { return }
        progressRateRing.append(sample)
        progressRateNextId += 1
        progressRateLastFetch = sample.timestamp
        if followLatest {
            let windowSec = ChartZoom.stops[chartZoomIdx]
            scrollX = max(0, sample.elapsedSec - windowSec)
        }
    }

    /// Append a freshly-built `TrainingChartSample`, update the
    /// rolling GPU-ms baseline that the next sample's `gpuBusy %`
    /// will diff against, and recompute the decimated frame so the
    /// chart picks up the new sample on this same tick.
    func appendTrainingChart(_ sample: TrainingChartSample, totalGpuMs: Double) {
        guard collectionEnabled else { return }
        trainingRing.append(sample)
        trainingChartNextId += 1
        prevChartTotalGpuMs = totalGpuMs
        if let v = sample.rollingLegalMass, v > legalMassMaxAllTime {
            legalMassMaxAllTime = v
        }
        recomputeDecimatedFrame()
    }

    // MARK: - Reset

    /// Clear every chart-layer field back to a fresh-session state.
    /// Retains the rings' first-block reserved capacity so the new
    /// session reuses the existing 24h block of storage.
    func reset() {
        progressRateRing.reset()
        trainingRing.reset()
        decimatedFrame = .empty
        scrollX = 0
        followLatest = true
        progressRateLastFetch = .distantPast
        progressRateNextId = 0
        trainingChartNextId = 0
        prevChartTotalGpuMs = 0
        arenaChartEvents = []
        activeArenaStartElapsed = nil
        currentDiversityHistogramBars = []
        legalMassMaxAllTime = 0
    }

    // MARK: - Decimation

    /// Recompute `decimatedFrame` from the current rings + visible
    /// window. Skips the assignment when the new frame is
    /// bit-identical to the current one (avoids triggering an
    /// unnecessary chart re-render on a no-op tick — e.g. a scroll
    /// jiggle that didn't actually move the visible bucket
    /// boundary).
    func recomputeDecimatedFrame() {
        let visibleLength = ChartZoom.stops[chartZoomIdx]
        let visibleStart = max(0, scrollX)
        let frame = ChartDecimator.decimate(
            trainingRing: trainingRing,
            progressRateRing: progressRateRing,
            visibleStart: visibleStart,
            visibleLength: visibleLength,
            trainingBucketBudget: ChartDecimator.maximumBucketCount,
            progressRateBucketBudget: ChartDecimator.maximumBucketCount
        )
        if frame != decimatedFrame {
            decimatedFrame = frame
        }
    }

    // MARK: - Scroll handling

    /// Called from the chart-grid's `chartScrollPosition` binding's
    /// `onChange` hook. Decides whether the user has scrolled away
    /// from the latest sample (suspending auto-follow) and
    /// re-decimates against the new visible window so the chart
    /// marks update without waiting for the next 1 Hz append.
    func handleScrollChange(_ newValue: Double) {
        let latest = progressRateRing.last?.elapsedSec ?? 0
        let windowSec = ChartZoom.stops[chartZoomIdx]
        let latestScrollX = max(0, latest - windowSec)
        let shouldFollow = abs(newValue - latestScrollX) < 1.0
        if followLatest != shouldFollow {
            followLatest = shouldFollow
        }
        recomputeDecimatedFrame()
    }

    // MARK: - Arena event hooks

    /// Called when an arena tournament starts. Drives the live
    /// blue band on the arena-activity chart.
    func recordArenaStarted(elapsedSec: Double) {
        guard collectionEnabled else { return }
        activeArenaStartElapsed = max(0, elapsedSec)
    }

    /// Called when an arena tournament finishes. Clears the live
    /// band marker and appends a completed-arena bar to the chart.
    func recordArenaCompleted(_ event: ArenaChartEvent) {
        guard collectionEnabled else { return }
        arenaChartEvents.append(event)
        activeArenaStartElapsed = nil
    }

    /// Cancel an in-progress live band without recording a
    /// completion (e.g. arena task cancelled mid-run).
    func cancelActiveArena() {
        guard collectionEnabled else { return }
        activeArenaStartElapsed = nil
    }

    /// Replace the latched diversity histogram bars with a fresh
    /// snapshot. Caller pre-checks that the bars have actually
    /// changed before calling so SwiftUI doesn't invalidate on a
    /// stable reading.
    func setDiversityHistogramBars(_ bars: [DiversityHistogramBar]) {
        guard collectionEnabled else { return }
        currentDiversityHistogramBars = bars
    }

    // MARK: - Zoom controls

    /// Most recent elapsed-second across either ring. Source for
    /// every `ChartZoom.*(forDataSec:)` decision so manual and
    /// auto zoom modes both clamp to the same data span.
    private var lastDataSec: Double {
        progressRateRing.last?.elapsedSec ?? trainingRing.last?.elapsedSec ?? 0
    }

    var canZoomIn: Bool { chartZoomIdx > 0 }

    var canZoomOut: Bool {
        chartZoomIdx < ChartZoom.maxZoomOutIndex(forDataSec: lastDataSec)
    }

    func zoomIn() {
        let dataSec = lastDataSec
        let newIdx = max(0, chartZoomIdx - 1)
        let clamped = ChartZoom.clamp(newIdx, forDataSec: dataSec)
        let changed = clamped != chartZoomIdx
        chartZoomIdx = clamped
        chartZoomAuto = false
        lastManualChartZoomAt = Date()
        if changed { recomputeDecimatedFrame() }
    }

    func zoomOut() {
        let dataSec = lastDataSec
        let maxIdx = ChartZoom.maxZoomOutIndex(forDataSec: dataSec)
        let next = min(chartZoomIdx + 1, maxIdx)
        let changed = next != chartZoomIdx
        chartZoomIdx = next
        chartZoomAuto = false
        lastManualChartZoomAt = Date()
        if changed { recomputeDecimatedFrame() }
    }

    /// Re-enable auto-zoom and immediately snap to the auto stop.
    func enableAutoZoom() {
        let dataSec = lastDataSec
        chartZoomAuto = true
        lastManualChartZoomAt = nil
        let autoIdx = ChartZoom.autoIndex(forDataSec: dataSec)
        if chartZoomIdx != autoIdx {
            chartZoomIdx = autoIdx
            recomputeDecimatedFrame()
        }
    }

    /// Toggle handler for the Auto checkbox. Flipping off is a
    /// manual action (stamps `lastManualChartZoomAt` so the
    /// 1-hour re-engage timer applies).
    func setAutoZoom(_ enabled: Bool) {
        if enabled {
            enableAutoZoom()
        } else {
            chartZoomAuto = false
            lastManualChartZoomAt = Date()
        }
    }

    /// Per-heartbeat tick that applies:
    ///   - the auto-re-engage rule (manual mode → auto after
    ///     `chartZoomAutoReengageSec` of inactivity),
    ///   - in auto mode, snap `chartZoomIdx` to the auto stop,
    ///   - in manual mode, clamp down if the data shrunk past the
    ///     `maxZoomOutIndex`.
    /// Returns `true` iff `chartZoomIdx` or `chartZoomAuto`
    /// actually changed, so the caller can resync the menu hub.
    @discardableResult
    func refreshZoomTick() -> Bool {
        let dataSec = lastDataSec
        let priorIdx = chartZoomIdx
        let priorAuto = chartZoomAuto

        if !chartZoomAuto,
           let last = lastManualChartZoomAt,
           Date().timeIntervalSince(last) >= Self.chartZoomAutoReengageSec {
            chartZoomAuto = true
            lastManualChartZoomAt = nil
        }

        if chartZoomAuto {
            let autoIdx = ChartZoom.autoIndex(forDataSec: dataSec)
            if chartZoomIdx != autoIdx {
                chartZoomIdx = autoIdx
            }
        } else {
            let maxIdx = ChartZoom.maxZoomOutIndex(forDataSec: dataSec)
            if chartZoomIdx > maxIdx {
                chartZoomIdx = maxIdx
            }
        }

        let changed = chartZoomIdx != priorIdx || chartZoomAuto != priorAuto
        if chartZoomIdx != priorIdx {
            recomputeDecimatedFrame()
        }
        return changed
    }
}
