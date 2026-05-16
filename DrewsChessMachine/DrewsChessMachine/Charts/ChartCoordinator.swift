import Foundation
import Observation
import SwiftUIFastCharts

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
    /// Shared-state object passed into every `FastLineChart` that
    /// participates in the chart grid's synchronized crosshair. Owns
    /// the single source of truth for the chart-grid's hover
    /// position (`hoveredX`); `hoveredSec` below is a computed
    /// pass-through so unmigrated Arena/Diversity tiles writing
    /// through it land on the same Observable storage. This
    /// collapses what used to be two `@Observable` properties +
    /// a bidirectional `.onChange` mirror in `TrainingChartGridView`
    /// into one — every hover update now fans out a single
    /// observation invalidation.
    let fastChartGroup: FastChartGroup = FastChartGroup()

    /// Hovered elapsed-second across every chart that participates
    /// in the synchronized crosshair; `nil` when no chart in the
    /// grid is currently under the cursor. Read/written by the
    /// still-on-Swift-Charts tiles (Arena activity, Arena win
    /// trend, Diversity histogram) and by `ChartZoomControlRow`'s
    /// hover-readout. Backed by `fastChartGroup.hoveredX` so the
    /// migrated and unmigrated tiles share one Observable.
    var hoveredSec: Double? {
        get { fastChartGroup.hoveredX }
        set { fastChartGroup.hoveredX = newValue }
    }
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

    /// Reference instant against which `TrainingChartSample.elapsedSec`
    /// and `ProgressRateSample.elapsedSec` are measured. Distinct
    /// from `parallelStats.sessionStart` (which is fresh on every
    /// Play-and-Train start, including resumes, so lifetime
    /// games/hr stays accurate per-segment): the chart anchor is
    /// back-dated on resume so a saved trajectory's elapsed-second
    /// axis lines up with the post-resume axis with no visible gap.
    /// On a fresh session this is `Date()` at construction and on
    /// every `reset()`; on resume `seedFromRestoredSession(...)`
    /// sets it to `Date() - lastRestoredElapsedSec`.
    ///
    /// This is the single anchor that ALL chart-axis values must
    /// use — training samples, progress-rate samples, and arena
    /// event start/end times. Routing any of those off
    /// `parallelStats.sessionStart` directly would re-introduce the
    /// cross-anchor coordinate-space bug that the comment in
    /// `UpperContentView.refreshTrainingChartIfNeeded` originally
    /// warned about (parallel-stats vs. currentSessionStart drift),
    /// just with different participants.
    var chartElapsedAnchor: Date = Date()

    /// Most recent wall-clock timestamp at which a progress-rate
    /// sample was appended. Throttle clock for the sampler — caps it
    /// at one append per second no matter how fast the heartbeat
    /// fires (it currently fires slower than that, so this is a
    /// ceiling, not the operating point).
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
    func appendTrainingChart(_ sample: TrainingChartSample, totalGpuMs: Double) async {
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
    /// session reuses the existing 24h block of storage. Re-stamps
    /// `chartElapsedAnchor` to `Date()` so a fresh session's chart
    /// axis starts at 0.
    func reset() {
        decimationTask?.cancel()
        decimationTask = nil
        decimationGeneration += 1

        progressRateRing.reset()
        trainingRing.reset()
        decimatedFrame = .empty
        scrollX = 0
        followLatest = true
        chartElapsedAnchor = Date()
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

    /// Background task managing the current decimation pass.
    private var decimationTask: Task<Void, Never>?
    /// Generation token to prevent stale decimation frames from being published after a reset or rapid updates.
    private var decimationGeneration: Int = 0

    /// Recompute `decimatedFrame` from the current rings + visible
    /// window. Captures the visible sample slices on the main actor
    /// and offloads the heavy $O(N)$ decimation pass to a background
    /// thread to keep the UI heartbeat responsive.
    func recomputeDecimatedFrame() {
        // Cancel any stale decimation in flight.
        decimationTask?.cancel()
        decimationGeneration += 1
        let generation = decimationGeneration

        let visibleLength = ChartZoom.stops[chartZoomIdx]
        let visibleStart = max(0, scrollX)
        let lower = max(0, visibleStart)
        let upper = lower + max(0.001, visibleLength)

        // Locate the visible index range in each ring using binary search.
        // We use `upper.nextUp` so a sample exactly at the boundary is inclusive.
        let tStart = trainingRing.firstIndex(elapsedSecAtLeast: lower) { $0.elapsedSec }
        let tEnd = trainingRing.firstIndex(elapsedSecAtLeast: upper.nextUp) { $0.elapsedSec }
        let pStart = progressRateRing.firstIndex(elapsedSecAtLeast: lower) { $0.elapsedSec }
        let pEnd = progressRateRing.firstIndex(elapsedSecAtLeast: upper.nextUp) { $0.elapsedSec }

        // Copy the visible sample slices while on the main actor. Even for
        // a full 24h window (86,400 samples), this memory copy is extremely
        // fast compared to the subsequent decimation reduction.
        var tSamples: [TrainingChartSample] = []
        tSamples.reserveCapacity(tEnd - tStart)
        for i in tStart..<tEnd { tSamples.append(trainingRing[i]) }

        var pSamples: [ProgressRateSample] = []
        pSamples.reserveCapacity(pEnd - pStart)
        for i in pStart..<pEnd { pSamples.append(progressRateRing[i]) }

        let lastT = trainingRing.last?.elapsedSec
        let lastP = progressRateRing.last?.elapsedSec

        decimationTask = Task.detached { [weak self] in
            // Heavy O(N) work runs on a background global executor.
            let frame = ChartDecimator.decimate(
                trainingSamples: tSamples,
                progressRateSamples: pSamples,
                visibleStart: lower,
                visibleEnd: upper,
                lastT: lastT,
                lastP: lastP,
                trainingBucketBudget: ChartDecimator.maximumBucketCount,
                progressRateBucketBudget: ChartDecimator.maximumBucketCount
            )

            // Switch back to the main actor to publish the result.
            if !Task.isCancelled {
                await MainActor.run {
                    guard let self = self else { return }
                    guard self.decimationGeneration == generation else { return }
                    if frame != self.decimatedFrame {
                        self.decimatedFrame = frame
                    }
                }
            }
        }
    }

    // MARK: - Scroll handling

    /// Called from the chart-grid's `chartScrollPosition` binding's
    /// `onChange` hook. Decides whether the user has scrolled away
    /// from the latest sample (suspending auto-follow) and
    /// re-decimates against the new visible window so the chart
    /// marks update without waiting for the next heartbeat append.
    func handleScrollChange(_ newValue: Double) {
        let latest = progressRateRing.last?.elapsedSec ?? 0
        let windowSec = ChartZoom.stops[chartZoomIdx]
        let latestScrollX = max(0, latest - windowSec)
        let shouldFollow = abs(newValue - latestScrollX) < 1.0
        if followLatest != shouldFollow {
            followLatest = shouldFollow
        }
        // Skip the recompute when the new `scrollX` lands at (or near)
        // the auto-follow target: per-heartbeat-tick this is the
        // common case — `appendProgressRate` bumped `scrollX` and
        // `appendTrainingChart` will recompute on the same tick.
        // User-gesture scrolls (`shouldFollow == false`) still
        // recompute here so the chart slides under the cursor without
        // waiting for the next sample append. This eliminates one of
        // the two per-tick decimation passes the old code did.
        if shouldFollow {
            return
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

    // MARK: - Session save/restore

    /// Build a `ChartCoordinatorSnapshot` of the current chart state.
    /// Returns `nil` when collection is disabled or no samples have
    /// been recorded yet, so the caller can skip writing chart files
    /// entirely rather than producing an empty `training_chart.json`.
    func buildSnapshot() -> ChartCoordinatorSnapshot? {
        guard collectionEnabled else { return nil }
        guard trainingRing.count > 0 || progressRateRing.count > 0 else { return nil }
        let trainingSamples: [TrainingChartSample]
        if trainingRing.count > 0 {
            trainingSamples = (0..<trainingRing.count).map { trainingRing[$0] }
        } else {
            trainingSamples = []
        }
        let progressSamples: [ProgressRateSample]
        if progressRateRing.count > 0 {
            progressSamples = (0..<progressRateRing.count).map { progressRateRing[$0] }
        } else {
            progressSamples = []
        }
        let lastTrainElapsed = trainingSamples.last?.elapsedSec ?? 0
        let lastProgressElapsed = progressSamples.last?.elapsedSec ?? 0
        let lastElapsed = max(lastTrainElapsed, lastProgressElapsed)
        return ChartCoordinatorSnapshot(
            trainingSamples: trainingSamples,
            progressRateSamples: progressSamples,
            arenaChartEvents: arenaChartEvents,
            legalMassMaxAllTime: legalMassMaxAllTime,
            lastElapsedSec: lastElapsed
        )
    }

    /// Seed this coordinator from a restored snapshot. Called once
    /// at session-resume time, after `reset()` and before the
    /// heartbeat starts pushing new samples. Bulk-fills both rings
    /// (using the ring's no-side-effect `bulkRestore`), restores
    /// arena events and the legal-mass running max, advances the
    /// id counters past the restored samples so new appends don't
    /// collide, back-dates `chartElapsedAnchor` so post-resume
    /// elapsedSec values continue monotonically from the saved
    /// trajectory, and recomputes the decimated frame once.
    ///
    /// Also pre-advances `scrollX` to land the visible window on the
    /// most recent restored sample. Without this the chart would
    /// briefly render at `scrollX = 0` after resume — showing the
    /// leftmost minute or two of restored data — until the first
    /// post-resume heartbeat tick re-runs the auto-follow math
    /// inside `appendProgressRate`. Same expression `appendProgressRate`
    /// uses on every new append, so the user sees the latest data
    /// immediately on Resume rather than a brief "rewind" flash.
    func seedFromRestoredSession(_ snapshot: ChartCoordinatorSnapshot) {
        trainingRing.bulkRestore(snapshot.trainingSamples)
        progressRateRing.bulkRestore(snapshot.progressRateSamples)
        arenaChartEvents = snapshot.arenaChartEvents
        legalMassMaxAllTime = snapshot.legalMassMaxAllTime
        trainingChartNextId = snapshot.trainingSamples.count
        progressRateNextId = snapshot.progressRateSamples.count
        chartElapsedAnchor = Date().addingTimeInterval(-snapshot.lastElapsedSec)
        if followLatest {
            // Defensive clamp: `chartZoomIdx` is normally bounded by
            // the zoom controls, but a corrupted persisted value
            // would otherwise trap on the unchecked subscript here.
            // Clamping (rather than early-returning) preserves the
            // invariant that `recomputeDecimatedFrame()` always runs
            // at the end of seedFromRestoredSession, so the first
            // post-resume render still sees an up-to-date frame.
            let safeIdx = min(max(0, chartZoomIdx), ChartZoom.stops.count - 1)
            let windowSec = ChartZoom.stops[safeIdx]
            scrollX = max(0, snapshot.lastElapsedSec - windowSec)
        }
        recomputeDecimatedFrame()
    }
}

/// Snapshot of every chart-coordinator field that gets persisted
/// alongside a session. Built on the main actor at save time
/// (rings are `@MainActor`-isolated, so the array copies happen
/// in `ChartCoordinator.buildSnapshot` rather than in the detached
/// save task) and handed across the actor boundary as a `Sendable`
/// value. Mirror-image of `ChartCoordinator.seedFromRestoredSession`'s
/// inputs.
///
/// Top-level (not nested inside `ChartCoordinator`) so it doesn't
/// inherit the coordinator's `@MainActor` isolation — it travels
/// to the detached save task, where the JSON encode actually runs.
struct ChartCoordinatorSnapshot: Sendable {
    let trainingSamples: [TrainingChartSample]
    let progressRateSamples: [ProgressRateSample]
    let arenaChartEvents: [ArenaChartEvent]
    let legalMassMaxAllTime: Double
    let lastElapsedSec: Double
}
