import Foundation

/// Post-arena telemetry emission, lifted verbatim out of `UpperContentView`.
///
/// These three functions only consume their arguments plus module-global
/// sinks (`SessionLogger.shared`, `DiagSampler`, the free function
/// `validateTournamentRecords`) — they touch no SwiftUI state — so they
/// live here rather than bloating the view. They are called from every
/// `runArenaParallel` exit leg (success, cancellation, thrown error) so a
/// mid-tournament failure still surfaces "how much concurrency did we get
/// before it died" and "were the captured games internally consistent".
enum ArenaTelemetryFormatter {

    /// Drain the per-batcher batch-size histograms and run the
    /// post-arena game-validity sweep, emitting one log line per
    /// batcher and one for the validation outcome. Called on every
    /// `runArenaParallel` exit leg — success, cancellation, AND
    /// thrown errors — so a mid-tournament throw still surfaces
    /// "how much concurrency did we get before it died" and
    /// "were the captured games internally consistent" rather
    /// than silently losing that diagnostic.
    ///
    /// The validity sweep is skipped under cancellation because
    /// partial records may be incomplete (a slot mid-game when
    /// cancelled didn't append a final move record); under errors
    /// we still run it because partial-but-completed games are
    /// well-formed and worth checking — if a slot threw mid-game,
    /// only its own record is missing, not the others'.
    ///
    /// `context` annotates the log line for non-success paths
    /// (e.g. "after error") so a log reader can tell at a glance
    /// the run didn't complete normally.
    static func emitPostRunTelemetry(
        candidateBatcher: BatchedMoveEvaluationSource,
        championBatcher: BatchedMoveEvaluationSource,
        gpuTimer: ArenaGpuTimer,
        recordsBox: TournamentRecordsBox,
        wasCancelled: Bool,
        context: String?,
        arenaStartTime: Date,
        trainingBox: TrainingLiveStatsBox?
    ) async {
        let suffix = context.map { " (\($0))" } ?? ""
        let candidateBatchStats = await candidateBatcher.snapshotBatchSizeStats()
        let championBatchStats = await championBatcher.snapshotBatchSizeStats()
        let candidateTimingStats = await candidateBatcher.snapshotBatchTimingStats()
        let championTimingStats = await championBatcher.snapshotBatchTimingStats()
        SessionLogger.shared.log(
            "[ARENA] batch sizes  candidate\(suffix): \(candidateBatchStats.formatLogLine())"
        )
        SessionLogger.shared.log(
            "[ARENA] batch sizes  champion \(suffix): \(championBatchStats.formatLogLine())"
        )

        // Joint GPU-wall timing: a single number for "wall time during
        // which AT LEAST one side was on the GPU." Emitted before the
        // per-side timing lines because it's the easier-to-read summary
        // — `nonGpu = wall - gpu` is exactly the time the GPU was idle
        // (CPU per-ply work, scheduler gaps, continuation-resume
        // backpressure), with no per-side double-counting. The
        // per-side lines that follow remain useful for spotting
        // candidate-vs-champion balance (run-time difference, wait
        // asymmetry).
        let arenaTotalSec = max(0, Date().timeIntervalSince(arenaStartTime))
        let gpuBusySec = gpuTimer.totalBusyMs() / 1000.0
        let nonGpuSec = max(0, arenaTotalSec - gpuBusySec)
        let gpuUtilPct = arenaTotalSec > 0 ? (gpuBusySec / arenaTotalSec) * 100.0 : 0.0
        SessionLogger.shared.log(String(
            format: "[ARENA] timing joint%@: wall=%.1fs gpu=%.1fs nonGpu=%.1fs (gpu_util=%.1f%%)",
            suffix,
            arenaTotalSec,
            gpuBusySec,
            nonGpuSec,
            gpuUtilPct
        ))

        // Per-side wall-clock breakdown: wait/run vs the total arena
        // duration. `other` is the residual that's neither wait nor
        // run — time this batcher had nothing to do (the OTHER batcher
        // was busy, or slots were doing CPU-side work between
        // submissions). Each batcher is independent so the two `other`
        // numbers don't sum to anything meaningful; they're each "how
        // much of arena wall time was this batcher idle". Use the
        // joint line above for true GPU saturation.
        emitTimingLine(label: "candidate", suffix: suffix, totalSec: arenaTotalSec, stats: candidateTimingStats)
        emitTimingLine(label: "champion ", suffix: suffix, totalSec: arenaTotalSec, stats: championTimingStats)

        // Fire-reason histogram: answers "is the coalescing-window
        // timer actually firing the barrier, or are we just hitting
        // count-met every time?" without forcing a reader to infer
        // it from the size histogram. Always emits all five reasons
        // so the cand/champ lines line up visually for asymmetry
        // checks. Healthy steady-state: `full` dominates, `timer`
        // small, `drain` small (only on tournament drain), `threshold`
        // and `refill` small or zero.
        SessionLogger.shared.log(
            "[ARENA] fire reasons candidate\(suffix): \(candidateBatchStats.formatFireReasonsLine())"
        )
        SessionLogger.shared.log(
            "[ARENA] fire reasons champion \(suffix): \(championBatchStats.formatFireReasonsLine())"
        )

        // Pre/post `expectedSlotCount` drift counters: how many fires
        // saw the slot-count change during the GPU await, plus the
        // largest such delta. Steady-state non-zero is healthy (games
        // ending mid-fire is the dominant cause). Asymmetry across
        // sides or a sustained `maxDelta > ~5` would point at the
        // coordination paths between the harvest loop and the batcher
        // actor.
        SessionLogger.shared.log(
            "[ARENA] expected-drift candidate\(suffix): \(candidateBatchStats.formatExpectedDriftLine())"
        )
        SessionLogger.shared.log(
            "[ARENA] expected-drift champion \(suffix): \(championBatchStats.formatExpectedDriftLine())"
        )

        if wasCancelled {
            // Partial records under cancellation can be
            // structurally incomplete — skip rather than emit
            // misleading "validation FAILED" lines.
            SessionLogger.shared.log(
                "[ARENA] validation skipped\(suffix): tournament was cancelled mid-run"
            )
            emitPostStatsLine(trainingBox: trainingBox, suffix: suffix)
            return
        }

        let report = validateTournamentRecords(recordsBox.snapshot())
        if report.passed {
            SessionLogger.shared.log(
                "[ARENA] validation passed\(suffix): \(report.gamesChecked) games, "
                + "\(report.totalMovesChecked) moves all legal in their position contexts"
            )
        } else {
            let detail = report.failureDescription ?? "(no detail)"
            SessionLogger.shared.log(
                "[ARENA] validation FAILED\(suffix): \(detail)"
            )
            trainingBox?.recordError("Arena validation failed: \(detail)")
        }

        emitPostStatsLine(trainingBox: trainingBox, suffix: suffix)
    }

    /// One-line stats snapshot taken at the moment an arena ends.
    /// Captures rolling per-step trainer timing means + RSS + VM
    /// region count, so a reader scanning the session log can see
    /// whether each arena boundary corresponds to a step-up in
    /// per-step `gpu`/`step` time, RSS, or IOAccelerator-tagged
    /// VM region count. Emitted from every `emitPostRunTelemetry`
    /// exit path (success, cancel, error) so a slowdown investigation
    /// has a per-arena trace independent of the 60 s [STATS] cadence.
    static func emitPostStatsLine(
        trainingBox: TrainingLiveStatsBox?,
        suffix: String
    ) {
        let trainingSnap = trainingBox?.snapshot()
        let timingStr: String
        if let snap = trainingSnap, let stepMs = snap.recentStepMs {
            timingStr = String(
                format: "step=%.1f gpu=%.1f prep=%.2f read=%.2f wait=%.2f n=%d",
                stepMs,
                snap.recentGpuRunMs ?? 0,
                snap.recentDataPrepMs ?? 0,
                snap.recentReadbackMs ?? 0,
                snap.recentQueueWaitMs ?? 0,
                snap.recentTimingSamples
            )
        } else {
            timingStr = "n=0"
        }
        let rssBytes = DiagSampler.currentResidentBytes()
        let memStr = String(
            format: "rss=%.2fGB",
            Double(rssBytes) / 1024.0 / 1024.0 / 1024.0
        )
        let vm = DiagSampler.currentVMRegionCount()
        let vmStr = String(
            format: "total=%u ioAccel=%u",
            vm.total, vm.ioAccelerator
        )
        SessionLogger.shared.log(
            "[STATS-ARENA-END]\(suffix) timing=(\(timingStr)) mem=(\(memStr)) vm=(\(vmStr))"
        )
    }

    /// Emit one `[ARENA] timing` line for a single batcher. Wall /
    /// wait / run / other render in seconds for readability (arenas
    /// are typically tens of seconds, ms-formatting buries the
    /// signal under trailing zeros). Per-batch means stay in ms
    /// because they're sub-second by nature.
    static func emitTimingLine(
        label: String,
        suffix: String,
        totalSec: Double,
        stats: BatchTimingStats
    ) {
        let waitSec = stats.totalWaitMs / 1000.0
        let runSec = stats.totalRunMs / 1000.0
        let otherSec = max(0, totalSec - waitSec - runSec)
        SessionLogger.shared.log(String(
            format: "[ARENA] timing %@%@: total=%.1fs wait=%.1fs run=%.1fs other=%.1fs (batches=%d meanWait=%.2fms meanRun=%.2fms)",
            label,
            suffix,
            totalSec,
            waitSec,
            runSec,
            otherSec,
            stats.totalBatches,
            stats.meanWaitMs,
            stats.meanRunMs
        ))
    }
}
