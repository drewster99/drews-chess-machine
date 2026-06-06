import Foundation

@MainActor
extension SessionController {
    /// One-time, in-memory back-fill of `TrainingChartSample.trainingStep`
    /// for samples collected before that field existed, so the combined
    /// Training-vs-Eval-Loss window can place the whole training-loss
    /// trajectory on its shared trainer-step axis.
    ///
    /// Anchors come from both Lichess probe overall series — each tick
    /// carries an absolute timestamp plus the authoritative
    /// `completedTrainSteps`, giving exact `(time, step)` ground-truth
    /// points — converted into the chart's elapsed-second frame via
    /// `chartElapsedAnchor`. A terminal anchor at the live
    /// `(elapsed, completedTrainSteps)` pins the region after the last
    /// tick to the true step total. See `TrainingStepBackfill` for the
    /// piecewise-linear interpolation (which stays correctly flat across
    /// arena pauses, unlike a naive even-spacing map).
    ///
    /// Idempotent in two senses, so it's safe to call on every fresh
    /// window open (the only caller): it never overwrites a sample that
    /// already has a step, and it fills `nil` steps for whatever session
    /// is currently loaded — so loading a different session mid-launch and
    /// reopening the window back-fills that session too. Filled samples
    /// persist on the next session save. With no anchors yet (very early,
    /// before the first probe tick and with no trainer) it leaves samples
    /// untouched so a later open retries. The O(count) scan runs only on
    /// this user-initiated action, not per frame.
    func backfillTrainingStepsIfNeeded() {
        guard let coord = chartCoordinator, coord.trainingRing.count > 0 else { return }

        let anchorDate = coord.chartElapsedAnchor
        var raw: [TrainingStepBackfill.Anchor] = []
        for series in [lichessProbeHistory.overallSeries, lichessProbeWideHistory.overallSeries] {
            for tick in series {
                guard let step = tick.trainingStep else { continue }
                raw.append(.init(
                    elapsedSec: tick.timestamp.timeIntervalSince(anchorDate),
                    step: step
                ))
            }
        }
        if let step = trainer?.completedTrainSteps {
            raw.append(.init(
                elapsedSec: max(0, Date().timeIntervalSince(anchorDate)),
                step: step
            ))
        }

        let anchors = TrainingStepBackfill.normalize(raw)
        guard !anchors.isEmpty else { return }  // retry on a later open

        var filled = 0
        let total = coord.trainingRing.count
        coord.trainingRing.transformEach { sample in
            guard sample.trainingStep == nil,
                  let step = TrainingStepBackfill.interpolatedStep(
                    elapsedSec: sample.elapsedSec, anchors: anchors
                  )
            else { return sample }
            filled += 1
            var s = sample
            s.trainingStep = step
            return s
        }
        guard filled > 0 else { return }
        SessionLogger.shared.log(
            "[BACKFILL] training-step back-fill: filled=\(filled)/\(total) anchors=\(anchors.count)"
        )
    }
}
