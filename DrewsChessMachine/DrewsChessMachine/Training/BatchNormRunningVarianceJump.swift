import Foundation

/// Rule 14 (`bn_running_variance_jump`), the pure reading: one live read's
/// running-variance ratios against the reads of the previous
/// `batchNormRunningVarianceJumpLookbackSteps` trainer steps. The evaluator
/// keeps the history and turns a reading into an assessment; nothing here
/// holds state. Design and evidence:
/// `documentation/plans-active/BN_RUNNING_VARIANCE_CHANGE_ALARM_PLAN.md`.
///
/// Why a change rule beside rule 8's level (1,000×): B-silu's channel 76 at
/// `blocks.2.bn1` went from 0.02× its site median at one live read to 154×
/// within 800 trainer steps, then 45,186× once the network broke. Rule 8 sees
/// it only when it is already past 1,000×; a channel that sits at 50× for
/// thousands of steps (clip1's channel 31, AgG3's channel 88) is harmless.
/// So the rule compares each channel with its **own** lowest ratio over the
/// lookback — not only the previous read, so a rise spread over several reads
/// and sparse or irregular cadences (GUI pauses, offline time-cadence logs)
/// give the same answer.
///
/// Two arms:
/// - **jump:** a channel now at ≥ `batchNormRunningVarianceJumpOutlierRatio`
///   that is ≥ `batchNormRunningVarianceJumpRiseFactor` × its baseline —
///   critical when it is at ≥ `batchNormRunningVarianceJumpCriticalRatio`;
/// - **outlier count:** the number of channels at ≥ the outlier ratio reached
///   `max(2 × baseline, baseline + 5)`, the baseline being the lowest count
///   in the lookback.
///
/// Offline (`--replay-health-log`) a live line names only its largest
/// channel, so a past read bounds every other channel by its largest ratio:
/// the baseline is then an upper bound, the jump arm a lower bound on the
/// app's (it never raises where the app would not), and only the current
/// line's largest channel can jump.
enum BatchNormRunningVarianceJump {

    /// One judged live read, kept for the lookback.
    struct HistoryEntry: Sendable, Equatable {
        let trainerStep: Int
        let channels: LayerHealthDigest.RunningVarianceChannels
    }

    /// One channel at or above the outlier ratio, with its baseline.
    struct JumpedChannel: Sendable, Equatable {
        let site: String
        let channel: Int
        /// The channel's lowest ratio over the lookback's reads, or an upper
        /// bound of it (`baselineIsExact` false, offline).
        let baseline: Double
        /// False when some read of the lookback did not name this channel
        /// (offline: it bounded the channel by its largest ratio).
        let baselineIsExact: Bool
        let ratio: Double

        /// `ratio / baseline`; infinite for a zero baseline.
        var riseFactor: Double {
            baseline > 0 ? ratio / baseline : .infinity
        }
    }

    struct Reading: Sendable, Equatable {
        /// Channels meeting the jump condition, ratio descending.
        let jumped: [JumpedChannel]
        /// The largest rise of any channel at or above the outlier ratio,
        /// whether or not it jumped (the `[HEALTH] check` line's
        /// `rvRiseMax=`); nil when no such channel has a baseline.
        let largestRise: JumpedChannel?
        /// The current read's outlier count; nil offline without the field.
        let outlierCount: Int?
        /// The lowest outlier count over the lookback's reads that carry one;
        /// nil when none does.
        let outlierBaseline: Int?

        /// The jump arm holds.
        var jumpHolds: Bool { !jumped.isEmpty }

        /// The jump arm holds at critical: a jumped channel is at or above
        /// the critical ratio.
        var jumpIsCritical: Bool {
            jumped.contains { $0.ratio >= TrainingHealthThresholds.batchNormRunningVarianceJumpCriticalRatio }
        }

        /// The outlier-count arm holds.
        var outlierCountRiseHolds: Bool {
            guard let outlierCount, let outlierBaseline else { return false }
            return BatchNormRunningVarianceJump.outlierCountRose(from: outlierBaseline, to: outlierCount)
        }
    }

    /// The outlier-count arm's condition: `count ≥ max(2 × baseline,
    /// baseline + 5)`. 1 → 2, 2 → 4, 3 → 6, 4 → 8 stay quiet; 0 → 5 raises.
    static func outlierCountRose(from baseline: Int, to count: Int) -> Bool {
        count >= max(
            TrainingHealthThresholds.batchNormRunningVarianceOutlierCountRiseFactor * baseline,
            baseline + TrainingHealthThresholds.batchNormRunningVarianceOutlierCountMinimumRise)
    }

    /// The jump arm's condition for one channel.
    static func channelJumped(ratio: Double, baseline: Double) -> Bool {
        ratio >= TrainingHealthThresholds.batchNormRunningVarianceJumpOutlierRatio
            && ratio >= TrainingHealthThresholds.batchNormRunningVarianceJumpRiseFactor * baseline
    }

    /// The history entries in the lookback of a read at `trainerStep`:
    /// trainer steps `[trainerStep − lookback, trainerStep)` (inclusive lower
    /// bound, as `TrainingHealthReference.make`, so consecutive 1,000-step
    /// checkpoints are exactly one lookback apart).
    static func lookback(_ history: [HistoryEntry], trainerStep: Int) -> [HistoryEntry] {
        let lowerBound = trainerStep - TrainingHealthThresholds.batchNormRunningVarianceJumpLookbackSteps
        return history.filter { $0.trainerStep >= lowerBound && $0.trainerStep < trainerStep }
    }

    /// The reading of `current` (read at `trainerStep`) against `history`;
    /// nil when no read is in the lookback (the rule then holds:
    /// `baseline=none`). Within one monitor every read has the same site
    /// layout; a history entry whose layout differs from the current one is
    /// a code bug and traps naming the sites.
    static func read(
        _ current: LayerHealthDigest.RunningVarianceChannels,
        history: [HistoryEntry],
        trainerStep: Int
    ) -> Reading? {
        let entries = lookback(history, trainerStep: trainerStep)
        guard !entries.isEmpty else { return nil }
        if case .everyChannel(let profile) = current.coverage {
            for entry in entries {
                guard case .everyChannel(let past) = entry.channels.coverage else { continue }
                guard past.hasSameLayout(as: profile) else {
                    preconditionFailure(
                        "bn_running_variance_jump: the BN site layout changed within one monitor "
                        + "(trainer step \(entry.trainerStep): \(past.sites.map { "\($0.site)(\($0.channelCount))" }); "
                        + "trainer step \(trainerStep): \(profile.sites.map { "\($0.site)(\($0.channelCount))" }))")
                }
            }
        }

        var jumped: [JumpedChannel] = []
        var largestRise: JumpedChannel?
        func consider(site: String, siteIndex: Int?, channel: Int, ratio: Double) {
            guard ratio >= TrainingHealthThresholds.batchNormRunningVarianceJumpOutlierRatio,
                  let bound = baseline(of: entries, site: site, siteIndex: siteIndex, channel: channel) else { return }
            let candidate = JumpedChannel(
                site: site, channel: channel, baseline: bound.value, baselineIsExact: bound.isExact, ratio: ratio)
            if largestRise.map({ candidate.riseFactor > $0.riseFactor }) ?? true {
                largestRise = candidate
            }
            if channelJumped(ratio: ratio, baseline: bound.value) {
                jumped.append(candidate)
            }
        }
        switch current.coverage {
        case .everyChannel(let profile):
            for (siteIndex, site) in profile.sites.enumerated() {
                guard let ratios = site.ratios else { continue }
                for (channel, ratio) in ratios.enumerated() {
                    guard let ratio else { continue }
                    consider(site: site.site, siteIndex: siteIndex, channel: channel, ratio: ratio)
                }
            }
        case .largestOnly(let site, let channel, let ratio):
            consider(site: site, siteIndex: nil, channel: channel, ratio: ratio)
        }
        jumped.sort { $0.ratio != $1.ratio ? $0.ratio > $1.ratio : ($0.site, $0.channel) < ($1.site, $1.channel) }

        let outlierBaseline = entries.compactMap(\.channels.outlierCount).min()
        return Reading(
            jumped: jumped, largestRise: largestRise,
            outlierCount: current.outlierCount, outlierBaseline: outlierBaseline)
    }

    /// The channel's lowest ratio over `entries`: exact where an entry holds
    /// the channel's own ratio, the entry's largest ratio (an upper bound)
    /// where it names another channel. An entry with no value for the channel
    /// (a non-finite variance, a site whose median is not positive) is left
    /// out; nil when no entry bounds the channel.
    private static func baseline(
        of entries: [HistoryEntry],
        site: String,
        siteIndex: Int?,
        channel: Int
    ) -> (value: Double, isExact: Bool)? {
        var lowest: Double?
        var isExact = true
        for entry in entries {
            let bound: Double?
            switch entry.channels.coverage {
            case .everyChannel(let profile):
                let pastSite: BatchNormRunningVarianceProfile.Site?
                if let siteIndex {
                    pastSite = profile.sites[siteIndex]
                } else {
                    pastSite = profile.sites.first { $0.site == site }
                }
                if let ratios = pastSite?.ratios, channel < ratios.count {
                    bound = ratios[channel]
                } else {
                    bound = nil
                }
            case .largestOnly(let pastSite, let pastChannel, let pastRatio):
                if pastSite == site && pastChannel == channel {
                    bound = pastRatio
                } else {
                    bound = pastRatio
                    isExact = false
                }
            }
            guard let bound else {
                // The read has no value for this channel; the minimum over the
                // remaining reads is then an upper bound of the true one.
                isExact = false
                continue
            }
            lowest = min(lowest ?? bound, bound)
        }
        return lowest.map { ($0, isExact) }
    }
}
