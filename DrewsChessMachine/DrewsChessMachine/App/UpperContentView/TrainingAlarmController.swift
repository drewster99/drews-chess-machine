import SwiftUI

/// Owns the in-app training-alarm subsystem, lifted out of `UpperContentView`.
///
/// Two raise paths feed the single banner:
///   - **Divergence detector** (`evaluate(from:)`): called every heartbeat tick
///     with the latest chart sample. Tracks consecutive warning / critical /
///     recovery streaks of (low policy entropy, high gradient norm) and
///     raises a `.warning` or `.critical` alarm once a streak threshold is hit;
///     auto-clears once a healthy-reading streak is reached — but only if the
///     banner currently shows one of *its own* titles (so it never wipes an
///     alarm raised by the legal-mass-collapse detector, which runs on its own
///     cadence and calls `raise(...)` directly).
///   - **Other detectors** (e.g. the legal-mass-collapse probe inside
///     `startRealTraining`): call `raise(severity:title:detail:)` directly.
///
/// `clear()` is the auto-clear / lifecycle-reset (logs `[ALARM] cleared`),
/// `dismiss()` is the user "I've seen it" gesture (logs `[ALARM] dismissed`,
/// also resets the divergence streaks so the alarm only re-raises on a *fresh*
/// deterioration), `resetStreaks()` zeroes the streak counters without touching
/// the banner (used by session-lifecycle resets that pair it with `clear()`),
/// and `silence()` mutes the periodic beep but leaves the banner up.
///
/// The constants below are the canonical alarm thresholds for the project;
/// `colorizedPanelBody` and the periodic `[ALARM]` log line in `startRealTraining`
/// reference `TrainingAlarmController.policyEntropyAlarmThreshold`.
@MainActor
@Observable
final class TrainingAlarmController {

    // MARK: - Thresholds (canonical alarm constants)

    /// ln(30) ≈ 3.4 nats in typical midgame; a fresh network starts at
    /// pEnt ≈ 1.9 nats empirically. A threshold of 1.0 flags genuine collapse
    /// (≈ 2.7 effective legal moves), leaving a ~0.9-nat margin below
    /// fresh-init baseline. Normal training concentrates the policy over time,
    /// so moderate decline from 1.9 is expected and healthy.
    nonisolated static let policyEntropyAlarmThreshold: Double = 1.0
    nonisolated static let divergenceAlarmGradNormWarningThreshold: Double = 50.0
    nonisolated static let divergenceAlarmGradNormCriticalThreshold: Double = 500.0
    /// Post-mask entropy critical floor: ≈ 1.6 effective legal moves.
    nonisolated static let divergenceAlarmEntropyCriticalThreshold: Double = 0.5
    nonisolated static let divergenceAlarmConsecutiveWarningSamples: Int = 3
    nonisolated static let divergenceAlarmConsecutiveCriticalSamples: Int = 2
    nonisolated static let divergenceAlarmRecoverySamples: Int = 10

    /// Alarm titles owned by `evaluate(from:)`. Anchored as named constants so
    /// the raise path and the ownership-scoped auto-clear check can't drift
    /// apart.
    nonisolated static let divergenceCriticalAlarmTitle = "Critical Training Divergence"
    nonisolated static let divergenceWarningAlarmTitle = "Training Divergence Warning"

    // MARK: - Observable state (read by the banner)

    private(set) var active: TrainingAlarm?
    private(set) var silenced = false

    // MARK: - Private detector / sound state

    private var divergenceWarningStreak = 0
    private var divergenceCriticalStreak = 0
    private var divergenceRecoveryStreak = 0
    private var alarmSoundTask: Task<Void, Never>?

    // MARK: - Divergence detector

    /// Evaluate the latest chart sample against the divergence thresholds,
    /// updating the streak counters and raising / auto-clearing the banner.
    func evaluate(from sample: TrainingChartSample) {
        evaluate(rollingPolicyEntropy: sample.rollingPolicyEntropy,
                 rollingGradNorm: sample.rollingGradNorm)
    }

    /// Core of `evaluate(from:)`, split out so tests can drive it with the two
    /// rolling-window values directly (no `TrainingChartSample` to construct).
    func evaluate(rollingPolicyEntropy entropy: Double?, rollingGradNorm gradNorm: Double?) {
        let warningOutOfLine =
            (entropy.map { $0 < Self.policyEntropyAlarmThreshold } ?? false)
            && (gradNorm.map { $0 > Self.divergenceAlarmGradNormWarningThreshold } ?? false)
        let criticalOutOfLine =
            (entropy.map { $0 < Self.divergenceAlarmEntropyCriticalThreshold } ?? false)
            || (gradNorm.map { $0 > Self.divergenceAlarmGradNormCriticalThreshold } ?? false)

        if criticalOutOfLine {
            divergenceCriticalStreak += 1
            divergenceWarningStreak = 0
            divergenceRecoveryStreak = 0
        } else if warningOutOfLine {
            divergenceWarningStreak += 1
            divergenceCriticalStreak = 0
            divergenceRecoveryStreak = 0
        } else {
            divergenceCriticalStreak = 0
            divergenceWarningStreak = 0
            divergenceRecoveryStreak += 1
        }

        if divergenceCriticalStreak >= Self.divergenceAlarmConsecutiveCriticalSamples {
            raise(
                severity: .critical,
                title: Self.divergenceCriticalAlarmTitle,
                detail: Self.alarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceWarningStreak >= Self.divergenceAlarmConsecutiveWarningSamples {
            raise(
                severity: .warning,
                title: Self.divergenceWarningAlarmTitle,
                detail: Self.alarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceRecoveryStreak >= Self.divergenceAlarmRecoverySamples {
            // Scope the auto-clear to alarms this evaluator actually raised.
            // Without this check, a healthy entropy / gNorm reading would wipe
            // out the banner from any other detector (e.g. the legal-mass
            // collapse detector that runs on a separate 15 s cadence) — the
            // user would see unrelated alarms appear and disappear as the
            // heartbeat races the 15 s probe cycle. Titles are the de-facto
            // ownership marker because every raise in this file uses a distinct
            // title string.
            let activeTitle = active?.title
            let isOurs = activeTitle == Self.divergenceCriticalAlarmTitle
                || activeTitle == Self.divergenceWarningAlarmTitle
            if isOurs {
                clear()
            }
        }
    }

    private static func alarmDetail(entropy: Double?, gradNorm: Double?) -> String {
        let entropyStr = entropy.map { String(format: "%.4f", $0) } ?? "--"
        let gradStr = gradNorm.map { String(format: "%.3f", $0) } ?? "--"
        return "policy entropy=\(entropyStr), gNorm=\(gradStr)"
    }

    // MARK: - Raise / clear / silence / dismiss

    /// Raise (or update) the banner. Used by `evaluate(from:)` and by other
    /// detectors (e.g. the legal-mass-collapse probe) directly.
    func raise(severity: TrainingAlarm.Severity, title: String, detail: String) {
        let next = TrainingAlarm(
            id: UUID(),
            severity: severity,
            title: title,
            detail: detail,
            raisedAt: Date()
        )
        let isNewAlarm = active == nil
        let titleOrSeverityChanged = active?.title != next.title
            || active?.severity != next.severity
        active = next
        // Log on first raise OR on title/severity change so the session log
        // captures every banner state the user could see. Periodic re-raises
        // with identical title+severity (and just updated numeric detail) don't
        // relog — those are already covered by the periodic [STATS] /
        // threshold-alarm log lines.
        if isNewAlarm || titleOrSeverityChanged {
            SessionLogger.shared.log("[ALARM] \(title): \(detail)")
        }
        startAlarmSoundLoopIfNeeded()
    }

    /// Auto-clear / lifecycle-reset path: clears the banner, unmutes, and
    /// cancels the beep loop. Does NOT touch the divergence streak counters
    /// (the recovery-streak auto-clear in `evaluate(from:)` deliberately leaves
    /// them alone).
    func clear() {
        if let prior = active {
            SessionLogger.shared.log("[ALARM] cleared: \(prior.title)")
        }
        active = nil
        silenced = false
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
    }

    func silence() {
        if let active {
            SessionLogger.shared.log("[ALARM] silenced: \(active.title)")
        }
        silenced = true
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
    }

    /// Clear the banner AND reset the divergence streak counters so the alarm
    /// only re-raises on a *fresh* deterioration from a healthy baseline.
    /// User-initiated "I've seen it, move on" gesture (banner Dismiss button).
    func dismiss() {
        if let active {
            SessionLogger.shared.log("[ALARM] dismissed: \(active.title)")
        }
        active = nil
        silenced = false
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
        resetStreaks()
    }

    /// Zero the divergence streak counters without touching the banner. Paired
    /// with `clear()` by session-lifecycle resets (start new session, promote,
    /// new game) that previously zeroed the `divergence*Streak` `@State`
    /// alongside `clearTrainingAlarm()`.
    func resetStreaks() {
        divergenceWarningStreak = 0
        divergenceCriticalStreak = 0
        divergenceRecoveryStreak = 0
    }

    // MARK: - Beep loop

    private func startAlarmSoundLoopIfNeeded() {
        guard active != nil, !silenced, alarmSoundTask == nil else { return }
        alarmSoundTask = Task {
            while !Task.isCancelled {
                await playAlarmBuzzBurst()
                do {
                    try await Task.sleep(for: .seconds(300))
                } catch {
                    return
                }
            }
        }
    }

    @MainActor
    private func playAlarmBuzzBurst() async {
        for _ in 0..<3 {
            if Task.isCancelled || active == nil || silenced { return }
            NSSound.beep()
            do {
                try await Task.sleep(for: .seconds(1.2))
            } catch {
                return
            }
        }
    }
}
