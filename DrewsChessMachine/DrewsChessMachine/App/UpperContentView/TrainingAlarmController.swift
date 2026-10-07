import SwiftUI

/// Owns the in-app training-alarm subsystem, lifted out of `UpperContentView`.
///
/// Two raise paths feed the single banner:
///   - **`evaluate(from:)` family** — called every heartbeat tick with the
///     latest chart sample; runs three streak-based detectors against it:
///     (1) **divergence** — low policy entropy + high gradient norm;
///     (2) **value-head saturation** — `vAbs → 1` (the W/D/L head calling
///     almost every position a clean win/loss); (3) **value-head draw
///     collapse** — `pD → 1` (the head calling everything a draw — the
///     failure the W/D/L representation was adopted to escape). Each tracks
///     consecutive warning / critical / recovery streaks and raises a
///     `.warning` or `.critical` alarm once a streak threshold is hit;
///     auto-clears once a healthy-reading streak is reached — but only if the
///     banner currently shows one of *that detector's own* titles (so a
///     recovery in one detector never wipes an alarm another detector — or
///     the legal-mass-collapse probe below — raised).
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
/// reference `TrainingHealthThresholds.policyEntropyAlarm`.
@MainActor
@Observable
final class TrainingAlarmController {

    // MARK: - Thresholds (canonical alarm constants)

    /// ln(30) ≈ 3.4 nats in typical midgame; a fresh network starts at
    /// pEnt ≈ 1.9 nats empirically. A threshold of 1.0 flags genuine collapse
    /// (≈ 2.7 effective legal moves), leaving a ~0.9-nat margin below
    /// fresh-init baseline. Normal training concentrates the policy over time,
    /// so moderate decline from 1.9 is expected and healthy.
    // The levels themselves (entropy, gradient-norm, vAbs and pD
    // thresholds) live in `TrainingHealthThresholds` and are applied by
    // `TrainingHealthDetectorConditions`, shared with the training-health
    // evaluator's rules 10–13 (OD-9). This controller keeps the banner's
    // heartbeat streaks and titles.
    nonisolated static let divergenceAlarmConsecutiveWarningSamples: Int = 3
    nonisolated static let divergenceAlarmConsecutiveCriticalSamples: Int = 2
    nonisolated static let divergenceAlarmRecoverySamples: Int = 10

    /// Value-head over-confidence thresholds — `vAbs = mean(|v|)` for
    /// the W/D/L head's derived scalar `v = p_win − p_loss ∈ [-1, +1]`
    /// (post-WDL switch: no tanh). `vAbs → 1` means the head is
    /// confidently classifying nearly every batch position as a clean
    /// win or loss — implausible for real chess (most positions are
    /// near-balanced), so a sustained high value is a "something is
    /// wrong" signal worth surfacing. (The *opposite* failure — the
    /// head calling everything a draw, `vAbs → 0` / `pD → 1` — is the
    /// collapse the WDL representation was adopted to escape; it shows
    /// up directly as `pD=` on the `[STATS]` line rather than through
    /// this alarm.) Thresholds kept at the prior `(0.97, 0.995)`
    /// levels; they no longer index a tanh-gradient and are just
    /// "very confident" / "near-degenerate" markers.

    /// Streak counts for the value-head saturation detector. Match the
    /// divergence detector's `(warning: 3, critical: 2, recovery: 10)`
    /// shape — saturation creeps in over many heartbeats so a 2-/3-sample
    /// streak is enough confirmation that it's not a single-batch blip.
    nonisolated static let valueAbsMeanSaturationConsecutiveWarningSamples: Int = 3
    nonisolated static let valueAbsMeanSaturationConsecutiveCriticalSamples: Int = 2
    nonisolated static let valueAbsMeanSaturationRecoverySamples: Int = 10

    /// Value-head **draw-collapse** thresholds — `pD = mean(p_draw)`,
    /// the batch-mean of the W/D/L head's softmax draw probability.
    /// This is the failure mode the W/D/L representation was adopted to
    /// escape (and the one the old scalar `tanh` head fell into on a
    /// draw-heavy buffer): the head learns "every position is a draw"
    /// (`pD → 1`, `p_win ≈ p_loss ≈ 0`, `vAbs → 0`) and stops supplying
    /// a useful policy-gradient baseline / value signal. A fresh head
    /// starts at `pD = 0.75` (the `[0, ln 6, 0]` bias init) and healthy
    /// training pulls it *down* as the head learns to call decisive
    /// games; a sustained `pD` well *above* the fresh-init level is the
    /// regression-toward-collapse signal. Warning at 0.92 (comfortably
    /// above 0.75 and any plausible healthy value on a draw-heavy
    /// buffer), critical at 0.97 (basically degenerate). Distinct from
    /// `valueAbsMeanSaturation` above, which catches the *opposite*
    /// extreme (`vAbs → 1`, over-confident win/loss everywhere).
    nonisolated static let valueDrawCollapseConsecutiveWarningSamples: Int = 3
    nonisolated static let valueDrawCollapseConsecutiveCriticalSamples: Int = 2
    nonisolated static let valueDrawCollapseRecoverySamples: Int = 10

    /// Alarm titles owned by `evaluate(from:)`. Anchored as named constants so
    /// the raise path and the ownership-scoped auto-clear check can't drift
    /// apart.
    nonisolated static let divergenceCriticalAlarmTitle = "Critical Training Divergence"
    nonisolated static let divergenceWarningAlarmTitle = "Training Divergence Warning"
    nonisolated static let valueAbsMeanSaturationCriticalAlarmTitle = "Critical Value-Head Saturation"
    nonisolated static let valueAbsMeanSaturationWarningAlarmTitle = "Value-Head Saturation Warning"
    nonisolated static let valueDrawCollapseCriticalAlarmTitle = "Critical Value-Head Draw Collapse"
    nonisolated static let valueDrawCollapseWarningAlarmTitle = "Value-Head Draw Collapse Warning"

    // MARK: - Observable state (read by the banner)

    private(set) var active: TrainingAlarm?
    private(set) var silenced = false

    /// The training-health monitor's active alarms, mirrored for the alarm
    /// list (`TrainingHealthAlarmList`). The monitor is the source; this is
    /// replaced wholesale on every evaluation's main-actor hop
    /// (`refreshHealth(from:)`), never edited.
    private(set) var healthAlarms: [TrainingHealthActiveAlarm] = []

    /// The rule whose health stop suspended training, for the list's header
    /// row; nil when training is not suspended by a health alarm.
    private(set) var healthSuspendedRule: TrainingHealthRule?

    /// True from Stop (`endHealthRun()`) until the next start
    /// (`beginHealthRun()`): `healthAlarms` then describe a run that has
    /// ended. Nothing evaluates after Stop, so they can no longer clear by
    /// recovery; they stay listed for review but never sound — otherwise
    /// Stop's `clear()` (which un-silences) would restart the beep loop for
    /// as long as the app sits stopped.
    private(set) var healthAlarmsAreFromAnEndedRun = false

    // MARK: - Private detector / sound state

    private var divergenceWarningStreak = 0
    private var divergenceCriticalStreak = 0
    private var divergenceRecoveryStreak = 0
    private var valueAbsMeanSaturationWarningStreak = 0
    private var valueAbsMeanSaturationCriticalStreak = 0
    private var valueAbsMeanSaturationRecoveryStreak = 0
    private var valueDrawCollapseWarningStreak = 0
    private var valueDrawCollapseCriticalStreak = 0
    private var valueDrawCollapseRecoveryStreak = 0
    private var alarmSoundTask: Task<Void, Never>?

    // MARK: - Divergence detector

    /// Evaluate the latest chart sample against the divergence thresholds,
    /// updating the streak counters and raising / auto-clearing the banner.
    func evaluate(from sample: TrainingChartSample) {
        evaluate(rollingPolicyEntropy: sample.rollingPolicyEntropy,
                 rollingGradNorm: sample.rollingGradNorm,
                 rollingValueAbsMean: sample.rollingValueAbsMean,
                 rollingValueProbDraw: sample.rollingValueProbDraw)
    }

    /// Core of `evaluate(from:)`, split out so tests can drive it with the
    /// rolling-window values directly (no `TrainingChartSample` to construct).
    func evaluate(rollingPolicyEntropy entropy: Double?,
                  rollingGradNorm gradNorm: Double?,
                  rollingValueAbsMean valueAbsMean: Double? = nil,
                  rollingValueProbDraw valueProbDraw: Double? = nil) {
        evaluateDivergence(entropy: entropy, gradNorm: gradNorm)
        evaluateValueAbsMeanSaturation(valueAbsMean: valueAbsMean)
        evaluateValueDrawCollapse(valueProbDraw: valueProbDraw)
    }

    private func evaluateDivergence(entropy: Double?, gradNorm: Double?) {
        let level = TrainingHealthDetectorConditions.divergenceLevel(entropy: entropy, gradientNorm: gradNorm)
        let warningOutOfLine = level == .warning
        let criticalOutOfLine = level == .critical

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
                detail: Self.divergenceAlarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceWarningStreak >= Self.divergenceAlarmConsecutiveWarningSamples {
            raise(
                severity: .warning,
                title: Self.divergenceWarningAlarmTitle,
                detail: Self.divergenceAlarmDetail(entropy: entropy, gradNorm: gradNorm)
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

    /// Value-head saturation detector. Symmetric to `evaluateDivergence` but
    /// keyed off `rollingValueAbsMean` — the rolling mean of `|v|` over batch
    /// outputs. A `nil` value (no chart sample with `vAbs` data yet) resets
    /// the streaks defensively so a sequence of nils never auto-clears a
    /// previously-raised banner via the recovery path. Ownership-scoped
    /// auto-clear uses the saturation-specific titles so it never wipes a
    /// divergence banner or a legal-mass-collapse banner.
    private func evaluateValueAbsMeanSaturation(valueAbsMean: Double?) {
        guard let vAbs = valueAbsMean else {
            valueAbsMeanSaturationWarningStreak = 0
            valueAbsMeanSaturationCriticalStreak = 0
            valueAbsMeanSaturationRecoveryStreak = 0
            return
        }
        let level = TrainingHealthDetectorConditions.valueSaturationLevel(valueAbsMean: vAbs)
        let critical = level == .critical
        let warning = level != nil
        if critical {
            valueAbsMeanSaturationCriticalStreak += 1
            valueAbsMeanSaturationWarningStreak = 0
            valueAbsMeanSaturationRecoveryStreak = 0
        } else if warning {
            valueAbsMeanSaturationWarningStreak += 1
            valueAbsMeanSaturationCriticalStreak = 0
            valueAbsMeanSaturationRecoveryStreak = 0
        } else {
            valueAbsMeanSaturationCriticalStreak = 0
            valueAbsMeanSaturationWarningStreak = 0
            valueAbsMeanSaturationRecoveryStreak += 1
        }

        if valueAbsMeanSaturationCriticalStreak >= Self.valueAbsMeanSaturationConsecutiveCriticalSamples {
            raise(
                severity: .critical,
                title: Self.valueAbsMeanSaturationCriticalAlarmTitle,
                detail: Self.valueAbsMeanSaturationAlarmDetail(valueAbsMean: vAbs)
            )
        } else if valueAbsMeanSaturationWarningStreak >= Self.valueAbsMeanSaturationConsecutiveWarningSamples {
            raise(
                severity: .warning,
                title: Self.valueAbsMeanSaturationWarningAlarmTitle,
                detail: Self.valueAbsMeanSaturationAlarmDetail(valueAbsMean: vAbs)
            )
        } else if valueAbsMeanSaturationRecoveryStreak >= Self.valueAbsMeanSaturationRecoverySamples {
            let activeTitle = active?.title
            let isOurs = activeTitle == Self.valueAbsMeanSaturationCriticalAlarmTitle
                || activeTitle == Self.valueAbsMeanSaturationWarningAlarmTitle
            if isOurs {
                clear()
            }
        }
    }

    /// Value-head **draw-collapse** detector. Symmetric to the saturation
    /// detector but keyed off `rollingValueProbDraw` (high = bad — the
    /// head is heading toward "everything is a draw"). A `nil` value
    /// resets the streaks defensively so a run of nils never auto-clears
    /// a previously-raised banner. Ownership-scoped auto-clear uses the
    /// draw-collapse titles so it never wipes a different detector's banner.
    private func evaluateValueDrawCollapse(valueProbDraw: Double?) {
        guard let pD = valueProbDraw else {
            valueDrawCollapseWarningStreak = 0
            valueDrawCollapseCriticalStreak = 0
            valueDrawCollapseRecoveryStreak = 0
            return
        }
        let level = TrainingHealthDetectorConditions.valueDrawSaturationLevel(valueProbDraw: pD)
        let critical = level == .critical
        let warning = level != nil
        if critical {
            valueDrawCollapseCriticalStreak += 1
            valueDrawCollapseWarningStreak = 0
            valueDrawCollapseRecoveryStreak = 0
        } else if warning {
            valueDrawCollapseWarningStreak += 1
            valueDrawCollapseCriticalStreak = 0
            valueDrawCollapseRecoveryStreak = 0
        } else {
            valueDrawCollapseCriticalStreak = 0
            valueDrawCollapseWarningStreak = 0
            valueDrawCollapseRecoveryStreak += 1
        }

        if valueDrawCollapseCriticalStreak >= Self.valueDrawCollapseConsecutiveCriticalSamples {
            raise(
                severity: .critical,
                title: Self.valueDrawCollapseCriticalAlarmTitle,
                detail: Self.valueDrawCollapseAlarmDetail(valueProbDraw: pD)
            )
        } else if valueDrawCollapseWarningStreak >= Self.valueDrawCollapseConsecutiveWarningSamples {
            raise(
                severity: .warning,
                title: Self.valueDrawCollapseWarningAlarmTitle,
                detail: Self.valueDrawCollapseAlarmDetail(valueProbDraw: pD)
            )
        } else if valueDrawCollapseRecoveryStreak >= Self.valueDrawCollapseRecoverySamples {
            let activeTitle = active?.title
            let isOurs = activeTitle == Self.valueDrawCollapseCriticalAlarmTitle
                || activeTitle == Self.valueDrawCollapseWarningAlarmTitle
            if isOurs {
                clear()
            }
        }
    }

    private static func divergenceAlarmDetail(entropy: Double?, gradNorm: Double?) -> String {
        let entropyStr = entropy.map { String(format: "%.4f", $0) } ?? "--"
        let gradStr = gradNorm.map { String(format: "%.3f", $0) } ?? "--"
        return "policy entropy=\(entropyStr), gNorm=\(gradStr)"
    }

    private static func valueAbsMeanSaturationAlarmDetail(valueAbsMean: Double) -> String {
        String(format: "vAbs=%.4f — W/D/L value head calling nearly every position a clean win/loss (p_win − p_loss ≈ ±1)",
               valueAbsMean)
    }

    private static func valueDrawCollapseAlarmDetail(valueProbDraw: Double) -> String {
        String(format: "pD=%.4f — W/D/L value head collapsing toward \"everything is a draw\" (the failure the W/D/L head was meant to escape; try a small draw_penalty / value_label_smoothing_epsilon)",
               valueProbDraw)
    }

    // MARK: - Raise / clear / silence / dismiss

    /// Raise (or update) the banner. Used by `evaluate(from:)` and by other
    /// detectors (e.g. the legal-mass-collapse probe) directly.
    func raise(severity: TrainingAlarm.Severity, title: String, detail: String) {
        defer { reconcileSound() }
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
    }

    /// Auto-clear / lifecycle-reset path: clears the banner, unmutes, and
    /// cancels the beep loop. Does NOT touch the divergence streak counters
    /// (the recovery-streak auto-clear in `evaluate(from:)` deliberately leaves
    /// them alone), nor the health alarms, which clear only by recovery: the
    /// loop restarts at once if a critical one remains.
    func clear() {
        if let prior = active {
            SessionLogger.shared.log("[ALARM] cleared: \(prior.title)")
        }
        active = nil
        silenced = false
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
        reconcileSound()
    }

    /// Mute the beep for whatever is sounding (the banner, a critical health
    /// alarm, or both). Reachable from the banner and from the health list.
    func silence() {
        if let active {
            SessionLogger.shared.log("[ALARM] silenced: \(active.title)")
        }
        if healthAlarms.contains(where: { $0.severity == .critical }) {
            SessionLogger.shared.log("[ALARM] silenced: training health alarms "
                + healthAlarms.map { "\($0.rule.rawValue):\($0.severity.rawValue)" }.joined(separator: ","))
        }
        silenced = true
        reconcileSound()
    }

    // MARK: - Training-health alarms

    /// Mirror `monitor`'s current active set (D2: re-read on every hop, so
    /// hops arriving out of order never show an older state).
    func refreshHealth(from monitor: TrainingHealthMonitor) {
        refreshHealth(monitor.activeAlarmsSnapshot())
    }

    /// Core of `refreshHealth(from:)`, for tests. A silence covers what was
    /// sounding when it was given: a rule that becomes critical now (newly
    /// raised, or escalated from a warning) un-silences, so it beeps (OD-8).
    func refreshHealth(_ alarms: [TrainingHealthActiveAlarm]) {
        let criticalBefore = Set(healthAlarms.filter { $0.severity == .critical }.map(\.rule))
        let newlyCritical = alarms.filter { $0.severity == .critical && !criticalBefore.contains($0.rule) }
        healthAlarms = alarms
        if silenced && !newlyCritical.isEmpty {
            SessionLogger.shared.log("[ALARM] silence ended: newly critical training health alarms "
                + newlyCritical.map(\.rule.rawValue).joined(separator: ","))
            silenced = false
        }
        reconcileSound()
    }

    func setHealthSuspension(_ rule: TrainingHealthRule?) {
        healthSuspendedRule = rule
    }

    /// A Play-and-Train start: its monitor's alarms are live and may sound,
    /// and no health stop has suspended it yet.
    func beginHealthRun() {
        healthAlarmsAreFromAnEndedRun = false
        healthSuspendedRule = nil
        reconcileSound()
    }

    /// Stop: the run's alarms stay listed for review, but quiet
    /// (`healthAlarmsAreFromAnEndedRun`), and the suspension header ends
    /// with the run.
    func endHealthRun() {
        healthAlarmsAreFromAnEndedRun = true
        healthSuspendedRule = nil
        reconcileSound()
    }

    /// Whether the beep loop should run: the banner shows an alarm or a
    /// critical health alarm of the running start is active, and the user
    /// has not silenced it. Warnings alone never beep (OD-8).
    var shouldSound: Bool {
        guard !silenced else { return false }
        if active != nil { return true }
        return !healthAlarmsAreFromAnEndedRun && healthAlarms.contains { $0.severity == .critical }
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
        reconcileSound()
    }

    /// The consecutive-sample counters behind the divergence, value-saturation
    /// and value-draw alarms: what a session save records so a resume keeps
    /// counting where the saved run was (determinism plan C1 #20).
    struct Streaks: Codable, Equatable, Sendable {
        var divergenceWarning = 0
        var divergenceCritical = 0
        var divergenceRecovery = 0
        var valueAbsMeanSaturationWarning = 0
        var valueAbsMeanSaturationCritical = 0
        var valueAbsMeanSaturationRecovery = 0
        var valueDrawCollapseWarning = 0
        var valueDrawCollapseCritical = 0
        var valueDrawCollapseRecovery = 0
    }

    /// The current streak counters.
    var streaks: Streaks {
        Streaks(
            divergenceWarning: divergenceWarningStreak,
            divergenceCritical: divergenceCriticalStreak,
            divergenceRecovery: divergenceRecoveryStreak,
            valueAbsMeanSaturationWarning: valueAbsMeanSaturationWarningStreak,
            valueAbsMeanSaturationCritical: valueAbsMeanSaturationCriticalStreak,
            valueAbsMeanSaturationRecovery: valueAbsMeanSaturationRecoveryStreak,
            valueDrawCollapseWarning: valueDrawCollapseWarningStreak,
            valueDrawCollapseCritical: valueDrawCollapseCriticalStreak,
            valueDrawCollapseRecovery: valueDrawCollapseRecoveryStreak)
    }

    /// Continue counting from `saved` (a resumed session's counters).
    func restore(streaks saved: Streaks) {
        divergenceWarningStreak = saved.divergenceWarning
        divergenceCriticalStreak = saved.divergenceCritical
        divergenceRecoveryStreak = saved.divergenceRecovery
        valueAbsMeanSaturationWarningStreak = saved.valueAbsMeanSaturationWarning
        valueAbsMeanSaturationCriticalStreak = saved.valueAbsMeanSaturationCritical
        valueAbsMeanSaturationRecoveryStreak = saved.valueAbsMeanSaturationRecovery
        valueDrawCollapseWarningStreak = saved.valueDrawCollapseWarning
        valueDrawCollapseCriticalStreak = saved.valueDrawCollapseCritical
        valueDrawCollapseRecoveryStreak = saved.valueDrawCollapseRecovery
    }

    /// Zero the divergence streak counters without touching the banner. Paired
    /// with `clear()` by session-lifecycle resets (start new session, promote,
    /// new game) that previously zeroed the `divergence*Streak` `@State`
    /// alongside `clearTrainingAlarm()`.
    func resetStreaks() {
        divergenceWarningStreak = 0
        divergenceCriticalStreak = 0
        divergenceRecoveryStreak = 0
        valueAbsMeanSaturationWarningStreak = 0
        valueAbsMeanSaturationCriticalStreak = 0
        valueAbsMeanSaturationRecoveryStreak = 0
        valueDrawCollapseWarningStreak = 0
        valueDrawCollapseCriticalStreak = 0
        valueDrawCollapseRecoveryStreak = 0
    }

    // MARK: - Beep loop

    /// The one place the beep loop starts or stops, called after every
    /// change to the banner, the health alarms or `silenced`: one loop for
    /// both sources, so a health-only critical alarm beeps although the
    /// banner shows nothing.
    private func reconcileSound() {
        guard shouldSound else {
            alarmSoundTask?.cancel()
            alarmSoundTask = nil
            return
        }
        guard alarmSoundTask == nil else { return }
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
            if Task.isCancelled || !shouldSound { return }
            NSSound.beep()
            do {
                try await Task.sleep(for: .seconds(1.2))
            } catch {
                return
            }
        }
    }
}
