import Foundation

/// The training-health log lines, shared by every path (GUI, corpus replay,
/// train-vs-UCI, the offline `--replay-health-log`), so the format cannot
/// drift between them. Fixed key=value formats; `TrainingHealthLogTests` pins
/// every line kind exactly, because these strings are a grep contract:
///
/// - `[ALARM] health <kind> rule=…` — one line per event. The tag stays
///   `[ALARM]`, so `grep '\[ALARM\] health'` selects exactly these.
/// - `[HEALTH] …` — the run-start config, the periodic `check` line (positive
///   evidence the checks ran, with the monitor's own cost), rewinds, stale
///   observations, failed live reads and rules that do not apply.
///
/// Pure: renders strings, never writes them.
enum TrainingHealthLog {

    static let alarmTag = "[ALARM] health"
    static let healthTag = "[HEALTH]"

    /// `--` for a value not measured; never 0.
    static let notMeasured = "--"

    // MARK: Events

    static func eventLine(_ event: TrainingHealthEvent) -> String {
        var fields = ["\(alarmTag) \(event.kind.rawValue) rule=\(event.rule.rawValue) severity=\(event.severity.rawValue)"]
        switch event.kind {
        case .raise, .escalate:
            fields.append("trainerStep=\(event.trainerStep)")
            fields.append("value=\(event.value)")
            if !event.detail.isEmpty { fields.append(event.detail) }
            fields.append("threshold=\(event.threshold)")
            fields.append("action=\(event.action.name)")
            fields.append("lr=\(learningRate(event.learningRate))")
            fields.append("mom=\(momentum(event.momentum))")
        case .worsen:
            fields.append("trainerStep=\(event.trainerStep)")
            fields.append("value=\(event.value)")
            if !event.detail.isEmpty { fields.append(event.detail) }
        case .active, .clear:
            fields.append("since=\(event.since.map(String.init) ?? notMeasured)")
            fields.append("trainerStep=\(event.trainerStep)")
            fields.append("value=\(event.value)")
            if !event.detail.isEmpty { fields.append(event.detail) }
        case .stop:
            fields.append("trainerStep=\(event.trainerStep)")
            fields.append("action=\(event.action.name)")
        }
        return fields.joined(separator: " ")
    }

    /// The effective LR as the step lines print it (`%.3g`), `--` when unknown.
    static func learningRate(_ value: Double?) -> String {
        guard let value, value.isFinite else { return notMeasured }
        return String(format: "%.3g", value)
    }

    /// The effective momentum (`%.4g`), `--` for a checkpoint evaluation.
    static func momentum(_ value: Double?) -> String {
        guard let value, value.isFinite else { return notMeasured }
        return String(format: "%.4g", value)
    }

    // MARK: Run start

    /// `[HEALTH] config …` at every run start, beside the `[RUN]` line.
    static func configLine(
        config: TrainingHealthConfig,
        path: String,
        valueFC1Applicability: TrainingHealthValueFC1Applicability
    ) -> String {
        let actions = TrainingHealthRule.allCases
            .map { "\($0.rawValue):\(config.actions[$0].name)" }
            .joined(separator: ",")
        let rule3: String
        switch valueFC1Applicability {
        case .applies:
            rule3 = "applies"
        case .doesNotApply(let activation):
            rule3 = "not_applicable(activation=\(activation.rawValue))"
        case .unknownFromLog:
            rule3 = "unknown(activation not in log)"
        }
        return "\(healthTag) config enabled=\(config.enabled) interval=\(config.checkIntervalSteps)"
            + " grace=\(config.learningGraceSteps) warmup=\(config.lrWarmupSteps)"
            + " momentum=\(String(format: "%.4g", config.momentumCoefficient)) path=\(path)"
            + " actions=\(actions) value_fc1_zero_velocity=\(rule3)"
    }

    /// The one line a run with alarms disabled writes at its start.
    static func disabledLine(path: String) -> String {
        "\(healthTag) alarms disabled path=\(path)"
    }

    // MARK: Monitor lines

    static func rewindLine(from: Int?, to: Int, generation: Int) -> String {
        "\(healthTag) trainer clock rewound \(from.map(String.init) ?? "none") -> \(to); generation \(generation); windows reset"
    }

    static func unannouncedRewindLine(from: Int, to: Int, generation: Int) -> String {
        "\(healthTag) trainer clock rewind detected by recordStep (not announced): \(from) -> \(to); generation \(generation)"
    }

    static func staleObservationLine(
        tier: LayerHealthDigest.Tier,
        trainerStep: Int,
        generation: Int,
        currentGeneration: Int,
        newestApplied: Int?,
        reason: String?
    ) -> String {
        let reasonText = reason.map { "; \($0)" } ?? ""
        return "\(healthTag) stale \(tier.rawValue) observation ignored: trainerStep=\(trainerStep) generation=\(generation)"
            + " (current generation \(currentGeneration), newest applied \(newestApplied.map(String.init) ?? "none")\(reasonText))"
    }

    static func liveReadFailedLine(boundary: Int?) -> String {
        let step = boundary.map(String.init) ?? "none"
        return "\(healthTag) live read failed at trainerStep=\(step); window boundary = last recorded step \(step); layer-health rules no data"
    }

    static func notApplicableLine(rule: TrainingHealthRule, reason: String) -> String {
        "\(healthTag) rule \(rule.rawValue) does not apply: \(reason)"
    }

    // MARK: Check line

    enum CheckMarker: String, Sendable {
        /// Written when the run ends, flushing the last partial interval.
        case final
        /// Written by an evaluation that finished after the final line.
        case late
    }

    struct CheckFields: Sendable {
        let trainerStep: Int
        let generation: Int
        let live: Int
        let checkpoint: Int
        let stale: Int
        let truncated: Int
        let costMs: Double
        let trainMs: Double
        let liveReadFailed: Int
        let lossMaxRatio: Double?
        let lossMedianRatio: Double?
        let gradientMaxRatio: Double?
        let noData: [TrainingHealthRule: Int]
        let active: [TrainingHealthActiveAlarm]
        let marker: CheckMarker?
    }

    /// `[HEALTH] check …` — counters since the previous check line.
    /// `evaluations` is `live + checkpoint` (committed evaluations).
    /// `dead_channels_sites` names every affected site while
    /// `dead_channels` is active.
    static func checkLine(_ fields: CheckFields) -> String {
        func ratio(_ value: Double?) -> String {
            value.map { String(format: "%.2f", $0) } ?? notMeasured
        }
        let noData = TrainingHealthRule.allCases
            .compactMap { rule -> String? in
                guard let count = fields.noData[rule], count > 0 else { return nil }
                return "\(rule.rawValue):\(count)"
            }
            .joined(separator: ",")
        let active = fields.active
            .map { "\($0.rule.rawValue):\($0.severity.rawValue)" }
            .joined(separator: ",")
        var line = "\(healthTag) check trainerStep=\(fields.trainerStep) generation=\(fields.generation)"
            + " evaluations=\(fields.live + fields.checkpoint) live=\(fields.live) checkpoint=\(fields.checkpoint)"
            + " stale=\(fields.stale) truncated=\(fields.truncated)"
            + " cost_ms=\(String(format: "%.1f", fields.costMs)) train_ms=\(String(format: "%.1f", fields.trainMs))"
            + " liveReadFailed=\(fields.liveReadFailed)"
            + " lossMaxRatio=\(ratio(fields.lossMaxRatio)) lossMedianRatio=\(ratio(fields.lossMedianRatio))"
            + " gradMaxRatio=\(ratio(fields.gradientMaxRatio))"
            + " nodata=\(noData.isEmpty ? "none" : noData)"
            + " active=\(active.isEmpty ? "none" : active)"
        if let dead = fields.active.first(where: { $0.rule == .deadChannels }) {
            line += " dead_channels_\(dead.detail)"
        }
        if let marker = fields.marker {
            line += " \(marker.rawValue)=true"
        }
        return line
    }

    // MARK: The dedicated value-FC1 read (D6)

    /// `[LAYER-HEALTH] value-fc1 …` — one dedicated value-FC1 velocity read;
    /// the offline replay parses it as a rule-3-only observation.
    static func valueFC1Line(
        trainerStep: Int,
        zeroVelocityUnitCount: Int,
        unitCount: Int,
        lowVelocityUnitCount: Int,
        readMs: Double,
        summaryMs: Double
    ) -> String {
        "\(LayerHealthLog.tag) value-fc1 trainerStep=\(trainerStep)"
            + " valueFC1ZeroVel=\(zeroVelocityUnitCount)/\(unitCount) lowVel=\(lowVelocityUnitCount)"
            + " readMs=\(String(format: "%.2f", readMs)) summaryMs=\(String(format: "%.2f", summaryMs))"
    }
}
