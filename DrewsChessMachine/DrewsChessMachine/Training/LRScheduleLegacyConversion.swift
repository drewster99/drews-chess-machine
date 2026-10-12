import Foundation
import TrainingParametersMacroSupport

// MARK: - Converting the previous single-decay schedule

/// The previous schedule's settings, as plain values: the cycle, its decay
/// envelope and the momentum-follow bounds, plus the warmup length the decay
/// was offset by. The one input to the converter, whatever carried the old
/// schedule (a trainer file, a session, a lineage snapshot, `parameters.json`
/// or `UserDefaults`).
struct LegacyLRSchedule: Sendable, Equatable {
    var lrEnabled: Bool
    var lrPeriodSteps: Int
    var lrCount: Int
    var lrMin: Double
    var lrMax: Double
    var lrInvert: Bool
    var momentumEnabled: Bool
    var momentumPeriodSteps: Int
    var momentumCount: Int
    var momentumMin: Double
    var momentumMax: Double
    var momentumInvert: Bool
    var lrPeakEnd: Double
    var lrTroughEnd: Double
    /// In cycle steps (trainer step − warmup); 0 = no decay.
    var decayHorizonSteps: Int
    var momentumFollowsLRCycle: Bool
    var momentumFollowStartLow: Double
    var momentumFollowStartHigh: Double
    var momentumFollowEndLow: Double
    var momentumFollowEndHigh: Double
    var lrWarmupSteps: Int
}

/// A legacy schedule the keyframes cannot express. The previous code ran
/// static momentum on exactly the steps where the follow bounds were
/// inverted; when that is only part of the run, no keyframe schedule (whose
/// bounds are always ordered) reproduces it. Refused rather than approximated
/// (owner decision Q-4).
struct LegacyLRScheduleConversionError: Error, Equatable, CustomStringConvertible {
    let description: String
}

extension LRMomentumCycle {
    /// The previous schedule, converted so that every step feeds bit-identical
    /// LR and momentum (owner decision Q-4; Validation V-2).
    ///
    /// With warmup `w`, horizon `H` (cycle steps), the keyframes are
    /// `(0, start)` `(w, start)` `(w + H, end)` — the keyframe at `w` holds the
    /// start values through warmup, where the old decay fraction was 0 —
    /// collapsing to `(0, start)` `(H, end)` when `w == 0` (no duplicate step)
    /// and to `(0, start)` alone when `H == 0`.
    ///
    /// Settings the previous code treated as "no cycle" convert to what it
    /// did. A cycle whose LR pair was unusable (a bound not above 0, a peak
    /// below its trough at the start or the end) ran the static LR, and its
    /// follow momentum (which needs the LR phase) ran static too: it becomes
    /// `lrEnabled == false`. Follow bounds inverted at every step ran static
    /// momentum: they become `momentumEnabled == false`. Values that are
    /// inert after that, and invalid, are replaced by `inertSubstitute`'s so
    /// the result validates; inert values that are valid are kept as they
    /// are, so re-enabling a disabled cycle later gives back its schedule.
    ///
    /// `notes` says what was converted and why, for the caller's log line.
    static func converting(_ legacy: LegacyLRSchedule) throws -> (cycle: LRMomentumCycle, notes: [String]) {
        var notes: [String] = []
        let w = max(0, legacy.lrWarmupSteps)
        let h = max(0, legacy.decayHorizonSteps)

        func lrPairValid(peak: Double, trough: Double) -> Bool {
            peak.isFinite && trough.isFinite && trough > 0 && peak >= trough
                && peak <= LRKeyframeSchedule.maximumLearningRate
        }
        func momentumValue(_ value: Double) -> Bool {
            value.isFinite && LRKeyframeSchedule.momentumRange.contains(value)
        }
        let startLRValid = lrPairValid(peak: legacy.lrMax, trough: legacy.lrMin)
        let endLRValid = h == 0 || lrPairValid(peak: legacy.lrPeakEnd, trough: legacy.lrTroughEnd)
        let lrValid = startLRValid && endLRValid

        let followValuesInRange = [legacy.momentumFollowStartLow, legacy.momentumFollowStartHigh]
            .allSatisfy(momentumValue)
            && (h == 0 || [legacy.momentumFollowEndLow, legacy.momentumFollowEndHigh].allSatisfy(momentumValue))
        let startOrdered = legacy.momentumFollowStartHigh >= legacy.momentumFollowStartLow
        let endOrdered = h == 0 || legacy.momentumFollowEndHigh >= legacy.momentumFollowEndLow
        let followValid = followValuesInRange && startOrdered && endOrdered

        var lrEnabled = legacy.lrEnabled
        var momentumEnabled = legacy.momentumEnabled

        if lrEnabled && legacy.lrPeriodSteps > 0 && !lrValid {
            // The old code returned nil here at every step: static LR, and no
            // phase for following momentum to read.
            lrEnabled = false
            notes.append("the LR cycle was inactive (unusable bounds: start \(legacy.lrMax)/\(legacy.lrMin)"
                         + (h > 0 ? ", end \(legacy.lrPeakEnd)/\(legacy.lrTroughEnd)" : "")
                         + "); converted as disabled")
        }
        let followDrivesMomentum = momentumEnabled && legacy.momentumFollowsLRCycle
            && lrEnabled && legacy.lrPeriodSteps > 0
        // With no decay (h == 0) the start bounds hold at every step, so an
        // inverted start pair is inverted throughout whatever the end pair.
        let invertedThroughout = !startOrdered
            && (h == 0 || legacy.momentumFollowEndHigh < legacy.momentumFollowEndLow)
        if followDrivesMomentum && !followValid {
            if followValuesInRange && invertedThroughout {
                // Inverted at the start and at the end (or with no decay);
                // the bounds move linearly between, so inverted at every
                // step: the old code fed the static momentum throughout.
                momentumEnabled = false
                notes.append("momentum following was inactive (bounds inverted at every step); converted as momentum cycling disabled")
            } else if followValuesInRange {
                throw LegacyLRScheduleConversionError(
                    description: "the momentum-follow bounds are inverted on only part of the run "
                        + "(start low \(legacy.momentumFollowStartLow) high \(legacy.momentumFollowStartHigh), "
                        + "end low \(legacy.momentumFollowEndLow) high \(legacy.momentumFollowEndHigh)); "
                        + "the previous code fed static momentum on those steps only, which no keyframe schedule can express")
            } else {
                throw LegacyLRScheduleConversionError(
                    description: "a momentum-follow bound is outside \(LRKeyframeSchedule.momentumRange)")
            }
        }

        let substitute = inertSubstitute
        let startPeak = lrValid ? legacy.lrMax : substitute.peak
        let startTrough = lrValid ? legacy.lrMin : substitute.trough
        let endPeak = lrValid ? legacy.lrPeakEnd : substitute.peak
        let endTrough = lrValid ? legacy.lrTroughEnd : substitute.trough
        let startLow = followValid ? legacy.momentumFollowStartLow : substitute.momentumAtPeak
        let startHigh = followValid ? legacy.momentumFollowStartHigh : substitute.momentumAtTrough
        let endLow = followValid ? legacy.momentumFollowEndLow : substitute.momentumAtPeak
        let endHigh = followValid ? legacy.momentumFollowEndHigh : substitute.momentumAtTrough
        if !lrValid {
            notes.append("inert LR bounds replaced by \(substitute.peak)/\(substitute.trough)")
        }
        if !followValid {
            notes.append("inert momentum-follow bounds replaced by \(substitute.momentumAtPeak)/\(substitute.momentumAtTrough)")
        }

        let start = { (step: Int) in
            LRKeyframe(step: step, peak: startPeak, trough: startTrough, momentumAtPeak: startLow, momentumAtTrough: startHigh)
        }
        let keyframes: [LRKeyframe]
        if h == 0 {
            keyframes = [start(0)]
        } else {
            let end = LRKeyframe(step: w + h, peak: endPeak, trough: endTrough, momentumAtPeak: endLow, momentumAtTrough: endHigh)
            keyframes = w == 0 ? [start(0), end] : [start(0), start(w), end]
        }
        let schedule = try LRKeyframeSchedule(keyframes: keyframes).validated()

        let cycle = LRMomentumCycle(
            lrEnabled: lrEnabled, lrPeriodSteps: legacy.lrPeriodSteps, lrCount: legacy.lrCount,
            lrInvert: legacy.lrInvert, lrKeyframes: schedule,
            momentumEnabled: momentumEnabled, momentumPeriodSteps: legacy.momentumPeriodSteps,
            momentumCount: legacy.momentumCount, momentumMin: legacy.momentumMin, momentumMax: legacy.momentumMax,
            momentumInvert: legacy.momentumInvert, momentumFollowsLRCycle: legacy.momentumFollowsLRCycle
        )
        return (cycle, notes)
    }

    /// The values that stand in for invalid ones the converted schedule never
    /// reads (a disabled channel's numbers): the previous defaults' start
    /// values, which validate.
    static let inertSubstitute = LRKeyframe(step: 0, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95)
}


// MARK: - The previous schedule's stored forms

/// The previous cycle's encoded form (`trainer_lr_momentum_cycle` in a
/// trainer file, `lrMomentumCycle` in a session file): the synthesized
/// `Codable` of the struct before LR keyframes. Read so old files stay
/// readable (owner rule) and converted at once; nothing writes it any more
/// (a session's legacy fields are nil when it is saved, so they are omitted).
struct LegacyLRMomentumCycleCoding: Codable, Equatable, Sendable {
    let lrEnabled: Bool
    let lrPeriodSteps: Int
    let lrCount: Int
    let lrMin: Double
    let lrMax: Double
    let lrInvert: Bool
    let momentumEnabled: Bool
    let momentumPeriodSteps: Int
    let momentumCount: Int
    let momentumMin: Double
    let momentumMax: Double
    let momentumInvert: Bool
}

/// The previous decay envelope's encoded form
/// (`trainer_lr_momentum_cycle_envelope`, session `lrMomentumCycleEnvelope`).
/// Read and converted, never written, like `LegacyLRMomentumCycleCoding`.
struct LegacyLRMomentumCycleEnvelopeCoding: Codable, Equatable, Sendable {
    let lrPeakEnd: Double
    let lrTroughEnd: Double
    let decayHorizonSteps: Int
    let momentumFollowsLRCycle: Bool
    let momentumFollowStartLow: Double
    let momentumFollowStartHigh: Double
    let momentumFollowEndLow: Double
    let momentumFollowEndHigh: Double

    /// What a file written before the envelope existed ran: no decay and no
    /// momentum following (the previous `LRMomentumCycleEnvelope.noDecay`,
    /// whose other values were inert).
    static let beforeTheEnvelope = LegacyLRMomentumCycleEnvelopeCoding(
        lrPeakEnd: 1.0e-4, lrTroughEnd: 1.0e-6, decayHorizonSteps: 0, momentumFollowsLRCycle: false,
        momentumFollowStartLow: 0.85, momentumFollowStartHigh: 0.95,
        momentumFollowEndLow: 0.90, momentumFollowEndHigh: 0.95)
}

extension LegacyLRSchedule {
    init(cycle: LegacyLRMomentumCycleCoding, envelope: LegacyLRMomentumCycleEnvelopeCoding?, lrWarmupSteps: Int) {
        let e = envelope ?? .beforeTheEnvelope
        self.init(
            lrEnabled: cycle.lrEnabled, lrPeriodSteps: cycle.lrPeriodSteps, lrCount: cycle.lrCount,
            lrMin: cycle.lrMin, lrMax: cycle.lrMax, lrInvert: cycle.lrInvert,
            momentumEnabled: cycle.momentumEnabled, momentumPeriodSteps: cycle.momentumPeriodSteps,
            momentumCount: cycle.momentumCount, momentumMin: cycle.momentumMin, momentumMax: cycle.momentumMax,
            momentumInvert: cycle.momentumInvert, lrPeakEnd: e.lrPeakEnd, lrTroughEnd: e.lrTroughEnd,
            decayHorizonSteps: e.decayHorizonSteps, momentumFollowsLRCycle: e.momentumFollowsLRCycle,
            momentumFollowStartLow: e.momentumFollowStartLow, momentumFollowStartHigh: e.momentumFollowStartHigh,
            momentumFollowEndLow: e.momentumFollowEndLow, momentumFollowEndHigh: e.momentumFollowEndHigh,
            lrWarmupSteps: lrWarmupSteps)
    }
}

// MARK: - The previous schedule's parameter keys

/// The nine parameter ids the keyframes replaced, their declared defaults
/// before the change, and the conversion of a parameter dictionary that
/// carries them (`parameters.json`, a settings file, a lineage snapshot, the
/// GUI's stored settings).
enum LegacyLRScheduleKeys {
    static let lrCycleMin = "lr_cycle_min"
    static let lrCycleMax = "lr_cycle_max"
    static let lrCyclePeakEnd = "lr_cycle_peak_end"
    static let lrCycleTroughEnd = "lr_cycle_trough_end"
    static let lrCycleDecayHorizonSteps = "lr_cycle_decay_horizon_steps"
    static let momentumFollowStartLow = "momentum_follow_start_low"
    static let momentumFollowStartHigh = "momentum_follow_start_high"
    static let momentumFollowEndLow = "momentum_follow_end_low"
    static let momentumFollowEndHigh = "momentum_follow_end_high"

    /// The removed ids, in the order the previous declarations listed them.
    static let removedIDs: [String] = [
        lrCycleMin, lrCycleMax, lrCyclePeakEnd, lrCycleTroughEnd, lrCycleDecayHorizonSteps,
        momentumFollowStartLow, momentumFollowStartHigh, momentumFollowEndLow, momentumFollowEndHigh,
    ]

    /// The previous declarations' defaults: what a removed key a partial
    /// file omits stands for. (The previous build took an omitted key from
    /// the stored settings, which no longer hold these keys.)
    static let previousDefaults: [String: ParameterValue] = [
        lrCycleMin: .double(0.001), lrCycleMax: .double(0.1), lrCyclePeakEnd: .double(1.0e-4),
        lrCycleTroughEnd: .double(1.0e-6), lrCycleDecayHorizonSteps: .int(1_000_000),
        momentumFollowStartLow: .double(0.85), momentumFollowStartHigh: .double(0.95),
        momentumFollowEndLow: .double(0.90), momentumFollowEndHigh: .double(0.95),
    ]

    /// The previous declarations' ranges (integer ranges as doubles), used
    /// to read the GUI's stored entries the way the previous build did.
    static let previousRanges: [String: ClosedRange<Double>] = [
        lrCycleMin: 1.0e-7...1.0, lrCycleMax: 1.0e-7...10.0, lrCyclePeakEnd: 1.0e-7...1.0,
        lrCycleTroughEnd: 1.0e-7...1.0, lrCycleDecayHorizonSteps: 0...1_000_000_000,
        momentumFollowStartLow: 0.0...0.99, momentumFollowStartHigh: 0.0...0.99,
        momentumFollowEndLow: 0.0...0.99, momentumFollowEndHigh: 0.0...0.99,
    ]

    /// A stored entry of a removed key as the previous build read it: an
    /// entry of the declared type inside the declared range is its value;
    /// any other was replaced by the previous default (the previous
    /// `read`), so it converts as that default, with the reason.
    static func previousStoredValue(_ id: String, object: Any) throws -> (value: ParameterValue, replacedBecause: String?) {
        guard let previousDefault = previousDefaults[id], let range = previousRanges[id] else {
            throw TrainingConfigError.invalidValue(id: id, reason: "not one of the previous LR schedule keys")
        }
        let parsed: ParameterValue
        do {
            parsed = try ParameterValue(jsonValue: object, id: id)
        } catch {
            return (previousDefault, "\(id) held \(object) (\(error)), so the previous build read its default")
        }
        let number: Double?
        switch (previousDefault, parsed) {
        case (.double, .double(let value)): number = value
        case (.double, .int(let value)): number = Double(value)   // a whole number for a real key, as the previous read took it
        case (.int, .int(let value)): number = Double(value)
        default: number = nil
        }
        guard let number, number.isFinite, range.contains(number) else {
            return (previousDefault, "\(id) held \(object), not a value in the previous range \(range), so the previous build read its default")
        }
        return (parsed, nil)
    }

    /// Writes a conversion note to the session log and to stderr: the
    /// conversions can run before a CLI path starts its session log (which
    /// drops earlier lines), and the stored-settings one runs only once.
    static func logNote(_ line: String) {
        SessionLogger.shared.log(line)
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// The current persisted value of a kept parameter, as a `ParameterValue`:
    /// what a partial `parameters.json` leaves a key at. Only the kept keys
    /// the conversion reads are answered.
    static func persistedSetting(_ id: String) -> ParameterValue? {
        func value<K: TrainingParameterKey>(_ key: K.Type) -> ParameterValue { K.encode(TrainingParameters.persistedValue(key)) }
        switch id {
        case LRCycleEnabled.id: return value(LRCycleEnabled.self)
        case LRCyclePeriodSteps.id: return value(LRCyclePeriodSteps.self)
        case LRCycleCount.id: return value(LRCycleCount.self)
        case LRCycleInvert.id: return value(LRCycleInvert.self)
        case MomentumCycleEnabled.id: return value(MomentumCycleEnabled.self)
        case MomentumCyclePeriodSteps.id: return value(MomentumCyclePeriodSteps.self)
        case MomentumCycleCount.id: return value(MomentumCycleCount.self)
        case MomentumCycleMin.id: return value(MomentumCycleMin.self)
        case MomentumCycleMax.id: return value(MomentumCycleMax.self)
        case MomentumCycleInvert.id: return value(MomentumCycleInvert.self)
        case MomentumFollowsLRCycle.id: return value(MomentumFollowsLRCycle.self)
        case LRWarmupSteps.id: return value(LRWarmupSteps.self)
        default: return nil
        }
    }

    /// `values` with any of the nine removed keys converted to
    /// `lr_cycle_keyframes` (exactly, `LRMomentumCycle.converting`), plus
    /// `lr_cycle_enabled` / `momentum_cycle_enabled` set false where the
    /// previous code ran that channel as "no cycle". Unchanged when no
    /// removed key is present. Refused when a removed key appears beside
    /// `lr_cycle_keyframes` (two sources for one schedule).
    ///
    /// A removed key the dictionary lacks takes its previous default; a kept
    /// key it lacks takes `currentValue` (the setting the text leaves in
    /// force). `notes` lists what was converted, for the caller's log line.
    static func converting(
        _ values: [String: ParameterValue],
        currentValue: (String) -> ParameterValue?
    ) throws -> (values: [String: ParameterValue], notes: [String]) {
        let present = removedIDs.filter { values[$0] != nil }
        guard !present.isEmpty else { return (values, []) }
        guard values[LRCycleKeyframes.id] == nil else {
            throw TrainingConfigError.invalidValue(
                id: LRCycleKeyframes.id,
                reason: "given together with the previous schedule's keys \(present.joined(separator: ", ")); use one or the other")
        }
        var notes: [String] = []
        let missingRemoved = removedIDs.filter { values[$0] == nil }
        if !missingRemoved.isEmpty {
            notes.append("missing \(missingRemoved.joined(separator: ", ")) taken from the previous defaults")
        }
        func raw(_ id: String) throws -> ParameterValue {
            if let v = values[id] { return v }
            if let v = previousDefaults[id] { return v }
            if let v = currentValue(id) { return v }
            throw TrainingConfigError.invalidValue(id: id, reason: "needed to convert the previous LR schedule keys, and not available")
        }
        func double(_ id: String) throws -> Double {
            switch try raw(id) {
            case .double(let x): return x
            case .int(let x): return Double(x)
            default: throw TrainingConfigError.wrongType(id: id)
            }
        }
        func int(_ id: String) throws -> Int {
            guard case .int(let x) = try raw(id) else { throw TrainingConfigError.wrongType(id: id) }
            return x
        }
        func bool(_ id: String) throws -> Bool {
            guard case .bool(let x) = try raw(id) else { throw TrainingConfigError.wrongType(id: id) }
            return x
        }
        let legacy = LegacyLRSchedule(
            lrEnabled: try bool(LRCycleEnabled.id), lrPeriodSteps: try int(LRCyclePeriodSteps.id),
            lrCount: try int(LRCycleCount.id), lrMin: try double(lrCycleMin), lrMax: try double(lrCycleMax),
            lrInvert: try bool(LRCycleInvert.id), momentumEnabled: try bool(MomentumCycleEnabled.id),
            momentumPeriodSteps: try int(MomentumCyclePeriodSteps.id), momentumCount: try int(MomentumCycleCount.id),
            momentumMin: try double(MomentumCycleMin.id), momentumMax: try double(MomentumCycleMax.id),
            momentumInvert: try bool(MomentumCycleInvert.id), lrPeakEnd: try double(lrCyclePeakEnd),
            lrTroughEnd: try double(lrCycleTroughEnd), decayHorizonSteps: try int(lrCycleDecayHorizonSteps),
            momentumFollowsLRCycle: try bool(MomentumFollowsLRCycle.id),
            momentumFollowStartLow: try double(momentumFollowStartLow),
            momentumFollowStartHigh: try double(momentumFollowStartHigh),
            momentumFollowEndLow: try double(momentumFollowEndLow), momentumFollowEndHigh: try double(momentumFollowEndHigh),
            lrWarmupSteps: try int(LRWarmupSteps.id))
        let converted: (cycle: LRMomentumCycle, notes: [String])
        do {
            converted = try LRMomentumCycle.converting(legacy)
        } catch let error as LegacyLRScheduleConversionError {
            throw TrainingConfigError.invalidValue(id: LRCycleKeyframes.id, reason: error.description)
        }
        var out = values
        for id in removedIDs { out[id] = nil }
        out[LRCycleKeyframes.id] = .lrKeyframes(converted.cycle.lrKeyframes)
        if converted.cycle.lrEnabled != legacy.lrEnabled {
            out[LRCycleEnabled.id] = .bool(converted.cycle.lrEnabled)
        }
        if converted.cycle.momentumEnabled != legacy.momentumEnabled {
            out[MomentumCycleEnabled.id] = .bool(converted.cycle.momentumEnabled)
        }
        notes.append(contentsOf: converted.notes)
        notes.append(LRMomentumCycleLogFormat.keyframesDescription(converted.cycle.lrKeyframes))
        return (out, notes)
    }
}
