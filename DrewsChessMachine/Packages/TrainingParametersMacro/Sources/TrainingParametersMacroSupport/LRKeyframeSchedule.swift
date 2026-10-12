import Foundation

/// One point of the LR cycle's long-run envelope: at trainer step `step` the
/// cycle swings between `trough` and `peak`, and — when momentum follows the
/// LR cycle — momentum sits at `momentumAtPeak` where the LR peaks and at
/// `momentumAtTrough` where it bottoms out.
///
/// The momentum pair is always present, inert unless momentum follows the LR
/// cycle (as the four follow keys it replaces always held values). Keeping it
/// unconditional means no key's validity depends on another key's value; the
/// parameter system validates one key at a time.
///
/// It lives in the macro's support package, not the app, because
/// `ParameterValue` is public and carries it, and the `@TrainingParameter`
/// declaration overload must be able to name it as a default's type.
public struct LRKeyframe: Sendable, Equatable, Hashable, Codable {
    /// Trainer step (not cycle step) at which these values hold exactly.
    public var step: Int
    public var peak: Double
    public var trough: Double
    public var momentumAtPeak: Double
    public var momentumAtTrough: Double

    public init(step: Int, peak: Double, trough: Double, momentumAtPeak: Double, momentumAtTrough: Double) {
        self.step = step
        self.peak = peak
        self.trough = trough
        self.momentumAtPeak = momentumAtPeak
        self.momentumAtTrough = momentumAtTrough
    }

    enum CodingKeys: String, CodingKey {
        case step, peak, trough
        case momentumAtPeak = "momentum_at_peak"
        case momentumAtTrough = "momentum_at_trough"
    }
}

/// The LR cycle's envelope over a run: a list of keyframes, geometric between
/// them for the LR bounds and linear for the momentum bounds, held after the
/// last (documentation/plans-active/LR_KEYFRAMES_PLAN.md, owner decisions K1–K3,
/// O-2, O-3, Q-3).
///
/// Why geometric for LR: an LR change acts on ratios — measured weight motion
/// scales in proportion to the LR — so a linear ramp from 1.0 to 0.1 would
/// spend nearly all its time near 1.0. Momentum's useful band is narrow and
/// was always interpolated linearly.
///
/// Why "held" is computed rather than stored: past the last keyframe the
/// bounds are the last segment evaluated at fraction 1.0, `b_{n-2} *
/// pow(b_{n-1} / b_{n-2}, 1.0)`, which can differ from `b_{n-1}` in the last
/// bit. That is exactly the arithmetic the previous single-decay envelope did
/// after its horizon, so a schedule converted from it feeds bit-identical
/// values at every step (owner decision Q-4: a resumed old run must not change
/// at all). Every keyframe except the last is reproduced exactly.
public struct LRKeyframeSchedule: Sendable, Equatable, Hashable, Codable {
    public var keyframes: [LRKeyframe]

    public init(keyframes: [LRKeyframe]) {
        self.keyframes = keyframes
    }

    /// Owner decision Q-3: any real schedule needs a handful; the bound keeps
    /// the editor, the log lines and the parameter snapshot small.
    public static let maximumKeyframeCount = 64
    /// The previous `lr_cycle_max` ceiling; an LR above it is a typo, not a
    /// schedule.
    public static let maximumLearningRate = 10.0
    /// The previous momentum-follow keys' range.
    public static let momentumRange: ClosedRange<Double> = 0.0...0.99

    /// The envelope at one trainer step.
    public struct Bounds: Sendable, Equatable {
        public let peak: Double
        public let trough: Double
        public let momentumAtPeak: Double
        public let momentumAtTrough: Double

        public init(peak: Double, trough: Double, momentumAtPeak: Double, momentumAtTrough: Double) {
            self.peak = peak
            self.trough = trough
            self.momentumAtPeak = momentumAtPeak
            self.momentumAtTrough = momentumAtTrough
        }
    }

    /// The bounds at trainer step `step`. Segment k covers
    /// `keyframes[k].step <= step < keyframes[k+1].step`; a step at or after
    /// the last keyframe is the last segment at fraction 1.0; a step before
    /// the first (impossible for a validated schedule, whose first keyframe
    /// is at 0) is the first segment at fraction 0.0. A single keyframe is
    /// its own segment, evaluated as `b * pow(b / b, 0.0)` — exactly `b`.
    ///
    /// Precondition: the schedule validates (non-empty, increasing steps).
    public func bounds(atTrainerStep step: Int) -> Bounds {
        let frames = keyframes
        precondition(!frames.isEmpty, "LRKeyframeSchedule.bounds on an empty schedule")
        if frames.count == 1 {
            let only = frames[0]
            return Self.interpolate(only, only, fraction: 0.0)
        }
        if step < frames[0].step {
            return Self.interpolate(frames[0], frames[1], fraction: 0.0)
        }
        let last = frames.count - 1
        if step >= frames[last].step {
            return Self.interpolate(frames[last - 1], frames[last], fraction: 1.0)
        }
        // Linear scan: at most 64 keyframes, and almost always 1–5.
        var k = 0
        while frames[k + 1].step <= step {
            k += 1
        }
        let a = frames[k]
        let b = frames[k + 1]
        let fraction = Double(step - a.step) / Double(b.step - a.step)
        return Self.interpolate(a, b, fraction: fraction)
    }

    /// The one place the segment arithmetic lives. The expressions and their
    /// operand order are the previous envelope's
    /// (`lrMax * pow(lrPeakEnd / lrMax, decay)`,
    /// `startLow + (endLow - startLow) * decay`); bit-identity with it depends
    /// on keeping them so.
    public static func interpolate(_ a: LRKeyframe, _ b: LRKeyframe, fraction f: Double) -> Bounds {
        Bounds(
            peak: a.peak * pow(b.peak / a.peak, f),
            trough: a.trough * pow(b.trough / a.trough, f),
            momentumAtPeak: a.momentumAtPeak + (b.momentumAtPeak - a.momentumAtPeak) * f,
            momentumAtTrough: a.momentumAtTrough + (b.momentumAtTrough - a.momentumAtTrough) * f
        )
    }

    /// The one validator: every writer of a schedule (parameter commits,
    /// `parameters.json`, `UserDefaults`, the converter, the editor) calls it.
    /// Returns the schedule unchanged, or throws the first problem found.
    @discardableResult
    public func validated() throws(LRKeyframeScheduleError) -> LRKeyframeSchedule {
        guard !keyframes.isEmpty else { throw LRKeyframeScheduleError.empty }
        guard keyframes.count <= Self.maximumKeyframeCount else {
            throw LRKeyframeScheduleError.tooManyKeyframes(count: keyframes.count, maximum: Self.maximumKeyframeCount)
        }
        guard keyframes[0].step == 0 else { throw LRKeyframeScheduleError.firstStepNotZero(step: keyframes[0].step) }
        for index in keyframes.indices {
            let k = keyframes[index]
            if index > 0, k.step <= keyframes[index - 1].step {
                throw LRKeyframeScheduleError.stepsNotIncreasing(index: index, step: k.step, previous: keyframes[index - 1].step)
            }
            for (field, value) in [("peak", k.peak), ("trough", k.trough),
                                   ("momentum_at_peak", k.momentumAtPeak), ("momentum_at_trough", k.momentumAtTrough)]
            where !value.isFinite {
                throw LRKeyframeScheduleError.notFinite(index: index, field: field)
            }
            guard k.trough > 0 else { throw LRKeyframeScheduleError.troughNotPositive(index: index, trough: k.trough) }
            guard k.peak >= k.trough else {
                throw LRKeyframeScheduleError.peakBelowTrough(index: index, peak: k.peak, trough: k.trough)
            }
            guard k.peak <= Self.maximumLearningRate else {
                throw LRKeyframeScheduleError.peakTooLarge(index: index, peak: k.peak, maximum: Self.maximumLearningRate)
            }
            for (field, value) in [("momentum_at_peak", k.momentumAtPeak), ("momentum_at_trough", k.momentumAtTrough)]
            where !Self.momentumRange.contains(value) {
                throw LRKeyframeScheduleError.momentumOutOfRange(index: index, field: field, value: value, range: Self.momentumRange)
            }
            guard k.momentumAtPeak <= k.momentumAtTrough else {
                throw LRKeyframeScheduleError.momentumAtPeakAboveTrough(index: index, atPeak: k.momentumAtPeak, atTrough: k.momentumAtTrough)
            }
        }
        return self
    }

    /// Codable as a bare array of keyframes — the form `parameters.json`,
    /// `UserDefaults`, session files and lineage records carry.
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.keyframes = try container.decode([LRKeyframe].self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(keyframes)
    }
}

/// Why a keyframe schedule was refused, naming the keyframe (0-based) and the
/// field, so the editor can mark the cell and a `parameters.json` error says
/// what to fix.
public enum LRKeyframeScheduleError: Error, Equatable, CustomStringConvertible {
    case empty
    case tooManyKeyframes(count: Int, maximum: Int)
    case firstStepNotZero(step: Int)
    case stepsNotIncreasing(index: Int, step: Int, previous: Int)
    case notFinite(index: Int, field: String)
    case troughNotPositive(index: Int, trough: Double)
    case peakBelowTrough(index: Int, peak: Double, trough: Double)
    case peakTooLarge(index: Int, peak: Double, maximum: Double)
    case momentumOutOfRange(index: Int, field: String, value: Double, range: ClosedRange<Double>)
    case momentumAtPeakAboveTrough(index: Int, atPeak: Double, atTrough: Double)

    /// The keyframe the problem is in, when there is one.
    public var keyframeIndex: Int? {
        switch self {
        case .empty, .tooManyKeyframes: return nil
        case .firstStepNotZero: return 0
        case .stepsNotIncreasing(let index, _, _), .notFinite(let index, _), .troughNotPositive(let index, _),
             .peakBelowTrough(let index, _, _), .peakTooLarge(let index, _, _), .momentumOutOfRange(let index, _, _, _),
             .momentumAtPeakAboveTrough(let index, _, _):
            return index
        }
    }

    public var description: String {
        switch self {
        case .empty:
            return "an LR schedule needs at least one keyframe"
        case .tooManyKeyframes(let count, let maximum):
            return "\(count) keyframes; at most \(maximum)"
        case .firstStepNotZero(let step):
            return "the first keyframe must be at step 0, not \(step)"
        case .stepsNotIncreasing(let index, let step, let previous):
            return "keyframe \(index): step \(step) is not after the previous keyframe's step \(previous)"
        case .notFinite(let index, let field):
            return "keyframe \(index): \(field) is not a finite number"
        case .troughNotPositive(let index, let trough):
            return "keyframe \(index): trough \(trough) must be greater than 0"
        case .peakBelowTrough(let index, let peak, let trough):
            return "keyframe \(index): peak \(peak) is below trough \(trough)"
        case .peakTooLarge(let index, let peak, let maximum):
            return "keyframe \(index): peak \(peak) is above \(maximum)"
        case .momentumOutOfRange(let index, let field, let value, let range):
            return "keyframe \(index): \(field) \(value) is outside \(range.lowerBound)…\(range.upperBound)"
        case .momentumAtPeakAboveTrough(let index, let atPeak, let atTrough):
            return "keyframe \(index): momentum at the LR peak \(atPeak) is above momentum at the LR trough \(atTrough)"
        }
    }
}
