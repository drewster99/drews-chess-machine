import Foundation
import TrainingParametersMacroSupport

/// One LR keyframe as the settings editor holds it: the five fields as text,
/// so a half-typed value is never lost or reformatted under the cursor. Save
/// parses the rows into an `LRKeyframeSchedule` and validates it with the
/// parameter's own validator (`schedule(from:)`).
struct LRKeyframeRowDraft: Identifiable, Equatable, Sendable {
    let id: UUID
    var step: String
    var peak: String
    var trough: String
    var momentumAtPeak: String
    var momentumAtTrough: String

    init(id: UUID = UUID(), step: String, peak: String, trough: String, momentumAtPeak: String, momentumAtTrough: String) {
        self.id = id
        self.step = step
        self.peak = peak
        self.trough = trough
        self.momentumAtPeak = momentumAtPeak
        self.momentumAtTrough = momentumAtTrough
    }

    /// A keyframe's values as text, each number in Swift's shortest exact
    /// form, so a seeded row parses back to the same `Double`.
    init(keyframe k: LRKeyframe) {
        self.init(step: String(k.step), peak: "\(k.peak)", trough: "\(k.trough)",
                  momentumAtPeak: "\(k.momentumAtPeak)", momentumAtTrough: "\(k.momentumAtTrough)")
    }

    /// Why rows could not become a schedule: a field that is not a number,
    /// or the schedule validator's refusal.
    enum Problem: Error, Equatable, CustomStringConvertible {
        case notANumber(row: Int, field: String, text: String)
        case invalid(LRKeyframeScheduleError)

        var description: String {
            switch self {
            case .notANumber(let row, let field, let text):
                return "keyframe \(row): \(field) \"\(text)\" is not a number"
            case .invalid(let error):
                return error.description
            }
        }
    }

    /// The rows as a validated schedule, or the first problem.
    static func schedule(from rows: [LRKeyframeRowDraft]) -> Result<LRKeyframeSchedule, Problem> {
        var frames: [LRKeyframe] = []
        for (index, row) in rows.enumerated() {
            func trimmed(_ s: String) -> String { s.trimmingCharacters(in: .whitespaces) }
            guard let step = Int(trimmed(row.step)) else {
                return .failure(.notANumber(row: index, field: "step", text: row.step))
            }
            var numbers: [Double] = []
            for (field, text) in [("peak", row.peak), ("trough", row.trough),
                                  ("momentum at peak", row.momentumAtPeak), ("momentum at trough", row.momentumAtTrough)] {
                guard let value = Double(trimmed(text)) else {
                    return .failure(.notANumber(row: index, field: field, text: text))
                }
                numbers.append(value)
            }
            frames.append(LRKeyframe(step: step, peak: numbers[0], trough: numbers[1],
                                     momentumAtPeak: numbers[2], momentumAtTrough: numbers[3]))
        }
        let schedule = LRKeyframeSchedule(keyframes: frames)
        do {
            return .success(try schedule.validated())
        } catch {
            return .failure(.invalid(error))
        }
    }

    /// A row to append after `rows`: the last row's values one LR-cycle
    /// period of `periodSteps` later, a starting point for the user to edit.
    static func appended(after rows: [LRKeyframeRowDraft], periodSteps: Int) -> LRKeyframeRowDraft {
        guard let last = rows.last else {
            return LRKeyframeRowDraft(keyframe: LRMomentumCycle.disabled.lrKeyframes.keyframes[0])
        }
        // A last step that isn't a usable step yet (still being typed, or
        // past the latest allowed) is copied as it is: the table shows the
        // problem on both rows rather than a step guessed from it.
        let lastStep = Int(last.step.trimmingCharacters(in: .whitespaces))
        let nextStep = lastStep.flatMap { $0 >= 0 && $0 <= LRKeyframeSchedule.maximumStep ? String($0 + max(1, periodSteps)) : nil }
        return LRKeyframeRowDraft(step: nextStep ?? last.step, peak: last.peak, trough: last.trough,
                                  momentumAtPeak: last.momentumAtPeak, momentumAtTrough: last.momentumAtTrough)
    }
}
