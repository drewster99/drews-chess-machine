import Foundation

/// One training parameter whose value differs between the run a CLI exact
/// resume continues (its parent's lineage snapshot) and the run doing the
/// resuming (its in-force snapshot, after the checkpoint's schedule was
/// adopted).
///
/// **Why this exists.** A `--resume-exact` is refused or reported NOT EXACT
/// only for what it cannot restore (streams, buffer, feed phase, build).
/// Training parameters are taken from the new run's `--parameters`, so a
/// resume that changed, say, `weight_decay` was logged `[RESUME] EXACT` with
/// no trace of the change anywhere. Each difference is now logged as one
/// `[RESUME-DIFF]` line, in the format the GUI resume already uses, so a
/// changed knob is visible in the session log of the segment that changed it.
/// Whether such a change should also be a resume gap is an open owner
/// decision; today it is only logged.
struct ParameterDifference: Equatable, Sendable {
    /// Which of the two snapshots hold the key.
    enum Presence: Equatable, Sendable {
        case both
        /// The parent recorded a key this build does not declare (removed or
        /// renamed since). Reported, never dropped.
        case parentOnly
        /// This build declares a key the parent's snapshot predates.
        case thisRunOnly
    }

    let id: String
    let presence: Presence
    /// The parent's value as text, nil when the parent has no such key.
    let parentValue: String?
    /// This run's value as text, nil when this build does not declare the key.
    let thisRunValue: String?
    /// The parent's value lies outside the range this build declares for the
    /// key (the range narrowed after the parent was written). The value is
    /// still what the parent trained with.
    let parentValueOutsideTodaysRange: Bool
    /// The key is a seed *setting* (`random_seed_mode` / `random_seed`). A
    /// run's actual seed is resolved separately (inherited from the parent's
    /// streams on an exact resume, or `--seed`) and logged on its `[RUN]`
    /// line, so a difference here is informational.
    let isSeedSetting: Bool

    /// The `[RESUME-DIFF]` log line for this difference.
    var logLine: String {
        var line = "[RESUME-DIFF] \(id): parent=\(parentValue ?? "absent") this_run=\(thisRunValue ?? "absent")"
        var notes: [String] = []
        switch presence {
        case .both: break
        case .parentOnly: notes.append("parent only: not a parameter of this build")
        case .thisRunOnly: notes.append("this run only: the parent's snapshot predates this parameter")
        }
        if parentValueOutsideTodaysRange {
            notes.append("parent value out of today's range")
        }
        if isSeedSetting {
            notes.append("seed setting, informational: the run's seed is on its [RUN] line")
        }
        if !notes.isEmpty {
            line += " (" + notes.joined(separator: "; ") + ")"
        }
        return line
    }
}

extension ParameterDifference {
    /// The `[RESUME-DIFF]` lines a CLI `--resume-exact` logs (corpus replay
    /// and train-vs-UCI share this one path): one per parameter whose value
    /// in `inForce` — this run's snapshot with the checkpoint's schedule
    /// adopted, so an adopted schedule is never reported — differs from the
    /// parent's recorded snapshot. A parent snapshot that cannot be compared
    /// refuses the resume rather than resuming with the comparison skipped.
    static func exactResumeLogLines(parent: LineageRecord.Parameters,
                                    inForce: TrainingParametersSnapshot) throws -> [String] {
        do {
            return try inForce.differences(fromLineage: parent).map(\.logLine)
        } catch {
            throw CLIRunRefusal(message: "--resume-exact: the checkpoint's parameter snapshot cannot be compared "
                + "with this run's: \(error.localizedDescription)")
        }
    }
}

/// Why a parent snapshot cannot be compared.
enum ParameterDifferenceError: Error, CustomStringConvertible, LocalizedError {
    /// `snapshot_json` is not a JSON object of parameter values.
    case unreadableParentSnapshot(detail: String)
    /// Every snapshot this build composes holds every declared parameter;
    /// one that does not is a bug, never compared as if the key were absent.
    case thisRunSnapshotMissing(id: String)

    var description: String {
        switch self {
        case .unreadableParentSnapshot(let detail):
            return "the parent's parameter snapshot is not a JSON object of parameter values: \(detail)"
        case .thisRunSnapshotMissing(let id):
            return "this run's parameter snapshot has no value for the declared parameter '\(id)'"
        }
    }

    var errorDescription: String? { description }
}

extension TrainingParametersSnapshot {
    /// Every parameter whose value differs between `parent` (the lineage
    /// snapshot of the run being resumed) and this snapshot, in id order.
    ///
    /// The parent's values are read through each declaration's *type*
    /// (`K.decode`), never through `validate`: a value outside today's range
    /// (a range narrowed since the parent was written) is what the parent
    /// trained with, so it is reported — marked out of range — and never
    /// aborts the resume. Both sides are compared in their decoded form, so a
    /// JSON integer written for a `Double` parameter equals the same `Double`.
    ///
    /// Seed settings are compared only when the parent's snapshot has them.
    ///
    /// The text is read with `JSONDecoder`, not `JSONSerialization`: the
    /// snapshot is written by `JSONSerialization`, which spells a `Double`
    /// with every significant digit (`0.0003` as `0.00029999999999999997`),
    /// and `JSONSerialization`'s own parser does not read every such
    /// spelling back to the same `Double`;
    /// comparing through it reported unchanged values as changed.
    ///
    /// Throws only when `snapshot_json` is not a JSON object of parameter
    /// values, or one of its values has the wrong type for its declaration.
    /// The record's own sha256 check already guarantees the text is the text
    /// that was written.
    func differences(fromLineage parent: LineageRecord.Parameters) throws -> [ParameterDifference] {
        let parentObject: [String: ParameterValue]
        do {
            parentObject = try JSONDecoder().decode([String: ParameterValue].self, from: Data(parent.snapshotJSON.utf8))
        } catch {
            throw ParameterDifferenceError.unreadableParentSnapshot(detail: String(describing: error))
        }
        let keysByID = Dictionary(uniqueKeysWithValues: TrainingParameters.allKeys.map { ($0.id, $0) })
        let seedSettingIDs: Set<String> = [RandomSeedModeParameter.id, RandomSeed.id]
        let thisRun = rawValueMap()
        var differences: [ParameterDifference] = []
        for id in Set(parentObject.keys).union(thisRun.keys).sorted() {
            let isSeedSetting = seedSettingIDs.contains(id)
            guard let key = keysByID[id] else {
                guard let parentRaw = parentObject[id] else { continue }
                differences.append(ParameterDifference(
                    id: id, presence: .parentOnly, parentValue: parentRaw.displayText,
                    thisRunValue: nil, parentValueOutsideTodaysRange: false, isSeedSetting: isSeedSetting))
                continue
            }
            guard let thisRunRaw = thisRun[id] else {
                throw ParameterDifferenceError.thisRunSnapshotMissing(id: id)
            }
            let thisRunValue = try Self.decodedByType(key, thisRunRaw)
            guard let parentRaw = parentObject[id] else {
                if isSeedSetting { continue }
                differences.append(ParameterDifference(
                    id: id, presence: .thisRunOnly, parentValue: nil, thisRunValue: thisRunValue.displayText,
                    parentValueOutsideTodaysRange: false, isSeedSetting: false))
                continue
            }
            let parentValue = try Self.decodedByType(key, parentRaw)
            guard parentValue != thisRunValue else { continue }
            differences.append(ParameterDifference(
                id: id, presence: .both, parentValue: parentValue.displayText, thisRunValue: thisRunValue.displayText,
                parentValueOutsideTodaysRange: try !Self.isWithinDeclaredRange(key, parentValue),
                isSeedSetting: isSeedSetting))
        }
        return differences
    }

    /// `raw` read as `K`'s declared type and written back in canonical form
    /// (`K.encode`), so equal values compare equal however they were written.
    private static func decodedByType<K: TrainingParameterKey>(_ key: K.Type, _ raw: ParameterValue) throws -> ParameterValue {
        K.encode(try K.decode(raw))
    }

    /// Whether `value` (already of `K`'s type) lies inside `K`'s declared
    /// range. Any failure other than "out of range" is rethrown.
    private static func isWithinDeclaredRange<K: TrainingParameterKey>(_ key: K.Type, _ value: ParameterValue) throws -> Bool {
        do {
            try K.definition.validate(value)
            return true
        } catch TrainingConfigError.outOfRange(_, _) {
            return false
        }
    }
}
