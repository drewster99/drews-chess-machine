//
//  RunProvenanceLine.swift
//  DrewsChessMachine
//
//  The one `[RUN]` line every run path logs when it starts (determinism plan
//  D4): which run and segment this is and what it continues, the build and
//  machine, the run seed, the parameters' hash, the totals the run starts
//  from, and the command line. Before it, a corpus-replay or train-vs-UCI
//  log carried no build, device or seed banner, and answering "what produced
//  this curve" meant cross-referencing files by hand; with it, the first
//  `[RUN]` line of any session log answers it.
//
//  Every path formats the line here, so the fields and their spelling are
//  the same in every log. A value that does not apply or was never recorded
//  is written out (`none`, `unrecorded`) rather than left out, so a reader
//  can tell "absent" from "forgot to log".
//

import Foundation

enum RunProvenanceLine {

    /// Shown for a value no record or run carries.
    static let none = "none"
    /// Shown for a total no predecessor recorded.
    static let unrecorded = "unrecorded"
    /// How many leading hex digits of a SHA-256 the line shows.
    static let shortHashLength = 12

    /// The line for a run path with a lineage segment (GUI Play-and-Train
    /// and `--train`, corpus replay, train-vs-UCI, `--derive-model`):
    /// `record` is the segment's record as it starts — its run, parent,
    /// build, device, parameters, totals and arguments — and `seed` the
    /// run's master seed, or nil for a path that draws nothing.
    static func line(record: LineageRecord, seed: RunRandomSeed?) -> String {
        var fields = ["path=\(record.invocation.pathKind.rawValue)",
                      "run=\(record.run.lineageRunID)",
                      "seg=\(record.run.segmentIndex)",
                      "(\(origin(of: record)))"]
        fields += buildAndDevice(build: record.build, device: record.device)
        fields.append(seedFields(seed))
        fields.append("params_sha=\(record.parameters.map { short($0.sha256) } ?? Self.none)")
        fields.append("cum_step=\(total(record.steps.cumTrainerStep))")
        fields.append("cum_games=\(total(record.fed.cumGames))")
        fields.append("cum_train_sec=\(total(record.time.cumTrainStepSec))")
        fields.append("argv=\(quoted(record.invocation.argv))")
        return "[RUN] " + fields.joined(separator: " ")
    }

    /// The line for a path with no lineage segment (`--uci`, which trains
    /// nothing and writes no model file).
    static func line(pathLabel: String, build: LineageRecord.Build, device: LineageRecord.Device,
                     argv: [String]) -> String {
        var fields = ["path=\(pathLabel)", "run=\(Self.none)"]
        fields += buildAndDevice(build: build, device: device)
        fields.append(seedFields(nil))
        fields.append("argv=\(quoted(LineageRecord.redactedArguments(argv)))")
        return "[RUN] " + fields.joined(separator: " ")
    }

    // MARK: - Fields

    /// How the segment began, and from which file.
    private static func origin(of record: LineageRecord) -> String {
        let parent: String
        if let recordedParent = record.parent {
            parent = "\(recordedParent.modelID) sha=\(recordedParent.contentSHA256.map(short) ?? Self.unrecorded)"
        } else {
            parent = Self.none
        }
        switch record.run.start {
        case .fresh:
            return "fresh"
        case .branch:
            return "branch from \(parent)"
        case .derive:
            return "derived from \(parent)"
        case .resume:
            if record.run.exactResume {
                return "exact resume of \(parent)"
            }
            return "resume of \(parent), not exact: \(ResumeExactness.tokenList(record.run.notExactItems))"
        }
    }

    private static func buildAndDevice(build: LineageRecord.Build, device: LineageRecord.Device) -> [String] {
        [
            "build=\(build.buildNumber)",
            "git=\(build.gitHash)",
            "dirty=\(build.gitDirty)",
            "device=\(quoted(device.cpu ?? Self.unrecorded))",
            "vm=\(device.isVirtualMachine.map { String($0) } ?? Self.unrecorded)",
            "os=\(quoted(device.osVersion))",
        ]
    }

    private static func seedFields(_ seed: RunRandomSeed?) -> String {
        seed?.provenanceFields ?? "seed=\(Self.none)"
    }

    private static func total(_ value: Int?) -> String {
        value.map(String.init) ?? Self.unrecorded
    }

    private static func total(_ value: Double?) -> String {
        value.map { String($0) } ?? Self.unrecorded
    }

    private static func short(_ hash: String) -> String {
        String(hash.prefix(shortHashLength))
    }

    /// `text` in double quotes, with embedded quotes and backslashes
    /// escaped so the field stays one token for a log reader.
    private static func quoted(_ text: String) -> String {
        "\"" + text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }

    private static func quoted(_ argv: [String]) -> String {
        quoted(argv.joined(separator: " "))
    }
}
