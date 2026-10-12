import Foundation
import TrainingParametersMacroSupport

/// `--show-lr-schedule [--parameters <file>] [--from <step>] [--to <step>]
/// [--buckets <n>] [--csv <path>]`: the LR schedule a run would be fed, as a
/// table, without starting the GUI or a run (LR_KEYFRAMES_PLAN.md, K4).
///
/// The values come from `LRMomentumCycleReadout.values` — the function the
/// trainer's feed uses — at every step, bucketed by `LRScheduleSample`, so
/// the table is what the trainer is fed (warmup and √batch included) and the
/// HTML step tables and experiment write-ups can read it instead of
/// re-deriving the schedule. Parameters: the declared defaults, overridden
/// by `--parameters` (read through the one parameters reader, so a file with
/// the previous schedule's keys is converted as everywhere else). Never the
/// machine's saved settings: the same command prints the same table anywhere.
enum ShowLRScheduleCLI {
    static let flag = "--show-lr-schedule"

    struct Options: Equatable {
        var parametersPath: String?
        var fromStep: Int?
        var toStep: Int?
        var bucketCount = 200
        var csvPath: String?
    }

    /// Runs and exits when `flag` is present; returns otherwise.
    static func handleIfRequested(rawArgs: [String]) {
        guard rawArgs.contains(flag) else { return }
        do {
            let options = try parse(rawArgs)
            let snapshot = try snapshot(parametersPath: options.parametersPath)
            let report = try render(snapshot: snapshot, options: options)
            if let csvPath = options.csvPath {
                let url = URL(fileURLWithPath: csvPath)
                let file = try FileSafety.createNewFile(at: url)
                try file.handle.write(contentsOf: Data(report.csv.utf8))
                try file.handle.close()
                FileHandle.standardOutput.write(Data((report.header + "wrote: \(url.path)\n").utf8))
            } else {
                FileHandle.standardOutput.write(Data((report.header + report.table).utf8))
            }
            Darwin.exit(0)
        } catch {
            FileHandle.standardError.write(Data("error: \(flag): \(error.localizedDescription)\n".utf8))
            Darwin.exit(2)
        }
    }

    struct UsageError: LocalizedError {
        let errorDescription: String?
    }

    static func parse(_ rawArgs: [String]) throws -> Options {
        var options = Options()
        var i = 0
        func value(after name: String) throws -> String {
            guard i + 1 < rawArgs.count else { throw UsageError(errorDescription: "\(name) needs a value") }
            return rawArgs[i + 1]
        }
        func integer(after name: String) throws -> Int {
            let text = try value(after: name)
            guard let n = Int(text), n >= 0 else { throw UsageError(errorDescription: "\(name) \(text) is not a whole number ≥ 0") }
            return n
        }
        while i < rawArgs.count {
            let arg = rawArgs[i]
            switch arg {
            case flag: i += 1
            case "--parameters": options.parametersPath = try value(after: arg); i += 2
            case "--from": options.fromStep = try integer(after: arg); i += 2
            case "--to": options.toStep = try integer(after: arg); i += 2
            case "--buckets":
                options.bucketCount = try integer(after: arg)
                guard options.bucketCount > 0 else { throw UsageError(errorDescription: "--buckets must be at least 1") }
                i += 2
            case "--csv": options.csvPath = try value(after: arg); i += 2
            default: throw UsageError(errorDescription: "does not accept '\(arg)'")
            }
        }
        if let from = options.fromStep, let to = options.toStep, to < from {
            throw UsageError(errorDescription: "--to \(to) is before --from \(from)")
        }
        return options
    }

    /// The declared defaults, overridden by the file's values.
    static func snapshot(parametersPath: String?) throws -> TrainingParametersSnapshot {
        guard let parametersPath else { return try TrainingParametersSnapshot.declaredDefaults(overriding: [:]) }
        let data = try Data(contentsOf: URL(fileURLWithPath: parametersPath))
        var values = try ParameterValue.parametersObject(fromJSON: data) { id in
            TrainingParameters.allKeys.first { $0.id == id }?.definition.defaultValue
        }
        // Run-level keys a parameters file may carry that are not parameters.
        values.removeValue(forKey: CliTrainingConfig.trainingTimeLimitKey)
        values.removeValue(forKey: CliTrainingConfig.trainingStepLimitKey)
        return try TrainingParametersSnapshot.declaredDefaults(overriding: values)
    }

    struct Report {
        let header: String
        let table: String
        let csv: String
        let sample: LRScheduleSample
    }

    /// The default range: through two LR-cycle periods past the last
    /// keyframe, and at least 50,000 steps.
    static func render(snapshot p: TrainingParametersSnapshot, options: Options) throws -> Report {
        let cycle = p.lrMomentumCycle
        let lastKeyframe = cycle.lrKeyframes.keyframes.last?.step ?? 0
        let from = options.fromStep ?? 0
        let to = options.toStep ?? max(50_000, lastKeyframe + 2 * max(1, cycle.lrPeriodSteps))
        guard let sample = LRScheduleSample.sample(steps: from...to, bucketCount: options.bucketCount, valueAt: { step in
            let fed = LRMomentumCycleReadout.values(
                completedTrainSteps: step, lrWarmupSteps: p.lrWarmupSteps, cycle: cycle,
                staticLearningRate: Float(p.learningRate), staticMomentum: Float(p.momentumCoeff),
                batchSize: p.trainingBatchSize, sqrtBatchScaling: p.sqrtBatchScalingLR,
                sqrtScaleBaseBatchSize: ChessTrainer.sqrtScaleBaseBatchSize)
            let values = cycle.values(completedTrainSteps: step, lrWarmupSteps: p.lrWarmupSteps)
            return .init(learningRate: Double(fed.learningRate), momentum: Double(fed.momentum),
                         lrPeak: values.lrPeak, lrTrough: values.lrTrough)
        }) else {
            throw UsageError(errorDescription: "sampling was cancelled")
        }
        var header = "LR schedule: \(LRMomentumCycleLogFormat.cycleDescription(cycle))\n"
        header += "warmup \(p.lrWarmupSteps) steps · static lr \(p.learningRate) · momentum \(p.momentumCoeff) · "
            + "batch \(p.trainingBatchSize) · sqrt-batch scaling \(p.sqrtBatchScalingLR ? "on" : "off")\n"
        if let warning = LRMomentumCycleLogFormat.troughFloorWarning(cycle) {
            header += warning + "\n"
        }
        func g(_ x: Double?) -> String { x.map { String(format: "%.4g", $0) } ?? "" }
        let widths = [10, 10, 10, 10, 10, 10, 10, 7, 7]
        func row(_ cells: [String]) -> String {
            zip(cells, widths).map { cell, width in
                String(repeating: " ", count: max(0, width - cell.count)) + cell
            }.joined(separator: " ") + "\n"
        }
        var table = row(["from", "to", "peak", "trough", "lr_min", "lr_max", "lr_last", "mom_min", "mom_max"])
        var csv = "step_from,step_to,peak,trough,lr_min,lr_max,lr_last,mom_min,mom_max\n"
        for b in sample.buckets {
            let cells = [String(b.firstStep), String(b.lastStep), g(b.lrPeak), g(b.lrTrough), g(b.lrMin), g(b.lrMax),
                         g(b.lrLast), String(format: "%.4f", b.momentumMin), String(format: "%.4f", b.momentumMax)]
            table += row(cells)
            csv += cells.joined(separator: ",") + "\n"
        }
        return Report(header: header, table: table, csv: csv, sample: sample)
    }
}
