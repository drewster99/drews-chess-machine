import Darwin
import Foundation

/// Headless numerics audit of saved checkpoints (`--analyze-numerics`),
/// invoked from `DrewsChessMachineApp.init`'s pre-flight, before any SwiftUI
/// or GUI setup. Head numerics plan, Phase 0.
///
/// `--analyze-numerics <path>` accepts a `.safetensors` / `.dcmmodel` file or
/// a folder; a folder covers every weight file beneath it. Each checkpoint is
/// identified by its own metadata (model id, training step), never by its
/// filename, and audited on its own embedded architecture.
///
/// For each checkpoint: the full result goes to a JSON file (under the
/// analyses folder, or `--numerics-out <dir>`), one compact JSON line to
/// stdout, and the text summary to stderr.
enum NumericsAuditCLI {

    static func runAndExit(
        path: String, corpusShardPath: String?, outDirectory: String?, staticOnly: Bool,
        policyTailPrecision: ChessNetwork.PolicyTailPrecision
    ) -> Never {
        SessionLogger.shared.start()

        let rootURL = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let targets: [URL]
        do {
            targets = try resolveTargets(rootURL: rootURL)
        } catch {
            FileHandle.standardError.write(Data("error: --analyze-numerics: \(error.localizedDescription)\n".utf8))
            Darwin.exit(83)
        }
        guard !targets.isEmpty else {
            FileHandle.standardError.write(Data("error: --analyze-numerics found no .safetensors/.dcmmodel under \(rootURL.path)\n".utf8))
            Darwin.exit(84)
        }
        let corpusURL = corpusShardPath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let outURL = outDirectory.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? CheckpointPaths.analysesDir

        FileHandle.standardError.write(Data(
            "[NUMERICS] \(targets.count) checkpoint(s)\(staticOnly ? ", static checks only" : ", policy tail \(policyTailPrecision.rawValue)")\(corpusURL.map { ", corpus \($0.path)" } ?? ", no corpus shard")\n".utf8
        ))

        var failures = 0
        var positionSets: [InputEncoding: NumericsAudit.PositionSet] = [:]
        for target in targets {
            do {
                // As stored: the audit measures the offset the file holds,
                // which loading for play would remove.
                let file = try CheckpointManager.loadModelFileAsStored(at: target)
                let arch = file.architecture
                var positions: NumericsAudit.PositionSet?
                if !staticOnly {
                    if let cached = positionSets[arch.inputEncoding] {
                        positions = cached
                    } else {
                        let built = try NumericsAudit.buildPositionSet(
                            encoding: arch.inputEncoding,
                            corpusShardURL: corpusURL,
                            lichessDirectory: .standard
                        )
                        positionSets[arch.inputEncoding] = built
                        positions = built
                    }
                }
                let planCount = arch.weightTensorPlan().count
                // Trainer files carry optimizer velocity after the plan's
                // tensors; the format/offset checks cover the model's own
                // weights, and the layer-health velocity checks read the rest.
                let weights = Array(file.weights.prefix(planCount))
                let velocity: LayerHealth.VelocitySource = file.includesOptimizerVelocity
                    ? .trainerVelocity(Array(file.weights.dropFirst(planCount)))
                    : .unavailable(reason: "a model file holds no optimizer state")
                // The trainer step where the file records one, else the step
                // it states — the same kind of number the GUI's audit of the
                // live trainer reports (its trainer clock). The header value
                // and how it was read go on the JSON line beside it.
                let stepReading = file.trainingStepReading
                let stepValue = stepReading.trainerStepOrStatedStep
                let modelID = file.modelID.isEmpty ? nil : file.modelID
                let label = "file:\(target.lastPathComponent)"
                let auditPositions = positions
                let result = try syncWait {
                    let names = try await variableNames(arch: arch)
                    return try await NumericsAudit.run(
                        names: names,
                        weights: weights,
                        arch: arch,
                        masters: nil,
                        mastersNote: "a saved model file holds one set of weights",
                        velocity: velocity,
                        positions: auditPositions,
                        dynamicSkippedReason: staticOnly ? "--numerics-static-only" : nil,
                        policyTailPrecision: policyTailPrecision,
                        modelLabel: label,
                        modelID: modelID,
                        trainingStep: stepValue
                    )
                }
                let jsonURL: URL
                switch SessionController.writeNumericsAuditJSON(result: result, modelLabel: modelID ?? label, directory: outURL) {
                case .success(let url): jsonURL = url
                case .failure(let error): throw error
                }
                FileHandle.standardError.write(Data("[NUMERICS] \(target.path)\n\(result.textSummary())\n\n".utf8))
                emit(compactLine(result: result, target: target, jsonURL: jsonURL, stepReading: stepReading))
            } catch {
                failures += 1
                emit(["event": "error", "model": target.path, "error": "\(error)"])
            }
        }

        SessionLogger.shared.shutdown()
        Darwin.exit(failures == 0 ? 0 : 85)
    }

    /// Graph variable names for `arch`, from a plain network build (the
    /// names the analyzers key on).
    private static func variableNames(arch: NetworkArchitecture) async throws -> [String] {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let network = try ChessNetwork(arch: arch, bnMode: .inference, initialization: .overwrittenByLoad)
                    continuation.resume(returning: (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name })
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func resolveTargets(rootURL: URL) throws -> [URL] {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDirectory) else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: rootURL.path])
        }
        let weightExtensions: Set<String> = ["safetensors", "dcmmodel"]
        guard isDirectory.boolValue else {
            return weightExtensions.contains(rootURL.pathExtension.lowercased()) ? [rootURL] : []
        }
        guard let enumerator = fm.enumerator(at: rootURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
            throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: rootURL.path])
        }
        var found: [URL] = []
        for case let url as URL in enumerator where weightExtensions.contains(url.pathExtension.lowercased()) {
            found.append(url)
        }
        return found.sorted { $0.path < $1.path }
    }

    private static func compactLine(result: NumericsAudit.Result, target: URL, jsonURL: URL,
                                    stepReading: ModelFileStepReading) -> [String: Any] {
        var line: [String: Any] = [
            "event": "numerics",
            "model": target.path,
            "overall": result.overallVerdict.rawValue,
            "json": jsonURL.path,
            "step_basis": stepReading.basis.rawValue,
        ]
        if let id = result.modelID { line["model_id"] = id }
        if let step = result.trainingStep { line["training_step"] = step }
        if let stated = stepReading.statedTrainingStep { line["stated_training_step"] = stated }
        if let segment = stepReading.segmentStep { line["segment_step"] = segment }
        if let offset = result.staticChecks.valueHeadOffset {
            line["value_offset_ratio_to_init"] = offset.ratioToInitExpectation
        }
        if let offset = result.staticChecks.policyHeadOffset {
            line["policy_offset_ratio_to_init"] = offset.ratioToInitExpectation
        }
        let health = result.layerHealth
        line["layer_health_dead_channels"] = health.deadChannelCount
        line["layer_health_mostly_off_channels"] = health.mostlyOffChannelCount
        line["layer_health_always_on_channels"] = health.alwaysOnChannelCount
        line["layer_health_non_finite_values"] = health.nonFiniteValueCount
        if let site = health.worstRunningVarianceSite, let ratio = site.runningVarianceMaxOverMedian {
            line["layer_health_running_variance_max_over_median"] = ratio
            line["layer_health_running_variance_worst_site"] = site.site
        }
        if let se = health.squeezeExcitationFC1 {
            line["layer_health_se_zero_velocity_units"] = se.reduce(0) { $0 + $1.zeroVelocityUnitCount }
        }
        if let value = health.valueFC1 {
            line["layer_health_value_fc1_zero_velocity_units"] = value.zeroVelocityUnitCount
        }
        if let dynamic = result.dynamicChecks {
            line["positions"] = dynamic.positions.total
            line["policy_tail_precision"] = result.policyTailPrecision
            if let value = dynamic.valueHead {
                for report in value where report.format != .fp32 {
                    line["value_ties_\(report.format.rawValue)"] = report.tieFraction
                    if let delta = report.crossEntropyDeltaVsFP32 { line["value_ce_delta_\(report.format.rawValue)"] = delta }
                }
            }
            for report in dynamic.policyHead where report.format != .fp32 {
                line["policy_top2_ties_\(report.format.rawValue)"] = report.top2TieFraction
                if let kl = report.klMean { line["policy_kl_\(report.format.rawValue)"] = kl }
            }
        }
        return line
    }

    private static func emit(_ object: [String: Any]) {
        do {
            let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            guard let text = String(data: data, encoding: .utf8) else {
                FileHandle.standardError.write(Data("error: JSON bytes are not UTF-8\n".utf8))
                return
            }
            print(text)
        } catch {
            FileHandle.standardError.write(Data("error: JSON encode failed: \(error.localizedDescription)\n".utf8))
        }
    }

    /// Bridge async → sync for this pre-flight path (mirrors
    /// `ProbeModelCLI.syncWait`): the main thread waits on a semaphore while
    /// the work runs in its own task.
    private static func syncWait<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = NumericsAuditSyncBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("NumericsAuditCLI.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}

/// The result slot `syncWait` hands across threads; written once by the task
/// before the semaphore is signalled, read after the wait returns.
private final class NumericsAuditSyncBox<T>: @unchecked Sendable {
    var success: T?
    var failure: Error?
}
