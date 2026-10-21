import Foundation

/// The headless commands of the model-description and Core ML export paths
/// (no window, then exit):
///
/// - `--describe-model <file> [<file> ...]` prints one JSON line per file:
///   `{"path": …, "description": {…}}` (`ModelDescription`, header-only) or
///   `{"path": …, "error": …}` for a file a load would refuse. Exits 0 when
///   every file was described, 1 when any was refused.
/// - `--export-inference-graph <file> --out <folder>` writes the file's
///   recorded inference graph, its weights and its description into a new
///   folder (`InferenceGraphExport`), the input of
///   `scripts/dcm_coreml_export.py`.
/// - `--coreml-parity <file> --coreml <model.mlpackage|.mlmodelc>
///   [--compute-units cpu|cpu-gpu|cpu-ne|all] [--tolerance <x>]` compares the
///   Core ML model with the file's MPSGraph network on the bundled test-set
///   positions, at every batch size the Core ML model accepts, and prints the
///   `CoreMLParityReport`s as JSON. With `--tolerance`, exits 3 when a raw-output
///   difference exceeds it or an output on either side is non-finite.
///
/// Usage and file errors exit 2.
enum InferenceGraphExportCLI {
    static let describeFlag = "--describe-model"
    static let exportFlag = "--export-inference-graph"
    static let parityFlag = "--coreml-parity"

    struct UsageError: LocalizedError {
        let errorDescription: String?
    }

    /// Runs and exits when one of the flags is present; returns otherwise.
    static func handleIfRequested(rawArgs: [String]) {
        let present = [describeFlag, exportFlag, parityFlag].filter(rawArgs.contains)
        guard let flag = present.first else { return }
        do {
            guard present.count == 1 else {
                throw UsageError(errorDescription: "\(present.joined(separator: " and ")) cannot be combined")
            }
            switch flag {
            case describeFlag:
                Darwin.exit(try runDescribe(rawArgs) ? 0 : 1)
            case exportFlag:
                try runExport(rawArgs)
            default:
                let outcome = try runParity(rawArgs)
                FileHandle.standardOutput.write(outcome.json)
                if let tolerance = outcome.tolerance,
                   !outcome.reports.allSatisfy({ $0.isWithin(tolerance: tolerance) }) {
                    FileHandle.standardError.write(Data(
                        "error: \(parityFlag): a difference exceeds --tolerance \(tolerance) or an output is non-finite\n".utf8))
                    Darwin.exit(3)
                }
            }
            Darwin.exit(0)
        } catch {
            FileHandle.standardError.write(Data("error: \(flag): \(text(of: error))\n".utf8))
            Darwin.exit(2)
        }
    }

    /// The error's own text: a `LocalizedError`'s description, else its
    /// `CustomStringConvertible` one (e.g. `SafetensorsModelIO.IOError`), never
    /// Foundation's generic "the operation couldn't be completed".
    static func text(of error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return String(describing: error)
    }

    /// The paths after `flag` (at least one; at most one unless `manyPaths`)
    /// and each `--name value` option in `allowed` (each at most once). Any
    /// other argument is refused. (`rawArgs` excludes the program path.)
    static func parse(_ rawArgs: [String], flag: String, manyPaths: Bool = false,
                      allowed: Set<String>) throws -> (paths: [String], options: [String: String]) {
        var paths: [String] = []
        var options: [String: String] = [:]
        var i = 0
        while i < rawArgs.count {
            let arg = rawArgs[i]
            if arg == flag {
                guard paths.isEmpty else { throw UsageError(errorDescription: "\(flag) given twice") }
                i += 1
                while i < rawArgs.count, !rawArgs[i].hasPrefix("--") {
                    paths.append(rawArgs[i])
                    i += 1
                }
                guard !paths.isEmpty else { throw UsageError(errorDescription: "\(flag) needs a model file") }
                guard manyPaths || paths.count == 1 else {
                    throw UsageError(errorDescription: "\(flag) takes one model file")
                }
                continue
            }
            guard allowed.contains(arg) else {
                throw UsageError(errorDescription: "\(flag) does not accept '\(arg)'")
            }
            guard options[arg] == nil else { throw UsageError(errorDescription: "\(arg) given twice") }
            guard i + 1 < rawArgs.count, !rawArgs[i + 1].hasPrefix("--") else {
                throw UsageError(errorDescription: "\(arg) needs a value")
            }
            options[arg] = rawArgs[i + 1]
            i += 2
        }
        guard !paths.isEmpty else { throw UsageError(errorDescription: "\(flag) needs a model file") }
        return (paths, options)
    }

    private static func fileURL(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    /// Prints one line per file; true when every file was described.
    private static func runDescribe(_ rawArgs: [String]) throws -> Bool {
        let (paths, _) = try parse(rawArgs, flag: describeFlag, manyPaths: true, allowed: [])
        var allDescribed = true
        for path in paths {
            var line: [String: Any] = ["path": path]
            do {
                line["description"] = try ModelDescription.describe(fileAt: fileURL(path)).jsonObject()
            } catch {
                line["error"] = text(of: error)
                allDescribed = false
            }
            var data = try JSONSerialization.data(withJSONObject: line, options: [.sortedKeys, .withoutEscapingSlashes])
            data.append(Data("\n".utf8))
            FileHandle.standardOutput.write(data)
        }
        return allDescribed
    }

    private static func runExport(_ rawArgs: [String]) throws {
        let (paths, options) = try parse(rawArgs, flag: exportFlag, allowed: ["--out"])
        guard let out = options["--out"] else { throw UsageError(errorDescription: "\(exportFlag) needs --out <new folder>") }
        let url = fileURL(paths[0])
        let folder = fileURL(out)
        let (description, file) = try describeAndLoad(url)
        let graph = try InferenceGraphExport.record(architecture: file.architecture)
        try InferenceGraphExport.write(graph: graph, weights: file.networkWeights, description: description, to: folder)
        FileHandle.standardOutput.write(Data(
            "wrote \(folder.path): \(graph.operations.count) operations, \(graph.weights.count) weights\n".utf8))
    }

    /// The model file at `url`, read once, as its description and its load: the
    /// two can't pair one file's description with another's weights (a rolling
    /// save renamed over the path between two reads), and a session's
    /// `champion.safetensors` takes its trainer file's tail in both, as the
    /// session load gives it (`CheckpointManager.sessionTrainerPolicyTail(forFileAt:)`).
    private static func describeAndLoad(_ url: URL) throws -> (ModelDescription, ModelCheckpointFile) {
        let data = try CheckpointManager.readModelFileBytes(at: url)
        let source = url.lastPathComponent
        let tail = CheckpointManager.isLegacyModelFile(data)
            ? nil : try CheckpointManager.sessionTrainerPolicyTail(forFileAt: url)
        let description = try ModelDescription.describe(bytes: data, source: source, sessionTrainerPolicyTail: tail)
        let file = try CheckpointManager.loadModelFile(fromBytes: data, source: source, sessionTrainerPolicyTail: tail)
        return (description, file)
    }

    struct ParityOutcome {
        let reports: [CoreMLParityReport]
        let tolerance: Float?
        let json: Data
    }

    private static func runParity(_ rawArgs: [String]) throws -> ParityOutcome {
        let (paths, options) = try parse(rawArgs, flag: parityFlag,
                                         allowed: ["--coreml", "--compute-units", "--tolerance"])
        guard let coreML = options["--coreml"] else {
            throw UsageError(errorDescription: "\(parityFlag) needs --coreml <model.mlpackage>")
        }
        var units = CoreMLParity.ComputeUnits.cpu
        if let raw = options["--compute-units"] {
            guard let parsed = CoreMLParity.ComputeUnits(rawValue: raw) else {
                throw UsageError(errorDescription: "--compute-units must be one of "
                    + CoreMLParity.ComputeUnits.allCases.map(\.rawValue).joined(separator: ", "))
            }
            units = parsed
        }
        var tolerance: Float?
        if let raw = options["--tolerance"] {
            guard let value = Float(raw), value.isFinite, value >= 0 else {
                throw UsageError(errorDescription: "--tolerance \(raw) is not a finite number >= 0")
            }
            tolerance = value
        }
        let coreMLURL = fileURL(coreML)
        let states = (LichessProbeData.largeSet + LichessProbeData.wideSet).map(\.state)
        let computeUnits = units
        // The file read and the graph build are long synchronous work: done
        // here, on the launching thread, never inside the task below.
        let (_, file) = try describeAndLoad(fileURL(paths[0]))
        let network = try ChessNetwork(arch: file.architecture, bnMode: .inference,
                                       initialization: .overwrittenByLoad)
        let weights = file.networkWeights
        let reports = try runBlocking {
            try await network.loadWeights(weights)
            return try await CoreMLParity.withModel(at: coreMLURL, computeUnits: computeUnits) { model in
                try await CoreMLParity.compare(network: network, model: model, states: states,
                                               computeUnits: computeUnits)
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var json = try encoder.encode(reports)
        json.append(Data("\n".utf8))
        return ParityOutcome(reports: reports, tolerance: tolerance, json: json)
    }
}
