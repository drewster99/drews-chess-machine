import Foundation

/// The headless commands of the Core ML export path (no window, then exit):
///
/// - `--describe-model <file>` prints the file's `ModelDescription` as JSON:
///   the architecture as this build resolves it (format-version rules and
///   legacy resolutions applied), expanded blocks and weight plan.
/// - `--export-inference-graph <file> --out <folder>` writes the file's
///   recorded inference graph, its weights and its description into a new
///   folder (`InferenceGraphExport`), the input of
///   `scripts/dcm_coreml_export.py`.
/// - `--coreml-parity <file> --coreml <model.mlpackage|.mlmodelc>
///   [--compute-units cpu|cpu-gpu|cpu-ne|all] [--tolerance <x>]` compares the
///   Core ML model with the file's MPSGraph network on the bundled test-set
///   positions and prints a `CoreMLParityReport` as JSON. With `--tolerance`,
///   exits 3 when a raw-output difference exceeds it or an output is non-finite.
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
                try runDescribe(rawArgs)
            case exportFlag:
                try runExport(rawArgs)
            default:
                let report = try runParity(rawArgs)
                FileHandle.standardOutput.write(report.json)
                if let tolerance = report.tolerance, !report.report.isWithin(tolerance: tolerance) {
                    FileHandle.standardError.write(Data(
                        "error: \(parityFlag): a difference exceeds --tolerance \(tolerance) or an output is non-finite\n".utf8))
                    Darwin.exit(3)
                }
            }
            Darwin.exit(0)
        } catch {
            FileHandle.standardError.write(Data("error: \(flag): \(error.localizedDescription)\n".utf8))
            Darwin.exit(2)
        }
    }

    /// The model path after `flag`, and every `--name value` pair in
    /// `allowed`; any other argument is refused. (`rawArgs` excludes the
    /// program path.)
    static func parse(_ rawArgs: [String], flag: String, allowed: Set<String>) throws -> (path: String, options: [String: String]) {
        var path: String?
        var options: [String: String] = [:]
        var i = 0
        while i < rawArgs.count {
            let arg = rawArgs[i]
            guard arg == flag || allowed.contains(arg) else {
                throw UsageError(errorDescription: "\(flag) does not accept '\(arg)'")
            }
            guard i + 1 < rawArgs.count, !rawArgs[i + 1].hasPrefix("--") else {
                throw UsageError(errorDescription: "\(arg) needs a value")
            }
            if arg == flag { path = rawArgs[i + 1] } else { options[arg] = rawArgs[i + 1] }
            i += 2
        }
        guard let path else { throw UsageError(errorDescription: "\(flag) needs a model file") }
        return (path, options)
    }

    private static func modelURL(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    private static func runDescribe(_ rawArgs: [String]) throws {
        let (path, _) = try parse(rawArgs, flag: describeFlag, allowed: [])
        let url = modelURL(path)
        let file = try CheckpointManager.loadModelFile(at: url)
        var json = try ModelDescription(file: file, source: url.lastPathComponent).encodedJSON()
        json.append(Data("\n".utf8))
        FileHandle.standardOutput.write(json)
    }

    private static func runExport(_ rawArgs: [String]) throws {
        let (path, options) = try parse(rawArgs, flag: exportFlag, allowed: ["--out"])
        guard let out = options["--out"] else { throw UsageError(errorDescription: "\(exportFlag) needs --out <new folder>") }
        let url = modelURL(path)
        let folder = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let file = try CheckpointManager.loadModelFile(at: url)
        let graph = try InferenceGraphExport.record(architecture: file.architecture)
        try InferenceGraphExport.write(graph: graph, weights: file.networkWeights,
                                       description: ModelDescription(file: file, source: url.lastPathComponent),
                                       to: folder)
        FileHandle.standardOutput.write(Data(
            "wrote \(folder.path): \(graph.operations.count) operations, \(graph.weights.count) weights\n".utf8))
    }

    struct ParityOutcome {
        let report: CoreMLParityReport
        let tolerance: Float?
        let json: Data
    }

    private static func runParity(_ rawArgs: [String]) throws -> ParityOutcome {
        let (path, options) = try parse(rawArgs, flag: parityFlag,
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
        let modelFileURL = modelURL(path)
        let coreMLURL = modelURL(coreML)
        let states = (LichessProbeData.largeSet + LichessProbeData.wideSet).map(\.state)
        let computeUnits = units
        // The file read and the graph build are long synchronous work: done
        // here, on the launching thread, never inside the task below.
        let file = try CheckpointManager.loadModelFile(at: modelFileURL)
        let network = try ChessNetwork(arch: file.architecture, bnMode: .inference,
                                       initialization: .overwrittenByLoad)
        let weights = file.networkWeights
        let report = try runBlocking {
            try await network.loadWeights(weights)
            let model = try await CoreMLParity.loadModel(at: coreMLURL, computeUnits: computeUnits)
            return try await CoreMLParity.compare(network: network, model: model, states: states, computeUnits: computeUnits)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var json = try encoder.encode(report)
        json.append(Data("\n".utf8))
        return ParityOutcome(report: report, tolerance: tolerance, json: json)
    }
}
