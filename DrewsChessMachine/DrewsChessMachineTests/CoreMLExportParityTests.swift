import XCTest
import Metal
@testable import DrewsChessMachine

/// End to end: a recorded inference graph, translated to Core ML by
/// `scripts/dcm_coreml_export.py`, computes what the MPSGraph network computes
/// — for every architecture option (`InferenceGraphTestArchitectures`).
///
/// Needs a Python with coremltools: `$DCM_COREML_PYTHON`, else
/// `~/.venvs/dcm-coreml/bin/python` (documentation/coreml-export.md). Without
/// one the tests skip, naming what is missing.
final class CoreMLExportParityTests: XCTestCase {
    /// fp32 Core ML against the fp32 MPSGraph differs only by float32
    /// accumulation order; the largest logit difference seen on real models
    /// is about 1e-3.
    private static let fp32Tolerance: Float = 5e-3

    private static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    private func requirePython() async throws -> URL {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let path = ProcessInfo.processInfo.environment["DCM_COREML_PYTHON"]
            ?? (NSHomeDirectory() as NSString).appendingPathComponent(".venvs/dcm-coreml/bin/python")
        guard FileManager.default.isExecutableFile(atPath: path) else {
            throw XCTSkip("no Python with coremltools at \(path) (set DCM_COREML_PYTHON)")
        }
        let check = try await runOffThePool(URL(fileURLWithPath: path), ["-I", "-c", "import coremltools"])
        guard check.status == 0 else { throw XCTSkip("\(path) cannot import coremltools: \(check.stderr)") }
        return URL(fileURLWithPath: path)
    }

    private struct ProcessResult: Sendable {
        let status: Int32
        let stderr: String
    }

    private static func run(_ executable: URL, _ arguments: [String]) throws -> ProcessResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        let errorPipe = Pipe()
        process.standardError = errorPipe
        try process.run()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return ProcessResult(status: process.terminationStatus, stderr: String(decoding: errorData, as: UTF8.self))
    }

    /// `run` on a GCD queue: waiting on a subprocess must not hold a
    /// cooperative-pool thread.
    private func runOffThePool(_ executable: URL, _ arguments: [String]) async throws -> ProcessResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do { continuation.resume(returning: try Self.run(executable, arguments)) } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func export(python: URL, graphFolder: URL, to package: URL, precision: String) async throws {
        let script = Self.repositoryRoot.appendingPathComponent("scripts/dcm_coreml_export.py").path
        let result = try await runOffThePool(
            python, ["-I", script, "--graph", graphFolder.path, "--out", package.path, "--precision", precision])
        XCTAssertEqual(result.status, 0, "exporter failed: \(result.stderr)")
    }

    /// The network's weights with every BatchNorm parameter and running
    /// statistic made different per channel (γ, β, mean around their usual
    /// ranges, variance positive). A freshly built network has γ = 1, β = 0,
    /// mean 0, variance 1 everywhere, under which swapping any two of them, or
    /// broadcasting them on the wrong axis, changes nothing.
    private func weightsWithDistinctBatchNormParameters(_ weights: [[Float]], plan: [WeightTensorSpec]) -> [[Float]] {
        var generator = DCMRandom(seed: 77)
        return zip(weights, plan).map { values, spec in
            let draw = { (low: Float, high: Float) in values.indices.map { _ in Float.random(in: low...high, using: &generator) } }
            switch spec.kind {
            case .bnAffine:
                return spec.name.hasSuffix(".weight") ? draw(0.5, 1.5) : draw(-0.5, 0.5)
            case .bnRunningStat:
                return spec.name.hasSuffix(".running_var") ? draw(0.25, 2.0) : draw(-0.75, 0.75)
            case .conv, .linear, .bias, .scalar:
                return values
            }
        }
    }

    private func makeScratchFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("CoreMLExportParityTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        addTeardownBlock {
            do { try FileManager.default.removeItem(at: folder) } catch {
                XCTFail("could not remove \(folder.path): \(error)")
            }
        }
        return folder
    }

    /// Protects: the exported Core ML model reproduces the network's policy
    /// logits, value and W/D/L probabilities, for every architecture option.
    /// Fails if: the recorder misdescribes an op (wrong operand order, padding,
    /// axis, weight layout) or the translator maps one wrongly — e.g.
    /// variance computed as the sample variance, slice on the wrong axis, a
    /// leaky slope lost, a concat in the wrong order, BatchNorm's mean, variance,
    /// γ and β swapped or broadcast on the wrong axis — at every Core ML batch
    /// size the export accepts (a reshape or slice right only at batch 1).
    func testFP32ExportMatchesTheNetworkForEveryArchitectureVariant() async throws {
        let python = try await requirePython()
        let scratch = try makeScratchFolder()
        let states = LichessProbeData.largeSet.map(\.state)
        for variant in try InferenceGraphTestArchitectures.all() {
            let network = try ChessNetwork(arch: variant.architecture, initialization: .seeded(initSeed: 33),
                                           graphRecorder: InferenceGraphRecorder())
            let graph = try XCTUnwrap(network.recordedInferenceGraph)
            let weights = weightsWithDistinctBatchNormParameters(
                try await network.exportWeights(), plan: variant.architecture.weightTensorPlan())
            try await network.loadWeights(weights)
            let folder = scratch.appendingPathComponent(variant.name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            try InferenceGraphExport.graphFileData(graph)
                .write(to: folder.appendingPathComponent(InferenceGraphExport.graphFileName))
            try InferenceGraphExport.weightsFileData(graph: graph, weights: weights)
                .write(to: folder.appendingPathComponent(InferenceGraphExport.weightsFileName))

            let package = scratch.appendingPathComponent("\(variant.name).mlpackage")
            try await export(python: python, graphFolder: folder, to: package, precision: "fp32")
            let reports = try await CoreMLParity.withModel(at: package, computeUnits: .cpu) { model in
                try await CoreMLParity.compare(network: network, model: model, states: states, computeUnits: .cpu)
            }
            XCTAssertGreaterThan(reports.count, 1, "\(variant.name): the export accepts more than batch size 1")
            for report in reports {
                XCTAssertTrue(report.isWithin(tolerance: Self.fp32Tolerance), "\(variant.name): \(report)")
            }
        }
    }
}
