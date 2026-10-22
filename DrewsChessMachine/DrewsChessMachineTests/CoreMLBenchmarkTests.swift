//
//  CoreMLBenchmarkTests.swift
//  DrewsChessMachineTests
//
//  `--coreml-benchmark` (`CoreMLBenchmark`, `InferenceGraphExportCLI+Benchmark`):
//  the config's refusals, the statistics, the rotated arm order, the input
//  ring and an end-to-end MPSGraph-only run. Core ML arms need an exported
//  package (Python + coremltools); their parity rules are
//  `CoreMLExportParityTests`' and the real run's.
//

import CoreML
import XCTest
@testable import DrewsChessMachine

final class CoreMLBenchmarkTests: XCTestCase {

    private let folder = URL(fileURLWithPath: "/bench/configs", isDirectory: true)

    private func config(_ object: [String: Any]) throws -> CoreMLBenchmark.Config {
        try CoreMLBenchmark.Config.parse(try JSONSerialization.data(withJSONObject: object), relativeTo: folder)
    }

    private var validObject: [String: Any] {
        [
            "batch_sizes": [64, 1, 8],
            "in_flight": [4, 1],
            "rounds": 3,
            "round_seconds": 1.5,
            "min_top_move_agreement": 0.95,
            "arms": [
                ["label": "mps", "engine": "mpsgraph", "model": "models/a.safetensors"],
                ["label": "ne", "engine": "coreml", "model": "/abs/a.safetensors", "coreml": "~/pkg.mlpackage",
                 "compute_units": "cpu-ne"],
            ],
        ]
    }

    /// Protects: a valid config parses to sorted sizes, resolved paths and
    /// typed backends, with Debug builds refused by default.
    /// Fails if: sizes stop being sorted, relative paths stop resolving
    /// against the config's folder, `~` stops expanding, or
    /// `allow_debug_build` stops defaulting to false when absent.
    func testAValidConfigParses() throws {
        let parsed = try config(validObject)
        XCTAssertEqual(parsed.batchSizes, [1, 8, 64])
        XCTAssertEqual(parsed.inFlight, [1, 4])
        XCTAssertEqual(parsed.rounds, 3)
        XCTAssertEqual(parsed.roundSeconds, 1.5)
        XCTAssertFalse(parsed.allowDebugBuild)
        XCTAssertEqual(parsed.arms[0], CoreMLBenchmark.Arm(
            label: "mps", model: URL(fileURLWithPath: "/bench/configs/models/a.safetensors"), backend: .mpsGraph))
        XCTAssertEqual(parsed.arms[1], CoreMLBenchmark.Arm(
            label: "ne", model: URL(fileURLWithPath: "/abs/a.safetensors"),
            backend: .coreML(package: URL(fileURLWithPath: ("~/pkg.mlpackage" as NSString).expandingTildeInPath),
                             computeUnits: .cpuAndNeuralEngine)))
    }

    /// Protects: every malformed config is refused with a reason, never run
    /// with a silently substituted value.
    /// Fails if: any of these edge cases stops being refused.
    func testMalformedConfigsAreRefused() throws {
        let armsKey = "arms"
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("unknown top-level key", { $0["round_secs"] = 1 }),
            ("missing rounds", { $0.removeValue(forKey: "rounds") }),
            ("zero rounds", { $0["rounds"] = 0 }),
            ("fractional rounds", { $0["rounds"] = 2.5 }),
            ("boolean rounds", { $0["rounds"] = true }),
            ("zero round_seconds", { $0["round_seconds"] = 0 }),
            ("round_seconds above the cap", { $0["round_seconds"] = 61 }),
            ("agreement above 1", { $0["min_top_move_agreement"] = 1.5 }),
            ("empty batch sizes", { $0["batch_sizes"] = [Int]() }),
            ("batch size 0", { $0["batch_sizes"] = [0, 8] }),
            ("batch size listed twice", { $0["batch_sizes"] = [8, 8] }),
            ("in-flight above the cap", { $0["in_flight"] = [1, 17] }),
            ("non-boolean allow_debug_build", { $0["allow_debug_build"] = 1 }),
            ("empty arms", { $0[armsKey] = [[String: Any]]() }),
            ("duplicate label", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[1]["label"] = "mps"
                object[armsKey] = arms
            }),
            ("unknown engine", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[0]["engine"] = "metal"
                object[armsKey] = arms
            }),
            ("Core ML key on an MPSGraph arm", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[0]["compute_units"] = "all"
                object[armsKey] = arms
            }),
            ("Core ML arm without compute units", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[1].removeValue(forKey: "compute_units")
                object[armsKey] = arms
            }),
            ("Core ML arm without a package", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[1].removeValue(forKey: "coreml")
                object[armsKey] = arms
            }),
            ("unknown compute units", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[1]["compute_units"] = "gpu"
                object[armsKey] = arms
            }),
            ("empty label", { object in
                var arms = object[armsKey] as? [[String: Any]] ?? []
                arms[0]["label"] = ""
                object[armsKey] = arms
            }),
        ]
        for (name, edit) in cases {
            var object = validObject
            edit(&object)
            XCTAssertThrowsError(try config(object), name) { error in
                XCTAssertTrue(error is CoreMLBenchmark.ConfigError, "\(name): \(error)")
            }
        }
        XCTAssertThrowsError(try CoreMLBenchmark.Config.parse(Data("[1]".utf8), relativeTo: folder))
        XCTAssertThrowsError(try CoreMLBenchmark.Config.parse(Data("{".utf8), relativeTo: folder))
    }

    /// Protects: the latency percentiles reported per cell.
    /// Fails if: the interpolation, the ends, or the one-value case change.
    func testQuantileInterpolatesBetweenRanks() {
        XCTAssertEqual(CoreMLBenchmark.quantile([5], 0.9), 5)
        XCTAssertEqual(CoreMLBenchmark.quantile([1, 2, 3, 4], 0), 1)
        XCTAssertEqual(CoreMLBenchmark.quantile([1, 2, 3, 4], 1), 4)
        XCTAssertEqual(CoreMLBenchmark.quantile([1, 2, 3, 4], 0.5), 2.5)
        XCTAssertEqual(CoreMLBenchmark.quantile([0, 10], 0.1), 1, accuracy: 1e-12)
    }

    /// Protects: every arm leads one round in turn, so drift is spread evenly.
    /// Fails if: the rotation stops covering every arm each round or stops
    /// changing the leader.
    func testArmOrderRotatesTheLeader() {
        XCTAssertEqual(CoreMLBenchmark.armOrder(armCount: 3, round: 0), [0, 1, 2])
        XCTAssertEqual(CoreMLBenchmark.armOrder(armCount: 3, round: 1), [1, 2, 0])
        XCTAssertEqual(CoreMLBenchmark.armOrder(armCount: 3, round: 5), [2, 0, 1])
        XCTAssertEqual(CoreMLBenchmark.armOrder(armCount: 1, round: 7), [0])
    }

    /// Protects: a cell's summary over its rounds.
    /// Fails if: latencies stop pooling across rounds, the throughput median
    /// or spread is computed from the wrong rounds, or the worst thermal state
    /// isn't the most severe one.
    func testACellSummarizesItsRounds() {
        let rounds = [
            CoreMLBenchmark.RoundMeasurement(latencies: [0.002, 0.004], positions: 100, wallSeconds: 1,
                                             thermalState: .nominal),
            CoreMLBenchmark.RoundMeasurement(latencies: [0.001], positions: 200, wallSeconds: 1, thermalState: .serious),
            CoreMLBenchmark.RoundMeasurement(latencies: [0.003], positions: 150, wallSeconds: 1, thermalState: .fair),
        ]
        let cell = CoreMLBenchmark.CellResult(arm: "a", batchSize: 8, inFlight: 1, rounds: rounds)
        XCTAssertEqual(cell.calls, 4)
        XCTAssertEqual(cell.rounds, 3)
        XCTAssertEqual(cell.latencyMinMs, 1, accuracy: 1e-9)
        XCTAssertEqual(cell.latencyMedianMs, 2.5, accuracy: 1e-9)
        XCTAssertEqual(cell.positionsPerSecondMedian, 150)
        XCTAssertEqual(cell.positionsPerSecondMin, 100)
        XCTAssertEqual(cell.positionsPerSecondMax, 200)
        XCTAssertEqual(cell.roundSpread, 2)
        XCTAssertEqual(cell.worstThermalState, "serious")
    }

    /// Protects: Core ML outputs are copied by `memcpy` only when their
    /// layout is plain row-major.
    /// Fails if: a padded stride (the Neural Engine's usual layout) is taken
    /// for contiguous, or a size-1 dimension's arbitrary stride is refused.
    func testContiguityRecognizesPaddedStrides() {
        func numbers(_ values: [Int]) -> [NSNumber] { values.map { NSNumber(value: $0) } }
        XCTAssertTrue(CoreMLBenchmark.isContiguous(shape: numbers([2, 3, 4]), strides: numbers([12, 4, 1])))
        XCTAssertFalse(CoreMLBenchmark.isContiguous(shape: numbers([2, 3]), strides: numbers([16, 1])))
        XCTAssertTrue(CoreMLBenchmark.isContiguous(shape: numbers([4, 1]), strides: numbers([1, 32])))
        XCTAssertFalse(CoreMLBenchmark.isContiguous(shape: numbers([2, 3]), strides: numbers([3])))
    }

    /// Protects: a batch starting near the ring's end reads the first
    /// positions after the last, contiguously.
    /// Fails if: the wrap copy is missing or misplaced, which would feed the
    /// last batches garbage.
    func testTheRingWrapsBatchesAroundItsEnd() {
        let encoded: [Float] = [10, 11, 20, 21, 30, 31]
        let ring = CoreMLBenchmark.BoardRing(encoded: encoded, positionCount: 3, maximumBatchSize: 5)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: ring.batch(at: 2), count: 10)),
                       [30, 31, 10, 11, 20, 21, 30, 31, 10, 11])
    }

    /// Protects: the run produces one cell per arm × batch size × in-flight
    /// count, each from every round, with every stream's calls counted, and
    /// writes its JSON only to a new file.
    /// Fails if: a cell, round or stream is dropped, first-call times stop
    /// being recorded per batch size, or `--json-out` overwrites a file.
    func testAnMPSGraphOnlyRunMeasuresEveryCell() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreml-benchmark-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        addTeardownBlock { try FileManager.default.removeItem(at: parent) }

        let arch = ArchitectureActivationSiteTests.tiny()
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float((tensorIndex * 7 + $0 % 13) % 5) * 0.01 + 0.01 }
        }
        let data = try SafetensorsModelIO.encode(
            modelID: "20261022-1-BNCH", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture"),
            weights: weights, architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        try data.write(to: parent.appendingPathComponent("tiny.safetensors"))
        let configObject: [String: Any] = [
            "batch_sizes": [1, 7], "in_flight": [1, 2], "rounds": 2, "round_seconds": 0.05,
            "min_top_move_agreement": 0.95, "allow_debug_build": true,
            "arms": [
                ["label": "first", "engine": "mpsgraph", "model": "tiny.safetensors"],
                ["label": "second", "engine": "mpsgraph", "model": "tiny.safetensors"],
            ],
        ]
        let configURL = parent.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: configObject).write(to: configURL)
        let out = parent.appendingPathComponent("result.json")

        try InferenceGraphExportCLI.runBenchmark(
            [InferenceGraphExportCLI.benchmarkFlag, configURL.path, "--json-out", out.path])

        let report = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: out)) as? [String: Any])
        let cells = try XCTUnwrap(report["cells"] as? [[String: Any]])
        XCTAssertEqual(cells.count, 2 * 2 * 2)
        for cell in cells {
            let inFlight = try XCTUnwrap(cell["in_flight"] as? Int)
            XCTAssertEqual(cell["rounds"] as? Int, 2)
            XCTAssertGreaterThanOrEqual(try XCTUnwrap(cell["calls"] as? Int), 2 * inFlight)
            XCTAssertGreaterThan(try XCTUnwrap(cell["positions_per_second_median"] as? Double), 0)
        }
        let arms = try XCTUnwrap(report["arms"] as? [[String: Any]])
        XCTAssertEqual(arms.map { $0["label"] as? String }, ["first", "second"])
        for arm in arms {
            XCTAssertEqual(Set(try XCTUnwrap(arm["first_call_ms"] as? [String: Double]).keys), ["1", "7"])
        }

        XCTAssertThrowsError(try InferenceGraphExportCLI.runBenchmark(
            [InferenceGraphExportCLI.benchmarkFlag, configURL.path, "--json-out", out.path]))
    }

    /// Protects: the text table keeps every cell row's columns under its
    /// header, whatever the arm labels' lengths.
    /// Fails if: the label column goes back to a fixed width a long label
    /// (`coreml-float16-mirror-gpu`, 25 characters) overflows, shifting that
    /// row's numbers right of the others'.
    func testTheTableAlignsRowsWithLongArmLabels() {
        let labels = ["mps", "coreml-float16-mirror-gpu"]
        let round = CoreMLBenchmark.RoundMeasurement(latencies: [0.001], positions: 64, wallSeconds: 1,
                                                     thermalState: .nominal)
        let report = InferenceGraphExportCLI.BenchmarkReport(
            build: "1", configuration: "Release", hardwareModel: nil, chip: nil, gpu: nil, gpuCores: nil,
            memoryBytes: nil, osVersion: "test", lowPowerMode: false, startedAt: "test", positions: 64,
            batchSizes: [64], inFlight: [1], rounds: 1, roundSeconds: 1, warmUpSeconds: 0,
            arms: labels.map {
                .init(label: $0, engine: "mpsgraph", model: "m", coreml: nil, computeUnits: nil, loadSeconds: 0,
                      firstCallMs: [:], outputConversions: 0, parity: nil, placement: nil)
            },
            cells: labels.map { CoreMLBenchmark.CellResult(arm: $0, batchSize: 64, inFlight: 1, rounds: [round]) })

        let lines = InferenceGraphExportCLI.table(report).split(separator: "\n", omittingEmptySubsequences: false)
        let blockStart = lines.firstIndex { $0 == "batch 64, in flight 1" }
        XCTAssertNotNil(blockStart)
        guard let blockStart else { return }
        let block = lines[(blockStart + 1)...(blockStart + 1 + labels.count)]
        XCTAssertEqual(Set(block.map(\.count)).count, 1, "header and rows differ in width:\n" + block.joined(separator: "\n"))
    }
}
