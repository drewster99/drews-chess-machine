//
//  SafetensorsDecodeStrictnessTests.swift
//  DrewsChessMachineTests
//
//  A model file's identity and trainer clock are read by two paths: the full
//  decode (GUI Load Model, session resume, every CLI start) and the
//  header-only parent read (`readParentFile`, the CLI lineage paths). Both
//  must apply one rule: a file without a `model_id`, or whose
//  `training_step` is not an integer, is refused rather than read as an
//  empty identity or an absent clock, and a valid file describes the same
//  lineage parent through either path.
//

import XCTest
@testable import DrewsChessMachine

final class SafetensorsDecodeStrictnessTests: XCTestCase {

    private static let architecture = NetworkArchitecture.preset(.nt8y_3x3stem)

    private func encoded(trainingStep: Int?, schedule: TrainerScheduleState? = nil) throws -> Data {
        let arch = Self.architecture
        var weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex + $0 % 5) * 0.25 }
        }
        if schedule != nil {
            weights += arch.trainableTensorPlan().map { [Float](repeating: 0.5, count: $0.elementCount) }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: trainingStep, parentModelID: "",
                                           notes: "fixture", trainerSchedule: schedule)
        return try SafetensorsModelIO.encode(
            modelID: "20261003-1-STRC", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: schedule != nil,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: schedule?.completedTrainSteps, corpus: nil))
    }

    /// `data` with one raw metadata value replaced (nil removes the key).
    private func withRawMetadata(_ data: Data, key: String, value: String?) throws -> Data {
        let (tensors, decoded) = try SafetensorsFile.decode(data)
        var metadata = decoded
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata[key] = value
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    private func decode(_ data: Data) throws -> SafetensorsModelIO.Decoded {
        try SafetensorsModelIO.decode(data, valueHead: .asStored, source: "strictness.safetensors")
    }

    func testDecodeRefusesAMalformedTrainingStep() throws {
        let valid = try encoded(trainingStep: 12)
        for raw in ["twelve", "1.5", ""] {
            let malformed = try withRawMetadata(valid, key: SafetensorsModelIO.Key.trainingStep, value: raw)
            XCTAssertThrowsError(try decode(malformed), "training_step '\(raw)'") { error in
                XCTAssertTrue(String(describing: error).contains("training_step"), "\(error)")
            }
        }
    }

    func testDecodeRefusesAMissingModelID() throws {
        let valid = try encoded(trainingStep: 12)
        let missing = try withRawMetadata(valid, key: SafetensorsModelIO.Key.modelID, value: nil)
        XCTAssertThrowsError(try decode(missing)) { error in
            XCTAssertTrue(String(describing: error).contains("model_id"), "\(error)")
        }
    }

    func testDecodeAndHeaderReadDescribeTheSameParent() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SafetensorsDecodeStrictnessTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer {
            do { try FileManager.default.removeItem(at: directory) }
            catch { XCTFail("could not remove \(directory.path): \(error)") }
        }
        let schedule = TrainerScheduleState(completedTrainSteps: 345, lrWarmupSteps: 100, lrMomentumCycle: .disabled)
        let cases: [(String, Data)] = [
            ("plain", try encoded(trainingStep: 12)),
            ("no-step", try encoded(trainingStep: nil)),
            ("trainer", try encoded(trainingStep: 12, schedule: schedule)),
        ]
        for (name, data) in cases {
            let url = directory.appendingPathComponent("\(name).safetensors")
            try data.write(to: url)
            let decodedParent = try decode(data).file.lineageParent
            let headerParent = try SafetensorsModelIO.readParentFile(at: url)
            XCTAssertEqual(decodedParent.modelID, headerParent.modelID, name)
            XCTAssertEqual(decodedParent.trainerCompletedSteps, headerParent.trainerCompletedSteps, name)
            XCTAssertEqual(decodedParent, headerParent, name)
        }
    }
}
