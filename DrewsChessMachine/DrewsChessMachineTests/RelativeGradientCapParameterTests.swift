//
//  RelativeGradientCapParameterTests.swift
//  DrewsChessMachineTests
//
//  The relative gradient cap's five parameters (plan X5): declarations and
//  absent values (a pre-feature checkpoint resumes at mode `off`), the
//  snapshot → `TrainerHyperparameters` → trainer path, the parameters-file
//  round trip, W > N refused where parameters enter a CLI run, and the
//  `--accept-inexact grad_norm_history` refusal.
//

import XCTest
@testable import DrewsChessMachine
import TrainingParametersMacroSupport

@MainActor
final class RelativeGradientCapParameterTests: XCTestCase {

    private func definition(_ id: String) throws -> TrainingParameterDefinition {
        try XCTUnwrap(TrainingParameters.allDefinitions.first { $0.id == id }, "no parameter \(id)")
    }

    func test_declarations() throws {
        let mode = try definition("relative_grad_clip_mode")
        XCTAssertEqual(mode.defaultValue, RelativeGradClipMode.encode(1))
        XCTAssertEqual(mode.intRange?.min, 0)
        XCTAssertEqual(mode.intRange?.max, 2)
        XCTAssertEqual(mode.category, "Optimizer")
        XCTAssertTrue(mode.liveTunable)
        XCTAssertEqual(try definition("relative_grad_clip_multiple").defaultValue, RelativeGradClipMultiple.encode(3.0))
        XCTAssertEqual(try definition("relative_grad_clip_window_steps").intRange?.max, 10_000)
        XCTAssertEqual(try definition("relative_grad_clip_window_steps").intRange?.min, 100)
        XCTAssertEqual(try definition("relative_grad_clip_min_history_steps").defaultValue, RelativeGradClipMinHistorySteps.encode(100))
        XCTAssertEqual(try definition("relative_grad_clip_floor").defaultValue, RelativeGradClipFloor.encode(0.5))
        for id in ["relative_grad_clip_multiple", "relative_grad_clip_window_steps",
                   "relative_grad_clip_min_history_steps", "relative_grad_clip_floor"] {
            XCTAssertEqual(try definition(id).category, "Optimizer", id)
            XCTAssertTrue(try definition(id).liveTunable, id)
        }
    }

    func test_absentValues_aPreFeatureCheckpointResumesWithTheCapOff() {
        XCTAssertEqual(RelativeGradClipMode.absentValue, .preFeature(0))
        XCTAssertEqual(TrainingParameterResolution.absentValue(of: RelativeGradClipMode.self), 0)
        XCTAssertEqual(RelativeGradClipMultiple.absentValue, .currentSetting)
        XCTAssertEqual(RelativeGradClipWindowSteps.absentValue, .currentSetting)
        XCTAssertEqual(RelativeGradClipMinHistorySteps.absentValue, .currentSetting)
        XCTAssertEqual(RelativeGradClipFloor.absentValue, .currentSetting)
        let resolved = TrainingParameterResolution.resolve(RelativeGradClipMode.self, saved: nil, current: 2)
        XCTAssertEqual(resolved.applied, 0)
    }

    func test_snapshotReachesTheTrainer() throws {
        let snapshot = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            "relative_grad_clip_mode": .int(2), "relative_grad_clip_multiple": .double(4),
            "relative_grad_clip_window_steps": .int(2000), "relative_grad_clip_min_history_steps": .int(200),
            "relative_grad_clip_floor": .double(0.25),
        ])
        let hyperparameters = try TrainerHyperparameters.validated(snapshot)
        XCTAssertEqual(hyperparameters.relativeGradientCap, try RelativeGradientCapConfiguration(
            mode: .clip, multiple: 4, windowSteps: 2000, minimumHistorySteps: 200, floor: 0.25).settings)
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), hyperparameters: hyperparameters,
                                       arch: .current, initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.relativeGradientCap, hyperparameters.relativeGradientCap)
        XCTAssertEqual(TrainerHyperparameters(currentlyAppliedTo: trainer), hyperparameters)
        XCTAssertEqual(hyperparameters.relativeGradientCap.compactDescription, "clip/k4/N2000/W200/floor0.25")
    }

    func test_minimumHistoryAboveWindow_isRefusedWhereParametersEnterARun() throws {
        let snapshot = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            "relative_grad_clip_window_steps": .int(1000), "relative_grad_clip_min_history_steps": .int(2000),
        ])
        XCTAssertThrowsError(try ReplayParams(snapshot)) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("relative_grad_clip_min_history_steps"), text)
            XCTAssertTrue(text.contains("relative_grad_clip_window_steps"), text)
        }
        XCTAssertThrowsError(try TrainerHyperparameters.validated(snapshot))
    }

    /// Review MAJOR 1: per-key validation let a parameters file (File ▸
    /// Load Parameters, GUI `--parameters`) set W > N, which then reached a
    /// trainer. The singleton refuses the pair before assigning anything.
    func test_applyRefusesMinimumHistoryAboveWindow_andAssignsNothing() throws {
        let p = TrainingParameters.shared
        let saved = p.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
        defer {
            do { try p.apply(saved) } catch { XCTFail("restore failed: \(error)") }
            TrainingParameters.suppressPersistence = false
        }
        p.relativeGradClipWindowSteps = 1000
        p.relativeGradClipMinHistorySteps = 100
        p.learningRate = 0.001
        XCTAssertThrowsError(try p.apply([
            "relative_grad_clip_min_history_steps": .int(2000), "learning_rate": .double(0.002),
        ])) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("relative_grad_clip_min_history_steps"), text)
            XCTAssertTrue(text.contains("relative_grad_clip_window_steps"), text)
        }
        XCTAssertEqual(p.relativeGradClipMinHistorySteps, 100)
        XCTAssertEqual(p.learningRate, 0.001, "a refused map assigns nothing")
        XCTAssertNoThrow(try p.apply(["relative_grad_clip_window_steps": .int(3000), "relative_grad_clip_min_history_steps": .int(2000)]))
        XCTAssertEqual(p.relativeGradClipMinHistorySteps, 2000)
    }

    /// The trainer validates its settings on every real-data step, so a pair
    /// that slipped past every entry point stops training with the error
    /// instead of crashing or clipping under a cap nobody chose.
    func test_trainerStepRefusesAnInvalidPair() async throws {
        let trainer = try RelativeGradientCapFixture.makeTrainer(
            hardMax: 15, configuration: try RelativeGradientCapFixture.configuration(.clip))
        trainer.relativeGradientCap = RelativeGradientCapSettings(
            modeRawValue: 2, multiple: 3, windowSteps: 100, minimumHistorySteps: 200, floor: 0.5)
        let buffer = RelativeGradientCapFixture.makeReplayBuffer(arch: .current)
        do {
            _ = try await trainer.trainStep(replayBuffer: buffer, batchSize: RelativeGradientCapFixture.batchSize)
            XCTFail("W > N must stop the step")
        } catch {
            XCTAssertEqual(error as? RelativeGradientCapConfigurationError,
                           .minimumHistoryAboveWindow(minimumHistorySteps: 200, windowSteps: 100))
        }
        XCTAssertEqual(trainer.completedTrainSteps, 0)
    }

    func test_parametersFileRoundTrip() throws {
        let json = try TrainingParameters.defaultsJSON()
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: json) as? [String: Any])
        XCTAssertEqual(object["relative_grad_clip_mode"] as? Int, 1)
        XCTAssertEqual(object["relative_grad_clip_multiple"] as? Double, 3)
        XCTAssertEqual(object["relative_grad_clip_window_steps"] as? Int, 1000)
        XCTAssertEqual(object["relative_grad_clip_min_history_steps"] as? Int, 100)
        XCTAssertEqual(object["relative_grad_clip_floor"] as? Double, 0.5)

        var edited = object
        edited["relative_grad_clip_mode"] = 2
        edited["relative_grad_clip_multiple"] = 4.5
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("relcap-params-\(UUID().uuidString)")
            .appendingPathExtension("json")
        let file = try FileSafety.createNewFile(at: url)
        let identity = file.identity
        addTeardownBlock {
            _ = try FileSafety.removeOwnedItem(at: url, identity: identity)
        }
        try file.handle.write(contentsOf: try JSONSerialization.data(withJSONObject: edited))
        try file.handle.close()
        let loaded = try CliTrainingConfig.load(from: url)
        let params = try ReplayParams(try TrainingParametersSnapshot.declaredDefaults(overriding: loaded.trainingParameters))
        XCTAssertEqual(params.trainer.relativeGradientCap.mode, .clip)
        XCTAssertEqual(params.trainer.relativeGradientCap.multiple, 4.5)
    }

    func test_acceptInexact_namesGradNormHistory() throws {
        let exactness = ResumeExactness(gaps: [.gradNormHistory])
        XCTAssertNotNil(exactness.refusal(accepting: []), "an unaccepted grad_norm_history gap refuses the resume")
        XCTAssertNil(exactness.refusal(accepting: try ResumeGap.parseAcceptList("grad_norm_history")))
    }
}
