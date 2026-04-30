//
//  CliTrainingConfigTests.swift
//  DrewsChessMachineTests
//
//  Pins the schema of the `--parameters <file>` JSON so downstream
//  tooling (and CLAUDE.md / scripts that pass these files in) has a
//  stable contract. The app applies each non-nil field to the matching
//  @AppStorage key on launch; if a test below starts failing because
//  a key name changed, the runtime override will silently no-op on
//  the old field name — which is the exact failure mode these tests
//  exist to catch.
//

import XCTest
@testable import DrewsChessMachine

final class CliTrainingConfigTests: XCTestCase {

    // MARK: - Full JSON round-trip

    func testAllFieldsDecode() throws {
        let json = #"""
        {
            "entropy_bonus": 1.5e-3,
            "grad_clip_max_norm": 42.0,
            "weight_decay": 2.5e-4,
            "K": 7.0,
            "learning_rate": 3e-5,
            "sqrt_batch_scaling_lr": true,
            "lr_warmup_steps": 250,
            "draw_penalty": 0.25,
            "self_play_start_tau": 1.1,
            "self_play_target_tau": 0.33,
            "self_play_tau_decay_per_ply": 0.055,
            "arena_start_tau": 0.9,
            "arena_target_tau": 0.2,
            "arena_tau_decay_per_ply": 0.04,
            "replay_ratio_target": 1.5,
            "replay_ratio_auto_adjust": false,
            "self_play_workers": 8,
            "training_step_delay_ms": 25,
            "training_batch_size": 2048,
            "replay_buffer_capacity": 500000,
            "replay_buffer_min_positions_before_training": 10000,
            "arena_promote_threshold": 0.6,
            "arena_games_per_tournament": 100,
            "arena_auto_interval_sec": 900,
            "candidate_probe_interval_sec": 30,
            "legal_mass_collapse_threshold": 0.997,
            "legal_mass_collapse_grace_seconds": 450,
            "legal_mass_collapse_no_improvement_probes": 7,
            "training_time_limit": 600
        }
        """#
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(CliTrainingConfig.self, from: data)

        XCTAssertEqual(cfg.entropyBonus, 1.5e-3)
        XCTAssertEqual(cfg.gradClipMaxNorm, 42.0)
        XCTAssertEqual(cfg.weightDecay, 2.5e-4)
        XCTAssertEqual(cfg.K, 7.0)
        XCTAssertEqual(cfg.learningRate, 3e-5)
        XCTAssertEqual(cfg.sqrtBatchScalingForLR, true)
        XCTAssertEqual(cfg.lrWarmupSteps, 250)
        XCTAssertEqual(cfg.drawPenalty, 0.25)
        XCTAssertEqual(cfg.selfPlayStartTau, 1.1)
        XCTAssertEqual(cfg.selfPlayTargetTau, 0.33)
        XCTAssertEqual(cfg.selfPlayTauDecayPerPly, 0.055)
        XCTAssertEqual(cfg.arenaStartTau, 0.9)
        XCTAssertEqual(cfg.arenaTargetTau, 0.2)
        XCTAssertEqual(cfg.arenaTauDecayPerPly, 0.04)
        XCTAssertEqual(cfg.replayRatioTarget, 1.5)
        XCTAssertEqual(cfg.replayRatioAutoAdjust, false)
        XCTAssertEqual(cfg.selfPlayWorkers, 8)
        XCTAssertEqual(cfg.trainingStepDelayMs, 25)
        XCTAssertEqual(cfg.trainingBatchSize, 2048)
        XCTAssertEqual(cfg.replayBufferCapacity, 500000)
        XCTAssertEqual(cfg.replayBufferMinPositionsBeforeTraining, 10000)
        XCTAssertEqual(cfg.arenaPromoteThreshold, 0.6)
        XCTAssertEqual(cfg.arenaGamesPerTournament, 100)
        XCTAssertEqual(cfg.arenaAutoIntervalSec, 900)
        XCTAssertEqual(cfg.candidateProbeIntervalSec, 30)
        XCTAssertEqual(cfg.legalMassCollapseThreshold, 0.997)
        XCTAssertEqual(cfg.legalMassCollapseGraceSeconds, 450)
        XCTAssertEqual(cfg.legalMassCollapseNoImprovementProbes, 7)
        XCTAssertEqual(cfg.trainingTimeLimitSec, 600)
    }

    // MARK: - Encode / decode round-trip
    //
    // The Save Parameters menu item builds a fully-populated
    // `CliTrainingConfig` from the live UI state and writes it via
    // `encodeJSON()`. The Load Parameters menu item then reads that
    // file back via `CliTrainingConfig.load(from:)`. Both ends must
    // agree on the JSON shape and field names — this test pins the
    // contract by populating every field, encoding to JSON,
    // decoding back, and checking each value round-trips byte-for-
    // byte. If a future refactor renames a CodingKey or drops a
    // field by accident, this test fails before any user-visible
    // breakage.
    func testEncodeDecodeRoundTripPreservesEveryField() throws {
        var cfg = CliTrainingConfig()
        cfg.entropyBonus = 1.5e-3
        cfg.gradClipMaxNorm = 42.0
        cfg.weightDecay = 2.5e-4
        cfg.K = 7.0
        cfg.learningRate = 3e-5
        cfg.sqrtBatchScalingForLR = true
        cfg.lrWarmupSteps = 250
        cfg.drawPenalty = 0.25
        cfg.selfPlayStartTau = 1.1
        cfg.selfPlayTargetTau = 0.33
        cfg.selfPlayTauDecayPerPly = 0.055
        cfg.arenaStartTau = 0.9
        cfg.arenaTargetTau = 0.2
        cfg.arenaTauDecayPerPly = 0.04
        cfg.replayRatioTarget = 1.5
        cfg.replayRatioAutoAdjust = false
        cfg.selfPlayWorkers = 8
        cfg.trainingStepDelayMs = 25
        cfg.trainingBatchSize = 2048
        cfg.replayBufferCapacity = 500000
        cfg.replayBufferMinPositionsBeforeTraining = 10000
        cfg.arenaPromoteThreshold = 0.6
        cfg.arenaGamesPerTournament = 100
        cfg.arenaAutoIntervalSec = 900
        cfg.candidateProbeIntervalSec = 30
        cfg.legalMassCollapseThreshold = 0.997
        cfg.legalMassCollapseGraceSeconds = 450
        cfg.legalMassCollapseNoImprovementProbes = 7
        cfg.trainingTimeLimitSec = 600

        let data = try cfg.encodeJSON()
        let decoded = try JSONDecoder().decode(CliTrainingConfig.self, from: data)

        XCTAssertEqual(decoded.entropyBonus, cfg.entropyBonus)
        XCTAssertEqual(decoded.gradClipMaxNorm, cfg.gradClipMaxNorm)
        XCTAssertEqual(decoded.weightDecay, cfg.weightDecay)
        XCTAssertEqual(decoded.K, cfg.K)
        XCTAssertEqual(decoded.learningRate, cfg.learningRate)
        XCTAssertEqual(decoded.sqrtBatchScalingForLR, cfg.sqrtBatchScalingForLR)
        XCTAssertEqual(decoded.lrWarmupSteps, cfg.lrWarmupSteps)
        XCTAssertEqual(decoded.drawPenalty, cfg.drawPenalty)
        XCTAssertEqual(decoded.selfPlayStartTau, cfg.selfPlayStartTau)
        XCTAssertEqual(decoded.selfPlayTargetTau, cfg.selfPlayTargetTau)
        XCTAssertEqual(decoded.selfPlayTauDecayPerPly, cfg.selfPlayTauDecayPerPly)
        XCTAssertEqual(decoded.arenaStartTau, cfg.arenaStartTau)
        XCTAssertEqual(decoded.arenaTargetTau, cfg.arenaTargetTau)
        XCTAssertEqual(decoded.arenaTauDecayPerPly, cfg.arenaTauDecayPerPly)
        XCTAssertEqual(decoded.replayRatioTarget, cfg.replayRatioTarget)
        XCTAssertEqual(decoded.replayRatioAutoAdjust, cfg.replayRatioAutoAdjust)
        XCTAssertEqual(decoded.selfPlayWorkers, cfg.selfPlayWorkers)
        XCTAssertEqual(decoded.trainingStepDelayMs, cfg.trainingStepDelayMs)
        XCTAssertEqual(decoded.trainingBatchSize, cfg.trainingBatchSize)
        XCTAssertEqual(decoded.replayBufferCapacity, cfg.replayBufferCapacity)
        XCTAssertEqual(decoded.replayBufferMinPositionsBeforeTraining, cfg.replayBufferMinPositionsBeforeTraining)
        XCTAssertEqual(decoded.arenaPromoteThreshold, cfg.arenaPromoteThreshold)
        XCTAssertEqual(decoded.arenaGamesPerTournament, cfg.arenaGamesPerTournament)
        XCTAssertEqual(decoded.arenaAutoIntervalSec, cfg.arenaAutoIntervalSec)
        XCTAssertEqual(decoded.candidateProbeIntervalSec, cfg.candidateProbeIntervalSec)
        XCTAssertEqual(decoded.legalMassCollapseThreshold, cfg.legalMassCollapseThreshold)
        XCTAssertEqual(decoded.legalMassCollapseGraceSeconds, cfg.legalMassCollapseGraceSeconds)
        XCTAssertEqual(decoded.legalMassCollapseNoImprovementProbes, cfg.legalMassCollapseNoImprovementProbes)
        XCTAssertEqual(decoded.trainingTimeLimitSec, cfg.trainingTimeLimitSec)
    }

    /// `encodeJSON()` emits sorted keys. A UI-saved file produced
    /// from the same values as an autotrain-generated parameters.json
    /// should byte-diff cleanly. Pin the property explicitly so a
    /// future encoder-options change can't silently regress it.
    func testEncodeJSONUsesSortedKeys() throws {
        var cfg = CliTrainingConfig()
        cfg.K = 5
        cfg.learningRate = 1e-4
        cfg.entropyBonus = 1e-3
        let data = try cfg.encodeJSON()
        let text = String(data: data, encoding: .utf8) ?? ""
        // K, entropy_bonus, learning_rate must appear in
        // alphabetical order (E < K < l).
        let entIdx = text.range(of: "entropy_bonus")?.lowerBound
        let kIdx = text.range(of: "\"K\"")?.lowerBound
        let lrIdx = text.range(of: "learning_rate")?.lowerBound
        XCTAssertNotNil(entIdx)
        XCTAssertNotNil(kIdx)
        XCTAssertNotNil(lrIdx)
        // Sorted alphabetically: K (uppercase) < entropy_bonus
        // (lowercase 'e') < learning_rate (lowercase 'l') under
        // ASCII byte ordering, which is what JSONEncoder.sortedKeys
        // uses.
        XCTAssertLessThan(kIdx!, entIdx!)
        XCTAssertLessThan(entIdx!, lrIdx!)
    }

    /// `encodeJSON()` should omit nil-valued Optional fields rather
    /// than emit them as `null`. Synthesized `Codable` for Optional
    /// uses `encodeIfPresent`, so this is the inherited behavior —
    /// pin it so a future custom encoder doesn't quietly regress
    /// the partial-config story.
    func testEncodeJSONOmitsNilFields() throws {
        var cfg = CliTrainingConfig()
        cfg.K = 5
        cfg.learningRate = 1e-4
        let data = try cfg.encodeJSON()
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("\"K\""))
        XCTAssertTrue(text.contains("\"learning_rate\""))
        XCTAssertFalse(text.contains("entropy_bonus"))
        XCTAssertFalse(text.contains("null"))
    }

    // MARK: - Partial JSON

    func testPartialJsonLeavesMissingFieldsNil() throws {
        let json = #"""
        {
            "learning_rate": 1e-4,
            "training_time_limit": 30
        }
        """#
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(CliTrainingConfig.self, from: data)

        XCTAssertEqual(cfg.learningRate, 1e-4)
        XCTAssertEqual(cfg.trainingTimeLimitSec, 30)
        XCTAssertNil(cfg.entropyBonus)
        XCTAssertNil(cfg.gradClipMaxNorm)
        XCTAssertNil(cfg.weightDecay)
        XCTAssertNil(cfg.K)
        XCTAssertNil(cfg.sqrtBatchScalingForLR)
        XCTAssertNil(cfg.lrWarmupSteps)
        XCTAssertNil(cfg.drawPenalty)
        XCTAssertNil(cfg.selfPlayStartTau)
        XCTAssertNil(cfg.selfPlayTargetTau)
        XCTAssertNil(cfg.selfPlayTauDecayPerPly)
        XCTAssertNil(cfg.arenaStartTau)
        XCTAssertNil(cfg.arenaTargetTau)
        XCTAssertNil(cfg.arenaTauDecayPerPly)
        XCTAssertNil(cfg.replayRatioTarget)
        XCTAssertNil(cfg.replayRatioAutoAdjust)
        XCTAssertNil(cfg.selfPlayWorkers)
        XCTAssertNil(cfg.trainingStepDelayMs)
        XCTAssertNil(cfg.trainingBatchSize)
        XCTAssertNil(cfg.replayBufferCapacity)
        XCTAssertNil(cfg.replayBufferMinPositionsBeforeTraining)
        XCTAssertNil(cfg.arenaPromoteThreshold)
        XCTAssertNil(cfg.arenaGamesPerTournament)
        XCTAssertNil(cfg.arenaAutoIntervalSec)
        XCTAssertNil(cfg.candidateProbeIntervalSec)
        XCTAssertNil(cfg.legalMassCollapseThreshold)
        XCTAssertNil(cfg.legalMassCollapseGraceSeconds)
        XCTAssertNil(cfg.legalMassCollapseNoImprovementProbes)
    }

    func testEmptyJsonIsValid() throws {
        let data = "{}".data(using: .utf8)!
        let cfg = try JSONDecoder().decode(CliTrainingConfig.self, from: data)
        XCTAssertNil(cfg.learningRate)
        XCTAssertNil(cfg.trainingTimeLimitSec)
    }

    // MARK: - Error paths

    func testMalformedJsonThrows() {
        let data = "{ not json ".data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode(CliTrainingConfig.self, from: data))
    }

    func testWrongTypeThrows() {
        // `self_play_workers` is Int; a string value must throw so
        // a user typo doesn't silently round-trip as nil.
        let json = #"""
        { "self_play_workers": "eight" }
        """#
        let data = json.data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode(CliTrainingConfig.self, from: data))
    }

    func testLoadFromFileRoundTrip() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-params-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let json = #"""
        { "K": 2.0, "training_time_limit": 7 }
        """#
        try json.data(using: .utf8)!.write(to: tmp)
        let cfg = try CliTrainingConfig.load(from: tmp)
        XCTAssertEqual(cfg.K, 2.0)
        XCTAssertEqual(cfg.trainingTimeLimitSec, 7)
    }

    func testLoadMissingFileThrows() {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).json")
        XCTAssertThrowsError(try CliTrainingConfig.load(from: tmp))
    }

    // MARK: - Summary string

    func testSummaryStringOnlyIncludesNonNil() {
        var cfg = CliTrainingConfig()
        cfg.learningRate = 1e-4
        cfg.trainingTimeLimitSec = 120
        let summary = cfg.summaryString()
        XCTAssertTrue(summary.contains("learning_rate=0.0001") || summary.contains("learning_rate=1e-04"))
        XCTAssertTrue(summary.contains("training_time_limit=120"))
        XCTAssertFalse(summary.contains("entropy_bonus"))
        XCTAssertFalse(summary.contains("K="))
    }

    func testSummaryStringEmptyConfig() {
        let cfg = CliTrainingConfig()
        XCTAssertEqual(cfg.summaryString(), "(empty)")
    }
}
