//
//  SessionCheckpointSchemaExpansionTests.swift
//  DrewsChessMachineTests
//
//  Round-trip tests for the eight Optional fields added to
//  `SessionCheckpointState` to close the autotrain reproducibility
//  gap (`lr_warmup_steps`, `sqrt_batch_scaling_lr`,
//  `replay_buffer_min_positions_before_training`,
//  `arena_auto_interval_sec`, `candidate_probe_interval_sec`,
//  `legal_mass_collapse_threshold`, `legal_mass_collapse_grace_seconds`,
//  `legal_mass_collapse_no_improvement_probes`).
//
//  Two invariants are pinned here:
//
//  1. **Round-trip fidelity** — a state populated with these fields,
//     encoded to JSON, and decoded back, preserves every value
//     exactly. If a future refactor renames a field or changes its
//     CodingKey, this test catches it before any user-visible
//     breakage.
//
//  2. **Back-compat with older `.dcmsession` files** — a
//     `session.json` written before the schema expansion (with none
//     of the new keys present) decodes successfully, with the new
//     fields landing as nil. The resume path then falls through to
//     the user's current `@AppStorage` values for those fields,
//     matching the pre-expansion behavior.
//

import XCTest
@testable import DrewsChessMachine

final class SessionCheckpointSchemaExpansionTests: XCTestCase {

    // MARK: - Round-trip with the new fields populated

    func testRoundTripPreservesAllExpansionFields() throws {
        let formatVersion = SessionCheckpointState.currentFormatVersion
        let jsonText = """
        {
          "formatVersion": \(formatVersion),
          "sessionID": "test-session",
          "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400,
          "elapsedTrainingSec": 3600,
          "trainingSteps": 12345,
          "selfPlayGames": 678,
          "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280,
          "batchSize": 1024,
          "learningRate": 5.0e-5,
          "promoteThreshold": 0.55,
          "arenaGames": 200,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.4},
          "arenaTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "lrWarmupSteps": 250,
          "sqrtBatchScalingForLR": true,
          "replayBufferMinPositionsBeforeTraining": 75000,
          "arenaAutoIntervalSec": 600,
          "candidateProbeIntervalSec": 30,
          "legalMassCollapseThreshold": 0.997,
          "legalMassCollapseGraceSeconds": 450,
          "legalMassCollapseNoImprovementProbes": 7,
          "championID": "20260430-01-AAAA",
          "trainerID": "20260430-01-AAAA-1",
          "arenaHistory": []
        }
        """
        let original = try SessionCheckpointState.decode(Data(jsonText.utf8))
        XCTAssertEqual(original.lrWarmupSteps, 250)
        XCTAssertEqual(original.sqrtBatchScalingForLR, true)
        XCTAssertEqual(original.replayBufferMinPositionsBeforeTraining, 75000)
        XCTAssertEqual(original.arenaAutoIntervalSec, 600)
        XCTAssertEqual(original.candidateProbeIntervalSec, 30)
        XCTAssertEqual(original.legalMassCollapseThreshold, 0.997)
        XCTAssertEqual(original.legalMassCollapseGraceSeconds, 450)
        XCTAssertEqual(original.legalMassCollapseNoImprovementProbes, 7)

        // Encode back to disk format and decode again. Every
        // expansion field must survive the round-trip with the
        // exact same value — a missing / mistyped CodingKey would
        // round-trip as nil and trip these assertions.
        let encoded = try original.encode()
        let decoded = try SessionCheckpointState.decode(encoded)

        XCTAssertEqual(decoded.lrWarmupSteps, original.lrWarmupSteps)
        XCTAssertEqual(decoded.sqrtBatchScalingForLR, original.sqrtBatchScalingForLR)
        XCTAssertEqual(
            decoded.replayBufferMinPositionsBeforeTraining,
            original.replayBufferMinPositionsBeforeTraining
        )
        XCTAssertEqual(decoded.arenaAutoIntervalSec, original.arenaAutoIntervalSec)
        XCTAssertEqual(decoded.candidateProbeIntervalSec, original.candidateProbeIntervalSec)
        XCTAssertEqual(decoded.legalMassCollapseThreshold, original.legalMassCollapseThreshold)
        XCTAssertEqual(
            decoded.legalMassCollapseGraceSeconds,
            original.legalMassCollapseGraceSeconds
        )
        XCTAssertEqual(
            decoded.legalMassCollapseNoImprovementProbes,
            original.legalMassCollapseNoImprovementProbes
        )
    }

    // MARK: - Back-compat: older session.json files decode

    /// Pre-expansion `.dcmsession` files (no expansion keys) must
    /// still decode cleanly, with the new fields landing as nil.
    /// On resume, the live code falls through to whatever the
    /// user's current `@AppStorage` values hold for those fields,
    /// matching the original behavior. This test pins the contract
    /// so a future schema change can't quietly require migration.
    func testLegacySessionWithoutExpansionFieldsDecodes() throws {
        let formatVersion = SessionCheckpointState.currentFormatVersion
        let jsonText = """
        {
          "formatVersion": \(formatVersion),
          "sessionID": "legacy-session",
          "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400,
          "elapsedTrainingSec": 100,
          "trainingSteps": 0,
          "selfPlayGames": 0,
          "selfPlayMoves": 0,
          "trainingPositionsSeen": 0,
          "batchSize": 4096,
          "learningRate": 5.0e-5,
          "promoteThreshold": 0.55,
          "arenaGames": 100,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.03, "floorTau": 0.4},
          "arenaTau": {"startTau": 1.0, "decayPerPly": 0.01, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "legacy-champion",
          "trainerID": "legacy-trainer",
          "arenaHistory": []
        }
        """
        let state = try SessionCheckpointState.decode(Data(jsonText.utf8))

        XCTAssertNil(state.lrWarmupSteps)
        XCTAssertNil(state.sqrtBatchScalingForLR)
        XCTAssertNil(state.replayBufferMinPositionsBeforeTraining)
        XCTAssertNil(state.arenaAutoIntervalSec)
        XCTAssertNil(state.candidateProbeIntervalSec)
        XCTAssertNil(state.legalMassCollapseThreshold)
        XCTAssertNil(state.legalMassCollapseGraceSeconds)
        XCTAssertNil(state.legalMassCollapseNoImprovementProbes)
    }

    // MARK: - Cross-format compat (parameters JSON ↔ session.json)

    /// The expansion fields use camelCase keys in `session.json`
    /// (matching the existing `SessionCheckpointState` style) and
    /// snake_case keys in the parameters JSON (matching the
    /// `--parameters` CLI shape). Confirm both formats are
    /// independently decodable so the autotrain → UI → CLI
    /// round-trip remains coherent. If the two were ever unified
    /// (e.g. by switching session.json to snake_case), the
    /// session-side keys here would have to change first.
    func testCrossFormatKeysAreIndependent() throws {
        // Parameters JSON uses snake_case keys.
        let parametersJSON = #"""
        {
            "lr_warmup_steps": 30,
            "sqrt_batch_scaling_lr": false,
            "legal_mass_collapse_threshold": 0.999
        }
        """#
        let cfg = try JSONDecoder().decode(
            CliTrainingConfig.self,
            from: Data(parametersJSON.utf8)
        )
        XCTAssertEqual(cfg.lrWarmupSteps, 30)
        XCTAssertEqual(cfg.sqrtBatchScalingForLR, false)
        XCTAssertEqual(cfg.legalMassCollapseThreshold, 0.999)

        // session.json uses camelCase keys for the same logical
        // fields. Mixing snake_case here would silently fail.
        let formatVersion = SessionCheckpointState.currentFormatVersion
        let sessionJSON = """
        {
          "formatVersion": \(formatVersion),
          "sessionID": "x",
          "savedAtUnix": 0,
          "sessionStartUnix": 0,
          "elapsedTrainingSec": 0,
          "trainingSteps": 0,
          "selfPlayGames": 0,
          "selfPlayMoves": 0,
          "trainingPositionsSeen": 0,
          "batchSize": 4096,
          "learningRate": 5.0e-5,
          "promoteThreshold": 0.55,
          "arenaGames": 100,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.03, "floorTau": 0.4},
          "arenaTau": {"startTau": 1.0, "decayPerPly": 0.01, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "lrWarmupSteps": 30,
          "sqrtBatchScalingForLR": false,
          "legalMassCollapseThreshold": 0.999,
          "championID": "x",
          "trainerID": "x",
          "arenaHistory": []
        }
        """
        let state = try SessionCheckpointState.decode(Data(sessionJSON.utf8))
        XCTAssertEqual(state.lrWarmupSteps, 30)
        XCTAssertEqual(state.sqrtBatchScalingForLR, false)
        XCTAssertEqual(state.legalMassCollapseThreshold, 0.999)
    }
}
