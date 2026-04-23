//
//  CliTrainingRecorderTests.swift
//  DrewsChessMachineTests
//
//  Pins the JSON shape of the `--output <file>` snapshot. The top-
//  level keys and the nested field names are the public contract
//  for downstream tooling; if a field rename slips through without
//  updating both the encoder CodingKeys and these tests, parsers
//  that ingest the JSON will break silently. Every assertion below
//  checks for a specific snake_case key in the encoded dictionary
//  so any drift shows up here first.
//

import XCTest
@testable import DrewsChessMachine

final class CliTrainingRecorderTests: XCTestCase {

    // Convenience: encode the recorder's JSON to `[String: Any]` for
    // key-presence assertions. Avoids coupling tests to the exact
    // numeric types the JSONSerialization produces.
    private func writeAndDecode(
        _ recorder: CliTrainingRecorder,
        totalTrainingSeconds: Double
    ) throws -> [String: Any] {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-recorder-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try recorder.writeJSON(to: tmp, totalTrainingSeconds: totalTrainingSeconds)
        let data = try Data(contentsOf: tmp)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "test", code: 1)
        }
        return json
    }

    // MARK: - Empty run

    func testEmptyRunWritesMinimalStructure() throws {
        let r = CliTrainingRecorder()
        r.setSessionID("20260421-1-ABCD")
        let json = try writeAndDecode(r, totalTrainingSeconds: 3.25)
        XCTAssertEqual(json["total_training_seconds"] as? Double, 3.25)
        XCTAssertEqual(json["training_elapsed_seconds"] as? Double, 3.25)
        XCTAssertEqual(json["session_id"] as? String, "20260421-1-ABCD")
        XCTAssertNil(json["training_steps"] as? Int)  // no stats line → nil
        XCTAssertNil(json["positions_trained"] as? Int)
        XCTAssertEqual((json["arena_results"] as? [Any])?.count, 0)
        XCTAssertEqual((json["stats"] as? [Any])?.count, 0)
        XCTAssertEqual((json["candidate_tests"] as? [Any])?.count, 0)
    }

    // MARK: - Stats line

    func testStatsLineAppendAndTopLevelMirroring() throws {
        let r = CliTrainingRecorder()
        r.appendStats(makeStats(steps: 42, positions: 1000))
        r.appendStats(makeStats(steps: 97, positions: 2500))
        let json = try writeAndDecode(r, totalTrainingSeconds: 60)
        // Top-level `training_steps` / `positions_trained` must
        // reflect the LAST stats line appended so the JSON answers
        // "what is the run at, right now?" without the caller
        // having to scan the whole stats array.
        XCTAssertEqual(json["training_steps"] as? Int, 97)
        XCTAssertEqual(json["positions_trained"] as? Int, 2500)
        let stats = json["stats"] as? [[String: Any]]
        XCTAssertEqual(stats?.count, 2)
        XCTAssertEqual(stats?[0]["steps"] as? Int, 42)
        XCTAssertEqual(stats?[1]["steps"] as? Int, 97)
        // Spot-check snake_case key names that downstream tooling
        // greps for.
        let line = stats?[0]
        XCTAssertNotNil(line?["learning_rate"])
        XCTAssertNotNil(line?["entropy_regularization_coeff"])
        XCTAssertNotNil(line?["grad_clip_max_norm"])
        XCTAssertNotNil(line?["policy_scale_k"])
        XCTAssertNotNil(line?["self_play_start_tau"])
        XCTAssertNotNil(line?["arena_floor_tau"])
        XCTAssertNotNil(line?["ratio_target"])
        XCTAssertNotNil(line?["buffer_count"])
        XCTAssertNotNil(line?["worker_count"])
    }

    // MARK: - Arena

    func testArenaAppendShape() throws {
        let r = CliTrainingRecorder()
        r.appendArena(makeArena(index: 1, promoted: true))
        r.appendArena(makeArena(index: 2, promoted: false))
        let json = try writeAndDecode(r, totalTrainingSeconds: 100)
        let arenas = json["arena_results"] as? [[String: Any]]
        XCTAssertEqual(arenas?.count, 2)
        let first = arenas?[0]
        XCTAssertEqual(first?["index"] as? Int, 1)
        XCTAssertEqual(first?["promoted"] as? Bool, true)
        XCTAssertNotNil(first?["score"])
        XCTAssertNotNil(first?["candidate_wins"])
        XCTAssertNotNil(first?["champion_wins"])
        XCTAssertNotNil(first?["self_play_start_tau"])
        XCTAssertNotNil(first?["arena_decay_per_ply"])
        XCTAssertNotNil(first?["diversity_unique_games"])
        XCTAssertNotNil(first?["arena_promote_threshold"])
    }

    // MARK: - Candidate test

    func testCandidateTestAppendShape() throws {
        let r = CliTrainingRecorder()
        r.appendCandidateTest(makeProbe(probeIndex: 1))
        r.appendCandidateTest(makeProbe(probeIndex: 2))
        let json = try writeAndDecode(r, totalTrainingSeconds: 45)
        let probes = json["candidate_tests"] as? [[String: Any]]
        XCTAssertEqual(probes?.count, 2)
        let p = probes?[0]
        XCTAssertEqual(p?["probe_index"] as? Int, 1)
        XCTAssertEqual(p?["probe_network_target"] as? String, "candidate")
        XCTAssertNotNil(p?["inference_time_ms"])
        let vh = p?["value_head"] as? [String: Any]
        XCTAssertNotNil(vh?["output"])
        let ph = p?["policy_head"] as? [String: Any]
        let stats = ph?["policy_stats"] as? [String: Any]
        XCTAssertNotNil(stats?["sum"])
        XCTAssertNotNil(stats?["top100_sum"])
        XCTAssertNotNil(stats?["above_uniform_count"])
        XCTAssertNotNil(stats?["legal_mass_sum"])
        XCTAssertNotNil(stats?["illegal_mass_sum"])
        let topRaw = ph?["top_raw"] as? [[String: Any]]
        // 10-move spec — must be populated with the sample we built.
        XCTAssertEqual(topRaw?.count, 10)
        XCTAssertEqual(topRaw?[0]["rank"] as? Int, 1)
        XCTAssertNotNil(topRaw?[0]["from"])
        XCTAssertNotNil(topRaw?[0]["to"])
        XCTAssertNotNil(topRaw?[0]["probability"])
        XCTAssertNotNil(topRaw?[0]["is_legal"])
    }

    // MARK: - Overwrite semantics

    func testWriteOverwritesExistingFile() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-recorder-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try "pre-existing junk".data(using: .utf8)!.write(to: tmp)
        let r = CliTrainingRecorder()
        r.setSessionID("S1")
        try r.writeJSON(to: tmp, totalTrainingSeconds: 5)
        let data = try Data(contentsOf: tmp)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["session_id"] as? String, "S1")
    }

    // MARK: - Counts snapshot

    func testCountsSnapshotMatchesAppends() {
        let r = CliTrainingRecorder()
        r.appendArena(makeArena(index: 1, promoted: false))
        r.appendStats(makeStats(steps: 1, positions: 10))
        r.appendStats(makeStats(steps: 2, positions: 20))
        r.appendCandidateTest(makeProbe(probeIndex: 1))
        let c = r.countsSnapshot()
        XCTAssertEqual(c.arenas, 1)
        XCTAssertEqual(c.stats, 2)
        XCTAssertEqual(c.probes, 1)
    }

    // MARK: - Fixtures

    private func makeStats(steps: Int, positions: Int) -> CliTrainingRecorder.StatsLine {
        CliTrainingRecorder.StatsLine(
            elapsedSec: 10,
            steps: steps,
            selfPlayGames: 5,
            positionsTrained: positions,
            avgLen: 42.5,
            rollingAvgLen: 40.0,
            gameLenP50: 40,
            gameLenP95: 95,
            bufferCount: 128,
            bufferCapacity: 10_000,
            policyLoss: -0.12,
            valueLoss: 0.33,
            policyEntropy: 6.0,
            gradGlobalNorm: 2.5,
            policyHeadWeightNorm: 1.2,
            policyLogitAbsMax: 3.3,
            playedMoveProb: 0.05,
            playedMoveProbPosAdv: 0.06,
            playedMoveProbPosAdvSkipped: 1,
            playedMoveProbNegAdv: 0.04,
            playedMoveProbNegAdvSkipped: 2,
            playedMoveCondWindowSize: 100,
            legalMass: 0.8,
            top1LegalFraction: 0.6,
            valueMean: 0.01,
            valueAbsMean: 0.5,
            vBaselineDelta: 0.02,
            advMean: 0.0,
            advStd: 0.5,
            advMin: -1.2,
            advMax: 1.3,
            advFracPositive: 0.5,
            advFracSmall: 0.2,
            advP05: -1.0,
            advP50: 0.0,
            advP95: 1.0,
            spStartTau: 1.0,
            spFloorTau: 0.4,
            spDecayPerPly: 0.03,
            arStartTau: 1.0,
            arFloorTau: 0.2,
            arDecayPerPly: 0.03,
            diversityUniqueGames: 100,
            diversityGamesInWindow: 200,
            diversityUniquePercent: 50,
            diversityAvgDivergencePly: 8.5,
            ratioTarget: 1.0,
            ratioCurrent: 1.05,
            ratioProductionRate: 300,
            ratioConsumptionRate: 310,
            ratioAutoAdjust: true,
            ratioComputedDelayMs: 50,
            whiteCheckmates: 1,
            blackCheckmates: 2,
            stalemates: 0,
            fiftyMoveDraws: 1,
            threefoldRepetitionDraws: 5,
            insufficientMaterialDraws: 0,
            batchSize: 128,
            learningRate: 5e-5,
            promoteThreshold: 0.55,
            arenaGames: 200,
            workerCount: 4,
            gradClipMaxNorm: 30,
            weightDecayC: 1e-4,
            entropyRegularizationCoeff: 1e-3,
            drawPenalty: 0.1,
            policyScaleK: 5,
            buildNumber: 42,
            trainerID: "T1",
            championID: "C1"
        )
    }

    private func makeArena(index: Int, promoted: Bool) -> CliTrainingRecorder.Arena {
        CliTrainingRecorder.Arena(
            index: index,
            finishedAtStep: 1000,
            gamesPlayed: 200,
            tournamentGames: 200,
            candidateWins: 50,
            championWins: 40,
            draws: 110,
            score: 0.525,
            drawRate: 0.55,
            elo: 15,
            eloLo: -10,
            eloHi: 40,
            scoreLo: 0.48,
            scoreHi: 0.57,
            candidateWinsAsWhite: 25,
            candidateDrawsAsWhite: 55,
            candidateLossesAsWhite: 20,
            candidateWinsAsBlack: 25,
            candidateDrawsAsBlack: 55,
            candidateLossesAsBlack: 20,
            candidateScoreAsWhite: 0.525,
            candidateScoreAsBlack: 0.525,
            promoted: promoted,
            promotionKind: promoted ? "automatic" : nil,
            promotedID: promoted ? "C2" : nil,
            durationSec: 120,
            candidateID: "Cand1",
            championID: "Champ1",
            trainerID: "Trainer1",
            learningRate: 5e-5,
            promoteThreshold: 0.55,
            batchSize: 128,
            workerCount: 4,
            spStartTau: 1.0,
            spFloorTau: 0.4,
            spDecayPerPly: 0.03,
            arStartTau: 1.0,
            arFloorTau: 0.2,
            arDecayPerPly: 0.03,
            diversityUniqueGames: 150,
            diversityGamesInWindow: 200,
            diversityUniquePercent: 75,
            diversityAvgDivergencePly: 10.2,
            buildNumber: 42
        )
    }

    private func makeProbe(probeIndex: Int) -> CliTrainingRecorder.CandidateTest {
        let top10 = (0..<10).map { i in
            CliTrainingRecorder.CandidateTest.TopMove(
                rank: i + 1,
                from: "e2",
                to: "e4",
                fromRow: 6, fromCol: 4, toRow: 4, toCol: 4,
                probability: 0.1 - 0.01 * Double(i),
                isLegal: i % 2 == 0
            )
        }
        let stats = CliTrainingRecorder.CandidateTest.PolicyStats(
            sum: 1.0,
            top100Sum: 0.8,
            aboveUniformCount: 8,
            legalMoveCount: 20,
            legalUniformThreshold: 0.05,
            legalMassSum: 0.95,
            illegalMassSum: 0.05,
            min: 1e-9,
            max: 0.2
        )
        return CliTrainingRecorder.CandidateTest(
            elapsedSec: 15.0 * Double(probeIndex),
            probeIndex: probeIndex,
            probeNetworkTarget: "candidate",
            inferenceTimeMs: 12.3,
            valueHead: .init(output: 0.05),
            policyHead: .init(policyStats: stats, topRaw: top10)
        )
    }
}
