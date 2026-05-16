//
//  CliTrainingConfigTests.swift
//  DrewsChessMachineTests
//
//  Pins the schema of the `--parameters <file>` JSON. The contract:
//    - Top-level flat snake_case keys.
//    - Each key matches a registered TrainingParameters id, except for
//      `training_time_limit` which is the session budget pulled out
//      separately by the CLI loader.
//    - JSON value type matches the TrainingParameterDefinition.type
//      (with a tolerance for ints used in double-typed parameters).
//    - Out-of-range values are rejected at apply time.
//
//  Rewritten 2026-05-01 alongside the TrainingParameters singleton
//  rewrite. The previous tests asserted per-field Codable round-trip
//  through a now-removed struct shape.
//

import XCTest
@testable import DrewsChessMachine

final class CliTrainingConfigTests: XCTestCase {

    // MARK: - Loader

    func testLoad_extractsValuesAndTimeLimit() throws {
        let json = #"""
        {
            "entropy_bonus": 0.0025,
            "grad_clip_max_norm": 42.0,
            "self_play_concurrency": 12,
            "sqrt_batch_scaling_lr": false,
            "training_time_limit": 600.0
        }
        """#
        let url = try writeTemp(json)
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try CliTrainingConfig.load(from: url)

        XCTAssertEqual(cfg.trainingTimeLimitSec, 600.0)
        XCTAssertEqual(cfg.trainingParameters.count, 4)
        XCTAssertEqual(cfg.trainingParameters["entropy_bonus"], .double(0.0025))
        XCTAssertEqual(cfg.trainingParameters["grad_clip_max_norm"], .double(42.0))
        XCTAssertEqual(cfg.trainingParameters["self_play_concurrency"], .int(12))
        XCTAssertEqual(cfg.trainingParameters["sqrt_batch_scaling_lr"], .bool(false))
    }

    func testLoad_emptyFile_yieldsEmptyMap() throws {
        let url = try writeTemp("{}")
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(cfg.trainingParameters.count, 0)
        XCTAssertNil(cfg.trainingTimeLimitSec)
    }

    func testLoad_unknownKey_passesThrough() throws {
        // Unknown ids are tolerated by the loader; they surface as
        // unknownParameter from TrainingParameters.apply later.
        let json = #"{ "this_id_does_not_exist": 1 }"#
        let url = try writeTemp(json)
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(cfg.trainingParameters["this_id_does_not_exist"], .int(1))
    }

    // MARK: - Summary

    func testSummary_sortsAndIncludesTimeLimit() {
        let cfg = CliTrainingConfig(
            trainingParameters: [
                "self_play_concurrency": .int(8),
                "entropy_bonus": .double(0.001)
            ],
            trainingTimeLimitSec: 300.0
        )
        let summary = cfg.summaryString()
        // Sorted by id for stable diffs.
        XCTAssertTrue(summary.contains("entropy_bonus=0.001"))
        XCTAssertTrue(summary.contains("self_play_concurrency=8"))
        XCTAssertTrue(summary.contains("training_time_limit=300.0"))
        // Sort order: entropy_bonus < self_play_concurrency < training_time_limit
        let idxEnt = summary.range(of: "entropy_bonus")!.lowerBound
        let idxWork = summary.range(of: "self_play_concurrency")!.lowerBound
        XCTAssertTrue(idxEnt < idxWork)
    }

    func testSummary_emptyShowsPlaceholder() {
        let cfg = CliTrainingConfig(trainingParameters: [:], trainingTimeLimitSec: nil)
        XCTAssertEqual(cfg.summaryString(), "(empty)")
    }

    // MARK: - helpers

    private func writeTemp(_ text: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
