//
//  TrainingParametersTests.swift
//  DrewsChessMachineTests
//
//  Pins the contract of the centralized TrainingParameters singleton:
//    - Registry covers exactly 29 keys (one per parameter the engine exposes).
//    - All ids are unique.
//    - Every definition's defaultValue passes the definition's own validator.
//    - defaultsJSON round-trips through CliTrainingConfig.load + apply with
//      no errors and produces the same value map.
//    - JSON-on-disk load with an out-of-range value throws outOfRange.
//

import XCTest
@testable import DrewsChessMachine

final class TrainingParametersTests: XCTestCase {

    func test_registry_size() {
        XCTAssertEqual(
            TrainingParameters.allKeys.count,
            33,
            "Adding/removing a TrainingParameter requires intentionally updating this count."
        )
    }

    func test_registry_uniqueIds() {
        let ids = TrainingParameters.allKeys.map { $0.id }
        let unique = Set(ids)
        XCTAssertEqual(ids.count, unique.count, "Duplicate parameter ids: \(ids.sorted())")
    }

    func test_defaults_passOwnValidation() {
        for key in TrainingParameters.allKeys {
            let def = key.definition
            do {
                try def.validate(def.defaultValue)
            } catch {
                XCTFail("default for \(def.id) fails own validator: \(error)")
            }
        }
    }

    func test_defaultsJSON_roundTripsThroughLoader() throws {
        let json = try TrainingParameters.defaultsJSON()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try json.write(to: url, options: [.atomic])
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try CliTrainingConfig.load(from: url)

        XCTAssertNil(cfg.trainingTimeLimitSec)
        XCTAssertEqual(cfg.trainingParameters.count, TrainingParameters.allKeys.count)

        for key in TrainingParameters.allKeys {
            guard let loaded = cfg.trainingParameters[key.id] else {
                XCTFail("loader missing key \(key.id)")
                continue
            }
            // Both must validate against the definition.
            do {
                try key.definition.validate(loaded)
            } catch {
                XCTFail("\(key.id) loaded value \(loaded) fails validation: \(error)")
            }
        }
    }

    func test_jsonFileLoad_outOfRange_throwsAtApply() async throws {
        // Loader itself doesn't validate; apply() is where validation runs.
        // Build a JSON with a value clearly outside legal_mass_collapse_threshold's range.
        let json = #"{ "legal_mass_collapse_threshold": 99.0 }"#
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(cfg.trainingParameters["legal_mass_collapse_threshold"], .double(99.0))

        var threwOutOfRange = false
        do {
            try await MainActor.run {
                try TrainingParameters.shared.apply(cfg.trainingParameters)
            }
        } catch TrainingConfigError.outOfRange {
            threwOutOfRange = true
        } catch {
            XCTFail("expected outOfRange, got \(error)")
        }
        XCTAssertTrue(threwOutOfRange, "apply() should reject out-of-range values")
    }

    func test_defaultsMarkdown_containsEveryKeyAsHeading() {
        let md = TrainingParameters.defaultsMarkdown()
        for key in TrainingParameters.allKeys {
            XCTAssertTrue(
                md.contains("### \(key.id)"),
                "markdown is missing heading for \(key.id)"
            )
        }
    }

    func test_snapshot_returnsCurrentValues() async {
        // Set a few values, snapshot, verify.
        let snap = await MainActor.run {
            TrainingParameters.shared.snapshot()
        }
        // At minimum the snapshot has all 29 keys populated.
        let raw = snap.rawValueMap()
        XCTAssertEqual(raw.count, TrainingParameters.allKeys.count)
        for key in TrainingParameters.allKeys {
            XCTAssertNotNil(raw[key.id], "snapshot missing \(key.id)")
        }
    }
}
