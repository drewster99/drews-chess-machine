//
//  ParametersFileRoundTripTests.swift
//  DrewsChessMachineTests
//
//  A parameters file the app writes reads back to exactly the values it
//  wrote. `--create-parameters-file` / `--show-default-parameters`
//  (`TrainingParameters.defaultsJSON`) and the settings save
//  (`TrainingParameters.save(to:)`) write a `Double` with every significant
//  digit (`0.0003` as `0.00029999999999999997`); the readers
//  (`CliTrainingConfig.load`, `TrainingParameters.load(from:)`) used to parse
//  that text with `JSONSerialization`, which read `weight_decay` 0.0003 back
//  as 0.0002999999999999999 — so a run from an untouched defaults file
//  recorded, hashed and resume-compared a value nobody set.
//
//  Every test that assigns `TrainingParameters.shared` snapshots it in setUp,
//  suppresses persistence, and restores it in tearDown.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ParametersFileRoundTripTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]
    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-parameters-round-trip-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: false)
        tempDir = dir
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    /// `values` applied to the settings and read back in the singleton's own
    /// typed encoding, so a whole-number `Double` written as a JSON integer
    /// compares equal to the same `Double`.
    private func typedValues(_ values: [String: ParameterValue]) throws -> [String: ParameterValue] {
        try TrainingParameters.shared.apply(values)
        return TrainingParameters.shared.snapshot().rawValueMap()
    }

    /// Write `values` the way every parameters writer does: each value's
    /// `jsonValue` through `JSONSerialization`, pretty-printed, sorted keys.
    private func writeParametersFile(_ values: [String: ParameterValue], name: String) throws -> URL {
        var object: [String: Any] = [:]
        for (id, value) in values {
            object[id] = value.jsonValue
        }
        let url = tempDir.appendingPathComponent(name)
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url)
        return url
    }

    // MARK: - Regression

    func testTheDefaultsFileReadsBackEveryDefaultExactly() throws {
        let url = tempDir.appendingPathComponent("parameters.json")
        try TrainingParameters.defaultsJSON().write(to: url)
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertNil(config.trainingTimeLimitSec)
        XCTAssertNil(config.trainingStepLimit)
        XCTAssertEqual(Set(config.trainingParameters.keys), Set(TrainingParameters.allKeys.map { $0.id }),
                       "the defaults file holds every declared parameter and nothing else")

        let declared = Dictionary(uniqueKeysWithValues: TrainingParameters.allDefinitions.map { ($0.id, $0.defaultValue) })
        let expected = try typedValues(declared)
        let loaded = try typedValues(config.trainingParameters)
        for id in expected.keys.sorted() {
            XCTAssertEqual(loaded[id], expected[id], "\(id) reads back as the declared default")
        }
    }

    func testTheSettingsSaveAndLoadRoundTripExactly() throws {
        let p = TrainingParameters.shared
        p.weightDecay = 0.0003
        p.valueLabelSmoothingEpsilon = 0.013
        p.learningRate = 0.000123456789012345
        let before = p.snapshot().rawValueMap()
        let url = tempDir.appendingPathComponent("settings.json")
        try p.save(to: url)
        try p.apply(Dictionary(uniqueKeysWithValues: TrainingParameters.allDefinitions.map { ($0.id, $0.defaultValue) }))
        try p.load(from: url)
        let after = p.snapshot().rawValueMap()
        for id in before.keys.sorted() {
            XCTAssertEqual(after[id], before[id], "\(id) reads back exactly as saved")
        }
    }

    // MARK: - Kinds and budgets

    /// The exact reader keeps the kinds `JSONSerialization` told apart: a
    /// number written with a fraction is a `Double` even when whole, an
    /// integer is an `Int`, true/false is a `Bool` (never 1/0), and a decimal
    /// string is a `UInt64`.
    func testTheReaderKeepsEachValuesKind() throws {
        let text = #"{"a": 42.0, "b": 42, "c": true, "d": "18446744073709551615", "e": 1e-3, "f": 0.00029999999999999997}"#
        let values = try ParameterValue.parametersObject(fromJSON: Data(text.utf8))
        XCTAssertEqual(values["a"], .double(42))
        XCTAssertEqual(values["b"], .int(42))
        XCTAssertEqual(values["c"], .bool(true))
        XCTAssertEqual(values["d"], .uint64(UInt64.max))
        XCTAssertEqual(values["e"], .double(0.001))
        XCTAssertEqual(values["f"], .double(0.0003))
        XCTAssertThrowsError(try ParameterValue.parametersObject(fromJSON: Data("[1]".utf8)), "not an object")
        XCTAssertThrowsError(try ParameterValue.parametersObject(fromJSON: Data(#"{"a": null}"#.utf8)), "no kind reads null")
    }

    func testTheBudgetKeysAreReadExactlyAndNeverTruncated() throws {
        let valid = tempDir.appendingPathComponent("budgets.json")
        try Data(#"{"training_time_limit": 600, "training_step_limit": 3000.0}"#.utf8).write(to: valid)
        let config = try CliTrainingConfig.load(from: valid)
        XCTAssertEqual(config.trainingTimeLimitSec, 600)
        XCTAssertEqual(config.trainingStepLimit, 3000)
        XCTAssertTrue(config.trainingParameters.isEmpty)
        for text in [#"{"training_step_limit": 2.5}"#, #"{"training_step_limit": true}"#, #"{"training_time_limit": true}"#] {
            let url = tempDir.appendingPathComponent("budget-\(UUID().uuidString).json")
            try Data(text.utf8).write(to: url)
            XCTAssertThrowsError(try CliTrainingConfig.load(from: url), text)
        }
    }

    /// Every `Double` parameter, at many values across its declared range,
    /// written the way the app writes parameters files and read back by the
    /// `--parameters` loader, is the same `Double`.
    func testEveryDoubleParameterRoundTripsExactlyThroughTheLoader() throws {
        var rng = DCMRandom(seed: 20261006)
        let doubleDefinitions = TrainingParameters.allDefinitions.filter { $0.type == .double }
        XCTAssertFalse(doubleDefinitions.isEmpty)
        for trial in 0..<200 {
            var written: [String: Double] = [:]
            for definition in doubleDefinitions {
                let range = try XCTUnwrap(definition.doubleRange, "\(definition.id) declares a range")
                written[definition.id] = Double.random(in: range.min...range.max, using: &rng)
            }
            let url = try writeParametersFile(written.mapValues { .double($0) }, name: "fuzz-\(trial).json")
            let loaded = try CliTrainingConfig.load(from: url).trainingParameters
            for (id, value) in written {
                switch loaded[id] {
                case .double(let read):
                    XCTAssertEqual(read, value, "trial \(trial): \(id) written \(value) read \(read)")
                case .int(let read):
                    XCTAssertEqual(Double(read), value, "trial \(trial): \(id) written \(value) read \(read)")
                case let other:
                    XCTFail("trial \(trial): \(id) written \(value) read \(String(describing: other))")
                }
            }
        }
    }
}
