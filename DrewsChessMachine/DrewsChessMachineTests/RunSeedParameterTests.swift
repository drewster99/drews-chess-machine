//
//  RunSeedParameterTests.swift
//  DrewsChessMachineTests
//
//  The run-seed settings (`random_seed_mode`, `random_seed`, `--seed`) and
//  the `UInt64` parameter value kind they need: the resolver's precedence
//  and log lines, the `--seed` parser, the mode enum pinned to its declared
//  range, and a seed above 2^53 surviving every JSON path exactly (a
//  `Double` round trip would silently change it).
//

import XCTest
@testable import DrewsChessMachine

final class RunSeedParameterTests: XCTestCase {

    /// 2^53 + 1: the first integer a `Double` cannot represent.
    private let seedAboveDoublePrecision: UInt64 = 9_007_199_254_740_993

    // MARK: - RunRandomSeed.resolve

    func test_resolve_seededModeUsesConfiguredSeed() {
        let seed = RunRandomSeed.resolve(
            mode: .seeded, configuredSeed: 1234, commandLineSeed: nil,
            drawSeed: { XCTFail("seeded mode must not draw"); return 0 })
        XCTAssertEqual(seed.masterSeed, 1234)
        XCTAssertEqual(seed.origin, .configured)
        XCTAssertEqual(seed.effectiveMode, .seeded)
    }

    func test_resolve_unseededModeDrawsAndRecordsTheDrawnSeed() {
        let seed = RunRandomSeed.resolve(
            mode: .unseeded, configuredSeed: 1234, commandLineSeed: nil, drawSeed: { 777 })
        XCTAssertEqual(seed.masterSeed, 777)
        XCTAssertEqual(seed.origin, .drawn)
        XCTAssertEqual(seed.effectiveMode, .unseeded)
        XCTAssertEqual(seed.logLines.last, "[RUN] seed=777 mode=unseeded(drawn) derivation=v1")
        XCTAssertEqual(seed.logLines.first,
                       "[PARAM] random_seed=1234 ignored: random_seed_mode=unseeded draws the run seed")
    }

    func test_resolve_commandLineSeedOverridesBothSettings() {
        for mode in RandomSeedMode.allCases {
            let seed = RunRandomSeed.resolve(
                mode: mode, configuredSeed: 1234, commandLineSeed: 42,
                drawSeed: { XCTFail("--seed must not draw"); return 0 })
            XCTAssertEqual(seed.masterSeed, 42, "mode \(mode)")
            XCTAssertEqual(seed.origin, .commandLine)
            XCTAssertEqual(seed.effectiveMode, .seeded)
            XCTAssertEqual(seed.logLines.last, "[RUN] seed=42 mode=seeded(--seed) derivation=v1")
        }
    }

    func test_logLine_configuredSeedHasNoParamLine() {
        let seed = RunRandomSeed.resolve(
            mode: .seeded, configuredSeed: UInt64.max, commandLineSeed: nil, drawSeed: { 0 })
        XCTAssertEqual(seed.logLines, ["[RUN] seed=\(UInt64.max) mode=seeded derivation=v1"])
    }

    func test_streams_deriveFromTheMasterSeed() {
        let seed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 99, commandLineSeed: nil, drawSeed: { 0 })
        XCTAssertEqual(seed.streams.generator(.sampler), DCMRandomStreams(masterSeed: 99).generator(.sampler))
    }

    // MARK: - --seed parsing

    func test_parseCommandLineSeed_acceptsTheFullUInt64Range() throws {
        XCTAssertEqual(try RunRandomSeed.parseCommandLineSeed("0"), 0)
        XCTAssertEqual(try RunRandomSeed.parseCommandLineSeed("18446744073709551615"), UInt64.max)
        XCTAssertEqual(try RunRandomSeed.parseCommandLineSeed("9007199254740993"), seedAboveDoublePrecision)
    }

    func test_parseCommandLineSeed_rejectsAnythingElse() {
        for text in ["", "-1", "18446744073709551616", "1.5", "0x10", " 5", "five"] {
            XCTAssertThrowsError(try RunRandomSeed.parseCommandLineSeed(text), "'\(text)'") { error in
                XCTAssertEqual(error as? RunRandomSeedError, .invalidCommandLineSeed(text))
            }
        }
    }

    // MARK: - Parameter declarations

    func test_randomSeedMode_rangeMatchesEnumCases() {
        guard let range = RandomSeedModeParameter.definition.intRange else {
            return XCTFail("random_seed_mode must declare an Int range")
        }
        XCTAssertEqual(range.min...range.max, RandomSeedMode.parameterRawValueRange)
    }

    func test_randomSeed_isUInt64WithTheFullRange() {
        let definition = RandomSeed.definition
        XCTAssertEqual(definition.type, .uint64)
        guard let range = definition.uint64Range else {
            return XCTFail("random_seed must declare a UInt64 range")
        }
        XCTAssertEqual(range.min, 0)
        XCTAssertEqual(range.max, UInt64.max)
        XCTAssertFalse(definition.liveTunable)
    }

    func test_randomSeed_parsedInDeclaredRange() {
        XCTAssertEqual(RandomSeed.parsedInDeclaredRange(" 18446744073709551615 "), UInt64.max)
        XCTAssertNil(RandomSeed.parsedInDeclaredRange("18446744073709551616"))
        XCTAssertNil(RandomSeed.parsedInDeclaredRange("-3"))
    }

    // MARK: - UInt64 values through JSON

    func test_uint64Value_codableRoundTripIsExact() throws {
        let value = ParameterValue.uint64(seedAboveDoublePrecision)
        let data = try JSONEncoder().encode(value)
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "\"9007199254740993\"")
        XCTAssertEqual(try JSONDecoder().decode(ParameterValue.self, from: data), value)
    }

    func test_jsonValue_decimalStringAndNumbersDecodeExactly() throws {
        XCTAssertEqual(try ParameterValue(jsonValue: "9007199254740993", id: "random_seed"),
                       .uint64(seedAboveDoublePrecision))
        XCTAssertEqual(try ParameterValue(jsonValue: NSNumber(value: UInt64.max), id: "random_seed"),
                       .uint64(UInt64.max))
        XCTAssertEqual(try ParameterValue(jsonValue: NSNumber(value: UInt64(5)), id: "random_seed"), .int(5))
        XCTAssertEqual(try RandomSeed.decode(.int(5)), 5)
        XCTAssertEqual(try RandomSeed.decode(.uint64(UInt64.max)), UInt64.max)
        XCTAssertThrowsError(try RandomSeed.decode(.int(-1)))
        XCTAssertThrowsError(try RandomSeed.decode(.double(5)))
        XCTAssertThrowsError(try ParameterValue(jsonValue: "not a number", id: "random_seed"))
    }

    func test_parametersFile_seedAboveDoublePrecisionLoadsExactly() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-run-seed-\(UUID().uuidString).json")
        defer {
            do { try FileManager.default.removeItem(at: url) } catch { XCTFail("cleanup: \(error)") }
        }
        let json = #"{"random_seed": "9007199254740993", "random_seed_mode": 1}"#
        try Data(json.utf8).write(to: url)
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(config.trainingParameters["random_seed"], .uint64(seedAboveDoublePrecision))
        XCTAssertEqual(config.trainingParameters["random_seed_mode"], .int(1))
    }

    // MARK: - GameSerialCounter

    func test_gameSerialCounter_handsOutConsecutiveSerials() {
        let counter = GameSerialCounter(firstSerial: 10)
        XCTAssertEqual(counter.next(), 10)
        XCTAssertEqual(counter.next(), 11)
        XCTAssertEqual(counter.nextSerial, 12)
    }
}
