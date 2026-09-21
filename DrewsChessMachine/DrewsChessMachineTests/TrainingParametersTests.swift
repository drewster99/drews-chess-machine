//
//  TrainingParametersTests.swift
//  DrewsChessMachineTests
//
//  Pins the contract of the centralized TrainingParameters singleton:
//    - Registry covers exactly 69 keys (one per parameter the engine exposes).
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
            69,
            "Adding/removing a TrainingParameter requires intentionally updating this count."
        )
    }

    // MARK: - Arena promotion criterion

    /// The `arena_promotion_criterion` parameter stores a raw `Int` because
    /// `ParameterType` has no enum case, so the declared range and
    /// `ArenaPromotionCriterion`'s cases are two separate declarations that
    /// can drift. If the range were ever narrower than the enum, a criterion
    /// would be unreachable through persistence; if it were wider,
    /// `init(persistedRawValue:)` would trap on a value the validator let
    /// through. Pin them together.
    func test_arenaPromotionCriterion_rangeMatchesEnumCases() {
        guard let range = ArenaPromotionCriterionParameter.definition.intRange else {
            return XCTFail("arena_promotion_criterion must declare an Int range")
        }
        let (low, high) = (range.min, range.max)
        XCTAssertEqual(low...high, ArenaPromotionCriterion.parameterRawValueRange)

        // Every value the validator accepts must have a case, and the cases
        // must be contiguous — a gap would pass validation and then trap.
        for raw in low...high {
            XCTAssertNotNil(
                ArenaPromotionCriterion(rawValue: raw),
                "raw value \(raw) is inside the declared range but has no case"
            )
        }
    }

    func test_arenaPromotionCriterion_defaultIsScoreThreshold() throws {
        let raw = try ArenaPromotionCriterionParameter.decode(
            ArenaPromotionCriterionParameter.definition.defaultValue
        )
        XCTAssertEqual(ArenaPromotionCriterion(persistedRawValue: raw), .scoreThreshold,
                       "existing sessions must keep today's promotion rule until the user opts in")
    }

    func test_arenaPromotionCriterion_logTokensAreUniqueAndStable() {
        let tokens = ArenaPromotionCriterion.allCases.map(\.logToken)
        XCTAssertEqual(Set(tokens).count, tokens.count, "log tokens must be distinguishable: \(tokens)")
        XCTAssertEqual(ArenaPromotionCriterion.scoreThreshold.logToken, "score")
        XCTAssertEqual(ArenaPromotionCriterion.sprt.logToken, "sprt")
    }

    // MARK: - SPRT parameter block

    /// The defaults must themselves form a valid `SPRTConfig`. A user who
    /// switches the criterion to SPRT and changes nothing else has to get a
    /// working test, not a throw at arena start. Built from the definitions
    /// rather than from `TrainingParameters.shared` so the assertion is about
    /// the declared defaults, not about whatever this machine has saved.
    func test_sprtDefaults_formAValidConfig() throws {
        let config = try defaultSPRTConfig()
        XCTAssertEqual(config.elo0, 0.0)
        XCTAssertEqual(config.elo1, 10.0)
        XCTAssertEqual(config.alpha, 0.05)
        XCTAssertEqual(config.beta, 0.05)
        XCTAssertEqual(config.minGames, 32)
        XCTAssertEqual(config.maxGames, 20000)
    }

    /// Per-field ranges cannot express `elo1 > elo0`, so a parameters.json
    /// that inverts the hypotheses passes every field validator. The failure
    /// has to surface when the config is built, rather than producing a test
    /// that silently never decides.
    func test_invertedHypothesesPassFieldValidationAndFailConfigConstruction() throws {
        XCTAssertNoThrow(try ArenaSPRTElo0.definition.validate(.double(30.0)))
        XCTAssertNoThrow(try ArenaSPRTElo1.definition.validate(.double(10.0)))

        let defaults = try defaultSPRTConfig()
        XCTAssertThrowsError(
            try ArenaSPRT.SPRTConfig(
                elo0: 30.0, elo1: 10.0,
                alpha: defaults.alpha, beta: defaults.beta,
                minGames: defaults.minGames, maxGames: defaults.maxGames
            )
        )
    }

    /// Same shape for the other cross-field constraint: each error rate is
    /// individually inside `0.001…0.5`, but together they leave no room
    /// between the boundaries.
    func test_errorRatesSummingToOnePassFieldValidationAndFailConfigConstruction() throws {
        XCTAssertNoThrow(try ArenaSPRTAlpha.definition.validate(.double(0.5)))
        XCTAssertNoThrow(try ArenaSPRTBeta.definition.validate(.double(0.5)))

        let defaults = try defaultSPRTConfig()
        XCTAssertThrowsError(
            try ArenaSPRT.SPRTConfig(
                elo0: defaults.elo0, elo1: defaults.elo1,
                alpha: 0.5, beta: 0.5,
                minGames: defaults.minGames, maxGames: defaults.maxGames
            )
        )
    }

    /// `0` is in range for the runaway guard and means "unbounded" — it must
    /// not be read as a cap below `minGames`.
    func test_zeroMaxGamesIsInRangeAndMeansUnbounded() throws {
        XCTAssertNoThrow(try ArenaSPRTMaxGames.definition.validate(.int(0)))

        let defaults = try defaultSPRTConfig()
        let unbounded = try ArenaSPRT.SPRTConfig(
            elo0: defaults.elo0, elo1: defaults.elo1,
            alpha: defaults.alpha, beta: defaults.beta,
            minGames: defaults.minGames, maxGames: 0
        )
        XCTAssertEqual(unbounded.maxGames, 0)
    }

    /// Exercises the snapshot accessor block end to end: the seven stored
    /// properties, `collectValues`, and `arenaSPRTConfig()`. Mutates the
    /// singleton, so it saves and restores every value it touches, in the
    /// style of `TrainingSettingsCycleValidationTests`.
    @MainActor
    func test_snapshotCarriesTheSPRTBlockIntoAConfig() throws {
        let p = TrainingParameters.shared
        let saved = (
            criterion: p.arenaPromotionCriterion,
            elo0: p.arenaSPRTElo0, elo1: p.arenaSPRTElo1,
            alpha: p.arenaSPRTAlpha, beta: p.arenaSPRTBeta,
            minGames: p.arenaSPRTMinGames, maxGames: p.arenaSPRTMaxGames
        )
        defer {
            p.arenaPromotionCriterion = saved.criterion
            p.arenaSPRTElo0 = saved.elo0
            p.arenaSPRTElo1 = saved.elo1
            p.arenaSPRTAlpha = saved.alpha
            p.arenaSPRTBeta = saved.beta
            p.arenaSPRTMinGames = saved.minGames
            p.arenaSPRTMaxGames = saved.maxGames
        }

        p.arenaPromotionCriterion = .sprt
        p.arenaSPRTElo0 = -5.0
        p.arenaSPRTElo1 = 25.0
        p.arenaSPRTAlpha = 0.01
        p.arenaSPRTBeta = 0.2
        p.arenaSPRTMinGames = 64
        p.arenaSPRTMaxGames = 0

        let snapshot = p.snapshot()
        XCTAssertEqual(snapshot.arenaPromotionCriterion, .sprt)

        let config = try snapshot.arenaSPRTConfig()
        XCTAssertEqual(config.elo0, -5.0)
        XCTAssertEqual(config.elo1, 25.0)
        XCTAssertEqual(config.alpha, 0.01)
        XCTAssertEqual(config.beta, 0.2)
        XCTAssertEqual(config.minGames, 64)
        XCTAssertEqual(config.maxGames, 0)
    }

    /// Every SPRT knob is snapshotted at arena start; the likelihood ratio is
    /// only meaningful against hypotheses that hold for the whole test, so a
    /// mid-arena edit must not reach a running tournament.
    func test_sprtParameters_areNotLiveTunable() {
        let ids: Set<String> = [
            ArenaPromotionCriterionParameter.id,
            ArenaSPRTElo0.id, ArenaSPRTElo1.id,
            ArenaSPRTAlpha.id, ArenaSPRTBeta.id,
            ArenaSPRTMinGames.id, ArenaSPRTMaxGames.id
        ]
        XCTAssertEqual(ids.count, 7, "ids must be distinct")
        var seen = 0
        for key in TrainingParameters.allKeys where ids.contains(key.id) {
            seen += 1
            XCTAssertFalse(key.definition.liveTunable, "\(key.id) must be snapshotted at arena start")
        }
        XCTAssertEqual(seen, ids.count, "registry is missing one of the SPRT parameters")
    }

    /// `SPRTConfig` built from the registry's declared defaults.
    private func defaultSPRTConfig() throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(
            elo0: try ArenaSPRTElo0.decode(ArenaSPRTElo0.definition.defaultValue),
            elo1: try ArenaSPRTElo1.decode(ArenaSPRTElo1.definition.defaultValue),
            alpha: try ArenaSPRTAlpha.decode(ArenaSPRTAlpha.definition.defaultValue),
            beta: try ArenaSPRTBeta.decode(ArenaSPRTBeta.definition.defaultValue),
            minGames: try ArenaSPRTMinGames.decode(ArenaSPRTMinGames.definition.defaultValue),
            maxGames: try ArenaSPRTMaxGames.decode(ArenaSPRTMaxGames.definition.defaultValue)
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
