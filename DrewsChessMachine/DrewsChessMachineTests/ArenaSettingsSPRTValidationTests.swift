//
//  ArenaSettingsSPRTValidationTests.swift
//  DrewsChessMachineTests
//
//  Guards `ArenaSettingsPopoverModel`'s handling of the SPRT block.
//
//  The interesting cases here are not the per-field ranges (those mirror the
//  parameter declarations and would fail loudly) but the transactional
//  behaviour around them: a cross-field failure must block the whole save
//  including the criterion flip, since a criterion switched to SPRT alongside
//  rejected hypotheses would run the new rule against the old numbers. And
//  the SPRT fields must be validated even while the score threshold is
//  selected, because otherwise a bad value saves silently and surfaces at the
//  start of the first SPRT arena — the worst place to discover a statistics
//  misconfiguration.
//
//  Pure model/validation logic, no Metal. `save()` writes through to
//  `TrainingParameters.shared`, so every test snapshots and restores what it
//  touches.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ArenaSettingsSPRTValidationTests: XCTestCase {

    private func makeModel() -> ArenaSettingsPopoverModel {
        ArenaSettingsPopoverModel(
            maxConcurrency: 4096,
            formatDurationSpec: { "\(Int($0))s" },
            parseDurationSpec: { Double($0.replacingOccurrences(of: "s", with: "")) }
        )
    }

    /// Snapshots every parameter `save()` can reach, so a failing assertion
    /// cannot leak into the next test or into the user's real settings.
    private func withRestoredParameters(_ body: () -> Void) {
        let p = TrainingParameters.shared
        let saved = (
            criterion: p.arenaPromotionCriterion,
            elo0: p.arenaSPRTElo0, elo1: p.arenaSPRTElo1,
            alpha: p.arenaSPRTAlpha, beta: p.arenaSPRTBeta,
            minGames: p.arenaSPRTMinGames, maxGames: p.arenaSPRTMaxGames,
            games: p.arenaGamesPerTournament,
            concurrency: p.arenaConcurrency,
            interval: p.arenaAutoIntervalSec,
            threshold: p.arenaPromoteThreshold,
            startTau: p.arenaStartTau, targetTau: p.arenaTargetTau,
            decay: p.arenaTauDecayPerPly
        )
        defer {
            p.arenaPromotionCriterion = saved.criterion
            p.arenaSPRTElo0 = saved.elo0
            p.arenaSPRTElo1 = saved.elo1
            p.arenaSPRTAlpha = saved.alpha
            p.arenaSPRTBeta = saved.beta
            p.arenaSPRTMinGames = saved.minGames
            p.arenaSPRTMaxGames = saved.maxGames
            p.arenaGamesPerTournament = saved.games
            p.arenaConcurrency = saved.concurrency
            p.arenaAutoIntervalSec = saved.interval
            p.arenaPromoteThreshold = saved.threshold
            p.arenaStartTau = saved.startTau
            p.arenaTargetTau = saved.targetTau
            p.arenaTauDecayPerPly = saved.decay
        }
        body()
    }

    /// Fills every non-SPRT field with something valid so a test can isolate
    /// the SPRT block as the only possible cause of failure.
    private func seedValidNonSPRTFields(_ model: ArenaSettingsPopoverModel) {
        model.gamesText = "400"
        model.concurrencyText = "64"
        model.intervalText = "900s"
        model.promoteThresholdText = "0.530"
        model.tauStartText = "0.60"
        model.tauDecayText = "0.020"
        model.tauFloorText = "0.20"
    }

    private func seedValidSPRTFields(_ model: ArenaSettingsPopoverModel) {
        model.sprtElo0Text = "0"
        model.sprtElo1Text = "10"
        model.sprtAlphaText = "0.050"
        model.sprtBetaText = "0.050"
        model.sprtMinGamesText = "32"
        model.sprtMaxGamesText = "20000"
    }

    // MARK: - Seeding

    func test_seedFromParams_roundTripsTheCriterionAndHypotheses() {
        withRestoredParameters {
            let p = TrainingParameters.shared
            p.arenaPromotionCriterion = .sprt
            p.arenaSPRTElo0 = -5
            p.arenaSPRTElo1 = 25
            p.arenaSPRTAlpha = 0.01
            p.arenaSPRTBeta = 0.2
            p.arenaSPRTMinGames = 64
            p.arenaSPRTMaxGames = 0

            let model = makeModel()
            model.seedFromParams()

            XCTAssertEqual(model.promotionCriterion, .sprt)
            XCTAssertEqual(model.sprtElo0Text, "-5")
            XCTAssertEqual(model.sprtElo1Text, "25", "whole Elo values should not read as 25.0")
            XCTAssertEqual(model.sprtAlphaText, "0.010")
            XCTAssertEqual(model.sprtBetaText, "0.200")
            XCTAssertEqual(model.sprtMinGamesText, "64")
            XCTAssertEqual(model.sprtMaxGamesText, "0")
            XCTAssertNil(model.sprtRelationError)
        }
    }

    // MARK: - Happy path

    func test_validSPRTBlockSavesAndFlipsTheCriterion() {
        withRestoredParameters {
            let p = TrainingParameters.shared
            p.arenaPromotionCriterion = .scoreThreshold

            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.sprtElo1Text = "20"
            model.promotionCriterion = .sprt
            model.isPresented = true
            model.save()

            XCTAssertFalse(model.isPresented, "a clean save dismisses the popover")
            XCTAssertNil(model.sprtRelationError)
            XCTAssertEqual(p.arenaPromotionCriterion, .sprt)
            XCTAssertEqual(p.arenaSPRTElo1, 20)
        }
    }

    // MARK: - Cross-field failures

    /// `elo1 > elo0` cannot be expressed as a per-field range: both values are
    /// individually legal. The relation has to be reported on its own line,
    /// since reddening one of the two boxes would be picking a culprit.
    func test_invertedHypotheses_blockSaveAndReportOnTheRelationNotAField() {
        withRestoredParameters {
            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.sprtElo0Text = "30"
            model.sprtElo1Text = "10"
            model.isPresented = true
            model.save()

            XCTAssertTrue(model.isPresented, "an invalid form keeps the popover open")
            XCTAssertNotNil(model.sprtRelationError)
            XCTAssertFalse(model.sprtElo0Error, "neither field is individually out of range")
            XCTAssertFalse(model.sprtElo1Error)
        }
    }

    func test_errorRatesSummingToOne_blockSave() {
        withRestoredParameters {
            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.sprtAlphaText = "0.500"
            model.sprtBetaText = "0.500"
            model.isPresented = true
            model.save()

            XCTAssertTrue(model.isPresented)
            XCTAssertNotNil(model.sprtRelationError)
            XCTAssertFalse(model.sprtAlphaError, "0.5 is inside the declared range")
            XCTAssertFalse(model.sprtBetaError)
        }
    }

    func test_maxGamesBelowMinGames_blockSave() {
        withRestoredParameters {
            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.sprtMinGamesText = "100"
            model.sprtMaxGamesText = "50"
            model.isPresented = true
            model.save()

            XCTAssertTrue(model.isPresented)
            XCTAssertNotNil(model.sprtRelationError)
        }
    }

    /// `0` means unbounded, so it must not be read as a cap below minGames.
    func test_zeroMaxGamesIsAcceptedAsUnbounded() {
        withRestoredParameters {
            let p = TrainingParameters.shared
            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.sprtMinGamesText = "100"
            model.sprtMaxGamesText = "0"
            model.isPresented = true
            model.save()

            XCTAssertFalse(model.isPresented)
            XCTAssertNil(model.sprtRelationError)
            XCTAssertEqual(p.arenaSPRTMaxGames, 0)
        }
    }

    // MARK: - The criterion is written last

    /// A rejected hypothesis set must not leave the criterion flipped. The
    /// next arena would then run SPRT against whatever was in
    /// `TrainingParameters` before — the new rule with the old numbers.
    func test_failedSPRTValidationDoesNotFlipTheCriterion() {
        withRestoredParameters {
            let p = TrainingParameters.shared
            p.arenaPromotionCriterion = .scoreThreshold

            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.sprtElo0Text = "30"     // inverted against elo1 = 10
            model.promotionCriterion = .sprt
            model.isPresented = true
            model.save()

            XCTAssertTrue(model.isPresented)
            XCTAssertEqual(
                p.arenaPromotionCriterion, .scoreThreshold,
                "the criterion must not flip while its hypotheses are rejected"
            )
        }
    }

    /// Same rule in the other direction: a failure anywhere in the form, not
    /// just in the SPRT block, holds the criterion back.
    func test_unrelatedFieldFailureAlsoHoldsBackTheCriterion() {
        withRestoredParameters {
            let p = TrainingParameters.shared
            p.arenaPromotionCriterion = .scoreThreshold

            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.concurrencyText = "not a number"
            model.promotionCriterion = .sprt
            model.isPresented = true
            model.save()

            XCTAssertTrue(model.isPresented)
            XCTAssertTrue(model.concurrencyError)
            XCTAssertEqual(p.arenaPromotionCriterion, .scoreThreshold)
        }
    }

    // MARK: - Validated even when inactive

    /// The SPRT fields stay editable under the score threshold. Letting a bad
    /// value through because it happens to be inactive defers the failure to
    /// the start of the first SPRT arena.
    func test_sprtFieldsAreValidatedEvenUnderTheScoreThreshold() {
        withRestoredParameters {
            let model = makeModel()
            model.seedFromParams()
            seedValidNonSPRTFields(model)
            seedValidSPRTFields(model)
            model.promotionCriterion = .scoreThreshold
            model.sprtAlphaText = "5"     // far outside 0.001…0.5
            model.isPresented = true
            model.save()

            XCTAssertTrue(model.isPresented, "an out-of-range inactive field still blocks save")
            XCTAssertTrue(model.sprtAlphaError)
        }
    }

    // MARK: - Hint

    func test_criterionHintDescribesTheSelectedRule() {
        let model = makeModel()
        model.seedFromParams()
        model.gamesText = "400"
        model.promoteThresholdText = "0.530"

        model.promotionCriterion = .scoreThreshold
        XCTAssertTrue(model.criterionHint.contains("400"))
        XCTAssertTrue(model.criterionHint.contains("0.530"))

        model.promotionCriterion = .sprt
        XCTAssertTrue(
            model.criterionHint.contains("unused"),
            "the SPRT hint must say which knobs stop applying: \(model.criterionHint)"
        )
    }
}
