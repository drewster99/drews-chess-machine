//
//  ArenaHistoryCodableTests.swift
//  DrewsChessMachineTests
//
//  Round-trip and back-compat tests for ArenaHistoryEntryCodable,
//  the on-disk representation of one arena result inside a saved
//  session's `session.json`. The back-compat story matters: when
//  this schema is extended (as it just was, adding six per-side
//  fields), legacy `.dcmsession` files written by older builds
//  must still decode without throwing and reconstruct a usable
//  TournamentRecord through the load path.
//
//  Tests cover:
//   - Round-trip: encode(entry) → JSON → decode → equal entry
//   - Forward compat: older JSON missing the new fields decodes
//     successfully with nil per-side counters
//   - Unknown keys in the JSON don't break decoding (ignored)
//   - The full load-path conversion (entry → TournamentRecord),
//     including the 0-default substitution for legacy entries
//

import XCTest
@testable import DrewsChessMachine

final class ArenaHistoryEntryCodableRoundTripTests: XCTestCase {

    private func makeFullEntry() -> ArenaHistoryEntryCodable {
        ArenaHistoryEntryCodable(
            finishedAtStep: 5000,
            candidateWins: 120,
            championWins: 45,
            draws: 35,
            score: 0.6875,
            promoted: true,
            promotedID: "20260420-3-ABCD",
            durationSec: 945.0,
            gamesPlayed: 200,
            promotionKind: "automatic",
            candidateWinsAsWhite: 65,
            candidateWinsAsBlack: 55,
            candidateLossesAsWhite: 20,
            candidateLossesAsBlack: 25,
            candidateDrawsAsWhite: 15,
            candidateDrawsAsBlack: 20
        )
    }

    func testFullEntryRoundTrip() throws {
        let original = makeFullEntry()
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testDecodeLegacyMissingPerSideFields() throws {
        // Simulate a `session.json` written before the per-side
        // fields existed. All six `candidate*As{White,Black}`
        // keys absent → must decode with nil, not throw.
        let legacyJson = """
        {
          "finishedAtStep": 5000,
          "candidateWins": 120,
          "championWins": 45,
          "draws": 35,
          "score": 0.6875,
          "promoted": true,
          "promotedID": "20260420-3-ABCD",
          "durationSec": 945.0,
          "gamesPlayed": 200,
          "promotionKind": "automatic"
        }
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: legacyJson)
        XCTAssertEqual(decoded.finishedAtStep, 5000)
        XCTAssertEqual(decoded.candidateWins, 120)
        XCTAssertEqual(decoded.draws, 35)
        XCTAssertNil(decoded.candidateWinsAsWhite)
        XCTAssertNil(decoded.candidateWinsAsBlack)
        XCTAssertNil(decoded.candidateLossesAsWhite)
        XCTAssertNil(decoded.candidateLossesAsBlack)
        XCTAssertNil(decoded.candidateDrawsAsWhite)
        XCTAssertNil(decoded.candidateDrawsAsBlack)
    }

    func testDecodeVeryLegacyMissingGamesPlayedAndPromotionKind() throws {
        // Pre-`gamesPlayed` / pre-`promotionKind` era: both
        // optional fields absent. Decoder must still succeed —
        // ContentView's load path reconstructs gamesPlayed from
        // W/L/D and defaults promotionKind to `.automatic` for
        // any entry with `promoted == true`.
        let veryLegacyJson = """
        {
          "finishedAtStep": 1000,
          "candidateWins": 30,
          "championWins": 20,
          "draws": 10,
          "score": 0.583,
          "promoted": false,
          "promotedID": null,
          "durationSec": 120.0
        }
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: veryLegacyJson)
        XCTAssertNil(decoded.gamesPlayed)
        XCTAssertNil(decoded.promotionKind)
        XCTAssertNil(decoded.candidateWinsAsWhite)
    }

    func testDecodeIgnoresUnknownKeys() throws {
        // Forward compat: a future field added to the schema
        // shouldn't break older builds that don't know about it.
        // JSONDecoder in default mode silently ignores unknown
        // keys — pin that behavior so a config change to strict
        // decoding would be caught here.
        let forwardJson = """
        {
          "finishedAtStep": 2000,
          "candidateWins": 10,
          "championWins": 10,
          "draws": 0,
          "score": 0.5,
          "promoted": false,
          "promotedID": null,
          "durationSec": 60.0,
          "gamesPlayed": 20,
          "promotionKind": null,
          "future_field": "hello",
          "another_unknown": 42
        }
        """.data(using: .utf8)!
        XCTAssertNoThrow(try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: forwardJson))
    }

    func testEncodedJSONContainsAllNewFields() throws {
        // Ensure the fresh-write path actually emits the new keys.
        // Writing is what future reads depend on — a silent drop
        // during encoding would invisibly truncate arena history.
        let original = makeFullEntry()
        let data = try JSONEncoder().encode(original)
        let json = String(data: data, encoding: .utf8)!
        XCTAssertTrue(json.contains("\"candidateWinsAsWhite\""))
        XCTAssertTrue(json.contains("\"candidateWinsAsBlack\""))
        XCTAssertTrue(json.contains("\"candidateLossesAsWhite\""))
        XCTAssertTrue(json.contains("\"candidateLossesAsBlack\""))
        XCTAssertTrue(json.contains("\"candidateDrawsAsWhite\""))
        XCTAssertTrue(json.contains("\"candidateDrawsAsBlack\""))
    }

    func testMixedPartialLegacy() throws {
        // A hand-edited or half-migrated file where only SOME of
        // the new fields are present. Decoder should populate the
        // present ones and leave the absent ones as nil.
        let mixedJson = """
        {
          "finishedAtStep": 7777,
          "candidateWins": 40, "championWins": 40, "draws": 20,
          "score": 0.5,
          "promoted": false, "promotedID": null, "durationSec": 300.0,
          "gamesPlayed": 100, "promotionKind": null,
          "candidateWinsAsWhite": 22,
          "candidateDrawsAsBlack": 9
        }
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: mixedJson)
        XCTAssertEqual(decoded.candidateWinsAsWhite, 22)
        XCTAssertEqual(decoded.candidateDrawsAsBlack, 9)
        XCTAssertNil(decoded.candidateWinsAsBlack)
        XCTAssertNil(decoded.candidateLossesAsWhite)
        XCTAssertNil(decoded.candidateLossesAsBlack)
        XCTAssertNil(decoded.candidateDrawsAsWhite)
    }

    func testDecodeLegacyMissingExtendedSummary() throws {
        // Files written before extendedSummary was persisted have
        // no `extendedSummary` key; the field must decode as nil so
        // ArenaDetailPopover knows to hide the histogram block.
        let legacyJson = """
        {
          "finishedAtStep": 4000,
          "candidateWins": 10, "championWins": 5, "draws": 5,
          "score": 0.625, "promoted": false, "promotedID": null,
          "durationSec": 200.0, "gamesPlayed": 20, "promotionKind": null
        }
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: legacyJson)
        XCTAssertNil(decoded.extendedSummary)
    }

    func testExtendedSummaryRoundTripPreservesBuckets() throws {
        // Build a small summary directly (bypassing the aggregator
        // — this test is just about the on-disk codec).
        let summary = ArenaExtendedSummary(
            wdlByLength: [
                ArenaWDLByLengthBucket(lowerInclusive: 0, upperInclusive: 19,
                                       wins: 2, draws: 1, losses: 0)
            ],
            valueByPly: [
                ArenaValueByPlyBucket(lowerInclusive: 0, upperInclusive: 4,
                                      mean: 0.25, meanPolicyProbability: 0.45,
                                      meanMaterialAdvantage: 1.5,
                                      wins: 2, draws: 1, losses: 0)
            ],
            valueByProgress: [
                ArenaValueByProgressBucket(lowerPercent: 0, upperPercent: 5,
                                           mean: 0.1, meanPolicyProbability: 0.55,
                                           wins: 1, draws: 1, losses: 1)
            ],
            valueByMaterialAdvantage: [
                ArenaValueByMaterialAdvantageBucket(advantage: 2, mean: 0.3,
                                                    wins: 3, draws: 0, losses: 1)
            ],
            valueByTotalMaterial: [
                ArenaValueByTotalMaterialBucket(lowerInclusive: 72, upperInclusive: 77,
                                                mean: 0.05, wins: 1, draws: 2, losses: 0)
            ]
        )
        let entry = ArenaHistoryEntryCodable(
            finishedAtStep: 1, candidateWins: 2, championWins: 0, draws: 1,
            score: 0.833, promoted: false, promotedID: nil,
            durationSec: 60.0, gamesPlayed: 3, promotionKind: nil,
            extendedSummary: summary
        )
        let data = try JSONEncoder().encode(entry)
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: data)
        XCTAssertEqual(decoded.extendedSummary, summary)
        // Round-tripped score property comes back identical.
        let decodedSummary = try XCTUnwrap(decoded.extendedSummary)
        XCTAssertEqual(
            decodedSummary.valueByPly[0].candidateScore,
            (2.0 + 0.5 * 1.0) / 3.0,
            accuracy: 1e-9
        )
    }

    func testExtendedSummaryDecodesLegacyJSONWithoutMaterialBreakdowns() throws {
        // Summaries persisted before the material breakdowns existed
        // carry no `valueByMaterialAdvantage` / `valueByTotalMaterial`
        // keys; the custom decoder must default them to empty rather
        // than failing the whole summary load.
        let legacy = #"""
        {"wdlByLength":[],"valueByPly":[],"valueByProgress":[]}
        """#
        let summary = try JSONDecoder().decode(
            ArenaExtendedSummary.self, from: Data(legacy.utf8))
        XCTAssertTrue(summary.valueByMaterialAdvantage.isEmpty)
        XCTAssertTrue(summary.valueByTotalMaterial.isEmpty)
    }

    func testValueBucketsDecodeLegacyJSONWithoutPolicyProbability() throws {
        // Buckets persisted before `meanPolicyProbability` existed
        // carry no such key; the Optional field must decode as nil
        // rather than failing the whole summary load.
        let legacyPly = #"""
        {"lowerInclusive":0,"upperInclusive":19,"mean":0.1,"wins":2,"draws":1,"losses":0}
        """#
        let ply = try JSONDecoder().decode(
            ArenaValueByPlyBucket.self, from: Data(legacyPly.utf8))
        XCTAssertNil(ply.meanPolicyProbability)
        XCTAssertEqual(ply.wins, 2)

        let legacyProgress = #"""
        {"lowerPercent":0,"upperPercent":5,"mean":0.1,"wins":1,"draws":1,"losses":1}
        """#
        let progress = try JSONDecoder().decode(
            ArenaValueByProgressBucket.self, from: Data(legacyProgress.utf8))
        XCTAssertNil(progress.meanPolicyProbability)
        XCTAssertEqual(progress.losses, 1)
    }

    func testZeroedPerSideCountsDistinctFromNil() throws {
        // An entry written with all-zero per-side counts must
        // round-trip as 0, NOT nil. The load path substitutes 0
        // for nil, so losing this distinction would silently
        // collapse "arena had 0 white games" into the "legacy file
        // with unknown per-side counts" bucket.
        let zeroed = ArenaHistoryEntryCodable(
            finishedAtStep: 1,
            candidateWins: 0, championWins: 0, draws: 0,
            score: 0, promoted: false, promotedID: nil,
            durationSec: 1.0, gamesPlayed: 0, promotionKind: nil,
            candidateWinsAsWhite: 0, candidateWinsAsBlack: 0,
            candidateLossesAsWhite: 0, candidateLossesAsBlack: 0,
            candidateDrawsAsWhite: 0, candidateDrawsAsBlack: 0
        )
        let data = try JSONEncoder().encode(zeroed)
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: data)
        XCTAssertEqual(decoded.candidateWinsAsWhite, 0)
        XCTAssertNotNil(decoded.candidateWinsAsWhite)
        XCTAssertEqual(decoded, zeroed)
    }
}

// MARK: - SPRT verdict persistence

/// Persisting a sequential test's verdict is not just persisting a decision:
/// an LLR of +3.1 says nothing without the hypotheses and error rates that
/// produced it, and those can be edited between the arena and the resume.
/// These tests pin that the config travels with the verdict, that a session
/// written before SPRT existed still loads, and that a file whose stored
/// config no longer forms a valid test loses the verdict rather than
/// acquiring a different one.
final class ArenaSPRTVerdictCodableTests: XCTestCase {

    private func makeConfig(
        elo0: Double = -5, elo1: Double = 25,
        alpha: Double = 0.01, beta: Double = 0.2,
        minGames: Int = 64, maxGames: Int = 500
    ) throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(
            elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
            minGames: minGames, maxGames: maxGames
        )
    }

    private func makeVerdict(
        _ decision: ArenaSPRT.Decision = .accept,
        llr: Double? = 4.5
    ) throws -> ArenaSPRT.Verdict {
        ArenaSPRT.Verdict(
            decision: decision, llr: llr,
            wins: 40, draws: 18, losses: 12,
            config: try makeConfig()
        )
    }

    func testVerdictRoundTripsIncludingItsHypotheses() throws {
        let original = try makeVerdict()
        let data = try JSONEncoder().encode(ArenaSPRTVerdictCodable(original))
        let decoded = try JSONDecoder().decode(ArenaSPRTVerdictCodable.self, from: data)
        XCTAssertEqual(decoded.verdict(), original)
    }

    func testEveryFinalDecisionRoundTrips() throws {
        for decision in [ArenaSPRT.Decision.accept, .reject, .inconclusive] {
            let original = try makeVerdict(decision)
            let decoded = ArenaSPRTVerdictCodable(original).verdict()
            XCTAssertEqual(decoded?.decision, decision)
        }
    }

    /// A guard-fired verdict on an unscoreable record has no ratio, and `nil`
    /// has to survive as `nil` — coercing it to 0.0 would read as "the
    /// evidence was exactly balanced", which is a claim the run never made.
    func testNilLLRSurvivesRatherThanBecomingZero() throws {
        let decoded = ArenaSPRTVerdictCodable(try makeVerdict(.inconclusive, llr: nil)).verdict()
        XCTAssertNotNil(decoded)
        XCTAssertNil(decoded?.llr)
    }

    /// `.continueTesting` is not a state a verdict can be in. A file claiming
    /// one is corrupt, and must lose the verdict rather than have it repaired
    /// into a decision the arena never reached.
    func testNonFinalStoredDecisionIsRefused() throws {
        var json = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(ArenaSPRTVerdictCodable(try makeVerdict()))
        ) as! [String: Any]
        json["decision"] = "continueTesting"
        let decoded = try JSONDecoder().decode(
            ArenaSPRTVerdictCodable.self,
            from: try JSONSerialization.data(withJSONObject: json)
        )
        XCTAssertNil(decoded.verdict())
    }

    func testUnknownStoredDecisionIsRefused() throws {
        var json = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(ArenaSPRTVerdictCodable(try makeVerdict()))
        ) as! [String: Any]
        json["decision"] = "somethingFromAFutureBuild"
        let decoded = try JSONDecoder().decode(
            ArenaSPRTVerdictCodable.self,
            from: try JSONSerialization.data(withJSONObject: json)
        )
        XCTAssertNil(decoded.verdict())
    }

    /// A hand-edited file can hold a config that no longer forms a valid test.
    /// Repairing it would silently reinterpret the stored LLR against
    /// different hypotheses than the ones that produced it.
    func testInvalidStoredConfigIsRefusedRatherThanRepaired() throws {
        var json = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(ArenaSPRTVerdictCodable(try makeVerdict()))
        ) as! [String: Any]
        json["elo1"] = -99.0   // now below elo0, so the hypotheses are inverted
        let decoded = try JSONDecoder().decode(
            ArenaSPRTVerdictCodable.self,
            from: try JSONSerialization.data(withJSONObject: json)
        )
        XCTAssertNil(decoded.verdict())
    }
}

// MARK: - Criterion on the history entry

final class ArenaHistoryCriterionCodableTests: XCTestCase {

    private func makeEntry(
        criterion: String?,
        sprt: ArenaSPRTVerdictCodable? = nil
    ) -> ArenaHistoryEntryCodable {
        ArenaHistoryEntryCodable(
            finishedAtStep: 5000,
            candidateWins: 40, championWins: 12, draws: 18,
            score: 0.7,
            promoted: false,
            promotedID: nil,
            durationSec: 120,
            gamesPlayed: 70,
            promotionKind: nil,
            promotionCriterion: criterion,
            sprt: sprt
        )
    }

    func testEntryWithSPRTRoundTrips() throws {
        let config = try ArenaSPRT.SPRTConfig(
            elo0: 0, elo1: 10, alpha: 0.05, beta: 0.05, minGames: 32, maxGames: 20000)
        let verdict = ArenaSPRT.Verdict(
            decision: .reject, llr: -3.2, wins: 10, draws: 40, losses: 20, config: config)
        let original = makeEntry(criterion: "sprt", sprt: ArenaSPRTVerdictCodable(verdict))

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: data)
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.promotionCriterion, "sprt")
        XCTAssertEqual(decoded.sprt?.verdict(), verdict)
    }

    /// Every session file written before SPRT existed carries neither field.
    /// They must still decode, with both absent rather than defaulted into
    /// something the arena did not do.
    func testLegacyEntryWithoutCriterionOrSPRTStillDecodes() throws {
        let legacy = """
        {
          "finishedAtStep": 5000,
          "candidateWins": 40,
          "championWins": 12,
          "draws": 18,
          "score": 0.7,
          "promoted": false,
          "durationSec": 120,
          "gamesPlayed": 70
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(ArenaHistoryEntryCodable.self, from: legacy)
        XCTAssertNil(decoded.promotionCriterion)
        XCTAssertNil(decoded.sprt)
        XCTAssertEqual(decoded.gamesPlayed, 70)
    }

    /// A threshold-mode arena stores its criterion but no verdict — that pair
    /// is meaningful and must not be confused with the legacy shape above.
    func testThresholdEntryStoresCriterionWithoutAVerdict() throws {
        let original = makeEntry(criterion: "score")
        let decoded = try JSONDecoder().decode(
            ArenaHistoryEntryCodable.self, from: try JSONEncoder().encode(original))
        XCTAssertEqual(decoded.promotionCriterion, "score")
        XCTAssertNil(decoded.sprt)
    }

    /// The stored token is the criterion's `logToken`, so the two must agree
    /// or a saved history will not load back as the rule that produced it.
    func testStoredTokensMatchTheEnumsOwnTokens() {
        for criterion in ArenaPromotionCriterion.allCases {
            let matched = ArenaPromotionCriterion.allCases.first { $0.logToken == criterion.logToken }
            XCTAssertEqual(matched, criterion, "logToken must round-trip for \(criterion)")
        }
    }
}
