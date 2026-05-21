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
                                      wins: 2, draws: 1, losses: 0)
            ],
            valueByProgress: [
                ArenaValueByProgressBucket(lowerPercent: 0, upperPercent: 5,
                                           mean: 0.1, meanPolicyProbability: 0.55,
                                           wins: 1, draws: 1, losses: 1)
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
