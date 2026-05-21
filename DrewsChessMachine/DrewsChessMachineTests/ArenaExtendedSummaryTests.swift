//
//  ArenaExtendedSummaryTests.swift
//  DrewsChessMachineTests
//
//  Pure-Swift tests for `ArenaSummaryAggregator.aggregate` — the
//  per-arena bucket aggregator behind the `[ARENA]` block lines and
//  the per-row arena-history histograms:
//   - W/D/L by game length (20-ply buckets, candidate perspective)
//   - candidate value scalar + arena-style score by absolute ply
//   - candidate value scalar + arena-style score by game progress
//

import XCTest
@testable import DrewsChessMachine

final class ArenaExtendedSummaryTests: XCTestCase {

    // MARK: - WDL by game length

    func testWDLByLengthBucketsSingleGameInFirstBucket() {
        let record = makeRecord(
            length: 10,           // 10-ply game → 0-19 bucket
            aIsWhite: true,
            result: .checkmate(winner: .white)  // candidate (white) wins
        )
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        // 10 closed buckets + 1 overflow = 11.
        XCTAssertEqual(summary.wdlByLength.count, 11)
        XCTAssertEqual(summary.wdlByLength[0].lowerInclusive, 0)
        XCTAssertEqual(summary.wdlByLength[0].upperInclusive, 19)
        XCTAssertEqual(summary.wdlByLength[0].wins, 1)
        XCTAssertEqual(summary.wdlByLength[0].draws, 0)
        XCTAssertEqual(summary.wdlByLength[0].losses, 0)

        // All later buckets empty.
        for i in 1..<11 {
            XCTAssertEqual(summary.wdlByLength[i].count, 0, "bucket \(i)")
        }
    }

    func testWDLByLengthBucketsCandidatePerspectiveBlackLoss() {
        // Candidate plays black, white wins by checkmate → loss for candidate.
        let record = makeRecord(
            length: 30,           // 20-39 bucket
            aIsWhite: false,
            result: .checkmate(winner: .white)
        )
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.wdlByLength[1].lowerInclusive, 20)
        XCTAssertEqual(summary.wdlByLength[1].upperInclusive, 39)
        XCTAssertEqual(summary.wdlByLength[1].wins, 0)
        XCTAssertEqual(summary.wdlByLength[1].losses, 1)
    }

    func testWDLByLengthBucketsDrawCategorization() {
        let stalemate = makeRecord(length: 50, aIsWhite: true, result: .stalemate)
        let fifty = makeRecord(length: 75, aIsWhite: false, result: .drawByFiftyMoveRule)
        let insuf = makeRecord(length: 60, aIsWhite: true, result: .drawByInsufficientMaterial)
        let rep = makeRecord(length: 65, aIsWhite: false, result: .drawByThreefoldRepetition)

        let summary = ArenaSummaryAggregator.aggregate(records: [stalemate, fifty, insuf, rep])

        // All four games land in the 40-59/60-79 buckets as draws.
        // 50 → bucket 2 (40-59), 60/65/75 → bucket 3 (60-79).
        XCTAssertEqual(summary.wdlByLength[2].draws, 1)  // 50-ply stalemate
        XCTAssertEqual(summary.wdlByLength[3].draws, 3)  // 60, 65, 75
        XCTAssertEqual(summary.wdlByLength[2].wins, 0)
        XCTAssertEqual(summary.wdlByLength[2].losses, 0)
    }

    func testWDLByLengthOverflowBucket() {
        // 250-ply game lands in the open-ended "200+" bucket (index 10).
        let record = makeRecord(
            length: 250,
            aIsWhite: true,
            result: .drawByThreefoldRepetition
        )
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        let overflow = summary.wdlByLength[10]
        XCTAssertNil(overflow.upperInclusive)
        XCTAssertEqual(overflow.lowerInclusive, 200)
        XCTAssertEqual(overflow.draws, 1)
    }

    func testWDLByLengthBoundaryAtExactly200Plies() {
        // 200-ply game lands in the overflow bucket (200 / 20 == 10).
        let record = makeRecord(
            length: 200,
            aIsWhite: true,
            result: .stalemate
        )
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.wdlByLength[10].draws, 1)
        XCTAssertEqual(summary.wdlByLength[9].count, 0)
    }

    // MARK: - Value by absolute ply

    func testValueByPlyMeansAcrossSingleGame() {
        // 4 candidate-to-move samples — plies 0, 2, 4 fall into the
        // first 20-ply bucket [0-19]; ply 25 falls into [20-39].
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.1),
            CandidateValueSample(ply: 2, value: 0.2),
            CandidateValueSample(ply: 4, value: 0.3),
            CandidateValueSample(ply: 25, value: -0.4)
        ]
        let record = makeRecord(length: 26, aIsWhite: true,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        // Bucket 0 (plies 0-19): mean = (0.1 + 0.2 + 0.3) / 3 = 0.2
        // Bucket 1 (plies 20-39): mean = -0.4
        XCTAssertEqual(summary.valueByPly.count, 2)
        XCTAssertEqual(summary.valueByPly[0].lowerInclusive, 0)
        XCTAssertEqual(summary.valueByPly[0].upperInclusive, 19)
        XCTAssertEqual(summary.valueByPly[0].count, 3)
        XCTAssertEqual(summary.valueByPly[0].mean, 0.2, accuracy: 1e-6)
        XCTAssertEqual(summary.valueByPly[1].lowerInclusive, 20)
        XCTAssertEqual(summary.valueByPly[1].upperInclusive, 39)
        XCTAssertEqual(summary.valueByPly[1].count, 1)
        XCTAssertEqual(summary.valueByPly[1].mean, -0.4, accuracy: 1e-6)
    }

    func testValueByPlyDropsEmptyTrailingBucketsButKeepsGaps() {
        // One sample at ply 0, one at ply 45 → 20-ply buckets [0-19]
        // and [40-59] have data, [20-39] is empty. The aggregator
        // filters empty buckets entirely, including the gap.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.1),
            CandidateValueSample(ply: 45, value: 0.5)
        ]
        let record = makeRecord(length: 46, aIsWhite: true,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByPly.count, 2)
        XCTAssertEqual(summary.valueByPly[0].lowerInclusive, 0)
        XCTAssertEqual(summary.valueByPly[0].upperInclusive, 19)
        XCTAssertEqual(summary.valueByPly[1].lowerInclusive, 40)
        XCTAssertEqual(summary.valueByPly[1].upperInclusive, 59)
    }

    func testValueByPlyAggregatesAcrossMultipleGames() {
        let g1 = makeRecord(length: 5, aIsWhite: true,
                            result: .checkmate(winner: .white),
                            samples: [CandidateValueSample(ply: 0, value: 0.2),
                                      CandidateValueSample(ply: 2, value: 0.4)])
        let g2 = makeRecord(length: 5, aIsWhite: true,
                            result: .checkmate(winner: .white),
                            samples: [CandidateValueSample(ply: 0, value: -0.2),
                                      CandidateValueSample(ply: 2, value: 0.0)])
        let summary = ArenaSummaryAggregator.aggregate(records: [g1, g2])

        // Bucket 0: 4 samples — mean = (0.2 + 0.4 - 0.2 + 0.0) / 4 = 0.1
        XCTAssertEqual(summary.valueByPly.count, 1)
        XCTAssertEqual(summary.valueByPly[0].count, 4)
        XCTAssertEqual(summary.valueByPly[0].mean, 0.1, accuracy: 1e-6)
    }

    // MARK: - Value by game progress

    func testValueByProgressBucketsSpanFullRangeAtFiveSamplePoints() {
        // 20-ply game with candidate-as-white → samples at plies
        // 0, 2, 4, 6, ..., 18 → percent = 0, 10, 20, ..., 90.
        // Each 5% bucket has 0 or 1 samples; only every-other bucket
        // populates.
        let samples: [CandidateValueSample] = stride(from: 0, to: 20, by: 2).map {
            CandidateValueSample(ply: $0, value: Float($0) * 0.05)
        }
        let record = makeRecord(length: 20, aIsWhite: true,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        // 10 samples → expect 10 buckets populated.
        XCTAssertEqual(summary.valueByProgress.count, 10)
        for (idx, bucket) in summary.valueByProgress.enumerated() {
            XCTAssertEqual(bucket.count, 1, "progress bucket \(idx)")
            // Bucket lower% = 10 * idx (because we sample every 2
            // plies in a 20-ply game = every 10%).
            XCTAssertEqual(bucket.lowerPercent, idx * 10)
            XCTAssertEqual(bucket.upperPercent, idx * 10 + 5)
        }
    }

    func testValueByProgressMeansAcrossSampleClumps() {
        // 100-ply game; six candidate samples clustered at the end:
        // plies 95, 96, 97, 98, 99 → percent 95, 96, 97, 98, 99 →
        // all in the [95, 100] bucket.
        // Plus ply 0 → bucket [0, 5).
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.0),
            CandidateValueSample(ply: 95, value: 0.6),
            CandidateValueSample(ply: 96, value: 0.7),
            CandidateValueSample(ply: 97, value: 0.8),
            CandidateValueSample(ply: 98, value: 0.9),
            CandidateValueSample(ply: 99, value: 1.0)
        ]
        let record = makeRecord(length: 100, aIsWhite: true,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByProgress.count, 2)

        let first = summary.valueByProgress[0]
        XCTAssertEqual(first.lowerPercent, 0)
        XCTAssertEqual(first.upperPercent, 5)
        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(first.mean, 0.0, accuracy: 1e-6)

        let last = summary.valueByProgress[1]
        XCTAssertEqual(last.lowerPercent, 95)
        XCTAssertEqual(last.upperPercent, 100)
        XCTAssertEqual(last.count, 5)
        XCTAssertEqual(last.mean, 0.8, accuracy: 1e-6) // (0.6+0.7+0.8+0.9+1.0)/5
    }

    func testValueByProgressClampsFinalPlyIntoLastBucket() {
        // Ply N-1 in an N-ply game → percent = ((N-1)*100)/N which
        // for small N can equal exactly 99 (or sit in 95-100). Pick
        // length 50 → ply 49 → percent 98 → last bucket.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 49, value: 0.5)
        ]
        let record = makeRecord(length: 50, aIsWhite: false,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByProgress.count, 1)
        XCTAssertEqual(summary.valueByProgress[0].lowerPercent, 95)
        XCTAssertEqual(summary.valueByProgress[0].upperPercent, 100)
    }

    // MARK: - Empty / edge cases

    func testEmptyRecordsProducesAllZeroLengthBucketsAndEmptyValueLists() {
        let summary = ArenaSummaryAggregator.aggregate(records: [])
        XCTAssertEqual(summary.wdlByLength.count, 11)
        for b in summary.wdlByLength {
            XCTAssertEqual(b.count, 0)
        }
        XCTAssertTrue(summary.valueByPly.isEmpty)
        XCTAssertTrue(summary.valueByProgress.isEmpty)
    }

    func testRecordWithNoSamplesContributesOnlyToLengthBucket() {
        let record = makeRecord(length: 10, aIsWhite: true,
                                result: .stalemate, samples: [])
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.wdlByLength[0].draws, 1)
        XCTAssertTrue(summary.valueByPly.isEmpty)
        XCTAssertTrue(summary.valueByProgress.isEmpty)
    }

    // MARK: - W/D/L attribution + candidateScore

    func testValueByPlyBucketAttributesEachSampleToGameOutcome() {
        // Two candidate plies in a game the candidate wins: both
        // plies must be counted as W in their bucket. The mean value
        // scalar is independent of the outcome.
        let win = makeRecord(length: 4, aIsWhite: true,
                             result: .checkmate(winner: .white),
                             samples: [CandidateValueSample(ply: 0, value: 0.2),
                                       CandidateValueSample(ply: 2, value: 0.4)])
        let loss = makeRecord(length: 4, aIsWhite: true,
                              result: .checkmate(winner: .black),
                              samples: [CandidateValueSample(ply: 0, value: -0.1),
                                        CandidateValueSample(ply: 2, value: -0.3)])
        let draw = makeRecord(length: 4, aIsWhite: false,
                              result: .stalemate,
                              samples: [CandidateValueSample(ply: 1, value: 0.05),
                                        CandidateValueSample(ply: 3, value: -0.05)])
        let summary = ArenaSummaryAggregator.aggregate(records: [win, loss, draw])

        // All six samples land in bucket [0-19].
        XCTAssertEqual(summary.valueByPly.count, 1)
        let bucket = summary.valueByPly[0]
        XCTAssertEqual(bucket.count, 6)
        XCTAssertEqual(bucket.wins, 2)
        XCTAssertEqual(bucket.losses, 2)
        XCTAssertEqual(bucket.draws, 2)
        // (W + 0.5D) / N = (2 + 1) / 6 = 0.5
        XCTAssertEqual(bucket.candidateScore, 0.5, accuracy: 1e-9)
    }

    func testValueByProgressBucketAttributesOutcomeToEveryCandidatePly() {
        // 20-ply game candidate plays white, wins. Candidate moves
        // on plies 0,2,...,18 → 10 samples spanning 0-90% of game.
        let samples: [CandidateValueSample] = stride(from: 0, to: 20, by: 2).map {
            CandidateValueSample(ply: $0, value: 0.0)
        }
        let rec = makeRecord(length: 20, aIsWhite: true,
                             result: .checkmate(winner: .white),
                             samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [rec])

        // Every populated progress bucket has wins == count and
        // candidateScore == 1.0 (it's the same game throughout).
        XCTAssertFalse(summary.valueByProgress.isEmpty)
        for b in summary.valueByProgress {
            XCTAssertEqual(b.wins, b.count, "bucket \(b.lowerPercent)")
            XCTAssertEqual(b.draws, 0)
            XCTAssertEqual(b.losses, 0)
            XCTAssertEqual(b.candidateScore, 1.0, accuracy: 1e-9,
                           "bucket \(b.lowerPercent)")
        }
    }

    func testWDLByLengthCandidateScoreUsesArenaIdentity() {
        // 1 win, 1 draw, 0 losses in the same length bucket →
        // (1 + 0.5) / 2 = 0.75.
        let win = makeRecord(length: 10, aIsWhite: true,
                             result: .checkmate(winner: .white))
        let draw = makeRecord(length: 15, aIsWhite: true, result: .stalemate)
        let summary = ArenaSummaryAggregator.aggregate(records: [win, draw])

        XCTAssertEqual(summary.wdlByLength[0].count, 2)
        XCTAssertEqual(summary.wdlByLength[0].candidateScore, 0.75, accuracy: 1e-9)
    }

    func testEmptyBucketsHaveZeroScore() {
        let summary = ArenaSummaryAggregator.aggregate(records: [])
        for b in summary.wdlByLength {
            XCTAssertEqual(b.count, 0)
            XCTAssertEqual(b.candidateScore, 0)
        }
    }

    // MARK: - Codable round-trip

    func testExtendedSummaryRoundTripsThroughJSON() throws {
        let win = makeRecord(length: 10, aIsWhite: true,
                             result: .checkmate(winner: .white),
                             samples: [CandidateValueSample(ply: 0, value: 0.3),
                                       CandidateValueSample(ply: 2, value: -0.1)])
        let draw = makeRecord(length: 30, aIsWhite: false,
                              result: .drawByThreefoldRepetition,
                              samples: [CandidateValueSample(ply: 1, value: 0.0),
                                        CandidateValueSample(ply: 5, value: 0.1)])
        let original = ArenaSummaryAggregator.aggregate(records: [win, draw])

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ArenaExtendedSummary.self, from: data)

        XCTAssertEqual(decoded, original)
        // Sanity: candidateScore is derived from counts, so it must
        // survive a round-trip even though it isn't stored.
        XCTAssertEqual(
            decoded.wdlByLength.map(\.candidateScore),
            original.wdlByLength.map(\.candidateScore)
        )
    }

    // MARK: - Mean policy probability

    func testValueByPlyMeanPolicyProbability() throws {
        // Two candidate plies in the first 20-ply bucket, each with a
        // known chosen-move policy probability → the bucket's
        // meanPolicyProbability is their average.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.1, policyProbability: 0.4),
            CandidateValueSample(ply: 2, value: 0.2, policyProbability: 0.6)
        ]
        let record = makeRecord(length: 3, aIsWhite: true,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByPly.count, 1)
        let policy = try XCTUnwrap(summary.valueByPly[0].meanPolicyProbability)
        XCTAssertEqual(policy, 0.5, accuracy: 1e-6)  // (0.4 + 0.6) / 2
    }

    func testValueByProgressMeanPolicyProbability() throws {
        // 20-ply game, candidate white. ply 0 → progress bucket
        // [0,5); ply 10 → [50,55). Each carries its own policy
        // probability, so each bucket's mean is that single value.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.0, policyProbability: 0.3),
            CandidateValueSample(ply: 10, value: 0.0, policyProbability: 0.7)
        ]
        let record = makeRecord(length: 20, aIsWhite: true,
                                result: .checkmate(winner: .white),
                                samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByProgress.count, 2)
        XCTAssertEqual(
            try XCTUnwrap(summary.valueByProgress[0].meanPolicyProbability),
            0.3, accuracy: 1e-6
        )
        XCTAssertEqual(
            try XCTUnwrap(summary.valueByProgress[1].meanPolicyProbability),
            0.7, accuracy: 1e-6
        )
    }

    // MARK: - Material breakdowns

    func testValueByMaterialAdvantageAggregates() {
        // Three candidate plies at advantages -2, 0, +2; white wins,
        // so every ply scores 1.0. Each distinct advantage is its own
        // bucket, emitted in ascending order.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.1, policyProbability: 0,
                                 materialAdvantage: -2, totalMaterial: 40),
            CandidateValueSample(ply: 2, value: 0.2, policyProbability: 0,
                                 materialAdvantage: 0, totalMaterial: 38),
            CandidateValueSample(ply: 4, value: 0.3, policyProbability: 0,
                                 materialAdvantage: 2, totalMaterial: 36)
        ]
        let record = makeRecord(length: 5, aIsWhite: true,
                                result: .checkmate(winner: .white), samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByMaterialAdvantage.map(\.advantage), [-2, 0, 2])
        for bucket in summary.valueByMaterialAdvantage {
            XCTAssertEqual(bucket.count, 1)
            XCTAssertEqual(bucket.candidateScore, 1.0, accuracy: 1e-9)
        }
    }

    func testValueByMaterialAdvantageClampsBeyondNine() {
        // Advantages past ±9 fold into the ±9 overflow buckets.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0, policyProbability: 0,
                                 materialAdvantage: 15, totalMaterial: 50),
            CandidateValueSample(ply: 2, value: 0, policyProbability: 0,
                                 materialAdvantage: -30, totalMaterial: 50)
        ]
        let record = makeRecord(length: 3, aIsWhite: true,
                                result: .stalemate, samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByMaterialAdvantage.map(\.advantage), [-9, 9])
    }

    func testValueByMaterialAdvantageMeanIsValueMean() {
        // Two plies at the same advantage → one bucket whose `mean`
        // is the value-scalar mean over those plies.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.2, policyProbability: 0,
                                 materialAdvantage: 3, totalMaterial: 40),
            CandidateValueSample(ply: 2, value: 0.4, policyProbability: 0,
                                 materialAdvantage: 3, totalMaterial: 40)
        ]
        let record = makeRecord(length: 3, aIsWhite: true,
                                result: .checkmate(winner: .white), samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByMaterialAdvantage.count, 1)
        let bucket = summary.valueByMaterialAdvantage[0]
        XCTAssertEqual(bucket.advantage, 3)
        XCTAssertEqual(bucket.count, 2)
        XCTAssertEqual(bucket.mean, 0.3, accuracy: 1e-6)
    }

    func testValueByTotalMaterialBucketsByWidthSix() {
        // totalMaterial 8 → bucket 1 ([6-11]); 40 → bucket 6 ([36-41]).
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0.1, policyProbability: 0,
                                 materialAdvantage: 0, totalMaterial: 8),
            CandidateValueSample(ply: 2, value: 0.5, policyProbability: 0,
                                 materialAdvantage: 0, totalMaterial: 40)
        ]
        let record = makeRecord(length: 3, aIsWhite: true,
                                result: .checkmate(winner: .white), samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByTotalMaterial.count, 2)
        XCTAssertEqual(summary.valueByTotalMaterial[0].lowerInclusive, 6)
        XCTAssertEqual(summary.valueByTotalMaterial[0].upperInclusive, 11)
        XCTAssertEqual(summary.valueByTotalMaterial[1].lowerInclusive, 36)
        XCTAssertEqual(summary.valueByTotalMaterial[1].upperInclusive, 41)
    }

    func testValueByPlyCarriesMeanMaterialAdvantage() throws {
        // Two candidate plies in the first 20-ply bucket at material
        // advantages 2 and 4 → the bucket's meanMaterialAdvantage is 3.
        let samples: [CandidateValueSample] = [
            CandidateValueSample(ply: 0, value: 0, policyProbability: 0,
                                 materialAdvantage: 2, totalMaterial: 40),
            CandidateValueSample(ply: 2, value: 0, policyProbability: 0,
                                 materialAdvantage: 4, totalMaterial: 40)
        ]
        let record = makeRecord(length: 3, aIsWhite: true,
                                result: .checkmate(winner: .white), samples: samples)
        let summary = ArenaSummaryAggregator.aggregate(records: [record])

        XCTAssertEqual(summary.valueByPly.count, 1)
        let material = try XCTUnwrap(summary.valueByPly[0].meanMaterialAdvantage)
        XCTAssertEqual(material, 3.0, accuracy: 1e-6)
    }

    func testArenaMaterialSummaryFromBoard() {
        // White: queen (9) + king (0). Black: rook (5) + king (0).
        var board: [Piece?] = Array(repeating: nil, count: 64)
        board[0] = Piece(type: .king, color: .white)
        board[1] = Piece(type: .queen, color: .white)
        board[2] = Piece(type: .king, color: .black)
        board[3] = Piece(type: .rook, color: .black)

        let white = ArenaMaterial.summary(board: board, candidateColor: .white)
        XCTAssertEqual(white.advantage, 4)   // 9 − 5
        XCTAssertEqual(white.total, 14)      // 9 + 5

        let black = ArenaMaterial.summary(board: board, candidateColor: .black)
        XCTAssertEqual(black.advantage, -4)  // 5 − 9
        XCTAssertEqual(black.total, 14)
    }

    // MARK: - Fixture helpers

    /// A `ChessMove` stub used only to pad `moveHistory` to a desired
    /// length — the aggregator never inspects the moves themselves,
    /// only `moveHistory.count`. Any legal-looking square pair works.
    private static let padMove = ChessMove(fromRow: 0, fromCol: 0, toRow: 0, toCol: 1, promotion: nil)

    private func makeRecord(
        length: Int,
        aIsWhite: Bool,
        result: GameResult,
        samples: [CandidateValueSample] = []
    ) -> TournamentGameRecord {
        let history = [ChessMove](repeating: Self.padMove, count: length)
        return TournamentGameRecord(
            gameIndex: 0,
            aIsWhite: aIsWhite,
            result: result,
            moveHistory: history,
            candidateValueSamples: samples
        )
    }
}

extension CandidateValueSample {
    /// Test convenience. Most aggregator tests exercise the value /
    /// W·D·L paths and don't depend on policy probability or material;
    /// these keep their fixtures terse. Tests that assert on those
    /// signals use the full memberwise init.
    init(ply: Int, value: Float) {
        self.init(ply: ply, value: value, policyProbability: 0,
                  materialAdvantage: 0, totalMaterial: 0)
    }

    init(ply: Int, value: Float, policyProbability: Float) {
        self.init(ply: ply, value: value, policyProbability: policyProbability,
                  materialAdvantage: 0, totalMaterial: 0)
    }
}
