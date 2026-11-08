//
//  CorpusEvaluationMetricsTests.swift
//  DrewsChessMachineTests
//
//  The `--eval-corpus` loss math (`CorpusEvaluationMetrics`): legal-masked
//  policy NLL and competition rank, W/D/L cross-entropy and accuracy, the
//  result from the side to move, ply buckets, accumulation and merging, and
//  game selection. Pure: no Metal, no files.
//

import Foundation
import XCTest
@testable import DrewsChessMachine

final class CorpusEvaluationMetricsTests: XCTestCase {

    private func policyScore(_ logits: [Float], played: Int) -> LegalPolicyScore {
        logits.withUnsafeBufferPointer {
            CorpusEvaluationMetrics.legalPolicyScore(legalLogits: $0, playedOffset: played)
        }
    }

    /// Reference legal-masked NLL, computed directly from the definition.
    private func referenceNLL(_ logits: [Float], played: Int) -> Double {
        let values = logits.map(Double.init)
        let denominator = values.map { Foundation.exp($0) }.reduce(0, +)
        return -Foundation.log(Foundation.exp(values[played]) / denominator)
    }

    // MARK: - Policy

    /// Protects: the NLL is −ln of the played move's legal-only softmax
    /// probability.
    /// Fails if: the log-sum-exp drops the max shift incorrectly, uses the
    /// wrong sign, or scores a different move than the played one.
    func testPolicyNLLMatchesTheSoftmaxDefinition() {
        let logits: [Float] = [0.5, -1.25, 2.0, 0.0]
        for played in logits.indices {
            XCTAssertEqual(policyScore(logits, played: played).negativeLogProbability,
                           referenceNLL(logits, played: played), accuracy: 1e-12)
        }
    }

    /// Protects: a position with one legal move costs nothing and is top-1,
    /// whatever its logit.
    /// Fails if: a forced move gets a non-zero NLL or a rank other than 1.
    func testOneLegalMoveHasZeroNLLAndRankOne() {
        for logit: Float in [-80, 0, 37.5] {
            let score = policyScore([logit], played: 0)
            XCTAssertEqual(score.negativeLogProbability, 0, accuracy: 1e-15)
            XCTAssertEqual(score.rank, 1)
            XCTAssertTrue(score.isInTop(1))
        }
    }

    /// Protects: the played move at the strict argmax is rank 1 and top-1.
    /// Fails if: the rank counts the played move against itself or counts
    /// smaller logits.
    func testPlayedArgmaxIsRankOne() {
        let score = policyScore([0.1, 3.0, -2.0, 2.9], played: 1)
        XCTAssertEqual(score.rank, 1)
        XCTAssertTrue(score.isInTop(1))
        XCTAssertLessThan(score.negativeLogProbability, Foundation.log(4.0))
    }

    /// Protects: rank is one more than the count of strictly larger legal
    /// logits, so top-5 means at most four moves beat the played one.
    /// Fails if: the boundary between top-5 and not is off by one.
    func testRankCountsStrictlyLargerLogits() {
        let logits: [Float] = [10, 9, 8, 7, 6, 5, 4]
        XCTAssertEqual(policyScore(logits, played: 4).rank, 5)
        XCTAssertTrue(policyScore(logits, played: 4).isInTop(CorpusEvaluationMetrics.wideTopK))
        XCTAssertEqual(policyScore(logits, played: 5).rank, 6)
        XCTAssertFalse(policyScore(logits, played: 5).isInTop(CorpusEvaluationMetrics.wideTopK))
    }

    /// Protects: tied logits share the best rank among them (competition
    /// ranking), so a tie at the top is top-1 for every tied move, and the
    /// result does not depend on move order.
    /// Fails if: ties are broken by position in the legal-move list.
    func testTiedLogitsShareTheBestRank() {
        let logits: [Float] = [1.5, 1.5, 1.5, -3]
        for played in 0..<3 {
            let score = policyScore(logits, played: played)
            XCTAssertEqual(score.rank, 1)
            XCTAssertEqual(score.negativeLogProbability, referenceNLL(logits, played: played), accuracy: 1e-12)
        }
        XCTAssertEqual(policyScore([2, 5, 5, 2], played: 3).rank, 3)
    }

    /// Protects: extreme logits give finite, exact NLLs (the max shift and
    /// Double accumulation).
    /// Fails if: the softmax exponentiates raw logits (overflow to inf or
    /// underflow to a zero probability and an infinite NLL).
    func testExtremeLogitsStayFinite() {
        let huge = policyScore([1000, -1000, 999], played: 0)
        XCTAssertEqual(huge.negativeLogProbability, Foundation.log(1 + Foundation.exp(-1.0)), accuracy: 1e-12)
        let unlikely = policyScore([1000, -1000], played: 1)
        XCTAssertEqual(unlikely.negativeLogProbability, 2000, accuracy: 1e-9)
        XCTAssertTrue(unlikely.negativeLogProbability.isFinite)
        let allTiny = policyScore([-5000, -5000], played: 0)
        XCTAssertEqual(allTiny.negativeLogProbability, Foundation.log(2.0), accuracy: 1e-12)
        let maxFloat = policyScore([Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude], played: 0)
        XCTAssertEqual(maxFloat.negativeLogProbability, 0, accuracy: 1e-12)
    }

    /// Protects: a uniform legal policy scores ln(number of legal moves).
    /// Fails if: illegal moves leak into the denominator (the caller hands
    /// only legal logits; the function must use exactly those).
    func testUniformLegalPolicyScoresLogOfLegalCount() {
        let score = policyScore([Float](repeating: 0.25, count: 30), played: 7)
        XCTAssertEqual(score.negativeLogProbability, Foundation.log(30.0), accuracy: 1e-12)
        XCTAssertEqual(score.rank, 1)
    }

    // MARK: - Value

    /// Protects: the W/D/L target from the side to move's point of view.
    /// Fails if: the result is taken from White's side for Black to move, or
    /// a draw maps to anything but draw.
    func testResultFromTheSideToMove() {
        XCTAssertEqual(WDLClass(outcome: .whiteWin, sideToMove: .white), .win)
        XCTAssertEqual(WDLClass(outcome: .whiteWin, sideToMove: .black), .loss)
        XCTAssertEqual(WDLClass(outcome: .blackWin, sideToMove: .black), .win)
        XCTAssertEqual(WDLClass(outcome: .blackWin, sideToMove: .white), .loss)
        XCTAssertEqual(WDLClass(outcome: .draw, sideToMove: .white), .draw)
        XCTAssertEqual(WDLClass(outcome: .draw, sideToMove: .black), .draw)
    }

    /// Protects: value CE is −ln p(result) on the reported probabilities, and
    /// the prediction is correct only when the result's probability is the
    /// largest.
    /// Fails if: the wrong slot is read for a class, or accuracy uses the
    /// wrong comparison.
    func testValueCrossEntropyAndAccuracy() {
        let probabilities = WDLProbabilities(win: 0.6, draw: 0.3, loss: 0.1)
        let win = CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: .win)
        XCTAssertEqual(win.crossEntropy, -Foundation.log(0.6), accuracy: 1e-12)
        XCTAssertTrue(win.isPredictionCorrect)
        let draw = CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: .draw)
        XCTAssertEqual(draw.crossEntropy, -Foundation.log(0.3), accuracy: 1e-12)
        XCTAssertFalse(draw.isPredictionCorrect)
        let loss = CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: .loss)
        XCTAssertEqual(loss.crossEntropy, -Foundation.log(0.1), accuracy: 1e-12)
        XCTAssertFalse(loss.isPredictionCorrect)
    }

    /// Protects: a tie for the largest W/D/L probability counts as correct
    /// for each tied class (the policy's tie rule).
    /// Fails if: ties are broken by class order.
    func testValueTieCountsAsCorrect() {
        let probabilities = WDLProbabilities(win: 0.4, draw: 0.4, loss: 0.2)
        XCTAssertTrue(CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: .win).isPredictionCorrect)
        XCTAssertTrue(CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: .draw).isPredictionCorrect)
        XCTAssertFalse(CorpusEvaluationMetrics.valueScore(probabilities: probabilities, result: .loss).isPredictionCorrect)
    }

    /// Protects: a zero probability on the result gives an infinite CE, which
    /// the CLI then reports as a non-finite result, never a number.
    /// Fails if: the probability is clamped to a floor (a silent default
    /// that would hide a saturated value head).
    func testZeroProbabilityGivesInfiniteCrossEntropy() {
        let score = CorpusEvaluationMetrics.valueScore(probabilities: WDLProbabilities(win: 1, draw: 0, loss: 0),
                                                       result: .loss)
        XCTAssertEqual(score.crossEntropy, .infinity)
        XCTAssertFalse(score.isPredictionCorrect)
    }

    // MARK: - Buckets and accumulation

    /// Protects: the bucket boundaries 0–19, 20–59, 60+.
    /// Fails if: a boundary ply lands in the neighbouring bucket.
    func testPlyBucketBoundaries() {
        XCTAssertEqual(CorpusEvaluationPlyBucket(ply: 0), .plies0To19)
        XCTAssertEqual(CorpusEvaluationPlyBucket(ply: 19), .plies0To19)
        XCTAssertEqual(CorpusEvaluationPlyBucket(ply: 20), .plies20To59)
        XCTAssertEqual(CorpusEvaluationPlyBucket(ply: 59), .plies20To59)
        XCTAssertEqual(CorpusEvaluationPlyBucket(ply: 60), .plies60AndLater)
        XCTAssertEqual(CorpusEvaluationPlyBucket(ply: 500), .plies60AndLater)
    }

    /// Protects: an empty accumulation has no means (never zeros that read as
    /// a perfect or failed score), and an empty tally's overall is empty.
    /// Fails if: `means` returns zeros or NaN for zero positions.
    func testEmptyAccumulationHasNoMeans() {
        XCTAssertNil(CorpusEvaluationAccumulator().means)
        XCTAssertEqual(CorpusEvaluationAccumulator().positions, 0)
        let tally = CorpusEvaluationTally()
        XCTAssertEqual(tally.overall.positions, 0)
        XCTAssertNil(tally.overall.means)
        XCTAssertNil(tally.accumulator(for: .plies20To59).means)
    }

    /// Protects: means are per-position averages of what was added, and the
    /// overall figures are exactly the merge of the buckets.
    /// Fails if: a bucket is double-counted or missing from the overall, or a
    /// counter (top-1, top-5, value accuracy) is mis-incremented.
    func testTallyMeansAndBucketMerge() throws {
        var tally = CorpusEvaluationTally()
        let even = WDLProbabilities(win: 0.5, draw: 0.25, loss: 0.25)
        let top1 = LegalPolicyScore(negativeLogProbability: 1, rank: 1)
        let top5 = LegalPolicyScore(negativeLogProbability: 2, rank: 5)
        let outside = LegalPolicyScore(negativeLogProbability: 3, rank: 6)
        let right = ValueScore(crossEntropy: 0.5, isPredictionCorrect: true)
        let wrong = ValueScore(crossEntropy: 1.5, isPredictionCorrect: false)
        tally.add(ply: 3, policy: top1, value: right, probabilities: even)
        tally.add(ply: 25, policy: top5, value: wrong, probabilities: even)
        tally.add(ply: 70, policy: outside, value: right, probabilities: WDLProbabilities(win: 0, draw: 1, loss: 0))
        tally.add(ply: 71, policy: top1, value: wrong, probabilities: even)

        let means = try XCTUnwrap(tally.overall.means)
        XCTAssertEqual(tally.overall.positions, 4)
        XCTAssertEqual(means.policyNLL, 7.0 / 4, accuracy: 1e-12)
        XCTAssertEqual(means.top1, 0.5, accuracy: 1e-12)
        XCTAssertEqual(means.wideTopK, 0.75, accuracy: 1e-12)
        XCTAssertEqual(means.valueCrossEntropy, 1.0, accuracy: 1e-12)
        XCTAssertEqual(means.valueAccuracy, 0.5, accuracy: 1e-12)
        XCTAssertEqual(means.meanWinProbability, 1.5 / 4, accuracy: 1e-12)
        XCTAssertEqual(means.meanDrawProbability, 1.75 / 4, accuracy: 1e-12)
        XCTAssertEqual(means.meanLossProbability, 0.75 / 4, accuracy: 1e-12)

        XCTAssertEqual(tally.accumulator(for: .plies0To19).positions, 1)
        XCTAssertEqual(tally.accumulator(for: .plies20To59).positions, 1)
        let late = try XCTUnwrap(tally.accumulator(for: .plies60AndLater).means)
        XCTAssertEqual(tally.accumulator(for: .plies60AndLater).positions, 2)
        XCTAssertEqual(late.policyNLL, 2, accuracy: 1e-12)
    }

    /// Protects: merging two tallies equals adding all their positions to one
    /// (how per-batch tallies combine into a model's result), and merging an
    /// empty tally changes nothing.
    /// Fails if: `merge` drops a bucket the receiver did not have yet.
    func testMergeEqualsAddingEverything() {
        let probabilities = WDLProbabilities(win: 0.25, draw: 0.5, loss: 0.25)
        let scores: [(Int, LegalPolicyScore, ValueScore)] = [
            (0, LegalPolicyScore(negativeLogProbability: 0.75, rank: 2), ValueScore(crossEntropy: 0.25, isPredictionCorrect: true)),
            (40, LegalPolicyScore(negativeLogProbability: 1.5, rank: 1), ValueScore(crossEntropy: 2, isPredictionCorrect: false)),
            (90, LegalPolicyScore(negativeLogProbability: 0.5, rank: 9), ValueScore(crossEntropy: 1, isPredictionCorrect: true)),
        ]
        var whole = CorpusEvaluationTally()
        var first = CorpusEvaluationTally()
        var second = CorpusEvaluationTally()
        for (offset, entry) in scores.enumerated() {
            whole.add(ply: entry.0, policy: entry.1, value: entry.2, probabilities: probabilities)
            if offset == 0 {
                first.add(ply: entry.0, policy: entry.1, value: entry.2, probabilities: probabilities)
            } else {
                second.add(ply: entry.0, policy: entry.1, value: entry.2, probabilities: probabilities)
            }
        }
        var merged = first
        merged.merge(second)
        XCTAssertEqual(merged, whole)
        merged.merge(CorpusEvaluationTally())
        XCTAssertEqual(merged, whole)
        var fromEmpty = CorpusEvaluationTally()
        fromEmpty.merge(whole)
        XCTAssertEqual(fromEmpty, whole)
    }

    // MARK: - Game selection

    /// Protects: the first / last N games in corpus order.
    /// Fails if: `--eval-from-end` is off by one at either end.
    func testGameSelectionRanges() throws {
        let first = try CorpusEvaluationGameSelection(origin: .start, requestedGames: 3, corpusGames: 10)
        XCTAssertEqual(first.gameRange, 0..<3)
        let last = try CorpusEvaluationGameSelection(origin: .end, requestedGames: 3, corpusGames: 10)
        XCTAssertEqual(last.gameRange, 7..<10)
        let all = try CorpusEvaluationGameSelection(origin: .end, requestedGames: 10, corpusGames: 10)
        XCTAssertEqual(all.gameRange, 0..<10)
    }

    /// Protects: a request larger than the corpus, or not positive, is
    /// refused rather than quietly evaluating fewer games.
    /// Fails if: the selection is clamped to the corpus size.
    func testGameSelectionRefusesImpossibleRequests() {
        XCTAssertThrowsError(try CorpusEvaluationGameSelection(origin: .start, requestedGames: 11, corpusGames: 10)) {
            XCTAssertEqual($0 as? CorpusEvaluationGameSelection.SelectionError,
                           .corpusTooSmall(requested: 11, corpusGames: 10))
        }
        XCTAssertThrowsError(try CorpusEvaluationGameSelection(origin: .end, requestedGames: 0, corpusGames: 10)) {
            XCTAssertEqual($0 as? CorpusEvaluationGameSelection.SelectionError, .nonPositiveGameCount(0))
        }
    }
}
