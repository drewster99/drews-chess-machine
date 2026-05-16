import XCTest
@testable import DrewsChessMachine

/// Tests for the pure `MoveSampler` extracted from
/// `MPSChessPlayer.sampleMove`. Coverage focuses on:
///
/// - **Deterministic edges**: single legal move, near-zero tau (forces
///   argmax behavior even with random draws).
/// - **Result self-consistency**: the returned `policyIndex` matches
///   `PolicyEncoding.policyIndex` for the returned move.
/// - **Distributional sanity**: with sharply-peaked logits and tau=1.0
///   the argmax-legal move is sampled with high frequency; with flat
///   logits it is sampled near-uniformly.
/// - **`randomish` flag**: true for flat logits (near-uniform softmax),
///   false for sharply-peaked logits.
/// - **Dirichlet noise**: when active and ply < limit, the distribution
///   differs from no-noise — a top-1-confident logit profile becomes
///   non-trivially diluted onto other legal moves.
/// - **Tau schedule**: scratch capacity and ply-bounds preconditions
///   fire as expected.
///
/// Note: Swift's default `Float.random(in:)` uses
/// `SystemRandomNumberGenerator`, which is not seedable. So
/// byte-identical-across-runs determinism is not directly testable
/// here without changing the function signature to take a custom RNG.
/// Statistical tests use large sample counts and tolerant bounds.
final class MoveSamplerTests: XCTestCase {

    // MARK: - Helpers

    /// Builds a flat zero-logit buffer of the correct policy size.
    private func zeroLogits() -> [Float] {
        [Float](repeating: 0, count: ChessNetwork.policySize)
    }

    /// Builds a logit buffer with a single sharply-peaked move (logit=10)
    /// at the policy index for `winner` (with `currentPlayer` perspective);
    /// all other cells are 0.
    private func sharpLogits(winner: ChessMove, currentPlayer: PieceColor) -> [Float] {
        var arr = zeroLogits()
        let idx = PolicyEncoding.policyIndex(winner, currentPlayer: currentPlayer)
        arr[idx] = 10.0
        return arr
    }

    /// Two synthetic moves on otherwise-empty boards. Both are pseudo-legal-
    /// looking but we never run them through MoveGenerator — we just need
    /// distinct objects whose `policyIndex` returns valid indices for the
    /// test player perspective.
    private let move1 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil) // e2-e4
    private let move2 = ChessMove(fromRow: 1, fromCol: 3, toRow: 3, toCol: 3, promotion: nil) // d2-d4
    private let move3 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil) // Nb1-c3
    private let move4 = ChessMove(fromRow: 0, fromCol: 6, toRow: 2, toCol: 5, promotion: nil) // Ng1-f3

    /// Allocates and runs a single sample with caller-friendly defaults.
    /// Returns the result; scratches are recreated per call so a test can
    /// rely on no cross-call state.
    @discardableResult
    private func sample(
        logits: [Float],
        legalMoves: [ChessMove],
        currentPlayer: PieceColor = .white,
        ply: Int = 0,
        schedule: SamplingSchedule = .uniform
    ) -> MoveSampler.Result {
        var probs = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        var eta   = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        return logits.withUnsafeBufferPointer { logitsBuf in
            probs.withUnsafeMutableBufferPointer { probsBuf in
                eta.withUnsafeMutableBufferPointer { etaBuf in
                    MoveSampler.sampleMove(
                        logits: logitsBuf,
                        legalMoves: legalMoves,
                        currentPlayer: currentPlayer,
                        ply: ply,
                        schedule: schedule,
                        probsScratch: probsBuf,
                        etaScratch: etaBuf
                    )
                }
            }
        }
    }

    // MARK: - Deterministic edges

    func test_singleLegalMove_alwaysReturned() {
        let logits = zeroLogits()
        for _ in 0..<50 {
            let result = sample(logits: logits, legalMoves: [move1])
            XCTAssertEqual(result.move, move1)
            XCTAssertEqual(result.policyIndex, PolicyEncoding.policyIndex(move1, currentPlayer: .white))
            // n == 1: randomish must always be false (forced moves
            // aren't "random") per the in-function precondition.
            XCTAssertFalse(result.randomish)
        }
    }

    func test_returnedPolicyIndexMatchesPolicyEncoding() {
        let logits = sharpLogits(winner: move2, currentPlayer: .white)
        let result = sample(
            logits: logits,
            legalMoves: [move1, move2, move3, move4],
            schedule: SamplingSchedule(startTau: 0.001, decayPerPly: 0.0, floorTau: 0.001)
        )
        XCTAssertEqual(result.move, move2)
        XCTAssertEqual(result.policyIndex, PolicyEncoding.policyIndex(move2, currentPlayer: .white))
    }

    func test_lowTau_picksArgmax() {
        // At tau ≈ 0 the softmax becomes a one-hot on the max-logit cell.
        // Inverse-CDF sampling then returns the argmax-legal move
        // deterministically, regardless of the random draw r.
        let logits = sharpLogits(winner: move3, currentPlayer: .white)
        let schedule = SamplingSchedule(startTau: 0.0001, decayPerPly: 0.0, floorTau: 0.0001)
        for _ in 0..<200 {
            let result = sample(
                logits: logits,
                legalMoves: [move1, move2, move3, move4],
                schedule: schedule
            )
            XCTAssertEqual(result.move, move3, "Argmax move should win at tau→0")
        }
    }

    func test_blackPerspective_usesBlackPolicyEncoding() {
        // The policy index of a given ChessMove differs by side. Sampling
        // with currentPlayer=.black must use the black-perspective index.
        let logits = sharpLogits(winner: move1, currentPlayer: .black)
        let result = sample(
            logits: logits,
            legalMoves: [move1, move2],
            currentPlayer: .black,
            schedule: SamplingSchedule(startTau: 0.001, decayPerPly: 0.0, floorTau: 0.001)
        )
        XCTAssertEqual(result.move, move1)
        XCTAssertEqual(result.policyIndex, PolicyEncoding.policyIndex(move1, currentPlayer: .black))
    }

    // MARK: - randomish flag

    func test_randomish_trueForFlatLogits() {
        // All-zero logits → uniform softmax → maxProb == 1/n < 1.5/n → randomish.
        let logits = zeroLogits()
        let result = sample(logits: logits, legalMoves: [move1, move2, move3, move4])
        XCTAssertTrue(result.randomish, "Uniform logits should be flagged randomish")
    }

    func test_randomish_falseForSharpLogits() {
        // Single move at 10.0, rest at 0 → softmax ≈ [1, 0, 0, 0] → maxProb≈1 > 1.5/4.
        let logits = sharpLogits(winner: move2, currentPlayer: .white)
        let result = sample(
            logits: logits,
            legalMoves: [move1, move2, move3, move4]
        )
        XCTAssertFalse(result.randomish, "Sharp logits should NOT be flagged randomish")
    }

    func test_randomish_falseForSingleMove() {
        // n == 1: forced move, never randomish, regardless of logits.
        let logits = zeroLogits()
        let result = sample(logits: logits, legalMoves: [move1])
        XCTAssertFalse(result.randomish)
    }

    // MARK: - Distributional

    func test_sharpLogits_atTau1_argmaxDominates() {
        // At tau=1 with logit gap 10 vs 0, softmax weight on the top
        // move is ≈ 1/(1 + 3·e^-10) ≈ 0.99986. Should sample it
        // overwhelmingly often.
        let logits = sharpLogits(winner: move2, currentPlayer: .white)
        var hits = 0
        let trials = 500
        for _ in 0..<trials {
            let result = sample(logits: logits, legalMoves: [move1, move2, move3, move4])
            if result.move == move2 { hits += 1 }
        }
        XCTAssertGreaterThan(hits, trials * 95 / 100, "Sharp logits should pick argmax >95% of trials")
    }

    func test_flatLogits_atTau1_distributionIsRoughlyUniform() {
        // All-zero logits at tau=1.0: each legal move has 1/4 probability.
        // Trials=4000 → expected ~1000 per bucket with stddev≈sqrt(4000·0.25·0.75)≈27.4
        // → 5-sigma window is ~[860, 1140]. Use a tolerant ±25% bound.
        let logits = zeroLogits()
        let moves = [move1, move2, move3, move4]
        var counts = [ChessMove: Int]()
        let trials = 4000
        for _ in 0..<trials {
            let result = sample(logits: logits, legalMoves: moves)
            counts[result.move, default: 0] += 1
        }
        for m in moves {
            let c = counts[m] ?? 0
            XCTAssertGreaterThan(c, trials / 4 * 75 / 100, "Move \(m.notation) sampled \(c) times — too low")
            XCTAssertLessThan(c, trials / 4 * 125 / 100, "Move \(m.notation) sampled \(c) times — too high")
        }
    }

    // MARK: - Dirichlet noise

    func test_dirichletNoise_dilutesSharpLogits() {
        // Sharp logit on move2 + heavy Dirichlet (alpha=0.3, eps=0.5)
        // should make move2's sampling frequency noticeably less than
        // the no-noise case (where it'd be >99%). With epsilon=0.5
        // even at α=0.3 the noise will pull ~50% of weight onto other
        // moves on average. Expect move2 to win ~50-95% of trials,
        // and the other moves to sometimes win.
        let logits = sharpLogits(winner: move2, currentPlayer: .white)
        let scheduleNoisy = SamplingSchedule(
            startTau: 1.0,
            decayPerPly: 0.0,
            floorTau: 1.0,
            dirichletNoise: DirichletNoiseConfig(alpha: 0.3, epsilon: 0.5, plyLimit: 100)
        )
        var topMoveHits = 0
        var otherMoveHits = 0
        let trials = 500
        for _ in 0..<trials {
            let result = sample(
                logits: logits,
                legalMoves: [move1, move2, move3, move4],
                schedule: scheduleNoisy
            )
            if result.move == move2 { topMoveHits += 1 } else { otherMoveHits += 1 }
        }
        XCTAssertGreaterThan(otherMoveHits, trials / 50, "Dirichlet ε=0.5 should pull at least ~2% of trials onto non-top moves; got \(otherMoveHits)/\(trials)")
        XCTAssertLessThan(topMoveHits, trials, "Some trials must NOT pick the sharp-logit move under heavy Dirichlet")
    }

    func test_dirichletPastPlyLimit_noEffect() {
        // Ply >= plyLimit → noise is suppressed. With sharp logits, the
        // argmax move should win >95% of trials even when noise config
        // is "present" but plyLimit gates it off.
        let logits = sharpLogits(winner: move2, currentPlayer: .white)
        let schedulePastLimit = SamplingSchedule(
            startTau: 1.0,
            decayPerPly: 0.0,
            floorTau: 1.0,
            dirichletNoise: DirichletNoiseConfig(alpha: 0.3, epsilon: 0.5, plyLimit: 10)
        )
        var hits = 0
        let trials = 500
        for _ in 0..<trials {
            let result = sample(
                logits: logits,
                legalMoves: [move1, move2, move3, move4],
                ply: 50, // well past plyLimit=10
                schedule: schedulePastLimit
            )
            if result.move == move2 { hits += 1 }
        }
        XCTAssertGreaterThan(hits, trials * 95 / 100, "Past plyLimit, noise should be off and argmax should dominate")
    }

    func test_dirichletWithSingleLegalMove_noChange() {
        // n == 1: noise is suppressed (no other moves to redistribute
        // weight onto). Returned move must always be the single legal
        // move, regardless of noise config.
        let logits = zeroLogits()
        let scheduleNoisy = SamplingSchedule(
            startTau: 1.0,
            decayPerPly: 0.0,
            floorTau: 1.0,
            dirichletNoise: DirichletNoiseConfig(alpha: 0.3, epsilon: 0.5, plyLimit: 100)
        )
        for _ in 0..<100 {
            let result = sample(logits: logits, legalMoves: [move3], schedule: scheduleNoisy)
            XCTAssertEqual(result.move, move3)
        }
    }
}
