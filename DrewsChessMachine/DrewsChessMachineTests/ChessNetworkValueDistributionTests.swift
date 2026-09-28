import XCTest
@testable import DrewsChessMachine

/// The single-pass W/D/L readback (`evaluateWithValueDistribution`) and the
/// concurrency guarantee of single-position `evaluate` (Lichess bot plan
/// §8.3, §8.6).
///
/// - The new path must produce exactly the policy `evaluate` produces, and
///   the same W/D/L the older second-pass `evaluateValueDistribution`
///   produces, from the same one forward pass.
/// - Several games evaluating on one network at once must get exactly the
///   results they would get one at a time. `ChessNetwork`'s doc comment
///   warned that concurrent callers need separate networks or explicit
///   serialization; the serial `executionQueue` is meant to be that
///   serialization, and this test proves it rather than trusting either
///   reading of the comment.
final class ChessNetworkValueDistributionTests: XCTestCase {

    private static let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ]

    private func boards(for net: ChessMPSNetwork) throws -> [[Float]] {
        try Self.fens.map { fen in
            BoardEncoder.encode(try FENParser.parse(fen), encoding: net.inputEncoding)
        }
    }

    private struct PolicyAndValue: Sendable, Equatable {
        let policy: [Float]
        let value: Float
    }

    private struct PolicyAndDistribution: Sendable {
        let policy: [Float]
        let win: Float
        let draw: Float
        let loss: Float
    }

    private static func evaluate(_ net: ChessMPSNetwork, _ board: [Float]) async throws -> PolicyAndValue {
        let box = SyncBox<PolicyAndValue?>(nil)
        try await net.evaluate(board: board) { policy, value in
            box.value = PolicyAndValue(policy: Array(policy), value: value)
        }
        return try XCTUnwrap(box.value, "evaluate did not invoke its consumer")
    }

    private static func evaluateWithDistribution(_ net: ChessMPSNetwork, _ board: [Float]) async throws -> PolicyAndDistribution {
        let box = SyncBox<PolicyAndDistribution?>(nil)
        try await net.evaluateWithValueDistribution(board: board) { policy, wdl in
            box.value = PolicyAndDistribution(policy: Array(policy), win: wdl.win, draw: wdl.draw, loss: wdl.loss)
        }
        return try XCTUnwrap(box.value, "evaluateWithValueDistribution did not invoke its consumer")
    }

    func testSinglePassMatchesPolicyAndSecondPassDistribution() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        for (index, board) in try boards(for: net).enumerated() {
            let scalarPath = try await Self.evaluate(net, board)
            let singlePass = try await Self.evaluateWithDistribution(net, board)
            let secondPass = try await net.evaluateValueDistribution(board: board)

            XCTAssertEqual(singlePass.policy, scalarPath.policy, "policy differs, position \(index)")
            XCTAssertEqual(singlePass.win, secondPass.win, accuracy: 1e-6, "p_win, position \(index)")
            XCTAssertEqual(singlePass.draw, secondPass.draw, accuracy: 1e-6, "p_draw, position \(index)")
            XCTAssertEqual(singlePass.loss, secondPass.loss, accuracy: 1e-6, "p_loss, position \(index)")
            // The graph may compute in bfloat16, whose spacing just below
            // one is 2^-8; each of the three probabilities is rounded
            // independently, so the sum can miss one by up to three
            // half-spacings.
            XCTAssertEqual(singlePass.win + singlePass.draw + singlePass.loss, 1, accuracy: 3 * 0x1p-9,
                           "W/D/L should sum to 1, position \(index)")
            // The graph's scalar and the two probabilities are each rounded
            // to bfloat16 on their own (spacing up to 2^-8 below one), so
            // the difference can miss the scalar by three half-spacings.
            XCTAssertEqual(singlePass.win - singlePass.loss, scalarPath.value, accuracy: 3 * 0x1p-9,
                           "scalar value is p_win − p_loss, position \(index)")
        }
    }

    func testConcurrentEvaluateOnOneNetworkMatchesSequential() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        let positions = try boards(for: net)
        // Several rounds over every position, so many calls are in flight
        // at once.
        let jobs = (0..<4).flatMap { _ in positions.indices }

        var sequential: [Int: PolicyAndValue] = [:]
        for index in positions.indices {
            sequential[index] = try await Self.evaluate(net, positions[index])
        }

        let concurrent = try await withThrowingTaskGroup(of: (Int, PolicyAndValue).self) { group in
            for index in jobs {
                let board = positions[index]
                group.addTask {
                    (index, try await Self.evaluate(net, board))
                }
            }
            var results: [(Int, PolicyAndValue)] = []
            for try await result in group {
                results.append(result)
            }
            return results
        }

        XCTAssertEqual(concurrent.count, jobs.count)
        for (index, result) in concurrent {
            XCTAssertEqual(result, sequential[index], "concurrent result differs from sequential, position \(index)")
        }
    }
}
