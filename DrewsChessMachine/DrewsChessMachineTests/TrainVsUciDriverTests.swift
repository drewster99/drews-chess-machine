//
//  TrainVsUciDriverTests.swift
//  DrewsChessMachineTests
//
//  Unit coverage for the pure, correctness-critical logic of the
//  train-vs-UCI driver: mapping a game result to the trainer's outcome
//  (+1 win / -1 loss / 0 draw) given which colour the trainer played.
//  A sign error here would poison every value/return target the driver
//  writes to the replay buffer, so it is pinned exhaustively.
//
//  The full driver flow (batched trainer eval, async opponent moves,
//  slot state machine) requires the MPSGraph network + live UCI engines
//  and is exercised end-to-end by the `--train-vs-uci` smoke run rather
//  than XCTest, per the project's Metal-dependent testing convention.
//

import XCTest
@testable import DrewsChessMachine

final class TrainVsUciDriverTests: XCTestCase {

    func testTrainerOutcomeCheckmate() {
        // Trainer wins when the checkmate winner is its own colour.
        XCTAssertEqual(
            TrainVsUciDriver.trainerOutcome(result: .checkmate(winner: .white), trainerColor: .white), 1)
        XCTAssertEqual(
            TrainVsUciDriver.trainerOutcome(result: .checkmate(winner: .black), trainerColor: .black), 1)
        // Trainer loses when the opponent's colour delivered mate.
        XCTAssertEqual(
            TrainVsUciDriver.trainerOutcome(result: .checkmate(winner: .black), trainerColor: .white), -1)
        XCTAssertEqual(
            TrainVsUciDriver.trainerOutcome(result: .checkmate(winner: .white), trainerColor: .black), -1)
    }

    func testTrainerOutcomeDrawsAreZeroForBothColors() {
        let draws: [GameResult] = [
            .stalemate,
            .drawByFiftyMoveRule,
            .drawByInsufficientMaterial,
            .drawByThreefoldRepetition,
        ]
        for draw in draws {
            XCTAssertEqual(
                TrainVsUciDriver.trainerOutcome(result: draw, trainerColor: .white), 0,
                "\(draw) should be 0 for a white trainer")
            XCTAssertEqual(
                TrainVsUciDriver.trainerOutcome(result: draw, trainerColor: .black), 0,
                "\(draw) should be 0 for a black trainer")
        }
    }
}
