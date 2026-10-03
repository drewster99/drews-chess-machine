//
//  TrainerDeclaredDefaultsTests.swift
//  DrewsChessMachineTests
//
//  A bare `ChessTrainer()` (tests, the timing sweeps) must start at the
//  declared training-parameter defaults wherever the trainer's default is
//  meant to match one, so the two cannot drift apart again: five trainer
//  defaults had drifted from their declarations unnoticed.
//

import XCTest
@testable import DrewsChessMachine

final class TrainerDeclaredDefaultsTests: XCTestCase {

    func testABareTrainerStartsAtTheDeclaredDefaults() throws {
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.learningRate, Float(LearningRate.declaredDefault))
        XCTAssertEqual(trainer.entropyRegularizationCoeff, Float(EntropyBonus.declaredDefault))
        XCTAssertEqual(trainer.policyLossWeight, Float(PolicyLossWeight.declaredDefault))
        XCTAssertEqual(trainer.valueLossWeight, Float(ValueLossWeight.declaredDefault))
        XCTAssertEqual(trainer.illegalMassPenaltyWeight, Float(IllegalMassWeight.declaredDefault))
        XCTAssertEqual(trainer.policyLabelSmoothingEpsilon, Float(PolicyLabelSmoothingEpsilon.declaredDefault))
        XCTAssertEqual(trainer.policyLabelSmoothingMode.rawValue, PolicyLabelSmoothingModeParameter.declaredDefault)
        XCTAssertEqual(trainer.policyLabelSmoothingPerMove, Float(PolicyLabelSmoothingPerMove.declaredDefault))
        XCTAssertEqual(trainer.policyLabelSmoothingPerMoveCap, Float(PolicyLabelSmoothingPerMoveCap.declaredDefault))
        XCTAssertEqual(trainer.useSignedAdvantageComplementCE, SignedAdvantageComplementCE.declaredDefault)
        XCTAssertEqual(trainer.sqrtBatchScalingForLR, SqrtBatchScalingLR.declaredDefault)
    }
}
