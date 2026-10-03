//
//  DropoutRunStreamTests.swift
//  DrewsChessMachineTests
//
//  The GUI builds its trainer once and reuses it for every Play-and-Train
//  run of a launch, so each run start hands the trainer its own seed's
//  `dropout` stream (`ChessTrainer.beginDropoutStream`). These pin that a
//  reused trainer then starts exactly where a trainer newly built with that
//  stream starts — whatever it was built with and however many steps it has
//  already taken.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class DropoutRunStreamTests: XCTestCase {

    /// One small group with a nonzero dropout multiplier, so the training
    /// graph draws dropout randomness (the shape `DropoutRNGStateTests` uses).
    private func archWithDropout() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.blockGroups = [
            BlockGroup(
                count: 2, channels: NetworkArchitecture.current.towerOutputChannels,
                conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .none, seReductionRatio: 4,
                useRezero: true, rezeroAlphaInit: 0.5,
                activationFunction: .relu, activationStyle: .pre,
                skipMerge: .cleanAdd, dropoutMultiplier: 0.5
            )
        ]
        return arch
    }

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    func testBeginningARunStreamMatchesATrainerBuiltWithIt() async throws {
        try requireMetal()
        let runStreams = DCMRandomStreams(masterSeed: 77)

        let built = try ChessTrainer(dropoutStream: runStreams.generator(.dropout), arch: archWithDropout(), initialization: .seeded(initSeed: 1))
        built.dropoutRate = 0.2
        let expected = try await built.captureDropoutState()

        let reused = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: archWithDropout(), initialization: .seeded(initSeed: 2))
        reused.dropoutRate = 0.2
        _ = try await reused.trainStep(batchSize: 8)
        _ = try await reused.trainStep(batchSize: 8)
        let afterTwoSteps = try await reused.captureDropoutState()
        XCTAssertNotEqual(afterTwoSteps, expected)

        try await reused.beginDropoutStream(runStreams.generator(.dropout))
        let afterBegin = try await reused.captureDropoutState()
        XCTAssertEqual(afterBegin, expected)

        _ = try await built.trainStep(batchSize: 8)
        _ = try await reused.trainStep(batchSize: 8)
        let reusedNext = try await reused.captureDropoutState()
        let builtNext = try await built.captureDropoutState()
        XCTAssertEqual(reusedNext, builtNext, "the two must advance identically from the shared start")
    }

    func testDifferentRunSeedsStartDifferentMasks() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: archWithDropout(), initialization: .seeded(initSeed: 1))
        try await trainer.beginDropoutStream(DCMRandomStreams(masterSeed: 5).generator(.dropout))
        let first = try await trainer.captureDropoutState()
        try await trainer.beginDropoutStream(DCMRandomStreams(masterSeed: 6).generator(.dropout))
        let second = try await trainer.captureDropoutState()
        XCTAssertNotEqual(second, first)
    }
}
