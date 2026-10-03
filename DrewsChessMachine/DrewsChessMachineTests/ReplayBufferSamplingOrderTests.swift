//
//  ReplayBufferSamplingOrderTests.swift
//  DrewsChessMachineTests
//
//  Two pieces of the trainer's minibatch draw that must not depend on
//  anything but the buffer's contents: uniform draws pick a logical
//  (age-ordered) index, not a ring position, and the length-tilt β is
//  solved from the length histogram in a fixed order.
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferSamplingOrderTests: XCTestCase {

    // MARK: - Logical index → ring slot

    func testBeforeTheRingFillsLogicalIndexIsTheSlot() {
        for logical in 0..<5 {
            XCTAssertEqual(ReplayBuffer.physicalSlot(logicalIndex: logical, storedCount: 5, capacity: 8, writeIndex: 5), logical)
        }
    }

    func testOnceFullLogicalZeroIsTheOldestSlotAndTheOrderWraps() {
        // Full ring of 8 whose next write goes to slot 3: slot 3 holds the
        // oldest position, slot 2 the newest.
        let expected = [3, 4, 5, 6, 7, 0, 1, 2]
        for (logical, slot) in expected.enumerated() {
            XCTAssertEqual(ReplayBuffer.physicalSlot(logicalIndex: logical, storedCount: 8, capacity: 8, writeIndex: 3), slot)
        }
    }

    func testAFullRingWithItsWritePointerAtZeroIsInSlotOrder() {
        for logical in 0..<6 {
            XCTAssertEqual(ReplayBuffer.physicalSlot(logicalIndex: logical, storedCount: 6, capacity: 6, writeIndex: 0), logical)
        }
    }

    // MARK: - Length-tilt β

    /// The same histogram, built in two different insertion orders (so the
    /// two dictionaries may iterate differently), gives a bit-identical β.
    func testLengthTiltBetaDependsOnlyOnTheHistogramContents() {
        let entries: [(UInt16, Int)] = [
            (12, 40), (37, 900), (61, 3_500), (88, 2_200), (140, 1_300),
            (201, 600), (260, 250), (333, 90), (412, 30), (515, 7),
        ]
        var forward: [UInt16: Int] = [:]
        for (length, count) in entries { forward[length] = count }
        var backward: [UInt16: Int] = [:]
        backward.reserveCapacity(1_024)
        for (length, count) in entries.reversed() { backward[length] = count }

        for target in [40, 70, 95, 120] {
            let a = ReplayBuffer.lengthTiltBeta(residentLengthHistogram: forward, target: target)
            let b = ReplayBuffer.lengthTiltBeta(residentLengthHistogram: backward, target: target)
            XCTAssertEqual(a.beta.bitPattern, b.beta.bitPattern, "target \(target)")
            XCTAssertEqual(a.infeasible, b.infeasible, "target \(target)")
            XCTAssertEqual(a.shortestResidentLength, 12)
            XCTAssertEqual(b.shortestResidentLength, 12)
        }
    }

    func testLengthTiltBetaOfAnEmptyHistogramIsNoTilt() {
        let result = ReplayBuffer.lengthTiltBeta(residentLengthHistogram: [:], target: 80)
        XCTAssertEqual(result.beta, 0)
        XCTAssertFalse(result.infeasible)
        XCTAssertEqual(result.shortestResidentLength, 0)
    }

    func testATargetBelowTheShortestGameIsInfeasible() {
        let result = ReplayBuffer.lengthTiltBeta(residentLengthHistogram: [50: 10, 90: 10], target: 40)
        XCTAssertTrue(result.infeasible)
        XCTAssertEqual(result.shortestResidentLength, 50)
    }
}
