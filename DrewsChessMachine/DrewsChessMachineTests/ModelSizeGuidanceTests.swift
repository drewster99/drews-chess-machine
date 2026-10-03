//
//  ModelSizeGuidanceTests.swift
//  DrewsChessMachineTests
//
//  The size guidance scaled to installed memory: the recommended range, the
//  reference batch's limit, the batch-size ladder below it, the "likely too
//  large" warning, and the one refusal — a training state larger than
//  physical memory. Every case is computed for an explicit memory size, so
//  the results do not depend on the machine running the tests.
//

import XCTest
@testable import DrewsChessMachine

final class ModelSizeGuidanceTests: XCTestCase {

    private let gigabyte: UInt64 = 1 << 30

    private func verdict(_ parameters: Int, memoryGB: UInt64) -> ModelSizeGuidance.Verdict {
        ModelSizeGuidance(parameterCount: parameters, physicalMemoryBytes: memoryGB * gigabyte).verdict
    }

    func testReferenceMachineThresholds() {
        XCTAssertEqual(verdict(8_445_748, memoryGB: 64), .withinRecommendedSize)
        XCTAssertEqual(verdict(15_000_000, memoryGB: 64), .withinRecommendedSize)
        XCTAssertEqual(verdict(15_000_001, memoryGB: 64), .trainsAtReferenceBatch)
        XCTAssertEqual(verdict(20_000_000, memoryGB: 64), .trainsAtReferenceBatch)
        XCTAssertEqual(verdict(20_000_001, memoryGB: 64), .trainsAtReducedBatch(batchSize: 2048))
        // √2 × 20M ≈ 28.28M at batch 2048; 40M at 1024; √8 × 20M ≈ 56.57M at 512.
        XCTAssertEqual(verdict(28_000_000, memoryGB: 64), .trainsAtReducedBatch(batchSize: 2048))
        XCTAssertEqual(verdict(30_000_000, memoryGB: 64), .trainsAtReducedBatch(batchSize: 1024))
        XCTAssertEqual(verdict(40_000_000, memoryGB: 64), .trainsAtReducedBatch(batchSize: 1024))
        XCTAssertEqual(verdict(56_000_000, memoryGB: 64), .trainsAtReducedBatch(batchSize: 512))
        XCTAssertEqual(verdict(57_000_000, memoryGB: 64), .likelyTooLargeToTrain)
    }

    func testThresholdsScaleWithMemory() {
        XCTAssertEqual(verdict(3_750_000, memoryGB: 16), .withinRecommendedSize)
        XCTAssertEqual(verdict(3_750_001, memoryGB: 16), .trainsAtReferenceBatch)
        XCTAssertEqual(verdict(5_000_001, memoryGB: 16), .trainsAtReducedBatch(batchSize: 2048))
        XCTAssertEqual(verdict(30_000_000, memoryGB: 128), .withinRecommendedSize)
        XCTAssertEqual(verdict(35_000_000, memoryGB: 128), .trainsAtReferenceBatch)
    }

    func testTrainingStateLargerThanMemoryIsTheOneRefusal() {
        // Four fp32 copies per parameter: exactly the memory fits, one more does not.
        let fitsExactly = Int(64 * gigabyte) / ModelSizeGuidance.trainingBytesPerParameter
        XCTAssertEqual(verdict(fitsExactly, memoryGB: 64), .likelyTooLargeToTrain)
        let guidance = ModelSizeGuidance(parameterCount: fitsExactly + 1, physicalMemoryBytes: 64 * gigabyte)
        XCTAssertEqual(guidance.verdict, .trainingStateExceedsPhysicalMemory)
        XCTAssertThrowsError(try guidance.requireTrainingStateFitsInPhysicalMemory()) { error in
            XCTAssertTrue(String(describing: error).contains("physical memory"), "\(error)")
        }
        XCTAssertNoThrow(try ModelSizeGuidance(parameterCount: fitsExactly, physicalMemoryBytes: 64 * gigabyte)
            .requireTrainingStateFitsInPhysicalMemory())
        XCTAssertEqual(verdict(Int.max, memoryGB: 64), .trainingStateExceedsPhysicalMemory)
    }

    func testReducedBatchesStopAtTheSmallestRecommended() {
        XCTAssertEqual(ModelSizeGuidance.reducedBatchSizes, [2048, 1024, 512])
    }

    func testLogLineCarriesTheVerdictAndThresholds() {
        let line = ModelSizeGuidance(parameterCount: 30_000_000, physicalMemoryBytes: 64 * gigabyte)
            .logLine(event: "test")
        XCTAssertTrue(line.hasPrefix("[ARCH] size guidance (test): parameters=30000000 physical_memory=64GB "), line)
        XCTAssertTrue(line.contains("recommended_max=15000000 batch4096_max=20000000 batch512_max=56568542 "), line)
        XCTAssertTrue(line.contains("verdict=trains_at_batch_1024 | "), line)
    }

    func testReadoutNamesTheMemoryAndTheLimit() {
        let within = ModelSizeGuidance(parameterCount: 8_000_000, physicalMemoryBytes: 64 * gigabyte).readout
        XCTAssertTrue(within.contains("15.0M") && within.contains("64 GB"), within)
        let reduced = ModelSizeGuidance(parameterCount: 30_000_000, physicalMemoryBytes: 64 * gigabyte).readout
        XCTAssertTrue(reduced.contains("batch 1024") && reduced.contains("40.0M"), reduced)
    }
}
