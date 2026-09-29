//
//  HeadLossGraphTests.swift
//  DrewsChessMachineTests
//
//  Pins the invariants of the training loss's targets and logit centering
//  (`HeadLossGraph`), which decide what the heads' shared logit offset feels:
//
//  - Every policy and value target row sums to exactly 1 in fp32, for any
//    legal-move count and ε.
//  - A single-legal-move position gets zero complement weight and a finite
//    complement target (never NaN, even at ε = 0).
//  - A cross-entropy on centered logits gives the shared (all-ones)
//    direction exactly zero gradient, even against a target that does not
//    sum to 1 and logits that carry a large offset.
//
//  The builders run on a small standalone graph with the production policy
//  width, so these are the production code paths, not mirrors.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class HeadLossGraphTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private func floatData(_ values: [Float]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func int32Data(_ values: [Int32]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func read(_ results: [MPSGraphTensor: MPSGraphTensorData], _ tensor: MPSGraphTensor,
                      count: Int, file: StaticString = #filePath, line: UInt = #line) throws -> [Float] {
        let data = try XCTUnwrap(results[tensor], "tensor missing from results", file: file, line: line)
        XCTAssertEqual(data.dataType, .float32, "loss-path tensors must be fp32", file: file, line: line)
        return ChessNetwork.readFloatsFP32(from: data, count: count)
    }

    // MARK: - Policy targets

    /// Rows with 1, 2, 3, 7, 29 and 218 legal moves (the played move always
    /// among them), at several ε.
    func testPolicyTargetsSumToOneAndSingleLegalHasZeroComplementWeight() throws {
        try requireMetal()
        let policySize = ChessNetwork.policySize
        let legalCounts = [1, 2, 3, 7, 29, 218]
        let batch = legalCounts.count
        var mask = [Float](repeating: 0, count: batch * policySize)
        var played = [Int32](repeating: 0, count: batch)
        for (row, count) in legalCounts.enumerated() {
            // Spread the legal cells across the policy with a stride so they
            // are not a contiguous block; the played move is the first one.
            for k in 0..<count { mask[row * policySize + (k * 17 + row) % policySize] = 1 }
            played[row] = Int32(row)   // the k = 0 legal cell
        }

        for epsilon: Float in [0, 0.05, 0.1, 1.0 / 3.0] {
            let graph = MPSGraph()
            let legalMask = graph.constant(
                floatData(mask), shape: [NSNumber(value: batch), NSNumber(value: policySize)], dataType: .float32)
            let movePlayed = graph.constant(int32Data(played), shape: [NSNumber(value: batch)], dataType: .int32)
            let eps = graph.constant(floatData([epsilon]), shape: [1], dataType: .float32)
            let targets = HeadLossGraph.policyTargets(
                graph: graph, movePlayed: movePlayed, legalMask: legalMask, epsilon: eps, policySize: policySize)
            let results = graph.run(
                feeds: [:],
                targetTensors: [targets.smoothed, targets.complement, targets.complementValid],
                targetOperations: nil
            )
            let smoothed = try read(results, targets.smoothed, count: batch * policySize)
            let complement = try read(results, targets.complement, count: batch * policySize)
            let valid = try read(results, targets.complementValid, count: batch)

            for (row, count) in legalCounts.enumerated() {
                let range = (row * policySize)..<((row + 1) * policySize)
                let smoothedSum = smoothed[range].reduce(0.0) { $0 + Double($1) }
                XCTAssertEqual(smoothedSum, 1, accuracy: 1e-6, "ε=\(epsilon) |legal|=\(count): positive target sum")
                XCTAssertTrue(smoothed[range].allSatisfy { $0.isFinite && $0 >= 0 })
                // No mass off the legal set.
                for i in range where mask[i] == 0 {
                    XCTAssertEqual(smoothed[i], 0, "ε=\(epsilon) |legal|=\(count): positive target mass on illegal cell")
                    XCTAssertEqual(complement[i], 0, "ε=\(epsilon) |legal|=\(count): complement mass on illegal cell")
                }
                XCTAssertTrue(complement[range].allSatisfy { $0.isFinite && $0 >= 0 },
                              "ε=\(epsilon) |legal|=\(count): complement target must be finite")
                if count == 1 {
                    XCTAssertEqual(valid[row], 0, "ε=\(epsilon): a forced move gets zero complement weight")
                } else {
                    XCTAssertEqual(valid[row], 1, "ε=\(epsilon) |legal|=\(count): complement weight")
                    let complementSum = complement[range].reduce(0.0) { $0 + Double($1) }
                    XCTAssertEqual(complementSum, 1, accuracy: 1e-6, "ε=\(epsilon) |legal|=\(count): complement sum")
                }
            }
        }
    }

    // MARK: - Value target

    func testValueTargetSumsToOneOnTheOutcomeSlot() throws {
        try requireMetal()
        let classes = 3
        // Win, draw, loss, and a draw rewritten by a partial drawPenalty.
        let zs: [Float] = [1, 0, -1, -0.1]
        let expectedSlot = [0, 1, 2, 1]
        for epsilon: Float in [0, 0.1, 1.0 / 3.0] {
            let graph = MPSGraph()
            let z = graph.constant(floatData(zs), shape: [NSNumber(value: zs.count), 1], dataType: .float32)
            let eps = graph.constant(floatData([epsilon]), shape: [1], dataType: .float32)
            let target = HeadLossGraph.valueTarget(graph: graph, z: z, epsilon: eps, classes: classes)
            let results = graph.run(feeds: [:], targetTensors: [target], targetOperations: nil)
            let values = try read(results, target, count: zs.count * classes)
            for row in 0..<zs.count {
                let slice = Array(values[(row * classes)..<((row + 1) * classes)])
                let sum = slice.reduce(0.0) { $0 + Double($1) }
                XCTAssertEqual(sum, 1, accuracy: 1e-6, "ε=\(epsilon) z=\(zs[row]): value target sum")
                let argmax = try XCTUnwrap(slice.indices.max { slice[$0] < slice[$1] })
                XCTAssertEqual(argmax, expectedSlot[row], "ε=\(epsilon) z=\(zs[row]): outcome slot")
            }
        }
    }

    // MARK: - Centering

    /// Gradient of `mean(CE(center(logits), y))` with respect to the raw
    /// logits: every row must sum to ~0. The target deliberately sums to less
    /// than 1 and the logits sit on a large shared offset — the two
    /// conditions that drive the offset without centering.
    func testCenteredCrossEntropyGivesSharedDirectionZeroGradient() throws {
        try requireMetal()
        let classes = 5
        let batch = 3
        let offset: Float = 500
        var logitValues = [Float](repeating: 0, count: batch * classes)
        for i in 0..<logitValues.count { logitValues[i] = offset + Float((i * 7) % 11) * 0.3 - 1.5 }
        // Rows sum to 0.9, 0.99 and 1.0.
        let targetValues: [Float] = [
            0.5, 0.2, 0.1, 0.1, 0.0,
            0.0, 0.0, 0.99, 0.0, 0.0,
            0.2, 0.2, 0.2, 0.2, 0.2,
        ]

        let graph = MPSGraph()
        let logits = graph.variable(
            with: floatData(logitValues),
            shape: [NSNumber(value: batch), NSNumber(value: classes)],
            dataType: .float32,
            name: "logits"
        )
        let labels = graph.constant(
            floatData(targetValues), shape: [NSNumber(value: batch), NSNumber(value: classes)], dataType: .float32)
        let (centered, meanPerRow) = HeadLossGraph.centerLogits(logits, graph: graph, name: "test_logits")
        let ce = graph.softMaxCrossEntropy(centered, labels: labels, axis: 1, reuctionType: .none, name: "ce")
        let loss = graph.mean(of: graph.reshape(ce, shape: [-1, 1], name: nil), axes: [0, 1], name: "loss")
        let gradients = graph.gradients(of: loss, with: [logits], name: "grads")
        let gradient = try XCTUnwrap(gradients[logits], "no gradient reached the logits")

        let results = graph.run(feeds: [:], targetTensors: [gradient, meanPerRow], targetOperations: nil)
        let g = try read(results, gradient, count: batch * classes)
        let means = try read(results, meanPerRow, count: batch)

        for row in 0..<batch {
            let slice = g[(row * classes)..<((row + 1) * classes)]
            let sum = slice.reduce(0.0) { $0 + Double($1) }
            let magnitude = slice.reduce(0.0) { $0 + abs(Double($1)) }
            XCTAssertGreaterThan(magnitude, 1e-3, "row \(row): the gradient itself must not vanish")
            XCTAssertEqual(sum, 0, accuracy: 1e-6 * max(1, magnitude), "row \(row): shared-direction gradient")
            XCTAssertEqual(Double(means[row]), Double(offset), accuracy: 2, "row \(row): the offset is reported")
        }
    }
}
