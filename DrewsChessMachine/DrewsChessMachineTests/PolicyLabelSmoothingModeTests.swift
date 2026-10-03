//
//  PolicyLabelSmoothingModeTests.swift
//  DrewsChessMachineTests
//
//  Pins the per-move policy label-smoothing form ("arm B" of
//  documentation/plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md) and the
//  parameter plumbing that selects it:
//
//  - Per-move targets: every row sums to 1 for every legal count 1…218; each
//    non-played legal move gets δ below the cap and an equal share of the cap
//    above it; a single legal move gives an exact one-hot; illegal cells are
//    exactly 0; the complement target puts the per-move floor on the played
//    move and the rest evenly on the other legal moves.
//  - Fixed-total mode through the mode-selecting builder is bit-identical to
//    the fixed-total builder, and both match the closed-form target.
//  - The mode is a fed selector: one built graph switches forms by feed.
//  - The three parameters: declared ids, defaults and ranges; the mode's
//    range pinned to its enum; parameters.json load + apply; the shared
//    trainer path; session-state Optional fields and the pre-feature
//    resolution.
//
//  The graph tests run the production builders on a small standalone graph
//  with the production policy width, in the style of `HeadLossGraphTests`.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class PolicyLabelSmoothingModeTests: XCTestCase {

    // MARK: - Fixture

    private let policySize = ChessNetwork.policySize
    /// Every legal-move count a chess position can have, one row each.
    private let legalCounts = Array(1...218)

    private struct Batch {
        let mask: [Float]
        let played: [Int32]
        let rows: Int
    }

    /// One row per entry of `counts`. The legal cells are spread across the
    /// policy with a stride so they are not a contiguous block; the played
    /// move is the first of them.
    private func makeBatch(_ counts: [Int]) -> Batch {
        let rows = counts.count
        var mask = [Float](repeating: 0, count: rows * policySize)
        var played = [Int32](repeating: 0, count: rows)
        for (row, count) in counts.enumerated() {
            for k in 0..<count { mask[row * policySize + (k * 17 + row) % policySize] = 1 }
            played[row] = Int32(row % policySize)
        }
        return Batch(mask: mask, played: played, rows: rows)
    }

    private struct Targets {
        let smoothed: [Float]
        let complement: [Float]
        let valid: [Float]
    }

    private func requireMetal() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        return device
    }

    private func floatData(_ values: [Float]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func int32Data(_ values: [Int32]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func scalar(_ graph: MPSGraph, _ value: Float) -> MPSGraphTensor {
        graph.constant(floatData([value]), shape: [1], dataType: .float32)
    }

    private func read(_ results: [MPSGraphTensor: MPSGraphTensorData], _ tensor: MPSGraphTensor,
                      count: Int, file: StaticString = #filePath, line: UInt = #line) throws -> [Float] {
        let data = try XCTUnwrap(results[tensor], "tensor missing from results", file: file, line: line)
        XCTAssertEqual(data.dataType, .float32, "loss-path tensors must be fp32", file: file, line: line)
        return ChessNetwork.readFloatsFP32(from: data, count: count)
    }

    private func batchInputs(_ graph: MPSGraph, _ batch: Batch) -> (legalMask: MPSGraphTensor, movePlayed: MPSGraphTensor) {
        let legalMask = graph.constant(
            floatData(batch.mask),
            shape: [NSNumber(value: batch.rows), NSNumber(value: policySize)],
            dataType: .float32
        )
        let movePlayed = graph.constant(int32Data(batch.played), shape: [NSNumber(value: batch.rows)], dataType: .int32)
        return (legalMask, movePlayed)
    }

    /// The mode-selecting production builder, with every smoothing scalar a
    /// constant.
    private func selectedTargets(
        _ batch: Batch,
        mode: PolicyLabelSmoothingMode,
        epsilon: Float = 0.1,
        perMove: Float,
        perMoveCap: Float
    ) throws -> Targets {
        _ = try requireMetal()
        let graph = MPSGraph()
        let inputs = batchInputs(graph, batch)
        let targets = HeadLossGraph.policyTargets(
            graph: graph,
            movePlayed: inputs.movePlayed,
            legalMask: inputs.legalMask,
            labelSmoothing: HeadLossGraph.PolicyLabelSmoothingInputs(
                perMoveSelector: scalar(graph, mode.graphSelectorValue),
                epsilon: scalar(graph, epsilon),
                perMove: scalar(graph, perMove),
                perMoveCap: scalar(graph, perMoveCap)
            ),
            policySize: policySize
        )
        let results = graph.run(
            feeds: [:],
            targetTensors: [targets.smoothed, targets.complement, targets.complementValid],
            targetOperations: nil
        )
        return Targets(
            smoothed: try read(results, targets.smoothed, count: batch.rows * policySize),
            complement: try read(results, targets.complement, count: batch.rows * policySize),
            valid: try read(results, targets.complementValid, count: batch.rows)
        )
    }

    /// The fixed-total-only builder — the code path every run used before
    /// the per-move form existed.
    private func fixedTotalTargets(_ batch: Batch, epsilon: Float) throws -> Targets {
        _ = try requireMetal()
        let graph = MPSGraph()
        let inputs = batchInputs(graph, batch)
        let targets = HeadLossGraph.policyTargets(
            graph: graph,
            movePlayed: inputs.movePlayed,
            legalMask: inputs.legalMask,
            epsilon: scalar(graph, epsilon),
            policySize: policySize
        )
        let results = graph.run(
            feeds: [:],
            targetTensors: [targets.smoothed, targets.complement, targets.complementValid],
            targetOperations: nil
        )
        return Targets(
            smoothed: try read(results, targets.smoothed, count: batch.rows * policySize),
            complement: try read(results, targets.complement, count: batch.rows * policySize),
            valid: try read(results, targets.complementValid, count: batch.rows)
        )
    }

    private func rowRange(_ row: Int) -> Range<Int> {
        (row * policySize)..<((row + 1) * policySize)
    }

    private func rowSum(_ values: [Float], _ row: Int) -> Double {
        values[rowRange(row)].reduce(0.0) { $0 + Double($1) }
    }

    /// The per-move mass each non-played legal move should get before
    /// renormalization: `min(δ, cap/(n − 1))`, in Double from the fp32 inputs.
    private func expectedPerAlternative(legalCount n: Int, perMove: Float, perMoveCap: Float) -> Double {
        guard n > 1 else { return 0 }
        return min(Double(perMove), Double(perMoveCap) / Double(n - 1))
    }

    // MARK: - Per-move positive target

    func testPerMoveTargetSumsToOneForEveryLegalCount() throws {
        let batch = makeBatch(legalCounts)
        let cases: [(perMove: Float, cap: Float)] = [(0.0033, 0.5), (0.0033, 0.02), (0.05, 0.9), (0, 0.5)]
        for (perMove, cap) in cases {
            let targets = try selectedTargets(batch, mode: .perMove, perMove: perMove, perMoveCap: cap)
            for (row, n) in legalCounts.enumerated() {
                XCTAssertEqual(rowSum(targets.smoothed, row), 1, accuracy: 1e-6,
                               "δ=\(perMove) cap=\(cap) |legal|=\(n): positive target sum")
                XCTAssertTrue(targets.smoothed[rowRange(row)].allSatisfy { $0.isFinite && $0 >= 0 },
                              "δ=\(perMove) cap=\(cap) |legal|=\(n): finite, non-negative")
            }
        }
    }

    /// Below the cap every alternative gets δ; above it they share the cap
    /// equally and the played move gets `1 − cap`. With the default δ and cap
    /// the cap engages from `n − 1 > cap/δ`; the small cap makes it engage
    /// at a handful of legal moves too.
    func testPerMoveMassIsDeltaBelowTheCapAndAnEqualShareOfTheCapAboveIt() throws {
        let batch = makeBatch(legalCounts)
        let cases: [(perMove: Float, cap: Float)] = [(0.0033, 0.5), (0.0033, 0.02)]
        for (perMove, cap) in cases {
            let targets = try selectedTargets(batch, mode: .perMove, perMove: perMove, perMoveCap: cap)
            var sawBelowCap = false
            var sawAboveCap = false
            for (row, n) in legalCounts.enumerated() where n > 1 {
                let per = expectedPerAlternative(legalCount: n, perMove: perMove, perMoveCap: cap)
                let capEngaged = Double(perMove) * Double(n - 1) > Double(cap)
                if capEngaged { sawAboveCap = true } else { sawBelowCap = true }
                let playedIndex = row * policySize + Int(batch.played[row])
                let expectedPlayed = 1 - per * Double(n - 1)
                XCTAssertEqual(Double(targets.smoothed[playedIndex]), expectedPlayed, accuracy: 2e-6,
                               "δ=\(perMove) cap=\(cap) |legal|=\(n): played-move target")
                for i in rowRange(row) where batch.mask[i] == 1 && i != playedIndex {
                    XCTAssertEqual(Double(targets.smoothed[i]), per, accuracy: per * 1e-5,
                                   "δ=\(perMove) cap=\(cap) |legal|=\(n): alternative target")
                }
                if capEngaged {
                    XCTAssertEqual(1 - Double(targets.smoothed[playedIndex]), Double(cap), accuracy: 2e-6,
                                   "δ=\(perMove) cap=\(cap) |legal|=\(n): total smoothing equals the cap")
                }
            }
            XCTAssertTrue(sawBelowCap, "fixture must cover positions below the cap (δ=\(perMove) cap=\(cap))")
            XCTAssertTrue(sawAboveCap, "fixture must cover positions above the cap (δ=\(perMove) cap=\(cap))")
        }
    }

    func testPerMoveSingleLegalMoveIsAnExactOneHot() throws {
        let batch = makeBatch([1])
        for perMove: Float in [0, 0.0033, 0.05] {
            let targets = try selectedTargets(batch, mode: .perMove, perMove: perMove, perMoveCap: 0.5)
            let playedIndex = Int(batch.played[0])
            for i in rowRange(0) {
                XCTAssertEqual(targets.smoothed[i], i == playedIndex ? 1 : 0, "δ=\(perMove): cell \(i)")
            }
            XCTAssertEqual(targets.valid[0], 0, "δ=\(perMove): a forced move gets zero complement weight")
            XCTAssertTrue(targets.complement[rowRange(0)].allSatisfy { $0.isFinite && $0 >= 0 },
                          "δ=\(perMove): the unused complement target must still be finite")
        }
    }

    func testPerMoveIllegalCellsAreExactlyZero() throws {
        let batch = makeBatch(legalCounts)
        let targets = try selectedTargets(batch, mode: .perMove, perMove: 0.0033, perMoveCap: 0.5)
        // Counted per row and asserted once per row: a per-cell assertion
        // over a million illegal cells costs far more than the check.
        for (row, n) in legalCounts.enumerated() {
            var positiveViolations = 0
            var complementViolations = 0
            for i in rowRange(row) where batch.mask[i] == 0 {
                if targets.smoothed[i] != 0 { positiveViolations += 1 }
                if targets.complement[i] != 0 { complementViolations += 1 }
            }
            XCTAssertEqual(positiveViolations, 0, "|legal|=\(n): positive target mass on illegal cells")
            XCTAssertEqual(complementViolations, 0, "|legal|=\(n): complement mass on illegal cells")
        }
    }

    // MARK: - Per-move complement target

    /// The mirror: the played move gets the per-move floor — the positive
    /// target's per-alternative mass, `min(δ, cap/(n − 1))` — and the other
    /// legal moves share the rest equally.
    func testPerMoveComplementPutsTheFloorOnThePlayedMoveAndTheRestOnTheOthers() throws {
        let batch = makeBatch(legalCounts)
        let cases: [(perMove: Float, cap: Float)] = [(0.0033, 0.5), (0.05, 0.02), (0, 0.5)]
        for (perMove, cap) in cases {
            let targets = try selectedTargets(batch, mode: .perMove, perMove: perMove, perMoveCap: cap)
            for (row, n) in legalCounts.enumerated() where n > 1 {
                let floor = expectedPerAlternative(legalCount: n, perMove: perMove, perMoveCap: cap)
                XCTAssertEqual(targets.valid[row], 1, "|legal|=\(n): complement weight")
                XCTAssertEqual(rowSum(targets.complement, row), 1, accuracy: 1e-6,
                               "δ=\(perMove) cap=\(cap) |legal|=\(n): complement sum")
                let playedIndex = row * policySize + Int(batch.played[row])
                XCTAssertEqual(Double(targets.complement[playedIndex]), floor, accuracy: 1e-7,
                               "δ=\(perMove) cap=\(cap) |legal|=\(n): played move's complement floor")
                let other = (1 - floor) / Double(n - 1)
                for i in rowRange(row) where batch.mask[i] == 1 && i != playedIndex {
                    XCTAssertEqual(Double(targets.complement[i]), other, accuracy: other * 1e-5,
                                   "δ=\(perMove) cap=\(cap) |legal|=\(n): other legal move's complement mass")
                }
            }
        }
    }

    /// "This move was bad" must never make the bad move the most favoured
    /// target. A floor of `min(δ, cap)` on the played move did, once δ
    /// reached `1/n` (δ 0.05 ties at 20 legal moves and wins above it). Swept
    /// over the declared ranges' corners and every legal-move count.
    func testPerMoveComplementNeverFavoursThePlayedMove() throws {
        let batch = makeBatch(legalCounts)
        for perMove: Float in [0.0033, 0.005, 0.01, 0.03, 0.05] {
            for cap: Float in [0.02, 0.5, 0.9] {
                let targets = try selectedTargets(batch, mode: .perMove, perMove: perMove, perMoveCap: cap)
                for (row, n) in legalCounts.enumerated() where n > 1 {
                    let playedIndex = row * policySize + Int(batch.played[row])
                    let played = targets.complement[playedIndex]
                    for i in rowRange(row) where batch.mask[i] == 1 && i != playedIndex {
                        XCTAssertLessThan(played, targets.complement[i],
                                          "δ=\(perMove) cap=\(cap) |legal|=\(n): played move must get less than every alternative")
                    }
                }
            }
        }
    }

    /// The complement's floor on the played move is exactly the positive
    /// target's per-alternative mass, `min(δ, cap/(n − 1))`.
    func testPerMoveComplementFloorEqualsThePositivePerAlternativeMass() throws {
        let batch = makeBatch(legalCounts)
        let cases: [(perMove: Float, cap: Float)] = [(0.0033, 0.5), (0.05, 0.02), (0.05, 0.9)]
        for (perMove, cap) in cases {
            let targets = try selectedTargets(batch, mode: .perMove, perMove: perMove, perMoveCap: cap)
            for (row, n) in legalCounts.enumerated() where n > 1 {
                let playedIndex = row * policySize + Int(batch.played[row])
                let floor = expectedPerAlternative(legalCount: n, perMove: perMove, perMoveCap: cap)
                XCTAssertEqual(Double(targets.complement[playedIndex]), floor, accuracy: 1e-7,
                               "δ=\(perMove) cap=\(cap) |legal|=\(n): played move's complement floor")
            }
        }
    }

    // MARK: - Fixed-total mode

    /// Fixed-total mode through the mode-selecting builder must hand training
    /// exactly the targets the fixed-total builder produces — bit for bit —
    /// and those must match the closed form
    /// `(1 − ε)·oneHot + ε·mask/|legal|` (complement:
    /// `(1 − ε)·otherMask/(|legal| − 1) + ε·mask/|legal|`), renormalized.
    func testFixedTotalModeIsBitIdenticalToTheFixedTotalBuilderAndMatchesTheFormula() throws {
        let batch = makeBatch(legalCounts)
        for epsilon: Float in [0, 0.03, 0.1, 0.9] {
            let selected = try selectedTargets(batch, mode: .fixedTotal, epsilon: epsilon, perMove: 0.0033, perMoveCap: 0.5)
            let reference = try fixedTotalTargets(batch, epsilon: epsilon)
            XCTAssertEqual(selected.smoothed.map(\.bitPattern), reference.smoothed.map(\.bitPattern),
                           "ε=\(epsilon): positive target must be bit-identical to the fixed-total builder")
            XCTAssertEqual(selected.complement.map(\.bitPattern), reference.complement.map(\.bitPattern),
                           "ε=\(epsilon): complement target must be bit-identical to the fixed-total builder")
            XCTAssertEqual(selected.valid, reference.valid, "ε=\(epsilon): complement weights")

            let eps = Double(epsilon)
            for (row, n) in legalCounts.enumerated() {
                let playedIndex = row * policySize + Int(batch.played[row])
                var positiveRaw = [Double](repeating: 0, count: policySize)
                var complementRaw = [Double](repeating: 0, count: policySize)
                for (j, i) in rowRange(row).enumerated() where batch.mask[i] == 1 {
                    let isPlayed = i == playedIndex
                    positiveRaw[j] = (isPlayed ? 1 - eps : 0) + eps / Double(n)
                    complementRaw[j] = (isPlayed ? 0 : (1 - eps) / Double(max(n - 1, 1))) + eps / Double(n)
                }
                let positiveSum = positiveRaw.reduce(0, +)
                let complementSum = complementRaw.reduce(0, +)
                // Largest deviation over the row, asserted once per row.
                var positiveError = 0.0
                var complementError = 0.0
                for (j, i) in rowRange(row).enumerated() {
                    positiveError = max(positiveError, abs(Double(selected.smoothed[i]) - positiveRaw[j] / positiveSum))
                    complementError = max(complementError, abs(Double(selected.complement[i]) - complementRaw[j] / complementSum))
                }
                XCTAssertLessThanOrEqual(positiveError, 1e-6, "ε=\(epsilon) |legal|=\(n): positive target vs formula")
                if n > 1 {
                    XCTAssertLessThanOrEqual(complementError, 1e-6, "ε=\(epsilon) |legal|=\(n): complement target vs formula")
                }
            }
        }
    }

    // MARK: - Live mode switch

    /// The mode is fed, not baked: one graph produces the fixed-total targets
    /// when fed 0 and the per-move targets when fed 1, matching graphs built
    /// with the selector as a constant. This is what lets a mode change reach
    /// a running trainer on its next step. Compared to fp32 rounding rather
    /// than bit for bit: a constant selector lets the graph compiler fold the
    /// select away, which may legitimately fuse the chosen branch
    /// differently.
    func testOneGraphSwitchesFormsByFeed() throws {
        let device = try requireMetal()
        let batch = makeBatch([1, 2, 30, 218])
        let graph = MPSGraph()
        let inputs = batchInputs(graph, batch)
        let selector = graph.placeholder(shape: [1], dataType: .float32, name: "selector")
        let targets = HeadLossGraph.policyTargets(
            graph: graph,
            movePlayed: inputs.movePlayed,
            legalMask: inputs.legalMask,
            labelSmoothing: HeadLossGraph.PolicyLabelSmoothingInputs(
                perMoveSelector: selector,
                epsilon: scalar(graph, 0.1),
                perMove: scalar(graph, 0.0033),
                perMoveCap: scalar(graph, 0.5)
            ),
            policySize: policySize
        )
        let graphDevice = MPSGraphDevice(mtlDevice: device)
        for mode in PolicyLabelSmoothingMode.allCases {
            let feed = MPSGraphTensorData(
                device: graphDevice,
                data: floatData([mode.graphSelectorValue]),
                shape: [1],
                dataType: .float32
            )
            let results = graph.run(
                feeds: [selector: feed],
                targetTensors: [targets.smoothed, targets.complement],
                targetOperations: nil
            )
            let smoothed = try read(results, targets.smoothed, count: batch.rows * policySize)
            let complement = try read(results, targets.complement, count: batch.rows * policySize)
            let expected = try selectedTargets(batch, mode: mode, epsilon: 0.1, perMove: 0.0033, perMoveCap: 0.5)
            let smoothedError = zip(smoothed, expected.smoothed).reduce(0.0) { max($0, abs(Double($1.0) - Double($1.1))) }
            let complementError = zip(complement, expected.complement).reduce(0.0) { max($0, abs(Double($1.0) - Double($1.1))) }
            XCTAssertLessThanOrEqual(smoothedError, 1e-7,
                                     "\(mode.logToken): fed selector must give the positive target a constant one gives")
            XCTAssertLessThanOrEqual(complementError, 1e-7,
                                     "\(mode.logToken): fed selector must give the complement target a constant one gives")
        }
    }

    // MARK: - Mode enum

    /// `policy_label_smoothing_mode` stores a raw `Int`, so the declared range
    /// and the enum's cases are two separate declarations that can drift. A
    /// narrower range would make a mode unreachable through persistence; a
    /// wider one would let the validator pass a value
    /// `init(persistedRawValue:)` traps on.
    func testModeParameterRangeMatchesTheEnumCases() throws {
        let range = try XCTUnwrap(PolicyLabelSmoothingModeParameter.definition.intRange,
                                  "policy_label_smoothing_mode must declare an Int range")
        XCTAssertEqual(range.min...range.max, PolicyLabelSmoothingMode.parameterRawValueRange)
        for raw in range.min...range.max {
            XCTAssertNotNil(PolicyLabelSmoothingMode(rawValue: raw), "raw value \(raw) is in range but has no case")
        }
    }

    func testLogTokensAreStableAndRoundTrip() {
        XCTAssertEqual(PolicyLabelSmoothingMode.fixedTotal.logToken, "fixed_total")
        XCTAssertEqual(PolicyLabelSmoothingMode.perMove.logToken, "per_move")
        for mode in PolicyLabelSmoothingMode.allCases {
            XCTAssertEqual(PolicyLabelSmoothingMode(logToken: mode.logToken), mode)
        }
        XCTAssertNil(PolicyLabelSmoothingMode(logToken: "per-move"))
        XCTAssertEqual(PolicyLabelSmoothingMode.fixedTotal.graphSelectorValue, 0)
        XCTAssertEqual(PolicyLabelSmoothingMode.perMove.graphSelectorValue, 1)
    }

    func testLogFieldsKeepThePLabelSmoothSpellingAndCarryAllFourValues() {
        let fields = PolicyLabelSmoothingMode.logFields(mode: .perMove, epsilon: 0.1, perMove: 0.0033, perMoveCap: 0.5)
        XCTAssertEqual(fields, "pLabelSmooth=0.1 pLabelSmoothMode=per_move pLabelSmoothPerMove=0.0033 pLabelSmoothPerMoveCap=0.5")
    }

    // MARK: - Parameter declarations

    func testParameterIdsDefaultsAndLiveTunability() throws {
        XCTAssertEqual(PolicyLabelSmoothingModeParameter.id, "policy_label_smoothing_mode")
        XCTAssertEqual(PolicyLabelSmoothingPerMove.id, "policy_label_smoothing_per_move")
        XCTAssertEqual(PolicyLabelSmoothingPerMoveCap.id, "policy_label_smoothing_per_move_cap")

        XCTAssertEqual(
            PolicyLabelSmoothingMode(persistedRawValue: PolicyLabelSmoothingModeParameter.declaredDefault),
            .fixedTotal,
            "fixed total stays the default until the A/B decides"
        )
        XCTAssertEqual(PolicyLabelSmoothingPerMove.declaredDefault, 0.0033)
        XCTAssertEqual(PolicyLabelSmoothingPerMoveCap.declaredDefault, 0.5)

        // Same treatment as ε, which is fed every step.
        XCTAssertTrue(PolicyLabelSmoothingEpsilon.definition.liveTunable)
        XCTAssertTrue(PolicyLabelSmoothingModeParameter.definition.liveTunable)
        XCTAssertTrue(PolicyLabelSmoothingPerMove.definition.liveTunable)
        XCTAssertTrue(PolicyLabelSmoothingPerMoveCap.definition.liveTunable)

        for id in [PolicyLabelSmoothingModeParameter.id, PolicyLabelSmoothingPerMove.id, PolicyLabelSmoothingPerMoveCap.id] {
            XCTAssertEqual(TrainingParameters.allKeys.filter { $0.id == id }.count, 1, "\(id) must be registered exactly once")
        }
    }

    func testValidationRejectsOutOfRangeValues() {
        let rejected: [(String, () throws -> Void)] = [
            ("mode -1", { try PolicyLabelSmoothingModeParameter.definition.validate(.int(-1)) }),
            ("mode 2", { try PolicyLabelSmoothingModeParameter.definition.validate(.int(2)) }),
            ("δ -0.001", { try PolicyLabelSmoothingPerMove.definition.validate(.double(-0.001)) }),
            ("δ 0.06", { try PolicyLabelSmoothingPerMove.definition.validate(.double(0.06)) }),
            ("cap -0.1", { try PolicyLabelSmoothingPerMoveCap.definition.validate(.double(-0.1)) }),
            ("cap 0.95", { try PolicyLabelSmoothingPerMoveCap.definition.validate(.double(0.95)) }),
            ("mode as double", { try PolicyLabelSmoothingModeParameter.definition.validate(.double(1)) }),
        ]
        for (label, validate) in rejected {
            XCTAssertThrowsError(try validate(), "\(label) must be rejected")
        }
        XCTAssertNoThrow(try PolicyLabelSmoothingModeParameter.definition.validate(.int(0)))
        XCTAssertNoThrow(try PolicyLabelSmoothingModeParameter.definition.validate(.int(1)))
        XCTAssertNoThrow(try PolicyLabelSmoothingPerMove.definition.validate(.double(0)))
        XCTAssertNoThrow(try PolicyLabelSmoothingPerMove.definition.validate(.double(0.05)))
        XCTAssertNoThrow(try PolicyLabelSmoothingPerMoveCap.definition.validate(.double(0)))
        XCTAssertNoThrow(try PolicyLabelSmoothingPerMoveCap.definition.validate(.double(0.9)))
    }

    /// `--show-default-parameters` / `--create-parameters-file` emit every
    /// registered key; the three new ones must appear with their declared
    /// defaults (the mode as its raw integer).
    func testDefaultsJSONCarriesTheNewKeys() throws {
        let json = try TrainingParameters.defaultsJSON()
        let object = try JSONSerialization.jsonObject(with: json)
        let dict = try XCTUnwrap(object as? [String: Any])
        XCTAssertEqual((dict["policy_label_smoothing_mode"] as? NSNumber)?.intValue, 0)
        XCTAssertEqual((dict["policy_label_smoothing_per_move"] as? NSNumber)?.doubleValue, 0.0033)
        XCTAssertEqual((dict["policy_label_smoothing_per_move_cap"] as? NSNumber)?.doubleValue, 0.5)
    }

    // MARK: - Session state

    private func sessionState(
        mode: String?,
        perMove: Float?,
        perMoveCap: Float?
    ) -> SessionCheckpointState {
        SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "test-session",
            savedAtUnix: 1_700_000_000,
            sessionStartUnix: 1_699_999_000,
            elapsedTrainingSec: 1000,
            trainingSteps: 1234,
            selfPlayGames: 10,
            selfPlayMoves: 600,
            trainingPositionsSeen: 1234 * 4096,
            batchSize: 4096,
            learningRate: 5e-5,
            promoteThreshold: 0.55,
            arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4,
            policyLabelSmoothingEpsilon: 0.1,
            policyLabelSmoothingMode: mode,
            policyLabelSmoothingPerMove: perMove,
            policyLabelSmoothingPerMoveCap: perMoveCap,
            championID: "champ-id",
            trainerID: "train-id",
            arenaHistory: []
        )
    }

    func testSessionStateRoundTripsTheNewFields() throws {
        let original = sessionState(mode: PolicyLabelSmoothingMode.perMove.logToken, perMove: 0.004, perMoveCap: 0.4)
        let decoded = try SessionCheckpointState.decode(try original.encode())
        XCTAssertEqual(decoded.policyLabelSmoothingMode, "per_move")
        XCTAssertEqual(decoded.policyLabelSmoothingPerMove, 0.004)
        XCTAssertEqual(decoded.policyLabelSmoothingPerMoveCap, 0.4)
        XCTAssertEqual(decoded, original, "whole struct must round-trip identically")
    }

    /// A session written before the fields existed has no such keys; it must
    /// still decode, with the fields nil.
    func testSessionStateWithoutTheNewKeysDecodesToNil() throws {
        let encoded = try sessionState(mode: "per_move", perMove: 0.004, perMoveCap: 0.4).encode()
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        for key in ["policyLabelSmoothingMode", "policyLabelSmoothingPerMove", "policyLabelSmoothingPerMoveCap"] {
            XCTAssertNotNil(object.removeValue(forKey: key), "fixture must have encoded \(key)")
        }
        let stripped = try JSONSerialization.data(withJSONObject: object)
        let decoded = try SessionCheckpointState.decode(stripped)
        XCTAssertNil(decoded.policyLabelSmoothingMode)
        XCTAssertNil(decoded.policyLabelSmoothingPerMove)
        XCTAssertNil(decoded.policyLabelSmoothingPerMoveCap)
        XCTAssertEqual(decoded.policyLabelSmoothingEpsilon, 0.1, "the pre-existing ε field is unaffected")
    }

    /// Absent mode ⇒ the session predates the per-move form ⇒ it trained with
    /// fixed-total smoothing, whatever the live mode is now.
    func testAbsentSessionModeResolvesToFixedTotal() {
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingMode(saved: nil), .fixedTotal)
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingMode(saved: .perMove), .perMove)
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingMode(saved: .fixedTotal), .fixedTotal)
    }
}

/// The parameter round trip through `TrainingParameters.shared` and the shared
/// trainer path. Snapshots the singleton first, suppresses `UserDefaults`
/// persistence for the duration, and restores the snapshot afterwards, as
/// `TrainerHyperparametersTests` does.
@MainActor
final class PolicyLabelSmoothingModeParameterTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private func writeTemporaryJSON(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-params-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: url)
        return url
    }

    private func remove(_ url: URL) {
        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            XCTFail("could not remove temp parameters file: \(error)")
        }
    }

    func testParametersJSONLoadsAndAppliesTheNewKeys() throws {
        let url = try writeTemporaryJSON(
            #"{ "policy_label_smoothing_mode": 1, "policy_label_smoothing_per_move": 0.004, "policy_label_smoothing_per_move_cap": 0.4 }"#
        )
        defer { remove(url) }
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(config.trainingParameters["policy_label_smoothing_mode"], .int(1))
        XCTAssertEqual(config.trainingParameters["policy_label_smoothing_per_move"], .double(0.004))
        XCTAssertEqual(config.trainingParameters["policy_label_smoothing_per_move_cap"], .double(0.4))

        let p = TrainingParameters.shared
        try p.apply(config.trainingParameters)
        XCTAssertEqual(p.policyLabelSmoothingMode, .perMove)
        XCTAssertEqual(p.policyLabelSmoothingPerMove, 0.004)
        XCTAssertEqual(p.policyLabelSmoothingPerMoveCap, 0.4)

        let snapshot = p.snapshot()
        XCTAssertEqual(snapshot.policyLabelSmoothingMode, .perMove)
        XCTAssertEqual(snapshot.policyLabelSmoothingPerMove, 0.004)
        XCTAssertEqual(snapshot.policyLabelSmoothingPerMoveCap, 0.4)
        XCTAssertEqual(snapshot.rawValueMap()["policy_label_smoothing_mode"], .int(1))
    }

    /// All-or-nothing: an out-of-range new key rejects the whole file and
    /// leaves the singleton untouched.
    func testParametersJSONRejectsAnOutOfRangeModeAndAppliesNothing() throws {
        let p = TrainingParameters.shared
        p.policyLabelSmoothingMode = .fixedTotal
        p.policyLabelSmoothingPerMove = 0.0033
        let url = try writeTemporaryJSON(
            #"{ "policy_label_smoothing_per_move": 0.01, "policy_label_smoothing_mode": 2 }"#
        )
        defer { remove(url) }
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertThrowsError(try p.apply(config.trainingParameters)) { error in
            guard case TrainingConfigError.outOfRange(let id, _) = error else {
                return XCTFail("expected outOfRange, got \(error)")
            }
            XCTAssertEqual(id, "policy_label_smoothing_mode")
        }
        XCTAssertEqual(p.policyLabelSmoothingMode, .fixedTotal)
        XCTAssertEqual(p.policyLabelSmoothingPerMove, 0.0033)
    }

    func testSingletonSetterRevertsAnOutOfRangeAssignment() {
        let p = TrainingParameters.shared
        p.policyLabelSmoothingPerMove = 0.004
        p.policyLabelSmoothingPerMove = 0.2
        XCTAssertEqual(p.policyLabelSmoothingPerMove, 0.004)
        p.policyLabelSmoothingPerMoveCap = 0.4
        p.policyLabelSmoothingPerMoveCap = 1.0
        XCTAssertEqual(p.policyLabelSmoothingPerMoveCap, 0.4)
    }

    /// Every training path (GUI, corpus replay, train-vs-UCI) configures its
    /// trainer through `TrainerHyperparameters`; the new fields must travel
    /// that path from the parameters to the trainer and back.
    func testTheSharedTrainerPathCarriesTheNewFields() throws {
        let p = TrainingParameters.shared
        p.policyLabelSmoothingMode = .perMove
        p.policyLabelSmoothingPerMove = 0.004
        p.policyLabelSmoothingPerMoveCap = 0.4
        let snapshot = p.snapshot()

        let hyperparameters = TrainerHyperparameters(snapshot)
        XCTAssertEqual(hyperparameters.policyLabelSmoothingMode, .perMove)
        XCTAssertEqual(hyperparameters.policyLabelSmoothingPerMove, Float(0.004))
        XCTAssertEqual(hyperparameters.policyLabelSmoothingPerMoveCap, Float(0.4))
        XCTAssertEqual(ReplayParams(snapshot).trainer, hyperparameters, "the CLI runners carry the same configuration")

        let cliTrainer = try ChessTrainer(hyperparameters: hyperparameters, arch: .current)
        XCTAssertEqual(TrainerHyperparameters(currentlyAppliedTo: cliTrainer), hyperparameters)

        let guiTrainer = try ChessTrainer(arch: .current)
        XCTAssertEqual(guiTrainer.policyLabelSmoothingMode, .fixedTotal, "a bare trainer starts in fixed-total mode")
        hyperparameters.apply(to: guiTrainer)
        XCTAssertEqual(guiTrainer.policyLabelSmoothingMode, .perMove)
        XCTAssertEqual(guiTrainer.policyLabelSmoothingPerMove, Float(0.004))
        XCTAssertEqual(guiTrainer.policyLabelSmoothingPerMoveCap, Float(0.4))
    }
}
