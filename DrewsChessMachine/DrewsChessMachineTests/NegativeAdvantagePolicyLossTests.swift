//
//  NegativeAdvantagePolicyLossTests.swift
//  DrewsChessMachineTests
//
//  Pins `negative_advantage_policy_loss` (`NegativeAdvantagePolicyLoss`), the
//  enum that replaced the Bool `signed_advantage_complement_ce`:
//
//  - The unlikelihood loss is −log Σ_{legal j ≠ played} p_j, and its logit
//    gradient is the closed form softmax(x) − softmax over the other legal
//    moves: +p on the played move and on illegal cells, −p_j(1/S − 1) on the
//    other legal moves (the CE gradient against the current policy
//    renormalized without the played move).
//  - A single-legal-move position gets exactly zero loss and gradient, never
//    NaN.
//  - One compiled graph switches between off, complement CE and unlikelihood
//    by the fed selector, each matching its closed form.
//  - The parameter: range pinned to the enum, tokens, declaration
//    (default complement_ce, live-tunable, pre-feature off), parameters.json,
//    the shared trainer path, the GUI's stored settings and session.json —
//    including every conversion of the old Bool (true → complement_ce,
//    false → off) and the refusal of both keys together.
//
//  The graph tests run the production builders (`HeadLossGraph`) on a small
//  standalone graph with the production policy width.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

final class NegativeAdvantagePolicyLossGraphTests: XCTestCase {

    private let policySize = ChessNetwork.policySize
    /// One legal move, two, a narrow position, a typical one.
    private let legalCounts = [1, 2, 5, 30]
    private let epsilon: Float = 0.1

    private struct Fixture {
        let logits: [Float]
        let mask: [Float]
        let played: [Int32]
        let rows: Int
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

    private func read(_ results: [MPSGraphTensor: MPSGraphTensorData], _ tensor: MPSGraphTensor,
                      count: Int, file: StaticString = #filePath, line: UInt = #line) throws -> [Float] {
        let data = try XCTUnwrap(results[tensor], "tensor missing from results", file: file, line: line)
        XCTAssertEqual(data.dataType, .float32, "loss-path tensors must be fp32", file: file, line: line)
        return ChessNetwork.readFloatsFP32(from: data, count: count)
    }

    /// Legal cells spread with a stride; the played move is the first legal
    /// cell. Logits vary over every cell, illegal ones included, so the
    /// illegal cells carry real softmax mass.
    private func makeFixture() -> Fixture {
        let rows = legalCounts.count
        var mask = [Float](repeating: 0, count: rows * policySize)
        var played = [Int32](repeating: 0, count: rows)
        var logits = [Float](repeating: 0, count: rows * policySize)
        for (row, count) in legalCounts.enumerated() {
            for k in 0..<count { mask[row * policySize + (k * 17 + row) % policySize] = 1 }
            played[row] = Int32(row % policySize)
            for i in 0..<policySize {
                logits[row * policySize + i] = Float(3 * sin(0.37 * Double(i) + Double(row)))
            }
        }
        return Fixture(logits: logits, mask: mask, played: played, rows: rows)
    }

    /// Double-precision log-softmax of one row.
    private func logSoftmax(_ row: ArraySlice<Float>) -> [Double] {
        let values = row.map { Double($0) }
        let maximum = values.max() ?? 0
        let logSum = maximum + log(values.reduce(0.0) { $0 + exp($1 - maximum) })
        return values.map { $0 - logSum }
    }

    private func rowSlice<T>(_ array: [T], _ row: Int) -> ArraySlice<T> {
        array[(row * policySize)..<((row + 1) * policySize)]
    }

    /// −log Σ_{legal j ≠ played} p_j, or 0 for a single legal move.
    private func expectedUnlikelihood(_ f: Fixture, row: Int) -> Double {
        guard legalCounts[row] > 1 else { return 0 }
        let logP = logSoftmax(rowSlice(f.logits, row))
        let mask = rowSlice(f.mask, row).map { $0 }
        let playedIndex = Int(f.played[row])
        var otherMass = 0.0
        for i in 0..<policySize where mask[i] == 1 && i != playedIndex { otherMass += exp(logP[i]) }
        return -log(otherMass)
    }

    /// The fixed-total complement CE, or 0 for a single legal move.
    private func expectedComplementCE(_ f: Fixture, row: Int) -> Double {
        let n = legalCounts[row]
        guard n > 1 else { return 0 }
        let logP = logSoftmax(rowSlice(f.logits, row))
        let mask = rowSlice(f.mask, row).map { $0 }
        let playedIndex = Int(f.played[row])
        let e = Double(epsilon)
        var target = [Double](repeating: 0, count: policySize)
        for i in 0..<policySize where mask[i] == 1 {
            target[i] = e / Double(n) + (i == playedIndex ? 0 : (1 - e) / Double(n - 1))
        }
        let total = target.reduce(0, +)
        var ce = 0.0
        for i in 0..<policySize where target[i] > 0 { ce -= target[i] / total * logP[i] }
        return ce
    }

    private struct Built {
        let graph: MPSGraph
        let logits: MPSGraphTensor
        let targets: HeadLossGraph.PolicyTargets
    }

    private func build(_ f: Fixture) -> Built {
        let graph = MPSGraph()
        let shape: [NSNumber] = [NSNumber(value: f.rows), NSNumber(value: policySize)]
        let logits = graph.variable(with: floatData(f.logits), shape: shape, dataType: .float32, name: "logits")
        let legalMask = graph.constant(floatData(f.mask), shape: shape, dataType: .float32)
        let movePlayed = graph.constant(int32Data(f.played), shape: [NSNumber(value: f.rows)], dataType: .int32)
        let eps = graph.constant(floatData([epsilon]), shape: [1], dataType: .float32)
        let targets = HeadLossGraph.policyTargets(
            graph: graph, movePlayed: movePlayed, legalMask: legalMask, epsilon: eps, policySize: policySize)
        return Built(graph: graph, logits: logits, targets: targets)
    }

    /// (a) The unlikelihood's value and its logit gradient against the closed
    /// forms, on every row with more than one legal move.
    func testUnlikelihoodValueAndGradientMatchTheClosedForm() throws {
        _ = try requireMetal()
        let f = makeFixture()
        let b = build(f)
        let perPosition = HeadLossGraph.policyUnlikelihoodPerPosition(
            graph: b.graph, centeredLogits: b.logits, targets: b.targets)
        let total = b.graph.reductionSum(with: perPosition, axes: [0, 1], name: "total")
        let gradient = try XCTUnwrap(b.graph.gradients(of: total, with: [b.logits], name: "grads")[b.logits])
        let results = b.graph.run(feeds: [:], targetTensors: [perPosition, gradient], targetOperations: nil)
        let loss = try read(results, perPosition, count: f.rows)
        let g = try read(results, gradient, count: f.rows * policySize)

        for row in 0..<f.rows where legalCounts[row] > 1 {
            XCTAssertEqual(Double(loss[row]), expectedUnlikelihood(f, row: row), accuracy: 1e-4, "row \(row): −log S")
            let logP = logSoftmax(rowSlice(f.logits, row))
            let mask = rowSlice(f.mask, row).map { $0 }
            let playedIndex = Int(f.played[row])
            var otherMass = 0.0
            for i in 0..<policySize where mask[i] == 1 && i != playedIndex { otherMass += exp(logP[i]) }
            let rowGradient = rowSlice(g, row).map { Double($0) }
            for i in 0..<policySize {
                let p = exp(logP[i])
                let isOther = mask[i] == 1 && i != playedIndex
                // softmax(x) − softmax_O(x): the CE gradient against the
                // policy renormalized over the other legal moves.
                let expected = isOther ? p - p / otherMass : p
                XCTAssertEqual(rowGradient[i], expected, accuracy: 1e-4 * abs(expected) + 1e-7,
                               "row \(row) cell \(i) (\(i == playedIndex ? "played" : isOther ? "other legal" : "illegal"))")
                if isOther {
                    XCTAssertLessThan(rowGradient[i], 0, "row \(row) cell \(i): another legal move must be raised")
                } else {
                    XCTAssertGreaterThan(rowGradient[i], 0, "row \(row) cell \(i): played / illegal must be lowered")
                }
            }
        }
    }

    /// (b) One legal move: zero loss, zero gradient, everything finite.
    func testSingleLegalMoveGivesZeroLossAndGradientAndNoNaN() throws {
        _ = try requireMetal()
        let f = makeFixture()
        let b = build(f)
        let perPosition = HeadLossGraph.policyUnlikelihoodPerPosition(
            graph: b.graph, centeredLogits: b.logits, targets: b.targets)
        let total = b.graph.reductionSum(with: perPosition, axes: [0, 1], name: "total")
        let gradient = try XCTUnwrap(b.graph.gradients(of: total, with: [b.logits], name: "grads")[b.logits])
        let results = b.graph.run(feeds: [:], targetTensors: [perPosition, gradient], targetOperations: nil)
        let loss = try read(results, perPosition, count: f.rows)
        let g = try read(results, gradient, count: f.rows * policySize)

        XCTAssertTrue(loss.allSatisfy(\.isFinite), "every row's loss must be finite")
        XCTAssertTrue(g.allSatisfy(\.isFinite), "every gradient must be finite")
        let singleRow = try XCTUnwrap(legalCounts.firstIndex(of: 1))
        XCTAssertEqual(loss[singleRow], 0, "a forced move teaches nothing")
        XCTAssertTrue(rowSlice(g, singleRow).allSatisfy { $0 == 0 }, "a forced move gets no gradient")
    }

    /// (c) One compiled graph, three fed selector values: off is exactly
    /// zero, complement CE and unlikelihood match their closed forms.
    func testOneGraphSwitchesTheNegativeBranchByFeed() throws {
        let device = try requireMetal()
        let f = makeFixture()
        let b = build(f)
        let selector = b.graph.placeholder(shape: [1], dataType: .float32, name: "selector")
        let perPosition = HeadLossGraph.negativeAdvantagePolicyLossPerPosition(
            graph: b.graph, centeredLogits: b.logits, targets: b.targets, selector: selector)
        let graphDevice = MPSGraphDevice(mtlDevice: device)
        for form in NegativeAdvantagePolicyLoss.allCases {
            let feed = MPSGraphTensorData(
                device: graphDevice, data: floatData([form.graphSelectorValue]), shape: [1], dataType: .float32)
            let results = b.graph.run(feeds: [selector: feed], targetTensors: [perPosition], targetOperations: nil)
            let loss = try read(results, perPosition, count: f.rows)
            for row in 0..<f.rows {
                let expected: Double
                switch form {
                case .off: expected = 0
                case .complementCE: expected = expectedComplementCE(f, row: row)
                case .unlikelihood: expected = expectedUnlikelihood(f, row: row)
                }
                if form == .off {
                    XCTAssertEqual(loss[row], 0, "off: row \(row) must be exactly zero")
                } else {
                    XCTAssertEqual(Double(loss[row]), expected, accuracy: 1e-4 * max(1, abs(expected)),
                                   "\(form.logToken): row \(row)")
                }
            }
        }
    }
}

/// The parameter and every conversion of the old Bool. Snapshots the
/// singleton first, suppresses `UserDefaults` persistence for the duration,
/// and restores the snapshot afterwards, as `TrainerHyperparametersTests`
/// does.
@MainActor
final class NegativeAdvantagePolicyLossParameterTests: XCTestCase {

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

    /// The old Bool and the form it converts to.
    private static let oldFlagConversions: [(Bool, NegativeAdvantagePolicyLoss)] = [(true, .complementCE), (false, .off)]

    private let newID = "negative_advantage_policy_loss"
    private let oldID = "signed_advantage_complement_ce"

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

    // MARK: Enum and declaration

    func testParameterRangeMatchesTheEnumCases() throws {
        let range = try XCTUnwrap(NegativeAdvantagePolicyLossParameter.definition.intRange)
        XCTAssertEqual(range.min...range.max, NegativeAdvantagePolicyLoss.parameterRawValueRange)
        for raw in range.min...range.max {
            XCTAssertNotNil(NegativeAdvantagePolicyLoss(rawValue: raw), "raw value \(raw) is in range but has no case")
        }
    }

    func testTokensAndSelectorValuesAreStable() {
        XCTAssertEqual(NegativeAdvantagePolicyLoss.off.logToken, "off")
        XCTAssertEqual(NegativeAdvantagePolicyLoss.complementCE.logToken, "complement_ce")
        XCTAssertEqual(NegativeAdvantagePolicyLoss.unlikelihood.logToken, "unlikelihood")
        for form in NegativeAdvantagePolicyLoss.allCases {
            XCTAssertEqual(NegativeAdvantagePolicyLoss(logToken: form.logToken), form)
            XCTAssertEqual(form.graphSelectorValue, Float(form.rawValue))
        }
        XCTAssertNil(NegativeAdvantagePolicyLoss(logToken: "complementCE"))
    }

    func testDeclaration() {
        XCTAssertEqual(NegativeAdvantagePolicyLossParameter.id, newID)
        XCTAssertEqual(NegativeAdvantagePolicyLoss(persistedRawValue: NegativeAdvantagePolicyLossParameter.declaredDefault),
                       .complementCE, "the default keeps today's training")
        XCTAssertTrue(NegativeAdvantagePolicyLossParameter.definition.liveTunable)
        XCTAssertEqual(TrainingParameterResolution.absentValue(of: NegativeAdvantagePolicyLossParameter.self),
                       NegativeAdvantagePolicyLoss.off.rawValue,
                       "a record with neither key predates the complement CE, which trained off")
        XCTAssertEqual(TrainingParameters.allKeys.filter { $0.id == newID }.count, 1)
        XCTAssertTrue(TrainingParameters.allKeys.allSatisfy { $0.id != oldID }, "the old key is not declared")
    }

    func testDefaultsJSONCarriesTheNewKeyOnly() throws {
        let object = try JSONSerialization.jsonObject(with: try TrainingParameters.defaultsJSON())
        let dict = try XCTUnwrap(object as? [String: Any])
        XCTAssertEqual((dict[newID] as? NSNumber)?.intValue, 1)
        XCTAssertNil(dict[oldID])
    }

    // MARK: parameters.json and lineage snapshots

    func testParametersJSONLoadsAndAppliesTheNewKey() throws {
        let url = try writeTemporaryJSON(#"{ "negative_advantage_policy_loss": 2 }"#)
        defer { remove(url) }
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(config.trainingParameters[newID], .int(2))
        let p = TrainingParameters.shared
        try p.apply(config.trainingParameters)
        XCTAssertEqual(p.negativeAdvantagePolicyLoss, .unlikelihood)
        XCTAssertEqual(p.snapshot().negativeAdvantagePolicyLoss, .unlikelihood)
        XCTAssertEqual(p.snapshot().rawValueMap()[newID], .int(2))
    }

    func testParametersJSONRejectsAnOutOfRangeValue() throws {
        let url = try writeTemporaryJSON(#"{ "negative_advantage_policy_loss": 3 }"#)
        defer { remove(url) }
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertThrowsError(try TrainingParameters.shared.apply(config.trainingParameters))
    }

    func testOldKeyInParameterJSONConvertsExactly() throws {
        for (flag, form) in Self.oldFlagConversions {
            let values = try ParameterValue.parametersObject(fromJSON: Data(#"{ "signed_advantage_complement_ce": \#(flag) }"#.utf8))
            XCTAssertEqual(values[newID], .int(form.rawValue), "\(oldID)=\(flag)")
            XCTAssertNil(values[oldID], "the old key never survives a read")
        }
    }

    func testOldKeyInAParametersFileLoadsConverted() throws {
        let url = try writeTemporaryJSON(#"{ "signed_advantage_complement_ce": false }"#)
        defer { remove(url) }
        let config = try CliTrainingConfig.load(from: url)
        XCTAssertEqual(config.trainingParameters[newID], .int(NegativeAdvantagePolicyLoss.off.rawValue))
        XCTAssertNil(config.trainingParameters[oldID])
    }

    func testBothKeysOrANonBoolOldKeyAreRefused() {
        XCTAssertThrowsError(try ParameterValue.parametersObject(
            fromJSON: Data(#"{ "signed_advantage_complement_ce": true, "negative_advantage_policy_loss": 1 }"#.utf8)))
        XCTAssertThrowsError(try ParameterValue.parametersObject(
            fromJSON: Data(#"{ "signed_advantage_complement_ce": 1 }"#.utf8)))
    }

    /// An old lineage snapshot at the same behavior is no difference; a
    /// changed behavior is reported.
    func testOldLineageSnapshotComparesByBehavior() throws {
        let p = TrainingParameters.shared
        p.negativeAdvantagePolicyLoss = .complementCE
        let current = p.snapshot()
        func parent(oldFlag: Bool) throws -> LineageRecord.Parameters {
            var values = current.rawValueMap()
            values[newID] = nil
            values[oldID] = .bool(oldFlag)
            for id in LineageRecord.Parameters.excludedParameterIDs { values[id] = nil }
            return try LineageRecord.Parameters(values: values)
        }
        XCTAssertTrue(try current.differences(fromLineage: parent(oldFlag: true)).allSatisfy { $0.id != newID },
                      "true is complement_ce: no difference")
        let changed = try current.differences(fromLineage: parent(oldFlag: false)).filter { $0.id == newID }
        XCTAssertEqual(changed.count, 1, "false is off: reported against complement_ce")
        XCTAssertTrue(try current.differences(fromLineage: parent(oldFlag: false)).allSatisfy { $0.id != oldID },
                      "the old key is converted, never reported as a removed parameter")
    }

    // MARK: Stored settings

    func testStoredValueRoundTrips() throws {
        let defaults = try makeTemporaryDefaults()
        TrainingParameters.store(.int(NegativeAdvantagePolicyLoss.unlikelihood.rawValue), forKey: newID, in: defaults)
        guard case .valid(let raw) = TrainingParameters.inspectStored(NegativeAdvantagePolicyLossParameter.self, in: defaults) else {
            return XCTFail("the stored value must read back as valid")
        }
        XCTAssertEqual(raw, NegativeAdvantagePolicyLoss.unlikelihood.rawValue)
    }

    func testStoredOldSettingConverts() throws {
        for (flag, form) in Self.oldFlagConversions {
            let defaults = try makeTemporaryDefaults()
            defaults.set(flag, forKey: oldID)
            TrainingParameters.convertLegacySignedAdvantageComplementCESetting(in: defaults)
            guard case .valid(let raw) = TrainingParameters.inspectStored(NegativeAdvantagePolicyLossParameter.self, in: defaults) else {
                return XCTFail("\(oldID)=\(flag) must convert to a stored \(newID)")
            }
            XCTAssertEqual(raw, form.rawValue, "\(oldID)=\(flag)")
            XCTAssertEqual(defaults.object(forKey: oldID) as? Bool, flag, "the old entry is left in place")
        }
    }

    func testStoredNewSettingWinsAndUnreadableOldSettingStoresNothing() throws {
        let both = try makeTemporaryDefaults()
        both.set(false, forKey: oldID)
        TrainingParameters.store(.int(NegativeAdvantagePolicyLoss.unlikelihood.rawValue), forKey: newID, in: both)
        TrainingParameters.convertLegacySignedAdvantageComplementCESetting(in: both)
        XCTAssertEqual(both.object(forKey: newID) as? Int, NegativeAdvantagePolicyLoss.unlikelihood.rawValue,
                       "a stored new setting is never overwritten by the old one")

        let unreadable = try makeTemporaryDefaults()
        unreadable.set("yes", forKey: oldID)
        TrainingParameters.convertLegacySignedAdvantageComplementCESetting(in: unreadable)
        XCTAssertNil(unreadable.object(forKey: newID), "the previous build ran its default, the new default")

        let none = try makeTemporaryDefaults()
        TrainingParameters.convertLegacySignedAdvantageComplementCESetting(in: none)
        XCTAssertNil(none.object(forKey: newID))
    }

    // MARK: Session files

    private func sessionState(_ token: String?) -> SessionCheckpointState {
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
            negativeAdvantagePolicyLoss: token,
            championID: "champ-id",
            trainerID: "train-id",
            arenaHistory: []
        ).withLineage(LineageRecord.sessionTestFixture)
    }

    private func jsonObject(_ state: SessionCheckpointState) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: try state.encode()) as? [String: Any])
    }

    func testSessionRoundTripsTheTokenAndNeverWritesTheOldKey() throws {
        let original = sessionState(NegativeAdvantagePolicyLoss.unlikelihood.logToken)
        let decoded = try SessionCheckpointState.decode(try original.encode())
        XCTAssertEqual(decoded.negativeAdvantagePolicyLoss, "unlikelihood")
        XCTAssertEqual(decoded, original)
        XCTAssertNil(try jsonObject(original)["signedAdvantageComplementCE"])
    }

    func testOldSessionFlagConvertsAndIsNotWrittenBack() throws {
        for (flag, form) in Self.oldFlagConversions {
            var object = try jsonObject(sessionState(nil))
            object["signedAdvantageComplementCE"] = flag
            let decoded = try SessionCheckpointState.decode(try JSONSerialization.data(withJSONObject: object))
            XCTAssertEqual(decoded.negativeAdvantagePolicyLoss, form.logToken, "signedAdvantageComplementCE=\(flag)")
            XCTAssertNil(decoded.signedAdvantageComplementCE)
            XCTAssertNil(try jsonObject(decoded)["signedAdvantageComplementCE"], "the old key is never written back")
        }
    }

    func testSessionWithBothKeysIsRefusedAndWithNeitherResolvesToOff() throws {
        var object = try jsonObject(sessionState("complement_ce"))
        object["signedAdvantageComplementCE"] = true
        XCTAssertThrowsError(try SessionCheckpointState.decode(try JSONSerialization.data(withJSONObject: object)))

        let neither = try SessionCheckpointState.decode(try sessionState(nil).encode())
        XCTAssertNil(neither.negativeAdvantagePolicyLoss)
        XCTAssertEqual(SessionCheckpointState.resolvedNegativeAdvantagePolicyLoss(saved: nil), .off)
    }

    func testUnknownSessionTokenIsReviewedAtLoad() {
        let findings = sessionState("sometimes").invalidSavedSettings(current: TrainingParameters.shared.snapshot())
        XCTAssertEqual(findings.filter { $0.id == SessionCheckpointState.SavedSettingID.negativeAdvantagePolicyLoss }.count, 1)
        XCTAssertTrue(sessionState("unlikelihood").invalidSavedSettings(current: TrainingParameters.shared.snapshot())
            .allSatisfy { $0.id != SessionCheckpointState.SavedSettingID.negativeAdvantagePolicyLoss })
    }

    // MARK: Trainer path

    /// Every training path configures its trainer through
    /// `TrainerHyperparameters`; a live change reaches a built trainer
    /// through `apply(to:)` without a rebuild.
    func testTheSharedTrainerPathCarriesItLive() throws {
        let p = TrainingParameters.shared
        p.negativeAdvantagePolicyLoss = .unlikelihood
        let snapshot = p.snapshot()
        let hyperparameters = TrainerHyperparameters(snapshot)
        XCTAssertEqual(hyperparameters.negativeAdvantagePolicyLoss, .unlikelihood)
        XCTAssertEqual(try ReplayParams(snapshot).trainer, hyperparameters, "the CLI runners carry the same configuration")

        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: .current, initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.negativeAdvantagePolicyLoss, .complementCE, "a bare trainer starts at the default")
        hyperparameters.apply(to: trainer)
        XCTAssertEqual(trainer.negativeAdvantagePolicyLoss, .unlikelihood)
        XCTAssertEqual(TrainerHyperparameters(currentlyAppliedTo: trainer), hyperparameters)
    }
}
