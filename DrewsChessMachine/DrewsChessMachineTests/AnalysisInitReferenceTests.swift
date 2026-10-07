import MetalPerformanceShadersGraph
import XCTest
@testable import DrewsChessMachine

/// The analyzers' init reference: the architecture built by the network
/// builder itself, so every "init" figure matches how the model really
/// started — whatever its final init, draw prior or other init options.
final class AnalysisInitReferenceTests: XCTestCase {

    /// An untrained model's snapshot, its reference from a fresh cache under
    /// `recorded`, checked against the network's variables.
    private func snapshot(network: ChessNetwork, recorded: ModelInitRecord?) async throws -> AnalyzedNetworkSnapshot {
        let names = (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name }
        let weights = try await network.exportWeights()
        let reference = try await AnalysisInitReferenceCache().reference(architecture: network.arch, initialization: recorded)
        try reference.requireMatches(variableNames: names, trainableCount: network.trainableVariables.count)
        return AnalyzedNetworkSnapshot(
            role: .trainer, modelID: "20261007-1-TEST", architecture: network.arch,
            policyTailPrecision: network.policyTailPrecision, names: names, weights: weights,
            trainableCount: network.trainableVariables.count, trainingStep: 0,
            trainingStepSource: "test", takenAt: Date(), initReference: reference)
    }

    /// With the model's own seed every trainable is at its init exactly —
    /// whether the model was built calibrated (GUI) or with identity BN
    /// statistics in training mode (a fresh CLI trainer) — and running
    /// statistics get no init figures either way.
    func testAnUntrainedModelWithItsSeedIsAtItsInitWhicheverWayItWasBuilt() async throws {
        let recorded = ModelInitRecord(initSeed: 7, scheme: WeightInitScheme.current)
        let networks = [
            try ChessMPSNetwork(.randomWeights(initSeed: 7), arch: .current).network,
            try ChessNetwork(arch: .current, bnMode: .training, initialization: .seeded(initSeed: 7)),
        ]
        for network in networks {
            let snapshot = try await snapshot(network: network, recorded: recorded)
            XCTAssertEqual(snapshot.initReference.basis, .modelInitSeed(7))
            let result = try NetworkWeightAnalyzer.run(snapshot: snapshot, modelLabel: snapshot.modelLabel)
            let variables = result.sections.flatMap(\.variables)
            XCTAssertEqual(variables.count, snapshot.names.count)
            for variable in variables {
                if variable.isRunningStatistic {
                    XCTAssertNil(variable.initL2Norm, variable.name)
                    XCTAssertNil(variable.l2NormRatioToInit, variable.name)
                    XCTAssertNil(variable.driftFromInit, variable.name)
                    XCTAssertFalse(variable.initExact, variable.name)
                } else {
                    XCTAssertTrue(variable.initExact, variable.name)
                    XCTAssertEqual(try XCTUnwrap(variable.driftFromInit, variable.name), 0, variable.name)
                    if let ratio = variable.l2NormRatioToInit { XCTAssertEqual(ratio, 1, accuracy: 1e-12, variable.name) }
                }
            }
            for section in result.sections {
                XCTAssertTrue(section.initExact, section.sectionName)
                if let ratio = section.totalL2RatioToInit { XCTAssertEqual(ratio, 1, accuracy: 1e-12, section.sectionName) }
            }
        }
    }

    /// Without the seed only tensors the builder did not draw are exact; a
    /// zero-initialized head has no ratio; the value bias starts at the
    /// model's own draw prior; sections holding drawn tensors are flagged.
    func testWithoutTheSeedOnlyTensorsNotDrawnFromItAreExact() async throws {
        var arch = NetworkArchitecture.current
        arch.policyHeadFinalInit = .zero
        arch.valueHeadDrawPrior = 0.5
        try arch.validate()
        let snapshot = try await snapshot(
            network: try ChessMPSNetwork(.randomWeights(initSeed: 11), arch: arch).network, recorded: nil)
        XCTAssertEqual(snapshot.initReference.basis, .fallbackSeed(.seedNotRecorded))
        XCTAssertFalse(snapshot.initReference.exactNames.contains("stem_conv_weights"))
        XCTAssertTrue(snapshot.initReference.exactNames.contains("value_wdl_fc2_bias"))

        let result = try NetworkWeightAnalyzer.run(snapshot: snapshot, modelLabel: snapshot.modelLabel)
        var byName: [String: NetworkWeightAnalyzer.Result.WeightStats] = [:]
        for variable in result.sections.flatMap(\.variables) { byName[variable.name] = variable }
        let stem = try XCTUnwrap(byName["stem_conv_weights"])
        XCTAssertFalse(stem.initExact)
        XCTAssertNil(stem.driftFromInit)
        XCTAssertGreaterThan(try XCTUnwrap(stem.initL2Norm), 0)
        XCTAssertEqual(result.sections.first { $0.sectionName == "stem" }?.initExact, false)
        XCTAssertEqual(result.stemInputChannelDetail?.initExact, false)

        let policyFinal = try XCTUnwrap(byName["policy_conv_weights"] ?? byName["policy_fc_weights"])
        XCTAssertTrue(policyFinal.initExact)
        XCTAssertEqual(policyFinal.initL2Norm, 0)
        XCTAssertNil(policyFinal.l2NormRatioToInit)
        XCTAssertEqual(try XCTUnwrap(policyFinal.driftFromInit), 0)

        let valueHead = try ValueHeadAnalyzer.run(snapshot: snapshot, modelLabel: snapshot.modelLabel)
        let bias = try XCTUnwrap(valueHead.fc2Bias)
        XCTAssertTrue(bias.initExact)
        // The prior as the model stores it: rounded to its compute dtype.
        let storedPrior = try XCTUnwrap(snapshot.values(named: "value_wdl_fc2_bias")).map(Double.init)
        XCTAssertEqual(bias.initial, storedPrior)
        for (stored, prior) in zip(storedPrior, NetworkArchitecture.wdlBiasPrior(drawProbability: 0.5)) {
            XCTAssertEqual(stored, Double(prior), accuracy: 0.004)
        }
        XCTAssertEqual(bias.delta, [0, 0, 0])
    }

    func testTheBasisFollowsTheRecordedInitialization() {
        XCTAssertEqual(AnalysisInitReference.basis(for: nil), .fallbackSeed(.seedNotRecorded))
        XCTAssertEqual(AnalysisInitReference.basis(for: ModelInitRecord(initSeed: 3, scheme: "dcm-init-0")),
                       .fallbackSeed(.otherScheme("dcm-init-0")))
        XCTAssertEqual(AnalysisInitReference.basis(for: ModelInitRecord(initSeed: 3, scheme: WeightInitScheme.current)),
                       .modelInitSeed(3))
    }

    /// The audit's value-bias init mean is the model's own draw prior as the
    /// model stores it, read from the reference; a reference for another
    /// architecture is refused.
    func testTheAuditReadsTheBiasInitMeanFromTheReference() async throws {
        var arch = NetworkArchitecture.current
        arch.valueHeadDrawPrior = 0.5
        try arch.validate()
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 5), arch: arch).network
        let weights = try await network.exportWeights()
        let cache = AnalysisInitReferenceCache()
        let reference = try await cache.reference(architecture: arch, initialization: nil)
        func audit(_ auditedArch: NetworkArchitecture) async throws -> NumericsAudit.Result {
            try await NumericsAudit.run(
                names: reference.variableNames, weights: weights, arch: auditedArch, initReference: reference,
                masters: nil, mastersNote: "test", velocity: .unavailable(reason: "test"), positions: nil,
                dynamicSkippedReason: "test", policyTailPrecision: network.policyTailPrecision,
                modelLabel: "test", modelID: nil, trainingStep: nil)
        }
        // The stored start: a bf16 model rounds the prior to its dtype.
        let storedBias = try XCTUnwrap(reference.initialValues["value_wdl_fc2_bias"])
        let storedMean = storedBias.reduce(0.0) { $0 + Double($1) } / Double(storedBias.count)
        let prior = NetworkArchitecture.wdlBiasPrior(drawProbability: 0.5)
        let priorMean = prior.reduce(0.0) { $0 + Double($1) } / Double(prior.count)
        let offset = try XCTUnwrap(try await audit(arch).staticChecks.valueHeadOffset)
        XCTAssertEqual(offset.biasInitMean, storedMean)
        XCTAssertEqual(offset.biasInitMean, priorMean, accuracy: 0.004)
        do {
            _ = try await audit(.current)
            XCTFail("expected the architecture mismatch")
        } catch NumericsAuditError.initReferenceArchitectureMismatch {
        }
    }

    func testTheStoredInComputeDtypeNote() {
        XCTAssertNil(NumericsAudit.storedInComputeDtypeNote(weights: [[0.5, 1.25]], dataType: .float32))
        // 1.0 + 2⁻¹⁰ needs more than bf16's 7 mantissa bits.
        XCTAssertNil(NumericsAudit.storedInComputeDtypeNote(weights: [[0.5, 1.0009765625]], dataType: .bFloat16))
        XCTAssertNotNil(NumericsAudit.storedInComputeDtypeNote(weights: [[0.5, -0.64453125], [2]], dataType: .bFloat16))
        XCTAssertNotNil(NumericsAudit.storedInComputeDtypeNote(weights: [[0.5, 1.0009765625]], dataType: .float16))
        XCTAssertNil(NumericsAudit.storedInComputeDtypeNote(weights: [[1e-9]], dataType: .float16))
    }
}
