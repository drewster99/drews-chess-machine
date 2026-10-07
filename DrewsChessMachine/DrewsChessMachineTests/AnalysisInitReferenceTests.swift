import MetalPerformanceShadersGraph
import XCTest
@testable import DrewsChessMachine

/// The analyzers' init reference: the architecture built by the network
/// builder itself, so every "init" figure matches how the model really
/// started — whatever its final init, draw prior or other init options.
final class AnalysisInitReferenceTests: XCTestCase {

    /// An untrained model's snapshot, its reference built from `recorded`.
    private func snapshot(arch: NetworkArchitecture, seed: UInt64, recorded: ModelInitRecord?) async throws -> AnalyzedNetworkSnapshot {
        let network = try ChessMPSNetwork(.randomWeights(initSeed: seed), arch: arch).network
        let names = (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name }
        let weights = try await network.exportWeights()
        let reference = try await AnalysisInitReference.build(architecture: arch, initialization: recorded, names: names)
        return AnalyzedNetworkSnapshot(
            role: .trainer, modelID: "20261007-1-TEST", architecture: arch,
            policyTailPrecision: network.policyTailPrecision, names: names, weights: weights,
            trainableCount: network.trainableVariables.count, trainingStep: 0,
            trainingStepSource: "test", takenAt: Date(), initReference: reference)
    }

    /// With the model's own seed the reference is the model's starting
    /// point: an untrained model has no drift and a ratio of 1 everywhere.
    /// BN running statistics come from a GPU calibration pass, so they are
    /// held to a tolerance rather than bit equality.
    func testAnUntrainedModelWithItsSeedIsAtItsInit() async throws {
        let snapshot = try await snapshot(arch: .current, seed: 7, recorded: ModelInitRecord(initSeed: 7, scheme: WeightInitScheme.current))
        XCTAssertEqual(snapshot.initReference.basis, .modelInitSeed(7))
        let result = try NetworkWeightAnalyzer.run(snapshot: snapshot, modelLabel: snapshot.modelLabel)
        let variables = result.sections.flatMap(\.variables)
        XCTAssertEqual(variables.count, snapshot.names.count)
        for variable in variables {
            XCTAssertTrue(variable.initExact, variable.name)
            let drift = try XCTUnwrap(variable.driftFromInit, variable.name)
            let runningStatistic = variable.name.hasSuffix("_running_mean") || variable.name.hasSuffix("_running_var")
            XCTAssertLessThanOrEqual(drift, runningStatistic ? 1e-3 * max(variable.l2Norm, 1) : 0, variable.name)
            if let ratio = variable.l2NormRatioToInit, !runningStatistic {
                XCTAssertEqual(ratio, 1, accuracy: 1e-9, variable.name)
            }
        }
        for section in result.sections {
            if let ratio = section.totalL2RatioToInit {
                XCTAssertEqual(ratio, 1, accuracy: 1e-9, section.sectionName)
            }
        }
    }

    /// Without the model's seed, tensors that don't depend on it are still
    /// exact and random ones are not; a zero-initialized policy head has an
    /// initial norm of zero and no ratio; the value head's starting bias is
    /// the model's own draw prior, not a hard-coded one.
    func testWithoutTheSeedOnlyDeterministicTensorsAreExact() async throws {
        var arch = NetworkArchitecture.current
        arch.policyHeadFinalInit = .zero
        arch.valueHeadDrawPrior = 0.5
        try arch.validate()
        let snapshot = try await snapshot(arch: arch, seed: 11, recorded: nil)
        guard case .otherSeeds = snapshot.initReference.basis else { return XCTFail("expected the other-seeds basis") }
        XCTAssertFalse(snapshot.initReference.exactNames.contains("stem_conv_weights"))
        XCTAssertTrue(snapshot.initReference.exactNames.contains("value_wdl_fc2_bias"))

        let result = try NetworkWeightAnalyzer.run(snapshot: snapshot, modelLabel: snapshot.modelLabel)
        let byName = Dictionary(uniqueKeysWithValues: result.sections.flatMap(\.variables).map { ($0.name, $0) })
        let stem = try XCTUnwrap(byName["stem_conv_weights"])
        XCTAssertFalse(stem.initExact)
        XCTAssertNil(stem.driftFromInit)
        XCTAssertGreaterThan(stem.initL2Norm, 0)

        let policyFinal = try XCTUnwrap(byName["policy_conv_weights"] ?? byName["policy_fc_weights"])
        XCTAssertTrue(policyFinal.initExact)
        XCTAssertEqual(policyFinal.initL2Norm, 0)
        XCTAssertNil(policyFinal.l2NormRatioToInit)
        XCTAssertEqual(try XCTUnwrap(policyFinal.driftFromInit), 0)

        let valueHead = try ValueHeadAnalyzer.run(snapshot: snapshot, modelLabel: snapshot.modelLabel)
        let bias = try XCTUnwrap(valueHead.fc2Bias)
        XCTAssertEqual(bias.initial, NetworkArchitecture.wdlBiasPrior(drawProbability: 0.5).map(Double.init))
        XCTAssertEqual(bias.delta, [0, 0, 0])
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
