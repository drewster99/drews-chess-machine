//
//  ArchitectureActivationSiteGraphTests.swift
//  DrewsChessMachineTests
//
//  The per-site activations (format v9) on the GPU: a legacy-decoded model
//  builds the identical forward pass; each head site changes only the head it
//  feeds; each site builds the function its own field names (checked
//  numerically against a CPU computation from the tapped site input, not by
//  op names); fresh trainables do not depend on the activations; leaky heads
//  train; a head-only change changes the behavior fingerprint.
//

import Foundation
import Metal
import XCTest
@testable import DrewsChessMachine

final class ArchitectureActivationSiteGraphTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// Policy logits and the W/D/L distribution, as bit patterns.
    private struct Outputs: Equatable {
        let policy: [UInt32]
        let value: [UInt32]
    }

    private func outputs(_ net: ChessNetwork, board: [Float]) async throws -> Outputs {
        let box = SyncBox<Outputs>(Outputs(policy: [], value: []))
        try await net.evaluateWithValueDistribution(board: board) { policy, wdl in
            box.value = Outputs(policy: policy.map(\.bitPattern),
                                value: [wdl.win, wdl.draw, wdl.loss].map(\.bitPattern))
        }
        return box.value
    }

    private func network(_ arch: NetworkArchitecture, weights: [[Float]], analysisTaps: Bool = false) async throws -> ChessNetwork {
        let net = try ChessNetwork(arch: arch, bnMode: .inference, initialization: .seeded(initSeed: 99),
                                   analysisTaps: analysisTaps)
        try await net.loadWeights(weights)
        return net
    }

    /// One weight set for `arch` with non-trivial batch-norm state: running
    /// variances away from 1, running means and β spread across both signs,
    /// and a non-zero value FC1 bias, so every function's negative side is
    /// exercised.
    private func perturbedWeights(_ arch: NetworkArchitecture) async throws -> [[Float]] {
        let seeded = try ChessNetwork(arch: arch, bnMode: .inference, initialization: .seeded(initSeed: 20261005))
        var weights = try await seeded.exportWeights()
        for (index, spec) in arch.weightTensorPlan().enumerated() {
            for element in weights[index].indices {
                let k = Float(element)
                if spec.name.hasSuffix(".running_var") {
                    weights[index][element] = 0.5 + 0.25 * k.truncatingRemainder(dividingBy: 4)
                } else if spec.name.hasSuffix(".running_mean") {
                    weights[index][element] = 0.1 * (k.truncatingRemainder(dividingBy: 3) - 1)
                } else if spec.kind == .bnAffine, spec.name.hasSuffix(".bias") {
                    weights[index][element] = 0.5 * (k.truncatingRemainder(dividingBy: 5) - 2)
                } else if spec.name == "value.fc1.bias" {
                    weights[index][element] = 0.1 * (k.truncatingRemainder(dividingBy: 5) - 2)
                }
            }
        }
        return weights
    }

    /// `count` deterministic pseudo-random boards in [0, 1).
    private func boards(_ arch: NetworkArchitecture, count: Int) -> [Float] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        return (0..<(count * arch.inputPlanes * 64)).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24)
        }
    }

    // MARK: - Legacy decode

    func testLegacyDecodedArchitectureBuildsTheIdenticalForwardPass() async throws {
        try requireMetal()
        let arch = ArchitectureActivationSiteTests.tiny(activation: .gelu, seStyle: .scaleAndBias)
        let weights = try await perturbedWeights(arch)
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        let encoded = try SafetensorsModelIO.encode(
            modelID: "20261005-1-LGCY", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: meta.trainerSchedule.map(\.completedTrainSteps), corpus: nil))
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(encoded)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata["dcm_format_version"] = "8"
        var object = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(try XCTUnwrap(metadata["architecture"]).utf8)) as? [String: Any])
        for site in ArchitectureActivationSite.allCases { object.removeValue(forKey: site.jsonKey) }
        object["activation_function"] = "gelu"
        metadata["architecture"] = String(
            decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
        let legacy = try SafetensorsModelIO.decode(
            try SafetensorsFile.encode(tensors: tensors, metadata: metadata),
            valueHead: .recenterUnlessMarked, source: "legacy.safetensors")
        XCTAssertEqual(legacy.architecture, arch)
        XCTAssertNotNil(legacy.architectureFormat.legacyLogLine)

        let board = boards(arch, count: 1)
        let current = try await outputs(try await network(arch, weights: weights), board: board)
        let resolved = try await outputs(try await network(legacy.architecture, weights: weights), board: board)
        XCTAssertEqual(current, resolved, "the resolved architecture must build the identical forward pass")
    }

    // MARK: - Head sites

    func testEachHeadSiteChangesOnlyTheHeadItFeeds() async throws {
        try requireMetal()
        let base = ArchitectureActivationSiteTests.tiny(seStyle: .scaleAndBias)
        let weights = try await perturbedWeights(base)
        let board = boards(base, count: 1)
        let reference = try await outputs(try await network(base, weights: weights), board: board)
        for site in [ArchitectureActivationSite.valueHeadConv, .valueHeadFC1Hidden, .policyHead, .towerEnd] {
            var arch = base
            try arch.setActivation(.leakyRelu, at: site)
            let changed = try await outputs(try await network(arch, weights: weights), board: board)
            switch site {
            case .valueHeadConv, .valueHeadFC1Hidden:
                XCTAssertEqual(changed.policy, reference.policy, "\(site) must not change the policy logits")
                XCTAssertNotEqual(changed.value, reference.value, "\(site) must change the value head")
            case .policyHead:
                XCTAssertEqual(changed.value, reference.value, "\(site) must not change the value head")
                XCTAssertNotEqual(changed.policy, reference.policy, "\(site) must change the policy logits")
            case .towerEnd:
                XCTAssertNotEqual(changed.value, reference.value, "the tower end feeds the value head")
                XCTAssertNotEqual(changed.policy, reference.policy, "the tower end feeds the policy head")
            case .stem, .featureSkipFusion:
                XCTFail("not a head site of this fixture")
            }
        }
    }

    // MARK: - Each site builds its function

    private static func apply(_ function: ActivationFunction, _ x: Double) -> Double {
        switch function {
        case .relu: return max(x, 0)
        case .leakyRelu: return x >= 0 ? x : ActivationFunction.leakyReLUNegativeSlope * x
        case .silu: return x / (1 + exp(-x))
        case .gelu: return 0.5 * x * (1 + erf(x / 2.0.squareRoot()))
        case .doesNotApply: preconditionFailure("does_not_apply is not a function")
        }
    }

    /// Inference-form BatchNorm (ε = 1e-5, as `ChessNetwork.batchNorm`) of an
    /// NCHW tensor of `count` positions, followed by `function`.
    private static func batchNormThen(
        _ function: ActivationFunction, input: [Float], count: Int, channels: Int,
        tensors: [String: [Float]], prefix: String
    ) throws -> [Double] {
        let gamma = try XCTUnwrap(tensors["\(prefix).weight"])
        let beta = try XCTUnwrap(tensors["\(prefix).bias"])
        let mean = try XCTUnwrap(tensors["\(prefix).running_mean"])
        let variance = try XCTUnwrap(tensors["\(prefix).running_var"])
        XCTAssertEqual(input.count, count * channels * 64)
        return input.indices.map { index in
            let c = (index / 64) % channels
            let normalized = (Double(input[index]) - Double(mean[c])) / (Double(variance[c]) + 1e-5).squareRoot()
            return apply(function, normalized * Double(gamma[c]) + Double(beta[c]))
        }
    }

    /// A bias-free 1×1 convolution, weight OIHW `[out, in, 1, 1]`.
    private static func conv1x1(_ input: [Double], count: Int, inChannels: Int, outChannels: Int, weight: [Float]) -> [Double] {
        var output = [Double](repeating: 0, count: count * outChannels * 64)
        for n in 0..<count {
            for o in 0..<outChannels {
                for square in 0..<64 {
                    var sum = 0.0
                    for i in 0..<inChannels {
                        sum += Double(weight[o * inChannels + i]) * input[(n * inChannels + i) * 64 + square]
                    }
                    output[(n * outChannels + o) * 64 + square] = sum
                }
            }
        }
        return output
    }

    /// The value FC1 (`flatten → x·W + b`, native `[in, out]` weight) and
    /// its hidden activation.
    private static func valueFC1(
        _ input: [Double], count: Int, inputs: Int, units: Int, weight: [Float], bias: [Float], function: ActivationFunction
    ) -> [Double] {
        var output = [Double](repeating: 0, count: count * units)
        for n in 0..<count {
            for j in 0..<units {
                var sum = Double(bias[j])
                for i in 0..<inputs {
                    sum += input[n * inputs + i] * Double(weight[i * units + j])
                }
                output[n * units + j] = apply(function, sum)
            }
        }
        return output
    }

    /// The CPU prediction of `site`'s tapped output, with `function` at
    /// `site` and every other site at the architecture's own value.
    private static func prediction(
        site: ArchitectureActivationSite, function: ActivationFunction, arch: NetworkArchitecture,
        taps: [String: [Float]], tensors: [String: [Float]], count: Int
    ) throws -> (expected: [Double], tapName: String) {
        func tap(_ name: String) throws -> [Float] { try XCTUnwrap(taps[name], "tap \(name)") }
        switch site {
        case .stem:
            return (try batchNormThen(function, input: try tap("stem_bn_input"), count: count,
                                      channels: arch.stemOutputChannels, tensors: tensors, prefix: "stem.bn"),
                    "stem_output")
        case .towerEnd:
            return (try batchNormThen(function, input: try tap("tower_final_bn_input"), count: count,
                                      channels: arch.towerOutputChannels, tensors: tensors, prefix: "tower_final_bn"),
                    "tower_output")
        case .featureSkipFusion:
            let fused = try batchNormThen(function, input: try tap("feature_skip_bn_input"), count: count,
                                          channels: arch.towerOutputChannels, tensors: tensors, prefix: "feature_skip.bn")
            return (conv1x1(fused, count: count, inChannels: arch.towerOutputChannels,
                            outChannels: arch.policyPreConvChannels,
                            weight: try XCTUnwrap(tensors["policy.pre_conv.weight"])),
                    "policy_pre_bn_input")
        case .policyHead:
            return (try batchNormThen(function, input: try tap("policy_pre_bn_input"), count: count,
                                      channels: arch.policyPreConvChannels, tensors: tensors, prefix: "policy.pre_bn"),
                    "policy_pre_act")
        case .valueHeadConv, .valueHeadFC1Hidden:
            let convFunction = site == .valueHeadConv ? function : arch.valueHeadConvActivation
            let hiddenFunction = site == .valueHeadFC1Hidden ? function : arch.valueHeadFC1HiddenActivation
            let conv = try batchNormThen(convFunction, input: try tap("value_bn_input"), count: count,
                                         channels: arch.valueHeadConvChannels, tensors: tensors, prefix: "value.bn")
            return (valueFC1(conv, count: count, inputs: arch.valueHeadConvChannels * 64,
                             units: arch.valueHeadHiddenUnits,
                             weight: try XCTUnwrap(tensors["value.fc1.weight"]),
                             bias: try XCTUnwrap(tensors["value.fc1.bias"]), function: hiddenFunction),
                    "value_fc1_act")
        }
    }

    func testEachSiteBuildsItsSelectedFunction() async throws {
        try requireMetal()
        let base = ArchitectureActivationSiteTests.fullSiteFixture()
        let weights = try await perturbedWeights(base)
        let tensors = Dictionary(uniqueKeysWithValues: zip(base.weightTensorPlan().map(\.name), weights))
        let count = 3
        let input = boards(base, count: count)
        for site in ArchitectureActivationSite.allCases {
            for function in ActivationFunction.functions {
                var arch = base
                try arch.setActivation(function, at: site)
                let net = try await network(arch, weights: weights, analysisTaps: true)
                let tapped = try await net.evaluateAnalysisTaps(boards: input, count: count)
                let taps = Dictionary(uniqueKeysWithValues: tapped.map { ($0.name, $0.values) })
                let (expected, tapName) = try Self.prediction(
                    site: site, function: function, arch: arch, taps: taps, tensors: tensors, count: count)
                let actual = try XCTUnwrap(taps[tapName], "tap \(tapName)")
                XCTAssertEqual(actual.count, expected.count, "\(site) \(function.rawValue)")
                let scale = max(expected.map(abs).max() ?? 0, 1e-30)
                let tolerance = 1e-5 * scale
                let error = zip(actual, expected).map { abs(Double($0) - $1) }.max() ?? 0
                XCTAssertLessThanOrEqual(error, tolerance, "\(site) \(function.rawValue): max error \(error)")
                // The tolerance is far below the gap to every other function,
                // so the check cannot pass with the wrong function built.
                for other in ActivationFunction.functions where other != function {
                    let (otherExpected, _) = try Self.prediction(
                        site: site, function: other, arch: arch, taps: taps, tensors: tensors, count: count)
                    let gap = zip(actual, otherExpected).map { abs(Double($0) - $1) }.max() ?? 0
                    XCTAssertGreaterThan(gap, 100 * tolerance, "\(site) \(function.rawValue) vs \(other.rawValue)")
                }
            }
        }
    }

    // MARK: - Fresh build

    func testFreshTrainablesAreActivationIndependent() async throws {
        try requireMetal()
        let base = ArchitectureActivationSiteTests.fullSiteFixture()
        var leakyEverywhere = base
        try leakyEverywhere.setActivationAtEveryExistingSite(.leakyRelu)
        var leakyHeads = base
        for site in [ArchitectureActivationSite.policyHead, .valueHeadConv, .valueHeadFC1Hidden] {
            try leakyHeads.setActivation(.leakyRelu, at: site)
        }
        let plan = base.weightTensorPlan()
        func fresh(_ arch: NetworkArchitecture) async throws -> [[Float]] {
            try await ChessMPSNetwork(.randomWeights(initSeed: 20261005), arch: arch).network.exportWeights()
        }
        let reference = try await fresh(base)
        let everywhere = try await fresh(leakyEverywhere)
        let heads = try await fresh(leakyHeads)
        for (index, spec) in plan.enumerated() where spec.kind != .bnRunningStat {
            XCTAssertEqual(everywhere[index].map(\.bitPattern), reference[index].map(\.bitPattern), spec.name)
        }
        for (index, spec) in plan.enumerated() {
            XCTAssertEqual(heads[index].map(\.bitPattern), reference[index].map(\.bitPattern),
                           "\(spec.name): only head activations changed, so every tensor (BN statistics included) must match")
        }
    }

    // MARK: - Training

    func testLeakyHeadsTrainOneFiniteStep() async throws {
        try requireMetal()
        var arch = ArchitectureActivationSiteTests.tiny(seStyle: .scaleAndBias)
        arch.computeDataType = .bFloat16
        for site in [ArchitectureActivationSite.policyHead, .valueHeadConv, .valueHeadFC1Hidden] {
            try arch.setActivation(.leakyRelu, at: site)
        }
        for precision in ChessNetwork.PolicyTailPrecision.allCases {
            let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), lrWarmupSteps: 0, arch: arch,
                                           initialization: .seeded(initSeed: 1), policyTailPrecision: precision)
            let timing = try await trainer.trainStep(batchSize: 8)
            XCTAssertTrue(timing.policyLoss.isFinite, "\(precision.rawValue): policy loss must be finite")
            XCTAssertTrue(timing.valueLoss.isFinite, "\(precision.rawValue): value loss must be finite")
        }
    }

    // MARK: - Behavior fingerprint

    func testAHeadActivationChangeChangesTheBehaviorFingerprint() async throws {
        try requireMetal()
        let relu = ArchitectureActivationSiteTests.tiny(seStyle: .scaleAndBias)
        var leakyValue = relu
        try leakyValue.setActivation(.leakyRelu, at: .valueHeadFC1Hidden)
        let base = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(arch: relu, policyTailPrecision: .float32FromPreBatchNorm),
            streamDerivation: DCMRandomStreams.self)
        let changed = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(arch: leakyValue, policyTailPrecision: .float32FromPreBatchNorm),
            streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(base.sha256, changed.sha256)
    }
}
