//
//  LayerHealthTests.swift
//  DrewsChessMachineTests
//
//  Pins the layer-health module (`LayerHealth` / `LayerHealthSummary`):
//
//  - Site enumeration agrees with `weightTensorPlan()` for every head style,
//    block style (pre / post / mixed), SE style, uniform and heterogeneous
//    towers, and both feature-skip fusions — same BN layers, same order,
//    same widths; SE FC1 / ReZero / value FC1 names resolve in the plan.
//  - Site activation tagging agrees with the BUILT graph: an activation op
//    consumes a BN op's output directly iff the site says it does.
//  - β/|γ| threshold classification, γ = 0 and non-finite channels.
//  - SiLU / GELU / unactivated sites report the classification as not
//    applicable, never as zero counts.
//  - Velocity dead units use the native [in, out] layout (column = unit).
//  - ReZero fraction of cap and saturation; NaN / Inf detection; the summary
//    always JSON-encodes.
//  - The trainer's live readback returns exactly the exported trainer
//    state's values at those names (masters under bf16, working under fp32).
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class LayerHealthTests: XCTestCase {

    // MARK: - Architectures

    private func tinyArch(
        style: BlockActivationStyle = .pre,
        activation: ActivationFunction = .relu,
        seStyle: SEStyle = .scaleAndBias,
        policy: PolicyHeadStyle = .intermediateConv,
        dtype: ComputeDataType = .float32
    ) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 2, stemConvKernelSize: 3,
            activationFunction: activation, blockActivationStyle: style,
            blockSkipMerge: style == .pre ? .cleanAdd : .activationGated,
            blockUseRezero: style == .pre, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: seStyle, blockSeReductionRatio: 4,
            policyHeadStyle: policy, policyPreConvChannels: 8,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 8,
            computeDataType: dtype)
    }

    /// Post-activation 16ch group (attenuate SE, leaky) into a
    /// pre-activation 32ch group (scale-and-bias SE, ReZero): a width
    /// transition, a style change, and a stem activation plus a tower-end BN.
    private func mixedArch() -> NetworkArchitecture {
        var arch = tinyArch(activation: .leakyRelu)
        arch.blockGroups = [
            BlockGroup(
                count: 1, channels: 16, conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .attenuateOnly, seReductionRatio: 4,
                useRezero: false, rezeroAlphaInit: 1,
                activationFunction: .leakyRelu, activationStyle: .post,
                skipMerge: .activationGated, dropoutMultiplier: 1),
            BlockGroup(
                count: 2, channels: 32, conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .scaleAndBias, seReductionRatio: 4,
                useRezero: true, rezeroAlphaInit: 0.4,
                activationFunction: .relu, activationStyle: .pre,
                skipMerge: .cleanAdd, dropoutMultiplier: 1,
                seActivation: .leakyRelu),
        ]
        arch.stemActivation = .leakyRelu
        return arch
    }

    private func compressSkipArch() -> NetworkArchitecture {
        var arch = tinyArch(policy: .fcBottleneck)
        arch.featureSkipSource = .stemOutput
        arch.featureSkipFusion = .compressConvBNReLU
        arch.featureSkipActivation = .relu
        arch.featureSkipToPolicyHead = true
        arch.featureSkipToValueHead = false
        arch.featureSkipToFinalBlock = false
        return arch
    }

    private func concatToFinalBlockArch() -> NetworkArchitecture {
        var arch = tinyArch(seStyle: .none, policy: .simpleConv)
        arch.featureSkipSource = .stemOutput
        arch.featureSkipFusion = .concatDirect
        arch.featureSkipToPolicyHead = false
        arch.featureSkipToValueHead = true
        arch.featureSkipToFinalBlock = true
        return arch
    }

    /// A tower whose every activation (main path, SE FC1, heads) is `fn`.
    private func smoothArch(_ fn: ActivationFunction) -> NetworkArchitecture {
        tinyArch(activation: fn)
    }

    private func allArchitectures() throws -> [(String, NetworkArchitecture)] {
        var list: [(String, NetworkArchitecture)] = NetworkArchitecture.Preset.allCases.map {
            ("preset \($0.rawValue)", NetworkArchitecture.preset($0))
        }
        for style in [BlockActivationStyle.pre, .post] {
            for se in SEStyle.allCases {
                for policy in PolicyHeadStyle.allCases {
                    list.append(("tiny \(style.rawValue) \(se.rawValue) \(policy.rawValue)",
                                 tinyArch(style: style, seStyle: se, policy: policy)))
                }
            }
        }
        list.append(("mixed post->pre", mixedArch()))
        list.append(("compress feature skip", compressSkipArch()))
        list.append(("concat to final block", concatToFinalBlockArch()))
        list.append(("silu", smoothArch(.silu)))
        list.append(("gelu", smoothArch(.gelu)))
        for (label, arch) in list {
            XCTAssertNoThrow(try arch.validate(), label)
        }
        return list
    }

    // MARK: - Site enumeration vs weightTensorPlan

    func testBatchNormSitesMatchThePlanRunningStatsInOrder() throws {
        for (label, arch) in try allArchitectures() {
            let plan = arch.weightTensorPlan()
            let planRunningMeans = plan.filter { $0.kind == .bnRunningStat && $0.name.hasSuffix(".running_mean") }
            let planLayers = planRunningMeans.map { String($0.name.dropLast(".running_mean".count)) }
            let sites = LayerHealth.batchNormSites(for: arch)
            XCTAssertEqual(sites.map(\.name), planLayers, "\(label): BN layers and their build order")
            let specByName = Dictionary(uniqueKeysWithValues: plan.map { ($0.name, $0) })
            for site in sites {
                XCTAssertEqual(specByName[site.runningMeanTensorName]?.shape, [site.channels], "\(label): \(site.name) width")
                XCTAssertEqual(specByName[site.gammaTensorName]?.kind, .bnAffine, "\(label): \(site.name) γ")
                XCTAssertEqual(specByName[site.betaTensorName]?.kind, .bnAffine, "\(label): \(site.name) β")
                XCTAssertEqual(specByName[site.runningVarianceTensorName]?.kind, .bnRunningStat, "\(label): \(site.name) running var")
                XCTAssertEqual(specByName[site.gammaTensorName]?.shape, [site.channels], "\(label): \(site.name) γ width")
            }
        }
    }

    func testHiddenUnitLayersAndReZeroSitesResolveInThePlan() throws {
        for (label, arch) in try allArchitectures() {
            let specByName = Dictionary(uniqueKeysWithValues: arch.weightTensorPlan().map { ($0.name, $0) })
            let seLayers = LayerHealth.squeezeExcitationFC1Layers(for: arch)
            XCTAssertEqual(seLayers.count, arch.expandedBlocks.filter { $0.seStyle != .none }.count, label)
            for layer in seLayers + [LayerHealth.valueFC1Layer(for: arch)] {
                let spec = specByName[layer.weightTensorName]
                XCTAssertEqual(spec?.kind, .linear, "\(label): \(layer.weightTensorName)")
                XCTAssertEqual(spec?.shape, [layer.inputCount, layer.unitCount], "\(label): \(layer.weightTensorName) is [in, out]")
            }
            for (blockIndex, spec) in arch.expandedBlocks.enumerated() where spec.seStyle != .none {
                let layer = seLayers.first { $0.blockIndex == blockIndex }
                XCTAssertEqual(layer?.activation, spec.seActivation, "\(label): block \(blockIndex) SE FC1 activation")
            }
            let reZero = LayerHealth.reZeroSites(for: arch)
            XCTAssertEqual(reZero.count, arch.expandedBlocks.filter(\.useRezero).count, label)
            for site in reZero {
                XCTAssertEqual(specByName[site.alphaTensorName]?.kind, .scalar, "\(label): \(site.alphaTensorName)")
                let spec = arch.expandedBlocks[site.blockIndex]
                XCTAssertEqual(site.cap, Double(spec.rezeroAlphaInit) * NetworkArchitecture.rezeroTanhCeilingMultiple, accuracy: 0)
            }
            let planNames = Set(specByName.keys)
            for name in LayerHealth.liveStateTensorNames(for: arch) {
                XCTAssertTrue(planNames.contains(name), "\(label): live tensor \(name) is in the plan")
            }
        }
    }

    func testActivationTaggingFollowsTheBlockAndHeadStyles() {
        let pre = LayerHealth.batchNormSites(for: tinyArch(style: .pre))
        XCTAssertNil(pre.first { $0.name == "stem.bn" }?.activation, "pre-act stem BN feeds block 0's BN, not an activation")
        XCTAssertEqual(pre.first { $0.name == "blocks.0.bn1" }?.activation, .relu)
        XCTAssertEqual(pre.first { $0.name == "blocks.1.bn2" }?.activation, .relu)
        XCTAssertEqual(pre.first { $0.name == "tower_final_bn" }?.activation, .relu)

        let post = LayerHealth.batchNormSites(for: tinyArch(style: .post))
        XCTAssertEqual(post.first { $0.name == "stem.bn" }?.activation, .relu)
        XCTAssertEqual(post.first { $0.name == "blocks.0.bn1" }?.activation, .relu)
        XCTAssertNil(post.first { $0.name == "blocks.0.bn2" }?.activation, "post-act BN2 feeds SE / the merge")
        XCTAssertNil(post.first { $0.name == "tower_final_bn" }, "post-act tail has no tower-end BN")

        XCTAssertNil(LayerHealth.batchNormSites(for: tinyArch(policy: .simpleConv)).first { $0.name == "policy.pre_bn" })
        XCTAssertEqual(LayerHealth.batchNormSites(for: tinyArch(policy: .fcBottleneck)).first { $0.name == "policy.pre_bn" }?.channels, 8)

        let mixed = LayerHealth.batchNormSites(for: mixedArch())
        XCTAssertEqual(mixed.first { $0.name == "stem.bn" }?.activation, .leakyRelu, "first group post-act -> stem act (tower activation)")
        XCTAssertEqual(mixed.first { $0.name == "blocks.0.bn1" }?.activation, .leakyRelu)
        XCTAssertEqual(mixed.first { $0.name == "blocks.1.bn1" }?.channels, 16, "pre-act BN1 normalizes the 16ch input at the transition")
        XCTAssertEqual(mixed.first { $0.name == "blocks.1.bn1" }?.activation, .relu)
        XCTAssertEqual(mixed.first { $0.name == "tower_final_bn" }?.channels, 32)

        let concat = LayerHealth.batchNormSites(for: concatToFinalBlockArch())
        XCTAssertEqual(concat.first { $0.name == "blocks.1.bn1" }?.channels, 32, "final-block concat widens BN1 by the stem width")
    }

    // MARK: - Site tagging vs the built graph

    /// Graph op names the builder gives a site's BN and its activation.
    private func graphNames(forSite site: String) -> (batchNorm: String, activation: String) {
        switch site {
        case "stem.bn": return ("stem_bn", "stem_act")
        case "tower_final_bn": return ("tower_final_bn", "tower_final_act")
        case "feature_skip.bn": return ("feature_skip_bn", "feature_skip_act")
        case "policy.pre_bn": return ("policy_pre_bn", "policy_pre_act")
        case "value.bn": return ("value_bn", "value_act")
        default:
            let parts = site.split(separator: ".")
            precondition(parts.count == 3 && parts[0] == "blocks", "unexpected site \(site)")
            let which = parts[2].dropFirst("bn".count)
            return ("block\(parts[1])_bn\(which)", "block\(parts[1])_act\(which)")
        }
    }

    /// Producer op name -> names of the ops that read its output, over every
    /// op reachable backward from the head outputs.
    private func consumerNames(of network: ChessNetwork) -> [String: Set<String>] {
        var consumers: [String: Set<String>] = [:]
        var visited = Set<ObjectIdentifier>()
        var stack: [MPSGraphOperation] = [
            network.policyOutput.operation, network.valueLogits.operation, network.valueOutput.operation,
        ]
        while let op = stack.popLast() {
            guard visited.insert(ObjectIdentifier(op)).inserted else { continue }
            for input in op.inputTensors {
                consumers[input.operation.name, default: []].insert(op.name)
                stack.append(input.operation)
            }
        }
        return consumers
    }

    func testSiteActivationTaggingMatchesTheBuiltGraph() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let architectures: [(String, NetworkArchitecture)] = [
            ("pre relu", tinyArch(style: .pre)),
            ("post relu", tinyArch(style: .post, policy: .simpleConv)),
            ("mixed post->pre leaky/relu", mixedArch()),
            ("compress feature skip, fc_bottleneck", compressSkipArch()),
            ("concat to final block", concatToFinalBlockArch()),
            ("silu", smoothArch(.silu)),
        ]
        for (label, arch) in architectures {
            let network = try ChessNetwork(arch: arch, bnMode: .inference, initialization: .seeded(initSeed: 1))
            let consumers = consumerNames(of: network)
            for site in LayerHealth.batchNormSites(for: arch) {
                let names = graphNames(forSite: site.name)
                guard let readers = consumers[names.batchNorm] else {
                    XCTFail("\(label): BN op \(names.batchNorm) for \(site.name) not found in the graph")
                    continue
                }
                if site.activation != nil {
                    XCTAssertTrue(readers.contains(names.activation),
                                  "\(label): \(names.activation) must read \(names.batchNorm) directly (readers: \(readers.sorted()))")
                } else {
                    let activationReaders = readers.filter { $0.hasSuffix("_act") || $0.hasSuffix("_act1") || $0.hasSuffix("_act2") }
                    XCTAssertTrue(activationReaders.isEmpty,
                                  "\(label): \(names.batchNorm) is tagged unactivated but is read by \(activationReaders.sorted())")
                }
            }
        }
    }

    // MARK: - Threshold classification

    func testBetaOverAbsGammaClassification() {
        let site = LayerHealth.BatchNormSite(name: "test.bn", channels: 12, activation: .relu)
        //            ch:   0     1     2     3    4    5    6    7    8    9     10          11
        let gamma: [Float] = [1,    1,    1,    1,   1,   1,   1,   0,   0,   -2,   .nan,       1]
        let beta: [Float] = [-3.5, -3.0, -2.5, -2.0, 0, 3.0, 3.5, 1,   0,   8,    0,          0.5]
        let health = LayerHealth.batchNormSiteHealth(
            site: site, gamma: gamma, beta: beta, runningVariance: [Float](repeating: 1, count: 12))
        XCTAssertEqual(health.classification, .classified)
        // dead: ch0 (< -3) and ch8 (γ = 0, β = 0 → constant 0).
        XCTAssertEqual(health.deadChannelCount, 2)
        // mostly off: ch1 (-3 is not < -3) and ch2.
        XCTAssertEqual(health.mostlyOffChannelCount, 2)
        // always on: ch6 (3.5), ch7 (γ = 0, β > 0), ch9 (8 / |−2| = 4 — |γ|, not γ).
        XCTAssertEqual(health.alwaysOnChannelCount, 3)
        XCTAssertEqual(health.zeroGammaChannelCount, 2)
        XCTAssertEqual(health.nonFiniteChannelCount, 1)
        XCTAssertEqual(health.unhealthyChannelCount, 7)
        XCTAssertEqual(health.minBetaOverAbsGamma, -3.5)
        XCTAssertEqual(health.minBetaOverAbsGammaChannel, 0)
        XCTAssertEqual(health.maxBetaOverAbsGamma, 4)
        XCTAssertEqual(health.maxBetaOverAbsGammaChannel, 9)
    }

    func testThresholdConstantsAreTheReportLabels() {
        XCTAssertEqual(LayerHealth.deadBetaOverAbsGamma, -3)
        XCTAssertEqual(LayerHealth.mostlyOffBetaOverAbsGamma, -2)
        XCTAssertEqual(LayerHealth.alwaysOnBetaOverAbsGamma, 3)
        XCTAssertEqual(LayerHealth.lowVelocityFractionOfReferenceUnitNorm, 0.05)
        XCTAssertEqual(LayerHealth.lowVelocityReferencePercentile, 90)
    }

    func testSmoothAndUnactivatedSitesReportNotApplicable() throws {
        let gamma: [Float] = [1, 1]
        let beta: [Float] = [-10, 10]
        let variance: [Float] = [1, 1]
        for fn in [ActivationFunction.silu, .gelu] {
            let health = LayerHealth.batchNormSiteHealth(
                site: .init(name: "s", channels: 2, activation: fn), gamma: gamma, beta: beta, runningVariance: variance)
            XCTAssertEqual(health.classification, .notApplicableSmoothActivation, fn.rawValue)
            XCTAssertNil(health.deadChannelCount, fn.rawValue)
            XCTAssertNil(health.mostlyOffChannelCount, fn.rawValue)
            XCTAssertNil(health.alwaysOnChannelCount, fn.rawValue)
            XCTAssertEqual(health.maxBetaOverAbsGamma, 10, "\(fn.rawValue): β/|γ| extremes are still reported")
        }
        let unactivated = LayerHealth.batchNormSiteHealth(
            site: .init(name: "s", channels: 2, activation: nil), gamma: gamma, beta: beta, runningVariance: variance)
        XCTAssertEqual(unactivated.classification, .notApplicableNoActivation)
        XCTAssertNil(unactivated.deadChannelCount)
        XCTAssertEqual(LayerHealth.classification(for: .leakyRelu), .classified)
        XCTAssertEqual(LayerHealth.classification(for: .relu), .classified)

        // A whole-tower SiLU net has no classified site, and the compact
        // line says so instead of printing zeros.
        let arch = smoothArch(.silu)
        let summary = try LayerHealth.summarizePlanAligned(
            arch: arch, baseWeights: syntheticWeights(arch), velocity: .unavailable(reason: "test"))
        XCTAssertEqual(summary.classifiedSites.count, 0)
        XCTAssertEqual(summary.batchNormSites.filter { $0.classification == .notApplicableSmoothActivation }.count,
                       summary.batchNormSites.filter { $0.activation != nil }.count)
        XCTAssertTrue(summary.compactLine().contains("dead=n/a off=n/a alwaysOn=n/a"), summary.compactLine())
        XCTAssertEqual(summary.velocityNotIncludedReason, "test")
    }

    // MARK: - Running variance

    func testRunningVarianceMaxOverMedian() {
        let site = LayerHealth.BatchNormSite(name: "v", channels: 6, activation: .relu)
        let health = LayerHealth.batchNormSiteHealth(
            site: site, gamma: [Float](repeating: 1, count: 6), beta: [Float](repeating: 0, count: 6),
            runningVariance: [1, 1, 1, 1, 1200, .infinity])
        XCTAssertEqual(health.runningVarianceMax, 1200)
        XCTAssertEqual(health.runningVarianceMaxChannel, 4)
        XCTAssertEqual(health.runningVarianceMedian, 1)
        XCTAssertEqual(health.runningVarianceMaxOverMedian, 1200)
        XCTAssertEqual(health.nonFiniteRunningVarianceCount, 1)

        let zeroMedian = LayerHealth.batchNormSiteHealth(
            site: LayerHealth.BatchNormSite(name: "z", channels: 3, activation: .relu),
            gamma: [1, 1, 1], beta: [0, 0, 0], runningVariance: [0, 0, 5])
        XCTAssertNil(zeroMedian.runningVarianceMaxOverMedian, "no ratio against a zero median")
    }

    // MARK: - Velocity (native [in, out] layout)

    /// A bimodal layer where most units are weak: against the median they
    /// would all look normal (the median unit is weak); against the 90th
    /// percentile they are counted. Mirrors the SE bottlenecks measured on
    /// 2026-10-02 (more than half of each block's units on the leaky trickle).
    func testLowVelocityIsMeasuredAgainstThe90thPercentileNotTheMedian() throws {
        let units = 10
        let layer = LayerHealth.HiddenUnitLayer(
            weightTensorName: "x.fc1.weight", blockIndex: 0, inputCount: 1, unitCount: units, activation: .leakyRelu)
        // One input row: 6 weak units at 0.01, 4 active units at 1.0.
        let velocity: [Float] = (0..<units).map { $0 < 6 ? 0.01 : 1.0 }
        let health = try LayerHealth.hiddenUnitVelocityHealth(layer: layer, velocity: velocity)
        XCTAssertEqual(health.zeroVelocityUnitCount, 0)
        XCTAssertEqual(health.lowVelocityUnitCount, 6, "every weak unit is below 5% of the active units' norm")
        let reference = try XCTUnwrap(health.referenceUnitVelocityNorm)
        XCTAssertGreaterThan(reference, 0.5, "the reference tracks the active units")
        let median = try XCTUnwrap(health.medianUnitVelocityNorm)
        XCTAssertLessThan(median, 0.05, "the median is itself a weak unit here")
    }

    func testZeroAndLowVelocityUnitsReadColumnsOfTheInOutLayout() throws {
        let layer = LayerHealth.HiddenUnitLayer(
            weightTensorName: "x.fc1.weight", blockIndex: 0, inputCount: 4, unitCount: 4, activation: .relu)
        // [in, out] row-major: element (input i, unit j) at i * 4 + j.
        // unit 0: 1.0 everywhere; unit 1: exactly 0 everywhere (dead);
        // unit 2: 1e-3 (nonzero, far below 5% of the median); unit 3: 2.0.
        var velocity = [Float](repeating: 0, count: 16)
        for i in 0..<4 {
            velocity[i * 4 + 0] = 1
            velocity[i * 4 + 1] = 0
            velocity[i * 4 + 2] = 1e-3
            velocity[i * 4 + 3] = 2
        }
        let health = try LayerHealth.hiddenUnitVelocityHealth(layer: layer, velocity: velocity)
        XCTAssertEqual(health.zeroVelocityUnitCount, 1)
        XCTAssertEqual(health.zeroVelocityUnits, [1])
        XCTAssertEqual(health.lowVelocityUnitCount, 1)
        XCTAssertEqual(health.nonFiniteUnitCount, 0)

        // The same zeros laid out as an INPUT row (row 1) touch every unit,
        // so no unit is dead: the layout, not the zero count, decides.
        var rowZero = [Float](repeating: 1, count: 16)
        for j in 0..<4 { rowZero[1 * 4 + j] = 0 }
        let rowHealth = try LayerHealth.hiddenUnitVelocityHealth(layer: layer, velocity: rowZero)
        XCTAssertEqual(rowHealth.zeroVelocityUnitCount, 0)

        var poisoned = velocity
        poisoned[2 * 4 + 3] = .nan
        let poisonedHealth = try LayerHealth.hiddenUnitVelocityHealth(layer: layer, velocity: poisoned)
        XCTAssertEqual(poisonedHealth.nonFiniteUnitCount, 1)

        XCTAssertThrowsError(try LayerHealth.hiddenUnitVelocityHealth(layer: layer, velocity: [0, 0, 0]))
    }

    // MARK: - ReZero

    func testReZeroFractionOfCapAndSaturation() {
        let site = LayerHealth.ReZeroSite(blockIndex: 3, alphaTensorName: "blocks.3.rezero_alpha", cap: 0.5)
        let atCap = LayerHealth.reZeroHealthEntry(site: site, alpha: 0.5)
        XCTAssertEqual(atCap.fractionOfCap ?? .nan, tanh(1.0), accuracy: 1e-12)
        XCTAssertEqual(atCap.effectiveAlpha ?? .nan, 0.5 * tanh(1.0), accuracy: 1e-12)
        XCTAssertEqual(atCap.saturated, false)
        let far = LayerHealth.reZeroHealthEntry(site: site, alpha: 5)
        XCTAssertEqual(far.saturated, true)
        let negative = LayerHealth.reZeroHealthEntry(site: site, alpha: -5)
        XCTAssertEqual(negative.saturated, true, "saturation is on |fraction|")
        let nan = LayerHealth.reZeroHealthEntry(site: site, alpha: .nan)
        XCTAssertNil(nan.alpha)
        XCTAssertNil(nan.fractionOfCap)
        XCTAssertNil(nan.saturated)
    }

    // MARK: - Full summaries

    /// Plan-aligned weights for `arch`: BN γ = 1, β = 0, running mean 0,
    /// running var 1, ReZero α = its init, everything else a small constant.
    private func syntheticWeights(_ arch: NetworkArchitecture) -> [[Float]] {
        let reZeroInit = Dictionary(uniqueKeysWithValues: LayerHealth.reZeroSites(for: arch).map {
            ($0.alphaTensorName, arch.expandedBlocks[$0.blockIndex].rezeroAlphaInit)
        })
        return arch.weightTensorPlan().map { spec -> [Float] in
            let value: Float
            switch spec.kind {
            case .bnAffine: value = spec.name.hasSuffix(".weight") ? 1 : 0
            case .bnRunningStat: value = spec.name.hasSuffix(".running_var") ? 1 : 0
            case .scalar: value = reZeroInit[spec.name] ?? 0
            case .conv, .linear, .bias: value = 0.01
            }
            return [Float](repeating: value, count: spec.elementCount)
        }
    }

    private func syntheticVelocity(_ arch: NetworkArchitecture) -> [[Float]] {
        arch.trainableTensorPlan().map { [Float](repeating: 1e-3, count: $0.elementCount) }
    }

    func testTrainerStateSummaryFindsInjectedFaultsAndEncodes() throws {
        let arch = tinyArch()
        let plan = arch.weightTensorPlan()
        let trainables = arch.trainableTensorPlan()
        var weights = syntheticWeights(arch)
        var velocity = syntheticVelocity(arch)
        func planIndex(_ name: String) throws -> Int {
            try XCTUnwrap(plan.firstIndex { $0.name == name }, name)
        }
        func trainableIndex(_ name: String) throws -> Int {
            try XCTUnwrap(trainables.firstIndex { $0.name == name }, name)
        }
        // One always-on policy pre-block channel (β = 10).
        let policyBeta = try planIndex("policy.pre_bn.bias")
        weights[policyBeta][5] = 10
        // One dead block-1 BN2 channel (β = -4).
        let blockBeta = try planIndex("blocks.1.bn2.bias")
        weights[blockBeta][2] = -4
        // One runaway running variance.
        let valueVariance = try planIndex("value.bn.running_var")
        weights[valueVariance][1] = 500
        // A NaN in a conv weight and an Inf in a velocity.
        let conv = try planIndex("blocks.0.conv1.weight")
        weights[conv][7] = .nan
        let stemVelocity = try trainableIndex("stem.conv.weight")
        velocity[stemVelocity][0] = .infinity
        // Block 0's SE FC1 ([16, 4]): unit 2's column of weight velocity is exactly 0.
        let fc1 = try trainableIndex("blocks.0.se_scalebias.fc1.weight")
        for input in 0..<16 { velocity[fc1][input * 4 + 2] = 0 }

        let summary = try LayerHealth.summarizeTrainerState(arch: arch, trainerWeights: weights + velocity)
        XCTAssertEqual(summary.scope, .allTensors)
        XCTAssertEqual(summary.alwaysOnChannelCount, 1)
        XCTAssertEqual(summary.deadChannelCount, 1)
        XCTAssertEqual(summary.worstRunningVarianceSite?.site, "value.bn")
        XCTAssertEqual(summary.worstRunningVarianceSite?.runningVarianceMaxOverMedian, 500)
        XCTAssertEqual(summary.nonFiniteValueCount, 2)
        XCTAssertEqual(summary.nonFiniteTensorCount, 2)
        XCTAssertEqual(summary.nonFiniteTensorNames, ["blocks.0.conv1.weight", "opt.stem.conv.weight.velocity"])
        XCTAssertEqual(summary.examinedTensorCount, plan.count + trainables.count)
        let se = try XCTUnwrap(summary.squeezeExcitationFC1)
        XCTAssertEqual(se.map(\.zeroVelocityUnitCount), [1, 0])
        XCTAssertEqual(se.first?.zeroVelocityUnits, [2])
        XCTAssertEqual(summary.valueFC1?.zeroVelocityUnitCount, 0)
        XCTAssertNil(summary.velocityNotIncludedReason)
        XCTAssertEqual(summary.reZero.count, 2)
        XCTAssertEqual(summary.reZero.first?.fractionOfCap ?? .nan, tanh(1.0), accuracy: 1e-6)
        XCTAssertNotNil(summary.largestMagnitudeTensors)

        let line = summary.compactLine()
        XCTAssertTrue(line.contains("alwaysOn=1"), line)
        XCTAssertTrue(line.contains("dead=1"), line)
        XCTAssertTrue(line.contains("maxBetaOverGamma=+10.00@policy.pre_bn[5]"), line)
        XCTAssertTrue(line.contains("rvMaxOverMedian=500.0@value.bn[1]"), line)
        XCTAssertTrue(line.contains("nonFinite=2"), line)
        XCTAssertTrue(line.contains("seZeroVel=1/8"), line)
        XCTAssertTrue(line.contains("worstSE=blocks.0(1/4)"), line)
        XCTAssertFalse(summary.detailedLines().isEmpty)

        // Every Double is finite by construction, so the default (strict)
        // JSONEncoder must accept the summary even with NaN/Inf inputs.
        let data = try JSONEncoder().encode(summary)
        let decoded = try JSONDecoder().decode(LayerHealthSummary.self, from: data)
        XCTAssertEqual(decoded, summary)

        let findings = NumericsAudit.layerHealthFindings(summary)
        XCTAssertTrue(findings.contains { $0.verdict == .bad }, "non-finite values are BAD")
        XCTAssertTrue(findings.contains { $0.subject == "policy.pre_bn" && $0.verdict == .degraded })
        XCTAssertTrue(findings.contains { $0.subject == "blocks.0.se_scalebias.fc1.weight" })
    }

    func testLiveScopeNeedsOnlyTheLiveTensors() throws {
        let arch = mixedArch()
        let plan = arch.weightTensorPlan()
        let all = Dictionary(uniqueKeysWithValues: zip(plan.map(\.name), syntheticWeights(arch)))
        var live: [String: [Float]] = [:]
        for name in LayerHealth.liveStateTensorNames(for: arch) {
            live[name] = all[name]
        }
        let summary = try LayerHealth.summarizeLiveState(arch: arch, tensors: live)
        XCTAssertEqual(summary.scope, .batchNormStateOnly)
        XCTAssertNil(summary.squeezeExcitationFC1)
        XCTAssertNil(summary.valueFC1)
        XCTAssertNil(summary.largestMagnitudeTensors)
        XCTAssertEqual(summary.velocityNotIncludedReason, LayerHealth.liveVelocityNotReadReason)
        XCTAssertEqual(summary.deadChannelCount, 0)
        XCTAssertEqual(summary.alwaysOnChannelCount, 0)
        XCTAssertEqual(summary.examinedTensorCount, live.count)
        XCTAssertFalse(summary.compactLine().contains("seZeroVel"))

        var missing = live
        missing.removeValue(forKey: "value.bn.weight")
        XCTAssertThrowsError(try LayerHealth.summarizeLiveState(arch: arch, tensors: missing)) { error in
            XCTAssertEqual(error as? LayerHealthError, .missingTensor("value.bn.weight"))
        }
        var unknown = live
        unknown["not.a.tensor"] = [0]
        XCTAssertThrowsError(try LayerHealth.summarizeLiveState(arch: arch, tensors: unknown)) { error in
            XCTAssertEqual(error as? LayerHealthError, .unknownTensor("not.a.tensor"))
        }
        var wrongSize = live
        wrongSize["value.bn.weight"] = [1]
        XCTAssertThrowsError(try LayerHealth.summarizeLiveState(arch: arch, tensors: wrongSize))
    }

    func testTrainerStateCountMismatchThrows() {
        let arch = tinyArch()
        XCTAssertThrowsError(try LayerHealth.summarizeTrainerState(arch: arch, trainerWeights: syntheticWeights(arch)))
    }

    // MARK: - CLI recorder

    func testRecorderWritesLayerHealthRecords() throws {
        let arch = tinyArch()
        let summary = try LayerHealth.summarizeTrainerState(
            arch: arch, trainerWeights: syntheticWeights(arch) + syntheticVelocity(arch))
        let recorder = CliTrainingRecorder()
        let empty = try JSONSerialization.jsonObject(with: recorder.encodedJSONData(totalTrainingSeconds: 1)) as? [String: Any]
        XCTAssertEqual((empty?["layer_health"] as? [Any])?.count, 0)
        recorder.appendLayerHealth(CliTrainingRecorder.LayerHealthRecord(
            step: 100, trainerStep: 1100, context: "replay-autosave", summary: summary))
        let json = try JSONSerialization.jsonObject(with: recorder.encodedJSONData(totalTrainingSeconds: 1)) as? [String: Any]
        let records = try XCTUnwrap(json?["layer_health"] as? [[String: Any]])
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records[0]["step"] as? Int, 100)
        XCTAssertEqual(records[0]["trainer_step"] as? Int, 1100)
        XCTAssertEqual(records[0]["context"] as? String, "replay-autosave")
        let encodedSummary = try XCTUnwrap(records[0]["summary"] as? [String: Any])
        XCTAssertEqual(encodedSummary["scope"] as? String, "all_tensors")
        XCTAssertNotNil(encodedSummary["batch_norm_sites"])
    }

    // MARK: - Trainer live readback (tiny GPU wiring)

    /// The live readback must return exactly what the trainer would persist
    /// at those names — the fp32 masters under bf16, the working variables
    /// under fp32 — before and after a step, with the matching step count.
    func testLiveReadbackMatchesTheExportedTrainerState() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        for dtype in [ComputeDataType.float32, .bFloat16] {
            let arch = tinyArch(dtype: dtype)
            let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), momentumCoeff: 0.9, lrWarmupSteps: 0, arch: arch, initialization: .seeded(initSeed: 1))
            // The synthetic-data `trainStep(batchSize:)` deliberately does not
            // advance `completedTrainSteps` (only real-data steps count, so
            // random-label smoke steps can't consume LR warmup); the readback
            // must report the trainer's own counter either way.
            var weightsBeforeStep: [[Float]] = []
            for expectedSteps in [0, 1] {
                if expectedSteps == 1 {
                    weightsBeforeStep = try await trainer.exportTrainerWeights()
                    _ = try await trainer.trainStep(batchSize: 16)
                    let weightsAfterStep = try await trainer.exportTrainerWeights()
                    XCTAssertNotEqual(weightsAfterStep, weightsBeforeStep,
                                      "\(dtype): the step must change the trainer state the readback is compared with")
                }
                let live = try await trainer.readLayerHealthLiveState()
                XCTAssertEqual(live.completedTrainSteps, trainer.completedTrainSteps, "\(dtype)")
                XCTAssertEqual(Set(live.tensors.keys), Set(LayerHealth.liveStateTensorNames(for: arch)), "\(dtype)")
                let exported = try await trainer.exportTrainerWeights()
                let plan = arch.weightTensorPlan()
                for (index, spec) in plan.enumerated() {
                    guard let values = live.tensors[spec.name] else { continue }
                    XCTAssertEqual(values, exported[index], "\(dtype) step \(expectedSteps): \(spec.name)")
                }
                let summary = try LayerHealth.summarizeLiveState(arch: arch, tensors: live.tensors)
                XCTAssertEqual(summary.nonFiniteValueCount, 0, "\(dtype)")
                if expectedSteps == 0 {
                    XCTAssertEqual(summary.deadChannelCount + summary.mostlyOffChannelCount + summary.alwaysOnChannelCount, 0,
                                   "\(dtype): γ = 1, β = 0 at init")
                    XCTAssertEqual(summary.reZero.first?.alpha ?? .nan, 0.5, accuracy: 0, "\(dtype): α at init")
                }
                let line = LayerHealthLog.liveLine(summary: summary, trainerStep: live.completedTrainSteps)
                XCTAssertTrue(line.hasPrefix("[LAYER-HEALTH] live trainerStep=\(live.completedTrainSteps) scope=batch_norm_state_only"), line)
            }
        }
    }
}
