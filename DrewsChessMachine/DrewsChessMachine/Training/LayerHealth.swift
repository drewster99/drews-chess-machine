import Foundation

// MARK: - Layer health
//
// An early warning for the layer-level failure modes measured on 2026-10-01
// (documentation/research/policy-head-2026-10-01/REPORT.md and
// experiments/20260929-se-style-ab/TENSOR-STATS.md):
//
//  (a) Dead ReLU units. A unit whose pre-activation is negative for every
//      input outputs 0 and gets exactly zero gradient, so it can never
//      recover. Measured in the SE excitation FC1 bottlenecks, where a large
//      share of some blocks' units had died. Two complementary signals catch
//      it: for a BN-fed ReLU channel, β/|γ| far below zero; for an FC hidden
//      unit (no BN in front of it), the unit's momentum velocity reaching
//      exactly zero — momentum decays geometrically while the gradient is
//      zero and lands on exact fp32 zero only after a long run of
//      consecutive zero-gradient steps, so an exact zero is strong evidence.
//  (b) Always-on units. A BN-fed ReLU channel whose β/|γ| is far above zero
//      is never zeroed: ReLU does nothing to it, so it is a linear
//      pass-through riding on a large constant, and the next layer turns that
//      constant into a hidden bias. In the policy pre-block these channels
//      carried a large part of the bf16 shared policy offset that softmax
//      cannot see.
//  (c) Runaway BN running variance. One channel's running variance far out
//      of scale with its layer's median says the conv feeding it grew one
//      row out of scale with its siblings (BN normalizes it away in the
//      forward, so nothing else shows it).
//  (d) ReZero α saturating at its tanh cap. The effective branch scale is
//      `C·tanh(α/C)`; reported as a fraction of C.
//  (e) NaN / Inf anywhere in the examined tensors.
//  (f) Head shared offsets are NOT recomputed here: the live monitors are
//      `pLogitMean` / `vLogitMean` on `[STATS]` / `[REPLAY]` / `[VS-UCI]`, and
//      the weight-level measure is `NumericsAudit`'s shared-offset report.
//
// WHY β/|γ|: after BatchNorm, channel k is modelled as normal with mean β[k]
// and spread |γ[k]|, so β/|γ| is how many spreads its typical value sits
// above zero, and the share of values ReLU zeroes is Φ(−β/|γ|). The
// thresholds below are the report's labels (dead, mostly off, always on).
// The model holds exactly only while the running statistics track the batch
// statistics, but it needs nothing beyond γ and β, which is what makes the
// live tier cheap. The classification is only meaningful where the
// activation has a hard zero/linear split — `relu` and `leaky_relu` (for
// which "dead" means "always on the small negative-slope side"). SiLU and
// GELU have no such split, so their sites report the classification as
// explicitly not applicable rather than as zero counts that would read as
// "healthy".
//
// SITE ENUMERATION is derived from the architecture alone and mirrors the
// graph builder (`ChessNetwork`): every BatchNorm in build order, each tagged
// with the activation that consumes its output directly (or none). The names
// are `weightTensorPlan()` module paths, so the BN tensors resolve against the
// same single source of truth the builder, the safetensors writer and the
// loader use. `LayerHealthTests` cross-checks the enumeration against
// `weightTensorPlan()` and against the built graph's op wiring.
//
// Two tiers consume this module:
//  - live (`Scope.batchNormStateOnly`): the BN γ/β/running stats plus the
//    ReZero α scalars only — a few KB read on the trainer's own serial queue
//    between steps (`ChessTrainer.readLayerHealthLiveState`).
//  - checkpoint (`Scope.allTensors`): every tensor of an already-exported
//    trainer state, including optimizer velocity when present, so the
//    velocity-based dead-unit checks and the full NaN sweep run with no extra
//    GPU reads.

/// Pure layer-health analysis: site enumeration, thresholds, and the
/// summary computation. No GPU, no files, no logging.
enum LayerHealth {

    // MARK: - Thresholds

    /// β/|γ| below which a ReLU-family BN channel counts as dead: ReLU zeroes
    /// all but a sliver of its values, so it is essentially never on.
    static let deadBetaOverAbsGamma: Double = -3

    /// β/|γ| below which a ReLU-family BN channel that is not dead counts as
    /// mostly off: ReLU zeroes the large majority of its values. Disjoint
    /// from dead — a channel is counted in exactly one of dead / mostly off.
    static let mostlyOffBetaOverAbsGamma: Double = -2

    /// β/|γ| above which a ReLU-family BN channel counts as always on: ReLU
    /// almost never zeroes it, so the channel is a linear pass-through
    /// carrying a constant.
    static let alwaysOnBetaOverAbsGamma: Double = 3

    /// An FC hidden unit whose weight-velocity column norm is nonzero but
    /// below this fraction of its layer's reference unit norm counts as a
    /// low-velocity unit — on its way to the exact-zero velocity of a dead
    /// unit, or barely trained (e.g. a leaky-ReLU unit living on the 1%
    /// negative-slope trickle).
    static let lowVelocityFractionOfReferenceUnitNorm: Double = 0.05

    /// The percentile of a layer's unit velocity norms used as the
    /// reference for "low". Not the median: in a bimodal layer where more
    /// than half the units are weak (the SE bottlenecks measured on
    /// 2026-10-02), the median unit is itself a weak one and nothing looks
    /// low against it. The 90th percentile tracks the active units.
    static let lowVelocityReferencePercentile: Double = 90

    /// A ReZero block whose |effective α| is at least this fraction of its
    /// tanh cap counts as saturated.
    static let reZeroSaturatedFractionOfCap: Double = 0.99

    /// How many tensors the largest-|value| list reports.
    static let largestMagnitudeTensorReportCount = 5

    /// How many non-finite tensor names a summary keeps.
    static let nonFiniteTensorNameReportCount = 5

    /// How many zero-velocity unit indexes a summary keeps per layer.
    static let zeroVelocityUnitIndexReportCount = 16

    /// Why the live tier carries no velocity checks.
    static let liveVelocityNotReadReason =
        "the live readback reads batch-norm state and ReZero α only"

    // MARK: - Site descriptions

    /// One BatchNorm layer, named by its `weightTensorPlan()` module path.
    struct BatchNormSite: Sendable, Equatable {
        /// Plan module path, e.g. `blocks.2.bn1` or `policy.pre_bn`.
        let name: String
        let channels: Int
        /// The activation that consumes this BN's output directly, or nil
        /// when the output feeds something else first (a pre-activation
        /// stem's BN feeds block 0's BN; a post-activation block's BN2 feeds
        /// the SE / skip merge).
        let activation: ActivationFunction?

        var gammaTensorName: String { "\(name).weight" }
        var betaTensorName: String { "\(name).bias" }
        var runningMeanTensorName: String { "\(name).running_mean" }
        var runningVarianceTensorName: String { "\(name).running_var" }
        var tensorNames: [String] {
            [gammaTensorName, betaTensorName, runningMeanTensorName, runningVarianceTensorName]
        }
    }

    /// A fully-connected hidden layer followed by an activation, whose
    /// weight (and velocity) is stored in the graph's native `[inputs,
    /// units]` row-major layout: unit `j`'s weights are column `j`, element
    /// `i * unitCount + j`.
    struct HiddenUnitLayer: Sendable, Equatable {
        let weightTensorName: String
        /// The residual block this layer belongs to; nil for a head layer.
        let blockIndex: Int?
        let inputCount: Int
        let unitCount: Int
        let activation: ActivationFunction
    }

    /// One block's ReZero scalar.
    struct ReZeroSite: Sendable, Equatable {
        let blockIndex: Int
        let alphaTensorName: String
        /// The forward's soft-bound asymptote `C` in `C·tanh(α/C)`.
        let cap: Double
    }

    /// Every BatchNorm in graph build order, mirroring `ChessNetwork`:
    /// stem BN (activated only when the first block is post-activation) →
    /// per block (pre: BN1→act, BN2→act; post: BN1→act, BN2 unactivated) →
    /// tower-end BN→act (pre-activation tail only) → feature-skip compress
    /// BN→act (compress fusion with a routed head only) → policy pre-block
    /// BN→act (`intermediate_conv` / `fc_bottleneck`) → value BN→act. Block
    /// sites use the block's own activation; every architecture-level site
    /// is tagged with its own field (`stemActivation`, `towerEndActivation`,
    /// `featureSkipActivation`, `policyHeadActivation`,
    /// `valueHeadConvActivation`), read only where the site exists — so
    /// `does_not_apply`, the value an absent site holds, never becomes a tag
    /// (an absent stem activation is `nil`: its BN feeds no activation).
    static func batchNormSites(for arch: NetworkArchitecture) -> [BatchNormSite] {
        var sites: [BatchNormSite] = []
        sites.append(BatchNormSite(
            name: "stem.bn",
            channels: arch.stemOutputChannels,
            activation: arch.hasStemActivation ? arch.stemActivation : nil))

        // Thread the incoming width exactly as `weightTensorPlan` does: a
        // pre-activation BN1 normalizes the block's INPUT (including any
        // final-block feature-skip concat), a post-activation BN1 the conv1
        // output.
        var incomingChannels = arch.stemOutputChannels
        for (blockIndex, spec) in arch.expandedBlocks.enumerated() {
            let blockInputChannels = incomingChannels + arch.blockSkipExtraInputChannels(blockIndex: blockIndex)
            let prefix = "blocks.\(blockIndex)"
            switch spec.activationStyle {
            case .pre:
                sites.append(BatchNormSite(
                    name: "\(prefix).bn1", channels: blockInputChannels, activation: spec.activationFunction))
                sites.append(BatchNormSite(
                    name: "\(prefix).bn2", channels: spec.channels, activation: spec.activationFunction))
            case .post:
                sites.append(BatchNormSite(
                    name: "\(prefix).bn1", channels: spec.channels, activation: spec.activationFunction))
                // BN2 feeds the SE and the skip merge; an `activation_gated`
                // merge activates the SUM, never this BN's output alone.
                sites.append(BatchNormSite(
                    name: "\(prefix).bn2", channels: spec.channels, activation: nil))
            }
            incomingChannels = spec.channels
        }

        if arch.hasTowerEndBN {
            sites.append(BatchNormSite(
                name: "tower_final_bn", channels: arch.towerOutputChannels, activation: arch.towerEndActivation))
        }
        if arch.featureSkipUsesCompressNode {
            sites.append(BatchNormSite(
                name: "feature_skip.bn", channels: arch.towerOutputChannels, activation: arch.featureSkipActivation))
        }
        switch arch.policyHeadStyle {
        case .simpleConv:
            break
        case .intermediateConv, .fcBottleneck:
            sites.append(BatchNormSite(
                name: "policy.pre_bn", channels: arch.policyPreConvChannels, activation: arch.policyHeadActivation))
        }
        sites.append(BatchNormSite(
            name: "value.bn", channels: arch.valueHeadConvChannels, activation: arch.valueHeadConvActivation))
        return sites
    }

    /// Every SE excitation FC1 (`C → C/r`, activated by the group's
    /// `seActivation`), one per block whose SE style is not `none`.
    static func squeezeExcitationFC1Layers(for arch: NetworkArchitecture) -> [HiddenUnitLayer] {
        var layers: [HiddenUnitLayer] = []
        for (blockIndex, spec) in arch.expandedBlocks.enumerated() {
            let module: String
            switch spec.seStyle {
            case .none:
                continue
            case .attenuateOnly:
                module = "se_attenuate"
            case .scaleAndBias:
                module = "se_scalebias"
            }
            layers.append(HiddenUnitLayer(
                weightTensorName: "blocks.\(blockIndex).\(module).fc1.weight",
                blockIndex: blockIndex,
                inputCount: spec.channels,
                unitCount: spec.channels / spec.seReductionRatio,
                activation: spec.seActivation))
        }
        return layers
    }

    /// The value head's FC1 (`flatten(conv channels × 64) → hidden`),
    /// activated by the model's `valueHeadFC1HiddenActivation` (a site every
    /// model has). The activation only labels the velocity row: zero-velocity
    /// counting measures the weight velocity whatever the function.
    static func valueFC1Layer(for arch: NetworkArchitecture) -> HiddenUnitLayer {
        HiddenUnitLayer(
            weightTensorName: "value.fc1.weight",
            blockIndex: nil,
            inputCount: arch.boardSize * arch.boardSize * arch.valueHeadConvChannels,
            unitCount: arch.valueHeadHiddenUnits,
            activation: arch.valueHeadFC1HiddenActivation)
    }

    /// Every block with ReZero, in block order.
    static func reZeroSites(for arch: NetworkArchitecture) -> [ReZeroSite] {
        var sites: [ReZeroSite] = []
        for (blockIndex, spec) in arch.expandedBlocks.enumerated() where spec.useRezero {
            sites.append(ReZeroSite(
                blockIndex: blockIndex,
                alphaTensorName: "blocks.\(blockIndex).rezero_alpha",
                cap: spec.rezeroTanhCeiling))
        }
        return sites
    }

    /// The plan tensor names the live tier reads: each BN site's γ, β,
    /// running mean and running variance, then each ReZero α.
    static func liveStateTensorNames(for arch: NetworkArchitecture) -> [String] {
        batchNormSites(for: arch).flatMap(\.tensorNames) + reZeroSites(for: arch).map(\.alphaTensorName)
    }

    /// The BN classification that applies to a site's activation. An
    /// exhaustive switch, so a new activation function must decide here.
    /// `does_not_apply` is unreachable: every site reads its activation
    /// field only where the site exists (`batchNormSites`), and `validate()`
    /// refuses `does_not_apply` at every existing site.
    static func classification(for activation: ActivationFunction?) -> LayerHealthSummary.ActivationClassification {
        guard let activation else { return .notApplicableNoActivation }
        switch activation {
        case .relu, .leakyRelu:
            return .classified
        case .silu, .gelu:
            return .notApplicableSmoothActivation
        case .doesNotApply:
            preconditionFailure("LayerHealth: 'does_not_apply' tags a batch-norm site that exists. "
                + "validate() refuses it at every existing site, so this is a defect.")
        }
    }

    // MARK: - Velocity input

    /// Where a summary's optimizer velocity comes from.
    enum VelocitySource: Sendable {
        /// One velocity tensor per trainable, in trainable order — the tail
        /// of a trainer-state export (`ChessTrainer.exportTrainerWeights`).
        case trainerVelocity([[Float]])
        /// No velocity; the reason is recorded in the summary.
        case unavailable(reason: String)
    }

    // MARK: - Entry points

    /// Summarize a full trainer-state export: the plan's tensors followed by
    /// one velocity tensor per trainable (`TrainerResumeSnapshot.trainerWeights`).
    static func summarizeTrainerState(
        arch: NetworkArchitecture,
        trainerWeights: [[Float]]
    ) throws -> LayerHealthSummary {
        let planCount = arch.weightTensorPlan().count
        let trainableCount = arch.trainableTensorPlan().count
        guard trainerWeights.count == planCount + trainableCount else {
            throw LayerHealthError.tensorCountMismatch(
                what: "trainer state (plan tensors + one velocity per trainable)",
                expected: planCount + trainableCount,
                got: trainerWeights.count)
        }
        return try summarizePlanAligned(
            arch: arch,
            baseWeights: Array(trainerWeights.prefix(planCount)),
            velocity: .trainerVelocity(Array(trainerWeights.suffix(trainableCount))))
    }

    /// Summarize plan-aligned base weights (the layout `exportWeights()`
    /// produces) over every tensor.
    static func summarizePlanAligned(
        arch: NetworkArchitecture,
        baseWeights: [[Float]],
        velocity: VelocitySource
    ) throws -> LayerHealthSummary {
        let plan = arch.weightTensorPlan()
        guard baseWeights.count == plan.count else {
            throw LayerHealthError.tensorCountMismatch(
                what: "plan-aligned weights", expected: plan.count, got: baseWeights.count)
        }
        var tensors: [String: [Float]] = [:]
        tensors.reserveCapacity(plan.count)
        for (spec, values) in zip(plan, baseWeights) {
            tensors[spec.name] = values
        }
        return try summarize(arch: arch, tensors: tensors, velocity: velocity, scope: .allTensors)
    }

    /// Summarize the live tier's readback (`liveStateTensorNames`).
    static func summarizeLiveState(
        arch: NetworkArchitecture,
        tensors: [String: [Float]]
    ) throws -> LayerHealthSummary {
        try summarize(
            arch: arch,
            tensors: tensors,
            velocity: .unavailable(reason: liveVelocityNotReadReason),
            scope: .batchNormStateOnly)
    }

    /// The core computation. `tensors` maps plan names to values and must
    /// hold at least every BN site's four tensors and every ReZero α; with
    /// `.allTensors` scope it must hold the whole plan. Throws on a missing
    /// tensor, a wrongly-sized one, or a name the plan does not know.
    static func summarize(
        arch: NetworkArchitecture,
        tensors: [String: [Float]],
        velocity: VelocitySource,
        scope: LayerHealthSummary.Scope
    ) throws -> LayerHealthSummary {
        let plan = arch.weightTensorPlan()
        let planNames = plan.map(\.name)
        let knownNames = Set(planNames)
        if let unknown = tensors.keys.sorted().first(where: { !knownNames.contains($0) }) {
            throw LayerHealthError.unknownTensor(unknown)
        }
        if scope == .allTensors, let missing = planNames.first(where: { tensors[$0] == nil }) {
            throw LayerHealthError.missingTensor(missing)
        }
        let expectedCountByName = Dictionary(uniqueKeysWithValues: plan.map { ($0.name, $0.elementCount) })
        for (name, values) in tensors {
            guard let expected = expectedCountByName[name] else {
                throw LayerHealthError.unknownTensor(name)
            }
            guard values.count == expected else {
                throw LayerHealthError.tensorSizeMismatch(name: name, expected: expected, got: values.count)
            }
        }

        // Velocity by the trainable's plan name; reported under its
        // persisted `opt.<name>.velocity` name.
        let trainables = arch.trainableTensorPlan()
        let velocityByWeightName: [String: [Float]]?
        let velocityNotIncludedReason: String?
        switch velocity {
        case .trainerVelocity(let values):
            guard values.count == trainables.count else {
                throw LayerHealthError.tensorCountMismatch(
                    what: "velocity tensors (one per trainable)", expected: trainables.count, got: values.count)
            }
            var byName: [String: [Float]] = [:]
            byName.reserveCapacity(trainables.count)
            for (spec, v) in zip(trainables, values) {
                guard v.count == spec.elementCount else {
                    throw LayerHealthError.tensorSizeMismatch(
                        name: SafetensorsModelIO.velocityTensorName(forTrainable: spec.name),
                        expected: spec.elementCount, got: v.count)
                }
                byName[spec.name] = v
            }
            velocityByWeightName = byName
            velocityNotIncludedReason = nil
        case .unavailable(let reason):
            velocityByWeightName = nil
            velocityNotIncludedReason = reason
        }

        func require(_ name: String) throws -> [Float] {
            guard let values = tensors[name] else { throw LayerHealthError.missingTensor(name) }
            return values
        }

        var batchNormHealth: [LayerHealthSummary.BatchNormSiteHealth] = []
        for site in batchNormSites(for: arch) {
            let gamma = try require(site.gammaTensorName)
            let beta = try require(site.betaTensorName)
            let runningVariance = try require(site.runningVarianceTensorName)
            // The plan sizes were checked above; this pins the site
            // enumeration's own channel count to them too.
            for (name, values) in [(site.gammaTensorName, gamma), (site.betaTensorName, beta),
                                   (site.runningVarianceTensorName, runningVariance)]
            where values.count != site.channels {
                throw LayerHealthError.tensorSizeMismatch(name: name, expected: site.channels, got: values.count)
            }
            batchNormHealth.append(batchNormSiteHealth(
                site: site, gamma: gamma, beta: beta, runningVariance: runningVariance))
        }

        var reZeroHealth: [LayerHealthSummary.ReZeroHealth] = []
        for site in reZeroSites(for: arch) {
            let values = try require(site.alphaTensorName)
            guard let alpha = values.first, values.count == 1 else {
                throw LayerHealthError.tensorSizeMismatch(name: site.alphaTensorName, expected: 1, got: values.count)
            }
            reZeroHealth.append(reZeroHealthEntry(site: site, alpha: alpha))
        }

        var squeezeExcitation: [LayerHealthSummary.HiddenUnitVelocityHealth]?
        var valueFC1: LayerHealthSummary.HiddenUnitVelocityHealth?
        if let velocityByWeightName {
            var seHealth: [LayerHealthSummary.HiddenUnitVelocityHealth] = []
            for layer in squeezeExcitationFC1Layers(for: arch) {
                guard let v = velocityByWeightName[layer.weightTensorName] else {
                    throw LayerHealthError.missingTensor(
                        SafetensorsModelIO.velocityTensorName(forTrainable: layer.weightTensorName))
                }
                let layerHealth = try hiddenUnitVelocityHealth(layer: layer, velocity: v)
                seHealth.append(layerHealth)
            }
            squeezeExcitation = seHealth
            let valueLayer = valueFC1Layer(for: arch)
            guard let v = velocityByWeightName[valueLayer.weightTensorName] else {
                throw LayerHealthError.missingTensor(
                    SafetensorsModelIO.velocityTensorName(forTrainable: valueLayer.weightTensorName))
            }
            valueFC1 = try hiddenUnitVelocityHealth(layer: valueLayer, velocity: v)
        }

        // NaN / Inf sweep and per-tensor max |value|, in plan order then
        // velocity in trainable order, so reported names are deterministic.
        var examinedTensorCount = 0
        var examinedValueCount = 0
        var nonFiniteValueCount = 0
        var nonFiniteTensorCount = 0
        var nonFiniteTensorNames: [String] = []
        var magnitudes: [LayerHealthSummary.TensorMagnitude] = []
        func sweep(_ name: String, _ values: [Float], recordMagnitude: Bool) {
            let scan = scanFiniteness(values)
            examinedTensorCount += 1
            examinedValueCount += values.count
            if scan.nonFiniteCount > 0 {
                nonFiniteValueCount += scan.nonFiniteCount
                nonFiniteTensorCount += 1
                if nonFiniteTensorNames.count < nonFiniteTensorNameReportCount {
                    nonFiniteTensorNames.append(name)
                }
            }
            if recordMagnitude, let maxAbs = scan.maxAbsFinite {
                magnitudes.append(LayerHealthSummary.TensorMagnitude(name: name, maxAbs: maxAbs))
            }
        }
        for name in planNames {
            guard let values = tensors[name] else { continue }
            sweep(name, values, recordMagnitude: scope == .allTensors)
        }
        if let velocityByWeightName {
            for spec in trainables {
                guard let values = velocityByWeightName[spec.name] else { continue }
                sweep(SafetensorsModelIO.velocityTensorName(forTrainable: spec.name), values, recordMagnitude: false)
            }
        }
        let largest: [LayerHealthSummary.TensorMagnitude]? = scope == .allTensors
            ? Array(magnitudes.sorted { $0.maxAbs > $1.maxAbs }.prefix(largestMagnitudeTensorReportCount))
            : nil

        return LayerHealthSummary(
            scope: scope,
            batchNormSites: batchNormHealth,
            squeezeExcitationFC1: squeezeExcitation,
            valueFC1: valueFC1,
            velocityNotIncludedReason: velocityNotIncludedReason,
            reZero: reZeroHealth,
            examinedTensorCount: examinedTensorCount,
            examinedValueCount: examinedValueCount,
            nonFiniteValueCount: nonFiniteValueCount,
            nonFiniteTensorCount: nonFiniteTensorCount,
            nonFiniteTensorNames: nonFiniteTensorNames,
            largestMagnitudeTensors: largest)
    }

    // MARK: - Per-site computations

    /// β/|γ| classification and running-variance spread for one BN site.
    ///
    /// A channel with a non-finite γ or β is counted in
    /// `nonFiniteChannelCount` and left out of everything else. A channel
    /// with γ = 0 outputs the constant β, so (for a classified site) it is
    /// always on when β > 0 and dead otherwise; it is counted in
    /// `zeroGammaChannelCount` and left out of the β/|γ| extremes, which
    /// would be infinite. Running variance: max and median over its finite
    /// values; max/median is nil when the median is not positive.
    static func batchNormSiteHealth(
        site: BatchNormSite,
        gamma: [Float],
        beta: [Float],
        runningVariance: [Float]
    ) -> LayerHealthSummary.BatchNormSiteHealth {
        let siteClassification = Self.classification(for: site.activation)
        let classifies = siteClassification == .classified
        var dead = 0
        var mostlyOff = 0
        var alwaysOn = 0
        var zeroGamma = 0
        var nonFiniteChannels = 0
        var minRatio: (value: Double, channel: Int)?
        var maxRatio: (value: Double, channel: Int)?
        for channel in 0..<site.channels {
            let g = gamma[channel]
            let b = beta[channel]
            guard g.isFinite, b.isFinite else {
                nonFiniteChannels += 1
                continue
            }
            if g == 0 {
                zeroGamma += 1
                if classifies {
                    if b > 0 { alwaysOn += 1 } else { dead += 1 }
                }
                continue
            }
            let ratio = Double(b) / abs(Double(g))
            if minRatio.map({ ratio < $0.value }) ?? true { minRatio = (ratio, channel) }
            if maxRatio.map({ ratio > $0.value }) ?? true { maxRatio = (ratio, channel) }
            guard classifies else { continue }
            if ratio < deadBetaOverAbsGamma {
                dead += 1
            } else if ratio < mostlyOffBetaOverAbsGamma {
                mostlyOff += 1
            } else if ratio > alwaysOnBetaOverAbsGamma {
                alwaysOn += 1
            }
        }

        var finiteVariances: [(value: Double, channel: Int)] = []
        finiteVariances.reserveCapacity(site.channels)
        var nonFiniteVariances = 0
        for channel in 0..<site.channels {
            let v = runningVariance[channel]
            if v.isFinite {
                finiteVariances.append((Double(v), channel))
            } else {
                nonFiniteVariances += 1
            }
        }
        let varianceMax = finiteVariances.max { $0.value < $1.value }
        let sortedVariances = finiteVariances.map(\.value).sorted()
        let varianceMedian: Double? = sortedVariances.isEmpty
            ? nil
            : NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: sortedVariances)
        let maxOverMedian: Double?
        if let varianceMax, let varianceMedian, varianceMedian > 0 {
            maxOverMedian = varianceMax.value / varianceMedian
        } else {
            maxOverMedian = nil
        }

        return LayerHealthSummary.BatchNormSiteHealth(
            site: site.name,
            activation: site.activation,
            channelCount: site.channels,
            classification: siteClassification,
            deadChannelCount: classifies ? dead : nil,
            mostlyOffChannelCount: classifies ? mostlyOff : nil,
            alwaysOnChannelCount: classifies ? alwaysOn : nil,
            zeroGammaChannelCount: zeroGamma,
            nonFiniteChannelCount: nonFiniteChannels,
            minBetaOverAbsGamma: minRatio?.value,
            minBetaOverAbsGammaChannel: minRatio?.channel,
            maxBetaOverAbsGamma: maxRatio?.value,
            maxBetaOverAbsGammaChannel: maxRatio?.channel,
            runningVarianceMax: varianceMax?.value,
            runningVarianceMaxChannel: varianceMax?.channel,
            runningVarianceMedian: varianceMedian,
            runningVarianceMaxOverMedian: maxOverMedian,
            nonFiniteRunningVarianceCount: nonFiniteVariances)
    }

    /// Zero- and low-velocity units of one FC hidden layer. `velocity` is in
    /// the weight's native `[inputs, units]` layout, so unit `j` is column
    /// `j`. A unit with any non-finite velocity entry is counted in
    /// `nonFiniteUnitCount` and left out of the norms. The median and the
    /// reference percentile cover every finite unit, zero ones included;
    /// "low" counts nonzero units below
    /// `lowVelocityFractionOfReferenceUnitNorm` × the reference
    /// (`lowVelocityReferencePercentile`) norm.
    static func hiddenUnitVelocityHealth(
        layer: HiddenUnitLayer,
        velocity: [Float]
    ) throws -> LayerHealthSummary.HiddenUnitVelocityHealth {
        let units = layer.unitCount
        let inputs = layer.inputCount
        guard velocity.count == inputs * units else {
            throw LayerHealthError.tensorSizeMismatch(
                name: SafetensorsModelIO.velocityTensorName(forTrainable: layer.weightTensorName),
                expected: inputs * units, got: velocity.count)
        }
        // Row-major accumulation: one pass over contiguous memory.
        var sumOfSquares = [Double](repeating: 0, count: units)
        var hasNonzero = [Bool](repeating: false, count: units)
        var hasNonFinite = [Bool](repeating: false, count: units)
        velocity.withUnsafeBufferPointer { buffer in
            for row in 0..<inputs {
                let rowStart = row * units
                for unit in 0..<units {
                    let v = buffer[rowStart + unit]
                    guard v.isFinite else {
                        hasNonFinite[unit] = true
                        continue
                    }
                    if v != 0 {
                        hasNonzero[unit] = true
                        let d = Double(v)
                        sumOfSquares[unit] += d * d
                    }
                }
            }
        }

        var zeroUnits: [Int] = []
        var nonFiniteUnits = 0
        var finiteNorms: [Double] = []
        finiteNorms.reserveCapacity(units)
        for unit in 0..<units {
            if hasNonFinite[unit] {
                nonFiniteUnits += 1
                continue
            }
            finiteNorms.append(sumOfSquares[unit].squareRoot())
            if !hasNonzero[unit] { zeroUnits.append(unit) }
        }
        let sortedNorms = finiteNorms.sorted()
        let median: Double? = sortedNorms.isEmpty
            ? nil
            : NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: sortedNorms)
        let reference: Double? = sortedNorms.isEmpty
            ? nil
            : NetworkWeightAnalyzer.percentile(p: lowVelocityReferencePercentile, sortedAscending: sortedNorms)
        let threshold = reference.map { $0 * lowVelocityFractionOfReferenceUnitNorm }
        var lowUnits = 0
        if let threshold {
            for unit in 0..<units
            where !hasNonFinite[unit] && hasNonzero[unit] && sumOfSquares[unit].squareRoot() < threshold {
                lowUnits += 1
            }
        }

        return LayerHealthSummary.HiddenUnitVelocityHealth(
            layer: layer.weightTensorName,
            blockIndex: layer.blockIndex,
            activation: layer.activation,
            inputCount: inputs,
            unitCount: units,
            zeroVelocityUnitCount: zeroUnits.count,
            zeroVelocityUnits: Array(zeroUnits.prefix(zeroVelocityUnitIndexReportCount)),
            lowVelocityUnitCount: lowUnits,
            nonFiniteUnitCount: nonFiniteUnits,
            medianUnitVelocityNorm: median,
            referenceUnitVelocityNorm: reference,
            lowVelocityThreshold: threshold)
    }

    /// One block's ReZero α against its cap. A non-finite α leaves every
    /// derived value nil (it is also counted by the NaN sweep).
    static func reZeroHealthEntry(site: ReZeroSite, alpha: Float) -> LayerHealthSummary.ReZeroHealth {
        guard alpha.isFinite, site.cap > 0 else {
            return LayerHealthSummary.ReZeroHealth(
                blockIndex: site.blockIndex, alpha: nil, cap: site.cap,
                effectiveAlpha: nil, fractionOfCap: nil, saturated: nil)
        }
        let rawAlpha = Double(alpha)
        let fraction = tanh(rawAlpha / site.cap)
        return LayerHealthSummary.ReZeroHealth(
            blockIndex: site.blockIndex,
            alpha: rawAlpha,
            cap: site.cap,
            effectiveAlpha: site.cap * fraction,
            fractionOfCap: fraction,
            saturated: abs(fraction) >= reZeroSaturatedFractionOfCap)
    }

    /// Count of non-finite values and the largest finite |value|.
    static func scanFiniteness(_ values: [Float]) -> (nonFiniteCount: Int, maxAbsFinite: Double?) {
        var nonFinite = 0
        var maxAbs: Float = 0
        var sawFinite = false
        values.withUnsafeBufferPointer { buffer in
            for v in buffer {
                if v.isFinite {
                    sawFinite = true
                    let a = abs(v)
                    if a > maxAbs { maxAbs = a }
                } else {
                    nonFinite += 1
                }
            }
        }
        return (nonFinite, sawFinite ? Double(maxAbs) : nil)
    }
}

// MARK: - Summary

/// The result of one layer-health pass. Every `Double` is finite by
/// construction (non-finite inputs are counted, never propagated), so the
/// summary always JSON-encodes.
struct LayerHealthSummary: Codable, Sendable, Equatable {

    /// Which tensors the summary saw.
    enum Scope: String, Codable, Sendable {
        /// BN γ/β/running stats and ReZero α only (the live tier). No
        /// velocity checks and no largest-|value| list.
        case batchNormStateOnly = "batch_norm_state_only"
        /// Every model tensor, plus optimizer velocity when available.
        case allTensors = "all_tensors"
    }

    /// Whether the dead / mostly-off / always-on classification applies.
    enum ActivationClassification: String, Codable, Sendable {
        /// ReLU or leaky ReLU consumes the BN output: classified.
        case classified
        /// SiLU or GELU: no hard on/off split, so not classified.
        case notApplicableSmoothActivation = "not_applicable_smooth_activation"
        /// No activation consumes the BN output directly.
        case notApplicableNoActivation = "not_applicable_no_activation"
    }

    struct BatchNormSiteHealth: Codable, Sendable, Equatable {
        let site: String
        let activation: ActivationFunction?
        let channelCount: Int
        let classification: ActivationClassification
        /// nil when `classification` is not `.classified`.
        let deadChannelCount: Int?
        let mostlyOffChannelCount: Int?
        let alwaysOnChannelCount: Int?
        let zeroGammaChannelCount: Int
        let nonFiniteChannelCount: Int
        /// Extremes of β/|γ| over finite channels with γ ≠ 0; nil when there
        /// are none.
        let minBetaOverAbsGamma: Double?
        let minBetaOverAbsGammaChannel: Int?
        let maxBetaOverAbsGamma: Double?
        let maxBetaOverAbsGammaChannel: Int?
        let runningVarianceMax: Double?
        let runningVarianceMaxChannel: Int?
        let runningVarianceMedian: Double?
        /// nil when the median running variance is not positive.
        let runningVarianceMaxOverMedian: Double?
        let nonFiniteRunningVarianceCount: Int

        /// Dead + mostly off + always on; nil when not classified.
        var unhealthyChannelCount: Int? {
            guard let dead = deadChannelCount, let off = mostlyOffChannelCount, let on = alwaysOnChannelCount else {
                return nil
            }
            return dead + off + on
        }

        enum CodingKeys: String, CodingKey {
            case site
            case activation
            case channelCount = "channel_count"
            case classification
            case deadChannelCount = "dead_channel_count"
            case mostlyOffChannelCount = "mostly_off_channel_count"
            case alwaysOnChannelCount = "always_on_channel_count"
            case zeroGammaChannelCount = "zero_gamma_channel_count"
            case nonFiniteChannelCount = "non_finite_channel_count"
            case minBetaOverAbsGamma = "min_beta_over_abs_gamma"
            case minBetaOverAbsGammaChannel = "min_beta_over_abs_gamma_channel"
            case maxBetaOverAbsGamma = "max_beta_over_abs_gamma"
            case maxBetaOverAbsGammaChannel = "max_beta_over_abs_gamma_channel"
            case runningVarianceMax = "running_variance_max"
            case runningVarianceMaxChannel = "running_variance_max_channel"
            case runningVarianceMedian = "running_variance_median"
            case runningVarianceMaxOverMedian = "running_variance_max_over_median"
            case nonFiniteRunningVarianceCount = "non_finite_running_variance_count"
        }
    }

    struct HiddenUnitVelocityHealth: Codable, Sendable, Equatable {
        /// The layer's weight tensor (plan name).
        let layer: String
        let blockIndex: Int?
        let activation: ActivationFunction
        let inputCount: Int
        let unitCount: Int
        /// Units whose every weight-velocity entry is exactly zero.
        let zeroVelocityUnitCount: Int
        /// The first `LayerHealth.zeroVelocityUnitIndexReportCount` of them.
        let zeroVelocityUnits: [Int]
        /// Nonzero units below the low-velocity threshold.
        let lowVelocityUnitCount: Int
        let nonFiniteUnitCount: Int
        /// Median weight-velocity column norm over finite units; nil when
        /// every unit is non-finite.
        let medianUnitVelocityNorm: Double?
        /// The `LayerHealth.lowVelocityReferencePercentile` unit norm the
        /// low-velocity threshold is a fraction of; nil when every unit is
        /// non-finite.
        let referenceUnitVelocityNorm: Double?
        let lowVelocityThreshold: Double?

        enum CodingKeys: String, CodingKey {
            case layer
            case blockIndex = "block_index"
            case activation
            case inputCount = "input_count"
            case unitCount = "unit_count"
            case zeroVelocityUnitCount = "zero_velocity_unit_count"
            case zeroVelocityUnits = "zero_velocity_units"
            case lowVelocityUnitCount = "low_velocity_unit_count"
            case nonFiniteUnitCount = "non_finite_unit_count"
            case medianUnitVelocityNorm = "median_unit_velocity_norm"
            case referenceUnitVelocityNorm = "reference_unit_velocity_norm"
            case lowVelocityThreshold = "low_velocity_threshold"
        }
    }

    struct ReZeroHealth: Codable, Sendable, Equatable {
        let blockIndex: Int
        /// The stored α; nil when non-finite (as are the derived values).
        let alpha: Double?
        let cap: Double
        /// `cap · tanh(α / cap)` — the scale the forward applies.
        let effectiveAlpha: Double?
        /// `tanh(α / cap)` = effective α as a signed fraction of the cap.
        let fractionOfCap: Double?
        let saturated: Bool?

        enum CodingKeys: String, CodingKey {
            case blockIndex = "block_index"
            case alpha
            case cap
            case effectiveAlpha = "effective_alpha"
            case fractionOfCap = "fraction_of_cap"
            case saturated
        }
    }

    struct TensorMagnitude: Codable, Sendable, Equatable {
        let name: String
        let maxAbs: Double

        enum CodingKeys: String, CodingKey {
            case name
            case maxAbs = "max_abs"
        }
    }

    let scope: Scope
    let batchNormSites: [BatchNormSiteHealth]
    /// nil when velocity was not included (see `velocityNotIncludedReason`).
    let squeezeExcitationFC1: [HiddenUnitVelocityHealth]?
    let valueFC1: HiddenUnitVelocityHealth?
    let velocityNotIncludedReason: String?
    let reZero: [ReZeroHealth]
    let examinedTensorCount: Int
    let examinedValueCount: Int
    let nonFiniteValueCount: Int
    let nonFiniteTensorCount: Int
    /// The first `LayerHealth.nonFiniteTensorNameReportCount` tensors with
    /// a non-finite value.
    let nonFiniteTensorNames: [String]
    /// The tensors with the largest finite |value|, largest first; nil for
    /// `.batchNormStateOnly`.
    let largestMagnitudeTensors: [TensorMagnitude]?

    enum CodingKeys: String, CodingKey {
        case scope
        case batchNormSites = "batch_norm_sites"
        case squeezeExcitationFC1 = "squeeze_excitation_fc1"
        case valueFC1 = "value_fc1"
        case velocityNotIncludedReason = "velocity_not_included_reason"
        case reZero = "rezero"
        case examinedTensorCount = "examined_tensor_count"
        case examinedValueCount = "examined_value_count"
        case nonFiniteValueCount = "non_finite_value_count"
        case nonFiniteTensorCount = "non_finite_tensor_count"
        case nonFiniteTensorNames = "non_finite_tensor_names"
        case largestMagnitudeTensors = "largest_magnitude_tensors"
    }

    // MARK: Rollups (derived, never stored)

    /// Sites the dead / mostly-off / always-on classification applies to.
    var classifiedSites: [BatchNormSiteHealth] {
        batchNormSites.filter { $0.classification == .classified }
    }
    var classifiedChannelCount: Int { classifiedSites.reduce(0) { $0 + $1.channelCount } }
    var deadChannelCount: Int { classifiedSites.reduce(0) { $0 + ($1.deadChannelCount ?? 0) } }
    var mostlyOffChannelCount: Int { classifiedSites.reduce(0) { $0 + ($1.mostlyOffChannelCount ?? 0) } }
    var alwaysOnChannelCount: Int { classifiedSites.reduce(0) { $0 + ($1.alwaysOnChannelCount ?? 0) } }

    /// The classified site with the most dead + mostly-off + always-on
    /// channels (earliest wins a tie); nil when every classified site is
    /// clean or there is none.
    var worstClassifiedSite: BatchNormSiteHealth? {
        var worst: BatchNormSiteHealth?
        for site in classifiedSites {
            guard let count = site.unhealthyChannelCount, count > 0 else { continue }
            if worst.flatMap(\.unhealthyChannelCount).map({ count > $0 }) ?? true {
                worst = site
            }
        }
        return worst
    }

    /// Largest β/|γ| over classified sites.
    var maxBetaOverAbsGammaSite: BatchNormSiteHealth? {
        classifiedSites
            .filter { $0.maxBetaOverAbsGamma != nil }
            .max { ($0.maxBetaOverAbsGamma ?? 0) < ($1.maxBetaOverAbsGamma ?? 0) }
    }

    /// Smallest β/|γ| over classified sites.
    var minBetaOverAbsGammaSite: BatchNormSiteHealth? {
        classifiedSites
            .filter { $0.minBetaOverAbsGamma != nil }
            .min { ($0.minBetaOverAbsGamma ?? 0) < ($1.minBetaOverAbsGamma ?? 0) }
    }

    /// Largest running-variance max/median over every BN site.
    var worstRunningVarianceSite: BatchNormSiteHealth? {
        batchNormSites
            .filter { $0.runningVarianceMaxOverMedian != nil }
            .max { ($0.runningVarianceMaxOverMedian ?? 0) < ($1.runningVarianceMaxOverMedian ?? 0) }
    }

    /// The ReZero block with the largest |fraction of cap|.
    var mostSaturatedReZero: ReZeroHealth? {
        reZero
            .filter { $0.fractionOfCap != nil }
            .max { abs($0.fractionOfCap ?? 0) < abs($1.fractionOfCap ?? 0) }
    }
    var saturatedReZeroCount: Int { reZero.filter { $0.saturated == true }.count }
}

// MARK: - Rendering

extension LayerHealthSummary {

    /// The compact one-line rendering (no tag): `scope`, `reluSites`
    /// (classified / all BN sites), `ch`, `dead`, `off`, `alwaysOn`, `worst`
    /// (the classified site with the most of those), `maxBetaOverGamma` /
    /// `minBetaOverGamma` (`value@site[channel]`), `rvMaxOverMedian`,
    /// `rezeroCapFrac` + `saturated`, `nonFinite`; with velocity also
    /// `seZeroVel`, `seLowVel`, `worstSE`, `valueFC1ZeroVel`; for
    /// `.allTensors` also `maxAbs` (`value@tensor`).
    func compactLine() -> String {
        var fields: [String] = ["scope=\(scope.rawValue)"]
        let classified = classifiedSites
        fields.append("reluSites=\(classified.count)/\(batchNormSites.count)")
        if classified.isEmpty {
            fields.append("dead=n/a off=n/a alwaysOn=n/a (no relu/leaky_relu BN sites)")
        } else {
            fields.append("ch=\(classifiedChannelCount)")
            fields.append("dead=\(deadChannelCount)")
            fields.append("off=\(mostlyOffChannelCount)")
            fields.append("alwaysOn=\(alwaysOnChannelCount)")
            if let worst = worstClassifiedSite {
                fields.append("worst=\(worst.site)(dead \(worst.deadChannelCount ?? 0) off \(worst.mostlyOffChannelCount ?? 0) on \(worst.alwaysOnChannelCount ?? 0))")
            } else {
                fields.append("worst=none")
            }
            if let site = maxBetaOverAbsGammaSite, let value = site.maxBetaOverAbsGamma,
               let channel = site.maxBetaOverAbsGammaChannel {
                fields.append("maxBetaOverGamma=\(Self.signed(value))@\(site.site)[\(channel)]")
            }
            if let site = minBetaOverAbsGammaSite, let value = site.minBetaOverAbsGamma,
               let channel = site.minBetaOverAbsGammaChannel {
                fields.append("minBetaOverGamma=\(Self.signed(value))@\(site.site)[\(channel)]")
            }
        }
        if let site = worstRunningVarianceSite, let ratio = site.runningVarianceMaxOverMedian,
           let channel = site.runningVarianceMaxChannel {
            fields.append("rvMaxOverMedian=\(String(format: "%.1f", ratio))@\(site.site)[\(channel)]")
        } else {
            fields.append("rvMaxOverMedian=n/a")
        }
        if reZero.isEmpty {
            fields.append("rezero=none")
        } else if let most = mostSaturatedReZero, let fraction = most.fractionOfCap {
            fields.append("rezeroCapFrac=\(String(format: "%.3f", abs(fraction)))@blocks.\(most.blockIndex)")
            fields.append("saturated=\(saturatedReZeroCount)/\(reZero.count)")
        } else {
            fields.append("rezeroCapFrac=n/a")
        }
        fields.append("nonFinite=\(nonFiniteValueCount)")
        if let se = squeezeExcitationFC1 {
            if se.isEmpty {
                fields.append("seZeroVel=none")
            } else {
                let zero = se.reduce(0) { $0 + $1.zeroVelocityUnitCount }
                let low = se.reduce(0) { $0 + $1.lowVelocityUnitCount }
                let units = se.reduce(0) { $0 + $1.unitCount }
                fields.append("seZeroVel=\(zero)/\(units)")
                fields.append("seLowVel=\(low)")
                if let worst = se.max(by: { $0.zeroVelocityUnitCount < $1.zeroVelocityUnitCount }),
                   worst.zeroVelocityUnitCount > 0, let block = worst.blockIndex {
                    fields.append("worstSE=blocks.\(block)(\(worst.zeroVelocityUnitCount)/\(worst.unitCount))")
                }
            }
        }
        if let value = valueFC1 {
            fields.append("valueFC1ZeroVel=\(value.zeroVelocityUnitCount)/\(value.unitCount)")
        }
        if squeezeExcitationFC1 == nil, valueFC1 == nil, scope == .allTensors {
            fields.append("velocity=n/a")
        }
        if let largest = largestMagnitudeTensors?.first {
            fields.append("maxAbs=\(Self.general(largest.maxAbs))@\(largest.name)")
        }
        return fields.joined(separator: " ")
    }

    /// The detailed multi-line rendering (no tag, no headline): a BN site
    /// table, the velocity checks, ReZero, the NaN sweep and the largest
    /// |value| tensors.
    func detailedLines() -> [String] {
        var lines: [String] = []
        let deadThreshold = Self.signed(LayerHealth.deadBetaOverAbsGamma)
        let mostlyOffThreshold = Self.signed(LayerHealth.mostlyOffBetaOverAbsGamma)
        let alwaysOnThreshold = Self.signed(LayerHealth.alwaysOnBetaOverAbsGamma)
        lines.append("batch-norm sites — β/|γ|: dead < \(deadThreshold), mostly off < \(mostlyOffThreshold), always on > \(alwaysOnThreshold); classified for relu/leaky_relu only")
        let nameWidth = max(4, batchNormSites.map(\.site.count).max() ?? 0)
        lines.append("  \(Self.batchNormHeader(nameWidth: nameWidth))")
        for site in batchNormSites {
            lines.append("  \(Self.batchNormRow(site, nameWidth: nameWidth))")
        }

        if let reason = velocityNotIncludedReason {
            lines.append("velocity checks: not included (\(reason))")
        } else {
            let fraction = Int((LayerHealth.lowVelocityFractionOfReferenceUnitNorm * 100).rounded())
            let percentile = Int(LayerHealth.lowVelocityReferencePercentile.rounded())
            lines.append("FC hidden-unit velocity — zero: every weight-velocity entry exactly 0; low: nonzero but < \(fraction)% of the layer's p\(percentile) unit norm")
            for layer in (squeezeExcitationFC1 ?? []) + [valueFC1].compactMap({ $0 }) {
                lines.append("  \(Self.velocityRow(layer))")
            }
            if squeezeExcitationFC1?.isEmpty == true {
                lines.append("  (no SE blocks)")
            }
        }

        if reZero.isEmpty {
            lines.append("ReZero: none")
        } else {
            lines.append("ReZero — effective α = cap·tanh(α/cap); saturated at |fraction| ≥ \(String(format: "%.2f", LayerHealth.reZeroSaturatedFractionOfCap))")
            for entry in reZero {
                lines.append("  \(Self.reZeroRow(entry))")
            }
        }

        var nonFiniteLine = "non-finite: \(nonFiniteValueCount) values in \(nonFiniteTensorCount) of \(examinedTensorCount) tensors (\(examinedValueCount) values examined)"
        if !nonFiniteTensorNames.isEmpty {
            nonFiniteLine += "; first: \(nonFiniteTensorNames.joined(separator: ", "))"
        }
        lines.append(nonFiniteLine)

        if let largest = largestMagnitudeTensors, !largest.isEmpty {
            let list = largest.map { "\($0.name) \(Self.general($0.maxAbs))" }.joined(separator: ", ")
            lines.append("largest |value|: \(list)")
        }
        return lines
    }

    /// Column widths of the BN site table, shared by the header and every
    /// row so the columns stay aligned.
    private enum BatchNormColumn {
        static let activation = 10
        static let count = 6
        static let extreme = 17
        static let varianceRatio = 9
    }

    private static func batchNormHeader(nameWidth: Int) -> String {
        let counts = ["ch", "dead", "off", "on", "zeroγ", "nonfin"]
            .map { leftPad($0, BatchNormColumn.count) }
            .joined(separator: " ")
        let extremes = rightPad("min β/|γ| [ch]", BatchNormColumn.extreme) + "  " + rightPad("max β/|γ| [ch]", BatchNormColumn.extreme)
        return "\(rightPad("site", nameWidth))  \(rightPad("act", BatchNormColumn.activation)) \(counts)  \(extremes)  rv max/median [ch]"
    }

    private static func batchNormRow(_ site: BatchNormSiteHealth, nameWidth: Int) -> String {
        let countValues: [Int?] = [
            site.channelCount, site.deadChannelCount, site.mostlyOffChannelCount,
            site.alwaysOnChannelCount, site.zeroGammaChannelCount, site.nonFiniteChannelCount,
        ]
        let counts = countValues
            .map { leftPad($0.map(String.init) ?? "n/a", BatchNormColumn.count) }
            .joined(separator: " ")
        let extremeMin = rightPad(extreme(site.minBetaOverAbsGamma, site.minBetaOverAbsGammaChannel), BatchNormColumn.extreme)
        let extremeMax = rightPad(extreme(site.maxBetaOverAbsGamma, site.maxBetaOverAbsGammaChannel), BatchNormColumn.extreme)
        let varianceRatio: String
        if let ratio = site.runningVarianceMaxOverMedian, let channel = site.runningVarianceMaxChannel {
            varianceRatio = "\(leftPad(String(format: "%.1f", ratio), BatchNormColumn.varianceRatio)) [\(channel)]"
        } else {
            varianceRatio = leftPad("n/a", BatchNormColumn.varianceRatio)
        }
        let activation = rightPad(site.activation?.rawValue ?? "-", BatchNormColumn.activation)
        var row = "\(rightPad(site.site, nameWidth))  \(activation) \(counts)  \(extremeMin)  \(extremeMax)  \(varianceRatio)"
        if site.nonFiniteRunningVarianceCount > 0 {
            row += "  (\(site.nonFiniteRunningVarianceCount) non-finite running var)"
        }
        return row
    }

    private static func extreme(_ value: Double?, _ channel: Int?) -> String {
        guard let value, let channel else { return "n/a" }
        return "\(String(format: "%+8.2f", value)) [\(channel)]"
    }

    private static func velocityRow(_ layer: HiddenUnitVelocityHealth) -> String {
        var row = "\(layer.layer)  \(layer.activation.rawValue)  units \(layer.unitCount)  zero \(layer.zeroVelocityUnitCount)"
        if !layer.zeroVelocityUnits.isEmpty {
            let shown = layer.zeroVelocityUnits.map(String.init).joined(separator: ",")
            let more = layer.zeroVelocityUnitCount > layer.zeroVelocityUnits.count ? ",…" : ""
            row += " [\(shown)\(more)]"
        }
        row += "  low \(layer.lowVelocityUnitCount)"
        if layer.nonFiniteUnitCount > 0 {
            row += "  non-finite \(layer.nonFiniteUnitCount)"
        }
        if let median = layer.medianUnitVelocityNorm {
            row += "  median unit norm \(general(median))"
        }
        return row
    }

    private static func reZeroRow(_ entry: ReZeroHealth) -> String {
        let block = "blocks.\(entry.blockIndex)"
        guard let alpha = entry.alpha, let effective = entry.effectiveAlpha, let fraction = entry.fractionOfCap else {
            return "\(block)  α non-finite  cap \(String(format: "%.4f", entry.cap))"
        }
        let saturated = entry.saturated == true ? "  SATURATED" : ""
        return "\(block)  α \(String(format: "%+.4f", alpha))  cap \(String(format: "%.4f", entry.cap))  effective \(String(format: "%+.4f", effective))  = \(String(format: "%+.3f", fraction)) of cap\(saturated)"
    }

    static func signed(_ value: Double) -> String { String(format: "%+.2f", value) }

    static func general(_ value: Double) -> String {
        let magnitude = abs(value)
        if magnitude != 0, magnitude >= 1e4 || magnitude < 1e-3 {
            return String(format: "%.3e", value)
        }
        return String(format: "%.4g", value)
    }

    private static func leftPad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : String(repeating: " ", count: width - text.count) + text
    }

    private static func rightPad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
    }
}

// MARK: - Errors

enum LayerHealthError: LocalizedError, Equatable {
    case missingTensor(String)
    case unknownTensor(String)
    case tensorSizeMismatch(name: String, expected: Int, got: Int)
    case tensorCountMismatch(what: String, expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .missingTensor(let name):
            return "layer health: tensor \(name) is missing"
        case .unknownTensor(let name):
            return "layer health: tensor \(name) is not in the architecture's weight plan"
        case .tensorSizeMismatch(let name, let expected, let got):
            return "layer health: tensor \(name) has \(got) values, expected \(expected)"
        case .tensorCountMismatch(let what, let expected, let got):
            return "layer health: \(got) \(what), expected \(expected)"
        }
    }
}
