//
//  InitOptionDerive.swift
//  DrewsChessMachine
//
//  `--derive-model` operations for the init-neutral options (determinism plan
//  B2.1, decision D-4): one shape-preserving operation per option plus
//  `--set-neutral-init`, each re-initializing ONLY the tensors its option owns
//  and copying every other tensor bit-exact, exactly as `--set-se-beta-init`
//  does. Every operation's tensor rewrites come from one function,
//  `InitOptionTensorRewrites.between`, which compares the option values of
//  the source and target architectures — so an operation can only ever rewrite
//  what its own field change implies, and the per-option and neutral
//  operations can never disagree about what a value means. The values written
//  are the graph builder's own (`WeightInitScheme`, `ChessNetwork`), so a
//  derived tensor equals what a fresh build of the target architecture would
//  give it.
//

import Foundation

// MARK: - Operations that draw weights

/// A derive operation that may draw fresh weights and so carries an init
/// seed (recorded in its derivation record). `--init-seed` replaces the
/// drawn seed of every such operation that `drawsWeights`.
protocol InitSeedableDeriveOperation: DeriveOperation {
    /// Whether this operation draws weights (so an init seed applies to it).
    var drawsWeights: Bool { get }
    /// This operation drawing under `seed` instead.
    func withInitSeed(_ seed: UInt64) -> Self
}

extension SetSEBetaInitDeriveOperation: InitSeedableDeriveOperation {}

// MARK: - Shared rewrites

enum InitOptionTensorRewrites {

    /// The rewrites that turn `source`-init tensors into `target`-init ones,
    /// for every init-neutral option whose value differs between the two
    /// (same tensor layout — the derive engine has checked it). Deterministic
    /// values (SE γ bias, last BN γ, identity-like projection, zero head
    /// finals, the draw-prior bias) need no seed; a He re-draw uses `initSeed`
    /// through the tensor's own `init/<name>` stream.
    static func between(source: NetworkArchitecture, target: NetworkArchitecture,
                        initSeed: UInt64) throws -> [DeriveTensorRewrite] {
        let plan = target.weightTensorPlan()
        var specByName: [String: WeightTensorSpec] = [:]
        for spec in plan { specByName[spec.name] = spec }
        func spec(_ name: String) throws -> WeightTensorSpec {
            guard let found = specByName[name] else { throw ModelDerivation.DeriveError.missingTensor(name: name) }
            return found
        }
        func whole(_ name: String, values: [Float], summary: String) throws -> DeriveTensorRewrite {
            let tensor = try spec(name)
            guard values.count == tensor.elementCount else {
                throw ModelDerivation.DeriveError.rewriteChangedElementCount(
                    name: name, before: tensor.elementCount, after: values.count)
            }
            return DeriveTensorRewrite(
                tensorName: name, rewrittenElementRanges: [0..<values.count], summary: summary,
                rewrite: { data in for index in values.indices { data[index] = values[index] } })
        }

        var rewrites: [DeriveTensorRewrite] = []
        let projectionBlocks = Set(target.skipProjectionBlockIndices)
        for (groupIndex, group) in target.blockGroups.enumerated() {
            let before = source.blockGroups[groupIndex]
            let blocks = target.blockRange(ofGroup: groupIndex)
            if before.seGammaBiasInit != group.seGammaBiasInit, group.seStyle != .none {
                let module = group.seStyle == .scaleAndBias ? "se_scalebias" : "se_attenuate"
                let values = try WeightInitScheme.seFC2BiasValues(group: group)
                let gammaHalf = 0..<group.channels
                for block in blocks {
                    let name = "blocks.\(block).\(module).fc2.bias"
                    _ = try spec(name)
                    rewrites.append(DeriveTensorRewrite(
                        tensorName: name, rewrittenElementRanges: [gammaHalf],
                        summary: "γ bias 0..<\(group.channels) set to \(group.seGammaBiasInit)",
                        rewrite: { data in for index in gammaHalf { data[index] = values[index] } }))
                }
            }
            if before.branchOutputInit != group.branchOutputInit {
                let gamma = ChessNetwork.lastBranchBatchNormGamma(group)
                for block in blocks {
                    let name = "blocks.\(block).bn2.weight"
                    rewrites.append(try whole(
                        name, values: [Float](repeating: gamma, count: try spec(name).elementCount),
                        summary: "last BN γ set to \(gamma) (\(group.branchOutputInit.rawValue))"))
                }
            }
            if before.skipProjectionInit != group.skipProjectionInit {
                for block in blocks where projectionBlocks.contains(block) {
                    let name = "blocks.\(block).skip_proj.weight"
                    let projection = try spec(name)
                    switch group.skipProjectionInit {
                    case .identityLike:
                        rewrites.append(try whole(
                            name,
                            values: WeightInitScheme.identityLikeProjectionValues(
                                outChannels: projection.shape[0], inChannels: projection.shape[1]),
                            summary: "identity-like skip projection"))
                    case .he:
                        rewrites.append(try whole(
                            name,
                            values: try WeightInitScheme.storedValues(initSeed: initSeed, spec: projection, distribution: .heNormal),
                            summary: "skip projection re-drawn He-normal"))
                    }
                }
            }
        }

        if source.policyHeadFinalInit != target.policyHeadFinalInit {
            let name = target.policyHeadStyle == .fcBottleneck ? "policy.fc.weight" : "policy.conv.weight"
            rewrites.append(try headFinal(name, init: target.policyHeadFinalInit, spec: try spec(name), initSeed: initSeed))
        }
        if source.valueHeadFinalInit != target.valueHeadFinalInit {
            let name = target.valueHeadStyle == .wdlSoftmax ? "value.wdl_fc2.weight" : "value.scalar_fc2.weight"
            rewrites.append(try headFinal(name, init: target.valueHeadFinalInit, spec: try spec(name), initSeed: initSeed))
        }
        if source.valueHeadDrawPrior != target.valueHeadDrawPrior, target.valueHeadStyle == .wdlSoftmax {
            rewrites.append(try whole(
                "value.wdl_fc2.bias",
                values: NetworkArchitecture.wdlBiasPrior(drawProbability: target.valueHeadDrawPrior),
                summary: "W/D/L bias set for draw prior \(target.valueHeadDrawPrior)"))
        }
        return rewrites
    }

    private static func headFinal(_ name: String, init finalInit: HeadFinalInit, spec: WeightTensorSpec,
                                  initSeed: UInt64) throws -> DeriveTensorRewrite {
        let values: [Float]
        let summary: String
        switch finalInit {
        case .zero:
            values = [Float](repeating: 0, count: spec.elementCount)
            summary = "final layer zeroed"
        case .he:
            values = try WeightInitScheme.storedValues(initSeed: initSeed, spec: spec, distribution: .heNormal)
            summary = "final layer re-drawn He-normal"
        }
        return DeriveTensorRewrite(
            tensorName: name, rewrittenElementRanges: [0..<values.count], summary: summary,
            rewrite: { data in for index in values.indices { data[index] = values[index] } })
    }

    /// Whether turning `source` into `target` draws any weights (a move back
    /// to a He init), so an init seed applies.
    static func drawsWeights(source: NetworkArchitecture, target: NetworkArchitecture) -> Bool {
        let projectionDraw = target.blockGroups.indices.contains { index in
            source.blockGroups[index].skipProjectionInit != target.blockGroups[index].skipProjectionInit
                && target.blockGroups[index].skipProjectionInit == .he
                && target.groupHasSkipProjection(index)
        }
        return projectionDraw
            || (source.policyHeadFinalInit != target.policyHeadFinalInit && target.policyHeadFinalInit == .he)
            || (source.valueHeadFinalInit != target.valueHeadFinalInit && target.valueHeadFinalInit == .he)
    }
}

// MARK: - Group selection

/// The groups a per-group option applies to: those named by `--group`
/// (each checked against `applies`), or every group it applies to.
private func selectedGroups(
    _ groupIndices: [Int]?, in architecture: NetworkArchitecture, operation: String, field: String,
    appliesTo applies: (Int) -> Bool, layerDescription: String
) throws -> [Int] {
    if let groupIndices {
        for index in groupIndices where !architecture.blockGroups.indices.contains(index) {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: operation,
                detail: "--group \(index) is out of range (the model has \(architecture.blockGroups.count) block groups, 0-based)")
        }
        for index in groupIndices where !applies(index) {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: operation, detail: "block group \(index) has no \(layerDescription), so \(field) has no effect there")
        }
        return groupIndices
    }
    let all = architecture.blockGroups.indices.filter(applies)
    guard !all.isEmpty else {
        throw ModelDerivation.DeriveError.operationNotApplicable(
            operation: operation, detail: "the model has no block group with a \(layerDescription)")
    }
    return all
}

private func groupsArgument(_ groupIndices: [Int]?, all: String) -> String {
    groupIndices.map { $0.map(String.init).joined(separator: ",") } ?? all
}

private func nothingToDerive(_ operation: String, _ detail: String) -> ModelDerivation.DeriveError {
    .operationNotApplicable(operation: operation, detail: "\(detail); nothing to derive")
}

// MARK: - Operation: set-se-gamma-bias-init

/// Sets `se_gamma_bias_init` on SE block groups and rewrites the γ half of
/// each affected block's SE FC2 bias to it. A constant: no seed.
struct SetSEGammaBiasInitDeriveOperation: DeriveOperation {
    let value: Float
    let groupIndices: [Int]?

    static let kind = DeriveOperationKind(
        name: "set-se-gamma-bias-init",
        flag: "--set-se-gamma-bias-init",
        valueSyntax: "<float>",
        summary: "Set se_gamma_bias_init on SE block groups (all of them, or those named by --group) and set the γ "
            + "half of each affected block's SE FC2 bias to it (0 = standard, gate sigmoid(0) at init; ln 9 ≈ 2.197 = "
            + "near-identity gate 0.9). Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["block_groups[].se_gamma_bias_init"],
        rewrittenTensorsDescription: "blocks.<i>.se_attenuate.fc2.bias, or blocks.<i>.se_scalebias.fc2.bias 0..C-1, "
            + "for every block i of an affected group",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            guard let parsed = Float(value), parsed.isFinite else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-se-gamma-bias-init", detail: "value '\(value)' is not a finite number")
            }
            return SetSEGammaBiasInitDeriveOperation(value: parsed, groupIndices: groupIndices)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        ["value": "\(value)", "groups": groupsArgument(groupIndices, all: "all with SE")]
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let groups = try selectedGroups(
            groupIndices, in: architecture, operation: kindName, field: "se_gamma_bias_init",
            appliesTo: { architecture.blockGroups[$0].seStyle != .none }, layerDescription: "SE block")
        guard groups.contains(where: { architecture.blockGroups[$0].seGammaBiasInit != value }) else {
            throw nothingToDerive(kindName, "every selected block group already has se_gamma_bias_init \(value)")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].seGammaBiasInit = value }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        try InitOptionTensorRewrites.between(source: source, target: target, initSeed: 0)
    }
}

// MARK: - Operation: set-branch-output-init

/// Sets `branch_output_init` on post-activation block groups and rewrites the
/// last BN γ (`bn2.weight`) of each affected block: 0 for
/// `zero_last_bn_gamma`, 1 for `standard`. A constant: no seed.
struct SetBranchOutputInitDeriveOperation: DeriveOperation {
    let value: BranchOutputInit
    let groupIndices: [Int]?

    static let kind = DeriveOperationKind(
        name: "set-branch-output-init",
        flag: "--set-branch-output-init",
        valueSyntax: BranchOutputInit.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set branch_output_init on post-activation block groups (all of them, or those named by --group) and "
            + "set each affected block's last BN γ to match: zero_last_bn_gamma = 0 (the branch adds nothing at step 0), "
            + "standard = 1. Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["block_groups[].branch_output_init"],
        rewrittenTensorsDescription: "blocks.<i>.bn2.weight for every block i of an affected group",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            guard let parsed = BranchOutputInit(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-branch-output-init",
                    detail: "value '\(value)' is not one of \(BranchOutputInit.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetBranchOutputInitDeriveOperation(value: parsed, groupIndices: groupIndices)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        ["value": value.rawValue, "groups": groupsArgument(groupIndices, all: "all post-activation")]
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let groups = try selectedGroups(
            groupIndices, in: architecture, operation: kindName, field: "branch_output_init",
            appliesTo: { architecture.blockGroups[$0].activationStyle == .post },
            layerDescription: "BN after its branch's last conv (post-activation)")
        guard groups.contains(where: { architecture.blockGroups[$0].branchOutputInit != value }) else {
            throw nothingToDerive(kindName, "every selected block group already has branch_output_init '\(value.rawValue)'")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].branchOutputInit = value }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        try InitOptionTensorRewrites.between(source: source, target: target, initSeed: 0)
    }
}

// MARK: - Operation: set-skip-projection-init

/// Sets `skip_projection_init` on block groups with a width-transition skip
/// projection and rewrites those projections: the identity-like constant, or
/// a He re-draw from the tensor's `init/<name>` stream under `initSeed`.
struct SetSkipProjectionInitDeriveOperation: InitSeedableDeriveOperation {
    let value: SkipProjectionInit
    let groupIndices: [Int]?
    let initSeed: UInt64

    var drawsWeights: Bool { value == .he }

    func withInitSeed(_ seed: UInt64) -> SetSkipProjectionInitDeriveOperation {
        SetSkipProjectionInitDeriveOperation(value: value, groupIndices: groupIndices, initSeed: seed)
    }

    static let kind = DeriveOperationKind(
        name: "set-skip-projection-init",
        flag: "--set-skip-projection-init",
        valueSyntax: SkipProjectionInit.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set skip_projection_init on block groups that change width (all of them, or those named by --group) and "
            + "re-initialize their 1x1 skip projections to match: identity_like = 1 from channel i to channel i, 0 elsewhere; "
            + "he = a fresh He-normal draw from --init-seed (or a drawn seed, recorded). Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["block_groups[].skip_projection_init"],
        rewrittenTensorsDescription: "blocks.<i>.skip_proj.weight for every block i of an affected group that has one",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            guard let parsed = SkipProjectionInit(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-skip-projection-init",
                    detail: "value '\(value)' is not one of \(SkipProjectionInit.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetSkipProjectionInitDeriveOperation(
                value: parsed, groupIndices: groupIndices, initSeed: WeightInitialization.drawnInitSeed())
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        var arguments = ["value": value.rawValue, "groups": groupsArgument(groupIndices, all: "all with a skip projection")]
        if drawsWeights {
            arguments["init_seed"] = String(initSeed)
            arguments["init_scheme"] = WeightInitScheme.current
        }
        return arguments
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let groups = try selectedGroups(
            groupIndices, in: architecture, operation: kindName, field: "skip_projection_init",
            appliesTo: { architecture.groupHasSkipProjection($0) }, layerDescription: "width-transition skip projection")
        guard groups.contains(where: { architecture.blockGroups[$0].skipProjectionInit != value }) else {
            throw nothingToDerive(kindName, "every selected block group already has skip_projection_init '\(value.rawValue)'")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].skipProjectionInit = value }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        try InitOptionTensorRewrites.between(source: source, target: target, initSeed: initSeed)
    }
}

// MARK: - Operations: head final layers

/// Sets `policy_head_final_init` and rewrites the policy head's final
/// projection: zeros, or a He re-draw under `initSeed`.
struct SetPolicyHeadFinalInitDeriveOperation: InitSeedableDeriveOperation {
    let value: HeadFinalInit
    let initSeed: UInt64

    var drawsWeights: Bool { value == .he }

    func withInitSeed(_ seed: UInt64) -> SetPolicyHeadFinalInitDeriveOperation {
        SetPolicyHeadFinalInitDeriveOperation(value: value, initSeed: seed)
    }

    static let kind = DeriveOperationKind(
        name: "set-policy-head-final-init",
        flag: "--set-policy-head-final-init",
        valueSyntax: HeadFinalInit.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set policy_head_final_init and re-initialize the policy head's final projection to match: zero = a "
            + "uniform policy at step 0; he = a fresh He-normal draw from --init-seed (or a drawn seed, recorded). "
            + "Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["policy_head_final_init"],
        rewrittenTensorsDescription: "policy.conv.weight (simple_conv, intermediate_conv) or policy.fc.weight (fc_bottleneck)",
        acceptsGroupSelection: false,
        make: { value, _ in
            guard let parsed = HeadFinalInit(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-policy-head-final-init",
                    detail: "value '\(value)' is not one of \(HeadFinalInit.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetPolicyHeadFinalInitDeriveOperation(value: parsed, initSeed: WeightInitialization.drawnInitSeed())
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        var arguments = ["value": value.rawValue]
        if drawsWeights {
            arguments["init_seed"] = String(initSeed)
            arguments["init_scheme"] = WeightInitScheme.current
        }
        return arguments
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard architecture.policyHeadFinalInit != value else {
            throw nothingToDerive(kindName, "policy_head_final_init is already '\(value.rawValue)'")
        }
        var edited = architecture
        edited.policyHeadFinalInit = value
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        try InitOptionTensorRewrites.between(source: source, target: target, initSeed: initSeed)
    }
}

/// Sets `value_head_final_init` and rewrites the value head's final FC
/// weight: zeros, or a He re-draw under `initSeed`.
struct SetValueHeadFinalInitDeriveOperation: InitSeedableDeriveOperation {
    let value: HeadFinalInit
    let initSeed: UInt64

    var drawsWeights: Bool { value == .he }

    func withInitSeed(_ seed: UInt64) -> SetValueHeadFinalInitDeriveOperation {
        SetValueHeadFinalInitDeriveOperation(value: value, initSeed: seed)
    }

    static let kind = DeriveOperationKind(
        name: "set-value-head-final-init",
        flag: "--set-value-head-final-init",
        valueSyntax: HeadFinalInit.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set value_head_final_init and re-initialize the value head's final FC weight to match: zero = the "
            + "value head outputs its W/D/L prior at step 0; he = a fresh He-normal draw from --init-seed (or a drawn "
            + "seed, recorded). The bias is untouched. Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["value_head_final_init"],
        rewrittenTensorsDescription: "value.wdl_fc2.weight (or value.scalar_fc2.weight)",
        acceptsGroupSelection: false,
        make: { value, _ in
            guard let parsed = HeadFinalInit(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-value-head-final-init",
                    detail: "value '\(value)' is not one of \(HeadFinalInit.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetValueHeadFinalInitDeriveOperation(value: parsed, initSeed: WeightInitialization.drawnInitSeed())
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        var arguments = ["value": value.rawValue]
        if drawsWeights {
            arguments["init_seed"] = String(initSeed)
            arguments["init_scheme"] = WeightInitScheme.current
        }
        return arguments
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard architecture.valueHeadFinalInit != value else {
            throw nothingToDerive(kindName, "value_head_final_init is already '\(value.rawValue)'")
        }
        var edited = architecture
        edited.valueHeadFinalInit = value
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        try InitOptionTensorRewrites.between(source: source, target: target, initSeed: initSeed)
    }
}

/// Sets `value_head_draw_prior` on a W/D/L head and rewrites its final bias to
/// `NetworkArchitecture.wdlBiasPrior`. A constant: no seed.
struct SetValueHeadDrawPriorDeriveOperation: DeriveOperation {
    let value: Float

    static let kind = DeriveOperationKind(
        name: "set-value-head-draw-prior",
        flag: "--set-value-head-draw-prior",
        valueSyntax: "<probability in (0, 1)>",
        summary: "Set value_head_draw_prior (the W/D/L head's initial draw probability; standard 0.75) and rewrite the "
            + "head's final bias to [0, ln(2p/(1-p)), 0]. Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["value_head_draw_prior"],
        rewrittenTensorsDescription: "value.wdl_fc2.bias",
        acceptsGroupSelection: false,
        make: { value, _ in
            guard let parsed = Float(value), parsed.isFinite else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-value-head-draw-prior", detail: "value '\(value)' is not a finite number")
            }
            return SetValueHeadDrawPriorDeriveOperation(value: parsed)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] { ["value": "\(value)"] }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard architecture.valueHeadStyle == .wdlSoftmax else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName, detail: "the value head is '\(architecture.valueHeadStyle.rawValue)', which has no draw class")
        }
        guard architecture.valueHeadDrawPrior != value else {
            throw nothingToDerive(kindName, "value_head_draw_prior is already \(value)")
        }
        var edited = architecture
        edited.valueHeadDrawPrior = value
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        try InitOptionTensorRewrites.between(source: source, target: target, initSeed: 0)
    }
}

// MARK: - Operation: set-neutral-init

/// Applies the Neutral init set (`NetworkArchitecture.withNeutralInit`, the
/// same function as the Build screen's "Neutral init" button) to every block
/// group and head, and rewrites the tensors of every option that changed.
/// Every neutral value is a constant, so it draws nothing. The draw prior is
/// not part of the set and is left as it is.
struct SetNeutralInitDeriveOperation: DeriveOperation {

    /// The only value the flag takes: the set applies to the whole model.
    static let value = "all"

    static let kind = DeriveOperationKind(
        name: "set-neutral-init",
        flag: "--set-neutral-init",
        valueSyntax: value,
        summary: "Apply the Neutral init set to every block group and head (the Build New Model screen's "
            + "\"Neutral init\"): near-identity SE gate bias, zero last BN γ on post-activation groups, identity-like "
            + "skip projections, zero policy and value final layers; the draw prior is left as it is. Every other "
            + "tensor is copied bit-exact.",
        changedArchitectureFields: [
            "block_groups[].se_gamma_bias_init", "block_groups[].branch_output_init",
            "block_groups[].skip_projection_init", "policy_head_final_init", "value_head_final_init",
        ],
        rewrittenTensorsDescription: "the tensors of every option the set changes (see the per-option operations)",
        acceptsGroupSelection: false,
        make: { value, _ in
            guard value == SetNeutralInitDeriveOperation.value else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-neutral-init", detail: "value '\(value)' must be '\(SetNeutralInitDeriveOperation.value)'")
            }
            return SetNeutralInitDeriveOperation()
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] { ["value": Self.value] }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let edited = architecture.withNeutralInit()
        guard edited != architecture else {
            throw nothingToDerive(kindName, "the model already has the neutral init")
        }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        precondition(!InitOptionTensorRewrites.drawsWeights(source: source, target: target),
                     "the Neutral set has only constant values")
        return try InitOptionTensorRewrites.between(source: source, target: target, initSeed: 0)
    }
}
