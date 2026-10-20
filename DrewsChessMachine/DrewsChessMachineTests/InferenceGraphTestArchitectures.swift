@testable import DrewsChessMachine

/// Small fp32 architectures that together reach every option the forward
/// graph builders branch on, for the inference-graph recording and Core ML
/// parity tests. Each is validated, so a variant the architecture rules refuse
/// fails loudly instead of quietly testing less.
enum InferenceGraphTestArchitectures {
    struct Variant {
        let name: String
        let architecture: NetworkArchitecture
    }

    private static func base(style: BlockActivationStyle = .pre, merge: BlockSkipMerge = .cleanAdd,
                             se: SEStyle = .none, rezero: Bool = false,
                             policy: PolicyHeadStyle = .intermediateConv, value: ValueHeadStyle = .wdlSoftmax,
                             encoding: InputEncoding = .basic24) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: encoding, channels: 32, numBlocks: 2, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: style,
            blockSkipMerge: merge, blockUseRezero: rezero, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 5,
            blockSeStyle: se, blockSeReductionRatio: 4,
            policyHeadStyle: policy, policyPreConvChannels: 16,
            valueHeadStyle: value, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32)
    }

    private static func finished(_ arch: NetworkArchitecture, activation: ActivationFunction) throws -> NetworkArchitecture {
        var arch = arch
        arch.clearActivationSitesTheTopologyLacks()
        try arch.setActivationAtEveryExistingSite(activation)
        try arch.validate()
        return arch
    }

    static func all() throws -> [Variant] {
        // Pre-activation, scale-and-bias SE, ReZero, intermediate_conv policy.
        var seRezero = base(se: .scaleAndBias, rezero: true, encoding: .basic30)
        for i in seRezero.blockGroups.indices { seRezero.blockGroups[i].seActivation = .relu }

        // Post-activation, activation-gated merge, attenuate-only SE with a
        // leaky FC1, simple_conv policy, scalar tanh value head, zero-γ init.
        var post = base(style: .post, merge: .activationGated, se: .attenuateOnly,
                        policy: .simpleConv, value: .scalarTanh)
        for i in post.blockGroups.indices {
            post.blockGroups[i].seActivation = .leakyRelu
            post.blockGroups[i].activationFunction = .leakyRelu
            post.blockGroups[i].branchOutputInit = .zeroLastBNGamma
        }

        // LayerNorm block output, SiLU main path, GELU at the sites,
        // fc_bottleneck policy.
        var layerNorm = base(policy: .fcBottleneck)
        for i in layerNorm.blockGroups.indices {
            layerNorm.blockGroups[i].outputNorm = .layerNorm
            layerNorm.blockGroups[i].activationFunction = .silu
        }

        // A width change between groups (1×1 skip projection) and the
        // compress feature skip into both heads.
        var widthCompress = base(se: .scaleAndBias)
        let group = widthCompress.blockGroups[0]
        var wide = group
        wide.channels = 48
        var first = group
        first.count = 1
        wide.count = 1
        first.seActivation = .gelu
        wide.seActivation = .gelu
        widthCompress.blockGroups = [first, wide]
        widthCompress.featureSkipSource = .stemOutput
        widthCompress.featureSkipFusion = .compressConvBNReLU
        widthCompress.featureSkipToPolicyHead = true
        widthCompress.featureSkipToValueHead = true

        // concat_direct feature skip into the policy head and the final block.
        var concat = base()
        concat.featureSkipSource = .stemOutput
        concat.featureSkipFusion = .concatDirect
        concat.featureSkipToPolicyHead = true
        concat.featureSkipToFinalBlock = true

        return [
            Variant(name: "pre-se-scalebias-rezero", architecture: try finished(seRezero, activation: .relu)),
            Variant(name: "post-gated-attenuate-simpleconv-tanh", architecture: try finished(post, activation: .leakyRelu)),
            Variant(name: "layernorm-silu-gelu-fcbottleneck", architecture: try finished(layerNorm, activation: .gelu)),
            Variant(name: "width-change-compress-skip", architecture: try finished(widthCompress, activation: .silu)),
            Variant(name: "concat-skip-policy-final-block", architecture: try finished(concat, activation: .relu)),
        ]
    }
}
