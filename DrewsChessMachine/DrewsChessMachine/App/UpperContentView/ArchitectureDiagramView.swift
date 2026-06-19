import SwiftUI

/// Read-only, live-updating schematic of an ENTIRE network, rendered from a
/// (valid) `NetworkArchitecture` (ARCHITECTURE_EXPANSION_PLAN.md Feature 2
/// Phase B). Two call sites share this renderer: the Build-New-Model draft
/// ("what am I about to build") and the About popover's current network
/// ("what am I running").
///
/// Vertical flow: input planes → stem → each block group as one cell (count
/// badge + the FULL recipe — both kernels, channels, SE, activation
/// function/style, skip merge, ReZero α, dropout multiplier; the same
/// no-silent-defaults rule as `architectureSummary`) → 1×1-projection
/// markers wherever adjacent widths differ → tower-end BN → policy and value
/// heads side by side. Box width is proportional to channel count so a WRN
/// width staircase is visible at a glance; every segment carries its
/// parameter count, summed from `weightTensorPlan()` (the single source of
/// truth for tensor shapes), with the grand total at the bottom.
struct ArchitectureDiagramView: View {
    let architecture: NetworkArchitecture

    /// Per-segment element counts, summed from the weight plan by name
    /// prefix (`stem.`, `blocks.N.`, `tower_final_bn`, `policy.`, `value.`).
    private struct Segments {
        var stem = 0
        var perGroup: [Int] = []
        var towerEndBN = 0
        var featureSkip = 0
        var policy = 0
        var value = 0
        var total = 0
    }

    private var segments: Segments {
        var s = Segments()
        var blockToGroup: [Int] = []
        for (gi, g) in architecture.blockGroups.enumerated() {
            blockToGroup.append(contentsOf: Array(repeating: gi, count: g.count))
        }
        s.perGroup = Array(repeating: 0, count: architecture.blockGroups.count)
        for spec in architecture.weightTensorPlan() {
            s.total += spec.elementCount
            if spec.name.hasPrefix("stem.") {
                s.stem += spec.elementCount
            } else if spec.name.hasPrefix("blocks.") {
                let digits = spec.name.dropFirst("blocks.".count).prefix(while: \.isNumber)
                if let b = Int(digits), b < blockToGroup.count {
                    s.perGroup[blockToGroup[b]] += spec.elementCount
                }
            } else if spec.name.hasPrefix("tower_final_bn") {
                s.towerEndBN += spec.elementCount
            } else if spec.name.hasPrefix("feature_skip.") {
                s.featureSkip += spec.elementCount
            } else if spec.name.hasPrefix("policy.") {
                s.policy += spec.elementCount
            } else if spec.name.hasPrefix("value.") {
                s.value += spec.elementCount
            }
        }
        return s
    }

    /// Channel scale for width-proportional boxes: the widest thing on the
    /// channel axis anywhere in the flow.
    private var maxChannelScale: Int {
        max(architecture.maxBlockChannels, architecture.inputPlanes)
    }

    private func barWidth(_ channels: Int) -> CGFloat {
        90 + CGFloat(channels) / CGFloat(max(1, maxChannelScale)) * 230
    }

    var body: some View {
        let segs = segments
        let arch = architecture
        VStack(spacing: 0) {
            cell(width: barWidth(arch.inputPlanes), emphasized: false) {
                line("input · \(arch.inputEncoding.rawValue)", bold: true)
                line("\(arch.inputPlanes) planes · 8×8")
            }
            connector()
            cell(width: barWidth(arch.stemOutputChannels), emphasized: false) {
                line("stem \(arch.stemConvKernelSize)×\(arch.stemConvKernelSize) conv", bold: true)
                line("\(arch.inputPlanes) → \(arch.stemOutputChannels)ch · BN\(arch.hasStemActivation ? " · \(arch.activationFunction.rawValue)" : "")")
                paramsLine(segs.stem)
            }
            ForEach(Array(arch.blockGroups.enumerated()), id: \.offset) { gi, g in
                let inC = gi == 0 ? arch.stemOutputChannels : arch.blockGroups[gi - 1].channels
                connector()
                if inC != g.channels {
                    transitionMarker(inC: inC, outC: g.channels)
                    connector()
                }
                groupCell(g, params: gi < segs.perGroup.count ? segs.perGroup[gi] : 0)
            }
            if arch.hasTowerEndBN {
                connector()
                cell(width: barWidth(arch.towerOutputChannels), emphasized: false) {
                    line("tower-end BN · \(arch.activationFunction.rawValue)", bold: true)
                    paramsLine(segs.towerEndBN)
                }
            }
            if arch.featureSkipEnabled {
                connector()
                featureSkipMarker(arch, compressParams: segs.featureSkip)
            }
            connector()
            HStack(alignment: .top, spacing: 16) {
                cell(width: 175, emphasized: false) {
                    line("policy · \(arch.policyHeadStyle.rawValue)", bold: true)
                    if arch.policyHeadStyle != .simpleConv {
                        line("\(arch.policyHeadInputChannels) → K=\(arch.policyPreConvChannels)")
                    }
                    line("→ \(arch.policySize) logits")
                    paramsLine(segs.policy)
                }
                cell(width: 175, emphasized: false) {
                    line("value · \(arch.valueHeadStyle.rawValue)", bold: true)
                    line("\(arch.valueHeadInputChannels) → \(arch.valueHeadConvChannels)ch → FC\(arch.valueHeadHiddenUnits)")
                    line("→ \(arch.valueHeadClasses) \(arch.valueHeadClasses == 3 ? "(W/D/L)" : "(scalar)")")
                    paramsLine(segs.value)
                }
            }
            Divider()
                .frame(width: 240)
                .padding(.vertical, 8)
            Text("Σ \(segs.total.formatted(.number)) params · \(arch.computeDataType.rawValue)")
                .font(.system(.caption, design: .monospaced).weight(.semibold))
                .monospacedDigit()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: Cells

    @ViewBuilder
    private func groupCell(_ g: BlockGroup, params: Int) -> some View {
        cell(width: barWidth(g.channels), emphasized: true) {
            line("×\(g.count) · @\(g.channels)ch", bold: true)
            line("conv \(g.conv1KernelSize)×\(g.conv1KernelSize) + \(g.conv2KernelSize)×\(g.conv2KernelSize)")
            line("\(seLabel(g)) · \(g.activationFunction.rawValue)/\(g.activationStyle.rawValue)")
            line("\(g.skipMerge.rawValue) · \(rezeroLabel(g))")
            line("drop×\(String(format: "%g", g.dropoutMultiplier))")
            paramsLine(params)
        }
    }

    private func seLabel(_ g: BlockGroup) -> String {
        switch g.seStyle {
        case .none: return "no-SE"
        case .attenuateOnly: return "SE/\(g.seReductionRatio)"
        case .scaleAndBias: return "SE+/\(g.seReductionRatio)"
        }
    }

    private func rezeroLabel(_ g: BlockGroup) -> String {
        g.useRezero ? "ReZero(\(String(format: "%.3g", g.rezeroAlphaInit)))" : "no-ReZero"
    }

    /// Marker for the optional feature skip: a single long concat skip carrying
    /// the source tensor (currently the stem output) into the routed head inputs.
    /// `concatDirect` adds no tensors — the routed heads' first convs already
    /// account for the widened input — so this marker is informational, no
    /// separate param count. Plain function (not `@ViewBuilder`): it builds the
    /// destination list imperatively and returns one styled `Text`.
    private func featureSkipMarker(_ arch: NetworkArchitecture, compressParams: Int) -> some View {
        let dests = [
            arch.featureSkipToPolicyHead ? "policy" : nil,
            arch.featureSkipToValueHead ? "value" : nil,
            arch.featureSkipToFinalBlock ? "finalBlock" : nil
        ].compactMap { $0 }.joined(separator: ",")
        // compress_conv_bn_relu adds a real fusion node; show its params. concatDirect
        // adds none (the routed heads/final-block widen in place), so show the +channels.
        let detail = compressParams > 0
            ? "\(compressParams.formatted(.number)) params"
            : "+\(arch.featureSkipSourceChannels)ch"
        return Text("⤳ skip \(arch.featureSkipSource.rawValue) → [\(dests)] · \(arch.featureSkipFusion.rawValue) (\(detail))")
            .font(.system(.caption2, design: .monospaced).weight(.semibold))
            .monospacedDigit()
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(Capsule().fill(Color.purple.opacity(0.18)))
            .overlay(Capsule().strokeBorder(Color.purple.opacity(0.5)))
    }

    @ViewBuilder
    private func transitionMarker(inC: Int, outC: Int) -> some View {
        Text("1×1 proj \(inC) → \(outC)")
            .font(.system(.caption2, design: .monospaced).weight(.semibold))
            .monospacedDigit()
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(Capsule().fill(Color.orange.opacity(0.18)))
            .overlay(Capsule().strokeBorder(Color.orange.opacity(0.5)))
    }

    @ViewBuilder
    private func cell<Content: View>(
        width: CGFloat, emphasized: Bool, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(spacing: 1) {
            content()
        }
        .padding(.vertical, 5)
        .padding(.horizontal, 8)
        .frame(width: width)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(emphasized ? Color.accentColor.opacity(0.10) : Color.primary.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(
                    emphasized ? Color.accentColor.opacity(0.45) : Color.primary.opacity(0.18)
                )
        )
    }

    @ViewBuilder
    private func line(_ text: String, bold: Bool = false) -> some View {
        Text(text)
            .font(.system(.caption2, design: .monospaced).weight(bold ? .semibold : .regular))
            .monospacedDigit()
            .lineLimit(1)
            .minimumScaleFactor(0.7)
    }

    @ViewBuilder
    private func paramsLine(_ count: Int) -> some View {
        Text("\(count.formatted(.number)) params")
            .font(.system(.caption2, design: .monospaced))
            .monospacedDigit()
            .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func connector() -> some View {
        Rectangle()
            .fill(Color.primary.opacity(0.3))
            .frame(width: 1, height: 10)
    }
}
