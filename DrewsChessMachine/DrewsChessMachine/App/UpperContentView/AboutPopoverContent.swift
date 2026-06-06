import SwiftUI

/// Popover content for the About menu item / button. Describes the **actual**
/// built network's architecture (not the static default) plus a couple of live
/// values from the current `ChessMPSNetwork` instance (network ID, build time)
/// when one exists.
struct AboutPopoverContent: View {
    let network: ChessMPSNetwork?

    /// The architecture of the live network, or the current preset when none is
    /// built yet — so the popover reflects what was actually built.
    private var arch: NetworkArchitecture { network?.network.arch ?? .current }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("About Drew's Chess Machine")
                .font(.headline)
            Text("Forward pass through a ~\(parameterCountMillionsText) parameter convolutional network using MPSGraph on the GPU.")
                .font(.callout)
            Divider()
            Text(arch.architectureSummary)
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
            Text("Parameters: \(parameterCountText)")
                .font(.system(.callout, design: .monospaced))
            if let net = network {
                Text("Network ID: \(net.identifier?.description ?? "–")")
                    .font(.system(.callout, design: .monospaced))
                Text("Build time: \(String(format: "%.1f ms", net.buildTimeMs))")
                    .font(.system(.callout, design: .monospaced))
            }
        }
        .padding(16)
        .frame(width: 540)
    }

    /// Exact persistent-tensor count with thousands separators plus a
    /// rounded-millions tag, e.g. "4,917,971 (~4.9M)". From the live arch.
    private var parameterCountText: String {
        let count = arch.parameterCount
        let grouped = count.formatted(.number.grouping(.automatic))
        return "\(grouped) (~\(parameterCountMillionsText))"
    }

    /// Rounded-millions form of the live arch's parameter count, e.g. "4.9M".
    private var parameterCountMillionsText: String {
        String(format: "%.1fM", Double(arch.parameterCount) / 1_000_000)
    }
}
