import SwiftUI

/// Popover content for the About menu item / button. Static text
/// describing the network architecture plus a couple of live values
/// from the current `ChessMPSNetwork` instance (network ID, build
/// time) when one exists.
struct AboutPopoverContent: View {
    let network: ChessMPSNetwork?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("About Drew's Chess Machine")
                .font(.headline)
            Text("Forward pass through a ~\(Self.parameterCountMillionsText) parameter convolutional network using MPSGraph on the GPU. Weights are randomly initialized (He initialization) — no training has occurred.")
                .font(.callout)
            Divider()
            Text("Architecture: \(ChessNetwork.inputPlanes)×\(ChessNetwork.boardSize)×\(ChessNetwork.boardSize) input → stem(\(ChessNetwork.channels)) → \(ChessNetwork.numBlocks) res+SE blocks → policy(\(ChessNetwork.policySize)) + value(\(ChessNetwork.valueHeadClasses))")
                .font(.system(.callout, design: .monospaced))
            Text("Parameters: \(Self.parameterCountText)")
                .font(.system(.callout, design: .monospaced))
            if let net = network {
                Text("Network ID: \(net.identifier?.description ?? "–")")
                    .font(.system(.callout, design: .monospaced))
                Text("Build time: \(String(format: "%.1f ms", net.buildTimeMs))")
                    .font(.system(.callout, design: .monospaced))
            }
        }
        .padding(16)
        .frame(width: 500)
    }

    /// Exact persistent-tensor count with thousands separators plus a
    /// rounded-millions tag, e.g. "4,917,971 (~4.9M)". Derived from
    /// `ChessNetwork.parameterCount` so it tracks the architecture.
    private static var parameterCountText: String {
        let count = ChessNetwork.parameterCount
        let grouped = count.formatted(.number.grouping(.automatic))
        return "\(grouped) (~\(parameterCountMillionsText))"
    }

    /// Rounded-millions form of `ChessNetwork.parameterCount`, e.g. "4.9M".
    private static var parameterCountMillionsText: String {
        let millions = Double(ChessNetwork.parameterCount) / 1_000_000
        return String(format: "%.1fM", millions)
    }
}
