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
            Text("Forward pass through a ~2.4M parameter convolutional network using MPSGraph on the GPU. Weights are randomly initialized (He initialization) — no training has occurred.")
                .font(.callout)
            Divider()
            Text("Architecture: 20×8×8 input → stem(128) → 8 res+SE blocks → policy(4864) + value(1)")
                .font(.system(.callout, design: .monospaced))
            Text("Parameters: ~2,400,000 (~2.4M)")
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
}
