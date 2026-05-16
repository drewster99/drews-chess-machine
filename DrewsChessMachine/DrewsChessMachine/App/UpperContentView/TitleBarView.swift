import SwiftUI

/// Title bar at the top of `UpperContentView`: build/git summary +
/// info popover on the left, self-play network ID / status / last-
/// saved indicator on the right. Extracted from `UpperContentView`'s
/// monolithic body so a parent re-render driven by `trainingStats`
/// or `gameSnapshot` doesn't force this view to recompute its
/// body — `Equatable` conformance lets SwiftUI's diff skip the
/// body when none of these inputs changed.
struct TitleBarView: View {
    /// The live champion network. Held for the info popover only;
    /// NOT compared by reference in `==` because weight-copy events
    /// (e.g. arena promotion) mutate `network.identifier` in place
    /// without changing the instance. The identifier is captured at
    /// construction time into the separate `networkIdentifier`
    /// property below so the Equatable short-circuit reflects ID
    /// changes rather than instance changes.
    let network: ChessMPSNetwork?
    /// `network.identifier` snapshotted at the moment the parent
    /// reconstructed `TitleBarView`. Drives the displayed "Self play
    /// ID" text and the Equatable comparison.
    let networkIdentifier: ModelID?
    let networkStatus: String
    let hasSavedCheckpoint: Bool
    let lastSavedDisplayString: String
    @Binding var showingInfoPopover: Bool

    var body: some View {
        HStack(spacing: 8) {
            Text(BuildInfo.summary)
                .font(.callout)
                .foregroundStyle(.secondary)
            Button(action: { showingInfoPopover.toggle() }) {
                Image(systemName: "info.circle")
                    .font(.title3)
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showingInfoPopover) {
                AboutPopoverContent(network: network)
            }
            Spacer()
            if network != nil {
                Text("Self play ID: \(networkIdentifier?.description ?? "–")")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Text(networkStatus.isEmpty ? "" : networkStatus.components(separatedBy: "\n").first ?? "")
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            HStack(spacing: 4) {
                if hasSavedCheckpoint {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                Text(lastSavedDisplayString)
                    .font(.callout)
                    .foregroundStyle(hasSavedCheckpoint ? AnyShapeStyle(Color.green) : AnyShapeStyle(.secondary))
                    .lineLimit(1)
            }
        }
    }

}

extension TitleBarView: Equatable {
    nonisolated static func == (lhs: TitleBarView, rhs: TitleBarView) -> Bool {
        lhs.networkIdentifier == rhs.networkIdentifier
            && lhs.networkStatus == rhs.networkStatus
            && lhs.hasSavedCheckpoint == rhs.hasSavedCheckpoint
            && lhs.lastSavedDisplayString == rhs.lastSavedDisplayString
            && lhs.network === rhs.network
    }
}
