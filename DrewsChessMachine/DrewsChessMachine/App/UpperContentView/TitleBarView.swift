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
    /// The champion's name, preset and file format
    /// (`SessionController.championNameplate`); nil while its weights have
    /// no recorded origin.
    let championNameplate: ModelNameplate?
    let networkStatus: String
    let hasSavedCheckpoint: Bool
    let lastSavedDisplayString: String
    /// The Lichess bot, for its status chip. The chip observes it
    /// directly, so bot changes never re-render the rest of the bar.
    let lichessBot: LichessBotController
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
                AboutPopoverContent(network: network, nameplate: championNameplate)
            }
            if let arch = network?.network.arch {
                Text(TitleBarView.architectureText(arch: arch, nameplate: championNameplate))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            LichessBotStatusChip(controller: lichessBot)
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

extension TitleBarView {
    /// "my-net · preset v4_5block_7x7 (edited) · format v12 · 3-block 9×9 ·
    /// 128ch · 8,271,279 params": the nameplate's text, when the champion has
    /// one, before the topology.
    static func architectureText(arch: NetworkArchitecture, nameplate: ModelNameplate?) -> String {
        guard let nameplate else { return arch.shortLabel }
        return nameplate.headerText + " · " + arch.shortLabel
    }
}

extension TitleBarView: Equatable {
    nonisolated static func == (lhs: TitleBarView, rhs: TitleBarView) -> Bool {
        lhs.networkIdentifier == rhs.networkIdentifier
            && lhs.championNameplate == rhs.championNameplate
            && lhs.networkStatus == rhs.networkStatus
            && lhs.hasSavedCheckpoint == rhs.hasSavedCheckpoint
            && lhs.lastSavedDisplayString == rhs.lastSavedDisplayString
            && lhs.network === rhs.network
            && lhs.lichessBot === rhs.lichessBot
    }
}
