import SwiftUI

// MARK: - Content View (composer)

/// Top-level window content. Owns the shared `ChartCoordinator` and
/// composes two sibling child views:
///
///   - `UpperContentView`: status, controls, board, parameter
///     editors, alarm banners — every UI element except the chart
///     grid. Drives the heartbeat that pushes new chart samples
///     into the coordinator.
///   - `LowerContentView`: the chart-zoom row + chart grid. Reads
///     the coordinator's decimated frame and binds to its scroll
///     position, hover, and zoom state. Mounted only while the
///     coordinator's `isActive` flag is true (mirrored from
///     `UpperContentView.realTraining`).
///
/// Carving the chart layer into a sibling view scopes its SwiftUI
/// body invalidation: when any unrelated `UpperContentView` `@State`
/// changes, the coordinator's chart-relevant fields don't change,
/// so SwiftUI skips re-evaluating `LowerContentView.body` even
/// though the parent (`ContentView`) re-ran.
struct ContentView: View {
    let commandHub: AppCommandHub
    let autoTrainOnLaunch: Bool
    let cliConfig: CliTrainingConfig?
    let cliOutputURL: URL?
    /// View > Show Training Graphs preference, forwarded from
    /// `DrewsChessMachineApp`'s `@AppStorage`. Gates the lower
    /// chart pane independently of `chartCoordinator.isActive`
    /// (which only reflects whether chart data is being collected),
    /// so the user can hide the pane during training to reclaim
    /// vertical space without stopping data capture.
    let showTrainingGraphs: Bool
    /// View > Collect Chart Data preference, forwarded from
    /// `DrewsChessMachineApp`'s `@AppStorage`. When `false`, the
    /// coordinator drops every per-tick sample on the floor and the
    /// underlying ring buffers stay at zero element storage —
    /// intended for clean perf-isolation runs where chart bookkeeping
    /// must not perturb the training hot path. Also force-hides the
    /// lower pane regardless of `showTrainingGraphs`, since rendering
    /// an empty axis with no data is just visual noise.
    let chartCollectionEnabled: Bool

    /// View > Show Policy Channels Panel preference, forwarded from
    /// `DrewsChessMachineApp`'s `@AppStorage`. When `true`, the
    /// 76-channel panel inside `UpperContentView` expands to fill
    /// the chart-pane area and `LowerContentView` is dropped from
    /// the layout entirely so the panel really does take over the
    /// bottom of the window.
    let showPolicyChannelsPanel: Bool

    /// Single source of truth for chart-layer state. Held here
    /// (rather than on `UpperContentView`) so it can be passed to
    /// both child views as a shared reference. Initial allocation
    /// happens once when SwiftUI constructs `ContentView`'s storage
    /// — the rings inside the coordinator each pre-reserve a 24h
    /// block, so subsequent appends stay reallocation-free.
    @State private var chartCoordinator = ChartCoordinator()

    var body: some View {
        VStack(spacing: 0) {
            UpperContentView(
                commandHub: commandHub,
                autoTrainOnLaunch: autoTrainOnLaunch,
                cliConfig: cliConfig,
                cliOutputURL: cliOutputURL,
                chartCoordinator: chartCoordinator
            )
            .frame(minHeight: 400)
            // The chart pane is dropped entirely when the
            // policy-channels panel is on — it's expanded to
            // take over the freed space inside UpperContentView,
            // so leaving even a zero-height LowerContentView in
            // the layout would still steal a divider line.
            if !showPolicyChannelsPanel {
                // Spacer comes BEFORE the chart pane so any extra
                // vertical space the window provides above the chart
                // pane's natural height pushes the chart pane down to
                // the window's bottom edge instead of leaving a gap
                // below the charts. When the chart pane is hidden
                // (training off, or training graphs disabled) the
                // Spacer collapses to 0 so the upper area can occupy
                // the full window height as before.
                Spacer()
                    .frame(maxHeight: (effectiveShowTrainingGraphs && chartCoordinator.isActive) ? nil : 0)
                VStack {
                    Divider()
                    LowerContentView(
                        promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
                        replayRatioTarget: TrainingParameters.shared.replayRatioTarget,
                        gradClipMaxNorm: TrainingParameters.shared.gradClipMaxNorm,
                        appMemoryTotalGB: Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024),
                        gpuMemoryTotalGB: Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024),
                        chartCoordinator: chartCoordinator
                    )
                }
                .opacity((effectiveShowTrainingGraphs && chartCoordinator.isActive) ? 1.0 : 0.0)
                .frame(height: !effectiveShowTrainingGraphs ? 0 : (chartCoordinator.isActive ? nil : 250))
                .padding(.bottom, 4)
            }
        }
        .onAppear {
            // Bootstrap the coordinator's gate from the @AppStorage
            // value at first appearance — the coordinator's init also
            // reads UserDefaults directly, but mirroring here keeps
            // the two sources of truth aligned even on the first frame.
            chartCoordinator.collectionEnabled = chartCollectionEnabled
        }
        .onChange(of: chartCollectionEnabled) { _, newValue in
            // Live-tunable: flipping off mid-run stops new appends
            // immediately; flipping on resumes from the next tick.
            // Existing samples in the rings (if any) are left alone
            // so a brief perf-isolation toggle doesn't lose history.
            chartCoordinator.collectionEnabled = newValue
        }
    }

    /// `showTrainingGraphs` AND-ed with `chartCollectionEnabled`:
    /// rendering the lower pane with no data flowing into it is just
    /// noise, so a disabled-collection state implies a hidden pane
    /// regardless of the user's Show Training Graphs preference.
    private var effectiveShowTrainingGraphs: Bool {
        showTrainingGraphs && chartCollectionEnabled
    }
}

#Preview {
    ContentView(
        commandHub: AppCommandHub(),
        autoTrainOnLaunch: false,
        cliConfig: nil,
        cliOutputURL: nil,
        showTrainingGraphs: true,
        chartCollectionEnabled: true,
        showPolicyChannelsPanel: false
    )
}
