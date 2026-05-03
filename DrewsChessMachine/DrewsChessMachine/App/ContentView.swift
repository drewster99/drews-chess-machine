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
            VStack {
                Divider()
                LowerContentView(
                    promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
                    replayRatioTarget: TrainingParameters.shared.replayRatioTarget,
                    appMemoryTotalGB: Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024),
                    gpuMemoryTotalGB: Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024),
                    chartCoordinator: chartCoordinator
                )
            }
            .opacity((showTrainingGraphs && chartCoordinator.isActive) ? 1.0 : 0.0)
            .frame(height: !showTrainingGraphs ? 0 : (chartCoordinator.isActive ? nil : 250))
            Spacer()
                .frame(maxHeight: (showTrainingGraphs && chartCoordinator.isActive) ? nil : 0)
        }
    }
}

#Preview {
    ContentView(
        commandHub: AppCommandHub(),
        autoTrainOnLaunch: false,
        cliConfig: nil,
        cliOutputURL: nil,
        showTrainingGraphs: true
    )
}
