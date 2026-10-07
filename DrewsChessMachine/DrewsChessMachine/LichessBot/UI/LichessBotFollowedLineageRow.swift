import SwiftUI

/// The follow-lineage source's chosen lineage in the model settings
/// (follow-lineage plan §3.9): which run and segment it follows, and what
/// that lineage's newest file is. The newest file comes from the running
/// bot's last check while the bot is online and plays exactly this lineage;
/// otherwise — the bot offline, or a lineage chosen in the draft but not yet
/// applied — the row reads the models folder's headers once, off the main
/// actor, and shows what a check would select. Nothing shown is stored in
/// settings.
struct LichessBotFollowedLineageRow: View {
    @Binding var followedLineage: LichessBotFollowedLineage?
    let isEnabled: Bool
    let controller: LichessBotController
    /// The folder whose headers the row reads (`Models/` in the app).
    let modelsDirectory: URL

    @State private var showingPicker = false
    @State private var resolution = LichessBotFollowedLineageResolution.idle

    var body: some View {
        LabeledContent("Lineage") {
            HStack(alignment: .firstTextBaseline) {
                LichessBotFollowedLineageSummary(followedLineage: followedLineage, status: shownStatus, resolution: resolution)
                Button("Choose Lineage…") {
                    showingPicker = true
                }
            }
        }
        .disabled(!isEnabled)
        .sheet(isPresented: $showingPicker) {
            LichessBotModelLinePicker(isPresented: $showingPicker, purpose: .chooseLineageToFollow) { entry in
                followedLineage = LichessBotModelLinePickerPurpose.followedLineage(of: entry)
            }
        }
        .task(id: LichessBotFollowedLineageResolution.Key(followed: followedLineage, live: liveStatus != nil)) {
            await resolve()
        }
    }

    /// The running bot's status, when it describes this row's lineage as
    /// the lineage in force.
    private var liveStatus: LichessBotLineageFollowStatus? {
        guard controller.isRunning, let followedLineage else { return nil }
        let applied: LichessBotFollowedLineage? = controller.settings.model.followedLineage
        guard applied == followedLineage, let status = controller.lineageFollowStatus else { return nil }
        return status.followed == followedLineage ? status : nil
    }

    private var shownStatus: LichessBotLineageFollowStatus? {
        liveStatus ?? resolution.status
    }

    private func resolve() async {
        guard let followedLineage, liveStatus == nil else {
            resolution = .idle
            return
        }
        resolution = .reading
        resolution = await LichessBotFollowedLineageResolution.resolve(followedLineage, in: modelsDirectory)
    }
}

/// What the row found reading the models folder for a lineage the running
/// bot isn't checking.
enum LichessBotFollowedLineageResolution: Equatable {
    case idle
    case reading
    case resolved(LichessBotLineageFollowStatus, anchorModelID: String?)

    /// What the row's resolution depends on: the lineage, and whether the
    /// running bot reports it instead.
    struct Key: Equatable {
        let followed: LichessBotFollowedLineage?
        let live: Bool
    }

    var status: LichessBotLineageFollowStatus? {
        if case .resolved(let status, _) = self { return status }
        return nil
    }

    var anchorModelID: String? {
        if case .resolved(_, let anchorModelID) = self { return anchorModelID }
        return nil
    }

    /// One scan of `directory` and the selection a check would make, through
    /// the follow-lineage source's own code (`LichessBotLineageFollower`),
    /// so the row and the bot can never disagree on what "newest" is. An
    /// unreadable folder is reported as that outcome, never as "none".
    static func resolve(_ followed: LichessBotFollowedLineage, in directory: URL) async -> LichessBotFollowedLineageResolution {
        let scan: Result<ModelFolderScan, Error>
        do {
            scan = .success(try await ModelFolderHeaderCache.scan(directory: directory, previous: .empty))
        } catch {
            scan = .failure(error)
        }
        var follower = LichessBotLineageFollower()
        let decision = follower.record(scan, followed: followed, notBelow: nil, checkedAt: Date(), now: .zero, consequence: "")
        let anchorModelID: String?
        if case .success(let result) = scan {
            anchorModelID = result.entries.first { entry in
                guard case .recorded(let position)? = entry.lineage else { return false }
                return position.segmentID == followed.anchorSegmentID
            }?.modelID
        } else {
            anchorModelID = nil
        }
        return .resolved(decision.status, anchorModelID: anchorModelID)
    }
}

/// The lineage row's text: the run and anchor, then the newest file or the
/// problem.
struct LichessBotFollowedLineageSummary: View {
    let followedLineage: LichessBotFollowedLineage?
    let status: LichessBotLineageFollowStatus?
    let resolution: LichessBotFollowedLineageResolution

    var body: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(identityText)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .help(helpText)
            Text(detailText)
                .font(.caption)
                .foregroundStyle(isProblem ? Color.orange : Color.secondary)
                .lineLimit(2)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
    }

    private var identityText: String {
        guard let followedLineage else { return "None" }
        let anchor = resolution.anchorModelID.map { "from \($0)" } ?? "from segment \(followedLineage.anchorSegmentID.prefix(8))"
        return "run \(followedLineage.lineageRunID.prefix(8)) \(anchor)"
    }

    private var helpText: String {
        guard let followedLineage else { return "" }
        return "run \(followedLineage.lineageRunID)\nsegment \(followedLineage.anchorSegmentID)"
    }

    private var detailText: String {
        guard followedLineage != nil else { return "" }
        if let status {
            return status.outcome.description
        }
        return resolution == .reading ? "Reading model headers…" : ""
    }

    private var isProblem: Bool {
        status?.outcome.isProblem ?? false
    }
}
