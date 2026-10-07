import Foundation

/// What the lineage picker chooses (follow-lineage plan §3.9): one model
/// file for the fixed-file source, or a segment whose run the follow-lineage
/// source follows from there on. Decides which rows are selectable and says
/// why the others are not, from the files' own lineage facts.
enum LichessBotModelLinePickerPurpose: Sendable, Equatable {
    case chooseFile
    case chooseLineageToFollow

    var title: String {
        switch self {
        case .chooseFile: return "Choose a model by lineage"
        case .chooseLineageToFollow: return "Choose a lineage to follow"
        }
    }

    /// The footer's text while nothing is selected.
    var selectionPrompt: String {
        switch self {
        case .chooseFile:
            return "Select a segment (its latest file), an earlier file, or a session champion"
        case .chooseLineageToFollow:
            return "Select a segment: follows this segment's run from here on, through every exact resume"
        }
    }

    var chooseButtonTitle: String {
        switch self {
        case .chooseFile: return "Use This Model"
        case .chooseLineageToFollow: return "Follow This Lineage"
        }
    }

    /// The file selecting `node` chooses — for following, the segment's
    /// latest file, whose lineage names the run and the anchor segment — or
    /// nil when the row can't be chosen for this purpose.
    func selectableEntry(for node: ModelLineageNode) -> ModelFileEntry? {
        switch self {
        case .chooseFile:
            return node.selectableEntry
        case .chooseLineageToFollow:
            guard case .segment(let line, _, _, _) = node.kind, Self.followablePosition(of: line.latest) != nil else {
                return nil
            }
            return line.latest
        }
    }

    /// Why `node` can't be chosen for this purpose, for the row's help; nil
    /// when it can.
    func unselectableReason(for node: ModelLineageNode) -> String? {
        guard selectableEntry(for: node) == nil else { return nil }
        switch (self, node.kind) {
        case (.chooseFile, .conflict), (.chooseLineageToFollow, .conflict):
            return "This model ID names more than one parent"
        case (.chooseFile, _):
            return nil
        case (.chooseLineageToFollow, .segment(let line, _, _, _)):
            return Self.unfollowableReason(of: line.latest)
        case (.chooseLineageToFollow, .file):
            return "Choose the segment row: a lineage is followed from a segment, not from one file"
        case (.chooseLineageToFollow, .sessionChampion):
            return "A GUI self-play session's champion — use the Champion or Live trainer source"
        }
    }

    /// The followed lineage the chosen entry names, for following.
    static func followedLineage(of entry: ModelFileEntry) -> LichessBotFollowedLineage? {
        followablePosition(of: entry).map { LichessBotFollowedLineage(lineageRunID: $0.lineageRunID, anchorSegmentID: $0.segmentID) }
    }

    private static func followablePosition(of entry: ModelFileEntry) -> ModelFileLineagePosition? {
        guard case .recorded(let position)? = entry.lineage, ModelLineageTip.followablePathKinds.contains(position.pathKind) else {
            return nil
        }
        return position
    }

    private static func unfollowableReason(of entry: ModelFileEntry) -> String {
        switch entry.lineage {
        case nil:
            return "No lineage facts were read for this file"
        case .unrecorded?:
            return "Written before lineage records: its run is unknown"
        case .unreadable(let reason)?:
            return "Unreadable lineage: \(reason)"
        case .recorded(let position)?:
            if position.pathKind == .gui {
                return "A GUI run — use the Champion or Live trainer source"
            }
            return "Written by \(position.pathKind.rawValue), not a training run"
        }
    }
}
