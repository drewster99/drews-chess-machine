import SwiftUI

/// Pick a model file by lineage (plan §9.1): a tree from each root seed
/// through its training segments (by `parent_model_id`), each segment
/// selecting its latest file and listing its earlier files, plus self-play
/// session champions under their base ModelID. Built only from the files'
/// metadata, never from their names. With `.chooseLineageToFollow` only a
/// training segment whose latest file records a followable lineage can be
/// chosen (follow-lineage plan §3.9); every other row says why not.
struct LichessBotModelLinePicker: View {
    @Binding var isPresented: Bool
    let purpose: LichessBotModelLinePickerPurpose
    let onChoose: (ModelFileEntry) -> Void

    @State private var tree: [ModelLineageNode]?
    @State private var modelCount = (lineages: 0, files: 0, sessions: 0)
    @State private var unreadable: [UnreadableModelFile] = []
    @State private var scanError: String?
    @State private var sessionsError: String?
    @State private var search = ""
    @State private var selection: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(purpose.title)
                .font(.title2.weight(.semibold))
            HStack {
                TextField("Filter by model ID or session (a seed's ID shows its whole family)", text: $search)
                    .textFieldStyle(.roundedBorder)
                Text(summary)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Text(sessionsError ?? "")
                .font(.callout)
                .foregroundStyle(.orange)
                .shown(sessionsError != nil)
            ZStack {
                ProgressView("Reading model headers…")
                    .shown(tree == nil && scanError == nil)
                Text(scanError ?? "")
                    .foregroundStyle(.red)
                    .shown(scanError != nil)
                List(selection: $selection) {
                    LichessBotUnreadableModelFilesSection(files: unreadable)
                    OutlineGroup(filteredTree, children: \.children) { node in
                        LichessBotLineageNodeRow(node: node)
                            .tag(node.id)
                            .selectionDisabled(purpose.selectableEntry(for: node) == nil)
                            .help(purpose.unselectableReason(for: node) ?? "")
                    }
                }
                .shown(tree != nil)
            }
            .frame(minHeight: 380)
            HStack {
                Text(selectedEntry?.url.lastPathComponent ?? purpose.selectionPrompt)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)
                Button(purpose.chooseButtonTitle) {
                    if let selectedEntry {
                        onChoose(selectedEntry)
                        isPresented = false
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(selectedEntry == nil)
            }
        }
        .padding(20)
        // Widened by the training method column
        // (`LichessBotModelFileRow.trainingMethodWidth`), so the other
        // columns keep the room they had before it.
        .frame(width: 1100 + LichessBotModelFileRow.trainingMethodWidth, height: 760)
        .task {
            await load()
        }
    }

    private func load() async {
        let scan: ModelFileCatalog.Scan
        do {
            scan = try await ModelFileCatalog.scan(directory: CheckpointPaths.modelsDir)
        } catch {
            scanError = "Could not list \(CheckpointPaths.modelsDir.path): \(error.localizedDescription)"
            return
        }
        var champions: [SessionChampion] = []
        var unreadableFiles = scan.unreadable
        do {
            let sessions = try await ModelFileCatalog.scanSessionChampionsInBackground(directory: CheckpointPaths.sessionsDir)
            champions = sessions.champions
            unreadableFiles += sessions.unreadable
        } catch {
            sessionsError = "Session champions not shown: could not list \(CheckpointPaths.sessionsDir.path): \(error.localizedDescription)"
        }
        for file in unreadableFiles {
            SessionLogger.shared.log("[MODELS] unreadable model file \(file.url.path): \(file.reason)")
        }
        unreadable = unreadableFiles
        modelCount = (scan.lines.count, scan.lines.reduce(0) { $0 + $1.files.count }, champions.count)
        tree = ModelLineageTree.build(lines: scan.lines, champions: champions)
    }

    private var filteredTree: [ModelLineageNode] {
        guard let tree else { return [] }
        let query = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !query.isEmpty else { return tree }
        return tree.filter { root in root.searchableIDs.contains { $0.contains(query) } }
    }

    private var selectedEntry: ModelFileEntry? {
        guard let selection, let tree, let node = Self.find(selection, in: tree) else { return nil }
        return purpose.selectableEntry(for: node)
    }

    private static func find(_ id: String, in nodes: [ModelLineageNode]) -> ModelLineageNode? {
        for node in nodes {
            if node.id == id { return node }
            if let found = find(id, in: node.children ?? []) { return found }
        }
        return nil
    }

    private var summary: String {
        guard tree != nil else { return "" }
        return "\(modelCount.lineages) model IDs · \(modelCount.files) files · \(modelCount.sessions) session champions"
    }
}

/// One row of the lineage tree.
struct LichessBotLineageNodeRow: View {
    let node: ModelLineageNode

    var body: some View {
        switch node.kind {
        case .segment(let line, let path, let isUntrained, let isBranchTip):
            LichessBotLineageSegmentRow(line: line, path: path, isUntrained: isUntrained, isBranchTip: isBranchTip, trainedBelow: node.trainedBelow)
        case .file(let entry):
            LichessBotModelFileRow(file: entry, showsModelID: false)
        case .sessionChampion(let champion):
            HStack(spacing: 12) {
                LichessBotModelFileRow(file: champion.entry, showsModelID: true)
                Text("self-play session \(champion.sessionName)")
                    .font(.caption)
                    .foregroundStyle(.purple)
                    .lineLimit(1)
            }
        case .conflict(let modelID, let parents):
            Label("\(modelID) names more than one parent: \(parents.joined(separator: ", "))", systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
        }
    }
}

/// A segment: its latest file, its step range and file count, and — for a
/// branch tip — the whole chain from the seed.
struct LichessBotLineageSegmentRow: View {
    let line: ModelLine
    let path: [String]
    let isUntrained: Bool
    let isBranchTip: Bool
    let trainedBelow: ModelLineageNode.TrainedBelow

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 12) {
                LichessBotModelFileRow(file: line.latest, showsModelID: true)
                Text(tagText)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(ModelLineageNode.isSeedOnly(isUntrained: isUntrained, trainedBelow: trainedBelow) ? Color.orange : Color.green)
            }
            Text(chainText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .shown(isBranchTip && path.count > 1)
        }
    }

    private var tagText: String {
        if isUntrained {
            guard !trainedBelow.isEmpty else {
                return "untrained (no training_step)"
            }
            var below: [String] = []
            if trainedBelow.trainedSegments > 0 {
                below.append("\(trainedBelow.trainedSegments) trained segment\(trainedBelow.trainedSegments == 1 ? "" : "s")")
            }
            if trainedBelow.sessionChampions > 0 {
                below.append("\(trainedBelow.sessionChampions) session champion\(trainedBelow.sessionChampions == 1 ? "" : "s")")
            }
            return "untrained seed · \(below.joined(separator: " · ")) below"
        }
        let steps = line.files.compactMap(\.trainingStep)
        var range = ""
        if let low = steps.min(), let high = steps.max() {
            range = low == high ? "step \(low)" : "steps \(low)–\(high)"
        }
        let kind = line.latest.creator.map { " · \($0)" } ?? ""
        return "\(line.files.count) file\(line.files.count == 1 ? "" : "s") · \(range)\(kind)\(isBranchTip ? " · latest on branch" : "")"
    }

    /// Each ModelID's final component (the part that names the line), seed
    /// first, joined by arrows.
    private var chainText: String {
        path.map { id in
            guard let last = id.split(separator: "-").last else { return id }
            return String(last)
        }.joined(separator: " → ")
    }
}

/// The model files that could not be read, each with its reason, first in
/// the list and expanded so a bad file is never silently missing from its
/// lineage. Not selectable: none of them can be loaded as-is.
struct LichessBotUnreadableModelFilesSection: View {
    let files: [UnreadableModelFile]
    @State private var isExpanded = true

    var body: some View {
        // A zero-size row still takes a List row's height, so the group is
        // present only when there is something to report.
        Section {
            ForEach(files.isEmpty ? [] : [files.count], id: \.self) { count in
                DisclosureGroup(
                    isExpanded: $isExpanded,
                    content: {
                        ForEach(files) { file in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(file.displayName)
                                    .font(.system(.callout, design: .monospaced))
                                    .textSelection(.enabled)
                                    .help(file.url.path)
                                Text(file.reason)
                                    .font(.callout)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                            .selectionDisabled()
                        }
                    },
                    label: {
                        Label("\(count) model file\(count == 1 ? "" : "s") could not be read", systemImage: "exclamationmark.triangle.fill")
                            .font(.callout.weight(.semibold))
                            .foregroundStyle(.orange)
                    }
                )
                .selectionDisabled()
            }
        }
    }
}

/// One model file's identity and strength: model ID, training step, date,
/// how it was trained (`ModelTrainingHistory`; blank when unknown), its
/// largest test set's figures (`ModelTestSetSummary`), architecture.
struct LichessBotModelFileRow: View {
    let file: ModelFileEntry
    let showsModelID: Bool

    var body: some View {
        // Every column is one line, so every row has the same height and
        // the disclosure chevron lines up with the text.
        HStack(spacing: 16) {
            Text(showsModelID ? file.modelID : "")
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .frame(width: 190, alignment: .leading)
            Text(file.trainingStep.map { "step \($0)" } ?? "no step")
                .font(.system(.callout, design: .monospaced))
                .frame(width: 130, alignment: .trailing)
            Text(Self.dateFormatter.string(from: file.createdAt ?? file.fileModifiedAt))
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 140, alignment: .leading)
            Text(file.trainingHistory?.displayText ?? "")
                .font(.callout)
                .frame(width: Self.trainingMethodWidth, alignment: .leading)
                .help(file.trainingHistory?.displayText.map { "Trained by \($0)" } ?? "Training method not recorded")
            ModelTestSetSummaryCells(summary: file.testSets ?? .notRecorded, textStyle: .callout)
            Text(file.architectureLabel)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .help(file.url.lastPathComponent)
    }

    /// Wide enough for a two-method chain ("corpus replay → self-play");
    /// a longer one truncates, with the whole chain in the tooltip.
    static let trainingMethodWidth: CGFloat = 190

    /// Fixed-width local date and time, so the column never wraps.
    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter
    }()
}
