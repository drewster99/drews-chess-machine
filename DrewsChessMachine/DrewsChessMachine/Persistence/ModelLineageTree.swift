import Foundation

/// One row of the model-lineage tree (plan §9.1).
struct ModelLineageNode: Identifiable, Sendable, Equatable {
    enum Kind: Sendable, Equatable {
        /// One `model_id`'s files. Selecting the row selects its latest file;
        /// the earlier files are its first children. `path` runs from the
        /// root seed to this segment.
        case segment(line: ModelLine, path: [String], isUntrained: Bool, isBranchTip: Bool)
        /// An earlier file of the enclosing segment.
        case file(ModelFileEntry)
        /// A self-play session's champion, placed under its base ModelID.
        case sessionChampion(SessionChampion)
        /// A `model_id` whose files name more than one parent. Never placed
        /// under either; its segment is this row's child.
        case conflict(modelID: String, parents: [String])
    }

    /// What was trained anywhere below a row. A seed's own file is untrained
    /// (a fresh build records no step) while its trained segments sit under
    /// it (collapsed, or as the next row when there is only one), so its row
    /// has to say what is under it or it reads as if the whole lineage were
    /// untrained.
    struct TrainedBelow: Sendable, Equatable {
        /// Descendant segments with at least one file that records a step.
        var trainedSegments = 0
        /// Self-play session champions placed anywhere below.
        var sessionChampions = 0

        var isEmpty: Bool { trainedSegments == 0 && sessionChampions == 0 }
    }

    let id: String
    let kind: Kind
    /// nil for a leaf, and in a picker's listing for a row with one row
    /// below it (`ModelLineageTree.flatteningSoleChildren`), so
    /// `OutlineGroup` shows no disclosure arrow.
    let children: [ModelLineageNode]?
    /// Counted once here from the children, which are built first, since
    /// every row's label reads it; copied unchanged by `init(listing:)`.
    let trainedBelow: TrainedBelow
    /// Listed at its parent's level, right below it, because it is the
    /// parent's only row (`ModelLineageTree.flatteningSoleChildren`).
    let isSoleChildOfRowAbove: Bool

    init(id: String, kind: Kind, children: [ModelLineageNode]?) {
        self.id = id
        self.kind = kind
        self.children = children
        self.isSoleChildOfRowAbove = false
        var below = TrainedBelow()
        for child in children ?? [] {
            below.trainedSegments += child.trainedBelow.trainedSegments
            below.sessionChampions += child.trainedBelow.sessionChampions
            switch child.kind {
            case .segment(_, _, let isUntrained, _):
                if !isUntrained { below.trainedSegments += 1 }
            case .sessionChampion:
                below.sessionChampions += 1
            case .file, .conflict:
                break
            }
        }
        self.trainedBelow = below
    }

    /// `node` kept on a search's path with only `children` (the rows below
    /// it that lead to a match): the tree's counts, so its label still says
    /// what was trained below it.
    fileprivate init(pruning node: ModelLineageNode, children: [ModelLineageNode]) {
        self.id = node.id
        self.kind = node.kind
        self.children = children
        self.trainedBelow = node.trainedBelow
        self.isSoleChildOfRowAbove = false
    }

    /// A branch tip listed above the tree (`recentBranchTips`): the same
    /// row, no children, a distinct id.
    fileprivate init(tipOf node: ModelLineageNode) {
        self.id = ModelLineageTree.tipIDPrefix + node.id
        self.kind = node.kind
        self.children = nil
        self.trainedBelow = node.trainedBelow
        self.isSoleChildOfRowAbove = false
    }

    /// This row moved by `flatteningSoleChildren`: its counts stay the
    /// tree's, so a seed whose one trained segment now sits beside it still
    /// says what was trained below it.
    fileprivate init(listing node: ModelLineageNode, children: [ModelLineageNode]?, isSoleChildOfRowAbove: Bool) {
        self.id = node.id
        self.kind = node.kind
        self.children = children
        self.trainedBelow = node.trainedBelow
        self.isSoleChildOfRowAbove = isSoleChildOfRowAbove
    }

    /// An untrained segment with nothing trained anywhere below it: what the
    /// tree sorts after the trained roots and the picker tags orange. The one
    /// rule for both.
    static func isSeedOnly(isUntrained: Bool, trainedBelow: TrainedBelow) -> Bool {
        isUntrained && trainedBelow.isEmpty
    }

    var isSeedOnly: Bool {
        guard case .segment(_, _, let isUntrained, _) = kind else { return false }
        return Self.isSeedOnly(isUntrained: isUntrained, trainedBelow: trainedBelow)
    }

    /// The file selecting this row chooses, if any.
    var selectableEntry: ModelFileEntry? {
        switch kind {
        case .segment(let line, _, _, _): return line.latest
        case .file(let entry): return entry
        case .sessionChampion(let champion): return champion.entry
        case .conflict: return nil
        }
    }

    /// This row's own searchable texts, lowercased: its model id (and a
    /// conflict's parents), a session's name, and the file names it stands
    /// for — a segment's latest file, an earlier file, a champion's file — so
    /// a run's file stem ("zlra-elitecyc") finds it (issue #20).
    var ownSearchTexts: [String] {
        let own: [String]
        switch kind {
        case .segment(let line, _, _, _): own = [line.modelID, line.latest.url.lastPathComponent]
        case .file(let entry): own = [entry.url.lastPathComponent]
        case .sessionChampion(let champion): own = [champion.entry.modelID, champion.sessionName, champion.entry.url.lastPathComponent]
        case .conflict(let modelID, let parents): own = [modelID] + parents
        }
        return own.map { $0.lowercased() }
    }

    /// Every searchable text in this subtree (`ownSearchTexts` of each row):
    /// a seed's id, or any file name below it, finds the whole family.
    var searchableIDs: [String] {
        ownSearchTexts + (children ?? []).flatMap(\.searchableIDs)
    }

    /// Whether this row itself matches a lowercased, trimmed query.
    func ownTextsContain(_ query: String) -> Bool {
        ownSearchTexts.contains { $0.contains(query) }
    }

    /// Whether this row's own model ID matches a lowercased, trimmed query:
    /// a segment's, a session champion's or a conflict's (a file row has
    /// none of its own). A model-ID match shows the row's whole family.
    func ownModelIDContains(_ query: String) -> Bool {
        switch kind {
        case .segment(let line, _, _, _): return line.modelID.lowercased().contains(query)
        case .sessionChampion(let champion): return champion.entry.modelID.lowercased().contains(query)
        case .conflict(let modelID, _): return modelID.lowercased().contains(query)
        case .file: return false
        }
    }

    /// The newest file anywhere in this subtree, for ordering.
    var newestActivity: Date {
        var own: Date
        switch kind {
        case .segment(let line, _, _, _): own = line.newestFileModifiedAt
        case .file(let entry): own = entry.fileModifiedAt
        case .sessionChampion(let champion): own = champion.entry.fileModifiedAt
        case .conflict: own = .distantPast
        }
        for child in children ?? [] {
            own = max(own, child.newestActivity)
        }
        return own
    }
}

/// Builds the seed → branch → segment tree from the files' own metadata
/// (`model_id`, `parent_model_id`, `training_step`), never from filenames
/// or the dashboards' registries (plan §9.1).
enum ModelLineageTree {

    static func build(lines: [ModelLine], champions: [SessionChampion]) -> [ModelLineageNode] {
        var parentsByID: [String: Set<String>] = [:]
        for line in lines {
            parentsByID[line.modelID] = Set(line.files.compactMap(\.parentModelID))
        }
        let knownIDs = Set(lines.map(\.modelID))
        var childIDs: [String: [String]] = [:]
        var roots: [String] = []
        var conflicts: [(modelID: String, parents: [String])] = []
        for line in lines {
            let parents = parentsByID[line.modelID] ?? []
            if parents.count > 1 {
                conflicts.append((line.modelID, parents.sorted()))
            } else if let parent = parents.first, knownIDs.contains(parent), parent != line.modelID {
                childIDs[parent, default: []].append(line.modelID)
            } else {
                roots.append(line.modelID)
            }
        }
        let lineByID = Dictionary(lines.map { ($0.modelID, $0) }, uniquingKeysWith: { first, _ in first })
        var championsByBase: [String: [SessionChampion]] = [:]
        var orphanChampions: [SessionChampion] = []
        for champion in champions {
            let base = baseModelID(ofChampion: champion.entry.modelID)
            if knownIDs.contains(base) {
                championsByBase[base, default: []].append(champion)
            } else {
                orphanChampions.append(champion)
            }
        }

        var visited: Set<String> = []
        func segmentNode(_ modelID: String, path: [String]) -> ModelLineageNode? {
            // A parent cycle would recurse forever; each id is placed once.
            guard let line = lineByID[modelID], visited.insert(modelID).inserted else { return nil }
            let fullPath = path + [modelID]
            let childSegments = (childIDs[modelID] ?? [])
                .compactMap { segmentNode($0, path: fullPath) }
                .sorted { $0.newestActivity > $1.newestActivity }
            let earlierFiles = line.files.dropFirst().map { ModelLineageNode(id: "file:\($0.url.path)", kind: .file($0), children: nil) }
            let sessionNodes = (championsByBase[modelID] ?? []).map(championNode)
            // Child lineages first: a branch continued below this segment is
            // what a reader looks for, and listed after the segment's earlier
            // files it sat below dozens of file rows (issue #20).
            let children = childSegments + earlierFiles + sessionNodes
            let isUntrained = line.files.allSatisfy { $0.trainingStep == nil }
            return ModelLineageNode(
                id: "segment:\(modelID)",
                kind: .segment(line: line, path: fullPath, isUntrained: isUntrained, isBranchTip: childSegments.isEmpty),
                children: children.isEmpty ? nil : children
            )
        }

        let rootNodes = roots.compactMap { segmentNode($0, path: []) }
        let conflictNodes = conflicts.map { conflict in
            ModelLineageNode(
                id: "conflict:\(conflict.modelID)",
                kind: .conflict(modelID: conflict.modelID, parents: conflict.parents),
                children: segmentNode(conflict.modelID, path: []).map { [$0] }
            )
        }
        // Ids reachable only through a cycle were never visited; list them
        // as roots rather than dropping them.
        let unplaced = lines.map(\.modelID).filter { !visited.contains($0) }.compactMap { segmentNode($0, path: []) }

        let trainedFirst = (rootNodes + unplaced).sorted { lhs, rhs in
            if lhs.isSeedOnly != rhs.isSeedOnly { return !lhs.isSeedOnly }
            return lhs.newestActivity > rhs.newestActivity
        }
        return conflictNodes + trainedFirst + orphanChampions.map(championNode)
    }

    /// The rows as a picker lists them: a row with exactly one row below it
    /// gets no disclosure, and that row follows it at the same level, marked
    /// `isSoleChildOfRowAbove` (owner request 2026-10-09: a one-item group
    /// is never something to expand). Applied at every depth, so a chain of
    /// single rows reads as a flat run. Ids, counts and search are the
    /// tree's, so filter and find on the tree, then flatten for display.
    static func flatteningSoleChildren(_ nodes: [ModelLineageNode]) -> [ModelLineageNode] {
        var rows: [ModelLineageNode] = []
        for node in nodes {
            appendListing(of: node, isSoleChildOfRowAbove: false, to: &rows)
        }
        return rows
    }

    /// One accumulator for a whole chain of single rows, so a long chain
    /// isn't copied once per level.
    private static func appendListing(of node: ModelLineageNode, isSoleChildOfRowAbove: Bool, to rows: inout [ModelLineageNode]) {
        guard let children = node.children, children.count > 1 else {
            rows.append(ModelLineageNode(listing: node, children: nil, isSoleChildOfRowAbove: isSoleChildOfRowAbove))
            if let soleChild = node.children?.first {
                appendListing(of: soleChild, isSoleChildOfRowAbove: true, to: &rows)
            }
            return
        }
        rows.append(ModelLineageNode(listing: node, children: flatteningSoleChildren(children), isSoleChildOfRowAbove: isSoleChildOfRowAbove))
    }

    /// The picker's rows for a search, then listed. A row whose own texts
    /// (`ownSearchTexts`: model id, session, file names) contain the query,
    /// case-insensitively, is kept with its whole subtree when the match is
    /// on its own model ID — a seed's id still shows its whole family — or
    /// when nothing below it matches; a row matching only on a file or
    /// session name, with matches below it, keeps just the paths to those
    /// matches. A row that doesn't match is kept only as the
    /// path to rows below it that do, with its other rows left out. So a deep
    /// match comes with just its ancestors, and the listing's sole-child rule
    /// then shows that chain as a flat run, visible without expanding
    /// anything (issue #20: a branch six levels down was unfindable). Pruning
    /// comes first because the listing moves a seed's only segment out of its
    /// subtree; searching that segment still shows the seed above it.
    static func listedRows(of tree: [ModelLineageNode], matching search: String) -> [ModelLineageNode] {
        flatteningSoleChildren(prunedTree(tree, matching: search))
    }

    /// `tree` cut down to the rows on a path to a match of `search` (see
    /// `listedRows`); the whole tree for an empty search. Unlisted, so a
    /// caller can also take `recentBranchTips` of it.
    static func prunedTree(_ tree: [ModelLineageNode], matching search: String) -> [ModelLineageNode] {
        let query = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !query.isEmpty else { return tree }
        return tree.compactMap { pruned($0, to: query) }
    }

    /// `node` cut down to the rows on a path to a match of `query`, or nil
    /// when nothing in it matches (see `listedRows`).
    /// A row matching only on a file or session name whose subtree also
    /// holds a match is cut down to the paths to those matches like any
    /// other row: otherwise a match below a matching ancestor (R-elitecyc's
    /// files under R-elite's, both holding "elite") would stay inside a
    /// collapsed group. A model-ID match keeps everything: the seed's own
    /// file and champion names carry its ID, so path-pruning there would
    /// drop its child segments, the family the search is for.
    private static func pruned(_ node: ModelLineageNode, to query: String) -> ModelLineageNode? {
        if node.ownModelIDContains(query) { return node }
        let keptChildren = (node.children ?? []).compactMap { pruned($0, to: query) }
        if node.ownTextsContain(query) {
            return keptChildren.isEmpty ? node : ModelLineageNode(pruning: node, children: keptChildren)
        }
        guard !keptChildren.isEmpty else { return nil }
        return ModelLineageNode(pruning: node, children: keptChildren)
    }

    /// The most recently active branch tips: every trained segment with no
    /// segment below it, newest activity first, at most `limit`. The picker
    /// lists them above the tree (issue #20) so the run being trained now is
    /// the first row, however deep it sits. Each is the tree's own node with
    /// its children left out (the tree row below carries them) and a `tip:`
    /// id, so a List can hold both rows.
    static func recentBranchTips(of tree: [ModelLineageNode], limit: Int) -> [ModelLineageNode] {
        var tips: [ModelLineageNode] = []
        func collect(_ nodes: [ModelLineageNode]) {
            for node in nodes {
                if case .segment(_, _, let isUntrained, let isBranchTip) = node.kind, isBranchTip, !isUntrained {
                    tips.append(node)
                }
                collect(node.children ?? [])
            }
        }
        collect(tree)
        return tips
            .sorted { $0.newestActivity > $1.newestActivity }
            .prefix(limit)
            .map { ModelLineageNode(tipOf: $0) }
    }

    /// The `tip:` id prefix of `recentBranchTips` rows.
    static let tipIDPrefix = "tip:"

    /// A self-play champion's lineage-root ModelID: the id with any
    /// trainer-generation suffix removed, as `ModelID.lineageRoot` defines
    /// it. Session champions record an empty `parent_model_id`, so this is
    /// the only link to their line.
    static func baseModelID(ofChampion modelID: String) -> String {
        ModelID(value: modelID).lineageRoot
    }

    private static func championNode(_ champion: SessionChampion) -> ModelLineageNode {
        ModelLineageNode(id: "session:\(champion.entry.url.path)", kind: .sessionChampion(champion), children: nil)
    }
}
