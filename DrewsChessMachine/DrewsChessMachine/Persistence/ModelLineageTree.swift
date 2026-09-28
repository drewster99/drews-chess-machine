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

    let id: String
    let kind: Kind
    /// nil for a leaf, so `OutlineGroup` shows no disclosure arrow.
    let children: [ModelLineageNode]?

    /// The file selecting this row chooses, if any.
    var selectableURL: URL? {
        switch kind {
        case .segment(let line, _, _, _): return line.latest.url
        case .file(let entry): return entry.url
        case .sessionChampion(let champion): return champion.entry.url
        case .conflict: return nil
        }
    }

    /// Every model id and session name in this subtree, lowercased, for
    /// search: typing a seed's id finds its whole family.
    var searchableIDs: [String] {
        var own: [String]
        switch kind {
        case .segment(let line, _, _, _): own = [line.modelID]
        case .file: own = []
        case .sessionChampion(let champion): own = [champion.entry.modelID, champion.sessionName]
        case .conflict(let modelID, let parents): own = [modelID] + parents
        }
        own = own.map { $0.lowercased() }
        return own + (children ?? []).flatMap(\.searchableIDs)
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
            let children = earlierFiles + childSegments + sessionNodes
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
            let lhsSeedOnly = isSeedOnly(lhs)
            let rhsSeedOnly = isSeedOnly(rhs)
            if lhsSeedOnly != rhsSeedOnly { return !lhsSeedOnly }
            return lhs.newestActivity > rhs.newestActivity
        }
        return conflictNodes + trainedFirst + orphanChampions.map(championNode)
    }

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

    /// An untrained seed with nothing trained under it.
    private static func isSeedOnly(_ node: ModelLineageNode) -> Bool {
        guard case .segment(_, _, let isUntrained, _) = node.kind else { return false }
        let hasTrainedDescendant = (node.children ?? []).contains { child in
            switch child.kind {
            case .segment, .sessionChampion: return true
            case .file(let entry): return entry.trainingStep != nil
            case .conflict: return false
            }
        }
        return isUntrained && !hasTrainedDescendant
    }
}
