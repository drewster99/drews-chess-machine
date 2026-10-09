import XCTest
@testable import DrewsChessMachine

/// Plan §9.1: the seed → branch → segment tree built from file metadata.
final class ModelLineageTreeTests: XCTestCase {

    private func entry(_ modelID: String, step: Int?, parent: String?, modified: TimeInterval) -> ModelFileEntry {
        ModelFileEntry(
            url: URL(fileURLWithPath: "/m/\(modelID)-\(step.map(String.init) ?? "none").safetensors"),
            modelID: modelID,
            trainingStep: step,
            createdAt: nil,
            architectureLabel: "a",
            fileModifiedAt: Date(timeIntervalSince1970: modified),
            parentModelID: parent,
            creator: step == nil ? "manual" : "replay"
        )
    }

    private func line(_ modelID: String, _ files: [ModelFileEntry]) -> ModelLine {
        ModelLine(modelID: modelID, files: files.sorted(by: ModelFileCatalog.isMoreAdvanced))
    }

    private func segmentIDs(_ nodes: [ModelLineageNode]) -> [String] {
        nodes.compactMap { node in
            if case .segment(let line, _, _, _) = node.kind { return line.modelID }
            return nil
        }
    }

    private func node(_ modelID: String, in nodes: [ModelLineageNode]) -> ModelLineageNode? {
        for node in nodes {
            if case .segment(let line, _, _, _) = node.kind, line.modelID == modelID { return node }
            if let found = self.node(modelID, in: node.children ?? []) { return found }
        }
        return nil
    }

    func testChainsBranchesAndEachFileOnce() throws {
        let lines = [
            line("S-seed", [entry("S-seed", step: nil, parent: nil, modified: 1)]),
            line("S-a", [entry("S-a", step: 100, parent: "S-seed", modified: 2), entry("S-a", step: 50, parent: "S-seed", modified: 1.5)]),
            line("S-b", [entry("S-b", step: 10, parent: "S-a", modified: 3)]),
            line("S-c", [entry("S-c", step: 7, parent: "S-seed", modified: 4)]),
        ]
        let tree = ModelLineageTree.build(lines: lines, champions: [])
        XCTAssertEqual(segmentIDs(tree), ["S-seed"])
        let seed = try XCTUnwrap(node("S-seed", in: tree))
        // Newest branch first.
        XCTAssertEqual(segmentIDs(seed.children ?? []), ["S-c", "S-a"])
        let a = try XCTUnwrap(node("S-a", in: tree))
        guard case .segment(let aLine, let path, let untrained, let tip) = a.kind else { return XCTFail("not a segment") }
        XCTAssertEqual(aLine.latest.trainingStep, 100)
        XCTAssertEqual(path, ["S-seed", "S-a"])
        XCTAssertFalse(untrained)
        XCTAssertFalse(tip, "S-b continues this branch")
        // The earlier file only; the latest is the segment row itself.
        let earlier = (a.children ?? []).compactMap { child -> Int? in
            if case .file(let file) = child.kind { return file.trainingStep }
            return nil
        }
        XCTAssertEqual(earlier, [50])
        guard case .segment(_, let bPath, _, let bTip) = try XCTUnwrap(node("S-b", in: tree)).kind else { return XCTFail("not a segment") }
        XCTAssertEqual(bPath, ["S-seed", "S-a", "S-b"])
        XCTAssertTrue(bTip)
    }

    /// A seed's own file is untrained, so its row counts what was trained
    /// below it: every trained segment at any depth and every session
    /// champion, but not an untrained child segment.
    func testTrainedBelowCountsTrainedSegmentsAndChampionsAtAnyDepth() throws {
        let lines = [
            line("20260901-1-SEED", [entry("20260901-1-SEED", step: nil, parent: nil, modified: 1)]),
            line("S-a", [entry("S-a", step: 100, parent: "20260901-1-SEED", modified: 2), entry("S-a", step: 50, parent: "20260901-1-SEED", modified: 1.5)]),
            line("S-b", [entry("S-b", step: 10, parent: "S-a", modified: 3)]),
            line("S-copy", [entry("S-copy", step: nil, parent: "20260901-1-SEED", modified: 4)]),
            line("L-lone", [entry("L-lone", step: nil, parent: nil, modified: 5)]),
        ]
        let champion = SessionChampion(sessionName: "sess", entry: entry("20260901-1-SEED-3", step: 900, parent: nil, modified: 6))
        let tree = ModelLineageTree.build(lines: lines, champions: [champion])

        let seed = try XCTUnwrap(node("20260901-1-SEED", in: tree))
        XCTAssertEqual(seed.trainedBelow, ModelLineageNode.TrainedBelow(trainedSegments: 2, sessionChampions: 1))
        XCTAssertEqual(try XCTUnwrap(node("S-a", in: tree)).trainedBelow, ModelLineageNode.TrainedBelow(trainedSegments: 1, sessionChampions: 0))
        XCTAssertTrue(try XCTUnwrap(node("S-b", in: tree)).trainedBelow.isEmpty)
        XCTAssertTrue(try XCTUnwrap(node("S-copy", in: tree)).trainedBelow.isEmpty)
        XCTAssertTrue(try XCTUnwrap(node("L-lone", in: tree)).trainedBelow.isEmpty)
    }

    func testMissingParentMakesARootAndUntrainedSeedsSortLast() {
        let lines = [
            line("X-lone-seed", [entry("X-lone-seed", step: nil, parent: nil, modified: 9)]),
            line("Y-orphan", [entry("Y-orphan", step: 5, parent: "Y-deleted", modified: 1)]),
        ]
        XCTAssertEqual(segmentIDs(ModelLineageTree.build(lines: lines, champions: [])), ["Y-orphan", "X-lone-seed"])
    }

    func testConflictingParentsAreReportedNotPicked() {
        let lines = [
            line("P1", [entry("P1", step: 1, parent: nil, modified: 1)]),
            line("P2", [entry("P2", step: 1, parent: nil, modified: 1)]),
            line("C", [entry("C", step: 2, parent: "P1", modified: 2), entry("C", step: 3, parent: "P2", modified: 3)]),
        ]
        let tree = ModelLineageTree.build(lines: lines, champions: [])
        guard case .conflict(let modelID, let parents) = tree.first?.kind else { return XCTFail("expected a conflict row first") }
        XCTAssertEqual(modelID, "C")
        XCTAssertEqual(parents, ["P1", "P2"])
    }

    func testSessionChampionsGoUnderTheirBaseModelID() throws {
        let lines = [line("20260727-1-Ejp0", [entry("20260727-1-Ejp0", step: 1_397_000, parent: nil, modified: 1)])]
        let champion = SessionChampion(sessionName: "20260917-sjIy-promote", entry: entry("20260727-1-Ejp0-66", step: 1_186_000, parent: nil, modified: 5))
        XCTAssertEqual(ModelLineageTree.baseModelID(ofChampion: "20260727-1-Ejp0-66"), "20260727-1-Ejp0")
        XCTAssertEqual(ModelLineageTree.baseModelID(ofChampion: "20260727-1-Ejp0"), "20260727-1-Ejp0")
        let root = try XCTUnwrap(ModelLineageTree.build(lines: lines, champions: [champion]).first)
        XCTAssertTrue((root.children ?? []).contains { child in
            if case .sessionChampion(let placed) = child.kind { return placed == champion }
            return false
        })
        XCTAssertTrue(root.searchableIDs.contains("20260917-sjiy-promote"))
    }

    /// A seed whose only child is an untrained copy has nothing trained
    /// below it: it sorts after trained roots, by the same rule that tags its
    /// row "untrained" in orange.
    func testASeedWithOnlyAnUntrainedChildSortsWithTheSeeds() {
        let lines = [
            line("Q-seed", [entry("Q-seed", step: nil, parent: nil, modified: 1)]),
            line("Q-copy", [entry("Q-copy", step: nil, parent: "Q-seed", modified: 10)]),
            line("R-trained", [entry("R-trained", step: 5, parent: nil, modified: 2)]),
        ]
        XCTAssertEqual(segmentIDs(ModelLineageTree.build(lines: lines, champions: [])), ["R-trained", "Q-seed"])
    }

    /// The rows as the picker lists them: a row with exactly one row below
    /// it has no disclosure and that row follows it, marked, at every depth;
    /// a row with several keeps them under it. Ids and counts are the tree's.
    func testASoleChildIsListedBelowItsRowInsteadOfUnderADisclosure() throws {
        let lines = [
            line("O-seed", [entry("O-seed", step: nil, parent: nil, modified: 1)]),
            line("O-run", [entry("O-run", step: 7000, parent: "O-seed", modified: 2)]),
            line("O-next", [entry("O-next", step: 9000, parent: "O-run", modified: 3)]),
            line("P-seed", [entry("P-seed", step: nil, parent: nil, modified: 1)]),
            line("P-a", [entry("P-a", step: 10, parent: "P-seed", modified: 3), entry("P-a", step: 5, parent: "P-seed", modified: 2.5)]),
            line("P-b", [entry("P-b", step: 20, parent: "P-seed", modified: 4)]),
        ]
        let tree = ModelLineageTree.build(lines: lines, champions: [])
        let listed = ModelLineageTree.flatteningSoleChildren(tree)
        func row(_ id: String, in nodes: [ModelLineageNode]) throws -> ModelLineageNode {
            try XCTUnwrap(nodes.first { $0.id == id }, id)
        }

        // A chain of single rows is a flat run; only the first is unmarked.
        XCTAssertEqual(listed.map(\.id), ["segment:P-seed", "segment:O-seed", "segment:O-run", "segment:O-next"])
        XCTAssertEqual(listed.map(\.isSoleChildOfRowAbove), [false, false, true, true])
        XCTAssertNil(try row("segment:O-seed", in: listed).children, "no disclosure for one row")
        XCTAssertNil(try row("segment:O-run", in: listed).children)
        XCTAssertEqual(try row("segment:O-seed", in: listed).trainedBelow, try XCTUnwrap(node("O-seed", in: tree)).trainedBelow,
                       "the seed still counts what was trained below it")
        XCTAssertEqual(try row("segment:O-seed", in: listed).trainedBelow.trainedSegments, 2)

        // Two rows below: they stay under the disclosure; one of them with
        // a single earlier file lists it beside itself.
        let pChildren = try XCTUnwrap(try row("segment:P-seed", in: listed).children)
        XCTAssertEqual(pChildren.map(\.id), ["segment:P-b", "segment:P-a", "file:/m/P-a-5.safetensors"])
        XCTAssertEqual(pChildren.map(\.isSoleChildOfRowAbove), [false, false, true])
        XCTAssertNil(try row("segment:P-a", in: pChildren).children)

        XCTAssertTrue(tree.allSatisfy { !$0.isSoleChildOfRowAbove }, "the tree itself is unchanged")
        XCTAssertNotNil(try XCTUnwrap(node("O-seed", in: tree)).children)
    }

    /// Every row of a listing at any depth, in order.
    private func allRows(_ nodes: [ModelLineageNode]) -> [ModelLineageNode] {
        nodes.flatMap { [$0] + allRows($0.children ?? []) }
    }

    /// The listing's edge cases: a conflict row, a session champion as the
    /// only row, a sole row that has several rows of its own, empty input,
    /// and that every tree id is listed exactly once (List selection and
    /// `OutlineGroup` identity rely on it).
    func testTheListingKeepsEveryRowOnceForEveryKind() throws {
        let lines = [
            line("P1", [entry("P1", step: 1, parent: nil, modified: 1)]),
            line("P2", [entry("P2", step: 1, parent: nil, modified: 1)]),
            line("C", [entry("C", step: 2, parent: "P1", modified: 2), entry("C", step: 3, parent: "P2", modified: 3)]),
            line("20261001-1-SEED", [entry("20261001-1-SEED", step: nil, parent: nil, modified: 1)]),
            line("R-run", [entry("R-run", step: 20, parent: "20261001-1-SEED", modified: 5), entry("R-run", step: 10, parent: "20261001-1-SEED", modified: 4)]),
            line("R-next", [entry("R-next", step: 30, parent: "R-run", modified: 6)]),
            line("20261002-1-SOLO", [entry("20261002-1-SOLO", step: 100, parent: nil, modified: 7)]),
        ]
        let champion = SessionChampion(sessionName: "sess", entry: entry("20261002-1-SOLO-2", step: 900, parent: nil, modified: 8))
        let tree = ModelLineageTree.build(lines: lines, champions: [champion])
        let listed = ModelLineageTree.flatteningSoleChildren(tree)

        let treeIDs = allRows(tree).map(\.id)
        let listedIDs = allRows(listed).map(\.id)
        XCTAssertEqual(Set(listedIDs).count, listedIDs.count, "no id twice")
        XCTAssertEqual(Set(listedIDs), Set(treeIDs), "every tree row listed")

        // A conflict always has one row: its segment, listed below it.
        let conflict = try XCTUnwrap(listed.firstIndex { $0.id == "conflict:C" })
        XCTAssertNil(listed[conflict].children)
        XCTAssertEqual(listed[conflict + 1].id, "segment:C")
        XCTAssertTrue(listed[conflict + 1].isSoleChildOfRowAbove)

        // A session champion as the only row; its parent still counts it.
        let solo = try XCTUnwrap(listed.firstIndex { $0.id == "segment:20261002-1-SOLO" })
        XCTAssertNil(listed[solo].children)
        XCTAssertEqual(listed[solo].trainedBelow.sessionChampions, 1)
        guard case .sessionChampion(let placed) = listed[solo + 1].kind else { return XCTFail("expected the champion below its segment") }
        XCTAssertEqual(placed, champion)
        XCTAssertTrue(listed[solo + 1].isSoleChildOfRowAbove)

        // A sole row with several rows of its own keeps its disclosure and
        // its copied count.
        let seed = try XCTUnwrap(listed.firstIndex { $0.id == "segment:20261001-1-SEED" })
        let run = listed[seed + 1]
        XCTAssertEqual(run.id, "segment:R-run")
        XCTAssertTrue(run.isSoleChildOfRowAbove)
        XCTAssertEqual(run.children?.map(\.id), ["file:/m/R-run-10.safetensors", "segment:R-next"])
        XCTAssertEqual(run.trainedBelow, try XCTUnwrap(node("R-run", in: tree)).trainedBelow)
        XCTAssertEqual(run.trainedBelow.trainedSegments, 1)

        XCTAssertEqual(ModelLineageTree.flatteningSoleChildren([]), [])
        let empty = ModelLineageNode(id: "x", kind: .conflict(modelID: "x", parents: []), children: [])
        XCTAssertEqual(ModelLineageTree.flatteningSoleChildren([empty]).map(\.children), [nil], "an empty list of rows is a leaf")
    }

    /// Search filters whole families before listing: the id of a seed's
    /// only segment finds the seed and that segment below it.
    func testSearchingASoleSegmentShowsItsSeedAboveIt() {
        let lines = [
            line("O-seed", [entry("O-seed", step: nil, parent: nil, modified: 1)]),
            line("O-run", [entry("O-run", step: 7000, parent: "O-seed", modified: 2)]),
            line("Z-other", [entry("Z-other", step: 5, parent: nil, modified: 3)]),
        ]
        let tree = ModelLineageTree.build(lines: lines, champions: [])
        XCTAssertEqual(ModelLineageTree.listedRows(of: tree, matching: "  o-RUN ").map(\.id), ["segment:O-seed", "segment:O-run"])
        XCTAssertEqual(ModelLineageTree.listedRows(of: tree, matching: "").map(\.id), ["segment:Z-other", "segment:O-seed", "segment:O-run"])
        XCTAssertEqual(ModelLineageTree.listedRows(of: tree, matching: "nothing"), [])
    }
}
