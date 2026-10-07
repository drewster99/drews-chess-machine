import XCTest
@testable import DrewsChessMachine

/// `ModelLineageTip.select` — the newest file of one followed lineage, from
/// the lineage records alone (follow-lineage plan §3.3). Every record here is
/// made by `LineageTracker`; filenames and modification dates are chosen to
/// disagree with the records wherever they could mislead.
final class ModelLineageTipTests: XCTestCase {

    private var fileCounter = 0

    /// A catalog entry for a file holding `record`, with hash `sha`.
    private func entry(_ record: LineageRecord, modelID: String, sha: String?, name: String? = nil, modified: TimeInterval = 1_790_000_000) -> ModelFileEntry {
        fileCounter += 1
        return ModelFileEntry(
            url: URL(fileURLWithPath: "/models/\(name ?? "file-\(fileCounter).safetensors")"),
            modelID: modelID,
            trainingStep: record.steps.segmentLocalStep,
            createdAt: nil,
            architectureLabel: "a",
            fileModifiedAt: Date(timeIntervalSince1970: modified),
            contentSHA256: sha,
            lineage: .recorded(ModelFileLineagePosition(record: record))
        )
    }

    private func anchor(_ record: LineageRecord) -> ModelLineageAnchor {
        ModelLineageAnchor(lineageRunID: record.run.lineageRunID, anchorSegmentID: record.run.segmentID)
    }

    private func newest(_ selection: ModelLineageTip.Selection) -> ModelLineageTip.Candidate? {
        if case .following(let newest, _, _) = selection.outcome { return newest }
        return nil
    }

    func testNewestIsTheHighestLocalStepInOneSegment() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let r1 = try run.record(localStep: 1000, at: 2_000)
        let r2 = try run.record(localStep: 2000, at: 3_000)
        let r3 = try run.record(localStep: 3000, at: 4_000)
        let entries = [entry(r1, modelID: run.modelID, sha: "a1"), entry(r3, modelID: run.modelID, sha: "a3"), entry(r2, modelID: run.modelID, sha: "a2")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(r1), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "a3")
        XCTAssertEqual(selection.candidateCount, 3)
    }

    func testFilenamesAndModificationDatesNeverDecide() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let older = try run.record(localStep: 1000, at: 2_000)
        let newer = try run.record(localStep: 9000, at: 9_000)
        let entries = [
            entry(older, modelID: run.modelID, sha: "old", name: "run-replay-step9000.safetensors", modified: 1_900_000_000),
            entry(newer, modelID: run.modelID, sha: "new", name: "run-replay-step1000.safetensors", modified: 1_700_000_000),
        ]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(older), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "new")
    }

    func testLaterSegmentOutranksAHigherCumulativeStepInItsParent() throws {
        let parentRun = try LineageTestRuns.fresh(modelID: "20261006-1-PRNT", startedUnix: 1_000)
        let early = try parentRun.record(localStep: 1000, at: 2_000)
        let late = try parentRun.record(localStep: 5000, at: 3_000)
        // The parent stopped; a resume of its step-1000 file started later.
        let child = try LineageTestRuns.resume(from: early, parentModelID: parentRun.modelID, parentSHA256: "p1000", modelID: "20261006-2-CHLD", startedUnix: 4_000)
        let childFile = try child.record(localStep: 500, at: 5_000)
        XCTAssertLessThan(try XCTUnwrap(childFile.steps.cumTrainerStep), try XCTUnwrap(late.steps.cumTrainerStep))
        let entries = [entry(early, modelID: parentRun.modelID, sha: "p1000"), entry(late, modelID: parentRun.modelID, sha: "p5000"), entry(childFile, modelID: child.modelID, sha: "c500")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(early), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "c500")
    }

    func testExactResumeIsFollowedAcrossANewModelID() throws {
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let handoff = try first.record(localStep: 3000, at: 2_000)
        let second = try LineageTestRuns.resume(from: handoff, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-2-SEG1", startedUnix: 3_000)
        let later = try second.record(localStep: 1000, at: 4_000)
        let entries = [entry(handoff, modelID: first.modelID, sha: "s0"), entry(later, modelID: second.modelID, sha: "s1")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(handoff), notBelow: nil)
        XCTAssertEqual(newest(selection)?.entry.modelID, "20261006-2-SEG1")
    }

    func testBranchIsNotFollowed() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-MAIN", startedUnix: 1_000)
        let main = try run.record(localStep: 1000, at: 2_000)
        let branch = try LineageTestRuns.branch(from: main, parentModelID: run.modelID, parentSHA256: "m", modelID: "20261006-2-BRCH", startedUnix: 3_000)
        let branched = try branch.record(localStep: 9000, at: 9_000)
        let entries = [entry(main, modelID: run.modelID, sha: "m"), entry(branched, modelID: branch.modelID, sha: "b")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(main), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "m", "a branch is a new run")
        XCTAssertEqual(selection.candidateCount, 1)
    }

    func testFilesBeforeTheAnchorSegmentAreNotCandidates() throws {
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let s0 = try first.record(localStep: 3000, at: 2_000)
        let second = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-2-SEG1", startedUnix: 3_000)
        let s1 = try second.record(localStep: 1000, at: 4_000)
        let entries = [entry(s0, modelID: first.modelID, sha: "s0"), entry(s1, modelID: second.modelID, sha: "s1")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(s1), notBelow: nil)
        XCTAssertEqual(selection.candidateCount, 1)
        XCTAssertEqual(newest(selection)?.contentSHA256, "s1")
    }

    func testForkAfterTheAnchorIsReportedWithBothSegments() throws {
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let s0 = try first.record(localStep: 3000, at: 2_000)
        let left = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-2-LEFT", startedUnix: 3_000)
        let right = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-3-RGHT", startedUnix: 3_100)
        let l = try left.record(localStep: 1000, at: 4_000)
        let r = try right.record(localStep: 2000, at: 4_100)
        let entries = [entry(s0, modelID: first.modelID, sha: "s0"), entry(l, modelID: left.modelID, sha: "l"), entry(r, modelID: right.modelID, sha: "r")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(s0), notBelow: nil)
        guard case .fork(let tips) = selection.outcome else {
            return XCTFail("expected a fork, got \(selection.outcome)")
        }
        XCTAssertEqual(Set(tips.map(\.modelID)), ["20261006-2-LEFT", "20261006-3-RGHT"])
        XCTAssertEqual(Set(tips.map(\.newestLocalStep)), [1000, 2000])

        // Anchoring on one sibling resolves it.
        let resolved = ModelLineageTip.select(entries: entries, followed: anchor(l), notBelow: nil)
        XCTAssertEqual(newest(resolved)?.contentSHA256, "l")
    }

    func testForkBeforeTheAnchorIsNotAFork() throws {
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let s0 = try first.record(localStep: 3000, at: 2_000)
        let left = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-2-LEFT", startedUnix: 3_000)
        let right = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-3-RGHT", startedUnix: 3_100)
        let l = try left.record(localStep: 1000, at: 4_000)
        let r = try right.record(localStep: 2000, at: 4_100)
        let entries = [entry(s0, modelID: first.modelID, sha: "s0"), entry(l, modelID: left.modelID, sha: "l"), entry(r, modelID: right.modelID, sha: "r")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(r), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "r", "the sibling outside the anchored segment's chain is not a candidate")
    }

    func testByteIdenticalFilesAreOneCandidate() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let save = try run.record(localStep: 31000, at: 2_000)
        let entries = [
            entry(save, modelID: run.modelID, sha: "same", name: "run-replay-step31000.safetensors"),
            entry(save, modelID: run.modelID, sha: "same", name: "run-replay-latest.safetensors"),
        ]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(save), notBelow: nil)
        guard case .following(let newest, let sameWeights, _) = selection.outcome else {
            return XCTFail("expected following, got \(selection.outcome)")
        }
        XCTAssertEqual(newest.entry.url.lastPathComponent, "run-replay-latest.safetensors", "path order between byte-identical copies")
        XCTAssertEqual(sameWeights.map(\.lastPathComponent), ["run-replay-latest.safetensors", "run-replay-step31000.safetensors"])
    }

    func testDistinctWeightsAtOneRankAreAConflict() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let save = try run.record(localStep: 31000, at: 2_000)
        let entries = [entry(save, modelID: run.modelID, sha: "one"), entry(save, modelID: run.modelID, sha: "two")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(save), notBelow: nil)
        guard case .conflict(let files) = selection.outcome else {
            return XCTFail("expected a conflict, got \(selection.outcome)")
        }
        XCTAssertEqual(Set(files.map(\.contentSHA256)), ["one", "two"])
    }

    func testNeverSelectsBelowThePlayingGeneration() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let onDisk = try run.record(localStep: 2000, at: 2_000)
        let playing = try run.record(localStep: 3000, at: 3_000)
        let rank = ModelLineageRank(segmentChain: ModelFileLineagePosition(record: playing).segmentChain, segmentLocalStep: 3000, recordedUnix: 3_000)
        let selection = ModelLineageTip.select(entries: [entry(onDisk, modelID: run.modelID, sha: "d")], followed: anchor(onDisk), notBelow: rank)
        guard case .keepPlaying(let newestOnDisk) = selection.outcome else {
            return XCTFail("expected keepPlaying, got \(selection.outcome)")
        }
        XCTAssertEqual(newestOnDisk.contentSHA256, "d")
    }

    func testNullCumulativeStepStillRanks() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let a = try run.record(localStep: 1000, at: 2_000)
        let b = try run.record(localStep: 2000, at: 3_000)
        var first = entry(a, modelID: run.modelID, sha: "a")
        var second = entry(b, modelID: run.modelID, sha: "b")
        // A run continuing unrecorded history records no cumulative step.
        for (index, record) in [a, b].enumerated() {
            let position = ModelFileLineagePosition(record: record)
            let nulled = ModelFileLineagePosition(
                lineageRunID: position.lineageRunID, segmentID: position.segmentID, segmentIndex: position.segmentIndex,
                segmentStartedUnix: position.segmentStartedUnix, segmentChain: position.segmentChain, handoffs: position.handoffs,
                segmentLocalStep: position.segmentLocalStep, cumTrainerStep: nil, recordedUnix: position.recordedUnix, pathKind: position.pathKind)
            if index == 0 { first.lineage = .recorded(nulled) } else { second.lineage = .recorded(nulled) }
        }
        let selection = ModelLineageTip.select(entries: [first, second], followed: anchor(a), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "b")
    }

    func testExcludedFilesAreCountedByReason() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let good = try run.record(localStep: 1000, at: 2_000)
        let gui = try LineageTestRuns.resume(from: good, parentModelID: run.modelID, parentSHA256: "g", modelID: "20261006-2-GUIS", startedUnix: 3_000, pathKind: .gui)
        let guiFile = try gui.record(localStep: 10, at: 4_000)
        var unrecorded = entry(good, modelID: "20261006-3-OLDF", sha: "u")
        unrecorded.lineage = .unrecorded(formatVersion: 6)
        var unreadable = entry(good, modelID: "20261006-4-BADL", sha: "x")
        unreadable.lineage = .unreadable(reason: "malformed lineage")
        let entries = [
            entry(good, modelID: run.modelID, sha: "g"),
            entry(guiFile, modelID: gui.modelID, sha: "gui"),
            entry(good, modelID: run.modelID, sha: nil),
            unrecorded,
            unreadable,
        ]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(good), notBelow: nil)
        XCTAssertEqual(newest(selection)?.contentSHA256, "g")
        XCTAssertEqual(selection.excluded, [.otherPathKind: 1, .noContentHash: 1, .unrecordedLineage: 1, .unreadableLineage: 1])
    }

    func testNoCandidatesIsNoFiles() throws {
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000)
        let record = try run.record(localStep: 1000, at: 2_000)
        let selection = ModelLineageTip.select(entries: [], followed: anchor(record), notBelow: nil)
        XCTAssertEqual(selection.outcome, .noFiles)
    }

    /// Two live continuations of one point: a child `--resume-exact`-ed from
    /// a step file while the parent kept training.
    func testParentThatKeptTrainingAfterAResumeIsAFork() throws {
        let parent = try LineageTestRuns.fresh(modelID: "20261006-1-PRNT", startedUnix: 1_000)
        let handoff = try parent.record(localStep: 3000, at: 2_000)
        let child = try LineageTestRuns.resume(from: handoff, parentModelID: parent.modelID, parentSHA256: "p3000", modelID: "20261006-2-CHLD", startedUnix: 3_000)
        let parentLater = try parent.record(localStep: 4000, at: 3_500)
        let childFile = try child.record(localStep: 1000, at: 4_000)
        let entries = [entry(handoff, modelID: parent.modelID, sha: "p3000"), entry(parentLater, modelID: parent.modelID, sha: "p4000"), entry(childFile, modelID: child.modelID, sha: "c1000")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(handoff), notBelow: nil)
        guard case .fork(let tips) = selection.outcome else {
            return XCTFail("expected a fork, got \(selection.outcome)")
        }
        XCTAssertEqual(Set(tips.map(\.modelID)), ["20261006-1-PRNT", "20261006-2-CHLD"])

        // Anchoring on the child resolves it.
        let resolved = ModelLineageTip.select(entries: entries, followed: anchor(childFile), notBelow: nil)
        XCTAssertEqual(newest(resolved)?.contentSHA256, "c1000")
    }

    func testResumeFromAnEarlierFileAfterTheParentStoppedIsFollowed() throws {
        let parent = try LineageTestRuns.fresh(modelID: "20261006-1-PRNT", startedUnix: 1_000)
        let handoff = try parent.record(localStep: 3000, at: 2_000)
        let parentLast = try parent.record(localStep: 5000, at: 2_500)
        let child = try LineageTestRuns.resume(from: handoff, parentModelID: parent.modelID, parentSHA256: "p3000", modelID: "20261006-2-CHLD", startedUnix: 3_000)
        let childFile = try child.record(localStep: 100, at: 4_000)
        let entries = [entry(handoff, modelID: parent.modelID, sha: "p3000"), entry(parentLast, modelID: parent.modelID, sha: "p5000"), entry(childFile, modelID: child.modelID, sha: "c100")]
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(handoff), notBelow: nil)
        guard case .following(let newest, _, let redos) = selection.outcome else {
            return XCTFail("expected following, got \(selection.outcome)")
        }
        XCTAssertEqual(newest.contentSHA256, "c100")
        XCTAssertEqual(redos, [ModelLineageTip.Redo(childSegmentID: childFile.run.segmentID, parentSegmentID: handoff.run.segmentID, resumedAtLocalStep: 3000, parentNewestLocalStep: 5000)])
    }

    func testCandidateNotPrefixOrderedWithThePlayingGenerationIsAFork() throws {
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let s0 = try first.record(localStep: 3000, at: 2_000)
        let left = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-2-LEFT", startedUnix: 3_000)
        let right = try LineageTestRuns.resume(from: s0, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-3-RGHT", startedUnix: 3_100)
        let playingRecord = try left.record(localStep: 1000, at: 4_000)
        let r = try right.record(localStep: 2000, at: 4_100)
        // The left sibling's files are gone; the right one's remain.
        let entries = [entry(s0, modelID: first.modelID, sha: "s0"), entry(r, modelID: right.modelID, sha: "r")]
        let playing = ModelLineageRank(segmentChain: ModelFileLineagePosition(record: playingRecord).segmentChain, segmentLocalStep: 1000, recordedUnix: 4_000)
        let selection = ModelLineageTip.select(entries: entries, followed: anchor(s0), notBelow: playing)
        guard case .divergedFromPlaying(let newest) = selection.outcome else {
            return XCTFail("expected a fork against what plays, got \(selection.outcome)")
        }
        XCTAssertEqual(newest.contentSHA256, "r")
    }
}
