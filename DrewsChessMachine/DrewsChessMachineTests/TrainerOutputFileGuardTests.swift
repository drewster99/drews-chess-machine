//
//  TrainerOutputFileGuardTests.swift
//  DrewsChessMachineTests
//
//  Corpus replay and train-vs-UCI wrote their step-enumerated checkpoints
//  with an atomic overwrite, and step numbers restart in every run, so a
//  resumed segment that reused an earlier segment's --out-model stem silently
//  replaced that segment's checkpoints (distinct v5 checkpoints from different
//  segments ended up sharing one step-file name). The rolling --out-model file
//  was likewise replaced whatever was there. These pin the rules that replaced
//  that: an enumerated write never lands on a file the run did not write; a
//  stem that already holds reachable step files refuses the run before
//  training; and an existing rolling file is replaced only when it is the
//  rolling file of the state the run continues — same model ID, same step —
//  (or with --overwrite-out-model), never when it is the start model or not a
//  regular file. An existing file of the same line at an earlier step was
//  once adopted too, so a mistyped --out-model naming an enumerated
//  checkpoint overwrote it; an --out-model named like an enumerated
//  checkpoint is now refused outright.
//

import XCTest
import Darwin
@testable import DrewsChessMachine

final class TrainerOutputFileGuardTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("TrainerOutputFileGuardTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    // MARK: Helpers

    /// A minimal safetensors file whose header carries `model_id` and,
    /// optionally, `training_step` — all the guard reads.
    @discardableResult
    private func writeModelFile(_ name: String, modelID: String, trainingStep: Int?) throws -> URL {
        var metadata = ["model_id": modelID]
        if let trainingStep { metadata["training_step"] = String(trainingStep) }
        let data = try SafetensorsFile.encode(
            tensors: [SafetensorsTensor(name: "w", shape: [2], data: [1, 2])],
            metadata: metadata)
        let url = root.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    private func identity(_ modelID: String, _ step: Int?) -> TrainerModelFileIdentity {
        TrainerModelFileIdentity(modelID: modelID, trainingStep: step)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    private func allEntryNames(in directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    /// Put a different file at `url` the way another process would: written
    /// beside it, then renamed over it. Both files exist at once, so the
    /// newcomer cannot reuse the original's inode number.
    private func swapInAnotherFile(at url: URL, contents: String) throws {
        let sibling = url.deletingLastPathComponent().appendingPathComponent("incoming-\(UUID().uuidString)")
        try Data(contents.utf8).write(to: sibling)
        XCTAssertEqual(Darwin.rename(sibling.path, url.path), 0, String(cString: strerror(errno)))
    }

    // MARK: Rolling-file ownership rule (pure)

    func testSameLineAtTheStartStepContinuesTheLineage() {
        XCTAssertEqual(
            TrainerOutputFileGuard.rollingOverwriteVerdict(existing: identity("A", 5000), startModel: identity("A", 5000)),
            .continuesLineage)
    }

    func testSameLineBehindTheStartStepIsRefused() {
        // An earlier checkpoint of the line being continued (an enumerated
        // step file or a copy of one), reached by a mistyped --out-model.
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("A", 4000), startModel: identity("A", 5000)) else {
            return XCTFail("an earlier checkpoint of the same line must not be overwritten with later training")
        }
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("A", 0), startModel: identity("A", 5000)) else {
            return XCTFail("an earlier checkpoint of the same line must not be overwritten with later training")
        }
    }

    func testFileAheadOfTheStartModelIsRefused() {
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("A", 6000), startModel: identity("A", 5000)) else {
            return XCTFail("a rolling file ahead of the start model holds training that exists only there")
        }
    }

    func testAnotherModelLineIsRefused() {
        // An earlier run of the same command: same parent, but its own model ID.
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("B", 1000), startModel: identity("A", 5000)) else {
            return XCTFail("another run's output must not be overwritten")
        }
    }

    func testFreshRunContinuesNoLine() {
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("A", 1000), startModel: nil) else {
            return XCTFail("a fresh-init run continues no model line")
        }
    }

    func testMissingTrainingStepIsRefused() {
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("A", nil), startModel: identity("A", 5000)) else {
            return XCTFail("without both steps there is no telling whether training is lost")
        }
        guard case .refused = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: identity("A", 5000), startModel: identity("A", nil)) else {
            return XCTFail("without both steps there is no telling whether training is lost")
        }
    }

    // MARK: Rolling-file checks against the filesystem

    func testAbsentOutModelIsCreatedNew() throws {
        let plan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: root.appendingPathComponent("run-replay-latest.safetensors"),
            startModelURL: nil, startModel: nil, overwriteAuthorized: false)
        XCTAssertEqual(plan.disposition, .createNew)
        XCTAssertNil(plan.existingFileIdentity)
    }

    func testOutModelEqualToStartModelIsRefusedEvenWithTheOverrideFlag() throws {
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: 5000)
        for overwrite in [false, true] {
            XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
                outModelURL: start, startModelURL: start, startModel: identity("A", 5000),
                overwriteAuthorized: overwrite)) { error in
                XCTAssertEqual(error as? TrainerOutputFileError, .outModelIsStartModel(path: start.path))
            }
        }
    }

    func testOutModelReachingTheStartModelThroughASymlinkOrDotsIsRefused() throws {
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: 5000)
        let link = root.appendingPathComponent("alias.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: start)
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: link, startModelURL: start, startModel: identity("A", 5000), overwriteAuthorized: true))

        try FileManager.default.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: true)
        let dotted = root.appendingPathComponent("sub/../seed.safetensors")
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: dotted, startModelURL: start, startModel: identity("A", 5000), overwriteAuthorized: true))
    }

    func testOutModelHardLinkedToTheStartModelIsRefused() throws {
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: 5000)
        let hardLink = root.appendingPathComponent("hardlink.safetensors")
        try FileManager.default.linkItem(at: start, to: hardLink)
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: hardLink, startModelURL: start, startModel: identity("A", 5000),
            overwriteAuthorized: true)) { error in
            XCTAssertEqual(error as? TrainerOutputFileError, .outModelIsStartModel(path: hardLink.path))
        }
    }

    func testOutModelDifferingOnlyInCaseIsRefusedOnACaseInsensitiveVolume() throws {
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: 5000)
        let upper = root.appendingPathComponent("SEED.safetensors")
        guard FileManager.default.fileExists(atPath: upper.path) else {
            throw XCTSkip("the temporary directory is on a case-sensitive volume")
        }
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: upper, startModelURL: start, startModel: identity("A", 5000), overwriteAuthorized: true))
    }

    func testDirectoryAtOutModelIsRefusedAndSurvivesEvenWithTheOverrideFlag() throws {
        let folder = root.appendingPathComponent("run-replay-latest.safetensors", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: folder.appendingPathComponent("keep.txt"))
        for overwrite in [false, true] {
            XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
                outModelURL: folder, startModelURL: nil, startModel: nil, overwriteAuthorized: overwrite)) { error in
                XCTAssertEqual(error as? TrainerOutputFileError,
                               .outModelNotARegularFile(path: folder.path, kind: .directory))
            }
        }
        XCTAssertTrue(isDirectory(folder))
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent("keep.txt").path))
    }

    func testSymlinkAtOutModelIsRefused() throws {
        let target = try writeModelFile("elsewhere.safetensors", modelID: "Z", trainingStep: 1)
        let link = root.appendingPathComponent("run-replay-latest.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: link, startModelURL: nil, startModel: nil, overwriteAuthorized: true)) { error in
            XCTAssertEqual(error as? TrainerOutputFileError,
                           .outModelNotARegularFile(path: link.path, kind: .symbolicLink))
        }
    }

    func testExistingRollingFileOfTheContinuedLineIsAdopted() throws {
        // Continuing a segment from one of its own checkpoints into its rolling
        // file: both carry the segment's model ID, the rolling file no further on.
        let start = try writeModelFile("v5-cont-replay-step336610.safetensors", modelID: "A", trainingStep: 336610)
        let rolling = try writeModelFile("v5-cont-replay-latest.safetensors", modelID: "A", trainingStep: 336610)
        let plan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: rolling, startModelURL: start, startModel: identity("A", 336610), overwriteAuthorized: false)
        XCTAssertEqual(plan.disposition, .continueLineage(existing: identity("A", 336610)))
        XCTAssertEqual(plan.existingFileIdentity, try FileSafety.existingItem(at: rolling)?.identity)
    }

    /// The reviewed hole: `--out-model …-replay-step29000.safetensors` (a
    /// typo for the rolling file) named an enumerated checkpoint of the line
    /// being continued, which the header rule adopted.
    func testExistingEnumeratedCheckpointOfTheContinuedLineIsRefused() throws {
        // The start model: a copy of the segment's step-30000 state.
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: 30000)
        // An earlier checkpoint of the line (the header rule refuses it too
        // now), and one at the start model's own step (which passes the
        // header rule; only its name gives it away).
        for step in [29000, 30000] {
            let enumerated = try writeModelFile("v5-cont-replay-step\(step).safetensors", modelID: "A", trainingStep: step)
            let before = try Data(contentsOf: enumerated)
            XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
                outModelURL: enumerated, startModelURL: start, startModel: identity("A", 30000),
                overwriteAuthorized: false)) { error in
                XCTAssertEqual(error as? TrainerOutputFileError,
                               .outModelNamedLikeAnEnumeratedCheckpoint(path: enumerated.path, step: step))
            }
            XCTAssertEqual(try Data(contentsOf: enumerated), before, "a refused check must not touch the file")
        }
    }

    func testEnumeratedNameIsRefusedEvenWhenNothingIsThereYet() throws {
        for name in ["run-replay-step29000.safetensors", "run-vsuci-step7.safetensors", "run-step3000.safetensors"] {
            let url = root.appendingPathComponent(name)
            XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
                outModelURL: url, startModelURL: nil, startModel: nil, overwriteAuthorized: false)) { error in
                guard case .outModelNamedLikeAnEnumeratedCheckpoint(_, let step)? = error as? TrainerOutputFileError else {
                    return XCTFail("expected outModelNamedLikeAnEnumeratedCheckpoint for \(name), got \(error)")
                }
                XCTAssertEqual(step, EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem: name))
            }
        }
    }

    func testEnumeratedNameIsAllowedWithTheOverrideFlag() throws {
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: 30000)
        let enumerated = try writeModelFile("v5-cont-replay-step29000.safetensors", modelID: "A", trainingStep: 29000)
        let plan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: enumerated, startModelURL: start, startModel: identity("A", 30000), overwriteAuthorized: true)
        guard case .overwriteAuthorized = plan.disposition else {
            return XCTFail("--overwrite-out-model must allow the name, replacing the file there")
        }
        XCTAssertEqual(plan.existingFileIdentity, try FileSafety.existingItem(at: enumerated)?.identity)

        let absent = root.appendingPathComponent("v5-cont-replay-step31000.safetensors")
        XCTAssertEqual(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: absent, startModelURL: start, startModel: identity("A", 30000),
            overwriteAuthorized: true).disposition, .createNew)
    }

    func testRerunOfTheSameCommandIsRefusedUnlessOverwriteIsPassed() throws {
        // First run: seed A → rolling file saved under its own new ID B.
        let start = try writeModelFile("seed.safetensors", modelID: "A", trainingStep: nil)
        let rolling = try writeModelFile("seed-replay-latest.safetensors", modelID: "B", trainingStep: 42000)
        let before = try Data(contentsOf: rolling)
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: rolling, startModelURL: start, startModel: identity("A", nil), overwriteAuthorized: false)) { error in
            guard case .outModelBelongsToAnotherRun(let path, _)? = error as? TrainerOutputFileError else {
                return XCTFail("expected outModelBelongsToAnotherRun, got \(error)")
            }
            XCTAssertEqual(path, rolling.path)
        }
        XCTAssertEqual(try Data(contentsOf: rolling), before, "a refused check must not touch the file")

        let plan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: rolling, startModelURL: start, startModel: identity("A", nil), overwriteAuthorized: true)
        guard case .overwriteAuthorized = plan.disposition else {
            return XCTFail("--overwrite-out-model must allow replacing another run's regular file")
        }
        XCTAssertNotNil(plan.existingFileIdentity)
    }

    func testUnreadableRegularFileIsRefusedUnlessOverwriteIsPassed() throws {
        let junk = root.appendingPathComponent("notes-replay-latest.safetensors")
        try Data("not a model".utf8).write(to: junk)
        XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: junk, startModelURL: nil, startModel: nil, overwriteAuthorized: false)) { error in
            guard case .outModelUnreadable? = error as? TrainerOutputFileError else {
                return XCTFail("expected outModelUnreadable, got \(error)")
            }
        }
        let plan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: junk, startModelURL: nil, startModel: nil, overwriteAuthorized: true)
        guard case .overwriteAuthorized = plan.disposition else {
            return XCTFail("expected overwriteAuthorized")
        }
    }

    // MARK: Rolling writer

    func testRollingWriterReplacesOnlyTheFileItOwns() throws {
        let url = root.appendingPathComponent("run-replay-latest.safetensors")
        let writer = RollingTrainerModelWriter(url: url, plan: RollingOutputPlan(disposition: .createNew, existingFileIdentity: nil))
        try writer.write(Data("one".utf8))
        try writer.write(Data("two".utf8))
        XCTAssertEqual(try Data(contentsOf: url), Data("two".utf8))

        // Something else replaces the file mid-run: the next save must refuse.
        try swapInAnotherFile(at: url, contents: "someone else")
        XCTAssertThrowsError(try writer.write(Data("three".utf8))) { error in
            XCTAssertEqual(error as? FileSafetyError, .fileChangedSinceWritten(path: url.path))
        }
        XCTAssertEqual(try Data(contentsOf: url), Data("someone else".utf8))
        XCTAssertEqual(try allEntryNames(in: root), ["run-replay-latest.safetensors"], "no staging file may be left behind")
    }

    func testRollingWriterPlannedAsNewRefusesAFileThatAppearedSinceTheCheck() throws {
        let url = root.appendingPathComponent("run-replay-latest.safetensors")
        let writer = RollingTrainerModelWriter(url: url, plan: RollingOutputPlan(disposition: .createNew, existingFileIdentity: nil))
        try Data("arrived later".utf8).write(to: url)
        XCTAssertThrowsError(try writer.write(Data("mine".utf8))) { error in
            XCTAssertEqual(error as? FileSafetyError, .alreadyExists(path: url.path, kind: .regularFile))
        }
        XCTAssertEqual(try Data(contentsOf: url), Data("arrived later".utf8))
    }

    // MARK: Enumerated naming

    func testEnumeratedNamesMatchTheHistoricalScheme() {
        let replay = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("20260702-Qeu8-replay-latest.safetensors"), runTag: "replay")
        XCTAssertEqual(replay.fileName(trainerStep: 41000), "20260702-Qeu8-replay-step41000.safetensors")
        let vsuci = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("foo-vsuci-latest.safetensors"), runTag: "vsuci")
        XCTAssertEqual(vsuci.fileName(trainerStep: 7), "foo-vsuci-step7.safetensors")
        let plain = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("foo.safetensors"), runTag: "replay")
        XCTAssertEqual(plain.fileName(trainerStep: 3000), "foo-step3000.safetensors")
    }

    func testEnumeratedNameParsingAcceptsExactlyThisStemsStepFiles() {
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("20260702-Qeu8-replay-latest.safetensors"), runTag: "replay")
        XCTAssertEqual(naming.trainerStep(ofFileName: "20260702-Qeu8-replay-step41000.safetensors"), 41000)
        XCTAssertEqual(naming.trainerStep(ofFileName: "20260702-Qeu8-replay-step0.safetensors"), 0)
        // A resumed segment's own stem, a sibling run, markers and padding are not this stem's files.
        XCTAssertNil(naming.trainerStep(ofFileName: "20260702-Qeu8-resume2-replay-step41000.safetensors"))
        XCTAssertNil(naming.trainerStep(ofFileName: "20260702-Qeu8e5-replay-step41000.safetensors"))
        XCTAssertNil(naming.trainerStep(ofFileName: "20260702-Qeu8-replay-step49374-DO-NOT-RESUME.safetensors"))
        XCTAssertNil(naming.trainerStep(ofFileName: "20260702-Qeu8-replay-step041000.safetensors"))
        XCTAssertNil(naming.trainerStep(ofFileName: "20260702-Qeu8-replay-latest.safetensors"))
        XCTAssertNil(naming.trainerStep(ofFileName: "20260702-Qeu8-replay-step.safetensors"))
    }

    func testEnumeratedNameParsingRoundTripsAStemThatRepeatsTheMarker() {
        // The default --out-model when the start model is itself a rolling file.
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("a-replay-latest-replay-latest.safetensors"), runTag: "replay")
        let name = naming.fileName(trainerStep: 5000)
        XCTAssertEqual(naming.trainerStep(ofFileName: name), 5000)
    }

    func testAnyStemParserRecognizesEveryEnumeratedShape() {
        // Built by the namer itself, for every run kind and stem shape.
        let rollingNames = [
            "20260702-Qeu8-replay-latest.safetensors",
            "foo-vsuci-latest.safetensors",
            "foo.safetensors",
            "a-replay-latest-replay-latest.safetensors",
            "a-replay-latest-b.safetensors",
        ]
        for rollingName in rollingNames {
            for runTag in EnumeratedCheckpointNaming.allRunTags {
                let naming = EnumeratedCheckpointNaming(
                    rollingOutputURL: root.appendingPathComponent(rollingName), runTag: runTag)
                for step in [0, 7, 41000] {
                    let name = naming.fileName(trainerStep: step)
                    XCTAssertEqual(EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem: name), step, name)
                }
            }
        }
    }

    func testAnyStemParserLeavesRollingAndOtherNamesAlone() {
        for name in [
            "20260702-Qeu8-replay-latest.safetensors",
            "foo-vsuci-latest.safetensors",
            "seed.safetensors",
            "run-replay-step.safetensors",
            "run-replay-step041000.safetensors",
            "-step5.safetensors",
            "run-replay-step29000.json",
            "run-replay-step29000",
            "20260628-v5_5block_7x7_lnout-step39000-frozen.safetensors",
            "stepper-step-up.safetensors",
        ] {
            XCTAssertNil(EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem: name), name)
        }
    }

    // MARK: Enumerated pre-flight

    func testStemWithReachableStepFilesRefusesTheRun() throws {
        let rolling = root.appendingPathComponent("v5-cont-replay-latest.safetensors")
        let naming = EnumeratedCheckpointNaming(rollingOutputURL: rolling, runTag: "replay")
        for step in [1000, 2000, 336610] {
            try Data("segment 1".utf8).write(to: naming.url(trainerStep: step))
        }
        // No step limit: every step is reachable.
        XCTAssertThrowsError(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: naming, segmentStartTrainerStep: 0, stepLimit: nil)) { error in
            guard case .enumeratedStepsAlreadyPresent(let firstPath, let count, _, _, let suggestion)?
                    = error as? TrainerOutputFileError else {
                return XCTFail("expected enumeratedStepsAlreadyPresent, got \(error)")
            }
            XCTAssertEqual(firstPath, naming.url(trainerStep: 1000).path)
            XCTAssertEqual(count, 3)
            XCTAssertTrue(suggestion.contains("v5-cont-resumeN-replay-latest.safetensors"), suggestion)
        }
        // A step limit below every existing step reaches none of them.
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(naming: naming, segmentStartTrainerStep: 0, stepLimit: 999))
        XCTAssertEqual(
            try TrainerOutputFileGuard.reachableEnumeratedCheckpoints(naming: naming, segmentStartTrainerStep: 0, stepLimit: 2000).map(\.step),
            [1000, 2000])
    }

    func testNewSegmentStemPassesThePreflight() throws {
        let old = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("v5-cont-replay-latest.safetensors"), runTag: "replay")
        try Data("segment 1".utf8).write(to: old.url(trainerStep: 1000))
        let resumed = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("v5-cont-resume2-replay-latest.safetensors"), runTag: "replay")
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(naming: resumed, segmentStartTrainerStep: 0, stepLimit: nil))
    }

    func testMissingOutputDirectoryHasNoStepFiles() throws {
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("not-yet/run-replay-latest.safetensors"), runTag: "replay")
        XCTAssertEqual(try TrainerOutputFileGuard.reachableEnumeratedCheckpoints(naming: naming, segmentStartTrainerStep: 0, stepLimit: nil), [])
    }

    // MARK: Enumerated writer

    func testEnumeratedWriteNeverOverwritesAnotherRunsFile() throws {
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("run-replay-latest.safetensors"), runTag: "replay")
        let earlier = naming.url(trainerStep: 1000)
        try Data("earlier segment".utf8).write(to: earlier)
        let writer = EnumeratedCheckpointWriter(naming: naming)
        XCTAssertThrowsError(try writer.write(Data("this run".utf8), trainerStep: 1000)) { error in
            XCTAssertEqual(error as? TrainerOutputFileError,
                           .enumeratedCheckpointExists(path: earlier.path, step: 1000))
        }
        XCTAssertEqual(try Data(contentsOf: earlier), Data("earlier segment".utf8))
        XCTAssertEqual(try allEntryNames(in: root), [earlier.lastPathComponent], "no staging file may be left behind")
    }

    func testEnumeratedWriteNeverReplacesAFolder() throws {
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("run-replay-latest.safetensors"), runTag: "replay")
        let folder = naming.url(trainerStep: 2000)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: folder.appendingPathComponent("keep.txt"))
        let writer = EnumeratedCheckpointWriter(naming: naming)
        XCTAssertThrowsError(try writer.write(Data("this run".utf8), trainerStep: 2000))
        XCTAssertTrue(isDirectory(folder))
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent("keep.txt").path))
    }

    func testFinalSaveOnTheLastAutosaveStepReplacesThisRunsOwnFile() throws {
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("run-replay-latest.safetensors"), runTag: "replay")
        let writer = EnumeratedCheckpointWriter(naming: naming)
        let first = try writer.write(Data("autosave".utf8), trainerStep: 5000)
        XCTAssertEqual(first.outcome, .created)
        let second = try writer.write(Data("final".utf8), trainerStep: 5000)
        XCTAssertEqual(second.outcome, .replacedThisRunsEarlierSave)
        XCTAssertEqual(try Data(contentsOf: naming.url(trainerStep: 5000)), Data("final".utf8))
    }

    func testOwnFileSwappedOutUnderTheRunIsNotReplaced() throws {
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent("run-replay-latest.safetensors"), runTag: "replay")
        let writer = EnumeratedCheckpointWriter(naming: naming)
        _ = try writer.write(Data("autosave".utf8), trainerStep: 5000)
        let url = naming.url(trainerStep: 5000)
        try swapInAnotherFile(at: url, contents: "someone else")
        XCTAssertThrowsError(try writer.write(Data("final".utf8), trainerStep: 5000)) { error in
            XCTAssertEqual(error as? FileSafetyError, .fileChangedSinceWritten(path: url.path))
        }
        XCTAssertEqual(try Data(contentsOf: url), Data("someone else".utf8))
    }

    // MARK: Header identity

    func testIdentityIsReadFromTheHeader() throws {
        let url = try writeModelFile("m.safetensors", modelID: "20260701-3-AbCd", trainingStep: 1234)
        XCTAssertEqual(try TrainerModelFileIdentity.read(from: url), identity("20260701-3-AbCd", 1234))
        let noStep = try writeModelFile("n.safetensors", modelID: "20260701-4-EfGh", trainingStep: nil)
        XCTAssertEqual(try TrainerModelFileIdentity.read(from: noStep), identity("20260701-4-EfGh", nil))
    }
}
