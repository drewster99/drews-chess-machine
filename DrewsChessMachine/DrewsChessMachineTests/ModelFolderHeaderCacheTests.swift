import XCTest
@testable import DrewsChessMachine

/// `ModelFolderHeaderCache` — a model folder's headers read incrementally
/// (follow-lineage plan §3.2): one listing and one `stat` per file, a header
/// read only for a new or changed file, the catalog's listing exactly, and
/// lineage facts from the `dcm_lineage` record, never its mirror keys.
final class ModelFolderHeaderCacheTests: XCTestCase {

    private func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ModelFolderHeaderCacheTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        return folder
    }

    /// A small safetensors file whose header is a model file's: model ID,
    /// format version, training step and lineage record. The record is not
    /// optional: every file written at the current format carries one
    /// (required from v7), and from v11 a file stating a `training_step`
    /// without one is unreadable, since its segment step is the record's.
    private func headerFile(modelID: String, step: Int, record: LineageRecord, extra: [String: String] = [:], payload: [Float] = [1, 2]) throws -> Data {
        var metadata: [String: String] = [
            "model_id": modelID,
            "created_at_unix": "1790000000",
            "training_step": String(step),
            SafetensorsModelIO.Key.formatVersion: SafetensorsModelIO.formatVersion,
            LineageRecord.metadataKey: try record.jsonText(),
        ]
        metadata.merge(extra) { _, new in new }
        return try SafetensorsFile.encode(tensors: [SafetensorsTensor(name: "w", shape: [payload.count], data: payload)], metadata: metadata)
    }

    private func scan(_ folder: URL, previous: ModelFolderHeaderCache = .empty) throws -> ModelFolderScan {
        try ModelFolderHeaderCache.scanSynchronously(directory: folder, previous: previous)
    }

    private func record(localStep: Int = 1000) throws -> LineageRecord {
        try LineageTestRuns.fresh(modelID: "20261006-1-AAAA", startedUnix: 1_000).record(localStep: localStep, at: 2_000)
    }

    func testStagingFilesAreNeverListed() throws {
        let folder = try makeFolder()
        let data = try headerFile(modelID: "20261006-1-AAAA", step: 1, record: try record(localStep: 1))
        try data.write(to: folder.appendingPathComponent("model.safetensors"))
        try data.write(to: folder.appendingPathComponent(".model.safetensors.\(UUID().uuidString).tmp"))
        try data.write(to: folder.appendingPathComponent("model2.safetensors.tmp"))
        try data.write(to: folder.appendingPathComponent(".hidden.safetensors"))
        let result = try scan(folder)
        XCTAssertEqual(result.listed, 1)
        XCTAssertEqual(result.entries.map(\.url.lastPathComponent), ["model.safetensors"])
        XCTAssertEqual(result.unreadable, [])
    }

    func testUnchangedFilesAreNotReadAgain() throws {
        let folder = try makeFolder()
        for index in 0..<3 {
            try headerFile(modelID: "20261006-\(index + 1)-AAAA", step: index, record: try record()).write(to: folder.appendingPathComponent("m\(index).safetensors"))
        }
        try Data("not a model".utf8).write(to: folder.appendingPathComponent("junk.safetensors"))
        let first = try scan(folder)
        XCTAssertEqual(first.headersRead, 4)
        XCTAssertEqual(first.reused, 0)
        let second = try scan(folder, previous: first.cache)
        XCTAssertEqual(second.headersRead, 0, "nothing changed, so no header is read")
        XCTAssertEqual(second.reused, 4)
        XCTAssertEqual(second.entries, first.entries)
        XCTAssertEqual(second.unreadable, first.unreadable)
    }

    func testAFileRenamedOverTheSamePathIsReadAgain() throws {
        let folder = try makeFolder()
        let url = folder.appendingPathComponent("run-replay-latest.safetensors")
        let first = try headerFile(modelID: "20261006-1-AAAA", step: 1000, record: try record(localStep: 1000))
        let second = try headerFile(modelID: "20261006-1-BBBB", step: 2000, record: try record(localStep: 2000))
        XCTAssertEqual(first.count, second.count, "same size, so only the identity tells them apart")
        try first.write(to: url)
        let modified = try XCTUnwrap(try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)
        let before = try scan(folder)
        _ = try FileSafety.replaceRegularFile(second, at: url, expectedIdentity: nil)
        try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: url.path)
        let after = try scan(folder, previous: before.cache)
        XCTAssertEqual(after.headersRead, 1, "a new inode is a new file")
        XCTAssertEqual(after.entries.map(\.modelID), ["20261006-1-BBBB"])
    }

    func testRemovedFileLeavesTheScan() throws {
        let folder = try makeFolder()
        let keep = folder.appendingPathComponent("keep.safetensors")
        let gone = folder.appendingPathComponent("gone.safetensors")
        try headerFile(modelID: "20261006-1-KEEP", step: 1, record: try record(localStep: 1)).write(to: keep)
        try headerFile(modelID: "20261006-1-GONE", step: 1, record: try record(localStep: 1)).write(to: gone)
        let before = try scan(folder)
        let listedGone = try XCTUnwrap(before.entries.first { $0.modelID == "20261006-1-GONE" }?.url)
        XCTAssertNotNil(before.cache.fingerprint(of: listedGone))
        try FileManager.default.removeItem(at: gone)
        let after = try scan(folder, previous: before.cache)
        XCTAssertEqual(after.entries.map(\.modelID), ["20261006-1-KEEP"])
        XCTAssertNil(after.cache.fingerprint(of: listedGone), "a removed file leaves the cache")
    }

    func testTruncatedFileIsUnreadableUntilItChanges() throws {
        let folder = try makeFolder()
        let url = folder.appendingPathComponent("copying.safetensors")
        let full = try headerFile(modelID: "20261006-1-AAAA", step: 1, record: try record(localStep: 1))
        try full.prefix(20).write(to: url)
        // The listing names the folder by its resolved path (/private/var),
        // so files are compared by name.
        let first = try scan(folder)
        XCTAssertEqual(first.unreadable.map(\.url.lastPathComponent), [url.lastPathComponent])
        XCTAssertEqual(first.headersRead, 1)
        let second = try scan(folder, previous: first.cache)
        XCTAssertEqual(second.headersRead, 0, "an unchanged unreadable file is not read again")
        XCTAssertEqual(second.unreadable.map(\.url.lastPathComponent), [url.lastPathComponent])
        try full.write(to: url)
        let third = try scan(folder, previous: second.cache)
        XCTAssertEqual(third.headersRead, 1, "it grew, so it is read again")
        XCTAssertEqual(third.entries.map(\.modelID), ["20261006-1-AAAA"])
        XCTAssertEqual(third.unreadable, [])
    }

    func testUnlistableFolderThrows() throws {
        let folder = try makeFolder().appendingPathComponent("absent", isDirectory: true)
        XCTAssertThrowsError(try scan(folder)) { error in
            guard case .folderUnreadable(let directory, _)? = error as? ModelFolderScanError else {
                return XCTFail("expected folderUnreadable, got \(error)")
            }
            XCTAssertEqual(directory, folder)
        }
    }

    func testCatalogScanMatchesTheUncachedScan() throws {
        let folder = try makeFolder()
        try headerFile(modelID: "20261006-1-AAAA", step: 1000, record: try record(localStep: 1000)).write(to: folder.appendingPathComponent("a-1000.safetensors"))
        try headerFile(modelID: "20261006-1-AAAA", step: 2000, record: try record(localStep: 2000)).write(to: folder.appendingPathComponent("a-2000.safetensors"))
        try headerFile(modelID: "20261006-2-BBBB", step: 5, record: try record(localStep: 5)).write(to: folder.appendingPathComponent("b.safetensors"))
        try Data("not a model".utf8).write(to: folder.appendingPathComponent("junk.safetensors"))
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("dir.safetensors"), withIntermediateDirectories: false)

        // The uncached reading: every listed file through the catalog's own
        // per-file reader.
        let listed = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "safetensors" }
        var expectedEntries: [URL: ModelFileEntry] = [:]
        var expectedUnreadable: Set<URL> = []
        for url in listed {
            do {
                expectedEntries[url] = try ModelFileCatalog.entry(for: url)
            } catch {
                expectedUnreadable.insert(url)
            }
        }
        let cold = try scan(folder)
        let warm = try scan(folder, previous: cold.cache)
        for result in [cold, warm] {
            XCTAssertEqual(Dictionary(uniqueKeysWithValues: result.entries.map { ($0.url, $0) }), expectedEntries)
            XCTAssertEqual(Set(result.unreadable.map(\.url)), expectedUnreadable)
        }
        let catalog = try ModelFileCatalog.scanSynchronously(directory: folder)
        XCTAssertEqual(Set(catalog.lines.flatMap(\.files).map(\.url)), Set(expectedEntries.keys))
        XCTAssertEqual(Set(catalog.unreadable.map(\.url)), expectedUnreadable)
        XCTAssertEqual(catalog.lines.first { $0.modelID == "20261006-1-AAAA" }?.files.map(\.trainingStep), [2000, 1000])
    }

    func testNonRegularItemsAreListedUnreadableNotDropped() throws {
        let folder = try makeFolder()
        let directory = folder.appendingPathComponent("x.safetensors")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        let dangling = folder.appendingPathComponent("y.safetensors")
        try FileManager.default.createSymbolicLink(at: dangling, withDestinationURL: folder.appendingPathComponent("nowhere.safetensors"))
        let result = try scan(folder)
        XCTAssertEqual(result.listed, 2)
        XCTAssertEqual(Set(result.unreadable.map(\.url.lastPathComponent)), ["x.safetensors", "y.safetensors"])
        XCTAssertEqual(result.entries, [])
    }

    func testLinkTargetChangeIsReadAgain() throws {
        let folder = try makeFolder()
        let targets = try makeFolder()
        let a = targets.appendingPathComponent("a.safetensors")
        let b = targets.appendingPathComponent("b.safetensors")
        try headerFile(modelID: "20261006-1-AAAA", step: 1, record: try record(localStep: 1)).write(to: a)
        try headerFile(modelID: "20261006-1-BBBB", step: 1, record: try record(localStep: 1)).write(to: b)
        let link = folder.appendingPathComponent("current.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: a)
        let before = try scan(folder)
        XCTAssertEqual(before.entries.map(\.modelID), ["20261006-1-AAAA"])
        try FileManager.default.removeItem(at: link)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: b)
        let after = try scan(folder, previous: before.cache)
        XCTAssertEqual(after.headersRead, 1, "the cache keys on what the link resolves to")
        XCTAssertEqual(after.entries.map(\.modelID), ["20261006-1-BBBB"])
    }

    /// A current-format file that states a `training_step` but carries no
    /// `dcm_lineage` record is malformed: from v11 its segment step is the
    /// record's, and lineage is required from v7. No writer of this app
    /// produces one (every v7+ header on the owner's Mac carries a record),
    /// so the scan must show it as an unreadable entry naming the missing
    /// record — never drop it, and never list it with a step read some
    /// other way.
    func testCurrentFormatFileWithoutItsLineageRecordIsListedUnreadable() throws {
        let folder = try makeFolder()
        let metadata: [String: String] = [
            "model_id": "20261006-1-AAAA",
            "created_at_unix": "1790000000",
            "training_step": "1000",
            SafetensorsModelIO.Key.formatVersion: SafetensorsModelIO.formatVersion,
        ]
        let data = try SafetensorsFile.encode(tensors: [SafetensorsTensor(name: "w", shape: [2], data: [1, 2])], metadata: metadata)
        try data.write(to: folder.appendingPathComponent("no-record.safetensors"))
        let result = try scan(folder)
        XCTAssertEqual(result.listed, 1)
        XCTAssertEqual(result.entries, [], "no step is invented for it")
        XCTAssertEqual(result.unreadable.map(\.url.lastPathComponent), ["no-record.safetensors"])
        let reason = try XCTUnwrap(result.unreadable.first?.reason)
        XCTAssertTrue(reason.contains(LineageRecord.metadataKey), "the reason names the missing record: \(reason)")
    }

    func testLineageFactsComeFromTheRecordNotTheMirrorKeys() throws {
        let folder = try makeFolder()
        let lineage = try record(localStep: 31000)
        let mirrors = [
            "lineage_run_id": "00000000-0000-0000-0000-00000000BEEF",
            "lineage_segment_index": "7",
            "cum_trainer_step": "1",
        ]
        try headerFile(modelID: "20261006-1-AAAA", step: 31000, record: lineage, extra: mirrors).write(to: folder.appendingPathComponent("m.safetensors"))
        let entry = try XCTUnwrap(try scan(folder).entries.first)
        guard case .recorded(let position)? = entry.lineage else {
            return XCTFail("expected a recorded lineage, got \(String(describing: entry.lineage))")
        }
        XCTAssertEqual(position.lineageRunID, lineage.run.lineageRunID)
        XCTAssertEqual(position.segmentIndex, 0)
        XCTAssertEqual(position.cumTrainerStep, lineage.steps.cumTrainerStep)
        XCTAssertEqual(position.segmentLocalStep, 31000)
        XCTAssertEqual(position.segmentChain, [lineage.run.segmentID])
        XCTAssertNotNil(entry.contentSHA256)
    }
}
