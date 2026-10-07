import XCTest
@testable import DrewsChessMachine

/// `ModelFileCatalog`: model files grouped into lines by the `model_id` in
/// their safetensors metadata, latest by training step — never by filename.
final class ModelFileCatalogTests: XCTestCase {

    private func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ModelFileCatalogTests-\(UUID().uuidString)", isDirectory: true)
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

    private func writeModel(_ folder: URL, name: String, modelID: String?, step: Int?) throws {
        var metadata: [String: String] = ["created_at_unix": "1759000000"]
        if let modelID { metadata["model_id"] = modelID }
        if let step { metadata["training_step"] = String(step) }
        metadata["architecture"] = String(decoding: try JSONEncoder().encode(NetworkArchitecture.current), as: UTF8.self)
        let data = try SafetensorsFile.encode(tensors: [SafetensorsTensor(name: "w", shape: [2], data: [1, 2])], metadata: metadata)
        try data.write(to: folder.appendingPathComponent(name))
    }

    func testLinesGroupByModelIDAndPickTheHighestStep() throws {
        let folder = try makeFolder()
        // Filenames deliberately disagree with the metadata.
        try writeModel(folder, name: "run-step9000.safetensors", modelID: "20260901-1-AAAA", step: 1000)
        try writeModel(folder, name: "run-step1000.safetensors", modelID: "20260901-1-AAAA", step: 9000)
        try writeModel(folder, name: "run-latest.safetensors", modelID: "20260901-1-AAAA", step: 5000)
        try writeModel(folder, name: "other.safetensors", modelID: "20260902-1-BBBB", step: nil)
        try writeModel(folder, name: "no-id.safetensors", modelID: nil, step: 3)
        try Data("not a model".utf8).write(to: folder.appendingPathComponent("junk.safetensors"))
        try Data("ignored".utf8).write(to: folder.appendingPathComponent("notes.txt"))

        let scan = try ModelFileCatalog.scanSynchronously(directory: folder)
        XCTAssertEqual(Set(scan.lines.map(\.modelID)), ["20260901-1-AAAA", "20260902-1-BBBB"])
        let line = try XCTUnwrap(scan.lines.first { $0.modelID == "20260901-1-AAAA" })
        XCTAssertEqual(line.files.map(\.trainingStep), [9000, 5000, 1000])
        XCTAssertEqual(line.latest.url.lastPathComponent, "run-step1000.safetensors")
        XCTAssertEqual(line.latest.architectureLabel, NetworkArchitecture.current.shortLabel)
        XCTAssertEqual(scan.unreadable.map(\.url.lastPathComponent), ["junk.safetensors", "no-id.safetensors"])
    }

    /// A session saved before safetensors holds `champion.dcmmodel`; it is
    /// listed like any other champion, not reported as missing. A session
    /// with no champion at all is reported under its own folder's name,
    /// since every session's champion has the same file name.
    func testSessionChampionsReadLegacyDcmmodelAndNameTheSession() throws {
        let folder = try makeFolder()
        let current = folder.appendingPathComponent("20261001-000000-20260930-1-NEWW-manual.dcmsession", isDirectory: true)
        let legacy = folder.appendingPathComponent("20260529-182349-20260525-2-IWkd-manual.dcmsession", isDirectory: true)
        let empty = folder.appendingPathComponent("20260601-000000-20260601-1-EMPT-manual.dcmsession", isDirectory: true)
        for session in [current, legacy, empty] {
            try FileManager.default.createDirectory(at: session, withIntermediateDirectories: false)
        }
        try writeModel(current, name: SessionCheckpointLayout.championFilename, modelID: "20260930-1-NEWW", step: 1000)
        let legacyMetadata = ModelCheckpointMetadata(creator: "manual", trainingStep: 4200, parentModelID: "20260525-1-PRNT", notes: "legacy")
        try ModelCheckpointFile(modelID: "20260525-2-IWkd", createdAtUnix: 1_780_000_000, metadata: legacyMetadata, weights: [[1, 2]])
            .encode()
            .write(to: legacy.appendingPathComponent(SessionCheckpointLayout.legacyChampionFilename))

        let scan = try ModelFileCatalog.scanSessionChampions(directory: folder)

        XCTAssertEqual(Set(scan.champions.map(\.entry.modelID)), ["20260930-1-NEWW", "20260525-2-IWkd"])
        let legacyChampion = try XCTUnwrap(scan.champions.first { $0.entry.modelID == "20260525-2-IWkd" })
        XCTAssertEqual(legacyChampion.sessionName, "20260529-182349-20260525-2-IWkd-manual")
        XCTAssertEqual(legacyChampion.entry.url.lastPathComponent, SessionCheckpointLayout.legacyChampionFilename)
        XCTAssertEqual(legacyChampion.entry.trainingStep, 4200)
        XCTAssertEqual(legacyChampion.entry.createdAt, Date(timeIntervalSince1970: 1_780_000_000))
        XCTAssertEqual(legacyChampion.entry.parentModelID, "20260525-1-PRNT")
        XCTAssertEqual(legacyChampion.entry.creator, "manual")
        XCTAssertNil(legacyChampion.entry.contentSHA256)
        XCTAssertEqual(legacyChampion.entry.lineage, .unrecorded(formatVersion: ArchitectureFormat.unversionedLegacyVersion))
        let legacyPreset = try XCTUnwrap(NetworkArchitecture.legacyDcmmodelArchHashes[ModelCheckpointFile.archHash(for: .current)])
        XCTAssertEqual(legacyChampion.entry.architectureLabel, NetworkArchitecture.preset(legacyPreset).shortLabel)

        XCTAssertEqual(scan.unreadable.map(\.displayName), ["20260601-000000-20260601-1-EMPT-manual.dcmsession/champion.safetensors"])
        XCTAssertEqual(scan.unreadable.map(\.reason), [ModelFileCatalogError.noSessionChampion.localizedDescription])
    }

    /// A damaged `.dcmmodel` champion is reported with the decoder's reason,
    /// never listed.
    func testCorruptLegacyChampionIsReported() throws {
        let folder = try makeFolder()
        let session = folder.appendingPathComponent("old.dcmsession", isDirectory: true)
        try FileManager.default.createDirectory(at: session, withIntermediateDirectories: false)
        var data = try ModelCheckpointFile(
            modelID: "20260525-2-IWkd", createdAtUnix: 1_780_000_000,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: 1, parentModelID: "", notes: ""),
            weights: [[1, 2]]).encode()
        data[data.count - 40] ^= 0xFF
        try data.write(to: session.appendingPathComponent(SessionCheckpointLayout.legacyChampionFilename))

        let scan = try ModelFileCatalog.scanSessionChampions(directory: folder)

        XCTAssertTrue(scan.champions.isEmpty)
        XCTAssertEqual(scan.unreadable.map(\.displayName), ["old.dcmsession/champion.dcmmodel"])
        XCTAssertTrue(try XCTUnwrap(scan.unreadable.first).reason.contains("not a readable .dcmmodel model"))
    }

    func testUnreadableDisplayNameOutsideASessionIsTheFileName() {
        let file = UnreadableModelFile(url: URL(fileURLWithPath: "/m/Models/x.safetensors"), reason: "r")
        XCTAssertEqual(file.displayName, "x.safetensors")
    }

    func testStepBeatsNoStep() {
        let date = Date()
        func entry(_ step: Int?, modified: Date) -> ModelFileEntry {
            ModelFileEntry(url: URL(fileURLWithPath: "/x/\(step ?? -1)"), modelID: "m", trainingStep: step, createdAt: nil, architectureLabel: "a", fileModifiedAt: modified)
        }
        XCTAssertTrue(ModelFileCatalog.isMoreAdvanced(entry(1, modified: date), entry(nil, modified: date.addingTimeInterval(100))))
        XCTAssertTrue(ModelFileCatalog.isMoreAdvanced(entry(nil, modified: date.addingTimeInterval(100)), entry(nil, modified: date)))
    }
}
