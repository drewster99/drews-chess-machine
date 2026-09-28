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

    func testStepBeatsNoStep() {
        let date = Date()
        func entry(_ step: Int?, modified: Date) -> ModelFileEntry {
            ModelFileEntry(url: URL(fileURLWithPath: "/x/\(step ?? -1)"), modelID: "m", trainingStep: step, createdAt: nil, architectureLabel: "a", fileModifiedAt: modified)
        }
        XCTAssertTrue(ModelFileCatalog.isMoreAdvanced(entry(1, modified: date), entry(nil, modified: date.addingTimeInterval(100))))
        XCTAssertTrue(ModelFileCatalog.isMoreAdvanced(entry(nil, modified: date.addingTimeInterval(100)), entry(nil, modified: date)))
    }
}
