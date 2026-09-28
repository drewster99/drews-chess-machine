import XCTest
@testable import DrewsChessMachine

/// `ModelFileCatalog.scanSynchronously` orders lines by their newest file,
/// not by the modification time of their highest-step file: a line whose
/// most recent write was an earlier step is still the most recently active.
final class ModelFileCatalogActivityOrderTests: XCTestCase {

    private func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ModelFileCatalogActivityOrderTests-\(UUID().uuidString)", isDirectory: true)
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

    private func writeModel(_ folder: URL, name: String, modelID: String, step: Int, modified: Date) throws {
        let metadata: [String: String] = [
            "model_id": modelID,
            "training_step": String(step),
            "architecture": String(decoding: try JSONEncoder().encode(NetworkArchitecture.current), as: UTF8.self)
        ]
        let data = try SafetensorsFile.encode(tensors: [SafetensorsTensor(name: "w", shape: [2], data: [1, 2])], metadata: metadata)
        let url = folder.appendingPathComponent(name)
        try data.write(to: url)
        try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: url.path)
    }

    func testLinesSortByTheirNewestFileNotTheirHighestStep() throws {
        let folder = try makeFolder()
        let base = Date(timeIntervalSince1970: 1_750_000_000)
        // Line A's highest step is its oldest file; its lower step was written last.
        try writeModel(folder, name: "a-high.safetensors", modelID: "20260901-1-AAAA", step: 2000, modified: base)
        try writeModel(folder, name: "a-low.safetensors", modelID: "20260901-1-AAAA", step: 1000, modified: base.addingTimeInterval(200))
        try writeModel(folder, name: "b.safetensors", modelID: "20260902-1-BBBB", step: 500, modified: base.addingTimeInterval(100))

        let scan = try ModelFileCatalog.scanSynchronously(directory: folder)
        XCTAssertEqual(scan.lines.map(\.modelID), ["20260901-1-AAAA", "20260902-1-BBBB"])
        XCTAssertEqual(scan.lines.first?.latest.trainingStep, 2000, "latest is still chosen by step")
        XCTAssertTrue(scan.unreadable.isEmpty)
    }
}
