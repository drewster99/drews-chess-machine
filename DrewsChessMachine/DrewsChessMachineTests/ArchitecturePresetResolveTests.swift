import XCTest
@testable import DrewsChessMachine

/// Covers `ArchitecturePresetStore.resolve(nameOrPath:)` — the resolution
/// behind `--new-model --architecture <value>`: a built-in preset name, a
/// user-saved preset name (with or without `.json`), or a path to a
/// `NamedArchitecture` JSON. Uses only built-in presets and temp files, so it
/// never touches the user's real `Presets/` folder.
final class ArchitecturePresetResolveTests: XCTestCase {

    private func writeTempJSON(_ named: NamedArchitecture, name: String = "myarch") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("arch-resolve-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("\(name).json")
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        try enc.encode(named).write(to: url)
        return url
    }

    func test_builtInName_resolves() throws {
        let (named, source) = try ArchitecturePresetStore.resolve(nameOrPath: "v4_5block_7x7")
        XCTAssertEqual(source, "v4_5block_7x7")
        XCTAssertEqual(named.architecture.parameterCount,
                       NetworkArchitecture.preset(.v4_5block_7x7).parameterCount)
    }

    func test_builtInName_withJsonSuffix_isStripped() throws {
        // `.json` on a NAME is tolerated → resolves to the same built-in.
        let (named, source) = try ArchitecturePresetStore.resolve(nameOrPath: "v4_5block_7x7.json")
        XCTAssertEqual(source, "v4_5block_7x7", "the .json suffix should be stripped from the name")
        XCTAssertEqual(named.architecture.parameterCount,
                       NetworkArchitecture.preset(.v4_5block_7x7).parameterCount)
    }

    func test_path_resolvesAndUsesFileStemAsSource() throws {
        let orig = NamedArchitecture(label: "T", architecture: .preset(.v3_8block_3x3))
        let url = try writeTempJSON(orig, name: "custom-net")
        let (named, source) = try ArchitecturePresetStore.resolve(nameOrPath: url.path)
        XCTAssertEqual(source, "custom-net", "source name should be the file stem")
        XCTAssertEqual(named.architecture.parameterCount, orig.architecture.parameterCount)
        XCTAssertEqual(named.architecture.inputEncoding, orig.architecture.inputEncoding)
    }

    func test_unknownNameOrPath_throwsNotFound() {
        XCTAssertThrowsError(
            try ArchitecturePresetStore.resolve(nameOrPath: "no-such-preset-\(UUID().uuidString)")
        ) { error in
            guard case ArchitecturePresetStore.StoreError.presetNotFound = error else {
                return XCTFail("expected .presetNotFound, got \(error)")
            }
        }
    }

    func test_malformedFileByPath_throws() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("arch-bad-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("bad.json")
        try Data("{ this is not valid json".utf8).write(to: url)
        // A file that exists but doesn't decode is an error, not a silent skip.
        XCTAssertThrowsError(try ArchitecturePresetStore.resolve(nameOrPath: url.path))
    }
}
