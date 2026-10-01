//
//  ParametersFileWriterTests.swift
//  DrewsChessMachineTests
//
//  `--create-parameters-file <path> --force` once deleted a whole folder:
//  given `documentation/`, it treated the folder as the JSON file's name and
//  `--force` removed it recursively to write the file in its place. The help
//  text documents the path as a folder ("default: ./"). These pin that a
//  folder path writes `parameters.json` / `parameters.md` inside the folder,
//  and that `--force` replaces only regular files, never a folder.
//

import XCTest
@testable import DrewsChessMachine

final class ParametersFileWriterTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ParametersFileWriterTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    private func isRegularFile(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && !directory.boolValue
    }

    /// A folder with something in it, to show it survives.
    private func makeFolder(_ name: String) throws -> URL {
        let folder = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: folder.appendingPathComponent("keep.txt"))
        return folder
    }

    func testForceWithAFolderPathWritesInsideItAndKeepsItsContents() throws {
        let folder = try makeFolder("documentation")
        let written = try ParametersFileWriter.writeDefaults(path: folder.path, force: true)
        XCTAssertTrue(isDirectory(folder), "the folder must survive")
        XCTAssertTrue(isRegularFile(folder.appendingPathComponent("keep.txt")), "the folder's contents must survive")
        XCTAssertEqual(written.json.standardizedFileURL, folder.appendingPathComponent("parameters.json").standardizedFileURL)
        XCTAssertEqual(written.markdown.standardizedFileURL, folder.appendingPathComponent("parameters.md").standardizedFileURL)
        XCTAssertTrue(isRegularFile(written.json))
        XCTAssertTrue(isRegularFile(written.markdown))
    }

    func testFolderPathWithTrailingSlashWritesInsideIt() throws {
        let folder = try makeFolder("docs")
        let written = try ParametersFileWriter.writeDefaults(path: folder.path + "/", force: false)
        XCTAssertEqual(written.json.standardizedFileURL, folder.appendingPathComponent("parameters.json").standardizedFileURL)
        XCTAssertTrue(isRegularFile(folder.appendingPathComponent("keep.txt")))
    }

    func testTrailingSlashForAMissingFolderIsAnError() throws {
        let missing = root.appendingPathComponent("missing", isDirectory: true)
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: missing.path + "/", force: true))
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing.path))
    }

    func testFilePathWritesTheFileAndASiblingMarkdown() throws {
        let path = root.appendingPathComponent("custom.json")
        let written = try ParametersFileWriter.writeDefaults(path: path.path, force: false)
        XCTAssertEqual(written.json.standardizedFileURL, path.standardizedFileURL)
        XCTAssertEqual(written.markdown.standardizedFileURL, root.appendingPathComponent("custom.md").standardizedFileURL)
        XCTAssertTrue(isRegularFile(written.json))
        XCTAssertTrue(isRegularFile(written.markdown))
    }

    func testExistingFileWithoutForceIsRefused() throws {
        let path = root.appendingPathComponent("parameters.json")
        try Data("mine".utf8).write(to: path)
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: path.path, force: false))
        XCTAssertEqual(try Data(contentsOf: path), Data("mine".utf8))
    }

    func testForceReplacesExistingRegularFiles() throws {
        let path = root.appendingPathComponent("parameters.json")
        try Data("old".utf8).write(to: path)
        try Data("old".utf8).write(to: root.appendingPathComponent("parameters.md"))
        _ = try ParametersFileWriter.writeDefaults(path: path.path, force: true)
        XCTAssertNotEqual(try Data(contentsOf: path), Data("old".utf8))
        XCTAssertNotEqual(try Data(contentsOf: root.appendingPathComponent("parameters.md")), Data("old".utf8))
    }

    /// The markdown destination is someone's file too: `--create-parameters-file
    /// ROADMAP` must not replace `ROADMAP.md` without `--force`.
    func testExistingMarkdownWithoutForceIsRefused() throws {
        let markdown = root.appendingPathComponent("ROADMAP.md")
        try Data("plans".utf8).write(to: markdown)
        let path = root.appendingPathComponent("ROADMAP")
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: path.path, force: false))
        XCTAssertEqual(try Data(contentsOf: markdown), Data("plans".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path), "nothing is written when the request is refused")
    }

    func testAPathEndingInMarkdownIsRefused() throws {
        let path = root.appendingPathComponent("notes.md")
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: path.path, force: true))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    }

    /// `notes.MD` makes the markdown path `notes.md` — the same file on a
    /// case-insensitive volume. The same-path check once compared case, so
    /// without `--force` the JSON was published and the markdown write then
    /// failed on it (a partial write despite "nothing is written when
    /// refused"), and with `--force` the markdown silently replaced the JSON.
    func testAPathEndingInMarkdownInAnyCaseIsRefusedAndNothingIsWritten() throws {
        for name in ["notes.MD", "notes.Md", "notes.mD"] {
            for force in [false, true] {
                let path = root.appendingPathComponent(name)
                XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: path.path, force: force)) { error in
                    XCTAssertEqual(error as? ParametersFileWriterError, .jsonAndMarkdownSamePath(path.path),
                                   "\(name), force: \(force)")
                }
                XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [],
                               "nothing may be written for \(name) (force: \(force))")
            }
        }
    }

    /// A JSON path with no extension whose staging name fits in `NAME_MAX`
    /// but whose markdown sibling's (three bytes longer) does not: the JSON
    /// publishes, then the markdown fails at staging. A deterministic way to
    /// fail the second write after the first succeeded.
    private func jsonPathWhoseMarkdownCannotBeStaged() -> URL {
        let length = Int(NAME_MAX) - FileSafety.temporarySiblingNameOverhead - ".md".utf8.count + 1
        return root.appendingPathComponent(String(repeating: "p", count: length))
    }

    func testMarkdownFailureAfterANewJSONRemovesThatJSONAgain() throws {
        let jsonURL = jsonPathWhoseMarkdownCannotBeStaged()
        let markdownURL = jsonURL.appendingPathExtension("md")
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: jsonURL.path, force: false)) { error in
            guard case .markdownNotWritten(let path, _, let jsonOutcome)? = error as? ParametersFileWriterError else {
                return XCTFail("expected markdownNotWritten, got \(error)")
            }
            XCTAssertEqual(path, markdownURL.path)
            XCTAssertTrue(jsonOutcome.contains("removed again"), jsonOutcome)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [],
                       "the JSON written before the markdown failed must be removed, and no staging file left")
    }

    func testMarkdownFailureAfterAForcedJSONReplacementSaysTheJSONWasReplaced() throws {
        let jsonURL = jsonPathWhoseMarkdownCannotBeStaged()
        try Data("old".utf8).write(to: jsonURL)
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: jsonURL.path, force: true)) { error in
            guard case .markdownNotWritten(_, _, let jsonOutcome)? = error as? ParametersFileWriterError else {
                return XCTFail("expected markdownNotWritten, got \(error)")
            }
            XCTAssertTrue(jsonOutcome.contains("cannot be restored"), jsonOutcome)
        }
        XCTAssertEqual(try Data(contentsOf: jsonURL), try TrainingParameters.defaultsJSON())
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [jsonURL.lastPathComponent],
                       "no staging file may be left behind")
    }

    /// A folder sitting at the markdown file's path is never removed either.
    func testForceRefusesToReplaceAFolderAtTheMarkdownPath() throws {
        let markdownFolder = try makeFolder("custom.md")
        let path = root.appendingPathComponent("custom.json")
        XCTAssertThrowsError(try ParametersFileWriter.writeDefaults(path: path.path, force: true))
        XCTAssertTrue(isRegularFile(markdownFolder.appendingPathComponent("keep.txt")))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path), "nothing is written when the request is refused")
    }
}
