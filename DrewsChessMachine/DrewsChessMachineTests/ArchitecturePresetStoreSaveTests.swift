//
//  ArchitecturePresetStoreSaveTests.swift
//  DrewsChessMachineTests
//
//  Build New Model's "Save as Preset" passed the user-typed name straight
//  into `Presets/<name>.json`: a name with `/` or `..` wrote outside the
//  Presets folder, and saving under an existing name silently replaced that
//  preset. These pin the name rules, that the file always lands directly in
//  the Presets folder, that an existing preset is replaced only with the
//  caller's explicit confirmation (`replacingExisting: true`), and that
//  only a regular file is ever replaced. Each test saves into its own
//  temporary Presets folder.
//

import XCTest
@testable import DrewsChessMachine

final class ArchitecturePresetStoreSaveTests: XCTestCase {

    private var root: URL!
    private var presets: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ArchitecturePresetStoreSaveTests-\(UUID().uuidString)", isDirectory: true)
        presets = root.appendingPathComponent("Presets", isDirectory: true)
        try FileManager.default.createDirectory(at: presets, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private let architecture = NetworkArchitecture.current

    @discardableResult
    private func save(_ name: String, label: String = "label", replacingExisting: Bool = false) throws -> URL {
        try ArchitecturePresetStore.save(
            name: name, label: label, architecture: architecture,
            replacingExisting: replacingExisting, presetsDirectory: presets)
    }

    private func entries(of folder: URL) throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: folder.path))
    }

    private func assertInvalidName(_ name: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try ArchitecturePresetStore.validatePresetName(name), file: file, line: line) { error in
            guard case ArchitecturePresetStore.StoreError.invalidPresetName(let reported, _) = error else {
                XCTFail("expected invalidPresetName for \(name.debugDescription), got \(error)", file: file, line: line)
                return
            }
            XCTAssertEqual(reported, name, file: file, line: line)
        }
    }

    // MARK: - Name rules

    func testNamesThatCouldEscapeOrHideAreRejected() {
        for name in ["", ".", "..", "../escape", "a/b", "/abs", "..\u{2215}x", ".hidden", "a:b", "a\\b", "tab\there", "new\nline"] {
            assertInvalidName(name)
        }
    }

    func testLeadingOrTrailingWhitespaceIsRejected() {
        for name in [" lead", "trail ", "\tlead"] {
            assertInvalidName(name)
        }
    }

    /// `resolve(nameOrPath:)` strips one `.json`, so a preset saved under a
    /// name ending in `.json` could never be reached by that name.
    func testNameEndingInJSONIsRejected() {
        assertInvalidName("mine.json")
        assertInvalidName("mine.JSON")
    }

    /// A save stages `<name>.json` under a longer hidden name first, so the
    /// bound is set by that name, not the final one. The longest accepted
    /// name must actually save — a name that fits only as `<name>.json`
    /// once passed the check and then failed at staging with
    /// `ENAMETOOLONG`.
    func testOverlongNameIsRejectedButTheLongestThatFitsIsAcceptedAndSaves() throws {
        let longestLength = Int(NAME_MAX) - ".json".utf8.count - FileSafety.temporarySiblingNameOverhead
        let longest = String(repeating: "a", count: longestLength)
        XCTAssertNoThrow(try ArchitecturePresetStore.validatePresetName(longest))
        assertInvalidName(longest + "a")
        // The old bound: fits as the final file name, but not as the staging name.
        assertInvalidName(String(repeating: "a", count: Int(NAME_MAX) - ".json".utf8.count))

        let url = try save(longest, label: "Longest")
        XCTAssertEqual(url.lastPathComponent, longest + ".json")
        XCTAssertEqual(try ArchitecturePresetStore.loadFile(at: url).label, "Longest")
        XCTAssertEqual(try entries(of: presets), [longest + ".json"], "no staging file may be left behind")
    }

    /// The overhead is measured from the name `FileSafety` actually stages
    /// under, so the bound above cannot drift from it.
    func testStagingNameOverheadMatchesTheRealStagingName() {
        let destination = presets.appendingPathComponent("mine.json")
        let staging = FileSafety.temporarySibling(of: destination)
        XCTAssertEqual(staging.lastPathComponent.utf8.count - destination.lastPathComponent.utf8.count,
                       FileSafety.temporarySiblingNameOverhead)
    }

    func testBuiltInNamesStayReserved() throws {
        let builtIn = NetworkArchitecture.Preset.current.rawValue
        XCTAssertThrowsError(try ArchitecturePresetStore.validatePresetName(builtIn)) { error in
            XCTAssertEqual(error as? ArchitecturePresetStore.StoreError, .reservedName(builtIn))
        }
    }

    func testOrdinaryNamesAreAccepted() throws {
        for name in ["mine", "my preset", "v5_7x7-wide.2", "café", "custom"] {
            XCTAssertNoThrow(try ArchitecturePresetStore.validatePresetName(name), name)
        }
    }

    // MARK: - Where the file lands

    func testTraversalNameWritesNothingAnywhere() throws {
        let before = try entries(of: root)
        XCTAssertThrowsError(try save("../escape"))
        XCTAssertEqual(try entries(of: root), before, "nothing may be written beside the Presets folder")
        XCTAssertTrue(try entries(of: presets).isEmpty)
    }

    func testSavedPresetIsADirectChildOfThePresetsFolderAndLoadsBack() throws {
        let url = try save("my preset", label: "Mine")
        XCTAssertEqual(url.deletingLastPathComponent().standardizedFileURL.path, presets.standardizedFileURL.path)
        XCTAssertEqual(url.lastPathComponent, "my preset.json")
        let loaded = try ArchitecturePresetStore.loadFile(at: url)
        XCTAssertEqual(loaded, NamedArchitecture(label: "Mine", architecture: architecture))
    }

    // MARK: - Never overwrite without confirmation

    func testSavingOverAnExistingPresetThrowsAndKeepsIt() throws {
        let url = try save("mine", label: "Original")
        let original = try Data(contentsOf: url)
        XCTAssertThrowsError(try save("mine", label: "Second")) { error in
            XCTAssertEqual(error as? ArchitecturePresetStore.StoreError, .presetAlreadyExists("mine"))
        }
        XCTAssertEqual(try Data(contentsOf: url), original, "an unconfirmed save must leave the existing preset byte-for-byte")
    }

    func testConfirmedReplacementOverwritesTheExistingPreset() throws {
        let url = try save("mine", label: "Original")
        try save("mine", label: "Replacement", replacingExisting: true)
        XCTAssertEqual(try ArchitecturePresetStore.loadFile(at: url).label, "Replacement")
        XCTAssertEqual(try entries(of: presets), ["mine.json"], "the replacement must not leave a temporary file behind")
    }

    func testFolderInThePresetsPlaceIsNeverReplacedEvenWhenConfirmed() throws {
        let folder = presets.appendingPathComponent("mine.json", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: folder.appendingPathComponent("keep.txt"))
        for replacing in [false, true] {
            XCTAssertThrowsError(try save("mine", replacingExisting: replacing)) { error in
                guard case ArchitecturePresetStore.StoreError.presetPathNotARegularFile(let name, _) = error else {
                    XCTFail("expected presetPathNotARegularFile (replacing: \(replacing)), got \(error)")
                    return
                }
                XCTAssertEqual(name, "mine")
            }
        }
        XCTAssertEqual(try Data(contentsOf: folder.appendingPathComponent("keep.txt")), Data("keep".utf8))
    }

    func testSymbolicLinkInThePresetsPlaceIsNeverReplacedOrFollowed() throws {
        let target = root.appendingPathComponent("elsewhere.json")
        try Data("not a preset".utf8).write(to: target)
        let link = presets.appendingPathComponent("mine.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        for replacing in [false, true] {
            XCTAssertThrowsError(try save("mine", replacingExisting: replacing)) { error in
                guard case ArchitecturePresetStore.StoreError.presetPathNotARegularFile(_, _) = error else {
                    XCTFail("expected presetPathNotARegularFile (replacing: \(replacing)), got \(error)")
                    return
                }
            }
        }
        XCTAssertEqual(try Data(contentsOf: target), Data("not a preset".utf8))
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: link.path), target.path)
    }
}
