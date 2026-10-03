//
//  FileSafetySameFileTests.swift
//  DrewsChessMachineTests
//
//  `FileSafety.mayNameTheSameFile` is the one check every writer uses before
//  writing two outputs, or an output beside an input, that must not be one
//  file: `--create-parameters-file` (JSON and markdown), the replay and
//  train-vs-UCI `--out-model` against `--start-model`, and `--probe-model`'s
//  two outputs against each other and against every probed checkpoint. It
//  must say "may be the same file" for every way two paths can reach one
//  file — equal ignoring case, through `..`, a hard link, a symbolic link to
//  the file, a symbolic link in a parent folder (`/tmp` is one on macOS),
//  even when the file does not exist yet — and "different" only for paths
//  that cannot be one file.
//

import XCTest
@testable import DrewsChessMachine

final class FileSafetySameFileTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileSafetySameFileTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func makeFile(_ name: String) throws -> URL {
        let url = root.appendingPathComponent(name)
        try Data(name.utf8).write(to: url)
        return url
    }

    func testCaseVariantsMayBeTheSameFile() throws {
        let lower = root.appendingPathComponent("notes.md")
        let upper = root.appendingPathComponent("NOTES.MD")
        XCTAssertTrue(try FileSafety.mayNameTheSameFile(lower, upper))
    }

    func testDotDotPathIsTheSameFile() throws {
        let file = try makeFile("a.json")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: false)
        let viaDotDot = URL(fileURLWithPath: root.path + "/sub/../a.json")
        XCTAssertTrue(try FileSafety.mayNameTheSameFile(file, viaDotDot))
    }

    func testHardLinksAreTheSameFile() throws {
        let file = try makeFile("a.json")
        let link = root.appendingPathComponent("b.json")
        try FileManager.default.linkItem(at: file, to: link)
        XCTAssertTrue(try FileSafety.mayNameTheSameFile(file, link))
    }

    func testSymbolicLinkToTheFileIsTheSameFile() throws {
        let file = try makeFile("a.json")
        let link = root.appendingPathComponent("b.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        XCTAssertTrue(try FileSafety.mayNameTheSameFile(file, link))
    }

    func testExistingFileThroughASymlinkedFolderIsTheSameFile() throws {
        let real = root.appendingPathComponent("real", isDirectory: true)
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: false)
        let link = root.appendingPathComponent("link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        let file = real.appendingPathComponent("a.json")
        try Data("a".utf8).write(to: file)
        XCTAssertTrue(try FileSafety.mayNameTheSameFile(file, URL(fileURLWithPath: link.path + "/a.json")))
    }

    func testNotYetExistingFileThroughASymlinkedFolderIsTheSameFile() throws {
        let real = root.appendingPathComponent("real", isDirectory: true)
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: false)
        let link = root.appendingPathComponent("link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        XCTAssertTrue(try FileSafety.mayNameTheSameFile(real.appendingPathComponent("new.json"),
                                                        URL(fileURLWithPath: link.path + "/new.json")))
    }

    func testDistinctExistingFilesAreDifferent() throws {
        let first = try makeFile("a.json")
        let second = try makeFile("b.json")
        XCTAssertFalse(try FileSafety.mayNameTheSameFile(first, second))
    }

    func testDistinctNotYetExistingPathsAreDifferent() throws {
        XCTAssertFalse(try FileSafety.mayNameTheSameFile(root.appendingPathComponent("a.json"),
                                                         root.appendingPathComponent("b.json")))
    }
}
