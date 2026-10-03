import XCTest
import Darwin
@testable import DrewsChessMachine

/// `--output` / `--overwrite-output`: a run's results.json destination is
/// checked before the run, an existing file is replaced only when the flag
/// authorizes it (and only that file), and results are never lost to — or
/// written over — a file that appeared during the run.
final class CliResultsOutputTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CliResultsOutputTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root, FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.removeItem(at: root)
        }
    }

    private func recorder(session: String) -> CliTrainingRecorder {
        let r = CliTrainingRecorder()
        r.setSessionID(session)
        return r
    }

    private func sessionID(in url: URL) throws -> String? {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        return json?["session_id"] as? String
    }

    // MARK: Pre-flight

    func testAFreePathIsAcceptedAsNew() throws {
        let output = try CliResultsOutput.preflight(url: root.appendingPathComponent("r.json"), overwriteAuthorized: false)
        XCTAssertNil(output.replacing)
    }

    func testAnExistingFileIsRefusedWithoutTheFlag() throws {
        let url = root.appendingPathComponent("r.json")
        try Data("earlier run".utf8).write(to: url)
        XCTAssertThrowsError(try CliResultsOutput.preflight(url: url, overwriteAuthorized: false)) { error in
            XCTAssertEqual(error as? CliResultsOutputError, .alreadyExists(path: url.path))
        }
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "earlier run")
    }

    func testAnExistingFileIsAcceptedForReplacementWithTheFlag() throws {
        let url = root.appendingPathComponent("r.json")
        try Data("earlier run".utf8).write(to: url)
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: true)
        XCTAssertEqual(output.replacing, try FileSafety.existingItem(at: url)?.identity)
    }

    func testAFolderAtThePathIsRefusedEvenWithTheFlag() throws {
        let url = root.appendingPathComponent("r.json", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        XCTAssertThrowsError(try CliResultsOutput.preflight(url: url, overwriteAuthorized: true)) { error in
            guard case .notARegularFile? = error as? FileSafetyError else {
                return XCTFail("expected notARegularFile, got \(error)")
            }
        }
    }

    func testAMissingFolderIsRefused() {
        let url = root.appendingPathComponent("missing/r.json")
        XCTAssertThrowsError(try CliResultsOutput.preflight(url: url, overwriteAuthorized: false)) { error in
            guard case .folderUnusable? = error as? CliResultsOutputError else {
                return XCTFail("expected folderUnusable, got \(error)")
            }
        }
    }

    // MARK: Writing

    func testWritingToAFreePathCreatesTheFile() throws {
        let url = root.appendingPathComponent("r.json")
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: false)
        let written = try recorder(session: "S1").write(to: output, totalTrainingSeconds: 1)
        XCTAssertEqual(written, url)
        XCTAssertEqual(try sessionID(in: url), "S1")
    }

    func testAnAuthorizedFileIsReplaced() throws {
        let url = root.appendingPathComponent("r.json")
        try Data("earlier run".utf8).write(to: url)
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: true)
        let written = try recorder(session: "S2").write(to: output, totalTrainingSeconds: 1)
        XCTAssertEqual(written, url)
        XCTAssertEqual(try sessionID(in: url), "S2")
    }

    func testAFileThatAppearedDuringTheRunIsKeptAndResultsGoBesideIt() throws {
        let url = root.appendingPathComponent("r.json")
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: false)
        try Data("appeared meanwhile".utf8).write(to: url)
        let written = try recorder(session: "S3").write(to: output, totalTrainingSeconds: 1)
        XCTAssertEqual(written, root.appendingPathComponent("r-2.json"))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "appeared meanwhile")
        XCTAssertEqual(try sessionID(in: written), "S3")
    }

    func testAnAuthorizedFileReplacedDuringTheRunIsKeptAndResultsGoBesideIt() throws {
        let url = root.appendingPathComponent("r.json")
        try Data("earlier run".utf8).write(to: url)
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: true)
        // Another process swaps a different file in: written beside, renamed over.
        let incoming = root.appendingPathComponent("incoming")
        try Data("someone else's".utf8).write(to: incoming)
        XCTAssertEqual(Darwin.rename(incoming.path, url.path), 0, String(cString: strerror(errno)))
        let written = try recorder(session: "S4").write(to: output, totalTrainingSeconds: 1)
        XCTAssertEqual(written, root.appendingPathComponent("r-2.json"))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "someone else's")
        XCTAssertEqual(try sessionID(in: written), "S4")
    }
}
