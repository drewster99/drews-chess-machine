//
//  ProbeSetFileParsingTests.swift
//  DrewsChessMachineTests
//
//  `--probe-model --probe-set-file <set.json>` reads a puzzle set file
//  through the one parser the bundled sets use
//  (`LichessProbeData.parseProbeSet`). A bundled set read as a file must come
//  out identical (same identity, fingerprint, probes and metadata), and a
//  malformed file must be a thrown error naming what is wrong — never the
//  bundled loader's precondition failure.
//

import XCTest
@testable import DrewsChessMachine

final class ProbeSetFileParsingTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProbeSetFileParsingTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    // MARK: - The bundled set, read as a file

    /// Protects: a set file and the bundled set it copies are the same set —
    /// same id, title, fingerprint, probes (names, categories, positions,
    /// moves) and metadata — so `--probe-set-file` results are comparable
    /// with `--probe-set wide` results.
    /// Fails if: the file path and the bundled path parse differently (a
    /// second parser, a different fingerprint line or probe name).
    func testTheBundledWideFileParsesToTheBundledWideSet() throws {
        let url = try XCTUnwrap(Bundle.main.url(forResource: "lichess_probes_wide", withExtension: "json"))
        let parsed = try LichessProbeData.loadProbeSetFile(at: url)
        let bundled = LichessProbeData.wide

        XCTAssertEqual(parsed.testSet.id, bundled.id)
        XCTAssertEqual(parsed.testSet.title, bundled.title)
        XCTAssertEqual(parsed.testSet.description, bundled.description)
        XCTAssertEqual(parsed.testSet.fingerprintSHA256, bundled.fingerprintSHA256)
        XCTAssertEqual(parsed.testSet.probes.count, bundled.probes.count)
        for (fromFile, fromBundle) in zip(parsed.testSet.probes, bundled.probes) {
            XCTAssertEqual(fromFile.name, fromBundle.name)
            XCTAssertEqual(fromFile.category, fromBundle.category)
            XCTAssertEqual(fromFile.acceptable, fromBundle.acceptable)
            XCTAssertEqual(FENParser.fen(from: fromFile.state), FENParser.fen(from: fromBundle.state))
        }
        XCTAssertEqual(parsed.metadata.count, bundled.probes.count)
        for probe in bundled.probes {
            let fromFile = try XCTUnwrap(parsed.metadata[probe.name])
            let fromBundle = try XCTUnwrap(LichessProbeData.metadata[probe.name])
            XCTAssertEqual(fromFile.id, fromBundle.id)
            XCTAssertEqual(fromFile.rating, fromBundle.rating)
            XCTAssertEqual(fromFile.theme, fromBundle.theme)
            XCTAssertEqual(fromFile.fen, fromBundle.fen)
            XCTAssertEqual(fromFile.bestMoveUci, fromBundle.bestMoveUci)
        }
    }

    /// Protects: a set file's battery is labelled with the file's metadata
    /// id and records the file it came from, and carries the file's own
    /// metadata (its ratings feed the puzzle Elo).
    /// Fails if: the label falls back to a bundled name, or the battery reads
    /// the bundled metadata instead of the file's.
    func testASetFileBatteryIsNamedByItsFile() throws {
        let url = try writeSetFile(id: "custom-set", puzzles: [Self.validPuzzle])
        let batteries = try ProbeModelCLI.batteries(for: .setFile(path: url.path))

        XCTAssertEqual(batteries.count, 1)
        let battery = try XCTUnwrap(batteries.first)
        XCTAssertEqual(battery.label, "custom-set")
        XCTAssertEqual(battery.testSet.id, "custom-set")
        XCTAssertEqual(battery.setFilePath, url.path)
        let probe = try XCTUnwrap(battery.testSet.probes.first)
        XCTAssertEqual(battery.puzzleMetadata[probe.name]?.rating, 1234)
        XCTAssertNil(LichessProbeData.metadata[probe.name], "a puzzle the bundled sets don't hold")
    }

    /// Protects: the bundled batteries keep the labels earlier outputs used
    /// (`200`, `wide`) and their order, and have no set file.
    /// Fails if: `--probe-set` output changes its `set` field or order.
    func testTheBundledBatteriesKeepTheirLabels() throws {
        XCTAssertEqual(try ProbeModelCLI.batteries(for: .bundled(.both)).map(\.label), ["200", "wide"])
        XCTAssertEqual(try ProbeModelCLI.batteries(for: .bundled(.set200)).map(\.label), ["200"])
        XCTAssertEqual(try ProbeModelCLI.batteries(for: .bundled(.wide)).map(\.label), ["wide"])
        XCTAssertEqual(try ProbeModelCLI.batteries(for: .bundled(.both)).map(\.testSet.fingerprintSHA256),
                       LichessProbeData.modelFileTestSets.map(\.fingerprintSHA256))
        XCTAssertTrue(try ProbeModelCLI.batteries(for: .bundled(.both)).allSatisfy { $0.setFilePath == nil })
    }

    // MARK: - Malformed files

    /// Protects: a missing file is a thrown `unreadable` error.
    /// Fails if: the read failure crashes or is swallowed.
    func testAMissingFileIsUnreadable() {
        let url = root.appendingPathComponent("absent.json")
        XCTAssertThrowsError(try LichessProbeData.loadProbeSetFile(at: url)) { error in
            guard case LichessProbeData.ProbeSetFileError.unreadable(let path, _) = error else {
                return XCTFail("expected unreadable, got \(error)")
            }
            XCTAssertEqual(path, url.path)
        }
    }

    /// Protects: bytes that are not JSON, and JSON without the set shape
    /// (no `metadata`, no `puzzles`, a puzzle missing a field or with a
    /// rating that is not a number), are `notASetFile`.
    /// Fails if: any of them crashes or parses.
    func testNonSetJSONIsNotASetFile() throws {
        let cases: [String] = [
            "not json",
            #"{"puzzles": []}"#,
            #"{"metadata": {"id": "x", "title": "t", "description": "d"}}"#,
            #"{"metadata": {"id": "x", "title": "t", "description": "d"}, "puzzles": [{"id": "a", "theme": "mateIn1", "rating": 1200, "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}]}"#,
            #"{"metadata": {"id": "x", "title": "t", "description": "d"}, "puzzles": [{"id": "a", "theme": "mateIn1", "rating": "high", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "best_move_uci": "a1a2"}]}"#,
        ]
        for text in cases {
            XCTAssertThrowsError(try LichessProbeData.parseProbeSet(jsonData: Data(text.utf8)), text) { error in
                guard case LichessProbeData.ProbeSetFileError.notASetFile = error else {
                    return XCTFail("expected notASetFile for \(text), got \(error)")
                }
            }
        }
    }

    /// Protects: a set with no puzzles is refused (nothing to probe; a
    /// summary of zero positions would read as a result).
    /// Fails if: an empty set parses.
    func testAnEmptySetIsRefused() throws {
        assertParseFails(puzzles: [], expected: .noPuzzles)
    }

    /// Protects: a repeated puzzle id is refused — the probe name embeds the
    /// id and the metadata is keyed by name, so a repeat would silently
    /// replace one puzzle's metadata.
    /// Fails if: duplicates parse.
    func testARepeatedPuzzleIDIsRefused() throws {
        assertParseFails(puzzles: [Self.validPuzzle, Self.validPuzzle], expected: .duplicatePuzzleID("zz001"))
    }

    /// Protects: a bad FEN, a bad UCI move and a theme outside the probe
    /// buckets each throw an error naming the puzzle.
    /// Fails if: any of them crashes (the bundled loader's precondition) or
    /// is accepted.
    func testBadPuzzleFieldsNameThePuzzle() throws {
        var badFEN = Self.validPuzzle
        badFEN["fen"] = "not a fen"
        XCTAssertThrowsError(try parse(puzzles: [badFEN])) { error in
            guard case LichessProbeData.ProbeSetFileError.badFEN(let id, let fen, _) = error else {
                return XCTFail("expected badFEN, got \(error)")
            }
            XCTAssertEqual(id, "zz001")
            XCTAssertEqual(fen, "not a fen")
        }

        var badMove = Self.validPuzzle
        badMove["best_move_uci"] = "z9z9"
        assertParseFails(puzzles: [badMove], expected: .badMove(puzzleID: "zz001", uci: "z9z9"))

        var badTheme = Self.validPuzzle
        badTheme["theme"] = "crushing"
        assertParseFails(puzzles: [badTheme], expected: .unknownTheme(puzzleID: "zz001", theme: "crushing"))
    }

    /// Protects: every malformed-file error describes itself (the CLI prints
    /// `localizedDescription`).
    /// Fails if: an error case has no description.
    func testEveryErrorHasADescription() {
        let errors: [LichessProbeData.ProbeSetFileError] = [
            .unreadable(path: "/x", reason: "r"), .notASetFile(reason: "r"), .noPuzzles,
            .duplicatePuzzleID("a"), .badFEN(puzzleID: "a", fen: "f", reason: "r"),
            .badMove(puzzleID: "a", uci: "u"), .unknownTheme(puzzleID: "a", theme: "t"),
        ]
        for error in errors {
            XCTAssertFalse((error.errorDescription ?? "").isEmpty, "\(error)")
        }
    }

    // MARK: - Helpers

    /// A legal position (white king a1, black king h1) with a
    /// legal king move, in a Lichess probe theme.
    private static let validPuzzle: [String: Any] = [
        "id": "zz001",
        "theme": "endgame",
        "rating": 1234,
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "best_move_uci": "a1a2",
        "popularity": 90,
    ]

    private func setFileData(id: String, puzzles: [[String: Any]]) throws -> Data {
        let object: [String: Any] = [
            "metadata": ["id": id, "title": "A test set", "description": "Written by the test.", "preset": "test"],
            "puzzles": puzzles,
        ]
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func writeSetFile(id: String, puzzles: [[String: Any]]) throws -> URL {
        let url = root.appendingPathComponent("\(id).json")
        try setFileData(id: id, puzzles: puzzles).write(to: url, options: [.withoutOverwriting])
        return url
    }

    private func parse(puzzles: [[String: Any]]) throws -> LichessProbeData.ParsedProbeSet {
        try LichessProbeData.parseProbeSet(jsonData: try setFileData(id: "t", puzzles: puzzles))
    }

    private func assertParseFails(
        puzzles: [[String: Any]],
        expected: LichessProbeData.ProbeSetFileError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try parse(puzzles: puzzles), file: file, line: line) { error in
            XCTAssertEqual(error as? LichessProbeData.ProbeSetFileError, expected, file: file, line: line)
        }
    }
}
