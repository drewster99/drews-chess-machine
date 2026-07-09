//
//  UCIEngineTests.swift
//  DrewsChessMachineTests
//
//  Guards that the UCI `go` path threads the engine's real ply history into
//  the board it encodes. UCIEngine is otherwise stdin/stdout-driven and so not
//  directly testable; `encodedBoardForGo` is the internal seam the handler
//  uses to build the position it evaluates.
//
//  The regression this pins: a history encoding (full10ply200) played from an
//  external GUI over UCI must see populated history planes, not the all-zero
//  "absent frame" signal it would get if the handler forgot to pass
//  `engine.recentStates` — a train/infer skew that silently weakens UCI play.
//

import XCTest
@testable import DrewsChessMachine

final class UCIEngineTests: XCTestCase {

    // e2-e4 and e7-e5 in absolute (row 0 = rank 8) coordinates.
    private let e4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
    private let e5 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)

    func testGoEncodingThreadsEngineHistoryForFull10Ply200() throws {
        // Play two plies so the engine accrues a non-empty recentStates window.
        let engine = ChessGameEngine()
        try engine.applyMoveAndAdvance(e4)
        try engine.applyMoveAndAdvance(e5)
        XCTAssertEqual(engine.recentStates.count, 2, "engine must retain prior plies as history")

        let encoded = UCIEngine.encodedBoardForGo(engine: engine, encoding: .full10ply200)
        XCTAssertEqual(encoded.count, BoardEncoder.tensorLength(for: .full10ply200))

        // History frames live in planes 20..199; with real history they must
        // carry occupancy mass. If the handler dropped the history argument,
        // they would be all-zero (the absent-frame signal) — the guarded bug.
        let historyMass = (20 * 64..<200 * 64).reduce(Float(0)) { $0 + encoded[$1] }
        XCTAssertGreaterThan(historyMass, 0,
                             "full10ply200 UCI encoding must populate history planes from engine.recentStates")

        // And the seam must be byte-identical to a direct state+history encode,
        // i.e. it really encodes `engine.state` with `engine.recentStates`.
        let direct = BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: .full10ply200)
        XCTAssertEqual(encoded, direct, "seam must encode state+history exactly as BoardEncoder does")
    }

    func testGoEncodingHasZeroHistoryAtGameStart() {
        // No prior plies → every history frame must be all-zero (the
        // absent-frame signal). Negative control for the test above.
        let engine = ChessGameEngine()
        XCTAssertEqual(engine.recentStates.count, 0)
        let encoded = UCIEngine.encodedBoardForGo(engine: engine, encoding: .full10ply200)
        for i in (20 * 64)..<(200 * 64) {
            XCTAssertEqual(encoded[i], 0.0, "absent history plane float \(i) must be zero at game start")
        }
    }

    // Regression: a UCI `setoption` value is free-form text that may contain
    // runs of spaces (e.g. a filesystem path). It must be the verbatim remainder
    // of the line after `value`, NOT a re-join of the space-split command tokens
    // (which collapses consecutive spaces and would corrupt such a path).
    func testSetOptionValuePreservesConsecutiveSpaces() {
        XCTAssertEqual(
            UCIEngine.setOptionValue(fromLine: "setoption name Model value /a/b  c/x.safetensors"),
            "/a/b  c/x.safetensors",
            "consecutive spaces inside a setoption value must be preserved")
        XCTAssertEqual(
            UCIEngine.setOptionValue(fromLine: "setoption name Model value /x/y   z"),
            "/x/y   z",
            "triple spaces must be preserved verbatim")
        // A value that itself contains the word "value": the first delimiter wins.
        XCTAssertEqual(
            UCIEngine.setOptionValue(fromLine: "setoption name Model value /my value/x"),
            "/my value/x")
        // The ordinary single-space path (common case) round-trips intact.
        XCTAssertEqual(
            UCIEngine.setOptionValue(fromLine: "setoption name Model value /Users/a/Application Support/m.safetensors"),
            "/Users/a/Application Support/m.safetensors")
        // No value text after `value` → empty string.
        XCTAssertEqual(UCIEngine.setOptionValue(fromLine: "setoption name Model value"), "")
    }

    // Bare model-name resolution against a Models/ directory listing. The key
    // case is that a run name (`Qeu8e`) resolves to its own '-replay-latest'
    // and does NOT get captured by a longer sibling run (`Qeu8e5`).
    func testResolveModelNameByRunNameDisambiguatesPrefixes() {
        let files = [
            "20260704-Qeu8e-replay-latest.safetensors",
            "20260704-Qeu8e5-replay-latest.safetensors",
            "20260704-Qeu8e5-replay-step14000.safetensors",
            "20260527-sMe9-26-manual.dcmmodel",
            "notes.txt",
        ]
        // run name → latest, with prefix disambiguation
        XCTAssertEqual(UCIModelLoader.resolveModelName("Qeu8e", among: files),
                       .found("20260704-Qeu8e-replay-latest.safetensors"))
        XCTAssertEqual(UCIModelLoader.resolveModelName("Qeu8e5", among: files),
                       .found("20260704-Qeu8e5-replay-latest.safetensors"))
        // exact filename
        XCTAssertEqual(UCIModelLoader.resolveModelName("20260704-Qeu8e5-replay-step14000.safetensors", among: files),
                       .found("20260704-Qeu8e5-replay-step14000.safetensors"))
        // filename without extension (.safetensors, then .dcmmodel)
        XCTAssertEqual(UCIModelLoader.resolveModelName("20260704-Qeu8e5-replay-step14000", among: files),
                       .found("20260704-Qeu8e5-replay-step14000.safetensors"))
        XCTAssertEqual(UCIModelLoader.resolveModelName("20260527-sMe9-26-manual", among: files),
                       .found("20260527-sMe9-26-manual.dcmmodel"))
        // nothing matches
        XCTAssertEqual(UCIModelLoader.resolveModelName("does-not-exist", among: files), .notFound)
        // two runs with the same trailing token → ambiguous
        XCTAssertEqual(
            UCIModelLoader.resolveModelName("Foo", among: ["a-Foo-replay-latest.safetensors",
                                                           "b-Foo-replay-latest.safetensors"]),
            .ambiguous(["a-Foo-replay-latest.safetensors", "b-Foo-replay-latest.safetensors"]))
    }
}
