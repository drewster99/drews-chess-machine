//
//  LichessProbeDataTests.swift
//  DrewsChessMachineTests
//
//  Smoke tests for the bundled 200-puzzle Lichess probe set. The data
//  is build-time content — these tests pin down that the JSON in the
//  app bundle parses cleanly, every entry decodes into a usable
//  `TacticalProbe`, theme buckets are balanced, and the UCI parser
//  round-trips a few hand-picked moves.
//

import XCTest
@testable import DrewsChessMachine

final class LichessProbeDataTests: XCTestCase {

    func testBundleHas200Puzzles() {
        XCTAssertEqual(LichessProbeData.largeSet.count, 200)
    }

    func testEveryProbeHasSingleAcceptableMove() {
        for probe in LichessProbeData.largeSet {
            XCTAssertEqual(
                probe.acceptable.count, 1,
                "probe '\(probe.name)' should have exactly one acceptable move"
            )
        }
    }

    func testEveryProbeBoardIs64Squares() {
        for probe in LichessProbeData.largeSet {
            XCTAssertEqual(
                probe.state.board.count, 64,
                "probe '\(probe.name)' board must be 64 cells"
            )
        }
    }

    func testThemeBucketsAreBalanced() {
        var counts: [ProbeCategory: Int] = [:]
        for probe in LichessProbeData.largeSet {
            counts[probe.category, default: 0] += 1
        }

        let expected: [ProbeCategory] = [
            .lichessMateIn1, .lichessHangingPiece, .lichessFork,
            .lichessPin, .lichessSkewer, .lichessOpening,
            .lichessMiddlegame, .lichessEndgame
        ]
        for c in expected {
            XCTAssertEqual(
                counts[c] ?? 0, 25,
                "expected 25 probes in category \(c.rawValue), got \(counts[c] ?? 0)"
            )
        }
    }

    func testEveryProbeAcceptedMoveIsOnTheBoard() {
        // Sanity: source/target squares are inside 0..7 row/col.
        for probe in LichessProbeData.largeSet {
            for move in probe.acceptable {
                XCTAssertTrue((0..<8).contains(move.fromRow))
                XCTAssertTrue((0..<8).contains(move.fromCol))
                XCTAssertTrue((0..<8).contains(move.toRow))
                XCTAssertTrue((0..<8).contains(move.toCol))
            }
        }
    }

    // MARK: - UCI parser

    func testParseUCIBasicMove() {
        // e2e4 = white pawn 2-square push. e2 = (row 6, col 4), e4 = (row 4, col 4).
        let m = LichessProbeData.parseUCI("e2e4")
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.fromRow, 6)
        XCTAssertEqual(m?.fromCol, 4)
        XCTAssertEqual(m?.toRow, 4)
        XCTAssertEqual(m?.toCol, 4)
        XCTAssertNil(m?.promotion)
    }

    func testParseUCIPromotionToQueen() {
        // a7a8q = white pawn promoting on a8. a7 = (row 1, col 0), a8 = (row 0, col 0).
        let m = LichessProbeData.parseUCI("a7a8q")
        XCTAssertEqual(m?.fromRow, 1)
        XCTAssertEqual(m?.fromCol, 0)
        XCTAssertEqual(m?.toRow, 0)
        XCTAssertEqual(m?.toCol, 0)
        XCTAssertEqual(m?.promotion, .queen)
    }

    func testParseUCIPromotionToKnight() {
        let m = LichessProbeData.parseUCI("h2h1n")
        XCTAssertEqual(m?.promotion, .knight)
    }

    func testParseUCIRejectsBadLength() {
        XCTAssertNil(LichessProbeData.parseUCI("e2"))
        XCTAssertNil(LichessProbeData.parseUCI("e2e4e5"))
    }

    func testParseUCIRejectsBadFile() {
        XCTAssertNil(LichessProbeData.parseUCI("i2i4"))
    }

    func testParseUCIRejectsBadRank() {
        XCTAssertNil(LichessProbeData.parseUCI("e0e4"))
        XCTAssertNil(LichessProbeData.parseUCI("e2e9"))
    }

    func testParseUCIRejectsBadPromotion() {
        XCTAssertNil(LichessProbeData.parseUCI("a7a8x"))
    }
}
