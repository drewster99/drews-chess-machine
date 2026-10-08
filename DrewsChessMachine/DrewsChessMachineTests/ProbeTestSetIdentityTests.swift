import XCTest
@testable import DrewsChessMachine

/// The bundled puzzle sets' identity (test-set results plan D5): id, title
/// and description come from each set file's `metadata` block, the
/// fingerprint pins the puzzle list, and `ProbeCategory` is the one mapping
/// between Lichess theme ids, buckets and display titles.
final class ProbeTestSetIdentityTests: XCTestCase {

    func testTheBundledSetsCarryTheirIdentity() {
        let set200 = LichessProbeData.set200
        XCTAssertEqual(set200.id, "lichess-200")
        XCTAssertEqual(set200.title, "Lichess puzzles, 200")
        XCTAssertFalse(set200.description.isEmpty)
        XCTAssertEqual(set200.probes.count, 200)

        let wide = LichessProbeData.wide
        XCTAssertEqual(wide.id, "lichess-wide")
        XCTAssertEqual(wide.title, "Lichess puzzles, wide")
        XCTAssertFalse(wide.description.isEmpty)
        XCTAssertEqual(wide.probes.count, 4435)

        XCTAssertEqual(LichessProbeData.modelFileTestSets.map(\.id), ["lichess-200", "lichess-wide"])
    }

    /// The expected values were computed from the set files by a separate
    /// Python reading (`hashlib.sha256` over the same per-puzzle lines), so
    /// a change to the line format or to the bundled puzzles fails here.
    func testTheFingerprintsMatchAnIndependentReading() {
        XCTAssertEqual(LichessProbeData.set200.fingerprintSHA256, "1c7cdab805263f4cad16a683a61b69ee3df00897b80fe4c5ce142690b125a9db")
        XCTAssertEqual(LichessProbeData.wide.fingerprintSHA256, "695db74acfdcab2295640ad5d57363df4d5ecc293b6b7aa2d19a4f8528a4fd4c")
    }

    func testTheFingerprintChangesWhenOnePuzzleChanges() {
        let line = LichessProbeData.fingerprintLine(id: "00008", theme: "mateIn1", rating: 1200, fen: "8/8/8/8/8/8/8/K6k w - - 0 1", bestMoveUci: "a1a2")
        let changedMove = LichessProbeData.fingerprintLine(id: "00008", theme: "mateIn1", rating: 1200, fen: "8/8/8/8/8/8/8/K6k w - - 0 1", bestMoveUci: "a1b1")
        let changedRating = LichessProbeData.fingerprintLine(id: "00008", theme: "mateIn1", rating: 1201, fen: "8/8/8/8/8/8/8/K6k w - - 0 1", bestMoveUci: "a1a2")
        let base = LichessProbeData.sha256Hex(line)
        XCTAssertEqual(base, LichessProbeData.sha256Hex(line), "stable")
        XCTAssertNotEqual(base, LichessProbeData.sha256Hex(changedMove))
        XCTAssertNotEqual(base, LichessProbeData.sha256Hex(changedRating))
        XCTAssertEqual(base.count, 64)
    }

    func testEveryLichessThemeRoundTripsAndHasATitle() throws {
        let lichess = ProbeCategory.allCases.filter { $0.lichessThemeID != nil }
        XCTAssertEqual(lichess.count, 13)
        for category in lichess {
            let themeID = try XCTUnwrap(category.lichessThemeID)
            XCTAssertEqual(ProbeCategory(lichessThemeID: themeID), category)
            XCTAssertFalse(category.title.isEmpty)
            XCTAssertNotEqual(category.title, category.rawValue, "a person-facing title, not the case name")
        }
        XCTAssertEqual(Set(lichess.map(\.title)).count, 13, "titles are distinct")
        XCTAssertNil(ProbeCategory(lichessThemeID: "notATheme"))
        XCTAssertEqual(ProbeCategory.lichessHangingPiece.title, "Hanging piece")
        XCTAssertEqual(ProbeCategory.lichessDiscoveredAttack.title, "Discovered attack")
    }

    func testEveryBundledPuzzleIsInALichessBucket() {
        for probe in LichessProbeData.set200.probes + LichessProbeData.wide.probes {
            XCTAssertNotNil(probe.category.lichessThemeID, probe.name)
        }
    }
}
