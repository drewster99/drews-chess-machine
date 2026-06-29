import XCTest
@testable import DrewsChessMachine

/// Ordering and retention rules for the Load Session picker's
/// lineage grouping (SESSION_PICKER_PLAN.md): groups newest-run
/// first, sessions within a group newest first, unreadable rows
/// retained instead of dropped.
@MainActor
final class SessionPickerModelTests: XCTestCase {

    /// Manifest with identity derived purely from the folder name
    /// (UTC timestamp prefix orders the rows), optionally overridden
    /// by an exact `savedAtUnix`.
    private func manifest(_ folderName: String, savedAtUnix: Double? = nil) -> SessionManifest {
        var dict: [String: Any] = [:]
        if let savedAtUnix { dict["savedAtUnix"] = savedAtUnix }
        return SessionManifest.extract(
            jsonDict: dict, folderName: folderName,
            disk: nil, srcBytes: nil, srcMTime: nil
        )
    }

    func testGroups_newestRunFirst_andSessionsNewestFirstWithinRun() {
        let rows = [
            manifest("20260601-100000-20260601-1-AAAA-manual.dcmsession"),
            manifest("20260603-100000-20260601-1-AAAA-periodic.dcmsession"),
            manifest("20260610-100000-20260609-2-BBBB-manual.dcmsession"),
            manifest("20260609-100000-20260609-2-BBBB-promote.dcmsession"),
        ]
        let groups = SessionPickerModel.makeGroups(from: rows)
        XCTAssertEqual(groups.map(\.lineageTag), ["BBBB", "AAAA"],
                       "run with the newest save sorts first")
        XCTAssertEqual(groups[0].sessions.map(\.trigger), ["manual", "promote"],
                       "within a run, newest save first")
        XCTAssertEqual(groups[1].sessions.map(\.trigger), ["periodic", "manual"])
        XCTAssertEqual(groups[0].newestDate, groups[0].sessions[0].savedAt)
    }

    func testGroups_groupArchitectureComesFromNewestSession() {
        let older = manifest("20260601-100000-20260601-1-AAAA-manual.dcmsession")
        let newerDict: [String: Any] = [
            "architecture": [
                "architectureVersion": 4, "channels": 128,
                "numBlocks": 5, "inputPlanes": 30
            ]
        ]
        let newer = SessionManifest.extract(
            jsonDict: newerDict,
            folderName: "20260605-100000-20260601-1-AAAA-manual.dcmsession",
            disk: nil, srcBytes: nil, srcMTime: nil
        )
        XCTAssertNil(older.architectureSummary)
        XCTAssertNotNil(newer.architectureSummary)
        let groups = SessionPickerModel.makeGroups(from: [older, newer])
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].architectureSummary, newer.architectureSummary)
    }

    func testGroups_unreadableRowsAreRetainedInTheirLineageGroup() {
        let bogus = URL(fileURLWithPath:
            "/nonexistent/20260608-100000-20260601-1-AAAA-manual.dcmsession")
        let unreadable = SessionManifest.extract(fromSessionFolder: bogus)
        XCTAssertNotNil(unreadable.loadError)
        let healthy = manifest("20260601-100000-20260601-1-AAAA-periodic.dcmsession")
        let groups = SessionPickerModel.makeGroups(from: [healthy, unreadable])
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].sessions.count, 2,
                       "unreadable row is listed, not dropped")
        XCTAssertEqual(groups[0].sessions[0].folderName, unreadable.folderName,
                       "unreadable row still sorts by its folder-name date")
    }

    func testGroups_nonconformingNamesGroupUnderQuestionMark() {
        let foreign = manifest("strange-folder.dcmsession")
        XCTAssertNil(foreign.lineageTag)
        let groups = SessionPickerModel.makeGroups(from: [foreign])
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].lineageTag, "?")
        XCTAssertEqual(groups[0].sessions.count, 1)
    }

    func testFolderURL_derivesFromScannedDirectoryOnly() {
        let model = SessionPickerModel()
        let m = manifest("20260601-100000-20260601-1-AAAA-manual.dcmsession")
        XCTAssertNil(model.folderURL(for: m), "no scan yet → no URL to fabricate")
    }
}
