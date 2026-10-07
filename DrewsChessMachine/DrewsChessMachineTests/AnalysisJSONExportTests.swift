import XCTest
@testable import DrewsChessMachine

/// Two exports of one label must never replace each other, whether or not
/// they land in the same second, and the name keeps its family/time/label shape.
final class AnalysisJSONExportTests: XCTestCase {
    private struct TestReport: AnalysisReport, Codable, Equatable {
        static var exportFamily: AnalysisJSONExport.Family { .numericsAudit }
        let value: Int
        func textSummary() -> String { "value=\(value)" }
    }
    private var root: URL!
    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("AnalysisJSONExportTests-\(UUID().uuidString)", isDirectory: true)
    }
    override func tearDownWithError() throws {
        if FileManager.default.fileExists(atPath: root.path) { try FileManager.default.removeItem(at: root) }
    }

    func testRepeatedExportsOfOneLabelNeverReplaceEachOther() throws {
        let first = try AnalysisJSONExport.publish(TestReport(value: 1), modelLabel: "trainer:20261007-1-AbCd", directory: root)
        let second = try AnalysisJSONExport.publish(TestReport(value: 2), modelLabel: "trainer:20261007-1-AbCd", directory: root)
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(try JSONDecoder().decode(TestReport.self, from: Data(contentsOf: first)), TestReport(value: 1))
        XCTAssertEqual(try JSONDecoder().decode(TestReport.self, from: Data(contentsOf: second)), TestReport(value: 2))
    }

    func testSummarizeAndPublishOffPoolReturnsTheSummaryAndTheFile() async throws {
        let outcome = await AnalysisJSONExport.summarizeAndPublishOffPool(TestReport(value: 7), modelLabel: "x", directory: root)
        XCTAssertEqual(outcome.summary, "value=7")
        let url = try outcome.written.get()
        XCTAssertEqual(try JSONDecoder().decode(TestReport.self, from: Data(contentsOf: url)), TestReport(value: 7))
    }

    func testFileStemNamesFamilyLocalTimeAndAPlainLabel() throws {
        var components = DateComponents()
        (components.year, components.month, components.day) = (2026, 10, 7)
        (components.hour, components.minute, components.second) = (12, 34, 56)
        let date = try XCTUnwrap(Calendar.current.date(from: components))
        XCTAssertEqual(AnalysisJSONExport.fileStem(family: .numericsAudit, modelLabel: "file:a b/c", at: date),
                       "numerics_audit_20261007-123456_file_a_b_c")
    }
}
