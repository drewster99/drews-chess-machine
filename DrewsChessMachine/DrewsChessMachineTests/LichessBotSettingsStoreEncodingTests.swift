import XCTest
@testable import DrewsChessMachine

/// The settings store overlays saved settings onto the JSON object of the
/// defaults. A value that does not encode as a JSON object is a defect in the
/// build, and it must surface as a thrown error the controller can show,
/// never a trap that takes down the whole app at launch.
final class LichessBotSettingsStoreEncodingTests: XCTestCase {
    func testANonObjectEncodingThrowsInsteadOfTrapping() {
        XCTAssertThrowsError(try LichessBotSettingsStore.jsonObject(encoding: 3))
        XCTAssertThrowsError(try LichessBotSettingsStore.jsonObject(encoding: ["a", "b"]))
    }

    func testTheDefaultSettingsEncodeAsAnObject() throws {
        let object = try LichessBotSettingsStore.jsonObject(encoding: LichessBotSettings())
        XCTAssertFalse(object.isEmpty)
    }
}
