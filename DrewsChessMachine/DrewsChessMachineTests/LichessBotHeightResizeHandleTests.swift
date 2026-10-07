import XCTest
@testable import DrewsChessMachine

/// The Record card's resize handle drags from the edge the operator sees,
/// not the stored height, so no part of a drag moves nothing.
final class LichessBotHeightResizeHandleTests: XCTestCase {
    private let range: ClosedRange<Double> = 160...1600

    func testADragStartsFromTheDisplayedEdge() {
        // The content's minimum holds the card at 400 although 200 is stored.
        XCTAssertEqual(LichessBotHeightResizeHandle.shownHeight(stored: 200, displayed: 400, range: range), 400)
        XCTAssertEqual(LichessBotHeightResizeHandle.shownHeight(stored: 560, displayed: 560, range: range), 560)
        XCTAssertEqual(LichessBotHeightResizeHandle.shownHeight(stored: 2000, displayed: nil, range: range), 1600)
    }

    func testADragEndingAboveTheContentsMinimumStoresTheDisplayedHeight() {
        XCTAssertEqual(LichessBotHeightResizeHandle.heightAfterDrag(stored: 300, displayed: 400, range: range), 400)
        XCTAssertEqual(LichessBotHeightResizeHandle.heightAfterDrag(stored: 700, displayed: 700, range: range), 700)
        XCTAssertEqual(LichessBotHeightResizeHandle.heightAfterDrag(stored: 1600, displayed: 1700, range: range), 1600)
        XCTAssertEqual(LichessBotHeightResizeHandle.heightAfterDrag(stored: 300, displayed: nil, range: range), 300)
    }
}
