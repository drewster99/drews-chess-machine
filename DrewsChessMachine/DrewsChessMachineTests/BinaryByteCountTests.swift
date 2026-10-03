//
//  BinaryByteCountTests.swift
//  DrewsChessMachineTests
//
//  `BinaryByteCount.text` rounds before it picks the unit and the decimal
//  count: a value that rounds up to 1024 of a unit is shown as 1.0 of the
//  next one, and a value that rounds up to 100 is shown without a decimal,
//  the same as any other value from 100 up.
//

import XCTest
@testable import DrewsChessMachine

final class BinaryByteCountTests: XCTestCase {

    func testAValueThatRoundsToTheNextUnitIsShownInIt() {
        XCTAssertEqual(BinaryByteCount.text(1_048_575), "1.0 MB")
        XCTAssertEqual(BinaryByteCount.text(1_073_741_823), "1.0 GB")
        XCTAssertEqual(BinaryByteCount.text(1023 * 1024 + 1000), "1.0 MB")
    }

    func testAValueThatRoundsUpToOneHundredHasNoDecimal() {
        XCTAssertEqual(BinaryByteCount.text(102_349), "100 KB")
        XCTAssertEqual(BinaryByteCount.text(Int(99.94 * 1024)), "99.9 KB")
    }

    func testUnitsBelowTheEdges() {
        XCTAssertEqual(BinaryByteCount.text(0), "0 B")
        XCTAssertEqual(BinaryByteCount.text(1023), "1023 B")
        XCTAssertEqual(BinaryByteCount.text(1024), "1.0 KB")
        XCTAssertEqual(BinaryByteCount.text(1023 * 1024), "1023 KB")
        XCTAssertEqual(BinaryByteCount.text(2048 * 1024 * 1024 * 1024), "2048 GB")
    }
}
