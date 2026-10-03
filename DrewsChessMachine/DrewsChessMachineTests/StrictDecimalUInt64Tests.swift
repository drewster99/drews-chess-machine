//
//  StrictDecimalUInt64Tests.swift
//  DrewsChessMachineTests
//
//  `UInt64(strictDecimal:)` is the one parser for seed text: digits only,
//  the way `String(value)` writes a UInt64. Swift's `UInt64(_:)` also takes
//  a leading `+` and a negative zero, which no writer here produces, so
//  `--seed +5` used to run seed 5 while `--init-seed +5` was refused.
//

import XCTest
@testable import DrewsChessMachine

final class StrictDecimalUInt64Tests: XCTestCase {

    func testDigitsParseOverTheWholeRange() {
        XCTAssertEqual(UInt64(strictDecimal: "0"), 0)
        XCTAssertEqual(UInt64(strictDecimal: "5"), 5)
        XCTAssertEqual(UInt64(strictDecimal: "007"), 7)
        XCTAssertEqual(UInt64(strictDecimal: "9007199254740993"), 9_007_199_254_740_993)
        XCTAssertEqual(UInt64(strictDecimal: String(UInt64.max)), UInt64.max)
        XCTAssertEqual(UInt64(strictDecimal: "123456"[...]), 123_456, "a substring parses like a string")
    }

    func testAnythingButDigitsIsRefused() {
        let refused = ["", "+5", "+0", "-0", "-1", " 5", "5 ", "1.5", "1e3", "0x10", "five",
                       "18446744073709551616", "\u{0663}"]
        for text in refused {
            XCTAssertNil(UInt64(strictDecimal: text), "'\(text)'")
        }
    }
}
