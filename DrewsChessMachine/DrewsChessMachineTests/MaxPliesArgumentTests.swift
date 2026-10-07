//
//  MaxPliesArgumentTests.swift
//  DrewsChessMachineTests
//
//  `--train-vs-uci --max-plies <n>` is the cap every game against the
//  engines is played to, and what the run's session.json records. A cap
//  below one ply plays no game; the driver used to raise it to 1 unseen
//  while session.json recorded the flag's value, so the value is refused
//  where it is parsed.
//

import XCTest
@testable import DrewsChessMachine

final class MaxPliesArgumentTests: XCTestCase {

    // MARK: - Regression

    func testACapBelowOnePlyIsRefused() {
        for text in ["0", "-1", "-400"] {
            XCTAssertThrowsError(try DrewsChessMachineApp.parseMaxPliesPerGame(text), "--max-plies \(text)") { error in
                XCTAssertTrue(error is CLIRunRefusal, "\(error)")
            }
        }
    }

    // MARK: - Accepted values

    func testAPositiveCapIsTheValueGiven() throws {
        XCTAssertEqual(try DrewsChessMachineApp.parseMaxPliesPerGame("1"), 1)
        XCTAssertEqual(try DrewsChessMachineApp.parseMaxPliesPerGame("400"), 400)
    }

    func testANonIntegerIsRefused() {
        for text in ["", "ten", "4.5", "1e3"] {
            XCTAssertThrowsError(try DrewsChessMachineApp.parseMaxPliesPerGame(text), "--max-plies '\(text)'")
        }
    }
}
