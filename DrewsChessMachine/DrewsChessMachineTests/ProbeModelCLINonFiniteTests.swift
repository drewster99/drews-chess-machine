//
//  ProbeModelCLINonFiniteTests.swift
//  DrewsChessMachineTests
//
//  A checkpoint whose weights went NaN — exactly what the probe exists to
//  examine after a training blow-up — yields NaN probabilities, NLLs and
//  value outputs. `JSONSerialization` does not throw on a NaN or infinite
//  number: it raises an Objective-C exception, which Swift cannot catch, so
//  the whole probe process aborted. Encoding a probe line must instead throw
//  a Swift error naming the non-finite fields, so the run can report that
//  checkpoint as failed and carry on with the rest.
//
//  Kept in its own class: before the fix this test aborts the test process,
//  which would take every other test in a shared class down with it.
//

import XCTest
@testable import DrewsChessMachine

final class ProbeModelCLINonFiniteTests: XCTestCase {

    func testEncodingALineWithNonFiniteNumbersThrowsInsteadOfAborting() {
        let record: [String: Any] = [
            "model": "/tmp/blown-up.safetensors",
            "nll": Double.nan,
            "avgProb": Double.infinity,
            "n": 4435,
        ]
        XCTAssertThrowsError(try ProbeModelCLI.encodeLine(record)) { error in
            XCTAssertTrue(error.localizedDescription.contains("nll"), "\(error.localizedDescription)")
            XCTAssertTrue(error.localizedDescription.contains("avgProb"), "\(error.localizedDescription)")
        }
    }

    func testEncodingAFiniteLineSucceeds() throws {
        let record: [String: Any] = ["model": "/tmp/fine.safetensors", "nll": 2.25, "n": 4435, "themes": ["fork": "3/9"]]
        let line = try ProbeModelCLI.encodeLine(record)
        XCTAssertEqual(String(decoding: line, as: UTF8.self),
                       #"{"model":"\/tmp\/fine.safetensors","n":4435,"nll":2.25,"themes":{"fork":"3\/9"}}"#)
    }
}
