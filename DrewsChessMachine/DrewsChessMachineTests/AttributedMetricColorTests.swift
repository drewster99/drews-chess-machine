//
//  AttributedMetricColorTests.swift
//  DrewsChessMachineTests
//
//  Tests for the multi-line body colorizer used on the Training /
//  Self Play text panels. The colorizer's correctness invariants:
//   - Non-matching lines pass through with no foreground-color
//     attribute.
//   - Matching lines split into (prefix, colored value, suffix)
//     with the numeric token carrying the expected color; the
//     rest of the line stays default.
//   - Line breaks and indentation are preserved exactly so
//     monospaced column alignment still lines up.
//   - Threshold bands from CLAUDE.md / the in-repo constants are
//     honored (entropy collapse below policyEntropyAlarmThreshold,
//     gNorm clip event at or above gradClipMaxNorm, etc.).
//
//  A bug here would silently stop warning the user when training
//  is collapsing — worth pinning.
//

import XCTest
import SwiftUI
@testable import DrewsChessMachine

final class AttributedMetricColorTests: XCTestCase {

    // Threshold set matching the current in-repo defaults — tests
    // calibrated against these specific bands so changes to the
    // defaults force a test update (which is the right behavior;
    // band changes are load-bearing for the UI).
    private var defaults: AttributedMetricColor.Thresholds {
        AttributedMetricColor.Thresholds.default(
            entropyCollapseBelow: 5.0,
            gradClipMaxNorm: 5.0
        )
    }

    // MARK: - Line-level helpers

    /// Extract the foreground color applied to the substring
    /// matching `token` in `attr`. Returns nil if no foreground
    /// attribute is set on that range.
    private func color(of token: String, in attr: AttributedString) -> Color? {
        guard let range = attr.range(of: token) else { return nil }
        return attr[range].foregroundColor
    }

    /// Convenience: colorize a single line with default thresholds.
    private func colorize(_ line: String) -> AttributedString {
        AttributedMetricColor.colorize(body: line, thresholds: defaults)
    }

    // MARK: - Pass-through

    func testUnrecognizedLineIsUnchanged() {
        let attr = colorize("  Some unrelated line 42.0")
        // No colored token — the string content should equal the
        // plain line and nothing has a non-nil foregroundColor.
        XCTAssertEqual(String(attr.characters), "  Some unrelated line 42.0")
    }

    func testLinePrefixPreserved() {
        // Leading spaces drive monospaced alignment — must survive.
        let attr = colorize("    v abs:         0.9500")
        XCTAssertTrue(String(attr.characters).hasPrefix("    v abs:"))
    }

    // MARK: - Entropy bands

    func testEntropyBelowCollapseThresholdIsRed() {
        let attr = colorize("  Entropy:     3.500000")
        XCTAssertEqual(color(of: "3.500000", in: attr), .red)
    }

    func testEntropyAtThresholdIsNotRed() {
        // Strictly below is red; exactly at the threshold is not.
        let attr = colorize("  Entropy:     5.000000")
        XCTAssertNil(color(of: "5.000000", in: attr))
    }

    func testEntropyInHealthyBandIsUncolored() {
        let attr = colorize("  Entropy:     7.500000")
        XCTAssertNil(color(of: "7.500000", in: attr))
    }

    func testEntropyNearUniformIsOrange() {
        // Above the flat threshold (default 8.20) → orange band.
        let attr = colorize("  Entropy:     8.250000")
        XCTAssertEqual(color(of: "8.250000", in: attr), .orange)
    }

    // MARK: - v abs bands

    func testVAbsHighIsRed() {
        let attr = colorize("    v abs:         0.9500")
        XCTAssertEqual(color(of: "0.9500", in: attr), .red)
    }

    func testVAbsWatchZoneIsOrange() {
        let attr = colorize("    v abs:         0.8000")
        XCTAssertEqual(color(of: "0.8000", in: attr), .orange)
    }

    func testVAbsLowIsUncolored() {
        let attr = colorize("    v abs:         0.3000")
        XCTAssertNil(color(of: "0.3000", in: attr))
    }

    func testVAbsDashPlaceholderDoesNotCrash() {
        // "--" is the standard placeholder when no data yet. Not
        // a number — colorizer must fall through to plain text.
        let attr = colorize("    v abs:         --")
        XCTAssertEqual(String(attr.characters), "    v abs:         --")
    }

    // MARK: - Grad norm bands

    func testGradNormBelowClipIsUncolored() {
        let attr = colorize("  Grad norm:   2.500")
        XCTAssertNil(color(of: "2.500", in: attr))
    }

    func testGradNormAtClipIsOrange() {
        // Matches gradClipMaxNorm (5.0) → clip event.
        let attr = colorize("  Grad norm:   5.000")
        XCTAssertEqual(color(of: "5.000", in: attr), .orange)
    }

    func testGradNormFarAboveClipIsRed() {
        // ≥ 2× clip ceiling → divergence candidate.
        let attr = colorize("  Grad norm:   12.500")
        XCTAssertEqual(color(of: "12.500", in: attr), .red)
    }

    // MARK: - Diversity bands

    func testDiversityHighPercentIsGreen() {
        let attr = colorize("  Diversity:   195/200 unique (97%)  avg diverge ply 8.3")
        XCTAssertEqual(color(of: "97%", in: attr), .green)
    }

    func testDiversityMidPercentIsOrange() {
        let attr = colorize("  Diversity:   60/100 unique (60%)  avg diverge ply 4.1")
        XCTAssertEqual(color(of: "60%", in: attr), .orange)
    }

    func testDiversityLowPercentIsRed() {
        let attr = colorize("  Diversity:   5/200 unique (2%)  avg diverge ply 0.5")
        XCTAssertEqual(color(of: "2%", in: attr), .red)
    }

    // MARK: - Multi-line body

    func testMultiLineBodyColorsOnlyRelevantLines() {
        let body = """
          Total loss:   1.234
          Entropy:     4.000000
          Grad norm:   5.500
          Some other:  99.9
        """
        let attr = AttributedMetricColor.colorize(body: body, thresholds: defaults)
        // Verify round-trip equality of the plain string.
        XCTAssertEqual(String(attr.characters), body)
        // Entropy collapse → red.
        XCTAssertEqual(color(of: "4.000000", in: attr), .red)
        // gNorm just above clip → orange.
        XCTAssertEqual(color(of: "5.500", in: attr), .orange)
        // Unrelated line: no color.
        XCTAssertNil(color(of: "99.9", in: attr))
        // Unrelated line: no color on the "1.234" token either.
        XCTAssertNil(color(of: "1.234", in: attr))
    }

    func testEmptyBodyReturnsEmpty() {
        let attr = AttributedMetricColor.colorize(body: "", thresholds: defaults)
        XCTAssertEqual(String(attr.characters), "")
    }

    func testTrailingNewlinePreserved() {
        let body = "  Entropy:     7.000000\n"
        let attr = AttributedMetricColor.colorize(body: body, thresholds: defaults)
        XCTAssertEqual(String(attr.characters), body)
    }
}
