import AppKit
import XCTest
@testable import DrewsChessMachine

/// The Overview's credits and outcomes lines keep every count in a fixed
/// width, so the words after a number stay put as the number grows, and the
/// padding character really is as wide as a digit in the line's font.
final class LichessBotChallengeCreditsLineTests: XCTestCase {

    /// A day's log of `created` bot challenges: all but the newest few sent
    /// over a minute before `now`, so the last-minute count stays within
    /// its budget's width; alternately accepted and declined.
    private func log(created: Int, withinLastMinute: Int, now: Date) -> LichessBotChallengeOutcomeLog {
        var log = LichessBotChallengeOutcomeLog()
        for index in 0..<created {
            let age: TimeInterval = index < withinLastMinute ? 5 : 120 + Double(index) * 60
            let id = "c\(index)"
            let sentAt = now.addingTimeInterval(-age)
            log.recordCreated(challengeID: id, opponentID: "bot\(index)", kind: .bot, at: sentAt)
            let outcome: LichessBotChallengeOutcome = index.isMultiple(of: 2) ? .accepted : .declined(.unstated)
            XCTAssertTrue(log.resolve(challengeID: id, outcome: outcome, at: sentAt.addingTimeInterval(1)))
        }
        return log
    }

    /// The distance from the start of `line` to the first `marker`.
    private func offset(of marker: String, in line: String) throws -> Int {
        let range = try XCTUnwrap(line.range(of: marker), "\"\(marker)\" not in \"\(line)\"")
        return line.distance(from: line.startIndex, to: range.lowerBound)
    }

    func testCountsKeepTheirWidthAsDigitsGrow() throws {
        let now = Date()
        let few = LichessBotChallengeCreditsLine.lines(log: log(created: 7, withinLastMinute: 1, now: now), now: now)
        let many = LichessBotChallengeCreditsLine.lines(log: log(created: 187, withinLastMinute: 12, now: now), now: now)
        let nbsp = "\u{00A0}"
        let perDay = LichessBotChallengeCredits.perDay
        let perMinute = LichessBotChallengeCredits.perMinute
        XCTAssertEqual(try offset(of: "of\(nbsp)\(perDay)", in: few.credits), try offset(of: "of\(nbsp)\(perDay)", in: many.credits))
        XCTAssertEqual(try offset(of: "of\(nbsp)\(perMinute)", in: few.credits), try offset(of: "of\(nbsp)\(perMinute)", in: many.credits))
        for word in ["accepted", "declined", "refused", "acceptance"] {
            XCTAssertEqual(try offset(of: word, in: few.outcomes), try offset(of: word, in: many.outcomes), word)
        }
        XCTAssertEqual(few.credits.count, many.credits.count)
        XCTAssertEqual(few.outcomes.count, many.outcomes.count)
    }

    func testTheAcceptanceRateKeepsItsWidth() throws {
        let now = Date()
        var none = LichessBotChallengeOutcomeLog()
        none.recordCreated(challengeID: "p", opponentID: "pending", kind: .bot, at: now.addingTimeInterval(-300))
        let noRate = LichessBotChallengeCreditsLine.lines(log: none, now: now)
        let half = LichessBotChallengeCreditsLine.lines(log: log(created: 2, withinLastMinute: 0, now: now), now: now)
        let all = LichessBotChallengeCreditsLine.lines(log: log(created: 1, withinLastMinute: 0, now: now), now: now)
        XCTAssertEqual(try offset(of: "acceptance", in: half.outcomes), try offset(of: "acceptance", in: all.outcomes))
        XCTAssertEqual(try offset(of: "acceptance", in: noRate.outcomes), try offset(of: "acceptance", in: all.outcomes))
    }

    func testPaddingRightAlignsToTheWidestValue() {
        let figureSpace = "\u{2007}"
        XCTAssertEqual(LichessBotChallengeCreditsLine.padded(7, toWidthOf: 200), figureSpace + figureSpace + "7")
        XCTAssertEqual(LichessBotChallengeCreditsLine.padded(200, toWidthOf: 200), "200")
        XCTAssertEqual(LichessBotChallengeCreditsLine.padded(1234, toWidthOf: 200), "1234", "a value wider than the widest is shown in full")
    }

    /// The padding is only invisible alignment if a figure space is as wide
    /// as a digit in the line's font.
    func testFigureSpaceIsAsWideAsADigit() {
        let size = NSFont.preferredFont(forTextStyle: .callout).pointSize
        let font = NSFont.monospacedDigitSystemFont(ofSize: size, weight: .regular)
        func width(_ text: String) -> CGFloat {
            NSAttributedString(string: text, attributes: [.font: font]).size().width
        }
        XCTAssertEqual(width("\u{2007}"), width("0"), accuracy: 0.01)
        XCTAssertEqual(width("1"), width("0"), accuracy: 0.01)
    }
}
