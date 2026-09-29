import XCTest
@testable import DrewsChessMachine

/// Lichess's speed for a clock, and the settings warning for checked speeds
/// the clock and increment bounds rule out.
final class LichessBotSpeedBoundsTests: XCTestCase {

    func testSpeedForClockFollowsTheEstimatedDuration() {
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 15, incrementSeconds: 0), .ultraBullet)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 60, incrementSeconds: 0), .bullet)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 120, incrementSeconds: 1), .bullet)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 180, incrementSeconds: 0), .blitz)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 300, incrementSeconds: 3), .blitz)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 600, incrementSeconds: 0), .rapid)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 900, incrementSeconds: 10), .rapid)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 1800, incrementSeconds: 0), .classical)
        // The boundaries are exclusive above: an estimate exactly on one
        // belongs to the slower speed.
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 30, incrementSeconds: 0), .bullet)
        XCTAssertEqual(LichessBotSpeed.forClock(limitSeconds: 1500, incrementSeconds: 0), .classical)
    }

    func testTheSheetsClockChoicesUseTheSameRule() {
        for choice in LichessBotChallengeSheet.ClockChoice.allCases {
            let seconds = choice.seconds
            XCTAssertEqual(choice.speed, LichessBotSpeed.forClock(limitSeconds: seconds.limit, incrementSeconds: seconds.increment), choice.rawValue)
        }
    }

    func testClockBoundsRuleOutFasterAndSlowerSpeeds() {
        var settings = LichessBotChallengeSettings()
        settings.allowedSpeeds = [.ultraBullet, .bullet, .blitz, .rapid, .classical]
        settings.minimumInitialSeconds = 180
        settings.maximumInitialSeconds = 1800
        settings.minimumIncrementSeconds = 0
        settings.maximumIncrementSeconds = 30
        XCTAssertEqual(settings.speedsRuledOutByClockBounds, [.ultraBullet, .bullet])

        settings.minimumInitialSeconds = 15
        settings.maximumInitialSeconds = 300
        settings.maximumIncrementSeconds = 0
        XCTAssertEqual(settings.speedsRuledOutByClockBounds, [.rapid, .classical], "the fastest corner reaches ultraBullet; the slowest stops at blitz")
    }

    func testUncheckedSpeedsAreNeverReported() {
        var settings = LichessBotChallengeSettings()
        settings.allowedSpeeds = [.blitz]
        settings.minimumInitialSeconds = 180
        settings.maximumInitialSeconds = 1800
        XCTAssertEqual(settings.speedsRuledOutByClockBounds, [])
    }

    func testCorrespondenceIsAlwaysRuledOut() {
        var settings = LichessBotChallengeSettings()
        settings.allowedSpeeds = [.blitz, .correspondence]
        XCTAssertEqual(settings.speedsRuledOutByClockBounds, [.correspondence])
    }
}
