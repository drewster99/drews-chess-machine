@testable import DrewsChessMachine
import XCTest

/// When the pending challenges' players' statuses are read, and what each
/// row shows from them.
final class LichessBotPendingOpponentStatusesTests: XCTestCase {
    private let start = Date(timeIntervalSince1970: 1_800_000_000)

    private func status(_ id: String, online: Bool?, playing: Bool?) -> LichessBotUserStatus {
        LichessBotUserStatus(id: id, name: id, title: "BOT", online: online, playing: playing)
    }

    /// Protects: no request while nothing is pending; the first pending
    /// challenge is read at once.
    /// Fails if: it reads with no ids, or waits the interval before a first read.
    func testReadsOnlyWithPendingChallenges() {
        let statuses = LichessBotPendingOpponentStatuses()
        XCTAssertFalse(statuses.needsRead(of: [], now: start))
        XCTAssertTrue(statuses.needsRead(of: ["a"], now: start))
    }

    /// Protects: reads are spaced by the interval, except that a newly
    /// pending player is read at once.
    /// Fails if: it rereads every poll, never rereads, or makes a new
    /// challenge's player wait the interval.
    func testIntervalAndNewPlayers() {
        var statuses = LichessBotPendingOpponentStatuses()
        statuses.noteReadStarted(at: start)
        statuses.noteRead(["a": status("a", online: true, playing: false)], at: start)
        let soon = start.addingTimeInterval(LichessBotPendingOpponentStatuses.readInterval - 1)
        XCTAssertFalse(statuses.needsRead(of: ["a"], now: soon))
        XCTAssertTrue(statuses.needsRead(of: ["a", "b"], now: soon), "a new pending player")
        XCTAssertTrue(statuses.needsRead(of: ["a"], now: start.addingTimeInterval(LichessBotPendingOpponentStatuses.readInterval)))
    }

    /// Protects: a failing read is retried once per interval, new players included.
    /// Fails if: a failure makes every poll send a request.
    func testFailureBacksOffForTheInterval() {
        var statuses = LichessBotPendingOpponentStatuses()
        statuses.noteReadStarted(at: start)
        statuses.noteReadFailed("offline")
        let soon = start.addingTimeInterval(1)
        XCTAssertFalse(statuses.needsRead(of: ["a"], now: soon))
        XCTAssertFalse(statuses.needsRead(of: ["a", "b"], now: soon))
        XCTAssertTrue(statuses.needsRead(of: ["a"], now: start.addingTimeInterval(LichessBotPendingOpponentStatuses.readInterval)))
        XCTAssertEqual(statuses.activity(of: "a"), .unknown("offline"))
    }

    /// Protects: what a row shows from Lichess's flags, matched by
    /// lowercased id (rows hold the username as typed).
    /// Fails if: playing isn't told from online, an absent `online` isn't
    /// offline, or the lookup is case-sensitive.
    func testActivityFromFlags() {
        var statuses = LichessBotPendingOpponentStatuses()
        XCTAssertEqual(statuses.activity(of: "Busy"), .notReadYet)
        statuses.noteRead([
            "busy": status("busy", online: true, playing: true),
            "idle": status("idle", online: true, playing: nil),
            "gone": status("gone", online: nil, playing: nil),
        ], at: start)
        XCTAssertEqual(statuses.activity(of: "Busy"), .playing(readAt: start))
        XCTAssertEqual(statuses.activity(of: "idle"), .onlineNotPlaying(readAt: start))
        XCTAssertEqual(statuses.activity(of: "gone"), .offline(readAt: start))
        XCTAssertEqual(statuses.activity(of: "never"), .notReadYet)
    }
}
