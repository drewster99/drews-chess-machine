@testable import DrewsChessMachine
import XCTest

/// Fill Open Slots' two targets: what counts toward the goal, when another
/// send is needed, and what the fill does after each pass outcome.
@MainActor
final class LichessBotFillOpenSlotsTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    private func request(_ count: Int, _ target: LichessBotFillOpenSlotsTarget) -> LichessBotFillOpenSlotsRequest {
        LichessBotFillOpenSlotsRequest(count: count, target: target, overridesTimeLimits: false)
    }

    /// Protects: starting games, the count may exceed the open slots and
    /// doesn't depend on them; sending challenges stays capped by them.
    /// Fails if: the games range follows the open slots, or is nil with none open.
    func testCountRangeByTarget() {
        XCTAssertEqual(LichessBotFillOpenSlotsRequest.countRange(target: .gamesStarted, openSlots: 0), 1...LichessBotFillOpenSlotsRequest.maximumGamesToStart)
        XCTAssertEqual(LichessBotFillOpenSlotsRequest.countRange(target: .gamesStarted, openSlots: 10), 1...LichessBotFillOpenSlotsRequest.maximumGamesToStart)
        XCTAssertEqual(LichessBotFillOpenSlotsRequest.countRange(target: .challengesSent, openSlots: 10), 1...10)
        XCTAssertNil(LichessBotFillOpenSlotsRequest.countRange(target: .challengesSent, openSlots: 0))
    }

    /// Protects: starting games counts accepted challenges only; a declined
    /// one is replaced, and pending ones hold off further sends.
    /// Fails if: sends count toward the goal, a decline isn't freed, or the
    /// pending challenges aren't counted against the remaining need.
    func testGamesStartedCountsAcceptancesAndReplacesDeclines() {
        let goal = request(2, .gamesStarted)
        var progress = LichessBotFillOpenSlotsProgress()
        XCTAssertTrue(progress.needsAnotherSend(goal))
        progress.noteSent(challengeID: "a")
        progress.noteSent(challengeID: "b")
        XCTAssertFalse(progress.needsAnotherSend(goal), "two pending could start both games")
        XCTAssertFalse(progress.isComplete(goal))
        XCTAssertTrue(progress.noteAnswered(challengeID: "a", accepted: false))
        XCTAssertTrue(progress.needsAnotherSend(goal), "a declined challenge is replaced")
        XCTAssertTrue(progress.noteAnswered(challengeID: "b", accepted: true))
        progress.noteSent(challengeID: "c")
        XCTAssertTrue(progress.noteAnswered(challengeID: "c", accepted: true))
        XCTAssertTrue(progress.isComplete(goal))
        XCTAssertEqual(progress.sent, 3)
        XCTAssertEqual(progress.accepted, 2)
        XCTAssertEqual(progress.unanswered, 1)
        XCTAssertTrue(progress.pendingChallengeIDs.isEmpty)
    }

    /// Protects: sending challenges is done at the count sent, whatever the answers.
    /// Fails if: it waits for acceptances or counts answers.
    func testChallengesSentCompletesAtTheCountSent() {
        let goal = request(2, .challengesSent)
        var progress = LichessBotFillOpenSlotsProgress()
        progress.noteSent(challengeID: "a")
        XCTAssertTrue(progress.needsAnotherSend(goal))
        progress.noteSent(challengeID: "b")
        XCTAssertFalse(progress.needsAnotherSend(goal))
        XCTAssertTrue(progress.isComplete(goal))
    }

    /// Protects: an answer to a challenge the fill didn't send, or a second
    /// answer to one (the acceptance and the game start both resolve it),
    /// changes nothing.
    /// Fails if: foreign or repeated answers are counted.
    func testForeignAndRepeatedAnswersAreIgnored() {
        var progress = LichessBotFillOpenSlotsProgress()
        progress.noteSent(challengeID: "a")
        progress.noteSent(challengeID: "a")
        XCTAssertEqual(progress.sent, 1, "a repeated send note")
        XCTAssertFalse(progress.noteAnswered(challengeID: "other", accepted: true))
        XCTAssertTrue(progress.noteAnswered(challengeID: "a", accepted: true))
        XCTAssertFalse(progress.noteAnswered(challengeID: "a", accepted: true))
        XCTAssertEqual(progress.accepted, 1)
        XCTAssertEqual(progress.unanswered, 0)
    }

    /// Protects: what each pass outcome does to each kind of fill.
    /// Fails if: starting games stops at a full house, the cap, the off-peak
    /// schedule, a missing candidate or a failed send; or sending challenges
    /// starts waiting on them; or a spacing wait ends either.
    func testNextStepAfterEachOutcome() {
        let later = now.addingTimeInterval(600)
        let retryAt = now.addingTimeInterval(LichessBotMatchmaking.minimumSendSpacing)
        typealias Controller = LichessBotController
        let cases: [(String, Controller.MatchmakingPassOutcome, LichessBotFillOpenSlotsTarget, Controller.FillOpenSlotsStep)] = [
            ("sent, games", .sent(username: "b"), .gamesStarted, .sendAgain),
            ("sent, challenges", .sent(username: "b"), .challengesSent, .sendAgain),
            ("spacing, games", .rateLimited(.spacing(until: later)), .gamesStarted, .wait(.sendLimit(until: later), until: later)),
            ("spacing, challenges", .rateLimited(.spacing(until: later)), .challengesSent, .wait(.sendLimit(until: later), until: later)),
            ("cap, games", .rateLimited(.hourlyCap(until: later, count: 4)), .gamesStarted, .wait(.sendLimit(until: later), until: later)),
            ("cap, challenges", .rateLimited(.hourlyCap(until: later, count: 4)), .challengesSent, .end(Controller.MatchmakingPassOutcome.rateLimited(.hourlyCap(until: later, count: 4)).text)),
            ("off-peak with a peak, games", .rateLimited(.offPeakPaused(peakStartsAt: later)), .gamesStarted, .wait(.sendLimit(until: later), until: later)),
            ("off-peak, no peak, games", .rateLimited(.offPeakPaused(peakStartsAt: nil)), .gamesStarted, .end(Controller.MatchmakingPassOutcome.rateLimited(.offPeakPaused(peakStartsAt: nil)).text)),
            ("no slot, games", .noOpenSlot, .gamesStarted, .waitForFreeSlot),
            ("no slot, challenges", .noOpenSlot, .challengesSent, .end(Controller.MatchmakingPassOutcome.noOpenSlot.text)),
            ("no candidate, games", .noCandidate("none"), .gamesStarted, .retry(until: retryAt)),
            ("no candidate, challenges", .noCandidate("none"), .challengesSent, .end("none")),
            ("failed, games", .failed("x at limit"), .gamesStarted, .retry(until: retryAt)),
            ("failed, challenges", .failed("x at limit"), .challengesSent, .end("failed: x at limit")),
            ("blocked, games", .blocked(reason: "held"), .gamesStarted, .end("paused: held")),
            ("stopped, games", .stopped("gone"), .gamesStarted, .end("stopped: gone")),
        ]
        for (name, outcome, target, expected) in cases {
            XCTAssertEqual(Controller.fillOpenSlotsNextStep(after: outcome, target: target, consecutiveMisses: 0, now: now), expected, name)
        }
    }

    /// Protects: starting games gives up after the last allowed miss in a
    /// row, not before and not never.
    /// Fails if: the miss limit is off by one or ignored.
    func testStartingGamesGivesUpAfterTheMissLimit() {
        let limit = 5
        let beforeLast = LichessBotController.fillOpenSlotsNextStep(after: .noCandidate("none"), target: .gamesStarted, consecutiveMisses: limit - 2, now: now)
        XCTAssertEqual(beforeLast, .retry(until: now.addingTimeInterval(LichessBotMatchmaking.minimumSendSpacing)))
        let last = LichessBotController.fillOpenSlotsNextStep(after: .noCandidate("none"), target: .gamesStarted, consecutiveMisses: limit - 1, now: now)
        guard case .end(let text) = last else {
            return XCTFail("expected the fill to end, got \(last)")
        }
        XCTAssertTrue(text.contains("gave up after \(limit) passes"), text)
    }

    /// Protects: the tally the Overview and the log show for each target.
    /// Fails if: starting games reports only sends, or the counts mix up.
    func testTallyText() {
        var progress = LichessBotFillOpenSlotsProgress()
        progress.noteSent(challengeID: "a")
        progress.noteSent(challengeID: "b")
        progress.noteSent(challengeID: "c")
        progress.noteAnswered(challengeID: "a", accepted: true)
        progress.noteAnswered(challengeID: "b", accepted: false)
        XCTAssertEqual(LichessBotController.fillOpenSlotsTally(request(4, .gamesStarted), progress: progress), "1 of 4 games started; 3 challenges sent, 1 pending, 1 not accepted")
        XCTAssertEqual(LichessBotController.fillOpenSlotsTally(request(4, .challengesSent), progress: progress), "sent 3 of 4")
    }

    /// Protects: a keep-going fill picks another bot after a pick refused
    /// as at its daily limit, without spending a miss; a plain fill still
    /// ends there with Lichess's reason.
    /// Fails if: such refusals count toward giving up (five bots at their
    /// limit in a row ended a 20-game fill after 3 games, 2026-11-06), or
    /// the plain fill stops reporting the limit.
    func testABotAtItsDailyLimitIsNotAMissWhenStartingGames() {
        let until = now.addingTimeInterval(3600)
        let refused = LichessBotController.MatchmakingPassOutcome.opponentAtBotLimit(username: "maia5", until: until)
        XCTAssertEqual(LichessBotController.fillOpenSlotsNextStep(after: refused, target: .gamesStarted, consecutiveMisses: 4, now: now), .pickAnotherBot)
        XCTAssertEqual(LichessBotController.fillOpenSlotsNextStep(after: refused, target: .challengesSent, consecutiveMisses: 0, now: now), .end(refused.text))
        XCTAssertTrue(refused.text.contains("maia5 is at Lichess' bot-vs-bot daily limit until"), refused.text)
    }
}
