import XCTest
@testable import DrewsChessMachine

/// `LichessBotChallengeQueue` and the controller's classification of a
/// queued send's failure (plan §7.3 A): order, duplicate rejection, skip
/// reasons, one send at a time, and stopping on a 429 then resuming.
final class LichessBotChallengeQueueTests: XCTestCase {

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    private func players(_ names: String...) -> [LichessBotChallengeQueue.Player] {
        names.map { LichessBotChallengeQueue.Player(username: $0, userID: $0) }
    }

    private func sendStep(_ queue: LichessBotChallengeQueue, freeSlots: Int = 1) throws -> LichessBotChallengeQueue.Entry {
        guard case .send(let entry) = queue.nextStep(sendingBlockedReason: nil, freeSlots: freeSlots) else {
            XCTFail("expected a send, got \(queue.nextStep(sendingBlockedReason: nil, freeSlots: freeSlots))")
            throw CancellationError()
        }
        return entry
    }

    func testEntriesKeepTheOrderGiven() throws {
        var queue = LichessBotChallengeQueue()
        let result = queue.add(players("Alice", "Bob", "Carol"), request: request, pendingUserIDs: [])
        XCTAssertEqual(result.added, ["Alice", "Bob", "Carol"])
        XCTAssertEqual(queue.entries.map(\.username), ["Alice", "Bob", "Carol"])
        XCTAssertEqual(queue.entries.map(\.userID), ["alice", "bob", "carol"], "ids are lowercased")
        XCTAssertEqual(try sendStep(queue).username, "Alice")
    }

    func testDuplicatesOfAQueuedPlayerAreNotAdded() {
        var queue = LichessBotChallengeQueue()
        let first = queue.add(players("Alice", "Bob", "alice"), request: request, pendingUserIDs: [])
        XCTAssertEqual(first.added, ["Alice", "Bob"])
        XCTAssertEqual(first.alreadyQueued, ["alice"], "the same player twice in one add")
        let second = queue.add(players("BOB", "Dave"), request: request, pendingUserIDs: [])
        XCTAssertEqual(second.added, ["Dave"])
        XCTAssertEqual(second.alreadyQueued, ["BOB"], "ids compare case-insensitively")
        XCTAssertEqual(queue.entries.map(\.username), ["Alice", "Bob", "Dave"])
    }

    func testDuplicatesOfAPendingOpponentAreNotAdded() {
        var queue = LichessBotChallengeQueue()
        let result = queue.add(players("Alice", "Bob"), request: request, pendingUserIDs: ["bob"])
        XCTAssertEqual(result.added, ["Alice"])
        XCTAssertEqual(result.alreadyPending, ["Bob"])
        XCTAssertEqual(queue.entries.map(\.username), ["Alice"])
    }

    func testAPlayerBeingSentIsStillADuplicate() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice"), request: request, pendingUserIDs: [])
        let alice = try sendStep(queue)
        XCTAssertTrue(queue.markSending(alice.id))
        let result = queue.add(players("Alice"), request: request, pendingUserIDs: [])
        XCTAssertEqual(result.alreadyQueued, ["Alice"])
        XCTAssertEqual(queue.entries.count, 1)
    }

    func testChoosingASkippedPlayerAgainRetriesThemAtTheEnd() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice", "Bob"), request: request, pendingUserIDs: [])
        let alice = try sendStep(queue)
        queue.markSending(alice.id)
        queue.record(.skipped(reason: "offline"), for: alice.id)
        let result = queue.add(players("Alice"), request: request, pendingUserIDs: [])
        XCTAssertEqual(result.added, ["Alice"])
        XCTAssertEqual(queue.entries.map(\.username), ["Bob", "Alice"])
        XCTAssertEqual(queue.entries.map(\.status), [.waiting, .waiting])
    }

    func testSkippedEntriesStayListedWithTheirReasonAndTheQueueMovesOn() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice", "Bob", "Carol"), request: request, pendingUserIDs: [])

        let alice = try sendStep(queue)
        queue.markSending(alice.id)
        queue.record(.skipped(reason: "offline"), for: alice.id)
        let bob = try sendStep(queue)
        XCTAssertEqual(bob.username, "Bob", "the queue moves on past a skipped entry")
        queue.markSending(bob.id)
        queue.record(.sent, for: bob.id)
        let carol = try sendStep(queue)
        queue.skip(carol.id, reason: "at its bot-game limit")

        XCTAssertEqual(queue.entries.map(\.username), ["Alice", "Carol"], "a sent entry leaves; skipped ones stay listed")
        XCTAssertEqual(queue.entries.map(\.status), [.skipped(reason: "offline"), .skipped(reason: "at its bot-game limit")])
        XCTAssertFalse(queue.hasEntriesToSend, "skipped entries are done")
        XCTAssertEqual(queue.nextStep(sendingBlockedReason: nil, freeSlots: 5), .idle)
    }

    func testADroppedEntryLeavesTheQueue() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice", "Bob"), request: request, pendingUserIDs: [])
        let alice = try sendStep(queue)
        queue.markSending(alice.id)
        queue.record(.dropped(reason: "HTTP 400"), for: alice.id)
        XCTAssertEqual(queue.entries.map(\.username), ["Bob"])
    }

    func testOnlyOneEntryIsSentAtATime() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice", "Bob"), request: request, pendingUserIDs: [])
        let alice = try sendStep(queue, freeSlots: 5)
        queue.markSending(alice.id)
        XCTAssertEqual(queue.nextStep(sendingBlockedReason: nil, freeSlots: 5), .wait(reason: LichessBotChallengeQueue.sendInProgressReason))
    }

    func testEntriesWaitForAFreeSlot() {
        var queue = LichessBotChallengeQueue()
        XCTAssertEqual(queue.nextStep(sendingBlockedReason: nil, freeSlots: 1), .idle)
        _ = queue.add(players("Alice"), request: request, pendingUserIDs: [])
        XCTAssertEqual(queue.nextStep(sendingBlockedReason: nil, freeSlots: 0), .wait(reason: LichessBotChallengeQueue.waitingForSlotReason))
        XCTAssertEqual(queue.nextStep(sendingBlockedReason: "draining", freeSlots: 3), .wait(reason: "draining"))
    }

    /// A 429 stops the queue with the entry kept in its place; once the
    /// hold is over, the same entry goes next.
    func testTheQueueStopsOnARateLimitAndResumesAfter() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice", "Bob"), request: request, pendingUserIDs: [])
        let alice = try sendStep(queue)
        queue.markSending(alice.id)
        queue.record(LichessBotController.queueOutcome(for: LichessBotGateError.rateLimited(cooldown: .seconds(60))), for: alice.id)

        XCTAssertEqual(queue.entries.map(\.username), ["Alice", "Bob"])
        XCTAssertEqual(queue.entries.map(\.status), [.waiting, .waiting])
        XCTAssertEqual(queue.nextStep(sendingBlockedReason: "rate-limit hold after a 429", freeSlots: 2), .wait(reason: "rate-limit hold after a 429"))
        XCTAssertEqual(try sendStep(queue, freeSlots: 2).id, alice.id, "the stopped entry is next once the hold ends")
    }

    func testCancelingAnEntryBeingSentIgnoresItsOutcome() throws {
        var queue = LichessBotChallengeQueue()
        _ = queue.add(players("Alice", "Bob"), request: request, pendingUserIDs: [])
        let alice = try sendStep(queue)
        queue.markSending(alice.id)
        queue.remove(alice.id)
        queue.record(.skipped(reason: "offline"), for: alice.id)
        XCTAssertEqual(queue.entries.map(\.username), ["Bob"])
        XCTAssertFalse(queue.markSending(alice.id))
    }

    // MARK: - Failure classification

    func testFailuresTiedToThePlayerSkipThem() {
        XCTAssertEqual(
            LichessBotController.queueOutcome(for: LichessBotControllerError.perOpponentGameLimit(username: "bob", limit: 1, committed: 1)),
            .skipped(reason: "already playing or challenging them (the per-opponent limit)")
        )
        XCTAssertEqual(LichessBotController.queueOutcome(for: LichessBotControllerError.opponentOffline("bob")), .skipped(reason: "offline"))
        let refusal = LichessBotAPIError.http(status: 400, message: "bob played 100 games against other bots today, please wait until 2026-09-29T06:57:07.895Z to challenge them.")
        guard case .skipped(let reason) = LichessBotController.queueOutcome(for: refusal) else {
            return XCTFail("a bot-limit refusal skips the player")
        }
        XCTAssertTrue(reason.hasPrefix("at its bot-game limit until"), reason)
    }

    func testStateChangesAndRateLimitsStopTheQueue() {
        for error: Error in [
            LichessBotGateError.rateLimited(cooldown: .seconds(60)),
            LichessBotGateError.closed(reason: "breaker"),
            LichessBotControllerError.notOnline,
            LichessBotControllerError.concurrentGameLimit(limit: 2, committed: 2),
            LichessBotControllerError.missingChallengeScope,
            CancellationError(),
        ] {
            guard case .stopped = LichessBotController.queueOutcome(for: error) else {
                XCTFail("\(error) must stop the queue")
                continue
            }
        }
    }

    func testOtherRefusalsDropTheEntry() {
        guard case .dropped = LichessBotController.queueOutcome(for: LichessBotAPIError.http(status: 400, message: "Challenge not allowed")) else {
            return XCTFail("a Lichess refusal drops the entry")
        }
        guard case .dropped = LichessBotController.queueOutcome(for: LichessBotControllerError.noSuchPlayer("ghost")) else {
            return XCTFail("an unknown player drops the entry")
        }
    }
}
