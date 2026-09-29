import XCTest
@testable import DrewsChessMachine

/// The outgoing-challenge outcome log: classifying a refused challenge POST
/// from the API error, credit costs by opponent kind, decline reason keys,
/// and the rolling-day and rolling-minute counts under a fixed clock.
final class LichessBotChallengeOutcomeTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_790_000_000)

    // MARK: - Classifying POST failures

    func test_429FromTheGate_isRateLimited() throws {
        let error = LichessBotGateError.rateLimited(cooldown: .seconds(60))
        let refusal = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: error))
        XCTAssertEqual(refusal.kind, .rateLimited)
        XCTAssertEqual(refusal.httpStatus, 429)
        XCTAssertEqual(refusal.text, error.localizedDescription)
    }

    func test_429AsHTTPError_isRateLimited() throws {
        let refusal = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.http(status: 429, message: "Too many requests")))
        XCTAssertEqual(refusal.kind, .rateLimited)
        XCTAssertEqual(refusal.text, "Too many requests")
    }

    func test_400BotDailyLimit_isSplitFromOther400s() throws {
        let limitText = "somebot played 100 games against other bots today, please wait until 2026-09-29T10:00:00Z to challenge them."
        let limit = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.http(status: 400, message: limitText)))
        XCTAssertEqual(limit, LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: limitText))

        let other = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.http(status: 400, message: "This user doesn't accept challenges")))
        XCTAssertEqual(other, LichessBotChallengeRefusal(kind: .badRequest, httpStatus: 400, text: "This user doesn't accept challenges"))

        let silent = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.http(status: 400, message: nil)))
        XCTAssertEqual(silent, LichessBotChallengeRefusal(kind: .badRequest, httpStatus: 400, text: nil))
    }

    func test_otherStatuses_areOtherHTTPStatus() throws {
        let notFound = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.http(status: 404, message: "Not found")))
        XCTAssertEqual(notFound.kind, .otherHTTPStatus)
        XCTAssertEqual(notFound.httpStatus, 404)
        let unauthorized = try XCTUnwrap(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.unauthorized(status: 403, message: "Missing scope")))
        XCTAssertEqual(unauthorized.kind, .otherHTTPStatus)
        XCTAssertEqual(unauthorized.httpStatus, 403)
    }

    func test_errorsWithoutALichessAnswer_areNotClassified() {
        XCTAssertNil(LichessBotChallengeRefusal.classify(postError: LichessBotGateError.closed(reason: "breaker")))
        XCTAssertNil(LichessBotChallengeRefusal.classify(postError: CancellationError()))
        XCTAssertNil(LichessBotChallengeRefusal.classify(postError: URLError(.timedOut)))
        XCTAssertNil(LichessBotChallengeRefusal.classify(postError: LichessBotAPIError.undecodableResponse(endpoint: "/api/challenge", detail: "x")))
    }

    // MARK: - Decline reasons

    func test_declineReasonKeys() {
        for reason in LichessBotDeclineReason.allCases {
            XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: reason.rawValue), .known(reason))
        }
        XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: "somethingNew"), .unrecognized("somethingNew"))
        XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: nil), .unstated)
        XCTAssertEqual(Set(LichessBotDeclineReason.allCases.map(\.rawValue)),
                       ["generic", "later", "tooFast", "tooSlow", "timeControl", "rated", "casual", "standard", "variant", "noBot", "onlyBot"])
    }

    // MARK: - Credit costs

    func test_creditCostByOpponentKind() {
        XCTAssertEqual(LichessBotChallengeCredits.cost(for: .bot), 1)
        XCTAssertEqual(LichessBotChallengeCredits.cost(for: .human(following: true)), 0)
        XCTAssertEqual(LichessBotChallengeCredits.cost(for: .human(following: false)), 5)
        XCTAssertEqual(LichessBotChallengeCredits.cost(for: .human(following: nil)), 5)
        XCTAssertEqual(LichessBotChallengeCredits.perDay, 200)
        XCTAssertEqual(LichessBotChallengeCredits.perMinute, 25)
    }

    func test_notCreatedAttempts_costNothing() {
        var log = LichessBotChallengeOutcomeLog()
        XCTAssertTrue(log.recordNotCreated(opponentID: "a", kind: .human(following: false), outcome: .offline, at: now))
        XCTAssertTrue(log.recordNotCreated(opponentID: "b", kind: .bot, outcome: .refused(LichessBotChallengeRefusal(kind: .rateLimited, httpStatus: 429, text: nil)), at: now))
        XCTAssertFalse(log.recordNotCreated(opponentID: "c", kind: .bot, outcome: .accepted, at: now))
        let summary = log.summary(now: now)
        XCTAssertEqual(summary.creditsLastDay, 0)
        XCTAssertEqual(summary.offline, 1)
        XCTAssertEqual(summary.refused, 1)
        XCTAssertEqual(summary.refusedByKind, [.rateLimited: 1])
        XCTAssertEqual(log.records.count, 2)
    }

    // MARK: - Resolving

    func test_resolve_firstAnswerWins() {
        var log = LichessBotChallengeOutcomeLog()
        log.recordCreated(challengeID: "c1", opponentID: "Bot1", kind: .bot, at: now)
        XCTAssertEqual(log.summary(now: now).pending, 1)
        XCTAssertTrue(log.resolve(challengeID: "c1", outcome: .accepted, at: now))
        XCTAssertFalse(log.resolve(challengeID: "c1", outcome: .canceled, at: now))
        XCTAssertFalse(log.resolve(challengeID: "unknown", outcome: .canceled, at: now))
        XCTAssertFalse(log.resolve(challengeID: "c1", outcome: .offline, at: now))
        let summary = log.summary(now: now)
        XCTAssertEqual(summary.accepted, 1)
        XCTAssertEqual(summary.canceled, 0)
        XCTAssertEqual(summary.pending, 0)
        XCTAssertEqual(log.records.first?.opponentID, "bot1")
    }

    // MARK: - Rolling windows

    func test_rollingDayAndMinute_withFixedClock() {
        var log = LichessBotChallengeOutcomeLog()
        // A day and a second ago: outside the day.
        log.recordCreated(challengeID: "old", opponentID: "a", kind: .bot, at: now.addingTimeInterval(-86_401))
        // Twenty hours ago, a human DCM doesn't follow: 5.
        log.recordCreated(challengeID: "h", opponentID: "h", kind: .human(following: false), at: now.addingTimeInterval(-72_000))
        log.resolve(challengeID: "h", outcome: .declined(.known(.noBot)), at: now.addingTimeInterval(-71_990))
        // Two minutes ago: in the day, not the minute.
        log.recordCreated(challengeID: "b1", opponentID: "b1", kind: .bot, at: now.addingTimeInterval(-120))
        log.resolve(challengeID: "b1", outcome: .declined(.known(.tooFast)), at: now.addingTimeInterval(-100))
        // Thirty seconds ago: in both.
        log.recordCreated(challengeID: "b2", opponentID: "b2", kind: .bot, at: now.addingTimeInterval(-30))
        log.resolve(challengeID: "b2", outcome: .accepted, at: now.addingTimeInterval(-10))
        log.recordCreated(challengeID: "f", opponentID: "f", kind: .human(following: true), at: now.addingTimeInterval(-20))

        let summary = log.summary(now: now)
        XCTAssertEqual(summary.creditsLastDay, 5 + 1 + 1 + 0)
        XCTAssertEqual(summary.creditsLastMinute, 1)
        XCTAssertEqual(summary.accepted, 1)
        XCTAssertEqual(summary.declined, 2)
        XCTAssertEqual(summary.pending, 1)
        XCTAssertEqual(summary.declinedByReason, [.known(.noBot): 1, .known(.tooFast): 1])
        XCTAssertEqual(summary.acceptanceRate, 1.0 / 3.0)

        // Once the human challenge is a day old it leaves the day; once the
        // latest bot challenge is a minute old it leaves the minute.
        XCTAssertEqual(log.summary(now: now.addingTimeInterval(4 * 3600 + 1)).creditsLastDay, 2)
        XCTAssertEqual(log.summary(now: now.addingTimeInterval(61)).creditsLastMinute, 0)
    }

    func test_windowEdges() {
        var log = LichessBotChallengeOutcomeLog()
        log.recordCreated(challengeID: "c", opponentID: "c", kind: .bot, at: now)
        XCTAssertEqual(log.summary(now: now.addingTimeInterval(59.999)).creditsLastMinute, 1)
        XCTAssertEqual(log.summary(now: now.addingTimeInterval(60)).creditsLastMinute, 0)
        XCTAssertEqual(log.summary(now: now.addingTimeInterval(86_399.999)).creditsLastDay, 1)
        XCTAssertEqual(log.summary(now: now.addingTimeInterval(86_400)).creditsLastDay, 0)
    }

    func test_prune_dropsRecordsOlderThanADay() {
        var log = LichessBotChallengeOutcomeLog()
        log.recordCreated(challengeID: "old", opponentID: "a", kind: .bot, at: now.addingTimeInterval(-86_400))
        log.recordCreated(challengeID: "new", opponentID: "b", kind: .bot, at: now.addingTimeInterval(-86_399))
        log.prune(now: now)
        XCTAssertEqual(log.records.map(\.challengeID), ["new"])
    }

    func test_acceptanceRate_nilWithoutAnswers() {
        var log = LichessBotChallengeOutcomeLog()
        XCTAssertNil(log.summary(now: now).acceptanceRate)
        log.recordCreated(challengeID: "p", opponentID: "p", kind: .bot, at: now)
        log.recordNotCreated(opponentID: "o", kind: .bot, outcome: .offline, at: now)
        XCTAssertNil(log.summary(now: now).acceptanceRate)
    }

    // MARK: - Persistence

    func test_saveAndLoad_roundTrip() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("challenge-outcomes-\(UUID().uuidString)", isDirectory: true)
        defer {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        let url = folder.appendingPathComponent("challenge-outcomes.json")
        XCTAssertEqual(try LichessBotChallengeOutcomeLog.load(from: url), LichessBotChallengeOutcomeLog())

        var log = LichessBotChallengeOutcomeLog()
        // Whole seconds: the file stores ISO-8601 times.
        log.recordCreated(challengeID: "c1", opponentID: "b", kind: .human(following: nil), at: now)
        log.resolve(challengeID: "c1", outcome: .declined(.unrecognized("newKey")), at: now)
        log.recordNotCreated(opponentID: "x", kind: .bot, outcome: .refused(LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: "t")), at: now)
        try log.save(to: url)
        XCTAssertEqual(try LichessBotChallengeOutcomeLog.load(from: url), log)
    }
}
