//
//  LichessBotChallengeReconstructionTests.swift
//  DrewsChessMachineTests
//
//  The back-fill (challenge-log plan §3.7): Algorithm v1 over synthetic
//  protocol lines, one per frozen message form, written exactly as
//  `LichessBotController` writes them; and the store that keeps
//  `Challenges/reconstructed-from-protocol.json` in step with its inputs,
//  writing only when the bytes change. File tests use a temporary folder
//  only.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeReconstructionTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    /// Display name and lowercased id differ only in case, as the real
    /// protocol messages mix them.
    private static let us = "drewschessmachine"
    private static let usName = "DrewsChessMachine"

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotChallengeReconstructionTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    // MARK: - Protocol lines

    /// One protocol day file's entries, in the order written.
    private struct ProtocolLines {
        let name: String
        private(set) var entries: [LichessBotProtocolEntry] = []

        init(name: String = "events-20261005.jsonl") {
            self.name = name
        }

        mutating func add(_ kind: LichessBotProtocolEventKind, _ message: String, fields: [String: String] = [:], at seconds: TimeInterval) {
            entries.append(LichessBotProtocolEntry(at: F.at(seconds), kind: kind, gameID: nil, message: message, fields: fields))
        }

        mutating func challenge(_ message: String, fields: [String: String] = [:], at seconds: TimeInterval) {
            add(.challenge, message, fields: fields, at: seconds)
        }

        mutating func stream(_ json: String, at seconds: TimeInterval) {
            add(.stream, json, fields: ["stream": "event"], at: seconds)
        }

        func data() throws -> Data {
            var data = Data()
            for entry in entries {
                data.append(try LichessBotJSONLines.encodeLine(entry))
            }
            return data
        }

        func file() throws -> LichessBotProtocolDayFile {
            LichessBotProtocolDayFile(name: name, data: try data())
        }
    }

    // Raw event-stream lines, shaped as Lichess sends them.

    private static func party(_ id: String, _ name: String) -> String {
        "{\"name\":\"\(name)\",\"title\":\"BOT\",\"id\":\"\(id)\",\"rating\":1500,\"online\":true}"
    }

    private static func challengeEvent(id: String, challenger: (id: String, name: String), dest: (id: String, name: String)) -> String {
        "{\"type\":\"challenge\",\"challenge\":{\"id\":\"\(id)\",\"url\":\"https://lichess.org/\(id)\",\"status\":\"created\","
            + "\"challenger\":\(party(challenger.id, challenger.name)),\"destUser\":\(party(dest.id, dest.name)),"
            + "\"variant\":{\"key\":\"standard\",\"name\":\"Standard\",\"short\":\"Std\"},\"rated\":true,\"speed\":\"blitz\","
            + "\"timeControl\":{\"type\":\"clock\",\"limit\":180,\"increment\":2,\"show\":\"3+2\"},\"color\":\"random\",\"finalColor\":\"white\"}}"
    }

    private static func outgoingEvent(id: String, to name: String) -> String {
        challengeEvent(id: id, challenger: (us, usName), dest: (name.lowercased(), name))
    }

    private static func incomingEvent(id: String, from name: String) -> String {
        challengeEvent(id: id, challenger: (name.lowercased(), name), dest: (us, usName))
    }

    private static func declinedEvent(id: String, key: String) -> String {
        "{\"type\":\"challengeDeclined\",\"challenge\":{\"id\":\"\(id)\",\"status\":\"declined\",\"declineReason\":\"I'm not accepting challenges at the moment.\",\"declineReasonKey\":\"\(key)\"}}"
    }

    private static func canceledEvent(id: String) -> String {
        "{\"type\":\"challengeCanceled\",\"challenge\":{\"id\":\"\(id)\",\"status\":\"canceled\"}}"
    }

    private static func gameStartEvent(id: String) -> String {
        "{\"type\":\"gameStart\",\"game\":{\"fullId\":\"[REDACTED]\",\"gameId\":\"\(id)\",\"source\":\"friend\"}}"
    }

    // `.challenge` messages, worded exactly as the controller writes them.

    private static func sendFields(id: String, rated: Bool = true, clock: String = "180+2", color: String = "random") -> [String: String] {
        ["id": id, "rated": "\(rated)", "clock": clock, "color": color]
    }

    private static func refusedOutcome(_ opponentID: String) -> String {
        "challenge outcome: \(opponentID) refused: bot daily game limit (400), HTTP 400: \(opponentID) played 100 games against other bots today, please wait until 2026-10-05T18:00:00.000Z to challenge them.; counted 1 credits (worst case)"
    }

    private static func botLimit(_ userID: String) -> String {
        "\(userID) is at its bot-game limit (100) until Oct 5, 2026 at 6:00:00\u{202F}PM"
    }

    private static func pick(_ name: String) -> String {
        "matchmaking pick: \(name) (blitz 1500) at 3+2, uniformly from 5 candidate(s); rating window 1200–1800 (DCM's blitz 1500 + offsets); excluded: DCM itself 1"
    }

    private func build(_ lines: ProtocolLines..., cutoff: Date? = nil, account: String = us) throws -> LichessBotChallengeReconstruction {
        LichessBotChallengeReconstruction.build(from: try lines.map { try $0.file() }, ourAccountID: account, liveLogFirstEntryAt: cutoff)
    }

    private func row(_ reconstruction: LichessBotChallengeReconstruction, challengeID: String,
                     file: StaticString = #filePath, line: UInt = #line) throws -> LichessBotReconstructedChallengeRow {
        try XCTUnwrap(LichessBotReconstructedChallengeLookup(reconstruction).row(challengeID: challengeID), file: file, line: line)
    }

    private func attempts(_ reconstruction: LichessBotChallengeReconstruction) -> [LichessBotReconstructedChallengeRow] {
        reconstruction.rows.filter { if case .notCreatedAttempt = $0.key { return true }; return false }
    }

    // MARK: - Direction and sends

    func testDirectionComesFromTheChallengerComparedCaseInsensitively() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "Out00001", to: "maia1"), at: 1)
        lines.challenge("challenge sent to maia1", fields: Self.sendFields(id: "Out00001"), at: 2)
        lines.stream(Self.incomingEvent(id: "Inc00001", from: "Someone"), at: 3)
        // The configured account in another case still names our own sends.
        let reconstruction = try build(lines, account: Self.usName)

        let outgoing = try row(reconstruction, challengeID: "Out00001")
        XCTAssertEqual(outgoing.direction, .outgoing)
        XCTAssertEqual(outgoing.opponent, LichessBotReconstructedOpponent(id: "maia1", name: "maia1"))
        XCTAssertEqual(outgoing.sendTerms, LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 180, clockIncrementSeconds: 2, color: .random))
        XCTAssertEqual(outgoing.challenge?.id, "Out00001")
        XCTAssertEqual(outgoing.state, .challenge(.open))
        XCTAssertEqual(outgoing.evidence, [.init(file: "events-20261005.jsonl", line: 1), .init(file: "events-20261005.jsonl", line: 2)])

        let incoming = try row(reconstruction, challengeID: "Inc00001")
        XCTAssertEqual(incoming.direction, .incoming)
        XCTAssertEqual(incoming.opponent, LichessBotReconstructedOpponent(id: "someone", name: "Someone"))
        XCTAssertNil(incoming.sender)
        XCTAssertNil(incoming.pickLineCheck)
        XCTAssertEqual(incoming.originConfidence, .certain)

        XCTAssertEqual(reconstruction.counts.outgoingCreatedRows, 1)
        XCTAssertEqual(reconstruction.counts.incomingRows, 1)
        XCTAssertEqual(reconstruction.counts.sendLines, 1)
        XCTAssertEqual(reconstruction.rows.map(\.key), [.challenge(id: "Out00001"), .challenge(id: "Inc00001")])
    }

    func testEachSenderCompanionPairsWithItsSendAcrossNameCase() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "MatchMk1", to: "EdwardKillick"), at: 1)
        lines.challenge("challenge sent to EdwardKillick", fields: Self.sendFields(id: "MatchMk1"), at: 2)
        lines.challenge("matchmaking sent a challenge to edwardkillick", fields: ["clock": "3+2", "rated": "true", "window": "x"], at: 2.01)
        lines.challenge("challenge sent to Cizme", fields: Self.sendFields(id: "Resend01", rated: false), at: 10)
        lines.challenge("matchmaking resent a challenge to cizme as casual", fields: ["clock": "3+2", "rated": "false", "color": "random"], at: 10.01)
        lines.challenge("challenge sent to QueuedOne", fields: Self.sendFields(id: "Queue001"), at: 20)
        lines.challenge("challenge queue: sent QUEUEDONE", at: 20.01)
        lines.challenge("challenge sent to Operator1", fields: Self.sendFields(id: "Oper0001"), at: 30)
        let reconstruction = try build(lines)

        XCTAssertEqual(try row(reconstruction, challengeID: "MatchMk1").sender, .attributed(sender: .matchmaking, confidence: .paired))
        XCTAssertEqual(try row(reconstruction, challengeID: "Resend01").sender, .attributed(sender: .matchmakingCasualResend, confidence: .paired))
        XCTAssertEqual(try row(reconstruction, challengeID: "Queue001").sender, .attributed(sender: .challengeQueue, confidence: .paired))
        let operatorRow = try row(reconstruction, challengeID: "Oper0001")
        XCTAssertEqual(operatorRow.sender, .attributed(sender: .byOperator, confidence: .inferredFromAbsence))
        XCTAssertEqual(operatorRow.originConfidence, .inferredFromAbsence)
        XCTAssertEqual(reconstruction.counts.companionsPaired, 3)
        XCTAssertTrue(reconstruction.unexplainedLines.isEmpty)
        // The paired companion is the row's evidence.
        XCTAssertEqual(try row(reconstruction, challengeID: "MatchMk1").evidence.map(\.line), [1, 2, 3])
        // Rows with no challenge event are outgoing from their send line.
        XCTAssertEqual(try row(reconstruction, challengeID: "Resend01").direction, .outgoing)
        XCTAssertNil(try row(reconstruction, challengeID: "Resend01").challenge)
    }

    func testCompanionsThatAreMissingLateOrAmbiguousStayUnpaired() throws {
        var lines = ProtocolLines()
        // A companion with no send before it.
        lines.challenge("matchmaking sent a challenge to Nobody", at: 1)
        // A companion more than five seconds after its send.
        lines.challenge("challenge sent to Slowpoke", fields: Self.sendFields(id: "Slow0001"), at: 10)
        lines.challenge("matchmaking sent a challenge to Slowpoke", at: 15.5)
        // Two sends to one player inside the window, then two companions:
        // each companion has two candidates, so neither is paired.
        lines.challenge("challenge sent to Twice", fields: Self.sendFields(id: "Twice001"), at: 20)
        lines.challenge("challenge sent to twice", fields: Self.sendFields(id: "Twice002"), at: 21)
        lines.challenge("matchmaking sent a challenge to TWICE", at: 21.5)
        lines.challenge("matchmaking sent a challenge to Twice", at: 22)
        let reconstruction = try build(lines)

        XCTAssertEqual(try row(reconstruction, challengeID: "Slow0001").sender, .attributed(sender: .byOperator, confidence: .inferredFromAbsence))
        XCTAssertEqual(try row(reconstruction, challengeID: "Twice001").sender, .ambiguousCompanion)
        XCTAssertEqual(try row(reconstruction, challengeID: "Twice002").sender, .ambiguousCompanion)
        XCTAssertNil(try row(reconstruction, challengeID: "Twice001").originConfidence)
        XCTAssertEqual(reconstruction.unexplainedLineCount(.companionWithoutSend), 2)
        XCTAssertEqual(reconstruction.unexplainedLineCount(.companionAmbiguous), 2)
        XCTAssertEqual(reconstruction.counts.companionsPaired, 0)
        XCTAssertEqual(reconstruction.unexplainedLines.map(\.line.line), [1, 3, 6, 7])
    }

    func testPickLineEvidenceWithinSixtySecondsBeforeTheSend() throws {
        var lines = ProtocolLines()
        lines.challenge(Self.pick("EdwardKillick"), at: 0)
        lines.challenge("challenge sent to edwardkillick", fields: Self.sendFields(id: "Picked01"), at: 59)
        lines.challenge("matchmaking sent a challenge to EdwardKillick", at: 59.01)
        lines.challenge(Self.pick("LatePick"), at: 100)
        lines.challenge("challenge sent to LatePick", fields: Self.sendFields(id: "TooLate1"), at: 161)
        lines.challenge("challenge sent to NoPick", fields: Self.sendFields(id: "NoPick01"), at: 200)
        let reconstruction = try build(lines)

        XCTAssertEqual(try row(reconstruction, challengeID: "Picked01").pickLineCheck, .found(.init(file: "events-20261005.jsonl", line: 1)))
        XCTAssertEqual(try row(reconstruction, challengeID: "TooLate1").pickLineCheck, .notFound)
        XCTAssertEqual(try row(reconstruction, challengeID: "NoPick01").pickLineCheck, .notFound)
        // A pick line is not evidence of a row: several sends could share it.
        XCTAssertFalse(try row(reconstruction, challengeID: "Picked01").evidence.contains(.init(file: "events-20261005.jsonl", line: 1)))
    }

    func testAnEchoWithNoSendLineHasNoSender() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "EchoOnly", to: "maia1"), at: 1)
        lines.stream(Self.gameStartEvent(id: "EchoOnly"), at: 2)
        let reconstruction = try build(lines)
        let echo = try row(reconstruction, challengeID: "EchoOnly")
        XCTAssertEqual(echo.sender, .noSendLine)
        XCTAssertNil(echo.pickLineCheck)
        XCTAssertEqual(echo.state, .challenge(.accepted(gameStarted: true)))
        XCTAssertEqual(reconstruction.counts.outgoingChallengesWithoutSendLine, 1)
        XCTAssertEqual(LichessBotReconstructedChallengeLookup(reconstruction).gameOrigin(gameID: "EchoOnly"), .outgoing(.noSendLine))
    }

    // MARK: - Not-created attempts

    func testNotCreatedAttemptsPairWithEachFailureCompanionOrAreInferredOperatorSends() throws {
        var lines = ProtocolLines()
        lines.challenge(Self.refusedOutcome("maia5"), fields: ["credits_day": "1/200", "credits_minute": "1/25"], at: 1)
        lines.challenge("matchmaking send to Maia5 failed: Maia5 is at Lichess's bot-vs-bot daily limit until 6:00 PM", at: 1.01)
        lines.challenge("challenge outcome: offlinebot offline", at: 10)
        lines.challenge("matchmaking send to OfflineBot stopped: the request gate is closed", at: 10.01)
        lines.challenge("challenge outcome: queued1 offline", at: 20)
        lines.challenge("challenge queue: skipped Queued1: Queued1 is offline", at: 20.01)
        lines.challenge("challenge outcome: queued2 offline", at: 30)
        lines.challenge("challenge queue: dropped QUEUED2: no such player", at: 30.01)
        lines.challenge("challenge outcome: queued3 offline", at: 40)
        lines.challenge("challenge queue: stopped at Queued3, which waits again: rate limited", at: 40.01)
        lines.challenge("challenge outcome: resent1 offline", at: 50)
        lines.challenge("matchmaking: casual resend to Resent1 not sent: Resent1 is offline", at: 50.01)
        lines.challenge("challenge outcome: byhand offline", at: 60)
        let reconstruction = try build(lines)

        let rows = attempts(reconstruction)
        XCTAssertEqual(rows.map(\.opponent?.id), ["maia5", "offlinebot", "queued1", "queued2", "queued3", "resent1", "byhand"])
        XCTAssertEqual(rows.map(\.sender), [
            .attributed(sender: .matchmaking, confidence: .paired),
            .attributed(sender: .matchmaking, confidence: .paired),
            .attributed(sender: .challengeQueue, confidence: .paired),
            .attributed(sender: .challengeQueue, confidence: .paired),
            .attributed(sender: .challengeQueue, confidence: .paired),
            .attributed(sender: .matchmakingCasualResend, confidence: .paired),
            .attributed(sender: .byOperator, confidence: .inferredFromAbsence),
        ])
        XCTAssertEqual(rows[0].state, .notCreated(.refused(LichessBotChallengeRefusal(
            kind: .botDailyGameLimit, httpStatus: 400,
            text: "maia5 played 100 games against other bots today, please wait until 2026-10-05T18:00:00.000Z to challenge them."
        ))))
        XCTAssertEqual(rows[1].state, .notCreated(.opponentOffline))
        XCTAssertEqual(rows.map(\.direction), Array(repeating: .outgoing, count: 7))
        XCTAssertEqual(rows[0].key, .notCreatedAttempt(.init(file: "events-20261005.jsonl", line: 1)))
        XCTAssertEqual(rows[0].evidence.map(\.line), [1, 2])
        XCTAssertEqual(reconstruction.counts.notCreatedRows, 7)
        XCTAssertEqual(reconstruction.counts.notCreatedFromOutcomeLines, 7)
        XCTAssertEqual(reconstruction.counts.failureCompanionsPaired, 6)
        XCTAssertTrue(reconstruction.unexplainedLines.isEmpty)
    }

    func testOutcomeLineAndBotLimitLineAreOneRefusalAndABotLimitLineAloneIsAnAttempt() throws {
        var lines = ProtocolLines()
        // Before outcome lines existed: the bot-limit line alone.
        lines.challenge(Self.pick("OldBot"), at: 0)
        lines.challenge(Self.botLimit("oldbot"), at: 1)
        lines.challenge("matchmaking send to OldBot failed: OldBot is at Lichess's bot-vs-bot daily limit until 6:00 PM", at: 1.01)
        // Since then: the bot-limit line, then the outcome line, for one
        // refusal.
        lines.challenge(Self.botLimit("newbot"), at: 10)
        lines.challenge(Self.refusedOutcome("newbot"), at: 10.001)
        let reconstruction = try build(lines)

        let rows = attempts(reconstruction)
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows[0].state, .notCreated(.botGameLimit(gamesPlayed: 100, untilAsLogged: "Oct 5, 2026 at 6:00:00\u{202F}PM")))
        XCTAssertEqual(rows[0].sender, .attributed(sender: .matchmaking, confidence: .paired))
        XCTAssertEqual(rows[0].pickLineCheck, .found(.init(file: "events-20261005.jsonl", line: 1)))
        XCTAssertEqual(rows[0].evidence.map(\.line), [2, 3])
        guard case .notCreated(.refused(let refusal)) = rows[1].state else {
            return XCTFail("expected a refusal, got \(rows[1].state)")
        }
        XCTAssertEqual(refusal.kind, .botDailyGameLimit)
        XCTAssertEqual(rows[1].sender, .attributed(sender: .byOperator, confidence: .inferredFromAbsence))
        XCTAssertEqual(rows[1].pickLineCheck, .notFound)
        XCTAssertEqual(rows[1].evidence.map(\.line), [4, 5])
        XCTAssertEqual(rows[1].key, .notCreatedAttempt(.init(file: "events-20261005.jsonl", line: 5)))
        XCTAssertEqual(reconstruction.counts.notCreatedFromBotLimitLines, 1)
        XCTAssertEqual(reconstruction.counts.notCreatedFromOutcomeLines, 1)
        XCTAssertEqual(reconstruction.counts.botLimitLinesBesideOutcomeLines, 1)
    }

    func testAFailureCompanionWithNoAttemptIsCountedNotGuessed() throws {
        var lines = ProtocolLines()
        // A send that failed before reaching Lichess has no outcome line.
        lines.challenge("matchmaking send to Ghost failed: Ghost: the bot is not online", at: 1)
        let reconstruction = try build(lines)
        XCTAssertTrue(reconstruction.rows.isEmpty)
        XCTAssertEqual(reconstruction.unexplainedLineCount(.failureCompanionWithoutAttempt), 1)
    }

    // MARK: - States

    func testAnswersAndWithdrawalsGiveTheLedgersStates() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "Declined", to: "Picky"), at: 1)
        lines.challenge("challenge sent to Picky", fields: Self.sendFields(id: "Declined"), at: 1.1)
        lines.stream(Self.declinedEvent(id: "Declined", key: "later"), at: 2)

        lines.stream(Self.outgoingEvent(id: "Canceled", to: "Quiet"), at: 10)
        lines.challenge("challenge sent to Quiet", fields: Self.sendFields(id: "Canceled"), at: 10.1)
        lines.stream(Self.canceledEvent(id: "Canceled"), at: 11)

        lines.stream(Self.outgoingEvent(id: "GoingOff", to: "Slow"), at: 20)
        lines.challenge("challenge sent to Slow", fields: Self.sendFields(id: "GoingOff"), at: 20.1)
        lines.challenge("withdrew challenge GoingOff on going offline", at: 21)

        lines.stream(Self.outgoingEvent(id: "TimedOut", to: "Cizme"), at: 30)
        lines.challenge("challenge sent to Cizme", fields: Self.sendFields(id: "TimedOut"), at: 30.1)
        lines.challenge("withdrawing unanswered challenge to cizme after 180 s", at: 210)
        lines.stream(Self.canceledEvent(id: "TimedOut"), at: 210.5)

        // A timeout whose cancel lost the race with the acceptance.
        lines.stream(Self.outgoingEvent(id: "LateAcc1", to: "Lamprook"), at: 300)
        lines.challenge("challenge sent to Lamprook", fields: Self.sendFields(id: "LateAcc1"), at: 300.1)
        lines.challenge("withdrawing unanswered challenge to Lamprook after 999 s", at: 1299)
        lines.stream(Self.gameStartEvent(id: "LateAcc1"), at: 1299.5)
        let reconstruction = try build(lines)

        XCTAssertEqual(try row(reconstruction, challengeID: "Declined").state, .challenge(.declined(.known(.later))))
        XCTAssertEqual(try row(reconstruction, challengeID: "Canceled").state, .challenge(.canceledOnLichessWithoutRecordedWithdrawal))
        XCTAssertEqual(try row(reconstruction, challengeID: "GoingOff").state, .challenge(.withdrawn(.goingOffline, .confirmed)))
        XCTAssertEqual(try row(reconstruction, challengeID: "TimedOut").state, .challenge(.withdrawn(.unansweredTimeout(seconds: 180), .confirmed)))
        let late = try row(reconstruction, challengeID: "LateAcc1")
        XCTAssertEqual(late.state, .challenge(.accepted(gameStarted: true)))
        XCTAssertEqual(late.notes, [.withdrawalAttempted(.unansweredTimeout(seconds: 999), nil)])
        XCTAssertTrue(reconstruction.unexplainedLines.isEmpty)
    }

    func testTimeoutWithdrawalNeedsExactlyOneOpenChallengeToThatPlayer() throws {
        var lines = ProtocolLines()
        // Answered before the withdrawal line: not open any more.
        lines.stream(Self.outgoingEvent(id: "Answered", to: "Alone"), at: 1)
        lines.stream(Self.declinedEvent(id: "Answered", key: "generic"), at: 2)
        lines.challenge("withdrawing unanswered challenge to Alone after 180 s", at: 200)
        // Two open challenges to one player.
        lines.stream(Self.outgoingEvent(id: "TwoOpen1", to: "Busy"), at: 300)
        lines.stream(Self.outgoingEvent(id: "TwoOpen2", to: "Busy"), at: 301)
        lines.challenge("withdrawing unanswered challenge to Busy after 180 s", at: 500)
        let reconstruction = try build(lines)

        XCTAssertEqual(reconstruction.unexplainedLineCount(.timeoutWithdrawalWithoutOpenChallenge), 1)
        XCTAssertEqual(reconstruction.unexplainedLineCount(.timeoutWithdrawalAmbiguous), 1)
        XCTAssertEqual(try row(reconstruction, challengeID: "TwoOpen1").state, .challenge(.open))
        XCTAssertEqual(try row(reconstruction, challengeID: "TwoOpen2").state, .challenge(.open))
    }

    func testIncomingDecisionsAndOwnEchoDecisionsSkipped() throws {
        var lines = ProtocolLines()
        lines.stream(Self.incomingEvent(id: "IncAcc01", from: "Friend"), at: 1)
        lines.challenge("friend: accept", fields: ["challenge": "IncAcc01"], at: 1.1)
        lines.stream(Self.gameStartEvent(id: "IncAcc01"), at: 2)

        lines.stream(Self.incomingEvent(id: "IncDec01", from: "Stranger"), at: 10)
        lines.challenge("stranger: decline (timeControl): blitz only", fields: ["challenge": "IncDec01"], at: 10.1)
        lines.stream(Self.declinedEvent(id: "IncDec01", key: "timeControl"), at: 10.2)

        lines.stream(Self.incomingEvent(id: "IncIgn01", from: "Lurker"), at: 20)
        lines.challenge("lurker: ignore: not now", fields: ["challenge": "IncIgn01"], at: 20.1)
        lines.stream(Self.canceledEvent(id: "IncIgn01"), at: 25)

        // Declined by DCM, accepted by hand on lichess.org.
        lines.stream(Self.incomingEvent(id: "IncHand1", from: "Owner"), at: 30)
        lines.challenge("owner: decline (casual): rated only", fields: ["challenge": "IncHand1"], at: 30.1)
        lines.stream(Self.gameStartEvent(id: "IncHand1"), at: 40)

        // Older builds decided on their own echo.
        lines.stream(Self.outgoingEvent(id: "OwnEcho1", to: "maia1"), at: 50)
        lines.challenge("drewschessmachine: accept", fields: ["challenge": "OwnEcho1"], at: 50.1)
        lines.challenge("DrewsChessMachine: ignore: our own outgoing challenge", fields: ["challenge": "OwnEcho1"], at: 50.2)
        lines.challenge("challenge sent to maia1", fields: Self.sendFields(id: "OwnEcho1"), at: 50.3)
        let reconstruction = try build(lines)

        let accepted = try row(reconstruction, challengeID: "IncAcc01")
        XCTAssertEqual(accepted.state, .challenge(.accepted(gameStarted: true)))
        XCTAssertEqual(accepted.decision, .accept)
        XCTAssertEqual(accepted.notes, [])
        let declined = try row(reconstruction, challengeID: "IncDec01")
        XCTAssertEqual(declined.decision, .decline(reason: .timeControl, rule: "blitz only"))
        XCTAssertEqual(declined.state, .challenge(.declined(.known(.timeControl))))
        let ignored = try row(reconstruction, challengeID: "IncIgn01")
        XCTAssertEqual(ignored.decision, .ignore(rule: "not now"))
        XCTAssertEqual(ignored.state, .challenge(.canceledByChallenger))
        let byHand = try row(reconstruction, challengeID: "IncHand1")
        XCTAssertEqual(byHand.state, .challenge(.accepted(gameStarted: true)))
        XCTAssertEqual(byHand.notes, [.acceptedOutsideDCM(dcmDecided: .decline(reason: .casual, rule: "rated only"))])

        let ownEcho = try row(reconstruction, challengeID: "OwnEcho1")
        XCTAssertEqual(ownEcho.direction, .outgoing)
        XCTAssertNil(ownEcho.decision)
        XCTAssertEqual(reconstruction.counts.skippedOwnEchoAcceptLines, 1)
        XCTAssertEqual(reconstruction.counts.skippedOwnEchoOtherDecisionLines, 1)
        XCTAssertTrue(reconstruction.unexplainedLines.isEmpty)
    }

    func testAnAcceptanceLineWithoutAGameStartIsAcceptedWithoutAStartedGame() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "AccLine1", to: "maia1"), at: 1)
        lines.challenge("challenge sent to maia1", fields: Self.sendFields(id: "AccLine1"), at: 1.1)
        lines.challenge("outgoing challenge accepted; game AccLine1", fields: ["challenge": "AccLine1"], at: 2)
        let reconstruction = try build(lines)
        XCTAssertEqual(try row(reconstruction, challengeID: "AccLine1").state, .challenge(.accepted(gameStarted: false)))
    }

    func testLinesNamingNoChallengeAndUnparsableFormsAreListed() throws {
        var lines = ProtocolLines()
        lines.stream(Self.gameStartEvent(id: "Tourney1"), at: 1)
        lines.challenge("withdrew challenge Unknown1 on going offline", at: 2)
        lines.challenge("challenge sent to NoFields", at: 3)
        lines.challenge("stranger: decline (notAReason): x", fields: ["challenge": "Whatever"], at: 4)
        lines.stream("{\"type\":\"challenge\",\"challenge\":{\"id\":\"Broken\"}}", at: 5)
        let reconstruction = try build(lines)
        XCTAssertTrue(reconstruction.rows.isEmpty)
        XCTAssertEqual(reconstruction.unexplainedLines.map(\.reason), [
            .namesNoMatchingChallenge, .namesNoMatchingChallenge, .unparsableDetails, .unparsableDetails, .undecodableStreamEvent,
        ])
        XCTAssertEqual(LichessBotReconstructedChallengeLookup(reconstruction).gameOrigin(gameID: "Tourney1"), .noReconstructedChallenge)
    }

    func testUndecodableAndUnterminatedLinesAreRecordedInTheInputs() throws {
        var lines = ProtocolLines()
        lines.challenge("challenge outcome: byhand offline", at: 1)
        var data = try lines.data()
        data.append(Data("not json\n".utf8))
        data.append(Data("{\"at\":\"2026-10".utf8))
        let reconstruction = LichessBotChallengeReconstruction.build(
            from: [LichessBotProtocolDayFile(name: "events-20261005.jsonl", data: data)], ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(reconstruction.inputs.count, 1)
        XCTAssertEqual(reconstruction.inputs[0].undecodableLines, [2])
        XCTAssertEqual(reconstruction.inputs[0].droppedTrailingByteCount, 14)
        XCTAssertEqual(reconstruction.inputs[0].byteCount, data.count)
        XCTAssertEqual(reconstruction.inputs[0].sha256.count, 64)
        XCTAssertEqual(reconstruction.rows.count, 1)
    }

    // MARK: - Cutoff

    func testCutoffIsTheLiveLogsFirstEntryNotItsDay() throws {
        var day = ProtocolLines(name: "events-20261005.jsonl")
        day.challenge("challenge outcome: before offline", at: 100)
        day.challenge("challenge outcome: after offline", at: 300)
        var next = ProtocolLines(name: "events-20261006.jsonl")
        next.add(.challenge, "challenge outcome: nextday offline", at: 86_400 + 10)
        let cutoff = F.at(200)
        let reconstruction = try build(day, cutoff: cutoff)
        XCTAssertEqual(attempts(reconstruction).map(\.opponent?.id), ["before"])
        XCTAssertEqual(reconstruction.counts.entriesAtOrAfterCutoff, 1)
        XCTAssertEqual(reconstruction.liveLogFirstEntryAt, cutoff)

        // Only day files on or before the cutoff's UTC day are inputs.
        let names = ["events-20261004.jsonl", "events-20261006.jsonl", "events-20261005.jsonl", "events-2026105.jsonl", "notes.txt"]
        XCTAssertEqual(LichessBotChallengeReconstruction.inputFileNames(from: names, liveLogFirstEntryAt: cutoff),
                       ["events-20261004.jsonl", "events-20261005.jsonl"])
        XCTAssertEqual(LichessBotChallengeReconstruction.inputFileNames(from: names, liveLogFirstEntryAt: nil),
                       ["events-20261004.jsonl", "events-20261005.jsonl", "events-20261006.jsonl"])
        // Without a cutoff, every entry of every file.
        XCTAssertEqual(attempts(try build(day, next)).map(\.opponent?.id), ["before", "after", "nextday"])
    }

    // MARK: - Determinism and the game join

    func testSameInputsGiveIdenticalBytesWhateverTheFileOrder() throws {
        var first = ProtocolLines(name: "events-20261005.jsonl")
        first.stream(Self.outgoingEvent(id: "Out00001", to: "maia1"), at: 1)
        first.challenge("challenge sent to maia1", fields: Self.sendFields(id: "Out00001"), at: 2)
        first.challenge("matchmaking sent a challenge to maia1", at: 2.1)
        first.stream(Self.gameStartEvent(id: "Out00001"), at: 3)
        var second = ProtocolLines(name: "events-20261006.jsonl")
        second.stream(Self.incomingEvent(id: "Inc00001", from: "Friend"), at: 86_401)
        second.challenge("friend: accept", fields: ["challenge": "Inc00001"], at: 86_402)
        second.challenge("challenge outcome: byhand offline", at: 86_403)

        let one = try build(first, second).encoded()
        let two = LichessBotChallengeReconstruction.build(from: [try second.file(), try first.file()], ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(one, try two.encoded())
        // A decoded file encodes back to the same bytes.
        XCTAssertEqual(try LichessBotChallengeReconstruction.decode(one).encoded(), one)
        XCTAssertFalse(String(decoding: one, as: UTF8.self).contains("generatedAt"))
    }

    func testGamesJoinTheirChallengesById() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "MatchMk1", to: "maia1"), at: 1)
        lines.challenge("challenge sent to maia1", fields: Self.sendFields(id: "MatchMk1"), at: 1.1)
        lines.challenge("matchmaking sent a challenge to maia1", at: 1.2)
        lines.stream(Self.outgoingEvent(id: "Oper0001", to: "maia2"), at: 2)
        lines.challenge("challenge sent to maia2", fields: Self.sendFields(id: "Oper0001"), at: 2.1)
        lines.stream(Self.incomingEvent(id: "Inc00001", from: "Friend"), at: 3)
        let lookup = LichessBotReconstructedChallengeLookup(try build(lines))

        XCTAssertEqual(lookup.gameOrigin(gameID: "MatchMk1"), .outgoing(.attributed(sender: .matchmaking, confidence: .paired)))
        XCTAssertEqual(lookup.gameOrigin(gameID: "matchmk1"), .noReconstructedChallenge, "ids are case-sensitive")
        let counts = lookup.gameCounts(gameIDs: ["MatchMk1", "Oper0001", "Inc00001", "Nowhere1"])
        XCTAssertEqual(counts[.matchmaking], 1)
        XCTAssertEqual(counts[.operatorInferred], 1)
        XCTAssertEqual(counts[.incoming], 1)
        XCTAssertEqual(counts[.unknown], 1)
        XCTAssertEqual(counts[.challengeQueue], 0)
        XCTAssertEqual(counts.total, 4)
    }

    // MARK: - The store

    private var directory: LichessBotDataDirectory {
        LichessBotDataDirectory(root: tempRoot)
    }

    private func writeProtocolFile(_ lines: ProtocolLines) throws {
        try lines.data().write(to: directory.protocolDirectory.appendingPathComponent(lines.name), options: .withoutOverwriting)
    }

    private func appendToProtocolFile(_ lines: ProtocolLines) throws {
        let handle = try FileHandle(forWritingTo: directory.protocolDirectory.appendingPathComponent(lines.name))
        try handle.seekToEnd()
        try handle.write(contentsOf: try lines.data())
        try handle.close()
    }

    private func storedFileState() throws -> (identity: FileSafety.FileIdentity, modified: Date, bytes: Data) {
        let url = directory.reconstructedChallengesURL
        let item = try XCTUnwrap(try FileSafety.existingItem(at: url))
        let modified = try XCTUnwrap(try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)
        return (item.identity, modified, try Data(contentsOf: url))
    }

    private func sampleDay() -> ProtocolLines {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "Out00001", to: "maia1"), at: 1)
        lines.challenge("challenge sent to maia1", fields: Self.sendFields(id: "Out00001"), at: 2)
        lines.challenge("challenge outcome: byhand offline", at: 3)
        return lines
    }

    func testStoreWritesOnceAndAnUnchangedRerunWritesNothing() throws {
        try directory.createDirectories()
        try writeProtocolFile(sampleDay())

        let first = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(first.outcome, .written)
        XCTAssertEqual(first.reconstruction.rows.count, 2)
        let before = try storedFileState()
        XCTAssertEqual(before.bytes, try first.reconstruction.encoded())

        // Make a rewrite visible in the modification time.
        Thread.sleep(forTimeInterval: 1.1)
        let second = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(second.outcome, .unchanged)
        XCTAssertEqual(try second.reconstruction.encoded(), try first.reconstruction.encoded())
        let after = try storedFileState()
        XCTAssertEqual(after.identity, before.identity)
        XCTAssertEqual(after.modified, before.modified)
        XCTAssertEqual(after.bytes, before.bytes)
        // The store writes only the reconstructed file.
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.challengesDirectory.path), ["reconstructed-from-protocol.json"])
    }

    func testStoreRegeneratesWhenAnInputGrows() throws {
        try directory.createDirectories()
        try writeProtocolFile(sampleDay())
        _ = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)

        var more = ProtocolLines()
        more.challenge("challenge outcome: another offline", at: 10)
        try appendToProtocolFile(more)
        let grown = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(grown.outcome, .written)
        XCTAssertEqual(grown.reconstruction.rows.count, 3)
        XCTAssertEqual(try storedFileState().bytes, try grown.reconstruction.encoded())
    }

    func testStoreRegeneratesForAnotherAccountOrCutoff() throws {
        try directory.createDirectories()
        try writeProtocolFile(sampleDay())
        let original = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil).outcome, .unchanged)

        let otherAccount = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: "maia1", liveLogFirstEntryAt: nil)
        XCTAssertEqual(otherAccount.outcome, .written)
        XCTAssertEqual(otherAccount.reconstruction.ourAccountID, "maia1")
        XCTAssertEqual(try row(otherAccount.reconstruction, challengeID: "Out00001").direction, .incoming)

        let cutoff = F.at(2.5)
        let withCutoff = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: cutoff)
        XCTAssertEqual(withCutoff.outcome, .written)
        XCTAssertEqual(withCutoff.reconstruction.rows.count, 1)
        XCTAssertEqual(withCutoff.reconstruction.liveLogFirstEntryAt, cutoff)
        XCTAssertEqual(try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: cutoff).outcome, .unchanged)

        let back = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(back.outcome, .written)
        XCTAssertEqual(back.reconstruction, original.reconstruction)
    }

    func testStoreRebuildsAnUndecodableFileAndRefusesAMissingAccount() throws {
        try directory.createDirectories()
        try writeProtocolFile(sampleDay())
        try Data("{\"algorithmVersion\":".utf8).write(to: directory.reconstructedChallengesURL, options: .withoutOverwriting)
        let rebuilt = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        XCTAssertEqual(rebuilt.outcome, .written)
        XCTAssertNotNil(rebuilt.undecodableStoredFile)
        XCTAssertEqual(try storedFileState().bytes, try rebuilt.reconstruction.encoded())

        XCTAssertThrowsError(try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: "", liveLogFirstEntryAt: nil)) { error in
            XCTAssertEqual(error as? LichessBotChallengeReconstructionError, .noAccountID)
        }
    }

    /// The file appears between the read (nothing there) and the publish:
    /// another instance won the create race.
    func testCreateRaceWithIdenticalBytesIsUnchangedAndWithOtherBytesIsReplaced() throws {
        try directory.createDirectories()
        let url = directory.reconstructedChallengesURL
        let bytes = try LichessBotChallengeReconstruction.build(from: [try sampleDay().file()], ourAccountID: Self.us, liveLogFirstEntryAt: nil).encoded()

        try bytes.write(to: url, options: .withoutOverwriting)
        let before = try storedFileState()
        XCTAssertEqual(try LichessBotChallengeReconstructionStore.writeIfChanged(bytes, at: url, storedBytes: nil), .unchanged)
        let after = try storedFileState()
        XCTAssertEqual(after.identity, before.identity)
        XCTAssertEqual(after.bytes, bytes)

        let older = Data("{}\n".utf8)
        try FileSafety.replaceRegularFile(older, at: url, expectedIdentity: nil)
        XCTAssertEqual(try LichessBotChallengeReconstructionStore.writeIfChanged(bytes, at: url, storedBytes: nil), .written)
        XCTAssertEqual(try Data(contentsOf: url), bytes)
    }

    func testStoreRefusesASymbolicLinkAmongTheInputs() throws {
        try directory.createDirectories()
        let target = tempRoot.appendingPathComponent("elsewhere.jsonl")
        try sampleDay().data().write(to: target, options: .withoutOverwriting)
        try FileManager.default.createSymbolicLink(at: directory.protocolDirectory.appendingPathComponent("events-20261005.jsonl"),
                                                   withDestinationURL: target)
        XCTAssertThrowsError(try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: Self.us, liveLogFirstEntryAt: nil)) { error in
            guard case FileSafetyError.notARegularFile = error else {
                return XCTFail("expected notARegularFile, got \(error)")
            }
        }
        XCTAssertNil(try FileSafety.existingItem(at: directory.reconstructedChallengesURL))
    }

    func testSummaryLine() throws {
        var lines = ProtocolLines()
        lines.stream(Self.outgoingEvent(id: "MatchMk1", to: "maia1"), at: 1)
        lines.challenge("challenge sent to maia1", fields: Self.sendFields(id: "MatchMk1"), at: 1.1)
        lines.challenge("matchmaking sent a challenge to maia1", at: 1.2)
        lines.stream(Self.incomingEvent(id: "Inc00001", from: "Friend"), at: 3)
        lines.challenge("challenge outcome: byhand offline", at: 4)
        let reconstruction = try build(lines)
        let games = LichessBotReconstructedChallengeLookup(reconstruction).gameCounts(gameIDs: ["MatchMk1", "Inc00001", "Nowhere1"])
        let line = LichessBotChallengeReconstructionStore.summaryLine(
            .init(outcome: .unchanged, reconstruction: reconstruction, undecodableStoredFile: nil), games: games)
        XCTAssertEqual(line, "[LICHESS-BOT] challenge history reconstructed: inputs=1 files (0.00 MB) rows=3 (outgoing created 1, not created 1, incoming 1); "
            + "games: incoming 1, matchmaking 1, queue 0, casual resend 0, operator (inferred) 0, outgoing sender not determined 0, unknown 1; unchanged")
    }
}
