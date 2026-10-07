//
//  LichessBotGameOriginResolverTests.swift
//  DrewsChessMachineTests
//
//  `LichessBotGameOriginResolver` (challenge-log plan §3.5): how each game
//  began, decided from the challenge ledger and the game's `gameStart` —
//  incoming, each outgoing sender, the POST race, echo-only challenges,
//  tournaments, unknown sources kept verbatim, undetermined at session end
//  with the gap the ledger's load status explains, resumed games, and at
//  most one write per game per run.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotGameOriginResolverTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private func ledger(_ events: [LichessBotChallengeLogEvent], loadStatus: LichessBotChallengeLedger.LoadStatus = .complete) -> LichessBotChallengeLedger {
        var ledger = LichessBotChallengeLedger(loadStatus: loadStatus)
        for (index, event) in events.enumerated() {
            ledger.apply(F.entry(event, at: TimeInterval(index)))
        }
        return ledger
    }

    private static func gameStart(_ gameID: String, source: String?, tournamentID: String? = nil) throws -> LichessBotGameEventInfo {
        var fields = [#""gameId":"\#(gameID)""#]
        if let source { fields.append(#""source":"\#(source)""#) }
        if let tournamentID { fields.append(#""tournamentId":"\#(tournamentID)""#) }
        return try JSONDecoder().decode(LichessBotGameEventInfo.self, from: Data(("{" + fields.joined(separator: ",") + "}").utf8))
    }

    func testAnIncomingChallengeIsAcceptedIncomingWhateverDCMDecided() {
        for decision in [LichessBotIncomingDecisionRecord.accept, .decline(reason: .tooFast, rule: "bullet"), .ignore(rule: "budget")] {
            var resolver = LichessBotGameOriginResolver()
            let ledger = ledger([F.received("inco1"), .incomingDecided(challengeID: "inco1", decision: decision)])
            XCTAssertEqual(resolver.sessionStarted(gameID: "inco1", recordedOrigin: nil, ledger: ledger),
                           .decided(.acceptedIncomingChallenge(challengeID: "inco1", challengerID: "maia1")))
        }
    }

    func testEachSenderIsTheOrigin() {
        let senders: [LichessBotChallengeSender] = [
            .challengeSheet, .casualResendOffer, .challengeQueue,
            .matchmaking(trigger: .automaticPass, fillMode: .everyFreeSlot),
            .matchmaking(trigger: .fillOpenSlots, fillMode: .everyFreeSlot),
            .matchmakingCasualResend,
        ]
        for sender in senders {
            var resolver = LichessBotGameOriginResolver()
            XCTAssertEqual(resolver.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: ledger([F.created(sender: sender)])),
                           .decided(.outgoingChallengeAccepted(challengeID: "AbCd1234", sender: sender)))
        }
    }

    func testThePOSTRaceWaitsThenDecidesWhenTheCreatedLineArrives() {
        var resolver = LichessBotGameOriginResolver()
        var ledger = ledger([.gameStarted(challengeID: "AbCd1234")])
        XCTAssertEqual(resolver.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: ledger), .waiting)
        XCTAssertTrue(resolver.isWaiting("AbCd1234"))
        ledger.apply(F.entry(F.created(sender: .challengeQueue), at: 5))
        XCTAssertEqual(resolver.challengeKnown(gameID: "AbCd1234", ledger: ledger),
                       .outgoingChallengeAccepted(challengeID: "AbCd1234", sender: .challengeQueue))
        XCTAssertNil(resolver.challengeKnown(gameID: "AbCd1234", ledger: ledger), "decided once per run")
        XCTAssertNil(resolver.sessionEnded(gameID: "AbCd1234", ledger: ledger))
    }

    func testEchoOnlyChallenges() {
        var attributed = LichessBotGameOriginResolver()
        let unanswered = ledger([.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .unansweredSend(attemptID: F.attemptID(1), sender: .challengeSheet))])
        XCTAssertEqual(attributed.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: unanswered),
                       .decided(.outgoingChallengeAccepted(challengeID: "AbCd1234", sender: .challengeSheet)))
        var unattributed = LichessBotGameOriginResolver()
        let notRecorded = ledger([.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded)])
        XCTAssertEqual(unattributed.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: notRecorded),
                       .decided(.outgoingChallengeSenderNotRecorded(challengeID: "AbCd1234")))
    }

    func testTournamentsAreDecidedFromTheGameStart() throws {
        for source in ["arena", "swiss"] {
            var resolver = LichessBotGameOriginResolver()
            resolver.noteGameStart(try Self.gameStart("t1", source: source, tournamentID: "tour9"))
            XCTAssertEqual(resolver.sessionStarted(gameID: "t1", recordedOrigin: nil, ledger: ledger([])),
                           .decided(.tournament(source: LichessBotOpenValue(raw: source), tournamentID: "tour9")))
        }
    }

    func testAnUnknownSourceIsKeptVerbatimWhenUndetermined() throws {
        var resolver = LichessBotGameOriginResolver()
        resolver.noteGameStart(try Self.gameStart("g1", source: "somethingNew"))
        let ledger = ledger([])
        XCTAssertEqual(resolver.sessionStarted(gameID: "g1", recordedOrigin: nil, ledger: ledger), .waiting)
        XCTAssertEqual(resolver.sessionEnded(gameID: "g1", ledger: ledger),
                       .undetermined(source: LichessBotOpenValue(raw: "somethingNew"), gap: .noChallengeRecord))
    }

    func testAMissOnAnIncompleteLedgerIsChallengeLogIncomplete() throws {
        let statuses: [LichessBotChallengeLedger.LoadStatus?] = [
            .partial(filesLeftOut: [.init(name: "challenges-20261004.jsonl", reason: "corrupt")]),
            .failed(reason: "unreadable"),
            nil,
        ]
        for status in statuses {
            var resolver = LichessBotGameOriginResolver()
            resolver.noteGameStart(try Self.gameStart("g1", source: "friend"))
            let ledger = status.map { ledger([], loadStatus: $0) }
            XCTAssertEqual(resolver.sessionStarted(gameID: "g1", recordedOrigin: nil, ledger: ledger), .waiting)
            XCTAssertEqual(resolver.sessionEnded(gameID: "g1", ledger: ledger),
                           .undetermined(source: LichessBotOpenValue(.friend), gap: .challengeLogIncomplete), "\(String(describing: status))")
        }
    }

    func testAResumedGameWithADeterminedOriginWritesNothing() {
        var resolver = LichessBotGameOriginResolver()
        let recorded = LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "AbCd1234", sender: .challengeQueue)
        XCTAssertEqual(resolver.sessionStarted(gameID: "AbCd1234", recordedOrigin: recorded, ledger: nil), .alreadyRecorded(recorded))
        XCTAssertNil(resolver.sessionEnded(gameID: "AbCd1234", ledger: nil))
    }

    func testAResumedGameWithNoneIsDecidedFromTheLedger() {
        var resolver = LichessBotGameOriginResolver()
        XCTAssertEqual(resolver.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: ledger([F.created()])),
                       .decided(.outgoingChallengeAccepted(challengeID: "AbCd1234", sender: F.automaticPass)))
    }

    func testAResumedUndeterminedOriginIsDecidedLaterOrNotWrittenAgain() throws {
        let undetermined = LichessBotGameOrigin.undetermined(source: LichessBotOpenValue(.friend), gap: .noChallengeRecord)
        var later = LichessBotGameOriginResolver()
        XCTAssertEqual(later.sessionStarted(gameID: "AbCd1234", recordedOrigin: undetermined, ledger: ledger([])), .waiting)
        XCTAssertEqual(later.challengeKnown(gameID: "AbCd1234", ledger: ledger([F.created()])),
                       .outgoingChallengeAccepted(challengeID: "AbCd1234", sender: F.automaticPass))

        var same = LichessBotGameOriginResolver()
        same.noteGameStart(try Self.gameStart("AbCd1234", source: "friend"))
        XCTAssertEqual(same.sessionStarted(gameID: "AbCd1234", recordedOrigin: undetermined, ledger: ledger([])), .waiting)
        XCTAssertNil(same.sessionEnded(gameID: "AbCd1234", ledger: ledger([])), "the journal already ends in that value")
    }

    func testADecidedGameIsNotDecidedAgainBySessionsLaterInTheRun() {
        var resolver = LichessBotGameOriginResolver()
        let ledger = ledger([F.created()])
        guard case .decided(let origin) = resolver.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: ledger) else {
            return XCTFail("expected a decision")
        }
        XCTAssertEqual(resolver.sessionStarted(gameID: "AbCd1234", recordedOrigin: nil, ledger: ledger), .alreadyRecorded(origin))
    }

    func testTokensAreFixedPerKind() {
        XCTAssertEqual(LichessBotGameOrigin.acceptedIncomingChallenge(challengeID: "a", challengerID: "b").token, "incoming")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "a", sender: .challengeSheet).token, "sheet")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "a", sender: .casualResendOffer).token, "casual-resend-offer")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "a", sender: .challengeQueue).token, "queue")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "a", sender: .matchmaking(trigger: .automaticPass, fillMode: .onlyWhenIdle)).token, "matchmaking-auto")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "a", sender: .matchmaking(trigger: .fillOpenSlots, fillMode: .everyFreeSlot)).token, "matchmaking-fill")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "a", sender: .matchmakingCasualResend).token, "matchmaking-casual-resend")
        XCTAssertEqual(LichessBotGameOrigin.outgoingChallengeSenderNotRecorded(challengeID: "a").token, "outgoing-sender-not-recorded")
        XCTAssertEqual(LichessBotGameOrigin.tournament(source: LichessBotOpenValue(.arena), tournamentID: nil).token, "tournament")
        XCTAssertEqual(LichessBotGameOrigin.undetermined(source: nil, gap: .noChallengeRecord).token, "undetermined")
    }

    func testTheGameStartSourceDecodesAsAnOpenValue() throws {
        XCTAssertEqual(try Self.gameStart("g", source: "friend").source?.known, .friend)
        XCTAssertEqual(try Self.gameStart("g", source: "importlive").source?.known, .importLive)
        XCTAssertNil(try Self.gameStart("g", source: "brandNew").source?.known)
        XCTAssertEqual(try Self.gameStart("g", source: "brandNew").source?.raw, "brandNew")
    }
}
