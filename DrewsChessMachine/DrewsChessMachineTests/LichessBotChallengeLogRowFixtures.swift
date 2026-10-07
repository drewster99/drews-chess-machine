//
//  LichessBotChallengeLogRowFixtures.swift
//  DrewsChessMachineTests
//
//  Data for the Challenge Log window's tests: a live ledger holding one row
//  of each kind, and protocol lines (written exactly as the controller
//  writes them) that rebuild into one reconstructed row of each kind, with
//  one challenge both hold.
//

import Foundation
@testable import DrewsChessMachine

enum LichessBotChallengeLogRowFixtures {
    private typealias F = LichessBotChallengeLogFixtures

    static let ourAccountID = "drewschessmachine"
    /// Pending in the controller: the open live row shows "Waiting".
    static let pendingID = "Pend0001"
    /// Held by both the ledger and the rebuilt history.
    static let sharedID = "AbCd1234"

    // MARK: Live

    /// One live row of each kind, at `F.at(1000)` onward.
    static func ledgerEntries() -> [LichessBotChallengeLogEntry] {
        [
            // Outgoing, accepted, its game seen starting.
            F.entry(F.created(sharedID), at: 1000),
            F.entry(.gameStarted(challengeID: sharedID), at: 1005),
            // Outgoing, still open, pending.
            F.entry(F.created(pendingID, sender: .challengeSheet), at: 1010),
            // Outgoing, declined with a key this build doesn't know.
            F.entry(F.created("Decl0001"), at: 1015),
            F.entry(.declinedOnLichess(challengeID: "Decl0001", reason: .unrecognized("nobot"), text: "no bots"), at: 1016),
            // Not created: the opponent was offline.
            F.entry(.outgoingNotCreated(attemptID: F.attemptID(1), opponentID: "offlinebot", sender: .challengeQueue,
                                        request: F.request, opponentKind: .bot, reason: .opponentOffline, creditCost: 0), at: 1020),
            // Incoming, declined by DCM.
            F.entry(F.received("InCo5678"), at: 1030),
            F.entry(.incomingDecided(challengeID: "InCo5678", decision: .decline(reason: .tooFast, rule: "too fast")), at: 1031),
            // Only Lichess's cancel: no direction.
            F.entry(.canceledOnLichess(challengeID: "Gone0001"), at: 1040),
            // DCM's echo, no send explains it.
            F.entry(.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(id: "Echo0001"), attribution: .notRecorded), at: 1050),
            // Withdrawn by the operator; Lichess said it was already gone.
            F.entry(F.created("Wdr00001", sender: .casualResendOffer), at: 1060),
            F.entry(.withdrawalRequested(challengeID: "Wdr00001", reason: .operatorCancel), at: 1061),
            F.entry(.withdrawalResult(challengeID: "Wdr00001", result: .alreadyGone(message: "not found")), at: 1062),
        ]
    }

    static func ledger() -> LichessBotChallengeLedger {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        ledger.apply(contentsOf: ledgerEntries())
        return ledger
    }

    // MARK: Rebuilt

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
        challengeEvent(id: id, challenger: (ourAccountID, "DrewsChessMachine"), dest: (name.lowercased(), name))
    }

    private static func sendFields(id: String) -> [String: String] {
        ["id": id, "rated": "true", "clock": "180+2", "color": "random"]
    }

    /// The protocol entries, oldest first, at `F.at(1)` … `F.at(70)`.
    static func protocolEntries() -> [LichessBotProtocolEntry] {
        func stream(_ json: String, at seconds: TimeInterval) -> LichessBotProtocolEntry {
            LichessBotProtocolEntry(at: F.at(seconds), kind: .stream, gameID: nil, message: json, fields: ["stream": "event"])
        }
        func challenge(_ message: String, fields: [String: String] = [:], at seconds: TimeInterval) -> LichessBotProtocolEntry {
            LichessBotProtocolEntry(at: F.at(seconds), kind: .challenge, gameID: nil, message: message, fields: fields)
        }
        return [
            // Matchmaking (paired), accepted.
            stream(outgoingEvent(id: "RecMm001", to: "EdwardKillick"), at: 1),
            challenge("challenge sent to EdwardKillick", fields: sendFields(id: "RecMm001"), at: 2),
            challenge("matchmaking sent a challenge to edwardkillick", at: 2.01),
            stream("{\"type\":\"gameStart\",\"game\":{\"fullId\":\"x\",\"gameId\":\"RecMm001\",\"source\":\"friend\"}}", at: 5),
            // The operator (inferred), never answered.
            challenge("challenge sent to Operator1", fields: sendFields(id: "RecOp001"), at: 10),
            // Incoming, accepted by DCM, game started.
            stream(challengeEvent(id: "RecIn001", challenger: ("someone", "Someone"), dest: (ourAccountID, "DrewsChessMachine")), at: 20),
            challenge("Someone: accept", fields: ["challenge": "RecIn001"], at: 21),
            stream("{\"type\":\"gameStart\",\"game\":{\"fullId\":\"y\",\"gameId\":\"RecIn001\",\"source\":\"friend\"}}", at: 22),
            // Not created: offline, no companion (the operator, inferred).
            challenge("challenge outcome: offlinebot offline", at: 30),
            // Outgoing, canceled with no withdrawal line.
            stream(outgoingEvent(id: "RecCx001", to: "Canceler"), at: 40),
            challenge("challenge sent to Canceler", fields: sendFields(id: "RecCx001"), at: 40.5),
            stream("{\"type\":\"challengeCanceled\",\"challenge\":{\"id\":\"RecCx001\",\"status\":\"canceled\"}}", at: 50),
            // The challenge the ledger also holds.
            stream(outgoingEvent(id: sharedID, to: "maia1"), at: 60),
            challenge("challenge sent to maia1", fields: sendFields(id: sharedID), at: 60.5),
        ]
    }

    static func protocolDayData() throws -> Data {
        var data = Data()
        for entry in protocolEntries() {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        return data
    }

    static func reconstruction() throws -> LichessBotChallengeReconstruction {
        let file = LichessBotProtocolDayFile(name: "events-20261005.jsonl", data: try protocolDayData())
        return LichessBotChallengeReconstruction.build(from: [file], ourAccountID: ourAccountID, liveLogFirstEntryAt: nil)
    }
}
