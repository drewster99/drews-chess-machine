//
//  LichessBotChallengeLogFixtures.swift
//  DrewsChessMachineTests
//
//  Shared values for the challenge-log tests: a fixed clock, parties,
//  snapshots, requests and entries.
//

import Foundation
@testable import DrewsChessMachine

enum LichessBotChallengeLogFixtures {
    /// 2026-10-05T12:00:00Z.
    static let start = Date(timeIntervalSince1970: 1_791_201_600)

    static func at(_ seconds: TimeInterval) -> Date {
        start.addingTimeInterval(seconds)
    }

    static let ourAccount = LichessBotChallengeParty(id: "drewschessmachine", name: "DrewsChessMachine", title: "BOT", rating: 1500, provisional: false)
    static let maia = LichessBotChallengeParty(id: "maia1", name: "maia1", title: "BOT", rating: 1400, provisional: nil)

    static func snapshot(id: String = "AbCd1234",
                         challenger: LichessBotChallengeParty = ourAccount,
                         destUser: LichessBotChallengeParty? = maia) -> LichessBotChallengeSnapshot {
        LichessBotChallengeSnapshot(
            id: id, challenger: challenger, destUser: destUser,
            variant: .init(.standard), rated: true, speed: .init(.blitz),
            timeControlType: .init(.clock), limitSeconds: 180, incrementSeconds: 2, daysPerTurn: nil,
            color: .init(.random), finalColor: .init(.white), initialFen: nil, rematchOf: nil
        )
    }

    static let request = LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 180, clockIncrementSeconds: 2, color: .random)

    static let automaticPass = LichessBotChallengeSender.matchmaking(trigger: .automaticPass, fillMode: .everyFreeSlot)

    /// `00000000-0000-0000-0000-0000000000NN` for `number` in 0…255.
    static func attemptID(_ number: UInt8) -> UUID {
        UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, number))
    }

    /// An entry as a fixed build wrote it.
    static func entry(_ event: LichessBotChallengeLogEvent, at seconds: TimeInterval) -> LichessBotChallengeLogEntry {
        LichessBotChallengeLogEntry(schemaVersion: 1, at: at(seconds), build: 2400, gitHash: "0123abc", event: event)
    }

    static func created(_ id: String = "AbCd1234", sender: LichessBotChallengeSender = automaticPass) -> LichessBotChallengeLogEvent {
        .outgoingCreated(challenge: snapshot(id: id), sender: sender, request: request, opponentKind: .bot, creditCost: 1)
    }

    static func received(_ id: String = "InCo5678") -> LichessBotChallengeLogEvent {
        .incomingReceived(challenge: snapshot(id: id, challenger: maia, destUser: ourAccount))
    }
}
