//
//  LichessBotGameOriginStyleTests.swift
//  DrewsChessMachineTests
//
//  The UI's words and glyphs for game origins and challenge-log rows
//  (challenge-log plan §3.9): every origin category has its glyph and labels
//  from `LichessBotGameOriginStyle` (the plan's table), the basis marker and
//  help, the unknown reasons in words (§3.6 step 5), the All Games window's
//  origin filter, sort and summary, and the Challenge Log's explicit wording
//  for every state the plan names.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotGameOriginStyleTests: XCTestCase {

    private typealias Style = LichessBotGameOriginStyle
    private typealias LogStyle = LichessBotChallengeLogStyle

    // MARK: - Origin style

    func testEveryCategoryHasThePlansGlyphAndShortLabel() {
        let expected: [LichessBotGameOriginCategory: (String, String)] = [
            .incoming: ("arrow.down.left", "Incoming"),
            .challengeSheet: ("arrow.up.right", "Sheet"),
            .casualResendOffer: ("arrow.uturn.right", "Resend offer"),
            .challengeQueue: ("list.bullet", "Queue"),
            .matchmaking: ("wand.and.stars", "Matchmaking"),
            .matchmakingCasualResend: ("arrow.uturn.right.circle", "Casual resend"),
            .outgoingSenderNotRecorded: ("arrow.up.right", "Outgoing (sender not recorded)"),
            .tournament: ("trophy", "Tournament"),
            .unknown: ("questionmark", "Unknown"),
        ]
        XCTAssertEqual(Set(expected.keys), Set(LichessBotGameOriginCategory.allCases))
        for category in LichessBotGameOriginCategory.allCases {
            XCTAssertEqual(Style.systemImage(for: category), expected[category]?.0, "\(category)")
            XCTAssertEqual(Style.shortLabel(for: category), expected[category]?.1, "\(category)")
            XCTAssertFalse(Style.longLabel(for: category).isEmpty, "\(category)")
        }
        XCTAssertEqual(Set(LichessBotGameOriginCategory.allCases.map(Style.longLabel(for:))).count, LichessBotGameOriginCategory.allCases.count)
        XCTAssertFalse(LichessBotGameOriginCategory.allCases.map(Style.systemImage(for:)).contains(Style.notYetKnownSystemImage))
    }

    func testThePresentationsLabelsComeFromTheStyle() {
        for category in LichessBotGameOriginCategory.allCases {
            let display = LichessBotGameOriginDisplay(category: category, detail: "detail text", basis: .challengeLog)
            let presentation = Style.presentation(of: display)
            XCTAssertEqual(presentation.systemImage, Style.systemImage(for: category))
            XCTAssertEqual(presentation.shortLabel, Style.shortLabel(for: category))
            XCTAssertEqual(presentation.longLabel, Style.longLabel(for: category))
            XCTAssertEqual(presentation.help, [Style.longLabel(for: category), "detail text", Style.basisText(.challengeLog)].joined(separator: "\n"))
        }
    }

    func testTheBasisMarkerAndColor() {
        let bases: [(LichessBotOriginBasis, marker: String, muted: Bool)] = [
            (.recorded, "", false),
            (.challengeLog, "", false),
            (.reconstructed(.certain), "", false),
            (.reconstructed(.paired), "", false),
            (.reconstructed(.inferredFromAbsence), "≈", true),
            (.unknown(.notRecorded), "", true),
            (.unknown(.playedBeforeOriginsWereRecorded), "", true),
            (.unknown(.gap(.noChallengeRecord)), "", true),
            (.unknown(.gap(.challengeLogIncomplete)), "", true),
        ]
        for (basis, marker, muted) in bases {
            let presentation = Style.presentation(of: LichessBotGameOriginDisplay(category: .challengeSheet, detail: "d", basis: basis))
            XCTAssertEqual(presentation.marker, marker, "\(basis)")
            XCTAssertEqual(presentation.isMuted, muted, "\(basis)")
            XCTAssertEqual(presentation.markedShortLabel, "Sheet" + marker)
            XCTAssertTrue(presentation.help.hasPrefix(Style.longLabel(for: .challengeSheet) + marker + "\nd\n"), presentation.help)
            XCTAssertTrue(presentation.help.hasSuffix(Style.basisText(basis)), presentation.help)
        }
    }

    func testRebuiltHelpGivesTheEvidenceInWords() {
        let inferred = Style.presentation(of: LichessBotGameOriginDisplay(category: .challengeSheet, detail: "d", basis: .reconstructed(.inferredFromAbsence)))
        XCTAssertTrue(inferred.help.contains("Reconstructed from the protocol log"), inferred.help)
        XCTAssertTrue(inferred.help.contains("'challenge sent to' with no matchmaking or queue line beside it"), inferred.help)
        for confidence in LichessBotReconstructionConfidence.allCases {
            XCTAssertTrue(Style.basisText(.reconstructed(confidence)).contains(Style.confidenceText(confidence)))
            XCTAssertTrue(Style.confidenceText(confidence).hasPrefix(Style.confidenceWord(confidence)))
        }
    }

    func testUnknownReasonsInThePlansWords() {
        XCTAssertEqual(Style.unknownReasonText(.playedBeforeOriginsWereRecorded),
                       "Unknown — played before origins were recorded; no challenge with this id in the protocol log.")
        XCTAssertEqual(Style.unknownReasonText(.notRecorded),
                       "Unknown — no origin recorded for this game, and no challenge with this id in the challenge log.")
        XCTAssertTrue(Style.unknownReasonText(.gap(.noChallengeRecord)).hasPrefix("Unknown — "))
        XCTAssertTrue(Style.unknownReasonText(.gap(.challengeLogIncomplete)).hasPrefix("Unknown — "))
        XCTAssertNotEqual(Style.unknownReasonText(.gap(.noChallengeRecord)), Style.unknownReasonText(.gap(.challengeLogIncomplete)))
    }

    func testAnUnknownOriginsLongLabelIsItsReason() {
        let reasons: [LichessBotOriginUnknownReason] = [.playedBeforeOriginsWereRecorded, .notRecorded, .gap(.noChallengeRecord), .gap(.challengeLogIncomplete)]
        for reason in reasons {
            let presentation = Style.presentation(of: LichessBotGameOriginDisplay(category: .unknown, detail: "game abc", basis: .unknown(reason)))
            XCTAssertEqual(presentation.shortLabel, "Unknown")
            XCTAssertEqual(presentation.longLabel, Style.unknownReasonText(reason))
            XCTAssertEqual(presentation.help, Style.unknownReasonText(reason) + "\ngame abc")
            XCTAssertTrue(presentation.isMuted)
            XCTAssertEqual(presentation.marker, "")
        }
    }

    func testNotYetKnownIsItsOwnPresentation() {
        let presentation = Style.presentation(of: nil)
        XCTAssertEqual(presentation.systemImage, Style.notYetKnownSystemImage)
        XCTAssertEqual(presentation.shortLabel, Style.notYetKnownLabel)
        XCTAssertEqual(presentation.help, Style.notYetKnownHelp)
        XCTAssertTrue(presentation.isMuted)
    }

    func testTheSenderCategoryMatchesTheDisplaysMapping() {
        let senders: [LichessBotChallengeSender] = [
            .challengeSheet, .casualResendOffer, .challengeQueue,
            .matchmaking(trigger: .automaticPass, fillMode: .everyFreeSlot), .matchmaking(trigger: .fillOpenSlots, fillMode: .onlyWhenIdle),
            .matchmakingCasualResend,
        ]
        for sender in senders {
            let display = LichessBotGameOriginDisplay.display(.outgoingChallengeAccepted(challengeID: "x", sender: sender), basis: .recorded)
            XCTAssertEqual(LichessBotGameOriginCategory(sender: sender), display.category, "\(sender)")
        }
    }

    func testTheLiveGamesDisplayIsItsRecordedOrigin() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        XCTAssertNil(game.originDisplay)
        XCTAssertTrue(LichessBotLiveView.menuTitle(for: game).hasSuffix(" · " + Style.notYetKnownPickerSuffix))
        let origin = LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "g1", sender: .challengeQueue)
        game.setOrigin(origin)
        XCTAssertEqual(game.originDisplay, LichessBotGameOriginDisplay.display(origin, basis: .recorded))
        XCTAssertTrue(LichessBotLiveView.menuTitle(for: game).hasSuffix(" · Queue"), LichessBotLiveView.menuTitle(for: game))
    }

    // MARK: - All Games window

    private static let gameFullTemplate = #"{"type":"gameFull","id":"GAMEID","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1759000000000,"white":{"id":"drewschessmachine","name":"drewschessmachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#

    private func summary(_ gameID: String) throws -> LichessBotGameSummary {
        var at = Date(timeIntervalSince1970: 1_759_000_000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        let entries: [LichessBotJournalEntry] = [
            .init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: gameID, build: 1, gitHash: "test", resumed: false)),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: Self.gameFullTemplate.replacingOccurrences(of: "GAMEID", with: gameID))),
            .init(at: next(), event: .finished(status: "resign", winner: "white", localDrawCondition: nil)),
        ]
        let record = try LichessBotRecordBuilder.build(
            gameID: gameID, journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: "drewschessmachine", checkedAt: Date()
        )
        return LichessBotGameSummary(record: record)
    }

    func testTheOriginFilterSortAndSummary() throws {
        let summaries = [try summary("g1"), try summary("g2"), try summary("g3"), try summary("g4")]
        let origins: [String: LichessBotGameOriginDisplay] = [
            "g1": LichessBotGameOriginDisplay(category: .matchmaking, detail: "d", basis: .recorded),
            "g2": LichessBotGameOriginDisplay(category: .incoming, detail: "d", basis: .reconstructed(.certain)),
            "g3": LichessBotGameOriginDisplay(category: .matchmaking, detail: "d", basis: .challengeLog),
        ]
        let all = LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .all)
        XCTAssertEqual(all.map(\.id), ["g1", "g2", "g3", "g4"])
        XCTAssertEqual(LichessBotAllGamesRow.summary(filedCount: 4, shown: all, filter: .all),
                       "4 filed games: Incoming 1 · Matchmaking 2 · Not yet known 1")

        let matchmaking = LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .only(.matchmaking))
        XCTAssertEqual(matchmaking.map(\.id), ["g1", "g3"])
        XCTAssertEqual(LichessBotAllGamesRow.summary(filedCount: 4, shown: matchmaking, filter: .only(.matchmaking)),
                       "2 of 4 filed games: Matchmaking 2")
        XCTAssertEqual(LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .only(.tournament)).count, 0)
        XCTAssertEqual(LichessBotAllGamesRow.summary(filedCount: 4, shown: [], filter: .only(.tournament)), "0 of 4 filed games")

        // Sorted by category order; not yet known last.
        let sorted = all.sorted(using: KeyPathComparator(\LichessBotAllGamesRow.originSortKey))
        XCTAssertEqual(sorted.map(\.originSortKey), [0, 4, 4, Int.max])
        XCTAssertEqual(sorted.first?.id, "g2")
        XCTAssertEqual(sorted.last?.id, "g4")
    }

    func testDisplayOrderIsTheCategorysPlaceInAllCases() {
        for (index, category) in LichessBotGameOriginCategory.allCases.enumerated() {
            XCTAssertEqual(category.displayOrder, index, "\(category)")
        }
    }

    // MARK: - Challenge Log wording

    func testOpenStatesAreSaidExplicitly() {
        XCTAssertEqual(LogStyle.stateText(.challenge(.open, isPending: true)), "Waiting")
        XCTAssertEqual(LogStyle.stateText(.challenge(.open, isPending: false)), "No answer recorded")
    }

    func testCancelAndWithdrawalStatesInThePlansWords() {
        XCTAssertEqual(LogStyle.stateText(.challenge(.canceledOnLichessWithoutRecordedWithdrawal, isPending: false)), "Withdrawn (reason not recorded)")
        XCTAssertEqual(LogStyle.stateText(.challenge(.withdrawn(.operatorCancel, nil), isPending: false)), "Withdrawn (by the operator), result not recorded")
        XCTAssertEqual(LogStyle.stateText(.challenge(.withdrawn(.unansweredTimeout(seconds: 60), .alreadyGone(message: nil)), isPending: false)), "No longer on Lichess")
        XCTAssertEqual(LogStyle.stateText(.challenge(.withdrawn(.goingOffline, .confirmed), isPending: false)), "Withdrawn (going offline)")
        XCTAssertEqual(LogStyle.stateText(.challenge(.canceledOnLichessDirectionNotRecorded, isPending: false)), "Canceled on Lichess (direction not recorded)")
        XCTAssertEqual(LogStyle.stateText(.challenge(.canceledByChallenger, isPending: false)), "Canceled by the challenger")
    }

    func testEveryStateHasTextAndHelp() {
        let refusal = LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: "limit")
        let states: [LichessBotChallengeLogRow.State] = [
            .challenge(.open, isPending: true), .challenge(.open, isPending: false),
            .challenge(.accepted(gameStarted: true), isPending: false), .challenge(.accepted(gameStarted: false), isPending: false),
            .challenge(.declined(.known(.tooFast)), isPending: false), .challenge(.declined(.unrecognized("nobot")), isPending: false),
            .challenge(.declined(.unstated), isPending: false),
            .challenge(.canceledByChallenger, isPending: false),
            .challenge(.withdrawn(.wentOfflineWhileSending, .failed(error: "e")), isPending: false),
            .challenge(.withdrawn(.goingOffline, .abandonedAtShutdown), isPending: false),
            .challenge(.notCreated(.opponentOffline), isPending: false), .challenge(.notCreated(.refused(refusal)), isPending: false),
            .challenge(.notCreated(.noAnswer(error: "timeout")), isPending: false),
            .challenge(.incomingDecided(.accept), isPending: false), .challenge(.incomingDecided(.decline(reason: .rated, rule: "r")), isPending: false),
            .challenge(.incomingDecided(.ignore(rule: "r")), isPending: false),
            .challenge(.canceledOnLichessWithoutRecordedWithdrawal, isPending: false),
            .challenge(.canceledOnLichessDirectionNotRecorded, isPending: false),
            .reconstructedNotCreated(.opponentOffline), .reconstructedNotCreated(.refused(refusal)),
            .reconstructedNotCreated(.botGameLimit(gamesPlayed: 100, untilAsLogged: "6 PM")),
        ]
        for state in states {
            XCTAssertFalse(LogStyle.stateText(state).isEmpty, "\(state)")
            let row = LichessBotChallengeLogRow(key: .challenge(id: "x"), at: Date(), direction: .outgoing, opponent: .notRecorded, terms: nil,
                                                initiative: .senderNotRecorded, state: state, notes: [.canceledOnLichess],
                                                anomalies: [.conflictingDirections], credits: .notRecorded, source: .live)
            let help = LogStyle.stateHelp(row)
            XCTAssertFalse(help.isEmpty, "\(state)")
            XCTAssertTrue(help.contains(LogStyle.noteText(.canceledOnLichess)), help)
            XCTAssertTrue(help.contains(LogStyle.anomalyText(.conflictingDirections)), help)
        }
        XCTAssertEqual(LogStyle.stateText(.challenge(.declined(.unrecognized("nobot")), isPending: false)), "Declined (nobot)")
    }

    func testTermsSourceCreditsAndSenderWording() {
        XCTAssertEqual(LogStyle.termsText(.init(rated: true, limitSeconds: 180, incrementSeconds: 2, daysPerTurn: nil, color: .init(.random))), "3+2 · rated · random")
        XCTAssertEqual(LogStyle.termsText(.init(rated: false, limitSeconds: 30, incrementSeconds: 0, daysPerTurn: nil, color: .init(.white))), "½+0 · casual · white")
        XCTAssertEqual(LogStyle.termsText(.init(rated: false, limitSeconds: 90, incrementSeconds: 1, daysPerTurn: nil, color: .init(.black))), "1.5+1 · casual · black")
        XCTAssertEqual(LogStyle.termsText(.init(rated: true, limitSeconds: nil, incrementSeconds: nil, daysPerTurn: 3, color: .init(.random))), "3 days per move · rated · random")
        XCTAssertEqual(LogStyle.termsText(nil), "Not recorded")

        XCTAssertEqual(LogStyle.sourceText(.live), "Live")
        XCTAssertEqual(LogStyle.sourceText(.reconstructed(.certain)), "Reconstructed (certain)")
        XCTAssertEqual(LogStyle.sourceText(.reconstructed(.paired)), "Reconstructed (paired)")
        XCTAssertEqual(LogStyle.sourceText(.reconstructed(.inferredFromAbsence)), "Reconstructed (inferred)")
        XCTAssertEqual(LogStyle.sourceText(.reconstructed(nil)), "Reconstructed (sender not determined)")

        XCTAssertEqual(LogStyle.creditsText(.spent(2)), "  2")
        XCTAssertEqual(LogStyle.creditsText(.notRecorded), "–")
        XCTAssertNotEqual(LogStyle.creditsHelp(.notRecorded), LogStyle.creditsHelp(.notApplicable))

        XCTAssertEqual(LogStyle.initiativeText(.sentBy(.challengeSheet)), Style.shortLabel(for: .challengeSheet))
        XCTAssertEqual(LogStyle.initiativeText(.sentBy(.matchmaking(trigger: .fillOpenSlots, fillMode: .onlyWhenIdle))), "Matchmaking (Fill Open Slots, only when idle)")
        XCTAssertEqual(LogStyle.initiativeText(.reconstructedSender(.attributed(sender: .byOperator, confidence: .inferredFromAbsence))), "Operator (sheet or resend)≈")
        XCTAssertEqual(LogStyle.initiativeText(.decided(.decline(reason: .tooFast, rule: "r"))), "DCM: decline (tooFast)")
        XCTAssertEqual(LogStyle.initiativeHelp(.decided(.decline(reason: .tooFast, rule: "too fast for me"))), "DCM declined it: too fast for me")
        XCTAssertEqual(LogStyle.directionLabel(.incoming), "IN")
        XCTAssertEqual(LogStyle.directionLabel(.outgoing), "OUT")
        XCTAssertEqual(LogStyle.directionLabel(nil), "–")
        for kind in LichessBotChallengeLogStateKind.allCases {
            XCTAssertFalse(LogStyle.label(kind).isEmpty)
        }
        for kind in LichessBotChallengeLogSenderKind.allCases {
            XCTAssertFalse(LogStyle.label(kind).isEmpty)
        }
    }

    func testFooterWording() {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        XCTAssertEqual(LogStyle.ledgerStatusText(ledger: nil, loadedFiles: nil), "Challenge log: loading…")
        let loaded = LichessBotChallengeLogRecorder.LoadedFiles(fileCount: 2, lineCount: 40, byteCount: 9000, skippedNewerLines: 1)
        XCTAssertEqual(LogStyle.ledgerStatusText(ledger: ledger, loadedFiles: loaded), "Challenge log at load: 2 file(s), 40 line(s), 1 newer line(s) skipped")
        XCTAssertFalse(LogStyle.ledgerStatusIsWarning(ledger))
        ledger = LichessBotChallengeLedger(loadStatus: .partial(filesLeftOut: [.init(name: "challenges-20261001.jsonl", reason: "line 3 doesn't decode")]))
        XCTAssertEqual(LogStyle.ledgerStatusText(ledger: ledger, loadedFiles: loaded),
                       "Challenge log at load: 2 file(s), 40 line(s), 1 newer line(s) skipped; 1 file(s) left out: challenges-20261001.jsonl (line 3 doesn't decode)")
        XCTAssertTrue(LogStyle.ledgerStatusIsWarning(ledger))
        ledger = LichessBotChallengeLedger(loadStatus: .failed(reason: "no access"))
        XCTAssertTrue(LogStyle.ledgerStatusText(ledger: ledger, loadedFiles: nil).contains("the read failed (no access)"))

        var counts = LichessBotChallengeLogCounts()
        counts.shown = 3
        counts.outgoing = 2
        counts.incoming = 1
        counts.live = 3
        counts.creditsSpent = 2
        XCTAssertEqual(LogStyle.countsText(counts, totalRowCount: 10),
                       "3 of 10 shown · outgoing 2 · incoming 1 · live 3 · reconstructed 0 · credits recorded 2")

        XCTAssertTrue(LogStyle.historyStatusText(.notBuilt, history: nil).contains("not built yet"))
        XCTAssertTrue(LogStyle.historyStatusText(.noAccount, history: nil).contains("no Lichess account"))
        XCTAssertTrue(LogStyle.historyStatusText(.failed("boom"), history: nil).contains("failed (boom)"))
        XCTAssertTrue(LogStyle.historyStatusIsWarning(.failed("boom")))
        XCTAssertFalse(LogStyle.historyStatusIsWarning(.rebuilding))
    }

    func testFooterHistoryWordingWithRows() throws {
        let history = try LichessBotChallengeLogRowFixtures.reconstruction()
        let text = LogStyle.historyStatusText(.ready(.unchanged, at: Date()), history: history)
        XCTAssertTrue(text.hasPrefix("Reconstructed history: \(history.rows.count) row(s) from 1 protocol file(s), no live log yet; up to date at "), text)
    }
}
