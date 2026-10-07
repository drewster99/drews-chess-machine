//
//  LichessBotChallengeReconstructionRealDataTests.swift
//  DrewsChessMachineTests
//
//  Validation of the back-fill against the bot's real data (challenge-log
//  plan §6.3): Algorithm v1 over the real `Protocol/events-*.jsonl`, joined
//  with the real filed games' ids, printed in the shape of the plan's
//  Appendix B / B.2 prototypes so the two can be compared on the same files.
//
//  Read-only, and in memory: it lists and reads files, and never runs the
//  store, so nothing is written anywhere. Skipped when the folder is absent
//  (any machine but the bot's). `DCM_CHALLENGE_RECONSTRUCTION_ROOT` (passed
//  to the test runner as `TEST_RUNNER_DCM_CHALLENGE_RECONSTRUCTION_ROOT`)
//  points it at a copy instead, so the prototype and this test can read
//  identical bytes while the bot keeps appending to the live files.
//
//  It asserts only what must hold on any data (every input line decodes,
//  every frozen form parses); the counts change as the bot plays, so they
//  are printed, not pinned.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeReconstructionRealDataTests: XCTestCase {

    private static let ourAccountID = "drewschessmachine"
    private static let rootOverrideKey = "DCM_CHALLENGE_RECONSTRUCTION_ROOT"

    private struct GameIDOnly: Decodable {
        let gameID: String
    }

    func testReconstructionOfTheRealProtocolLog() throws {
        let root: URL
        if let override = ProcessInfo.processInfo.environment[Self.rootOverrideKey] {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            // Only read from: listed and read below, never written.
            root = LichessBotDataDirectory.standard.root
        }
        let directory = LichessBotDataDirectory(root: root)
        guard try FileSafety.existingItem(at: directory.protocolDirectory)?.kind == .directory else {
            throw XCTSkip("no Lichess bot protocol log at \(directory.protocolDirectory.path)")
        }

        let listed = try LichessBotChallengeReconstructionStore.listInputs(in: directory, liveLogFirstEntryAt: nil)
        let files = try listed.map { LichessBotProtocolDayFile(name: $0.name, data: try Data(contentsOf: $0.url)) }
        let start = ContinuousClock.now
        let reconstruction = LichessBotChallengeReconstruction.build(from: files, ourAccountID: Self.ourAccountID, liveLogFirstEntryAt: nil)
        let elapsed = ContinuousClock.now - start

        let gameIDs = try filedGameIDs(in: directory.gamesDirectory)
        let lookup = LichessBotReconstructedChallengeLookup(reconstruction)
        let games = lookup.gameCounts(gameIDs: gameIDs)

        XCTAssertFalse(reconstruction.rows.isEmpty)
        for input in reconstruction.inputs {
            XCTAssertEqual(input.undecodableLines, [], "\(input.file)")
        }
        XCTAssertEqual(reconstruction.unexplainedLineCount(.unparsableDetails), 0,
                       "\(reconstruction.unexplainedLines.filter { $0.reason == .unparsableDetails }.map(\.line))")
        XCTAssertEqual(reconstruction.unexplainedLineCount(.undecodableStreamEvent), 0)
        XCTAssertEqual(Set(gameIDs).count, gameIDs.count, "a game filed twice")

        print(LichessBotChallengeReconstructionStore.summaryLine(
            .init(outcome: .unchanged, reconstruction: reconstruction, undecodableStoredFile: nil), games: games))
        print("[RECONSTRUCTION-VALIDATION] root=\(root.path) inputs=\(reconstruction.inputs.map { "\($0.file):\($0.byteCount)" }) "
              + "build_ms=\(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)")
        for line in breakdown(reconstruction, games: games, gameCount: gameIDs.count) {
            print("[RECONSTRUCTION-VALIDATION] \(line)")
        }
    }

    /// Every filed game record's id (`Games/YYYY/MM/*.json`).
    private func filedGameIDs(in gamesDirectory: URL) throws -> [String] {
        guard try FileSafety.existingItem(at: gamesDirectory) != nil else { return [] }
        let decoder = JSONDecoder()
        var ids: [String] = []
        for year in try FileManager.default.contentsOfDirectory(atPath: gamesDirectory.path).sorted() {
            let yearURL = gamesDirectory.appendingPathComponent(year, isDirectory: true)
            guard try FileSafety.existingItem(at: yearURL)?.kind == .directory else { continue }
            for month in try FileManager.default.contentsOfDirectory(atPath: yearURL.path).sorted() {
                let monthURL = yearURL.appendingPathComponent(month, isDirectory: true)
                guard try FileSafety.existingItem(at: monthURL)?.kind == .directory else { continue }
                for name in try FileManager.default.contentsOfDirectory(atPath: monthURL.path).sorted() where name.hasSuffix(".json") {
                    let data = try Data(contentsOf: monthURL.appendingPathComponent(name, isDirectory: false))
                    ids.append(try decoder.decode(GameIDOnly.self, from: data).gameID)
                }
            }
        }
        return ids
    }

    /// The counts the plan's Appendix B and B.2 print, from the
    /// reconstruction.
    private func breakdown(_ reconstruction: LichessBotChallengeReconstruction,
                           games: LichessBotReconstructedGameCounts,
                           gameCount: Int) -> [String] {
        var outgoingStates: [String: Int] = [:]
        var incomingStates: [String: Int] = [:]
        var notCreated = 0
        var notCreatedPaired = 0
        var notCreatedInferred = 0
        var notCreatedOther = 0
        var pickFoundBySender: [String: Int] = [:]
        var pickMissingBySender: [String: Int] = [:]
        var outgoingSendersBySender: [String: Int] = [:]
        for row in reconstruction.rows {
            switch (row.key, row.direction, row.state) {
            case (.notCreatedAttempt, _, _):
                notCreated += 1
                switch row.sender {
                case .attributed(_, .paired)?: notCreatedPaired += 1
                case .attributed(_, .inferredFromAbsence)?: notCreatedInferred += 1
                default: notCreatedOther += 1
                }
            case (.challenge, .outgoing, .challenge(let state)):
                outgoingStates[String(describing: state), default: 0] += 1
                outgoingSendersBySender[String(describing: row.sender), default: 0] += 1
                switch row.pickLineCheck {
                case .found?: pickFoundBySender[String(describing: row.sender), default: 0] += 1
                case .notFound?: pickMissingBySender[String(describing: row.sender), default: 0] += 1
                case nil: break
                }
            case (.challenge, .incoming, .challenge(let state)):
                switch state {
                case .accepted:
                    incomingStates["accepted", default: 0] += 1
                case .open, .declined, .canceledByChallenger, .withdrawn, .notCreated, .incomingDecided,
                     .canceledOnLichessWithoutRecordedWithdrawal, .canceledOnLichessDirectionNotRecorded:
                    let decision = row.decision.map { decision -> String in
                        switch decision {
                        case .accept: return "accept"
                        case .decline: return "decline"
                        case .ignore: return "ignore"
                        }
                    } ?? "undecided"
                    incomingStates["\(decision) (\(state))", default: 0] += 1
                }
            case (.challenge, _, .notCreated):
                incomingStates["impossible", default: 0] += 1
            }
        }
        let counts = reconstruction.counts
        var unexplained: [String: Int] = [:]
        for line in reconstruction.unexplainedLines {
            unexplained[line.reason.rawValue, default: 0] += 1
        }
        return [
            "rows \(reconstruction.rows.count): outgoing created \(counts.outgoingCreatedRows), not created \(counts.notCreatedRows), incoming \(counts.incomingRows)",
            "outgoing states \(outgoingStates.sorted { $0.key < $1.key })",
            "outgoing senders \(outgoingSendersBySender.sorted { $0.key < $1.key })",
            "pick line found \(pickFoundBySender.sorted { $0.key < $1.key }); missing \(pickMissingBySender.sorted { $0.key < $1.key })",
            "not created \(notCreated) paired \(notCreatedPaired) inferred \(notCreatedInferred) other \(notCreatedOther) "
                + "(outcome lines \(counts.notCreatedFromOutcomeLines), bot-limit lines \(counts.notCreatedFromBotLimitLines), "
                + "bot-limit beside outcome \(counts.botLimitLinesBesideOutcomeLines))",
            "incoming \(incomingStates.sorted { $0.key < $1.key })",
            "send lines \(counts.sendLines), companions paired \(counts.companionsPaired), failure companions paired \(counts.failureCompanionsPaired), "
                + "own-echo accepts skipped \(counts.skippedOwnEchoAcceptLines), other own-echo decisions skipped \(counts.skippedOwnEchoOtherDecisionLines)",
            "unexplained \(unexplained.sorted { $0.key < $1.key })",
            "games \(gameCount): incoming \(games[.incoming]), matchmaking \(games[.matchmaking]), queue \(games[.challengeQueue]), "
                + "casual resend \(games[.matchmakingCasualResend]), operator (inferred) \(games[.operatorInferred]), "
                + "sender not determined \(games[.outgoingSenderNotDetermined]), unknown \(games[.unknown])",
        ]
    }
}
