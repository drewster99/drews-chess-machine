import XCTest
@testable import DrewsChessMachine

/// A move source standing for a follow-lineage generation: its info carries
/// the followed lineage, and it records which plies it decided.
final class LichessBotFollowedGenerationMoveSource: LichessBotMoveSource, @unchecked Sendable {
    let info: LichessBotGenerationInfo
    let decidedPlies = SyncBox<[Int]>([])

    init(generationID: Int, modelID: String, step: Int, followed: LichessBotFollowedLineage) {
        info = LichessBotGenerationInfo(
            generationID: generationID,
            sourceKind: .followLineage,
            modelID: modelID,
            trainingStep: step,
            snapshotAt: Date(timeIntervalSince1970: 0),
            architectureSummary: "test",
            filePath: "/models/\(modelID).safetensors",
            fileSHA256: "file-\(generationID)",
            valueHeadRecenteredOnLoad: false,
            lineage: LichessBotGenerationLineage(
                lineageRunID: followed.lineageRunID, segmentID: followed.anchorSegmentID, segmentIndex: 0,
                segmentChain: [followed.anchorSegmentID], segmentLocalStep: step, recordedUnix: Int64(step),
                cumTrainerStep: step, contentSHA256: "content-\(generationID)", followed: followed)
        )
    }

    func decide(_ request: LichessBotMoveRequest, schedule: SamplingSchedule) async throws -> LichessBotMoveDecision {
        decidedPlies.modify { $0.append(request.ply) }
        guard let uci = request.legalMoves.map(\.uci).sorted().first else {
            throw LichessBotMoveChooserError.noLegalMoves
        }
        return LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 1, topMoves: [],
            win: 0.3, draw: 0.4, loss: 0.3,
            temperature: schedule.floorTau, legalMoveCount: request.legalMoves.count, randomish: false,
            encodeMilliseconds: 0, inferenceMilliseconds: 0, sampleMilliseconds: 0
        )
    }
}

/// Mid-game refresh for the follow-lineage source (follow-lineage plan
/// §3.8): with the toggle on, a game switches to a newer generation of the
/// same followed lineage, already built; never to another lineage's, never
/// with the toggle off.
final class LichessBotFollowLineageMidGameTests: XCTestCase {

    private let lineageA = LichessBotFollowedLineage(lineageRunID: "RUN-A", anchorSegmentID: "SEG-A")
    private let lineageB = LichessBotFollowedLineage(lineageRunID: "RUN-B", anchorSegmentID: "SEG-B")

    /// Plays one game (our two moves, then the opponent resigns) with
    /// `first` playing and `latest` as the newest built generation.
    private func play(first: any LichessBotMoveSource, latest: any LichessBotMoveSource, followed: LichessBotFollowedLineage, midGameRefresh: Bool) async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .finish(status: "resign", winner: "white")])
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.model.source = .followLineage
        settings.model.followedLineage = followed
        settings.model.midGameRefresh = midGameRefresh
        let frozen = settings
        let observer = LichessBotRecordingGameObserver()
        let session = LichessBotGameSession(
            gameID: LichessBotFakeGameServer.gameID,
            ourAccountID: LichessBotFakeGameServer.botID,
            api: server,
            moveSource: first,
            latestMoveSource: { latest },
            settingsProvider: { frozen },
            observer: observer,
            time: LichessBotManualTime(),
            onTurnStatus: { _, _ in },
            pacing: { LichessBotMovePacingSnapshot() },
            carryover: .newGame
        )
        let run = Task { await session.run() }
        for _ in 0..<2000 where observer.finishedStatus == nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertNotNil(observer.finishedStatus, "the game ends")
        await run.value
    }

    func testFollowedLineageGameSwitchesToANewerGenerationOfTheSameLineage() async throws {
        let first = LichessBotFollowedGenerationMoveSource(generationID: 1, modelID: "20261006-1-AAAA", step: 1000, followed: lineageA)
        let newer = LichessBotFollowedGenerationMoveSource(generationID: 2, modelID: "20261006-1-AAAA", step: 2000, followed: lineageA)
        try await play(first: first, latest: newer, followed: lineageA, midGameRefresh: true)
        XCTAssertEqual(newer.decidedPlies.value, [0, 2])
        XCTAssertEqual(first.decidedPlies.value, [])
    }

    func testGameNeverSwitchesToAnotherFollowedLineage() async throws {
        let first = LichessBotFollowedGenerationMoveSource(generationID: 1, modelID: "20261006-1-AAAA", step: 1000, followed: lineageA)
        let other = LichessBotFollowedGenerationMoveSource(generationID: 2, modelID: "20261006-2-BBBB", step: 9000, followed: lineageB)
        try await play(first: first, latest: other, followed: lineageB, midGameRefresh: true)
        XCTAssertEqual(first.decidedPlies.value, [0, 2], "the game keeps the lineage it started with after the source is re-pointed")
        XCTAssertEqual(other.decidedPlies.value, [])
    }

    func testGameNeverSwitchesWithMidGameRefreshOff() async throws {
        let first = LichessBotFollowedGenerationMoveSource(generationID: 1, modelID: "20261006-1-AAAA", step: 1000, followed: lineageA)
        let newer = LichessBotFollowedGenerationMoveSource(generationID: 2, modelID: "20261006-1-AAAA", step: 2000, followed: lineageA)
        try await play(first: first, latest: newer, followed: lineageA, midGameRefresh: false)
        XCTAssertEqual(first.decidedPlies.value, [0, 2])
        XCTAssertEqual(newer.decidedPlies.value, [])
    }

    func testGameNeverSwitchesToAnOlderGeneration() async throws {
        let first = LichessBotFollowedGenerationMoveSource(generationID: 3, modelID: "20261006-1-AAAA", step: 3000, followed: lineageA)
        let older = LichessBotFollowedGenerationMoveSource(generationID: 2, modelID: "20261006-1-AAAA", step: 2000, followed: lineageA)
        try await play(first: first, latest: older, followed: lineageA, midGameRefresh: true)
        XCTAssertEqual(first.decidedPlies.value, [0, 2])
        XCTAssertEqual(older.decidedPlies.value, [])
    }

    /// A newer generation can hold an older file of the lineage: after the
    /// file that plays is deleted and the source is switched away and back,
    /// the slots build the lineage's newest remaining file under a higher
    /// generation ID. A game in progress never steps back to it (§3.8).
    func testGameNeverSwitchesToANewerGenerationOfAnOlderFile() async throws {
        let first = LichessBotFollowedGenerationMoveSource(generationID: 2, modelID: "20261006-1-AAAA", step: 2000, followed: lineageA)
        let olderFile = LichessBotFollowedGenerationMoveSource(generationID: 3, modelID: "20261006-1-AAAA", step: 1000, followed: lineageA)
        try await play(first: first, latest: olderFile, followed: lineageA, midGameRefresh: true)
        XCTAssertEqual(first.decidedPlies.value, [0, 2])
        XCTAssertEqual(olderFile.decidedPlies.value, [])
    }
}
