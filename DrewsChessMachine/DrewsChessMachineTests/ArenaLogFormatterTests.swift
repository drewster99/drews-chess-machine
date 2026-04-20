//
//  ArenaLogFormatterTests.swift
//  DrewsChessMachineTests
//
//  Tests for the arena log formatter extracted from ContentView's
//  logArenaResult. Covers both outputs:
//   - formatKVLine(...)      — the machine-readable grep target
//   - formatHumanReadable    — the block displayed in the session log
//   - formatVerdict / formatDuration / drawRateFraction — helpers
//
//  The KV line is parse-facing and any silent change in key names
//  or ordering breaks downstream tooling — pinning every expected
//  key here forces an intentional test update when the schema
//  actually changes.
//

import XCTest
@testable import DrewsChessMachine

final class ArenaLogFormatterHelperTests: XCTestCase {

    func testFormatDurationBelowMinute() {
        XCTAssertEqual(ArenaLogFormatter.formatDuration(0), "0:00")
        XCTAssertEqual(ArenaLogFormatter.formatDuration(5), "0:05")
        XCTAssertEqual(ArenaLogFormatter.formatDuration(59), "0:59")
    }

    func testFormatDurationOverMinute() {
        XCTAssertEqual(ArenaLogFormatter.formatDuration(60), "1:00")
        XCTAssertEqual(ArenaLogFormatter.formatDuration(125), "2:05")
        XCTAssertEqual(ArenaLogFormatter.formatDuration(3725), "62:05")  // 1h 2m 5s formatted as m:ss
    }

    func testFormatDurationTruncatesFractionalSeconds() {
        XCTAssertEqual(ArenaLogFormatter.formatDuration(65.9), "1:05")   // truncate, not round
    }

    func testFormatVerdictKept() {
        let r = makeRecord(promoted: false, kind: nil, pid: nil)
        XCTAssertEqual(ArenaLogFormatter.formatVerdict(record: r), "kept")
    }

    func testFormatVerdictPromotedAutoWithID() {
        let r = makeRecord(promoted: true, kind: .automatic, pid: stubID("20260420-3-A1B2"))
        XCTAssertEqual(ArenaLogFormatter.formatVerdict(record: r), "PROMOTED(auto)=20260420-3-A1B2")
    }

    func testFormatVerdictPromotedManualWithID() {
        let r = makeRecord(promoted: true, kind: .manual, pid: stubID("20260420-9-ZZZZ"))
        XCTAssertEqual(ArenaLogFormatter.formatVerdict(record: r), "PROMOTED(manual)=20260420-9-ZZZZ")
    }

    func testFormatVerdictPromotedWithoutIDFallback() {
        // Defensive: promoted=true with no pid (shouldn't normally
        // happen) still must not crash. No ID tail, no trailing "=".
        let r = makeRecord(promoted: true, kind: .automatic, pid: nil)
        XCTAssertEqual(ArenaLogFormatter.formatVerdict(record: r), "PROMOTED(auto)")
    }

    func testDrawRateFraction() {
        let r = makeRecord(wins: 80, draws: 100, losses: 20)  // 200 games
        XCTAssertEqual(ArenaLogFormatter.drawRateFraction(record: r), 0.5, accuracy: 1e-12)
    }

    func testDrawRateFractionEmptyTournament() {
        let r = makeRecord(wins: 0, draws: 0, losses: 0)
        XCTAssertEqual(ArenaLogFormatter.drawRateFraction(record: r), 0, accuracy: 1e-12)
    }

    // MARK: - Fixture helpers

    private func stubID(_ s: String) -> ModelID {
        ModelID(value: s)
    }

    private func makeRecord(
        wins: Int = 0, draws: Int = 0, losses: Int = 0,
        promoted: Bool = false,
        kind: PromotionKind? = nil,
        pid: ModelID? = nil,
        durationSec: Double = 60
    ) -> TournamentRecord {
        TournamentRecord(
            finishedAtStep: 0,
            gamesPlayed: wins + draws + losses,
            candidateWins: wins,
            championWins: losses,
            draws: draws,
            score: ArenaEloStats.score(wins: wins, draws: draws, losses: losses),
            promoted: promoted,
            promotionKind: kind,
            promotedID: pid,
            durationSec: durationSec,
            candidateWinsAsWhite: 0, candidateWinsAsBlack: 0,
            candidateLossesAsWhite: 0, candidateLossesAsBlack: 0,
            candidateDrawsAsWhite: 0, candidateDrawsAsBlack: 0
        )
    }
}

// MARK: - KV Line

final class ArenaLogFormatterKVTests: XCTestCase {

    private func makeRecord() -> TournamentRecord {
        // Controlled fixture matching the user's worked example:
        // 1000 games, 312W / 401D / 287L, split by side deterministically.
        TournamentRecord(
            finishedAtStep: 12345,
            gamesPlayed: 1000,
            candidateWins: 312,
            championWins: 287,
            draws: 401,
            score: ArenaEloStats.score(wins: 312, draws: 401, losses: 287),
            promoted: false,
            promotionKind: nil,
            promotedID: nil,
            durationSec: 945.0,
            candidateWinsAsWhite: 170, candidateWinsAsBlack: 142,
            candidateLossesAsWhite: 130, candidateLossesAsBlack: 157,
            candidateDrawsAsWhite: 200, candidateDrawsAsBlack: 201
        )
    }

    func testKVLineContainsEveryDocumentedKey() {
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 7,
            candidateID: "CAND-1",
            championID: "CHAMP-1",
            trainerID: "TRAIN-1",
            buildNumber: 230
        )
        // Every documented key from the log format — none may be
        // silently dropped.
        let expectedKeys = [
            "step=", "games=", "w=", "d=", "l=",
            "score=", "elo=", "elo_lo=", "elo_hi=",
            "draw_rate=",
            "cand_white_w=", "cand_white_d=", "cand_white_l=",
            "cand_black_w=", "cand_black_d=", "cand_black_l=",
            "cand_white_score=", "cand_black_score=",
            "promoted=", "kind=", "dur_sec=", "build=",
            "candidate=", "champion=", "trainer="
        ]
        for key in expectedKeys {
            XCTAssertTrue(line.contains(key), "kv line missing key \(key)")
        }
    }

    func testKVLinePrefixShape() {
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 7,
            candidateID: "CAND-1", championID: "CHAMP-1", trainerID: "TRAIN-1",
            buildNumber: 230
        )
        XCTAssertTrue(line.hasPrefix("[ARENA] #7 kv step=12345"),
            "prefix must let greppers key on '[ARENA] #N kv'")
    }

    func testKVLineRecordCounterValues() {
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertTrue(line.contains("games=1000"))
        XCTAssertTrue(line.contains("w=312"))
        XCTAssertTrue(line.contains("d=401"))
        XCTAssertTrue(line.contains("l=287"))
        XCTAssertTrue(line.contains("dur_sec=945.0"))
    }

    func testKVLinePerSideBreakdown() {
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertTrue(line.contains("cand_white_w=170"))
        XCTAssertTrue(line.contains("cand_white_d=200"))
        XCTAssertTrue(line.contains("cand_white_l=130"))
        XCTAssertTrue(line.contains("cand_black_w=142"))
        XCTAssertTrue(line.contains("cand_black_d=201"))
        XCTAssertTrue(line.contains("cand_black_l=157"))
    }

    func testKVLinePromotionFields() {
        var r = makeRecord()
        r = TournamentRecord(
            finishedAtStep: r.finishedAtStep,
            gamesPlayed: r.gamesPlayed,
            candidateWins: r.candidateWins,
            championWins: r.championWins,
            draws: r.draws,
            score: r.score,
            promoted: true,
            promotionKind: .automatic,
            promotedID: ModelID(value: "20260420-3-XYZ1"),
            durationSec: r.durationSec,
            candidateWinsAsWhite: r.candidateWinsAsWhite,
            candidateWinsAsBlack: r.candidateWinsAsBlack,
            candidateLossesAsWhite: r.candidateLossesAsWhite,
            candidateLossesAsBlack: r.candidateLossesAsBlack,
            candidateDrawsAsWhite: r.candidateDrawsAsWhite,
            candidateDrawsAsBlack: r.candidateDrawsAsBlack
        )
        let line = ArenaLogFormatter.formatKVLine(
            record: r, index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertTrue(line.contains("promoted=1"))
        XCTAssertTrue(line.contains("kind=automatic"))
    }

    func testKVLineKeptHasKindNone() {
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),  // not promoted
            index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertTrue(line.contains("promoted=0"))
        XCTAssertTrue(line.contains("kind=none"))
    }

    func testKVLineEloNanForDegenerateSample() {
        // 0W / 0D / 200L → score 0 → Elo endpoints all undefined.
        // Must render as literal "nan" (documented parser signal),
        // not "—".
        let r = TournamentRecord(
            finishedAtStep: 1,
            gamesPlayed: 200,
            candidateWins: 0, championWins: 200, draws: 0,
            score: 0,
            promoted: false,
            promotionKind: nil,
            promotedID: nil,
            durationSec: 1,
            candidateWinsAsWhite: 0, candidateWinsAsBlack: 0,
            candidateLossesAsWhite: 100, candidateLossesAsBlack: 100,
            candidateDrawsAsWhite: 0, candidateDrawsAsBlack: 0
        )
        let line = ArenaLogFormatter.formatKVLine(
            record: r, index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertTrue(line.contains("elo=nan"))
        XCTAssertTrue(line.contains("elo_lo=nan"))
        XCTAssertTrue(line.contains("elo_hi=nan"))
    }

    func testKVLineIDPassThrough() {
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 1,
            candidateID: "20260420-5-AAAA",
            championID: "20260420-4-BBBB",
            trainerID: "20260420-6-CCCC",
            buildNumber: 999
        )
        XCTAssertTrue(line.contains("candidate=20260420-5-AAAA"))
        XCTAssertTrue(line.contains("champion=20260420-4-BBBB"))
        XCTAssertTrue(line.contains("trainer=20260420-6-CCCC"))
        XCTAssertTrue(line.contains("build=999"))
    }

    func testKVLineDrawRateMatchesRecord() {
        // User-spec: 401 draws / 1000 games → 0.4010
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertTrue(line.contains("draw_rate=0.4010"))
    }

    func testKVLineIsSingleLine() {
        // Must not contain newlines — it's a single log line.
        let line = ArenaLogFormatter.formatKVLine(
            record: makeRecord(),
            index: 1,
            candidateID: "x", championID: "y", trainerID: "z",
            buildNumber: 1
        )
        XCTAssertFalse(line.contains("\n"))
    }
}

// MARK: - Human-readable block

final class ArenaLogFormatterHumanReadableTests: XCTestCase {

    private func makeParameters(build: Int = 230) -> ArenaLogFormatter.Parameters {
        ArenaLogFormatter.Parameters(
            batchSize: 4096,
            learningRate: 1e-4,
            promoteThreshold: 0.55,
            tournamentGames: 200,
            spStartTau: 1.0, spFloorTau: 0.4, spDecayPerPly: 0.025,
            arStartTau: 1.0, arFloorTau: 0.2, arDecayPerPly: 0.025,
            workerCount: 8,
            buildNumber: build
        )
    }

    private func makeDiversity() -> ArenaLogFormatter.Diversity {
        ArenaLogFormatter.Diversity(
            uniqueGames: 195, gamesInWindow: 200,
            uniquePercent: 97.5, avgDivergencePly: 8.3
        )
    }

    private func makeRecord() -> TournamentRecord {
        TournamentRecord(
            finishedAtStep: 5000,
            gamesPlayed: 200,
            candidateWins: 120, championWins: 45, draws: 35,
            score: ArenaEloStats.score(wins: 120, draws: 35, losses: 45),
            promoted: true,
            promotionKind: .automatic,
            promotedID: ModelID(value: "20260420-3-ABCD"),
            durationSec: 945.0,
            candidateWinsAsWhite: 65, candidateWinsAsBlack: 55,
            candidateLossesAsWhite: 20, candidateLossesAsBlack: 25,
            candidateDrawsAsWhite: 15, candidateDrawsAsBlack: 20
        )
    }

    func testLineCountAndOrder() {
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 3,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        XCTAssertEqual(lines.count, 13)
        XCTAssertTrue(lines[0].hasPrefix("[ARENA] #3 Candidate vs Champion"))
        XCTAssertTrue(lines[1].contains("Games:"))
        XCTAssertTrue(lines[2].contains("Result:"))
        XCTAssertTrue(lines[3].contains("Score:"))
        XCTAssertTrue(lines[4].contains("Elo diff:"))
        XCTAssertTrue(lines[5].contains("Draw rate:"))
        XCTAssertEqual(lines[6], "[ARENA]     By side:")
        XCTAssertTrue(lines[7].contains("Candidate as white"))
        XCTAssertTrue(lines[8].contains("Candidate as black"))
        XCTAssertTrue(lines[9].contains("batch=4096"))
        XCTAssertTrue(lines[10].contains("candidate=C"))
        XCTAssertTrue(lines[11].contains("diversity:"))
        XCTAssertTrue(lines[12].contains("Verdict:"))
    }

    func testEveryLineBeginsWithArenaTag() {
        // Every line starts with "[ARENA]" so a tag-filter in the
        // log analyzer picks up the whole block atomically.
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        for (i, line) in lines.enumerated() {
            XCTAssertTrue(line.hasPrefix("[ARENA]"), "line \(i) missing tag: \(line)")
        }
    }

    func testResultLineFormat() {
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        // Ticket-specified format: "W wins / D draws / L losses"
        // — rendered concretely as "120W / 35D / 45L" from candidate
        // perspective.
        XCTAssertTrue(lines[2].contains("120W / 35D / 45L"))
    }

    func testVerdictLineCarriesDurationAndPromotedID() {
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        XCTAssertTrue(lines.last!.contains("PROMOTED(auto)=20260420-3-ABCD"))
        XCTAssertTrue(lines.last!.contains("dur=15:45"))
    }

    func testBySideDashWhenNoGames() {
        // Early-abort record with no games on the black side — the
        // ticket asks for "—" in that case rather than "0.0%".
        let r = TournamentRecord(
            finishedAtStep: 1, gamesPlayed: 3,
            candidateWins: 2, championWins: 1, draws: 0,
            score: 0.667,
            promoted: false, promotionKind: nil, promotedID: nil,
            durationSec: 10,
            candidateWinsAsWhite: 2, candidateWinsAsBlack: 0,
            candidateLossesAsWhite: 1, candidateLossesAsBlack: 0,
            candidateDrawsAsWhite: 0, candidateDrawsAsBlack: 0
        )
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: r, index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        XCTAssertTrue(lines[8].contains("Candidate as black: —"),
            "empty black side should render as em-dash, not 0.0%")
    }

    func testParamsLineCarriesSessionContext() {
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(build: 307), diversity: makeDiversity()
        )
        let paramsLine = lines[9]
        XCTAssertTrue(paramsLine.contains("batch=4096"))
        XCTAssertTrue(paramsLine.contains("promote>=0.55"))
        XCTAssertTrue(paramsLine.contains("games=200"))
        XCTAssertTrue(paramsLine.contains("workers=8"))
        XCTAssertTrue(paramsLine.contains("build=307"))
        XCTAssertTrue(paramsLine.contains("sp.tau=1.00/0.40/0.025"))
        XCTAssertTrue(paramsLine.contains("ar.tau=1.00/0.20/0.025"))
    }

    func testIDsLineFormat() {
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 1,
            candidateID: "CAND-ID", championID: "CHAMP-ID", trainerID: "TRAIN-ID",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        XCTAssertTrue(lines[10].contains("candidate=CAND-ID"))
        XCTAssertTrue(lines[10].contains("champion=CHAMP-ID"))
        XCTAssertTrue(lines[10].contains("trainer=TRAIN-ID"))
    }

    func testDiversityLineFormat() {
        let lines = ArenaLogFormatter.formatHumanReadable(
            record: makeRecord(), index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
        XCTAssertTrue(lines[11].contains("unique=195/200"))
        XCTAssertTrue(lines[11].contains("(98%)"))
        XCTAssertTrue(lines[11].contains("avgDiverge=8.3"))
    }
}
