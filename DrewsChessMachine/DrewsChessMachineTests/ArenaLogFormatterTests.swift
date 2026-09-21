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

// MARK: - SPRT reporting

/// The `[ARENA]` block is the only durable record of why an arena decided what
/// it did, so these tests are mostly about the three non-promoting SPRT
/// outcomes staying distinguishable from each other and from a threshold-mode
/// "kept". Collapsing any two of them turns "the guard fired on ambiguous
/// evidence" into "the candidate was rejected", which is a different claim.
final class ArenaLogFormatterSPRTTests: XCTestCase {

    private func makeConfig(
        elo0: Double = 0, elo1: Double = 10,
        alpha: Double = 0.05, beta: Double = 0.05,
        minGames: Int = 32, maxGames: Int = 20000
    ) throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(
            elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
            minGames: minGames, maxGames: maxGames
        )
    }

    private func makeRecord(
        criterion: ArenaPromotionCriterion?,
        verdict: ArenaSPRT.Verdict?,
        promoted: Bool = false,
        gamesPlayed: Int = 60
    ) -> TournamentRecord {
        var record = TournamentRecord(
            finishedAtStep: 5000,
            gamesPlayed: gamesPlayed,
            candidateWins: 20, championWins: 10, draws: gamesPlayed - 30,
            score: ArenaEloStats.score(wins: 20, draws: gamesPlayed - 30, losses: 10),
            promoted: promoted,
            promotionKind: promoted ? .automatic : nil,
            promotedID: promoted ? ModelID(value: "20260420-3-ABCD") : nil,
            durationSec: 120,
            candidateWinsAsWhite: 10, candidateWinsAsBlack: 10,
            candidateLossesAsWhite: 5, candidateLossesAsBlack: 5,
            candidateDrawsAsWhite: (gamesPlayed - 30) / 2,
            candidateDrawsAsBlack: (gamesPlayed - 30) - (gamesPlayed - 30) / 2
        )
        record.promotionCriterion = criterion
        record.sprtVerdict = verdict
        return record
    }

    private func verdict(
        _ decision: ArenaSPRT.Decision,
        llr: Double?,
        wins: Int = 20, draws: Int = 12, losses: Int = 10,
        config: ArenaSPRT.SPRTConfig
    ) -> ArenaSPRT.Verdict {
        ArenaSPRT.Verdict(
            decision: decision, llr: llr,
            wins: wins, draws: draws, losses: losses, config: config
        )
    }

    private func makeParameters() -> ArenaLogFormatter.Parameters {
        ArenaLogFormatter.Parameters(
            batchSize: 4096, learningRate: 1e-4,
            promoteThreshold: 0.53, tournamentGames: 400,
            spStartTau: 1.0, spFloorTau: 0.5, spDecayPerPly: 0.007,
            arStartTau: 0.6, arFloorTau: 0.2, arDecayPerPly: 0.02,
            workerCount: 8, buildNumber: 2400
        )
    }

    private func makeDiversity() -> ArenaLogFormatter.Diversity {
        ArenaLogFormatter.Diversity(
            uniqueGames: 58, gamesInWindow: 60,
            uniquePercent: 96.7, avgDivergencePly: 6.1
        )
    }

    private func humanBlock(_ record: TournamentRecord) -> [String] {
        ArenaLogFormatter.formatHumanReadable(
            record: record, index: 1,
            candidateID: "C", championID: "M", trainerID: "T",
            parameters: makeParameters(), diversity: makeDiversity()
        )
    }

    // MARK: Criterion token

    /// Legacy records carry no criterion. They all predate SPRT, so they ran
    /// the threshold — reporting them as "unknown" would be less accurate,
    /// not more.
    func testLegacyRecordWithoutCriterionReportsScore() {
        XCTAssertEqual(
            ArenaLogFormatter.criterionToken(makeRecord(criterion: nil, verdict: nil)),
            "score"
        )
    }

    func testCriterionTokenAppearsInBothOutputs() throws {
        let scoreRecord = makeRecord(criterion: .scoreThreshold, verdict: nil)
        XCTAssertTrue(
            humanBlock(scoreRecord).contains { $0.contains("crit=score") },
            "the params line must name the criterion"
        )
        XCTAssertTrue(
            ArenaLogFormatter.formatKVLine(
                record: scoreRecord, index: 1,
                candidateID: "C", championID: "M", trainerID: "T", buildNumber: 2400
            ).contains("crit=score")
        )

        let config = try makeConfig()
        let sprtRecord = makeRecord(
            criterion: .sprt,
            verdict: verdict(.accept, llr: 3.21, config: config),
            promoted: true
        )
        XCTAssertTrue(humanBlock(sprtRecord).contains { $0.contains("crit=sprt") })
        XCTAssertTrue(
            ArenaLogFormatter.formatKVLine(
                record: sprtRecord, index: 1,
                candidateID: "C", championID: "M", trainerID: "T", buildNumber: 2400
            ).contains("crit=sprt")
        )
    }

    // MARK: Threshold mode is untouched

    func testThresholdModeEmitsNoSPRTLinesOrFields() {
        let record = makeRecord(criterion: .scoreThreshold, verdict: nil)
        XCTAssertTrue(ArenaLogFormatter.formatSPRTBlock(record: record).isEmpty)
        XCTAssertEqual(ArenaLogFormatter.formatSPRTKV(record: record), "")

        let kv = ArenaLogFormatter.formatKVLine(
            record: record, index: 1,
            candidateID: "C", championID: "M", trainerID: "T", buildNumber: 2400
        )
        XCTAssertFalse(kv.contains("sprt"), "no sprt_* keys in threshold mode: \(kv)")
        XCTAssertEqual(ArenaLogFormatter.formatVerdict(record: record), "kept")
    }

    func testThresholdModeStillPrintsTheScheduledDenominator() {
        let lines = humanBlock(makeRecord(criterion: .scoreThreshold, verdict: nil))
        XCTAssertTrue(
            lines.contains { $0.contains("Games: 60/400") },
            "threshold mode reports games against the schedule: \(lines)"
        )
    }

    /// Under SPRT there was no schedule, so printing "60/400" would report a
    /// denominator the run never had.
    func testSPRTModeDoesNotInventAScheduledDenominator() throws {
        let record = makeRecord(
            criterion: .sprt,
            verdict: verdict(.reject, llr: -3.5, config: try makeConfig())
        )
        let lines = humanBlock(record)
        XCTAssertFalse(lines.contains { $0.contains("Games: 60/400") })
        XCTAssertTrue(lines.contains { $0.contains("Games: 60 (SPRT") })
    }

    // MARK: The three non-promoting outcomes stay distinct

    func testRejectInconclusiveAndUndecidedRenderDifferently() throws {
        let config = try makeConfig()
        let rejected = makeRecord(
            criterion: .sprt, verdict: verdict(.reject, llr: -3.5, config: config))
        let inconclusive = makeRecord(
            criterion: .sprt, verdict: verdict(.inconclusive, llr: 0.4, config: config))
        let undecided = makeRecord(criterion: .sprt, verdict: nil)

        let verdicts = [rejected, inconclusive, undecided].map(ArenaLogFormatter.formatVerdict)
        XCTAssertEqual(Set(verdicts).count, 3, "all three must be distinguishable: \(verdicts)")
        for line in verdicts {
            XCTAssertFalse(line.isEmpty)
            XCTAssertTrue(line.hasPrefix("kept"), "none of these promote: \(line)")
        }

        // And specifically: an inconclusive run must not be LABELLED a
        // rejection. Matching on the bare word "reject" would be wrong here —
        // the label deliberately contains "rejection", because saying that is
        // the whole point of the wording.
        let inconclusiveText = ArenaLogFormatter.formatVerdict(record: inconclusive)
        XCTAssertTrue(inconclusiveText.contains("inconclusive"))
        XCTAssertFalse(inconclusiveText.contains("SPRT reject"))
        XCTAssertNotEqual(inconclusiveText, ArenaLogFormatter.formatVerdict(record: rejected))
    }

    func testInconclusiveBlockSaysItIsNotARejection() throws {
        let record = makeRecord(
            criterion: .sprt,
            verdict: verdict(.inconclusive, llr: 0.4, config: try makeConfig())
        )
        let block = ArenaLogFormatter.formatSPRTBlock(record: record)
        XCTAssertTrue(
            block.contains { $0.contains("NOT a rejection") },
            "the guard's meaning must be spelled out, not inferred: \(block)"
        )
    }

    func testUndecidedSPRTRunSaysSoRatherThanShowingAnEmptyBlock() {
        let block = ArenaLogFormatter.formatSPRTBlock(
            record: makeRecord(criterion: .sprt, verdict: nil))
        XCTAssertEqual(block.count, 1)
        XCTAssertTrue(block[0].contains("no verdict"))
        XCTAssertEqual(ArenaLogFormatter.formatSPRTKV(
            record: makeRecord(criterion: .sprt, verdict: nil)), "sprt=none ")
    }

    // MARK: The drain is reported, not hidden

    /// At `concurrency > 1` the verdict's tally and the tournament's differ.
    /// The block has to say which one carried the decision.
    func testDrainedGamesAreCalledOutWhenTheTalliesDiffer() throws {
        let config = try makeConfig()
        let record = makeRecord(
            criterion: .sprt,
            verdict: verdict(.accept, llr: 3.1, wins: 20, draws: 12, losses: 10, config: config),
            promoted: true,
            gamesPlayed: 60
        )
        let block = ArenaLogFormatter.formatSPRTBlock(record: record)
        XCTAssertTrue(block.contains { $0.contains("decided at game 42") })
        XCTAssertTrue(
            block.contains { $0.contains("18 further game(s) drained") },
            "60 played − 42 at decision = 18: \(block)"
        )
        XCTAssertTrue(block.contains { $0.contains("not evidence") })
    }

    func testNoDrainWordingWhenTheTalliesAgree() throws {
        let config = try makeConfig()
        let record = makeRecord(
            criterion: .sprt,
            verdict: verdict(.accept, llr: 3.1, wins: 20, draws: 12, losses: 10, config: config),
            promoted: true,
            gamesPlayed: 42
        )
        let block = ArenaLogFormatter.formatSPRTBlock(record: record)
        XCTAssertTrue(block.contains { $0.contains("decided at game 42") })
        XCTAssertFalse(block.contains { $0.contains("drained") })
    }

    // MARK: Hypotheses travel with the verdict

    func testBlockCarriesTheHypothesesSoTheLLRIsInterpretable() throws {
        let config = try makeConfig(elo0: -5, elo1: 25, alpha: 0.01, beta: 0.2, minGames: 64, maxGames: 0)
        let record = makeRecord(
            criterion: .sprt, verdict: verdict(.accept, llr: 4.5, config: config), promoted: true)
        let block = ArenaLogFormatter.formatSPRTBlock(record: record)
        let joined = block.joined(separator: "\n")

        XCTAssertTrue(joined.contains("H0: elo=-5"))
        XCTAssertTrue(joined.contains("H1: elo=+25"))
        XCTAssertTrue(joined.contains("alpha=0.0100"))
        XCTAssertTrue(joined.contains("beta=0.2000"))
        XCTAssertTrue(joined.contains("minGames=64"))
        XCTAssertTrue(joined.contains("maxGames=unbounded"), "0 must render as unbounded, not 0")
        XCTAssertTrue(joined.contains("bounds="), "the LLR is meaningless without its bounds")
    }

    func testKVCarriesEveryConfigFieldAndTheDecisionTally() throws {
        let config = try makeConfig(elo0: -5, elo1: 25, alpha: 0.01, beta: 0.2, minGames: 64, maxGames: 500)
        let record = makeRecord(
            criterion: .sprt,
            verdict: verdict(.accept, llr: 4.5, wins: 20, draws: 12, losses: 10, config: config),
            promoted: true
        )
        let kv = ArenaLogFormatter.formatKVLine(
            record: record, index: 1,
            candidateID: "C", championID: "M", trainerID: "T", buildNumber: 2400
        )
        for expected in [
            "sprt=accept", "sprt_llr=4.5000",
            "sprt_games=42", "sprt_w=20", "sprt_d=12", "sprt_l=10",
            "sprt_elo0=-5.0000", "sprt_elo1=25.0000",
            "sprt_alpha=0.0100", "sprt_beta=0.2000",
            "sprt_min_games=64", "sprt_max_games=500",
            "sprt_lower=", "sprt_upper="
        ] {
            XCTAssertTrue(kv.contains(expected), "missing \(expected) in: \(kv)")
        }
        XCTAssertFalse(kv.contains("\n"), "the KV line must stay one line")
    }

    /// A guard-fired verdict on an unscoreable record has no ratio. The KV
    /// line renders `nan` rather than omitting the key, so a parser sees the
    /// same column count on every SPRT row.
    func testUndefinedLLRRendersAsNanInKVAndIsNamedInTheBlock() throws {
        let config = try makeConfig(maxGames: 64)
        let record = makeRecord(
            criterion: .sprt,
            verdict: verdict(.inconclusive, llr: nil, wins: 64, draws: 0, losses: 0, config: config)
        )
        let kv = ArenaLogFormatter.formatKVLine(
            record: record, index: 1,
            candidateID: "C", championID: "M", trainerID: "T", buildNumber: 2400
        )
        XCTAssertTrue(kv.contains("sprt_llr=nan"))
        XCTAssertTrue(
            ArenaLogFormatter.formatSPRTBlock(record: record)
                .contains { $0.contains("zero-variance") },
            "the reason the ratio is undefined must be stated"
        )
    }

    // MARK: Elo formatting

    func testEloHypothesesRenderWholeWhenWholeAndSignedAlways() {
        XCTAssertEqual(ArenaLogFormatter.formatElo(0), "+0")
        XCTAssertEqual(ArenaLogFormatter.formatElo(10), "+10")
        XCTAssertEqual(ArenaLogFormatter.formatElo(-5), "-5")
        XCTAssertEqual(ArenaLogFormatter.formatElo(2.5), "+2.5")
        XCTAssertEqual(ArenaLogFormatter.formatElo(-2.5), "-2.5")
    }
}
