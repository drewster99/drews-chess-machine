import XCTest
@testable import DrewsChessMachine

/// Parsing and sparse semantics of the offline replay
/// (`TrainingHealthLogReplay`, the alarms plan D4).
final class TrainingHealthLogReplayTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private func options(flag: Bool = false) throws -> TrainingHealthLogReplay.Options {
        TrainingHealthLogReplay.Options(segmentStepAsTrainerStep: flag, config: try S.config())
    }

    private func replay(_ texts: [String], flag: Bool = false) throws -> TrainingHealthLogReplay.Output {
        try TrainingHealthLogReplay.run(
            texts.enumerated().map { TrainingHealthLogReplay.Source(name: "log\($0.offset).txt", text: $0.element) },
            options: try options(flag: flag))
    }

    private func parse(_ line: String) throws -> TrainingHealthLogReplay.ParsedLine {
        try TrainingHealthLogReplay.parseLine(TrainingHealthLogReplay.stripTimestamp(line), file: "t.txt", number: 7)
    }

    private let replayRow = "09:06:07.934  [REPLAY] step=50 loss=6.4783 pLoss=5.0650 vLoss=0.9002 pEnt=2.519 pIllM=0.5131 playedP=0.068 pW=0.47 pD=0.06 pL=0.46 vAbs=0.015 pLogitMean=-0.0173 vLogitMean=0.6120 gNorm=1.235 lr=0.5 ms=2665.3 buf=500000 plies=668192 games=10125 rejected=0 skipped=0 epoch=0 mom=0.8500 lrCyc[pk=1.0e+01,tr=1.0e-02] trainerStep=50"

    // MARK: Parsing

    func testParsesAReplayRow() throws {
        guard case .stepRow(let row) = try parse(replayRow) else { return XCTFail("not a step row") }
        XCTAssertEqual(row.segmentStep, 50)
        XCTAssertEqual(row.trainerStep, 50)
        XCTAssertEqual(row.loss, 6.4783)
        XCTAssertEqual(row.illegalMassPenalty, 0.5131)
        XCTAssertEqual(row.gradGlobalNorm, 1.235)
        XCTAssertEqual(row.policyLogitMean, -0.0173)
        XCTAssertEqual(row.learningRate, 0.5)
        XCTAssertEqual(row.momentum, 0.85)
        XCTAssertEqual(row.totalMs, 2665.3)
        XCTAssertTrue(row.absentFields.isEmpty)
        XCTAssertFalse(row.isTrainVsUci)
    }

    func testParsesATrainVsUciRowWhichCarriesNoIllegalMass() throws {
        let line = "10:00:00.000  [VS-UCI] step=50 loss=4.0000 pLoss=3.0000 vLoss=1.0000 pEnt=2.000 playedP=0.100 pLogitMean=0.1000 vLogitMean=0.2000 gNorm=0.700 lr=0.01 ms=900.0 buf=1000 mom=0.9000 trainerStep=1050"
        guard case .stepRow(let row) = try parse(line) else { return XCTFail("not a step row") }
        XCTAssertTrue(row.isTrainVsUci)
        XCTAssertEqual(row.trainerStep, 1050)
        XCTAssertNil(row.illegalMassPenalty)
        XCTAssertEqual(row.absentFields, ["pIllM"])
    }

    func testNotMeasuredIsNeverZero() throws {
        let line = "12:18:52.631  [REPLAY] step=1 loss=8.5902 pLoss=6.7620 vLoss=0.8880 pEnt=-- pIllM=0.9401 playedP=-- pW=-- pD=-- pL=-- vAbs=-- pLogitMean=-- vLogitMean=-- gNorm=0.071 lr=5.14 ms=3546.7 buf=500000 plies=759988 games=11453 rejected=0 skipped=0 epoch=0 mom=0.8500 lrCyc[pk=1.0e+01,tr=1.0e-02] trainerStep=514"
        guard case .stepRow(let row) = try parse(line) else { return XCTFail("not a step row") }
        XCTAssertNil(row.policyLogitMean)
        XCTAssertTrue(row.absentFields.isEmpty, "-- is present but not measured, not absent")
        XCTAssertEqual(row.illegalMassPenalty, 0.9401)
    }

    func testParsesALiveLine() throws {
        let line = "09:06:07.937  [LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=9/10 ch=1040 dead=12 off=2 alwaysOn=0 worst=value.bn(dead 12 off 2 on 0) maxBetaOverGamma=+0.33@policy.pre_bn[16] minBetaOverGamma=-106.97@value.bn[8] rvMaxOverMedian=29.8@stem.bn[31] rezero=none nonFinite=0"
        guard case .live(let step, let health) = try parse(line) else { return XCTFail("not a live line") }
        XCTAssertEqual(step, 50)
        XCTAssertEqual(health.reluClassifiedSiteCount, 9)
        XCTAssertEqual(health.reluChannelCount, 1040)
        XCTAssertEqual(health.reluDeadCount, 12)
        XCTAssertEqual(health.worstSite, "value.bn")
        XCTAssertEqual(health.worstSiteDead, 12)
        XCTAssertNil(health.parked)
        XCTAssertEqual(health.nonFiniteValueCount, 0)
        XCTAssertEqual(health.runningVariance, LayerHealthDigest.RunningVarianceRunaway(maxOverMedian: 29.8, site: "stem.bn"))
    }

    func testParsesALiveLineWithNoReluSitesAndParkedCounts() throws {
        let line = "[LAYER-HEALTH] live trainerStep=20700 scope=batch_norm_state_only reluSites=0/10 dead=n/a off=n/a alwaysOn=n/a (no relu/leaky_relu BN sites) rvMaxOverMedian=n/a rezero=none nonFinite=0 parkedSites=9/10 parkedCh=1040 parked=21 parkedOff=8 parkedBy=policy.pre_bn:20/128,value.bn:1/16"
        guard case .live(_, let health) = try parse(line) else { return XCTFail("not a live line") }
        XCTAssertEqual(health.reluClassifiedSiteCount, 0)
        XCTAssertNil(health.reluDeadCount)
        XCTAssertNil(health.runningVariance)
        let parked = try XCTUnwrap(health.parked)
        XCTAssertEqual(parked.siteCount, 9)
        XCTAssertEqual(parked.channelCount, 1040)
        XCTAssertEqual(parked.parkedCount, 21)
        XCTAssertEqual(parked.sites.map(\.site), ["policy.pre_bn", "value.bn"])
        XCTAssertEqual(parked.sites.map(\.parked), [20, 1])
        XCTAssertEqual(parked.sites.map(\.channels), [128, 16])
    }

    func testParsesACheckpointHeadline() throws {
        let line = "13:06:00.000  [LAYER-HEALTH] checkpoint replay-autosave step=1000 trainerStep=1513 scope=all_tensors reluSites=9/10 ch=1040 dead=350 off=108 alwaysOn=10 worst=policy.pre_bn(dead 91 off 16 on 0) rvMaxOverMedian=466888.2@value.bn[12] rezero=none nonFinite=0 seZeroVel=none valueFC1ZeroVel=128/128 maxAbs=3.582e+08@policy.pre_bn.running_var"
        guard case .checkpointHeadline(let context, let step, let trainerStep, let health) = try parse(line) else {
            return XCTFail("not a checkpoint headline")
        }
        XCTAssertEqual(context, "replay-autosave")
        XCTAssertEqual(step, 1000)
        XCTAssertEqual(trainerStep, 1513)
        XCTAssertEqual(health.valueFC1, LayerHealthDigest.ValueFC1Velocity(zeroVelocityUnitCount: 128, unitCount: 128))
        if case .checkpointFailed = try parse("[LAYER-HEALTH] checkpoint replay-final step=3 trainerStep=3 failed: layer health: tensor x is missing") {
        } else {
            XCTFail("a failed pass is not an observation")
        }
    }

    func testParsesTableRowsWithNotApplicableColumns() {
        let stem = TrainingHealthLogReplay.tableRow("stem.bn         -             128    n/a    n/a    n/a      0      0    -17.38 [93]        +22.85 [87]          243.2 [78]")
        XCTAssertEqual(stem, TrainingHealthLogReplay.TableRow(site: "stem.bn", activation: "-", channelCount: 128, deadCount: nil))
        let value = TrainingHealthLogReplay.tableRow("value.bn        relu           16     14      1      0      0      0   -153.63 [8]          -1.56 [5]        466888.2 [12]")
        XCTAssertEqual(value, TrainingHealthLogReplay.TableRow(site: "value.bn", activation: "relu", channelCount: 16, deadCount: 14))
        let silu = TrainingHealthLogReplay.tableRow("blocks.0.bn1    silu          128    n/a    n/a    n/a      0      0     -0.25 [5]          +0.12 [31]            1.6 [14]")
        XCTAssertEqual(silu?.deadCount, nil)
        XCTAssertEqual(silu?.activation, "silu")
    }

    func testMalformedLinesAreRejectedNamingTheLine() throws {
        let bad = "[REPLAY] step=50 loss=abc trainerStep=50"
        XCTAssertThrowsError(try replay([bad])) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError,
                           .malformedLine(file: "log0.txt", line: 1, reason: "loss=abc is not a number"))
        }
        XCTAssertThrowsError(try replay(["[LAYER-HEALTH] live trainerStep=50 worst=value.bn(dead 12 off"])) { error in
            guard case .malformedLine(let file, let line, _) = error as? TrainingHealthLogReplayError else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(file, "log0.txt")
            XCTAssertEqual(line, 1)
        }
        XCTAssertThrowsError(try replay(["[REPLAY] step=50 pIllM=0.5 trainerStep=50"])) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError,
                           .malformedLine(file: "log0.txt", line: 1, reason: "step row without loss="))
        }
    }

    func testRowsWithoutTrainerStepNeedTheFlag() throws {
        let rows = [
            "[RUN] path=replay build=2100 git=abc",
            "[REPLAY] step=1 loss=3.5312 pIllM=0.0029 gNorm=0.789 lr=0.0002 ms=3580.4",
            "[REPLAY] step=50 loss=3.4844 pIllM=0.0029 gNorm=0.805 lr=0.01 ms=3414.3",
        ].joined(separator: "\n")
        XCTAssertThrowsError(try replay([rows])) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError,
                           .unsupportedFormat(file: "log0.txt", build: "2100", missingField: "trainerStep"))
            XCTAssertEqual(error.localizedDescription,
                           "unsupported log format: log0.txt (build 2100 from its [APP]/[RUN] line): step rows carry no trainerStep")
        }
        XCTAssertNoThrow(try replay([rows], flag: true))
        let backwards = rows + "\n[REPLAY] step=40 loss=3.4 pIllM=0.003 gNorm=0.8 lr=0.01 ms=3000.0"
        XCTAssertThrowsError(try replay([backwards], flag: true)) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError, .nonIncreasingSegmentSteps(file: "log0.txt", line: 4))
        }
    }

    func testLegacyRowsReportAbsentFieldsAsNoData() throws {
        let rows = (1...10).map { index in
            "[REPLAY] step=\(index * 50) loss=3.5 pIllM=0.003 gNorm=0.8 lr=0.01 ms=3000.0 trainerStep=\(index * 50)"
        }.joined(separator: "\n")
        let output = try replay([rows])
        XCTAssertTrue(output.header.contains("log0.txt: pLogitMean absent in 10 of 10 rows: policy_offset_drift no data on them"),
                      output.header.joined(separator: "\n"))
        XCTAssertTrue(output.events.isEmpty)
        let final = try XCTUnwrap(output.lines.last?.text)
        XCTAssertTrue(final.contains("policy_offset_drift:10"), final)
    }

    func testLogWithNeitherRowsNorLayerHealthIsRefused() {
        XCTAssertThrowsError(try replay(["[REPLAY] starting offline corpus replay over 1 corpus path(s)"])) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError, .noData(file: "log0.txt"))
        }
    }

    func testConflictingChannelCountsWithinOneRunAreRefused() {
        let text = [
            "[RUN] path=replay build=1",
            "[LAYER-HEALTH] checkpoint replay-autosave step=1000 trainerStep=1000 reluSites=1/1 ch=16 dead=0 nonFinite=0",
            "[LAYER-HEALTH]   batch-norm sites — β/|γ|: dead < -3.00, mostly off < -2.00, always on > +3.00; classified for relu/leaky_relu only",
            "[LAYER-HEALTH]     site            act            ch   dead    off     on  zeroγ nonfin  min β/|γ| [ch]     max β/|γ| [ch]     rv max/median [ch]",
            "[LAYER-HEALTH]     value.bn        relu           16      0      0      0      0      0     -1.00 [1]          +1.00 [2]              1.0 [3]",
            "[LAYER-HEALTH] checkpoint replay-autosave step=2000 trainerStep=2000 reluSites=1/1 ch=32 dead=0 nonFinite=0",
            "[LAYER-HEALTH]   batch-norm sites — β/|γ|: dead < -3.00, mostly off < -2.00, always on > +3.00; classified for relu/leaky_relu only",
            "[LAYER-HEALTH]     site            act            ch   dead    off     on  zeroγ nonfin  min β/|γ| [ch]     max β/|γ| [ch]     rv max/median [ch]",
            "[LAYER-HEALTH]     value.bn        relu           32      0      0      0      0      0     -1.00 [1]          +1.00 [2]              1.0 [3]",
        ].joined(separator: "\n")
        XCTAssertThrowsError(try replay([text])) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError,
                           .conflictingChannelCounts(site: "value.bn", first: "log0.txt:5 (16)", second: "log0.txt:9 (32)"))
        }
    }

    func testExcerptHeaderLinesAreSkippedOnlyInAnExcerpt() throws {
        let excerpt = [
            TrainingHealthLogReplay.excerptHeaderMarker,
            "# source: x.txt",
            "[REPLAY] step=50 loss=3.5 pIllM=0.003 gNorm=0.8 lr=0.01 ms=3000.0 trainerStep=50",
        ].joined(separator: "\n")
        XCTAssertNoThrow(try replay([excerpt]))
        let file = try TrainingHealthLogReplay.parse(TrainingHealthLogReplay.Source(name: "x", text: excerpt))
        XCTAssertEqual(file.lines.count, 1)
        let notExcerpt = try TrainingHealthLogReplay.parse(TrainingHealthLogReplay.Source(
            name: "y", text: "# not a marker\n[REPLAY] step=50 loss=3.5 trainerStep=50"))
        XCTAssertEqual(notExcerpt.lines.count, 2, "a session log's # line is just another line")
    }

    // MARK: Sparse semantics

    func testLiveLineUsesTheRunsCheckpointChannelCounts() throws {
        // The live line at 50 names value.bn with 12 dead; only the table
        // later in the same run knows value.bn has 16 channels.
        let text = [
            "[RUN] path=replay build=2323",
            "[REPLAY] step=50 loss=6.4 pIllM=0.5 gNorm=1.2 pLogitMean=-0.01 lr=0.5 ms=2000.0 mom=0.85 trainerStep=50",
            "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=9/10 ch=1040 dead=12 off=2 alwaysOn=0 worst=value.bn(dead 12 off 2 on 0) rvMaxOverMedian=29.8@stem.bn[31] rezero=none nonFinite=0",
            "[LAYER-HEALTH] checkpoint replay-abort step=513 trainerStep=513 scope=all_tensors reluSites=9/10 ch=1040 dead=339 off=101 alwaysOn=5 rvMaxOverMedian=466888.2@value.bn[12] rezero=none nonFinite=0 seZeroVel=none valueFC1ZeroVel=0/128",
            "[LAYER-HEALTH]   batch-norm sites — β/|γ|: dead < -3.00, mostly off < -2.00, always on > +3.00; classified for relu/leaky_relu only",
            "[LAYER-HEALTH]     site            act            ch   dead    off     on  zeroγ nonfin  min β/|γ| [ch]     max β/|γ| [ch]     rv max/median [ch]",
            "[LAYER-HEALTH]     value.bn        relu           16     14      1      0      0      0   -153.63 [8]          -1.56 [5]        466888.2 [12]",
            "[LAYER-HEALTH]   FC hidden-unit velocity — zero: every weight-velocity entry exactly 0; low: nonzero but < 5% of the layer's p90 unit norm",
            "[LAYER-HEALTH]     value.fc1.weight  relu  units 128  zero 0  low 62  median unit norm 2.459e-17",
        ].joined(separator: "\n")
        let output = try replay([text])
        let raise = try XCTUnwrap(output.events.first { $0.rule == .deadChannels && $0.kind == .raise })
        XCTAssertEqual(raise.trainerStep, 50)
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertEqual(raise.detail, "sites=value.bn(12/16) coverage=relu_leaky_relu_only")
        XCTAssertTrue(output.lines.first?.text.hasSuffix("value_fc1_zero_velocity=applies") == true)
    }

    func testParkedFieldsGiveEverySiteAndEveryActivation() throws {
        let text = [
            "[RUN] path=replay build=2400",
            "[REPLAY] step=50 loss=3.9 pIllM=0.005 gNorm=0.4 pLogitMean=0.1 lr=0.5 ms=2000.0 mom=0.85 trainerStep=21050",
            "[LAYER-HEALTH] live trainerStep=21050 scope=batch_norm_state_only reluSites=2/10 ch=144 dead=21 off=8 alwaysOn=0 worst=policy.pre_bn(dead 20 off 7 on 0) rvMaxOverMedian=12.0@blocks.2.bn1[76] rezero=none nonFinite=0 parkedSites=9/10 parkedCh=1040 parked=24 parkedOff=9 parkedBy=blocks.2.bn1:3/128,policy.pre_bn:20/128,value.bn:1/16",
        ].joined(separator: "\n")
        let output = try replay([text])
        let raise = try XCTUnwrap(output.events.first { $0.rule == .deadChannels })
        XCTAssertEqual(raise.value, "dead=24/1040")
        XCTAssertEqual(raise.detail, "sites=policy.pre_bn(20/128),value.bn(1/16),blocks.2.bn1(3/128)")
        XCTAssertEqual(raise.severity, .warning, "24/1040 = 2.3%; the largest site is 15.6%")
    }

    func testSegmentOneAloneHasNoDiagnosticFields() throws {
        let rows = (0..<5).map { index in
            "[REPLAY] step=\(index * 50 + 1) loss=8.5 pEnt=-- pIllM=0.94 pLogitMean=-- vLogitMean=-- gNorm=0.5 lr=5 ms=1900.0 mom=0.85 trainerStep=\(514 + index * 50)"
        }.joined(separator: "\n")
        let output = try replay(["[RUN] path=replay build=2323\n" + rows])
        let final = try XCTUnwrap(output.lines.last?.text)
        XCTAssertTrue(final.contains("policy_offset_drift:5"), final)
    }

    // MARK: The overall arm's denominator (review M1)

    private let batchNormTableHeader = [
        "[LAYER-HEALTH]   batch-norm sites — β/|γ|: dead < -3.00, mostly off < -2.00, always on > +3.00; classified for relu/leaky_relu only",
        "[LAYER-HEALTH]     site            act            ch   dead    off     on  zeroγ nonfin  min β/|γ| [ch]     max β/|γ| [ch]     rv max/median [ch]",
    ]

    func testOverallArmDividesByEveryActivatedChannelOfTheRunsTable() throws {
        // A mixed tower logged before the parked counts: the live line counts
        // the two leaky_relu sites only (144 channels); the run's table says
        // activations consume 128 + 128 + 16 = 272 channels (stem.bn feeds
        // none). 12/272 = 4.4% overall and 12/128 = 9.4% at the site: a
        // warning, not the 12/144 = 8.3% critical.
        let text = ([
            "[RUN] path=replay build=2330",
            "[REPLAY] step=50 loss=3.9 pIllM=0.005 gNorm=0.4 pLogitMean=0.1 lr=0.5 ms=2000.0 mom=0.85 trainerStep=50",
            "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=2/4 ch=144 dead=12 off=0 alwaysOn=0 worst=policy.pre_bn(dead 12 off 0 on 0) rvMaxOverMedian=3.0@stem.bn[1] rezero=none nonFinite=0",
            "[LAYER-HEALTH] checkpoint replay-autosave step=1000 trainerStep=1000 scope=all_tensors reluSites=2/4 ch=144 dead=12 off=0 alwaysOn=0 rvMaxOverMedian=3.0@stem.bn[1] rezero=none nonFinite=0 seZeroVel=none valueFC1ZeroVel=0/128",
        ] + batchNormTableHeader + [
            "[LAYER-HEALTH]     stem.bn         -             128    n/a    n/a    n/a      0      0     -1.00 [1]          +1.00 [2]              3.0 [1]",
            "[LAYER-HEALTH]     blocks.0.bn1    silu          128    n/a    n/a    n/a      0      0     -1.00 [1]          +1.00 [2]              1.0 [3]",
            "[LAYER-HEALTH]     policy.pre_bn   leaky_relu    128     12      0      0      0      0     -9.00 [1]          +1.00 [2]              1.0 [3]",
            "[LAYER-HEALTH]     value.bn        leaky_relu     16      0      0      0      0      0     -1.00 [1]          +1.00 [2]              1.0 [3]",
        ]).joined(separator: "\n")
        let output = try replay([text])
        let raise = try XCTUnwrap(output.events.first { $0.rule == .deadChannels && $0.kind == .raise })
        XCTAssertEqual(raise.trainerStep, 50)
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.value, "dead=12/272")
    }

    func testOverallArmHasNoDataWhenTheRunHasNoChannelTable() throws {
        // No checkpoint table in the run: the denominator of every activated
        // channel is unknown, so the overall arm has no data (never 20/144).
        // policy.pre_bn's channel count is unknown too, so only the warning
        // (any parked channel) can be judged.
        let text = [
            "[RUN] path=replay build=2330",
            "[REPLAY] step=50 loss=3.9 pIllM=0.005 gNorm=0.4 pLogitMean=0.1 lr=0.5 ms=2000.0 mom=0.85 trainerStep=50",
            "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=2/10 ch=144 dead=20 off=12 alwaysOn=0 worst=policy.pre_bn(dead 19 off 7 on 0) rvMaxOverMedian=3.0@stem.bn[1] rezero=none nonFinite=0",
        ].joined(separator: "\n")
        let output = try replay([text])
        let raise = try XCTUnwrap(output.events.first { $0.rule == .deadChannels && $0.kind == .raise })
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.value, "dead=20/--")
        XCTAssertEqual(raise.detail, "sites=policy.pre_bn(19/?) coverage=relu_leaky_relu_only")
    }

    func testConflictingActivationsWithinOneRunAreRefused() {
        // The pre-scan's activation per site is the overall denominator's
        // source, so two answers within one run are refused, not merged.
        let text = ([
            "[RUN] path=replay build=1",
            "[LAYER-HEALTH] checkpoint replay-autosave step=1000 trainerStep=1000 reluSites=1/1 ch=16 dead=0 nonFinite=0",
        ] + batchNormTableHeader + [
            "[LAYER-HEALTH]     value.bn        relu           16      0      0      0      0      0     -1.00 [1]          +1.00 [2]              1.0 [3]",
            "[LAYER-HEALTH] checkpoint replay-autosave step=2000 trainerStep=2000 reluSites=0/1 dead=n/a nonFinite=0",
        ] + batchNormTableHeader + [
            "[LAYER-HEALTH]     value.bn        silu           16    n/a    n/a    n/a      0      0     -1.00 [1]          +1.00 [2]              1.0 [3]",
        ]).joined(separator: "\n")
        XCTAssertThrowsError(try replay([text])) { error in
            XCTAssertEqual(error as? TrainingHealthLogReplayError,
                           .conflictingActivations(site: "value.bn", first: "log0.txt:5 (relu)", second: "log0.txt:9 (silu)"))
        }
    }

    // MARK: The dedicated value-FC1 line (review M3)

    func testValueFC1LinesCarryTheTrainedCountSoGuiLogsReplayRule3() throws {
        // A GUI log has no step rows; its dedicated value-FC1 reads carry the
        // steps this process trained, which is rule 3's gate.
        let text = [
            "[RUN] path=gui build=2400",
            "[LAYER-HEALTH] value-fc1 trainerStep=5000 trained=150 valueFC1ZeroVel=128/128 lowVel=0 readMs=1.00 summaryMs=0.50",
            "[LAYER-HEALTH] value-fc1 trainerStep=6000 trained=1150 valueFC1ZeroVel=128/128 lowVel=0 readMs=1.00 summaryMs=0.50",
        ].joined(separator: "\n")
        let output = try replay([text])
        let raise = try XCTUnwrap(output.events.first { $0.rule == .valueFC1ZeroVelocity && $0.kind == .raise })
        XCTAssertEqual(raise.trainerStep, 6000, "150 trained is below the 200-step gate; 1,150 is past it")
        XCTAssertEqual(raise.severity, .critical)
    }

    func testValueFC1LineWithoutTheTrainedCountIsMalformed() {
        XCTAssertThrowsError(try parse(
            "[LAYER-HEALTH] value-fc1 trainerStep=5000 valueFC1ZeroVel=0/128 lowVel=0 readMs=1.00 summaryMs=0.50")) { error in
            guard case .malformedLine(_, let line, _) = error as? TrainingHealthLogReplayError else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(line, 7)
        }
    }

    func testHeaderNamesRule3sOnlySourceInALogWithoutStepRows() throws {
        let text = [
            "[RUN] path=gui build=2400",
            "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=9/10 ch=1040 dead=0 off=0 alwaysOn=0 worst=none rvMaxOverMedian=3.0@stem.bn[1] rezero=none nonFinite=0",
        ].joined(separator: "\n")
        let output = try replay([text])
        XCTAssertTrue(output.header.contains(
            "log0.txt: no [REPLAY]/[VS-UCI] step rows: layer-health rules only (non_finite, dead_channels, bn_running_variance_runaway, bn_running_variance_jump); value_fc1_zero_velocity only from [LAYER-HEALTH] value-fc1 lines and replay-/vsuci- checkpoints (0 value-fc1 lines in this log)"),
            output.header.joined(separator: "\n"))
    }

    func testLogsPassedTogetherAreOneEvaluatorAndLaterRunsAreFresh() throws {
        let first = "[RUN] path=replay build=1\n[REPLAY] step=50 loss=3 pIllM=0.4 gNorm=1 ms=1.0 trainerStep=50\n[REPLAY] step=100 loss=3 pIllM=0.9 gNorm=1 ms=1.0 trainerStep=100"
        let second = "[RUN] path=replay build=1\n[REPLAY] step=1 loss=3 pIllM=0.9 gNorm=1 ms=1.0 trainerStep=150"
        let together = try replay([first, second])
        XCTAssertEqual(together.events.filter { $0.rule == .illegalMass }.map(\.trainerStep), [150],
                       "the second log continues the first's sustain and running minimum")
        let fresh = try replay([first + "\n" + second])
        XCTAssertTrue(fresh.events.filter { $0.rule == .illegalMass }.isEmpty, "a second [RUN] in one log starts a fresh evaluator")
    }

    // MARK: bn_running_variance_jump offline (BN_RUNNING_VARIANCE_CHANGE_ALARM_PLAN D5)

    func testLiveLineKeepsTheLargestChannelAndTheOutlierCount() throws {
        let line = "[LAYER-HEALTH] live trainerStep=19800 scope=batch_norm_state_only reluSites=2/10 ch=144 dead=0 off=0 alwaysOn=0 worst=none rvMaxOverMedian=154.0@blocks.2.bn1[76] rvOver10xMedian=6 rezero=none nonFinite=0"
        guard case .live(_, let health) = try parse(line) else { return XCTFail("not a live line") }
        XCTAssertEqual(health.runningVariance, LayerHealthDigest.RunningVarianceRunaway(maxOverMedian: 154, site: "blocks.2.bn1"))
        XCTAssertEqual(health.runningVarianceLargestChannel, 76)
        XCTAssertEqual(health.runningVarianceOutlierCount, 6)
        XCTAssertEqual(health.runningVarianceChannels, LayerHealthDigest.RunningVarianceChannels(
            coverage: .largestOnly(site: "blocks.2.bn1", channel: 76, ratio: 154), outlierCount: 6))
    }

    func testLiveLineWithoutTheOutlierFieldHasNoCount() throws {
        let line = "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=9/10 ch=1040 dead=0 off=0 alwaysOn=0 worst=none rvMaxOverMedian=3.0@stem.bn[1] rezero=none nonFinite=0"
        guard case .live(_, let health) = try parse(line) else { return XCTFail("not a live line") }
        XCTAssertEqual(health.runningVarianceLargestChannel, 1)
        XCTAssertNil(health.runningVarianceOutlierCount)
        let noRatio = "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=0/10 dead=n/a off=n/a alwaysOn=n/a (no relu/leaky_relu BN sites) rvMaxOverMedian=n/a rvOver10xMedian=0 rezero=none nonFinite=0"
        guard case .live(_, let empty) = try parse(noRatio) else { return XCTFail("not a live line") }
        XCTAssertNil(empty.runningVarianceChannels, "no largest channel: rule 14 has no data on the line")
        XCTAssertEqual(empty.runningVarianceOutlierCount, 0)
    }

    func testMalformedRunningVarianceChannelIsRejected() {
        XCTAssertThrowsError(try parse(
            "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only rvMaxOverMedian=3.0@stem.bn[x] rezero=none nonFinite=0"))
    }

    func testHeaderStatesTheRunningVarianceJumpLowerBound() throws {
        let text = [
            "[RUN] path=gui build=2400",
            "[LAYER-HEALTH] live trainerStep=50 scope=batch_norm_state_only reluSites=9/10 ch=1040 dead=0 off=0 alwaysOn=0 worst=none rvMaxOverMedian=3.0@stem.bn[1] rezero=none nonFinite=0",
            "[LAYER-HEALTH] live trainerStep=100 scope=batch_norm_state_only reluSites=9/10 ch=1040 dead=0 off=0 alwaysOn=0 worst=none rvMaxOverMedian=3.0@stem.bn[1] rvOver10xMedian=0 rezero=none nonFinite=0",
        ].joined(separator: "\n")
        let output = try replay([text])
        XCTAssertTrue(output.header.contains(
            "log0.txt: bn_running_variance_jump: the jump arm sees only each live line's largest channel (a lower bound on the app's); the outlier-count arm has no data on 1 of 2 live lines (no rvOver10xMedian=)"),
            output.header.joined(separator: "\n"))
    }

    /// Offline, only the largest channel can jump, against the past lines'
    /// largest ratios; the count arm reads `rvOver10xMedian=`.
    func testOfflineReplayRaisesTheJumpFromLiveLines() throws {
        func live(_ step: Int, _ ratio: String, _ outliers: Int) -> String {
            "[LAYER-HEALTH] live trainerStep=\(step) scope=batch_norm_state_only reluSites=2/10 ch=144 dead=0 off=0 alwaysOn=0 worst=none rvMaxOverMedian=\(ratio) rvOver10xMedian=\(outliers) rezero=none nonFinite=0"
        }
        let text = (["[RUN] path=gui build=2400"]
            + stride(from: 2000, through: 2900, by: 100).map { live($0, "60.0@blocks.2.bn1[31]", 5) }
            + [live(3000, "700.0@blocks.2.bn1[76]", 11)]).joined(separator: "\n")
        let events = try replay([text]).events.filter { $0.rule == .batchNormRunningVarianceJump }
        let raise = try XCTUnwrap(events.first)
        XCTAssertEqual(raise.kind, .raise)
        XCTAssertEqual(raise.trainerStep, 3000)
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertEqual(raise.value, "jumped=1 outliers=11/5")
        XCTAssertEqual(raise.threshold, "jump>=10xmin&ratio>=100|outliers>=2xmin&+5")
        XCTAssertEqual(raise.detail, "channels=blocks.2.bn1[76]:<=60.00->700.0")
    }
}
