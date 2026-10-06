//
//  ResumeEquivalenceTests.swift
//  DrewsChessMachineTests
//
//  The correctness gate for exact resume (determinism plan C6 / P12): a corpus
//  replay run trained N+M steps without stopping must end in the same state as
//  the same run trained N steps, saved, and continued M steps with
//  `--resume-exact` in a fresh trainer, buffer and feeder — the way a new
//  process would. Both sides run the real replay loop
//  (`CorpusReplayRunner.runReplay`) over a small synthetic corpus written for
//  the test, from one small start model, under one configured run seed, and
//  the files they end with are compared:
//
//  - every tensor (fp32 masters, batch-norm statistics, optimizer velocity),
//    through the data region's SHA-256 when the determinism probe finds a
//    training step bit-reproducible, else tensor by tensor within the
//    probe's tolerance;
//  - the run's stream positions: the replay sampler, the trainer's dropout
//    stream and its Philox state — equal exactly in either mode, because they
//    advance by draw count, not by arithmetic;
//  - the corpus feed position (epoch, next game, feed phase) and the
//    cumulative step, game and position totals;
//  - the resumed segment records itself as an exact resume with no gaps.
//
//  Equal stream positions after the same number of steps, from the same seed,
//  mean every step drew the same number of values; with the feed position and
//  the final weights also equal, every step drew the same batch indices and
//  the same dropout masks — a different batch or mask anywhere would move the
//  weights. Per-step index and mask equality after a buffer refill is also
//  pinned directly, without the GPU, by `ReplayBufferResumeEquivalenceTests`.
//
//  Variants: a resume inside the first epoch at a KL-probe step; a resume early
//  in the second epoch (the refeed window reaches back across the wrap); the
//  same under a two-pass epoch budget instead of a step limit; and probes on
//  versus off (a probe must not move any stream or weight). Two resumes that
//  cannot be exact are refused rather than run: one whose epoch budget the
//  checkpoint's run has already spent, and one whose rebuilt buffer would
//  not hold what the saved run's held (a run started part-way into the
//  corpus that saved before its buffer filled). Replay samples under the
//  run's sampling constraints, so the in-epoch resume is also run with the
//  declared defaults, with every constraint binding, and with material
//  stratification.
//
//  Not covered here, and why: a resume refused for a gap (a parameter change,
//  a pre-lineage file) throws a `CLIRunRefusal` out of `runReplay` — pinned
//  in-process by `CorpusReplayRefusalTests`, and at the decision level (the
//  gap list) by `ExactResumeCompletionTests`; the GUI and train-vs-UCI state
//  round trips (no trajectory claim, D-1) are pinned by `ExactResumeCompletionTests`,
//  `GuiResumeGapsTests`, `SessionSaveReplayBufferTests` and
//  `RunObservabilityResumeTests`. `scripts/resume_equivalence.sh` runs the
//  same comparison through the shipped binary on a real corpus.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ResumeEquivalenceTests: XCTestCase {

    private var tempDir: URL!
    private var corpusDir: URL!
    private var startModelURL: URL!

    /// The run seed both sides are configured with.
    private static let runSeed: UInt64 = 0xD5C0_FFEE

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-resume-equivalence-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
        corpusDir = try Self.writeCorpus(in: dir)
        startModelURL = try await Self.writeStartModel(in: dir)
    }

    override func tearDown() async throws {
        if let tempDir {
            do { try FileManager.default.removeItem(at: tempDir) } catch { /* best-effort cleanup */ }
        }
        try await super.tearDown()
    }

    // MARK: - Fixtures

    /// The smallest architecture that still has every part a resume must
    /// carry: a residual block with ReZero and an SE block, batch norm,
    /// dropout sites, both heads. fp32, so the determinism probe can find a
    /// training step bit-reproducible.
    static let architecture = NetworkArchitecture(
        inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
        activationFunction: .relu, blockActivationStyle: .pre,
        blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
        blockConv1KernelSize: 3, blockConv2KernelSize: 3,
        blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
        policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
        valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
        computeDataType: .float32
    )

    /// Games in the synthetic corpus, and their length range. With the
    /// replay parameters below an epoch lasts a few dozen steps, so an epoch
    /// wrap is reachable in a short test.
    static let corpusGames = 80
    static let gamePlyRange = 24...56

    /// Deterministic pseudo-random legal games from the starting position (a
    /// fixed LCG picks each move and each game's length), ended early at mate
    /// or stalemate, recorded into a sealed corpus split over several shards.
    static func writeCorpus(in dir: URL) throws -> URL {
        var lcg: UInt64 = 0x5EED_0F_C0_7215
        func next() -> UInt64 {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return lcg >> 33
        }
        var games: [GameRecord] = []
        for gameIndex in 0..<corpusGames {
            let span = UInt64(gamePlyRange.count)
            let targetPlies = gamePlyRange.lowerBound + Int(next() % span)
            var state = GameState.starting
            var moves: [ChessMove] = []
            var reason: GameTerminationReason = .maxPlies
            while moves.count < targetPlies {
                let legal = MoveGenerator.legalMoves(for: state)
                if legal.isEmpty {
                    reason = .checkmate
                    break
                }
                let move = legal[Int(next() % UInt64(legal.count))]
                moves.append(move)
                state = MoveGenerator.applyMove(move, to: state)
            }
            let outcome: GameOutcome = [.whiteWin, .draw, .blackWin][gameIndex % 3]
            games.append(GameRecord(moves: moves, outcome: outcome, terminationReason: reason))
        }
        let corpus = try GameCorpus.create(name: "resume-equivalence",
                                           comment: "synthetic games for ResumeEquivalenceTests",
                                           shardSoftLimitBytes: 4096,
                                           parentDirectory: dir)
        try corpus.beginSource(kind: "selfPlay")
        for game in games { try corpus.append(game) }
        try corpus.seal()
        return corpus.directory
    }

    /// A start model of `architecture` with seeded weights and no trainer
    /// state: every run below begins as a new branch from it.
    static func writeStartModel(in dir: URL) async throws -> URL {
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 11), arch: architecture)
        let weights = try await net.network.exportWeights()
        let data = try SafetensorsModelIO.encode(
            modelID: "20261003-1-RQST",
            createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "",
                                              notes: "resume-equivalence start model"),
            weights: weights,
            architecture: architecture,
            includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let url = dir.appendingPathComponent("start.safetensors")
        try data.write(to: url, options: .withoutOverwriting)
        return url
    }

    /// The four sampling parameters a run samples under.
    struct Sampling {
        let maxPerGame: Int
        let maxDrawPercent: Int
        let targetLength: Int
        let stratify: Bool

        /// Their declared defaults.
        static let declared = Sampling(
            maxPerGame: MaxPliesFromAnyOneGame.declaredDefault,
            maxDrawPercent: MaxDrawPercentPerBatch.declaredDefault,
            targetLength: TargetSampledGameLengthPlies.declaredDefault,
            stratify: ReplayBufferStratifyByMaterial.declaredDefault)

        var parameterValues: [String: ParameterValue] {
            [
                MaxPliesFromAnyOneGame.id: MaxPliesFromAnyOneGame.encode(maxPerGame),
                MaxDrawPercentPerBatch.id: MaxDrawPercentPerBatch.encode(maxDrawPercent),
                TargetSampledGameLengthPlies.id: TargetSampledGameLengthPlies.encode(targetLength),
                ReplayBufferStratifyByMaterial.id: ReplayBufferStratifyByMaterial.encode(stratify),
            ]
        }
    }

    /// The run parameters: every parameter at its declared default — never
    /// anyone's saved settings — with `sampling` applied and the knobs this
    /// test depends on pinned: a small batch and buffer, dropout on, a short
    /// warmup, and the KL probe and batch-stats diagnostics either both on at
    /// a short cadence or both off.
    func replayParams(probesOn: Bool, sampling: Sampling) throws -> ReplayParams {
        let pinned: [String: ParameterValue] = [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            ReplayRatioTarget.id: .double(0.48),
            DropoutRate.id: .double(0.1),
            LRWarmupSteps.id: .int(5),
            KLProbeInterval.id: .int(probesOn ? 3 : 0),
            BatchStatsInterval.id: .int(probesOn ? 5 : 0),
        ]
        return try ReplayParams(TrainingParametersSnapshot.declaredDefaults(
            overriding: sampling.parameterValues.merging(pinned) { _, pinnedValue in pinnedValue }))
    }

    private func config(stepLimit: Int, startModel: URL, resumeExact: Bool, out: String) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpusDir],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: resumeExact,
            acceptInexact: [],
            outModelPath: tempDir.appendingPathComponent(out).path,
            overwriteOutModel: false,
            runModelID: "20261003-2-RQEV",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: Self.runSeed, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    /// A run configuration with the budget and start position of the
    /// caller's choosing: a step limit, an epoch budget (`--epochs`), both
    /// or neither (a single pass), and an optional `--start-game-index`.
    private func config(stepLimit: Int?, epochs: Int?, startGameIndex: Int?, startModel: URL, resumeExact: Bool,
                        out: String) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpusDir],
            stepLimit: stepLimit,
            epochs: epochs,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: startGameIndex,
            resumeExact: resumeExact,
            acceptInexact: [],
            outModelPath: tempDir.appendingPathComponent(out).path,
            overwriteOutModel: false,
            runModelID: "20261003-2-RQEV",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: Self.runSeed, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    /// Run the real replay loop and return the file it ended with.
    private func run(stepLimit: Int, from startModel: URL, resumeExact: Bool, out: String,
                     probesOn: Bool = true, sampling: Sampling
    ) async throws -> (url: URL, result: CorpusReplayRunner.Result) {
        let cfg = config(stepLimit: stepLimit, startModel: startModel, resumeExact: resumeExact, out: out)
        let result = try await CorpusReplayRunner.runReplay(
            config: cfg, params: try replayParams(probesOn: probesOn, sampling: sampling), abort: ReplayAbortFlag())
        XCTAssertEqual(result.steps, stepLimit, "\(out): the run trains its whole step limit")
        return (tempDir.appendingPathComponent(out), result)
    }

    // MARK: - Determinism probe

    /// How two runs' tensors must compare: bit for bit when one training step
    /// from identical state reproduces itself on this machine, otherwise
    /// within a relative tolerance.
    enum TensorAgreement: Equatable {
        case bitExact
        case tolerance(relative: Float)
    }

    /// Two trainers built from the same weights, dropout stream and batch
    /// source take one real-data step each; their exported state (masters,
    /// statistics, velocity) decides whether the GPU path is bit-reproducible
    /// here (plan C6 step 1).
    private func determinismProbe() async throws -> TensorAgreement {
        let start = try CheckpointManager.loadModelFile(at: startModelURL)
        let p = try replayParams(probesOn: true, sampling: .declared)
        var exports: [[[Float]]] = []
        for _ in 0..<2 {
            let trainer = try ChessTrainer(
                dropoutStream: DCMRandom(seed: 3), hyperparameters: p.trainer,
                arch: Self.architecture, initialization: .overwrittenByLoad)
            try await trainer.loadBaseWeightsResetVelocity(start.networkWeights)
            let buffer = try Self.fixtureBuffer(sampler: DCMRandom(seed: 5))
            let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: p.trainingBatchSize)
            XCTAssertNotNil(timing, "the probe buffer holds more than a batch, so the step trains")
            exports.append(try await trainer.exportTrainerWeights())
        }
        let identical = exports[0].count == exports[1].count
            && zip(exports[0], exports[1]).allSatisfy { $0.map(\.bitPattern) == $1.map(\.bitPattern) }
        let agreement: TensorAgreement = identical ? .bitExact : .tolerance(relative: 1.0e-6)
        print("[RESUME-EQUIV] determinism probe: \(agreement)")
        return agreement
    }

    /// A buffer of real positions from the synthetic corpus's games, fed
    /// through the same feeder the replay runner uses.
    static func fixtureBuffer(sampler: DCMRandom) throws -> ReplayBuffer {
        let buffer = ReplayBuffer(capacity: 512, inputEncoding: architecture.inputEncoding, sampler: sampler)
        var lcg: UInt64 = 0x0FEE_D0_5EED
        func next() -> UInt64 {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return lcg >> 33
        }
        let encoding = architecture.inputEncoding
        let floatsPerBoard = BoardEncoder.tensorLength(for: encoding)
        let count = 512
        var boards = [Float](repeating: 0, count: count * floatsPerBoard)
        var moves = [Int32](repeating: 0, count: count)
        var plies = [UInt16](repeating: 0, count: count)
        let taus = [Float](repeating: 1.0, count: count)
        var hashes = [UInt64](repeating: 0, count: count)
        let materials = [UInt8](repeating: 32, count: count)
        var outcomes = [Float](repeating: 0, count: count)
        var state = GameState.starting
        var ply = 0
        var filled = 0
        while filled < count {
            let legal = MoveGenerator.legalMoves(for: state)
            if legal.isEmpty || ply >= 60 {
                state = GameState.starting
                ply = 0
                continue
            }
            let move = legal[Int(next() % UInt64(legal.count))]
            boards.replaceSubrange(filled * floatsPerBoard ..< (filled + 1) * floatsPerBoard,
                                   with: BoardEncoder.encode(state, encoding: encoding))
            moves[filled] = Int32(PolicyEncoding.policyIndex(move, currentPlayer: state.currentPlayer))
            plies[filled] = UInt16(ply)
            hashes[filled] = next()
            outcomes[filled] = Float(Int(next() % 3)) - 1.0
            state = MoveGenerator.applyMove(move, to: state)
            ply += 1
            filled += 1
        }
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { pl in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { mc in
        outcomes.withUnsafeBufferPointer { o in
            guard let bb = b.baseAddress, let mb = m.baseAddress, let pb = pl.baseAddress,
                  let tb = t.baseAddress, let hb = h.baseAddress, let mcb = mc.baseAddress,
                  let ob = o.baseAddress else {
                preconditionFailure("fixture arrays are non-empty, so every base address exists")
            }
            buffer.append(boards: bb, policyIndices: mb, plyIndices: pb, samplingTaus: tb,
                          stateHashes: hb, materialCounts: mcb, gameLength: UInt16(count),
                          workerId: 0, intraWorkerGameIndex: 0, outcomes: ob, count: count)
        }}}}}}}
        return buffer
    }

    // MARK: - Comparison

    /// What a finished run's file says about where the run stands.
    private struct EndState {
        let contentSHA256: String
        let tensors: [[Float]]
        let record: LineageRecord
        let streams: LineageRecord.RunStreams
        let corpus: LineageRecord.CorpusPosition
    }

    private func endState(_ url: URL, _ what: String) throws -> EndState {
        let file = try CheckpointManager.loadModelFileAsStored(at: url)
        guard let provenance = file.safetensorsProvenance, let sha = provenance.contentSHA256 else {
            XCTFail("\(what): a replay save always stamps its content hash")
            throw CocoaError(.fileReadCorruptFile)
        }
        guard let record = file.lineageParent.lineage.record else {
            XCTFail("\(what): a replay save always carries a lineage record")
            throw CocoaError(.fileReadCorruptFile)
        }
        guard let streams = record.rng.streams else {
            XCTFail("\(what): a replay save always records the run's streams")
            throw CocoaError(.fileReadCorruptFile)
        }
        guard let corpus = record.fed.corpus else {
            XCTFail("\(what): a replay save always records its corpus position")
            throw CocoaError(.fileReadCorruptFile)
        }
        return EndState(contentSHA256: sha, tensors: file.weights, record: record, streams: streams, corpus: corpus)
    }

    /// Assert the interrupted run ended where the uninterrupted one did.
    private func assertSameEnd(_ resumed: EndState, _ uninterrupted: EndState,
                               agreement: TensorAgreement, _ what: String) {
        // Streams advance by draw count, so they agree exactly in either mode.
        XCTAssertEqual(resumed.streams.masterSeed, uninterrupted.streams.masterSeed, "\(what): run seed")
        XCTAssertEqual(resumed.streams.samplerState, uninterrupted.streams.samplerState,
                       "\(what): replay sampler stream position — a different number of batch draws")
        XCTAssertEqual(resumed.streams.dropoutStreamState, uninterrupted.streams.dropoutStreamState,
                       "\(what): trainer dropout stream position")
        XCTAssertEqual(resumed.record.rng.dropoutPhiloxState, uninterrupted.record.rng.dropoutPhiloxState,
                       "\(what): dropout Philox state — a different dropout mask sequence")
        // The feed: the same games fed, the same phase into the next step.
        XCTAssertEqual(resumed.corpus.epoch, uninterrupted.corpus.epoch, "\(what): corpus epoch")
        XCTAssertEqual(resumed.corpus.nextGameIndex, uninterrupted.corpus.nextGameIndex, "\(what): next game")
        XCTAssertEqual(resumed.corpus.shard, uninterrupted.corpus.shard, "\(what): shard")
        XCTAssertEqual(resumed.corpus.populatedPlies, uninterrupted.corpus.populatedPlies, "\(what): buffer fill")
        XCTAssertEqual(resumed.corpus.feedAheadPositions, uninterrupted.corpus.feedAheadPositions, "\(what): feed phase")
        // Totals continue across the resume.
        XCTAssertEqual(resumed.record.steps.cumTrainerStep, uninterrupted.record.steps.cumTrainerStep,
                       "\(what): cumulative trainer step")
        XCTAssertEqual(resumed.record.fed.cumGames, uninterrupted.record.fed.cumGames, "\(what): cumulative games fed")
        XCTAssertEqual(resumed.record.fed.cumPositions, uninterrupted.record.fed.cumPositions,
                       "\(what): cumulative positions fed")
        // The resumed segment is an exact resume with nothing left out.
        XCTAssertTrue(resumed.record.run.exactResume, "\(what): the resumed segment is recorded as exact")
        XCTAssertEqual(resumed.record.run.notExactItems, [], "\(what): no resume gaps")
        // Every tensor: masters, batch-norm statistics, velocity.
        switch agreement {
        case .bitExact:
            XCTAssertEqual(resumed.contentSHA256, uninterrupted.contentSHA256,
                           "\(what): every tensor bit-identical (the determinism probe found a step bit-reproducible)")
        case .tolerance(let relative):
            XCTAssertEqual(resumed.tensors.count, uninterrupted.tensors.count, "\(what): tensor count")
            for (index, (r, u)) in zip(resumed.tensors, uninterrupted.tensors).enumerated() {
                XCTAssertEqual(r.count, u.count, "\(what): tensor \(index) size")
                let scale = max(u.map { abs($0) }.max() ?? 0, Float.leastNormalMagnitude)
                let worst = zip(r, u).map { abs($0 - $1) }.max() ?? 0
                XCTAssertLessThanOrEqual(worst / scale, relative, "\(what): tensor \(index) within tolerance")
            }
        }
    }

    // MARK: - Tests

    /// N + M straight through versus N, save, `--resume-exact`, M — resuming
    /// inside the first epoch, at a KL-probe step.
    func testAResumeInsideAnEpochEndsWhereTheUninterruptedRunEnds() async throws {
        let agreement = try await determinismProbe()
        let n = 21, m = 20
        XCTAssertTrue(ChessTrainer.isKLProbeStep(stepIndex: n, interval: 3),
                      "the save lands on a KL-probe step, so the probe's place in the stream is tested")
        let straight = try await run(stepLimit: n + m, from: startModelURL, resumeExact: false, out: "straight.safetensors", sampling: .declared)
        let first = try await run(stepLimit: n, from: startModelURL, resumeExact: false, out: "first.safetensors", sampling: .declared)
        let firstEnd = try endState(first.url, "first segment")
        XCTAssertEqual(firstEnd.corpus.epoch, 0, "this case resumes inside the first epoch")
        let second = try await run(stepLimit: m, from: first.url, resumeExact: true, out: "second.safetensors", sampling: .declared)
        assertSameEnd(try endState(second.url, "resumed"), try endState(straight.url, "uninterrupted"),
                      agreement: agreement, "in-epoch resume")
    }

    /// The save lands early in the second epoch, so the resume's refeed
    /// window reaches back into the first epoch's last games (C1 #6).
    func testAResumeEarlyInTheSecondEpochEndsWhereTheUninterruptedRunEnds() async throws {
        let agreement = try await determinismProbe()
        // The fixture corpus's plies, over the per-step feed after the
        // prefill, fill an epoch well before this step limit, so the save
        // lands a few games into the second epoch.
        let n = 50, m = 12
        let first = try await run(stepLimit: n, from: startModelURL, resumeExact: false, out: "wrap-first.safetensors", sampling: .declared)
        let firstEnd = try endState(first.url, "second-epoch first segment")
        XCTAssertEqual(firstEnd.corpus.epoch, 1, "the save is in the second epoch")
        // The refeed window covers at least one and a half buffers of the
        // longest games; a next game inside that many games means the window
        // reaches back across the wrap.
        let narrowestWindow = Int((1.5 * 2000 / Double(Self.gamePlyRange.upperBound)).rounded(.up))
        XCTAssertLessThan(firstEnd.corpus.nextGameIndex, narrowestWindow,
                          "the save is early enough in the epoch that the refeed crosses the wrap")
        let straight = try await run(stepLimit: n + m, from: startModelURL, resumeExact: false,
                                     out: "wrap-straight.safetensors", sampling: .declared)
        let second = try await run(stepLimit: m, from: first.url, resumeExact: true, out: "wrap-second.safetensors", sampling: .declared)
        assertSameEnd(try endState(second.url, "second-epoch resumed"),
                      try endState(straight.url, "second-epoch uninterrupted"),
                      agreement: agreement, "second-epoch resume")
    }

    /// The comparison can tell a resume that drops state from one that keeps
    /// it: the same split continued as a new branch (`--start-model` without
    /// `--resume-exact`) restarts the sampler and dropout streams from the
    /// seed and the feed from the corpus start, and must not end where the
    /// uninterrupted run ends.
    func testABranchFromTheSaveDoesNotEndWhereTheUninterruptedRunEnds() async throws {
        let n = 9, m = 6
        let straight = try await run(stepLimit: n + m, from: startModelURL, resumeExact: false,
                                     out: "control-straight.safetensors", sampling: .declared)
        let first = try await run(stepLimit: n, from: startModelURL, resumeExact: false, out: "control-first.safetensors", sampling: .declared)
        let branch = try await run(stepLimit: m, from: first.url, resumeExact: false, out: "control-branch.safetensors", sampling: .declared)
        let branchEnd = try endState(branch.url, "branch")
        let straightEnd = try endState(straight.url, "control uninterrupted")
        XCTAssertNotEqual(branchEnd.streams.samplerState, straightEnd.streams.samplerState,
                          "a branch restarts the sampler stream, so the comparison must see it")
        XCTAssertNotEqual(branchEnd.corpus.nextGameIndex, straightEnd.corpus.nextGameIndex,
                          "a branch restarts the feed, so the comparison must see it")
        XCTAssertNotEqual(branchEnd.contentSHA256, straightEnd.contentSHA256,
                          "a branch trains on other batches, so its tensors must differ")
        XCTAssertFalse(branchEnd.record.run.exactResume, "a branch is not recorded as an exact resume")
    }

    /// The KL probe and the batch-stats diagnostics run extra GPU work on
    /// their steps; they must draw nothing from the run's streams and move no
    /// weight (C2 probe-isolation rule).
    func testProbesOnOrOffDrawTheSameBatchesAndMasks() async throws {
        let agreement = try await determinismProbe()
        let steps = 15
        let on = try await run(stepLimit: steps, from: startModelURL, resumeExact: false, out: "probes-on.safetensors",
                               probesOn: true, sampling: .declared)
        let off = try await run(stepLimit: steps, from: startModelURL, resumeExact: false, out: "probes-off.safetensors",
                                probesOn: false, sampling: .declared)
        let onEnd = try endState(on.url, "probes on")
        let offEnd = try endState(off.url, "probes off")
        XCTAssertEqual(onEnd.streams.samplerState, offEnd.streams.samplerState, "probes drew from the sampler stream")
        XCTAssertEqual(onEnd.streams.dropoutStreamState, offEnd.streams.dropoutStreamState,
                       "probes drew from the dropout stream")
        XCTAssertEqual(onEnd.record.rng.dropoutPhiloxState, offEnd.record.rng.dropoutPhiloxState,
                       "probes advanced the dropout Philox state")
        XCTAssertEqual(onEnd.corpus.nextGameIndex, offEnd.corpus.nextGameIndex, "probes changed the feed")
        if agreement == .bitExact {
            XCTAssertEqual(onEnd.contentSHA256, offEnd.contentSHA256, "probes changed the weights")
        }
    }

    // MARK: - Sampling constraints

    /// Corpus replay samples under the run's sampling constraints. With a
    /// per-game cap of one, a batch can take only one position from each
    /// game; the fixture buffer holds far fewer games than a batch would
    /// draw from uniformly without repeats, so the cap rejects draws and the
    /// sampler stream ends somewhere other than in a run whose constraints
    /// are all inactive (a cap at the range maximum, above the batch size;
    /// no draw cap; no length target), which samples uniformly.
    func testReplaySamplesUnderTheRunsPerGameCap() async throws {
        let steps = 12
        let capped = try await run(
            stepLimit: steps, from: startModelURL, resumeExact: false, out: "cap-one.safetensors", probesOn: false,
            sampling: Sampling(maxPerGame: 1, maxDrawPercent: 100, targetLength: 0, stratify: false))
        let capRangeMaximum = try XCTUnwrap(MaxPliesFromAnyOneGame.definition.intRange).max
        XCTAssertGreaterThanOrEqual(capRangeMaximum, try replayParams(probesOn: false, sampling: .declared).trainingBatchSize,
                                    "a cap at the range maximum is inactive for the harness's batch")
        let inactive = try await run(
            stepLimit: steps, from: startModelURL, resumeExact: false, out: "cap-inactive.safetensors", probesOn: false,
            sampling: Sampling(maxPerGame: capRangeMaximum, maxDrawPercent: 100, targetLength: 0, stratify: false))
        XCTAssertNotEqual(try endState(capped.url, "per-game cap of one").streams.samplerState,
                          try endState(inactive.url, "inactive constraints").streams.samplerState,
                          "a per-game cap of one must change how replay draws its batches")
    }

    /// N + M straight through versus N, save, `--resume-exact`, M, under each
    /// set of sampling constraints: the declared defaults; every constraint
    /// binding (a per-game cap of one, a draw cap, and a length target below
    /// the fixture's mean game length); and material stratification.
    func testAResumeUnderSamplingConstraintsEndsWhereTheUninterruptedRunEnds() async throws {
        let agreement = try await determinismProbe()
        let cases: [(name: String, sampling: Sampling)] = [
            ("declared-defaults", .declared),
            ("all-binding", Sampling(maxPerGame: 1, maxDrawPercent: 50,
                                     targetLength: Self.gamePlyRange.lowerBound + 6, stratify: false)),
            ("stratified", Sampling(maxPerGame: Sampling.declared.maxPerGame,
                                    maxDrawPercent: Sampling.declared.maxDrawPercent,
                                    targetLength: Sampling.declared.targetLength, stratify: true)),
        ]
        let n = 21, m = 20
        for c in cases {
            let straight = try await run(stepLimit: n + m, from: startModelURL, resumeExact: false,
                                         out: "\(c.name)-straight.safetensors", sampling: c.sampling)
            let first = try await run(stepLimit: n, from: startModelURL, resumeExact: false,
                                      out: "\(c.name)-first.safetensors", sampling: c.sampling)
            let second = try await run(stepLimit: m, from: first.url, resumeExact: true,
                                       out: "\(c.name)-second.safetensors", sampling: c.sampling)
            assertSameEnd(try endState(second.url, "\(c.name) resumed"),
                          try endState(straight.url, "\(c.name) uninterrupted"),
                          agreement: agreement, "\(c.name) resume")
        }
    }

    // MARK: - Epoch budget and buffer fill

    /// A run's epoch budget counts from the start of its lineage, so a
    /// resume of a checkpoint saved in the second epoch, with no step limit
    /// and the default budget of one pass, has nothing left to train. It is
    /// refused before any training or save — not run to an immediate end
    /// whose save would record a corpus position behind its parent's.
    func testAResumeWhoseEpochBudgetIsSpentIsRefused() async throws {
        let first = try await run(stepLimit: 50, from: startModelURL, resumeExact: false,
                                  out: "budget-first.safetensors", sampling: .declared)
        XCTAssertEqual(try endState(first.url, "budget first segment").corpus.epoch, 1, "the save is in the second epoch")
        let resumedOut = "budget-resumed.safetensors"
        let cfg = config(stepLimit: nil, epochs: nil, startGameIndex: nil, startModel: first.url, resumeExact: true,
                         out: resumedOut)
        do {
            let result = try await CorpusReplayRunner.runReplay(
                config: cfg, params: try replayParams(probesOn: true, sampling: .declared), abort: ReplayAbortFlag())
            XCTFail("the resume ran (\(result.steps) steps) although its epoch budget was spent")
        } catch CorpusReplayError.exactResumeEpochBudgetSpent(let savedEpoch, let epochLimit) {
            XCTAssertEqual(savedEpoch, 1)
            XCTAssertEqual(epochLimit, 1)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: tempDir.appendingPathComponent(resumedOut).path),
                       "a refused resume writes no model")
    }

    /// Two passes straight through versus a step-limited first segment
    /// ending in the second epoch, continued with `--resume-exact` under the
    /// same two-pass budget: the resume crosses no wrap of its own, trains
    /// to the end of the second pass, and ends where the uninterrupted run
    /// ends.
    func testACrossEpochResumeUnderAnEpochBudgetEndsWhereTheUninterruptedRunEnds() async throws {
        let agreement = try await determinismProbe()
        let n = 50
        let params = try replayParams(probesOn: true, sampling: .declared)
        let straight = try await CorpusReplayRunner.runReplay(
            config: config(stepLimit: nil, epochs: 2, startGameIndex: nil, startModel: startModelURL, resumeExact: false,
                           out: "epochs-straight.safetensors"),
            params: params, abort: ReplayAbortFlag())
        XCTAssertGreaterThan(straight.steps, n, "two passes take more steps than the first segment")
        let first = try await run(stepLimit: n, from: startModelURL, resumeExact: false,
                                  out: "epochs-first.safetensors", sampling: .declared)
        XCTAssertEqual(try endState(first.url, "epochs first segment").corpus.epoch, 1, "the save is in the second epoch")
        let second = try await CorpusReplayRunner.runReplay(
            config: config(stepLimit: nil, epochs: 2, startGameIndex: nil, startModel: first.url, resumeExact: true,
                           out: "epochs-second.safetensors"),
            params: params, abort: ReplayAbortFlag())
        XCTAssertEqual(second.steps, straight.steps - n, "the resume trains the rest of the second pass")
        let straightEnd = try endState(tempDir.appendingPathComponent("epochs-straight.safetensors"), "two passes straight")
        XCTAssertEqual(straightEnd.corpus.epoch, 2, "a finished two-pass run stands at the start of a third pass")
        XCTAssertEqual(straightEnd.corpus.nextGameIndex, 0)
        assertSameEnd(try endState(tempDir.appendingPathComponent("epochs-second.safetensors"), "two passes resumed"),
                      straightEnd, agreement: agreement, "epoch-budget resume")
    }

    /// A run started part-way into the corpus that saves before its buffer
    /// fills holds only the games it fed; the resume's refeed window
    /// reaches back before that start, so the rebuilt buffer holds more.
    /// The resume must not continue as exact with a buffer the saved run
    /// never had.
    func testAResumeOfARunStartedMidCorpusIsNotReportedExact() async throws {
        let startGame = Self.corpusGames / 2
        let firstOut = "midcorpus-first.safetensors"
        let firstResult = try await CorpusReplayRunner.runReplay(
            config: config(stepLimit: 3, epochs: nil, startGameIndex: startGame, startModel: startModelURL,
                           resumeExact: false, out: firstOut),
            params: try replayParams(probesOn: true, sampling: .declared), abort: ReplayAbortFlag())
        XCTAssertEqual(firstResult.steps, 3)
        let firstEnd = try endState(tempDir.appendingPathComponent(firstOut), "mid-corpus first segment")
        XCTAssertLessThan(firstEnd.corpus.populatedPlies, firstEnd.corpus.bufferCapacity,
                          "the first segment saves before its buffer fills")
        let resumedOut = "midcorpus-resumed.safetensors"
        do {
            let result = try await CorpusReplayRunner.runReplay(
                config: config(stepLimit: 3, epochs: nil, startGameIndex: nil,
                               startModel: tempDir.appendingPathComponent(firstOut), resumeExact: true, out: resumedOut),
                params: try replayParams(probesOn: true, sampling: .declared), abort: ReplayAbortFlag())
            XCTFail("the resume ran (\(result.steps) steps) with a buffer the saved run never had")
        } catch CorpusReplayError.exactResumeBufferMismatch(let saved, let rebuilt, _) {
            XCTAssertEqual(saved, firstEnd.corpus.populatedPlies)
            XCTAssertNotEqual(rebuilt, saved)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: tempDir.appendingPathComponent(resumedOut).path),
                       "a refused resume writes no model")
    }
}
