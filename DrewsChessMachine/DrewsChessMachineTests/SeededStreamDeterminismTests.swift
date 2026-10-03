//
//  SeededStreamDeterminismTests.swift
//  DrewsChessMachineTests
//
//  Every training-relevant random draw comes from a named, seeded stream
//  (determinism plan, Part A3). These pin what that buys: the replay
//  buffer's `sample()` sequence is a function of its sampler seed and can be
//  continued from a captured state; `MoveSampler` (including Dirichlet noise)
//  is a function of the generator it is handed; a game's moves depend only on
//  its own stream, not on how many games run beside it; the BN-calibration
//  warmup batch and the entropy probe's subsample are reproducible; and the
//  Gamma sampler behind the Dirichlet noise still has the right moments.
//

import XCTest
@testable import DrewsChessMachine

final class SeededStreamDeterminismTests: XCTestCase {

    // MARK: - Replay buffer sampler

    /// Positions per fixture game, and fixture games. Each position's policy
    /// index is its global insertion index, so a sampled batch's moves name
    /// exactly which positions were drawn.
    private let pliesPerGame = 20
    private let gameCount = 30

    private func makeFilledBuffer(sampler: DCMRandom) -> ReplayBuffer {
        let buffer = ReplayBuffer(capacity: pliesPerGame * gameCount, sampler: sampler)
        let fpb = buffer.floatsPerBoard
        let n = pliesPerGame
        let boards = UnsafeMutablePointer<Float>.allocate(capacity: n * fpb)
        let moves = UnsafeMutablePointer<Int32>.allocate(capacity: n)
        let plies = UnsafeMutablePointer<UInt16>.allocate(capacity: n)
        let taus = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let hashes = UnsafeMutablePointer<UInt64>.allocate(capacity: n)
        let materials = UnsafeMutablePointer<UInt8>.allocate(capacity: n)
        let outcomes = UnsafeMutablePointer<Float>.allocate(capacity: n)
        defer {
            boards.deallocate(); moves.deallocate(); plies.deallocate(); taus.deallocate()
            hashes.deallocate(); materials.deallocate(); outcomes.deallocate()
        }
        for game in 0..<gameCount {
            boards.initialize(repeating: 0, count: n * fpb)
            for i in 0..<n {
                let global = game * n + i
                boards[i * fpb] = Float(global)
                moves[i] = Int32(global)
                plies[i] = UInt16(i)
                taus[i] = 1
                hashes[i] = UInt64(global) &* 0x9E37_79B9_7F4A_7C15
                // Spread positions over the material buckets so the
                // stratified path has several buckets to draw from.
                materials[i] = UInt8(8 + (global % 24))
                outcomes[i] = game % 3 == 0 ? 0 : (i % 2 == 0 ? 1 : -1)
            }
            buffer.append(
                boards: boards, policyIndices: moves, plyIndices: plies, samplingTaus: taus,
                stateHashes: hashes, materialCounts: materials, gameLength: UInt16(n),
                workerId: 0, intraWorkerGameIndex: UInt32(game), outcomes: outcomes, count: n)
        }
        return buffer
    }

    /// The positions (by insertion index) of `batches` consecutive batches.
    private func drawSequence(_ buffer: ReplayBuffer, batches: Int, batchSize: Int) -> [[Int32]] {
        let fpb = buffer.reconstructedStride
        let boards = UnsafeMutablePointer<Float>.allocate(capacity: batchSize * fpb)
        let moves = UnsafeMutablePointer<Int32>.allocate(capacity: batchSize)
        let zs = UnsafeMutablePointer<Float>.allocate(capacity: batchSize)
        defer { boards.deallocate(); moves.deallocate(); zs.deallocate() }
        var sequence: [[Int32]] = []
        for _ in 0..<batches {
            XCTAssertTrue(buffer.sample(count: batchSize, intoBoards: boards, moves: moves, zs: zs))
            sequence.append(Array(UnsafeBufferPointer(start: moves, count: batchSize)))
        }
        return sequence
    }

    /// Each sampling path the buffer has: plain uniform, per-game cap and
    /// draw cap (the budget path), length tilt, and material stratification.
    private let constraintVariants: [(String, ReplayBuffer.SamplingConstraints)] = [
        ("uniform", ReplayBuffer.SamplingConstraints(maxPerGame: .max, maxDrawPercent: 100, targetMeanGameLengthPlies: 0)),
        ("budget", ReplayBuffer.SamplingConstraints(maxPerGame: 3, maxDrawPercent: 25, targetMeanGameLengthPlies: 0)),
        ("tilt", ReplayBuffer.SamplingConstraints(maxPerGame: .max, maxDrawPercent: 100, targetMeanGameLengthPlies: 30)),
    ]

    func test_sample_sameSeedGivesTheSameSequenceOnEveryPath() {
        for (name, constraints) in constraintVariants {
            let a = makeFilledBuffer(sampler: DCMRandom(seed: 7))
            let b = makeFilledBuffer(sampler: DCMRandom(seed: 7))
            a.setSamplingConstraints(constraints)
            b.setSamplingConstraints(constraints)
            XCTAssertEqual(drawSequence(a, batches: 5, batchSize: 32),
                           drawSequence(b, batches: 5, batchSize: 32), name)
        }
    }

    func test_sample_stratifiedPathIsSeeded() {
        let a = makeFilledBuffer(sampler: DCMRandom(seed: 7))
        let b = makeFilledBuffer(sampler: DCMRandom(seed: 7))
        let bucketCount = ReplayBufferAnalyzer.materialBuckets.count
        let constraints = ReplayBuffer.SamplingConstraints(
            maxPerGame: .max, maxDrawPercent: 100, targetMeanGameLengthPlies: 0,
            materialBucketWeights: [Float](repeating: 1 / Float(bucketCount), count: bucketCount))
        a.setSamplingConstraints(constraints)
        b.setSamplingConstraints(constraints)
        XCTAssertEqual(drawSequence(a, batches: 5, batchSize: 32), drawSequence(b, batches: 5, batchSize: 32))
    }

    func test_sample_differentSeedsGiveDifferentSequences() {
        let a = makeFilledBuffer(sampler: DCMRandom(seed: 7))
        let b = makeFilledBuffer(sampler: DCMRandom(seed: 8))
        XCTAssertNotEqual(drawSequence(a, batches: 3, batchSize: 32), drawSequence(b, batches: 3, batchSize: 32))
    }

    func test_sample_restoredSamplerStateContinuesTheSequence() {
        let uninterrupted = makeFilledBuffer(sampler: DCMRandom(seed: 11))
        let full = drawSequence(uninterrupted, batches: 6, batchSize: 16)

        let first = makeFilledBuffer(sampler: DCMRandom(seed: 11))
        let head = drawSequence(first, batches: 3, batchSize: 16)
        let saved = first.samplerState()

        let resumed = makeFilledBuffer(sampler: DCMRandom(seed: 999))
        resumed.restoreSamplerState(saved)
        let tail = drawSequence(resumed, batches: 3, batchSize: 16)
        XCTAssertEqual(head + tail, full)
    }

    // MARK: - MoveSampler

    private let noisySchedule = SamplingSchedule(
        startTau: 1.0, decayPerPly: 0, floorTau: 1.0,
        dirichletNoise: DirichletNoiseConfig(alpha: 0.3, epsilon: 0.25, plyLimit: 30))

    /// Fixed, non-uniform logits for the starting position's legal moves.
    private func startingPositionInputs() -> (logits: [Float], legalMoves: [ChessMove]) {
        let legalMoves = MoveGenerator.legalMoves(for: .starting)
        var logits = [Float](repeating: 0, count: ChessNetwork.policySize)
        for (i, move) in legalMoves.enumerated() {
            let index = PolicyEncoding.policyIndex(move, currentPlayer: .white)
            logits[index] = Float(i % 5) * 0.4
        }
        return (logits, legalMoves)
    }

    private func sampleMoves(count: Int, ply: Int, rng: inout DCMRandom) -> [ChessMove] {
        let (logits, legalMoves) = startingPositionInputs()
        var probs = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        var eta = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        var moves: [ChessMove] = []
        for _ in 0..<count {
            let result = logits.withUnsafeBufferPointer { logitsBuf in
                probs.withUnsafeMutableBufferPointer { probsBuf in
                    eta.withUnsafeMutableBufferPointer { etaBuf in
                        MoveSampler.sampleMove(
                            logits: logitsBuf, legalMoves: legalMoves, currentPlayer: .white, ply: ply,
                            schedule: noisySchedule, probsScratch: probsBuf, etaScratch: etaBuf, rng: &rng)
                    }
                }
            }
            moves.append(result.move)
        }
        return moves
    }

    func test_moveSampler_sameGeneratorGivesTheSameMovesWithDirichletNoise() {
        var a = DCMRandom(seed: 21)
        var b = DCMRandom(seed: 21)
        let movesA = sampleMoves(count: 200, ply: 0, rng: &a)
        XCTAssertEqual(movesA, sampleMoves(count: 200, ply: 0, rng: &b))
        XCTAssertEqual(a, b, "both generators must have advanced identically")
        XCTAssertGreaterThan(Set(movesA.map(\.uci)).count, 5, "the fixture must actually sample several moves")
    }

    func test_moveSampler_differentGeneratorsGiveDifferentMoves() {
        var a = DCMRandom(seed: 21)
        var b = DCMRandom(seed: 22)
        XCTAssertNotEqual(sampleMoves(count: 50, ply: 0, rng: &a), sampleMoves(count: 50, ply: 0, rng: &b))
    }

    /// Gamma(α, 1) has mean α and variance α, on both sides of the α = 1
    /// boost. Tolerances are several standard errors at this sample size.
    func test_sampleGamma_momentsMatchTheDistribution() {
        var rng = DCMRandom(seed: 5)
        let n = 200_000
        for alpha: Float in [0.3, 1.0, 2.5] {
            var sum = 0.0
            var sumSquares = 0.0
            for _ in 0..<n {
                let g = Double(MoveSampler.sampleGamma(alpha: alpha, rng: &rng))
                XCTAssertGreaterThan(g, 0)
                sum += g
                sumSquares += g * g
            }
            let mean = sum / Double(n)
            let variance = sumSquares / Double(n) - mean * mean
            XCTAssertEqual(mean, Double(alpha), accuracy: 0.02 * max(1, Double(alpha)), "mean, alpha \(alpha)")
            XCTAssertEqual(variance, Double(alpha), accuracy: 0.05 * max(1, Double(alpha)), "variance, alpha \(alpha)")
        }
    }

    // MARK: - Per-game streams

    /// The moves a game samples depend only on its own stream: the game with
    /// serial 3 plays the same moves alone (K = 1) as in slot 3 of eight games
    /// sampled in lockstep (K = 8), given identical network outputs.
    func test_gameStream_isIndependentOfHowManyGamesRunBesideIt() {
        let streams = DCMRandomStreams(masterSeed: 1234)
        let plies = 40

        var alone = streams.generator(.selfPlayGame(serial: 3))
        var aloneMoves: [ChessMove] = []
        for ply in 0..<plies {
            aloneMoves.append(contentsOf: sampleMoves(count: 1, ply: ply, rng: &alone))
        }

        var slots = (0..<8).map { streams.generator(.selfPlayGame(serial: $0)) }
        var slotMoves = [[ChessMove]](repeating: [], count: 8)
        for ply in 0..<plies {
            for slot in 0..<8 {
                slotMoves[slot].append(contentsOf: sampleMoves(count: 1, ply: ply, rng: &slots[slot]))
            }
        }
        XCTAssertEqual(slotMoves[3], aloneMoves)
        XCTAssertNotEqual(slotMoves[2], aloneMoves, "distinct serials must get distinct streams")
    }

    func test_streamNames_areDistinctPerGameAndArena() {
        let streams = DCMRandomStreams(masterSeed: 1)
        let generators = [
            streams.generator(.selfPlayGame(serial: 0)),
            streams.generator(.selfPlayGame(serial: 1)),
            streams.generator(.arenaGame(arenaIndex: 0, gameIndex: 0)),
            streams.generator(.arenaGame(arenaIndex: 1, gameIndex: 0)),
            streams.generator(.arenaGame(arenaIndex: 0, gameIndex: 1)),
            streams.generator(.trainVsUciGame(serial: 0)),
            streams.generator(.sampler),
        ]
        for i in generators.indices {
            for j in generators.indices where j > i {
                XCTAssertNotEqual(generators[i], generators[j], "streams \(i) and \(j)")
            }
        }
    }

    // MARK: - BN calibration and probes

    func test_bnCalibrationWarmupBatch_isAFunctionOfItsStream() {
        var a = DCMRandomStreams.batchNormCalibrationGenerator(initSeed: 17)
        var b = DCMRandomStreams.batchNormCalibrationGenerator(initSeed: 17)
        var c = DCMRandomStreams.batchNormCalibrationGenerator(initSeed: 18)
        let batchA = ChessMPSNetwork.warmupBatch(encoding: .basic30, random: &a)
        XCTAssertEqual(batchA, ChessMPSNetwork.warmupBatch(encoding: .basic30, random: &b))
        XCTAssertNotEqual(batchA, ChessMPSNetwork.warmupBatch(encoding: .basic30, random: &c))
    }

    func test_entropyProbeRandom_isAFunctionOfRunSeedAndStep() {
        let runSeed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 5, commandLineSeed: nil, drawSeed: { 0 })
        let a = ReplayBufferAnalyzer.entropyProbeRandom(runSeed: runSeed, trainerStep: 100)
        let b = ReplayBufferAnalyzer.entropyProbeRandom(runSeed: runSeed, trainerStep: 100)
        let c = ReplayBufferAnalyzer.entropyProbeRandom(runSeed: runSeed, trainerStep: 101)
        XCTAssertEqual(a.random, b.random)
        XCTAssertNotEqual(a.random, c.random)
        XCTAssertEqual(a.description, "stream probe.entropy_by_bucket.100 of run seed 5")
        let unseeded = ReplayBufferAnalyzer.entropyProbeRandom(runSeed: nil, trainerStep: 100)
        XCTAssertTrue(unseeded.description.hasPrefix("seeded from the system"))
    }
}
