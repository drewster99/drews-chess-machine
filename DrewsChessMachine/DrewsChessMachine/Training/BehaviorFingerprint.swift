//
//  BehaviorFingerprint.swift
//  DrewsChessMachine
//
//  Whether this process computes what the process that wrote a checkpoint
//  computed (determinism plan C1 #33). A different build or OS is a reason
//  a resume might not continue the saved run — but a rebuild that changed
//  nothing the run computes is not, and flagging every rebuild as a `build`
//  gap would make `--accept-inexact build` routine and meaningless. So every
//  trainer-state save records a behavior fingerprint, and a resume under a
//  different build or OS recomputes it: equal fingerprints mean the change
//  is not a gap; a different fingerprint, or a checkpoint without one, is.
//
//  The fingerprint is the SHA-256 of a fixed micro-computation that runs the
//  code a training run's trajectory depends on, on fixed inputs and seeds:
//
//  1. Board encoding: a fixed game — castling, an en-passant capture and a
//     threefold-repetition shuffle — encoded position by position, with its
//     history, under the checkpoint's input encoding.
//  2. Seeded draws: replay-buffer batches from a buffer of those positions
//     under uniform, material-stratified and length-tilted constraints, from
//     the run streams' `sampler` stream (so the stream derivation is
//     covered); move choices with Dirichlet noise from a self-play game
//     stream; and the dropout Philox state MPSGraph derives from a seed.
//  3. Training: a fixed tiny network, with the checkpoint's compute data
//     type and the running policy-tail precision, initialized from a fixed
//     init seed, trained one SGD step (with dropout) on that buffer — its
//     losses and its exported weights and velocity, bit for bit.
//
//  Everything that can change these bytes — encoder, move generation,
//  sampler, stream derivation, MoveSampler and its Dirichlet draw, MPSGraph
//  random state, initialization, graph build, the optimizer — changes the
//  fingerprint. The recipe is versioned (`recipe`): a recipe change makes a
//  saved fingerprint incomparable, which is treated as different, never as
//  equal.
//
//  The recipe's numbers (seeds, hyperparameters, the tiny architecture) are
//  constants of the recipe, deliberately not the run's settings or the
//  parameters' declared defaults, so a fingerprint depends on code and
//  platform only. It is computed once per process for each (input encoding,
//  compute data type, policy-tail precision) and cached.
//

import CryptoKit
import Foundation
import Metal

enum BehaviorFingerprint {

    /// The recipe version. Bump it whenever the micro-computation changes;
    /// fingerprints of different recipes never compare equal.
    static let recipe = 1

    /// A computed fingerprint, as a lineage record stores it
    /// (`rng.behavior_fingerprint`).
    struct Record: Codable, Equatable, Hashable, Sendable {
        let recipe: Int
        let sha256: String

        /// Whether `other` is the same behavior: same recipe and same hash.
        func matches(_ other: Record) -> Bool {
            recipe == other.recipe && sha256 == other.sha256
        }
    }

    /// What the fingerprint is computed for: the checkpoint's numerics.
    struct Settings: Hashable, Sendable {
        let inputEncoding: InputEncoding
        let computeDataType: ComputeDataType
        let policyTailPrecision: ChessNetwork.PolicyTailPrecision

        init(inputEncoding: InputEncoding, computeDataType: ComputeDataType,
             policyTailPrecision: ChessNetwork.PolicyTailPrecision) {
            self.inputEncoding = inputEncoding
            self.computeDataType = computeDataType
            self.policyTailPrecision = policyTailPrecision
        }

        /// The settings of a trainer of `arch` under `policyTailPrecision`.
        init(arch: NetworkArchitecture, policyTailPrecision: ChessNetwork.PolicyTailPrecision) {
            self.init(inputEncoding: arch.inputEncoding, computeDataType: arch.computeDataType,
                      policyTailPrecision: policyTailPrecision)
        }
    }

    enum FingerprintError: Error, LocalizedError {
        case recipeMoveIllegal(String)
        case noMetalDevice
        case noCommandQueue
        case bufferTooSmall
        case trainStepSkipped

        var errorDescription: String? {
            switch self {
            case .recipeMoveIllegal(let uci):
                return "behavior fingerprint: the recipe's move \(uci) is not legal in its position"
            case .noMetalDevice:
                return "behavior fingerprint: no Metal device"
            case .noCommandQueue:
                return "behavior fingerprint: could not create a Metal command queue"
            case .bufferTooSmall:
                return "behavior fingerprint: the recipe's buffer holds fewer positions than a batch"
            case .trainStepSkipped:
                return "behavior fingerprint: the training step did not run"
            }
        }
    }

    // MARK: - Recipe constants

    /// The fixed game: castling on both sides, an en-passant capture
    /// (`e5d6`), then a knight shuffle that repeats the position.
    static let recipeGame: [String] = [
        "e2e4", "g8f6", "e4e5", "d7d5", "e5d6", "e7d6", "g1f3", "f8e7", "f1e2", "e8g8", "e1g1",
        "b8c6", "b1c3", "c6b8", "c3b1", "b8c6", "b1c3", "c6b8", "c3b1",
    ]
    private static let masterSeed: UInt64 = 0x0DC3_F1A6_E7B1_0001
    private static let initSeed: UInt64 = 0x0DC3_F1A6_E7B1_0002
    private static let philoxSeed = 0x5EED
    private static let gameOutcomes: [Float] = [1, 0, -1, 1]
    private static let batchSize = 16
    private static let drawsPerConstraint = 3
    private static let movesSampled = 64

    // MARK: - Compute

    private static let cache = SyncBox<[Settings: Record]>([:])

    /// The fingerprint of this process for `settings`, computed on first
    /// use and cached for the life of the process. The CPU and graph-build
    /// work runs on a dispatch queue, never on a Swift concurrency thread.
    static func compute(for settings: Settings) async throws -> Record {
        if let cached = cache.value[settings] { return cached }
        let record = try await computeUncached(for: settings, streamDerivation: DCMRandomStreams.self)
        cache.modify { $0[settings] = record }
        return record
    }

    /// The micro-computation itself. `streams` names where the seeded
    /// draws come from — production passes `DCMRandomStreams`; a test can
    /// pass another derivation to show the fingerprint covers it.
    static func computeUncached<Streams: FingerprintStreamSource>(
        for settings: Settings, streamDerivation: Streams.Type
    ) async throws -> Record {
        let prepared: (hasher: SHA256, buffer: ReplayBuffer, trainer: ChessTrainer) =
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do {
                        continuation.resume(returning: try prepare(settings: settings, streams: Streams.self))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        var hasher = prepared.hasher
        guard let timing = try await prepared.trainer.trainStep(replayBuffer: prepared.buffer, batchSize: batchSize) else {
            throw FingerprintError.trainStepSkipped
        }
        for value in [timing.loss, timing.policyLoss, timing.valueLoss, timing.policyEntropy] {
            append(value.bitPattern, to: &hasher)
        }
        for tensor in try await prepared.trainer.exportTrainerWeights() {
            append(tensor, to: &hasher)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return Record(recipe: recipe, sha256: digest)
    }

    private static let queue = DispatchQueue(label: "drewschess.behavior-fingerprint", qos: .userInitiated)

    /// Steps 1 and 2, and the tiny trainer for step 3 (built, not yet
    /// trained). Runs on `queue`.
    private static func prepare<Streams: FingerprintStreamSource>(
        settings: Settings, streams: Streams.Type
    ) throws -> (hasher: SHA256, buffer: ReplayBuffer, trainer: ChessTrainer) {
        var hasher = SHA256()
        let header = "dcm-behavior-fingerprint recipe=\(recipe) encoding=\(settings.inputEncoding.rawValue) "
            + "compute=\(settings.computeDataType.rawValue) policy_tail=\(settings.policyTailPrecision.rawValue)"
        hasher.update(data: Data(header.utf8))
        let runStreams = Streams.streams(masterSeed: masterSeed)

        // 1. Board encoding under the checkpoint's encoding, and the game's
        //    positions in the tiny network's encoding for the buffer.
        let engine = ChessGameEngine()
        var bufferGame: [(board: [Float], policyIndex: Int32, materialCount: UInt8)] = []
        for uci in recipeGame {
            append(BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: settings.inputEncoding),
                   to: &hasher)
            let mover = engine.state.currentPlayer
            guard let move = ChessMove.parseUCI(uci, legal: engine.currentLegalMoves) else {
                throw FingerprintError.recipeMoveIllegal(uci)
            }
            var materialCount = 0
            for case let piece? in engine.state.board where piece.type != .pawn { materialCount += 1 }
            bufferGame.append((
                board: BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: tinyArchitecture.inputEncoding),
                policyIndex: Int32(PolicyEncoding.policyIndex(move, currentPlayer: mover)),
                materialCount: UInt8(materialCount)))
            do {
                try engine.applyMoveAndAdvance(move)
            } catch {
                throw FingerprintError.recipeMoveIllegal(uci)
            }
        }
        append(BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: settings.inputEncoding),
               to: &hasher)

        // 2. Seeded draws: replay-buffer batches under each constraint kind.
        let buffer = ReplayBuffer(capacity: 128, inputEncoding: tinyArchitecture.inputEncoding,
                                  sampler: runStreams.generator(.sampler))
        for (gameIndex, outcome) in gameOutcomes.enumerated() {
            appendGame(bufferGame, outcome: outcome, gameIndex: gameIndex, to: buffer)
        }
        let constraints: [ReplayBuffer.SamplingConstraints] = [
            .unconstrained, stratifiedConstraints(),
            ReplayBuffer.SamplingConstraints(maxPerGame: .max, maxDrawPercent: 100,
                                             targetMeanGameLengthPlies: 10, materialBucketWeights: nil),
        ]
        for constraint in constraints {
            buffer.setSamplingConstraints(constraint)
            for _ in 0..<drawsPerConstraint {
                try appendDraw(from: buffer, to: &hasher)
            }
        }
        buffer.setSamplingConstraints(.unconstrained)

        //    Move choices with Dirichlet noise.
        let legalMoves = MoveGenerator.legalMoves(for: .starting)
        var logits = [Float](repeating: 0, count: ChessNetwork.policySize)
        for (i, move) in legalMoves.enumerated() {
            logits[PolicyEncoding.policyIndex(move, currentPlayer: .white)] = Float(i % 7) * 0.3 - 0.9
        }
        let schedule = SamplingSchedule(startTau: 0.8, decayPerPly: 0, floorTau: 0.8,
                                        dirichletNoise: DirichletNoiseConfig(alpha: 0.3, epsilon: 0.25, plyLimit: 30))
        var gameRandom = runStreams.generator(.selfPlayGame(serial: 0))
        var probs = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        var eta = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        for _ in 0..<movesSampled {
            let result = logits.withUnsafeBufferPointer { logitsBuffer in
                probs.withUnsafeMutableBufferPointer { probsBuffer in
                    eta.withUnsafeMutableBufferPointer { etaBuffer in
                        MoveSampler.sampleMove(
                            logits: logitsBuffer, legalMoves: legalMoves, currentPlayer: .white, ply: 0,
                            schedule: schedule, probsScratch: probsBuffer, etaScratch: etaBuffer, rng: &gameRandom)
                    }
                }
            }
            append(UInt32(result.policyIndex), to: &hasher)
            append(result.chosenProbability.bitPattern, to: &hasher)
        }

        //    The dropout Philox state MPSGraph derives from a seed.
        guard let device = MTLCreateSystemDefaultDevice() else { throw FingerprintError.noMetalDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw FingerprintError.noCommandQueue }
        let philox = try DropoutPhiloxState.derived(fromSeed: philoxSeed, device: device, commandQueue: commandQueue)
        for word in philox.words { append(UInt32(bitPattern: word), to: &hasher) }

        // 3. The tiny trainer, with recipe hyperparameters.
        var arch = tinyArchitecture
        arch.computeDataType = settings.computeDataType
        let trainer = try ChessTrainer(
            dropoutStream: runStreams.generator(.dropout),
            learningRate: 0.01,
            entropyRegularizationCoeff: 0.001,
            drawPenalty: 0.1,
            weightDecayC: 0.0001,
            gradClipMaxNorm: 10,
            policyLossWeight: 1,
            valueLossWeight: 1,
            illegalMassPenaltyWeight: 0.01,
            policyLabelSmoothingEpsilon: 0.05,
            policyLabelSmoothingMode: .fixedTotal,
            policyLabelSmoothingPerMove: 0.002,
            policyLabelSmoothingPerMoveCap: 0.1,
            valueLabelSmoothingEpsilon: 0.05,
            momentumCoeff: 0.9,
            useSignedAdvantageComplementCE: false,
            sqrtBatchScalingForLR: false,
            lrWarmupSteps: 0,
            arch: arch,
            initialization: .seeded(initSeed: initSeed),
            executableOptimizationLevel: .level1,
            splitWorkingWeightSync: true,
            policyTailPrecision: settings.policyTailPrecision,
            disableAutoLayoutConversion: false,
            reducedPrecisionFastMathRaw: nil)
        trainer.dropoutRate = 0.1
        trainer.batchStatsInterval = 0
        trainer.klProbeInterval = 0
        trainer.lrMomentumCycle = .disabled
        return (hasher, buffer, trainer)
    }

    /// The recipe's fixed tiny network (the checkpoint's compute data type
    /// replaces `computeDataType`).
    static let tinyArchitecture: NetworkArchitecture = NetworkArchitecture(
        inputEncoding: .basic30, channels: 16, numBlocks: 2, stemConvKernelSize: 3,
        activationFunction: .relu, blockActivationStyle: .pre,
        blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
        blockConv1KernelSize: 3, blockConv2KernelSize: 3,
        blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
        policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
        valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
        computeDataType: .float32
    )

    private static func stratifiedConstraints() -> ReplayBuffer.SamplingConstraints {
        let n = ReplayBufferAnalyzer.materialBuckets.count
        var weights = [Float](repeating: 0, count: n)
        for i in 0..<n { weights[i] = Float(i + 1) / Float(n * (n + 1) / 2) }
        return ReplayBuffer.SamplingConstraints(maxPerGame: .max, maxDrawPercent: 100,
                                                targetMeanGameLengthPlies: 0, materialBucketWeights: weights)
    }

    private static func appendGame(_ game: [(board: [Float], policyIndex: Int32, materialCount: UInt8)],
                                   outcome: Float, gameIndex: Int, to buffer: ReplayBuffer) {
        let n = game.count
        let boards = game.flatMap(\.board)
        let moves = game.map(\.policyIndex)
        let plies = (0..<n).map { UInt16($0) }
        let taus = [Float](repeating: 0.8, count: n)
        let hashes = game.map { position in
            position.board.withUnsafeBufferPointer { b -> UInt64 in
                guard let base = b.baseAddress else { preconditionFailure("an encoded board is never empty") }
                return ReplayBuffer.hashBoard(base, count: b.count)
            }
        }
        let materials = game.map(\.materialCount)
        // Outcomes alternate sign by mover, as a real flush writes them.
        let outcomes = (0..<n).map { $0 % 2 == 0 ? outcome : -outcome }
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { pl in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { ma in
        outcomes.withUnsafeBufferPointer { o in
            guard let b = b.baseAddress, let m = m.baseAddress, let pl = pl.baseAddress,
                  let t = t.baseAddress, let h = h.baseAddress, let ma = ma.baseAddress,
                  let o = o.baseAddress else {
                preconditionFailure("the recipe game is not empty")
            }
            buffer.append(
                boards: b, policyIndices: m, plyIndices: pl, samplingTaus: t,
                stateHashes: h, materialCounts: ma, gameLength: UInt16(n),
                workerId: 0, intraWorkerGameIndex: UInt32(gameIndex),
                outcomes: o, count: n)
        }}}}}}}
    }

    private static func appendDraw(from buffer: ReplayBuffer, to hasher: inout SHA256) throws {
        let floatsPerBoard = BoardEncoder.tensorLength(for: tinyArchitecture.inputEncoding)
        var boards = [Float](repeating: 0, count: batchSize * floatsPerBoard)
        var moves = [Int32](repeating: 0, count: batchSize)
        var zs = [Float](repeating: 0, count: batchSize)
        var hashes = [UInt64](repeating: 0, count: batchSize)
        let drew = boards.withUnsafeMutableBufferPointer { b -> Bool in
        moves.withUnsafeMutableBufferPointer { m in
        zs.withUnsafeMutableBufferPointer { z in
        hashes.withUnsafeMutableBufferPointer { h in
            guard let b = b.baseAddress, let m = m.baseAddress, let z = z.baseAddress, let h = h.baseAddress else {
                preconditionFailure("the draw arrays are not empty")
            }
            return buffer.sample(count: batchSize, intoBoards: b, moves: m, zs: z, hashes: h)
        }}}}
        guard drew else { throw FingerprintError.bufferTooSmall }
        for move in moves { append(UInt32(bitPattern: move), to: &hasher) }
        for z in zs { append(z.bitPattern, to: &hasher) }
        for hash in hashes { append(hash, to: &hasher) }
    }

    private static func append<Value: FixedWidthInteger>(_ value: Value, to hasher: inout SHA256) {
        withUnsafeBytes(of: value.littleEndian) { hasher.update(bufferPointer: $0) }
    }

    private static func append(_ values: [Float], to hasher: inout SHA256) {
        for value in values { append(value.bitPattern, to: &hasher) }
    }
}

/// Where the fingerprint's seeded draws come from: the run streams'
/// derivation. Production uses `DCMRandomStreams`.
protocol FingerprintStreamSource {
    associatedtype Streams: FingerprintStreams
    static func streams(masterSeed: UInt64) -> Streams
}

/// The named streams the fingerprint draws from.
protocol FingerprintStreams {
    func generator(_ stream: DCMStream) -> DCMRandom
}

extension DCMRandomStreams: FingerprintStreams {}

extension DCMRandomStreams: FingerprintStreamSource {
    static func streams(masterSeed: UInt64) -> DCMRandomStreams {
        DCMRandomStreams(masterSeed: masterSeed)
    }
}
