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
//  2. Seeded draws: replay-buffer batches from a buffer of games of
//     different lengths cut from that game, under uniform, material-
//     stratified and length-tilted constraints and under the per-game cap
//     and the draw cap, from the run streams' `sampler` stream (so the stream derivation is
//     covered); move choices with Dirichlet noise from a self-play game
//     stream; and the dropout Philox state MPSGraph derives from a seed.
//  3. Training: the checkpoint's own architecture — its block groups,
//     convolutions, SE, ReZero, activations, heads, input encoding and
//     compute data type — under the running policy-tail precision,
//     initialized from a fixed init seed (never the checkpoint's trained
//     weights), trained one SGD step (with dropout, which includes the
//     value-baseline forward pass) on a batch drawn from that buffer — its
//     losses and its exported weights and velocity, bit for bit. Recipe 1
//     trained a fixed tiny network instead, which left any change in a
//     block type it lacked undetected.
//
//  Everything that can change these bytes — encoder, move generation,
//  sampler, stream derivation, MoveSampler and its Dirichlet draw, MPSGraph
//  random state, initialization, graph build, the optimizer — changes the
//  fingerprint. The recipe is versioned (`recipe`): a recipe change makes a
//  saved fingerprint incomparable, which is treated as different, never as
//  equal.
//
//  The recipe's numbers (seeds, hyperparameters) are constants of the
//  recipe, deliberately not the run's settings or the parameters' declared
//  defaults, so a fingerprint depends on code, platform and the checkpoint's
//  architecture only. It is computed once per process for each (architecture,
//  policy-tail precision) and cached.
//

import CryptoKit
import Foundation
import Metal

enum BehaviorFingerprint {

    /// The recipe version. Bump it whenever the micro-computation changes;
    /// fingerprints of different recipes never compare equal. The current
    /// recipe runs the per-game cap, the draw cap and a binding length tilt
    /// over games of different lengths, which the previous one did not:
    /// from it on, corpus replay and train-vs-UCI sample under those
    /// constraints, which builds of the previous recipe did not.
    static let recipe = 3

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

    /// What the fingerprint is computed for: the checkpoint's architecture
    /// (which carries its input encoding and compute data type) and the
    /// policy-tail precision its trainer runs under.
    struct Settings: Hashable, Sendable {
        let architecture: NetworkArchitecture
        let policyTailPrecision: ChessNetwork.PolicyTailPrecision

        init(arch: NetworkArchitecture, policyTailPrecision: ChessNetwork.PolicyTailPrecision) {
            self.architecture = arch
            self.policyTailPrecision = policyTailPrecision
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
    /// The plies of each recipe-buffer game, by outcome: openings of the
    /// recipe game, of different lengths so the length tilt has something to
    /// down-weight.
    private static let recipeGamePlies = [19, 11, 15, 7]
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

    /// Steps 1 and 2, and the checkpoint-architecture trainer for step 3 (built, not yet
    /// trained). Runs on `queue`.
    private static func prepare<Streams: FingerprintStreamSource>(
        settings: Settings, streams: Streams.Type
    ) throws -> (hasher: SHA256, buffer: ReplayBuffer, trainer: ChessTrainer) {
        var hasher = SHA256()
        // The architecture itself is not hashed: what it computes is, below.
        // Hashing its serialized form would make a change in how it is
        // written (a new Codable field) read as a change in behavior.
        let header = "dcm-behavior-fingerprint recipe=\(recipe) policy_tail=\(settings.policyTailPrecision.rawValue)"
        hasher.update(data: Data(header.utf8))
        let arch = settings.architecture
        let encoding = arch.inputEncoding
        let runStreams = Streams.streams(masterSeed: masterSeed)

        // 1. Board encoding under the checkpoint's encoding; each position's
        //    stored frame goes to the buffer.
        let bufferGame = try replayRecipeGame(encoding: encoding) { encoded in
            append(encoded, to: &hasher)
        }

        // 2. Seeded draws: replay-buffer batches under each constraint kind.
        let buffer = recipeBuffer(bufferGame, encoding: encoding, runStreams: runStreams)
        try appendConstrainedDraws(from: buffer, constraints: samplerConstraints, encoding: encoding, to: &hasher)

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

        // 3. A trainer of the checkpoint's architecture, from the recipe's
        //    init seed, with the recipe's hyperparameters.
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

    /// The constraints the recipe's sampler draws run under, in order.
    static let samplerConstraints: [ReplayBuffer.SamplingConstraints] = [
        .unconstrained, stratifiedConstraints(),
        ReplayBuffer.SamplingConstraints(maxPerGame: .max, maxDrawPercent: 100,
                                         targetMeanGameLengthPlies: 10, materialBucketWeights: nil),
        // The constrained path at the sampling parameters' declared values
        // when this recipe was written (recipe constants, deliberately not
        // read from the declarations): a per-game cap below the batch size,
        // no draw cap, a length target above every recipe game.
        ReplayBuffer.SamplingConstraints(maxPerGame: 10, maxDrawPercent: 100,
                                         targetMeanGameLengthPlies: 999, materialBucketWeights: nil),
        // Every cap binding at once, still fillable from the recipe buffer: a
        // per-game cap the recipe games together just cover, a draw cap below
        // the buffer's draw share, and a length target below the buffer's
        // position-weighted mean game length.
        ReplayBuffer.SamplingConstraints(maxPerGame: 5, maxDrawPercent: 15,
                                         targetMeanGameLengthPlies: 12, materialBucketWeights: nil),
    ]

    /// Play the recipe game under `encoding`, handing each encoded position
    /// (and the final one) to `eachEncoding`, and return its plies as the
    /// replay buffer stores them: one mover-relative frame per ply (the start
    /// of the full encoding), from which the buffer rebuilds a history stack
    /// at sample time — the layout `ActiveGame` writes.
    private static func replayRecipeGame(
        encoding: InputEncoding, eachEncoding: (([Float]) -> Void)
    ) throws -> [(board: [Float], policyIndex: Int32, materialCount: UInt8)] {
        let storedFrameFloats = encoding.planesPerFrame * ChessNetwork.boardSize * ChessNetwork.boardSize
        let engine = ChessGameEngine()
        var bufferGame: [(board: [Float], policyIndex: Int32, materialCount: UInt8)] = []
        for uci in recipeGame {
            let encoded = BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: encoding)
            eachEncoding(encoded)
            let mover = engine.state.currentPlayer
            guard let move = ChessMove.parseUCI(uci, legal: engine.currentLegalMoves) else {
                throw FingerprintError.recipeMoveIllegal(uci)
            }
            var materialCount = 0
            for case let piece? in engine.state.board where piece.type != .pawn { materialCount += 1 }
            bufferGame.append((
                board: Array(encoded.prefix(storedFrameFloats)),
                policyIndex: Int32(PolicyEncoding.policyIndex(move, currentPlayer: mover)),
                materialCount: UInt8(materialCount)))
            do {
                try engine.applyMoveAndAdvance(move)
            } catch {
                throw FingerprintError.recipeMoveIllegal(uci)
            }
        }
        eachEncoding(BoardEncoder.encode(engine.state, history: engine.recentStates, encoding: encoding))
        return bufferGame
    }

    /// The recipe buffer: one game per recipe outcome, each the opening
    /// `recipeGamePlies[i]` plies of the recipe game — games of different
    /// lengths, so the length tilt has long games to down-weight — sampled
    /// from the run streams' `sampler` stream.
    private static func recipeBuffer(
        _ bufferGame: [(board: [Float], policyIndex: Int32, materialCount: UInt8)],
        encoding: InputEncoding, runStreams: some FingerprintStreams
    ) -> ReplayBuffer {
        precondition(recipeGamePlies.count == gameOutcomes.count && recipeGamePlies.allSatisfy { $0 <= bufferGame.count },
                     "each recipe outcome has a game no longer than the recipe game")
        let buffer = ReplayBuffer(capacity: 128, inputEncoding: encoding,
                                  sampler: runStreams.generator(.sampler))
        for (gameIndex, outcome) in gameOutcomes.enumerated() {
            appendGame(Array(bufferGame.prefix(recipeGamePlies[gameIndex])), outcome: outcome,
                       gameIndex: gameIndex, to: buffer)
        }
        return buffer
    }

    /// The sampler draws under each of `constraints` in turn, appended to
    /// `hasher`; the buffer is left unconstrained for the training step.
    private static func appendConstrainedDraws(
        from buffer: ReplayBuffer, constraints: [ReplayBuffer.SamplingConstraints],
        encoding: InputEncoding, to hasher: inout SHA256
    ) throws {
        for constraint in constraints {
            buffer.setSamplingConstraints(constraint)
            for _ in 0..<drawsPerConstraint {
                try appendDraw(from: buffer, encoding: encoding, to: &hasher)
            }
        }
        buffer.setSamplingConstraints(.unconstrained)
    }

    /// The digest of the recipe's sampler draws alone, under `constraints`,
    /// for the basic encoding and the standard stream derivation — what a
    /// change in how the sampler applies a constraint changes.
    static func samplerDrawsDigest(constraints: [ReplayBuffer.SamplingConstraints]) throws -> String {
        let encoding = InputEncoding.basic30
        let bufferGame = try replayRecipeGame(encoding: encoding) { _ in }
        let buffer = recipeBuffer(bufferGame, encoding: encoding,
                                  runStreams: DCMRandomStreams.streams(masterSeed: masterSeed))
        var hasher = SHA256()
        try appendConstrainedDraws(from: buffer, constraints: constraints, encoding: encoding, to: &hasher)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func stratifiedConstraints() -> ReplayBuffer.SamplingConstraints {
        let n = ReplayBufferAnalyzer.materialBuckets.count
        var weights = [Float](repeating: 0, count: n)
        for i in 0..<n { weights[i] = Float(i + 1) / Float(n * (n + 1) / 2) }
        return ReplayBuffer.SamplingConstraints(maxPerGame: .max, maxDrawPercent: 100,
                                                targetMeanGameLengthPlies: 0, materialBucketWeights: weights)
    }

    /// Append one game the way `ActiveGame.flush` does: newest ply first,
    /// each row's ply index its game ply, outcomes signed by mover.
    private static func appendGame(_ forward: [(board: [Float], policyIndex: Int32, materialCount: UInt8)],
                                   outcome: Float, gameIndex: Int, to buffer: ReplayBuffer) {
        let n = forward.count
        let game = Array(forward.reversed())
        let boards = game.flatMap(\.board)
        let moves = game.map(\.policyIndex)
        let plies = (0..<n).map { UInt16(n - 1 - $0) }
        let taus = [Float](repeating: 0.8, count: n)
        let hashes = game.map { position in
            position.board.withUnsafeBufferPointer { b -> UInt64 in
                guard let base = b.baseAddress else { preconditionFailure("an encoded board is never empty") }
                return ReplayBuffer.hashBoard(base, count: b.count)
            }
        }
        let materials = game.map(\.materialCount)
        // Outcomes signed by the row's mover (white moves on even plies).
        let outcomes = plies.map { $0 % 2 == 0 ? outcome : -outcome }
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

    private static func appendDraw(from buffer: ReplayBuffer, encoding: InputEncoding,
                                   to hasher: inout SHA256) throws {
        let floatsPerBoard = BoardEncoder.tensorLength(for: encoding)
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

    /// The floats' IEEE bit patterns, little-endian, in one update (Apple
    /// silicon stores them that way, so the array's bytes are exactly that).
    private static func append(_ values: [Float], to hasher: inout SHA256) {
        values.withUnsafeBytes { hasher.update(bufferPointer: $0) }
    }
}

/// Where the fingerprint's seeded draws come from: the run streams'
/// derivation. Production uses `DCMRandomStreams`. A source is used only
/// through its metatype, which `computeUncached` hands to the
/// fingerprint's dispatch queue, so the metatype must be sendable.
protocol FingerprintStreamSource: SendableMetatype {
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
