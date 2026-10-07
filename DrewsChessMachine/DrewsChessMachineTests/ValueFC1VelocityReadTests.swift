import Metal
import XCTest
@testable import DrewsChessMachine

/// `ChessTrainer.readTrainableVelocity(named:)`, the training-health
/// monitor's dedicated value-FC1 read (alarms plan D6): it returns exactly
/// the velocity a full export holds for that tensor, and it is an observer —
/// a trainer that reads after every step ends bit-identical (weights,
/// velocity, BN statistics, dropout Philox state) to one that never reads.
final class ValueFC1VelocityReadTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// Small, with dropout (so the training graph advances the Philox
    /// state) and a ReLU value hidden layer.
    private func arch() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.blockGroups = [
            BlockGroup(
                count: 1, channels: 16,
                conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .none, seReductionRatio: 4,
                useRezero: true, rezeroAlphaInit: 0.5,
                activationFunction: .relu, activationStyle: .pre,
                skipMerge: .cleanAdd, dropoutMultiplier: 0.5
            )
        ]
        arch.valueHeadConvChannels = 4
        arch.valueHeadHiddenUnits = 8
        arch.valueHeadFC1HiddenActivation = .relu
        return arch
    }

    private func makeTrainer() throws -> ChessTrainer {
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 11), momentumCoeff: 0.9, lrWarmupSteps: 0,
            arch: arch(), initialization: .seeded(initSeed: 3))
        trainer.dropoutRate = 0.2
        return trainer
    }

    func testReadEqualsTheExportedVelocityOfThatTensor() async throws {
        try requireMetal()
        let trainer = try makeTrainer()
        let name = LayerHealth.valueFC1Layer(for: trainer.arch).weightTensorName
        let index = try XCTUnwrap(trainer.arch.trainableTensorPlan().map(\.name).firstIndex(of: name))
        for step in 0..<3 {
            if step > 0 { _ = try await trainer.trainStep(batchSize: 8) }
            let read = try await trainer.readTrainableVelocity(named: name)
            let exported = try await trainer.exportVelocitySnapshot()
            XCTAssertEqual(read.velocity, exported[index], "after \(step) steps")
            XCTAssertEqual(read.completedTrainSteps, trainer.completedTrainSteps)
            if step > 0 {
                XCTAssertTrue(read.velocity.contains { $0 != 0 }, "a trained step leaves nonzero velocity")
            }
        }
    }

    func testReadRefusesANameOutsideThePlan() async throws {
        try requireMetal()
        let trainer = try makeTrainer()
        do {
            _ = try await trainer.readTrainableVelocity(named: "value.fc9.weight")
            XCTFail("an unknown tensor must be refused")
        } catch ChessTrainerError.layerHealthTensorNotInPlan(let name) {
            XCTAssertEqual(name, "value.fc9.weight")
        }
    }

    /// Probe isolation: reading after every step changes nothing the trainer
    /// holds, including the dropout stream (no RNG advance). Real-data steps
    /// from identically seeded replay buffers, so the trainers see the same
    /// batches (the synthetic `trainStep(batchSize:)` draws its random
    /// batch from the system generator, so two trainers would never agree).
    /// A third, plain trainer is the control for the GPU's own determinism:
    /// when the two plain trainers agree bit for bit the reading one must
    /// too; when they do not (a GPU shared with other work), the reading one
    /// may differ from a plain one by no more than the plain ones differ
    /// from each other.
    func testReadingAfterEveryStepLeavesTheTrainerBitIdentical() async throws {
        try requireMetal()
        let reading = try makeTrainer()
        let plain = try makeTrainer()
        let control = try makeTrainer()
        let name = LayerHealth.valueFC1Layer(for: reading.arch).weightTensorName
        let readingBuffer = makeReplayBuffer(arch: reading.arch)
        let plainBuffer = makeReplayBuffer(arch: plain.arch)
        let controlBuffer = makeReplayBuffer(arch: control.arch)
        for _ in 0..<4 {
            _ = try await reading.trainStep(replayBuffer: readingBuffer, batchSize: 16)
            _ = try await reading.readTrainableVelocity(named: name)
            _ = try await plain.trainStep(replayBuffer: plainBuffer, batchSize: 16)
            _ = try await control.trainStep(replayBuffer: controlBuffer, batchSize: 16)
        }
        XCTAssertEqual(reading.completedTrainSteps, 4)
        XCTAssertEqual(plain.completedTrainSteps, 4)
        let readingState = try await reading.exportTrainerWeights()
        let plainState = try await plain.exportTrainerWeights()
        let controlState = try await control.exportTrainerWeights()
        if plainState == controlState {
            XCTAssertEqual(readingState, plainState, "weights, BN statistics and velocity must be bit-identical")
        } else {
            let gpuSpread = Self.largestDifference(plainState, controlState)
            let readingSpread = Self.largestDifference(readingState, plainState)
            XCTAssertLessThanOrEqual(
                readingSpread, gpuSpread,
                "the GPU is not deterministic here (plain vs control \(gpuSpread)); the read must not add to it")
        }
        let readingDropout = try await reading.captureDropoutState()
        let plainDropout = try await plain.captureDropoutState()
        XCTAssertEqual(readingDropout, plainDropout, "the read must not advance the dropout stream")
    }

    private static func largestDifference(_ a: [[Float]], _ b: [[Float]]) -> Float {
        precondition(a.count == b.count && zip(a, b).allSatisfy { $0.count == $1.count }, "same trainer shape")
        var largest: Float = 0
        for (x, y) in zip(a, b) {
            for (u, v) in zip(x, y) {
                largest = max(largest, abs(u - v))
            }
        }
        return largest
    }

    /// Deterministic pseudo-random legal games from the start position, the
    /// played move as the target (`ExactResumeTests`' fixture): real
    /// positions, because a real-data step builds each sample's legal mask
    /// from the decoded board.
    private func makeReplayBuffer(arch: NetworkArchitecture) -> ReplayBuffer {
        let positions = 256
        let encoding = arch.inputEncoding
        let floatsPerBoard = BoardEncoder.tensorLength(for: encoding)
        var boards = [Float](repeating: 0, count: positions * floatsPerBoard)
        var moves = [Int32](repeating: 0, count: positions)
        var plies = [UInt16](repeating: 0, count: positions)
        let taus = [Float](repeating: 1.0, count: positions)
        var hashes = [UInt64](repeating: 0, count: positions)
        let materials = [UInt8](repeating: 32, count: positions)
        var outcomes = [Float](repeating: 0, count: positions)
        var lcg: UInt64 = 0x0FC1_7E10_C17A_5EED
        func next() -> UInt64 {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return lcg >> 33
        }
        var state = GameState.starting
        var ply = 0
        var filled = 0
        while filled < positions {
            let legal = MoveGenerator.legalMoves(for: state)
            if legal.isEmpty || ply >= 80 {
                state = GameState.starting
                ply = 0
                continue
            }
            let move = legal[Int(next() % UInt64(legal.count))]
            let encoded = BoardEncoder.encode(state, encoding: encoding)
            boards.replaceSubrange(filled * floatsPerBoard ..< (filled + 1) * floatsPerBoard, with: encoded)
            moves[filled] = Int32(PolicyEncoding.policyIndex(move, currentPlayer: state.currentPlayer))
            plies[filled] = UInt16(ply)
            hashes[filled] = next()
            outcomes[filled] = Float(Int(next() % 3)) - 1.0
            state = MoveGenerator.applyMove(move, to: state)
            ply += 1
            filled += 1
        }
        let buffer = ReplayBuffer(capacity: positions, inputEncoding: encoding, sampler: DCMRandom(seed: 5))
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { p in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { mc in
        outcomes.withUnsafeBufferPointer { o in
            guard let bb = b.baseAddress, let mb = m.baseAddress, let pb = p.baseAddress,
                  let tb = t.baseAddress, let hb = h.baseAddress, let mcb = mc.baseAddress,
                  let ob = o.baseAddress else {
                preconditionFailure("fixture arrays are non-empty, so every base address exists")
            }
            buffer.append(
                boards: bb, policyIndices: mb, plyIndices: pb, samplingTaus: tb,
                stateHashes: hb, materialCounts: mcb,
                gameLength: UInt16(positions),
                workerId: 0, intraWorkerGameIndex: 0,
                outcomes: ob, count: positions)
        }}}}}}}
        return buffer
    }

    /// The whole dedicated-read path: the read, its summary, the
    /// `[LAYER-HEALTH] value-fc1` line and a committed rule-3 observation.
    func testDedicatedReadLogsTheLineAndFeedsRuleThree() async throws {
        try requireMetal()
        let trainer = try makeTrainer()
        let monitor = TrainingHealthMonitor(valueFC1Applicability: .applies)
        let lines = SyncBox<[String]>([])
        let config = try TrainingHealthTestSupport.config()
        let outcome = await TrainingHealthReads.valueFC1Read(
            trainer: trainer, monitor: monitor, config: config, attemptTrainerStep: 0,
            log: { line in lines.modify { $0.append(line.text) } })
        guard case .evaluated(let evaluation) = outcome else {
            return XCTFail("the read must succeed: \(outcome)")
        }
        let line = try XCTUnwrap(lines.value.first { $0.hasPrefix("[LAYER-HEALTH] value-fc1 ") })
        XCTAssertTrue(line.contains(" trained=0 "), line)
        XCTAssertTrue(line.contains(" valueFC1ZeroVel=8/8 "), "no step trained: every unit reads zero — \(line)")
        XCTAssertEqual(monitor.segmentSummary().evaluations, 1)
        // 0 steps trained by this process: rule 3 has no data, nothing raised.
        let committed = try XCTUnwrap(evaluation, "the observation must be committed")
        XCTAssertTrue(committed.events.isEmpty)
    }

    func testRetryWaitsAFullIntervalAfterAFailure() {
        var retry = TrainingHealthValueFC1ReadRetry()
        XCTAssertTrue(retry.allowsRead(atTrainerStep: 1000))
        retry.record(.failed(trainerStep: 1000))
        XCTAssertFalse(retry.allowsRead(atTrainerStep: 1001))
        XCTAssertFalse(retry.allowsRead(atTrainerStep: 1999))
        XCTAssertTrue(retry.allowsRead(atTrainerStep: 2000))
        XCTAssertTrue(retry.allowsRead(atTrainerStep: 900), "a rewound clock allows the read at once")
        retry.record(.evaluated(nil))
        XCTAssertTrue(retry.allowsRead(atTrainerStep: 1001))
    }
}
