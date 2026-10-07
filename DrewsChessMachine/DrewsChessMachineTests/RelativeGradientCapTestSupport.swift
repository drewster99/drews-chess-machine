//
//  RelativeGradientCapTestSupport.swift
//  DrewsChessMachineTests
//
//  Fixtures shared by the relative gradient cap's trainer-level tests: a
//  replay buffer of real positions (the real-data step builds each sample's
//  legal mask from the decoded board, so random planes will not do) and a
//  trainer configured for a given relative-cap mode.
//

import XCTest
@testable import DrewsChessMachine

enum RelativeGradientCapFixture {
    static let batchSize = 32
    static let replayPositions = 512

    /// Deterministic pseudo-random legal games from the starting position (a
    /// fixed LCG picks each move; a game restarts after 80 plies or when no
    /// legal move is left), each position encoded with the played move as its
    /// policy target; the buffer's sampler is seeded, so two buffers built
    /// here draw the same batches.
    static func makeReplayBuffer(arch: NetworkArchitecture) -> ReplayBuffer {
        let encoding = arch.inputEncoding
        let floatsPerBoard = BoardEncoder.tensorLength(for: encoding)
        var boards = [Float](repeating: 0, count: replayPositions * floatsPerBoard)
        var moves = [Int32](repeating: 0, count: replayPositions)
        var plies = [UInt16](repeating: 0, count: replayPositions)
        let taus = [Float](repeating: 1.0, count: replayPositions)
        var hashes = [UInt64](repeating: 0, count: replayPositions)
        let materials = [UInt8](repeating: 32, count: replayPositions)
        var outcomes = [Float](repeating: 0, count: replayPositions)
        var lcg: UInt64 = 0x0123_4567_89AB_CDEF
        func next() -> UInt64 {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return lcg >> 33
        }
        var state = GameState.starting
        var ply = 0
        var filled = 0
        while filled < replayPositions {
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
        let buffer = ReplayBuffer(capacity: replayPositions, inputEncoding: encoding, sampler: DCMRandom(seed: 1))
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
                gameLength: UInt16(replayPositions),
                workerId: 0, intraWorkerGameIndex: 0,
                outcomes: ob, count: replayPositions)
        }}}}}}}
        return buffer
    }

    /// A trainer seeded with `initSeed`, no weight decay (so the velocity
    /// after one step from zero is exactly the clipped gradient), the hard
    /// max `hardMax` and the relative cap `configuration`.
    static func makeTrainer(initSeed: UInt64 = 1, hardMax: Float,
                            configuration: RelativeGradientCapConfiguration) throws -> ChessTrainer {
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1),
            weightDecayC: 0,
            gradClipMaxNorm: hardMax,
            relativeGradientCap: configuration,
            arch: .current,
            initialization: .seeded(initSeed: initSeed)
        )
        return trainer
    }

    static func configuration(_ mode: RelativeGradientCapMode, k: Double = 1, n: Int = 4, w: Int = 2,
                              floor: Double = 1.0e-6) throws -> RelativeGradientCapConfiguration {
        try RelativeGradientCapConfiguration(mode: mode, multiple: k, windowSteps: n, minimumHistorySteps: w, floor: floor)
    }

    /// One real-data step; fails the test if the buffer could not fill a batch.
    static func step(_ trainer: ChessTrainer, _ buffer: ReplayBuffer,
                     file: StaticString = #filePath, line: UInt = #line) async throws -> TrainStepTiming {
        let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize)
        return try XCTUnwrap(timing, "the fixture buffer holds more than one batch", file: file, line: line)
    }
}
