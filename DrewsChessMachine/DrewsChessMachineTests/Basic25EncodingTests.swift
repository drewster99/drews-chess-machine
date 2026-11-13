//
//  Basic25EncodingTests.swift
//  DrewsChessMachineTests
//
//  `basic25` is `basic24` plus plane 24, 1.0 on every square: a board-edge
//  marker for zero-padded convolutions (owner decision 2026-11-12). These
//  tests pin that its first 24 planes are exactly `basic24`'s, that the
//  replay buffer stores only those 24 planes yet samples the full 25 (the
//  constant plane written at sample time, through save and load too), that a
//  25-plane model builds, evaluates and round-trips, and that new models
//  default to it.
//

import XCTest
@testable import DrewsChessMachine

final class Basic25EncodingTests: XCTestCase {

    private static let area = 64
    private static let basic24Floats = 24 * area
    private static let basic25Floats = 25 * area

    // MARK: - Structure

    func testStructure() {
        let e = InputEncoding.basic25
        XCTAssertEqual(e.planeCount, 25)
        XCTAssertEqual(e.historyFrameCount, 1)
        XCTAssertEqual(e.planesPerFrame, 24)
        XCTAssertEqual(e.tailPlanes, .boardEdgeMarker)
        XCTAssertEqual(e.tailPlaneCount, 1)
        XCTAssertEqual(BoardEncoder.tensorLength(for: e), Self.basic25Floats)
        XCTAssertEqual(e.channelNames.count, 25)
        XCTAssertEqual(e.shortChannelNames.count, 25)
        XCTAssertEqual(e.analyzerPlaneLabels.count, 25)
    }

    /// The plane groups are `basic24`'s, then one group for plane 24 alone;
    /// together they cover exactly planes 0…24.
    func testPlaneGroupsAreBasic24sThenTheBoardEdgePlane() {
        let groups = InputEncoding.basic25.planeGroups
        XCTAssertEqual(Array(groups.dropLast()), InputEncoding.basic24.planeGroups)
        XCTAssertEqual(groups.last?.range, 24...24)
        var covered: [Int] = []
        for group in groups { covered += Array(group.range) }
        XCTAssertEqual(covered, Array(0...24))
    }

    func testChannelNamesAreBasic24sThenTheBoardEdgePlane() {
        let pairs: [([String], [String])] = [
            (InputEncoding.basic25.channelNames, InputEncoding.basic24.channelNames),
            (InputEncoding.basic25.shortChannelNames, InputEncoding.basic24.shortChannelNames),
            (InputEncoding.basic25.analyzerPlaneLabels, InputEncoding.basic24.analyzerPlaneLabels),
        ]
        for (basic25, basic24) in pairs {
            XCTAssertEqual(Array(basic25.prefix(24)), basic24)
            XCTAssertFalse(basic24.contains(basic25[24]), "plane 24's name must not repeat a basic24 name")
        }
    }

    // MARK: - Encoder

    /// Positions covering both sides to move, castling rights held and lost,
    /// an en-passant square, a repetition (4-ply knight shuffle) and a
    /// nonzero halfmove clock, plus deterministic pseudo-random games.
    private static func positions() throws -> [GameState] {
        var out: [GameState] = [.starting]
        let afterE4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil), to: .starting)
        out.append(afterE4)                                  // black to move, EP square set
        out.append(MoveGenerator.applyMove(
            ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil), to: afterE4))
        // Only some castling rights, a halfmove clock, and an EP square.
        out.append(try FENParser.parse("r3k2r/pppq1ppp/8/3pP3/8/8/PPP2PPP/R3K2R w Kq d6 0 9"))
        out.append(try FENParser.parse("4k3/8/8/8/8/8/8/4K3 b - - 37 80"))
        // Nf3 Nf6 Ng1 Ng8, twice: the start position recurs 4 and 8 plies ago.
        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        let shuffle = [(7, 6, 5, 5), (0, 6, 2, 5), (5, 5, 7, 6), (2, 5, 0, 6)]
        for (fromRow, fromCol, toRow, toCol) in shuffle + shuffle {
            _ = try engine.applyMoveAndAdvance(
                ChessMove(fromRow: fromRow, fromCol: fromCol, toRow: toRow, toCol: toCol, promotion: nil))
            out.append(engine.state)
        }
        out += try pseudoRandomGame(seed: 0x9E37_79B9_7F4A_7C15, plies: 120).map(\.state)
        return out
    }

    /// One ply of a played game as the self-play driver records it.
    private struct PlayedPly {
        let ply: Int
        let state: GameState
    }

    /// Every position (including the start) of a deterministic pseudo-random
    /// game of up to `plies` plies.
    private static func pseudoRandomGame(seed initialSeed: UInt64, plies: Int) throws -> [PlayedPly] {
        var seed = initialSeed
        func next() -> UInt64 {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return seed >> 33
        }
        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        var out = [PlayedPly(ply: 0, state: engine.state)]
        for ply in 1...plies {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            if legal.isEmpty { break }
            _ = try engine.applyMoveAndAdvance(legal[Int(next() % UInt64(legal.count))])
            out.append(PlayedPly(ply: ply, state: engine.state))
        }
        return out
    }

    func testPlanes0Through23AreBasic24AndPlane24IsAllOnes() throws {
        var sawRepetition = false
        var sawBlackToMove = false
        for state in try Self.positions() {
            let basic25 = BoardEncoder.encode(state, encoding: .basic25)
            let basic24 = BoardEncoder.encode(state, encoding: .basic24)
            XCTAssertEqual(basic25.count, Self.basic25Floats)
            XCTAssertEqual(Array(basic25.prefix(Self.basic24Floats)), basic24)
            XCTAssertEqual(Array(basic25[Self.basic24Floats...]), [Float](repeating: 1, count: Self.area))
            if basic24[(20 * Self.area)...].contains(1) { sawRepetition = true }
            if state.currentPlayer == .black { sawBlackToMove = true }
        }
        XCTAssertTrue(sawRepetition, "the fixtures must include a temporal repetition")
        XCTAssertTrue(sawBlackToMove, "the fixtures must include black to move")
    }

    /// The caller-buffer form zero-fills first, so a reused scratch holding
    /// garbage still encodes exactly what the array form does.
    func testEncodingIntoAReusedBufferMatchesTheArrayForm() throws {
        var scratch = [Float](repeating: .nan, count: Self.basic25Floats)
        for state in try Self.positions() {
            scratch.withUnsafeMutableBufferPointer { buffer in
                BoardEncoder.encode(state, into: buffer, encoding: .basic25)
            }
            XCTAssertEqual(scratch, BoardEncoder.encode(state, encoding: .basic25))
        }
    }

    // MARK: - Replay buffer

    func testTheBufferStoresBasic24sPlanesAndSamples25() {
        let buffer = ReplayBuffer(capacity: 8, inputEncoding: .basic25, sampler: DCMRandom(seed: 1))
        XCTAssertEqual(buffer.floatsPerBoard, Self.basic24Floats)
        XCTAssertEqual(buffer.reconstructedStride, Self.basic25Floats)
        XCTAssertEqual(buffer.bytesPerPosition,
                       ReplayBuffer(capacity: 8, inputEncoding: .basic24, sampler: DCMRandom(seed: 1)).bytesPerPosition,
                       "the constant plane must cost no buffer memory")
        // A stored stride alone (the verify scratch) still resolves to basic24.
        XCTAssertEqual(ReplayBuffer.singleFrameEncoding(forStoredStride: Self.basic24Floats), .basic24)
    }

    /// Append one game as `ActiveGame.flush` does: one block in reverse ply
    /// order, each row the first `floatsPerBoard` floats of the full encoding.
    private func appendGame(_ game: [PlayedPly], to buffer: ReplayBuffer, gameIndex: UInt32) throws {
        let stride = buffer.floatsPerBoard
        let total = game.count
        var boards = [Float](repeating: 0, count: total * stride)
        var policy = [Int32](repeating: 0, count: total)
        var plies = [UInt16](repeating: 0, count: total)
        let taus = [Float](repeating: 1, count: total)
        var hashes = [UInt64](repeating: 0, count: total)
        let materials = [UInt8](repeating: 0, count: total)
        var outcomes = [Float](repeating: 0, count: total)
        for row in 0..<total {
            let played = game[total - 1 - row]
            let frame = Array(BoardEncoder.encode(played.state, encoding: .basic25).prefix(stride))
            boards.replaceSubrange((row * stride)..<((row + 1) * stride), with: frame)
            policy[row] = Int32(played.ply)
            plies[row] = UInt16(played.ply)
            hashes[row] = try frame.withUnsafeBufferPointer { pointer in
                ReplayBuffer.hashBoard(try XCTUnwrap(pointer.baseAddress), count: stride)
            }
            outcomes[row] = played.ply % 2 == 0 ? 1 : -1
        }
        try boards.withUnsafeBufferPointer { b in
        try policy.withUnsafeBufferPointer { pi in
        try plies.withUnsafeBufferPointer { pl in
        try taus.withUnsafeBufferPointer { t in
        try hashes.withUnsafeBufferPointer { h in
        try materials.withUnsafeBufferPointer { m in
        try outcomes.withUnsafeBufferPointer { o in
            buffer.append(
                boards: try XCTUnwrap(b.baseAddress), policyIndices: try XCTUnwrap(pi.baseAddress),
                plyIndices: try XCTUnwrap(pl.baseAddress), samplingTaus: try XCTUnwrap(t.baseAddress),
                stateHashes: try XCTUnwrap(h.baseAddress), materialCounts: try XCTUnwrap(m.baseAddress),
                gameLength: UInt16(total), workerId: 3, intraWorkerGameIndex: gameIndex,
                outcomes: try XCTUnwrap(o.baseAddress), count: total)
        }}}}}}}
    }

    /// Every resident position, materialized through the buffer's own path,
    /// equals the encoder's full 25-plane encoding.
    private func assertEveryPositionMaterializesAsTheEncoderWrites(
        _ game: [PlayedPly], in buffer: ReplayBuffer, gameIndex: UInt32,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let packedId = ReplayBuffer.packWorkerGameId(workerId: 3, gameIndex: gameIndex)
        for played in game {
            var got = [Float](repeating: .nan, count: Self.basic25Floats)
            let found = try got.withUnsafeMutableBufferPointer { pointer in
                buffer.reconstructStack(forWorkerGameId: packedId, plyIndex: UInt16(played.ply),
                                        into: try XCTUnwrap(pointer.baseAddress))
            }
            XCTAssertTrue(found, "no resident slot for ply \(played.ply)", file: file, line: line)
            XCTAssertEqual(got, BoardEncoder.encode(played.state, encoding: .basic25),
                           "ply \(played.ply)", file: file, line: line)
        }
    }

    func testStoredPositionsMaterializeAsTheEncoderWrites() throws {
        let game = try Self.pseudoRandomGame(seed: 0x1234_5678_9ABC_DEF1, plies: 60)
        let buffer = ReplayBuffer(capacity: 256, inputEncoding: .basic25, sampler: DCMRandom(seed: 2))
        try appendGame(game, to: buffer, gameIndex: 1)
        try assertEveryPositionMaterializesAsTheEncoderWrites(game, in: buffer, gameIndex: 1)
    }

    /// `sample` (the trainer's path) writes 25 planes per position: the
    /// stored 24, then plane 24 all ones — no plane left over from the
    /// staging's previous contents.
    func testSampledPositionsCarryTheStoredPlanesAndTheBoardEdgePlane() throws {
        let game = try Self.pseudoRandomGame(seed: 0x0F0F_1234_ABCD_0001, plies: 60)
        let buffer = ReplayBuffer(capacity: 256, inputEncoding: .basic25, sampler: DCMRandom(seed: 3))
        try appendGame(game, to: buffer, gameIndex: 1)
        let count = 32
        var boards = [Float](repeating: .nan, count: count * Self.basic25Floats)
        var moves = [Int32](repeating: 0, count: count)
        var zs = [Float](repeating: 0, count: count)
        var plies = [UInt16](repeating: 0, count: count)
        let drew = try boards.withUnsafeMutableBufferPointer { b in
        try moves.withUnsafeMutableBufferPointer { m in
        try zs.withUnsafeMutableBufferPointer { z in
        try plies.withUnsafeMutableBufferPointer { p in
            buffer.sample(count: count, intoBoards: try XCTUnwrap(b.baseAddress), moves: try XCTUnwrap(m.baseAddress),
                          zs: try XCTUnwrap(z.baseAddress), plies: try XCTUnwrap(p.baseAddress))
        }}}}
        XCTAssertTrue(drew)
        let stateByPly = Dictionary(uniqueKeysWithValues: game.map { ($0.ply, $0.state) })
        for i in 0..<count {
            let sampled = Array(boards[(i * Self.basic25Floats)..<((i + 1) * Self.basic25Floats)])
            let state = try XCTUnwrap(stateByPly[Int(plies[i])])
            XCTAssertEqual(sampled, BoardEncoder.encode(state, encoding: .basic25), "sample \(i), ply \(plies[i])")
            XCTAssertEqual(Array(sampled[Self.basic24Floats...]), [Float](repeating: 1, count: Self.area))
        }
    }

    /// The saved file records basic24's stored stride and basic25's tag; it
    /// restores into a basic25 buffer that samples all 25 planes, and into
    /// the stride-only verify scratch (which resolves to basic24).
    func testBufferFileRoundTrips() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-basic25-replay-\(UUID().uuidString).bin")
        defer {
            do {
                if FileManager.default.fileExists(atPath: file.path) {
                    try FileManager.default.removeItem(at: file)
                }
            } catch {
                XCTFail("could not remove \(file.path): \(error)")
            }
        }
        let game = try Self.pseudoRandomGame(seed: 0xDEAD_BEEF_0000_0025, plies: 40)
        let buffer = ReplayBuffer(capacity: 128, inputEncoding: .basic25, sampler: DCMRandom(seed: 4))
        try appendGame(game, to: buffer, gameIndex: 7)
        _ = try buffer.write(to: file)
        XCTAssertEqual(try ReplayBuffer.peekFloatsPerBoard(at: file), Self.basic24Floats)

        let restored = ReplayBuffer(capacity: 128, inputEncoding: .basic25, sampler: DCMRandom(seed: 5))
        try restored.restore(from: file)
        XCTAssertEqual(restored.count, game.count)
        try assertEveryPositionMaterializesAsTheEncoderWrites(game, in: restored, gameIndex: 7)

        let verifyScratch = ReplayBuffer(capacity: 128, floatsPerBoard: Self.basic24Floats, sampler: DCMRandom(seed: 6))
        XCTAssertEqual(verifyScratch.inputEncoding, .basic24)
        try verifyScratch.restore(from: file)
        XCTAssertEqual(verifyScratch.count, game.count)
    }

    // MARK: - Models

    private static func smallArchitecture() -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: .basic25, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32
        )
    }

    func testA25PlaneNetworkBuildsAndEvaluates() async throws {
        let arch = Self.smallArchitecture()
        try arch.validate()
        XCTAssertEqual(arch.inputPlanes, 25)
        XCTAssertEqual(arch.weightTensorPlan().first { $0.name == "stem.conv.weight" }?.shape, [16, 25, 3, 3])
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 1), arch: arch)
        let board = BoardEncoder.encode(.starting, encoding: .basic25)
        try await net.evaluate(board: board) { policyBuf, value in
            XCTAssertEqual(policyBuf.count, arch.policySize)
            XCTAssertTrue(value.isFinite)
        }
    }

    /// The training graph builds for a 25-plane stem and takes a 25-plane
    /// champion's weights.
    func testA25PlaneTrainerBuildsAndLoadsChampionWeights() async throws {
        let arch = Self.smallArchitecture()
        let champion = try ChessMPSNetwork(.randomWeights(initSeed: 2), arch: arch)
        let weights = try await champion.network.exportWeights()
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: arch, initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.arch.inputPlanes, 25)
        try await trainer.network.loadWeights(weights)
    }

    func testModelFileRoundTripKeepsBasic25() throws {
        let arch = Self.smallArchitecture()
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex + $0 % 7) * 0.125 }
        }
        let data = try SafetensorsModelIO.encode(
            modelID: "20261112-1-ENC25", createdAtUnix: 1_794_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: ""),
            weights: weights, architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let decoded = try SafetensorsModelIO.decode(data, valueHead: .asStored, source: "fixture")
        XCTAssertEqual(decoded.architecture, arch)
        XCTAssertEqual(decoded.architecture.inputEncoding, .basic25)
    }

    // MARK: - New-model default

    /// Newly built models default to the 25-plane encoding; everything else
    /// is the current preset, which (like every existing model) keeps its
    /// 30-plane encoding.
    func testNewModelDefaultIsTheCurrentPresetWithBasic25() {
        var expected = NetworkArchitecture.current
        XCTAssertEqual(expected.inputEncoding, .basic30)
        expected.inputEncoding = .basic25
        XCTAssertEqual(NetworkArchitecture.newModelDefault, expected)
    }
}
