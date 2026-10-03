//
//  Basic24EncodingTests.swift
//  DrewsChessMachineTests
//
//  `basic24` is `basic30` without the six temporal-repetition planes that can
//  never fire in a legal game (GitHub issue #10): repetition 1, 2, 3, 5, 7
//  and 9 plies ago. These tests pin the claim those planes are always zero,
//  that `basic24` carries exactly `basic30`'s information in its own plane
//  order, and that a 24-plane model builds, evaluates and round-trips while
//  30-plane models are unaffected.
//

import XCTest
@testable import DrewsChessMachine

final class Basic24EncodingTests: XCTestCase {

    private static let area = 64

    // MARK: - Structure

    func testStructure() {
        let e = InputEncoding.basic24
        XCTAssertEqual(e.planeCount, 24)
        XCTAssertEqual(e.historyFrameCount, 1)
        XCTAssertEqual(e.planesPerFrame, 24)
        XCTAssertEqual(e.tailPlaneCount, 0)
        XCTAssertEqual(BoardEncoder.tensorLength(for: e), 24 * Self.area)
        XCTAssertEqual(e.channelNames.count, 24)
        XCTAssertEqual(e.shortChannelNames.count, 24)
        XCTAssertEqual(e.analyzerPlaneLabels.count, 24)
    }

    /// The kept distances are exactly the ply distances in the ten-ply
    /// window at which a strict duplicate is possible: even (same side to
    /// move) and at least four (a two-ply return is impossible).
    func testKeptDistancesAreTheEvenDistancesFromFour() {
        let derived = (1...10).filter { $0 % 2 == 0 && $0 >= 4 }
        XCTAssertEqual(InputEncoding.possibleRepetitionPlyDistances, derived)
    }

    func testChannelNamesMatchTheKeptBasic30Planes() {
        let basic30 = InputEncoding.basic30.channelNames
        let basic24 = InputEncoding.basic24.channelNames
        XCTAssertEqual(Array(basic24.prefix(20)), Array(basic30.prefix(20)))
        for (offset, distance) in InputEncoding.possibleRepetitionPlyDistances.enumerated() {
            XCTAssertEqual(basic24[20 + offset], basic30[19 + distance])
        }
    }

    // MARK: - Content parity with basic30

    /// Every position of `cycle` (alternating white and black moves, each
    /// `(fromRow, fromCol, toRow, toCol)`) played `repeats` times from
    /// `start`, past every threefold (the engine leaves draws to the caller).
    private static func cyclePositions(
        from start: GameState, _ cycle: [(Int, Int, Int, Int)], repeats: Int
    ) throws -> [GameState] {
        let engine = ChessGameEngine(state: start, adjudication: .serverAuthoritative)
        var out: [GameState] = []
        for _ in 0..<repeats {
            for (fromRow, fromCol, toRow, toCol) in cycle {
                let move = ChessMove(fromRow: fromRow, fromCol: fromCol, toRow: toRow, toCol: toCol, promotion: nil)
                XCTAssertTrue(MoveGenerator.legalMoves(for: engine.state).contains(move), "fixture move \(move) is illegal")
                _ = try engine.applyMoveAndAdvance(move)
                out.append(engine.state)
            }
        }
        return out
    }

    /// Positions covering the start, ordinary play, and repeated positions
    /// at every possible ply distance (4-, 6-, 8- and 10-ply cycles), plus
    /// deterministic pseudo-random games.
    private static func positions() throws -> [GameState] {
        var out: [GameState] = [.starting]
        // A 4-ply knight shuffle from the opening position: Nf3 Nf6 Ng1 Ng8.
        out += try cyclePositions(from: .starting,
                                  [(7, 6, 5, 5), (0, 6, 2, 5), (5, 5, 7, 6), (2, 5, 0, 6)], repeats: 3)
        // King tours of 2, 3, 4 and 5 moves per side (4-, 6-, 8- and 10-ply
        // cycles; a knight needs an even number of moves to return, so the
        // odd tours use kings). White e1 → …, black e8 → … in step.
        let kings = try FENParser.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        let tours: [[(Int, Int, Int, Int)]] = [
            // e1-d1-e1 / e8-d8-e8
            [(7, 4, 7, 3), (0, 4, 0, 3), (7, 3, 7, 4), (0, 3, 0, 4)],
            // e1-d1-d2-e1 / e8-d8-d7-e8
            [(7, 4, 7, 3), (0, 4, 0, 3), (7, 3, 6, 3), (0, 3, 1, 3), (6, 3, 7, 4), (1, 3, 0, 4)],
            // e1-d1-d2-e2-e1 / e8-d8-d7-e7-e8
            [(7, 4, 7, 3), (0, 4, 0, 3), (7, 3, 6, 3), (0, 3, 1, 3), (6, 3, 6, 4), (1, 3, 1, 4),
             (6, 4, 7, 4), (1, 4, 0, 4)],
            // e1-d1-c2-d3-e2-e1 / e8-d8-c7-d6-e7-e8
            [(7, 4, 7, 3), (0, 4, 0, 3), (7, 3, 6, 2), (0, 3, 1, 2), (6, 2, 5, 3), (1, 2, 2, 3),
             (5, 3, 6, 4), (2, 3, 1, 4), (6, 4, 7, 4), (1, 4, 0, 4)],
        ]
        for tour in tours {
            out += try cyclePositions(from: kings, tour, repeats: 3)
        }
        // Deterministic pseudo-random games.
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return seed >> 33
        }
        for _ in 0..<24 {
            let engine = ChessGameEngine(adjudication: .serverAuthoritative)
            for _ in 0..<80 {
                let legal = MoveGenerator.legalMoves(for: engine.state)
                if legal.isEmpty { break }
                _ = try engine.applyMoveAndAdvance(legal[Int(next() % UInt64(legal.count))])
                out.append(engine.state)
            }
        }
        return out
    }

    private func plane(_ tensor: [Float], _ index: Int) -> ArraySlice<Float> {
        tensor[(index * Self.area)..<((index + 1) * Self.area)]
    }

    func testBasic24CarriesExactlyTheBasic30Information() throws {
        let positions = try Self.positions()
        var firedAtDistance = Set<Int>()
        for state in positions {
            let basic30 = BoardEncoder.encode(state, encoding: .basic30)
            let basic24 = BoardEncoder.encode(state, encoding: .basic24)
            XCTAssertEqual(Array(basic24.prefix(20 * Self.area)), Array(basic30.prefix(20 * Self.area)))
            for (offset, distance) in InputEncoding.possibleRepetitionPlyDistances.enumerated() {
                XCTAssertEqual(plane(basic24, 20 + offset), plane(basic30, 19 + distance),
                               "basic24 plane \(20 + offset) must equal basic30's \(distance)-ply repetition plane")
                if plane(basic30, 19 + distance).contains(1) { firedAtDistance.insert(distance) }
            }
            for distance in [1, 2, 3, 5, 7, 9] {
                XCTAssertTrue(plane(basic30, 19 + distance).allSatisfy { $0 == 0 },
                              "basic30's \(distance)-ply repetition plane fired; it can never be 1 in legal play")
            }
        }
        XCTAssertEqual(firedAtDistance, Set(InputEncoding.possibleRepetitionPlyDistances),
                       "the fixtures must exercise every kept repetition plane")
    }

    // MARK: - Models

    private static func smallArchitecture(_ encoding: InputEncoding) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: encoding, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32
        )
    }

    func testA24PlaneNetworkBuildsAndEvaluates() async throws {
        let arch = Self.smallArchitecture(.basic24)
        try arch.validate()
        XCTAssertEqual(arch.inputPlanes, 24)
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 1), arch: arch)
        let board = BoardEncoder.encode(.starting, encoding: .basic24)
        try await net.evaluate(board: board) { policyBuf, value in
            XCTAssertEqual(policyBuf.count, arch.policySize)
            XCTAssertTrue(value.isFinite)
        }
    }

    /// The training graph (autodiff over the whole network) builds for a
    /// 24-plane stem and takes a 24-plane champion's weights.
    func testA24PlaneTrainerBuildsAndLoadsChampionWeights() async throws {
        let arch = Self.smallArchitecture(.basic24)
        let champion = try ChessMPSNetwork(.randomWeights(initSeed: 2), arch: arch)
        let weights = try await champion.network.exportWeights()
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: arch, initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.arch.inputPlanes, 24)
        try await trainer.network.loadWeights(weights)
    }

    func testModelFileRoundTripKeepsEachEncoding() throws {
        for encoding in [InputEncoding.basic24, .basic30] {
            let arch = Self.smallArchitecture(encoding)
            let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
                (0..<spec.elementCount).map { Float(tensorIndex + $0 % 7) * 0.125 }
            }
            let data = try SafetensorsModelIO.encode(
                modelID: "20261002-1-ENC\(encoding.planeCount)", createdAtUnix: 1_790_000_000,
                metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: ""),
                weights: weights, architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
            let decoded = try SafetensorsModelIO.decode(data, valueHead: .asStored, source: "fixture")
            XCTAssertEqual(decoded.architecture, arch)
            XCTAssertEqual(decoded.architecture.inputEncoding, encoding)
        }
    }

    func testReplayBufferUsesThe24PlaneStride() {
        let buffer = ReplayBuffer(capacity: 8, inputEncoding: .basic24, sampler: DCMRandom(seed: 1))
        XCTAssertEqual(buffer.floatsPerBoard, 24 * Self.area)
        XCTAssertEqual(ReplayBuffer.singleFrameEncoding(forStoredStride: 24 * Self.area), .basic24)
        XCTAssertEqual(ReplayBuffer.singleFrameEncoding(forStoredStride: 30 * Self.area), .basic30)
    }

    // MARK: - New-model default

    /// Newly built models default to the 24-plane encoding; everything else
    /// is the current preset, which (like every existing model) keeps its
    /// 30-plane encoding.
    func testNewModelDefaultIsTheCurrentPresetWith24Planes() {
        var expected = NetworkArchitecture.current
        XCTAssertEqual(expected.inputEncoding, .basic30)
        expected.inputEncoding = .basic24
        XCTAssertEqual(NetworkArchitecture.newModelDefault, expected)
    }
}
