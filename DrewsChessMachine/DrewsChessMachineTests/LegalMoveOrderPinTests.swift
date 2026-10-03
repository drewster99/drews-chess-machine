//
//  LegalMoveOrderPinTests.swift
//  DrewsChessMachineTests
//
//  Pins the ORDER of `MoveGenerator.legalMoves(for:)` (determinism plan, A6
//  rows O12 and O13). Two seeded consumers map a random index onto that
//  array: `MoveSampler`'s inverse-CDF walks the legal moves in order, and the
//  BN-calibration walk picks `legalMoves[random.nextBounded(count)]`. With a
//  seeded stream both are reproducible only while the generator emits the
//  same moves in the same order; a reordering (a different board-scan order,
//  a `Set` creeping into the path) would silently change every seeded game
//  and every fresh network's BN statistics without failing any set-based
//  move-generation test.
//
//  Each FEN's ordered UCI list is reduced to the first 16 hex digits of its
//  SHA-256. An intended change to generation order must update these digests
//  deliberately, and is a determinism-breaking change for seeded runs.
//

import CryptoKit
import XCTest
@testable import DrewsChessMachine

final class LegalMoveOrderPinTests: XCTestCase {

    /// Twenty positions covering every move kind: castling both ways and
    /// through attacked squares, en passant (including a pinned en-passant
    /// pawn), promotions and underpromotions, check, double check, pins,
    /// checkmate and stalemate (no legal moves).
    static let pinnedPositions: [(fen: String, digest: String)] = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "a8c9c678855c347d"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "7c1ed2f21cb3d444"),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "1e1fdd5e03255322"),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "5d2cebe37b4704af"),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", "57981249889833c8"),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", "e152d2b82bfd9baa"),
        ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "5cdca166d74a6de5"),
        ("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1", "397f25a77cb9a8e5"),
        ("4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1", "ad8771e7c3297890"),
        ("4k3/1P6/8/8/8/8/6p1/4K3 b - - 0 1", "8a4e64d8350477f1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "2a28702271d8825b"),
        ("r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1", "92a0cbff522711e5"),
        ("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1", "f567f0f5f44867c5"),
        ("4k3/8/8/8/1b6/8/3N4/4K2r w - - 0 1", "55df12673abf706f"),
        ("4k3/8/8/8/8/8/3PPP2/q3K3 w - - 0 1", "e3b0c44298fc1c14"),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "e3b0c44298fc1c14"),
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "63556953b683f7f6"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "5ee98e9f4800ecbe"),
        ("8/P7/8/8/8/8/7p/K6k b - - 0 1", "a5b40418631fb2f0"),
        ("2kr3r/p1ppqpb1/bn2Qnp1/3PN3/1p2P3/2N5/PPPBBPPP/R3K2R b KQ - 3 2", "25b3d48ecf6f2f7a"),
    ]

    static func orderDigest(_ moves: [ChessMove]) -> String {
        let joined = moves.map(\.uci).joined(separator: " ")
        let hash = SHA256.hash(data: Data(joined.utf8))
        return hash.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    func test_legalMoveOrder_isPinnedForFixedPositions() throws {
        for (fen, expected) in Self.pinnedPositions {
            let moves = MoveGenerator.legalMoves(for: try FENParser.parse(fen))
            let digest = Self.orderDigest(moves)
            XCTAssertEqual(digest, expected, "legal-move order changed for \(fen): \(moves.map(\.uci))")
        }
    }

    /// The pin above is only meaningful if repeated generation is stable
    /// within one process too.
    func test_legalMoveOrder_isStableAcrossRepeatedCalls() throws {
        for (fen, _) in Self.pinnedPositions {
            let state = try FENParser.parse(fen)
            let first = MoveGenerator.legalMoves(for: state)
            for _ in 0..<3 {
                XCTAssertEqual(MoveGenerator.legalMoves(for: state), first, fen)
            }
        }
    }
}
