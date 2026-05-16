import XCTest
@testable import DrewsChessMachine

/// Verifies the pointer-flavored `evaluateBatched` overload added in
/// Phase 2 of the self-play tick-driver rework produces byte-identical
/// policy + value outputs to the legacy `[Float]` overload.
///
/// The pointer overload exists so the tick driver can hand its
/// long-lived `UnsafeMutablePointer<Float>` tick-scratch directly to
/// the GPU path without a `[Float]` allocation per fire. Bit-identical
/// output is the load-bearing invariant — any divergence would mean
/// the new path is silently reshaping or reordering data.
final class ChessMPSNetworkPointerEvaluateTests: XCTestCase {

    /// Build a deterministic batch of K boards: K copies of the
    /// standard starting position, encoded with the engine's
    /// `BoardEncoder`. Same input for both paths.
    private func makeBatch(count: Int) -> [Float] {
        let boardFloats = BoardEncoder.tensorLength
        var batch = [Float](repeating: 0, count: count * boardFloats)
        let state = GameState.starting
        var oneBoard = [Float](repeating: 0, count: boardFloats)
        oneBoard.withUnsafeMutableBufferPointer { buf in
            BoardEncoder.encode(state, into: buf)
        }
        for i in 0..<count {
            let dstOffset = i * boardFloats
            for j in 0..<boardFloats {
                batch[dstOffset + j] = oneBoard[j]
            }
        }
        return batch
    }

    func test_pointerOverload_byteIdenticalToArrayOverload() async throws {
        let net = try ChessMPSNetwork(.randomWeights)
        let count = 8
        let batch = makeBatch(count: count)

        // Path 1: the existing [Float] overload.
        nonisolated(unsafe) var policyArr: [Float] = []
        nonisolated(unsafe) var valueArr: [Float] = []
        try await net.evaluateBatched(batchBoards: batch, count: count) { policyBuf, valueBuf in
            policyArr = Array(policyBuf)
            valueArr = Array(valueBuf)
        }

        // Path 2: the new pointer overload, fed the exact same bytes
        // out of a separately-allocated `UnsafeMutablePointer<Float>`
        // (mimics what the tick driver will do with its tick-scratch).
        let floatCount = count * BoardEncoder.tensorLength
        let scratch = UnsafeMutablePointer<Float>.allocate(capacity: floatCount)
        defer {
            scratch.deinitialize(count: floatCount)
            scratch.deallocate()
        }
        scratch.initialize(repeating: 0, count: floatCount)
        for i in 0..<floatCount {
            scratch[i] = batch[i]
        }

        nonisolated(unsafe) var policyPtr: [Float] = []
        nonisolated(unsafe) var valuePtr: [Float] = []
        try await net.evaluateBatched(
            batchBoardsPointer: UnsafePointer(scratch),
            floatCount: floatCount,
            count: count
        ) { policyBuf, valueBuf in
            policyPtr = Array(policyBuf)
            valuePtr = Array(valueBuf)
        }

        XCTAssertEqual(policyArr.count, count * ChessNetwork.policySize)
        XCTAssertEqual(valueArr.count, count)
        XCTAssertEqual(policyPtr.count, policyArr.count, "Pointer-overload policy count differs from Array-overload")
        XCTAssertEqual(valuePtr.count, valueArr.count, "Pointer-overload value count differs from Array-overload")

        // Bit-exact. The pointer overload forwards into the same
        // `internalEvaluate(batchBoards: UnsafeBufferPointer<Float>, ...)`
        // the [Float] path eventually reaches, so any divergence is a
        // wrapping bug (wrong count, wrong stride).
        for i in 0..<policyArr.count {
            XCTAssertEqual(
                policyPtr[i], policyArr[i],
                "Policy divergence at index \(i): pointer=\(policyPtr[i]) array=\(policyArr[i])"
            )
        }
        for i in 0..<valueArr.count {
            XCTAssertEqual(
                valuePtr[i], valueArr[i],
                "Value divergence at index \(i): pointer=\(valuePtr[i]) array=\(valueArr[i])"
            )
        }
    }
}
