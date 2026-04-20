//
//  MPSGraphReshapeLayoutTests.swift
//  DrewsChessMachineTests
//
//  Verifies empirically that MPSGraph's `reshape` op treats a 4D
//  `[B, C, H, W]` tensor as NCHW row-major (C-order, last dim
//  contiguous) when flattening to `[B, C*H*W]`. This is the load-
//  bearing assumption behind `PolicyEncoding.policyIndex = c*64 + r*8
//  + col` matching the flat `[batch, policySize]` policy output that
//  the training graph one-hots against.
//
//  Apple's public MPSGraph docs do NOT formally guarantee the flatten
//  ordering. If MPSGraph ever used column-major flatten, or inserted a
//  transpose to NHWC internally around reshape, every training-time
//  one-hot would target the wrong class and every sampling-time logit
//  lookup would read a different class than the network thought it
//  emitted — with NO compile-time or runtime error. These tests run
//  the ops on a real MTLDevice and assert the round-trip is an
//  identity permutation under the assumed formula.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class MPSGraphReshapeLayoutTests: XCTestCase {

    // MARK: - Test 1: [B, C, H, W] -> [B, C*H*W] is NCHW row-major
    //
    // Build a constant tensor of shape [1, 76, 8, 8] where the value
    // at (0, c, r, col) equals c * 64 + r * 8 + col (= the flat index
    // under NCHW row-major). Reshape to [1, 4864] and assert
    // out[0, i] == Float(i) for every i. Any other flatten traversal
    // (NHWC, column-major, etc.) would permute the output and the
    // equality would fail at many positions.
    func testReshapeIsNCHWRowMajorForward() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let dtype = MPSDataType.float32

        let C = ChessNetwork.policyChannels   // 76
        let H = 8
        let W = 8
        let flatSize = C * H * W              // 4864
        XCTAssertEqual(flatSize, ChessNetwork.policySize,
                       "policySize should equal policyChannels * 64")

        // Fill a buffer shaped [1, C, H, W] under the assumption that
        // index (0, c, r, col) sits at linear offset c*H*W + r*W + col.
        // If that assumption holds, reshape to [1, flatSize] and a
        // readback will produce [0, 1, 2, ..., flatSize - 1].
        var values = [Float](repeating: 0, count: flatSize)
        for c in 0..<C {
            for r in 0..<H {
                for col in 0..<W {
                    let offset = c * H * W + r * W + col
                    values[offset] = Float(offset)
                }
            }
        }

        let input = graph.constant(
            makeFloatData(values),
            shape: [1, NSNumber(value: C), NSNumber(value: H), NSNumber(value: W)],
            dataType: dtype
        )
        let flat = graph.reshape(
            input,
            shape: [1, NSNumber(value: flatSize)],
            name: "flatten_nchw"
        )

        let results = graph.run(
            with: cmdQueue,
            feeds: [:],
            targetTensors: [flat],
            targetOperations: nil
        )
        guard let out = results[flat] else {
            XCTFail("graph.run did not return the reshape output")
            return
        }
        let read = readFloats(out, count: flatSize)

        var firstMismatch: (i: Int, got: Float, expected: Float)?
        for i in 0..<flatSize where read[i] != Float(i) {
            firstMismatch = (i, read[i], Float(i))
            break
        }
        if let m = firstMismatch {
            XCTFail(
                "MPSGraph reshape from [1, \(C), \(H), \(W)] to [1, \(flatSize)] "
                + "does NOT match NCHW row-major. First mismatch at flat index \(m.i): "
                + "got \(m.got), expected \(m.expected). "
                + "PolicyEncoding.policyIndex = c*64 + r*8 + col would be WRONG. "
                + "Training targets and sampling logits would be permuted against each other."
            )
        }
    }

    // MARK: - Test 2: [B, C*H*W] -> [B, C, H, W] inverts the forward
    //
    // The reverse check. Build a flat tensor [1, 4864] whose element i
    // equals i. Reshape to [1, 76, 8, 8]. Assert out[0, c, r, col] ==
    // c*64 + r*8 + col. Catches a scenario where forward reshape
    // happens to produce the right layout but reverse reshape does not
    // (e.g., if reshape's flatten is a view but unflatten copies).
    func testReshapeIsNCHWRowMajorReverse() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let dtype = MPSDataType.float32

        let C = ChessNetwork.policyChannels
        let H = 8
        let W = 8
        let flatSize = C * H * W

        var flatValues = [Float](repeating: 0, count: flatSize)
        for i in 0..<flatSize {
            flatValues[i] = Float(i)
        }

        let flatInput = graph.constant(
            makeFloatData(flatValues),
            shape: [1, NSNumber(value: flatSize)],
            dataType: dtype
        )
        let unflat = graph.reshape(
            flatInput,
            shape: [1, NSNumber(value: C), NSNumber(value: H), NSNumber(value: W)],
            name: "unflatten_nchw"
        )

        let results = graph.run(
            with: cmdQueue,
            feeds: [:],
            targetTensors: [unflat],
            targetOperations: nil
        )
        guard let out = results[unflat] else {
            XCTFail("graph.run did not return the reshape output")
            return
        }
        let read = readFloats(out, count: flatSize)

        var firstMismatch: (c: Int, r: Int, col: Int, got: Float, expected: Float)?
        outer: for c in 0..<C {
            for r in 0..<H {
                for col in 0..<W {
                    let linear = c * H * W + r * W + col
                    let expected = Float(linear)
                    if read[linear] != expected {
                        firstMismatch = (c, r, col, read[linear], expected)
                        break outer
                    }
                }
            }
        }
        if let m = firstMismatch {
            XCTFail(
                "Unflatten [1, \(flatSize)] -> [1, \(C), \(H), \(W)] does NOT place "
                + "flat index i at (c=\(m.c), r=\(m.r), col=\(m.col)) where "
                + "c*64 + r*8 + col = \(m.c * 64 + m.r * 8 + m.col). "
                + "Got \(m.got), expected \(m.expected)."
            )
        }
    }

    // MARK: - Test 3: End-to-end through PolicyEncoding + oneHot +
    // softMaxCrossEntropy
    //
    // Checks that the policy-head reshape formula `c*64 + r*8 + col`
    // agrees with what MPSGraph's oneHot + softMaxCrossEntropy
    // actually consume. For a single known move we:
    //   1. Compute (chan, r, col) via PolicyEncoding.encode.
    //   2. Compute idx via PolicyEncoding.policyIndex (= c*64+r*8+col).
    //   3. Build a [1, 76, 8, 8] logits tensor that is zero everywhere
    //      except value L at (0, chan, r, col).
    //   4. Reshape to [1, 4864] (same op the ChessNetwork policy head
    //      uses) and feed to graph.softMaxCrossEntropy alongside a
    //      one-hot at flat index `idx`.
    //   5. The closed-form CE loss for "all zeros except logit L at
    //      the one-hot's target class, over a D-way softmax" is
    //          -L + log(exp(L) + (D-1) * exp(0))
    //      Verify numerically. Any mismatch in reshape order would
    //      place logit L at the wrong flat position, the one-hot
    //      would target a class with logit 0, and the loss would be
    //          0 - log(exp(L) + (D-1)) + L·0... wait — specifically,
    //      CE = -log softmax(logits)[target] = -log(1 / (exp(L) +
    //      D-1)) = log(exp(L) + D-1). I.e., no leading -L term when
    //      the target class has logit 0 instead of L. So a layout
    //      mismatch is detected as a large loss discrepancy.
    func testEndToEndPolicyEncodingMatchesReshapeAndOneHot() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let cmdQueue = device.makeCommandQueue()!
        let graph = MPSGraph()
        let dtype = MPSDataType.float32

        // Pick a move whose (channel, row, col) exercises a non-zero
        // channel, non-zero row, non-zero col — so a "dumb" flatten
        // that gets any axis ordering wrong (e.g. col*76*8 + r*76 +
        // chan under a HWCN traversal) would land at a different flat
        // index. 1. e4 in the starting position is too bland (fromRow=6
        // which is rich, fromCol=4 rich, but channel 4 only exercises
        // the low queen-style block). Use a knight move on a
        // mid-board square to hit a knight channel (56..63).
        let state: GameState = .starting
        // Pick Nf3 (knight from g1 = (row=7, col=6) to f3 = (row=5, col=5)):
        // in encoder frame (white to move, no flip), from (7, 6) to (5, 5),
        // dr = -2, dc = -1 = knight "up-left" = jump index 7.
        // Channel = 56 + 7 = 63.
        let move = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
        // Confirm the move is legal in the starting position so this
        // test stays meaningful.
        XCTAssertTrue(
            MoveGenerator.legalMoves(for: state).contains(move),
            "Nf3 must be legal in the starting position"
        )
        let (chan, r, col) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
        let idx = PolicyEncoding.policyIndex(move, currentPlayer: state.currentPlayer)
        XCTAssertEqual(idx, chan * 64 + r * 8 + col,
                       "policyIndex formula must match c*64+r*8+col by construction")

        let C = ChessNetwork.policyChannels
        let H = 8
        let W = 8
        let flatSize = C * H * W

        // Build logits [1, C, H, W] with a single non-zero at the
        // encoder-frame (chan, r, col) position. The magnitude L is
        // chosen large enough that a wrong-target CE loss (where the
        // one-hot hits a zero-logit cell) would differ from the
        // correct CE loss by multiple orders of magnitude.
        let L: Float = 5.0
        var logitsValues = [Float](repeating: 0, count: flatSize)
        logitsValues[chan * H * W + r * W + col] = L

        let logits4D = graph.constant(
            makeFloatData(logitsValues),
            shape: [1, NSNumber(value: C), NSNumber(value: H), NSNumber(value: W)],
            dataType: dtype
        )
        // Exact same reshape the ChessNetwork policy head uses.
        let logitsFlat = graph.reshape(
            logits4D,
            shape: [1, NSNumber(value: flatSize)],
            name: "policy_flatten_test"
        )

        // One-hot at flat index `idx`, depth `flatSize`, axis 1 on
        // an input of shape [1] — produces [1, flatSize].
        let indicesData = makeInt32Data([Int32(idx)])
        let indices = graph.constant(
            indicesData,
            shape: [1],
            dataType: .int32
        )
        let oneHot = graph.oneHot(
            withIndicesTensor: indices,
            depth: flatSize,
            axis: 1,
            dataType: dtype,
            onValue: 1.0,
            offValue: 0.0,
            name: "test_onehot"
        )
        let ce = graph.softMaxCrossEntropy(
            logitsFlat,
            labels: oneHot,
            axis: 1,
            reuctionType: .none,
            name: "test_ce"
        )

        let results = graph.run(
            with: cmdQueue,
            feeds: [:],
            targetTensors: [ce],
            targetOperations: nil
        )
        guard let ceOut = results[ce] else {
            XCTFail("graph.run did not return CE output")
            return
        }
        let ceValues = readFloats(ceOut, count: 1)

        // Correct-alignment expected loss:
        //   softmax(logits)[idx] = exp(L) / (exp(L) + (D-1))
        //   CE = -log softmax[idx] = -L + log(exp(L) + (D-1))
        let D = Float(flatSize)
        let correctLoss = -L + log(exp(L) + (D - 1.0))

        // Wrong-alignment expected loss (one-hot target lands at a
        // cell whose logit is 0 instead of L):
        //   softmax[wrongIdx] = 1 / (exp(L) + (D-1))
        //   CE = log(exp(L) + (D-1))
        // i.e. correctLoss + L. At L = 5 this is ~5 nats larger,
        // far outside any rounding tolerance.
        let wrongLoss = log(exp(L) + (D - 1.0))

        XCTAssertEqual(
            ceValues[0], correctLoss, accuracy: 1e-3,
            "End-to-end policy encoding / reshape / oneHot / softMaxCrossEntropy "
            + "produced CE=\(ceValues[0]); expected \(correctLoss) under "
            + "NCHW row-major flatten with one-hot at c*64+r*8+col. "
            + "A mis-aligned encoding would yield \(wrongLoss)."
        )
    }

    // MARK: - Helpers

    private func makeFloatData(_ values: [Float]) -> Data {
        var v = values
        return v.withUnsafeMutableBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size)
        }
    }

    private func makeInt32Data(_ values: [Int32]) -> Data {
        var v = values
        return v.withUnsafeMutableBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Int32>.size)
        }
    }

    private func readFloats(_ tensorData: MPSGraphTensorData, count: Int) -> [Float] {
        let nda = tensorData.mpsndarray()
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { buf in
            nda.readBytes(buf.baseAddress!, strideBytes: nil)
        }
        return out
    }
}
