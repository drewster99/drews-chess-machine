import Accelerate
import SwiftUI

/// `SessionController`'s engine-diagnostics battery — split out of
/// `SessionController.swift` to keep that file navigable. On-demand correctness
/// probes (PolicyEncoding round-trips, repetition detection, encoder shapes,
/// network forward-pass shapes) and the policy-conditioning probe; results go
/// to the session log with `[DIAG]` prefix.
extension SessionController {

    // MARK: - Engine Diagnostics

    /// Run a one-shot battery of correctness probes and log results
    /// with `[DIAG]` prefix. Designed to be triggered on demand after
    /// significant code changes (architecture refactors, encoder
    /// changes) so the user can confirm the engine still passes basic
    /// invariants without waiting for a full training session to
    /// surface any regression.
    ///
    /// Probes:
    ///   1. PolicyEncoding round-trip across all legal moves at the
    ///      starting position.
    ///   2. PolicyEncoding round-trip in a position with promotions.
    ///   3. PolicyEncoding distinct-index check (no two legal moves
    ///      share an index).
    ///   4. ChessGameEngine 3-fold detection on knight shuffle.
    ///   5. BoardEncoder produces correct tensor length.
    ///   6. Network forward pass shape check (if a network exists).
    ///
    /// Designed to complete in well under a second so the user sees
    /// immediate pass/fail feedback. Results go to the session log,
    /// not to a dialog — the log is the canonical record.
    func runEngineDiagnostics() {
        SessionLogger.shared.log("[BUTTON] Engine Diagnostics")
        // Wrap in a Task so we can `await` the network's async
        // evaluate cleanly. Pure-logic probes run synchronously
        // inside; only the network probe needs the await. Failures
        // are reported via the [DIAG] log lines, not via UI alerts.
        let networkRef = network
        Task {
            await self.runEngineDiagnosticsAsync(net: networkRef)
        }
    }

    private func runEngineDiagnosticsAsync(net: ChessMPSNetwork?) async {
        SessionLogger.shared.log("[DIAG] === Engine diagnostics begin ===")
        var failed = 0
        var ran = 0

        func check(_ name: String, _ predicate: () throws -> Bool) {
            ran += 1
            do {
                if try predicate() {
                    SessionLogger.shared.log("[DIAG] PASS  \(name)")
                } else {
                    SessionLogger.shared.log("[DIAG] FAIL  \(name)")
                    failed += 1
                }
            } catch {
                SessionLogger.shared.log("[DIAG] FAIL  \(name): \(error.localizedDescription)")
                failed += 1
            }
        }

        // 1. PolicyEncoding round-trip on starting position.
        check("PolicyEncoding round-trip at starting position") {
            let state = GameState.starting
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state),
                      decoded == move else { return false }
            }
            return !legals.isEmpty
        }

        // 2. Round-trip with promotions on the board.
        check("PolicyEncoding round-trip with promotions") {
            var board: [Piece?] = Array(repeating: nil, count: 64)
            board[7 * 8 + 0] = Piece(type: .king, color: .white)
            board[0 * 8 + 7] = Piece(type: .king, color: .black)
            for col in 1..<7 { board[1 * 8 + col] = Piece(type: .pawn, color: .white) }
            let state = GameState(
                board: board, currentPlayer: .white,
                whiteKingsideCastle: false, whiteQueensideCastle: false,
                blackKingsideCastle: false, blackQueensideCastle: false,
                enPassantSquare: nil, halfmoveClock: 0
            )
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state),
                      decoded == move else { return false }
            }
            // Verify all 4 promotion variants are distinct
            let promos = legals.filter { $0.promotion != nil && $0.fromCol == 1 && $0.toCol == 1 }
            let promoIndices = Set(promos.map {
                PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer)
            })
            return promos.count == 4 && promoIndices.count == 4
        }

        // 3. Distinct policy indices for all legal moves.
        check("PolicyEncoding produces distinct indices for legal moves") {
            let legals = MoveGenerator.legalMoves(for: .starting)
            let indices = legals.map { PolicyEncoding.policyIndex($0, currentPlayer: .white) }
            return Set(indices).count == indices.count
        }

        // 4. 3-fold detection via knight shuffle.
        check("ChessGameEngine detects 3-fold via knight shuffle") {
            let engine = ChessGameEngine()
            let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
            let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
            let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
            let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)
            for _ in 0..<2 {
                _ = try engine.applyMoveAndAdvance(nf3)
                _ = try engine.applyMoveAndAdvance(nc6)
                _ = try engine.applyMoveAndAdvance(ng1)
                _ = try engine.applyMoveAndAdvance(nb8)
            }
            if case .drawByThreefoldRepetition = engine.result { return true }
            return false
        }

        // 5. BoardEncoder shape check.
        check("BoardEncoder produces tensorLength floats (= \(BoardEncoder.tensorLength))") {
            let tensor = BoardEncoder.encode(.starting)
            return tensor.count == BoardEncoder.tensorLength
        }

        // 6. Network forward-pass shape (only if a network is built).
        if let net = net {
            ran += 1
            do {
                let board = BoardEncoder.encode(.starting)
                nonisolated(unsafe) var policy: [Float] = []
                try await net.evaluate(board: board) { policyBuf, _ in
                    policy = Array(policyBuf)
                }
                if policy.count == ChessNetwork.policySize {
                    SessionLogger.shared.log(
                        "[DIAG] PASS  Network forward-pass produces \(ChessNetwork.policySize) logits"
                    )
                } else {
                    SessionLogger.shared.log(
                        "[DIAG] FAIL  Network forward-pass: expected \(ChessNetwork.policySize) logits, got \(policy.count)"
                    )
                    failed += 1
                }
            } catch {
                SessionLogger.shared.log(
                    "[DIAG] FAIL  Network forward-pass error: \(error.localizedDescription)"
                )
                failed += 1
            }
        } else {
            SessionLogger.shared.log("[DIAG] SKIP  Network forward-pass shape (no network built yet)")
        }

        SessionLogger.shared.log("[DIAG] === Engine diagnostics done: \(ran - failed)/\(ran) passed ===")
    }

    /// Run a one-shot "is the policy head producing position-conditional
    /// output?" probe. Two very different positions go through the live
    /// champion network in inference mode; the policy outputs are
    /// compared for L1 distance, max single-cell |Δ|, and value-head Δ.
    /// If the policy outputs are essentially identical (avg per-cell
    /// |Δ| < 1e-4) the policy head has collapsed to a position-agnostic
    /// constant — that's the symptom we've been chasing in the masked
    /// CE / entropy debugging. Healthy networks emit meaningfully
    /// different policies for unrelated boards.
    func runPolicyConditioningDiagnostic() {
        SessionLogger.shared.log("[BUTTON] Policy Conditioning Probe")
        let networkRef = network
        Task {
            await self.runPolicyConditioningDiagnosticAsync(net: networkRef)
        }
    }

    private func runPolicyConditioningDiagnosticAsync(net: ChessMPSNetwork?) async {
        SessionLogger.shared.log("[DIAG] === Policy-conditioning probe begin ===")
        guard let net else {
            SessionLogger.shared.log("[DIAG] SKIP  No network built yet")
            SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
            return
        }

        // Position 1: white-to-move starting position. Plain, common,
        // policy is well-defined.
        let pos1 = GameState.starting

        // Position 2: a midgame-ish black-to-move position with very
        // different piece layout — different side to move, different
        // material, different square occupancies — so every input plane
        // looks different from pos1.
        var midboard: [Piece?] = Array(repeating: nil, count: 64)
        midboard[0 * 8 + 4] = Piece(type: .king, color: .black)
        midboard[7 * 8 + 4] = Piece(type: .king, color: .white)
        midboard[3 * 8 + 3] = Piece(type: .queen, color: .black)
        midboard[4 * 8 + 5] = Piece(type: .knight, color: .white)
        midboard[2 * 8 + 6] = Piece(type: .rook, color: .white)
        midboard[5 * 8 + 1] = Piece(type: .bishop, color: .black)
        let pos2 = GameState(
            board: midboard, currentPlayer: .black,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        )

        do {
            let board1 = BoardEncoder.encode(pos1)
            let board2 = BoardEncoder.encode(pos2)
            nonisolated(unsafe) var policy1: [Float] = []
            nonisolated(unsafe) var value1: Float = 0
            try await net.evaluate(board: board1) { policyBuf, v in
                policy1 = Array(policyBuf)
                value1 = v
            }
            nonisolated(unsafe) var policy2: [Float] = []
            nonisolated(unsafe) var value2: Float = 0
            try await net.evaluate(board: board2) { policyBuf, v in
                policy2 = Array(policyBuf)
                value2 = v
            }

            guard policy1.count == policy2.count else {
                SessionLogger.shared.log(
                    "[DIAG] FAIL  policy length mismatch: \(policy1.count) vs \(policy2.count)"
                )
                SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
                return
            }

            // Per-position summary stats — mean via vDSP_meanv,
            // variance via vDSP_measqv (mean of squares) after
            // subtracting mean with vDSP_vsadd.
            var mean1: Float = 0
            var mean2: Float = 0
            var meanSq1: Float = 0
            var meanSq2: Float = 0
            policy1.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                let n = vDSP_Length(buf.count)
                vDSP_meanv(base, 1, &mean1, n)
                var negMean = -mean1
                // Use a separate temporary so we don't mutate policy1.
                var centered = [Float](repeating: 0, count: buf.count)
                centered.withUnsafeMutableBufferPointer { cBuf in
                    guard let cBase = cBuf.baseAddress else { return }
                    vDSP_vsadd(base, 1, &negMean, cBase, 1, n)
                    vDSP_measqv(cBase, 1, &meanSq1, n)
                }
            }
            policy2.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                let n = vDSP_Length(buf.count)
                vDSP_meanv(base, 1, &mean2, n)
                var negMean = -mean2
                var centered = [Float](repeating: 0, count: buf.count)
                centered.withUnsafeMutableBufferPointer { cBuf in
                    guard let cBase = cBuf.baseAddress else { return }
                    vDSP_vsadd(base, 1, &negMean, cBase, 1, n)
                    vDSP_measqv(cBase, 1, &meanSq2, n)
                }
            }
            let std1 = meanSq1.squareRoot()
            let std2 = meanSq2.squareRoot()

            // L1 distance + max single-cell |Δ| + per-cell average.
            var l1: Double = 0
            var maxAbsDiff: Double = 0
            var maxAbsIdx: Int = 0
            for i in 0..<policy1.count {
                let d = Double(abs(policy1[i] - policy2[i]))
                l1 += d
                if d > maxAbsDiff {
                    maxAbsDiff = d
                    maxAbsIdx = i
                }
            }
            let avgPerCellDiff = l1 / Double(policy1.count)

            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   pos1: mean=%+0.4f std=%.4f, pos2: mean=%+0.4f std=%.4f",
                    mean1, std1, mean2, std2
                )
            )
            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   policy Δ: L1=%.3f, maxAbs=%.4f at idx=%d, avg per-cell |Δ|=%.6f",
                    l1, maxAbsDiff, maxAbsIdx, avgPerCellDiff
                )
            )
            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   value Δ: pos1=%+0.4f, pos2=%+0.4f, Δ=%+0.4f",
                    value1, value2, value1 - value2
                )
            )

            // Pass criterion: avg per-cell |Δ| above the noise floor
            // (1e-4 is generous — a randomly-initialized network with
            // logit std ~2.5 should easily produce avg |Δ| ≈ 1.0+ on
            // unrelated inputs). Below 1e-4 means the policy head's
            // output is effectively independent of the input — exactly
            // the failure mode we've been hypothesizing.
            let policyConditional = avgPerCellDiff >= 1e-4
            if policyConditional {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] PASS  Policy head is position-conditional (avg per-cell |Δ| = %.6f, threshold ≥ 1e-4)",
                        avgPerCellDiff
                    )
                )
            } else {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] FAIL  Policy head appears position-AGNOSTIC (avg per-cell |Δ| = %.6f, < 1e-4 threshold) — outputs are effectively the same regardless of input",
                        avgPerCellDiff
                    )
                )
            }

            // Also probe the value head: a healthy value head should
            // give meaningfully different scalars for two unrelated
            // positions. A pinned-to-zero value head would give |Δ|≈0.
            let valueDelta = abs(Double(value1 - value2))
            if valueDelta < 1e-4 {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] WARN  Value head also looks position-agnostic (|Δ|=%.6f)",
                        valueDelta
                    )
                )
            }
        } catch {
            SessionLogger.shared.log(
                "[DIAG] FAIL  Policy probe error: \(error.localizedDescription)"
            )
        }

        SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
    }

}
