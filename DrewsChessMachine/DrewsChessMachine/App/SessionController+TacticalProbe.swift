import Foundation

// MARK: - Tactical Probe — Types

/// Coarse buckets for the starter probe set. Used only to label log
/// lines so a grep can pull out, e.g., "how do mate-in-1 verdicts
/// look across the last N probes?"
enum ProbeCategory: String, Sendable {
    case mateInOne
    case hangingPieceCapture
    case forcedPromotion
}

/// A single hand-built tactical position with an unambiguous "right
/// move." We do not have a FEN parser — every fixture is constructed
/// cell-by-cell via the `Piece(type:color:)` + manual `GameState`
/// init path already used by `SessionController+Diagnostics.swift`.
///
/// `acceptable` is a set because forced-promotion positions could
/// reasonably accept any non-stalemating promotion, and avoid-
/// stalemate positions might have multiple safe mates. For mate-in-1
/// and free-capture fixtures the set has exactly one element.
struct TacticalProbe: Sendable {
    /// Full name including the canonical correct move, e.g.
    /// `"Free knight, Be4xc6"`. Used in `[TACTICAL]` log lines and
    /// in the monitor's popover heading where having both the
    /// description and the correct move on the same line is useful.
    let name: String
    /// Short description without the move, e.g. `"Free knight"`. Used
    /// in the monitor's PROBE column where ACTUAL / EXPECTED columns
    /// already display the moves themselves, so repeating the move
    /// in the name would be redundant noise.
    let shortDescription: String
    let category: ProbeCategory
    let state: GameState
    let acceptable: Set<ChessMove>
}

/// Bucketed result of running one probe against the network. The
/// thresholds are deliberately conservative — `correctAndConfident`
/// requires the network to put more than half its legal-masked mass on
/// the expected move(s). At current training state (`pEnt ≈ 1.7`, ~5
/// effective moves) we expect mostly `correctButFlat` results on
/// mate-in-1 positions; if even those come back `wrong`, the network
/// can't represent sharp policy and we have a capacity question.
enum ProbeVerdict: String, Sendable {
    case correctAndConfident   // expected move ranked #1 AND combined prob ≥ 0.5
    case correctButFlat        // expected move ranked #1 AND combined prob <  0.5
    case correctInTop5         // expected move ranked 2..5
    case wrong                 // expected move ranked >5 or not found in legals
    case error                 // forward pass failed / network missing
}

/// One probe's full result. The top-5 list is *legal-masked-and-
/// renormalized* — i.e. the probabilities sum to ~1.0 across just the
/// legal moves — so `prob` values read like "of the legal moves
/// available, the network put X% here." This is the right framing for
/// "passive vs decisive play" intuition; the raw 4864-cell softmax
/// would be polluted by ~1% illegal mass and harder to read.
struct ProbeResult: Sendable {
    let probe: TacticalProbe
    /// Top-5 legal moves by probability, descending. Each `prob` is
    /// from the legal-masked renormalized distribution.
    let topMoves: [TopMoveEntry]
    /// 1-indexed rank of the BEST acceptable move among legals (lowest
    /// rank wins). Nil if no acceptable move is legal in this position
    /// — that would be a fixture bug.
    let expectedRank: Int?
    /// Sum of legal-masked probabilities across all `acceptable` moves
    /// that are legal here.
    let expectedProb: Float
    /// Number of legal moves in the position.
    let legalCount: Int
    /// Shannon entropy of the legal-masked renormalized distribution,
    /// in nats. Compare to `uniformLegalEntropy` to see "how close to
    /// uniform is the network's read on this position."
    let legalEntropyNats: Float
    /// `ln(legalCount)` — the entropy a uniformly-flat policy would
    /// have over this position's legal set.
    let uniformLegalEntropy: Float
    /// Sum of raw-softmax probability assigned to illegal moves. Same
    /// metric as the `pIllM` field in `[STATS]` but computed on this
    /// single position.
    let illegalMass: Float
    /// Value head's W/D/L softmax, from a second forward pass via
    /// `ChessMPSNetwork.evaluateValueDistribution`. Cost ~1-2ms.
    let valueWDL: (win: Float, draw: Float, loss: Float)
    let verdict: ProbeVerdict

    struct TopMoveEntry: Sendable {
        let move: ChessMove
        let prob: Float
    }
}

// MARK: - Tactical Probe — Analysis (pure, no SessionController)

/// Given the raw 4864-cell policy softmax for `state`, mask to legal
/// moves, renormalize, and assemble a `ProbeResult`. Pure function —
/// no side effects, no logging.
///
/// `acceptableMoves` is the fixture's stated correct-answer set.
/// `wdl` is the value head's distribution from a separate forward pass.
private func buildProbeResult(
    probe: TacticalProbe,
    rawPolicy: [Float],
    wdl: (win: Float, draw: Float, loss: Float)
) -> ProbeResult {
    let state = probe.state
    let legals = MoveGenerator.legalMoves(for: state)

    // Build legal-only (move, raw-prob) pairs by indexing the raw
    // policy at each legal move's policy index. `policyIndex` uses
    // the encoder frame (rotates board for black-to-move); the raw
    // policy was produced by the same encoder frame, so this is the
    // matching read.
    var legalEntries: [(move: ChessMove, raw: Float)] = []
    legalEntries.reserveCapacity(legals.count)
    var legalSum: Float = 0
    for move in legals {
        let idx = PolicyEncoding.policyIndex(move, currentPlayer: state.currentPlayer)
        let p = rawPolicy[idx]
        legalEntries.append((move, p))
        legalSum += p
    }

    let illegalMass = max(0.0, 1.0 - legalSum)

    // Renormalize so legal probs sum to 1. Guard against the
    // pathological "all legal mass is zero" case — shouldn't happen
    // post-mask but cheap to defend.
    var normalized: [(move: ChessMove, prob: Float)] = legalEntries.map { ($0.move, 0) }
    if legalSum > 1e-12 {
        for i in normalized.indices {
            normalized[i].prob = legalEntries[i].raw / legalSum
        }
    } else if !legalEntries.isEmpty {
        // Fall back to uniform if the network put literally zero
        // mass on every legal move (extreme collapse case).
        let u = 1.0 / Float(legalEntries.count)
        for i in normalized.indices { normalized[i].prob = u }
    }

    // Sort descending by prob for ranking + top-5 extraction.
    normalized.sort { $0.prob > $1.prob }

    let top5 = normalized.prefix(5).map {
        ProbeResult.TopMoveEntry(move: $0.move, prob: $0.prob)
    }

    // Find the BEST (lowest 1-indexed) rank among acceptable moves
    // that are actually legal here. If none are legal, the fixture is
    // buggy — surface that via `expectedRank=nil` rather than crash.
    var bestRank: Int?
    var combinedProb: Float = 0
    for (i, entry) in normalized.enumerated() {
        if probe.acceptable.contains(entry.move) {
            combinedProb += entry.prob
            if bestRank == nil { bestRank = i + 1 }
        }
    }

    // Legal-masked entropy in nats. Empty-legal case shouldn't
    // happen (would mean stalemate/checkmate — the position itself
    // is terminal and shouldn't be a fixture), but guard anyway.
    var entropyNats: Float = 0
    for entry in normalized where entry.prob > 0 {
        entropyNats -= entry.prob * log(entry.prob)
    }
    let uniformEntropy = legals.isEmpty ? 0 : log(Float(legals.count))

    let verdict: ProbeVerdict
    if let r = bestRank {
        if r == 1 {
            verdict = combinedProb >= 0.5 ? .correctAndConfident : .correctButFlat
        } else if r <= 5 {
            verdict = .correctInTop5
        } else {
            verdict = .wrong
        }
    } else {
        verdict = .wrong
    }

    return ProbeResult(
        probe: probe,
        topMoves: top5,
        expectedRank: bestRank,
        expectedProb: combinedProb,
        legalCount: legals.count,
        legalEntropyNats: entropyNats,
        uniformLegalEntropy: uniformEntropy,
        illegalMass: illegalMass,
        valueWDL: wdl,
        verdict: verdict
    )
}

// MARK: - Tactical Probe — Runner (shared by one-shot + monitor)

/// Pure namespace for the per-probe runner. Wrapped in an enum (not
/// a free function) so the call site reads `TacticalProbeRunner.run(probe, against: net)`
/// — unambiguous from `SessionController.runTacticalProbe()`, the
/// no-arg instance method that drives the one-shot menu battery.
enum TacticalProbeRunner {

/// Run one probe through `net` and assemble its result. Two forward
/// passes per probe: one for the policy logits (the path
/// `MPSChessPlayer.chooseMove` uses) and one for the W/D/L value-head
/// distribution (the diagnostic-only path). Total cost ~3ms.
///
/// Pure async function — no actor isolation. The forward passes
/// serialize on `net`'s `executionQueue`, so concurrent invocations
/// from the one-shot menu path and the periodic monitor watcher
/// don't race. Errors are logged under `[TACTICAL]` and the function
/// returns a `.error`-verdict result so the caller's batch summary
/// stays well-formed.
static func run(_ probe: TacticalProbe, against net: ChessMPSNetwork) async -> ProbeResult {
    let boardTensor = BoardEncoder.encode(probe.state)

    // Forward pass #1: raw policy logits. Box them into a Sendable
    // holder so the @Sendable consume closure can write into a value
    // we read after the await — same pattern as
    // `ChessRunner.evaluate(board:state:pieces:)`.
    let logitsBox = LogitsBox()
    do {
        try await net.evaluate(board: boardTensor) { logitsBuf, _ in
            logitsBox.set(Array(logitsBuf))
        }
    } catch {
        SessionLogger.shared.log("[TACTICAL] evaluate failed for '\(probe.name)': \(error)")
        return ProbeResult(
            probe: probe,
            topMoves: [],
            expectedRank: nil,
            expectedProb: 0,
            legalCount: 0,
            legalEntropyNats: 0,
            uniformLegalEntropy: 0,
            illegalMass: 0,
            valueWDL: (0, 0, 0),
            verdict: .error
        )
    }

    // Forward pass #2: value-head W/D/L distribution.
    let wdl: (win: Float, draw: Float, loss: Float)
    do {
        wdl = try await net.evaluateValueDistribution(board: boardTensor)
    } catch {
        SessionLogger.shared.log(
            "[TACTICAL] evaluateValueDistribution failed for '\(probe.name)': \(error) — continuing with W/D/L=0/0/0"
        )
        wdl = (0, 0, 0)
    }

    let rawLogits = logitsBox.take()
    let rawPolicy = ChessRunner.softmax(rawLogits)
    return buildProbeResult(probe: probe, rawPolicy: rawPolicy, wdl: wdl)
}

}   // end enum TacticalProbeRunner

// MARK: - Tactical Probe — SessionController extension

extension SessionController {

    /// Run the tactical-probe battery against the current champion
    /// network and log results under `[TACTICAL]`. Designed as a
    /// manual one-shot click, mirroring `runEngineDiagnostics()`:
    /// short, no UI side-effects, all output goes to the session log.
    ///
    /// Runs against `network` (the champion) — that's what plays the
    /// human and what the arena scores against. The trainer-side
    /// weights mutate constantly and would give a moving-target read.
    /// For a longitudinal version of this probe (firing automatically
    /// at every arena boundary), see the planned per-arena hook in
    /// `SessionController+Arena.swift` — not implemented yet.
    func runTacticalProbe() {
        SessionLogger.shared.log("[BUTTON] Tactical Probe")
        let net = network
        let modelLabel = network?.identifier?.description ?? "<no-id>"
        Task {
            await self.runTacticalProbeAsync(net: net, modelLabel: modelLabel)
        }
    }

    private func runTacticalProbeAsync(net: ChessMPSNetwork?, modelLabel: String) async {
        guard let net else {
            SessionLogger.shared.log("[TACTICAL] no champion network — build one first")
            return
        }

        let probes = TacticalProbeData.standardSet
        SessionLogger.shared.log(
            "[TACTICAL] === begin n=\(probes.count) net=champion model=\(modelLabel) ==="
        )

        var verdictCounts: [ProbeVerdict: Int] = [:]
        for (i, probe) in probes.enumerated() {
            let result = await TacticalProbeRunner.run(probe, against: net)
            verdictCounts[result.verdict, default: 0] += 1
            logProbeResult(index: i + 1, total: probes.count, result: result)
        }

        let summary = ProbeVerdict.allOrderedForReport
            .map { v in "\(v.rawValue)=\(verdictCounts[v] ?? 0)" }
            .joined(separator: " ")
        SessionLogger.shared.log("[TACTICAL] === summary: \(summary) ===")
    }

    /// Log one probe's result as a short multi-line block. Each line
    /// is prefixed `[TACTICAL]` so a grep pulls the whole battery.
    private func logProbeResult(index: Int, total: Int, result: ProbeResult) {
        let log = SessionLogger.shared
        let p = result.probe
        let pct = String(format: "%.3f", result.expectedProb)
        let rank = result.expectedRank.map(String.init) ?? "—"
        let ent = String(format: "%.2f", result.legalEntropyNats)
        let unif = String(format: "%.2f", result.uniformLegalEntropy)
        let illM = String(format: "%.3f", result.illegalMass)
        let w = String(format: "%.2f", result.valueWDL.win)
        let d = String(format: "%.2f", result.valueWDL.draw)
        let l = String(format: "%.2f", result.valueWDL.loss)

        log.log(
            "[TACTICAL] [\(index)/\(total)] \(p.category.rawValue) \"\(p.name)\" -> \(result.verdict.rawValue)"
        )
        log.log(
            "[TACTICAL]      legals=\(result.legalCount) illegalMass=\(illM) "
            + "entropy=\(ent)/\(unif) nats W/D/L=\(w)/\(d)/\(l)"
        )
        let expectedMoveSummary: String
        if let firstAcceptable = p.acceptable.sorted(by: { $0.notation < $1.notation }).first {
            expectedMoveSummary = firstAcceptable.notation
                + (p.acceptable.count > 1 ? " (+\(p.acceptable.count - 1) more)" : "")
        } else {
            expectedMoveSummary = "—"
        }
        log.log(
            "[TACTICAL]      expect=\(expectedMoveSummary) rank=\(rank) prob=\(pct)"
        )
        if !result.topMoves.isEmpty {
            let top = result.topMoves.map { entry in
                "\(entry.move.notation)=\(String(format: "%.3f", entry.prob))"
            }.joined(separator: " ")
            log.log("[TACTICAL]      top: \(top)")
        }
    }
}

// MARK: - Sendable box for cross-await logit handoff

/// Tiny `OSAllocatedUnfairLock`-free holder for the `[Float]` logits
/// captured inside the `@Sendable` consume closure of
/// `ChessMPSNetwork.evaluate`. The closure runs synchronously on the
/// network's execution queue and the `await` doesn't return until the
/// closure has completed and the continuation has resumed, so a single
/// reader after the await sees the write — but the strict-concurrency
/// checker still needs a `Sendable` channel for the captured-var
/// assignment. `final class` + `nonisolated(unsafe)` is the smallest
/// such channel here. Same pattern as `ChessRunner.evaluate`'s
/// `nonisolated(unsafe) var logits`.
private final class LogitsBox: @unchecked Sendable {
    nonisolated(unsafe) private var value: [Float] = []
    func set(_ v: [Float]) { value = v }
    func take() -> [Float] { value }
}

// MARK: - Verdict ordering for report

private extension ProbeVerdict {
    /// Stable order for the per-battery summary line so the verdict
    /// histogram reads "best → worst" left to right.
    static let allOrderedForReport: [ProbeVerdict] = [
        .correctAndConfident,
        .correctButFlat,
        .correctInTop5,
        .wrong,
        .error
    ]
}
