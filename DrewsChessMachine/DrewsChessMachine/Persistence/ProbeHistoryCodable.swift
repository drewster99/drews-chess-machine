import Foundation

// MARK: - ProbeResult Codable mirror

/// Round-trippable, position-free mirror of one `ProbeResult`.
///
/// `ProbeResult` itself is deliberately not `Codable`: it carries a
/// full `GameState` (the board), a `Set<ChessMove>`, and a
/// `(win, draw, loss)` tuple — none of which encode cleanly, and all of
/// which are reconstructable from the probe's *name* at load time (see
/// `reconstruct(using:)`). So a session checkpoint stores only the
/// scalar measurement surface plus the probe name; the board, the
/// acceptable-move set, the short description, etc. are rebuilt from the
/// immutable fixture sets (`TacticalProbeData.standardSet`,
/// `LichessProbeData.largeSet`) on resume. Top moves round-trip as UCI
/// strings through `ChessMove.uci` / `LichessProbeData.parseUCI(_:)`.
///
/// This keeps `session.json` small (a name + a dozen scalars per row
/// instead of a 64-square board) and means the board popover still
/// works after a resume — the reconstructed `ProbeResult.probe.state`
/// is the same fixture position the live run probed.
struct ProbeResultCodable: Codable, Equatable, Sendable {
    let probeName: String
    let verdict: ProbeVerdict
    let expectedRank: Int?
    let expectedProb: Float
    let legalCount: Int
    let legalEntropyNats: Float
    let uniformLegalEntropy: Float
    let illegalMass: Float
    let valueWin: Float
    let valueDraw: Float
    let valueLoss: Float
    let topMoves: [TopMoveCodable]

    struct TopMoveCodable: Codable, Equatable, Sendable {
        let uci: String
        let prob: Float
    }

    init(_ result: ProbeResult) {
        probeName = result.probe.name
        verdict = result.verdict
        expectedRank = result.expectedRank
        expectedProb = result.expectedProb
        legalCount = result.legalCount
        legalEntropyNats = result.legalEntropyNats
        uniformLegalEntropy = result.uniformLegalEntropy
        illegalMass = result.illegalMass
        valueWin = result.valueWDL.win
        valueDraw = result.valueWDL.draw
        valueLoss = result.valueWDL.loss
        topMoves = result.topMoves.map {
            TopMoveCodable(uci: $0.move.uci, prob: $0.prob)
        }
    }

    /// Rebuild a full `ProbeResult` using a name→fixture index (build it
    /// once per restore via `ProbeFixtureIndex.build()`). Returns `nil`
    /// when the probe name no longer resolves — e.g. the bundled Lichess
    /// set was re-curated since the session was saved, or a hand-built
    /// fixture was renamed. The caller skips those rows rather than
    /// substituting a bogus position. Top moves whose stored UCI fails
    /// to parse are dropped individually (they only feed display / the
    /// popover's arrow overlay, not any correctness invariant).
    func reconstruct(using probeIndex: [String: TacticalProbe]) -> ProbeResult? {
        guard let probe = probeIndex[probeName] else { return nil }
        let moves: [ProbeResult.TopMoveEntry] = topMoves.compactMap { tm in
            guard let move = LichessProbeData.parseUCI(tm.uci) else { return nil }
            return ProbeResult.TopMoveEntry(move: move, prob: tm.prob)
        }
        return ProbeResult(
            probe: probe,
            topMoves: moves,
            expectedRank: expectedRank,
            expectedProb: expectedProb,
            legalCount: legalCount,
            legalEntropyNats: legalEntropyNats,
            uniformLegalEntropy: uniformLegalEntropy,
            illegalMass: illegalMass,
            valueWDL: (win: valueWin, draw: valueDraw, loss: valueLoss),
            verdict: verdict
        )
    }
}

/// Name→fixture lookup spanning both probe sets, used to reconstruct
/// `ProbeResult.probe` (including its `GameState`) on resume.
enum ProbeFixtureIndex {
    /// Build the combined index. Accessing `LichessProbeData.largeSet`
    /// triggers the one-shot bundle parse — fine at app runtime.
    /// Unit tests that only exercise the hand-built tactical fixtures
    /// should construct their own index from `TacticalProbeData.standardSet`
    /// to stay bundle-independent.
    static func build() -> [String: TacticalProbe] {
        var out: [String: TacticalProbe] = [:]
        for probe in TacticalProbeData.standardSet { out[probe.name] = probe }
        for probe in LichessProbeData.largeSet { out[probe.name] = probe }
        return out
    }
}

// MARK: - History snapshots

/// Serialized form of `LichessProbeHistory`, embedded in
/// `SessionCheckpointState` so the 200-puzzle monitor's charts, per-theme
/// rows, and latest detail table survive a save/resume cycle.
struct LichessProbeHistorySnapshot: Codable, Equatable, Sendable {
    /// Per-theme aggregate series, keyed by `ProbeCategory.rawValue`
    /// (rather than the enum) so the on-disk JSON is a clean string-keyed
    /// object and an unknown future theme decodes/skips gracefully.
    let perTheme: [String: [LichessProbeHistory.Entry]]
    let overall: [LichessProbeHistory.OverallTickSample]
    let latestResults: [ProbeResultCodable]
    let latestTimestamp: Date?
    let latestModelLabel: String?
    let latestTrainingStep: Int?
    let latestPositionsTrained: Int?
    let latestActiveTrainingSec: Double?
    let latestArenaCount: Int?
    let latestPromotionCount: Int?
}

/// Serialized form of `TacticalProbeHistory`, embedded in
/// `SessionCheckpointState`. Each entry keeps the full per-probe
/// `ProbeResult` (as a `ProbeResultCodable`) so the monitor's rows,
/// spark series, and board popover all restore on resume.
struct TacticalProbeHistorySnapshot: Codable, Equatable, Sendable {
    struct EntryCodable: Codable, Equatable, Sendable {
        let timestamp: Date
        let result: ProbeResultCodable
    }
    /// Per-probe-name series, keyed by `TacticalProbe.name`.
    let entries: [String: [EntryCodable]]
}
