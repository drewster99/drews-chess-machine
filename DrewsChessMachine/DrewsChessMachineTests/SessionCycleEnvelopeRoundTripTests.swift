import XCTest
@testable import DrewsChessMachine

/// `session.json` must reproduce the LR/momentum cycle a save wrote,
/// envelope included. `LRMomentumCycle` leaves its envelope out of its own
/// encoded form (so sessions written before the envelope existed still
/// decode) and the envelope travels as the separate `lrMomentumCycleEnvelope`
/// field; decoding has to put the two back together, the way the trainer's
/// own safetensors metadata does. Otherwise every save made under a decaying
/// envelope fails `CheckpointManager.saveSession`'s round-trip check.
final class SessionCycleEnvelopeRoundTripTests: XCTestCase {

    private func makeState() throws -> SessionCheckpointState {
        let jsonText = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20261003-1-EnvR",
          "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400,
          "elapsedTrainingSec": 3600,
          "trainingSteps": 12345,
          "selfPlayGames": 678,
          "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280,
          "batchSize": 1024,
          "learningRate": 5.0e-5,
          "promoteThreshold": 0.55,
          "arenaGames": 200,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.4},
          "arenaTau": {"startTau": 1.0, "decayPerPly": 0.05, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "20261003-1-EnvR",
          "trainerID": "20261003-1-EnvR-1",
          "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(jsonText.utf8))
    }

    private let decayingEnvelope = LRMomentumCycleEnvelope(
        lrPeakEnd: 1.0e-4,
        lrTroughEnd: 1.0e-6,
        decayHorizonSteps: 500_000,
        momentumFollowsLRCycle: true,
        momentumFollowStartLow: 0.85,
        momentumFollowStartHigh: 0.95,
        momentumFollowEndLow: 0.9,
        momentumFollowEndHigh: 0.98
    )

    /// The shape every save writes: the cycle carries its envelope and the
    /// envelope is also stored on its own.
    func testACycleWithADecayingEnvelopeRoundTrips() throws {
        var state = try makeState()
        var cycle = LRMomentumCycle.disabled
        cycle.lrEnabled = true
        cycle.envelope = decayingEnvelope
        state.lrMomentumCycle = cycle
        state.lrMomentumCycleEnvelope = decayingEnvelope

        let decoded = try SessionCheckpointState.decode(state.encode())

        XCTAssertEqual(decoded.lrMomentumCycle?.envelope, decayingEnvelope)
        XCTAssertEqual(decoded, state)
    }

    /// A session written before the envelope existed carries a cycle and no
    /// envelope: the cycle decodes with the no-decay, no-follow envelope it
    /// actually ran under.
    func testACycleWithoutASavedEnvelopeDecodesWithoutDecay() throws {
        var state = try makeState()
        var cycle = LRMomentumCycle.disabled
        cycle.lrEnabled = true
        state.lrMomentumCycle = cycle
        state.lrMomentumCycleEnvelope = nil

        let decoded = try SessionCheckpointState.decode(state.encode())

        XCTAssertEqual(decoded.lrMomentumCycle?.envelope, .noDecay)
        XCTAssertNil(decoded.lrMomentumCycleEnvelope)
        XCTAssertEqual(decoded, state)
    }
}
