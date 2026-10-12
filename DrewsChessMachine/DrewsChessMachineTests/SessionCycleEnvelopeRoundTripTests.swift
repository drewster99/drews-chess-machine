import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

/// `session.json` must reproduce the LR/momentum schedule a save wrote,
/// decay included, and must still read a session written before LR
/// keyframes.
///
/// A save writes one field, `lrSchedule` (`LRMomentumCycle`'s persisted form:
/// cycle, keyframes and momentum following); decoding it has to give back the
/// same schedule, or every save made under a decaying schedule fails
/// `CheckpointManager.saveSession`'s round-trip check.
///
/// A session written before LR keyframes carries the previous form instead:
/// the cycle as `lrMomentumCycle` and its decay envelope as the separate
/// `lrMomentumCycleEnvelope` (absent in a session written before the
/// envelope existed). `decode` converts the pair into `lrSchedule` exactly
/// (LR_KEYFRAMES_PLAN.md, owner decision Q-4, "Old files") and clears the
/// legacy fields, so nothing writes them again; a file carrying both forms,
/// or an envelope without its cycle, is refused.
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

    private let decayingEnvelope = LegacyLRMomentumCycleEnvelopeReference(
        lrPeakEnd: 1.0e-4,
        lrTroughEnd: 1.0e-6,
        decayHorizonSteps: 500_000,
        momentumFollowsLRCycle: true,
        momentumFollowStartLow: 0.85,
        momentumFollowStartHigh: 0.95,
        momentumFollowEndLow: 0.9,
        momentumFollowEndHigh: 0.98
    )

    /// The previous all-off cycle with the LR channel on — the cycle these
    /// tests always used.
    private var lrOnlyCycle: LegacyLRMomentumCycleReference {
        var cycle = LegacyLRMomentumCycleReference.disabledReference
        cycle.lrEnabled = true
        return cycle
    }

    // MARK: - The previous form, as a session written before LR keyframes

    /// `lrMomentumCycle` exactly as the previous build wrote it: the
    /// synthesized camelCase `Codable` of the old struct.
    private func legacyCycleJSON(_ cycle: LegacyLRMomentumCycleReference) -> [String: Any] {
        [
            "lrEnabled": cycle.lrEnabled,
            "lrPeriodSteps": cycle.lrPeriodSteps,
            "lrCount": cycle.lrCount,
            "lrMin": cycle.lrMin,
            "lrMax": cycle.lrMax,
            "lrInvert": cycle.lrInvert,
            "momentumEnabled": cycle.momentumEnabled,
            "momentumPeriodSteps": cycle.momentumPeriodSteps,
            "momentumCount": cycle.momentumCount,
            "momentumMin": cycle.momentumMin,
            "momentumMax": cycle.momentumMax,
            "momentumInvert": cycle.momentumInvert,
        ]
    }

    /// `lrMomentumCycleEnvelope` exactly as the previous build wrote it.
    private func legacyEnvelopeJSON(_ envelope: LegacyLRMomentumCycleEnvelopeReference) -> [String: Any] {
        [
            "lrPeakEnd": envelope.lrPeakEnd,
            "lrTroughEnd": envelope.lrTroughEnd,
            "decayHorizonSteps": envelope.decayHorizonSteps,
            "momentumFollowsLRCycle": envelope.momentumFollowsLRCycle,
            "momentumFollowStartLow": envelope.momentumFollowStartLow,
            "momentumFollowStartHigh": envelope.momentumFollowStartHigh,
            "momentumFollowEndLow": envelope.momentumFollowEndLow,
            "momentumFollowEndHigh": envelope.momentumFollowEndHigh,
        ]
    }

    /// A session.json: the base state (and `lrSchedule`, when the state has
    /// one) with `fields` added at the top level.
    private func sessionJSON(_ state: SessionCheckpointState, adding fields: [String: Any]) throws -> Data {
        let object = try JSONSerialization.jsonObject(with: state.encode())
        guard var top = object as? [String: Any] else {
            XCTFail("session.json did not encode as a JSON object")
            throw CocoaError(.coderReadCorrupt)
        }
        for (key, value) in fields {
            top[key] = value
        }
        return try JSONSerialization.data(withJSONObject: top)
    }

    /// The top-level keys of an encoded session.
    private func topLevelKeys(_ data: Data) throws -> Set<String> {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let top = object as? [String: Any] else {
            XCTFail("session.json did not encode as a JSON object")
            throw CocoaError(.coderReadCorrupt)
        }
        return Set(top.keys)
    }

    private func assertLegacyRefusal(_ data: Data, detailContains expected: String,
                                     file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try SessionCheckpointState.decode(data), file: file, line: line) { error in
            guard case SessionCheckpointError.legacyLRSchedule(let detail) = error else {
                XCTFail("expected SessionCheckpointError.legacyLRSchedule, got \(error)", file: file, line: line)
                return
            }
            XCTAssertTrue(detail.contains(expected), "detail: \(detail)", file: file, line: line)
        }
    }

    // MARK: - Round trip

    /// Was `testACycleWithADecayingEnvelopeRoundTrips` (cycle + separate
    /// envelope field). LR keyframes replaced the envelope with the keyframes
    /// inside `lrSchedule`, so this now round-trips the same decaying
    /// schedule in its keyframe form.
    func testAScheduleWithADecayRoundTrips() throws {
        var state = try makeState()
        var reference = lrOnlyCycle
        reference.envelope = decayingEnvelope
        let schedule = try LRMomentumCycle.converted(from: reference)
        XCTAssertEqual(schedule.lrKeyframes.keyframes.count, 2, "a decay is two keyframes at warmup 0")
        state.lrSchedule = schedule

        let encoded = try state.encode()
        let decoded = try SessionCheckpointState.decode(encoded)

        XCTAssertEqual(decoded.lrSchedule, schedule)
        XCTAssertEqual(decoded, state)
        let keys = try topLevelKeys(encoded)
        XCTAssertTrue(keys.contains("lrSchedule"))
        XCTAssertFalse(keys.contains("lrMomentumCycle"), "nothing writes the previous form")
        XCTAssertFalse(keys.contains("lrMomentumCycleEnvelope"), "nothing writes the previous form")
    }

    // MARK: - Sessions written before LR keyframes

    /// Was `testACycleWithoutASavedEnvelopeDecodesWithoutDecay`, which built
    /// the cycle-without-envelope state in memory. The in-memory form no
    /// longer exists, so the same session is now read from the JSON the
    /// previous build wrote: a cycle and no envelope converts to the no-decay,
    /// no-follow schedule it actually ran under, and the legacy fields are
    /// cleared.
    func testACycleWithoutASavedEnvelopeDecodesWithoutDecay() throws {
        let base = try makeState()
        let data = try sessionJSON(base, adding: [
            "lrWarmupSteps": 0,
            "lrMomentumCycle": legacyCycleJSON(lrOnlyCycle),
        ])

        let decoded = try SessionCheckpointState.decode(data)

        let schedule = try XCTUnwrap(decoded.lrSchedule)
        XCTAssertEqual(schedule, try LRMomentumCycle.converted(from: lrOnlyCycle))
        XCTAssertEqual(schedule.lrKeyframes, LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.03, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        ]), "no decay: one keyframe at the cycle's own bounds")
        XCTAssertFalse(schedule.momentumFollowsLRCycle, "a session before the envelope never followed")
        XCTAssertTrue(schedule.lrEnabled)
        XCTAssertNil(decoded.lrMomentumCycle)
        XCTAssertNil(decoded.lrMomentumCycleEnvelope)
    }

    /// The previous form with a decaying envelope and a warmup decodes to the
    /// exactly converted schedule: the start values held through warmup, the
    /// end values at warmup + horizon (the old decay counted from the end of
    /// warmup). The re-encoded session carries only the new form.
    func testAPreviousFormSessionWithAnEnvelopeDecodesToTheConvertedSchedule() throws {
        let base = try makeState()
        let warmup = 100
        let data = try sessionJSON(base, adding: [
            "lrWarmupSteps": warmup,
            "lrMomentumCycle": legacyCycleJSON(lrOnlyCycle),
            "lrMomentumCycleEnvelope": legacyEnvelopeJSON(decayingEnvelope),
        ])

        let decoded = try SessionCheckpointState.decode(data)

        var reference = lrOnlyCycle
        reference.envelope = decayingEnvelope
        let schedule = try XCTUnwrap(decoded.lrSchedule)
        XCTAssertEqual(schedule, try LRMomentumCycle.converted(from: reference, lrWarmupSteps: warmup))
        XCTAssertEqual(schedule.lrKeyframes, LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.03, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 100, peak: 0.03, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 500_100, peak: 1.0e-4, trough: 1.0e-6, momentumAtPeak: 0.9, momentumAtTrough: 0.98),
        ]))
        XCTAssertTrue(schedule.momentumFollowsLRCycle)
        XCTAssertNil(decoded.lrMomentumCycle)
        XCTAssertNil(decoded.lrMomentumCycleEnvelope)

        let reencoded = try decoded.encode()
        let keys = try topLevelKeys(reencoded)
        XCTAssertFalse(keys.contains("lrMomentumCycle"))
        XCTAssertFalse(keys.contains("lrMomentumCycleEnvelope"))
        XCTAssertEqual(try SessionCheckpointState.decode(reencoded), decoded)
    }

    /// A file carrying both forms has two sources for one schedule; nothing
    /// writes one, so it is refused rather than one silently winning.
    func testASessionCarryingBothFormsIsRefused() throws {
        var base = try makeState()
        base.lrSchedule = try LRMomentumCycle.converted(from: lrOnlyCycle)
        let data = try sessionJSON(base, adding: [
            "lrWarmupSteps": 0,
            "lrMomentumCycle": legacyCycleJSON(lrOnlyCycle),
        ])

        assertLegacyRefusal(data, detailContains: "both lrSchedule and the previous form")
    }

    /// An envelope without its cycle cannot be converted.
    func testAnEnvelopeWithoutItsCycleIsRefused() throws {
        let base = try makeState()
        let data = try sessionJSON(base, adding: [
            "lrWarmupSteps": 0,
            "lrMomentumCycleEnvelope": legacyEnvelopeJSON(decayingEnvelope),
        ])

        assertLegacyRefusal(data, detailContains: "a decay envelope without its cycle")
    }

    /// Follow bounds inverted on only part of the run fed static momentum on
    /// those steps only; no keyframe schedule expresses that, so the session
    /// is refused rather than approximated (owner decision Q-4).
    func testAPartlyInvertedFollowScheduleIsRefused() throws {
        let base = try makeState()
        var cycle = lrOnlyCycle
        cycle.momentumEnabled = true
        var envelope = decayingEnvelope
        envelope.momentumFollowStartLow = 0.95
        envelope.momentumFollowStartHigh = 0.85
        let data = try sessionJSON(base, adding: [
            "lrWarmupSteps": 0,
            "lrMomentumCycle": legacyCycleJSON(cycle),
            "lrMomentumCycleEnvelope": legacyEnvelopeJSON(envelope),
        ])

        assertLegacyRefusal(data, detailContains: "inverted on only part of the run")
    }
}
