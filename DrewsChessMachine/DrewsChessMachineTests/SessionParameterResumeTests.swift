//
//  SessionParameterResumeTests.swift
//  DrewsChessMachineTests
//
//  Pins resume parameter resolution: one resolver decides, per key, what a
//  resumed session runs with when its checkpoint carries — or lacks — a value
//  for that key, and every decision is logged as one `[RESUME-DIFF]` line.
//
//  The incident these guard against (Exp 6): a session saved before channel
//  dropout existed was resumed while the live setting was 0.7, and the run
//  silently trained with dropout it had never used. The pre-feature value is
//  now declared on the key (`absentValue`) and applied for that run only —
//  never written into the user's saved settings, which still hold 0.7 for
//  the next fresh run.
//
//  Every test snapshots `TrainingParameters.shared` and restores it
//  afterwards. The one test that lets persistence run restores the user's
//  saved dropout setting exactly, even if the code under test misbehaves.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class SessionParameterResumeTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]
    private var logLines: [String] = []

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
        logLines = []
    }

    override func tearDown() async throws {
        TrainingParameters.suppressPersistence = true
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private func makeResume() -> SessionParameterResume {
        SessionParameterResume(parameters: TrainingParameters.shared, log: { [weak self] line in
            self?.logLines.append(line)
        })
    }

    private func line(for id: String) -> String? {
        logLines.first { $0.hasPrefix("[RESUME-DIFF] \(id):") }
    }

    /// Exp 6: a legacy session without a dropout key, resumed while the live
    /// rate is 0.7, runs at the pre-feature rate 0 — and the user's saved
    /// setting is not rewritten.
    func testLegacySessionWithoutDropoutResumesAtPreFeatureRateWithoutTouchingSavedSettings() {
        let p = TrainingParameters.shared
        p.dropoutRate = 0.7
        let defaults = UserDefaults.standard
        let storedBefore = defaults.object(forKey: DropoutRate.id)
        TrainingParameters.suppressPersistence = false
        defer {
            TrainingParameters.suppressPersistence = true
            if let storedBefore {
                defaults.set(storedBefore, forKey: DropoutRate.id)
            } else {
                defaults.removeObject(forKey: DropoutRate.id)
            }
        }

        let applied = makeResume().restore(DropoutRate.self, savedFloat: nil, into: \.dropoutRate)

        XCTAssertEqual(applied, 0.0)
        XCTAssertEqual(p.dropoutRate, 0.0)
        let storedAfter = defaults.object(forKey: DropoutRate.id)
        XCTAssertEqual(storedAfter as? NSObject, storedBefore as? NSObject,
                       "a pre-feature value applies to the resumed run only; the saved setting must not change")
        let diff = line(for: DropoutRate.id)
        XCTAssertNotNil(diff)
        XCTAssertTrue(diff?.contains("saved=absent applied=0.0 current=0.7") ?? false, diff ?? "no line")
    }

    func testSavedValueIsRestored() {
        let p = TrainingParameters.shared
        p.weightDecay = 0.0003
        let resume = makeResume()

        let applied = resume.restore(WeightDecay.self, savedFloat: Float(0.0001), into: \.weightDecay)

        XCTAssertEqual(applied, 0.0001)
        XCTAssertEqual(p.weightDecay, 0.0001)
        XCTAssertTrue(resume.notExactParameterIDs.isEmpty)
        XCTAssertTrue(line(for: WeightDecay.id)?.contains("saved=0.0001 applied=0.0001 current=0.0003 (from session)") ?? false,
                      line(for: WeightDecay.id) ?? "no line")
    }

    func testOperationalKnobAbsentFromSessionKeepsTheLiveSetting() {
        let p = TrainingParameters.shared
        p.arenaAutoIntervalSec = 1200
        let resume = makeResume()

        let applied = resume.restore(ArenaAutoIntervalSec.self, saved: nil, into: \.arenaAutoIntervalSec)

        XCTAssertEqual(applied, 1200)
        XCTAssertEqual(p.arenaAutoIntervalSec, 1200)
        XCTAssertTrue(resume.notExactParameterIDs.isEmpty)
        XCTAssertTrue(line(for: ArenaAutoIntervalSec.id)?.contains("saved=absent applied=1200.0 current=1200.0") ?? false,
                      line(for: ArenaAutoIntervalSec.id) ?? "no line")
    }

    func testTrainingMathKeyWithNoKnownValueIsReportedNotExact() {
        let p = TrainingParameters.shared
        p.entropyBonus = 0.001
        let resume = makeResume()

        let applied = resume.restore(EntropyBonus.self, savedFloat: nil, into: \.entropyBonus)

        XCTAssertEqual(applied, 0.001)
        XCTAssertEqual(resume.notExactParameterIDs, [EntropyBonus.id])
        XCTAssertTrue(line(for: EntropyBonus.id)?.contains("NOT EXACT") ?? false, line(for: EntropyBonus.id) ?? "no line")
    }

    func testUnboundedPreFeatureBehaviorResolvesToTheDeclaredMaximum() {
        let p = TrainingParameters.shared
        p.maxPliesFromAnyOneGame = 10
        guard let maximum = MaxPliesFromAnyOneGame.definition.intRange?.max else {
            return XCTFail("max_plies_from_any_one_game is declared without a range")
        }

        let applied = makeResume().restore(MaxPliesFromAnyOneGame.self, saved: nil, into: \.maxPliesFromAnyOneGame)

        XCTAssertEqual(applied, maximum)
        XCTAssertEqual(p.maxPliesFromAnyOneGame, maximum)
    }

    /// Every pre-feature value a key declares must be one the key can hold;
    /// a declaration outside the range would be rejected at resume time.
    func testEveryDeclaredAbsentValueLiesInsideItsDeclaredRange() {
        for key in TrainingParameters.allKeys {
            assertAbsentValueIsValid(key)
        }
    }

    private func assertAbsentValueIsValid<K: TrainingParameterKey>(_ key: K.Type) {
        let resolved = TrainingParameterResolution.resolve(K.self, saved: nil, current: K.declaredDefault)
        guard resolved.source == .preFeature else { return }
        XCTAssertTrue(K.isWithinDeclaration(resolved.applied), "\(K.id): absent value \(resolved.applied) is outside its declared range")
    }

    /// The pre-feature resolvers the session file still exposes agree with
    /// the keys' declarations (one source of truth).
    func testSessionFileResolversAgreeWithTheDeclarations() {
        XCTAssertEqual(SessionCheckpointState.resolvedDropoutRate(saved: nil), 0)
        XCTAssertEqual(SessionCheckpointState.resolvedMomentumCoeff(saved: nil), 0)
        XCTAssertEqual(SessionCheckpointState.resolvedIllegalMassPenaltyWeight(saved: nil), 0)
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingEpsilon(saved: nil), 0)
        XCTAssertEqual(SessionCheckpointState.resolvedValueLabelSmoothingEpsilon(saved: nil), 0)
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingMode(saved: nil), .fixedTotal)
        XCTAssertFalse(SessionCheckpointState.resolvedSignedAdvantageComplementCE(savedFlag: nil))
        XCTAssertEqual(SessionCheckpointState.resolvedMaxPliesFromAnyOneGame(saved: nil),
                       MaxPliesFromAnyOneGame.definition.intRange?.max)
    }
}
