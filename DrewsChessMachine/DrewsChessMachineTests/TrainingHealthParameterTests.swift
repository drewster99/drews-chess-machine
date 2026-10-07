import XCTest
@testable import DrewsChessMachine
import TrainingParametersMacroSupport

/// The twelve training-health parameters (alarms plan, Part K): their
/// declarations, the one rule → key mapping on each side (snapshot and
/// singleton), the resolution into `TrainingHealthConfig`, the
/// parameters-file round trip, and the session save → resume path.
///
/// Tests that touch `TrainingParameters.shared` snapshot it and restore it
/// with persistence suppressed (the `GuiSessionResumeTests` pattern), so the
/// user's saved settings are never written.
@MainActor
final class TrainingHealthParameterTests: XCTestCase {

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

    /// Every action key, by rule, written out independently of the
    /// production mappings so a crossed wire in either fails here.
    private static let actionKeyIDs: [TrainingHealthRule: String] = [
        .nonFinite: "training_health_action_non_finite",
        .deadChannels: "training_health_action_dead_channels",
        .valueFC1ZeroVelocity: "training_health_action_value_fc1_zero_velocity",
        .illegalMass: "training_health_action_illegal_mass",
        .gradientCollapse: "training_health_action_gradient_collapse",
        .lossSpike: "training_health_action_loss_spike",
        .policyOffsetDrift: "training_health_action_policy_offset_drift",
        .batchNormRunningVarianceRunaway: "training_health_action_bn_running_variance_runaway",
        .gradientSpike: "training_health_action_gradient_spike",
        .divergence: "training_health_action_divergence",
        .valueSaturation: "training_health_action_value_saturation",
        .valueDrawSaturation: "training_health_action_value_draw_saturation",
        .legalMassStall: "training_health_action_legal_mass_stall",
        .batchNormRunningVarianceJump: "training_health_action_bn_running_variance_jump",
    ]

    private func definition(_ id: String) throws -> TrainingParameterDefinition {
        try XCTUnwrap(TrainingParameters.allDefinitions.first { $0.id == id }, "no parameter \(id)")
    }

    // MARK: Declarations

    func testTheTwelveKeysAreDeclaredWithTheirDefaultsAndRanges() throws {
        let enabled = try definition("training_health_alarms_enabled")
        XCTAssertEqual(enabled.defaultValue, TrainingHealthAlarmsEnabled.encode(true))
        XCTAssertEqual(enabled.category, "Health")
        XCTAssertTrue(enabled.liveTunable)

        let interval = try definition("training_health_check_interval_steps")
        XCTAssertEqual(interval.defaultValue, TrainingHealthCheckIntervalSteps.encode(1000))
        XCTAssertEqual(interval.intRange?.min, 50)
        XCTAssertEqual(interval.intRange?.max, 100_000)
        XCTAssertTrue(interval.liveTunable)

        let grace = try definition("training_health_learning_grace_steps")
        XCTAssertEqual(grace.defaultValue, TrainingHealthLearningGraceSteps.encode(1000))
        XCTAssertEqual(grace.intRange?.min, 0)
        XCTAssertEqual(grace.intRange?.max, 100_000)

        for rule in TrainingHealthRule.allCases {
            let id = try XCTUnwrap(Self.actionKeyIDs[rule])
            let action = try definition(id)
            XCTAssertEqual(action.defaultValue, ParameterValue.int(TrainingHealthAction.log.rawValue), id)
            XCTAssertEqual(action.category, "Health", id)
            XCTAssertTrue(action.liveTunable, id)
        }
        let healthIDs = TrainingParameters.allDefinitions.filter { $0.category == "Health" }.map(\.id)
        XCTAssertEqual(healthIDs.count, 17)
    }

    func testEveryHealthKeyKeepsTheCurrentSettingWhenAbsent() {
        XCTAssertEqual(TrainingHealthAlarmsEnabled.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthCheckIntervalSteps.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthLearningGraceSteps.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionNonFinite.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionDeadChannels.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionValueFC1ZeroVelocity.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionIllegalMass.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionGradientCollapse.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionLossSpike.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionPolicyOffsetDrift.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionBatchNormRunningVarianceRunaway.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionGradientSpike.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionDivergence.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionValueSaturation.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionValueDrawSaturation.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionLegalMassStall.absentValue, .currentSetting)
        XCTAssertEqual(TrainingHealthActionBatchNormRunningVarianceJump.absentValue, .currentSetting)
    }

    /// The action keys store a raw `Int`; the declared range must be
    /// exactly the enum's cases, or validation would admit a value that
    /// traps in `TrainingHealthAction(persistedRawValue:)`.
    func testActionRangeMatchesTheEnumCases() throws {
        for rule in TrainingHealthRule.allCases {
            let id = try XCTUnwrap(Self.actionKeyIDs[rule])
            let range = try XCTUnwrap(try definition(id).intRange, id)
            XCTAssertEqual(range.min...range.max, TrainingHealthAction.parameterRawValueRange, id)
            for raw in range.min...range.max {
                XCTAssertNotNil(TrainingHealthAction(rawValue: raw), "\(id): \(raw) has no case")
            }
        }
    }

    /// Each rule maps to its own key on the snapshot side and to its own
    /// stored property on the singleton: setting one rule's action changes
    /// that rule's reading and no other.
    func testEveryRuleHasExactlyOneActionKey() throws {
        for rule in TrainingHealthRule.allCases {
            let id = try XCTUnwrap(Self.actionKeyIDs[rule])
            let snapshot = try TrainingParametersSnapshot.declaredDefaults(
                overriding: [id: .int(TrainingHealthAction.stopOnAny.rawValue)])
            for other in TrainingHealthRule.allCases {
                XCTAssertEqual(
                    snapshot.trainingHealthAction(for: other), other == rule ? .stopOnAny : .log,
                    "snapshot: setting \(id) changed \(other.rawValue)")
            }

            let p = TrainingParameters.shared
            for each in TrainingHealthRule.allCases {
                p[keyPath: TrainingParameters.trainingHealthActionKeyPath(for: each)] = .log
            }
            p[keyPath: TrainingParameters.trainingHealthActionKeyPath(for: rule)] = .stopOnCritical
            let values = p.snapshot().rawValueMap()
            for other in TrainingHealthRule.allCases {
                let otherID = try XCTUnwrap(Self.actionKeyIDs[other])
                XCTAssertEqual(
                    values[otherID], .int(other == rule ? TrainingHealthAction.stopOnCritical.rawValue : 0),
                    "singleton: setting \(rule.rawValue) wrote \(otherID)")
            }
        }
        XCTAssertEqual(Set(Self.actionKeyIDs.values).count, TrainingHealthRule.allCases.count)
    }

    // MARK: Resolution

    func testConfigResolvesFromASnapshot() throws {
        let snapshot = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            "training_health_alarms_enabled": .bool(false),
            "training_health_check_interval_steps": .int(250),
            "training_health_learning_grace_steps": .int(40),
            LRWarmupSteps.id: LRWarmupSteps.encode(300),
            MomentumCoeff.id: MomentumCoeff.encode(0.85),
            "training_health_action_illegal_mass": .int(1),
            "training_health_action_dead_channels": .int(2),
        ])
        let config = try TrainingHealthConfig(snapshot)
        XCTAssertFalse(config.enabled)
        XCTAssertEqual(config.checkIntervalSteps, 250)
        XCTAssertEqual(config.learningGraceSteps, 40)
        XCTAssertEqual(config.lrWarmupSteps, 300)
        XCTAssertEqual(config.momentumCoefficient, 0.85)
        XCTAssertEqual(config.learningGateTrainerStep, 340)
        XCTAssertEqual(config.actions[.illegalMass], .stopOnCritical)
        XCTAssertEqual(config.actions[.deadChannels], .stopOnAny)
        XCTAssertEqual(config.actions[.nonFinite], .log)
    }

    func testDefaultConfigIsEnabledAndLogsOnly() throws {
        let config = try TrainingHealthConfig(TrainingParametersSnapshot.declaredDefaults(overriding: [:]))
        XCTAssertTrue(config.enabled)
        XCTAssertEqual(config.checkIntervalSteps, 1000)
        XCTAssertEqual(config.learningGraceSteps, 1000)
        for rule in TrainingHealthRule.allCases {
            XCTAssertEqual(config.actions[rule], .log, rule.rawValue)
        }
    }

    func testOutOfRangeActionIsRefusedByValidation() {
        XCTAssertThrowsError(try TrainingParametersSnapshot.declaredDefaults(
            overriding: ["training_health_action_loss_spike": .int(3)]))
        XCTAssertThrowsError(try TrainingParametersSnapshot.declaredDefaults(
            overriding: ["training_health_check_interval_steps": .int(49)]))
    }

    // MARK: parameters.json

    func testParametersFileRoundTripsTheHealthKeys() throws {
        let json = try TrainingParameters.defaultsJSON()
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: json) as? [String: Any])
        XCTAssertEqual(object["training_health_alarms_enabled"] as? Bool, true)
        XCTAssertEqual(object["training_health_check_interval_steps"] as? Int, 1000)
        XCTAssertEqual(object["training_health_learning_grace_steps"] as? Int, 1000)
        for id in Self.actionKeyIDs.values {
            XCTAssertEqual(object[id] as? Int, 0, id)
        }

        // An edited file: actions and the interval changed, then loaded.
        var edited = object
        edited["training_health_action_gradient_collapse"] = 1
        edited["training_health_check_interval_steps"] = 500
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("health-params-\(UUID().uuidString)")
            .appendingPathExtension("json")
        let file = try FileSafety.createNewFile(at: url)
        let identity = file.identity
        addTeardownBlock {
            _ = try FileSafety.removeOwnedItem(at: url, identity: identity)
        }
        try file.handle.write(contentsOf: try JSONSerialization.data(withJSONObject: edited))
        try file.handle.close()
        let loaded = try CliTrainingConfig.load(from: url)
        let snapshot = try TrainingParametersSnapshot.declaredDefaults(overriding: loaded.trainingParameters)
        XCTAssertEqual(snapshot.trainingHealthAction(for: .gradientCollapse), .stopOnCritical)
        XCTAssertEqual(snapshot.trainingHealthCheckIntervalSteps, 500)
    }

    // MARK: Session save → resume

    private func sessionState(extraFields: String) throws -> SessionCheckpointState {
        let json = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20261006-1-HLTH", "savedAtUnix": 1790000000,
          "sessionStartUnix": 1789996400, "elapsedTrainingSec": 3600,
          "trainingSteps": 500, "selfPlayGames": 40, "selfPlayMoves": 3000,
          "trainingPositionsSeen": 16000, "batchSize": 32, "learningRate": 0.001,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.02},
          "selfPlayWorkerCount": 6,
          \(extraFields)
          "championID": "20261006-1-HLTH", "trainerID": "20261006-2-HLTH", "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(json.utf8))
    }

    private func makeResume() -> SessionParameterResume {
        SessionParameterResume(parameters: TrainingParameters.shared, log: { [weak self] line in
            self?.logLines.append(line)
        })
    }

    /// The save writes every setting (actions by name) and a resume restores
    /// each through the resolver with one `[RESUME-DIFF]` line.
    func testSessionSaveWritesAndResumeRestoresTheHealthSettings() throws {
        let actions = TrainingHealthActions { rule in
            rule == .illegalMass ? .stopOnCritical : (rule == .nonFinite ? .stopOnAny : .log)
        }
        let saved = try sessionState(extraFields: "").withTrainingHealthSettings(
            enabled: false, checkIntervalSteps: 300, learningGraceSteps: 70, actions: actions)
        XCTAssertEqual(saved.trainingHealthActionIllegalMass, "stop_on_critical")
        XCTAssertEqual(saved.trainingHealthActionNonFinite, "stop_on_any")
        XCTAssertEqual(saved.trainingHealthActionLossSpike, "log")
        let decoded = try SessionCheckpointState.decode(try saved.encode())
        XCTAssertEqual(decoded.trainingHealthCheckIntervalSteps, 300)

        let p = TrainingParameters.shared
        p.trainingHealthAlarmsEnabled = true
        p.trainingHealthCheckIntervalSteps = 1000
        p.trainingHealthLearningGraceSteps = 1000
        for rule in TrainingHealthRule.allCases {
            p[keyPath: TrainingParameters.trainingHealthActionKeyPath(for: rule)] = .log
        }
        makeResume().applyGuiSession(decoded, acceptedReplacements: [])
        XCTAssertFalse(p.trainingHealthAlarmsEnabled)
        XCTAssertEqual(p.trainingHealthCheckIntervalSteps, 300)
        XCTAssertEqual(p.trainingHealthLearningGraceSteps, 70)
        XCTAssertEqual(p.trainingHealthActionIllegalMass, .stopOnCritical)
        XCTAssertEqual(p.trainingHealthActionNonFinite, .stopOnAny)
        XCTAssertEqual(p.trainingHealthActionGradientSpike, .log)
        for id in ["training_health_alarms_enabled", "training_health_check_interval_steps",
                   "training_health_learning_grace_steps"] + Array(Self.actionKeyIDs.values) {
            XCTAssertEqual(logLines.filter { $0.hasPrefix("[RESUME-DIFF] \(id):") }.count, 1, id)
        }
    }

    /// A session written before the alarms keeps the live settings.
    func testSessionWithoutHealthSettingsKeepsTheCurrentOnes() throws {
        let p = TrainingParameters.shared
        p.trainingHealthCheckIntervalSteps = 750
        p.trainingHealthActionDeadChannels = .stopOnCritical
        makeResume().applyGuiSession(try sessionState(extraFields: ""), acceptedReplacements: [])
        XCTAssertEqual(p.trainingHealthCheckIntervalSteps, 750)
        XCTAssertEqual(p.trainingHealthActionDeadChannels, .stopOnCritical)
    }

    /// An unknown saved action name is a load-time finding; accepting it
    /// keeps the current actions and says so per key.
    func testUnknownSavedActionIsAFindingAndAcceptingItKeepsTheCurrentActions() throws {
        let state = try sessionState(extraFields: #""trainingHealthActionLossSpike": "explode","#)
        let findings = state.invalidSavedSettings(
            current: try TrainingParametersSnapshot.declaredDefaults(overriding: [:]))
        let finding = try XCTUnwrap(findings.first { $0.id == SessionCheckpointState.SavedSettingID.trainingHealthActions })
        XCTAssertTrue(finding.found.contains("loss_spike=explode"), finding.found)

        let p = TrainingParameters.shared
        p.trainingHealthActionLossSpike = .stopOnAny
        makeResume().applyGuiSession(
            state, acceptedReplacements: [SessionCheckpointState.SavedSettingID.trainingHealthActions])
        XCTAssertEqual(p.trainingHealthActionLossSpike, .stopOnAny)
        XCTAssertEqual(logLines.filter { $0.hasPrefix("[RESUME-DIFF] training_health_action_loss_spike:") }.count, 1)
    }
}
