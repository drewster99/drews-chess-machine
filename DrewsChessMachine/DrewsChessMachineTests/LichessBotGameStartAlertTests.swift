import XCTest
@testable import DrewsChessMachine

/// Game-start alerts for human games from an accepted challenge: which
/// alert each origin and opponent kind gets, the once-per-game tracker, the
/// repeat-count validation, and saved settings without the new fields.
final class LichessBotGameStartAlertTests: XCTestCase {

    private static let incoming = LichessBotGameOrigin.acceptedIncomingChallenge(challengeID: "g1", challengerID: "someone")
    private static let outgoingOrigins: [LichessBotGameOrigin] = [
        .outgoingChallengeAccepted(challengeID: "g1", sender: .challengeSheet),
        .outgoingChallengeAccepted(challengeID: "g1", sender: .casualResendOffer),
        .outgoingChallengeAccepted(challengeID: "g1", sender: .challengeQueue),
        .outgoingChallengeAccepted(challengeID: "g1", sender: .matchmaking(trigger: .automaticPass, fillMode: .everyFreeSlot)),
        .outgoingChallengeAccepted(challengeID: "g1", sender: .matchmaking(trigger: .fillOpenSlots, fillMode: .everyFreeSlot)),
        .outgoingChallengeAccepted(challengeID: "g1", sender: .matchmakingCasualResend),
        .outgoingChallengeSenderNotRecorded(challengeID: "g1"),
    ]
    private static let noChallengeOrigins: [LichessBotGameOrigin] = [
        .tournament(source: LichessBotOpenValue(.arena), tournamentID: "t1"),
        .tournament(source: LichessBotOpenValue(.swiss), tournamentID: nil),
        .undetermined(source: LichessBotOpenValue(.friend), gap: .noChallengeRecord),
        .undetermined(source: nil, gap: .challengeLogIncomplete),
    ]

    // MARK: - Which alert

    func testIncomingChallengeFromAHumanIsWeAccepted() {
        XCTAssertEqual(LichessBotGameStartAlert.alert(origin: Self.incoming, opponent: .human), .weAcceptedHumanChallenge)
    }

    func testEveryOutgoingOriginWithAHumanIsHumanAccepted() {
        for origin in Self.outgoingOrigins {
            XCTAssertEqual(LichessBotGameStartAlert.alert(origin: origin, opponent: .human), .humanAcceptedOurChallenge, "\(origin)")
        }
    }

    func testBotsAndLichessAINeverAlert() {
        for origin in [Self.incoming] + Self.outgoingOrigins + Self.noChallengeOrigins {
            XCTAssertNil(LichessBotGameStartAlert.alert(origin: origin, opponent: .bot), "\(origin)")
            XCTAssertNil(LichessBotGameStartAlert.alert(origin: origin, opponent: .lichessAI), "\(origin)")
        }
    }

    func testTournamentAndUndeterminedNeverAlert() {
        for origin in Self.noChallengeOrigins {
            XCTAssertNil(LichessBotGameStartAlert.alert(origin: origin, opponent: .human), "\(origin)")
        }
    }

    /// The opponent kind comes from the shared rule, so a titled human is
    /// human and a BOT title or an AI level is not.
    func testOpponentKindRuleFeedsTheAlert() {
        XCTAssertEqual(LichessBotGameStartAlert.alert(origin: Self.incoming, opponent: LichessBotOpponentKind.of(title: "GM", aiLevel: nil)), .weAcceptedHumanChallenge)
        XCTAssertNil(LichessBotGameStartAlert.alert(origin: Self.incoming, opponent: LichessBotOpponentKind.of(title: "BOT", aiLevel: nil)))
        XCTAssertNil(LichessBotGameStartAlert.alert(origin: Self.incoming, opponent: LichessBotOpponentKind.of(title: nil, aiLevel: 3)))
    }

    func testEachAlertReadsItsOwnSound() {
        var alerts = LichessBotAlertSettings()
        alerts.humanAcceptedOurChallengeSound = LichessBotRepeatingAlertSound(soundName: "Glass", repeatCount: 3)
        alerts.weAcceptedHumanChallengeSound = LichessBotRepeatingAlertSound(soundName: "Ping", repeatCount: 2)
        XCTAssertEqual(LichessBotGameStartAlert.humanAcceptedOurChallenge.sound(in: alerts), LichessBotRepeatingAlertSound(soundName: "Glass", repeatCount: 3))
        XCTAssertEqual(LichessBotGameStartAlert.weAcceptedHumanChallenge.sound(in: alerts), LichessBotRepeatingAlertSound(soundName: "Ping", repeatCount: 2))
    }

    func testDefaultsAreSilentAndPlayOnce() {
        let alerts = LichessBotSettings().alerts
        for alert in LichessBotGameStartAlert.allCases {
            XCTAssertNil(alert.sound(in: alerts).soundName, "\(alert)")
            XCTAssertEqual(alert.sound(in: alerts).repeatCount, 1, "\(alert)")
        }
    }

    // MARK: - Tracker

    func testOriginThenOpponentAlertsOnce() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertEqual(tracker.opponentKnown(.human, gameID: "g1"), .weAcceptedHumanChallenge)
        XCTAssertFalse(tracker.isWaiting("g1"))
        // A re-sent gameFull (stream resync) and a repeated origin never alert again.
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
    }

    func testOpponentThenOriginAlertsOnce() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
        XCTAssertEqual(tracker.originKnown(Self.outgoingOrigins[0], gameID: "g1"), .humanAcceptedOurChallenge)
        XCTAssertNil(tracker.originKnown(Self.outgoingOrigins[0], gameID: "g1"))
    }

    func testResumedGameNeverAlerts() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: false)
        XCTAssertFalse(tracker.isWaiting("g1"))
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
    }

    /// A new session that learned its opponent, then a resumed session
    /// (the bot went offline and back online), then the origin: the
    /// resumed game must not alert.
    func testAWaitingGameResumedInALaterSessionNeverAlerts() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
        tracker.sessionStarted(gameID: "g1", isNewGame: false)
        XCTAssertFalse(tracker.isWaiting("g1"))
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
    }

    func testAlertedGameRestartedInTheSameLaunchNeverAlertsAgain() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertEqual(tracker.opponentKnown(.human, gameID: "g1"), .weAcceptedHumanChallenge)
        tracker.sessionEnded(gameID: "g1")
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertFalse(tracker.isWaiting("g1"))
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
    }

    func testSessionEndingBeforeBothFactsSettlesSilently() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
        tracker.sessionEnded(gameID: "g1")
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
    }

    func testABotGameSettlesWithoutAlerting() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertNil(tracker.opponentKnown(.bot, gameID: "g1"))
        XCTAssertFalse(tracker.isWaiting("g1"))
    }

    func testFactsForAGameWithNoSessionAreIgnored() {
        var tracker = LichessBotGameStartAlertTracker()
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g1"))
        XCTAssertFalse(tracker.isWaiting("g1"))
    }

    func testGamesAreTrackedApart() {
        var tracker = LichessBotGameStartAlertTracker()
        tracker.sessionStarted(gameID: "g1", isNewGame: true)
        tracker.sessionStarted(gameID: "g2", isNewGame: true)
        XCTAssertNil(tracker.originKnown(Self.incoming, gameID: "g1"))
        XCTAssertNil(tracker.opponentKnown(.human, gameID: "g2"))
        XCTAssertEqual(tracker.originKnown(Self.outgoingOrigins[0], gameID: "g2"), .humanAcceptedOurChallenge)
        XCTAssertTrue(tracker.isWaiting("g1"))
        XCTAssertEqual(tracker.opponentKnown(.human, gameID: "g1"), .weAcceptedHumanChallenge)
    }

    // MARK: - Repeat-count validation

    func testRepeatCountRangeIsOneToTen() {
        XCTAssertEqual(LichessBotLimits.alertSoundRepeatCountRange, 1...10)
    }

    func testRepeatCountsOutsideTheRangeAreInvalidOnTheAlertsTab() {
        let baseline = LichessBotSettings.testBaseline()
        for keyPath in [\LichessBotAlertSettings.humanAcceptedOurChallengeSound, \LichessBotAlertSettings.weAcceptedHumanChallengeSound] {
            for count in [0, -1, 11] {
                var draft = baseline
                draft.alerts[keyPath: keyPath].repeatCount = count
                XCTAssertFalse(draft.validationProblems().isEmpty, "\(keyPath) \(count)")
                XCTAssertEqual(LichessBotSettingsTab.tabsWithProblems(in: draft, comparedWith: baseline), [.alerts], "\(keyPath) \(count)")
            }
            for count in [1, 10] {
                var draft = baseline
                draft.alerts[keyPath: keyPath] = LichessBotRepeatingAlertSound(soundName: "Glass", repeatCount: count)
                XCTAssertEqual(draft.validationProblems(), [], "\(keyPath) \(count)")
            }
        }
    }

    /// An invalid count is rejected even with no sound chosen: the stored
    /// value must always be one the Stepper can show.
    func testInvalidRepeatCountWithNoSoundIsStillInvalid() {
        var draft = LichessBotSettings.testBaseline()
        draft.alerts.weAcceptedHumanChallengeSound = LichessBotRepeatingAlertSound(soundName: nil, repeatCount: 0)
        XCTAssertFalse(draft.validationProblems().isEmpty)
    }

    // MARK: - Saved settings

    /// Settings saved before these fields existed load, with the two alert
    /// sounds filled from the defaults and reported.
    func testSaveWithoutTheAlertFieldsLoads() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings.testBaseline()
        settings.alerts.humanChallengeSoundName = "Blow"
        let data = try JSONEncoder().encode(settings)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var alerts = try XCTUnwrap(object["alerts"] as? [String: Any])
        alerts["humanAcceptedOurChallengeSound"] = nil
        alerts["weAcceptedHumanChallengeSound"] = nil
        object["alerts"] = alerts
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)
        let loaded = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(loaded.settings.alerts.humanChallengeSoundName, "Blow", "everything saved is kept")
        XCTAssertEqual(loaded.settings.alerts.humanAcceptedOurChallengeSound, LichessBotRepeatingAlertSound())
        XCTAssertEqual(loaded.settings.alerts.weAcceptedHumanChallengeSound, LichessBotRepeatingAlertSound())
        XCTAssertEqual(loaded.filledFromDefaults, ["alerts.humanAcceptedOurChallengeSound", "alerts.weAcceptedHumanChallengeSound"])
        XCTAssertEqual(loaded.ignoredSavedKeys, [])
    }

    /// A saved alert with no sound (the nil name omitted) keeps its repeat
    /// count, and a chosen sound round-trips.
    func testAlertSoundsRoundTrip() throws {
        let defaults = try makeTemporaryDefaults()
        var settings = LichessBotSettings.testBaseline()
        settings.alerts.humanAcceptedOurChallengeSound = LichessBotRepeatingAlertSound(soundName: "Glass", repeatCount: 4)
        settings.alerts.weAcceptedHumanChallengeSound = LichessBotRepeatingAlertSound(soundName: nil, repeatCount: 7)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let loaded = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(loaded.settings, settings)
        XCTAssertEqual(loaded.filledFromDefaults, [])
        XCTAssertEqual(loaded.ignoredSavedKeys, [])
    }
}
