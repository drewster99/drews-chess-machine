import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeAlertTests: XCTestCase {
    private let ourAccountID = "drewschessmachine"
    private let alerts = LichessBotAlertSettings(botChallengeSoundName: "Ping", humanChallengeSoundName: "Glass")

    func testBotChallengerGetsBotSound() {
        let name = LichessBotChallengeAlert.soundName(challengerID: "somebot", challengerTitle: "BOT", ourAccountID: ourAccountID, alerts: alerts)
        XCTAssertEqual(name, "Ping")
    }

    func testUntitledChallengerGetsHumanSound() {
        let name = LichessBotChallengeAlert.soundName(challengerID: "someone", challengerTitle: nil, ourAccountID: ourAccountID, alerts: alerts)
        XCTAssertEqual(name, "Glass")
    }

    func testTitledHumanGetsHumanSound() {
        let name = LichessBotChallengeAlert.soundName(challengerID: "grandmaster", challengerTitle: "GM", ourAccountID: ourAccountID, alerts: alerts)
        XCTAssertEqual(name, "Glass")
    }

    func testOwnOutgoingEchoIsSilentEvenThoughWeAreABot() {
        let name = LichessBotChallengeAlert.soundName(challengerID: "DrewsChessMachine", challengerTitle: "BOT", ourAccountID: ourAccountID, alerts: alerts)
        XCTAssertNil(name)
        XCTAssertTrue(LichessBotChallengeAlert.isOwnOutgoingEcho(challengerID: "DrewsChessMachine", ourAccountID: "DREWSCHESSMACHINE"))
    }

    func testNoneSelectedPlaysNothing() {
        let silent = LichessBotAlertSettings(botChallengeSoundName: nil, humanChallengeSoundName: nil)
        XCTAssertNil(LichessBotChallengeAlert.soundName(challengerID: "somebot", challengerTitle: "BOT", ourAccountID: ourAccountID, alerts: silent))
        XCTAssertNil(LichessBotChallengeAlert.soundName(challengerID: "someone", challengerTitle: nil, ourAccountID: ourAccountID, alerts: silent))
    }

    func testDefaultsAreSilent() {
        let defaults = LichessBotSettings().alerts
        XCTAssertNil(defaults.botChallengeSoundName)
        XCTAssertNil(defaults.humanChallengeSoundName)
    }
}
