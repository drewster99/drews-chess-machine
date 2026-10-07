//
//  RunBudgetAndSavedDelayTests.swift
//  DrewsChessMachineTests
//
//  Two values a lineage record states that used to be stated wrongly
//  (hyperparameter recording plan P4):
//
//  - O-21: corpus replay enforces no wall-clock limit, yet it accepted a
//    parameters file's `training_time_limit` and ignored it, so the run's
//    recorded budget would have read as one it honoured. It now refuses.
//  - O-20: the replay-ratio controller's saved starting delay read 50 ms when
//    nothing was saved (and when the stored value was not an integer), a
//    silent first-launch value the record would have stated as the
//    controller's state. It now reads "none saved" — the start then uses
//    `training_step_delay_ms` and says so — and refuses a malformed value.
//

import XCTest
@testable import DrewsChessMachine

final class RunBudgetAndSavedDelayTests: XCTestCase {

    // MARK: - O-21

    func testCorpusReplayRefusesAParametersFileWithATimeLimit() throws {
        let timed = CliTrainingConfig(trainingParameters: [:], trainingTimeLimitSec: 600, trainingStepLimit: nil)
        let refusal = try XCTUnwrap(timed.corpusReplayRefusal(parametersPath: "/runs/p.json"),
                                    "a time limit corpus replay would ignore is refused")
        XCTAssertTrue(refusal.message.contains(CliTrainingConfig.trainingTimeLimitKey), refusal.message)
        XCTAssertTrue(refusal.message.contains("/runs/p.json"), refusal.message)

        let stepped = CliTrainingConfig(trainingParameters: [:], trainingTimeLimitSec: nil, trainingStepLimit: 5_000)
        XCTAssertNil(stepped.corpusReplayRefusal(parametersPath: "/runs/p.json"))
    }

    // MARK: - O-20

    func testNothingSavedReadsAsNoSavedDelay() throws {
        let defaults = try makeTemporaryDefaults()
        XCTAssertNil(try SessionController.savedAutoComputedDelayMs(in: defaults))
    }

    func testASavedDelayReadsBack() throws {
        let defaults = try makeTemporaryDefaults()
        defaults.set(137, forKey: SessionController.lastAutoComputedDelayMsKey)
        XCTAssertEqual(try SessionController.savedAutoComputedDelayMs(in: defaults), 137)
    }

    func testAMalformedSavedDelayIsAnError() throws {
        let defaults = try makeTemporaryDefaults()
        defaults.set("137", forKey: SessionController.lastAutoComputedDelayMsKey)
        XCTAssertThrowsError(try SessionController.savedAutoComputedDelayMs(in: defaults)) { error in
            guard case ReplayRatioInitialDelay.StoredDelayError.notAnInteger = error else {
                return XCTFail("\(error)")
            }
        }
    }
}
