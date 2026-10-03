//
//  WorkerPauseGateHoldTests.swift
//  DrewsChessMachineTests
//
//  A pause gate is not counted: one `resume()` releases the worker whoever
//  paused it. A session save releases self-play from more than one place
//  (when its replay buffer is written, and on every way it can end), so its
//  hold must resume the gate once only — a second resume would release a
//  pause another coordinator took in between.
//

import XCTest
@testable import DrewsChessMachine

final class WorkerPauseGateHoldTests: XCTestCase {

    func testAHoldResumesItsGateOnlyOnce() async {
        let gate = WorkerPauseGate()
        let stop = SyncBox<Bool>(false)
        let workerDone = DispatchGroup()
        DispatchQueue.global(qos: .userInitiated).async(group: workerDone) {
            while !stop.value {
                if gate.isRequestedToPause {
                    gate.markWaiting()
                    while gate.isRequestedToPause && !stop.value { usleep(200) }
                    gate.markRunning()
                } else {
                    usleep(200)
                }
            }
        }

        let firstPaused = await gate.pauseAndWait(timeoutMs: 5_000)
        XCTAssertTrue(firstPaused)
        let hold = WorkerPauseGateHold(gate)
        hold.release()
        XCTAssertFalse(gate.isRequestedToPause, "the first release resumes the gate")

        let secondPaused = await gate.pauseAndWait(timeoutMs: 5_000)
        XCTAssertTrue(secondPaused, "another coordinator pauses the gate")
        hold.release()
        XCTAssertTrue(gate.isRequestedToPause, "a second release must not resume another coordinator's pause")

        gate.resume()
        stop.value = true
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            workerDone.notify(queue: .global()) { continuation.resume() }
        }
    }
}
