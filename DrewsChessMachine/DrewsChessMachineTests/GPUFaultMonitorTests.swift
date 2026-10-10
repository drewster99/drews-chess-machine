//
//  GPUFaultMonitorTests.swift
//  DrewsChessMachineTests
//
//  The system-log layer of GPU fault detection (GPU fault forensics plan,
//  A3) and what a run records about it. The fixtures are the messages macOS
//  logged in the training processes during the three GPU resets of
//  2026-10-09 (19:36:49, 21:37:09, 22:59:04), copied from `log show`.
//

import XCTest
@testable import DrewsChessMachine

final class GPUFaultMonitorTests: XCTestCase {

    /// Metal's messages from the 2026-10-09 resets, one per command buffer.
    private static let incidentMessages = [
        "Execution of the command buffer was aborted due to an error during execution. "
            + "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)",
        "Execution of the command buffer was aborted due to an error during execution. "
            + "Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)",
    ]

    /// Lines that appeared around the faults and must not count.
    private static let otherMessages = [
        "IOGPUMetalError: <private>",
        "HALC_ProxyIOContext::IOWorkLoop: skipping cycle due to overload",
        "[SP-TICK] network error: skipping tick",
        "",
    ]

    func testIncidentMessagesAreFaultsAndNeighboursAreNot() {
        for message in Self.incidentMessages {
            XCTAssertTrue(GPUFaultMonitor.isFaultMessage(message), message)
        }
        for message in Self.otherMessages {
            XCTAssertFalse(GPUFaultMonitor.isFaultMessage(message), message)
        }
    }

    func testAvailabilityLabels() {
        XCTAssertEqual(GPUFaultMonitor.Availability.notStarted.label, "not started")
        XCTAssertEqual(GPUFaultMonitor.Availability.available.label, "available")
        XCTAssertEqual(GPUFaultMonitor.Availability.unavailable("no store").label, "unavailable: no store")
    }

    func testAMonitorThatNeverStartedRecordsNothingAndSaysSo() async {
        let ledger = GPUFaultLedger()
        let monitor = GPUFaultMonitor(ledger: ledger)
        await monitor.checkNow()
        XCTAssertEqual(ledger.latestSequence, 0)
        XCTAssertEqual(monitor.availability, .notStarted)
    }

    // MARK: - The run's view

    func testWatchSeesOnlyFaultsAfterItsStart() {
        let ledger = GPUFaultLedger()
        ledger.record(.systemLog(message: Self.incidentMessages[0]))
        let watch = GPUFaultWatch(ledger: ledger, monitor: GPUFaultMonitor(ledger: ledger))
        XCTAssertNil(watch.firstFault)
        let fault = ledger.record(.submission(stage: "training step", detail: "first=error"))
        ledger.record(.systemLog(message: Self.incidentMessages[1]))
        XCTAssertEqual(watch.firstFault, fault)
        XCTAssertEqual(watch.faultsSinceStart.count, 2)
    }

    func testReportCarriesMonitorAvailabilityStopAndEveryFault() throws {
        let ledger = GPUFaultLedger()
        let watch = GPUFaultWatch(ledger: ledger, monitor: GPUFaultMonitor(ledger: ledger))
        let none = watch.report(stoppedAtTrainerStep: nil, crashDumps: [])
        XCTAssertEqual(none.monitor, "not started")
        XCTAssertEqual(none.faults, [])
        XCTAssertNil(none.stoppedAtTrainerStep)

        ledger.record(.systemLog(message: Self.incidentMessages[0]),
                      at: Date(timeIntervalSince1970: 1_791_597_429.5))
        let report = watch.report(stoppedAtTrainerStep: 45_975, crashDumps: ["/tmp/dump"])
        XCTAssertEqual(report.faults.count, 1)
        XCTAssertEqual(report.faults[0].source, "system log")
        XCTAssertEqual(report.faults[0].detail, Self.incidentMessages[0])
        XCTAssertTrue(report.faults[0].time.hasSuffix(".500Z"), report.faults[0].time)

        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(report)) as? [String: Any]
        XCTAssertEqual(json?["stopped_at_trainer_step"] as? Int, 45_975)
        XCTAssertEqual(json?["crash_dumps"] as? [String], ["/tmp/dump"])
        XCTAssertEqual(json?["monitor"] as? String, "not started")
    }

    // MARK: - The suspension a GPU fault uses

    func testDivergenceSuspensionSkipsTheAutosaveSoSuspectWeightsAreNeverSaved() {
        let suspension = TrainingSuspension.divergence(reason: "GPU fault (training step): …")
        XCTAssertTrue(suspension.skipsPeriodicAutosave)
        XCTAssertEqual(suspension.arenaSkipLabel, "divergence")
    }

    // MARK: - [MEM] line

    func testMemoryLineShowsBase2SizesAndThePollCost() {
        let reading = MemoryStatusMonitor.Reading(
            footprintBytes: 3 * 1_073_741_824,
            gpuAllocatedBytes: 1_610_612_736,
            gpuRecommendedMaxBytes: 96 * 1_073_741_824,
            swapUsedBytes: 27_136_098_304,
            swapTotalBytes: 27_917_287_424,
            pressure: .warning,
            thermal: .fair,
            faultMonitorPollsMs: [2_100, 1_800, 1_900])
        XCTAssertEqual(
            reading.line(trigger: "periodic"),
            "[MEM] periodic footprint=3.00GB gpu=1.50GB/96.00GB swap=25.27GB/26.00GB "
                + "pressure=warning thermal=fair gpuFaultPolls=(n=3 p50=1900ms max=2100ms)")
        let unread = MemoryStatusMonitor.Reading(
            footprintBytes: 0, gpuAllocatedBytes: nil, gpuRecommendedMaxBytes: nil,
            swapUsedBytes: nil, swapTotalBytes: nil, pressure: .normal, thermal: .nominal,
            faultMonitorPollsMs: [])
        XCTAssertEqual(
            unread.line(trigger: "start"),
            "[MEM] start footprint=0.00GB gpu=n/a/n/a swap=n/a/n/a pressure=normal thermal=nominal gpuFaultPolls=(none)")
    }
}
