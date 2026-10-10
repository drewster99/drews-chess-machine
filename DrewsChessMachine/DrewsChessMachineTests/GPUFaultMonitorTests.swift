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

    // MARK: - Runs and pauses

    /// A log the test writes entries into. A query returns the entries
    /// logged at or after its start, oldest first, as `OSLogStore` does.
    private final class ScriptedFaultLog: GPUFaultLogSource, @unchecked Sendable {
        private let entries = SyncBox<[GPUFaultMonitor.LogEntry]>([])

        @discardableResult
        func add(_ message: String, at date: Date) -> GPUFaultMonitor.LogEntry {
            let entry = GPUFaultMonitor.LogEntry(date: date, message: message)
            entries.modify { $0.append(entry) }
            return entry
        }

        func faultEntries(from date: Date) throws -> [GPUFaultMonitor.LogEntry] {
            entries.read { all in all.filter { $0.date >= date }.sorted { $0.date < $1.date } }
        }
    }

    /// A monitor over `log` whose pause never runs its closing poll during
    /// a test, unless the test asks for one.
    private static func makeMonitor(
        ledger: GPUFaultLedger, log: ScriptedFaultLog, closingPollDelaySeconds: TimeInterval = 3_600
    ) -> GPUFaultMonitor {
        GPUFaultMonitor(ledger: ledger, openLogSource: { log }, closingPollDelaySeconds: closingPollDelaySeconds)
    }

    /// The bug this pins: a GUI Stop pauses the monitor and the next start
    /// resumed it reading from the last read, so a fault macOS logged
    /// between the runs was recorded after the new run's baseline and
    /// stopped that run at once. A fault older than the new run's look-back
    /// window belongs to no run: it is logged, never recorded.
    func testAFaultLoggedBetweenRunsBeforeTheLookBackWindowIsNotTheNextRunsFault() async {
        let ledger = GPUFaultLedger()
        let log = ScriptedFaultLog()
        let monitor = Self.makeMonitor(ledger: ledger, log: log)
        defer { monitor.pause() }

        monitor.start(lookBack: 60)
        await monitor.checkNow()
        monitor.pause()
        // Logged after run 1's last read (which reached back
        // `readOverlapSeconds`) and before run 2 starts; with no look-back
        // for run 2, it is outside run 2's window, as an hours-old fault is
        // outside a 60 s window.
        log.add(Self.incidentMessages[0], at: Date().addingTimeInterval(-0.5))

        monitor.start(lookBack: 0)
        let watch = GPUFaultWatch(ledger: ledger, monitor: monitor)
        await monitor.checkNow()
        XCTAssertNil(watch.firstFault)
        XCTAssertEqual(ledger.latestSequence, 0)
    }

    /// The look-back's purpose survives the fix: a fault logged shortly
    /// before a resumed run started (during the load before it, or in the
    /// stopped run's last seconds) counts for that run.
    func testAFaultInsideTheLookBackWindowBeforeAResumeCountsForTheNewRun() async {
        let ledger = GPUFaultLedger()
        let log = ScriptedFaultLog()
        let monitor = Self.makeMonitor(ledger: ledger, log: log)
        defer { monitor.pause() }

        monitor.start(lookBack: 60)
        await monitor.checkNow()
        monitor.pause()
        let entry = log.add(Self.incidentMessages[1], at: Date().addingTimeInterval(-0.5))

        monitor.start(lookBack: 60)
        let watch = GPUFaultWatch(ledger: ledger, monitor: monitor)
        await monitor.checkNow()
        XCTAssertEqual(watch.faultsSinceStart.map(\.time), [entry.date])
        XCTAssertEqual(watch.faultsSinceStart.map(\.source), [.systemLog(message: entry.message)])
    }

    /// First start: faults inside the look-back count; one in the read
    /// overlap just before it is reported but not counted; an older one is
    /// never read.
    func testFirstStartCountsOnlyTheLookBackWindow() async {
        let ledger = GPUFaultLedger()
        let log = ScriptedFaultLog()
        let now = Date()
        log.add(Self.incidentMessages[0], at: now.addingTimeInterval(-70))
        log.add(Self.incidentMessages[0], at: now.addingTimeInterval(-62))
        let inWindow = log.add(Self.incidentMessages[1], at: now.addingTimeInterval(-30))
        let monitor = Self.makeMonitor(ledger: ledger, log: log)
        defer { monitor.pause() }

        monitor.start(lookBack: 60)
        let watch = GPUFaultWatch(ledger: ledger, monitor: monitor)
        await monitor.checkNow()
        XCTAssertEqual(watch.faultsSinceStart.map(\.time), [inWindow.date])
    }

    /// An entry recorded by one run and returned again by the next run's
    /// overlapping read is not recorded twice.
    func testAnEntryReadAgainAfterAResumeIsNotRecordedTwice() async {
        let ledger = GPUFaultLedger()
        let log = ScriptedFaultLog()
        let monitor = Self.makeMonitor(ledger: ledger, log: log)
        defer { monitor.pause() }

        monitor.start(lookBack: 60)
        log.add(Self.incidentMessages[0], at: Date().addingTimeInterval(-1))
        await monitor.checkNow()
        XCTAssertEqual(ledger.latestSequence, 1)
        monitor.pause()

        monitor.start(lookBack: 60)
        let watch = GPUFaultWatch(ledger: ledger, monitor: monitor)
        await monitor.checkNow()
        XCTAssertNil(watch.firstFault)
        XCTAssertEqual(ledger.latestSequence, 1)
    }

    /// A pause's closing poll records a fault from the run's last seconds
    /// (logged after its last poll) before the next run begins, so that run
    /// doesn't count it; after the closing poll, `checkNow` reads nothing.
    func testThePausesClosingPollRecordsTheRunsLastFaultAndThenReadsNothing() async {
        let ledger = GPUFaultLedger()
        let log = ScriptedFaultLog()
        let monitor = Self.makeMonitor(ledger: ledger, log: log, closingPollDelaySeconds: 0)
        defer { monitor.pause() }

        monitor.start(lookBack: 60)
        await monitor.checkNow()
        let lastSeconds = log.add(Self.incidentMessages[0], at: Date().addingTimeInterval(-0.5))
        monitor.pause()
        let recorded = XCTNSPredicateExpectation(
            predicate: NSPredicate { _, _ in ledger.latestSequence == 1 }, object: nil)
        await fulfillment(of: [recorded], timeout: 10)
        XCTAssertEqual(ledger.faults(after: 0).map(\.time), [lastSeconds.date])

        log.add(Self.incidentMessages[1], at: Date())
        await monitor.checkNow()
        XCTAssertEqual(ledger.latestSequence, 1, "a paused monitor whose closing poll ran reads nothing")
    }

    func testStartWindow() {
        let now = Date(timeIntervalSince1970: 1_791_600_000)
        // First start: read and count from the look-back.
        XCTAssertEqual(
            GPUFaultMonitor.startWindow(lastReadFrom: nil, now: now, lookBack: 60),
            .init(readFrom: now - 60, countFrom: now - 60, unread: nil))
        // A pause shorter than the look-back is read in full.
        XCTAssertEqual(
            GPUFaultMonitor.startWindow(lastReadFrom: now - 20, now: now, lookBack: 60),
            .init(readFrom: now - 20, countFrom: now - 60, unread: nil))
        // Last read exactly at the window's start: nothing unread.
        XCTAssertEqual(
            GPUFaultMonitor.startWindow(lastReadFrom: now - 60, now: now, lookBack: 60),
            .init(readFrom: now - 60, countFrom: now - 60, unread: nil))
        // A three-hour pause: only the window is read; the rest is reported.
        XCTAssertEqual(
            GPUFaultMonitor.startWindow(lastReadFrom: now - 10_800, now: now, lookBack: 60),
            .init(readFrom: now - 60, countFrom: now - 60,
                  unread: DateInterval(start: now - 10_800, end: now - 60)))
    }

    func testNewFaultEntriesCountFromTheWindowStartAndReportEachEntryOnce() {
        let countFrom = Date(timeIntervalSince1970: 1_791_600_000)
        let atStart = GPUFaultMonitor.LogEntry(date: countFrom, message: Self.incidentMessages[0])
        let justBefore = GPUFaultMonitor.LogEntry(date: countFrom - 0.001, message: Self.incidentMessages[1])
        let notAFault = GPUFaultMonitor.LogEntry(date: countFrom + 1, message: Self.otherMessages[0])
        var seen = Set<GPUFaultMonitor.LogEntry>()

        let first = GPUFaultMonitor.newFaultEntries([justBefore, atStart, notAFault], countFrom: countFrom, seen: &seen)
        XCTAssertEqual(first, [.init(entry: justBefore, counts: false), .init(entry: atStart, counts: true)])
        XCTAssertEqual(seen, [justBefore, atStart])

        let again = GPUFaultMonitor.newFaultEntries([justBefore, atStart], countFrom: countFrom, seen: &seen)
        XCTAssertEqual(again, [])
    }

    /// After Stop, only a fault from the run's last seconds (up to the
    /// closing poll) is the stopped trainer's; a later one (Play Game,
    /// another process's GPU reset) is not, and one from before the run
    /// began never was.
    func testAStoppedRunOwnsOnlyFaultsUpToItsTail() {
        let ledger = GPUFaultLedger()
        let stoppedAt = Date(timeIntervalSince1970: 1_791_600_000)
        let tail = StoppedRunGPUFaultWatch.tailSeconds
        ledger.record(.systemLog(message: Self.incidentMessages[0]), at: stoppedAt - 30)
        let watch = GPUFaultWatch(ledger: ledger, monitor: GPUFaultMonitor(ledger: ledger))
        let stopped = StoppedRunGPUFaultWatch(watch: watch, stoppedAt: stoppedAt)
        XCTAssertNil(stopped.firstRunFault, "a fault recorded before the run began")

        ledger.record(.systemLog(message: Self.incidentMessages[1]), at: stoppedAt + tail + 0.001)
        XCTAssertNil(stopped.firstRunFault, "a fault after the tail is not the stopped run's")

        let atTailEnd = ledger.record(.systemLog(message: Self.incidentMessages[1]), at: stoppedAt + tail)
        XCTAssertEqual(stopped.firstRunFault, atTailEnd)
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

    /// A failed `task_info` read shows `n/a`, never a footprint of zero.
    func testMemoryLineShowsAnUnreadFootprintAsNotAvailable() {
        let reading = MemoryStatusMonitor.Reading(
            footprintBytes: nil, gpuAllocatedBytes: 1_610_612_736, gpuRecommendedMaxBytes: 96 * 1_073_741_824,
            swapUsedBytes: 27_136_098_304, swapTotalBytes: 27_917_287_424, pressure: .critical, thermal: .serious,
            faultMonitorPollsMs: [])
        XCTAssertEqual(
            reading.line(trigger: "pressure-change"),
            "[MEM] pressure-change footprint=n/a gpu=1.50GB/96.00GB swap=25.27GB/26.00GB "
                + "pressure=critical thermal=serious gpuFaultPolls=(none)")
    }

    /// The reader the `[MEM]` line uses succeeds in a live process, and
    /// the zero-on-failure reader built on it does too.
    func testPhysFootprintReadSucceedsInThisProcess() throws {
        let bytes = try ChessTrainer.readPhysFootprintBytes().get()
        XCTAssertGreaterThan(bytes, 0)
        XCTAssertGreaterThan(ChessTrainer.currentPhysFootprintBytes(), 0)
    }
}
