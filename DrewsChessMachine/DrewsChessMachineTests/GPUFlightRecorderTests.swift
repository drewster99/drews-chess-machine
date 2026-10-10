//
//  GPUFlightRecorderTests.swift
//  DrewsChessMachineTests
//
//  `GPUFlightRecorder` keeps the last couple of minutes of GPU submissions so
//  a fault can be matched to the work on the GPU when it happened (owner
//  request 2026-10-10, after three hangs during arenas). These pin: which
//  submissions a fault's account lists (running in the window before it,
//  including one that never finished), that one reset's several reports are
//  accounted once, that a failure isn't overwritten by a later success,
//  what the history keeps and drops, the labels, that a real checked
//  submission is recorded with its network, stage and size, and the crash
//  dump's `gpu-submissions.json`.
//

import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import XCTest
@testable import DrewsChessMachine

final class GPUFlightRecorderTests: XCTestCase {

    private let faultTime = Date(timeIntervalSince1970: 1_800_000_000)

    private func fault(_ sequence: Int, at time: Date) -> GPUFaultLedger.Fault {
        GPUFaultLedger.Fault(sequence: sequence, time: time, source: .systemLog(message: "test fault"))
    }

    func testAccountListsTheSubmissionsRunningBeforeTheFaultAndOneThatNeverFinished() {
        let recorder = GPUFlightRecorder()
        let doneLongBefore = recorder.begin(stage: .batchedInference, queue: "champion (self-play)",
                                            work: .positions(64), at: faultTime.addingTimeInterval(-10))
        recorder.finish(doneLongBefore, outcome: .completed, at: faultTime.addingTimeInterval(-9.9))
        let hung = recorder.begin(stage: .batchedInference, queue: "arena candidate",
                                  work: .positionsOnNewlyCompiledShape(37), at: faultTime.addingTimeInterval(-5))
        recorder.finish(hung, outcome: .failed("first=error"), at: faultTime.addingTimeInterval(-0.5))
        let neverFinished = recorder.begin(stage: .trainingStep, queue: "trainer",
                                           work: .positions(4096), at: faultTime.addingTimeInterval(-1))
        _ = recorder.begin(stage: .weightExport, queue: "trainer", work: .notPerPosition,
                           at: faultTime.addingTimeInterval(1))

        let lines = recorder.account(for: fault(1, at: faultTime))

        XCTAssertTrue(lines[0].hasPrefix("[GPU-INFLIGHT] fault #1 at "), lines[0])
        XCTAssertTrue(lines[0].contains(": 2 submission(s) running in the 2 s before it (4 recorded since "), lines[0])
        let listed = lines.filter { $0.hasPrefix("[GPU-INFLIGHT]   seq=") }
        XCTAssertEqual(listed.count, 2, lines.joined(separator: "\n"))
        XCTAssertEqual(listed[0], "[GPU-INFLIGHT]   seq=\(hung) stage=batched inference queue=\"arena candidate\" "
                       + "n=37 new-shape started=-5000ms ran=4500ms outcome=failed (first=error)")
        XCTAssertEqual(listed[1], "[GPU-INFLIGHT]   seq=\(neverFinished) stage=training step queue=\"trainer\" "
                       + "n=4096 started=-1000ms still-running outcome=running")
        XCTAssertTrue(lines.contains("[GPU-INFLIGHT] last 10 s by network and stage:"), lines.joined(separator: "\n"))
        XCTAssertTrue(lines.contains("[GPU-INFLIGHT]   queue=\"arena candidate\" stage=batched inference count=1 "
                                     + "sizes=[37] newShapes=1 longestMs=4500"), lines.joined(separator: "\n"))
        XCTAssertTrue(lines.contains("[GPU-INFLIGHT]   queue=\"champion (self-play)\" stage=batched inference count=1 "
                                     + "sizes=[64] newShapes=0 longestMs=100"), lines.joined(separator: "\n"))
        XCTAssertTrue(lines.contains("[GPU-INFLIGHT]   queue=\"trainer\" stage=training step count=1 "
                                     + "sizes=[4096] newShapes=0 longestMs=n/a stillRunning=1"),
                      lines.joined(separator: "\n"))
        XCTAssertFalse(lines.contains { $0.contains("weight export") }, "started after the fault")
    }

    func testOneResetsSeveralReportsAreAccountedOnce() {
        let recorder = GPUFlightRecorder()
        _ = recorder.begin(stage: .batchedInference, queue: "q", work: .positions(8), at: faultTime.addingTimeInterval(-1))

        XCTAssertGreaterThan(recorder.account(for: fault(1, at: faultTime)).count, 1)
        let repeated = recorder.account(for: fault(2, at: faultTime.addingTimeInterval(3)))
        XCTAssertEqual(repeated.count, 1)
        XCTAssertTrue(repeated[0].hasPrefix("[GPU-INFLIGHT] fault #2 at "), repeated[0])
        XCTAssertTrue(repeated[0].contains(": within 5 s of fault #1 ("), repeated[0])
        // An earlier report of the same reset (the monitor reads macOS's
        // log time, which can precede a failed check) coalesces too.
        XCTAssertEqual(recorder.account(for: fault(3, at: faultTime.addingTimeInterval(-2))).count, 1)
        XCTAssertGreaterThan(recorder.account(for: fault(4, at: faultTime.addingTimeInterval(10))).count, 1)
    }

    func testFinishKeepsTheFirstTimeAndAFailure() throws {
        let recorder = GPUFlightRecorder()
        let sequence = recorder.begin(stage: .valueBaseline, queue: "q", work: .positions(16), at: faultTime)
        recorder.finish(sequence, outcome: .failed("handler error"), at: faultTime.addingTimeInterval(1))
        recorder.finish(sequence, outcome: .completed, gpuMilliseconds: 12, at: faultTime.addingTimeInterval(2))

        let record = try XCTUnwrap(recorder.snapshot().first)
        XCTAssertEqual(record.finishedAt, faultTime.addingTimeInterval(1))
        XCTAssertEqual(record.outcome, .failed("handler error"))
        XCTAssertEqual(record.gpuMilliseconds, 12)
    }

    func testHistoryDropsOldFinishedRecordsButKeepsRunningOnes() {
        let recorder = GPUFlightRecorder()
        let old = recorder.begin(stage: .batchedInference, queue: "q", work: .positions(1), at: faultTime)
        recorder.finish(old, outcome: .completed, at: faultTime.addingTimeInterval(1))
        let running = recorder.begin(stage: .batchedInference, queue: "q", work: .positions(2),
                                     at: faultTime.addingTimeInterval(1))
        let recent = recorder.begin(stage: .batchedInference, queue: "q", work: .positions(3),
                                    at: faultTime.addingTimeInterval(1 + GPUFlightRecorder.retentionSeconds + 1))

        XCTAssertEqual(recorder.snapshot().map(\.sequence), [running, recent])
        // Finishing a dropped record is ignored.
        recorder.finish(old, outcome: .failed("late"), at: faultTime.addingTimeInterval(500))
        XCTAssertEqual(recorder.snapshot().map(\.sequence), [running, recent])
        XCTAssertEqual(recorder.evictedWhileRunning, 0)
    }

    func testCapacityBoundsTheHistoryAndCountsRunningRecordsItDrops() {
        let recorder = GPUFlightRecorder()
        for _ in 0...GPUFlightRecorder.capacity {
            _ = recorder.begin(stage: .batchedInference, queue: "q", work: .positions(1), at: faultTime)
        }
        let kept = recorder.snapshot()
        XCTAssertEqual(kept.count, GPUFlightRecorder.capacity)
        XCTAssertEqual(kept.first?.sequence, 2)
        XCTAssertEqual(recorder.evictedWhileRunning, 1)
    }

    func testWorkSizeLabels() {
        XCTAssertEqual(GPUWorkSize.notPerPosition.label, "n/a")
        XCTAssertEqual(GPUWorkSize.positions(37).label, "n=37")
        XCTAssertEqual(GPUWorkSize.positionsOnNewlyCompiledShape(37).label, "n=37 new-shape")
    }

    func testCheckedSubmissionIsRecordedWithItsNetworkStageAndSize() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        queue.label = "flight recorder test network"
        let recorder = GPUFlightRecorder()
        let graph = MPSGraph()
        let input = graph.placeholder(shape: [4], dataType: .float32, name: "x")
        let output = graph.addition(input, input, name: nil)
        let bytes = [Float](repeating: 1, count: 4).withUnsafeBufferPointer { Data(buffer: $0) }
        let feed = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: bytes, shape: [4],
                                      dataType: .float32)

        let submission = try GPUSubmission(queue: queue, stage: .analysisTaps, work: .positions(4), recorder: recorder)
        XCTAssertEqual(submission.commandBuffer.commandBuffer.label,
                       "analysis taps · flight recorder test network · n=4")
        XCTAssertEqual(recorder.snapshot(), [], "recorded when the encode begins, not at creation")
        _ = graph.encode(to: submission.commandBuffer, feeds: [input: feed], targetTensors: [output],
                         targetOperations: nil, executionDescriptor: submission.graphExecutionDescriptor)
        submission.commit()
        try submission.verify()

        let records = recorder.snapshot()
        XCTAssertEqual(records.count, 1)
        let record = try XCTUnwrap(records.first)
        XCTAssertEqual(record.stage, .analysisTaps)
        XCTAssertEqual(record.queue, "flight recorder test network")
        XCTAssertEqual(record.work, .positions(4))
        XCTAssertEqual(record.outcome, .completed)
        let finishedAt = try XCTUnwrap(record.finishedAt)
        XCTAssertGreaterThanOrEqual(finishedAt, record.startedAt)
        XCTAssertNotNil(record.gpuMilliseconds)
    }

    func testCrashDumpSubmissionsFileListsTheHistory() throws {
        let recorder = GPUFlightRecorder()
        let done = recorder.begin(stage: .trainingStep, queue: "trainer", work: .positions(4096), at: faultTime)
        recorder.finish(done, outcome: .completed, gpuMilliseconds: 250, at: faultTime.addingTimeInterval(0.3))
        _ = recorder.begin(stage: .batchedInference, queue: "arena champion", work: .positionsOnNewlyCompiledShape(12),
                           at: faultTime.addingTimeInterval(0.1))

        let data = try CrashDumpWriter.gpuSubmissionsFile(recorder)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["retention_seconds"] as? Double, GPUFlightRecorder.retentionSeconds)
        XCTAssertEqual(object["evicted_while_running"] as? Int, 0)
        let submissions = try XCTUnwrap(object["submissions"] as? [[String: Any]])
        XCTAssertEqual(submissions.count, 2)
        XCTAssertEqual(submissions[0]["stage"] as? String, "training step")
        XCTAssertEqual(submissions[0]["queue"] as? String, "trainer")
        XCTAssertEqual(submissions[0]["work"] as? String, "n=4096")
        XCTAssertEqual(submissions[0]["outcome"] as? String, "completed")
        XCTAssertEqual(submissions[0]["gpu_ms"] as? Double, 250)
        XCTAssertEqual(submissions[0]["started_at"] as? String, GPUFlightRecorder.timestamp(faultTime))
        XCTAssertEqual(submissions[0]["finished_at"] as? String,
                       GPUFlightRecorder.timestamp(faultTime.addingTimeInterval(0.3)))
        XCTAssertEqual(submissions[1]["work"] as? String, "n=12 new-shape")
        XCTAssertEqual(submissions[1]["outcome"] as? String, "running")
        XCTAssertTrue(submissions[1]["finished_at"] is NSNull)
    }
}
