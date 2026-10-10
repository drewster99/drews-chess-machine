//
//  GPUSubmissionTests.swift
//  DrewsChessMachineTests
//
//  `GPUSubmission` is the one checked path for GPU work (GPU fault forensics
//  plan, A2). These pin: the failure rule over every reachable signal (pure,
//  so it is tested without forcing a GPU fault, which can't be done safely);
//  that the checked replacements return exactly what `MPSGraph.run` /
//  `MPSGraphExecutable.run` returned, including for work MPSGraph splits
//  across several command buffers; and the fault ledger's ordering.
//

import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import XCTest
@testable import DrewsChessMachine

final class GPUSubmissionTests: XCTestCase {

    // MARK: - Failure rule

    private static func report(
        first: MTLCommandBufferStatus = .completed,
        last: MTLCommandBufferStatus = .completed,
        split: Bool = true,
        handler: GPUSubmission.HandlerOutcome = .succeeded,
        firstError: GPUErrorDescription? = nil,
        lastError: GPUErrorDescription? = nil
    ) -> GPUSubmissionReport {
        GPUSubmissionReport(
            stage: .trainingStep, firstStatus: first, firstError: firstError, lastStatus: last,
            lastError: lastError, splitIntoSeveralBuffers: split, handler: handler, gpuMilliseconds: 12,
            firstBufferMilliseconds: 4, lastBufferMilliseconds: 6)
    }

    private static let hang = GPUErrorDescription(
        domain: "MTLCommandBufferErrorDomain", code: 3,
        message: "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)", encoderStates: [])

    func testEverySignalCompletedPasses() {
        XCTAssertNil(Self.report().failure)
        XCTAssertNil(Self.report(split: false).failure)
    }

    func testAnyBufferNotCompletedFails() {
        let notCompleted: [MTLCommandBufferStatus] = [.error, .notEnqueued, .enqueued, .committed, .scheduled]
        for status in notCompleted {
            XCTAssertEqual(Self.report(first: status).failure?.status, status, "first \(status.rawValue)")
            XCTAssertEqual(Self.report(last: status).failure?.status, status, "last \(status.rawValue)")
        }
        // Both failed: the last buffer's status is the one reported.
        XCTAssertEqual(Self.report(first: .committed, last: .error).failure?.status, .error)
    }

    func testHandlerErrorOrSilenceFailsEvenWhenBuffersCompleted() {
        let failed = Self.report(handler: .failed(Self.hang)).failure
        XCTAssertEqual(failed?.status, .completed)
        XCTAssertTrue(failed?.detail.contains("kIOGPUCommandBufferCallbackErrorHang") ?? false)
        let silent = Self.report(handler: .pending).failure
        XCTAssertTrue(silent?.detail.contains("handler=did-not-report") ?? false)
    }

    func testLogLineNamesStageEveryBufferAndTheMetalError() {
        let line = Self.report(last: .error, handler: .failed(Self.hang), lastError: Self.hang).logLine
        XCTAssertTrue(line.hasPrefix("[GPU-ERR] stage=training step "), line)
        XCTAssertTrue(line.contains("first=completed"), line)
        XCTAssertTrue(line.contains("last=error (MTLCommandBufferErrorDomain#3"), line)
        XCTAssertTrue(line.contains("handler=error"), line)
        XCTAssertTrue(line.contains("gpuMs=12"), line)
        // An unsplit submission has one buffer: the last is the first.
        XCTAssertTrue(Self.report(first: .error, split: false).logLine.contains("last=same"))
    }

    func testSlowLineTestsEachBufferNotTheSpan() {
        func report(first: Double?, last: Double?, span: Double?, split: Bool = true) -> GPUSubmissionReport {
            GPUSubmissionReport(
                stage: .batchedInference, firstStatus: .completed, firstError: nil, lastStatus: .completed,
                lastError: nil, splitIntoSeveralBuffers: split, handler: .succeeded, gpuMilliseconds: span,
                firstBufferMilliseconds: first, lastBufferMilliseconds: last)
        }
        // A long span made of short buffers (CPU encode gaps between them) is not slow.
        XCTAssertNil(report(first: 900, last: 1_200, span: 9_000).slowLine(thresholdMs: 5_000))
        let slowLast = report(first: 900, last: 5_100, span: 9_000).slowLine(thresholdMs: 5_000)
        XCTAssertEqual(slowLast, "[GPU-SLOW] stage=batched inference firstMs=900 lastMs=5100 spanMs=9000")
        XCTAssertNotNil(report(first: 5_000, last: nil, span: nil, split: false).slowLine(thresholdMs: 5_000))
        XCTAssertNil(report(first: nil, last: nil, span: nil).slowLine(thresholdMs: 5_000))
    }

    func testErrorDescriptionKeepsDomainCodeAndMessage() {
        let error = NSError(domain: "MTLCommandBufferErrorDomain", code: 5,
                            userInfo: [NSLocalizedDescriptionKey: "Discarded (victim of GPU error/recovery)"])
        let described = GPUErrorDescription(error)
        XCTAssertEqual(described.domain, "MTLCommandBufferErrorDomain")
        XCTAssertEqual(described.code, 5)
        XCTAssertEqual(described.message, "Discarded (victim of GPU error/recovery)")
        XCTAssertEqual(described.encoderStates, [])
    }

    func testSingleBufferRuleMatchesRequireCompleted() {
        let all: [MTLCommandBufferStatus] = [.notEnqueued, .enqueued, .committed, .scheduled, .completed, .error]
        for status in all {
            let passes = GPUSubmissionReport.bufferCompleted(status)
            do {
                try ChessNetwork.requireCompleted(status: status, error: nil, stage: "test")
                XCTAssertTrue(passes, "status \(status.rawValue)")
            } catch {
                XCTAssertFalse(passes, "status \(status.rawValue)")
            }
        }
    }

    // MARK: - Same results as the unchecked calls

    private func requireMetal() throws -> (MTLDevice, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        return (device, queue)
    }

    private static func floats(_ data: MPSGraphTensorData, count: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: count)
        out.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            data.mpsndarray().readBytes(base, strideBytes: nil)
        }
        return out
    }

    private static func tensorData(_ values: [Float], shape: [NSNumber], device: MTLDevice) -> MPSGraphTensorData {
        let bytes = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: bytes, shape: shape, dataType: .float32)
    }

    /// A graph of `depth` chained matmul + add steps on a 64×64 input:
    /// small at depth 1, and deep enough at 400 that MPSGraph splits it
    /// across several command buffers (the call audit's probe: 4 extra at
    /// ~400 ops).
    private static func chainGraph(depth: Int) -> (MPSGraph, MPSGraphTensor, MPSGraphTensor) {
        let graph = MPSGraph()
        let input = graph.placeholder(shape: [64, 64], dataType: .float32, name: "x")
        let scale = graph.constant(1.0 / 64.0, shape: [64, 64], dataType: .float32)
        var value = input
        for _ in 0..<depth {
            value = graph.addition(graph.matrixMultiplication(primary: value, secondary: scale, name: nil),
                                   input, name: nil)
        }
        return (graph, input, value)
    }

    func testRunGraphMatchesGraphRunForSmallAndSplitWork() throws {
        let (device, queue) = try requireMetal()
        let input = (0..<(64 * 64)).map { Float($0 % 17) / 17 }
        for depth in [1, 400] {
            let (graph, placeholder, output) = Self.chainGraph(depth: depth)
            let feeds = [placeholder: Self.tensorData(input, shape: [64, 64], device: device)]
            let unchecked = graph.run(with: queue, feeds: feeds, targetTensors: [output], targetOperations: nil)
            let checked = try GPUSubmission.runGraph(
                graph, on: queue, feeds: feeds, targetTensors: [output], targetOperations: nil, stage: .analysisTaps,
                work: .notPerPosition)
            let expected = try XCTUnwrap(unchecked[output])
            let actual = try XCTUnwrap(checked[output])
            XCTAssertEqual(Self.floats(actual, count: 64 * 64).map(\.bitPattern),
                           Self.floats(expected, count: 64 * 64).map(\.bitPattern), "depth \(depth)")
        }
    }

    func testRunExecutableMatchesExecutableRunForSmallAndSplitWork() throws {
        let (device, queue) = try requireMetal()
        let input = (0..<(64 * 64)).map { Float($0 % 13) / 13 }
        for depth in [1, 400] {
            let (graph, placeholder, output) = Self.chainGraph(depth: depth)
            let executable = graph.compile(
                with: MPSGraphDevice(mtlDevice: device),
                feeds: [placeholder: MPSGraphShapedType(shape: [64, 64], dataType: .float32)],
                targetTensors: [output], targetOperations: nil, compilationDescriptor: nil)
            let inputs = [Self.tensorData(input, shape: [64, 64], device: device)]
            let unchecked = executable.run(with: queue, inputs: inputs, results: nil, executionDescriptor: nil)
            let checked = try GPUSubmission.runExecutable(
                executable, on: queue, inputs: inputs, results: nil, stage: .batchedInference,
                work: .notPerPosition)
            XCTAssertEqual(checked.count, 1)
            XCTAssertEqual(Self.floats(checked[0], count: 64 * 64).map(\.bitPattern),
                           Self.floats(unchecked[0], count: 64 * 64).map(\.bitPattern), "depth \(depth)")
        }
    }

    /// The value baseline's form: the executable writes into a caller-owned
    /// result buffer (`results:`), small and split.
    func testRunExecutableIntoCallerOwnedResultsMatchesRun() throws {
        let (device, queue) = try requireMetal()
        let input = (0..<(64 * 64)).map { Float($0 % 11) / 11 }
        for depth in [1, 400] {
            let (graph, placeholder, output) = Self.chainGraph(depth: depth)
            let executable = graph.compile(
                with: MPSGraphDevice(mtlDevice: device),
                feeds: [placeholder: MPSGraphShapedType(shape: [64, 64], dataType: .float32)],
                targetTensors: [output], targetOperations: nil, compilationDescriptor: nil)
            let inputs = [Self.tensorData(input, shape: [64, 64], device: device)]
            let expected = executable.run(with: queue, inputs: inputs, results: nil, executionDescriptor: nil)
            let owned = Self.tensorData([Float](repeating: 0, count: 64 * 64), shape: [64, 64], device: device)
            _ = try GPUSubmission.runExecutable(
                executable, on: queue, inputs: inputs, results: [owned], stage: .valueBaseline,
                work: .notPerPosition)
            XCTAssertEqual(Self.floats(owned, count: 64 * 64).map(\.bitPattern),
                           Self.floats(expected[0], count: 64 * 64).map(\.bitPattern), "depth \(depth)")
        }
    }

    /// The weight load's / syncs' form: the work is target operations
    /// (variable assigns), with a variable as the dummy target.
    func testRunGraphWithTargetOperationsRunsTheAssigns() throws {
        let (_, queue) = try requireMetal()
        let graph = MPSGraph()
        let zeros = [Float](repeating: 0, count: 4).withUnsafeBufferPointer { Data(buffer: $0) }
        let variable = graph.variable(with: zeros, shape: [4], dataType: .float32, name: "v")
        let values: [Float] = [1, 2, 3, 4]
        let written = graph.constant(values.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [4], dataType: .float32)
        let assign = graph.assign(variable, tensor: written, name: "assign_v")
        _ = try GPUSubmission.runGraph(
            graph, on: queue, feeds: [:], targetTensors: [variable], targetOperations: [assign], stage: .weightLoad,
            work: .notPerPosition)
        let read = try GPUSubmission.runGraph(
            graph, on: queue, feeds: [:], targetTensors: [variable], targetOperations: nil, stage: .weightExport,
            work: .notPerPosition)
        XCTAssertEqual(Self.floats(try XCTUnwrap(read[variable]), count: 4), values)
    }

    func testSuccessfulSubmissionRecordsNoFault() throws {
        let (device, queue) = try requireMetal()
        let before = GPUFaultLedger.shared.latestSequence
        let (graph, placeholder, output) = Self.chainGraph(depth: 400)
        let submission = try GPUSubmission(queue: queue, stage: .analysisTaps, work: .notPerPosition)
        _ = graph.encode(
            to: submission.commandBuffer,
            feeds: [placeholder: Self.tensorData([Float](repeating: 1, count: 64 * 64), shape: [64, 64], device: device)],
            targetTensors: [output], targetOperations: nil,
            executionDescriptor: submission.graphExecutionDescriptor)
        submission.commit()
        submission.waitUntilCompleted()
        let gpuMs = try submission.verify()
        XCTAssertGreaterThanOrEqual(gpuMs, 0)
        XCTAssertEqual(GPUFaultLedger.shared.latestSequence, before)
    }

    // MARK: - Fault ledger

    func testLedgerNumbersFaultsInOrderAndReturnsThoseAfterASequence() {
        let ledger = GPUFaultLedger()
        XCTAssertEqual(ledger.latestSequence, 0)
        let first = ledger.record(.submission(stage: "training step", detail: "first=error"))
        let second = ledger.record(.systemLog(message: "Discarded (victim of GPU error/recovery)"))
        XCTAssertEqual(first.sequence, 1)
        XCTAssertEqual(second.sequence, 2)
        XCTAssertEqual(ledger.latestSequence, 2)
        XCTAssertEqual(ledger.faults(after: 0).map(\.sequence), [1, 2])
        XCTAssertEqual(ledger.faults(after: 1).map(\.sequence), [2])
        XCTAssertEqual(ledger.faults(after: 2), [])
    }

    func testLedgerKeepsOnlyTheNewestFaultsButKeepsCounting() {
        let ledger = GPUFaultLedger()
        let total = GPUFaultLedger.retainedFaultCount + 5
        for index in 0..<total {
            ledger.record(.systemLog(message: "fault \(index)"))
        }
        XCTAssertEqual(ledger.latestSequence, total)
        let kept = ledger.faults(after: 0)
        XCTAssertEqual(kept.count, GPUFaultLedger.retainedFaultCount)
        XCTAssertEqual(kept.first?.sequence, 6)
        XCTAssertEqual(kept.last?.sequence, total)
    }
}
