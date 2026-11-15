//
//  GraphAuditCLI.swift
//  DrewsChessMachine
//
//  `--graph-audit <config.json> --graph-audit-out <new file>`: the fp64 graph
//  audit (`GraphAudit`), one JSON line per op and per gradient check, headless.
//  `--graph-audit-selftest`: checks every float64 backward reference against
//  central finite differences of its own float64 forward reference, so the
//  references the audit judges MPSGraph by are themselves verified.
//

import Foundation

enum GraphAuditCLI {
    static let modeFlag = "--graph-audit"
    static let outFlag = "--graph-audit-out"
    static let selfTestFlag = "--graph-audit-selftest"

    private enum ExitCode: Int32 {
        case usage = 2
        case configUnusable = 3
        case outputUnusable = 4
        case caseFailed = 5
        case gpuFault = 36
        case selfTestFailed = 6
    }

    static func handleIfRequested(rawArgs: [String]) {
        if rawArgs.contains(selfTestFlag) {
            runSelfTestAndExit()
        }
        guard let index = rawArgs.firstIndex(of: modeFlag) else { return }
        let allowed: Set<String> = [modeFlag, outFlag]
        if let bad = rawArgs.first(where: { $0.hasPrefix("--") && !allowed.contains($0) }) {
            fail("error: \(modeFlag) does not accept '\(bad)'", code: .usage, loggerStarted: false)
        }
        guard index + 1 < rawArgs.count, !rawArgs[index + 1].hasPrefix("--") else {
            fail("error: \(modeFlag) needs a config file", code: .usage, loggerStarted: false)
        }
        guard let outIndex = rawArgs.firstIndex(of: outFlag), outIndex + 1 < rawArgs.count else {
            fail("error: \(modeFlag) needs \(outFlag) <new file>", code: .usage, loggerStarted: false)
        }
        runAndExit(configPath: rawArgs[index + 1], outPath: rawArgs[outIndex + 1])
    }

    private static func runAndExit(configPath: String, outPath: String) -> Never {
        SessionLogger.shared.start()
        let config: GraphAudit.Config
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: (configPath as NSString).expandingTildeInPath))
            config = try JSONDecoder().decode(GraphAudit.Config.self, from: data)
        } catch {
            fail("error: config \(configPath): \(error.localizedDescription)", code: .configUnusable)
        }
        let output: FileSafety.OpenedForWriting
        do {
            output = try FileSafety.openForWritingReportingCreation(
                at: URL(fileURLWithPath: (outPath as NSString).expandingTildeInPath), existingRegularFile: .refuse)
        } catch {
            fail("error: \(outFlag) \(outPath): \(error.localizedDescription)", code: .outputUnusable)
        }
        func emit(_ record: GraphAudit.Record) throws {
            try ProbeModelCLI.appendLine(try ProbeModelCLI.encodeLine(record), to: output.handle)
        }
        let limits = GraphAudit.Limits(
            chunkBytes: config.chunkBytes ?? 1 << 30,
            footprintBytes: UInt64((config.memoryLimitGiB ?? 12) * Double(1 << 30)))
        let faultWatch = GPUFaultWatch.startForRun()
        var failed: [String] = []
        for auditCase in config.cases {
            progress("case \(auditCase.label) start")
            do {
                try GraphAudit.run(auditCase, limits: limits, emit: emit, progress: progress)
            } catch {
                failed.append(auditCase.label)
                progress("case \(auditCase.label) FAILED: \(error.localizedDescription)")
                do {
                    try emit(["event": "case_error", "case": auditCase.label, "error": error.localizedDescription])
                } catch {
                    fail("error: writing \(outFlag): \(error.localizedDescription)", code: .outputUnusable)
                }
            }
            let fault: GPUFaultLedger.Fault?
            do {
                fault = try runBlocking { await faultWatch.barrier() }
            } catch {
                fail("error: the GPU fault barrier failed: \(error.localizedDescription)", code: .gpuFault)
            }
            if let fault {
                fail("GPU fault during case \(auditCase.label) (\(fault.summary)); results after it are unreliable",
                     code: .gpuFault)
            }
        }
        do {
            try output.handle.close()
        } catch {
            fail("error: closing \(outFlag): \(error.localizedDescription)", code: .outputUnusable)
        }
        guard failed.isEmpty else {
            fail("\(failed.count) case(s) failed: \(failed.joined(separator: ", "))", code: .caseFailed)
        }
        progress("done: \(config.cases.count) case(s)")
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - Self-test

    /// Each op kind's float64 VJP against central finite differences of its
    /// float64 forward, on small random operands, along random directions.
    private static func runSelfTestAndExit() -> Never {
        var generator = DCMRandom(seed: 20261114)
        func random(_ shape: [Int], positive: Bool = false) -> Tensor64 {
            Tensor64(shape: shape, values: (0..<shape.reduce(1, *)).map { _ in
                let v = Double.random(in: -1...1, using: &generator)
                // Keep away from activation kinks, and positive where required.
                return positive ? 0.5 + abs(v) : (abs(v) < 0.05 ? v + 0.1 : v)
            })
        }
        let conv3 = GraphOperationKind.convolution2D(input: "x", weights: "w", strideY: 1, strideX: 1, dilationY: 1,
                                                     dilationX: 1, groups: 1, paddingTop: 1, paddingBottom: 1,
                                                     paddingLeft: 1, paddingRight: 1)
        let conv1 = GraphOperationKind.convolution2D(input: "x", weights: "w", strideY: 1, strideX: 1, dilationY: 1,
                                                     dilationX: 1, groups: 1, paddingTop: 0, paddingBottom: 0,
                                                     paddingLeft: 0, paddingRight: 0)
        let cases: [(String, GraphOperationKind, [Tensor64], [Int])] = [
            ("conv3x3", conv3, [random([2, 3, 5, 5]), random([4, 3, 3, 3])], [2, 4, 5, 5]),
            ("conv1x1", conv1, [random([2, 3, 4, 4]), random([2, 3, 1, 1])], [2, 2, 4, 4]),
            ("normalize", .normalize(input: "x", mean: "m", variance: "v", gamma: "g", beta: "b", epsilon: 1e-5),
             [random([2, 3, 2, 2]), random([1, 3, 1, 1]), random([1, 3, 1, 1], positive: true),
              random([1, 3, 1, 1]), random([1, 3, 1, 1])], [2, 3, 2, 2]),
            ("mean", .mean(input: "x", axes: [0, 2, 3]), [random([2, 3, 2, 2])], [1, 3, 1, 1]),
            ("variance", .variance(input: "x", axes: [0, 2, 3]), [random([2, 3, 2, 2])], [1, 3, 1, 1]),
            ("reductionSum", .reductionSum(input: "x", axis: 1), [random([2, 3, 4])], [2, 1, 4]),
            ("matmul", .matrixMultiplication(primary: "a", secondary: "b"), [random([3, 4]), random([4, 5])], [3, 5]),
            ("matmulBatched", .matrixMultiplication(primary: "a", secondary: "b"),
             [random([2, 3, 4]), random([4, 5])], [2, 3, 5]),
            ("add", .addition(primary: "a", secondary: "b"), [random([2, 3, 2, 2]), random([1, 3, 1, 1])], [2, 3, 2, 2]),
            ("subtract", .subtraction(primary: "a", secondary: "b"), [random([2, 3]), random([3])], [2, 3]),
            ("multiply", .multiplication(primary: "a", secondary: "b"),
             [random([2, 3, 2, 2]), random([2, 3, 1, 1])], [2, 3, 2, 2]),
            ("divide", .division(primary: "a", secondary: "b"), [random([2, 3]), random([2, 3], positive: true)], [2, 3]),
            ("tanh", .tanh(input: "x"), [random([3, 5])], [3, 5]),
            ("sigmoid", .sigmoid(input: "x"), [random([3, 5])], [3, 5]),
            ("erf", .erf(input: "x"), [random([3, 5])], [3, 5]),
            ("relu", .reLU(input: "x"), [random([3, 5])], [3, 5]),
            ("leakyRelu", .leakyReLU(input: "x", alpha: 0.1), [random([3, 5])], [3, 5]),
            ("softmax", .softMax(input: "x", axis: 1), [random([3, 6])], [3, 6]),
            ("concat", .concat(inputs: ["a", "b"], dimension: 1), [random([2, 2, 3]), random([2, 4, 3])], [2, 6, 3]),
            ("slice", .slice(input: "x", dimension: 1, start: 1, length: 2), [random([2, 4, 3])], [2, 2, 3]),
            ("reshape", .reshape(input: "x", shape: [6, 2]), [random([3, 4])], [6, 2]),
            ("cast", .cast(input: "x"), [random([3, 4])], [3, 4]),
        ]
        var worst = 0.0
        var failures: [String] = []
        for (name, kind, operands, outShape) in cases {
            do {
                let y = try GraphAudit.forward(kind, operands: operands, outputShape: outShape, outputType: .float32)
                let r = random(y.shape)
                let grads = try GraphAudit.vjp(kind, operands: operands, dY: r)
                func objective(_ xs: [Tensor64]) throws -> Double {
                    // float64 objective (cast's float32 rounding would hide the derivative).
                    let out: Tensor64
                    if case .cast = kind { out = xs[0] } else {
                        out = try GraphAudit.forward(kind, operands: xs, outputShape: outShape, outputType: .float32)
                    }
                    return zip(out.values, r.values).reduce(0) { $0 + $1.0 * $1.1 }
                }
                for (j, grad) in grads.enumerated() {
                    guard let grad else { continue }
                    let d = random(operands[j].shape)
                    let eps = 1e-6
                    var plus = operands, minus = operands
                    for k in 0..<d.count {
                        plus[j].values[k] += eps * d.values[k]
                        minus[j].values[k] -= eps * d.values[k]
                    }
                    let numeric = (try objective(plus) - objective(minus)) / (2 * eps)
                    let analytic = zip(grad.values, d.values).reduce(0) { $0 + $1.0 * $1.1 }
                    let rel = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-12)
                    worst = max(worst, rel)
                    let status = rel < 1e-6 ? "ok" : "FAIL"
                    if rel >= 1e-6 { failures.append("\(name)[\(j)]") }
                    print("\(status) \(name) operand \(j): numeric \(numeric) analytic \(analytic) rel \(rel)")
                }
            } catch {
                failures.append(name)
                print("FAIL \(name): \(error.localizedDescription)")
            }
        }
        // Forward spot checks against hand-computed values.
        do {
            let x = Tensor64(shape: [1, 1, 3, 3], values: (1...9).map(Double.init))
            let w = Tensor64(shape: [1, 1, 3, 3], values: [Double](repeating: 1, count: 9))
            let y = try GraphAudit.forward(conv3, operands: [x, w], outputShape: [1, 1, 3, 3], outputType: .float32)
            // Center = sum 1..9 = 45; corner (0,0) = 1+2+4+5 = 12.
            let ok = y.values[4] == 45 && y.values[0] == 12
            print("\(ok ? "ok" : "FAIL") conv forward spot check: center \(y.values[4]) corner \(y.values[0])")
            if !ok { failures.append("conv forward spot check") }
            let v = GraphAuditReference.reduceVariance(Tensor64(shape: [4], values: [1, 2, 3, 4]), axes: [0])
            let vOK = v.values[0] == 1.25
            print("\(vOK ? "ok" : "FAIL") population variance spot check: \(v.values[0])")
            if !vOK { failures.append("variance spot check") }
            let b = GraphAuditReference.bfloat16Rounded(1.00390625)  // halfway between 1 and 1+2^-7: ties to even
            let bOK = b == 1.0 && GraphAuditReference.bfloat16Rounded(1.01171875) == 1.015625
            print("\(bOK ? "ok" : "FAIL") bfloat16 rounding spot check: \(b)")
            if !bOK { failures.append("bfloat16 rounding") }
        } catch {
            failures.append("spot checks: \(error.localizedDescription)")
        }
        print("worst relative difference \(worst); \(failures.isEmpty ? "all passed" : "FAILED: " + failures.joined(separator: ", "))")
        Darwin.exit(failures.isEmpty ? 0 : ExitCode.selfTestFailed.rawValue)
    }

    private static func progress(_ message: String) {
        FileHandle.standardError.write(Data("[GRAPH-AUDIT] \(message)\n".utf8))
        SessionLogger.shared.log("[GRAPH-AUDIT] \(message)")
    }

    private static func fail(_ message: String, code: ExitCode, loggerStarted: Bool = true) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        if loggerStarted {
            SessionLogger.shared.log("[GRAPH-AUDIT] \(message)")
            SessionLogger.shared.shutdown()
        }
        Darwin.exit(code.rawValue)
    }
}
