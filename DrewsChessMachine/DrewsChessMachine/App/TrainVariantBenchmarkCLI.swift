import Darwin
import Foundation

/// `--train-variant-benchmark [--architecture <preset|file> | --start-model <file>]
/// [--batch <n>] [--seconds-per-variant <s>] [--warmup <n>] [--json-out <new file>]`:
/// what each kind of training step (`TrainingStepType`: plain, diagnostics,
/// probe) costs in step time, and what the one compiled training executable
/// costs in device memory, then exit (no GUI).
///
/// Why it exists (owner request 2026-11-22): a training run mixes the step
/// types — the diagnostic readback on step-line steps
/// (`ChessTrainer.isDiagnosticsStep`), the KL probe's forward on probe steps
/// (`ChessTrainer.isKLProbeStep`) — so its log can't say what each costs.
/// It first measured the four executables training used to compile per batch
/// size (plain / diagnostics / KL stash / both: ~4.1–4.6 GB of GPU heap each,
/// experiments/20261122-memory-footprint); training now compiles one that
/// always computes the diagnostics and the stash
/// (`ChessTrainer.trainingExecutables`), and this measures that system. The
/// output keeps the old format (one row per step type where there was one
/// per variant) so the two runs compare side by side.
///
/// **Production code only.** One throwaway `ChessTrainer` built in this
/// process (never a live trainer; no saved state, setting or file is read for
/// training or written, apart from the optional `--json-out` file, which must
/// be new) runs the random-data step `--sweep` uses
/// (`trainStep(batchSize:includeDiagnostics:)` → `runPreparedStep`, the step
/// every path runs). The step type is selected through the step's own inputs:
/// `includeDiagnostics` (the one parameter added for this benchmark; the
/// random-data step otherwise always reads diagnostics back) and
/// `klProbeInterval` (1 = every step probes, 0 = never), the knob training
/// itself uses. Hyperparameters are the declared defaults, as in `--sweep`;
/// none of them change the compiled graph's shape.
///
/// **Per step type, one after another** (plain, diagnostics, probe; owner spec
/// 2026-11-22: solo and continuous, not interleaved):
///   1. reload the starting weights with zero velocity, so every type starts
///      from the same weights (random-data SGD drifts them);
///   2. compile the executable (`compileSyntheticTrainingExecutable`),
///      reading `MTLDevice.currentAllocatedSize` before and after — a real
///      compile for the first type, a cache hit (`*`) for the others, since
///      every type runs the same executable;
///   3. one first step, then the allocation again (the executable's heap
///      appears at its first run, so only the first type's row shows it);
///   4. the remaining `--warmup` steps, untimed;
///   5. steps back to back for `--seconds-per-variant` wall seconds (the last
///      step may overrun; the window is the measured wall time), recording
///      each step's `TrainStepTiming`.
///
/// **What the numbers mean.** `totalMs` is the whole step (data prep, the
/// executable, the dropout-RNG advance, readback and, on probe steps, the KL
/// probe's forward pass); `gpuRunMs` is the training executable plus, except
/// on probe steps, the dropout-RNG advance (a probe step advances after its
/// probe, outside `gpuRunMs`); `klProbeMs` the probe's extra pass. The data is
/// random, so the step lacks the real-data path's value-baseline forward pass;
/// the differences between step types are what this measures. Device
/// allocation is a point reading of live allocations (`currentAllocatedSize`),
/// not a peak: MPSGraph may allocate and free transient buffers inside a step.
/// Allocation deltas are cumulative across rows. Step types run in a fixed
/// order, so thermal drift lands on the later ones; each record carries the
/// thermal state at the window's start and end.
///
/// Output: a fixed-width table on stdout, one `[BENCH]` line per step type in
/// the session log, and with `--json-out` one JSON document. Build
/// configuration matters for CPU-side time (data prep, readback); the
/// executable's GPU time does not depend on it.
enum TrainVariantBenchmarkCLI {

    static let modeFlag = "--train-variant-benchmark"
    static let architectureFlag = "--architecture"
    static let startModelFlag = "--start-model"
    static let batchFlag = "--batch"
    static let secondsFlag = "--seconds-per-variant"
    static let warmupFlag = "--warmup"
    static let jsonOutFlag = "--json-out"

    static let defaultBatchSize = 4096
    static let defaultSecondsPerVariant = 120.0
    static let defaultWarmupSteps = 5

    private enum ExitCode: Int32 {
        case usage = 130
        case modelUnusable = 131
        case outputUnusable = 132
        case benchmarkFailed = 133
        case outputWriteFailed = 134
    }

    /// Where the benchmarked weights come from.
    enum WeightsSource: Equatable {
        /// The current preset (`NetworkArchitecture.current`), fresh weights.
        case currentPreset
        /// `--architecture`: a preset name or architecture file, fresh weights.
        case architecture(String)
        /// `--start-model`: a model file's architecture and base weights.
        case startModel(String)
    }

    struct Options: Equatable {
        var source: WeightsSource = .currentPreset
        var batchSize = TrainVariantBenchmarkCLI.defaultBatchSize
        var secondsPerVariant = TrainVariantBenchmarkCLI.defaultSecondsPerVariant
        var warmupSteps = TrainVariantBenchmarkCLI.defaultWarmupSteps
        var jsonOutPath: String?
    }

    struct UsageError: Error, LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// One step type's measurements.
    struct VariantResult {
        let variant: TrainingStepType
        let compilation: ChessTrainer.TrainingExecutableCompilation
        let firstStepMs: Double
        let gpuAllocatedBytesAfterFirstStep: UInt64
        let warmupSteps: Int
        let windowSeconds: Double
        let totalMs: TimingDistribution
        let gpuRunMs: TimingDistribution
        let readbackMs: TimingDistribution
        let dataPrepMs: TimingDistribution
        /// Nil on step types that don't run the KL probe.
        let klProbeMs: TimingDistribution?
        let gpuAllocatedBytesAfterWindow: UInt64
        /// `phys_footprint` after the window; nil when `task_info` refused.
        let footprintBytesAfterWindow: UInt64?
        let thermalAtStart: ProcessInfo.ThermalState
        let thermalAtEnd: ProcessInfo.ThermalState

        var steps: Int { totalMs.count }
        var stepsPerSecond: Double { windowSeconds > 0 ? Double(steps) / windowSeconds : 0 }
    }

    // MARK: - Entry

    /// When `rawArgs` hold `--train-variant-benchmark`, runs it and exits
    /// (never returns); otherwise returns without doing anything.
    @MainActor
    static func handleIfRequested(rawArgs: [String]) {
        guard rawArgs.contains(modeFlag) else { return }
        let options: Options
        do {
            options = try parseOptions(rawArgs)
        } catch {
            fail("error: \(error.localizedDescription)", code: .usage, loggerStarted: false)
        }
        runAndExit(options)
    }

    static func parseOptions(_ rawArgs: [String]) throws -> Options {
        let valueFlags: Set<String> = [architectureFlag, startModelFlag, batchFlag, secondsFlag, warmupFlag, jsonOutFlag]
        var options = Options()
        var seen = Set<String>()
        var index = 0
        while index < rawArgs.count {
            let argument = rawArgs[index]
            if argument == modeFlag {
                index += 1
                continue
            }
            guard valueFlags.contains(argument) else {
                throw UsageError(message: "\(modeFlag) does not accept '\(argument)'")
            }
            guard seen.insert(argument).inserted else {
                throw UsageError(message: "\(argument) given more than once")
            }
            guard index + 1 < rawArgs.count, !rawArgs[index + 1].hasPrefix("--") else {
                throw UsageError(message: "\(argument) requires a value")
            }
            let value = rawArgs[index + 1]
            switch argument {
            case architectureFlag:
                options.source = .architecture(value)
            case startModelFlag:
                options.source = .startModel(value)
            case batchFlag:
                guard let n = Int(value), n > 0 else {
                    throw UsageError(message: "\(batchFlag) requires a positive integer")
                }
                options.batchSize = n
            case secondsFlag:
                guard let s = Double(value), s.isFinite, s > 0 else {
                    throw UsageError(message: "\(secondsFlag) requires a positive number of seconds")
                }
                options.secondsPerVariant = s
            case warmupFlag:
                // At least one: the first step is the "after first run"
                // allocation reading.
                guard let n = Int(value), n >= 1 else {
                    throw UsageError(message: "\(warmupFlag) requires an integer >= 1")
                }
                options.warmupSteps = n
            case jsonOutFlag:
                options.jsonOutPath = value
            default:
                preconditionFailure("TrainVariantBenchmarkCLI: unhandled value flag \(argument)")
            }
            index += 2
        }
        if seen.contains(architectureFlag) && seen.contains(startModelFlag) {
            throw UsageError(message: "\(architectureFlag) and \(startModelFlag) cannot be combined; give one")
        }
        return options
    }

    // MARK: - Run

    @MainActor
    private static func runAndExit(_ options: Options) -> Never {
        SessionLogger.shared.start()
        let dirty = BuildInfo.gitDirty ? "*" : ""
        SessionLogger.shared.log(
            "[BENCH] train-variant-benchmark launched build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirty) "
            + "batch=\(options.batchSize) secondsPerVariant=\(options.secondsPerVariant) warmup=\(options.warmupSteps) "
            + "source=\(describe(options.source))"
        )

        // Refuse an existing output before minutes of measurement, not after.
        // `FileSafety.writeNewFile` at the end is still the authority.
        let jsonOutURL = options.jsonOutPath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        if let jsonOutURL {
            var status = stat()
            if lstat(jsonOutURL.path, &status) == 0 {
                fail("error: \(jsonOutFlag) \(jsonOutURL.path) already exists; give a new file", code: .outputUnusable)
            }
        }

        let architecture: NetworkArchitecture
        let startModelBaseWeights: [[Float]]?
        do {
            switch options.source {
            case .currentPreset:
                architecture = .current
                startModelBaseWeights = nil
            case .architecture(let nameOrPath):
                architecture = try ArchitecturePresetStore.resolveArchitecture(nameOrPath: nameOrPath).named.architecture
                startModelBaseWeights = nil
            case .startModel(let path):
                let file = try CheckpointManager.loadModelFile(
                    at: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
                architecture = file.architecture
                startModelBaseWeights = file.weights
            }
        } catch {
            fail("error: \(describe(options.source)): \(error.localizedDescription)", code: .modelUnusable)
        }
        SessionLogger.shared.logArchitecture(event: "train-variant-benchmark", arch: architecture)
        SessionLogger.shared.log(architecture.policyTailLogLine)

        print("Training step-type benchmark: one training executable (diagnostics and KL stash always computed; random data, training-mode BN)")
        print("  source: \(describe(options.source))   compute: \(architecture.computeDataType.rawValue)"
              + "   batch: \(options.batchSize)   warmup: \(options.warmupSteps)"
              + String(format: "   window: %.0f s per step type", options.secondsPerVariant))
        print("  " + architecture.architectureSummary)
        print("")

        let results: [VariantResult]
        do {
            results = try measureAll(options: options, architecture: architecture,
                                     startModelBaseWeights: startModelBaseWeights)
        } catch {
            fail("error: benchmark failed: \(error)", code: .benchmarkFailed)
        }

        printTable(results)
        for result in results {
            SessionLogger.shared.log(logLine(result, batchSize: options.batchSize))
        }

        if let jsonOutURL {
            do {
                let document = jsonDocument(results, options: options, architecture: architecture)
                let data = try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
                try FileSafety.writeNewFile(data, at: jsonOutURL)
                print("\n  wrote \(jsonOutURL.path)")
            } catch {
                fail("error: writing \(jsonOutFlag) \(jsonOutURL.path): \(error)", code: .outputWriteFailed)
            }
        }
        SessionLogger.shared.log("[BENCH] train-variant-benchmark done stepTypes=\(results.count)")
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    /// Build the trainer, then measure every step type in order. Each GPU
    /// operation is one `runBlocking` (the trainer's work runs on its own
    /// serial queue behind continuations), so nothing long runs in a `Task`.
    private static func measureAll(
        options: Options,
        architecture: NetworkArchitecture,
        startModelBaseWeights: [[Float]]?
    ) throws -> [VariantResult] {
        let trainer = try ChessTrainer(
            // A throughput benchmark: no run, so no run seed.
            dropoutStream: DCMRandom.seededFromSystem(),
            arch: architecture,
            initialization: startModelBaseWeights == nil ? .drawnSeed() : .overwrittenByLoad
        )
        let baseCount = trainer.network.trainableVariables.count + trainer.network.bnRunningStatsVariables.count
        let startingWeights: [[Float]]
        if let startModelBaseWeights {
            // A trainer-state file carries velocity after the base tensors;
            // the base prefix is what a fork loads.
            guard startModelBaseWeights.count >= baseCount else {
                throw UsageError(message: "\(startModelFlag) holds \(startModelBaseWeights.count) tensors; "
                                 + "the architecture needs \(baseCount)")
            }
            startingWeights = Array(startModelBaseWeights.prefix(baseCount))
        } else {
            startingWeights = try runBlocking { try await trainer.network.exportWeights() }
        }
        let device = trainer.network.metalDevice
        let batchSize = options.batchSize

        var results: [VariantResult] = []
        for variant in TrainingStepType.allCases {
            progress("\(variant.label): compiling")
            trainer.klProbeInterval = variant.klProbeInterval
            try runBlocking { try await trainer.loadBaseWeightsResetVelocity(startingWeights) }
            let compilation = try runBlocking {
                try await trainer.compileSyntheticTrainingExecutable(batchSize: batchSize)
            }
            let firstStep = try runBlocking {
                try await trainer.trainStep(batchSize: batchSize, includeDiagnostics: variant.includeDiagnostics)
            }
            let allocatedAfterFirstStep = UInt64(device.currentAllocatedSize)
            for _ in 1..<options.warmupSteps {
                _ = try runBlocking {
                    try await trainer.trainStep(batchSize: batchSize, includeDiagnostics: variant.includeDiagnostics)
                }
            }

            progress(String(format: "%@: timing for %.0f s", variant.label, options.secondsPerVariant))
            let thermalAtStart = ProcessInfo.processInfo.thermalState
            var timings: [TrainStepTiming] = []
            let windowStart = CFAbsoluteTimeGetCurrent()
            while CFAbsoluteTimeGetCurrent() - windowStart < options.secondsPerVariant {
                timings.append(try runBlocking {
                    try await trainer.trainStep(batchSize: batchSize, includeDiagnostics: variant.includeDiagnostics)
                })
            }
            let windowSeconds = CFAbsoluteTimeGetCurrent() - windowStart
            let thermalAtEnd = ProcessInfo.processInfo.thermalState
            // The loop runs at least once (the window is positive), so every
            // distribution below has samples.
            guard let totalMs = TimingDistribution(samples: timings.map(\.totalMs)),
                  let gpuRunMs = TimingDistribution(samples: timings.map(\.gpuRunMs)),
                  let readbackMs = TimingDistribution(samples: timings.map(\.readbackMs)),
                  let dataPrepMs = TimingDistribution(samples: timings.map(\.dataPrepMs))
            else {
                preconditionFailure("TrainVariantBenchmarkCLI: a timed window completed no step")
            }
            let footprint: UInt64?
            switch ChessTrainer.readPhysFootprintBytes() {
            case .success(let bytes): footprint = bytes
            case .failure(let error):
                footprint = nil
                SessionLogger.shared.log("[BENCH] \(variant.label): phys_footprint unreadable: kern_return \(error.kernReturn)")
            }
            let result = VariantResult(
                variant: variant,
                compilation: compilation,
                firstStepMs: firstStep.totalMs,
                gpuAllocatedBytesAfterFirstStep: allocatedAfterFirstStep,
                warmupSteps: options.warmupSteps,
                windowSeconds: windowSeconds,
                totalMs: totalMs,
                gpuRunMs: gpuRunMs,
                readbackMs: readbackMs,
                dataPrepMs: dataPrepMs,
                // A probe step whose probe failed reports 0 (logged
                // `[KL-PROBE] skipped`); those are kept, not hidden.
                klProbeMs: variant.runsKLProbe ? TimingDistribution(samples: timings.map(\.klProbeMs)) : nil,
                gpuAllocatedBytesAfterWindow: UInt64(device.currentAllocatedSize),
                footprintBytesAfterWindow: footprint,
                thermalAtStart: thermalAtStart,
                thermalAtEnd: thermalAtEnd
            )
            progress(String(format: "%@: %d steps, median %.1f ms", variant.label, result.steps, totalMs.median))
            results.append(result)
        }
        return results
    }

    // MARK: - Output

    private static let mebibyte = 1_048_576.0

    private static func mb(_ bytes: UInt64) -> String {
        String(format: "%.0f", Double(bytes) / mebibyte)
    }

    private static func signedMB(_ delta: Int64) -> String {
        String(format: "%+.0f", Double(delta) / mebibyte)
    }

    private static func rightAligned(_ text: String, _ width: Int) -> String {
        String(repeating: " ", count: max(0, width - text.count)) + text
    }

    private static func printTable(_ results: [VariantResult]) {
        let plainMedian = results.first(where: { $0.variant == .plain })?.totalMs.median
        print("Step time (ms; median / p10 / p90 over each step type's window)")
        print("  " + "step type".padding(toLength: 22, withPad: " ", startingAt: 0)
              + rightAligned("steps", 7) + rightAligned("steps/s", 9)
              + rightAligned("total med", 13) + rightAligned("p10", 8) + rightAligned("p90", 8)
              + rightAligned("vs plain", 10)
              + rightAligned("gpu med", 12) + rightAligned("p10", 8) + rightAligned("p90", 8)
              + rightAligned("readback", 11) + rightAligned("kl probe", 10))
        for r in results {
            let ratio = plainMedian.map { String(format: "%7.3fx", r.totalMs.median / $0) } ?? "     n/a"
            let kl = r.klProbeMs.map { String(format: "%8.1f", $0.median) } ?? "       -"
            print(String(
                format: "  %@ %6d %8.3f  %11.1f %7.1f %7.1f  %@  %10.1f %7.1f %7.1f  %9.1f  %@",
                r.variant.label.padding(toLength: 22, withPad: " ", startingAt: 0),
                r.steps, r.stepsPerSecond,
                r.totalMs.median, r.totalMs.p10, r.totalMs.p90, ratio,
                r.gpuRunMs.median, r.gpuRunMs.p10, r.gpuRunMs.p90,
                r.readbackMs.median, kl
            ))
        }
        print("")
        print("Device allocation (MB = 1024^2 B; currentAllocatedSize, cumulative across step types in this order;"
              + " one executable serves every type)")
        print("  " + "step type".padding(toLength: 22, withPad: " ", startingAt: 0)
              + rightAligned("compile ms", 11) + " "
              + rightAligned("before", 8) + rightAligned("+compile", 10) + rightAligned("+first step", 13)
              + rightAligned("after window", 14) + rightAligned("footprint", 11) + "  thermal")
        for r in results {
            let c = r.compilation
            let compileDelta = signedByteDelta(from: c.gpuAllocatedBytesBeforeCompile, to: c.gpuAllocatedBytesAfterCompile)
            let firstDelta = signedByteDelta(from: c.gpuAllocatedBytesAfterCompile, to: r.gpuAllocatedBytesAfterFirstStep)
            let footprint = r.footprintBytesAfterWindow.map(mb) ?? "n/a"
            print("  " + r.variant.label.padding(toLength: 22, withPad: " ", startingAt: 0)
                  + String(format: " %10.0f", c.compileMs) + (c.alreadyCompiled ? "*" : " ")
                  + rightAligned(mb(c.gpuAllocatedBytesBeforeCompile), 8)
                  + rightAligned(signedMB(compileDelta), 10)
                  + rightAligned(signedMB(firstDelta), 13)
                  + rightAligned(mb(r.gpuAllocatedBytesAfterWindow), 14)
                  + rightAligned(footprint, 11)
                  + "  " + thermalName(r.thermalAtStart) + "->" + thermalName(r.thermalAtEnd))
        }
        if results.contains(where: { $0.compilation.alreadyCompiled }) {
            print("  * already compiled (cache hit): no compile was measured")
        }
    }

    private static func logLine(_ r: VariantResult, batchSize: Int) -> String {
        let c = r.compilation
        let kl = r.klProbeMs.map { String(format: "%.1f", $0.median) } ?? "none"
        return String(
            format: "[BENCH] stepType=%@ batch=%d steps=%d window=%.1fs stepsPerSec=%.3f "
                + "totalMs=med:%.1f/p10:%.1f/p90:%.1f gpuRunMs=med:%.1f/p10:%.1f/p90:%.1f "
                + "readbackMs=med:%.1f dataPrepMs=med:%.1f klProbeMs=med:%@ firstStepMs=%.0f "
                + "compileMs=%.0f%@ gpuAlloc=before:%@MB compile:%@MB firstStep:%@MB afterWindow:%@MB "
                + "footprint=%@MB thermal=%@->%@",
            r.variant.label, batchSize, r.steps, r.windowSeconds, r.stepsPerSecond,
            r.totalMs.median, r.totalMs.p10, r.totalMs.p90,
            r.gpuRunMs.median, r.gpuRunMs.p10, r.gpuRunMs.p90,
            r.readbackMs.median, r.dataPrepMs.median, kl, r.firstStepMs,
            c.compileMs, c.alreadyCompiled ? "(cached)" : "",
            mb(c.gpuAllocatedBytesBeforeCompile),
            signedMB(signedByteDelta(from: c.gpuAllocatedBytesBeforeCompile, to: c.gpuAllocatedBytesAfterCompile)),
            signedMB(signedByteDelta(from: c.gpuAllocatedBytesAfterCompile, to: r.gpuAllocatedBytesAfterFirstStep)),
            mb(r.gpuAllocatedBytesAfterWindow),
            r.footprintBytesAfterWindow.map(mb) ?? "n/a",
            thermalName(r.thermalAtStart), thermalName(r.thermalAtEnd)
        )
    }

    private static func jsonDocument(
        _ results: [VariantResult],
        options: Options,
        architecture: NetworkArchitecture
    ) -> [String: Any] {
        func distribution(_ d: TimingDistribution) -> [String: Any] {
            ["count": d.count, "mean": d.mean, "p10": d.p10, "median": d.median, "p90": d.p90,
             "min": d.min, "max": d.max]
        }
        let variants: [[String: Any]] = results.map { r in
            let c = r.compilation
            var record: [String: Any] = [
                "variant": r.variant.label,
                "include_diagnostics": r.variant.includeDiagnostics,
                "runs_kl_probe": r.variant.runsKLProbe,
                "steps": r.steps,
                "window_sec": r.windowSeconds,
                "steps_per_sec": r.stepsPerSecond,
                "warmup_steps": r.warmupSteps,
                "first_step_ms": r.firstStepMs,
                "compile_ms": c.compileMs,
                "already_compiled": c.alreadyCompiled,
                "total_ms": distribution(r.totalMs),
                "gpu_run_ms": distribution(r.gpuRunMs),
                "readback_ms": distribution(r.readbackMs),
                "data_prep_ms": distribution(r.dataPrepMs),
                "gpu_allocated_bytes_before_compile": c.gpuAllocatedBytesBeforeCompile,
                "gpu_allocated_bytes_after_compile": c.gpuAllocatedBytesAfterCompile,
                "gpu_allocated_bytes_after_first_step": r.gpuAllocatedBytesAfterFirstStep,
                "gpu_allocated_bytes_after_window": r.gpuAllocatedBytesAfterWindow,
                "thermal_at_start": thermalName(r.thermalAtStart),
                "thermal_at_end": thermalName(r.thermalAtEnd),
            ]
            record["kl_probe_ms"] = r.klProbeMs.map(distribution) ?? NSNull()
            record["footprint_bytes_after_window"] = r.footprintBytesAfterWindow.map { $0 as Any } ?? NSNull()
            return record
        }
        return [
            "build": BuildInfo.buildNumber,
            "git": BuildInfo.gitHash,
            "git_dirty": BuildInfo.gitDirty,
            "source": describe(options.source),
            "architecture_summary": architecture.architectureSummary,
            "compute_data_type": architecture.computeDataType.rawValue,
            "policy_tail_precision": architecture.policyTailPrecision.rawValue,
            "batch_size": options.batchSize,
            "seconds_per_variant": options.secondsPerVariant,
            "training_executables": "one per batch size (diagnostics and KL stash always computed)",
            "variants": variants,
        ]
    }

    private static func describe(_ source: WeightsSource) -> String {
        switch source {
        case .currentPreset: return "current preset (fresh weights)"
        case .architecture(let value): return "\(architectureFlag) \(value) (fresh weights)"
        case .startModel(let path): return "\(startModelFlag) \(path)"
        }
    }

    private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        MemoryStatusMonitor.Reading.name(of: state)
    }

    private static func progress(_ message: String) {
        FileHandle.standardError.write(Data("[BENCH] \(message)\n".utf8))
    }

    private static func fail(_ message: String, code: ExitCode, loggerStarted: Bool = true) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        if loggerStarted {
            SessionLogger.shared.log("[BENCH] \(message)")
            SessionLogger.shared.shutdown()
        }
        Darwin.exit(code.rawValue)
    }
}
