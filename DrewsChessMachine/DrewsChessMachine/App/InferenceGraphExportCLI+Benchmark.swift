import Foundation

/// `--coreml-benchmark <config.json> [--json-out <new file>]`: times Core ML
/// exports against the MPSGraph network on the same weights and positions
/// (`CoreMLBenchmark`; config format and fairness rules in
/// `documentation/plans-completed/COREML_BENCHMARK_PLAN.md`). Prints a table per
/// cell to stdout and progress to stderr; `--json-out` writes the full report
/// to a new file (never replacing one).
extension InferenceGraphExportCLI {
    static let benchmarkFlag = "--coreml-benchmark"

    /// The untimed warm-up before each arm's first round in a cell.
    static let benchmarkWarmUpSeconds = 0.5

    struct BenchmarkReport: Encodable, Sendable {
        struct ArmReport: Encodable, Sendable {
            let label: String
            let engine: String
            let model: String
            let coreml: String?
            let computeUnits: String?
            /// MPSGraph: graph build plus weight load of its first network.
            /// Core ML: compile plus load.
            let loadSeconds: Double
            /// The first call at each batch size, ms, keyed by size. Cold for
            /// an MPSGraph arm (its timed network's first run of that shape).
            /// Not cold for a Core ML arm: its parity check has already run
            /// every batch size on the same model.
            var firstCallMs: [String: Double]
            /// Core ML outputs that couldn't be copied by `memcpy` (not float32
            /// or not contiguous) and were converted.
            var outputConversions: Int
            let parity: [CoreMLParityReport]?
            /// Core ML's planned device per operation (`MLComputePlan`).
            let placement: CoreMLBenchmark.DevicePlacement?

            enum CodingKeys: String, CodingKey {
                case label, engine, model, coreml, parity, placement
                case computeUnits = "compute_units"
                case loadSeconds = "load_seconds"
                case firstCallMs = "first_call_ms"
                case outputConversions = "output_conversions"
            }
        }

        let build: String
        let configuration: String
        let hardwareModel: String?
        let chip: String?
        let gpu: String?
        let gpuCores: Int?
        let memoryBytes: UInt64?
        let osVersion: String
        let lowPowerMode: Bool
        let startedAt: String
        let positions: Int
        let batchSizes: [Int]
        let inFlight: [Int]
        let rounds: Int
        let roundSeconds: Double
        let warmUpSeconds: Double
        var arms: [ArmReport]
        var cells: [CoreMLBenchmark.CellResult]

        enum CodingKeys: String, CodingKey {
            case build, configuration, chip, gpu, positions, rounds, arms, cells
            case hardwareModel = "hardware_model"
            case gpuCores = "gpu_cores"
            case memoryBytes = "memory_bytes"
            case osVersion = "os_version"
            case lowPowerMode = "low_power_mode"
            case startedAt = "started_at"
            case batchSizes = "batch_sizes"
            case inFlight = "in_flight"
            case roundSeconds = "round_seconds"
            case warmUpSeconds = "warm_up_seconds"
        }
    }

    static func runBenchmark(_ rawArgs: [String]) throws {
        let (paths, options) = try parse(rawArgs, flag: benchmarkFlag, allowed: ["--json-out"])
        let configURL = fileURL(paths[0])
        let jsonOut = options["--json-out"].map(fileURL)
        if let jsonOut, try FileSafety.existingItem(at: jsonOut) != nil {
            throw UsageError(errorDescription: "--json-out \(jsonOut.path) already exists")
        }
        let config = try CoreMLBenchmark.Config.parse(
            try Data(contentsOf: configURL), relativeTo: configURL.deletingLastPathComponent())
        if BuildInfo.configuration == "Debug", !config.allowDebugBuild {
            throw CoreMLBenchmark.BenchmarkError.debugBuild
        }

        let states = (LichessProbeData.largeSet + LichessProbeData.wideSet).map(\.state)
        let largestBatch = config.batchSizes[config.batchSizes.count - 1]
        let streamCount = config.inFlight[config.inFlight.count - 1]
        let clock = ContinuousClock()

        // Model files, graph builds and encoding are long synchronous work:
        // done here, on the launching thread, never inside the task below.
        struct LoadedModel {
            let file: ModelCheckpointFile
            let ring: CoreMLBenchmark.BoardRing
        }
        var models: [URL: LoadedModel] = [:]
        for arm in config.arms where models[arm.model] == nil {
            progress("loading \(arm.model.lastPathComponent)")
            let (_, file) = try describeAndLoad(arm.model)
            let encoding = file.architecture.inputEncoding
            var encoded: [Float] = []
            encoded.reserveCapacity(states.count * BoardEncoder.tensorLength(for: encoding))
            for state in states { encoded += BoardEncoder.encode(state, encoding: encoding) }
            models[arm.model] = LoadedModel(
                file: file,
                ring: CoreMLBenchmark.BoardRing(encoded: encoded, positionCount: states.count,
                                                maximumBatchSize: largestBatch))
        }
        // Per MPSGraph arm: one network per stream. Per model a Core ML arm
        // checks against: one parity network.
        var armNetworks: [String: (networks: [ChessNetwork], buildSeconds: Double)] = [:]
        var parityNetworks: [URL: ChessNetwork] = [:]
        for arm in config.arms {
            guard let loaded = models[arm.model] else { preconditionFailure("every arm's model was loaded") }
            switch arm.backend {
            case .mpsGraph:
                let start = clock.now
                var networks: [ChessNetwork] = []
                for _ in 0..<streamCount {
                    networks.append(try ChessNetwork(arch: loaded.file.architecture, bnMode: .inference,
                                                     initialization: .overwrittenByLoad))
                }
                armNetworks[arm.label] = (networks, CoreMLBenchmark.seconds(clock.now - start) / Double(streamCount))
            case .coreML:
                if parityNetworks[arm.model] == nil {
                    parityNetworks[arm.model] = try ChessNetwork(arch: loaded.file.architecture, bnMode: .inference,
                                                                 initialization: .overwrittenByLoad)
                }
            }
        }

        let hardware = HardwareInfo.current
        let startedAt = ISO8601DateFormatter().string(from: Date())
        let report = BenchmarkReport(
            build: "\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(BuildInfo.gitDirty ? "*" : "")",
            configuration: BuildInfo.configuration, hardwareModel: hardware.hardwareModel, chip: hardware.cpuBrand,
            gpu: hardware.gpuModel, gpuCores: hardware.gpuCoreCount, memoryBytes: hardware.memoryBytes,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, startedAt: startedAt,
            positions: states.count, batchSizes: config.batchSizes, inFlight: config.inFlight, rounds: config.rounds,
            roundSeconds: config.roundSeconds, warmUpSeconds: benchmarkWarmUpSeconds, arms: [], cells: [])
        let modelsByURL = models.mapValues { ($0.file.networkWeights, $0.ring) }
        let networksByArm = armNetworks
        let parityByModel = parityNetworks

        let finished = try runBlocking {
            try await runBenchmarkLoaded(config: config, states: states, models: modelsByURL,
                                         armNetworks: networksByArm, parityNetworks: parityByModel, report: report)
        }

        FileHandle.standardOutput.write(Data(table(finished).utf8))
        if let jsonOut {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            var json = try encoder.encode(finished)
            json.append(Data("\n".utf8))
            _ = try FileSafety.writeNewFile(json, at: jsonOut)
            progress("wrote \(jsonOut.path)")
        }
    }

    private static func progress(_ text: String) {
        FileHandle.standardError.write(Data("[coreml-benchmark] \(text)\n".utf8))
    }

    private static func runBenchmarkLoaded(
        config: CoreMLBenchmark.Config, states: [GameState],
        models: [URL: (weights: [[Float]], ring: CoreMLBenchmark.BoardRing)],
        armNetworks: [String: (networks: [ChessNetwork], buildSeconds: Double)],
        parityNetworks: [URL: ChessNetwork], report initial: BenchmarkReport
    ) async throws -> BenchmarkReport {
        let clock = ContinuousClock()
        var report = initial
        // Weight loads count toward an MPSGraph arm's load time (stream 0's).
        var mpsLoadSeconds: [String: Double] = [:]
        for arm in config.arms {
            guard let model = models[arm.model] else { preconditionFailure("every arm's model was loaded") }
            if let entry = armNetworks[arm.label] {
                for (index, network) in entry.networks.enumerated() {
                    let start = clock.now
                    try await network.loadWeights(model.weights)
                    if index == 0 { mpsLoadSeconds[arm.label] = entry.buildSeconds + CoreMLBenchmark.seconds(clock.now - start) }
                }
            }
        }
        for (url, network) in parityNetworks {
            guard let model = models[url] else { preconditionFailure("every parity model was loaded") }
            try await network.loadWeights(model.weights)
        }

        let coreMLArms = config.arms.filter { if case .coreML = $0.backend { return true } else { return false } }
        return try await withCoreMLModels(coreMLArms[...], loaded: [:]) { loaded in
            var prepared: [CoreMLBenchmark.PreparedArm] = []
            for arm in config.arms {
                guard let model = models[arm.model] else { preconditionFailure("every arm's model was loaded") }
                let architecture = (armNetworks[arm.label]?.networks[0] ?? parityNetworks[arm.model])?.arch
                guard let architecture else { preconditionFailure("every arm has a network for its model") }
                switch arm.backend {
                case .mpsGraph:
                    guard let entry = armNetworks[arm.label], let loadSeconds = mpsLoadSeconds[arm.label] else {
                        preconditionFailure("every MPSGraph arm has networks")
                    }
                    prepared.append(.mpsGraph(networks: entry.networks, ring: model.ring,
                                              valueClasses: architecture.valueHeadClasses))
                    report.arms.append(BenchmarkReport.ArmReport(
                        label: arm.label, engine: arm.backend.engineName, model: arm.model.path, coreml: nil,
                        computeUnits: nil, loadSeconds: loadSeconds, firstCallMs: [:], outputConversions: 0,
                        parity: nil, placement: nil))
                case .coreML(let package, let units):
                    guard let entry = loaded[arm.label], let parityNetwork = parityNetworks[arm.model] else {
                        preconditionFailure("every Core ML arm was loaded and has a parity network")
                    }
                    let (shared, loadSeconds, placement) = entry
                    let accepted = try CoreMLParity.batchSizes(of: shared.model)
                    for size in config.batchSizes where !accepted.contains(size) {
                        throw CoreMLBenchmark.BenchmarkError.batchSizeNotAccepted(arm: arm.label, size: size,
                                                                                  accepted: accepted)
                    }
                    progress("parity \(arm.label) at \(config.batchSizes.count) batch sizes")
                    let parity = try await CoreMLParity.compare(network: parityNetwork, model: shared.model,
                                                                states: states, computeUnits: units,
                                                                batchSizes: config.batchSizes)
                    for result in parity {
                        let comparable = result.positions - result.positionsWithoutLegalMoves
                            - max(result.nonFiniteCoreMLPositions, result.nonFiniteReferencePositions)
                        let agreement = comparable > 0 ? Double(result.topMoveAgreements) / Double(comparable) : 0
                        guard result.nonFiniteCoreMLPositions == 0, result.nonFiniteReferencePositions == 0 else {
                            throw CoreMLBenchmark.BenchmarkError.parityFailed(
                                arm: arm.label, detail: "non-finite outputs at batch \(result.batchSize)")
                        }
                        guard agreement >= config.minimumTopMoveAgreement else {
                            throw CoreMLBenchmark.BenchmarkError.parityFailed(
                                arm: arm.label, detail: "top-move agreement \(agreement) at batch \(result.batchSize) "
                                    + "is below \(config.minimumTopMoveAgreement)")
                        }
                    }
                    prepared.append(.coreML(model: shared, ring: model.ring, planes: architecture.inputPlanes,
                                            valueClasses: architecture.valueHeadClasses))
                    report.arms.append(BenchmarkReport.ArmReport(
                        label: arm.label, engine: arm.backend.engineName, model: arm.model.path, coreml: package.path,
                        computeUnits: units.rawValue, loadSeconds: loadSeconds, firstCallMs: [:],
                        outputConversions: 0, parity: parity, placement: placement))
                }
            }

            let positionCount = states.count
            var cell = 0
            for batchSize in config.batchSizes {
                for (inFlightIndex, inFlight) in config.inFlight.enumerated() {
                    defer { cell += 1 }
                    for (index, arm) in prepared.enumerated() {
                        let first = try await CoreMLBenchmark.warmUp(arm, label: report.arms[index].label,
                                                                     batchSize: batchSize, inFlight: inFlight,
                                                                     warmUpSeconds: benchmarkWarmUpSeconds)
                        if inFlightIndex == 0 { report.arms[index].firstCallMs[String(batchSize)] = first * 1000 }
                    }
                    var rounds = Array(repeating: [CoreMLBenchmark.RoundMeasurement](), count: prepared.count)
                    for round in 0..<config.rounds {
                        for index in CoreMLBenchmark.armOrder(armCount: prepared.count, cell: cell, round: round,
                                                              roundsPerCell: config.rounds) {
                            let (measurement, conversions) = try await CoreMLBenchmark.measureRound(
                                prepared[index], label: report.arms[index].label, batchSize: batchSize,
                                inFlight: inFlight, round: round,
                                seconds: config.roundSeconds, positionCount: positionCount)
                            rounds[index].append(measurement)
                            report.arms[index].outputConversions += conversions
                        }
                    }
                    for (index, armRounds) in rounds.enumerated() {
                        report.cells.append(CoreMLBenchmark.CellResult(
                            arm: report.arms[index].label, batchSize: batchSize, inFlight: inFlight, rounds: armRounds))
                    }
                    progress("batch \(batchSize) in-flight \(inFlight) done")
                }
            }
            return report
        }
    }

    /// Runs `body` with every Core ML arm's model compiled and loaded (and the
    /// seconds that took), each removed again by `CoreMLParity.withModel`.
    private typealias LoadedCoreMLModel = (CoreMLBenchmark.SharedModel, Double, CoreMLBenchmark.DevicePlacement)

    /// The placement is read after the load is timed, so it isn't counted in
    /// the arm's load seconds.
    private static func withCoreMLModels<T: Sendable>(
        _ arms: ArraySlice<CoreMLBenchmark.Arm>, loaded: [String: LoadedCoreMLModel],
        _ body: ([String: LoadedCoreMLModel]) async throws -> T
    ) async throws -> T {
        guard let arm = arms.first else { return try await body(loaded) }
        guard case .coreML(let package, let units) = arm.backend else {
            preconditionFailure("only Core ML arms are loaded here")
        }
        progress("compiling and loading \(arm.label) (\(units.rawValue))")
        let clock = ContinuousClock()
        let start = clock.now
        return try await CoreMLParity.withCompiledModel(at: package, computeUnits: units) { model, compiled in
            let loadSeconds = CoreMLBenchmark.seconds(clock.now - start)
            let placement = try await CoreMLBenchmark.placement(ofCompiledModelAt: compiled, computeUnits: units)
            var next = loaded
            next[arm.label] = (CoreMLBenchmark.SharedModel(model), loadSeconds, placement)
            return try await withCoreMLModels(arms.dropFirst(), loaded: next, body)
        }
    }

    /// One block per batch size × in-flight cell: each arm's latency, its
    /// throughput, and its speed relative to the config's first arm.
    static func table(_ report: BenchmarkReport) -> String {
        var lines: [String] = []
        lines.append("Core ML vs MPSGraph — build \(report.build) (\(report.configuration)), \(report.chip ?? "unknown chip"), "
            + "\(report.osVersion), \(report.positions) positions, \(report.rounds) × \(report.roundSeconds) s rounds")
        lines.append("")
        // Sized to the longest label so a long one doesn't shift its row's columns.
        let labelWidth = max(22, (report.arms.map(\.label.count).max() ?? 0) + 2)
        lines.append(pad("arm", labelWidth) + pad("load s", 9, right: true) + pad("parity top-1", 14, right: true)
            + "  planned placement (share of estimated cost)")
        for arm in report.arms {
            let agreement = arm.parity.map { reports -> String in
                let worst = reports.map { Double($0.topMoveAgreements)
                    / Double(max(1, $0.positions - $0.positionsWithoutLegalMoves)) }.min() ?? 0
                return String(format: "%.2f%%", worst * 100)
            } ?? "reference"
            let placement = arm.placement.map { placement -> String in
                let total = placement.estimatedCost.values.reduce(0, +)
                guard total > 0 else { return "no costed operations" }
                return placement.estimatedCost.sorted { $0.value > $1.value }
                    .map { "\($0.key) " + String(format: "%.0f%%", $0.value / total * 100) }.joined(separator: ", ")
            } ?? "MPSGraph (GPU)"
            lines.append(pad(arm.label, labelWidth) + pad(String(format: "%.2f", arm.loadSeconds), 9, right: true)
                + pad(agreement, 14, right: true) + "  " + placement)
        }
        let baseline = report.arms.first?.label
        var cellGroups: [String: [CoreMLBenchmark.CellResult]] = [:]
        var groupOrder: [String] = []
        for cell in report.cells {
            let key = "batch \(cell.batchSize), in flight \(cell.inFlight)"
            if cellGroups[key] == nil { groupOrder.append(key) }
            cellGroups[key, default: []].append(cell)
        }
        for key in groupOrder {
            guard let cells = cellGroups[key] else { continue }
            let baselineRate = cells.first { $0.arm == baseline }?.positionsPerSecondMedian
            lines.append("")
            lines.append(key)
            lines.append(pad("arm", labelWidth) + pad("median ms", 11, right: true) + pad("p10 ms", 10, right: true)
                + pad("p90 ms", 10, right: true) + pad("positions/h", 15, right: true)
                + pad("vs \(baseline ?? "-")", 20, right: true) + pad("spread", 8, right: true)
                + pad("thermal", 10, right: true))
            for cell in cells {
                let relative = baselineRate.map { $0 > 0 ? String(format: "%.2f×", cell.positionsPerSecondMedian / $0) : "-" } ?? "-"
                lines.append(pad(cell.arm, labelWidth)
                    + pad(String(format: "%.3f", cell.latencyMedianMs), 11, right: true)
                    + pad(String(format: "%.3f", cell.latencyP10Ms), 10, right: true)
                    + pad(String(format: "%.3f", cell.latencyP90Ms), 10, right: true)
                    + pad(String(format: "%.0f", cell.positionsPerSecondMedian * 3600), 15, right: true)
                    + pad(relative, 20, right: true)
                    + pad(String(format: "%.2f", cell.roundSpread), 8, right: true)
                    + pad(cell.worstThermalState, 10, right: true))
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func pad(_ text: String, _ width: Int, right: Bool = false) -> String {
        guard text.count < width else { return right ? " " + text : text + " " }
        let fill = String(repeating: " ", count: width - text.count)
        return right ? fill + text : text + fill
    }
}
