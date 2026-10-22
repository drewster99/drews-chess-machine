import CoreML
import Foundation

/// Times an exported Core ML model against the app's MPSGraph forward pass on
/// the same weights and positions (`--coreml-benchmark`;
/// `documentation/plans-completed/COREML_BENCHMARK_PLAN.md`).
///
/// What makes the comparison fair, and why each rule exists:
///
/// - **One work boundary for both engines.** A timed call starts with encoded
///   boards in CPU memory and ends with every output (policy logits, value,
///   W/D/L) copied into the caller's own CPU buffers. MPSGraph runs through
///   `evaluateBatched(batchBoardsPointer:…)`, the self-play hot path, whose
///   `consume` copies the outputs out; Core ML gets a preallocated float32
///   `MLMultiArray` filled by `memcpy` and its outputs copied out the same way.
///   Encoding is shared work and stays outside the clock.
/// - **Correct before fast.** Every Core ML arm passes `CoreMLParity` against
///   its own model file's MPSGraph network, at every batch size it is timed at,
///   before it is timed: a fast wrong model measures nothing.
/// - **Warm, and drift spread evenly.** Each cell (batch size × in-flight
///   count) warms every arm untimed, then runs `rounds` rounds whose arm order
///   rotates by one each round, so heat or a background burst lands on every
///   arm alike. No random draws: the order and each stream's starting position
///   follow from the round and stream indices.
/// - **Concurrency reported apart.** In-flight N runs N concurrent streams:
///   N `ChessNetwork`s for MPSGraph (a network serializes its own calls on its
///   queue), N concurrent predictions on one `MLModel` for Core ML (Core ML's
///   async prediction is safe to call concurrently). Each is that engine's own
///   way to overlap work; single-stream numbers are never mixed with them.
/// - **A Debug build is refused** unless the config allows it: Swift debug code
///   would slow only the MPSGraph side's CPU work (Core ML is a system
///   framework, always optimized).
enum CoreMLBenchmark {

    struct ConfigError: LocalizedError, Equatable {
        let errorDescription: String?
        init(_ text: String) { errorDescription = text }
    }

    enum Backend: Sendable, Equatable {
        case mpsGraph
        case coreML(package: URL, computeUnits: CoreMLParity.ComputeUnits)

        var engineName: String {
            switch self {
            case .mpsGraph: return Config.mpsGraphEngine
            case .coreML: return Config.coreMLEngine
            }
        }
    }

    struct Arm: Sendable, Equatable {
        let label: String
        /// The model file: the weights an MPSGraph arm runs, and the parity
        /// reference of a Core ML arm.
        let model: URL
        let backend: Backend
    }

    /// A benchmark run's settings, read from the config file.
    struct Config: Sendable, Equatable {
        static let mpsGraphEngine = "mpsgraph"
        static let coreMLEngine = "coreml"
        static let maximumInFlight = 16
        static let maximumRoundSeconds = 60.0

        /// Ascending, distinct.
        let batchSizes: [Int]
        /// Ascending, distinct.
        let inFlight: [Int]
        let rounds: Int
        let roundSeconds: Double
        /// The least fraction of positions on which a Core ML arm must rank the
        /// same legal move first as its MPSGraph reference, at every batch size.
        let minimumTopMoveAgreement: Double
        let allowDebugBuild: Bool
        let arms: [Arm]

        private static let topLevelKeys: Set<String> = [
            "batch_sizes", "in_flight", "rounds", "round_seconds", "min_top_move_agreement",
            "allow_debug_build", "arms",
        ]
        private static let mpsGraphArmKeys: Set<String> = ["label", "engine", "model"]
        private static let coreMLArmKeys: Set<String> = ["label", "engine", "model", "coreml", "compute_units"]

        /// Parses a config. Relative paths are read against `folder` (the
        /// config file's own folder); `~` is expanded. Every key is required
        /// except `allow_debug_build` (an opt-in, false when absent); an
        /// unknown key is refused, so a misspelled option can't be ignored.
        static func parse(_ data: Data, relativeTo folder: URL) throws -> Config {
            let parsed: Any
            do {
                parsed = try JSONSerialization.jsonObject(with: data)
            } catch {
                throw ConfigError("the config is not JSON: \(error.localizedDescription)")
            }
            guard let object = parsed as? [String: Any] else { throw ConfigError("the config is not a JSON object") }
            try refuseUnknownKeys(of: object, allowed: topLevelKeys, in: "the config")

            let batchSizes = try ascendingDistinctPositiveIntegers(object["batch_sizes"], key: "batch_sizes")
            let inFlight = try ascendingDistinctPositiveIntegers(object["in_flight"], key: "in_flight")
            guard let largestInFlight = inFlight.last, largestInFlight <= maximumInFlight else {
                throw ConfigError("in_flight values must be at most \(maximumInFlight)")
            }
            let rounds = try integer(object["rounds"], key: "rounds")
            guard rounds >= 1 else { throw ConfigError("rounds must be at least 1") }
            let roundSeconds = try number(object["round_seconds"], key: "round_seconds")
            guard roundSeconds > 0, roundSeconds <= maximumRoundSeconds else {
                throw ConfigError("round_seconds must be above 0 and at most \(maximumRoundSeconds)")
            }
            let agreement = try number(object["min_top_move_agreement"], key: "min_top_move_agreement")
            guard (0...1).contains(agreement) else { throw ConfigError("min_top_move_agreement must be within 0...1") }
            var allowDebugBuild = false
            if let raw = object["allow_debug_build"] {
                guard let flag = raw as? NSNumber, CFGetTypeID(flag) == CFBooleanGetTypeID() else {
                    throw ConfigError("allow_debug_build must be true or false")
                }
                allowDebugBuild = flag.boolValue
            }

            guard let rawArms = object["arms"] as? [Any], !rawArms.isEmpty else {
                throw ConfigError("arms must be a non-empty list")
            }
            var arms: [Arm] = []
            var labels: Set<String> = []
            for (index, rawArm) in rawArms.enumerated() {
                let arm = try parseArm(rawArm, index: index, relativeTo: folder)
                guard labels.insert(arm.label).inserted else { throw ConfigError("arm label '\(arm.label)' is used twice") }
                arms.append(arm)
            }
            return Config(batchSizes: batchSizes, inFlight: inFlight, rounds: rounds, roundSeconds: roundSeconds,
                          minimumTopMoveAgreement: agreement, allowDebugBuild: allowDebugBuild, arms: arms)
        }

        private static func parseArm(_ raw: Any, index: Int, relativeTo folder: URL) throws -> Arm {
            let place = "arm \(index)"
            guard let object = raw as? [String: Any] else { throw ConfigError("\(place) is not a JSON object") }
            let label = try string(object["label"], key: "label", in: place)
            let engine = try string(object["engine"], key: "engine", in: place)
            let model = path(try string(object["model"], key: "model", in: place), relativeTo: folder)
            switch engine {
            case mpsGraphEngine:
                try refuseUnknownKeys(of: object, allowed: mpsGraphArmKeys, in: "\(place) (\(engine))")
                return Arm(label: label, model: model, backend: .mpsGraph)
            case coreMLEngine:
                try refuseUnknownKeys(of: object, allowed: coreMLArmKeys, in: "\(place) (\(engine))")
                let package = path(try string(object["coreml"], key: "coreml", in: place), relativeTo: folder)
                let rawUnits = try string(object["compute_units"], key: "compute_units", in: place)
                guard let units = CoreMLParity.ComputeUnits(rawValue: rawUnits) else {
                    throw ConfigError("\(place): compute_units must be one of "
                        + CoreMLParity.ComputeUnits.allCases.map(\.rawValue).joined(separator: ", "))
                }
                return Arm(label: label, model: model, backend: .coreML(package: package, computeUnits: units))
            default:
                throw ConfigError("\(place): engine must be \(mpsGraphEngine) or \(coreMLEngine), not '\(engine)'")
            }
        }

        private static func refuseUnknownKeys(of object: [String: Any], allowed: Set<String>, in place: String) throws {
            let unknown = Set(object.keys).subtracting(allowed).sorted()
            guard unknown.isEmpty else {
                throw ConfigError("\(place) has unknown key(s): \(unknown.joined(separator: ", "))")
            }
        }

        private static func path(_ raw: String, relativeTo folder: URL) -> URL {
            let expanded = (raw as NSString).expandingTildeInPath
            return expanded.hasPrefix("/") ? URL(fileURLWithPath: expanded)
                : folder.appendingPathComponent(expanded).standardizedFileURL
        }

        private static func string(_ raw: Any?, key: String, in place: String) throws -> String {
            guard let text = raw as? String, !text.isEmpty else {
                throw ConfigError("\(place): \(key) must be a non-empty string")
            }
            return text
        }

        /// A JSON number that isn't a boolean (`JSONSerialization` hands both
        /// back as `NSNumber`).
        private static func number(_ raw: Any?, key: String) throws -> Double {
            guard let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
                  value.doubleValue.isFinite else {
                throw ConfigError("\(key) must be a number")
            }
            return value.doubleValue
        }

        private static func integer(_ raw: Any?, key: String) throws -> Int {
            let value = try number(raw, key: key)
            guard value == value.rounded(), abs(value) < 1e15 else { throw ConfigError("\(key) must be a whole number") }
            return Int(value)
        }

        private static func ascendingDistinctPositiveIntegers(_ raw: Any?, key: String) throws -> [Int] {
            guard let list = raw as? [Any], !list.isEmpty else { throw ConfigError("\(key) must be a non-empty list") }
            let values = try list.map { try integer($0, key: key) }
            guard values.allSatisfy({ $0 >= 1 }) else { throw ConfigError("\(key) values must be at least 1") }
            guard Set(values).count == values.count else { throw ConfigError("\(key) lists a value twice") }
            return values.sorted()
        }
    }

    // MARK: - Device placement

    /// Where Core ML plans to run a model's operations under a compute-units
    /// setting (`MLComputePlan`). The setting only says which devices Core ML
    /// may use; this says which it prefers for each operation, so a "Neural
    /// Engine" arm that in fact runs on the CPU is visible in the report.
    /// Operations with no device (constants) are counted apart. The plan is
    /// Core ML's preference for the model's default shape; it is not a trace
    /// of where each timed prediction ran.
    struct DevicePlacement: Encodable, Sendable, Equatable {
        /// Preferred device → operation count.
        var operations: [String: Int] = [:]
        /// Preferred device → summed estimated cost (Core ML's relative
        /// weights, summing to about 1 over the costed operations).
        var estimatedCost: [String: Double] = [:]
        var operationsWithoutDevice = 0

        enum CodingKeys: String, CodingKey {
            case operations
            case estimatedCost = "estimated_cost"
            case operationsWithoutDevice = "operations_without_device"
        }
    }

    static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "neural_engine"
        @unknown default: return "unknown"
        }
    }

    static func placement(ofCompiledModelAt url: URL,
                          computeUnits: CoreMLParity.ComputeUnits) async throws -> DevicePlacement {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits.mlComputeUnits
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        var placement = DevicePlacement()
        guard case .program(let program) = plan.modelStructure else { return placement }
        var pending = program.functions.values.map(\.block)
        while let block = pending.popLast() {
            for operation in block.operations {
                pending += operation.blocks
                guard let usage = plan.deviceUsage(for: operation) else {
                    placement.operationsWithoutDevice += 1
                    continue
                }
                let device = deviceName(usage.preferred)
                placement.operations[device, default: 0] += 1
                if let cost = plan.estimatedCost(of: operation) {
                    placement.estimatedCost[device, default: 0] += cost.weight
                }
            }
        }
        return placement
    }

    // MARK: - Statistics

    /// The `fraction` quantile of `sorted` (ascending, non-empty), linearly
    /// interpolated between the two nearest ranks.
    static func quantile(_ sorted: [Double], _ fraction: Double) -> Double {
        precondition(!sorted.isEmpty, "a quantile needs at least one value")
        precondition((0...1).contains(fraction), "a quantile fraction is within 0...1")
        let position = fraction * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(lower + 1, sorted.count - 1)
        let weight = position - Double(lower)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * weight
    }

    /// The arm indices in the order round `round` runs them: the configured
    /// order rotated left by `round`, so each arm leads in turn.
    static func armOrder(armCount: Int, round: Int) -> [Int] {
        precondition(armCount >= 1 && round >= 0, "an order needs arms and a round index")
        return (0..<armCount).map { ($0 + round) % armCount }
    }

    /// One timed round of one arm in one cell.
    struct RoundMeasurement: Sendable, Equatable {
        /// Every timed call's latency, seconds.
        let latencies: [Double]
        let positions: Int
        /// From the round's start to its last call's end.
        let wallSeconds: Double
        let thermalState: ProcessInfo.ThermalState

        var positionsPerSecond: Double { wallSeconds > 0 ? Double(positions) / wallSeconds : 0 }
    }

    /// One arm's numbers in one cell, over all its rounds.
    struct CellResult: Encodable, Sendable, Equatable {
        let arm: String
        let batchSize: Int
        let inFlight: Int
        let calls: Int
        let rounds: Int
        let latencyMedianMs: Double
        let latencyP10Ms: Double
        let latencyP90Ms: Double
        let latencyMinMs: Double
        /// Median, min and max of the rounds' positions per second.
        let positionsPerSecondMedian: Double
        let positionsPerSecondMin: Double
        let positionsPerSecondMax: Double
        /// Max over min round throughput: how steady the cell was.
        let roundSpread: Double
        let worstThermalState: String

        enum CodingKeys: String, CodingKey {
            case arm, calls, rounds
            case batchSize = "batch_size"
            case inFlight = "in_flight"
            case latencyMedianMs = "latency_median_ms"
            case latencyP10Ms = "latency_p10_ms"
            case latencyP90Ms = "latency_p90_ms"
            case latencyMinMs = "latency_min_ms"
            case positionsPerSecondMedian = "positions_per_second_median"
            case positionsPerSecondMin = "positions_per_second_min"
            case positionsPerSecondMax = "positions_per_second_max"
            case roundSpread = "round_spread"
            case worstThermalState = "worst_thermal_state"
        }

        /// The cell's summary; `rounds` must be non-empty and each must have
        /// timed at least one call.
        init(arm: String, batchSize: Int, inFlight: Int, rounds: [RoundMeasurement]) {
            precondition(!rounds.isEmpty && rounds.allSatisfy { !$0.latencies.isEmpty },
                         "a cell needs rounds that each timed a call")
            let latencies = rounds.flatMap(\.latencies).sorted()
            let throughputs = rounds.map(\.positionsPerSecond).sorted()
            self.arm = arm
            self.batchSize = batchSize
            self.inFlight = inFlight
            calls = latencies.count
            self.rounds = rounds.count
            latencyMedianMs = CoreMLBenchmark.quantile(latencies, 0.5) * 1000
            latencyP10Ms = CoreMLBenchmark.quantile(latencies, 0.1) * 1000
            latencyP90Ms = CoreMLBenchmark.quantile(latencies, 0.9) * 1000
            latencyMinMs = latencies[0] * 1000
            positionsPerSecondMedian = CoreMLBenchmark.quantile(throughputs, 0.5)
            positionsPerSecondMin = throughputs[0]
            positionsPerSecondMax = throughputs[throughputs.count - 1]
            roundSpread = throughputs[0] > 0 ? throughputs[throughputs.count - 1] / throughputs[0] : .infinity
            let worst = rounds.map(\.thermalState).max { $0.rawValue < $1.rawValue } ?? .nominal
            worstThermalState = CoreMLBenchmark.name(of: worst)
        }
    }

    static func name(of state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown(\(state.rawValue))"
        }
    }

    // MARK: - Inputs and outputs

    /// The encoded positions every stream reads, laid out so any batch starting
    /// at a position index below `positionCount` is contiguous: the first
    /// `maximumBatchSize` positions are repeated after the last. Written once
    /// at init, then only read (concurrently, by every stream).
    final class BoardRing: @unchecked Sendable {
        let positionCount: Int
        let floatsPerPosition: Int
        private let storage: UnsafeMutableBufferPointer<Float>

        init(encoded: [Float], positionCount: Int, maximumBatchSize: Int) {
            precondition(positionCount >= 1 && encoded.count % positionCount == 0, "whole encoded positions")
            self.positionCount = positionCount
            floatsPerPosition = encoded.count / positionCount
            let ringPositions = positionCount + maximumBatchSize
            storage = .allocate(capacity: ringPositions * floatsPerPosition)
            for start in stride(from: 0, to: ringPositions, by: positionCount) {
                let count = min(positionCount, ringPositions - start)
                encoded.withUnsafeBufferPointer { source in
                    guard let base = source.baseAddress, let destination = storage.baseAddress else { return }
                    (destination + start * floatsPerPosition)
                        .update(from: base, count: count * floatsPerPosition)
                }
            }
        }

        deinit { storage.deallocate() }

        /// The batch starting at position `index` (below `positionCount`).
        func batch(at index: Int) -> UnsafePointer<Float> {
            precondition((0..<positionCount).contains(index), "a batch starts inside the ring")
            guard let base = storage.baseAddress else { preconditionFailure("an allocated ring has a base address") }
            return UnsafePointer(base + index * floatsPerPosition)
        }
    }

    /// One stream's output buffers, sized for the largest batch. Used by one
    /// stream at a time: each call's copy finishes before the call returns.
    final class OutputBuffers: @unchecked Sendable {
        let policy: UnsafeMutableBufferPointer<Float>
        let value: UnsafeMutableBufferPointer<Float>
        let valueProbabilities: UnsafeMutableBufferPointer<Float>

        init(maximumBatchSize: Int, valueClasses: Int) {
            policy = .allocate(capacity: maximumBatchSize * ChessNetwork.policySize)
            value = .allocate(capacity: maximumBatchSize)
            valueProbabilities = .allocate(capacity: maximumBatchSize * valueClasses)
        }

        deinit {
            policy.deallocate()
            value.deallocate()
            valueProbabilities.deallocate()
        }

        func copy(policy source: UnsafeBufferPointer<Float>, value valueSource: UnsafeBufferPointer<Float>,
                  valueProbabilities probabilitiesSource: UnsafeBufferPointer<Float>) {
            _ = UnsafeMutableBufferPointer(rebasing: policy[..<source.count]).initialize(from: source)
            _ = UnsafeMutableBufferPointer(rebasing: value[..<valueSource.count]).initialize(from: valueSource)
            _ = UnsafeMutableBufferPointer(rebasing: valueProbabilities[..<probabilitiesSource.count])
                .initialize(from: probabilitiesSource)
        }
    }

    /// An `MLModel` handed to concurrent streams. Core ML documents its async
    /// prediction as safe to call concurrently on one model, which is the
    /// whole point of the in-flight mode.
    final class SharedModel: @unchecked Sendable {
        let model: MLModel
        init(_ model: MLModel) { self.model = model }
    }

    enum BenchmarkError: LocalizedError {
        case debugBuild
        case batchSizeNotAccepted(arm: String, size: Int, accepted: [Int])
        case parityFailed(arm: String, detail: String)
        case inputNotContiguous(arm: String)

        var errorDescription: String? {
            switch self {
            case .debugBuild:
                return "this is a Debug build: its Swift code would slow only the MPSGraph side. Use a Release build, "
                    + "or set \"allow_debug_build\": true for a run whose timings don't matter"
            case .batchSizeNotAccepted(let arm, let size, let accepted):
                return "arm \(arm): the Core ML model doesn't accept batch size \(size) (it accepts \(accepted))"
            case .parityFailed(let arm, let detail):
                return "arm \(arm) failed the parity check: \(detail)"
            case .inputNotContiguous(let arm):
                return "arm \(arm): Core ML allocated a non-contiguous float32 input array"
            }
        }
    }

    /// An arm ready to time.
    enum PreparedArm: Sendable {
        /// One network per stream, so N streams can overlap.
        case mpsGraph(networks: [ChessNetwork], ring: BoardRing, valueClasses: Int)
        case coreML(model: SharedModel, ring: BoardRing, planes: Int, valueClasses: Int)
    }

    /// What one stream did in one round.
    struct StreamMeasurement: Sendable {
        var latencies: [Double] = []
        var positions = 0
        var end: ContinuousClock.Instant
        var outputConversions = 0
    }

    /// Calls the arm back to back from `startPosition` until `deadline`
    /// (at least `minimumCalls` calls), timing each call.
    static func runStream(_ arm: PreparedArm, stream: Int, batchSize: Int, startPosition: Int,
                          deadline: ContinuousClock.Instant, minimumCalls: Int) async throws -> StreamMeasurement {
        let clock = ContinuousClock()
        var measurement = StreamMeasurement(end: clock.now)
        switch arm {
        case .mpsGraph(let networks, let ring, let valueClasses):
            let network = networks[stream]
            let outputs = OutputBuffers(maximumBatchSize: batchSize, valueClasses: valueClasses)
            let floatCount = batchSize * ring.floatsPerPosition
            var position = startPosition
            while measurement.latencies.count < minimumCalls || clock.now < deadline {
                let start = clock.now
                try await network.evaluateBatched(batchBoardsPointer: ring.batch(at: position), floatCount: floatCount,
                                                  count: batchSize) { policy, value, probabilities in
                    outputs.copy(policy: policy, value: value, valueProbabilities: probabilities)
                }
                measurement.latencies.append(seconds(clock.now - start))
                measurement.positions += batchSize
                position = (position + batchSize) % ring.positionCount
            }
        case .coreML(let shared, let ring, let planes, let valueClasses):
            let input = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: planes), 8, 8],
                                         dataType: .float32)
            guard isContiguous(shape: input.shape, strides: input.strides) else {
                throw BenchmarkError.inputNotContiguous(arm: "stream \(stream)")
            }
            let features = try MLDictionaryFeatureProvider(
                dictionary: [CoreMLParity.inputName: MLFeatureValue(multiArray: input)])
            let outputs = OutputBuffers(maximumBatchSize: batchSize, valueClasses: valueClasses)
            let floatCount = batchSize * ring.floatsPerPosition
            var position = startPosition
            while measurement.latencies.count < minimumCalls || clock.now < deadline {
                let start = clock.now
                let source = ring.batch(at: position)
                input.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, _ in
                    guard let base = destination.baseAddress else { return }
                    base.update(from: source, count: floatCount)
                }
                let result = try await shared.model.prediction(from: features)
                measurement.outputConversions += try copyOutput(
                    result, InferenceGraphOutputName.policy.rawValue, into: outputs.policy,
                    count: batchSize * ChessNetwork.policySize)
                measurement.outputConversions += try copyOutput(
                    result, InferenceGraphOutputName.value.rawValue, into: outputs.value, count: batchSize)
                measurement.outputConversions += try copyOutput(
                    result, InferenceGraphOutputName.valueProbabilities.rawValue, into: outputs.valueProbabilities,
                    count: batchSize * valueClasses)
                measurement.latencies.append(seconds(clock.now - start))
                measurement.positions += batchSize
                position = (position + batchSize) % ring.positionCount
            }
        }
        measurement.end = clock.now
        return measurement
    }

    /// Copies a Core ML output into `destination`: a contiguous float32 array
    /// by `memcpy`, anything else (fp16, padded strides) by conversion.
    /// Returns 1 when it had to convert, so the results show the slower path.
    private static func copyOutput(_ provider: MLFeatureProvider, _ name: String,
                                   into destination: UnsafeMutableBufferPointer<Float>, count: Int) throws -> Int {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw CoreMLParity.ParityError.missingOutput(name)
        }
        guard array.count == count, count <= destination.count, let target = destination.baseAddress else {
            throw CoreMLParity.ParityError.outputSize(name: name, have: array.count, want: count)
        }
        if array.dataType == .float32, isContiguous(shape: array.shape, strides: array.strides) {
            array.withUnsafeBufferPointer(ofType: Float.self) { source in
                guard let base = source.baseAddress else { return }
                target.update(from: base, count: count)
            }
            return 0
        }
        let scalars = MLShapedArray<Float>(converting: array).scalars
        scalars.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            target.update(from: base, count: count)
        }
        return 1
    }

    /// Whether `strides` are the row-major strides of `shape`.
    static func isContiguous(shape: [NSNumber], strides: [NSNumber]) -> Bool {
        guard shape.count == strides.count else { return false }
        var expected = 1
        for (dimension, stride) in zip(shape.reversed(), strides.reversed()) {
            guard dimension.intValue == 1 || stride.intValue == expected else { return false }
            expected *= dimension.intValue
        }
        return true
    }

    static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) * 1e-18
    }

    /// Runs `inFlight` streams of `arm` together for one round. Each stream
    /// starts at its own position, offset by stream and round so consecutive
    /// rounds see different batches.
    static func measureRound(_ arm: PreparedArm, batchSize: Int, inFlight: Int, round: Int, seconds roundSeconds: Double,
                             positionCount: Int) async throws -> (RoundMeasurement, outputConversions: Int) {
        let clock = ContinuousClock()
        let start = clock.now
        let deadline = start + .milliseconds(Int64((roundSeconds * 1000).rounded()))
        let streams = try await withThrowingTaskGroup(of: StreamMeasurement.self) { group in
            for stream in 0..<inFlight {
                let startPosition = (stream * 1009 + round * 331) % positionCount
                group.addTask {
                    try await runStream(arm, stream: stream, batchSize: batchSize, startPosition: startPosition,
                                        deadline: deadline, minimumCalls: 1)
                }
            }
            var all: [StreamMeasurement] = []
            for try await measurement in group { all.append(measurement) }
            return all
        }
        let end = streams.map(\.end).max() ?? start
        return (RoundMeasurement(latencies: streams.flatMap(\.latencies), positions: streams.map(\.positions).reduce(0, +),
                                 wallSeconds: seconds(end - start), thermalState: ProcessInfo.processInfo.thermalState),
                streams.map(\.outputConversions).reduce(0, +))
    }

    /// Untimed calls before a cell's first round: every stream at least twice
    /// and for `warmUpSeconds`, so compilation (MPSGraph's per-shape executable,
    /// Core ML's per-shape plan) and caches are behind it. Returns stream 0's
    /// first call, seconds.
    static func warmUp(_ arm: PreparedArm, batchSize: Int, inFlight: Int, warmUpSeconds: Double) async throws -> Double {
        let clock = ContinuousClock()
        let deadline = clock.now + .milliseconds(Int64((warmUpSeconds * 1000).rounded()))
        let streams = try await withThrowingTaskGroup(of: (Int, StreamMeasurement).self) { group in
            for stream in 0..<inFlight {
                group.addTask {
                    (stream, try await runStream(arm, stream: stream, batchSize: batchSize, startPosition: 0,
                                                 deadline: deadline, minimumCalls: 2))
                }
            }
            var all: [(Int, StreamMeasurement)] = []
            for try await entry in group { all.append(entry) }
            return all
        }
        guard let first = streams.first(where: { $0.0 == 0 })?.1.latencies.first else {
            preconditionFailure("stream 0 made at least two calls")
        }
        return first
    }
}
