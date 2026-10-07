import Foundation

/// One network's weights as the analyzers see them: exported once and shared
/// by every analysis of the same request, with the facts that say whose
/// weights they are and from when.
///
/// **Why one export.** The analyzers used to export the network separately,
/// each at its own moment, while SGD kept running on the trainer — so the
/// weight analysis and the numerics audit of one button press described
/// different weights, and the step they recorded (read from two counters at
/// the press) described neither. A trainer snapshot here comes from
/// `ChessTrainer.exportAnalysisState()`, which reads the weights, the number
/// of SGD steps in them, the fp32 masters and the velocity in one turn of the
/// trainer's queue.
struct AnalyzedNetworkSnapshot: Sendable {
    enum Role: String, Codable, Sendable {
        case champion
        case trainer
    }

    let role: Role
    let modelID: String?
    /// Carries the policy tail precision the weights are built under.
    let architecture: NetworkArchitecture
    /// Graph variable names, in `exportWeights()` order: the trainables, then
    /// the BN running statistics.
    let names: [String]
    let weights: [[Float]]
    /// How many leading entries of `names` / `weights` are trainables; the
    /// rest are BN running statistics, which are not weights.
    let trainableCount: Int
    /// The trainer step these weights were taken at, nil when not known.
    let trainingStep: Int?
    /// Where `trainingStep` comes from, or why it is unknown.
    let trainingStepSource: String
    let takenAt: Date
    /// The architecture's initial trainable values (`AnalysisInitReference`),
    /// checked against `names` / `trainableCount` when the snapshot is taken.
    let initReference: AnalysisInitReference

    /// "champion:<model ID>" / "trainer:<model ID>": the label analyses
    /// carry and their file names embed.
    var modelLabel: String { Self.modelLabel(role: role, modelID: modelID) }

    static func modelLabel(role: Role, modelID: String?) -> String {
        "\(role.rawValue):\(modelID ?? "<no-id>")"
    }

    /// The values of `name`, or nil when the network has no such variable.
    func values(named name: String) -> [Float]? {
        names.firstIndex(of: name).map { weights[$0] }
    }

    func isRunningStatistic(at index: Int) -> Bool {
        index >= trainableCount
    }

    /// Where variable `index` started, from the init reference: the one
    /// lookup every analyzer uses.
    func start(ofVariableAt index: Int) throws -> AnalysisInitReference.Start {
        if isRunningStatistic(at: index) { return .runningStatistic }
        let name = names[index]
        guard let initial = initReference.initialValues[name], initial.count == weights[index].count else {
            throw AnalysisInitReferenceError.noInitialValues(name)
        }
        return .trainable(initial: initial, exact: initReference.exactNames.contains(name))
    }
}

/// The optimizer state a numerics audit compares a snapshot's working weights
/// with, taken in the same cut as those weights.
enum AnalysisOptimizerState: Sendable {
    /// An inference network (the champion): no masters, no velocity.
    case inferenceNetwork
    /// The trainer's fp32 masters (nil when it computes in fp32 and keeps
    /// none), parallel to the snapshot's names, and its velocity, one tensor
    /// per trainable — read in the same trainer-queue turn as the weights.
    case trainer(masters: [[Float]]?, velocity: [[Float]])
}

/// The initial values of an architecture's trainable tensors, made by the
/// network builder itself — a plain `ChessNetwork` built under an init seed,
/// the build every fresh model's trainables come from
/// (`ChessMPSNetwork(.randomWeights)`, a fresh `ChessTrainer`,
/// `--new-model`) — never by a second copy of its rules. Every "init" figure
/// an analyzer reports (initial L2 norms, ratios to init, drift, the starting
/// value-head bias, the audit's head-bias init mean) is read from here, so a
/// model built with another final init, draw prior, branch or skip init, or
/// SE bias init is compared with its own starting point.
///
/// **Trainables only.** BN running statistics have no seed-determined start:
/// a GUI build calibrates them on a warmup forward pass
/// (`ChessMPSNetwork.calibratedFreshNetwork`), a GPU result not
/// bit-reproducible across chips, OS builds or policy-tail precisions, while
/// a fresh CLI trainer and a graft start them at mean 0 / variance 1, and
/// nothing a model records says which start it had. So the reference holds
/// none for them and the analyzers report no init figure for them.
///
/// **Exactness.** With the model's recorded seed under the current scheme
/// (`ModelInitRecord`, `WeightInitScheme.current`) every trainable's initial
/// values are exact. Without it the reference is built under
/// `fallbackSeed`: a tensor the builder did not draw from the seed (biases,
/// BN γ/β, ReZero α, the draw prior, a zero-initialized head, an
/// identity-like projection — `RandomTensorRole.isDrawnFromInitSeed`) is still
/// exact, as the current builder makes it for the model's architecture (format
/// versions resolve older files to the init they were built with); a drawn
/// tensor gets a draw of the same distribution, flagged inexact, and no drift.
struct AnalysisInitReference: Sendable {
    /// Why a reference was built without the model's own seed.
    enum FallbackReason: Sendable, Hashable, CustomStringConvertible {
        case seedNotRecorded
        case otherScheme(String)

        var description: String {
            switch self {
            case .seedNotRecorded: return "the model's init seed is not recorded"
            case .otherScheme(let scheme): return "the model's init scheme \(scheme) is not \(WeightInitScheme.current)"
            }
        }
    }

    enum Basis: Sendable, Hashable {
        /// Built under the model's own init seed: every trainable exact.
        case modelInitSeed(UInt64)
        /// The model's seed is unknown: built under `fallbackSeed`, exact
        /// only for tensors not drawn from the seed.
        case fallbackSeed(FallbackReason)
    }

    /// Where a variable started, as the analyzers see it.
    enum Start: Sendable {
        case trainable(initial: [Float], exact: Bool)
        /// A BN running statistic, whose start depends on how the model was
        /// built (see the type doc), so it has no init figures.
        case runningStatistic
    }

    /// One seeded build of an architecture: what `AnalysisInitReferenceCache`
    /// shares between analyses of the same request.
    struct Build: Sendable {
        let architecture: NetworkArchitecture
        let seed: UInt64
        /// Every graph variable name in `exportWeights()` order: trainables,
        /// then BN running statistics. Unique (checked).
        let variableNames: [String]
        let trainableCount: Int
        /// Each trainable's initial values, by graph variable name.
        let initialValues: [String: [Float]]
        /// The trainables the builder drew from the seed.
        let drawnNames: Set<String>
    }

    /// The seed a reference is built under when the model's is unknown.
    static let fallbackSeed: UInt64 = 1

    let architecture: NetworkArchitecture
    let basis: Basis
    let variableNames: [String]
    let trainableCount: Int
    let initialValues: [String: [Float]]
    /// Trainables whose initial values are known exactly.
    let exactNames: Set<String>

    init(build: Build, basis: Basis) {
        architecture = build.architecture
        self.basis = basis
        variableNames = build.variableNames
        trainableCount = build.trainableCount
        initialValues = build.initialValues
        switch basis {
        case .modelInitSeed: exactNames = Set(build.initialValues.keys)
        case .fallbackSeed: exactNames = Set(build.initialValues.keys).subtracting(build.drawnNames)
        }
    }

    static func basis(for initialization: ModelInitRecord?) -> Basis {
        guard let initialization else { return .fallbackSeed(.seedNotRecorded) }
        guard initialization.scheme == WeightInitScheme.current else {
            return .fallbackSeed(.otherScheme(initialization.scheme))
        }
        return .modelInitSeed(initialization.initSeed)
    }

    static func buildSeed(for basis: Basis) -> UInt64 {
        switch basis {
        case .modelInitSeed(let seed): return seed
        case .fallbackSeed: return fallbackSeed
        }
    }

    /// A description for the exports.
    var basisDescription: String {
        let runningStatistics = "BN running statistics have no init figures (their start depends on how the model was built)"
        switch basis {
        case .modelInitSeed(let seed):
            return "the model's own init seed \(seed) (\(WeightInitScheme.current)): exact initial values for every trainable; \(runningStatistics)"
        case .fallbackSeed(let reason):
            return "\(reason); trainables not drawn from the seed exact, drawn ones from a draw of the same distribution (no drift); \(runningStatistics)"
        }
    }

    /// Throws unless `names` / `trainableCount` (an analyzed network's, in
    /// export order) are this architecture's.
    func requireMatches(variableNames names: [String], trainableCount count: Int) throws {
        guard names == variableNames, count == trainableCount else {
            throw AnalysisInitReferenceError.variablesDiffer(
                analyzed: names.count, reference: variableNames.count,
                analyzedTrainables: count, referenceTrainables: trainableCount)
        }
    }

    /// The mean of trainable `name`'s initial values, which must be exact:
    /// the audit's head-bias yardstick.
    func exactInitialMean(of name: String) throws -> Double {
        guard let values = initialValues[name], !values.isEmpty else {
            throw AnalysisInitReferenceError.noInitialValues(name)
        }
        guard exactNames.contains(name) else { throw AnalysisInitReferenceError.initialValuesNotExact(name) }
        return values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
    }

    /// One plain seeded build. Graph construction is long synchronous work,
    /// so it runs on a GCD queue; the export runs on the network's own queue.
    static func makeBuild(architecture: NetworkArchitecture, seed: UInt64) async throws -> Build {
        let network: ChessNetwork = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(with: Swift.Result(catching: {
                    try ChessNetwork(arch: architecture, initialization: .seeded(initSeed: seed))
                }))
            }
        }
        let trainableNames = network.trainableVariables.map { $0.operation.name }
        let variableNames = trainableNames + network.bnRunningStatsVariables.map { $0.operation.name }
        var seen = Set<String>()
        for name in variableNames where !seen.insert(name).inserted {
            throw AnalysisInitReferenceError.duplicateVariableName(name)
        }
        // Plan entries pair with variables by position (`validateAgainstPlan`
        // checked it at build); `randomTensorRoles` is keyed by plan name.
        let plan = network.arch.weightTensorPlan()
        guard plan.count == variableNames.count else {
            throw AnalysisInitReferenceError.planDiffers(plan: plan.count, variables: variableNames.count)
        }
        var drawnNames = Set<String>()
        for (index, name) in trainableNames.enumerated()
        where network.randomTensorRoles[plan[index].name]?.isDrawnFromInitSeed == true {
            drawnNames.insert(name)
        }
        let weights = try await network.exportWeights()
        guard weights.count == variableNames.count else {
            throw AnalysisInitReferenceError.exportCount(expected: variableNames.count, got: weights.count)
        }
        var initialValues: [String: [Float]] = [:]
        initialValues.reserveCapacity(trainableNames.count)
        for (name, values) in zip(trainableNames, weights) { initialValues[name] = values }
        return Build(architecture: architecture, seed: seed, variableNames: variableNames,
                     trainableCount: trainableNames.count, initialValues: initialValues, drawnNames: drawnNames)
    }
}

/// Init-reference builds shared by every analysis of one request (Run All's
/// champion and trainer, a CLI folder of files from one run): one build per
/// (architecture, seed), in flight or finished. Make one per request; it
/// holds every build it made until it is released.
final class AnalysisInitReferenceCache: Sendable {
    private struct Key: Hashable, Sendable {
        let architecture: NetworkArchitecture
        let seed: UInt64
    }

    private let builds = SyncBox<[Key: Task<AnalysisInitReference.Build, Error>]>([:])

    func reference(architecture: NetworkArchitecture, initialization: ModelInitRecord?) async throws -> AnalysisInitReference {
        let basis = AnalysisInitReference.basis(for: initialization)
        let key = Key(architecture: architecture, seed: AnalysisInitReference.buildSeed(for: basis))
        // Creating the Task only schedules it; nothing in the locked section
        // suspends, and the Task body never touches the box.
        let build = builds.mutate { builds in
            if let existing = builds[key] { return existing }
            let started = Task { try await AnalysisInitReference.makeBuild(architecture: key.architecture, seed: key.seed) }
            builds[key] = started
            return started
        }
        return AnalysisInitReference(build: try await build.value, basis: basis)
    }
}

enum AnalysisInitReferenceError: LocalizedError {
    case variablesDiffer(analyzed: Int, reference: Int, analyzedTrainables: Int, referenceTrainables: Int)
    case duplicateVariableName(String)
    case planDiffers(plan: Int, variables: Int)
    case exportCount(expected: Int, got: Int)
    case noInitialValues(String)
    case initialValuesNotExact(String)

    var errorDescription: String? {
        switch self {
        case .variablesDiffer(let analyzed, let reference, let analyzedTrainables, let referenceTrainables):
            return "The init reference built from the architecture has \(reference) variables (\(referenceTrainables) trainable) where the analyzed network has \(analyzed) (\(analyzedTrainables) trainable), or their names differ"
        case .duplicateVariableName(let name):
            return "The architecture's build has two variables named \(name)"
        case .planDiffers(let plan, let variables):
            return "The architecture's tensor plan has \(plan) entries for \(variables) graph variables"
        case .exportCount(let expected, let got):
            return "The init reference's export returned \(got) tensors for \(expected) variables"
        case .noInitialValues(let name):
            return "The init reference has no initial values of the analyzed size for \(name)"
        case .initialValuesNotExact(let name):
            return "The init reference does not know \(name)'s initial values exactly"
        }
    }
}
