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
/// `ChessTrainer.exportWeightsWithCompletedSteps()`, which pairs the weights
/// with the number of SGD steps in them.
struct AnalyzedNetworkSnapshot: Sendable {
    enum Role: String, Codable, Sendable {
        case champion
        case trainer
    }

    let role: Role
    let modelID: String?
    let architecture: NetworkArchitecture
    let policyTailPrecision: ChessNetwork.PolicyTailPrecision
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
    /// The same architecture's initial weights (`AnalysisInitReference`).
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
}

/// The initial weights of an architecture, made by the network builder
/// itself — `ChessMPSNetwork(.randomWeights(initSeed:))`, the path every
/// fresh model is built by, BN calibration included — never by a second copy
/// of its rules. Every "init" figure an analyzer reports (initial L2 norms,
/// ratios to init, drift from init, the starting value-head bias) is read
/// from here, so a model built with another final init, draw prior, branch
/// or skip init, or SE bias init is compared with its own starting point.
///
/// With the model's recorded init seed (`ModelInitRecord`, scheme
/// `WeightInitScheme.current`) the builder reproduces its exact initial
/// values, so every tensor's drift is exact — the BN running statistics to
/// the float tolerance of their GPU calibration pass. Without one, the
/// reference is built under two other seeds: a tensor whose values agree
/// between them is deterministic (biases, BN γ/β, the draw prior,
/// zero-initialized heads, identity-like projections) and its drift is
/// exact; a random one — and the BN running statistics, which the build
/// calibrates on a seeded warmup — gets an initial value from a draw of the
/// same distribution, and no drift.
struct AnalysisInitReference: Sendable {
    enum Basis: Sendable, Equatable {
        /// Built under the model's own init seed: exact for every tensor.
        case modelInitSeed(UInt64)
        /// The model's seed is unknown: built under other seeds, exact only
        /// for deterministic tensors.
        case otherSeeds(String)
    }

    let basis: Basis
    /// Initial values by graph variable name.
    let initialValues: [String: [Float]]
    /// Tensors whose initial values are known exactly.
    let exactNames: Set<String>

    /// A description for the exports.
    var basisDescription: String {
        switch basis {
        case .modelInitSeed(let seed):
            return "the model's own init seed \(seed) (\(WeightInitScheme.current)): exact initial values"
        case .otherSeeds(let why):
            return "\(why); deterministic tensors exact, random tensors from a draw of the same distribution (no drift)"
        }
    }

    /// Seeds for a reference built without the model's own: two, so the
    /// tensors that do not depend on the seed can be told apart.
    static let fallbackSeeds: (UInt64, UInt64) = (1, 2)

    /// Build the reference for `architecture` by building it, under
    /// `initialization`'s seed when it is one the current scheme can
    /// reproduce. `names` are the analyzed network's variable names, which
    /// the reference must match. Graph construction and calibration are long
    /// synchronous work, so they run on a GCD queue.
    static func build(architecture: NetworkArchitecture, initialization: ModelInitRecord?,
                      names: [String]) async throws -> AnalysisInitReference {
        if let initialization, initialization.scheme == WeightInitScheme.current {
            let values = try await initialValues(architecture: architecture, seed: initialization.initSeed, names: names)
            return AnalysisInitReference(basis: .modelInitSeed(initialization.initSeed),
                                         initialValues: values, exactNames: Set(names))
        }
        let why: String
        if let initialization {
            why = "the model's init scheme \(initialization.scheme) is not \(WeightInitScheme.current)"
        } else {
            why = "the model's init seed is not recorded"
        }
        let first = try await initialValues(architecture: architecture, seed: fallbackSeeds.0, names: names)
        let second = try await initialValues(architecture: architecture, seed: fallbackSeeds.1, names: names)
        let exact = Set(names.filter { first[$0] == second[$0] })
        return AnalysisInitReference(basis: .otherSeeds(why), initialValues: first, exactNames: exact)
    }

    private static func initialValues(architecture: NetworkArchitecture, seed: UInt64,
                                      names: [String]) async throws -> [String: [Float]] {
        let built: ChessMPSNetwork = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                do {
                    continuation.resume(returning: try ChessMPSNetwork(.randomWeights(initSeed: seed), arch: architecture))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        let network = built.network
        let referenceNames = (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name }
        guard referenceNames == names else {
            throw AnalysisInitReferenceError.variablesDiffer(analyzed: names.count, reference: referenceNames.count)
        }
        let weights = try await network.exportWeights()
        guard weights.count == names.count else {
            throw AnalysisInitReferenceError.variablesDiffer(analyzed: names.count, reference: weights.count)
        }
        return Dictionary(uniqueKeysWithValues: zip(names, weights))
    }
}

enum AnalysisInitReferenceError: LocalizedError {
    /// The architecture's fresh build does not have the analyzed network's
    /// variables — the network was not built from this architecture.
    case variablesDiffer(analyzed: Int, reference: Int)

    var errorDescription: String? {
        switch self {
        case .variablesDiffer(let analyzed, let reference):
            return "The init reference built from the architecture has \(reference) variables (or names) where the analyzed network has \(analyzed)"
        }
    }
}
