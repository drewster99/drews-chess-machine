//
//  RelativeGradientCap.swift
//  DrewsChessMachine
//
//  The relative gradient-norm cap (`documentation/plans-active/
//  RELATIVE_GRADIENT_CAP_PLAN.md`): each real-data SGD step is clipped at
//
//      min(Gradient Clip Max Norm, max(floor, k × median of the last N
//          pre-clip global gradient norms))
//
//  once the trailing window holds at least W entries. Everything here is pure
//  (no Metal, no logging): the trainer owns a `GradientNormHistory`, asks
//  `GradientCapPolicy.decide` for the next step's cap on its execution queue,
//  feeds `GradientCapDecision.fedCap` through the existing
//  `gradClipMaxNorm` scalar placeholder, and appends the step's pre-clip norm
//  and fed cap in the same block as its clock increment.
//
//  Why a cap relative to the run's own norms rather than a lower fixed cap: a
//  fixed cap of 1.0 stopped B-silu's step-20,600 blowup (E-0020), but early
//  training runs gradient norms of ~30 at step 1 and 1–3 for hundreds of
//  steps, so a fixed cap tight enough to catch a 7× spike late in training
//  would clip nearly every early step. And under the LR cycle a fixed cap
//  permits the same gradient norm at LR 1.0 as at LR 0.001 — a 1,000× larger
//  update — while a trailing median follows the phase.
//
//  Why the history holds PRE-clip norms: a post-clip history ratchets — each
//  clip records a value at the cap, pulling the median down, which lowers the
//  next cap and clips more. A median of pre-clip norms ignores up to half the
//  window, so an isolated spike moves it by one rank, not by its size.
//

import Foundation

// MARK: - Mode

/// What the relative cap does. One three-state parameter rather than an
/// enable flag plus a log flag, because "log only" is needed on its own (to
/// measure per-step spread without changing training math) and two booleans
/// would interact.
enum RelativeGradientCapMode: Int, Sendable, CaseIterable, Equatable {
    /// Only the hard max (`grad_clip_max_norm`) clips; the relative cap is
    /// not computed.
    case off = 0
    /// The relative cap is computed and each step it would clip is logged,
    /// but the hard max is what the graph is fed: training math is the same
    /// as `off`.
    case logOnly = 1
    /// The relative cap is fed to the graph.
    case clip = 2

    /// The token log lines and `results.json` use.
    var token: String {
        switch self {
        case .off: return "off"
        case .logOnly: return "log_only"
        case .clip: return "clip"
        }
    }

    /// The settings popover's segment label.
    var displayName: String {
        switch self {
        case .off: return "Off"
        case .logOnly: return "Log only"
        case .clip: return "Clip"
        }
    }
}

// MARK: - Configuration

enum RelativeGradientCapConfigurationError: Error, CustomStringConvertible, LocalizedError, Equatable {
    case unknownMode(Int)
    case minimumHistoryAboveWindow(minimumHistorySteps: Int, windowSteps: Int)
    case windowAboveCapacity(windowSteps: Int, capacity: Int)
    case nonPositiveWindow(windowSteps: Int)
    case nonPositiveMinimumHistory(minimumHistorySteps: Int)
    case invalidMultiple(Double)
    case invalidFloor(Double)

    var description: String {
        switch self {
        case .unknownMode(let raw):
            return "\(RelativeGradClipMode.id) = \(raw) is not a mode (0 = off, 1 = log only, 2 = clip)"
        case .minimumHistoryAboveWindow(let w, let n):
            return "\(RelativeGradClipMinHistorySteps.id) (\(w)) must be ≤ \(RelativeGradClipWindowSteps.id) (\(n)): "
                + "a window of \(n) steps can never hold \(w) entries, so the relative cap could never apply"
        case .windowAboveCapacity(let n, let capacity):
            return "\(RelativeGradClipWindowSteps.id) (\(n)) exceeds the gradient-norm history capacity (\(capacity))"
        case .nonPositiveWindow(let n):
            return "\(RelativeGradClipWindowSteps.id) (\(n)) must be positive"
        case .nonPositiveMinimumHistory(let w):
            return "\(RelativeGradClipMinHistorySteps.id) (\(w)) must be positive"
        case .invalidMultiple(let k):
            return "\(RelativeGradClipMultiple.id) (\(k)) must be finite and positive"
        case .invalidFloor(let floor):
            return "\(RelativeGradClipFloor.id) (\(floor)) must be finite and positive"
        }
    }

    var errorDescription: String? { description }
}

/// The relative cap's five settings, validated together. The throwing `init`
/// is the one place the cross-parameter rule (W ≤ N) lives, so the CLI, the
/// GUI popover and the trainer can never disagree about what is valid.
struct RelativeGradientCapConfiguration: Sendable, Equatable {
    let mode: RelativeGradientCapMode
    /// k: the multiple of the trailing median.
    let multiple: Double
    /// N: how many most recent real-data steps the median is over.
    let windowSteps: Int
    /// W: the minimum number of entries in the window before the relative
    /// term applies (warm-up).
    let minimumHistorySteps: Int
    /// Lower bound on the relative term.
    let floor: Double

    init(mode: RelativeGradientCapMode, multiple: Double, windowSteps: Int, minimumHistorySteps: Int, floor: Double) throws {
        guard multiple.isFinite, multiple > 0 else { throw RelativeGradientCapConfigurationError.invalidMultiple(multiple) }
        guard floor.isFinite, floor > 0 else { throw RelativeGradientCapConfigurationError.invalidFloor(floor) }
        guard windowSteps > 0 else { throw RelativeGradientCapConfigurationError.nonPositiveWindow(windowSteps: windowSteps) }
        guard minimumHistorySteps > 0 else {
            throw RelativeGradientCapConfigurationError.nonPositiveMinimumHistory(minimumHistorySteps: minimumHistorySteps)
        }
        guard windowSteps <= GradientNormHistory.capacity else {
            throw RelativeGradientCapConfigurationError.windowAboveCapacity(windowSteps: windowSteps, capacity: GradientNormHistory.capacity)
        }
        guard minimumHistorySteps <= windowSteps else {
            throw RelativeGradientCapConfigurationError.minimumHistoryAboveWindow(
                minimumHistorySteps: minimumHistorySteps, windowSteps: windowSteps)
        }
        self.mode = mode
        self.multiple = multiple
        self.windowSteps = windowSteps
        self.minimumHistorySteps = minimumHistorySteps
        self.floor = floor
    }

    /// The configuration from raw parameter values (the mode as its declared
    /// Int code).
    init(modeRawValue: Int, multiple: Double, windowSteps: Int, minimumHistorySteps: Int, floor: Double) throws {
        guard let mode = RelativeGradientCapMode(rawValue: modeRawValue) else {
            throw RelativeGradientCapConfigurationError.unknownMode(modeRawValue)
        }
        try self.init(mode: mode, multiple: multiple, windowSteps: windowSteps,
                      minimumHistorySteps: minimumHistorySteps, floor: floor)
    }

    /// The five parameters' declared defaults, validated.
    static func declaredDefaults() throws -> RelativeGradientCapConfiguration {
        try RelativeGradientCapConfiguration(
            modeRawValue: RelativeGradClipMode.declaredDefault,
            multiple: RelativeGradClipMultiple.declaredDefault,
            windowSteps: RelativeGradClipWindowSteps.declaredDefault,
            minimumHistorySteps: RelativeGradClipMinHistorySteps.declaredDefault,
            floor: RelativeGradClipFloor.declaredDefault
        )
    }

    /// The raw settings this configuration was validated from.
    var settings: RelativeGradientCapSettings {
        RelativeGradientCapSettings(modeRawValue: mode.rawValue, multiple: multiple, windowSteps: windowSteps,
                                    minimumHistorySteps: minimumHistorySteps, floor: floor)
    }
}

/// The relative cap's five settings as the parameters hold them, before the
/// cross-parameter check. This is what travels through
/// `TrainerHyperparameters` and what the trainer holds: every entry point
/// (`TrainingParameters.apply`, the settings popover, a session resume, CLI
/// start, GUI trainer setup) refuses W > N with an error, and the trainer
/// validates the settings again on every real-data step
/// (`validated()`), so a pair that slipped past them stops training with
/// that error rather than crashing the app or training under a cap nobody
/// chose.
struct RelativeGradientCapSettings: Sendable, Equatable {
    let modeRawValue: Int
    let multiple: Double
    let windowSteps: Int
    let minimumHistorySteps: Int
    let floor: Double

    init(modeRawValue: Int, multiple: Double, windowSteps: Int, minimumHistorySteps: Int, floor: Double) {
        self.modeRawValue = modeRawValue
        self.multiple = multiple
        self.windowSteps = windowSteps
        self.minimumHistorySteps = minimumHistorySteps
        self.floor = floor
    }

    /// The five parameters' declared defaults.
    static let declaredDefaults = RelativeGradientCapSettings(
        modeRawValue: RelativeGradClipMode.declaredDefault,
        multiple: RelativeGradClipMultiple.declaredDefault,
        windowSteps: RelativeGradClipWindowSteps.declaredDefault,
        minimumHistorySteps: RelativeGradClipMinHistorySteps.declaredDefault,
        floor: RelativeGradClipFloor.declaredDefault
    )

    /// The validated configuration, or the error naming the parameters.
    func validated() throws -> RelativeGradientCapConfiguration {
        try RelativeGradientCapConfiguration(modeRawValue: modeRawValue, multiple: multiple, windowSteps: windowSteps,
                                             minimumHistorySteps: minimumHistorySteps, floor: floor)
    }

    /// The mode, nil for a code outside 0…2.
    var mode: RelativeGradientCapMode? { RelativeGradientCapMode(rawValue: modeRawValue) }

    /// The mode's log token (`invalid(<code>)` for a code outside 0…2).
    var modeToken: String { mode?.token ?? "invalid(\(modeRawValue))" }

    /// `relClip=<mode>/k<k>/N<N>/W<W>/floor<f>` for the HPARAMS lines.
    var compactDescription: String {
        "\(modeToken)/k\(RelativeGradientCapLogFormat.number(multiple))/N\(windowSteps)/W\(minimumHistorySteps)"
            + "/floor\(RelativeGradientCapLogFormat.number(floor))"
    }
}

// MARK: - Decision

/// The cap one SGD step is fed, and why.
struct GradientCapDecision: Sendable, Equatable {
    /// Which term of `min(hardMax, max(floor, k × median))` set the decided
    /// cap.
    enum Binding: String, Sendable, Equatable {
        case hard
        case relative
        case floor
    }

    /// What the graph's `gradClipMaxNorm` placeholder is fed: `decidedCap`
    /// in mode `clip`, the hard max otherwise.
    let fedCap: Float
    /// The rule's cap (the hard max when the relative term does not apply).
    let decidedCap: Float
    let binding: Binding
    /// The trailing median of pre-clip norms the relative term used; nil when
    /// none was computed (mode `off`, warm-up, or the synthetic path).
    let referenceMedian: Double?
    /// Entries the median was taken over (0 when none was computed).
    let referenceCount: Int
    let mode: RelativeGradientCapMode
    /// The hard max the decision was made against.
    let hardMax: Float
    /// k and the floor the decision was made with, for the `[GRAD-CLIP]`
    /// line.
    let multiple: Double
    let floor: Double

    /// The graph clipped this step: its pre-clip norm exceeded the fed cap —
    /// the same comparison the graph's `maximum(norm, cap)` makes.
    func clipped(preClipNorm: Float) -> Bool { preClipNorm > fedCap }

    /// The rule's cap would have clipped this step (in `logOnly`, the step
    /// the relative cap would have acted on).
    func wouldClip(preClipNorm: Float) -> Bool { preClipNorm > decidedCap }

    /// A decision that feeds the hard max with no relative term: the
    /// synthetic random-data path (GPU sweeps, smoke tests), which records
    /// nothing in the history and so has no median to consult.
    static func hardMaxOnly(hardMax: Float) -> GradientCapDecision {
        GradientCapDecision(fedCap: hardMax, decidedCap: hardMax, binding: .hard, referenceMedian: nil,
                            referenceCount: 0, mode: .off, hardMax: hardMax, multiple: .nan, floor: .nan)
    }

    // Equality over the doubles that are `.nan` on the synthetic path would
    // make `hardMaxOnly` unequal to itself; compare their bit patterns.
    static func == (lhs: GradientCapDecision, rhs: GradientCapDecision) -> Bool {
        lhs.fedCap.bitPattern == rhs.fedCap.bitPattern
            && lhs.decidedCap.bitPattern == rhs.decidedCap.bitPattern
            && lhs.binding == rhs.binding
            && lhs.referenceMedian?.bitPattern == rhs.referenceMedian?.bitPattern
            && lhs.referenceCount == rhs.referenceCount
            && lhs.mode == rhs.mode
            && lhs.hardMax.bitPattern == rhs.hardMax.bitPattern
            && lhs.multiple.bitPattern == rhs.multiple.bitPattern
            && lhs.floor.bitPattern == rhs.floor.bitPattern
    }
}

/// The cap rule (plan R1). Pure and deterministic: the same configuration,
/// hard max and history give the same decision, which is what lets an exact
/// resume (history restored) feed the uninterrupted run's caps.
enum GradientCapPolicy {
    /// The decision for the real-data step that will be trainer step
    /// `nextTrainerStep`. All arithmetic is in `Double`, converted to `Float`
    /// once.
    static func decide(
        configuration: RelativeGradientCapConfiguration,
        hardMax: Float,
        history: GradientNormHistory,
        nextTrainerStep: Int
    ) -> GradientCapDecision {
        func hardOnly(median: Double?, count: Int) -> GradientCapDecision {
            GradientCapDecision(fedCap: hardMax, decidedCap: hardMax, binding: .hard, referenceMedian: median,
                                referenceCount: count, mode: configuration.mode, hardMax: hardMax,
                                multiple: configuration.multiple, floor: configuration.floor)
        }
        guard configuration.mode != .off else { return hardOnly(median: nil, count: 0) }
        // The one trailing-median definition (shared with the `gradient_spike`
        // rule): values at steps s−N … s−1, at least W of them, any span.
        let policy = TrailingReferencePolicy(
            lookbackSteps: configuration.windowSteps,
            minimumRecords: configuration.minimumHistorySteps,
            minimumSpanSteps: 0
        )
        let window = history.window(endingBefore: nextTrainerStep, count: configuration.windowSteps)
        guard let reference = TrainingHealthReference.make(window, windowStart: nextTrainerStep, policy: policy) else {
            return hardOnly(median: nil, count: 0)
        }
        let hard = Double(hardMax)
        let scaled = configuration.multiple * reference.median
        let relative = max(configuration.floor, scaled)
        let cap = min(hard, relative)
        let binding: GradientCapDecision.Binding
        if hard <= relative {
            binding = .hard
        } else if configuration.floor >= scaled {
            binding = .floor
        } else {
            binding = .relative
        }
        let decided = binding == .hard ? hardMax : Float(cap)
        return GradientCapDecision(
            fedCap: configuration.mode == .clip ? decided : hardMax,
            decidedCap: decided,
            binding: binding,
            referenceMedian: reference.median,
            referenceCount: reference.recordCount,
            mode: configuration.mode,
            hardMax: hardMax,
            multiple: configuration.multiple,
            floor: configuration.floor
        )
    }
}

// MARK: - History

enum GradientNormHistoryError: Error, CustomStringConvertible, LocalizedError, Equatable {
    /// An append whose step is not the one after the last recorded step: the
    /// trainer clock moved without the history (a bug), so the step stops
    /// rather than mixing two trajectories in one median.
    case discontinuity(expected: Int, got: Int)
    case nonFiniteNorm(trainerStep: Int, value: Float)
    case nonFiniteCap(trainerStep: Int, value: Float)
    case nonPositiveStep(Int)
    /// A history that does not end at the trainer clock it is restored to or
    /// saved with.
    case clockMismatch(historyLastStep: Int, trainerClock: Int)
    case malformed(String)

    var description: String {
        switch self {
        case .discontinuity(let expected, let got):
            return "gradient-norm history discontinuity: expected trainer step \(expected), got \(got) "
                + "(the trainer clock moved without the history)"
        case .nonFiniteNorm(let step, let value):
            return "gradient-norm history: non-finite pre-clip norm \(value) at trainer step \(step)"
        case .nonFiniteCap(let step, let value):
            return "gradient-norm history: non-finite fed cap \(value) at trainer step \(step)"
        case .nonPositiveStep(let step):
            return "gradient-norm history: trainer step \(step) is not positive"
        case .clockMismatch(let last, let clock):
            return "gradient-norm history ends at trainer step \(last) but the trainer clock is \(clock)"
        case .malformed(let detail):
            return "\(GradientNormHistory.metadataKey) is malformed: \(detail)"
        }
    }

    var errorDescription: String? { description }
}

/// Every real-data SGD step's pre-clip global gradient norm and the cap the
/// step was fed, contiguous in trainer step, oldest first, bounded at
/// `capacity` entries. It is trainer state (the next step's cap is a function
/// of it): saved in every trainer-state file, restored by every exact resume,
/// rewound by a GUI promotion with the weights and clock.
struct GradientNormHistory: Sendable, Equatable {
    /// The most entries kept: the window parameter's declared maximum, so any
    /// valid N is covered (derived from the declaration, not a second
    /// constant).
    static let capacity: Int = RelativeGradClipWindowSteps.declaredClosedRange.upperBound

    /// The `__metadata__` key of a trainer-state file that carries it.
    static let metadataKey = "trainer_grad_norm_history"
    static let formatVersion = 1

    /// The trainer step of the newest entry; nil exactly when empty.
    private(set) var lastTrainerStep: Int?
    private(set) var preClipNorms: [Float]
    private(set) var fedCaps: [Float]

    /// An empty history.
    init() {
        lastTrainerStep = nil
        preClipNorms = []
        fedCaps = []
    }

    var count: Int { preClipNorms.count }
    var isEmpty: Bool { preClipNorms.isEmpty }

    /// The trainer step of the oldest entry, nil when empty.
    var firstTrainerStep: Int? {
        lastTrainerStep.map { $0 - preClipNorms.count + 1 }
    }

    /// Throws unless `trainerStep` may be appended next: positive, and the
    /// step after the last recorded one (any positive step when empty). The
    /// trainer checks this before running a step, so a clock moved without
    /// the history stops training before any weight changes.
    func checkContinues(toTrainerStep trainerStep: Int) throws {
        guard trainerStep > 0 else { throw GradientNormHistoryError.nonPositiveStep(trainerStep) }
        if let last = lastTrainerStep, trainerStep != last + 1 {
            throw GradientNormHistoryError.discontinuity(expected: last + 1, got: trainerStep)
        }
    }

    /// Record trainer step `trainerStep`. It must be the step after the last
    /// recorded one (any positive step when empty), and both values finite;
    /// the oldest entry is dropped past `capacity`.
    mutating func append(trainerStep: Int, preClipNorm: Float, fedCap: Float) throws {
        try checkContinues(toTrainerStep: trainerStep)
        guard preClipNorm.isFinite else { throw GradientNormHistoryError.nonFiniteNorm(trainerStep: trainerStep, value: preClipNorm) }
        guard fedCap.isFinite else { throw GradientNormHistoryError.nonFiniteCap(trainerStep: trainerStep, value: fedCap) }
        preClipNorms.append(preClipNorm)
        fedCaps.append(fedCap)
        if preClipNorms.count > Self.capacity {
            preClipNorms.removeFirst(preClipNorms.count - Self.capacity)
            fedCaps.removeFirst(fedCaps.count - Self.capacity)
        }
        lastTrainerStep = trainerStep
    }

    /// The pre-clip norms of the (at most `count`) entries at trainer steps
    /// `step − count … step − 1`, oldest first, in the form
    /// `TrainingHealthReference.make` takes.
    func window(endingBefore step: Int, count: Int) -> [(trainerStep: Int, value: Float?)] {
        guard let first = firstTrainerStep, count > 0 else { return [] }
        let lower = max(first, step - count)
        let upper = min(lastTrainerStep ?? (first - 1), step - 1)
        guard lower <= upper else { return [] }
        return (lower...upper).map { trainerStep in
            (trainerStep: trainerStep, value: preClipNorms[trainerStep - first])
        }
    }

    /// The largest pre-clip norm and the number of clipped steps (pre-clip
    /// norm above the fed cap) among the recorded steps in `trainerSteps`,
    /// and how many recorded steps that range covered — the step lines'
    /// `gNormMax=` / `clips=`, so a spike between two logged steps still
    /// shows in the next line.
    func summary(trainerSteps: ClosedRange<Int>) -> (maxPreClipNorm: Float?, clipped: Int, steps: Int) {
        guard let first = firstTrainerStep, let last = lastTrainerStep else { return (nil, 0, 0) }
        let lower = max(first, trainerSteps.lowerBound)
        let upper = min(last, trainerSteps.upperBound)
        guard lower <= upper else { return (nil, 0, 0) }
        var maxNorm: Float?
        var clipped = 0
        for step in lower...upper {
            let norm = preClipNorms[step - first]
            if maxNorm.map({ norm > $0 }) ?? true { maxNorm = norm }
            if norm > fedCaps[step - first] { clipped += 1 }
        }
        return (maxNorm, clipped, upper - lower + 1)
    }

    /// The newest entry's fed cap, nil when empty.
    var lastFedCap: Float? { fedCaps.last }

    /// Throws unless the history is empty or ends exactly at `trainerClock`:
    /// a history saved with, or restored to, a trainer whose clock it does
    /// not end at belongs to another trajectory.
    func checkEnds(atTrainerClock trainerClock: Int) throws {
        if let last = lastTrainerStep, last != trainerClock {
            throw GradientNormHistoryError.clockMismatch(historyLastStep: last, trainerClock: trainerClock)
        }
    }

    // MARK: Metadata form

    private struct Payload: Codable {
        let version: Int
        let lastTrainerStep: Int?
        let preClipNorms: [Float]
        let fedCaps: [Float]

        enum CodingKeys: String, CodingKey {
            case version
            case lastTrainerStep = "last_trainer_step"
            case preClipNorms = "pre_clip_norms"
            case fedCaps = "fed_caps"
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(version, forKey: .version)
            // Written as JSON null when empty, so the key set is fixed.
            try container.encode(lastTrainerStep, forKey: .lastTrainerStep)
            try container.encode(preClipNorms, forKey: .preClipNorms)
            try container.encode(fedCaps, forKey: .fedCaps)
        }
    }

    /// `{"fed_caps":[…],"last_trainer_step":…,"pre_clip_norms":[…],"version":1}`.
    /// `JSONEncoder` writes each `Float` in its shortest round-trip form, so
    /// every value decodes back bit-exactly.
    func metadataValue() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(Payload(version: Self.formatVersion, lastTrainerStep: lastTrainerStep,
                                              preClipNorms: preClipNorms, fedCaps: fedCaps))
        return String(decoding: data, as: UTF8.self)
    }

    /// The history a metadata value holds. Refuses another version, unequal
    /// array lengths, non-finite values, more than `capacity` entries, a
    /// non-positive step, entries without a last step (or a last step
    /// without entries), and a last step smaller than the entry count (the
    /// first entry would precede trainer step 1).
    init(metadataValue text: String) throws {
        let payload: Payload
        do {
            payload = try JSONDecoder().decode(Payload.self, from: Data(text.utf8))
        } catch {
            throw GradientNormHistoryError.malformed(error.localizedDescription)
        }
        guard payload.version == Self.formatVersion else {
            throw GradientNormHistoryError.malformed("version \(payload.version) (this build reads \(Self.formatVersion))")
        }
        guard payload.preClipNorms.count == payload.fedCaps.count else {
            throw GradientNormHistoryError.malformed(
                "pre_clip_norms has \(payload.preClipNorms.count) entries but fed_caps has \(payload.fedCaps.count)")
        }
        guard payload.preClipNorms.count <= Self.capacity else {
            throw GradientNormHistoryError.malformed("\(payload.preClipNorms.count) entries exceed the capacity \(Self.capacity)")
        }
        guard payload.preClipNorms.allSatisfy(\.isFinite), payload.fedCaps.allSatisfy(\.isFinite) else {
            throw GradientNormHistoryError.malformed("a value is not finite")
        }
        switch payload.lastTrainerStep {
        case nil:
            guard payload.preClipNorms.isEmpty else {
                throw GradientNormHistoryError.malformed("entries without last_trainer_step")
            }
        case .some(let last):
            guard last > 0 else { throw GradientNormHistoryError.malformed("last_trainer_step \(last) is not positive") }
            guard !payload.preClipNorms.isEmpty else {
                throw GradientNormHistoryError.malformed("last_trainer_step \(last) without entries")
            }
            guard last >= payload.preClipNorms.count else {
                throw GradientNormHistoryError.malformed(
                    "last_trainer_step \(last) is smaller than the entry count \(payload.preClipNorms.count)")
            }
        }
        lastTrainerStep = payload.lastTrainerStep
        preClipNorms = payload.preClipNorms
        fedCaps = payload.fedCaps
    }

    /// The history a file's `__metadata__` carries, nil when it has none (a
    /// plain model file, or a trainer file written before the relative cap).
    static func decode(fromMetadata metadata: [String: String]) throws -> GradientNormHistory? {
        guard let text = metadata[metadataKey] else { return nil }
        return try GradientNormHistory(metadataValue: text)
    }
}

/// Where a resume's gradient-norm history comes from.
enum GradNormHistoryResumeState: Sendable, Equatable {
    /// Read from the saving trainer.
    case restored(GradientNormHistory)
    /// The source carries none (every trainer file written before the
    /// relative cap). The resumed trainer starts an empty history: W steps
    /// without the relative term — a resume gap only in mode `clip`, where
    /// it changes the fed caps.
    case notInCheckpoint

    /// The history, nil when none.
    var history: GradientNormHistory? {
        switch self {
        case .restored(let history): return history
        case .notInCheckpoint: return nil
        }
    }
}
