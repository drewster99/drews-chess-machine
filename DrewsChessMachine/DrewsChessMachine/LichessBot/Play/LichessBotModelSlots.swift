import CryptoKit
import Foundation

/// Weights copied out of a live network, with their identity.
struct LichessBotWeightsSnapshot: Sendable {
    let weights: [[Float]]
    let architecture: NetworkArchitecture
    let modelID: String
    let trainingStep: Int?
}

/// Where champion and trainer weights come from. The app implements this
/// over `SessionController`; tests supply fakes.
protocol LichessBotModelProvider: Sendable {
    /// The live champion's ModelID, or nil if no champion exists.
    func championModelID() async -> String?
    func championSnapshot() async throws -> LichessBotWeightsSnapshot
    func trainerSnapshot() async throws -> LichessBotWeightsSnapshot
}

enum LichessBotModelError: LocalizedError, Equatable {
    case noChampion
    case noTrainer
    case noFileSelected
    /// A loaded model file carried no decode-time value-head centering
    /// result, so the generation could not record whether its weights match
    /// the file's bytes. Every decoder sets it; seeing this is a bug.
    case valueHeadCenteringUnknown(String)

    var errorDescription: String? {
        switch self {
        case .noChampion:
            return "No champion network exists. Build or load one first."
        case .noTrainer:
            return "No trainer exists. Start Play-and-Train at least once."
        case .noFileSelected:
            return "No model file is selected."
        case .valueHeadCenteringUnknown(let path):
            return "Model file \(path) was decoded without a value-head centering result."
        }
    }
}

/// A snapshot of weights in its own inference network: the unit a game
/// plays with (plan §9). Immutable. A game keeps the generation it started
/// with for its whole length (unless live-trainer mid-game refresh is on),
/// and a generation's network is freed when nothing references it.
final class LichessBotModelGeneration: LichessBotMoveSource {
    let info: LichessBotGenerationInfo
    let network: ChessMPSNetwork

    init(info: LichessBotGenerationInfo, network: ChessMPSNetwork) {
        self.info = info
        self.network = network
    }

    func decide(_ request: LichessBotMoveRequest, schedule: SamplingSchedule) async throws -> LichessBotMoveDecision {
        try await LichessBotMoveChooser.choose(request, network: network, schedule: schedule)
    }
}

/// What a model build is doing right now, in words for the operator ("loading
/// <file>", "building the network"). Called from whatever task runs the
/// build; the receiver hops to its own actor.
typealias LichessBotModelBuildProgress = @Sendable (String) -> Void

/// A built network and everything its generation records, except the
/// generation's number, which the slots assign when they publish it (so a
/// failed build never uses one up).
struct LichessBotBuiltModel: Sendable {
    let sourceKind: LichessBotModelSourceKind
    let snapshot: LichessBotWeightsSnapshot
    let filePath: String?
    let fileSHA256: String?
    let valueHeadRecenteredOnLoad: Bool?
    let network: ChessMPSNetwork
    /// From the start of the build to the network being ready.
    let milliseconds: Double

    func generation(id: Int) -> LichessBotModelGeneration {
        let info = LichessBotGenerationInfo(
            generationID: id,
            sourceKind: sourceKind,
            modelID: snapshot.modelID,
            trainingStep: snapshot.trainingStep,
            snapshotAt: Date(),
            architectureSummary: snapshot.architecture.architectureSummary,
            filePath: filePath,
            fileSHA256: fileSHA256,
            valueHeadRecenteredOnLoad: valueHeadRecenteredOnLoad
        )
        return LichessBotModelGeneration(info: info, network: network)
    }

    /// The session-log line announcing generation `id`.
    func readyLine(generationID id: Int, reason: String) -> String {
        "[LICHESS-BOT] model generation \(id) ready: \(sourceKind.rawValue) \(snapshot.modelID) step=\(snapshot.trainingStep.map(String.init) ?? "-") (\(reason)) snapshot ms=\(String(format: "%.1f", milliseconds))"
    }
}

/// Builds the network for a model source: the one build path the first
/// generation (`LichessBotModelSlots.prepare`) and every later one share.
struct LichessBotGenerationBuilder: Sendable {
    let provider: any LichessBotModelProvider
    let time: any LichessBotTimeSource
    let loader: LichessBotModelFileLoader

    func build(for settings: LichessBotModelSettings, progress: LichessBotModelBuildProgress?) async throws -> LichessBotBuiltModel {
        let started = time.now()

        let snapshot: LichessBotWeightsSnapshot
        var filePath: String?
        var fileSHA256: String?
        var valueHeadRecenteredOnLoad: Bool?
        switch settings.source {
        case .champion:
            progress?("exporting the champion's weights")
            snapshot = try await provider.championSnapshot()
        case .trainerSnapshot, .liveTrainer:
            progress?("exporting the trainer's weights")
            snapshot = try await provider.trainerSnapshot()
        case .file:
            guard let path = settings.filePath, !path.isEmpty else {
                throw LichessBotModelError.noFileSelected
            }
            let url = URL(fileURLWithPath: path)
            progress?("loading \(url.lastPathComponent)")
            let loaded = try await loader.load(at: url)
            snapshot = loaded.snapshot
            filePath = path
            fileSHA256 = loaded.sha256
            valueHeadRecenteredOnLoad = loaded.valueHeadRecentered
        }

        progress?("building the network")
        let network = try await InferenceNetworkFactory.build(loading: snapshot.weights, arch: snapshot.architecture)
        return LichessBotBuiltModel(
            sourceKind: settings.source,
            snapshot: snapshot,
            filePath: filePath,
            fileSHA256: fileSHA256,
            valueHeadRecenteredOnLoad: valueHeadRecenteredOnLoad,
            network: network,
            milliseconds: LichessBotBackoff.seconds(time.now() - started) * 1000
        )
    }
}

/// Reads, decodes and hashes a model file for a generation (follow-lineage
/// plan §3.5), off the cooperative thread pool. Decoding goes through
/// `CheckpointManager.loadModelFile(fromBytes:source:)`, the single
/// model-file loader.
struct LichessBotModelFileLoader: Sendable {
    /// What a load produced. `valueHeadRecentered` reports whether decode
    /// changed the value head (see `ValueHeadRecentering`), so the
    /// generation records that its weights no longer match the hashed
    /// bytes.
    struct Loaded: Sendable {
        let snapshot: LichessBotWeightsSnapshot
        let sha256: String
        let valueHeadRecentered: Bool
    }

    /// Reads a file's bytes: the file itself in the app; scripted bytes in a
    /// test.
    let readBytes: @Sendable (URL) throws -> Data

    static let live = LichessBotModelFileLoader(readBytes: { url in
        try CheckpointManager.readModelFileBytes(at: url)
    })

    func load(at url: URL) async throws -> Loaded {
        let readBytes = self.readBytes
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // One read, decoded and hashed: the hash names exactly
                    // the bytes played. A second read could see another file
                    // renamed over the path in between (a rolling save).
                    // `Data(contentsOf:)` keeps reading the inode it opened,
                    // so the bytes are one file, and the decode's
                    // `content_sha256` check refuses a half-copied one.
                    let bytes = try readBytes(url)
                    let file = try CheckpointManager.loadModelFile(fromBytes: bytes, source: url.lastPathComponent)
                    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                    let snapshot = LichessBotWeightsSnapshot(
                        weights: file.networkWeights,
                        architecture: file.architecture,
                        modelID: file.modelID,
                        // The file's trainer step where it records one, else
                        // the step it states (`ModelFileStepReading`) — the
                        // same kind of number the live sources pass.
                        trainingStep: file.trainingStepReading.trainerStepOrStatedStep
                    )
                    guard let centering = file.valueHeadCentering else {
                        throw LichessBotModelError.valueHeadCenteringUnknown(url.path)
                    }
                    let recentered: Bool
                    if case .recentered = centering { recentered = true } else { recentered = false }
                    continuation.resume(returning: Loaded(snapshot: snapshot, sha256: digest, valueHeadRecentered: recentered))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

/// Owns the current model generation and builds new ones (plan §9, E15,
/// E40–E44; follow-lineage plan §3.10).
///
/// - **Champion** re-snapshots only when the champion's ModelID changes (a
///   promotion); champion weights change at no other time.
/// - **Trainer snapshot** is taken when the source is selected and on an
///   explicit re-snapshot.
/// - **Live trainer** re-snapshots on a cadence. Each snapshot holds the lock
///   SGD needs for the length of the weight export, so it is deliberately
///   not per move.
/// - **File** loads through `CheckpointManager.loadModelFile`, the single
///   model-file loader.
///
/// **Slots never exist without a generation.** The only way to get slots is
/// `prepare`, which builds the first generation before it returns, and a
/// generation is only ever replaced by a newer one that finished building.
/// The bot builds its slots before it opens the event stream, so while it is
/// online every new game reads `current` — never a build, never a wait,
/// never "no model" — and no challenge is ever declined for want of one
/// (owner rule OD-18). A source change while online is picked up by the poll
/// loop's `refreshIfDue`, which builds the new source's generation while
/// `current` keeps serving the old one; a failed switch keeps the old one
/// and throws, so the controller alarms and retries.
actor LichessBotModelSlots {
    private let builder: LichessBotGenerationBuilder
    private let log: @Sendable (String) -> Void

    /// The generation new games use.
    private(set) var current: LichessBotModelGeneration
    /// The model settings in force: `current`'s generation source, with
    /// the newest values of the fields that don't select weights (the
    /// refresh interval, the mid-game toggle).
    private var currentSettings: LichessBotModelSettings
    private var lastSnapshotAt: Duration
    private var nextGenerationID: Int
    /// A build in flight, joined by every caller that asks for the same
    /// generation source, so two callers never build two networks for one
    /// snapshot (two callers differing only in, say, the refresh interval
    /// want the same weights).
    private var pendingBuild: (source: LichessBotGenerationSource, task: Task<LichessBotModelGeneration, Error>)?

    private init(builder: LichessBotGenerationBuilder, first: LichessBotModelGeneration, settings: LichessBotModelSettings, log: @escaping @Sendable (String) -> Void) {
        self.builder = builder
        self.log = log
        self.current = first
        self.currentSettings = settings
        self.lastSnapshotAt = builder.time.now()
        self.nextGenerationID = first.info.generationID + 1
    }

    /// Build the first generation for `settings` and return slots holding
    /// it. Throws whatever the build throws (no champion, no trainer, a file
    /// that won't load): the caller stays offline with that error.
    ///
    /// `loader` reads model files; the app passes `.live` (the file itself),
    /// a test may count or script the reads.
    static func prepare(
        for settings: LichessBotModelSettings,
        provider: any LichessBotModelProvider,
        time: any LichessBotTimeSource,
        loader: LichessBotModelFileLoader = .live,
        log: @escaping @Sendable (String) -> Void,
        progress: LichessBotModelBuildProgress? = nil
    ) async throws -> LichessBotModelSlots {
        let builder = LichessBotGenerationBuilder(provider: provider, time: time, loader: loader)
        let built = try await builder.build(for: settings, progress: progress)
        let firstID = 1
        let slots = LichessBotModelSlots(builder: builder, first: built.generation(id: firstID), settings: settings, log: log)
        log(built.readyLine(generationID: firstID, reason: "going online"))
        return slots
    }

    /// Run a build, or join the one already running for the same
    /// generation source.
    private func rebuild(for settings: LichessBotModelSettings, reason: String) async throws -> LichessBotModelGeneration {
        let source = settings.generationSource
        if let pendingBuild, pendingBuild.source == source {
            return try await pendingBuild.task.value
        }
        let builder = self.builder
        let task = Task {
            let built = try await builder.build(for: settings, progress: nil)
            return self.publish(built, settings: settings, reason: reason)
        }
        pendingBuild = (source, task)
        defer {
            if pendingBuild?.source == source {
                pendingBuild = nil
            }
        }
        return try await task.value
    }

    /// Make a finished build the current generation.
    private func publish(_ built: LichessBotBuiltModel, settings: LichessBotModelSettings, reason: String) -> LichessBotModelGeneration {
        let generationID = nextGenerationID
        nextGenerationID += 1
        let generation = built.generation(id: generationID)
        current = generation
        currentSettings = settings
        lastSnapshotAt = builder.time.now()
        log(built.readyLine(generationID: generationID, reason: reason))
        return generation
    }

    /// Bring the current generation up to date with `settings`: switch to
    /// another source when the settings name one, re-snapshot when the
    /// source calls for it (a new champion ModelID, a live-trainer interval
    /// elapsed). Cheap to call often; does nothing otherwise. Throws when a
    /// switch or refresh fails, or the champion is gone; `current` keeps
    /// playing either way.
    ///
    /// A build already in flight for the same weights is joined: the result
    /// (or error) is that build's, never a success reported before it ends.
    /// One for other weights is waited out first, then these settings are
    /// handled as usual.
    @discardableResult
    func refreshIfDue(for settings: LichessBotModelSettings) async throws -> LichessBotModelRefreshOutcome {
        if let pendingBuild {
            if pendingBuild.source == settings.generationSource {
                let joined = try await pendingBuild.task.value
                return .joinedBuildInFlight(joined.info)
            }
            // Its outcome is its own caller's to report; these settings
            // are handled below, after it.
            _ = await pendingBuild.task.result
        }
        guard currentSettings.generationSource == settings.generationSource else {
            let built = try await rebuild(for: settings, reason: "source changed to \(settings.source.rawValue)")
            return .built(built.info)
        }
        // Same weights: the interval and the mid-game toggle in force are
        // the newest, with no rebuild.
        currentSettings = settings
        switch settings.source {
        case .champion:
            guard let championID = await builder.provider.championModelID() else {
                throw LichessBotModelError.noChampion
            }
            if championID != current.info.modelID {
                let built = try await rebuild(for: settings, reason: "champion changed to \(championID)")
                return .built(built.info)
            }
        case .liveTrainer:
            if builder.time.now() - lastSnapshotAt >= .seconds(settings.liveTrainerRefreshIntervalSeconds) {
                let built = try await rebuild(for: settings, reason: "live-trainer interval elapsed")
                return .built(built.info)
            }
        case .trainerSnapshot, .file:
            break
        }
        return .unchanged
    }
}

/// What `LichessBotModelSlots.refreshIfDue` did. Every case is a success;
/// a failure throws.
enum LichessBotModelRefreshOutcome: Equatable, Sendable {
    /// Nothing was due: the current generation is up to date.
    case unchanged
    /// This call built a new generation, now current.
    case built(LichessBotGenerationInfo)
    /// A build already in flight for the same weights finished with this
    /// generation, now current.
    case joinedBuildInFlight(LichessBotGenerationInfo)
}
