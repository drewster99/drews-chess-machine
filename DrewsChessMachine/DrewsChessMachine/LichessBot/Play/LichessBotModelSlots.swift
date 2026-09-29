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
    /// Whether a trainer exists (it may be idle; plan E44).
    func trainerAvailable() async -> Bool
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

/// Owns the current model generation and builds new ones (plan §9, E15,
/// E40–E44).
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
/// New games take the newest ready generation. Building happens before a
/// generation is published, so a game never waits on a network build for
/// its first move (E15).
actor LichessBotModelSlots {
    private let provider: any LichessBotModelProvider
    private let time: any LichessBotTimeSource
    private let log: @Sendable (String) -> Void

    private(set) var current: LichessBotModelGeneration?
    private var currentSettings: LichessBotModelSettings?
    private var lastSnapshotAt: Duration?
    private var nextGenerationID = 1
    /// A build in flight, shared by every caller that needs it, so
    /// concurrent callers (a challenge acceptance and a `gameStart` arriving
    /// together) never build two networks for one snapshot.
    private var pendingBuild: (settings: LichessBotModelSettings, task: Task<LichessBotModelGeneration, Error>)?

    init(provider: any LichessBotModelProvider, time: any LichessBotTimeSource, log: @escaping @Sendable (String) -> Void) {
        self.provider = provider
        self.time = time
        self.log = log
    }

    /// The generation new games should use, building one first if there is
    /// none for `settings` yet.
    func ready(for settings: LichessBotModelSettings) async throws -> LichessBotModelGeneration {
        if let current, currentSettings == settings {
            return current
        }
        return try await rebuild(for: settings, reason: currentSettings == nil ? "first use" : "source changed")
    }

    /// Run a build, or join the one already running for the same settings.
    private func rebuild(for settings: LichessBotModelSettings, reason: String) async throws -> LichessBotModelGeneration {
        if let pendingBuild, pendingBuild.settings == settings {
            return try await pendingBuild.task.value
        }
        let task = Task { try await self.build(for: settings, reason: reason) }
        pendingBuild = (settings, task)
        defer {
            if pendingBuild?.settings == settings {
                pendingBuild = nil
            }
        }
        return try await task.value
    }

    /// Re-snapshot when the source calls for it: a new champion ModelID, or
    /// a live-trainer interval elapsed. Cheap to call often; does nothing
    /// otherwise.
    func refreshIfDue(for settings: LichessBotModelSettings) async throws {
        guard let current, currentSettings == settings, pendingBuild == nil else { return }
        switch settings.source {
        case .champion:
            let championID = await provider.championModelID()
            if let championID, championID != current.info.modelID {
                _ = try await rebuild(for: settings, reason: "champion changed to \(championID)")
            }
        case .liveTrainer:
            if let last = lastSnapshotAt, time.now() - last >= .seconds(settings.liveTrainerRefreshIntervalSeconds) {
                _ = try await rebuild(for: settings, reason: "live-trainer interval elapsed")
            }
        case .trainerSnapshot, .file:
            break
        }
    }

    /// The operator's "Re-snapshot now".
    func forceRefresh(for settings: LichessBotModelSettings) async throws -> LichessBotModelGeneration {
        try await rebuild(for: settings, reason: "operator re-snapshot")
    }

    /// Whether `settings`' source can produce weights right now (plan §9:
    /// an unavailable source means declining challenges, never silently
    /// using a different one).
    func sourceAvailable(for settings: LichessBotModelSettings) async -> Bool {
        switch settings.source {
        case .champion:
            return await provider.championModelID() != nil
        case .trainerSnapshot, .liveTrainer:
            return await provider.trainerAvailable()
        case .file:
            return !(settings.filePath ?? "").isEmpty
        }
    }

    private func build(for settings: LichessBotModelSettings, reason: String) async throws -> LichessBotModelGeneration {
        let started = time.now()

        let snapshot: LichessBotWeightsSnapshot
        var filePath: String?
        var fileSHA256: String?
        var valueHeadRecenteredOnLoad: Bool?
        switch settings.source {
        case .champion:
            snapshot = try await provider.championSnapshot()
        case .trainerSnapshot, .liveTrainer:
            snapshot = try await provider.trainerSnapshot()
        case .file:
            guard let path = settings.filePath, !path.isEmpty else {
                throw LichessBotModelError.noFileSelected
            }
            let loaded = try await Self.loadFile(at: URL(fileURLWithPath: path))
            snapshot = loaded.snapshot
            filePath = path
            fileSHA256 = loaded.sha256
            valueHeadRecenteredOnLoad = loaded.valueHeadRecentered
        }

        let network = try await InferenceNetworkFactory.build(loading: snapshot.weights, arch: snapshot.architecture)
        let generationID = nextGenerationID
        nextGenerationID += 1
        let info = LichessBotGenerationInfo(
            generationID: generationID,
            sourceKind: settings.source,
            modelID: snapshot.modelID,
            trainingStep: snapshot.trainingStep,
            snapshotAt: Date(),
            architectureSummary: snapshot.architecture.architectureSummary,
            filePath: filePath,
            fileSHA256: fileSHA256,
            valueHeadRecenteredOnLoad: valueHeadRecenteredOnLoad
        )
        let generation = LichessBotModelGeneration(info: info, network: network)
        current = generation
        currentSettings = settings
        lastSnapshotAt = time.now()
        let milliseconds = LichessBotBackoff.seconds(time.now() - started) * 1000
        log("[LICHESS-BOT] model generation \(generationID) ready: \(settings.source.rawValue) \(snapshot.modelID) step=\(snapshot.trainingStep.map(String.init) ?? "-") (\(reason)) snapshot ms=\(String(format: "%.1f", milliseconds))")
        return generation
    }

    /// Load and hash a model file off the cooperative thread pool.
    /// `valueHeadRecentered` reports whether decode changed the value head
    /// (see `ValueHeadRecentering`), so the generation records that its
    /// weights no longer match the hashed bytes.
    private static func loadFile(
        at url: URL
    ) async throws -> (snapshot: LichessBotWeightsSnapshot, sha256: String, valueHeadRecentered: Bool) {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let file = try CheckpointManager.loadModelFile(at: url)
                    let bytes = try Data(contentsOf: url)
                    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                    let snapshot = LichessBotWeightsSnapshot(
                        weights: file.networkWeights,
                        architecture: file.architecture,
                        modelID: file.modelID,
                        trainingStep: file.metadata.trainingStep
                    )
                    guard let centering = file.valueHeadCentering else {
                        throw LichessBotModelError.valueHeadCenteringUnknown(url.path)
                    }
                    let recentered: Bool
                    if case .recentered = centering { recentered = true } else { recentered = false }
                    continuation.resume(returning: (snapshot, digest, recentered))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
