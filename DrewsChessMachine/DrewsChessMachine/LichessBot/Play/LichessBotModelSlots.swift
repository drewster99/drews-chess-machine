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

/// What one generation is built from, once the settings are resolved. The
/// follow-lineage source resolves to a file only after a check of the models
/// folder has selected one (`LichessBotGenerationSourceResolution`).
enum LichessBotGenerationInput: Sendable {
    case champion
    case trainerSnapshot
    case liveTrainer
    case file(path: String)
    case followed(LichessBotLineageFollowTarget)

    var sourceKind: LichessBotModelSourceKind {
        switch self {
        case .champion: return .champion
        case .trainerSnapshot: return .trainerSnapshot
        case .liveTrainer: return .liveTrainer
        case .file: return .file
        case .followed: return .followLineage
        }
    }
}

/// How model settings select a generation: directly, or — for the
/// follow-lineage source — through a check of the models folder that picks
/// the file. Settings that can select nothing (a file source with no file,
/// a follow-lineage source with no lineage) throw here, before any build.
enum LichessBotGenerationSourceResolution: Sendable {
    case direct(LichessBotGenerationInput)
    case followLineage(LichessBotFollowedLineage)

    init(_ settings: LichessBotModelSettings) throws {
        switch settings.source {
        case .champion:
            self = .direct(.champion)
        case .trainerSnapshot:
            self = .direct(.trainerSnapshot)
        case .liveTrainer:
            self = .direct(.liveTrainer)
        case .file:
            guard let path = settings.filePath, !path.isEmpty else {
                throw LichessBotModelError.noFileSelected
            }
            self = .direct(.file(path: path))
        case .followLineage:
            guard let followed = settings.followedLineage, !followed.lineageRunID.isEmpty, !followed.anchorSegmentID.isEmpty else {
                throw LichessBotLineageFollowError.noLineageChosen
            }
            self = .followLineage(followed)
        }
    }
}

/// A built network and everything its generation records, except the
/// generation's number, which the slots assign when they publish it (so a
/// failed build never uses one up).
struct LichessBotBuiltModel: Sendable {
    let sourceKind: LichessBotModelSourceKind
    let snapshot: LichessBotWeightsSnapshot
    let filePath: String?
    let fileSHA256: String?
    let valueHeadRecenteredOnLoad: Bool?
    /// Where the file played sits in its run, for a file that records it.
    let lineage: LichessBotGenerationLineage?
    /// For the follow-lineage source: what the load found that the check
    /// did not (a newer save of the lineage renamed over the selected path).
    let loadNote: String?
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
            valueHeadRecenteredOnLoad: valueHeadRecenteredOnLoad,
            lineage: lineage
        )
        return LichessBotModelGeneration(info: info, network: network)
    }

    /// The session-log line announcing generation `id`. A file-backed
    /// generation adds the file, its lineage position (or the file's own
    /// hash when it records none) and the architecture.
    func readyLine(generationID id: Int, reason: String) -> String {
        var line = "[LICHESS-BOT] model generation \(id) ready: \(sourceKind.rawValue) \(snapshot.modelID) step=\(snapshot.trainingStep.map(String.init) ?? "-") (\(reason)) snapshot ms=\(String(format: "%.1f", milliseconds))"
        if let filePath {
            line += " file=\(URL(fileURLWithPath: filePath).lastPathComponent)"
            if let lineage {
                line += " seg=\(lineage.segmentIndex) cum=\(lineage.cumTrainerStep.map(String.init) ?? "null") sha=\(lineage.contentSHA256.prefix(12))"
            } else if let fileSHA256 {
                line += " lineage=unrecorded file_sha=\(fileSHA256.prefix(12))"
            }
            line += " arch=\(snapshot.architecture.architectureSummary)"
        }
        if let loadNote {
            line += " (\(loadNote))"
        }
        return line
    }
}

/// Builds an inference network from weights. The app builds a real one
/// (`InferenceNetworkFactory`); a test can make the build fail after a file
/// loaded, to tell a network failure from a file failure.
struct LichessBotInferenceNetworkFactory: Sendable {
    let build: @Sendable (_ weights: [[Float]], _ architecture: NetworkArchitecture) async throws -> ChessMPSNetwork

    static let live = LichessBotInferenceNetworkFactory(build: { weights, architecture in
        try await InferenceNetworkFactory.build(loading: weights, arch: architecture)
    })
}

/// Builds the network for a model source: the one build path the first
/// generation (`LichessBotModelSlots.prepare`) and every later one share.
struct LichessBotGenerationBuilder: Sendable {
    let provider: any LichessBotModelProvider
    let time: any LichessBotTimeSource
    let loader: LichessBotModelFileLoader
    let networkFactory: LichessBotInferenceNetworkFactory
    /// Lists the models folder for the follow-lineage source.
    let folderScanner: any LichessBotModelFolderScanning

    func build(_ input: LichessBotGenerationInput, progress: LichessBotModelBuildProgress?) async throws -> LichessBotBuiltModel {
        let started = time.now()

        let snapshot: LichessBotWeightsSnapshot
        var filePath: String?
        var fileSHA256: String?
        var valueHeadRecenteredOnLoad: Bool?
        var lineage: LichessBotGenerationLineage?
        var loadNote: String?
        switch input {
        case .champion:
            progress?("exporting the champion's weights")
            snapshot = try await provider.championSnapshot()
        case .trainerSnapshot, .liveTrainer:
            progress?("exporting the trainer's weights")
            snapshot = try await provider.trainerSnapshot()
        case .file(let path):
            let url = URL(fileURLWithPath: path)
            progress?("loading \(url.lastPathComponent)")
            let loaded = try await loader.load(at: url)
            snapshot = loaded.snapshot
            filePath = path
            fileSHA256 = loaded.sha256
            valueHeadRecenteredOnLoad = loaded.valueHeadRecentered
            lineage = Self.recordedLineage(of: loaded.parent)
        case .followed(let target):
            progress?("loading \(target.url.lastPathComponent)")
            let loaded: LichessBotModelFileLoader.Loaded
            do {
                loaded = try await loader.load(at: target.url)
            } catch {
                throw LichessBotLineageFollowError.fileFailedToLoad(file: target.url, reason: error.localizedDescription)
            }
            let verified = try Self.verify(loaded.parent, against: target)
            snapshot = loaded.snapshot
            filePath = target.url.path
            fileSHA256 = loaded.sha256
            valueHeadRecenteredOnLoad = loaded.valueHeadRecentered
            lineage = verified.lineage
            loadNote = verified.note
        }

        progress?("building the network")
        let network = try await networkFactory.build(snapshot.weights, snapshot.architecture)
        return LichessBotBuiltModel(
            sourceKind: input.sourceKind,
            snapshot: snapshot,
            filePath: filePath,
            fileSHA256: fileSHA256,
            valueHeadRecenteredOnLoad: valueHeadRecenteredOnLoad,
            lineage: lineage,
            loadNote: loadNote,
            network: network,
            milliseconds: LichessBotBackoff.seconds(time.now() - started) * 1000
        )
    }

    /// A fixed file's lineage position, when its header records one with a
    /// content hash (follow-lineage plan OD-11). A file written before
    /// lineage records plays as before, with no position.
    static func recordedLineage(of parent: LineageTracker.ParentFile) -> LichessBotGenerationLineage? {
        guard case .recorded(let record) = parent.lineage, let contentSHA256 = parent.contentSHA256 else {
            return nil
        }
        return LichessBotGenerationLineage(position: ModelFileLineagePosition(record: record), contentSHA256: contentSHA256, followed: nil)
    }

    /// Check the file actually decoded against what the check selected
    /// (follow-lineage plan §3.4). The selected path may have been renamed
    /// over between the check and the load: a newer save of the same
    /// segment, or of a later segment of the lineage, passes and is what the
    /// generation records (with a note); anything else — another run, a
    /// sibling segment that took over the same `--out-model` path, an older
    /// save — is refused, and the next check rescans.
    static func verify(_ parent: LineageTracker.ParentFile, against target: LichessBotLineageFollowTarget) throws -> (lineage: LichessBotGenerationLineage, note: String?) {
        func refuse(_ reason: String) -> LichessBotLineageFollowError {
            .followedFileChangedDuringLoad(file: target.url, reason: reason)
        }
        guard case .recorded(let record) = parent.lineage else {
            throw refuse("the file read records no lineage")
        }
        let position = ModelFileLineagePosition(record: record)
        let followed = target.followed
        guard position.lineageRunID == followed.lineageRunID else {
            throw refuse("the file read belongs to run \(position.lineageRunID), not \(followed.lineageRunID)")
        }
        guard position.segmentChain.contains(followed.anchorSegmentID) else {
            throw refuse("the file read is from segment \(position.segmentID), which does not descend from the chosen segment")
        }
        guard ModelLineageTip.followablePathKinds.contains(position.pathKind) else {
            throw refuse("the file read was written by a \(position.pathKind.rawValue) run")
        }
        guard let contentSHA256 = parent.contentSHA256 else {
            throw refuse("the file read records no content_sha256")
        }
        guard position.segmentChain.starts(with: target.candidate.position.segmentChain) else {
            throw refuse("the file read is from segment \(position.segmentID), not the selected segment \(target.candidate.position.segmentID) or a later one")
        }
        let loadedRank = ModelLineageRank(segmentChain: position.segmentChain, segmentLocalStep: position.segmentLocalStep, recordedUnix: position.recordedUnix)
        guard !target.rank.isAbove(loadedRank) else {
            throw refuse("the file read (step \(position.segmentLocalStep)) is older than the selected one (step \(target.candidate.position.segmentLocalStep))")
        }
        let note = contentSHA256 == target.candidate.contentSHA256 ? nil : "file changed since the check: loaded step \(position.segmentLocalStep)"
        return (LichessBotGenerationLineage(position: position, contentSHA256: contentSHA256, followed: followed), note)
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
        /// The decoded file's identity and lineage, from the same bytes.
        let parent: LineageTracker.ParentFile
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
                        trainingStep: file.metadata.trainingStep
                    )
                    guard let centering = file.valueHeadCentering else {
                        throw LichessBotModelError.valueHeadCenteringUnknown(url.path)
                    }
                    let recentered: Bool
                    if case .recentered = centering { recentered = true } else { recentered = false }
                    continuation.resume(returning: Loaded(snapshot: snapshot, sha256: digest, valueHeadRecentered: recentered, parent: file.lineageParent))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

/// Owns the current model generation and builds new ones (plan §9, E15,
/// E40–E44; follow-lineage plan §3.4, §3.10).
///
/// - **Champion** re-snapshots only when the champion's ModelID changes (a
///   promotion); champion weights change at no other time.
/// - **Trainer snapshot** is taken when the source is selected.
/// - **Live trainer** re-snapshots on a cadence. Each snapshot holds the lock
///   SGD needs for the length of the weight export, so it is deliberately
///   not per move.
/// - **File** loads through `CheckpointManager.loadModelFile`, the single
///   model-file loader.
/// - **Follow lineage** checks the models folder on its own cadence for the
///   newest file of one training run (from a chosen segment on, through
///   every exact resume) and loads it when it is newer than what plays. A
///   check it can't vouch for (no file, a fork, the folder unreadable)
///   throws, so the controller alarms and backs off, and the last good
///   generation keeps playing; it never steps back to an older file.
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
    /// refresh and check intervals, the mid-game toggle).
    private var currentSettings: LichessBotModelSettings
    private var lastSnapshotAt: Duration
    private var nextGenerationID: Int
    /// A build in flight, joined by every caller that asks for the same
    /// generation source, so two callers never build two networks for one
    /// snapshot (two callers differing only in, say, the refresh interval
    /// want the same weights).
    private var pendingBuild: (source: LichessBotGenerationSource, task: Task<LichessBotModelGeneration, Error>)?
    /// The follow-lineage source's memory between checks: the folder's
    /// header cache, the last status, files that failed to load.
    private var follower: LichessBotLineageFollower
    /// A lineage check in flight, joined by every caller checking the same
    /// lineage, so concurrent callers share one scan.
    private var pendingLineageCheck: (followed: LichessBotFollowedLineage, task: Task<LichessBotLineageFollower.Decision, Never>)?

    /// What the follow-lineage source found at its last check; nil until a
    /// lineage was checked.
    var lineageFollowStatus: LichessBotLineageFollowStatus? {
        follower.status
    }

    private init(builder: LichessBotGenerationBuilder, first: LichessBotModelGeneration, settings: LichessBotModelSettings, follower: LichessBotLineageFollower, log: @escaping @Sendable (String) -> Void) {
        self.builder = builder
        self.log = log
        self.current = first
        self.currentSettings = settings
        self.lastSnapshotAt = builder.time.now()
        self.nextGenerationID = first.info.generationID + 1
        self.follower = follower
    }

    /// How a problem line ends while nothing plays yet: going online stops.
    private static let offlineConsequence = "the bot stays offline"

    /// Build the first generation for `settings` and return slots holding
    /// it. Throws whatever the build throws (no champion, no trainer, a file
    /// that won't load, a lineage with no file it can vouch for): the caller
    /// stays offline with that error. For the follow-lineage source the
    /// check's scan cache and status go to the slots with the generation, so
    /// the first poll doesn't re-read every header.
    ///
    /// `folderScanner` lists the models folder for the follow-lineage source
    /// (the app's scans `Models/`; a test scans its own folder). `loader`
    /// reads model files and `networkFactory` builds networks; the app passes
    /// the live ones, a test may count, script or fail them.
    static func prepare(
        for settings: LichessBotModelSettings,
        provider: any LichessBotModelProvider,
        time: any LichessBotTimeSource,
        folderScanner: any LichessBotModelFolderScanning,
        loader: LichessBotModelFileLoader = .live,
        networkFactory: LichessBotInferenceNetworkFactory = .live,
        log: @escaping @Sendable (String) -> Void,
        progress: LichessBotModelBuildProgress? = nil
    ) async throws -> LichessBotModelSlots {
        let builder = LichessBotGenerationBuilder(provider: provider, time: time, loader: loader, networkFactory: networkFactory, folderScanner: folderScanner)
        var follower = LichessBotLineageFollower()
        let input: LichessBotGenerationInput
        switch try LichessBotGenerationSourceResolution(settings) {
        case .direct(let direct):
            input = direct
        case .followLineage(let followed):
            progress?("reading model headers")
            let scan = await scanResult(of: folderScanner, previous: follower.cache)
            let decision = follower.record(scan, followed: followed, notBelow: nil, checkedAt: Date(), now: time.now(), consequence: offlineConsequence)
            decision.logLines.forEach(log)
            guard let target = try requireTarget(decision, playingThisLineage: false) else {
                throw LichessBotLineageFollowError.unavailable(decision.status.outcome)
            }
            input = .followed(target)
        }
        let built: LichessBotBuiltModel
        do {
            built = try await builder.build(input, progress: progress)
        } catch let error as LichessBotLineageFollowError {
            if case .followed(let target) = input, let reason = error.fileFailureReason {
                log(follower.recordLoadFailure(target, reason: reason, consequence: offlineConsequence))
            }
            throw error
        }
        let firstID = 1
        let slots = LichessBotModelSlots(builder: builder, first: built.generation(id: firstID), settings: settings, follower: follower, log: log)
        log(built.readyLine(generationID: firstID, reason: "going online"))
        return slots
    }

    /// One scan of the models folder, its failure kept as a value: an
    /// unreadable folder is an outcome of the check, not an exception.
    private static func scanResult(of scanner: any LichessBotModelFolderScanning, previous: ModelFolderHeaderCache) async -> Result<ModelFolderScan, Error> {
        do {
            return .success(try await scanner.scan(previous: previous))
        } catch {
            return .failure(error)
        }
    }

    /// The file a check selected, or nil when nothing newer is to be built
    /// while this lineage plays (the newest on disk ranks below it, or the
    /// newest failed to load and was already reported). Every other outcome
    /// throws, so the caller stays offline or the poll loop alarms.
    private static func requireTarget(_ decision: LichessBotLineageFollower.Decision, playingThisLineage: Bool) throws -> LichessBotLineageFollowTarget? {
        if let target = decision.target {
            return target
        }
        let outcome = decision.status.outcome
        switch outcome {
        case .keepPlaying where playingThisLineage, .newestFailedToLoad where playingThisLineage:
            return nil
        default:
            throw LichessBotLineageFollowError.unavailable(outcome)
        }
    }

    /// Run a build, or join the one already running for the same
    /// generation source.
    private func rebuild(_ input: LichessBotGenerationInput, settings: LichessBotModelSettings, reason: String) async throws -> LichessBotModelGeneration {
        let source = settings.generationSource
        if let pendingBuild, pendingBuild.source == source {
            return try await pendingBuild.task.value
        }
        let builder = self.builder
        let task = Task {
            do {
                let built = try await builder.build(input, progress: nil)
                return self.publish(built, settings: settings, reason: reason)
            } catch let error as LichessBotLineageFollowError {
                // Once, here, not per joined caller: a file that failed is
                // remembered until it changes on disk.
                if case .followed(let target) = input, let reason = error.fileFailureReason {
                    self.log(self.follower.recordLoadFailure(target, reason: reason, consequence: self.playingConsequence))
                }
                throw error
            }
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
    /// elapsed, a newer file of the followed lineage at a due check). Cheap
    /// to call often; does nothing otherwise. Throws when a switch or
    /// refresh fails, the champion is gone, or the followed lineage has no
    /// file it can vouch for; `current` keeps playing either way.
    ///
    /// A build already in flight for the same weights is joined: the result
    /// (or error) is that build's, never a success reported before it ends.
    /// One for other weights is waited out first, then these settings are
    /// handled as usual.
    @discardableResult
    func refreshIfDue(for settings: LichessBotModelSettings) async throws -> LichessBotModelRefreshOutcome {
        try await refresh(for: settings, forceLineageCheck: false)
    }

    /// The operator's "Check Now" for the follow-lineage source (plan
    /// OD-12): `refreshIfDue` with the lineage check run now instead of at
    /// its interval. Every other rule applies unchanged.
    @discardableResult
    func checkLineageNow(for settings: LichessBotModelSettings) async throws -> LichessBotModelRefreshOutcome {
        try await refresh(for: settings, forceLineageCheck: true)
    }

    private func refresh(for settings: LichessBotModelSettings, forceLineageCheck: Bool) async throws -> LichessBotModelRefreshOutcome {
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
            return try await switchSource(to: settings)
        }
        // Same weights: the intervals and the mid-game toggle in force are
        // the newest, with no rebuild.
        currentSettings = settings
        switch settings.source {
        case .champion:
            guard let championID = await builder.provider.championModelID() else {
                throw LichessBotModelError.noChampion
            }
            if championID != current.info.modelID {
                let built = try await rebuild(.champion, settings: settings, reason: "champion changed to \(championID)")
                return .built(built.info)
            }
        case .liveTrainer:
            if builder.time.now() - lastSnapshotAt >= .seconds(settings.liveTrainerRefreshIntervalSeconds) {
                let built = try await rebuild(.liveTrainer, settings: settings, reason: "live-trainer interval elapsed")
                return .built(built.info)
            }
        case .trainerSnapshot, .file:
            break
        case .followLineage:
            guard let followed = settings.followedLineage else {
                throw LichessBotLineageFollowError.noLineageChosen
            }
            let due = forceLineageCheck
                || follower.isDue(now: builder.time.now(), interval: .seconds(settings.lineageCheckIntervalSeconds))
                || newestKnownFileIsNotPlaying(of: followed)
            if due {
                return try await followNewest(of: followed, settings: settings, reason: "newer file of the followed lineage")
            }
        }
        return .unchanged
    }

    /// Build the generation of the source `settings` name instead of the
    /// current one's.
    private func switchSource(to settings: LichessBotModelSettings) async throws -> LichessBotModelRefreshOutcome {
        let reason = "source changed to \(settings.source.rawValue)"
        switch try LichessBotGenerationSourceResolution(settings) {
        case .direct(let input):
            let built = try await rebuild(input, settings: settings, reason: reason)
            return .built(built.info)
        case .followLineage(let followed):
            return try await followNewest(of: followed, settings: settings, reason: reason)
        }
    }

    /// Check the followed lineage and build its newest file when it is not
    /// what plays and ranks above it.
    private func followNewest(of followed: LichessBotFollowedLineage, settings: LichessBotModelSettings, reason: String) async throws -> LichessBotModelRefreshOutcome {
        let decision = await checkLineage(followed)
        // Read after the check: a build may have finished while it ran.
        let playing = playingRank(of: followed)
        guard let target = try Self.requireTarget(decision, playingThisLineage: playing != nil) else {
            return .unchanged
        }
        if let playing, let lineage = current.info.lineage {
            if lineage.contentSHA256 == target.candidate.contentSHA256 || !target.rank.isAbove(playing) {
                return .unchanged
            }
        }
        let built = try await rebuild(.followed(target), settings: settings, reason: reason)
        return .built(built.info)
    }

    /// Scan the models folder and fold the result into the follower, or
    /// join the check already running for the same lineage.
    private func checkLineage(_ followed: LichessBotFollowedLineage) async -> LichessBotLineageFollower.Decision {
        if let pending = pendingLineageCheck {
            if pending.followed == followed {
                return await pending.task.value
            }
            _ = await pending.task.value
        }
        let scanner = builder.folderScanner
        let previous = follower.cache
        let task = Task {
            let scan = await Self.scanResult(of: scanner, previous: previous)
            return self.recordCheck(scan, followed: followed)
        }
        pendingLineageCheck = (followed, task)
        let decision = await task.value
        if pendingLineageCheck?.task == task {
            pendingLineageCheck = nil
        }
        return decision
    }

    private func recordCheck(_ scan: Result<ModelFolderScan, Error>, followed: LichessBotFollowedLineage) -> LichessBotLineageFollower.Decision {
        // `current` is read here, after the scan returned, never before it:
        // a build may have finished meanwhile.
        let decision = follower.record(scan, followed: followed, notBelow: playingRank(of: followed), checkedAt: Date(), now: builder.time.now(), consequence: playingConsequence)
        decision.logLines.forEach(log)
        return decision
    }

    /// Whether the last check selected a file of `followed` that is not
    /// what plays: its build failed for a reason that isn't the file's (a
    /// network build), so the next poll retries it rather than waiting a
    /// whole check interval while the failure is cleared as "nothing due".
    private func newestKnownFileIsNotPlaying(of followed: LichessBotFollowedLineage) -> Bool {
        guard let status = follower.status, status.followed == followed, case .following(let newest) = status.outcome else {
            return false
        }
        return current.info.lineage?.followed != followed || current.info.lineage?.contentSHA256 != newest.contentSHA256
    }

    /// The playing generation's rank when it plays `followed`, else nil.
    private func playingRank(of followed: LichessBotFollowedLineage) -> ModelLineageRank? {
        guard current.info.sourceKind == .followLineage, let lineage = current.info.lineage, lineage.followed == followed else {
            return nil
        }
        return ModelLineageRank(segmentChain: lineage.segmentChain, segmentLocalStep: lineage.segmentLocalStep, recordedUnix: lineage.recordedUnix)
    }

    /// How a problem line ends while online: the generation that keeps
    /// playing.
    private var playingConsequence: String {
        "still playing generation \(current.info.generationID) (\(current.info.modelID) step \(current.info.trainingStep.map(String.init) ?? "-"))"
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
