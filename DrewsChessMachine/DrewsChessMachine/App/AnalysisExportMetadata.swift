import Foundation

/// Cross-cutting context embedded in every analysis export (replay-buffer,
/// value-head, and network-weight JSON), under the top-level
/// `exportMetadata` key.
///
/// The analyzers themselves are pure functions over a network or a buffer
/// — they have no idea how many SGD steps the trainer has taken, how the
/// network is shaped, or how long the session has run, because those
/// facts live on the live stats boxes / static arch constants the
/// `SessionController` can reach. So the controller snapshots this struct
/// once at export time (on the main actor) and stamps it onto each
/// analyzer `Result` before the JSON is written. When several analyses
/// are written in one `Run All` pass they all share the *same* snapshot,
/// so values line up exactly across the files produced together.
///
/// The blocks are organized by what they describe:
/// - `analyzedWeights` — the weights this file analyzed: whose (role and
///   model ID), the training step they were taken at, their architecture,
///   and the init reference the analyzer compared them with. Absent from an
///   export that analyzed no network (the replay buffer's).
/// - `build` — the binary that wrote the file.
/// - `model`, `selfPlay`, `training` — the SESSION at export time: its
///   champion and trainer IDs, its self-play volume (games the champions
///   generated) and its training progress. They describe neither the
///   analyzed weights nor their step; `analyzedWeights` does. With Swift's
///   synthesized `Codable`, a `nil` sub-block (or field) omits its key, so a
///   snapshot taken with no training run doesn't carry those sections.
struct AnalysisExportMetadata: Codable, Sendable {

    /// Bumped when the export schema changes in a way a downstream reader
    /// needs to branch on. v2 introduced the nested block layout.
    let schemaVersion: Int

    /// Build-time provenance — what binary produced this export.
    let build: Build

    /// The session's champion and trainer at export time.
    let model: Model

    /// The weights this file analyzed; nil for an export of no network.
    var analyzedWeights: AnalyzedWeights?

    /// The session's lifetime self-play volume (the champions' games, not
    /// the analyzed network's). `nil` when no self-play stats box exists.
    let selfPlay: SelfPlay?

    /// The session's training progress and training-loop config. `nil` when
    /// no trainer / live stats box exists.
    let training: Training?

    // MARK: - Analyzed weights

    struct AnalyzedWeights: Codable, Sendable {
        /// "champion" or "trainer".
        let role: String
        let modelID: String?
        /// The trainer step the analyzed weights were taken at; nil when not
        /// known (`trainingStepSource` says why).
        let trainingStep: Int?
        let trainingStepSource: String
        let takenAtISO8601: String
        let architecture: Architecture
        /// What the analyzer's "init" figures are measured against
        /// (`AnalysisInitReference.basisDescription`): trainables only; BN
        /// running statistics have none.
        let initReference: String

        init(_ snapshot: AnalyzedNetworkSnapshot) {
            role = snapshot.role.rawValue
            modelID = snapshot.modelID
            trainingStep = snapshot.trainingStep
            trainingStepSource = snapshot.trainingStepSource
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            takenAtISO8601 = iso.string(from: snapshot.takenAt)
            architecture = Architecture(snapshot.architecture)
            initReference = snapshot.initReference.basisDescription
        }
    }

    /// This session-level snapshot stamped with the weights a file analyzed.
    func describing(_ snapshot: AnalyzedNetworkSnapshot) -> AnalysisExportMetadata {
        var copy = self
        copy.analyzedWeights = AnalyzedWeights(snapshot)
        return copy
    }

    // MARK: - Build

    struct Build: Codable, Sendable {
        let buildNumber: Int
        let buildTimestamp: String
        let gitHash: String
        let gitBranch: String
        let gitIsDirty: Bool
    }

    // MARK: - Model

    struct Model: Codable, Sendable {
        /// `ModelID.description` of the session's champion, if one is loaded.
        let championModelID: String?
        /// `ModelID.description` of the session's trainer, if one exists.
        let trainerModelID: String?
    }

    // MARK: - Architecture

    struct Architecture: Codable, Sendable {
        /// Total persistent-tensor element count (`NetworkArchitecture.parameterCount`).
        let parameterCount: Int
        /// Every block in the tower, all groups summed (`NetworkArchitecture.numBlocks`).
        let numBlocks: Int
        /// The LAST group's width — the tower's output channels the heads read.
        let channels: Int
        /// The FIRST group's conv1 kernel.
        let convKernelSize: Int
        let inputPlanes: Int
        let boardSize: Int
        let policyChannels: Int
        let policySize: Int
        /// The first group's SE reduction ratio; omitted when that group has
        /// no SE, where the stored ratio means nothing.
        let seReductionRatio: Int?
        let valueHead: ValueHead
        /// Generated one-line summary (`NetworkArchitecture.architectureSummary`).
        let summary: String

        struct ValueHead: Codable, Sendable {
            let classes: Int
            let convChannels: Int
            let hiddenUnits: Int
        }

        init(_ arch: NetworkArchitecture) {
            parameterCount = arch.parameterCount
            // `numBlocks` is the whole tower, `channels` the last group's
            // width, `convKernelSize` / `seReductionRatio` the first group's;
            // mixed towers carry the full structure in `summary`.
            numBlocks = arch.numBlocks
            channels = arch.towerOutputChannels
            convKernelSize = arch.blockGroups[0].conv1KernelSize
            inputPlanes = arch.inputPlanes
            boardSize = arch.boardSize
            policyChannels = arch.policyChannels
            policySize = arch.policySize
            seReductionRatio = arch.blockGroups[0].seStyle == .none ? nil : arch.blockGroups[0].seReductionRatio
            valueHead = ValueHead(classes: arch.valueHeadClasses, convChannels: arch.valueHeadConvChannels,
                                  hiddenUnits: arch.valueHeadHiddenUnits)
            summary = arch.architectureSummary
        }
    }

    // MARK: - Self-play

    struct SelfPlay: Codable, Sendable {
        /// Games the session's champions generated (restored across
        /// resumes; matches `[STATS] spGames=`).
        let totalGames: Int
        /// Positions (plies) generated across the model's life
        /// (matches `[STATS] spMoves=`).
        let totalMoves: Int
        /// Games actually kept into the replay buffer after the
        /// draw-keep filter (`<= totalGames`; matches `spGamesEm=`).
        let emittedGames: Int
        /// Positions kept into the replay buffer (`<= totalMoves`;
        /// matches `spMovesEm=`).
        let emittedMoves: Int
    }

    // MARK: - Training

    struct Training: Codable, Sendable {
        /// The session's cumulative SGD steps at export time — restored
        /// across resumes. Not the analyzed weights' step
        /// (`AnalyzedWeights.trainingStep`).
        let trainingSteps: Int
        /// Lifetime active training wall-time (the status bar's "Active
        /// training time"): sum of training-segment durations + the
        /// active one. Excludes stopped time; restored across resumes.
        /// `nil` when no `CheckpointController` / segment history exists.
        let cumulativeTrainingSeconds: Double?
        /// Positions (plies) consumed per SGD step by the active
        /// Play-and-Train run — its run-start capture, which a settings edit
        /// during the run does not change. nil (omitted) when no run is
        /// active; `batchSizeSetting` is written instead.
        let batchSize: Int?
        /// The `training_batch_size` setting, which applies at the next
        /// Play-and-Train start; written only when no run is active, so it
        /// is never mistaken for a batch a run trained at. Added in v3.
        let batchSizeSetting: Int?
        /// Arena score a candidate must reach to be promoted
        /// (`TrainingParameters.shared.arenaPromoteThreshold`).
        let promoteThreshold: Double
        /// Current resident position count of the replay buffer the
        /// trainer samples from. `nil` when no buffer is loaded.
        let replayBufferPlies: Int?
    }

    /// Current schema version emitted by this build. v3: `training.batchSize`
    /// is the active run's and omitted with no run active, when
    /// `training.batchSizeSetting` holds the setting instead. v4: the
    /// top-level `architecture` (the champion's, on every file, with a
    /// hand-written `notes` line) is replaced by `analyzedWeights`, which
    /// describes the weights the file analyzed — their architecture, step and
    /// init reference; `seReductionRatio` is omitted for an SE-less group.
    /// v4 also changed the analyzer bodies: init figures come from the init
    /// reference (`initExact`, `driftFromInit` exact only); weight-analyzer
    /// section totals cover trainables, with `runningStatsL2Norm` apart;
    /// per-channel / per-plane / per-column init norms are per entry; the
    /// value head's `currentSoftmax` / `initialSoftmax` became
    /// `biasOnlySoftmax` / `initialBiasOnlySoftmax`. v5: BN running statistics
    /// have no init figures (`initL2Norm` null, `isRunningStatistic` true);
    /// `SectionSummary.initExact` and an `initExact` on the stem, conv and
    /// value-fc2 details flag init figures that include same-distribution
    /// draws; the value head's fc2 bias `initial` / `initialBiasOnlySoftmax` /
    /// `delta` are null unless exact; the numerics audit's `biasInitMean` is
    /// the reference's stored initial mean (in the model's compute dtype).
    /// v6: the architecture drops `architectureVersion` (the retired
    /// v3/v4/v5 display label), and its `summary` no longer starts with it.
    static let currentSchemaVersion = 6
}
