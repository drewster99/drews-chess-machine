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
/// The blocks are organized by lifetime and source:
/// - `build` / `architecture` — always present; pure build-time facts.
/// - `model` — model identities; fields nil when nothing is loaded.
/// - `selfPlay` / `training` — present only when a self-play / training
///   context exists. With Swift's synthesized `Codable`, a `nil`
///   sub-block (or field) omits its key entirely, so a metadata snapshot
///   taken with no training run simply doesn't carry those sections.
struct AnalysisExportMetadata: Codable, Sendable {

    /// Bumped when the export schema changes in a way a downstream reader
    /// needs to branch on. v2 introduced the nested block layout.
    let schemaVersion: Int

    /// Build-time provenance — what binary produced this export.
    let build: Build

    /// Model identities at export time.
    let model: Model

    /// Network architecture — derived entirely from compile-time
    /// constants, so it always describes the binary that wrote the file.
    let architecture: Architecture

    /// Lifetime self-play volume. `nil` when no self-play stats box
    /// exists (e.g. export taken before any run started).
    let selfPlay: SelfPlay?

    /// Training progress + key training-loop config. `nil` when no
    /// trainer / live stats box exists.
    let training: Training?

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
        /// `ModelID.description` of the live champion, if one is loaded.
        let championModelID: String?
        /// `ModelID.description` of the trainer, if one exists.
        let trainerModelID: String?
    }

    // MARK: - Architecture

    struct Architecture: Codable, Sendable {
        /// Human v-number for display (4 = pre-activation, 3 = post-activation).
        /// Identity is the embedded config / `summary`, not a hash (see plan §6).
        let architectureVersion: Int
        /// Total persistent-tensor element count (`NetworkArchitecture.parameterCount`).
        let parameterCount: Int
        let numBlocks: Int
        let channels: Int
        let convKernelSize: Int
        let inputPlanes: Int
        let boardSize: Int
        let policyChannels: Int
        let policySize: Int
        let seReductionRatio: Int
        let valueHead: ValueHead
        /// Generated one-line summary (`NetworkArchitecture.architectureSummary`).
        let summary: String
        /// Hand-maintained qualitative note; omitted when empty.
        let notes: String?

        struct ValueHead: Codable, Sendable {
            let classes: Int
            let convChannels: Int
            let hiddenUnits: Int
        }
    }

    // MARK: - Self-play

    struct SelfPlay: Codable, Sendable {
        /// Games generated across the model's life (restored across
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
        /// Cumulative SGD steps — restored across resumes.
        let trainingSteps: Int
        /// Lifetime active training wall-time (the status bar's "Active
        /// training time"): sum of training-segment durations + the
        /// active one. Excludes stopped time; restored across resumes.
        /// `nil` when no `CheckpointController` / segment history exists.
        let cumulativeTrainingSeconds: Double?
        /// Positions (plies) consumed per SGD step
        /// (`TrainingParameters.shared.trainingBatchSize`).
        let batchSize: Int
        /// Arena score a candidate must reach to be promoted
        /// (`TrainingParameters.shared.arenaPromoteThreshold`).
        let promoteThreshold: Double
        /// Current resident position count of the replay buffer the
        /// trainer samples from. `nil` when no buffer is loaded.
        let replayBufferPlies: Int?
    }

    /// Current schema version emitted by this build.
    static let currentSchemaVersion = 2
}
