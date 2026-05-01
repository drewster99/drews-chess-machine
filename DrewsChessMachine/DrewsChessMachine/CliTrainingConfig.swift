import Foundation

/// Hyperparameter set loaded from the `--parameters <file>` JSON on
/// launch. Every field is optional so a caller can supply a partial
/// file that overrides only the knobs they care about — unspecified
/// fields fall back to whatever the UI's `@AppStorage` values already
/// hold (which is either the last-session value or the compiled-in
/// default). The field names match the keys the user asked for in the
/// CLI spec and are the canonical set documented in the README; the
/// CodingKeys map the snake_case JSON keys to the camelCase Swift
/// properties so the file reads naturally in shell scripts.
///
/// `trainingTimeLimitSec` is special: it only takes effect when
/// `--output` is also supplied, because a time limit without an
/// output destination would silently terminate the process without
/// producing any artifact — which is worse than refusing to honor
/// it.
struct CliTrainingConfig: Codable, Sendable {
    // MARK: - Training loss / optimization

    var entropyBonus: Double? = nil
    var gradClipMaxNorm: Double? = nil
    var weightDecay: Double? = nil
    var K: Double? = nil
    var learningRate: Double? = nil
    var drawPenalty: Double? = nil
    /// Adam-family LR rule: when true, the optimizer feeds
    /// `learning_rate * sqrt(batch_size / 4096)` each step. The
    /// stored `learning_rate` stays the base value at the 4096 pivot.
    /// Weight decay is intentionally NOT batch-scaled (standard
    /// AdamW convention is to keep wd fixed across batch sizes).
    var sqrtBatchScalingForLR: Bool? = nil
    /// Linear LR warmup length in training steps. Per-step multiplier
    /// is `min(1, completedTrainSteps / lr_warmup_steps)`. Zero
    /// disables warmup entirely (full LR from step 0).
    var lrWarmupSteps: Int? = nil

    // MARK: - Self-play sampling schedule

    var selfPlayStartTau: Double? = nil
    var selfPlayTargetTau: Double? = nil
    var selfPlayTauDecayPerPly: Double? = nil

    // MARK: - Arena sampling schedule

    var arenaStartTau: Double? = nil
    var arenaTargetTau: Double? = nil
    var arenaTauDecayPerPly: Double? = nil

    // MARK: - Training loop shape

    var replayRatioTarget: Double? = nil
    var replayRatioAutoAdjust: Bool? = nil
    var selfPlayWorkers: Int? = nil
    var trainingStepDelayMs: Int? = nil
    var trainingBatchSize: Int? = nil
    var replayBufferCapacity: Int? = nil
    var replayBufferMinPositionsBeforeTraining: Int? = nil

    // MARK: - Arena cadence / outcome

    var arenaPromoteThreshold: Double? = nil
    var arenaGamesPerTournament: Int? = nil
    var arenaAutoIntervalSec: Double? = nil
    /// Number of arena games run concurrently per tournament. K>1
    /// enables batched inference on the per-network batchers; K=1 is
    /// the legacy serial behavior. Optional for back-compat with
    /// older parameters.json files.
    var arenaConcurrency: Int? = nil

    // MARK: - Probe

    var candidateProbeIntervalSec: Double? = nil

    // MARK: - Legal-mass collapse detector

    /// Illegal-mass threshold above which a probe counts as a "bad"
    /// reading. Default 0.999 (i.e. legal mass ≤ 0.001).
    var legalMassCollapseThreshold: Double? = nil
    /// Grace period (seconds) measured from the first observed SGD
    /// step before the detector starts firing. Default 300 s.
    var legalMassCollapseGraceSeconds: Double? = nil
    /// Number of consecutive probes (without legal-mass improvement)
    /// required to trip the abort. Default 5.
    var legalMassCollapseNoImprovementProbes: Int? = nil

    // MARK: - Time budget

    /// After this many wall-clock seconds inside the Play-and-Train
    /// session, the runtime writes the JSON snapshot to the
    /// `--output` path and calls `exit(0)`. No effect without
    /// `--output`. Nil = no deadline.
    var trainingTimeLimitSec: Double? = nil

    enum CodingKeys: String, CodingKey {
        case entropyBonus = "entropy_bonus"
        case gradClipMaxNorm = "grad_clip_max_norm"
        case weightDecay = "weight_decay"
        case K
        case learningRate = "learning_rate"
        case drawPenalty = "draw_penalty"
        case sqrtBatchScalingForLR = "sqrt_batch_scaling_lr"
        case lrWarmupSteps = "lr_warmup_steps"

        case selfPlayStartTau = "self_play_start_tau"
        case selfPlayTargetTau = "self_play_target_tau"
        case selfPlayTauDecayPerPly = "self_play_tau_decay_per_ply"

        case arenaStartTau = "arena_start_tau"
        case arenaTargetTau = "arena_target_tau"
        case arenaTauDecayPerPly = "arena_tau_decay_per_ply"

        case replayRatioTarget = "replay_ratio_target"
        case replayRatioAutoAdjust = "replay_ratio_auto_adjust"
        case selfPlayWorkers = "self_play_workers"
        case trainingStepDelayMs = "training_step_delay_ms"
        case trainingBatchSize = "training_batch_size"
        case replayBufferCapacity = "replay_buffer_capacity"
        case replayBufferMinPositionsBeforeTraining = "replay_buffer_min_positions_before_training"

        case arenaPromoteThreshold = "arena_promote_threshold"
        case arenaGamesPerTournament = "arena_games_per_tournament"
        case arenaAutoIntervalSec = "arena_auto_interval_sec"
        case arenaConcurrency = "arena_concurrency"

        case candidateProbeIntervalSec = "candidate_probe_interval_sec"

        case legalMassCollapseThreshold = "legal_mass_collapse_threshold"
        case legalMassCollapseGraceSeconds = "legal_mass_collapse_grace_seconds"
        case legalMassCollapseNoImprovementProbes = "legal_mass_collapse_no_improvement_probes"

        case trainingTimeLimitSec = "training_time_limit"
    }

    /// Load and decode a parameters JSON file from disk. Throws on
    /// I/O failure or malformed JSON — callers surface the error to
    /// the session log and abort the launch rather than silently
    /// running with defaults, because a misnamed field in the file
    /// is the kind of thing a scripted run would fail to notice
    /// until the end of a long session.
    static func load(from url: URL) throws -> CliTrainingConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(CliTrainingConfig.self, from: data)
    }

    /// Encode this config to JSON `Data`. Sorted keys so a UI-saved
    /// file diffs cleanly against an autotrain-saved file with the
    /// same values, and pretty-printed so a human can hand-edit it.
    /// Optional fields with nil values are omitted from the output
    /// (Swift's synthesized `encode(to:)` for Optional uses
    /// `encodeIfPresent`), so a partial config produces a partial
    /// file — matching the existing partial-override semantics on
    /// the load side.
    func encodeJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    /// Human-readable summary of the non-nil fields for the
    /// `[APP]` banner — shows the caller what the runtime actually
    /// picked up, so a typo in the key name (which makes the field
    /// stay nil) is immediately visible at launch.
    func summaryString() -> String {
        var parts: [String] = []
        func add<T>(_ label: String, _ v: T?) {
            guard let v else { return }
            parts.append("\(label)=\(v)")
        }
        add("entropy_bonus", entropyBonus)
        add("grad_clip_max_norm", gradClipMaxNorm)
        add("weight_decay", weightDecay)
        add("K", K)
        add("learning_rate", learningRate)
        add("draw_penalty", drawPenalty)
        add("sqrt_batch_scaling_lr", sqrtBatchScalingForLR)
        add("lr_warmup_steps", lrWarmupSteps)
        add("self_play_start_tau", selfPlayStartTau)
        add("self_play_target_tau", selfPlayTargetTau)
        add("self_play_tau_decay_per_ply", selfPlayTauDecayPerPly)
        add("arena_start_tau", arenaStartTau)
        add("arena_target_tau", arenaTargetTau)
        add("arena_tau_decay_per_ply", arenaTauDecayPerPly)
        add("replay_ratio_target", replayRatioTarget)
        add("replay_ratio_auto_adjust", replayRatioAutoAdjust)
        add("self_play_workers", selfPlayWorkers)
        add("training_step_delay_ms", trainingStepDelayMs)
        add("training_batch_size", trainingBatchSize)
        add("replay_buffer_capacity", replayBufferCapacity)
        add("replay_buffer_min_positions_before_training", replayBufferMinPositionsBeforeTraining)
        add("arena_promote_threshold", arenaPromoteThreshold)
        add("arena_games_per_tournament", arenaGamesPerTournament)
        add("arena_auto_interval_sec", arenaAutoIntervalSec)
        add("arena_concurrency", arenaConcurrency)
        add("candidate_probe_interval_sec", candidateProbeIntervalSec)
        add("legal_mass_collapse_threshold", legalMassCollapseThreshold)
        add("legal_mass_collapse_grace_seconds", legalMassCollapseGraceSeconds)
        add("legal_mass_collapse_no_improvement_probes", legalMassCollapseNoImprovementProbes)
        add("training_time_limit", trainingTimeLimitSec)
        return parts.isEmpty ? "(empty)" : parts.joined(separator: " ")
    }
}
