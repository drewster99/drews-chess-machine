import SwiftUI

/// Owns long-lived session networks, lifted out of `UpperContentView` as the
/// first slice (Stage 4a) of the session-lifecycle decomposition.
///
/// **Scope of this slice.** `SessionController` holds the three life-of-app
/// inference networks the arena and the candidate-test probe run against
/// (`candidateInferenceNetwork` / `arenaChampionNetwork` / `probeInferenceNetwork`,
/// plus the probe's `ChessRunner`), and the `performBuild()` static used by the
/// build paths. These are the network pieces that have no entanglement with the
/// rest of `UpperContentView`'s `@State` — they're lazily built on the first
/// Play-and-Train start and never torn down, and the view only reads them or
/// nil-coalesces a lazy build.
///
/// **Deliberately still on the view (for now).** The champion `network` + its
/// `runner`, `networkStatus`, `isBuilding`, and the `buildNetwork()` /
/// `ensureChampionBuilt()` flow stay on `UpperContentView` this slice: the
/// champion-network property is named `network` and referenced ~150 places, and
/// moving it cleanly wants a property rename first so a mechanical `network` →
/// `session.network` rewrite can't collide with `network:` argument labels.
/// That, plus the trainer / arena / parallel-stats buckets and the two giant
/// orchestration methods (`startRealTraining`, `runArenaParallel`), land in
/// follow-up Stage 4 slices. At that point the popover / auto-resume /
/// checkpoint controllers can take `weak var session` to drop their own
/// view-capturing closures.
@MainActor
@Observable
final class SessionController {

    // MARK: - Inference networks (life-of-app caches)

    /// Inference-mode network used as the arena's "candidate side" player. The
    /// trainer's current SGD weights are copied into it at each arena start so
    /// the candidate plays a coherent, stable snapshot. Built lazily on the
    /// first Play-and-Train start and cached for the life of the app.
    var candidateInferenceNetwork: ChessMPSNetwork?

    /// Inference-mode network dedicated to the candidate-test probe — kept
    /// separate from `candidateInferenceNetwork` so the probe doesn't have to
    /// pause for the whole duration of every arena (which produced a visible
    /// discontinuity in the probe trajectory across arena boundaries). Built
    /// lazily, cached for the app's life.
    var probeInferenceNetwork: ChessMPSNetwork?

    /// `ChessRunner` wrapping `probeInferenceNetwork`, used by
    /// `fireCandidateProbeIfNeeded` via the same `performInference` path as the
    /// forward-pass demo.
    var probeRunner: ChessRunner?

    /// Inference-mode network holding a snapshot of the champion's weights for
    /// the arena's "champion side" — copied once at arena start so the live
    /// champion stays free for continuous self-play during the tournament.
    /// Built lazily, cached for the app's life.
    var arenaChampionNetwork: ChessMPSNetwork?

    // MARK: - Parallel-worker stats / diversity (Stage 4b)

    /// Live-progress snapshot from the parallel self-play workers, mirrored
    /// from `parallelWorkerStatsBox` by the UI heartbeat. `nil` outside of a
    /// Play-and-Train session.
    var parallelStats: ParallelWorkerStatsBox.Snapshot?

    /// Lock-protected counter box shared across the parallel self-play and
    /// training worker tasks. Workers call `recordSelfPlayGame` /
    /// `recordTrainingStep`; the heartbeat polls `snapshot()` and mirrors into
    /// `parallelStats`. Created on Play-and-Train start, `nil` otherwise.
    var parallelWorkerStatsBox: ParallelWorkerStatsBox?

    /// Rolling-window game-diversity tracker for self-play. Fed by every
    /// self-play worker at game end; snapshot polled by the heartbeat for
    /// display and by the stats logger for `[STATS]` lines. `nil` outside a
    /// Play-and-Train session.
    var selfPlayDiversityTracker: GameDiversityTracker?

    // MARK: - Arena coordination boxes (Stage 4b)

    /// Cancellation-aware flag set while an arena tournament is in flight. The
    /// candidate-test probe checks this and skips firing so probe and arena
    /// never contend on the candidate inference network. `nil` between
    /// Play-and-Train sessions.
    var arenaActiveFlag: ArenaActiveFlag?

    /// Trigger inbox the arena coordinator polls — set by the training
    /// worker's auto-interval check and by the Run Arena button. `nil` between
    /// Play-and-Train sessions.
    var arenaTriggerBox: ArenaTriggerBox?

    /// User-override inbox for an in-flight arena. The Abort / Promote buttons
    /// (visible only while an arena is running) write to this box;
    /// `runArenaParallel` polls it to break the game loop early and to branch
    /// on promote-vs-no-promote once the driver returns. `nil` between
    /// Play-and-Train sessions.
    var arenaOverrideBox: ArenaOverrideBox?

    /// `true` while an arena is running — mirror of `arenaActiveFlag` the
    /// heartbeat maintains for UI purposes (disabling Run Arena, suppressing
    /// on-screen probe activity).
    var isArenaRunning: Bool = false

    // MARK: - Build

    /// The actual network construction. Runs on a detached `.userInitiated`
    /// task at the call sites (MPSGraph build is long synchronous work), so
    /// this is `nonisolated`.
    nonisolated static func performBuild() -> Result<ChessMPSNetwork, Error> {
        Result { try ChessMPSNetwork(.randomWeights) }
    }
}
