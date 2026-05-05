import Foundation

/// Aggregate signature of every signal that should re-fire
/// `syncMenuCommandHubState()`. Originally each signal had its
/// own `.onChange` modifier; that produced a 13-deep generic
/// chain that the Swift type-checker spent ~1.2 s solving.
/// Packing them into one Equatable struct collapses the chain
/// to a single `.onChange(of: menuHubSignature)` and preserves
/// the same observation semantics — any field changing is a
/// `!=` on the whole struct, which fires the same handler.
struct MenuHubSignature: Equatable {
    var isBuilding: Bool
    var continuousPlay: Bool
    var continuousTraining: Bool
    var sweepRunning: Bool
    var realTraining: Bool
    var isArenaRunning: Bool
    var checkpointSaveInFlight: Bool
    var isTrainingOnce: Bool
    var isEvaluating: Bool
    var gameIsPlaying: Bool
    var hasNetwork: Bool
    var hasPendingLoadedSession: Bool
    var autoResumeStateVersion: Int
    /// Mirrors `arenaRecoveryInProgress` — fires the menu sync
    /// when the Tools menu's "Recover Arena History from Logs"
    /// item should toggle its disabled state.
    var arenaRecoveryInProgress: Bool
}
