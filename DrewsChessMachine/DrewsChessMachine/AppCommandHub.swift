import Foundation
import Observation

/// Bridge between the top-level `.commands { ... }` menu DSL (which
/// lives on the `App` / `Scene`) and `ContentView` (which owns all
/// the state and action functions the menus need to trigger). SwiftUI
/// does not give menu commands direct access to view state, so we
/// route through this `@Observable` hub: `ContentView` assigns its
/// action functions into the closure slots on `.onAppear`, and a
/// small sync pushes the subset of `ContentView` state that governs
/// enable/disable logic into the hub's mirrored flags. The menu
/// builder reads flags for `.disabled` modifiers and calls closures
/// for button actions.
///
/// State-flag fan-out is deliberate rather than piping the whole
/// `ContentView` in: it keeps the menu DSL from needing to reach
/// into any private computed state, and it means the hub compiles
/// on its own without dragging `ChessNetwork` / `ChessTrainer`
/// headers into the app-level scene definition.
@Observable
@MainActor
final class AppCommandHub {

    // MARK: - Mirrored state flags

    /// `true` once the inference network has been built (or loaded).
    var networkReady: Bool = false
    /// Matches `ContentView.isBusy` — any long-running activity.
    var isBusy: Bool = false
    /// Mid-build network init.
    var isBuilding: Bool = false
    /// A Play Game / Play Continuous game is in flight.
    var gameIsPlaying: Bool = false
    /// Play Continuous loop active.
    var continuousPlay: Bool = false
    /// Train Continuous loop active.
    var continuousTraining: Bool = false
    /// Batch-size sweep active.
    var sweepRunning: Bool = false
    /// Play-and-Train session active.
    var realTraining: Bool = false
    /// Arena tournament in flight.
    var isArenaRunning: Bool = false
    /// A checkpoint save is in flight (guards against overlapping
    /// saves — same semantics as the inline button's disabled gate).
    var checkpointSaveInFlight: Bool = false
    /// A session was loaded but its Play-and-Train has not yet been
    /// resumed. Toggles the Train menu item label between "Play and
    /// Train" (cold start) and "Continue Training" (resume).
    var pendingLoadedSessionExists: Bool = false

    /// `true` when there is a `LastSessionPointer` on disk whose
    /// target `.dcmsession` folder still exists, the app is not
    /// already mid-training, and no auto-resume load is in flight.
    /// Drives the File menu's "Resume training from autosave" item
    /// enable/disable state. Refreshed at the same cadence as the
    /// other command-hub mirrored flags (menu-command-hub sync).
    var canResumeFromAutosave: Bool = false

    // Chart-zoom gating. Disabled when no Play-and-Train session is
    // live (the chart grid is hidden), and at the zoom-in / zoom-out
    // extremes so the menu items dim at the boundaries.
    var chartZoomInAvailable: Bool = false
    var chartZoomOutAvailable: Bool = false
    var chartZoomAutoAvailable: Bool = false

    // MARK: - Action closures

    var buildNetwork: () -> Void = {}
    var runForwardPass: () -> Void = {}
    var playSingleGame: () -> Void = {}
    var startContinuousPlay: () -> Void = {}
    var trainOnce: () -> Void = {}
    var startContinuousTraining: () -> Void = {}
    var startRealTraining: () -> Void = {}
    var startSweep: () -> Void = {}
    var stopAnyContinuous: () -> Void = {}
    var runArena: () -> Void = {}
    var abortArena: () -> Void = {}
    var promoteCandidate: () -> Void = {}
    var saveSession: () -> Void = {}
    var saveChampion: () -> Void = {}
    var loadSession: () -> Void = {}
    var loadModel: () -> Void = {}
    /// File menu > Load Parameters… — opens a JSON file picker, then
    /// applies every field present in the file as an override on top
    /// of the currently-effective values (same semantics as the CLI
    /// `--parameters <file>` flag).
    var loadParameters: () -> Void = {}
    /// File menu > Save Parameters… — opens a JSON file save panel
    /// and writes the current configuration as a fully-populated
    /// parameters JSON file (same shape as the CLI `--parameters`
    /// input format).
    var saveParameters: () -> Void = {}
    /// File menu > Resume training from autosave. Auto-loads the
    /// most recently saved `.dcmsession` and immediately starts
    /// Play-and-Train on it. Same code path the launch-time
    /// countdown sheet uses.
    var resumeFromAutosave: () -> Void = {}
    var revealSaves: () -> Void = {}
    /// Run the engine diagnostics probe — encoder/decoder round-trips,
    /// repetition tracking, network forward-pass shape check. All
    /// output goes to the session log with `[DIAG]` prefix. Suitable
    /// for one-shot health checks after major code changes.
    var runEngineDiagnostics: () -> Void = {}
    /// One-shot probe: feed two distinctly different positions through
    /// the live champion network, log the L1 distance / max per-cell
    /// |Δ| / value-head Δ between their policy outputs, and pass/fail
    /// based on whether the outputs are meaningfully position-conditional
    /// or essentially identical (the latter would indicate the policy
    /// head has collapsed to a position-agnostic constant). Output is
    /// logged with `[DIAG]` prefix, same pattern as
    /// `runEngineDiagnostics`.
    var runPolicyConditioningDiagnostic: () -> Void = {}

    // Chart-zoom actions. Wired to the View menu's ⌘= / ⌘- items
    // and the Auto toggle. A manual in/out press flips auto off
    // and stamps the recency timer that controls the 1-hour
    // auto-re-engage; pressing Auto re-enables it directly.
    var chartZoomIn: () -> Void = {}
    var chartZoomOut: () -> Void = {}
    var chartZoomEnableAuto: () -> Void = {}
}
