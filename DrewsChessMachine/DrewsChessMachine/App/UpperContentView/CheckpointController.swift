import SwiftUI

/// Owns the in-app checkpoint subsystem, lifted out of `UpperContentView`.
///
/// **This is the first half of the extraction.** Stage 3c part 1 (this file)
/// hosts the checkpoint **status display** (`checkpointStatusMessage` /
/// `checkpointStatusKind` + the auto-clear timer + the `[CHECKPOINT-ERR]` echo)
/// and the **slow-save watchdog** that flips a still-running save to a
/// `.slowProgress` row after 10 s. The big save / load / segments / periodic-
/// save logic is still on `UpperContentView` and calls into this controller via
/// `checkpoint.setCheckpointStatus(_:kind:)` / `.startSlowSaveWatchdog(label:)`
/// / `.cancelSlowSaveWatchdog()` / `.checkpointSaveInFlight = …`. Part 2 of the
/// extraction will move the save/load methods into this same controller.
@MainActor
@Observable
final class CheckpointController {

    /// Status row driven by the save/load paths. Auto-clears after a
    /// kind-dependent lifetime via the `Task { … Task.sleep }` in `setCheckpointStatus`.
    private(set) var checkpointStatusMessage: String?
    private(set) var checkpointStatusKind: CheckpointStatusKind = .progress

    /// True while a save is in flight, so the slow-save watchdog can tell a
    /// completed-fast save from one that's actually stuck on disk. Set / cleared
    /// by the save paths (still on `UpperContentView`) at their entry / exit
    /// points. `var` (not `private(set)`) so those external sites can write it.
    var checkpointSaveInFlight: Bool = false

    /// Handle to the in-flight slow-save watchdog `Task`. `private` — the only
    /// way to start / stop one is through the `startSlowSaveWatchdog(label:)` /
    /// `cancelSlowSaveWatchdog()` methods, which keep the cancel + nil-out
    /// invariants together.
    private var slowSaveWatchdogTask: Task<Void, Never>?

    /// Slow-save watchdog deadline. If a save has not completed
    /// within this many seconds of starting, the status row flips to
    /// `.slowProgress` and a `[CHECKPOINT-WARN]` line is logged
    /// exactly once per save (no progressive warnings — completion
    /// will eventually flip the row to success/error and restore
    /// normal styling). Calibrated to the typical save cost: a
    /// healthy session save (two ~10 MB `.dcmmodel` files plus a 35
    /// MB replay buffer at 500k positions) takes well under a second
    /// on SSD; 10 s leaves headroom for the post-promotion path's
    /// `.utility`-priority detached task to be scheduled under load
    /// without firing false-positive warnings, while still surfacing
    /// genuinely stuck saves promptly.
    nonisolated static let slowSaveWatchdogSeconds: Int = 10

    /// Surface a status message on the checkpoint status row, auto-clearing
    /// after a kind-dependent lifetime so a transient save success line doesn't
    /// linger past usefulness. Errors are also echoed to the session log
    /// (`[CHECKPOINT-ERR]`) so a 12-s on-screen line that auto-clears is still
    /// recoverable from the persistent log file. Success lifetime is 20 s — long
    /// enough for the user to glance up and confirm the save actually landed —
    /// versus 6 s for progress lines and 12 s for errors.
    func setCheckpointStatus(_ message: String, kind: CheckpointStatusKind) {
        checkpointStatusMessage = message
        checkpointStatusKind = kind
        // Always echo errors to the session log so a transient on-screen
        // error message that auto-clears in 12 seconds is still
        // recoverable from the persistent log file. (Some callsites
        // also log their own more-detailed [CHECKPOINT] line — minor
        // duplication is fine; visibility is the priority.)
        if kind == .error {
            SessionLogger.shared.log("[CHECKPOINT-ERR] \(message)")
        }
        // Auto-clear after a kind-dependent lifetime. Grabs the
        // current message at schedule time so a later message isn't
        // wiped out by an earlier one's timer.
        let snapshotMessage = message
        let lifetimeSeconds: Int
        switch kind {
        case .progress: lifetimeSeconds = 6
        // Slow-save status persists noticeably longer than a normal
        // progress line — the user is presumably waiting on it, and a
        // 6-second auto-clear in the middle of a stuck save would just
        // leave them confused about whether anything is still happening.
        case .slowProgress: lifetimeSeconds = 120
        case .success: lifetimeSeconds = 20
        case .error: lifetimeSeconds = 12
        }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(lifetimeSeconds))
            if self.checkpointStatusMessage == snapshotMessage {
                self.checkpointStatusMessage = nil
                self.checkpointStatusKind = .progress
            }
        }
    }

    /// Start a watchdog that warns the user if the save tagged
    /// `label` has not completed within `slowSaveWatchdogSeconds`.
    /// Every save path's completion branch must call `cancelSlowSaveWatchdog()`
    /// so a fast save's watchdog body never runs. Calling this while a previous
    /// watchdog is still pending cancels the previous one — only one save can
    /// be in flight at a time, and the most recent label is what should appear
    /// if it stalls.
    func startSlowSaveWatchdog(label: String) {
        slowSaveWatchdogTask?.cancel()
        let deadline = Self.slowSaveWatchdogSeconds
        slowSaveWatchdogTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(deadline))
            } catch {
                // The save completed before the deadline — its completion path
                // called `cancelSlowSaveWatchdog()`, which cancelled this Task.
                // `Task.sleep` throws `CancellationError`. Exit silently; the
                // fast-save case is the common one.
                return
            }
            if Task.isCancelled { return }
            // If the save already finished and emitted a final success/error
            // status, don't clobber it. We only flip to .slowProgress if the
            // row still shows the original "Saving…" line.
            guard self.checkpointSaveInFlight else { return }
            SessionLogger.shared.log(
                "[CHECKPOINT-WARN] \(label) still running after \(deadline)s — disk busy or replay buffer large?"
            )
            self.setCheckpointStatus(
                "Saving \(label)… (still running, \(deadline)s+)",
                kind: .slowProgress
            )
        }
    }

    /// Cancel the slow-save watchdog if any. Safe to call on any
    /// completion path — including success, error, and timeout
    /// branches that don't involve `slowSaveWatchdogTask` directly.
    func cancelSlowSaveWatchdog() {
        slowSaveWatchdogTask?.cancel()
        slowSaveWatchdogTask = nil
    }

    // MARK: - Parameter import / export (Stage 3c part 2a)

    /// Drives the Load Parameters file importer sheet (File menu →
    /// Load Parameters…). Loads a parameters JSON file with the same
    /// shape as the CLI `--parameters` flag and applies every named
    /// field as an override on top of the currently-effective values.
    var showingLoadParametersImporter: Bool = false

    /// Drives the Save Parameters file exporter sheet (File menu →
    /// Save Parameters…). Set to `true` after `parametersDocumentForExport`
    /// has been populated with a freshly-encoded snapshot of the
    /// current configuration.
    var showingSaveParametersExporter: Bool = false

    /// Pre-encoded JSON document handed to the Save Parameters file
    /// exporter. Built on the main actor at the moment the user
    /// invokes the menu item, so the encoded values reflect the
    /// session's state at that instant rather than at file-save time
    /// (which can be seconds later if the user takes a while to pick
    /// a destination).
    var parametersDocumentForExport: CliParametersDocument?

    /// Wired by `UpperContentView` to apply a `CliTrainingConfig` over the
    /// currently-effective parameters (mirroring the launch-time `--parameters`
    /// flag's behavior) and return the list of fields that actually changed
    /// (label / before / after triples for the user-visible summary line).
    var onApplyOverrides: (CliTrainingConfig) -> [(label: String, before: String, after: String)] = { _ in [] }

    /// File menu > Load Parameters… handler. Decodes the picked JSON
    /// file as a `CliTrainingConfig` and applies every named field on
    /// top of the currently-effective configuration. Mirrors the
    /// launch-time `--parameters` flag's behavior exactly, so the
    /// `[APP] --parameters override: …` log lines emitted by
    /// `applyCliConfigOverrides` show up in the session log identically
    /// whether the file was loaded at launch or via this menu item.
    func handleLoadParametersPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            setCheckpointStatus(
                "Load Parameters cancelled: \(error.localizedDescription)",
                kind: .error
            )
        case .success(let urls):
            guard let url = urls.first else { return }
            let needsAccess = url.startAccessingSecurityScopedResource()
            defer {
                if needsAccess { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let cfg = try CliTrainingConfig.load(from: url)
                SessionLogger.shared.log(
                    "[BUTTON] Load Parameters from \(url.lastPathComponent): \(cfg.summaryString())"
                )
                let changes = onApplyOverrides(cfg)
                // Surface both the count and the field labels in the
                // status row. `applyCliConfigOverrides` already logs
                // a per-field `[APP] --parameters override: …` line
                // for each entry plus a summary; this row is the
                // user-visible mirror of that summary so they don't
                // have to grep the session log to know what landed.
                if changes.isEmpty {
                    setCheckpointStatus(
                        "Loaded \(url.lastPathComponent): no parameters changed",
                        kind: .success
                    )
                } else {
                    let labels = changes.map(\.label).joined(separator: ", ")
                    setCheckpointStatus(
                        "Loaded \(url.lastPathComponent): \(changes.count) parameter\(changes.count == 1 ? "" : "s") changed (\(labels))",
                        kind: .success
                    )
                }
            } catch {
                setCheckpointStatus(
                    "Load Parameters failed: \(error.localizedDescription)",
                    kind: .error
                )
                SessionLogger.shared.log(
                    "[CHECKPOINT-ERR] Load Parameters from \(url.lastPathComponent) failed: \(error.localizedDescription)"
                )
            }
        }
    }

    /// File menu > Save Parameters… handler. Builds a fully-populated
    /// `CliTrainingConfig` from the current `TrainingParameters.shared`
    /// values, encodes it to JSON, stashes the bytes in
    /// `parametersDocumentForExport`, and triggers the file exporter.
    /// The exporter UI handles destination selection; on completion,
    /// `handleSaveParametersExportResult` logs success/failure.
    func handleSaveParametersMenuAction() {
        do {
            let snap = TrainingParameters.shared.snapshot().rawValueMap()
            var dict: [String: Any] = [:]
            for (id, raw) in snap {
                switch raw {
                case .bool(let x): dict[id] = x
                case .int(let x): dict[id] = x
                case .double(let x): dict[id] = x
                }
            }
            let data = try JSONSerialization.data(
                withJSONObject: dict,
                options: [.prettyPrinted, .sortedKeys]
            )
            parametersDocumentForExport = CliParametersDocument(data: data)
            showingSaveParametersExporter = true
            SessionLogger.shared.log("[BUTTON] Save Parameters")
        } catch {
            setCheckpointStatus(
                "Save Parameters failed (encode): \(error.localizedDescription)",
                kind: .error
            )
        }
    }

    // MARK: - Session identity / segments (Stage 3c part 2b)

    /// Stable ID for the current Play-and-Train session — minted on start,
    /// inherited on resume. Threaded into `[STATS]` / `[ARENA]` / `[SAVE]` log
    /// lines and the session save path so per-launch state can be correlated
    /// across files.
    var currentSessionID: String?

    /// Wall-clock instant of the most recent successful save of any flavor.
    /// `nil` until a save lands; shown in the status bar.
    var lastSavedAt: Date?

    /// Wall-clock at which the current session started — used to derive the
    /// elapsed-training counter for back-dating on resume.
    var currentSessionStart: Date?

    /// Closed training segments accumulated this session. Appended on Stop,
    /// pre-save (so the snapshot includes the in-flight segment with a clean
    /// close), and session-end.
    var completedTrainingSegments: [SessionCheckpointState.TrainingSegment] = []

    /// `nil` when no Play-and-Train run is currently active in-memory; non-nil
    /// while a run is in progress. Constructed in `beginActiveTrainingSegment`
    /// and consumed in `closeActiveTrainingSegment`.
    var activeSegmentStart: ActiveSegmentStart?

    /// Training-step count at the moment the current segment started. Read by
    /// the live `Run Totals` rate display so a resumed session shows
    /// segment-local rates, not lifetime ones over post-resume time.
    var trainingStepsAtSegmentStart: Int = 0

    /// In-memory record of an active training segment's start metadata. Kept on
    /// the controller (rather than synthesized at close time) so a build /
    /// session resume that lands in the middle of a segment still preserves
    /// the original starting counter snapshots.
    struct ActiveSegmentStart {
        let startUnix: Int64
        let startDate: Date
        let startingTrainingStep: Int
        let startingTotalPositions: Int
        let startingSelfPlayGames: Int
        let buildNumber: Int?
        let buildGitHash: String?
        let buildGitDirty: Bool?
    }

    /// Wired by `UpperContentView` to read `trainingStats?.steps` at call time.
    var trainingStepsProvider: () -> Int? = { nil }
    /// Wired to read `replayBuffer?.totalPositionsAdded` at call time.
    var totalPositionsAddedProvider: () -> Int? = { nil }
    /// Wired to read `parallelStats?.selfPlayGames` at call time.
    var selfPlayGamesProvider: () -> Int? = { nil }
    /// Wired to read `trainingBox?.snapshot()` at close-segment time so the
    /// segment record captures the closing entropy / loss / gNorm.
    var trainingBoxSnapshotProvider: () -> TrainingLiveStatsBox.Snapshot? = { nil }

    /// Begin a new training segment when Play-and-Train starts.
    /// Captures starting counter snapshots and the active build/git
    /// metadata so the resulting segment can be attributed to a
    /// specific code version after-the-fact.
    func beginActiveTrainingSegment() {
        let now = Date()
        let bufferAdded = totalPositionsAddedProvider() ?? 0
        let selfPlayGames = selfPlayGamesProvider() ?? 0
        let trainingStep = trainingStepsProvider() ?? 0
        activeSegmentStart = ActiveSegmentStart(
            startUnix: Int64(now.timeIntervalSince1970),
            startDate: now,
            startingTrainingStep: trainingStep,
            startingTotalPositions: bufferAdded,
            startingSelfPlayGames: selfPlayGames,
            buildNumber: BuildInfo.buildNumber,
            buildGitHash: BuildInfo.gitHash,
            buildGitDirty: BuildInfo.gitDirty
        )
        SessionLogger.shared.log(
            "[SEGMENT] start (segment #\(completedTrainingSegments.count + 1)) "
            + "step=\(activeSegmentStart?.startingTrainingStep ?? 0) "
            + "build=\(BuildInfo.buildNumber)"
        )
    }

    /// Close the in-progress segment with current end-of-segment
    /// counters and append it to `completedTrainingSegments`. Idempotent
    /// — if no segment is active, returns silently. Called from Stop,
    /// from the save path, and from session-end. `reason` is only used
    /// for the log line; the segment data itself is reason-agnostic.
    func closeActiveTrainingSegment(reason: String) {
        guard let start = activeSegmentStart else { return }
        let now = Date()
        let endUnix = Int64(now.timeIntervalSince1970)
        let durationSec = max(0, now.timeIntervalSince(start.startDate))
        let liveSnap = trainingBoxSnapshotProvider()
        let bufferAdded = totalPositionsAddedProvider() ?? start.startingTotalPositions
        let endLoss: Double? = {
            guard let p = liveSnap?.rollingPolicyLoss,
                  let v = liveSnap?.rollingValueLoss else { return nil }
            return p + v
        }()
        let segment = SessionCheckpointState.TrainingSegment(
            startUnix: start.startUnix,
            endUnix: endUnix,
            durationSec: durationSec,
            startingTrainingStep: start.startingTrainingStep,
            endingTrainingStep: trainingStepsProvider() ?? start.startingTrainingStep,
            startingTotalPositions: start.startingTotalPositions,
            endingTotalPositions: bufferAdded,
            startingSelfPlayGames: start.startingSelfPlayGames,
            endingSelfPlayGames: selfPlayGamesProvider() ?? start.startingSelfPlayGames,
            buildNumber: start.buildNumber,
            buildGitHash: start.buildGitHash,
            buildGitDirty: start.buildGitDirty,
            endPolicyEntropy: liveSnap?.rollingPolicyEntropy,
            endLossTotal: endLoss,
            endGradNorm: liveSnap?.rollingGradGlobalNorm
        )
        completedTrainingSegments.append(segment)
        activeSegmentStart = nil
        SessionLogger.shared.log(
            String(format: "[SEGMENT] close (%@) duration=%.1fs steps=%d -> %d positions=%d -> %d",
                   reason,
                   durationSec,
                   segment.startingTrainingStep,
                   segment.endingTrainingStep,
                   segment.startingTotalPositions,
                   segment.endingTotalPositions)
        )
    }

    /// Total active training wall-time across all segments, including the
    /// currently-running one if any. Excludes any time when training was
    /// stopped — sum of segment durations only.
    var cumulativeActiveTrainingSec: Double {
        let completed = completedTrainingSegments.reduce(0.0) { $0 + $1.durationSec }
        let active = activeSegmentStart.map { Date().timeIntervalSince($0.startDate) } ?? 0
        return completed + max(0, active)
    }

    /// Total run count: segments closed + 1 if a run is currently active.
    /// Useful for "this session has had N runs."
    var cumulativeRunCount: Int {
        completedTrainingSegments.count + (activeSegmentStart != nil ? 1 : 0)
    }

    /// Completion handler for the Save Parameters file exporter.
    /// Logs success or failure to the session log; user-visible
    /// status appears in the checkpoint status row.
    func handleSaveParametersExportResult(_ result: Result<URL, Error>) {
        parametersDocumentForExport = nil
        switch result {
        case .success(let url):
            setCheckpointStatus(
                "Saved parameters to \(url.lastPathComponent)",
                kind: .success
            )
            SessionLogger.shared.log(
                "[CHECKPOINT] Saved parameters: \(url.lastPathComponent)"
            )
        case .failure(let error):
            // SwiftUI's file exporter surfaces user-cancellation as
            // a failure with `.userCancelled` — don't treat that as
            // an error in the UI. Only real I/O failures get the
            // red status.
            if let cocoa = error as? CocoaError, cocoa.code == .userCancelled {
                return
            }
            setCheckpointStatus(
                "Save Parameters failed: \(error.localizedDescription)",
                kind: .error
            )
            SessionLogger.shared.log(
                "[CHECKPOINT-ERR] Save Parameters failed: \(error.localizedDescription)"
            )
        }
    }
}
