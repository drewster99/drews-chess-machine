import AppKit
import Foundation

/// `SessionController`'s offline replay-buffer analyzer hook — wired to
/// the `Analyze Replay Buffer…` Debug menu item. Runs
/// `ReplayBufferAnalyzer.run` on the currently-loaded buffer, writes a
/// timestamped JSON file under `~/Library/Application Support/
/// DrewsChessMachine/Analyses/`, logs an `[ANALYSIS]` text-summary
/// block to the session log, and surfaces an NSAlert with a
/// Reveal-in-Finder action so the JSON file is one click away.
///
/// The analyzer's pass is sub-second on a 1 M-position buffer but
/// holds the buffer's lock for the duration, so it runs in a detached
/// Task to keep the main actor responsive even if the buffer's lock
/// has a long-running contender (e.g. an in-flight append from the
/// self-play emit fan-in). Alert + log surfacing hops back to the
/// main actor after the heavy work completes.
extension SessionController {

    /// Entry point invoked by the Debug menu item. Runs the analyzer
    /// against `replayBuffer`; if no buffer is loaded, surfaces an
    /// explanatory alert instead of silently doing nothing.
    func analyzeReplayBufferToFile() {
        SessionLogger.shared.log("[BUTTON] Analyze Replay Buffer")
        guard let buf = replayBuffer else {
            Self.presentAnalyzeAlert(
                title: "Analyze Replay Buffer",
                message: "No replay buffer is loaded. Start Play-and-Train or load a saved session first.",
                revealURL: nil
            )
            return
        }
        // Snapshot the network ref alongside the buffer so the detached
        // task can run the live-network entropy probe (analysis #7).
        // The network is `@unchecked Sendable` and the ref captured
        // here outlives the closure regardless of any concurrent
        // session changes. The trainer ref is captured for the same
        // reason — the entropy probe runs against a fresh inference
        // network loaded with trainer weights when one is available,
        // since the champion's policy is frozen between promotions
        // and would produce a misleading bit-stable entropy section.
        let netForEntropy = network
        let trainerForEntropy = trainer
        let modelLabel = netForEntropy?.identifier?.description ?? "<no-id>"

        Task.detached(priority: .utility) {
            // Off-main heavy walk + JSON write. Both the analyzer pass
            // and the file I/O are bounded — a sub-second analyzer pass
            // and a ~MB-sized JSON write — but neither belongs on the
            // main actor while it could be driving UI updates from the
            // training loop. When a network is available we also run
            // the stratified policy-entropy probe (a few extra seconds
            // of forward passes) so analysis #7 lands in the same JSON.
            let entropyProbe = await Self.buildTrainerEntropyProbeNetwork(
                trainer: trainerForEntropy
            )
            let result: ReplayBufferAnalyzer.Result
            do {
                if let probe = entropyProbe {
                    result = try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                        buffer: buf,
                        network: probe.network,
                        modelLabel: modelLabel,
                        entropyModelLabel: probe.label
                    )
                } else if let net = netForEntropy {
                    result = try await ReplayBufferAnalyzer.runWithPolicyEntropy(
                        buffer: buf,
                        network: net,
                        modelLabel: modelLabel
                    )
                } else {
                    result = ReplayBufferAnalyzer.run(
                        buffer: buf,
                        modelLabel: modelLabel
                    )
                }
            } catch {
                // Policy-entropy forward pass failed (e.g., MPSGraph
                // transient error). Fall back to the pure-buffer
                // analysis so we still produce a file — the [ANALYSIS]
                // log will note the entropy failure separately so the
                // user can see why section (7) is missing.
                SessionLogger.shared.log("[ANALYSIS] Policy-entropy probe failed: \(error). Falling back to pure-buffer analysis.")
                result = ReplayBufferAnalyzer.run(
                    buffer: buf,
                    modelLabel: modelLabel
                )
            }
            let summary = result.textSummary()
            let writeOutcome = Self.writeAnalysisJSON(
                result: result,
                modelLabel: modelLabel
            )

            await MainActor.run {
                // Log the text summary verbatim — each line gets the
                // SessionLogger's timestamp prefix so the [ANALYSIS]
                // block is grep-distinct from STATS / ARENA noise.
                SessionLogger.shared.log("[ANALYSIS] === Replay buffer analysis begin ===")
                for line in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                    SessionLogger.shared.log("[ANALYSIS] \(line)")
                }
                SessionLogger.shared.log("[ANALYSIS] === Replay buffer analysis end ===")

                switch writeOutcome {
                case .success(let url):
                    SessionLogger.shared.log("[ANALYSIS] Saved JSON: \(url.path)")
                    Self.presentAnalyzeAlert(
                        title: "Replay Buffer Analysis Complete",
                        message: """
                            Saved JSON to:
                            \(url.path)

                            A text summary was written to the session log under [ANALYSIS]; \
                            click Reveal in Finder to open the JSON in the output folder.
                            """,
                        revealURL: url
                    )
                case .failure(let err):
                    SessionLogger.shared.log("[ANALYSIS] JSON write failed: \(err)")
                    Self.presentAnalyzeAlert(
                        title: "Replay Buffer Analysis — JSON Write Failed",
                        message: """
                            The analyzer ran and a text summary was written to the session \
                            log, but writing the JSON file failed:

                            \(err.localizedDescription)
                            """,
                        revealURL: nil
                    )
                }
            }
        }
    }

    // MARK: - JSON write

    /// Encode `result` and write it to a timestamped JSON file under
    /// `CheckpointPaths.analysesDir`. Creates the directory on demand.
    /// The filename embeds both a timestamp and a sanitized form of
    /// the model label so consecutive analyses don't collide and
    /// remain attributable to a specific model snapshot.
    nonisolated private static func writeAnalysisJSON(
        result: ReplayBufferAnalyzer.Result,
        modelLabel: String
    ) -> Result<URL, Error> {
        let fm = FileManager.default
        let dir = CheckpointPaths.analysesDir
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            return .failure(error)
        }

        let stamp = filenameTimestamp()
        let safeModel = modelLabel
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: ":", with: "_")
        let filename = "replay_analysis_\(stamp)_\(safeModel).json"
        let url = dir.appendingPathComponent(filename)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            let data = try encoder.encode(result)
            try data.write(to: url, options: [.atomic])
            return .success(url)
        } catch {
            return .failure(error)
        }
    }

    /// Filesystem-safe timestamp for analysis output filenames. Format
    /// matches the project's other timestamped artifacts (session
    /// checkpoints under `Sessions/`, etc.) so files sort lexicographically
    /// by time.
    nonisolated private static func filenameTimestamp() -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        df.locale = Locale(identifier: "en_US_POSIX")
        return df.string(from: Date())
    }

    // MARK: - Alert + Reveal in Finder

    /// Present a modal NSAlert. If `revealURL` is non-nil, adds a
    /// "Reveal in Finder" button that opens Finder with the file
    /// selected; otherwise only the default OK button is shown.
    /// Must be called on the main actor (NSAlert is an AppKit type).
    @MainActor
    private static func presentAnalyzeAlert(
        title: String,
        message: String,
        revealURL: URL?
    ) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        if revealURL != nil {
            alert.addButton(withTitle: "Reveal in Finder")
        }
        let response = alert.runModal()
        if let url = revealURL,
           response == .alertSecondButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }
}
