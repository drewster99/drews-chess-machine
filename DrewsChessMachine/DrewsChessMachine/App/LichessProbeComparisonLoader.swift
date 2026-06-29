import AppKit
import Foundation
import UniformTypeIdentifiers

/// Modal file picker + JSON parser for the Lichess Probe Detail
/// window's "Compare…" button. Mirrors `LichessProbeExporter`'s
/// destination: the panel opens at
/// `Application Support/DrewsChessMachine/Performance/LichessProbes/`
/// so the user can re-load a snapshot they exported earlier without
/// hunting through Finder.
///
/// Schema versions 2 through 4 are accepted — the JSON shape changed
/// enough between v1 and v2 that supporting both would mean two
/// parsers, and nobody has v1 files in the wild yet. v3 added the
/// top-level `training_step`; v4 added `positions_trained`,
/// `active_training_sec`, `arena_count`, and `promotion_count`. All
/// of v3/v4's additions are top-level scalars that `LoadedPayload`
/// silently ignores, so v2/v3/v4 decode identically here. Other
/// versions surface a friendly error alert rather than a Codable
/// decoding crash.
@MainActor
enum LichessProbeComparisonLoader {

    /// Open a panel, parse the selected JSON, return the loaded
    /// comparison snapshot. Returns nil on user cancel OR on any
    /// parse / schema-version failure (with an error alert surfaced).
    static func loadFromFile() -> LichessProbeComparison? {
        let panel = NSOpenPanel()
        panel.title = "Choose a Lichess Probe JSON to compare"
        panel.prompt = "Compare"
        panel.allowedContentTypes = [.json]
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = LichessProbeExporter.performanceLichessProbesDir

        guard panel.runModal() == .OK, let url = panel.url else {
            SessionLogger.shared.log("[TACTICAL-LICHESS] compare cancelled")
            return nil
        }

        return load(from: url, announce: true)
    }

    /// Parse a previously-exported Lichess Probe JSON at a known path —
    /// no panel. Used by `loadFromFile` (manual Compare…) and by the
    /// Detail window's auto-compare. `announce` gates the error alerts:
    /// the auto path passes `false` so a missing/garbage file logs
    /// quietly instead of popping a warning.
    static func load(from url: URL, announce: Bool = true) -> LichessProbeComparison? {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            reportLoadError(announce, "Could not read file",
                "Failed to read \(url.lastPathComponent):\n\n\(error.localizedDescription)")
            return nil
        }

        let decoded: LichessProbeComparison.LoadedPayload
        do {
            decoded = try JSONDecoder().decode(
                LichessProbeComparison.LoadedPayload.self,
                from: data
            )
        } catch {
            reportLoadError(announce, "Could not parse file",
                "\(url.lastPathComponent) is not a valid Lichess Probe export.\n\n\(error.localizedDescription)")
            return nil
        }

        // v2/v3/v4 decode identically here (v3/v4's extra top-level
        // fields aren't in `LoadedPayload`); accept all three.
        guard (2...4).contains(decoded.schemaVersion) else {
            reportLoadError(announce, "Unsupported schema",
                "\(url.lastPathComponent) uses schema version \(decoded.schemaVersion). Only versions 2-4 are supported.")
            return nil
        }

        let comparison = LichessProbeComparison(payload: decoded, sourceURL: url)
        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] compare loaded: file=\(url.lastPathComponent) "
            + "puzzles=\(comparison.byPuzzleId.count) "
            + "model=\(decoded.modelLabel ?? "<unknown>") "
            + "tick=\(decoded.tickTimestamp)"
        )
        return comparison
    }

    /// Either pop a warning alert (manual path) or log quietly (auto path).
    private static func reportLoadError(_ announce: Bool, _ title: String, _ message: String) {
        if announce {
            presentLoaderError(title: title, message: message)
        } else {
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS] auto-compare \(title): "
                + message.replacingOccurrences(of: "\n", with: " ")
            )
        }
    }

    private static func presentLoaderError(title: String, message: String) {
        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] compare error: \(title) — \(message.replacingOccurrences(of: "\n", with: " | "))"
        )
        NonBlockingAlert.presentWarning(title: title, message: message)
    }
}
