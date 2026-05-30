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
/// Only schema v2 files are accepted — the JSON shape changed enough
/// between v1 and v2 that supporting both would mean two parsers, and
/// nobody has v1 files in the wild yet. Other versions surface a
/// friendly error alert rather than a Codable decoding crash.
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

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            presentLoaderError(
                title: "Could not read file",
                message: "Failed to read \(url.lastPathComponent):\n\n\(error.localizedDescription)"
            )
            return nil
        }

        let decoded: LichessProbeComparison.LoadedPayload
        do {
            decoded = try JSONDecoder().decode(
                LichessProbeComparison.LoadedPayload.self,
                from: data
            )
        } catch {
            presentLoaderError(
                title: "Could not parse file",
                message: """
                    \(url.lastPathComponent) is not a valid Lichess Probe export.

                    \(error.localizedDescription)
                    """
            )
            return nil
        }

        // v3 only adds the optional top-level `training_step`, which
        // `LoadedPayload` doesn't decode — so v2 and v3 files are read
        // identically here. Accept both.
        guard (2...3).contains(decoded.schemaVersion) else {
            presentLoaderError(
                title: "Unsupported schema",
                message: """
                    \(url.lastPathComponent) uses schema version \(decoded.schemaVersion).
                    Only versions 2-3 are supported.
                    """
            )
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

    private static func presentLoaderError(title: String, message: String) {
        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] compare error: \(title) — \(message.replacingOccurrences(of: "\n", with: " | "))"
        )
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        _ = alert.runModal()
    }
}
