import Foundation
import Observation

/// State for the Load Session picker sheet: scans the sessions
/// directory on a background queue, resolves a `SessionManifest` per
/// folder (embedded manifest → index cache → full extraction, in that
/// order of cost), and delivers rows incrementally so the first cold
/// scan populates progressively instead of beachballing.
///
/// Grouping is by saved-lineage tag ("runs are the mental model"):
/// each group is one run, newest run first, sessions within a run in
/// reverse-chronological order.
@MainActor
@Observable
final class SessionPickerModel {

    private(set) var manifests: [SessionManifest] = []
    private(set) var isScanning = false
    private(set) var scannedCount = 0
    private(set) var totalCount = 0
    var selectedID: String?

    /// The directory the current manifests came from — the single
    /// source of truth for turning a manifest back into a folder URL
    /// (Load, Reveal in Finder). Set by `beginScan`.
    private(set) var scanDirectory: URL?

    /// Generation token: bumping it orphans any in-flight scan so a
    /// re-opened sheet can rescan without racing the previous pass.
    private var scanGeneration = 0

    /// Heavy JSON parsing happens here, not on the cooperative pool —
    /// a legacy `session.json` runs ~16 MB and the first cold scan can
    /// touch ~80 of them.
    private static let indexQueue = DispatchQueue(
        label: "drewschess.session-index", qos: .userInitiated
    )

    var selectedManifest: SessionManifest? {
        guard let selectedID else { return nil }
        return manifests.first { $0.id == selectedID }
    }

    /// Absolute URL of a listed session's folder, derived from the
    /// scanned directory (nil before any scan).
    func folderURL(for manifest: SessionManifest) -> URL? {
        scanDirectory?.appendingPathComponent(manifest.folderName, isDirectory: true)
    }

    struct RunGroup: Identifiable {
        let id: String
        let lineageTag: String
        let architectureSummary: String?
        let newestDate: Date?
        let sessions: [SessionManifest]
    }

    /// Lineage-grouped view of `manifests`: groups ordered newest-run
    /// first, sessions within a group newest first.
    var groups: [RunGroup] {
        Self.makeGroups(from: manifests)
    }

    /// Pure grouping logic, separated from the scan state so the
    /// ordering rules are unit-testable (SessionPickerModelTests).
    static func makeGroups(from manifests: [SessionManifest]) -> [RunGroup] {
        let byTag = Dictionary(grouping: manifests) { $0.lineageTag ?? "?" }
        let built = byTag.map { tag, items -> RunGroup in
            let sorted = items.sorted {
                ($0.savedAt ?? .distantPast) > ($1.savedAt ?? .distantPast)
            }
            return RunGroup(
                id: tag,
                lineageTag: tag,
                architectureSummary: sorted.first?.architectureSummary,
                newestDate: sorted.first?.savedAt,
                sessions: sorted
            )
        }
        return built.sorted {
            ($0.newestDate ?? .distantPast) > ($1.newestDate ?? .distantPast)
        }
    }

    /// Kick off (or restart) a directory scan. Safe to call every time
    /// the sheet opens — warm scans are nearly instant (embedded
    /// manifests + index cache), and the generation token cancels
    /// delivery from any superseded pass.
    func beginScan(directory: URL) {
        scanGeneration += 1
        let generation = scanGeneration
        isScanning = true
        manifests = []
        scannedCount = 0
        totalCount = 0
        scanDirectory = directory

        Self.indexQueue.async {
            let fm = FileManager.default
            let folders = ((try? fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )) ?? [])
            .filter { $0.pathExtension == "dcmsession" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }  // newest names first

            Task { @MainActor [weak self] in
                guard let self, self.scanGeneration == generation else { return }
                self.totalCount = folders.count
                if folders.isEmpty { self.isScanning = false }
            }

            for url in folders {
                let manifest = SessionManifest.resolve(sessionFolder: url)
                Task { @MainActor [weak self] in
                    guard let self, self.scanGeneration == generation else { return }
                    self.manifests.append(manifest)
                    self.scannedCount += 1
                    // Compare against the captured count, not
                    // `totalCount` — that property is set by a separate
                    // main-actor task and is not guaranteed to have
                    // landed before the first row delivery.
                    if self.scannedCount >= folders.count {
                        self.isScanning = false
                    }
                }
            }
        }
    }
}
