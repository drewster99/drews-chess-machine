import AppKit
import Foundation

/// JSON exporter for the latest tick of the Lichess Probe set.
///
/// Writes a single timestamped file under
/// `~/Library/Application Support/DrewsChessMachine/Performance/LichessProbes/`
/// (the canonical destination — no NSSavePanel; one place to look for
/// every exported probe snapshot). Filename shape:
/// `LichessProbe_yyyy-MM-dd_HH-mm-ss_<hash>.json`, where `<hash>` is the
/// first 8 hex chars of a per-export UUID. The full UUID is also
/// embedded in the JSON under `export_id`, so filename ↔ content
/// identity is checkable.
///
/// Schema v4 — each `puzzles[]` entry is `{ puzzle: {...}, probe_result: {...} }`
/// for readability and to keep the "what the puzzle is" data lexically
/// separated from "how the network performed on it." Top-level metadata
/// includes the export UUID, the tick timestamp, the model label, the
/// trainer step the tick was taken at (`training_step`, added in v3; nil
/// when the tick predates any trainer), four training-progress fields
/// added in v4 (`positions_trained`, `active_training_sec`, `arena_count`,
/// `promotion_count` — mirroring the status-bar cells of the same name
/// at tick time), and the build that produced the data so an exported
/// file can stand alone for analysis later.
///
/// User flow: the Lichess Probe Detail window's "Export latest…"
/// button calls `exportLatest`. On success an NSAlert with Reveal in
/// Finder pops; on failure an error alert pops with the underlying
/// error message. Cancel doesn't exist — there's no save panel any
/// more, so the only "cancel" branch is "no tick recorded yet" which
/// logs + beeps without writing anything.
@MainActor
enum LichessProbeExporter {

    static func exportLatest(history: LichessProbeHistory) {
        SessionLogger.shared.log("[BUTTON] Export Lichess Probe Results")

        guard !history.latestPerPuzzleResults.isEmpty,
              let tickTimestamp = history.latestTickTimestamp else {
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS] export skipped: no tick recorded yet"
            )
            NSSound.beep()
            return
        }

        let exportID = UUID()
        let filename = filename(for: tickTimestamp, exportID: exportID)

        let dir = performanceLichessProbesDir
        do {
            try FileManager.default.createDirectory(
                at: dir,
                withIntermediateDirectories: true
            )
        } catch {
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS] export failed: could not create \(dir.path): \(error.localizedDescription)"
            )
            presentExportAlert(
                title: "Export failed",
                message: "Could not create the export folder:\n\n\(dir.path)\n\n\(error.localizedDescription)",
                revealURL: nil
            )
            return
        }

        let url = dir.appendingPathComponent(filename)

        do {
            let data = try buildJSON(
                history: history,
                tickTimestamp: tickTimestamp,
                exportID: exportID
            )
            try data.write(to: url, options: .atomic)
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS] export wrote \(data.count) bytes to \(url.path)"
            )
            presentExportAlert(
                title: "Export complete",
                message: """
                    Wrote \(history.latestPerPuzzleResults.count) puzzles \
                    (\(formattedSize(data.count))) to

                    \(url.path)

                    Click Reveal in Finder to open the containing folder \
                    with the file selected.
                    """,
                revealURL: url
            )
        } catch {
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS] export failed: \(error.localizedDescription)"
            )
            presentExportAlert(
                title: "Export failed",
                message: "Could not write the JSON file:\n\n\(url.path)\n\n\(error.localizedDescription)",
                revealURL: nil
            )
        }
    }

    // MARK: - Destination directory

    /// `~/Library/Application Support/DrewsChessMachine/Performance/LichessProbes/`.
    /// Created on demand at first export. Sibling of the existing
    /// `Sessions/` / `Models/` / `Analyses/` folders used by
    /// CheckpointManager and the analyzer commands.
    static var performanceLichessProbesDir: URL {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support
            .appendingPathComponent("DrewsChessMachine", isDirectory: true)
            .appendingPathComponent("Performance", isDirectory: true)
            .appendingPathComponent("LichessProbes", isDirectory: true)
    }

    // MARK: - Filename

    private static func filename(for tickTimestamp: Date, exportID: UUID) -> String {
        let stamp = filenameTimestampFormatter.string(from: tickTimestamp)
        let shortHash = exportID.uuidString.lowercased()
            .replacingOccurrences(of: "-", with: "")
            .prefix(8)
        return "LichessProbe_\(stamp)_\(shortHash).json"
    }

    private static let filenameTimestampFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return df
    }()

    private static let isoTimestampFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    // MARK: - NSAlert with Reveal in Finder

    private static func presentExportAlert(
        title: String,
        message: String,
        revealURL: URL?
    ) {
        NonBlockingAlert.presentInformational(
            title: title,
            message: message,
            revealURL: revealURL
        )
    }

    private static func formattedSize(_ bytes: Int) -> String {
        let kb = Double(bytes) / 1024.0
        if kb < 1024 {
            return String(format: "%.1f KB", kb)
        } else {
            return String(format: "%.2f MB", kb / 1024.0)
        }
    }

    // MARK: - JSON build

    private static func buildJSON(
        history: LichessProbeHistory,
        tickTimestamp: Date,
        exportID: UUID
    ) throws -> Data {
        let entries = history.latestPerPuzzleResults.map(buildPuzzleEntry(_:))
        let payload = ExportPayload(
            schemaVersion: 4,
            exportId: exportID.uuidString.lowercased(),
            generatedAt: isoTimestampFormatter.string(from: Date()),
            tickTimestamp: isoTimestampFormatter.string(from: tickTimestamp),
            modelLabel: history.latestTickModelLabel,
            trainingStep: history.latestTickTrainingStep,
            positionsTrained: history.latestTickPositionsTrained,
            activeTrainingSec: history.latestTickActiveTrainingSec,
            arenaCount: history.latestTickArenaCount,
            promotionCount: history.latestTickPromotionCount,
            appBuild: AppBuildBlock.current,
            probeCount: entries.count,
            puzzles: entries
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(payload)
    }

    private static func buildPuzzleEntry(_ result: ProbeResult) -> PuzzleEntry {
        let probe = result.probe
        let meta = LichessProbeData.metadata[probe.name]
        let expectedNotation = probe.acceptable
            .sorted(by: { $0.notation < $1.notation })
            .first?.notation
        let topMoves = result.topMoves.map { entry in
            TopMoveBlock(
                uci: uciString(entry.move),
                notation: entry.move.notation,
                prob: entry.prob
            )
        }
        let puzzle = PuzzleBlock(
            id: meta?.id ?? probe.name,
            theme: meta?.theme ?? probe.shortDescription,
            themeCategory: probe.category.rawValue,
            ratingGlicko2: meta?.rating,
            fen: meta?.fen,
            expectedMoveUci: meta?.bestMoveUci,
            expectedMoveNotation: expectedNotation
        )
        let probeResult = ProbeResultBlock(
            verdict: result.verdict.rawValue,
            expectedRank: result.expectedRank,
            expectedProb: result.expectedProb,
            legalCount: result.legalCount,
            illegalMass: result.illegalMass,
            legalEntropyNats: result.legalEntropyNats,
            uniformLegalEntropy: result.uniformLegalEntropy,
            valueWdl: ValueWDLBlock(
                win: result.valueWDL.win,
                draw: result.valueWDL.draw,
                loss: result.valueWDL.loss
            ),
            topMoves: topMoves
        )
        return PuzzleEntry(puzzle: puzzle, probeResult: probeResult)
    }

    /// Reconstruct UCI long-algebraic from a `ChessMove`. Delegates to
    /// `ChessMove.uci`, the single source of truth for the encoding
    /// shared with session-checkpoint probe persistence.
    private static func uciString(_ move: ChessMove) -> String {
        move.uci
    }

    // MARK: - Codable payloads

    private struct ExportPayload: Encodable {
        let schemaVersion: Int
        let exportId: String
        let generatedAt: String
        let tickTimestamp: String
        let modelLabel: String?
        /// Trainer step count at the moment the exported tick was
        /// recorded. nil when the tick ran before a trainer existed.
        let trainingStep: Int?
        /// Total positions consumed by the trainer at tick time —
        /// `training_step × trainingBatchSize`. Added in schema v4.
        /// nil if `training_step` is nil.
        let positionsTrained: Int?
        /// Cumulative active training wall-time in seconds at tick
        /// time. Added in schema v4. nil if no checkpoint controller
        /// was attached at tick time.
        let activeTrainingSec: Double?
        /// Total arena tournaments completed at tick time. Added in
        /// schema v4.
        let arenaCount: Int?
        /// Subset of `arena_count` where the candidate was promoted.
        /// Added in schema v4.
        let promotionCount: Int?
        let appBuild: AppBuildBlock
        let probeCount: Int
        let puzzles: [PuzzleEntry]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case exportId = "export_id"
            case generatedAt = "generated_at"
            case tickTimestamp = "tick_timestamp"
            case modelLabel = "model_label"
            case trainingStep = "training_step"
            case positionsTrained = "positions_trained"
            case activeTrainingSec = "active_training_sec"
            case arenaCount = "arena_count"
            case promotionCount = "promotion_count"
            case appBuild = "app_build"
            case probeCount = "probe_count"
            case puzzles
        }
    }

    /// Top-level "what app built this snapshot" block — passes through
    /// the static `BuildInfo` fields the auto-generated build script
    /// produces so an exported file is reproducible without consulting
    /// the live app.
    private struct AppBuildBlock: Encodable {
        let buildNumber: Int
        let buildDate: String
        let buildTimestamp: String
        let gitHash: String
        let gitBranch: String
        let gitDirty: Bool
        let summary: String

        enum CodingKeys: String, CodingKey {
            case buildNumber = "build_number"
            case buildDate = "build_date"
            case buildTimestamp = "build_timestamp"
            case gitHash = "git_hash"
            case gitBranch = "git_branch"
            case gitDirty = "git_dirty"
            case summary
        }

        static var current: AppBuildBlock {
            AppBuildBlock(
                buildNumber: BuildInfo.buildNumber,
                buildDate: BuildInfo.buildDate,
                buildTimestamp: BuildInfo.buildTimestamp,
                gitHash: BuildInfo.gitHash,
                gitBranch: BuildInfo.gitBranch,
                gitDirty: BuildInfo.gitDirty,
                summary: BuildInfo.summary
            )
        }
    }

    /// One per row in `puzzles[]`. Hierarchical: a `puzzle` sub-object
    /// describes the position + bookmove, a `probe_result` sub-object
    /// describes the network's behavior on it.
    private struct PuzzleEntry: Encodable {
        let puzzle: PuzzleBlock
        let probeResult: ProbeResultBlock

        enum CodingKeys: String, CodingKey {
            case puzzle
            case probeResult = "probe_result"
        }
    }

    private struct PuzzleBlock: Encodable {
        /// Original Lichess `PuzzleId` from the puzzle DB CSV — an
        /// opaque ~5-char alphanumeric assigned by Lichess; we don't
        /// generate these. The Lichess URL for the puzzle is
        /// `https://lichess.org/training/<id>`.
        let id: String
        /// Lichess theme tag (e.g. `mateIn1`, `hangingPiece`).
        let theme: String
        /// In-app `ProbeCategory` rawValue (e.g. `lichessMateIn1`) —
        /// the bucket the curation script assigned the puzzle to.
        let themeCategory: String
        /// Lichess puzzle rating (Glicko-2 family — Elo-like number).
        let ratingGlicko2: Int?
        let fen: String?
        let expectedMoveUci: String?
        let expectedMoveNotation: String?

        enum CodingKeys: String, CodingKey {
            case id
            case theme
            case themeCategory = "theme_category"
            case ratingGlicko2 = "rating_glicko2"
            case fen
            case expectedMoveUci = "expected_move_uci"
            case expectedMoveNotation = "expected_move_notation"
        }
    }

    private struct ProbeResultBlock: Encodable {
        let verdict: String
        let expectedRank: Int?
        let expectedProb: Float
        let legalCount: Int
        let illegalMass: Float
        /// Shannon entropy (in nats) of the legal-masked-and-
        /// renormalized policy distribution. Measures how spread the
        /// network's probability is across legal moves: 0 = totally
        /// committed, higher = flatter.
        let legalEntropyNats: Float
        /// `ln(legal_count)` — the entropy a uniformly-flat policy
        /// would have over this position's legal set. Theoretical
        /// max for `legal_entropy_nats`; the ratio
        /// `legal_entropy_nats / uniform_legal_entropy` reads as
        /// "fraction of uniform," with 1.0 = totally flat and 0
        /// = perfect commitment.
        let uniformLegalEntropy: Float
        let valueWdl: ValueWDLBlock
        let topMoves: [TopMoveBlock]

        enum CodingKeys: String, CodingKey {
            case verdict
            case expectedRank = "expected_rank"
            case expectedProb = "expected_prob"
            case legalCount = "legal_count"
            case illegalMass = "illegal_mass"
            case legalEntropyNats = "legal_entropy_nats"
            case uniformLegalEntropy = "uniform_legal_entropy"
            case valueWdl = "value_wdl"
            case topMoves = "top_moves"
        }
    }

    private struct ValueWDLBlock: Encodable {
        let win: Float
        let draw: Float
        let loss: Float
    }

    private struct TopMoveBlock: Encodable {
        let uci: String
        let notation: String
        let prob: Float
    }
}
