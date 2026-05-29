import AppKit
import Foundation
import UniformTypeIdentifiers

/// JSON exporter for the latest tick of the Lichess Probe set.
///
/// Writes a single file containing the snapshot timestamp, the model
/// id of the network the tick ran against, and a per-puzzle array with
/// the raw bundle metadata (id, theme, rating, FEN, bookmove UCI) plus
/// every observed field from the corresponding `ProbeResult`
/// (verdict, expected rank/prob, masked-entropy, value W/D/L, full top-5
/// move list). Designed to round-trip with the curation pipeline:
/// re-running the script would regenerate the same source bundle, and
/// the exported entries cross-reference back via puzzle id.
///
/// User flow: menu item "Export Latest Lichess Probe Results…" or
/// the "Export latest…" button in the Detail window opens an
/// `NSSavePanel`; on confirmation the JSON is written and a success
/// `[TACTICAL-LICHESS]` line is logged. Cancel / write failure logs
/// an error and surfaces nothing on screen.
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

        let panel = NSSavePanel()
        panel.title = "Export Latest Lichess Probe Results"
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = defaultFilename(tickTimestamp: tickTimestamp)
        panel.isExtensionHidden = false

        let response = panel.runModal()
        guard response == .OK, let url = panel.url else {
            SessionLogger.shared.log("[TACTICAL-LICHESS] export cancelled")
            return
        }

        do {
            let data = try buildJSON(
                history: history,
                tickTimestamp: tickTimestamp
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
                message: "Could not write the JSON file:\n\n\(error.localizedDescription)",
                revealURL: nil
            )
        }
    }

    // MARK: - NSAlert with Reveal in Finder

    /// Modal alert mirroring the `presentAnalyzeAlert` pattern used by
    /// the existing analyzer commands. Two-button flow: OK and Reveal
    /// in Finder; Reveal is only added when `revealURL` is non-nil.
    private static func presentExportAlert(
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
        if let url = revealURL, response == .alertSecondButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }

    private static func formattedSize(_ bytes: Int) -> String {
        let kb = Double(bytes) / 1024.0
        if kb < 1024 {
            return String(format: "%.1f KB", kb)
        } else {
            return String(format: "%.2f MB", kb / 1024.0)
        }
    }

    // MARK: - Filename

    private static func defaultFilename(tickTimestamp: Date) -> String {
        let stamp = filenameTimestampFormatter.string(from: tickTimestamp)
        return "lichess-probe-\(stamp).json"
    }

    private static let filenameTimestampFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        return df
    }()

    private static let isoTimestampFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    // MARK: - JSON shape

    private static func buildJSON(
        history: LichessProbeHistory,
        tickTimestamp: Date
    ) throws -> Data {
        let entries = history.latestPerPuzzleResults.map(buildEntry(_:))
        let payload = ExportPayload(
            schemaVersion: 1,
            generatedAt: isoTimestampFormatter.string(from: Date()),
            tickTimestamp: isoTimestampFormatter.string(from: tickTimestamp),
            modelLabel: history.latestTickModelLabel,
            probeCount: entries.count,
            puzzles: entries
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(payload)
    }

    private static func buildEntry(_ result: ProbeResult) -> ExportEntry {
        let probe = result.probe
        let meta = LichessProbeData.metadata[probe.name]
        let expectedNotation = probe.acceptable
            .sorted(by: { $0.notation < $1.notation })
            .first?.notation
        let topMoves = result.topMoves.map { entry in
            ExportTopMove(uci: uciString(entry.move), notation: entry.move.notation, prob: entry.prob)
        }
        return ExportEntry(
            puzzleId: meta?.id ?? probe.name,
            theme: meta?.theme ?? probe.shortDescription,
            themeCategory: probe.category.rawValue,
            rating: meta?.rating,
            fen: meta?.fen,
            expectedMoveUci: meta?.bestMoveUci,
            expectedMoveNotation: expectedNotation,
            verdict: result.verdict.rawValue,
            expectedRank: result.expectedRank,
            expectedProb: result.expectedProb,
            legalCount: result.legalCount,
            illegalMass: result.illegalMass,
            legalEntropyNats: result.legalEntropyNats,
            uniformLegalEntropy: result.uniformLegalEntropy,
            valueWin: result.valueWDL.win,
            valueDraw: result.valueWDL.draw,
            valueLoss: result.valueWDL.loss,
            topMoves: topMoves
        )
    }

    /// Reconstruct UCI long-algebraic from a `ChessMove`. Mirror of the
    /// loader's `parseUCI`: `<from-square><to-square>[promotion]`.
    private static func uciString(_ move: ChessMove) -> String {
        let from = squareName(row: move.fromRow, col: move.fromCol)
        let to = squareName(row: move.toRow, col: move.toCol)
        let promo: String
        switch move.promotion {
        case .queen?:  promo = "q"
        case .rook?:   promo = "r"
        case .bishop?: promo = "b"
        case .knight?: promo = "n"
        default:       promo = ""
        }
        return "\(from)\(to)\(promo)"
    }

    private static func squareName(row: Int, col: Int) -> String {
        // row 0 = rank 8, row 7 = rank 1 — same convention as
        // `BoardEncoder.squareName`. Reimplemented here to avoid
        // pulling in BoardEncoder for one helper.
        let file = Character(UnicodeScalar(UInt8(97 + col)))
        let rank = 8 - row
        return "\(file)\(rank)"
    }

    // MARK: - Codable payloads

    private struct ExportPayload: Encodable {
        let schemaVersion: Int
        let generatedAt: String
        let tickTimestamp: String
        let modelLabel: String?
        let probeCount: Int
        let puzzles: [ExportEntry]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case generatedAt = "generated_at"
            case tickTimestamp = "tick_timestamp"
            case modelLabel = "model_label"
            case probeCount = "probe_count"
            case puzzles
        }
    }

    private struct ExportEntry: Encodable {
        let puzzleId: String
        let theme: String
        let themeCategory: String
        let rating: Int?
        let fen: String?
        let expectedMoveUci: String?
        let expectedMoveNotation: String?
        let verdict: String
        let expectedRank: Int?
        let expectedProb: Float
        let legalCount: Int
        let illegalMass: Float
        let legalEntropyNats: Float
        let uniformLegalEntropy: Float
        let valueWin: Float
        let valueDraw: Float
        let valueLoss: Float
        let topMoves: [ExportTopMove]

        enum CodingKeys: String, CodingKey {
            case puzzleId = "puzzle_id"
            case theme
            case themeCategory = "theme_category"
            case rating
            case fen
            case expectedMoveUci = "expected_move_uci"
            case expectedMoveNotation = "expected_move_notation"
            case verdict
            case expectedRank = "expected_rank"
            case expectedProb = "expected_prob"
            case legalCount = "legal_count"
            case illegalMass = "illegal_mass"
            case legalEntropyNats = "legal_entropy_nats"
            case uniformLegalEntropy = "uniform_legal_entropy"
            case valueWin = "value_win"
            case valueDraw = "value_draw"
            case valueLoss = "value_loss"
            case topMoves = "top_moves"
        }
    }

    private struct ExportTopMove: Encodable {
        let uci: String
        let notation: String
        let prob: Float
    }
}
