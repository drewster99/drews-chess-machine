import Foundation

/// `--list-sessions [--sessions-dir <folder>] [--tail <n> | --all] [--json]`:
/// the saved `.dcmsession` folders, oldest save first, as the app reads them,
/// then exit (no GUI).
///
/// Each row is the session's `SessionManifest` — the same reading the Load
/// Session picker shows (`SessionManifest.resolve`) — plus the run's lineage
/// record as the session's `trainer.safetensors` carries it (read from the
/// header, the way a session load reads it) and whether `replay_buffer.bin`
/// is in the folder. Nothing here re-reads `session.json` by its own rules.
///
/// The table shows, per save: folder, size, save time (local), trigger,
/// trainer and champion IDs, `session.json`'s training-step count beside the
/// trainer file's trainer step (`cum`), training hours, self-play games,
/// (`selfPlayGames`), arenas, promotions, the champion's puzzle pElo (largest test set), the
/// replay buffer (`yes` / `no`, or `MISSING` / `UNLISTED` when session.json
/// and the folder disagree) and the lineage (path, segment, gaps). `--json`
/// prints the rows instead, each with the whole manifest.
///
/// Exits 0 when every listed row was read, 1 when any listed row carries an
/// error (it is listed with it, never skipped), 2 on a usage error or an
/// unreadable sessions folder.
enum ListSessionsCLI {
    static let flag = "--list-sessions"
    static let defaultTail = 20

    struct UsageError: LocalizedError {
        let errorDescription: String?
    }

    struct Options: Equatable {
        var sessionsDirectory: URL
        /// Nil: every session.
        var tail: Int?
        var json: Bool
    }

    /// The run's lineage as the session's trainer file records it.
    enum TrainerLineage {
        /// The record as decoded, and as stored in the header (what `--json`
        /// prints: an older schema's record is shown in its own layout, never
        /// re-encoded).
        case recorded(LineageRecord, storedJSON: String)
        /// The trainer file was written before lineage records.
        case unrecorded(formatVersion: Int)
        /// A legacy `.dcmmodel` trainer file (no header to read a record from).
        case legacyTrainerFile
        case unreadable(String)
    }

    struct Row {
        let folder: URL
        let manifest: SessionManifest
        let lineage: TrainerLineage
        let replayBufferFilePresent: Bool

        /// Why this row is incomplete, if it is.
        var errors: [String] {
            var errors: [String] = []
            if let loadError = manifest.loadError { errors.append("session.json: \(loadError)") }
            if case .unreadable(let reason) = lineage { errors.append("trainer file: \(reason)") }
            return errors
        }

        /// `yes` / `no`, or the disagreement between what session.json
        /// declares and what the folder holds.
        var bufferLabel: String {
            switch (manifest.replayBufferDeclared, replayBufferFilePresent) {
            case (true?, false): return "MISSING"
            case (false?, true): return "UNLISTED"
            case (_, true): return "yes"
            case (_, false): return "no"
            }
        }
    }

    /// Runs and exits when `--list-sessions` is present; returns otherwise.
    static func handleIfRequested(rawArgs: [String]) {
        guard rawArgs.contains(flag) else { return }
        do {
            let options = try parse(rawArgs)
            let rows = listedRows(try rows(in: options.sessionsDirectory), tail: options.tail)
            if options.json {
                FileHandle.standardOutput.write(try jsonData(rows))
            } else {
                for line in render(rows) { FileHandle.standardOutput.write(Data((line + "\n").utf8)) }
            }
            if rows.isEmpty {
                FileHandle.standardError.write(Data("no .dcmsession folders in \(options.sessionsDirectory.path)\n".utf8))
            }
            Darwin.exit(rows.contains { !$0.errors.isEmpty } ? 1 : 0)
        } catch {
            FileHandle.standardError.write(Data("error: \(flag): \(error.localizedDescription)\n".utf8))
            Darwin.exit(2)
        }
    }

    static func parse(_ rawArgs: [String]) throws -> Options {
        guard let start = rawArgs.firstIndex(of: flag) else {
            throw UsageError(errorDescription: "\(flag) is missing")
        }
        var options = Options(sessionsDirectory: CheckpointPaths.sessionsDir, tail: defaultTail, json: false)
        var seen: Set<String> = []
        var index = start + 1
        func value(_ name: String) throws -> String {
            guard index + 1 < rawArgs.count else { throw UsageError(errorDescription: "\(name) needs a value") }
            index += 1
            return rawArgs[index]
        }
        while index < rawArgs.count {
            let argument = rawArgs[index]
            guard seen.insert(argument).inserted else {
                throw UsageError(errorDescription: "\(argument) is given twice")
            }
            switch argument {
            case "--sessions-dir":
                options.sessionsDirectory = URL(fileURLWithPath: (try value(argument) as NSString).expandingTildeInPath,
                                                isDirectory: true)
            case "--tail":
                let raw = try value(argument)
                guard let count = Int(raw), count >= 1 else {
                    throw UsageError(errorDescription: "--tail \(raw) is not a whole number of at least 1")
                }
                options.tail = count
            case "--all":
                options.tail = nil
            case "--json":
                options.json = true
            default:
                throw UsageError(errorDescription: "unknown argument \(argument) (usage: \(flag) [--sessions-dir <folder>] [--tail <n> | --all] [--json])")
            }
            index += 1
        }
        if seen.contains("--tail") && seen.contains("--all") {
            throw UsageError(errorDescription: "--tail and --all cannot be combined")
        }
        return options
    }

    /// One row per `.dcmsession` folder directly in `directory`, oldest save
    /// first (a folder with no known save time first of all), then by name.
    static func rows(in directory: URL) throws -> [Row] {
        let folders = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
        ).filter { url in
            guard url.pathExtension == "dcmsession" else { return false }
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
        }
        let rows = folders.map(row(for:))
        return rows.sorted { a, b in
            switch (a.manifest.savedAt, b.manifest.savedAt) {
            case let (x?, y?) where x != y: return x < y
            case (nil, _?): return true
            case (_?, nil): return false
            default: return a.manifest.folderName < b.manifest.folderName
            }
        }
    }

    static func row(for folder: URL) -> Row {
        Row(folder: folder,
            manifest: SessionManifest.resolve(sessionFolder: folder),
            lineage: trainerLineage(in: folder),
            replayBufferFilePresent: FileManager.default.fileExists(
                atPath: SessionCheckpointLayout.replayBufferURL(in: folder).path))
    }

    static func listedRows(_ rows: [Row], tail: Int?) -> [Row] {
        guard let tail else { return rows }
        return Array(rows.suffix(tail))
    }

    /// The trainer file's lineage, read from its header with every rule a
    /// load applies to a header (`SafetensorsModelIO.inspect`).
    static func trainerLineage(in folder: URL) -> TrainerLineage {
        let trainer = SessionCheckpointLayout.existingTrainerURL(in: folder)
        if trainer.lastPathComponent == SessionCheckpointLayout.legacyTrainerFilename {
            return FileManager.default.fileExists(atPath: trainer.path)
                ? .legacyTrainerFile : .unreadable("the session holds no trainer file")
        }
        do {
            let header = try SafetensorsFile.readHeader(at: trainer)
            let inspected = try SafetensorsModelIO.inspect(
                header: header, source: trainer.lastPathComponent, sessionTrainerPolicyTail: nil)
            switch inspected.provenance.lineage {
            case .recorded(let record):
                guard let stored = header.metadata[LineageRecord.metadataKey] else {
                    return .unreadable("the header has no \(LineageRecord.metadataKey), though a record was decoded from it")
                }
                return .recorded(record, storedJSON: stored)
            case .unrecorded(let formatVersion): return .unrecorded(formatVersion: formatVersion)
            }
        } catch {
            return .unreadable(String(describing: error))
        }
    }

    // MARK: - Text

    static func lineageText(_ lineage: TrainerLineage) -> String {
        switch lineage {
        case .recorded(let record, _):
            var text = "\(record.invocation.pathKind.rawValue) seg=\(record.run.segmentIndex)"
            text += " cum=" + (record.steps.cumTrainerStep.map(String.init) ?? "unrecorded")
            if !record.run.notExactItems.isEmpty {
                text += " not-exact:" + record.run.notExactItems.joined(separator: ",")
            }
            return text
        case .unrecorded(let formatVersion): return "none (format v\(formatVersion))"
        case .legacyTrainerFile: return "none (.dcmmodel)"
        case .unreadable: return "ERROR"
        }
    }

    /// `bytes` in base-2 units, as `du -h` shows them.
    static func binarySize(_ bytes: Int64) -> String {
        let units: [(String, Double)] = [("GB", 1024 * 1024 * 1024), ("MB", 1024 * 1024), ("KB", 1024)]
        for (unit, scale) in units where Double(bytes) >= scale {
            return String(format: "%.1f %@", Double(bytes) / scale, unit)
        }
        return "\(bytes) B"
    }

    private static let localTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter
    }()

    static let columns = ["folder", "size", "saved (local)", "trigger", "trainer -> champion", "steps", "cum",
                          "train h", "games", "arenas", "promotions", "pElo", "buffer", "lineage"]
    /// Columns aligned right (numbers).
    private static let numericColumns: Set<String> = ["size", "steps", "cum", "train h", "games", "arenas",
                                                      "promotions", "pElo"]

    /// The table's lines: a header, then one line per row. A row with an
    /// error shows its folder, size and save time, then `ERROR: …`.
    static func render(_ rows: [Row]) -> [String] {
        let cells: [[String]] = rows.map(cells(of:))
        var widths = columns.map(\.count)
        for (row, rowCells) in zip(rows, cells) where row.errors.isEmpty {
            for (column, cell) in rowCells.enumerated() { widths[column] = max(widths[column], cell.count) }
        }
        for (row, rowCells) in zip(rows, cells) where !row.errors.isEmpty {
            for column in 0..<3 { widths[column] = max(widths[column], rowCells[column].count) }
        }
        func line(_ values: [String]) -> String {
            zip(columns, zip(values, widths)).map { column, cell in
                let (value, width) = cell
                let padding = String(repeating: " ", count: max(0, width - value.count))
                return numericColumns.contains(column) ? padding + value : value + padding
            }.joined(separator: "  ").trimmingTrailingSpaces()
        }
        var lines = [line(columns)]
        for (row, rowCells) in zip(rows, cells) {
            if row.errors.isEmpty {
                lines.append(line(rowCells))
            } else {
                lines.append(line(Array(rowCells.prefix(3))) + "  ERROR: " + row.errors.joined(separator: "; "))
            }
        }
        return lines
    }

    static func cells(of row: Row) -> [String] {
        let m = row.manifest
        let cum: String
        if case .recorded(let record, _) = row.lineage {
            cum = record.steps.cumTrainerStep.map(String.init) ?? "—"
        } else {
            cum = "—"
        }
        return [
            m.folderName,
            m.diskBytes.map(binarySize) ?? "—",
            m.savedAt.map { localTimeFormatter.string(from: $0) } ?? "—",
            m.trigger ?? "(renamed)",
            "\(m.trainerID ?? "—") -> \(m.championID ?? "—")",
            m.trainingSteps.map(String.init) ?? "—",
            cum,
            m.elapsedTrainingSec.map { String(format: "%.1f", $0 / 3600) } ?? "—",
            m.selfPlayGames.map(String.init) ?? "—",
            m.arenaCount.map(String.init) ?? "—",
            m.promotionCount.map(String.init) ?? "—",
            m.championTestSets?.pEloText ?? "—",
            row.bufferLabel,
            lineageText(row.lineage),
        ]
    }

    // MARK: - JSON

    /// The rows as a JSON array: each the folder's path, the whole manifest,
    /// the trainer file's lineage (the record as the app decoded it, or why
    /// there is none), the buffer label and the row's errors.
    static func jsonData(_ rows: [Row]) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .secondsSince1970
        var objects: [[String: Any]] = []
        for row in rows {
            let lineage: [String: Any]
            switch row.lineage {
            case .recorded(_, let stored):
                lineage = ["status": "recorded",
                           "record": try JSONSerialization.jsonObject(with: Data(stored.utf8))]
            case .unrecorded(let formatVersion):
                lineage = ["status": "unrecorded", "format_version": formatVersion]
            case .legacyTrainerFile:
                lineage = ["status": "legacy_trainer_file"]
            case .unreadable(let reason):
                lineage = ["status": "unreadable", "reason": reason]
            }
            objects.append([
                "path": row.folder.path,
                "manifest": try JSONSerialization.jsonObject(with: try encoder.encode(row.manifest)),
                "trainer_lineage": lineage,
                "replay_buffer_file": row.replayBufferFilePresent,
                "buffer": row.bufferLabel,
                "errors": row.errors,
            ])
        }
        var data = try JSONSerialization.data(withJSONObject: objects, options: [.prettyPrinted, .sortedKeys,
                                                                                .withoutEscapingSlashes])
        data.append(Data("\n".utf8))
        return data
    }
}

private extension String {
    func trimmingTrailingSpaces() -> String {
        var text = self
        while text.last == " " { text.removeLast() }
        return text
    }
}
