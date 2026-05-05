import Foundation

/// One-shot recovery scanner that walks the per-launch session
/// logs (`~/Library/Logs/DrewsChessMachine/dcm_log_*.txt`),
/// extracts the `[ARENA] #N kv …` lines, and builds a
/// `step → Recovered` map. The map is consumed by the Arena
/// History UI to backfill `finishedAt` / `candidateID` /
/// `championID` on `TournamentRecord` entries that were appended
/// or persisted by builds before those fields existed.
///
/// The scanner is read-only (no mutation, no logging) — caller
/// owns the merge/save loop. Designed to run on a background
/// queue: a worst-case scan over hundreds of MB of logs takes a
/// fraction of a second on Apple Silicon, but we still don't
/// block the UI.
///
/// Date math: each log filename embeds a launch wall-clock stamp
/// (`dcm_log_yyyymmdd-HHMMSS.txt`). Each line inside carries an
/// `HH:MM:SS.SSS` prefix in local time. Long sessions can cross
/// midnight; the scanner walks lines in file order and increments
/// a day-offset counter every time the line's seconds-of-day goes
/// backward relative to the previous line. That makes the
/// reconstructed `Date` correct for sessions of any duration up
/// to (24h × 2³¹) — i.e., orders of magnitude beyond any
/// realistic dcm_log file.
enum ArenaLogRecovery {

    /// One recovered arena, keyed by `step` in the result map.
    struct Recovered: Sendable {
        let finishedAt: Date
        let candidateID: String?
        let championID: String?
        // Verification scalars used by the merge step to reject a
        // step-collision (same step, different W/D/L → different
        // arena, e.g. a re-run after a crash that didn't survive
        // the next save).
        let candidateWins: Int
        let draws: Int
        let championWins: Int
    }

    /// Default logs directory used by the live `SessionLogger`.
    /// The app is currently *not* sandboxed despite the stale
    /// comment in `SessionLogger.swift`; logs land at
    /// `~/Library/Logs/DrewsChessMachine/`.
    static func defaultLogsDirectory() throws -> URL {
        let library = try FileManager.default.url(
            for: .libraryDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        )
        return library
            .appendingPathComponent("Logs", isDirectory: true)
            .appendingPathComponent("DrewsChessMachine", isDirectory: true)
    }

    /// Scan every `dcm_log_*.txt` under `logsDirectory` and return
    /// a map keyed by training-step at the time of arena finish.
    /// When the same step appears in multiple log files (e.g. a
    /// session was resumed and the same checkpoint replayed), the
    /// chronologically latest line wins so the recovered
    /// `finishedAt` matches what was most recently saved into the
    /// session checkpoint.
    static func scan(logsDirectory: URL) -> [Int: Recovered] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: logsDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [:] }

        // Iterate in filename order so the day-offset bookkeeping
        // inside any single file stays consistent. Across files we
        // still take "latest finishedAt wins" so out-of-order
        // discovery on disk doesn't lose data.
        let logFiles = entries
            .filter {
                $0.lastPathComponent.hasPrefix("dcm_log_")
                && $0.pathExtension.lowercased() == "txt"
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var result: [Int: Recovered] = [:]
        for url in logFiles {
            mergeFile(url: url, into: &result)
        }
        return result
    }

    // MARK: - Per-file merge

    private static func mergeFile(url: URL, into result: inout [Int: Recovered]) {
        guard let launchDate = launchDate(fromFilename: url.lastPathComponent),
              let contents = try? String(contentsOf: url, encoding: .utf8) else {
            return
        }

        // `launchDate` is the start-of-launch wall-clock time; we
        // anchor the first observed line to "the same calendar
        // day as the launch" (most kv lines share that day) and
        // bump dayOffset on each backward-jump in seconds-of-day.
        var dayOffset = 0
        var prevSecOfDay: TimeInterval = -1
        let calendar = Calendar(identifier: .gregorian)
        let launchDayStart = calendar.startOfDay(for: launchDate)

        // Walk lines without copying the giant string. Substring
        // works fine here — we only retain small pieces of it.
        for raw in contents.split(separator: "\n", omittingEmptySubsequences: true) {
            guard raw.contains("[ARENA]"),
                  let kvStart = raw.range(of: " kv step=") else { continue }

            // Parse leading `HH:MM:SS.SSS` (or `HH:MM:SS`).
            guard let (secOfDay, _) = parseLineTime(raw) else { continue }
            if secOfDay < prevSecOfDay { dayOffset += 1 }
            prevSecOfDay = secOfDay

            // Build the absolute line date by anchoring to
            // `launchDayStart + dayOffset` and adding secondsOfDay.
            guard let lineDay = calendar.date(
                byAdding: .day, value: dayOffset, to: launchDayStart
            ) else { continue }
            let lineDate = lineDay.addingTimeInterval(secOfDay)

            // Parse the kv body. Layout is space-separated
            // key=value pairs (no quotes, no spaces inside values
            // — see `ArenaLogFormatter.formatKVLine`).
            let kvBody = raw[kvStart.upperBound...]
            // Re-prepend "step=" since we matched on " kv step=".
            let body = "step=" + String(kvBody)
            let kv = parseKV(body)

            guard let step = kv["step"].flatMap(Int.init),
                  let cw = kv["w"].flatMap(Int.init),
                  let dw = kv["d"].flatMap(Int.init),
                  let cl = kv["l"].flatMap(Int.init) else { continue }

            // Latest-wins on collision: only overwrite if the new
            // line is strictly newer than what we have. Equal
            // timestamps fall through to the existing entry —
            // arbitrary tie-breaking but stable for re-scans.
            if let existing = result[step], existing.finishedAt >= lineDate {
                continue
            }

            result[step] = Recovered(
                finishedAt: lineDate,
                candidateID: kv["candidate"],
                championID: kv["champion"],
                candidateWins: cw,
                draws: dw,
                championWins: cl
            )
        }
    }

    // MARK: - Parsing helpers

    /// Extract the launch-time `Date` from a log filename of the
    /// form `dcm_log_yyyymmdd-HHMMSS.txt`. Returns `nil` if the
    /// filename doesn't fit the pattern (unrelated file in the
    /// logs directory, copy with a renamed extension, etc.).
    static func launchDate(fromFilename name: String) -> Date? {
        // Expected: dcm_log_<8 digits>-<6 digits>.txt
        let prefix = "dcm_log_"
        let suffix = ".txt"
        guard name.hasPrefix(prefix), name.hasSuffix(suffix) else { return nil }
        let stamp = name
            .dropFirst(prefix.count)
            .dropLast(suffix.count)
        // 20260505-131313 → "20260505-131313"
        let parts = stamp.split(separator: "-")
        guard parts.count == 2,
              parts[0].count == 8,
              parts[1].count == 6 else { return nil }

        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        fmt.timeZone = TimeZone.current
        return fmt.date(from: String(stamp))
    }

    /// Parse the leading `HH:MM:SS.SSS` (or `HH:MM:SS`) time
    /// prefix of a log line into seconds-of-day. Returns
    /// `(secondsOfDay, prefixEndIndex)` or `nil` on parse failure.
    /// Tolerant of variable whitespace separating the timestamp
    /// from the message body.
    static func parseLineTime(_ line: Substring) -> (TimeInterval, Substring.Index)? {
        // Walk char-by-char rather than splitting the whole line
        // on whitespace — keeps the hot path allocation-free.
        var idx = line.startIndex
        let end = line.endIndex
        func nextDigits(_ count: Int) -> Int? {
            var n = 0
            for _ in 0..<count {
                guard idx < end, let d = line[idx].asciiDigit else { return nil }
                n = n * 10 + d
                idx = line.index(after: idx)
            }
            return n
        }
        func consume(_ ch: Character) -> Bool {
            guard idx < end, line[idx] == ch else { return false }
            idx = line.index(after: idx)
            return true
        }

        guard let h = nextDigits(2), consume(":"),
              let m = nextDigits(2), consume(":"),
              let s = nextDigits(2) else { return nil }
        var fracMs: TimeInterval = 0
        if idx < end, line[idx] == "." {
            idx = line.index(after: idx)
            guard let ms = nextDigits(3) else { return nil }
            fracMs = TimeInterval(ms) / 1000.0
        }
        let secOfDay = TimeInterval(h * 3600 + m * 60 + s) + fracMs
        return (secOfDay, idx)
    }

    /// Parse a space-separated key=value body into a dictionary.
    /// No quoting / escaping support — matches the kv line's
    /// fixed format in `ArenaLogFormatter.formatKVLine`.
    static func parseKV(_ body: String) -> [String: String] {
        var out: [String: String] = [:]
        for token in body.split(separator: " ", omittingEmptySubsequences: true) {
            guard let eq = token.firstIndex(of: "=") else { continue }
            let key = String(token[..<eq])
            let value = String(token[token.index(after: eq)...])
            out[key] = value
        }
        return out
    }
}

private extension Character {
    /// `0..<10` if this is an ASCII digit, else `nil`.
    /// Used by `parseLineTime` for an allocation-free time-prefix
    /// scan that doesn't go through `String(...)` or
    /// `Int(String(...))`.
    var asciiDigit: Int? {
        guard let ascii = self.asciiValue else { return nil }
        let zero: UInt8 = 0x30
        let nine: UInt8 = 0x39
        guard ascii >= zero, ascii <= nine else { return nil }
        return Int(ascii - zero)
    }
}
