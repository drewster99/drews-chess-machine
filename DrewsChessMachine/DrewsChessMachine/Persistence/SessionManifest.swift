import Foundation

/// One saved `.dcmsession`, summarized for the Load Session picker.
///
/// Two production paths (SESSION_PICKER_PLAN.md):
/// - **New saves**: `CheckpointManager` writes `manifest.json` into the
///   session folder at save time, from values it already has in hand.
/// - **Legacy sessions**: a background indexer extracts the same fields
///   from the (potentially ~16 MB) `session.json` once, then caches the
///   result OUTSIDE the immutable session folder in
///   `…/DrewsChessMachine/SessionIndex/<folderName>.json`.
///
/// Every field beyond identity is Optional: sessions span many schema
/// vintages, and the picker renders "—" for what a given save can't
/// provide rather than refusing to list it. A session whose
/// `session.json` cannot be read at all still gets a manifest, carrying
/// `loadError` — visible in the picker as an unreadable row, never
/// silently dropped.
struct SessionManifest: Codable, Hashable, Sendable, Identifiable {

    var id: String { folderName }

    /// What a manifest holds and how it was read. Bumped whenever a field is
    /// added or a reading changes; `resolve` uses a stored manifest (a
    /// session's `manifest.json`, an index-cache entry) only at this version
    /// and extracts again from `session.json` otherwise, so a manifest stored
    /// by another build never answers for this one.
    static let currentSchema = 2
    /// The schema this manifest was made at; nil in one made before the field.
    var schema: Int? = nil

    // MARK: Identity

    /// The folder's name as it is now (`resolve` sets it from the folder,
    /// whatever name a stored manifest was written under).
    private(set) var folderName: String
    /// The session ID `session.json` records; nil for a session.json without one.
    var sessionID: String? = nil
    /// Save trigger from the folder name (`manual`, `periodic`, `promote`,
    /// `sigusr2`, `vsuci-final`, …) — whatever tag the save used; the picker
    /// badges known values and shows the raw string otherwise. Nil when the
    /// name is not `<timestamp>-<sessionID>-<trigger>` for this session's own
    /// ID (a renamed folder). Recomputed by `resolve` from the current name.
    private(set) var trigger: String?
    /// Save instant. Folder-name timestamps are UTC by convention;
    /// `savedAtUnix` from `session.json` wins when available (exact).
    let savedAt: Date?
    /// The 4-char ModelID suffix of the SAVED lineage (e.g. `5K7Z`) —
    /// the picker's grouping key, per the "runs are the mental model"
    /// design decision. Recomputed by `resolve`.
    private(set) var lineageTag: String?

    // MARK: A — Architecture

    /// Canonical `architectureSummary` when the save embeds the
    /// runtime-config form; a synthesized legacy line (version, blocks,
    /// channels, planes) for pre-runtime-config saves.
    let architectureSummary: String?
    let parameterCount: Int?
    let inputPlanes: Int?
    let channels: Int?
    let numBlocks: Int?

    // MARK: B — Run / progress / lineage

    /// The model's name and starting preset from the session's lineage
    /// record (`ModelNaming`; unrecorded for a record before lineage schema
    /// 4 or a session without one). nil when this manifest predates the
    /// field (an embedded `manifest.json` or index-cache entry written
    /// before model naming) or the stored naming did not decode (logged).
    let modelNaming: LineageRecord.Recorded<ModelNaming>?
    let championID: String?
    let trainerID: String?
    let trainingSteps: Int?
    let elapsedTrainingSec: Double?
    let emittedGames: Int?
    /// `session.json`'s `selfPlayGames`: the self-play games played, kept in
    /// the buffer or not (`emittedGames` counts only the kept ones), since
    /// the current champion was promoted — both start over at each promotion
    /// (`resetSelfPlayGameStatsForNewChampion`); the run's lifetime count is
    /// its lineage record's `fed.cum_games`. Nil in a manifest written
    /// before this field.
    var selfPlayGames: Int? = nil
    let emittedPositions: Int?
    let bufferStored: Int?
    let bufferCapacity: Int?
    /// Whether `session.json` says this save wrote `replay_buffer.bin`
    /// (`hasReplayBuffer`). Nil for a save from before the field, and in a
    /// manifest written before this one carried it.
    var replayBufferDeclared: Bool? = nil
    let buildNumber: Int?
    let buildGitHash: String?
    let buildGitDirty: Bool?
    let buildGitBranch: String?

    // MARK: C — Performance

    let arenaCount: Int?
    let promotionCount: Int?
    /// Latest 200-set / wide-set MLE puzzle-Elo at save time, when the
    /// save carries probe history with finite values.
    let latestPElo200: Double?
    let latestPEloWide: Double?
    /// The test-set results each model file of the save carries (largest
    /// set, `ModelTestSetSummary`), read from the files' headers. Nil in a
    /// manifest written before these fields; `.notRecorded` for a save
    /// whose files predate test-set results.
    var championTestSets: ModelTestSetSummary? = nil
    var trainerTestSets: ModelTestSetSummary? = nil
    let whiteCheckmates: Int?
    let blackCheckmates: Int?
    let drawCount: Int?

    // MARK: D — Hyperparameter sampling

    let learningRate: Double?
    let batchSize: Int?
    let weightDecay: Double?
    let dropoutRate: Double?
    let momentumCoeff: Double?
    let promoteThreshold: Double?
    let selfPlayWorkerCount: Int?

    // MARK: Disk / cache validity / errors

    let diskBytes: Int64?
    /// Size + mtime of the `session.json` this manifest was derived
    /// from. Sessions are immutable, so a match means the cached
    /// manifest is still valid.
    let sourceJSONBytes: Int64?
    let sourceJSONMTime: Double?
    /// Non-nil when `session.json` was unreadable/undecodable — the
    /// picker shows the row with this message instead of dropping it.
    let loadError: String?
}

// MARK: - Folder-name parsing

extension SessionManifest {

    /// Parses `<yyyyMMdd>-<HHmmss>-<sessionID>-<trigger>.dcmsession`, the
    /// name `CheckpointManager.makeSessionDirectoryName` writes, where the
    /// session ID is a ModelID (`yyyymmdd-N-XXXX`) and the trigger is
    /// everything after it: a save tag may itself hold hyphens
    /// (`vsuci-periodic`, `promote-keep` after a hand rename). The lineage
    /// tag is the session ID's 4-character suffix.
    ///
    /// A name not of that shape (a folder renamed by hand) gets a nil
    /// lineage tag and trigger, and a save time only when its first two
    /// components are one: it is still listed, unannotated, instead of
    /// having a trigger and tag read out of arbitrary words.
    static func parseFolderName(
        _ name: String
    ) -> (savedAt: Date?, lineageTag: String?, trigger: String?) {
        guard name.hasSuffix(".dcmsession") else { return (nil, nil, nil) }
        let stem = String(name.dropLast(".dcmsession".count))
        let parts = stem.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        guard parts.count >= 2 else { return (nil, nil, nil) }
        let savedAt = timestamp(date: parts[0], time: parts[1])
        guard savedAt != nil, parts.count >= 6,
              let tag = modelIDTag(parts[2], parts[3], parts[4]) else {
            return (savedAt, nil, nil)
        }
        let trigger = parts[5...].joined(separator: "-")
        guard !trigger.isEmpty else { return (savedAt, tag, nil) }
        return (savedAt, tag, trigger)
    }

    /// The UTC instant a folder name's `yyyyMMdd` / `HHmmss` components state
    /// (`CheckpointManager.filenameTimestampFormatter` writes them in UTC), or
    /// nil when they are not that.
    private static func timestamp(date dateStr: String, time timeStr: String) -> Date? {
        guard dateStr.count == 8, timeStr.count == 6,
              isDigits(dateStr), isDigits(timeStr),
              let utc = TimeZone(identifier: "UTC") else { return nil }
        var comps = DateComponents()
        comps.year = Int(dateStr.prefix(4))
        comps.month = Int(dateStr.dropFirst(4).prefix(2))
        comps.day = Int(dateStr.dropFirst(6))
        comps.hour = Int(timeStr.prefix(2))
        comps.minute = Int(timeStr.dropFirst(2).prefix(2))
        comps.second = Int(timeStr.dropFirst(4))
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = utc
        return cal.date(from: comps)
    }

    /// The 4-character tag of a ModelID's three components
    /// (`yyyymmdd`, counter, tag), or nil when they are not a ModelID.
    private static func modelIDTag(_ date: String, _ counter: String, _ tag: String) -> String? {
        guard date.count == 8, isDigits(date), isDigits(counter),
              !tag.isEmpty, tag.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber) }) else { return nil }
        return tag
    }

    /// Non-empty and only ASCII digits.
    private static func isDigits(_ text: String) -> Bool {
        !text.isEmpty && text.allSatisfy { $0.isASCII && $0.isNumber }
    }

    /// The lineage tag of a session ID (a ModelID), or nil when it is not one.
    static func lineageTag(ofSessionID sessionID: String) -> String? {
        let parts = sessionID.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 3 else { return nil }
        return modelIDTag(parts[0], parts[1], parts[2])
    }

    /// The trigger and lineage tag of `folderName` checked against the
    /// session ID `session.json` records: a folder whose name does not carry
    /// its own session's ID was renamed (or holds another session's files),
    /// so its name states no trigger; its lineage tag is the recorded
    /// session's, so a renamed save still groups with its run.
    static func identity(
        folderName: String, recordedSessionID: String?
    ) -> (savedAt: Date?, lineageTag: String?, trigger: String?) {
        let parsed = parseFolderName(folderName)
        guard let recordedSessionID else { return parsed }
        let named = parsed.trigger != nil
            && folderName.dropFirst("yyyyMMdd-HHmmss-".count).hasPrefix(recordedSessionID + "-")
        return (parsed.savedAt, lineageTag(ofSessionID: recordedSessionID) ?? parsed.lineageTag,
                named ? parsed.trigger : nil)
    }
}

// MARK: - Extraction from session.json

extension SessionManifest {

    /// Build a manifest by reading a session folder from disk. Heavy
    /// (parses `session.json`, which can be ~16 MB) — call off the main
    /// actor. Never throws: failures produce a manifest with `loadError`
    /// set so the picker can show the folder as unreadable.
    static func extract(fromSessionFolder url: URL) -> SessionManifest {
        extractSessionJSON(fromSessionFolder: url).withTestSets(fromSessionFolder: url)
    }

    /// The `session.json` part of `extract(fromSessionFolder:)`.
    private static func extractSessionJSON(fromSessionFolder url: URL) -> SessionManifest {
        let folderName = url.lastPathComponent
        let parsed = parseFolderName(folderName)
        let jsonURL = url.appendingPathComponent("session.json")
        let fm = FileManager.default

        var srcBytes: Int64? = nil
        var srcMTime: Double? = nil
        if let attrs = try? fm.attributesOfItem(atPath: jsonURL.path) {
            srcBytes = (attrs[.size] as? NSNumber)?.int64Value
            srcMTime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970
        }
        let disk = directorySizeBytes(url)

        let dict: [String: Any]
        do {
            let data = try Data(contentsOf: jsonURL)
            guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return unreadable(folderName: folderName, parsed: parsed, disk: disk,
                                  srcBytes: srcBytes, srcMTime: srcMTime,
                                  error: "session.json is not a JSON object")
            }
            dict = obj
        } catch {
            return unreadable(folderName: folderName, parsed: parsed, disk: disk,
                              srcBytes: srcBytes, srcMTime: srcMTime,
                              error: error.localizedDescription)
        }
        return extract(jsonDict: dict, folderName: folderName,
                       disk: disk, srcBytes: srcBytes, srcMTime: srcMTime)
    }

    /// Core extraction from an already-parsed `session.json` object.
    /// Shared by the legacy-folder indexer above and the write-at-save
    /// path (`CheckpointManager` derives `manifest.json` from the exact
    /// bytes it just wrote, so there is exactly one extraction code path
    /// and the manifest can never drift from the file it summarizes).
    /// Decode a snake_case `NetworkArchitecture` embedded in an old
    /// `session.json`, as legacy (see `extract`). A failure is logged and
    /// returns nil, so the caller falls back to the scalar summary.
    private static func decodeEmbeddedArchitecture(_ arch: [String: Any], folderName: String) -> NetworkArchitecture? {
        let source = "\(folderName)/session.json"
        do {
            let archData = try JSONSerialization.data(withJSONObject: arch)
            return try ArchitectureFormat.makeDecoder(format: ArchitectureFormat.DecodeFormat(
                formatVersion: ArchitectureFormat.unversionedLegacyVersion, source: source)
            ).decode(NetworkArchitecture.self, from: archData)
        } catch {
            SessionLogger.shared.log("[SESSIONS] \(source): embedded architecture unreadable (\(String(describing: error))); showing the scalar summary")
            return nil
        }
    }

    /// `session.json`'s `lineage.model_naming`, decoded with the record's
    /// own strictness: unrecorded when there is no lineage or its schema
    /// predates the field; nil (logged) when it is present but malformed.
    private static func modelNaming(in dict: [String: Any], folderName: String) -> LineageRecord.Recorded<ModelNaming>? {
        guard let lineage = dict["lineage"] as? [String: Any],
              let stored = lineage["model_naming"] else { return .unrecorded }
        do {
            let data = try JSONSerialization.data(withJSONObject: stored)
            return try JSONDecoder().decode(LineageRecord.Recorded<ModelNaming>.self, from: data)
        } catch {
            SessionLogger.shared.log("[SESSION-INDEX] \(folderName): lineage.model_naming does not decode: \(error)")
            return nil
        }
    }

    static func extract(
        jsonDict dict: [String: Any],
        folderName: String,
        disk: Int64?,
        srcBytes: Int64?,
        srcMTime: Double?
    ) -> SessionManifest {
        func int(_ k: String) -> Int? { (dict[k] as? NSNumber)?.intValue }
        func dbl(_ k: String) -> Double? { (dict[k] as? NSNumber)?.doubleValue }
        func str(_ k: String) -> String? { dict[k] as? String }
        func bool(_ k: String) -> Bool? { (dict[k] as? NSNumber)?.boolValue }

        let parsed = identity(folderName: folderName, recordedSessionID: str("sessionID"))

        // Architecture: runtime-config saves embed the snake_case
        // NetworkArchitecture (decode it for the canonical summary);
        // legacy saves carry a camelCase scalar dict. `session.json` carries
        // no architecture format version and no current writer embeds a
        // NetworkArchitecture here (saves write `ArchitectureMetadata`), so
        // any embedded one predates format v4 and decodes as legacy. Display
        // only — resolutions are not logged.
        var archSummary: String? = nil
        var paramCount: Int? = nil
        var inputPlanes: Int? = nil
        var chans: Int? = nil
        var blocks: Int? = nil
        if let arch = dict["architecture"] as? [String: Any] {
            func aInt(_ k: String) -> Int? { (arch[k] as? NSNumber)?.intValue }
            if arch["input_encoding"] != nil,
               let decoded = decodeEmbeddedArchitecture(arch, folderName: folderName) {
                archSummary = decoded.architectureSummary
                paramCount = decoded.parameterCount
                inputPlanes = decoded.inputPlanes
                chans = decoded.towerOutputChannels
                blocks = decoded.numBlocks
            } else {
                // Legacy camelCase scalars; synthesize a readable line.
                chans = aInt("channels")
                blocks = aInt("numBlocks")
                inputPlanes = aInt("inputPlanes")
                paramCount = aInt("parameterCount")
                if let b = blocks, let c = chans {
                    let planes = inputPlanes.map { " in \($0) planes" } ?? ""
                    archSummary = "\(b)x[\(c)ch]\(planes) (legacy)"
                }
            }
        }

        // Performance: prefer the probe history's own promotion counter;
        // fall back to counting promoted arena entries.
        var arenaCount = (dict["arenaHistory"] as? [Any])?.count
        var promotionCount: Int? = nil
        var pElo200: Double? = nil
        var pEloWide: Double? = nil
        func probeFields(_ key: String) -> (promos: Int?, arenas: Int?, pElo: Double?) {
            guard let h = dict[key] as? [String: Any] else { return (nil, nil, nil) }
            let promos = (h["latestPromotionCount"] as? NSNumber)?.intValue
            let arenas = (h["latestArenaCount"] as? NSNumber)?.intValue
            // The latest tick's puzzleElo can be non-finite (±inf for
            // all-correct/all-wrong, NaN for a tick with no rated puzzles).
            // Walk back to the most recent FINITE value rather than
            // discarding the whole history when only the final tick is an
            // edge case.
            var elo: Double? = nil
            if let overall = h["overall"] as? [[String: Any]] {
                for entry in overall.reversed() {
                    if let e = (entry["puzzleElo"] as? NSNumber)?.doubleValue, e.isFinite {
                        elo = e
                        break
                    }
                }
            }
            return (promos, arenas, elo)
        }
        let p200 = probeFields("lichessProbeHistory")
        let pWide = probeFields("lichessProbeWideHistory")
        pElo200 = p200.pElo
        pEloWide = pWide.pElo
        // Promotions: the arena history records each one as it happens, but an
        // older session's history may hold only its latest arenas, while the
        // probe history's counter is lifetime but updated only at probe ticks
        // (one promotion behind on a save made right after a promotion). So
        // the history's count wins when the history holds every arena the
        // probe counted; the probe's otherwise; the history's when there is
        // no probe counter.
        let arenaEntries = dict["arenaHistory"] as? [Any]
        // Tolerate a non-`[[String:Any]]` arenaHistory shape (legacy /
        // heterogeneous): count only the entries that parse as promoted.
        let historyPromotions = arenaEntries.map { entries in
            entries.reduce(0) { acc, e in
                acc + (((e as? [String: Any])?["promoted"] as? NSNumber)?.boolValue == true ? 1 : 0)
            }
        }
        let probePromotions = p200.promos ?? pWide.promos
        let historyIsComplete = (arenaEntries?.count ?? 0) > 0
            && (arenaEntries?.count ?? 0) >= (p200.arenas ?? pWide.arenas ?? 0)
        promotionCount = historyIsComplete ? historyPromotions : (probePromotions ?? historyPromotions)
        // An empty arenaHistory array yields count 0 (non-nil) and would
        // mask the probe-history fallback; treat 0 as "no arena data here"
        // and prefer the probe's count when it has one.
        if (arenaCount ?? 0) == 0 { arenaCount = p200.arenas ?? arenaCount }

        // Draw counters are written as one atomic block, so a partial subset
        // means an unexpected/hand-edited schema — report nil ("—") rather
        // than a misleadingly-low "total" that silently omits a category.
        let drawParts = [int("stalemates"), int("fiftyMoveDraws"),
                         int("threefoldRepetitionDraws"), int("insufficientMaterialDraws")]
        let draws: [Int] = drawParts.contains(where: { $0 == nil }) ? [] : drawParts.compactMap { $0 }
        let savedAt = dbl("savedAtUnix").map { Date(timeIntervalSince1970: $0) } ?? parsed.savedAt

        var manifest = SessionManifest(
            folderName: folderName,
            trigger: parsed.trigger,
            savedAt: savedAt,
            lineageTag: parsed.lineageTag,
            architectureSummary: archSummary,
            parameterCount: paramCount,
            inputPlanes: inputPlanes,
            channels: chans,
            numBlocks: blocks,
            modelNaming: modelNaming(in: dict, folderName: folderName),
            championID: str("championID"),
            trainerID: str("trainerID"),
            trainingSteps: int("trainingSteps"),
            elapsedTrainingSec: dbl("elapsedTrainingSec"),
            emittedGames: int("emittedGames"),
            emittedPositions: int("emittedPositions"),
            bufferStored: int("replayBufferStoredCount"),
            bufferCapacity: int("replayBufferCapacity"),
            buildNumber: int("buildNumber"),
            buildGitHash: str("buildGitHash"),
            buildGitDirty: bool("buildGitDirty"),
            buildGitBranch: str("buildGitBranch"),
            arenaCount: arenaCount,
            promotionCount: promotionCount,
            latestPElo200: pElo200,
            latestPEloWide: pEloWide,
            whiteCheckmates: int("whiteCheckmates"),
            blackCheckmates: int("blackCheckmates"),
            drawCount: draws.isEmpty ? nil : draws.reduce(0, +),
            learningRate: dbl("learningRate"),
            batchSize: int("batchSize"),
            weightDecay: dbl("weightDecayCoeff"),
            dropoutRate: dbl("dropoutRate"),
            momentumCoeff: dbl("momentumCoeff"),
            promoteThreshold: dbl("promoteThreshold"),
            selfPlayWorkerCount: int("selfPlayWorkerCount"),
            diskBytes: disk,
            sourceJSONBytes: srcBytes,
            sourceJSONMTime: srcMTime,
            loadError: nil
        )
        manifest.replayBufferDeclared = bool("hasReplayBuffer")
        manifest.schema = currentSchema
        manifest.sessionID = str("sessionID")
        manifest.selfPlayGames = int("selfPlayGames")
        return manifest
    }

    private static func unreadable(
        folderName: String,
        parsed: (savedAt: Date?, lineageTag: String?, trigger: String?),
        disk: Int64?, srcBytes: Int64?, srcMTime: Double?, error: String
    ) -> SessionManifest {
        SessionManifest(
            folderName: folderName, trigger: parsed.trigger,
            savedAt: parsed.savedAt, lineageTag: parsed.lineageTag,
            architectureSummary: nil, parameterCount: nil, inputPlanes: nil,
            channels: nil, numBlocks: nil, modelNaming: nil, championID: nil, trainerID: nil,
            trainingSteps: nil, elapsedTrainingSec: nil, emittedGames: nil,
            emittedPositions: nil, bufferStored: nil, bufferCapacity: nil,
            buildNumber: nil, buildGitHash: nil, buildGitDirty: nil,
            buildGitBranch: nil, arenaCount: nil, promotionCount: nil,
            latestPElo200: nil, latestPEloWide: nil, whiteCheckmates: nil,
            blackCheckmates: nil, drawCount: nil, learningRate: nil,
            batchSize: nil, weightDecay: nil, dropoutRate: nil,
            momentumCoeff: nil, promoteThreshold: nil,
            selfPlayWorkerCount: nil, diskBytes: disk,
            sourceJSONBytes: srcBytes, sourceJSONMTime: srcMTime,
            loadError: error
        )
    }

    /// This manifest with the test-set summaries of the session's model
    /// files (`championTestSets`, `trainerTestSets`), read from their headers.
    func withTestSets(fromSessionFolder url: URL) -> SessionManifest {
        var copy = self
        copy.championTestSets = .ofModelFile(at: SessionCheckpointLayout.championURL(in: url))
        copy.trainerTestSets = .ofModelFile(at: SessionCheckpointLayout.trainerURL(in: url))
        return copy
    }

    struct ManifestEncodingError: Error, CustomStringConvertible {
        let description: String
    }

    /// Encode a `manifest.json` payload for a session whose
    /// `session.json` bytes were just written (the write-at-save path).
    /// `sessionDirURL` is the (staging) directory holding the session's
    /// files — its current contents define the recorded disk size.
    static func makeManifestData(
        sessionJSON: Data,
        folderName: String,
        sessionDirURL: URL,
        sessionJSONURL: URL
    ) throws -> Data {
        guard let dict = try JSONSerialization.jsonObject(with: sessionJSON) as? [String: Any] else {
            throw ManifestEncodingError(description: "session.json payload is not a JSON object")
        }
        var mtime: Double? = nil
        if let attrs = try? FileManager.default.attributesOfItem(atPath: sessionJSONURL.path) {
            mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970
        }
        let manifest = extract(
            jsonDict: dict,
            folderName: folderName,
            disk: directorySizeBytes(sessionDirURL),
            srcBytes: Int64(sessionJSON.count),
            srcMTime: mtime
        ).withTestSets(fromSessionFolder: sessionDirURL)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(manifest)
    }

    /// Shallow sum of the session folder's file sizes. Session folders
    /// are flat with a handful of files, so this is cheap despite the
    /// multi-GB replay buffer inside.
    static func directorySizeBytes(_ url: URL) -> Int64? {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: url.path) else { return nil }
        var total: Int64 = 0
        for n in names {
            let p = url.appendingPathComponent(n).path
            if let attrs = try? fm.attributesOfItem(atPath: p),
               let s = (attrs[.size] as? NSNumber)?.int64Value {
                total += s
            }
        }
        return total
    }
}

// MARK: - Index cache (legacy sessions)

extension SessionManifest {

    /// Cache directory for manifests of sessions that don't carry their
    /// own `manifest.json`. Lives OUTSIDE the session folders so legacy
    /// saves stay byte-immutable.
    static func indexCacheDirectory() -> URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        )[0]
        return base
            .appendingPathComponent("DrewsChessMachine", isDirectory: true)
            .appendingPathComponent("SessionIndex", isDirectory: true)
    }

    /// Manifest for one session folder, resolved cheapest-first:
    /// 1. `manifest.json` inside the folder (new saves write it),
    /// 2. a valid index-cache entry (size+mtime of `session.json` match),
    /// 3. full extraction from `session.json` (then cached).
    /// Heavy in case 3 — call off the main actor.
    ///
    /// A stored manifest is used only at `currentSchema`, and its
    /// folder-name fields are recomputed from the folder's current name
    /// (`relabeled(forFolder:)`): they were parsed when it was written, under
    /// the name the folder had then and by the parser of that build.
    static func resolve(sessionFolder url: URL) -> SessionManifest {
        let fm = FileManager.default
        let embedded = url.appendingPathComponent("manifest.json")
        if let m = storedManifest(at: embedded), m.schema == currentSchema {
            return m.relabeled(forFolder: url)
        }

        let jsonURL = url.appendingPathComponent("session.json")
        var curBytes: Int64? = nil
        var curMTime: Double? = nil
        if let attrs = try? fm.attributesOfItem(atPath: jsonURL.path) {
            curBytes = (attrs[.size] as? NSNumber)?.int64Value
            curMTime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970
        }
        let cacheURL = indexCacheDirectory()
            .appendingPathComponent(url.lastPathComponent + ".json")
        if let m = storedManifest(at: cacheURL), m.schema == currentSchema,
           m.sourceJSONBytes == curBytes, m.sourceJSONMTime == curMTime,
           m.loadError == nil {
            return m.relabeled(forFolder: url)
        }

        let extracted = extract(fromSessionFolder: url)
        if extracted.loadError == nil {
            do {
                try fm.createDirectory(at: indexCacheDirectory(), withIntermediateDirectories: true)
                let data = try JSONEncoder().encode(extracted)
                try data.write(to: cacheURL, options: .atomic)
            } catch {
                SessionLogger.shared.log(
                    "[SESSION-INDEX] cache write failed for \(url.lastPathComponent): \(error.localizedDescription)"
                )
            }
        }
        return extracted
    }

    /// A stored manifest (`manifest.json` in a session folder, or an index-cache
    /// entry), or nil when there is none. One that is present but does not
    /// read is logged and treated as absent: the manifest is then extracted
    /// from `session.json` again, which is what it summarizes.
    private static func storedManifest(at url: URL) -> SessionManifest? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            return try JSONDecoder().decode(SessionManifest.self, from: Data(contentsOf: url))
        } catch {
            SessionLogger.shared.log("[SESSION-INDEX] \(url.path) does not read (\(error.localizedDescription)); extracting from session.json")
            return nil
        }
    }

    /// This manifest for the folder at `url`: its current name, and the
    /// trigger and lineage tag that name states for this session's ID.
    func relabeled(forFolder url: URL) -> SessionManifest {
        var copy = self
        copy.folderName = url.lastPathComponent
        let identity = Self.identity(folderName: copy.folderName, recordedSessionID: sessionID)
        copy.trigger = identity.trigger
        copy.lineageTag = identity.lineageTag
        return copy
    }

    /// Delete index-cache entries whose session folder no longer exists.
    /// Cache files live outside the (immutable) session folders, so a
    /// deleted `.dcmsession` would otherwise leave its `<folderName>.json`
    /// orphaned forever. `liveFolderNames` is the set of `.dcmsession`
    /// folder names found in the current scan; any cache file not backed by
    /// one of them is reclaimed. Best-effort: a failed delete is logged and
    /// skipped (the orphan is harmless — the picker never reads it).
    static func pruneIndexCache(liveFolderNames: Set<String>) {
        let fm = FileManager.default
        let dir = indexCacheDirectory()
        guard let entries = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return }
        for cacheURL in entries where cacheURL.pathExtension == "json" {
            // Cache file is "<folderName>.json"; the backing folder is the
            // name with the ".json" suffix stripped (folder names already
            // carry their own ".dcmsession" extension).
            let backingFolder = cacheURL.deletingPathExtension().lastPathComponent
            guard !liveFolderNames.contains(backingFolder) else { continue }
            do {
                try fm.removeItem(at: cacheURL)
                SessionLogger.shared.log(
                    "[SESSION-INDEX] pruned orphan cache entry \(cacheURL.lastPathComponent)"
                )
            } catch {
                SessionLogger.shared.log(
                    "[SESSION-INDEX] orphan cache prune failed for \(cacheURL.lastPathComponent): \(error.localizedDescription)"
                )
            }
        }
    }
}
