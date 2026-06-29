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

    // MARK: Identity

    let folderName: String
    /// Save trigger parsed from the folder name (`manual`, `periodic`,
    /// `promote`, `sigusr2`, …) — whatever suffix the save used; the
    /// picker badges known values and shows the raw string otherwise.
    let trigger: String?
    /// Save instant. Folder-name timestamps are UTC by convention;
    /// `savedAtUnix` from `session.json` wins when available (exact).
    let savedAt: Date?
    /// The 4-char ModelID suffix of the SAVED lineage (e.g. `5K7Z`) —
    /// the picker's grouping key, per the "runs are the mental model"
    /// design decision.
    let lineageTag: String?

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

    let championID: String?
    let trainerID: String?
    let trainingSteps: Int?
    let elapsedTrainingSec: Double?
    let emittedGames: Int?
    let emittedPositions: Int?
    let bufferStored: Int?
    let bufferCapacity: Int?
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

    /// Parses `<yyyyMMdd>-<HHmmss>-<yyyymmdd>-<N>-<XXXX>-<trigger>.dcmsession`.
    /// Returns nil components for names that don't match (foreign folders
    /// still get listed, just unannotated).
    static func parseFolderName(
        _ name: String
    ) -> (savedAt: Date?, lineageTag: String?, trigger: String?) {
        guard name.hasSuffix(".dcmsession") else { return (nil, nil, nil) }
        let stem = String(name.dropLast(".dcmsession".count))
        let parts = stem.split(separator: "-").map(String.init)
        guard parts.count >= 6 else { return (nil, nil, nil) }
        let dateStr = parts[0], timeStr = parts[1]
        var savedAt: Date? = nil
        if dateStr.count == 8, timeStr.count == 6,
           dateStr.allSatisfy(\.isNumber), timeStr.allSatisfy(\.isNumber) {
            var comps = DateComponents()
            comps.year = Int(dateStr.prefix(4))
            comps.month = Int(dateStr.dropFirst(4).prefix(2))
            comps.day = Int(dateStr.dropFirst(6))
            comps.hour = Int(timeStr.prefix(2))
            comps.minute = Int(timeStr.dropFirst(2).prefix(2))
            comps.second = Int(timeStr.dropFirst(4))
            var cal = Calendar(identifier: .gregorian)
            // Session folder timestamps are UTC (established empirically;
            // see ARCH_EXPERIMENTS Exp 4 §2 note "Session filenames are
            // UTC-stamped").
            cal.timeZone = TimeZone(identifier: "UTC") ?? .current
            savedAt = cal.date(from: comps)
        }
        // The trigger is the LAST hyphen component; the lineage tag is the
        // one before it (ModelID = yyyymmdd-N-XXXX occupies parts 2..4 of
        // a conforming name, but counting from the end tolerates future
        // extra middle segments).
        let trigger = parts[parts.count - 1]
        let lineage = parts[parts.count - 2]
        return (savedAt, lineage, trigger)
    }
}

// MARK: - Extraction from session.json

extension SessionManifest {

    /// Build a manifest by reading a session folder from disk. Heavy
    /// (parses `session.json`, which can be ~16 MB) — call off the main
    /// actor. Never throws: failures produce a manifest with `loadError`
    /// set so the picker can show the folder as unreadable.
    static func extract(fromSessionFolder url: URL) -> SessionManifest {
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
    static func extract(
        jsonDict dict: [String: Any],
        folderName: String,
        disk: Int64?,
        srcBytes: Int64?,
        srcMTime: Double?
    ) -> SessionManifest {
        let parsed = parseFolderName(folderName)

        func int(_ k: String) -> Int? { (dict[k] as? NSNumber)?.intValue }
        func dbl(_ k: String) -> Double? { (dict[k] as? NSNumber)?.doubleValue }
        func str(_ k: String) -> String? { dict[k] as? String }
        func bool(_ k: String) -> Bool? { (dict[k] as? NSNumber)?.boolValue }

        // Architecture: runtime-config saves embed the snake_case
        // NetworkArchitecture (decode it for the canonical summary);
        // legacy saves carry a camelCase scalar dict.
        var archSummary: String? = nil
        var paramCount: Int? = nil
        var inputPlanes: Int? = nil
        var chans: Int? = nil
        var blocks: Int? = nil
        if let arch = dict["architecture"] as? [String: Any] {
            func aInt(_ k: String) -> Int? { (arch[k] as? NSNumber)?.intValue }
            if arch["input_encoding"] != nil,
               let archData = try? JSONSerialization.data(withJSONObject: arch),
               let decoded = try? JSONDecoder().decode(NetworkArchitecture.self, from: archData) {
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
                let v = aInt("architectureVersion").map { "v\($0) " } ?? ""
                if let b = blocks, let c = chans {
                    let planes = inputPlanes.map { " in \($0) planes" } ?? ""
                    archSummary = "\(v)\(b)x[\(c)ch]\(planes) (legacy)"
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
        promotionCount = p200.promos ?? pWide.promos
        // An empty arenaHistory array yields count 0 (non-nil) and would
        // mask the probe-history fallback; treat 0 as "no arena data here"
        // and prefer the probe's count when it has one.
        if (arenaCount ?? 0) == 0 { arenaCount = p200.arenas ?? arenaCount }
        // Tolerate a non-`[[String:Any]]` arenaHistory shape (legacy /
        // heterogeneous): count only the entries that parse as promoted,
        // rather than failing the whole cast and leaving promotionCount nil.
        if promotionCount == nil, let arenas = dict["arenaHistory"] as? [Any] {
            promotionCount = arenas.reduce(0) { acc, e in
                acc + (((e as? [String: Any])?["promoted"] as? NSNumber)?.boolValue == true ? 1 : 0)
            }
        }

        // Draw counters are written as one atomic block, so a partial subset
        // means an unexpected/hand-edited schema — report nil ("—") rather
        // than a misleadingly-low "total" that silently omits a category.
        let drawParts = [int("stalemates"), int("fiftyMoveDraws"),
                         int("threefoldRepetitionDraws"), int("insufficientMaterialDraws")]
        let draws: [Int] = drawParts.contains(where: { $0 == nil }) ? [] : drawParts.compactMap { $0 }
        let savedAt = dbl("savedAtUnix").map { Date(timeIntervalSince1970: $0) } ?? parsed.savedAt

        return SessionManifest(
            folderName: folderName,
            trigger: parsed.trigger,
            savedAt: savedAt,
            lineageTag: parsed.lineageTag,
            architectureSummary: archSummary,
            parameterCount: paramCount,
            inputPlanes: inputPlanes,
            channels: chans,
            numBlocks: blocks,
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
            channels: nil, numBlocks: nil, championID: nil, trainerID: nil,
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
        )
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
    static func resolve(sessionFolder url: URL) -> SessionManifest {
        let fm = FileManager.default
        let embedded = url.appendingPathComponent("manifest.json")
        if let data = try? Data(contentsOf: embedded),
           let m = try? JSONDecoder().decode(SessionManifest.self, from: data) {
            return m
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
        if let data = try? Data(contentsOf: cacheURL),
           let m = try? JSONDecoder().decode(SessionManifest.self, from: data),
           m.sourceJSONBytes == curBytes, m.sourceJSONMTime == curMTime,
           m.loadError == nil {
            return m
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
