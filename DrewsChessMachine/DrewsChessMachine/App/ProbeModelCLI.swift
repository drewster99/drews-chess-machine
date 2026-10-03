import Darwin
import Foundation

/// Headless tactical-probe evaluation of saved checkpoints — invoked from
/// `DrewsChessMachineApp.init`'s pre-flight branch on `--probe-model`,
/// before any SwiftUI / Metal GUI setup.
///
/// Investigation tool (not shipped UX) for the mid-run-blowup forensics:
/// the Lichess probe batteries only ever ran against the LIVE trainer
/// during training, so every run that predates the probes (or whose
/// session logs are gone) has no out-of-distribution measurement. This
/// flag retro-fills that hole: load a saved champion, run the 200-puzzle
/// and/or wide (~4,435-puzzle) battery through the exact same
/// `TacticalProbeRunner.runBatch` path the live watchers use, and emit
/// one JSON object per (checkpoint × set) — directly comparable to the
/// `[TACTICAL-LICHESS] tick` lines in historical session logs.
///
/// `--probe-model <path>` accepts:
///   - a `.dcmmodel` / `.safetensors` weight file,
///   - a `.dcmsession` directory (its champion file is probed),
///   - any other directory (every `*.dcmsession` directly inside it is
///     probed — the whole-Sessions-folder sweep).
///
/// Each checkpoint builds a fresh inference network from its own
/// embedded/legacy architecture, so checkpoints of different
/// architectures and encodings can be swept in one invocation.
///
/// `--probe-positions-out <file>` additionally writes one JSON line per
/// (checkpoint × set × position) — the probe's rank, probability and NLL of
/// the bookmove, top-1 move and probability, legal-masked entropy, illegal
/// mass, max |logit|, verdict, value W/D/L, and the puzzle's id / theme /
/// rating — so two checkpoints can be compared position by position (paired
/// tests, calibration, entropy by position type) instead of only through
/// battery means. Each battery's `nll` is exactly the mean of its
/// positions' `nll` (both use `ProbeBookmoveNLL`).
///
/// Interpretation reminder (learned the hard way on the KbHZ resume):
/// NLL is sharpness-confounded — a sharper policy pays more nats for the
/// same mistakes — so cross-model comparisons should read `argmax` /
/// `avgRank` alongside `nll`.
enum ProbeModelCLI {

    enum ProbeSet: String {
        case set200 = "200"
        case wide
        case both
    }

    /// `--probe-out` and `--probe-positions-out` refuse an existing file
    /// unless this flag is also given; the parser in `DrewsChessMachineApp`
    /// passes its presence as `replaceExistingOut`.
    static let replaceExistingOutFlag = "--probe-out-overwrite"

    /// Per-position output flag (see the type doc).
    static let positionsOutFlag = "--probe-positions-out"

    /// Run the probes and exit. `outPath`, when given, receives the same
    /// JSON lines as stdout; `positionsOutPath`, when given, receives the
    /// per-position lines (never printed to stdout — a wide battery is
    /// thousands of lines). Both are opened before any checkpoint is loaded
    /// (`openOutputs`), and each must be new — or, with `replaceExistingOut`,
    /// an existing *regular file*, which is emptied. Neither may be a probed
    /// checkpoint, and they may not be one file, whatever the flags; a refused
    /// pair creates and empties nothing.
    ///
    /// Exit status says how the run ended, so a script never mistakes a
    /// partial run for a complete one: an unusable pair of outputs ends the run
    /// before probing; a failed or unencodable write to either output ends it
    /// at once; a checkpoint that fails to load or probe, or whose results hold
    /// non-finite numbers (a blown-up net), is reported as an `"event":"error"`
    /// line — nothing of it is written to the positions file — and the sweep
    /// carries on, then exits non-zero naming every failed checkpoint.
    static func runAndExit(
        modelPath: String,
        set: ProbeSet,
        outPath: String?,
        positionsOutPath: String?,
        replaceExistingOut: Bool
    ) -> Never {
        SessionLogger.shared.start()

        if replaceExistingOut && outPath == nil && positionsOutPath == nil {
            FileHandle.standardError.write(Data(
                "error: \(replaceExistingOutFlag) needs --probe-out <file> or \(positionsOutFlag) <file>\n".utf8
            ))
            SessionLogger.shared.shutdown()
            Darwin.exit(66)
        }

        let expanded = (modelPath as NSString).expandingTildeInPath
        let rootURL = URL(fileURLWithPath: expanded)
        let targets = resolveTargets(rootURL: rootURL)
        guard !targets.isEmpty else {
            FileHandle.standardError.write(Data(
                "error: --probe-model found no .dcmmodel/.safetensors/.dcmsession under \(rootURL.path)\n".utf8
            ))
            SessionLogger.shared.shutdown()
            Darwin.exit(61)
        }

        let outputs: ProbeOutputFiles
        do {
            outputs = try openOutputs(summaryPath: outPath,
                                      positionsPath: positionsOutPath,
                                      probeTargets: targets,
                                      replaceExisting: replaceExistingOut)
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.shutdown()
            Darwin.exit(65)
        }
        let handle = outputs.summaryHandle
        let positionsHandle = outputs.positionsHandle

        /// Print one already-encoded summary line and append it to
        /// `--probe-out`; a failed write ends the run.
        func emit(_ line: Data) {
            guard let text = String(data: line, encoding: .utf8) else {
                FileHandle.standardError.write(Data("error: a probe output line is not UTF-8\n".utf8))
                SessionLogger.shared.shutdown()
                Darwin.exit(68)
            }
            print(text)
            if let handle {
                do {
                    try appendLine(line, to: handle)
                } catch {
                    FileHandle.standardError.write(Data(
                        "error: write to --probe-out failed: \(error.localizedDescription); the file may end in a partial line\n".utf8
                    ))
                    SessionLogger.shared.shutdown()
                    Darwin.exit(68)
                }
            }
        }

        /// Report one checkpoint as failed: an `"event":"error"` summary line
        /// (its fields are strings, so it always encodes).
        func emitFailure(of target: URL, reason: String) {
            let line: Data
            do {
                line = try encodeLine(["event": "error", "model": target.path, "error": reason])
            } catch {
                FileHandle.standardError.write(Data(
                    "error: could not encode the failure record for \(target.path): \(error.localizedDescription)\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(68)
            }
            emit(line)
        }

        FileHandle.standardError.write(Data(
            "[PROBE-MODEL] \(targets.count) checkpoint(s), set=\(set.rawValue)\n".utf8
        ))

        func writePositions(_ lines: [Data], to positionsHandle: FileHandle) {
            var data = Data()
            for line in lines {
                data.append(line)
                data.append(Data("\n".utf8))
            }
            do {
                try positionsHandle.write(contentsOf: data)
                try positionsHandle.synchronize()
            } catch {
                FileHandle.standardError.write(Data(
                    "error: write to \(positionsOutFlag) failed: \(error.localizedDescription)\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(67)
            }
        }

        let wantPositions = positionsHandle != nil
        var failedCheckpoints: [String] = []
        for target in targets {
            let outcome: ProbeOutcome
            do {
                outcome = try syncWait {
                    try await probeOne(weightFileURL: target, set: set, includePositions: wantPositions)
                }
            } catch {
                failedCheckpoints.append(target.path)
                emitFailure(of: target, reason: "\(error)")
                continue
            }
            // Every line of the checkpoint is encoded before any is written,
            // so a checkpoint whose results cannot be encoded (non-finite
            // numbers) leaves nothing partial in either output.
            let summaryLines: [Data]
            let positionLines: [Data]
            do {
                summaryLines = try outcome.summaries.map(encodeLine)
                positionLines = try outcome.positions.map(encodeLine)
            } catch {
                failedCheckpoints.append(target.path)
                emitFailure(of: target, reason: error.localizedDescription)
                continue
            }
            for line in summaryLines { emit(line) }
            if let positionsHandle {
                writePositions(positionLines, to: positionsHandle)
            }
        }

        SessionLogger.shared.shutdown()
        guard failedCheckpoints.isEmpty else {
            FileHandle.standardError.write(Data(
                ("[PROBE-MODEL] \(failedCheckpoints.count) of \(targets.count) checkpoint(s) failed: "
                    + failedCheckpoints.joined(separator: ", ") + "\n").utf8
            ))
            Darwin.exit(69)
        }
        Darwin.exit(0)
    }

    /// The run's two optional output files, open for writing.
    struct ProbeOutputFiles {
        let summaryHandle: FileHandle?
        let positionsHandle: FileHandle?
    }

    /// Why the probe's outputs could not be opened, or a line not encoded.
    enum ProbeOutputError: LocalizedError, Equatable {
        case outputIsAProbedCheckpoint(flag: String, path: String, checkpoint: String)
        case outputsMayBeTheSameFile(summaryPath: String, positionsPath: String)
        case outputExists(flag: String, path: String)
        case outputNotARegularFile(flag: String, path: String, kind: FileSafety.ItemKind)
        case outputOpenFailed(flag: String, path: String, reason: String)
        /// A refusal after the first output was created, whose removal then
        /// failed: the created file is still there.
        case refusedButCreatedOutputRemains(refusal: String, cleanupFailure: String)
        case nonFiniteValues(keys: [String])
        case notJSONEncodable(keys: [String])

        var errorDescription: String? {
            switch self {
            case let .outputIsAProbedCheckpoint(flag, path, checkpoint):
                return "\(flag) \(path) is the checkpoint \(checkpoint) being probed; refusing to write over it"
            case let .outputsMayBeTheSameFile(summaryPath, positionsPath):
                return "--probe-out \(summaryPath) and \(positionsOutFlag) \(positionsPath) may be the same file; "
                    + "they must be different files"
            case let .outputExists(flag, path):
                return "\(flag) \(path) already exists; pass \(replaceExistingOutFlag) to replace it"
            case let .outputNotARegularFile(flag, path, kind):
                return "\(flag) \(path) is a \(kind), not a regular file; refusing to write to it"
            case let .outputOpenFailed(flag, path, reason):
                return "\(flag) \(path): \(reason)"
            case let .refusedButCreatedOutputRemains(refusal, cleanupFailure):
                return "\(refusal); the output file this run had already created could not be removed: \(cleanupFailure)"
            case let .nonFiniteValues(keys):
                return "non-finite value(s) in: \(keys.joined(separator: ", "))"
            case let .notJSONEncodable(keys):
                return "value(s) JSON cannot represent in: \(keys.joined(separator: ", "))"
            }
        }
    }

    /// Validate and open the run's output files, before any checkpoint is
    /// loaded. In order, and with nothing created or emptied until every
    /// check has passed:
    ///
    /// 1. No output may name a probed checkpoint — directly, through `..` or a
    ///    symbolic link, by case, or as a hard link — whatever
    ///    `replaceExisting` says: opening it would empty the checkpoint.
    /// 2. The two outputs may not name one file.
    /// 3. An existing output must be a regular file, and replacing it needs
    ///    `replaceExisting`.
    ///
    /// Then the outputs are opened. A failure opening the second removes the
    /// first when this call created it (by identity, never by name), and the
    /// two open files' identities are compared, which catches one file reached
    /// two ways that no path check saw (a new file through a linked folder).
    ///
    /// What remains: checks 1–3 look at paths, so something linked into place
    /// between them and the opens is not seen by checks 1 and 3; check 2 is
    /// backed by the descriptor comparison. An existing summary file that
    /// another process swaps for a link to a checkpoint in that window would
    /// still be emptied.
    static func openOutputs(summaryPath: String?,
                            positionsPath: String?,
                            probeTargets: [URL],
                            replaceExisting: Bool) throws -> ProbeOutputFiles {
        let summaryFlag = "--probe-out"
        let summaryURL = summaryPath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let positionsURL = positionsPath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let outputs: [(flag: String, url: URL)] =
            [summaryURL.map { (summaryFlag, $0) }, positionsURL.map { (positionsOutFlag, $0) }].compactMap { $0 }

        for output in outputs {
            for target in probeTargets where try FileSafety.mayNameTheSameFile(output.url, target) {
                throw ProbeOutputError.outputIsAProbedCheckpoint(flag: output.flag, path: output.url.path, checkpoint: target.path)
            }
        }
        if let summaryURL, let positionsURL, try FileSafety.mayNameTheSameFile(summaryURL, positionsURL) {
            throw ProbeOutputError.outputsMayBeTheSameFile(summaryPath: summaryURL.path, positionsPath: positionsURL.path)
        }
        for output in outputs {
            guard let existing = try FileSafety.existingItem(at: output.url) else { continue }
            guard existing.kind == .regularFile else {
                throw ProbeOutputError.outputNotARegularFile(flag: output.flag, path: output.url.path, kind: existing.kind)
            }
            guard replaceExisting else {
                throw ProbeOutputError.outputExists(flag: output.flag, path: output.url.path)
            }
        }

        let policy: FileSafety.ExistingRegularFilePolicy = replaceExisting ? .truncate : .refuse
        let summary = try summaryURL.map { try openOutput(at: $0, flag: summaryFlag, policy: policy) }
        let positions: FileSafety.OpenedForWriting?
        do {
            positions = try positionsURL.map { try openOutput(at: $0, flag: positionsOutFlag, policy: policy) }
        } catch {
            // The checks above found nothing at the positions path, so a file
            // there now may be the summary file just created, reached by a
            // second path none of the checks saw.
            var refusal = error
            if case ProbeOutputError.outputExists = error, let summary, let summaryURL, let positionsURL {
                do {
                    if try FileSafety.resolvedIdentity(at: positionsURL) == summary.identity {
                        refusal = ProbeOutputError.outputsMayBeTheSameFile(summaryPath: summaryURL.path,
                                                                           positionsPath: positionsURL.path)
                    }
                } catch let identityError {
                    FileHandle.standardError.write(Data(
                        "warning: could not read the identity of \(positionsURL.path): \(identityError.localizedDescription)\n".utf8
                    ))
                }
            }
            throw abandon(summary, at: summaryURL, refusal: refusal)
        }
        if let summary, let positions, summary.identity == positions.identity, let summaryURL, let positionsURL {
            closeAfterRefusal(positions, at: positionsURL)
            throw abandon(summary, at: summaryURL,
                          refusal: ProbeOutputError.outputsMayBeTheSameFile(summaryPath: summaryURL.path,
                                                                            positionsPath: positionsURL.path))
        }
        return ProbeOutputFiles(summaryHandle: summary?.handle, positionsHandle: positions?.handle)
    }

    /// Open one output, mapping FileSafety's refusals to this tool's flags.
    private static func openOutput(at url: URL,
                                   flag: String,
                                   policy: FileSafety.ExistingRegularFilePolicy) throws -> FileSafety.OpenedForWriting {
        do {
            return try FileSafety.openForWritingReportingCreation(at: url, existingRegularFile: policy)
        } catch FileSafetyError.notARegularFile(_, let kind) {
            throw ProbeOutputError.outputNotARegularFile(flag: flag, path: url.path, kind: kind)
        } catch FileSafetyError.alreadyExists(_, let kind) {
            guard kind == .regularFile else {
                throw ProbeOutputError.outputNotARegularFile(flag: flag, path: url.path, kind: kind)
            }
            throw ProbeOutputError.outputExists(flag: flag, path: url.path)
        }
    }

    /// Close an output that is being given up on, and remove it when this run
    /// created it. Returns the error to throw: `refusal`, or — when the
    /// created file could not be removed — one saying it is still there.
    private static func abandon(_ opened: FileSafety.OpenedForWriting?, at url: URL?, refusal: Error) -> Error {
        guard let opened, let url else { return refusal }
        closeAfterRefusal(opened, at: url)
        guard opened.createdByThisCall else { return refusal }
        do {
            try FileSafety.removeOwnedItem(at: url, identity: opened.identity)
            return refusal
        } catch {
            return ProbeOutputError.refusedButCreatedOutputRemains(refusal: refusal.localizedDescription,
                                                                   cleanupFailure: error.localizedDescription)
        }
    }

    private static func closeAfterRefusal(_ opened: FileSafety.OpenedForWriting, at url: URL) {
        do {
            try opened.handle.close()
        } catch {
            FileHandle.standardError.write(Data("warning: closing \(url.path): \(error.localizedDescription)\n".utf8))
        }
    }

    /// One JSON output line (without its newline) for `record`, keys sorted.
    /// Every CLI that writes JSON lines encodes through here.
    ///
    /// `JSONSerialization` raises an Objective-C exception — which Swift
    /// cannot catch, so the process aborts — for a NaN or infinite number or
    /// any other value JSON cannot hold. A probe of a checkpoint whose weights
    /// blew up produces exactly such numbers, so the record is checked first
    /// and the offending keys are thrown as a Swift error.
    static func encodeLine(_ record: [String: Any]) throws -> Data {
        let nonFinite = nonFiniteKeyPaths(in: record, prefix: "")
        guard nonFinite.isEmpty else { throw ProbeOutputError.nonFiniteValues(keys: nonFinite) }
        guard JSONSerialization.isValidJSONObject(record) else {
            throw ProbeOutputError.notJSONEncodable(keys: record.keys.sorted())
        }
        return try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    }

    /// The key paths (`a.b` for nested dictionaries, `a[i]` in arrays) of
    /// every NaN or infinite number in `value`, sorted.
    private static func nonFiniteKeyPaths(in value: Any, prefix: String) -> [String] {
        switch value {
        case let number as Double:
            return number.isFinite ? [] : [prefix]
        case let number as Float:
            return number.isFinite ? [] : [prefix]
        case let dictionary as [String: Any]:
            return dictionary.sorted { $0.key < $1.key }.flatMap { entry in
                nonFiniteKeyPaths(in: entry.value, prefix: prefix.isEmpty ? entry.key : "\(prefix).\(entry.key)")
            }
        case let array as [Any]:
            return array.enumerated().flatMap { nonFiniteKeyPaths(in: $0.element, prefix: "\(prefix)[\($0.offset)]") }
        default:
            return []
        }
    }

    /// Append `line` and a newline to `handle` and flush it.
    static func appendLine(_ line: Data, to handle: FileHandle) throws {
        try handle.write(contentsOf: line)
        try handle.write(contentsOf: Data("\n".utf8))
        try handle.synchronize()
    }

    /// Expand the user's path into the list of weight files to probe.
    /// Precedence: weight file as-is; `.dcmsession` dir → its champion;
    /// other dir → champions of all `*.dcmsession` children, sorted by
    /// name (the save-timestamp prefix makes that chronological).
    private static func resolveTargets(rootURL: URL) -> [URL] {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDir) else { return [] }
        if !isDir.boolValue {
            let ext = rootURL.pathExtension.lowercased()
            return (ext == "dcmmodel" || ext == "safetensors") ? [rootURL] : []
        }
        if rootURL.pathExtension.lowercased() == "dcmsession" {
            let champion = SessionCheckpointLayout.existingChampionURL(in: rootURL)
            return fm.fileExists(atPath: champion.path) ? [champion] : []
        }
        let children: [URL]
        do {
            children = try fm.contentsOfDirectory(
                at: rootURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
            )
        } catch {
            FileHandle.standardError.write(Data(
                "error: cannot list \(rootURL.path): \(error.localizedDescription)\n".utf8
            ))
            return []
        }
        return children
            .filter { $0.pathExtension.lowercased() == "dcmsession" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { session in
                let champion = SessionCheckpointLayout.existingChampionURL(in: session)
                return fm.fileExists(atPath: champion.path) ? champion : nil
            }
    }

    /// One checkpoint's output: a summary per battery, and — when requested —
    /// one record per position of every battery, in battery order.
    private struct ProbeOutcome {
        let summaries: [[String: Any]]
        let positions: [[String: Any]]
    }

    /// Load one checkpoint into a fresh inference network (built from the
    /// checkpoint's own embedded/legacy architecture) and run the
    /// requested batteries in a single batched forward pass, exactly like
    /// `LichessProbeWatcher.tickOnce`. Returns one JSON-ready dictionary
    /// per battery, plus per-position records when `includePositions`.
    private static func probeOne(weightFileURL: URL, set: ProbeSet, includePositions: Bool) async throws -> ProbeOutcome {
        let file = try CheckpointManager.loadModelFile(at: weightFileURL)
        let network = try ChessMPSNetwork(.overwrittenByLoad, arch: file.architecture)
        // Trainer files carry optimizer velocity after the base block;
        // inference needs only the leading trainables + BN running stats
        // (same prefix rule as UCIModelLoader.buildAndLoad).
        let baseCount = network.network.trainableVariables.count
            + network.network.bnRunningStatsVariables.count
        guard file.weights.count >= baseCount else {
            throw ProbeModelError.weightCountTooSmall(have: file.weights.count, need: baseCount)
        }
        try await network.network.loadWeights(Array(file.weights.prefix(baseCount)))

        // Battery layout mirrors the live watcher: one combined encode +
        // one batched forward, split back per set afterward.
        let primary: [TacticalProbe] = set == .wide ? [] : LichessProbeData.largeSet
        let wide: [TacticalProbe] = set == .set200 ? [] : LichessProbeData.wideSet
        let probes = primary + wide
        let encoding = network.inputEncoding
        var input = [Float]()
        input.reserveCapacity(probes.count * BoardEncoder.tensorLength(for: encoding))
        for probe in probes {
            input.append(contentsOf: BoardEncoder.encode(probe.state, encoding: encoding))
        }

        let batch = await TacticalProbeRunner.runBatch(probes, encodedInput: input, against: network)
        guard batch.results.count == probes.count else {
            throw ProbeModelError.resultCountMismatch(have: batch.results.count, want: probes.count)
        }

        // Per-position logit-abs-max aligns 1:1 with `batch.results`; an
        // older/error path may hand back an empty array — slice defensively.
        let logitAbsMax = batch.logitAbsMaxPerPos.count == batch.results.count
            ? batch.logitAbsMaxPerPos : []

        var emitted: [[String: Any]] = []
        var positions: [[String: Any]] = []
        let batteries: [(label: String, range: Range<Int>)] = [
            ("200", 0..<primary.count),
            ("wide", primary.count..<probes.count),
        ]
        for battery in batteries where !battery.range.isEmpty {
            let results = Array(batch.results[battery.range])
            let logits = logitAbsMax.isEmpty ? [] : Array(logitAbsMax[battery.range])
            emitted.append(summary(
                of: results, logitAbsMaxPerPos: logits,
                setLabel: battery.label, file: file, weightFileURL: weightFileURL, gpuMs: batch.gpuMs
            ))
            if includePositions {
                for (index, result) in results.enumerated() {
                    positions.append(positionRecord(
                        result,
                        index: index,
                        setLabel: battery.label,
                        logitAbsMax: logits.isEmpty ? nil : logits[index],
                        modelID: file.modelID,
                        modelPath: weightFileURL.path
                    ))
                }
            }
        }
        return ProbeOutcome(summaries: emitted, positions: positions)
    }

    /// One position's `--probe-positions-out` record. `index` is the
    /// position's 0-based place in its battery (the bundled set's order, the
    /// pairing key between checkpoints). `nll` is
    /// `ProbeBookmoveNLL.nats(expectedProb:)`, so a battery's mean `nll`
    /// equals its summary `nll`. Fields that do not exist for a position are
    /// omitted rather than defaulted: `expectedRank` for a probe whose
    /// acceptable move is not legal or that errored, `top1Move`/`top1Prob`
    /// for an errored probe, the puzzle fields for a probe with no Lichess
    /// metadata, `logitAbsMax` when the forward pass did not return it.
    static func positionRecord(
        _ result: ProbeResult,
        index: Int,
        setLabel: String,
        logitAbsMax: Float?,
        modelID: String,
        modelPath: String
    ) -> [String: Any] {
        var record: [String: Any] = [
            "model": modelPath,
            "modelID": modelID,
            "set": setLabel,
            "index": index,
            "name": result.probe.name,
            "category": result.probe.category.rawValue,
            "legalCount": result.legalCount,
            "expectedProb": Double(result.expectedProb),
            "nll": ProbeBookmoveNLL.nats(expectedProb: result.expectedProb),
            "entropyNats": Double(result.legalEntropyNats),
            "uniformEntropyNats": Double(result.uniformLegalEntropy),
            "illegalMass": Double(result.illegalMass),
            "verdict": result.verdict.rawValue,
            "valueWin": Double(result.valueWDL.win),
            "valueDraw": Double(result.valueWDL.draw),
            "valueLoss": Double(result.valueWDL.loss),
        ]
        if let rank = result.expectedRank {
            record["expectedRank"] = rank
        }
        if let top = result.topMoves.first {
            record["top1Move"] = top.move.uci
            record["top1Prob"] = Double(top.prob)
        }
        if let meta = LichessProbeData.metadata[result.probe.name] {
            record["puzzleId"] = meta.id
            record["theme"] = meta.theme
            record["rating"] = meta.rating
        }
        if let logitAbsMax {
            record["logitAbsMax"] = Double(logitAbsMax)
        }
        return record
    }

    /// Fold one battery's results into the same overall metrics the
    /// `[TACTICAL-LICHESS] tick` log line reports.
    private static func summary(
        of results: [ProbeResult],
        logitAbsMaxPerPos: [Float],
        setLabel: String,
        file: ModelCheckpointFile,
        weightFileURL: URL,
        gpuMs: Double
    ) -> [String: Any] {
        let aggregates = LichessProbeHistory.aggregates(from: results)
        let overall = LichessProbeOverallSummary(folding: aggregates)
        let pairs: [(rating: Int, correct: Bool)] = results.compactMap {
            guard let meta = LichessProbeData.metadata[$0.probe.name] else { return nil }
            let correct = $0.verdict == .correctAndConfident || $0.verdict == .correctButFlat
            return (rating: meta.rating, correct: correct)
        }
        let elo = LichessProbeHistory.mlePuzzleElo(pairs: pairs)

        var themes: [String: String] = [:]
        for agg in aggregates {
            themes[agg.theme.rawValue] = "\(agg.argmaxCorrect)/\(agg.total)"
        }

        var obj: [String: Any] = [
            "model": weightFileURL.path,
            "session": weightFileURL.deletingLastPathComponent().lastPathComponent,
            "modelID": file.modelID,
            "params": file.architecture.parameterCount,
            "encoding": file.architecture.inputEncoding.rawValue,
            "policyTailPrecision": ChessNetwork.PolicyTailPrecision.process.rawValue,
            "set": setLabel,
            "n": overall.totalProbes,
            "argmaxCorrect": overall.argmaxCorrect,
            "top5Correct": overall.top5Correct,
            "avgProb": Double(overall.avgExpectedProb),
            "nll": Double(overall.meanNegLogProb),
            "gpuMs": gpuMs,
            "themes": themes,
        ]
        if let avgRank = overall.avgExpectedRank {
            obj["avgRank"] = avgRank
        }
        if elo.isFinite {
            obj["pElo"] = elo
        }
        // Policy-logit magnitude over this battery. `mean` matches the
        // trainer's `pLogitAbsMax` diagnostic (mean over positions of the
        // per-position max |logit|); `peak` is the worst single position,
        // the more sensitive read on logit blow-up. Emitted only when the
        // per-position array survived the forward pass.
        if let peak = logitAbsMaxPerPos.max() {
            let sum = logitAbsMaxPerPos.reduce(0, +)
            obj["policy_logit_abs_max"] = Double(sum / Float(logitAbsMaxPerPos.count))
            obj["policy_logit_abs_max_peak"] = Double(peak)
        }
        return obj
    }

    private enum ProbeModelError: Swift.Error, CustomStringConvertible {
        case weightCountTooSmall(have: Int, need: Int)
        case resultCountMismatch(have: Int, want: Int)

        var description: String {
            switch self {
            case .weightCountTooSmall(let have, let need):
                return "weight file has \(have) tensors but the network needs at least \(need)"
            case .resultCountMismatch(let have, let want):
                return "batched probe returned \(have) results for \(want) probes"
            }
        }
    }

    /// Bridge async → sync (mirrors `ArchSweepCLI.syncWait`).
    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = ProbeModelSyncBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("ProbeModelCLI.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}

private final class ProbeModelSyncBox<T>: @unchecked Sendable {
    var success: T?
    var failure: Error?
}
