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
    /// thousands of lines). Both are opened before any checkpoint is loaded,
    /// and each must be new — or, with `replaceExistingOut`, an existing
    /// *regular file*, which is emptied first. A directory, symbolic link or
    /// other non-regular item there is always refused. Any failure to open
    /// either ends the process with a non-zero exit before the (long) probe
    /// run starts, and a failed per-position write ends it immediately, so a
    /// run never completes with its results silently undelivered or a
    /// per-position file silently truncated.
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
            Darwin.exit(61)
        }

        if let outPath, let positionsOutPath {
            let summaryURL = URL(fileURLWithPath: (outPath as NSString).expandingTildeInPath).standardizedFileURL
            let positionsURL = URL(fileURLWithPath: (positionsOutPath as NSString).expandingTildeInPath).standardizedFileURL
            if summaryURL.path == positionsURL.path {
                FileHandle.standardError.write(Data(
                    "error: --probe-out and \(positionsOutFlag) must be different files\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(65)
            }
        }

        func openOutput(_ path: String, flagName: String) -> FileHandle {
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            do {
                return try FileSafety.openForWriting(
                    at: url,
                    existingRegularFile: replaceExistingOut ? .truncate : .refuse
                )
            } catch FileSafetyError.alreadyExists(path: let path, kind: .regularFile) {
                FileHandle.standardError.write(Data(
                    "error: \(flagName) \(path) already exists; pass \(replaceExistingOutFlag) to replace it\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(65)
            } catch {
                FileHandle.standardError.write(Data(
                    "error: \(flagName): \(error.localizedDescription)\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(65)
            }
        }

        let handle: FileHandle? = outPath.map { openOutput($0, flagName: "--probe-out") }
        let positionsHandle: FileHandle? = positionsOutPath.map { openOutput($0, flagName: positionsOutFlag) }

        func emit(_ obj: [String: Any]) {
            let data: Data
            do {
                data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
            } catch {
                FileHandle.standardError.write(Data(
                    "error: JSON encode failed: \(error.localizedDescription)\n".utf8
                ))
                return
            }
            guard let line = String(data: data, encoding: .utf8) else {
                FileHandle.standardError.write(Data("error: JSON bytes are not UTF-8\n".utf8))
                return
            }
            print(line)
            if let handle {
                do {
                    try handle.write(contentsOf: data)
                    try handle.write(contentsOf: Data("\n".utf8))
                    try handle.synchronize()
                } catch {
                    FileHandle.standardError.write(Data(
                        "error: write to --probe-out failed: \(error.localizedDescription)\n".utf8
                    ))
                }
            }
        }

        FileHandle.standardError.write(Data(
            "[PROBE-MODEL] \(targets.count) checkpoint(s), set=\(set.rawValue)\n".utf8
        ))

        func writePositions(_ records: [[String: Any]], to positionsHandle: FileHandle) {
            do {
                var data = Data()
                for record in records {
                    data.append(try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]))
                    data.append(Data("\n".utf8))
                }
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
        for target in targets {
            do {
                let outcome = try syncWait {
                    try await probeOne(weightFileURL: target, set: set, includePositions: wantPositions)
                }
                for obj in outcome.summaries { emit(obj) }
                if let positionsHandle {
                    writePositions(outcome.positions, to: positionsHandle)
                }
            } catch {
                emit([
                    "event": "error",
                    "model": target.path,
                    "error": "\(error)",
                ])
            }
        }

        SessionLogger.shared.shutdown()
        Darwin.exit(0)
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
        let network = try ChessMPSNetwork(.randomWeights, arch: file.architecture)
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
        if !logitAbsMaxPerPos.isEmpty {
            let sum = logitAbsMaxPerPos.reduce(0, +)
            obj["policy_logit_abs_max"] = Double(sum / Float(logitAbsMaxPerPos.count))
            obj["policy_logit_abs_max_peak"] = Double(logitAbsMaxPerPos.max() ?? 0)
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
