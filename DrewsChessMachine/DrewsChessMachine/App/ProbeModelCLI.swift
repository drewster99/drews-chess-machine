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

    static func runAndExit(modelPath: String, set: ProbeSet, outPath: String?) -> Never {
        SessionLogger.shared.start()

        let expanded = (modelPath as NSString).expandingTildeInPath
        let rootURL = URL(fileURLWithPath: expanded)
        let targets = resolveTargets(rootURL: rootURL)
        guard !targets.isEmpty else {
            FileHandle.standardError.write(Data(
                "error: --probe-model found no .dcmmodel/.safetensors/.dcmsession under \(rootURL.path)\n".utf8
            ))
            Darwin.exit(61)
        }

        var handle: FileHandle?
        if let outPath {
            let expandedOut = (outPath as NSString).expandingTildeInPath
            FileManager.default.createFile(atPath: expandedOut, contents: nil)
            handle = FileHandle(forWritingAtPath: expandedOut)
        }

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

        for target in targets {
            do {
                let results = try syncWait { try await probeOne(weightFileURL: target, set: set) }
                for obj in results { emit(obj) }
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

    /// Load one checkpoint into a fresh inference network (built from the
    /// checkpoint's own embedded/legacy architecture) and run the
    /// requested batteries in a single batched forward pass, exactly like
    /// `LichessProbeWatcher.tickOnce`. Returns one JSON-ready dictionary
    /// per battery.
    private static func probeOne(weightFileURL: URL, set: ProbeSet) async throws -> [[String: Any]] {
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

        var emitted: [[String: Any]] = []
        if !primary.isEmpty {
            emitted.append(summary(
                of: Array(batch.results[0..<primary.count]),
                setLabel: "200", file: file, weightFileURL: weightFileURL, gpuMs: batch.gpuMs
            ))
        }
        if !wide.isEmpty {
            emitted.append(summary(
                of: Array(batch.results[primary.count...]),
                setLabel: "wide", file: file, weightFileURL: weightFileURL, gpuMs: batch.gpuMs
            ))
        }
        return emitted
    }

    /// Fold one battery's results into the same overall metrics the
    /// `[TACTICAL-LICHESS] tick` log line reports.
    private static func summary(
        of results: [ProbeResult],
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
