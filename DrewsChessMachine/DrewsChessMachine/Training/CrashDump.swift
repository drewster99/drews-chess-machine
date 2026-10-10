import Foundation
import OSLog

/// Why a crash dump was written.
enum CrashDumpReason: String, Sendable, Codable {
    /// A GPU fault while the trainer was active (a failed submission, or
    /// macOS reporting a hang / discarded buffer); training stops.
    case gpuFault = "gpu-fault"
    /// A non-finite loss or gradient; training stops.
    case nonFinite = "non-finite"
    /// A pre-clip gradient norm far above its reference; training goes on.
    case nearMiss = "near-miss"
}

/// Everything a crash dump records besides what `CrashDumpWriter` reads
/// itself (the log tail, the system log, macOS's GPU reports, the trainer's
/// weights). Built by the training path that saw the problem.
struct CrashDumpContext: Sendable {
    let reason: CrashDumpReason
    /// The error or condition, verbatim.
    let detail: String
    /// `replay`, `vsuci` or `gui`.
    let pathKind: String
    let modelID: String
    /// The trainer's completed-step clock when the dump was taken.
    let trainerStep: Int
    /// The trainer step the dumped batch belongs to: the failing step for a
    /// failure the step itself reported; a later step for a fault seen late
    /// (the faulted step's batch is gone — its hash, if it was taken, is in
    /// `recent_batch_hashes`).
    let batchTrainerStep: Int
    let learningRate: Double
    let momentum: Double
    /// The run's `[RUN]` line (seed, parameters hash, build, device, lineage
    /// run and segment).
    let runProvenance: String?
    let batch: CapturedTrainingBatch?
    let batchHashes: [BatchHashChain.Entry]
    let gradientNorms: GradientNormHistory?
    let faults: [GPUFaultLedger.Fault]
    /// Parts that couldn't be gathered, and why.
    let notes: [String]
}

/// A copy of the batch a training step consumed (boards, played-move
/// indices, outcomes after the draw-penalty rewrite).
struct CapturedTrainingBatch: Sendable {
    let batchSize: Int
    let floatsPerBoard: Int
    let boards: [Float]
    let moves: [Int32]
    let outcomes: [Float]
}

/// Writes a crash dump: one folder under `CrashDumps/` with everything known
/// about a training failure, for investigating it afterwards (GPU fault
/// forensics plan, Part C; owner request 2026-10-09).
///
/// Order matters, because a dump is written while something is wrong:
/// 1. the parts already in memory — `manifest.json`, `batch.safetensors`,
///    `log-tail.txt` (the session logger is flushed first),
///    `system-log.txt` (this process's system-log entries for the last two
///    minutes) and copies of macOS's `gpuEvent` reports for this process —
///    are staged in `<name>.dcmcrash.tmp/` and published by an atomic rename;
/// 2. then the trainer's weights and optimizer velocity are read back from
///    the GPU into `weights-after.safetensors` with a per-tensor non-finite
///    census, under a time limit: right after a GPU hang the read can hang
///    too, and the dump must never block the halt. A read that fails or runs
///    out of time leaves `weights-after-error.txt` instead.
///
/// `batch.safetensors` and `weights-after.safetensors` are plain safetensors
/// (`SafetensorsFile`), not DrewsChessMachine model files: no lineage, no
/// test-set results, never loadable as a model. Nothing here deletes or
/// overwrites anything: names are created exclusively (`FileSafety`), with a
/// `-2`, `-3`, … suffix if a name is taken. A failure is logged
/// (`[CRASH-DUMP-ERR]`) and returned as nil; it never throws into the halt.
enum CrashDumpWriter {
    /// How long the weights read may take before the dump gives up on it.
    static let weightsReadTimeLimitSeconds: UInt64 = 60
    /// How many session-log lines `log-tail.txt` keeps.
    static let logTailLines = 5_000
    /// How far back `system-log.txt` reads this process's system log.
    static let systemLogLookBackSeconds: TimeInterval = 120

    /// Writes the dump into `directory` and returns its folder, or nil when
    /// it couldn't be written (logged). `weights` reads the trainer's weights
    /// and velocity; it is called last, under the time limit.
    /// Where dumps go: `CrashDumps/` in the app; a scratch folder under
    /// XCTest, so a test that trips a dump never writes into the user's
    /// folder.
    static var defaultDirectory: URL {
        XCTestHostDetection.isRunningUnderXCTest
            ? FileManager.default.temporaryDirectory.appendingPathComponent("DrewsChessMachine-XCTest-CrashDumps",
                                                                            isDirectory: true)
            : CheckpointPaths.crashDumpsDir
    }

    static func write(
        _ context: CrashDumpContext,
        in directory: URL = CrashDumpWriter.defaultDirectory,
        now: Date = Date(),
        weights: (@Sendable () async throws -> (weights: [[Float]], velocity: [[Float]]))?
    ) async -> URL? {
        let folder: URL
        do {
            // Seconds of file and log I/O: on a utility queue, never on a
            // Swift concurrency thread.
            folder = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Error>) in
                DispatchQueue.global(qos: .utility).async {
                    do {
                        continuation.resume(returning: try writeMemoryParts(context, in: directory, now: now))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } catch {
            SessionLogger.shared.log("[CRASH-DUMP-ERR] could not write the crash dump: \(error.localizedDescription)")
            return nil
        }
        guard let weights else {
            SessionLogger.shared.log("[CRASH-DUMP] wrote \(folder.path) (\(context.reason.rawValue); no weights)")
            return folder
        }
        SessionLogger.shared.log("[CRASH-DUMP] wrote \(folder.path) (\(context.reason.rawValue)); reading the weights next")
        await writeWeights(into: folder, read: weights)
        return folder
    }

    // MARK: - Memory parts

    static func folderName(for context: CrashDumpContext, now: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: now))-\(context.modelID)-step\(context.trainerStep)-\(context.reason.rawValue)"
    }

    private static func writeMemoryParts(_ context: CrashDumpContext, in directory: URL, now: Date) throws -> URL {
        try CheckpointPaths.ensureDirectory(directory)
        let base = folderName(for: context, now: now)
        // A free final name and a free staging name, with -2, -3, … on a clash.
        var attempt = 1
        var finalURL: URL
        var stagingURL: URL
        while true {
            let name = attempt == 1 ? base : "\(base)-\(attempt)"
            finalURL = directory.appendingPathComponent("\(name).\(CheckpointPaths.crashDumpPathExtension)", isDirectory: true)
            stagingURL = directory.appendingPathComponent(
                "\(name)\(CheckpointPaths.crashDumpStagingSuffix)", isDirectory: true)
            if try FileSafety.existingItem(at: finalURL) == nil {
                do {
                    try FileSafety.createNewDirectory(at: stagingURL)
                    break
                } catch FileSafetyError.alreadyExists {
                    // Another dump's staging, or a leftover: try the next name.
                }
            }
            attempt += 1
            guard attempt <= 100 else {
                throw CrashDumpError.noFreeName(base)
            }
        }

        try FileSafety.writeNewFile(manifest(context, now: now), at: stagingURL.appendingPathComponent("manifest.json"))
        if let batch = context.batch {
            try FileSafety.writeNewFile(try batchFile(batch, context: context),
                                        at: stagingURL.appendingPathComponent("batch.safetensors"))
        }
        SessionLogger.shared.flush()
        try FileSafety.writeNewFile(Data(sessionLogTail().utf8), at: stagingURL.appendingPathComponent("log-tail.txt"))
        try FileSafety.writeNewFile(Data(systemLogText(now: now).utf8),
                                    at: stagingURL.appendingPathComponent("system-log.txt"))
        copyGPUEventReports(into: stagingURL, since: now.addingTimeInterval(-systemLogLookBackSeconds))
        try FileSafety.renameWithoutReplacing(from: stagingURL, to: finalURL)
        return finalURL
    }

    private static func manifest(_ context: CrashDumpContext, now: Date) throws -> Data {
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let manifest = CrashDumpManifest(
            formatVersion: 1,
            reason: context.reason.rawValue,
            detail: context.detail,
            writtenAt: iso.string(from: now),
            pathKind: context.pathKind,
            modelID: context.modelID,
            trainerStep: context.trainerStep,
            batchTrainerStep: context.batchTrainerStep,
            batchHash: context.batch.map {
                BatchHashChain.batchHash(boards: $0.boards, moves: $0.moves, outcomes: $0.outcomes)
            },
            learningRate: context.learningRate,
            momentum: context.momentum,
            build: "\(BuildInfo.buildNumber) \(BuildInfo.gitHash)",
            runProvenance: context.runProvenance,
            recentBatchHashes: context.batchHashes.map {
                CrashDumpManifest.BatchHash(trainerStep: $0.trainerStep, batchHash: $0.batchHash, chain: $0.chain)
            },
            gradientNorms: context.gradientNorms.map {
                CrashDumpManifest.GradientNorms(lastTrainerStep: $0.lastTrainerStep, preClipNorms: $0.preClipNorms,
                                                fedCaps: $0.fedCaps)
            },
            gpuFaults: context.faults.map(GPUFaultReport.Fault.init),
            gpuFaultMonitor: GPUFaultMonitor.shared.availability.label,
            otherProcesses: otherDrewsChessMachineProcesses(),
            notes: context.notes,
            weightsAfter: context.reason == .nearMiss
                ? "not read (a near miss: training goes on)"
                : "read after this manifest: weights-after.safetensors (with weights-after-census.json), "
                    + "or weights-after-error.txt")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(manifest)
    }

    private static func batchFile(_ batch: CapturedTrainingBatch, context: CrashDumpContext) throws -> Data {
        try SafetensorsFile.encode(
            tensors: [
                SafetensorsTensor(name: "boards", shape: [batch.batchSize, batch.floatsPerBoard], data: batch.boards),
                SafetensorsTensor(name: "moves", shape: [batch.batchSize], data: batch.moves.map { Float($0) }),
                SafetensorsTensor(name: "outcomes", shape: [batch.batchSize], data: batch.outcomes),
            ],
            metadata: [
                "dcm_crash_dump_batch": "1",
                "trainer_step": String(context.batchTrainerStep),
                "note": "Not a model file. boards: the network input planes per position; moves: played-move "
                    + "policy indices (exact integers); outcomes: as trained (after the draw-penalty rewrite).",
            ])
    }

    /// At most this many bytes are read from the end of the session log
    /// (a multi-day log is far larger than the tail kept).
    static let logTailMaximumBytes: UInt64 = 8 * 1_048_576

    /// The last `logTailLines` lines of this process's session log, from at
    /// most its last `logTailMaximumBytes`.
    private static func sessionLogTail() -> String {
        guard let path = SessionLogger.shared.activeLogPath else {
            return "(no session log file)\n"
        }
        let url = URL(fileURLWithPath: path)
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer {
                do {
                    try handle.close()
                } catch {
                    SessionLogger.shared.log("[CRASH-DUMP-ERR] closing \(url.path): \(error.localizedDescription)")
                }
            }
            let size = try handle.seekToEnd()
            let start = size > logTailMaximumBytes ? size - logTailMaximumBytes : 0
            try handle.seek(toOffset: start)
            let data = try handle.readToEnd() ?? Data()
            var lines = String(decoding: data, as: UTF8.self).split(separator: "\n", omittingEmptySubsequences: false)
            // A read that starts mid-file starts mid-line: drop that partial line.
            if start > 0, !lines.isEmpty { lines.removeFirst() }
            return lines.suffix(logTailLines).joined(separator: "\n")
        } catch {
            return "(could not read \(url.path): \(error.localizedDescription))\n"
        }
    }

    /// This process's system-log entries for the last two minutes.
    private static func systemLogText(now: Date) -> String {
        do {
            let store = try OSLogStore(scope: .currentProcessIdentifier)
            let entries = try store.getEntries(at: store.position(date: now.addingTimeInterval(-systemLogLookBackSeconds)))
            var lines: [String] = []
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
            for case let entry as OSLogEntryLog in entries {
                lines.append("\(formatter.string(from: entry.date)) [\(entry.subsystem):\(entry.category)] \(entry.composedMessage)")
            }
            return lines.joined(separator: "\n") + "\n"
        } catch {
            return "(could not read this process's system log: \(error.localizedDescription))\n"
        }
    }

    /// Copies macOS's GPU event reports for this process written since
    /// `since` (`gpuEvent-*` in the system and the user DiagnosticReports
    /// folders, when readable — they are often gone within hours). A report
    /// counts as this process's when it names this pid. Failures are noted in
    /// a file, not thrown.
    private static func copyGPUEventReports(into folder: URL, since: Date) {
        let pid = ProcessInfo.processInfo.processIdentifier
        let pidMarkers = ["\"pid\":\(pid)", "\"pid\" : \(pid)", "\"pid\": \(pid)"]
        let reportFolders = [
            URL(fileURLWithPath: "/Library/Logs/DiagnosticReports", isDirectory: true),
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Logs/DiagnosticReports", isDirectory: true),
        ]
        var notes: [String] = []
        for reports in reportFolders {
            let entries: [URL]
            do {
                entries = try FileManager.default.contentsOfDirectory(
                    at: reports, includingPropertiesForKeys: [.contentModificationDateKey], options: [])
            } catch {
                notes.append("could not list \(reports.path): \(error.localizedDescription)")
                continue
            }
            for entry in entries where entry.lastPathComponent.hasPrefix("gpuEvent-") {
                do {
                    let modified = try entry.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
                    guard let modified, modified >= since else { continue }
                    let data = try Data(contentsOf: entry)
                    let text = String(decoding: data.prefix(16_384), as: UTF8.self)
                    guard pidMarkers.contains(where: { text.contains($0) }) else { continue }
                    try FileSafety.writeNewFile(data, at: folder.appendingPathComponent(entry.lastPathComponent))
                } catch {
                    notes.append("\(entry.path): \(error.localizedDescription)")
                }
            }
        }
        if !notes.isEmpty {
            do {
                try FileSafety.writeNewFile(Data((notes.joined(separator: "\n") + "\n").utf8),
                                            at: folder.appendingPathComponent("gpu-event-reports-not-copied.txt"))
            } catch {
                SessionLogger.shared.log("[CRASH-DUMP-ERR] could not note the GPU reports not copied: \(error.localizedDescription)")
            }
        }
    }

    /// `pid command…` of every other DrewsChessMachine process, secret-bearing
    /// option values redacted. The command line comes from `ps` as one string
    /// (paths may contain spaces, e.g. `Application Support`); redaction is
    /// applied to its space-separated words, which is exact for options
    /// (they and secret values contain no spaces).
    private static func otherDrewsChessMachineProcesses() -> [String] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-axo", "pid=,command="]
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
        } catch {
            return ["(could not run ps: \(error.localizedDescription))"]
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let me = ProcessInfo.processInfo.processIdentifier
        return String(decoding: data, as: UTF8.self)
            .split(separator: "\n")
            .compactMap { line -> String? in
                let trimmed = line.drop { $0 == " " }
                guard let space = trimmed.firstIndex(of: " "), let pid = Int32(trimmed[..<space]), pid != me else {
                    return nil
                }
                let command = String(trimmed[trimmed.index(after: space)...])
                guard command.contains("/Contents/MacOS/DrewsChessMachine") else { return nil }
                let words = command.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
                return "\(pid) " + LineageRecord.redactedArguments(words).joined(separator: " ")
            }
    }

    // MARK: - Weights (GPU reads, last)

    private static func writeWeights(
        into folder: URL,
        read: @escaping @Sendable () async throws -> (weights: [[Float]], velocity: [[Float]])
    ) async {
        let outcome = await readWithTimeLimit(read)
        switch outcome {
        case .success(let values):
            do {
                var tensors: [SafetensorsTensor] = []
                var census: [WeightsCensusEntry] = []
                for (index, values) in values.weights.enumerated() {
                    tensors.append(SafetensorsTensor(name: "weight.\(index)", shape: [values.count], data: values))
                    census.append(WeightsCensusEntry(name: "weight.\(index)", values: values))
                }
                for (index, values) in values.velocity.enumerated() {
                    tensors.append(SafetensorsTensor(name: "velocity.\(index)", shape: [values.count], data: values))
                    census.append(WeightsCensusEntry(name: "velocity.\(index)", values: values))
                }
                try FileSafety.writeNewFile(
                    try SafetensorsFile.encode(tensors: tensors, metadata: [
                        "dcm_crash_dump_weights": "1",
                        "note": "Not a model file: the trainer's variables in its export order, flattened, "
                            + "as they were after the failure.",
                    ]),
                    at: folder.appendingPathComponent("weights-after.safetensors"))
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try FileSafety.writeNewFile(try encoder.encode(census),
                                            at: folder.appendingPathComponent("weights-after-census.json"))
                let bad = census.filter { $0.nonFinite > 0 }
                SessionLogger.shared.log("[CRASH-DUMP] weights written: \(census.count) tensors, "
                    + "\(bad.count) with non-finite values\(bad.isEmpty ? "" : " (first: \(bad[0].name))")")
            } catch {
                writeWeightsError("writing the weights failed: \(error.localizedDescription)", into: folder)
            }
        case .failure(let reason):
            writeWeightsError(reason, into: folder)
        }
    }

    private static func writeWeightsError(_ reason: String, into folder: URL) {
        SessionLogger.shared.log("[CRASH-DUMP-ERR] weights not written: \(reason)")
        do {
            try FileSafety.writeNewFile(Data((reason + "\n").utf8), at: folder.appendingPathComponent("weights-after-error.txt"))
        } catch {
            SessionLogger.shared.log("[CRASH-DUMP-ERR] could not write weights-after-error.txt: \(error.localizedDescription)")
        }
    }

    enum WeightsReadOutcome: Sendable {
        case success((weights: [[Float]], velocity: [[Float]]))
        case failure(String)
    }

    /// Runs `read` with a time limit. A read still running at the limit is
    /// abandoned, not cancelled (a GPU wait can't be cancelled); the halt that
    /// follows ends the process.
    static func readWithTimeLimit(
        _ read: @escaping @Sendable () async throws -> (weights: [[Float]], velocity: [[Float]]),
        limitSeconds: UInt64 = weightsReadTimeLimitSeconds
    ) async -> WeightsReadOutcome {
        let resumed = SyncBox(false)
        return await withCheckedContinuation { (continuation: CheckedContinuation<WeightsReadOutcome, Never>) in
            @Sendable func finish(_ outcome: WeightsReadOutcome) {
                let first = resumed.mutate { done -> Bool in
                    if done { return false }
                    done = true
                    return true
                }
                if first { continuation.resume(returning: outcome) }
            }
            Task.detached(priority: .userInitiated) {
                do {
                    finish(.success(try await read()))
                } catch {
                    finish(.failure("reading the weights failed: \(error.localizedDescription)"))
                }
            }
            Task.detached(priority: .utility) {
                do {
                    try await Task.sleep(for: .seconds(Double(limitSeconds)))
                } catch {
                    return
                }
                finish(.failure("reading the weights took longer than \(limitSeconds) s; abandoned"))
            }
        }
    }
}

enum CrashDumpError: LocalizedError {
    case noFreeName(String)

    var errorDescription: String? {
        switch self {
        case .noFreeName(let base):
            return "no free crash-dump folder name for \(base) after 100 attempts"
        }
    }
}

/// `manifest.json`.
struct CrashDumpManifest: Encodable, Sendable {
    struct BatchHash: Encodable, Sendable {
        let trainerStep: Int
        let batchHash: String
        let chain: String?

        enum CodingKeys: String, CodingKey {
            case trainerStep = "trainer_step"
            case batchHash = "batch_hash"
            case chain
        }
    }

    struct GradientNorms: Encodable, Sendable {
        let lastTrainerStep: Int?
        let preClipNorms: [Float]
        let fedCaps: [Float]

        enum CodingKeys: String, CodingKey {
            case lastTrainerStep = "last_trainer_step"
            case preClipNorms = "pre_clip_norms"
            case fedCaps = "fed_caps"
        }
    }

    let formatVersion: Int
    let reason: String
    let detail: String
    let writtenAt: String
    let pathKind: String
    let modelID: String
    let trainerStep: Int
    let batchTrainerStep: Int
    let batchHash: String?
    let learningRate: Double
    let momentum: Double
    let build: String
    let runProvenance: String?
    let recentBatchHashes: [BatchHash]
    let gradientNorms: GradientNorms?
    let gpuFaults: [GPUFaultReport.Fault]
    let gpuFaultMonitor: String
    let otherProcesses: [String]
    let notes: [String]
    let weightsAfter: String

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case reason, detail
        case writtenAt = "written_at"
        case pathKind = "path_kind"
        case modelID = "model_id"
        case trainerStep = "trainer_step"
        case batchTrainerStep = "batch_trainer_step"
        case batchHash = "batch_hash"
        case learningRate = "learning_rate"
        case momentum
        case build
        case runProvenance = "run_provenance"
        case recentBatchHashes = "recent_batch_hashes"
        case gradientNorms = "gradient_norms"
        case gpuFaults = "gpu_faults"
        case gpuFaultMonitor = "gpu_fault_monitor"
        case otherProcesses = "other_processes"
        case notes
        case weightsAfter = "weights_after"
    }
}

/// One tensor's line in `weights-after-census.json`.
struct WeightsCensusEntry: Encodable, Sendable {
    let name: String
    let count: Int
    let nonFinite: Int
    let maxAbsFinite: Float

    init(name: String, values: [Float]) {
        self.name = name
        count = values.count
        var nonFinite = 0
        var maxAbs: Float = 0
        for value in values {
            if value.isFinite {
                maxAbs = max(maxAbs, abs(value))
            } else {
                nonFinite += 1
            }
        }
        self.nonFinite = nonFinite
        maxAbsFinite = maxAbs
    }

    enum CodingKeys: String, CodingKey {
        case name, count
        case nonFinite = "non_finite"
        case maxAbsFinite = "max_abs_finite"
    }
}

// MARK: - Gathering a dump from a trainer

extension CrashDumpWriter {
    /// A pre-clip gradient norm at least this many times the relative cap's
    /// reference median writes a near-miss dump (GPU fault forensics plan,
    /// C1). The three GPU resets on 2026-10-09 corrupted steps to 10.3M×,
    /// 32.7M× and 37.5M× their reference; ordinary clips are under 10×.
    static let nearMissReferenceMultiple: Double = 1_000
    /// At most one near-miss dump per this many trainer steps.
    static let nearMissMinimumSpacingSteps = 1_000

    /// Whether a completed step is a near miss: its pre-clip gradient norm
    /// reached `nearMissReferenceMultiple` × the cap's reference median.
    /// False while the cap has no reference yet (warm-up, or mode off).
    static func isNearMiss(preClipNorm: Float, decision: GradientCapDecision) -> Bool {
        guard let reference = decision.referenceMedian, reference > 0 else { return false }
        return Double(preClipNorm) >= nearMissReferenceMultiple * reference
    }

    /// Gathers a dump's contents from `trainer` and writes it. Nothing here
    /// throws: a part that can't be read is noted in the manifest. The
    /// weights are read only for a stopping reason (a near miss leaves them,
    /// since training goes on and they are in the next checkpoint anyway).
    static func dump(
        reason: CrashDumpReason,
        detail: String,
        pathKind: String,
        trainer: ChessTrainer,
        batchSize: Int,
        batchTrainerStep: Int,
        runProvenance: String?,
        faults: [GPUFaultLedger.Fault]
    ) async -> URL? {
        var notes: [String] = []
        let completed = trainer.completedTrainSteps
        let batch: CapturedTrainingBatch?
        do {
            batch = try await trainer.captureLastBatch()
        } catch {
            batch = nil
            notes.append("batch not captured: \(error.localizedDescription)")
        }
        let gradientNorms: GradientNormHistory?
        do {
            gradientNorms = try await trainer.exportGradNormHistory()
        } catch {
            gradientNorms = nil
            notes.append("gradient-norm history not read: \(error.localizedDescription)")
        }
        let context = CrashDumpContext(
            reason: reason,
            detail: detail,
            pathKind: pathKind,
            modelID: trainer.identifier?.description ?? "no-model-id",
            trainerStep: completed,
            batchTrainerStep: batchTrainerStep,
            learningRate: Double(trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: completed)),
            momentum: Double(trainer.effectiveMomentum(completedSteps: completed)),
            runProvenance: runProvenance,
            batch: batch,
            batchHashes: await trainer.batchHashes.recentEntries(),
            gradientNorms: gradientNorms,
            faults: faults,
            notes: notes)
        let weights: (@Sendable () async throws -> (weights: [[Float]], velocity: [[Float]]))?
        if reason == .nearMiss {
            weights = nil
        } else {
            weights = {
                (weights: try await trainer.exportTrainerWeights(), velocity: try await trainer.exportVelocitySnapshot())
            }
        }
        return await write(context, weights: weights)
    }
}
