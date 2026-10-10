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
    /// Nil when they couldn't be read (the reason is in `notes`); empty when
    /// no batch was hashed yet.
    let batchHashes: [BatchHashChain.Entry]?
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
///    minutes) and copies of macOS's `gpuEvent` reports naming this app —
///    are staged in `<name>.dcmcrash.tmp/` and published by an atomic rename.
///    `dump` reads the trainer state among them (the staged batch, the
///    gradient-norm history, the recent batch hashes) beforehand, on the
///    trainer's and the hash chain's serial queues, under a time limit of
///    its own: those reads wait behind whatever their queue is running, and
///    on the trainer's queue after a GPU hang that can be a GPU wait that
///    never returns;
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
    /// How long each trainer-state read (the staged batch, the gradient-norm
    /// history, the recent batch hashes) may take before the dump goes on
    /// without it. The three run at once, so together they take at most
    /// this long. On a healthy trainer queue they wait behind at most the
    /// step in flight, and healthy batch-4,096 corpus-replay steps logged
    /// `ms=` up to 15 s (34 s once) on 2026-10-01 and 2026-10-08: a shorter
    /// limit could abandon the batch — the dump's main part — on a queue
    /// that was only busy.
    static let trainerStateReadTimeLimitSeconds: UInt64 = 60
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
        weightsTimeLimitSeconds: UInt64 = CrashDumpWriter.weightsReadTimeLimitSeconds,
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
        await writeWeights(into: folder, limitSeconds: weightsTimeLimitSeconds, read: weights)
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
        copyGPUEventReports(into: stagingURL, since: now.addingTimeInterval(-systemLogLookBackSeconds),
                            from: gpuEventReportFolders, processName: ProcessInfo.processInfo.processName)
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
            recentBatchHashes: context.batchHashes.map { entries in
                entries.map {
                    CrashDumpManifest.BatchHash(trainerStep: $0.trainerStep, batchHash: $0.batchHash, chain: $0.chain)
                }
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

    /// The DiagnosticReports folders macOS writes `gpuEvent-*` reports into;
    /// `copyGPUEventReports` also searches each one's `Retired/`.
    static let gpuEventReportFolders = [
        URL(fileURLWithPath: "/Library/Logs/DiagnosticReports", isDirectory: true),
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DiagnosticReports", isDirectory: true),
    ]

    /// macOS records a process name cut to this many characters (the
    /// kernel's `MAXCOMLEN`): `DrewsChessMachin`.
    static let reportedProcessNameLength = 16

    /// Copies macOS's GPU event reports naming this app written since
    /// `since`, from each of `reportFolders` and its `Retired/` (macOS moves
    /// a report there once it has been handled — all three of 2026-10-09's
    /// were there the next morning, none in the top folder).
    ///
    /// A report names only the process macOS blamed (`process_name`, cut to
    /// 16 characters) and no pid, so with several DrewsChessMachine
    /// processes running a report may be another one's — which is what a
    /// dump in a process hit as a bystander needs: the report of the process
    /// that caused the reset. `system-log.txt` (this process's entries) and
    /// the manifest's other-process list tell which it was. A report whose
    /// body can't be read as JSON, and any other failure, is noted in a
    /// file, not thrown; a folder that doesn't exist is not a failure.
    static func copyGPUEventReports(into folder: URL, since: Date, from reportFolders: [URL], processName: String) {
        let reportedName = String(processName.prefix(reportedProcessNameLength))
        var notes: [String] = []
        let searched = reportFolders.flatMap { [$0, $0.appendingPathComponent("Retired", isDirectory: true)] }
        for reports in searched {
            let entries: [URL]
            do {
                entries = try FileManager.default.contentsOfDirectory(
                    at: reports, includingPropertiesForKeys: [.contentModificationDateKey], options: [])
            } catch CocoaError.fileReadNoSuchFile {
                continue
            } catch {
                notes.append("could not list \(reports.path): \(error.localizedDescription)")
                continue
            }
            for entry in entries where entry.lastPathComponent.hasPrefix("gpuEvent-") {
                do {
                    let modified = try entry.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
                    guard let modified, modified >= since else { continue }
                    let data = try Data(contentsOf: entry)
                    guard let name = gpuEventReportProcessName(data) else {
                        notes.append("\(entry.path): no process_name in the report body")
                        continue
                    }
                    guard name == reportedName else { continue }
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

    /// `process_name` from a `.ips` report: a JSON header line, then a JSON
    /// body. nil when the body isn't a JSON object with a string
    /// `process_name`.
    static func gpuEventReportProcessName(_ report: Data) -> String? {
        guard let newline = report.firstIndex(of: UInt8(ascii: "\n")) else { return nil }
        let body = report[report.index(after: newline)...]
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: Data(body))
        } catch {
            return nil
        }
        guard let fields = object as? [String: Any] else { return nil }
        return fields["process_name"] as? String
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
        limitSeconds: UInt64,
        read: @escaping @Sendable () async throws -> (weights: [[Float]], velocity: [[Float]])
    ) async {
        switch await withTimeLimit(seconds: limitSeconds, read) {
        case .finished(let values):
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
        case .failed(let error):
            writeWeightsError("reading the weights failed: \(error.localizedDescription)", into: folder)
        case .timedOut(let seconds):
            writeWeightsError("reading the weights took longer than \(seconds) s; abandoned", into: folder)
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

    // MARK: - Time limits

    /// How a read under `withTimeLimit` ended.
    enum TimeLimitedOutcome<Value: Sendable>: Sendable {
        case finished(Value)
        case failed(any Error)
        /// Still running at the limit: abandoned.
        case timedOut(limitSeconds: UInt64)
    }

    /// Runs `work` with a time limit. Work still running at the limit is
    /// abandoned, not cancelled: neither a GPU wait nor a block queued behind
    /// one on a serial `DispatchQueue` can be cancelled. It goes on in its
    /// detached task and its result, if it ever comes, is dropped — so the
    /// caller never waits for it (a task group would: a group's scope waits
    /// for every child). `work` must therefore only read and return a copy,
    /// never write into the dump or change any state: it may finish after
    /// the dump, and the halt behind it, have gone on. A `--train` or CLI
    /// halt then ends the process; an interactive GUI run keeps it, and the
    /// abandoned read finishes whenever its queue frees up.
    static func withTimeLimit<Value: Sendable>(
        seconds limitSeconds: UInt64,
        _ work: @escaping @Sendable () async throws -> Value
    ) async -> TimeLimitedOutcome<Value> {
        let resumed = SyncBox(false)
        return await withCheckedContinuation { (continuation: CheckedContinuation<TimeLimitedOutcome<Value>, Never>) in
            // The first of the work and the timer resumes; the other is a
            // no-op.
            @Sendable func finish(_ outcome: TimeLimitedOutcome<Value>) {
                let first = resumed.mutate { done -> Bool in
                    if done { return false }
                    done = true
                    return true
                }
                if first { continuation.resume(returning: outcome) }
            }
            let timer = Task.detached(priority: .utility) {
                do {
                    try await Task.sleep(for: .seconds(Double(limitSeconds)))
                } catch {
                    // Cancelled: the work finished first.
                    return
                }
                finish(.timedOut(limitSeconds: limitSeconds))
            }
            Task.detached(priority: .userInitiated) {
                let outcome: TimeLimitedOutcome<Value>
                do {
                    outcome = .finished(try await work())
                } catch {
                    outcome = .failed(error)
                }
                finish(outcome)
                timer.cancel()
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
    /// Absent when they couldn't be read (`notes` says why).
    let recentBatchHashes: [BatchHash]?
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
    ///
    /// Bounded in time apart from the file and log I/O of `write`: the
    /// trainer-state reads take at most `trainerStateReadTimeLimitSeconds`
    /// together, the weights read at most `weightsReadTimeLimitSeconds`.
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
        let completed = trainer.completedTrainSteps
        let state = await readTrainerState(
            batch: { try await trainer.captureLastBatch() },
            gradientNorms: { try await trainer.exportGradNormHistory() },
            batchHashes: { await trainer.batchHashes.recentEntries() })
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
            batch: state.batch,
            batchHashes: state.batchHashes,
            gradientNorms: state.gradientNorms,
            faults: faults,
            notes: state.notes)
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

    /// The trainer state a dump records, and why any part is missing.
    struct TrainerState: Sendable {
        /// Nil when none was staged yet, or when it couldn't be read.
        let batch: CapturedTrainingBatch?
        let gradientNorms: GradientNormHistory?
        let batchHashes: [BatchHashChain.Entry]?
        /// One line per part that couldn't be read.
        let notes: [String]
    }

    /// Reads the staged batch, the gradient-norm history and the recent
    /// batch hashes at once, each under `limitSeconds` (`withTimeLimit`).
    /// The first two run on the trainer's serial queue, behind any step or
    /// read in flight; the hashes on the hash chain's queue, behind any
    /// hashes still pending. Running them at once bounds the whole read by
    /// one limit, not three; their order on the trainer's queue doesn't
    /// matter (both only copy).
    static func readTrainerState(
        batch readBatch: @escaping @Sendable () async throws -> CapturedTrainingBatch?,
        gradientNorms readGradientNorms: @escaping @Sendable () async throws -> GradientNormHistory,
        batchHashes readBatchHashes: @escaping @Sendable () async -> [BatchHashChain.Entry],
        limitSeconds: UInt64 = trainerStateReadTimeLimitSeconds
    ) async -> TrainerState {
        async let batchOutcome = withTimeLimit(seconds: limitSeconds, readBatch)
        async let gradientNormsOutcome = withTimeLimit(seconds: limitSeconds, readGradientNorms)
        async let batchHashesOutcome = withTimeLimit(seconds: limitSeconds, readBatchHashes)

        var notes: [String] = []
        let batch: CapturedTrainingBatch?
        switch await batchOutcome {
        case .finished(let captured):
            batch = captured
        case .failed(let error):
            batch = nil
            notes.append("batch not captured: \(error.localizedDescription)")
        case .timedOut(let seconds):
            batch = nil
            notes.append("batch not captured: the trainer's queue did not answer within \(seconds) s; abandoned")
        }
        let gradientNorms: GradientNormHistory?
        switch await gradientNormsOutcome {
        case .finished(let history):
            gradientNorms = history
        case .failed(let error):
            gradientNorms = nil
            notes.append("gradient-norm history not read: \(error.localizedDescription)")
        case .timedOut(let seconds):
            gradientNorms = nil
            notes.append("gradient-norm history not read: the trainer's queue did not answer within \(seconds) s; "
                + "abandoned")
        }
        let batchHashes: [BatchHashChain.Entry]?
        switch await batchHashesOutcome {
        case .finished(let entries):
            batchHashes = entries
        case .failed(let error):
            batchHashes = nil
            notes.append("recent batch hashes not read: \(error.localizedDescription)")
        case .timedOut(let seconds):
            batchHashes = nil
            notes.append("recent batch hashes not read: the batch-hash queue did not answer within \(seconds) s; "
                + "abandoned")
        }
        return TrainerState(batch: batch, gradientNorms: gradientNorms, batchHashes: batchHashes, notes: notes)
    }
}
