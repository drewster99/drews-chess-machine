import Foundation

/// Lists the models folder for the follow-lineage source. Tests script a
/// folder; the app scans `Models/`.
protocol LichessBotModelFolderScanning: Sendable {
    func scan(previous: ModelFolderHeaderCache) async throws -> ModelFolderScan
}

/// Scans one folder with `ModelFolderHeaderCache` on its own serial utility
/// queue, off the cooperative thread pool: a scan `stat`s every file and,
/// the first time, reads thousands of headers (follow-lineage plan §1.5).
struct LichessBotModelsFolderScanner: LichessBotModelFolderScanning {
    let directory: URL
    private let queue: DispatchQueue

    init(directory: URL) {
        self.directory = directory
        self.queue = DispatchQueue(label: "drewschess.lichessbot.models-folder", qos: .utility)
    }

    func scan(previous: ModelFolderHeaderCache) async throws -> ModelFolderScan {
        let directory = self.directory
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result { try ModelFolderHeaderCache.scanSynchronously(directory: directory, previous: previous) })
            }
        }
    }
}

/// One file of a followed lineage, as the status line and the log name it.
struct LichessBotLineageFile: Sendable, Equatable {
    let url: URL
    let modelID: String
    let segmentID: String
    let segmentIndex: Int
    let segmentLocalStep: Int
    let cumTrainerStep: Int?
    let contentSHA256: String

    init(_ candidate: ModelLineageTip.Candidate) {
        url = candidate.entry.url
        modelID = candidate.entry.modelID
        segmentID = candidate.position.segmentID
        segmentIndex = candidate.position.segmentIndex
        segmentLocalStep = candidate.position.segmentLocalStep
        cumTrainerStep = candidate.position.cumTrainerStep
        contentSHA256 = candidate.contentSHA256
    }

    /// "<file> (<model_id>) seg k segment step s cum n|—": `s` is the
    /// writing segment's own step (the lineage record's
    /// `segment_local_step`), `n` the run's cumulative trainer step.
    var description: String {
        "\(url.lastPathComponent) (\(modelID)) seg \(segmentIndex) segment step \(segmentLocalStep) cum \(cumTrainerStep.map(String.init) ?? "—")"
    }
}

/// What the follow-lineage source found at its last check (follow-lineage
/// plan §3.4): shown on the Overview, logged on change.
struct LichessBotLineageFollowStatus: Sendable, Equatable {
    enum Outcome: Sendable, Equatable {
        /// `newest` is the lineage's newest file.
        case following(newest: LichessBotLineageFile)
        /// The newest file on disk ranks below the generation playing (a
        /// newer file was deleted): keep playing it, never step back.
        case keepPlaying(newestOnDisk: LichessBotLineageFile)
        case noFiles
        /// Two continuations of the lineage (or the newest file is on
        /// another branch of the run than the generation playing); each
        /// entry names one. The operator re-anchors.
        case fork(continuations: [String])
        /// Different weights claim one segment, step and save time.
        case conflict(files: [LichessBotLineageFile])
        case folderUnreadable(String)
        /// The newest file did not load; it is not retried until it changes
        /// on disk, and nothing older is played instead.
        case newestFailedToLoad(file: LichessBotLineageFile, reason: String)

        /// Whether the source can't vouch for a newer file: the bot keeps
        /// playing its last good generation, and the status shows a
        /// problem.
        var isProblem: Bool {
            switch self {
            case .following, .keepPlaying:
                return false
            case .noFiles, .fork, .conflict, .folderUnreadable, .newestFailedToLoad:
                return true
            }
        }

        /// Whether a check that finds this throws, so the poll loop's
        /// backoff and alarm apply on every check while it lasts. A newest
        /// file that failed to load threw once, when the load failed; later
        /// checks report it without alarming again until the file changes.
        var throwsOnCheck: Bool {
            switch self {
            case .noFiles, .fork, .conflict, .folderUnreadable:
                return true
            case .following, .keepPlaying, .newestFailedToLoad:
                return false
            }
        }

        var description: String {
            switch self {
            case .following(let newest):
                return "newest \(newest.description)"
            case .keepPlaying(let newest):
                return "the newest file on disk, \(newest.description), ranks below the generation playing"
            case .noFiles:
                return "no file of the followed lineage is in the models folder"
            case .fork(let continuations):
                return "the lineage forked after the chosen segment: \(continuations.joined(separator: "; ")); choose one to follow"
            case .conflict(let files):
                return "different weights claim the same save: \(files.map(\.description).joined(separator: "; "))"
            case .folderUnreadable(let reason):
                return "the models folder can't be read: \(reason)"
            case .newestFailedToLoad(let file, let reason):
                return "the newest file \(file.description) could not be loaded: \(reason)"
            }
        }
    }

    let followed: LichessBotFollowedLineage
    let checkedAt: Date
    let outcome: Outcome
    /// Files of the lineage that were candidates.
    let candidateCount: Int
    /// Files of the run left out, by reason.
    let excluded: [ModelLineageTip.Exclusion: Int]
    let listed: Int
    let headersRead: Int
    let reused: Int
    let scanMilliseconds: Double

    /// "reason:n,…" in the exclusions' declared order, or "none".
    var excludedText: String {
        let parts = ModelLineageTip.Exclusion.allCases.compactMap { reason in
            excluded[reason].map { "\(reason.rawValue):\($0)" }
        }
        return parts.isEmpty ? "none" : parts.joined(separator: ",")
    }
}

enum LichessBotLineageFollowError: LocalizedError, Equatable {
    /// The settings name the follow-lineage source but no lineage.
    case noLineageChosen
    /// The check found no file it can vouch for.
    case unavailable(LichessBotLineageFollowStatus.Outcome)
    /// The selected file did not read or decode (or its content hash did
    /// not match its data).
    case fileFailedToLoad(file: URL, reason: String)
    /// The file read at the selected path is not the followed lineage's at
    /// or above the selected position: another lineage's file, or a
    /// sibling segment's, took the path between the check and the load.
    case followedFileChangedDuringLoad(file: URL, reason: String)

    var errorDescription: String? {
        switch self {
        case .noLineageChosen:
            return "No lineage is chosen to follow."
        case .unavailable(let outcome):
            return "Follow lineage: \(outcome.description)"
        case .fileFailedToLoad(let file, let reason):
            return "Follow lineage: \(file.lastPathComponent) could not be loaded: \(reason)"
        case .followedFileChangedDuringLoad(let file, let reason):
            return "Follow lineage: \(file.lastPathComponent) changed between the check and the load: \(reason)"
        }
    }

    /// Why the file itself failed, when the error is about the file: the
    /// file is then not retried until it changes on disk. Nil for the other
    /// cases. A failure after the file decoded and verified (building the
    /// network) is not a `LichessBotLineageFollowError` and says nothing
    /// about the file.
    var fileFailureReason: String? {
        switch self {
        case .fileFailedToLoad(_, let reason), .followedFileChangedDuringLoad(_, let reason):
            return reason
        case .noLineageChosen, .unavailable:
            return nil
        }
    }
}

/// The file a check selected for the slots to load.
struct LichessBotLineageFollowTarget: Sendable, Equatable {
    let followed: LichessBotFollowedLineage
    let candidate: ModelLineageTip.Candidate
    /// The copy to load: among byte-identical copies, the first in path
    /// order that is not known to fail.
    let url: URL
    /// That copy's fingerprint at the check, to remember it if it fails.
    let fingerprint: ModelFileFingerprint

    /// The candidate's rank, which the loaded file must equal or exceed.
    var rank: ModelLineageRank {
        candidate.rank
    }
}

/// The follow-lineage source's memory between checks (follow-lineage plan
/// §3.4): the folder's header cache, the last status, when the last check
/// ran, and the files that failed to load (by fingerprint, so a file is
/// retried only once it changes on disk). A value the slots own; its
/// methods are synchronous and do no I/O, so the slots actor and the
/// pre-online `prepare` share them.
struct LichessBotLineageFollower: Sendable {
    private(set) var cache: ModelFolderHeaderCache = .empty
    private(set) var status: LichessBotLineageFollowStatus?
    private(set) var lastCheckAt: Duration?
    /// Files that failed to load, with the reason.
    private var failedFiles: [ModelFileFingerprint: String] = [:]
    /// Redos (a child resumed below its stopped parent's newest file)
    /// already logged, by child segment.
    private var loggedRedos: Set<String> = []

    /// What one check decided.
    struct Decision: Sendable {
        let status: LichessBotLineageFollowStatus
        /// The file to load when the outcome is `.following`.
        let target: LichessBotLineageFollowTarget?
        /// Session-log lines to write: the status when it changed, a redo
        /// the first time it is seen.
        let logLines: [String]
    }

    /// Whether a check is due: never checked, `interval` elapsed, or the
    /// last check found a problem that throws (so each retry the poll loop's
    /// backoff allows is a real check, and the problem stays reported until
    /// a check finds it gone).
    func isDue(now: Duration, interval: Duration) -> Bool {
        guard let lastCheckAt else { return true }
        if status?.outcome.throwsOnCheck == true {
            return true
        }
        return now - lastCheckAt >= interval
    }

    /// Fold a check into the follower: the folder's scan, or why it could
    /// not be listed. `notBelow` is the playing generation's rank when it
    /// plays this lineage, read after the scan returned. `consequence` ends
    /// a problem's log line ("still playing generation 3 (…)", "the bot
    /// stays offline").
    mutating func record(
        _ scanResult: Result<ModelFolderScan, Error>,
        followed: LichessBotFollowedLineage,
        notBelow: ModelLineageRank?,
        checkedAt: Date,
        now: Duration,
        consequence: String
    ) -> Decision {
        lastCheckAt = now
        let scan: ModelFolderScan
        switch scanResult {
        case .success(let result):
            scan = result
        case .failure(let error):
            let newStatus = LichessBotLineageFollowStatus(
                followed: followed, checkedAt: checkedAt, outcome: .folderUnreadable(error.localizedDescription),
                candidateCount: 0, excluded: [:], listed: 0, headersRead: 0, reused: 0, scanMilliseconds: 0)
            let lines = statusLines(newStatus, consequence: consequence)
            status = newStatus
            return Decision(status: newStatus, target: nil, logLines: lines)
        }
        cache = scan.cache
        let anchor = ModelLineageAnchor(lineageRunID: followed.lineageRunID, anchorSegmentID: followed.anchorSegmentID)
        let selection = ModelLineageTip.select(entries: scan.entries, followed: anchor, notBelow: notBelow)
        var lines: [String] = []
        let outcome: LichessBotLineageFollowStatus.Outcome
        var target: LichessBotLineageFollowTarget?
        switch selection.outcome {
        case .following(let newest, let sameWeights, let redos):
            for redo in redos where !loggedRedos.contains(redo.childSegmentID) {
                loggedRedos.insert(redo.childSegmentID)
                lines.append("[LICHESS-BOT] lineage follow: segment \(redo.childSegmentID) resumed \(redo.parentSegmentID) at segment step \(redo.resumedAtLocalStep), below its newest segment step \(redo.parentNewestLocalStep); following \(redo.childSegmentID)")
            }
            // Among byte-identical copies, the first that is not known to
            // fail. None left: the newest weights failed; never step back.
            var usable: (url: URL, fingerprint: ModelFileFingerprint)?
            var failureReasons: [String] = []
            for url in sameWeights {
                guard let fingerprint = scan.cache.fingerprint(of: url) else {
                    // Every listed entry has a fingerprint in the scan that
                    // listed it; without one there is nothing to remember a
                    // failure by, so the copy is not used.
                    failureReasons.append("\(url.lastPathComponent) has no fingerprint in the scan")
                    continue
                }
                if let reason = failedFiles[fingerprint] {
                    failureReasons.append(reason)
                } else {
                    usable = (url, fingerprint)
                    break
                }
            }
            let file = LichessBotLineageFile(newest)
            if let usable {
                outcome = .following(newest: file)
                target = LichessBotLineageFollowTarget(followed: followed, candidate: newest, url: usable.url, fingerprint: usable.fingerprint)
            } else {
                outcome = .newestFailedToLoad(file: file, reason: failureReasons.joined(separator: "; "))
            }
        case .keepPlaying(let newestOnDisk):
            outcome = .keepPlaying(newestOnDisk: LichessBotLineageFile(newestOnDisk))
        case .fork(let tips):
            outcome = .fork(continuations: tips.map { "segment \($0.segmentID) (\($0.modelID)) at segment step \($0.newestLocalStep), \($0.file.lastPathComponent)" })
        case .divergedFromPlaying(let newest):
            outcome = .fork(continuations: ["\(LichessBotLineageFile(newest).description) is on another branch of the run than the generation playing"])
        case .conflict(let files):
            outcome = .conflict(files: files.map(LichessBotLineageFile.init))
        case .noFiles:
            outcome = .noFiles
        }
        let newStatus = LichessBotLineageFollowStatus(
            followed: followed, checkedAt: checkedAt, outcome: outcome,
            candidateCount: selection.candidateCount, excluded: selection.excluded,
            listed: scan.listed, headersRead: scan.headersRead, reused: scan.reused,
            scanMilliseconds: scan.elapsedMilliseconds)
        lines += statusLines(newStatus, consequence: consequence)
        status = newStatus
        return Decision(status: newStatus, target: target, logLines: lines)
    }

    /// The target's file failed to load (or verify): remember it until it
    /// changes on disk, and report the newest as failed. Returns the log
    /// line.
    mutating func recordLoadFailure(_ target: LichessBotLineageFollowTarget, reason: String, consequence: String) -> String {
        failedFiles[target.fingerprint] = reason
        if let current = status {
            status = LichessBotLineageFollowStatus(
                followed: current.followed, checkedAt: current.checkedAt,
                outcome: .newestFailedToLoad(file: LichessBotLineageFile(target.candidate), reason: reason),
                candidateCount: current.candidateCount, excluded: current.excluded,
                listed: current.listed, headersRead: current.headersRead, reused: current.reused,
                scanMilliseconds: current.scanMilliseconds)
        }
        return "[LICHESS-BOT] lineage follow: \(target.url.lastPathComponent) could not be loaded: \(reason); not retried until it changes; \(consequence)"
    }

    /// The log lines for `newStatus`, only when it differs from the last
    /// one in lineage, outcome or newest file (not every check).
    private func statusLines(_ newStatus: LichessBotLineageFollowStatus, consequence: String) -> [String] {
        if let status, status.outcome == newStatus.outcome, status.followed == newStatus.followed {
            return []
        }
        let wasProblem = status?.outcome.isProblem ?? false
        switch newStatus.outcome {
        case .following(let newest):
            var lines: [String] = []
            if wasProblem {
                lines.append("[LICHESS-BOT] lineage follow available again: \(newStatus.outcome.description)")
            }
            lines.append("[LICHESS-BOT] lineage follow: run=\(newStatus.followed.lineageRunID) anchor=\(newStatus.followed.anchorSegmentID) newest=\(newest.url.lastPathComponent) model=\(newest.modelID) seg=\(newest.segmentIndex) segment_step=\(newest.segmentLocalStep) cum=\(newest.cumTrainerStep.map(String.init) ?? "null") sha=\(newest.contentSHA256.prefix(12)) files=\(newStatus.candidateCount) excluded=\(newStatus.excludedText) scan ms=\(String(format: "%.1f", newStatus.scanMilliseconds)) headers read=\(newStatus.headersRead) reused=\(newStatus.reused)")
            return lines
        case .keepPlaying:
            return ["[LICHESS-BOT] lineage follow: \(newStatus.outcome.description); \(consequence)"]
        case .noFiles, .fork, .conflict, .folderUnreadable, .newestFailedToLoad:
            return ["[LICHESS-BOT] lineage follow problem: \(newStatus.outcome.description); \(consequence)"]
        }
    }
}
