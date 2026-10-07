import Foundation

/// The run and segment a followed lineage starts from: files of this run
/// whose segment chain contains `anchorSegmentID` are the lineage.
struct ModelLineageAnchor: Sendable, Equatable {
    let lineageRunID: String
    let anchorSegmentID: String
}

/// Where a file (or the weights playing) sits in its run, in the order
/// "newest" uses: a deeper segment chain is later work; within one segment,
/// the higher local step; at one step, the later save (follow-lineage plan
/// §2.3). Two ranks are comparable only when one chain is a prefix of the
/// other — otherwise they are on different branches of the run (a fork).
struct ModelLineageRank: Sendable, Equatable {
    /// Earlier segments' IDs, oldest first, then the file's own.
    let segmentChain: [String]
    let segmentLocalStep: Int
    let recordedUnix: Int64

    /// Whether one chain is a prefix of the other.
    func isOnTheSameBranch(as other: ModelLineageRank) -> Bool {
        let shorter = min(segmentChain.count, other.segmentChain.count)
        return Array(segmentChain.prefix(shorter)) == Array(other.segmentChain.prefix(shorter))
    }

    /// Whether this rank is later work than `other`. Only meaningful on the
    /// same branch.
    func isAbove(_ other: ModelLineageRank) -> Bool {
        if segmentChain.count != other.segmentChain.count {
            return segmentChain.count > other.segmentChain.count
        }
        if segmentLocalStep != other.segmentLocalStep {
            return segmentLocalStep > other.segmentLocalStep
        }
        return recordedUnix > other.recordedUnix
    }
}

/// Chooses the newest file of one followed lineage from a model folder's
/// headers (follow-lineage plan §3.3). Pure: it reads only the lineage facts
/// the catalog took from each file's `dcm_lineage` record — never a
/// filename, a modification time or the flat mirror keys — and decides
/// nothing it cannot vouch for: two live continuations of the lineage are a
/// fork, two different files claiming one save are a conflict, and a newest
/// file below what is already playing is reported, never selected.
enum ModelLineageTip {

    /// A file of the followed lineage.
    struct Candidate: Sendable, Equatable {
        let entry: ModelFileEntry
        let position: ModelFileLineagePosition
        let contentSHA256: String

        var rank: ModelLineageRank {
            ModelLineageRank(segmentChain: position.segmentChain, segmentLocalStep: position.segmentLocalStep, recordedUnix: position.recordedUnix)
        }
    }

    /// Why a file is not a candidate, for the status line.
    enum Exclusion: String, Sendable, CaseIterable {
        /// A file of the run written by a path that is not followed (a GUI
        /// run, a derived or new model).
        case otherPathKind = "other path kind"
        /// A file of the run whose header has no `content_sha256`.
        case noContentHash = "no content hash"
        /// A file whose lineage record does not decode (its run is unknown).
        case unreadableLineage = "unreadable lineage"
        /// A file written before lineage records (its run is unknown).
        case unrecordedLineage = "written before lineage records"
    }

    /// The newest file of one segment of the lineage, for naming a fork.
    struct SegmentTip: Sendable, Equatable {
        let segmentID: String
        let modelID: String
        let newestLocalStep: Int
        let file: URL
    }

    /// A child segment that resumed its parent from a file below the
    /// parent's newest, after the parent had stopped (a redo): followed, and
    /// worth one log line.
    struct Redo: Sendable, Equatable {
        let childSegmentID: String
        let parentSegmentID: String
        let resumedAtLocalStep: Int
        let parentNewestLocalStep: Int
    }

    enum Outcome: Sendable, Equatable {
        /// `newest` is the lineage's newest file; `sameWeights` lists every
        /// file of the same rank with the same `content_sha256` (the rolling
        /// file and its step copy), `newest`'s URL first.
        case following(newest: Candidate, sameWeights: [URL], redos: [Redo])
        /// The newest file on disk ranks below what is playing (its newer
        /// file was deleted): keep playing.
        case keepPlaying(newestOnDisk: Candidate)
        /// Two continuations of the lineage: the operator re-anchors.
        case fork(tips: [SegmentTip])
        /// The newest file is on another branch of the run than the
        /// generation playing: a fork against what plays.
        case divergedFromPlaying(newest: Candidate)
        /// Different weights claim the same segment, step and save time.
        case conflict(files: [Candidate])
        /// No file of the lineage.
        case noFiles
    }

    struct Selection: Sendable, Equatable {
        let outcome: Outcome
        let candidateCount: Int
        let excluded: [Exclusion: Int]
    }

    /// Select the newest file of `followed` among `entries`. `notBelow` is
    /// the rank of the generation playing, when it is of this lineage: a
    /// best candidate below it is `.keepPlaying`, and one on another branch
    /// is a fork against it.
    static func select(entries: [ModelFileEntry], followed: ModelLineageAnchor, notBelow: ModelLineageRank?) -> Selection {
        var candidates: [Candidate] = []
        var excluded: [Exclusion: Int] = [:]
        for entry in entries {
            switch entry.lineage {
            case nil:
                // Only an entry built outside the catalog (a test) has no
                // lineage facts; the catalog sets them on every entry.
                continue
            case .unrecorded?:
                excluded[.unrecordedLineage, default: 0] += 1
            case .unreadable?:
                excluded[.unreadableLineage, default: 0] += 1
            case .recorded(let position)?:
                guard position.lineageRunID == followed.lineageRunID,
                      position.segmentChain.contains(followed.anchorSegmentID) else { continue }
                guard position.pathKind == .replay || position.pathKind == .vsuci else {
                    excluded[.otherPathKind, default: 0] += 1
                    continue
                }
                guard let sha = entry.contentSHA256, !sha.isEmpty else {
                    excluded[.noContentHash, default: 0] += 1
                    continue
                }
                candidates.append(Candidate(entry: entry, position: position, contentSHA256: sha))
            }
        }
        func selection(_ outcome: Outcome) -> Selection {
            Selection(outcome: outcome, candidateCount: candidates.count, excluded: excluded)
        }
        guard !candidates.isEmpty else { return selection(.noFiles) }

        // Each segment's files, newest first (every file of a segment
        // carries the same chain).
        var bySegment: [String: [Candidate]] = [:]
        for candidate in candidates {
            bySegment[candidate.position.segmentID, default: []].append(candidate)
        }
        for segmentID in bySegment.keys {
            bySegment[segmentID]?.sort { $0.rank.isAbove($1.rank) || ($0.rank == $1.rank && $0.entry.url.path < $1.entry.url.path) }
        }
        let segmentHeads = bySegment.values.compactMap(\.first)
        func tip(_ head: Candidate) -> SegmentTip {
            SegmentTip(segmentID: head.position.segmentID, modelID: head.entry.modelID, newestLocalStep: head.position.segmentLocalStep, file: head.entry.url)
        }

        // (a) Every segment must lie on one branch.
        for (index, first) in segmentHeads.enumerated() {
            for second in segmentHeads[(index + 1)...] where !first.rank.isOnTheSameBranch(as: second.rank) {
                let leaves = segmentHeads.filter { head in
                    !segmentHeads.contains { other in
                        other.position.segmentID != head.position.segmentID
                            && other.position.segmentChain.count > head.position.segmentChain.count
                            && other.rank.isOnTheSameBranch(as: head.rank)
                    }
                }
                return selection(.fork(tips: leaves.map(tip).sorted { $0.segmentID < $1.segmentID }))
            }
        }

        // (b) A parent that kept training after a child resumed it is a
        // second continuation; (c) one that had stopped is a redo.
        var redos: [Redo] = []
        var handoffs: [LineageHandoff] = []
        for head in segmentHeads {
            for handoff in head.position.handoffs where !handoffs.contains(handoff) {
                handoffs.append(handoff)
            }
        }
        for handoff in handoffs {
            guard let parentFiles = bySegment[handoff.fromSegmentID] else { continue }
            let beyond = parentFiles.filter { $0.position.segmentLocalStep > handoff.atLocalStep }
            guard let parentNewest = beyond.first else { continue }
            if beyond.contains(where: { $0.position.recordedUnix >= handoff.toStartedUnix }) {
                let child = segmentHeads.filter { $0.position.segmentChain.contains(handoff.toSegmentID) }
                    .max { $1.rank.isAbove($0.rank) }
                let tips = [tip(parentNewest)] + (child.map { [tip($0)] } ?? [])
                return selection(.fork(tips: tips))
            }
            redos.append(Redo(childSegmentID: handoff.toSegmentID, parentSegmentID: handoff.fromSegmentID,
                              resumedAtLocalStep: handoff.atLocalStep, parentNewestLocalStep: parentNewest.position.segmentLocalStep))
        }

        // Rank; the top rank's files with one content hash are one candidate.
        let sorted = candidates.sorted { $0.rank.isAbove($1.rank) || ($0.rank == $1.rank && $0.entry.url.path < $1.entry.url.path) }
        let best = sorted[0]
        let top = sorted.filter { $0.rank == best.rank }
        let distinctWeights = Set(top.map(\.contentSHA256))
        guard distinctWeights.count == 1 else {
            return selection(.conflict(files: top))
        }

        if let notBelow {
            guard best.rank.isOnTheSameBranch(as: notBelow) else {
                return selection(.divergedFromPlaying(newest: best))
            }
            if notBelow.isAbove(best.rank) {
                return selection(.keepPlaying(newestOnDisk: best))
            }
        }
        return selection(.following(newest: best, sameWeights: top.map(\.entry.url), redos: redos))
    }
}
