//
//  ResumeExactness.swift
//  DrewsChessMachine
//
//  What a resume could not restore, decided in one place for every path
//  (determinism plan C3). A resume is EXACT when it continues the saved run's
//  complete training state — weights, optimizer and schedule clock, the run's
//  random streams where they left off, the replay buffer in age order and,
//  for corpus replay, the feed phase — and NOT EXACT otherwise, with each
//  missing piece named by a `ResumeGap`. Every path logs exactly one
//  `[RESUME] EXACT` or `[RESUME] NOT EXACT: …` line and records the gaps in
//  the segment's lineage record (`run.not_exact_items`).
//
//  Per path (plan C3, decisions D-1 and D-7): corpus replay and train-vs-UCI
//  refuse a `--resume-exact` whose gaps are not all named by
//  `--accept-inexact`; a GUI resume is state-exact at most and never refuses.
//  A corpus whose content changed under a resume is not a gap: it is refused
//  whatever `--accept-inexact` names.
//

import Foundation

/// One piece of training state a resume did not restore.
enum ResumeGap: String, CaseIterable, Sendable {
    /// The run's streams (master seed, sampler, per-game serials) are not in
    /// the checkpoint, so the resumed run draws from a different seed.
    case rngSampler = "rng_sampler"
    /// The training graph's dropout Philox state is not the saved one.
    case dropoutState = "dropout_state"
    /// Corpus replay: the feed phase restarts instead of continuing.
    case feedCarry = "feed_carry"
    /// GUI / train-vs-UCI: the replay buffer was not saved with the
    /// checkpoint, so training refills from new games.
    case buffer
    /// GUI / train-vs-UCI: the per-game stream serials (and, in the GUI, the
    /// arena count) were not recorded, so later games reuse streams the
    /// saved run already drew from.
    case serials
    /// GUI: the arena-trigger and periodic-save clocks restart.
    case clocks
    /// The checkpoint carries no full training-parameter snapshot, or the
    /// resume trains under parameters that change what the saved state means
    /// (corpus replay: a different feed per step).
    case params
    /// The checkpoint carries no lineage record (written before lineage).
    case lineage
    /// A different build than the one that wrote the checkpoint.
    case build
    /// A different OS version than the one that wrote the checkpoint.
    case os
    /// The policy-head tail precision differs from the checkpoint's, or the
    /// checkpoint predates recording it.
    case policyTail = "policy_tail"

    /// The token logged and recorded for this gap.
    var token: String { rawValue }

    /// Parse an `--accept-inexact` list: comma-separated tokens. An unknown
    /// token is an error naming the valid ones.
    static func parseAcceptList(_ text: String) throws -> Set<ResumeGap> {
        var accepted = Set<ResumeGap>()
        for part in text.split(separator: ",", omittingEmptySubsequences: false) {
            let token = part.trimmingCharacters(in: .whitespaces)
            guard let gap = ResumeGap(rawValue: token) else {
                throw ResumeExactnessError.unknownAcceptToken(token)
            }
            accepted.insert(gap)
        }
        return accepted
    }

    /// `dropoutState` when a resume restores no dropout Philox state.
    static func dropoutGaps(restoring dropoutRNG: DropoutRNGResumeState) -> [ResumeGap] {
        switch dropoutRNG {
        case .philox: return []
        case .notInCheckpoint: return [.dropoutState]
        }
    }

    /// How the resuming process compares with the one that wrote `record`
    /// (plan C1 #33): `gaps` holds `build` / `os` when the build (git hash,
    /// dirty flag, build number) or OS version differs **and** the behavior
    /// fingerprints do not match — a different fingerprint, a saved file
    /// without one, or one of another recipe (`BehaviorFingerprint`). A
    /// change whose fingerprint matches computes what the saved run computed
    /// and is not a gap. `logLines` reports every change either way.
    static func environmentGaps(writtenBy record: LineageRecord,
                                runningBuild: LineageRecord.Build,
                                runningDevice: LineageRecord.Device,
                                runningFingerprint: BehaviorFingerprint.Record) -> EnvironmentComparison {
        let buildChanged = record.build.gitHash != runningBuild.gitHash
            || record.build.gitDirty != runningBuild.gitDirty
            || record.build.buildNumber != runningBuild.buildNumber
        let osChanged = record.device.osVersion != runningDevice.osVersion
        let saved = record.rng.behaviorFingerprint
        let fingerprintMatches = saved?.matches(runningFingerprint) == true
        let fingerprintText: String
        if fingerprintMatches {
            fingerprintText = "behavior fingerprint matches"
        } else if let saved {
            fingerprintText = saved.recipe == runningFingerprint.recipe
                ? "behavior fingerprint differs"
                : "behavior fingerprint recipe \(saved.recipe) cannot be compared with recipe \(runningFingerprint.recipe)"
        } else {
            fingerprintText = "the checkpoint records no behavior fingerprint"
        }
        var gaps: [ResumeGap] = []
        var logLines: [String] = []
        if buildChanged {
            logLines.append("[RESUME] build changed (\(record.build.buildNumber) \(record.build.gitHash)"
                + "\(record.build.gitDirty ? "+dirty" : "") → \(runningBuild.buildNumber) \(runningBuild.gitHash)"
                + "\(runningBuild.gitDirty ? "+dirty" : "")), \(fingerprintText)")
            if !fingerprintMatches { gaps.append(.build) }
        }
        if osChanged {
            logLines.append("[RESUME] OS changed (\(record.device.osVersion) → \(runningDevice.osVersion)), \(fingerprintText)")
            if !fingerprintMatches { gaps.append(.os) }
        }
        return EnvironmentComparison(gaps: gaps, logLines: logLines)
    }
}

/// The build/OS comparison of a resume (`ResumeGap.environmentGaps`): the
/// gaps it adds and the lines it logs.
struct EnvironmentComparison: Equatable, Sendable {
    let gaps: [ResumeGap]
    let logLines: [String]
}

/// The outcome of a resume, as every path reports it.
struct ResumeExactness: Equatable, Sendable {
    /// Missing pieces, without duplicates, in `ResumeGap.allCases` order.
    let gaps: [ResumeGap]

    init(gaps: [ResumeGap]) {
        let present = Set(gaps)
        self.gaps = ResumeGap.allCases.filter(present.contains)
    }

    /// The decision for a resume of `parent` missing `gaps`: a parent written
    /// before lineage also lacks `lineage`. Every path decides through this —
    /// the `[RESUME]` line, the `--resume-exact` refusal and the segment's
    /// recorded `not_exact_items` (which the `[RUN]` line reports) all come
    /// from the one value.
    static func resume(of parent: LineageTracker.ParentFile, gaps: [ResumeGap]) -> ResumeExactness {
        switch parent.lineage {
        case .recorded: return ResumeExactness(gaps: gaps)
        case .unrecorded: return ResumeExactness(gaps: gaps + [.lineage])
        }
    }

    /// Gap tokens as both the `[RESUME]` and the `[RUN]` line list them.
    static func tokenList(_ tokens: [String]) -> String {
        tokens.joined(separator: ", ")
    }

    var isExact: Bool { gaps.isEmpty }

    /// The tokens recorded in the lineage record's `not_exact_items`.
    var tokens: [String] { gaps.map(\.token) }

    /// The one `[RESUME]` line every path logs.
    var logLine: String {
        isExact ? "[RESUME] EXACT" : "[RESUME] NOT EXACT: " + Self.tokenList(tokens)
    }

    /// The GUI status bar's note for a running segment that began with this
    /// resume: nil for an exact one, `resumed not exact: <gaps>` otherwise.
    var statusBarNote: String? {
        isExact ? nil : "resumed not exact: " + Self.tokenList(tokens)
    }

    /// The refusal a `--resume-exact` gets when `accepted` does not name every
    /// gap (decision D-7), or nil when the resume may proceed.
    func refusal(accepting accepted: Set<ResumeGap>) -> String? {
        let unaccepted = gaps.filter { !accepted.contains($0) }
        guard !unaccepted.isEmpty else { return nil }
        let list = unaccepted.map(\.token).joined(separator: ",")
        return "--resume-exact cannot continue this checkpoint exactly: it lacks \(list). "
            + "Pass --accept-inexact \(list) to resume anyway (each missing piece then starts fresh and the "
            + "segment is recorded as not exact), or continue it as a new branch without --resume-exact."
    }
}

enum ResumeExactnessError: Error, Equatable, LocalizedError {
    case unknownAcceptToken(String)

    var errorDescription: String? {
        switch self {
        case .unknownAcceptToken(let token):
            let valid = ResumeGap.allCases.map(\.token).joined(separator: ", ")
            return "--accept-inexact: unknown item \"\(token)\" (valid: \(valid))"
        }
    }
}
