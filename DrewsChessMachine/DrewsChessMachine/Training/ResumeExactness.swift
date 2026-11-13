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
    /// The run's streams (master seed, sampler, per-game serials) are not
    /// continued — not in the checkpoint, or (GUI) in it but not usable
    /// (`--seed` names another seed, or another stream derivation) — so the
    /// resumed run draws from a different seed.
    case rngSampler = "rng_sampler"
    /// The training graph's dropout Philox state is not the saved one.
    case dropoutState = "dropout_state"
    /// Corpus replay: the feed phase restarts instead of continuing.
    case feedCarry = "feed_carry"
    /// GUI / train-vs-UCI: the replay buffer was not saved with the
    /// checkpoint, or (GUI) its restore failed, so training refills from
    /// new games.
    case buffer
    /// GUI / train-vs-UCI: the per-game stream serials (and, in the GUI, the
    /// arena count) were not recorded or not continued, so later games
    /// reuse streams the saved run already drew from.
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
    /// A different Mac than the one that wrote the checkpoint: hardware
    /// model, chip, virtual-machine flag or GPU name. (The GPU name does not
    /// carry the GPU core count, so two configurations of one chip compare
    /// equal.)
    case device
    // (`policy_tail`, a resume-time policy-tail precision other than the
    // checkpoint's, was removed with format v12: a resume builds the trainer
    // from the file's own architecture, which carries the tail, so the two
    // cannot differ. `--accept-inexact policy_tail` is refused as an unknown
    // token; recorded `not_exact_items` strings naming it still decode. A
    // file whose architecture a derive changed after its trainer state was
    // captured is a different thing, `policyTailPrecision` below.)
    /// The checkpoint carries no relative gradient-cap history
    /// (`trainer_grad_norm_history`; every trainer file written before the
    /// relative cap) and the resumed run's cap mode is `clip`, so its first
    /// W steps feed the hard max where the uninterrupted run fed the relative
    /// cap. In `off` and `log_only` the history does not change what is fed,
    /// so its absence is logged, not a gap.
    case gradNormHistory = "grad_norm_history"
    /// CLI `--resume-exact`: the configured LR schedule (warmup length,
    /// cycle, keyframes) feeds different values from the checkpoint's, and
    /// the configured one is used (owner decision O-4). Compared in behavior
    /// form (owner decision Q-5): an inert difference is not this gap.
    case schedule
    /// The checkpoint's `compute_data_type` is not the one its trainer state
    /// was captured under: a `--derive-model --set-compute-dtype` of a
    /// trainer-state file (`TrainerStateCapture`). The state is restored
    /// exactly; every later step computes in another precision, so the
    /// trajectory is not the saved run's. Raised whenever the two differ, on
    /// the same build, OS and Mac too (owner decision 2026-11-12).
    case computeDataType = "compute_data_type"
    /// The checkpoint's `policy_tail_precision` is not the one its trainer
    /// state was captured under (a derive, as for `computeDataType`).
    case policyTailPrecision = "policy_tail_precision"

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

    /// `gradNormHistory` when a resume restores no gradient-norm history and
    /// the resumed run clips with the relative cap (the only mode in which
    /// the history decides what the graph is fed).
    static func gradNormHistoryGaps(restoring history: GradNormHistoryResumeState,
                                    runningMode mode: RelativeGradientCapMode) -> [ResumeGap] {
        switch (history, mode) {
        case (.notInCheckpoint, .clip): return [.gradNormHistory]
        case (.restored, _), (.notInCheckpoint, .off), (.notInCheckpoint, .logOnly): return []
        }
    }

    /// `dropoutState` when a resume restores no dropout Philox state.
    static func dropoutGaps(restoring dropoutRNG: DropoutRNGResumeState) -> [ResumeGap] {
        switch dropoutRNG {
        case .philox: return []
        case .notInCheckpoint: return [.dropoutState]
        }
    }

    /// How the resuming process compares with the one that wrote `record`
    /// (plan C1 #33): `gaps` holds `build` / `os` / `device` when the build
    /// (git hash, dirty flag, build number), the OS version or the Mac
    /// (hardware model, chip, virtual-machine flag, GPU name) differs **and**
    /// the behavior fingerprints do not match — a different fingerprint, a
    /// saved file without one, or one of another recipe
    /// (`BehaviorFingerprint`). A change whose fingerprint matches computes
    /// what the saved run computed and is not a gap. `logLines` reports every
    /// change either way.
    ///
    /// A device field the system did not report is recorded as null, and is
    /// compared as a value: null against null is no change, null against a
    /// value is one.
    static func environmentGaps(writtenBy record: LineageRecord,
                                runningBuild: LineageRecord.Build,
                                runningDevice: LineageRecord.Device,
                                runningFingerprint: BehaviorFingerprint.Record) -> EnvironmentComparison {
        // The diff hash identifies uncommitted code (gap 6): two dirty
        // builds of one commit are different code unless their hashes
        // agree. An unrecorded hash (a schema-2 build) cannot be shown equal,
        // so it counts as a change; the fingerprint escape below still
        // applies.
        let diffChanged: Bool
        switch (record.build.gitDiffSHA256, runningBuild.gitDiffSHA256) {
        case (.recorded(let saved), .recorded(let running)): diffChanged = saved != running
        default: diffChanged = true
        }
        let buildChanged = record.build.gitHash != runningBuild.gitHash
            || record.build.gitDirty != runningBuild.gitDirty
            || record.build.buildNumber != runningBuild.buildNumber
            || diffChanged
        let osChanged = record.device.osVersion != runningDevice.osVersion
        let deviceChanged = record.device.hardwareModel != runningDevice.hardwareModel
            || record.device.cpu != runningDevice.cpu
            || record.device.isVirtualMachine != runningDevice.isVirtualMachine
            || record.device.gpu != runningDevice.gpu
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
        if deviceChanged {
            logLines.append("[RESUME] device changed (\(deviceText(record.device)) → \(deviceText(runningDevice))), "
                + fingerprintText)
            if !fingerprintMatches { gaps.append(.device) }
        }
        return EnvironmentComparison(gaps: gaps, logLines: logLines)
    }

    /// The device fields `environmentGaps` compares, as its log line shows
    /// them; a field the system did not report shows as `unreported`.
    private static func deviceText(_ device: LineageRecord.Device) -> String {
        func field(_ value: String?) -> String { value ?? "unreported" }
        return "hw_model=\(field(device.hardwareModel)) chip=\(field(device.cpu))"
            + " is_vm=\(field(device.isVirtualMachine.map(String.init))) gpu_name=\(field(device.gpu))"
    }
}

extension ResumeGap {
    /// The build / OS / device comparison and the precision comparison of a
    /// CLI `--resume-exact` of a trainer-state file of `fileArchitecture`
    /// (the architecture the resumed trainer is built from) whose record is
    /// `record`. The behavior fingerprint is computed for the architecture
    /// the trainer state was captured under (`TrainerStateCapture`), which
    /// is the one the record's fingerprint describes. Throws
    /// `TrainerStateCapture.CaptureError` for an inconsistent file.
    static func environmentAndPrecisionGaps(writtenBy record: LineageRecord,
                                            fileArchitecture: NetworkArchitecture,
                                            runningBuild: LineageRecord.Build,
                                            runningDevice: LineageRecord.Device) async throws -> EnvironmentComparison {
        let capture = try TrainerStateCapture(fileArchitecture: fileArchitecture, record: record)
        let environment = environmentGaps(
            writtenBy: record, runningBuild: runningBuild, runningDevice: runningDevice,
            runningFingerprint: try await BehaviorFingerprint.compute(for: .init(arch: capture.architecture)))
        let precision = precisionGaps(capture: capture, runningArchitecture: fileArchitecture)
        return EnvironmentComparison(gaps: environment.gaps + precision.gaps,
                                     logLines: environment.logLines + precision.logLines)
    }

    /// `computeDataType` / `policyTailPrecision` when the architecture a
    /// resume trains under (`running`, the checkpoint's own) computes in
    /// another precision than the one its trainer state was captured under,
    /// with one `[RESUME]` line per changed field naming both values and the
    /// derive(s) that changed it. No gap and no line when they agree — a
    /// training save, or derives that changed the precision and back.
    static func precisionGaps(capture: TrainerStateCapture,
                              runningArchitecture running: NetworkArchitecture) -> EnvironmentComparison {
        let derives = capture.derivations.map(\.modelID).joined(separator: ", ")
        var gaps: [ResumeGap] = []
        var logLines: [String] = []
        if capture.architecture.computeDataType != running.computeDataType {
            gaps.append(.computeDataType)
            logLines.append("[RESUME] \(ResumeGap.computeDataType.token): the trainer state was captured under "
                + "\(capture.architecture.computeDataType.rawValue); this file computes in "
                + "\(running.computeDataType.rawValue) (changed by derive \(derives))")
        }
        if capture.architecture.policyTailPrecision != running.policyTailPrecision {
            gaps.append(.policyTailPrecision)
            logLines.append("[RESUME] \(ResumeGap.policyTailPrecision.token): the trainer state was captured under "
                + "\(capture.architecture.policyTailPrecision.rawValue); this file has "
                + "\(running.policyTailPrecision.rawValue) (changed by derive \(derives))")
        }
        return EnvironmentComparison(gaps: gaps, logLines: logLines)
    }
}

/// The architecture a trainer-state file's trainer state was captured
/// under, and the derives that changed the file's architecture since — the
/// one rule a `--resume-exact` uses to tell a training save from a
/// precision-only derive of one (`--derive-model --set-compute-dtype` /
/// `--set-policy-tail-precision` on a trainer-state file).
///
/// Such a derive carries the source's lineage record unchanged (the run,
/// segment, clock, build, device and behavior fingerprint that captured the
/// state) and appends a derivation entry whose `trainer_state` names that
/// segment and clock and records the source's architecture. A resumed
/// segment gets a new segment ID, so the entries naming the record's own
/// segment are exactly the derives since the capture, oldest first, and the
/// first one's source architecture is the architecture of the last save
/// where training happened — however many derives were chained after it.
/// With no such entry, the file is a training save and its own architecture
/// is the captured one.
///
/// The captured architecture is what the behavior fingerprint is computed
/// for (the record's fingerprint was computed for it), so `build` / `os` /
/// `device` keep meaning "this build computes what the saved build computed
/// for that state"; the precision change itself is `precisionGaps`.
struct TrainerStateCapture: Equatable, Sendable {
    /// The architecture the trainer state was captured under.
    let architecture: NetworkArchitecture
    /// The trainer-state derives since the capture, oldest first; empty for
    /// a training save.
    let derivations: [ModelDerivation.DerivationRecord]

    enum CaptureError: Error, Equatable, LocalizedError {
        case clockMismatch(derive: String, recorded: Int, recordClock: Int?)
        case unreadableArchitecture(derive: String, detail: String)
        case changedBeyondPrecision(derive: String)

        var errorDescription: String? {
            switch self {
            case .clockMismatch(let derive, let recorded, let recordClock):
                return "derivation \(derive) carried the trainer state of this file's lineage segment at trainer step "
                    + "\(recorded), but the record's clock is \(recordClock.map(String.init) ?? "unrecorded"); "
                    + "the file is inconsistent and cannot be resumed exactly"
            case .unreadableArchitecture(let derive, let detail):
                return "derivation \(derive): the architecture its trainer state was captured under is unreadable: \(detail)"
            case .changedBeyondPrecision(let derive):
                return "derivation \(derive): the architecture the trainer state was captured under differs from this "
                    + "file's in more than compute_data_type and policy_tail_precision; the trainer state does not "
                    + "belong to this architecture"
            }
        }
    }

    /// The capture of the trainer state in a file of `fileArchitecture`
    /// whose lineage record is `record`. Throws — never reads as "no gap" —
    /// when a matching entry disagrees with the record's clock, its
    /// architecture is unreadable, or it differs from the file's in anything
    /// but the precision fields (no derive that carries trainer state can
    /// change anything else).
    init(fileArchitecture: NetworkArchitecture, record: LineageRecord) throws {
        let since: [(derivation: ModelDerivation.DerivationRecord, state: ModelDerivation.CarriedTrainerState)] =
            record.derivationHistory.compactMap { derivation in
                guard let state = derivation.trainerState, state.segmentID == record.run.segmentID else { return nil }
                return (derivation, state)
            }
        guard let first = since.first else {
            self.architecture = fileArchitecture
            self.derivations = []
            return
        }
        for (derivation, state) in since where state.trainerCompletedSteps != record.steps.cumTrainerStep {
            throw CaptureError.clockMismatch(derive: derivation.modelID, recorded: state.trainerCompletedSteps,
                                             recordClock: record.steps.cumTrainerStep)
        }
        let captured: NetworkArchitecture
        do {
            captured = try first.state.decodedSourceArchitecture()
        } catch {
            throw CaptureError.unreadableArchitecture(derive: first.derivation.modelID, detail: String(describing: error))
        }
        let capturedAtFilePrecision: NetworkArchitecture
        do {
            capturedAtFilePrecision = try captured.withComputeDataType(
                fileArchitecture.computeDataType, tail: fileArchitecture.policyTailPrecision)
        } catch {
            throw CaptureError.changedBeyondPrecision(derive: first.derivation.modelID)
        }
        guard capturedAtFilePrecision == fileArchitecture else {
            throw CaptureError.changedBeyondPrecision(derive: first.derivation.modelID)
        }
        self.architecture = captured
        self.derivations = since.map(\.derivation)
    }
}

/// The build/OS/device comparison of a resume (`ResumeGap.environmentGaps`):
/// the gaps it adds and the lines it logs.
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
