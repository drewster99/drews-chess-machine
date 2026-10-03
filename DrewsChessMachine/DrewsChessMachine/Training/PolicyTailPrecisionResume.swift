import Foundation

/// How a resume treats the policy-head tail precision a trainer file was
/// saved under (`trainer_policy_tail_precision`) against the precision this
/// process runs (`ChessNetwork.PolicyTailPrecision.process`).
///
/// The setting is not an architecture field — the same weights load under
/// either value — but on bf16 / fp16 models it changes the policy head's
/// arithmetic, so a lineage resumed under the other value is not the run it
/// continues. One pure decision shared by every resume path:
///
/// - **Corpus replay and train-vs-UCI `--resume-exact`** promise the
///   uninterrupted run, so a recorded mismatch refuses, naming the flag that
///   matches the file. A file written before the setting was recorded cannot
///   be checked; it is reported loudly and the resume proceeds (refusing would
///   strand every existing checkpoint).
/// - **GUI resume** never refuses (decision D-1 of the determinism plan: GUI
///   resumes are state-exact at most); a mismatch or an unrecorded value is
///   reported as a `NOT EXACT` item.
enum PolicyTailPrecisionResume {

    enum Finding: Equatable, Sendable {
        case matches
        case differs(saved: ChessNetwork.PolicyTailPrecision)
        case unrecorded
    }

    static func finding(
        saved: ChessNetwork.PolicyTailPrecision?,
        running: ChessNetwork.PolicyTailPrecision
    ) -> Finding {
        guard let saved else { return .unrecorded }
        return saved == running ? .matches : .differs(saved: saved)
    }

    /// The decision for a CLI `--resume-exact`: the line to log, and the
    /// refusal message when the resume must not proceed.
    static func exactResumeDecision(
        saved: ChessNetwork.PolicyTailPrecision?,
        running: ChessNetwork.PolicyTailPrecision
    ) -> (logLine: String, refusal: String?) {
        switch finding(saved: saved, running: running) {
        case .matches:
            return ("[RESUME-NUMERICS] policy_tail_precision=\(running.rawValue) matches the checkpoint", nil)
        case .differs(let saved):
            let refusal = "--resume-exact: the checkpoint was trained with policy tail precision \(saved.rawValue) "
                + "but this run uses \(running.rawValue); rerun with \(ChessNetwork.PolicyTailPrecision.flag) \(saved.rawValue) "
                + "to continue it exactly"
            return ("[RESUME-NUMERICS] ERROR \(refusal)", refusal)
        case .unrecorded:
            return (
                "[RESUME-NUMERICS] WARNING the checkpoint does not record its policy tail precision (written before "
                    + "the setting was saved); this run uses \(running.rawValue) — if the checkpoint trained under the "
                    + "other value this resume is not exact",
                nil
            )
        }
    }

    /// The resume gap the finding is: none when the precision matches,
    /// `policy_tail` when it differs or the checkpoint predates recording it
    /// (determinism plan C3 — a gap `--accept-inexact policy_tail` accepts).
    static func gaps(saved: ChessNetwork.PolicyTailPrecision?,
                     running: ChessNetwork.PolicyTailPrecision) -> [ResumeGap] {
        finding(saved: saved, running: running) == .matches ? [] : [.policyTail]
    }

    /// The `[RESUME-NUMERICS]` line a CLI exact resume logs. A mismatch is a
    /// `policy_tail` gap: the one `[RESUME]` line names it and the resume is
    /// refused unless `--accept-inexact` names it.
    static func exactResumeLogLine(saved: ChessNetwork.PolicyTailPrecision?,
                                   running: ChessNetwork.PolicyTailPrecision) -> String {
        switch finding(saved: saved, running: running) {
        case .matches:
            return "[RESUME-NUMERICS] policy_tail_precision=\(running.rawValue) matches the checkpoint"
        case .differs(let saved):
            return "[RESUME-NUMERICS] the checkpoint was trained with policy tail precision \(saved.rawValue) "
                + "but this run uses \(running.rawValue) (gap policy_tail); \(ChessNetwork.PolicyTailPrecision.flag) "
                + "\(saved.rawValue) continues it exactly"
        case .unrecorded:
            return "[RESUME-NUMERICS] the checkpoint does not record its policy tail precision (written before "
                + "the setting was saved); this run uses \(running.rawValue) (gap policy_tail)"
        }
    }

    /// The line a GUI resume logs, or nil when the precision matches.
    static func guiNotExactLine(
        saved: ChessNetwork.PolicyTailPrecision?,
        running: ChessNetwork.PolicyTailPrecision
    ) -> String? {
        switch finding(saved: saved, running: running) {
        case .matches:
            return nil
        case .differs(let saved):
            return "[RESUME] NOT EXACT: policy_tail saved=\(saved.rawValue) running=\(running.rawValue)"
        case .unrecorded:
            return "[RESUME] NOT EXACT: policy_tail saved=unrecorded running=\(running.rawValue)"
        }
    }
}
