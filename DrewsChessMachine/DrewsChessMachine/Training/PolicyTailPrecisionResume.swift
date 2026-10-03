import Foundation

/// How a resume treats the policy-head tail precision a trainer file was
/// saved under (`trainer_policy_tail_precision`) against the precision this
/// process runs (`ChessNetwork.PolicyTailPrecision.process`).
///
/// The setting is not an architecture field — the same weights load under
/// either value — but on bf16 / fp16 models it changes the policy head's
/// arithmetic, so a lineage resumed under the other value is not the run it
/// continues. One pure finding shared by every resume path: a mismatch, or a
/// file written before the setting was recorded, is the `policy_tail` resume
/// gap (`ResumeExactness`) — named in the one `[RESUME]` line, refused by a
/// CLI `--resume-exact` unless `--accept-inexact policy_tail` names it, and
/// reported (never refused) by a GUI resume (decision D-1).
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
}
