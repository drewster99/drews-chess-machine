import SwiftUI

/// Champion ID / Trainer ID / Network architecture block — surfaced
/// in the auto-resume sheet so the user can verify the model
/// lineage and the architecture shape that produced the session
/// before confirming the resume.
///
/// Architecture line falls back to "unknown (session predates
/// architecture metadata)" for sessions saved before
/// `ArchitectureMetadata` was added to `session.json`. The build
/// block below already flags a build-version mismatch separately —
/// when a mismatch fires, the architecture line here describes the
/// *saved* arch, so the two together let the user see exactly what
/// changed.
struct AutoResumeModelBlockView: View {
    let summary: SessionResumeSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            AutoResumeStatRowView(label: "Champion", value: summary.championID)
            AutoResumeStatRowView(label: "Trainer", value: summary.trainerID)
            AutoResumeStatRowView(label: "Network", value: archDescription)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.secondary.opacity(0.08))
        )
    }

    /// `v4 · 12 blocks · 128 channels · SE/4 · 3.9M params` when
    /// architecture metadata is present, or a "(unknown)" fallback
    /// when the session predates the field.
    private var archDescription: String {
        guard let arch = summary.architecture else {
            return "unknown (saved before arch metadata)"
        }
        let paramsStr = AutoResumeFormat.count(arch.parameterCount) + " params"
        return "v\(arch.architectureVersion) · "
            + "\(arch.numBlocks) blocks · "
            + "\(arch.channels) channels · "
            + "SE/\(arch.seReductionRatio) · "
            + paramsStr
    }
}
