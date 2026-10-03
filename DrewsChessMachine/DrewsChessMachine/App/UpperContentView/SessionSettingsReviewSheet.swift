import SwiftUI

/// A session load stopped on saved settings that cannot be used as found. The
/// user resumes with the current settings in their place, or does not resume.
struct SessionSettingsReviewSheet: View {
    let review: SessionSettingsReview
    let onResumeWithReplacements: () -> Void
    let onDoNotResume: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("This session has saved settings that can't be used")
                .font(.title3.weight(.semibold))
            Text(review.sessionURL.lastPathComponent)
                .font(.callout.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Text("Its session.json holds the values below, which the app can't resume with. You can resume using your current settings in their place — the session file is not changed — or not resume.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            List(review.findings) { finding in
                InvalidSettingRow(setting: finding, replacementLabel: "Replace with")
            }
            .frame(minHeight: 160)
            HStack {
                Button("Don't Resume", role: .cancel, action: onDoNotResume)
                Spacer()
                Button("Resume with Replacements", action: onResumeWithReplacements)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 600, minHeight: 380)
    }
}
