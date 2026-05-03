import SwiftUI

/// Build line + version-mismatch callout. Compares the persisted
/// `buildNumber` and `buildGitHash` against the current
/// `BuildInfo` constants; on mismatch (either field) surfaces a
/// yellow warning with the current build's number and short hash,
/// so the user knows resume is straddling a code-version boundary.
/// `buildGitDirty == true` on the saved build appends a `(dirty)`
/// marker. A session predating the build-info schema
/// (`buildNumber == nil`) renders as "Build: unknown" with no
/// warning.
struct AutoResumeBuildBlockView: View {
    let summary: SessionResumeSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let savedNum = summary.buildNumber {
                let savedHash = summary.buildGitHash ?? "?"
                let dirtyTag = (summary.buildGitDirty == true) ? " (dirty)" : ""
                let savedLine = "Trained under app build #\(savedNum) (\(savedHash))\(dirtyTag)"
                Text(savedLine)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                let mismatch = (savedNum != BuildInfo.buildNumber)
                    || (summary.buildGitHash != nil && summary.buildGitHash != BuildInfo.gitHash)
                if mismatch {
                    Text("⚠ Current app build is #\(BuildInfo.buildNumber) (\(BuildInfo.gitHash)) — last save used a different version.")
                        .font(.callout)
                        .foregroundStyle(Color.yellow)
                }
            } else {
                Text("Trained under app build: unknown (saved by an older version)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
