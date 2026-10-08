import SwiftUI

/// The session picker detail's test-set results: each model file of the
/// save with its largest set's figures, as recorded in the file at save time
/// (`SessionManifest.championTestSets` / `trainerTestSets`).
struct SessionPickerTestSetsSection: View {
    let manifest: SessionManifest

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("TEST SETS AT SAVE")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            SessionPickerTestSetsFileRow(label: "Champion", summary: manifest.championTestSets ?? .notRecorded)
            SessionPickerTestSetsFileRow(label: "Trainer", summary: manifest.trainerTestSets ?? .notRecorded)
        }
    }
}

/// One file's line in `SessionPickerTestSetsSection`.
struct SessionPickerTestSetsFileRow: View {
    let label: String
    let summary: ModelTestSetSummary

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 130, alignment: .leading)
            Text(summary.line)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(summary.largestSet == nil ? Color.secondary : Color.primary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .help(summary.help)
        }
    }
}
