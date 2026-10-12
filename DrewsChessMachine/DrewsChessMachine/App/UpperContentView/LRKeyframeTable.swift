import SwiftUI
import TrainingParametersMacroSupport

/// The LR keyframe editor on the Cycling tab (documentation/plans-active/
/// LR_KEYFRAMES_PLAN.md, owner decision O-6): one row per keyframe — trainer
/// step, peak LR, trough LR, and the momentum at the LR peak and at the LR
/// trough (used while momentum follows the LR cycle) — with add and delete.
/// The text is validated on Save, as a whole schedule, by the parameter's own
/// validator; `error` shows its message.
struct LRKeyframeTable: View {
    @Binding var rows: [LRKeyframeRowDraft]
    let error: String?
    let periodSteps: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Grid(alignment: .trailing, horizontalSpacing: 6, verticalSpacing: 4) {
                LRKeyframeTableHeader()
                ForEach($rows, id: \.id) { $row in
                    LRKeyframeTableRow(row: $row, canDelete: rows.count > 1) {
                        rows.removeAll { $0.id == row.id }
                    }
                }
            }
            HStack {
                Button("Add Keyframe") {
                    rows.append(LRKeyframeRowDraft.appended(after: rows, periodSteps: periodSteps))
                }
                .controlSize(.small)
                .disabled(rows.count >= LRKeyframeSchedule.maximumKeyframeCount)
                Text(error ?? "")
                    .font(.system(size: 11))
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// The column titles of `LRKeyframeTable`.
private struct LRKeyframeTableHeader: View {
    var body: some View {
        GridRow {
            Text("Step")
            Text("Peak LR")
            Text("Trough LR")
            Text("μ at peak")
            Text("μ at trough")
            Text("")
        }
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(.secondary)
    }
}

/// One editable keyframe of `LRKeyframeTable`.
private struct LRKeyframeTableRow: View {
    @Binding var row: LRKeyframeRowDraft
    let canDelete: Bool
    let delete: () -> Void

    var body: some View {
        GridRow {
            TextField("", text: $row.step).frame(width: 80)
            TextField("", text: $row.peak).frame(width: 80)
            TextField("", text: $row.trough).frame(width: 80)
            TextField("", text: $row.momentumAtPeak).frame(width: 60)
            TextField("", text: $row.momentumAtTrough).frame(width: 60)
            Button(action: delete) {
                Image(systemName: "minus.circle")
            }
            .buttonStyle(.borderless)
            .disabled(!canDelete)
            .help("Delete this keyframe")
        }
        .font(.system(size: 12, design: .monospaced))
        .multilineTextAlignment(.trailing)
        .textFieldStyle(.roundedBorder)
    }
}
