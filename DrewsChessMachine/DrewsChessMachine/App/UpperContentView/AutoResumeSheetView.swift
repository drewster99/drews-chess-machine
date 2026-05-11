import SwiftUI

/// The "Resume last training session?" sheet shown at app launch when a
/// `LastSessionPointer` still names an on-disk `.dcmsession`. Extracted out of
/// `UpperContentView` (it used to be a `autoResumeSheetContentView() -> AnyView`
/// helper, written that way only to keep the `.sheet { … }` call site from
/// inflating the already-huge body's type-inference cost — a concrete `View`
/// struct does that job better and drops the `AnyView`).
///
/// The countdown / pointer / summary state and the actual resume + dismiss
/// logic stay on `UpperContentView` for now; this view just renders them and
/// reports the two button presses back through closures.
struct AutoResumeSheetView: View {
    /// Pointer the sheet is offering to resume (the exact one captured when the
    /// sheet was presented, not whatever `LastSessionPointer.read()` returns now).
    let pointer: LastSessionPointer
    /// Lightweight peek of the target session's `session.json`, or `nil` if the
    /// peek failed — in which case the sheet falls back to a pointer-only layout
    /// so a corrupt session still gets the resume prompt rendered.
    let summary: SessionResumeSummary?
    /// Seconds remaining on the auto-resume countdown.
    let countdownRemaining: Int
    /// "Not Now" — dismiss the sheet and cancel the countdown.
    let onDismiss: () -> Void
    /// "Resume Training" (or the countdown firing) — tear down the sheet and
    /// start the load-and-resume chain.
    let onResume: () -> Void

    var body: some View {
        let savedAtString: String = {
            let f = DateFormatter()
            f.dateStyle = .medium
            f.timeStyle = .short
            return f.string(from: Date(timeIntervalSince1970: TimeInterval(pointer.savedAtUnix)))
        }()
        let agoString = AutoResumeFormat.relativeAgo(savedAtUnix: pointer.savedAtUnix)
        let plural = (countdownRemaining == 1 ? "" : "s")
        let sessionLine = "Session: \(pointer.sessionID)"
        let savedLine = "Saved \(agoString) (\(savedAtString))"
        let folderLine = pointer.directoryURL.lastPathComponent
        let countdownLine = "Training will automatically resume in \(countdownRemaining) second\(plural)."
        let resumeLabel = "Resume Training (\(countdownRemaining))"

        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text("Resume last training session?")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                AutoResumeTriggerBadgeView(trigger: pointer.trigger)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(sessionLine)
                if let summary {
                    Text(AutoResumeFormat.startedLine(sessionStartUnix: summary.sessionStartUnix))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Text(savedLine)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if let summary {
                AutoResumeProgressBlockView(summary: summary)
                AutoResumeBuildBlockView(summary: summary)
            }
            Button {
                CheckpointManager.revealInFinder(pointer.directoryURL)
            } label: {
                Text(folderLine)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .underline()
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .buttonStyle(.plain)
            .pointerStyle(.link)
            .help("Reveal in Finder")
            Text(countdownLine)
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Spacer()
                Button("Not Now") { onDismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(resumeLabel) { onResume() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 520)
    }
}
