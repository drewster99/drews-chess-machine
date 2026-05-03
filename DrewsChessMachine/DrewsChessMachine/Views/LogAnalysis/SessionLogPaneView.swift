import SwiftUI

/// Top pane of the Log Analysis window: displays the raw session-log
/// file contents in a monospaced scroll view, with a header line
/// naming the log file.
struct SessionLogPaneView: View {
    let logPath: String
    let logContent: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Session log · \((logPath as NSString).lastPathComponent)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            .padding(.bottom, 2)
            ScrollView {
                Text(logContent)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
        }
        .frame(minHeight: 150)
    }
}
