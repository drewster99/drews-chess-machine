import SwiftUI

/// Bottom pane of the Log Analysis window: displays Claude's
/// markdown-formatted response as it streams in, plus a small
/// "Analyzing…" indicator while the subprocess is running and an
/// error fallback if the run failed.
struct ClaudeAnalysisPaneView: View {
    let isAnalyzing: Bool
    let errorMessage: String?
    let response: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text("Claude analysis")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if isAnalyzing {
                    ProgressView()
                        .controlSize(.small)
                    Text("Analyzing…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            .padding(.bottom, 2)
            ScrollView {
                Group {
                    if let errorMessage {
                        Text("Error: \(errorMessage)")
                            .font(.body)
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    } else if !response.isEmpty {
                        MarkdownText(content: response)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    } else if isAnalyzing {
                        Text("Waiting for Claude to finish…")
                            .foregroundStyle(.secondary)
                            .padding(8)
                    } else {
                        Text("No response.")
                            .foregroundStyle(.secondary)
                            .padding(8)
                    }
                }
            }
        }
        .frame(minHeight: 150)
    }
}
