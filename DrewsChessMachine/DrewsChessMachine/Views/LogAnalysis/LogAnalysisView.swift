import SwiftUI

struct LogAnalysisView: View {
    @ObservedObject var viewModel: LogAnalysisViewModel

    var body: some View {
        VSplitView {
            SessionLogPaneView(
                logPath: viewModel.logPath,
                logContent: viewModel.logContent
            )
            ClaudeAnalysisPaneView(
                isAnalyzing: viewModel.isAnalyzing,
                errorMessage: viewModel.errorMessage,
                response: viewModel.claudeResponse
            )
        }
    }
}
