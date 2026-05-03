import SwiftUI

/// Right-hand stats panel under the board: header line, an
/// optional block of training-control rows (only visible while
/// `realTraining` is true), and the colorized body text. The
/// control rows are state-heavy enough — ~9 rows of
/// `ParameterTextField` editors plus their apply callbacks — that
/// the caller supplies them as a `@ViewBuilder` closure rather
/// than threading every binding through this view's init.
struct TrainingStatsColumn<Controls: View>: View {
    let header: String
    let bodyText: AttributedString
    let realTraining: Bool
    @ViewBuilder var controls: () -> Controls

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(header)
            if realTraining {
                controls()
            }
            Text(bodyText)
        }
        .frame(minWidth: 260, alignment: .topLeading)
    }
}
