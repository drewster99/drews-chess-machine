import SwiftUI

/// A model file's test-set figures in a picker row: pElo, NLL, top-1 and
/// top-5 of its largest set (`ModelTestSetSummary`), each labeled (the
/// pickers have no column headers) in fixed-width columns so rows align.
/// The tooltip names the set and gives its counts, or says why there are no
/// figures. Shared by the Lichess bot model picker and the session picker.
struct ModelTestSetSummaryCells: View {
    let summary: ModelTestSetSummary
    /// Monospaced, at the row's text style.
    let textStyle: Font.TextStyle

    var body: some View {
        HStack(spacing: 10) {
            Text("pElo \(summary.pEloText)")
                .frame(width: textStyle == .caption ? 66 : 80, alignment: .leading)
            Text("NLL \(summary.nllText)")
                .frame(width: textStyle == .caption ? 70 : 84, alignment: .leading)
            Text("top-1 \(summary.top1Text)")
                .frame(width: textStyle == .caption ? 84 : 100, alignment: .leading)
            Text("top-5 \(summary.top5Text)")
                .frame(width: textStyle == .caption ? 84 : 100, alignment: .leading)
        }
        .font(.system(textStyle, design: .monospaced))
        .foregroundStyle(summary.largestSet == nil ? Color.secondary : Color.primary)
        .help(summary.help)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(summary.line)
    }
}
