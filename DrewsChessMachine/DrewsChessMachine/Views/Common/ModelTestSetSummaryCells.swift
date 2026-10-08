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
        let widths = ColumnWidths(textStyle)
        HStack(spacing: 10) {
            Text("pElo \(summary.pEloText)")
                .frame(width: widths.pElo, alignment: .leading)
            Text("NLL \(summary.nllText)")
                .frame(width: widths.nll, alignment: .leading)
            Text("top-1 \(summary.top1Text)")
                .frame(width: widths.topK, alignment: .leading)
            Text("top-5 \(summary.top5Text)")
                .frame(width: widths.topK, alignment: .leading)
        }
        .font(.system(textStyle, design: .monospaced))
        .foregroundStyle(summary.largestSet == nil ? Color.secondary : Color.primary)
        .help(summary.help)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(summary.line)
    }

    /// Column widths that fit the widest label and figure ("pElo all ✓",
    /// "NLL 18.421", "top-5 100.0%") in the row's monospaced text style.
    private struct ColumnWidths {
        let pElo: CGFloat
        let nll: CGFloat
        let topK: CGFloat

        init(_ textStyle: Font.TextStyle) {
            switch textStyle {
            case .caption, .caption2, .footnote:
                pElo = 66
                nll = 70
                topK = 84
            default:
                pElo = 80
                nll = 84
                topK = 100
            }
        }
    }
}
