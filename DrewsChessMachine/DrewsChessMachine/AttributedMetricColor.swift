import SwiftUI

/// Builds `AttributedString` bodies for the training and self-play
/// text panels, coloring a small, deliberate set of diagnostic
/// metrics where the band tells the reader something useful:
///   - red     → a clearly-bad value that warrants attention
///   - orange  → a watch-zone value that could degrade
///   - green   → a clearly-healthy value
///   - default → everything else
///
/// The scope is intentionally narrow — only metrics with established
/// healthy/warning/alarm bands in the project's documentation and
/// session-log discipline. Every other value stays default so the
/// panels don't devolve into a rainbow.
///
/// Thresholds used here are calibrated against the running-values
/// the user has described as healthy / alarming in CLAUDE.md and
/// the in-tree `policyEntropyAlarmThreshold` constant, not invented.
enum AttributedMetricColor {

    /// Thresholds for the metrics the colorizer understands. Held
    /// as a parameter struct so tests and future callers can tweak
    /// bands without editing the parser.
    struct Thresholds {
        /// Policy entropy "collapse" band — below this is red.
        /// Matches `ContentView.policyEntropyAlarmThreshold` (1.0
        /// in-repo, calibrated for post-mask / legal-only entropy).
        /// The "flat" orange band (`entropyFlatAbove`) is currently
        /// derived from `ln(policySize)` — an upper bound that only
        /// applies to the unmasked logit space, so in practice the
        /// orange band never fires for post-mask pEnt. Collapse-red
        /// is the operative band.
        var entropyCollapseBelow: Double
        var entropyFlatAbove: Double
        /// Value-head absolute mean saturation bands. Above 0.9 is
        /// effectively saturated; 0.7-0.9 is watch-zone.
        var vAbsRedAbove: Double
        var vAbsOrangeAbove: Double
        /// Gradient L2 norm vs. the configured clip ceiling. At or
        /// above the ceiling is a clip event (orange). Well above
        /// (≥ 2× ceiling) is red — unclipped that would be a
        /// divergence candidate.
        var gradClipMaxNorm: Double
        /// Game-diversity unique-percent bands (0–100).
        var diversityRedBelow: Double
        var diversityOrangeBelow: Double
        var diversityGreenAbove: Double

        static func `default`(entropyCollapseBelow: Double, gradClipMaxNorm: Double) -> Thresholds {
            // Derived from the live policy width so the band tracks
            // the head shape automatically. 0.12 nats below uniform
            // matches the original calibration (8.20 vs ln(4096)≈8.32
            // in the prior 4096-cell head).
            let uniformEntropy = log(Double(ChessNetwork.policySize))
            return Thresholds(
                entropyCollapseBelow: entropyCollapseBelow,
                entropyFlatAbove: uniformEntropy - 0.12,
                vAbsRedAbove: 0.90,
                vAbsOrangeAbove: 0.70,
                gradClipMaxNorm: gradClipMaxNorm,
                diversityRedBelow: 30,
                diversityOrangeBelow: 80,
                diversityGreenAbove: 80
            )
        }
    }

    /// Colorize a plain-text multi-line body produced by
    /// `trainingStatsText().body`. Any line that matches a known
    /// metric prefix has its value span recolored; everything else
    /// is copied through with no attributes. Preserves line breaks
    /// exactly so monospaced column alignment still lines up.
    static func colorize(
        body: String,
        thresholds: Thresholds
    ) -> AttributedString {
        var out = AttributedString("")
        let lines = body.components(separatedBy: "\n")
        for (i, line) in lines.enumerated() {
            out.append(colorizeLine(line, thresholds: thresholds))
            if i < lines.count - 1 {
                out.append(AttributedString("\n"))
            }
        }
        return out
    }

    // MARK: - Line-level coloring

    private static func colorizeLine(
        _ line: String,
        thresholds t: Thresholds
    ) -> AttributedString {
        // Known metric labels and their value extraction/coloring.
        // Each handler returns nil if the line doesn't match, in
        // which case we fall through to a plain-text copy.
        if let attr = colorEntropyLine(line, t: t) { return attr }
        if let attr = colorVAbsLine(line, t: t) { return attr }
        if let attr = colorGradNormLine(line, t: t) { return attr }
        if let attr = colorDiversityLine(line, t: t) { return attr }
        return AttributedString(line)
    }

    /// `  Entropy:     8.123456` (Last Step section). Red below
    /// the collapse threshold, orange very close to uniform
    /// (`>= entropyFlatAbove`), else default.
    private static func colorEntropyLine(
        _ line: String,
        t: Thresholds
    ) -> AttributedString? {
        let prefix = "  Entropy:"
        guard line.hasPrefix(prefix) else { return nil }
        guard let (valueRange, value) = trailingNumber(in: line) else { return nil }
        let color = colorForEntropy(value, t: t)
        return assembleColored(line: line, numberRange: valueRange, color: color)
    }

    /// `    v abs:         0.1234`. Red above 0.9, orange above 0.7.
    private static func colorVAbsLine(
        _ line: String,
        t: Thresholds
    ) -> AttributedString? {
        let prefix = "    v abs:"
        guard line.hasPrefix(prefix) else { return nil }
        guard let (valueRange, value) = trailingNumber(in: line) else { return nil }
        let color: Color?
        if value >= t.vAbsRedAbove { color = .red }
        else if value >= t.vAbsOrangeAbove { color = .orange }
        else { color = nil }
        return assembleColored(line: line, numberRange: valueRange, color: color)
    }

    /// `  Grad norm:   2.345`. Orange at/above clip ceiling, red
    /// at ≥ 2× clip ceiling. Leaves default styling below the clip.
    private static func colorGradNormLine(
        _ line: String,
        t: Thresholds
    ) -> AttributedString? {
        let prefix = "  Grad norm:"
        guard line.hasPrefix(prefix) else { return nil }
        guard let (valueRange, value) = trailingNumber(in: line) else { return nil }
        let color: Color?
        if value >= 2 * t.gradClipMaxNorm { color = .red }
        else if value >= t.gradClipMaxNorm { color = .orange }
        else { color = nil }
        return assembleColored(line: line, numberRange: valueRange, color: color)
    }

    /// `  Diversity:   195/200 unique (97.5%)  avg diverge ply 8.3`.
    /// Colors the percentage token only — that's the headline for
    /// "is self-play collapsing" — and leaves the rest of the line
    /// alone.
    private static func colorDiversityLine(
        _ line: String,
        t: Thresholds
    ) -> AttributedString? {
        let prefix = "  Diversity:"
        guard line.hasPrefix(prefix) else { return nil }
        // Find "(NN.N%)" token and extract the percentage.
        guard let openParen = line.firstIndex(of: "("),
              let percentIdx = line[openParen...].firstIndex(of: "%") else {
            return nil
        }
        // Range from just after the open-paren through the `%` sign.
        let startIdx = line.index(after: openParen)
        guard let value = Double(line[startIdx..<percentIdx]) else { return nil }
        let color: Color?
        if value < t.diversityRedBelow { color = .red }
        else if value < t.diversityOrangeBelow { color = .orange }
        else if value > t.diversityGreenAbove { color = .green }
        else { color = nil }
        let tokenRange = startIdx..<line.index(after: percentIdx)
        return assembleColored(line: line, numberRange: tokenRange, color: color)
    }

    // MARK: - Helpers

    private static func colorForEntropy(_ v: Double, t: Thresholds) -> Color? {
        if v < t.entropyCollapseBelow { return .red }
        if v >= t.entropyFlatAbove { return .orange }
        return nil
    }

    /// Strip trailing whitespace-ish characters from the line and
    /// return the last whitespace-delimited token as a Double if it
    /// parses. Returns the token's character range in the original
    /// string so the caller can overlay colored attributes on the
    /// number only. Returns nil if the trailing token isn't a
    /// parseable number.
    private static func trailingNumber(in line: String) -> (Range<String.Index>, Double)? {
        guard let lastSpaceIdx = line.lastIndex(where: { $0 == " " }) else {
            return nil
        }
        let valueStart = line.index(after: lastSpaceIdx)
        let valueEnd = line.endIndex
        guard valueStart < valueEnd else { return nil }
        let token = String(line[valueStart..<valueEnd])
        guard let value = Double(token) else { return nil }
        return (valueStart..<valueEnd, value)
    }

    /// Build an `AttributedString` where the substring at `numberRange`
    /// carries the given foreground color (or the whole line is plain
    /// text if color is nil). Assembles from three slices
    /// (prefix / value / suffix) rather than trying to find the
    /// value token inside an already-constructed `AttributedString`,
    /// which is fragile when the value repeats or collides with
    /// similar substrings earlier in the line.
    private static func assembleColored(
        line: String,
        numberRange: Range<String.Index>,
        color: Color?
    ) -> AttributedString {
        guard let color else { return AttributedString(line) }
        let prefix = String(line[line.startIndex..<numberRange.lowerBound])
        let value = String(line[numberRange])
        let suffix = String(line[numberRange.upperBound..<line.endIndex])
        var out = AttributedString(prefix)
        var valueAttr = AttributedString(value)
        valueAttr.foregroundColor = color
        out.append(valueAttr)
        out.append(AttributedString(suffix))
        return out
    }
}
