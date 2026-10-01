import Foundation

extension NumericsAudit.Result {

    /// How many findings the text summary lists; the JSON has them all.
    static let summaryFindingLimit = 25

    /// The `[NUMERICS]` log block: headline numbers per check and the worst
    /// findings.
    func textSummary() -> String {
        typealias A = NumericsAudit
        var lines: [String] = []
        let id = modelID.map { " id \($0)" } ?? ""
        let step = trainingStep.map { " step \($0)" } ?? ""
        lines.append("model \(modelLabel)\(id)\(step), compute \(computeDataType)\(dynamicChecks == nil ? "" : ", policy tail \(policyTailPrecision)")")
        lines.append("overall: \(overallVerdict.rawValue)")

        if let offset = staticChecks.valueHeadOffset {
            lines.append("value fc2 shared offset: mean-row ratio \(A.fmt(offset.ratio)) = \(A.fmt(offset.ratioToInitExpectation))x init, bias mean \(A.fmt(offset.biasMean)) (init \(A.fmt(offset.biasInitMean))) -> \(offset.verdict.rawValue)")
        }
        if let offset = staticChecks.policyHeadOffset {
            lines.append("policy final shared offset: mean-row ratio \(A.fmt(offset.ratio)) = \(A.fmt(offset.ratioToInitExpectation))x init, bias mean \(A.fmt(offset.biasMean)) -> \(offset.verdict.rawValue)")
        }
        if let worst = staticChecks.batchNormStats.max(by: { $0.maxMeanToStd < $1.maxMeanToStd }) {
            lines.append("batch-norm stats: worst |mean|/std \(A.fmt(worst.maxMeanToStd)) at \(worst.layerName) channel \(worst.maxMeanToStdChannel) -> \(worst.verdict.rawValue)")
        }
        if !staticChecks.reZero.isEmpty {
            let perFormat = NumericFormat.allCases.map { format in
                "\(format.rawValue) \(staticChecks.reZero.filter { $0.saturatesInFormat[format.rawValue] == true }.count)"
            }.joined(separator: ", ")
            lines.append("ReZero: \(staticChecks.reZero.count) blocks; tanh bound rounds to 1 in \(perFormat)")
        }
        if let divergence = staticChecks.masterDivergence {
            if let top = divergence.first {
                lines.append("masters vs working: largest divergence \(A.fmt(top.maxDifferenceInBF16Steps)) bf16 steps in \(top.name)")
            } else {
                lines.append("masters vs working: no tensors compared")
            }
        } else if let note = staticChecks.mastersNote {
            lines.append("masters vs working: not compared (\(note))")
        }

        if let dynamic = dynamicChecks {
            let p = dynamic.positions
            var positionLine = "positions: \(p.total) (start \(p.startPosition), corpus \(p.corpusPositions), Lichess \(p.lichessPositions) from \(p.lichessGames) games; \(p.withValueTarget) with a game result)"
            if let note = p.corpusNote { positionLine += "; corpus: \(note)" }
            if let note = p.lichessNote { positionLine += "; Lichess: \(note)" }
            if !p.lichessGamesSkipped.isEmpty { positionLine += "; \(p.lichessGamesSkipped.count) Lichess game(s) skipped (listed in the JSON)" }
            lines.append(positionLine)
            for (format, error) in dynamic.formatBuildErrors.sorted(by: { $0.key < $1.key }) {
                lines.append("\(format) network not built: \(error)")
            }
            if let value = dynamic.valueHead {
                for report in value {
                    let median = report.sharedLogitPercentiles.count > 2 ? report.sharedLogitPercentiles[2] : nil
                    let wdl = report.startPositionWDL.map { String(format: "%.3f", $0) }.joined(separator: "/")
                    lines.append("value \(report.format.rawValue): ties \(A.fmt(report.tieFraction)), CE \(A.fmt(report.crossEntropyMean)) (delta \(A.fmt(report.crossEntropyDeltaVsFP32))), mean |dv| \(A.fmt(report.meanAbsDeltaV)), argmax changed \(A.fmt(report.argmaxChangedFraction)), shared logit median \(A.fmt(median)), start W/D/L \(wdl) -> \(report.verdict.rawValue)")
                }
            } else if let note = dynamic.valueHeadNote {
                lines.append("value head: \(note)")
            }
            for report in dynamic.policyHead {
                let level = report.legalMeanPercentiles.count > 2 ? report.legalMeanPercentiles[2] : nil
                lines.append("policy \(report.format.rawValue): KL \(A.fmt(report.klMean)), top-2 ties \(A.fmt(report.top2TieFraction)), top-1 changed \(A.fmt(report.top1ChangedFraction)), legal level median \(A.fmt(level)), legal spread median \(A.fmt(report.legalSpreadMedian)), all-move mean median \(A.fmt(report.allMoveMeanMedian)) -> \(report.verdict.rawValue)")
            }
            lines.append("activation taps: \(dynamic.taps.count)")
        } else if let reason = dynamicSkippedReason {
            lines.append("dynamic checks skipped: \(reason)")
        }

        let bad = findings.filter { $0.verdict == .bad }.count
        let degraded = findings.filter { $0.verdict == .degraded }.count
        lines.append("findings: \(bad) BAD, \(degraded) degraded")
        for finding in findings.prefix(Self.summaryFindingLimit) {
            let format = finding.format.map { " [\($0.rawValue)]" } ?? ""
            lines.append("  \(finding.verdict.rawValue) \(finding.area): \(finding.subject)\(format) — \(finding.detail)")
        }
        if findings.count > Self.summaryFindingLimit {
            lines.append("  (\(findings.count - Self.summaryFindingLimit) more in the JSON)")
        }
        return lines.joined(separator: "\n")
    }
}
