import Foundation

extension SessionController {

    /// Run the 200-puzzle Lichess probe battery against the live
    /// champion network and log a per-theme breakdown under
    /// `[TACTICAL-LICHESS]`. The manual companion of
    /// `runTacticalProbe()` — same trigger model (menu item, one click)
    /// but a heavier battery with aggregate-only output instead of
    /// per-probe rows.
    ///
    /// Runs against `network` (the champion). The same rationale as
    /// `runTacticalProbe()`: the trainer's weights mutate constantly
    /// and would give a moving-target read; the user clicks this
    /// expecting "what is the deployed network actually doing right
    /// now." The periodic watcher (`startLichessProbeWatcher()`) is
    /// the one that targets `candidate` by default so the monitor's
    /// trend line tracks the trainer's evolution between promotions.
    func runLichessProbe() {
        SessionLogger.shared.log("[BUTTON] Run Lichess Probe (200-puzzle)")
        let net = network
        let modelLabel = network?.identifier?.description ?? "<no-id>"
        Task {
            await self.runLichessProbeAsync(net: net, modelLabel: modelLabel)
        }
    }

    private func runLichessProbeAsync(
        net: ChessMPSNetwork?,
        modelLabel: String
    ) async {
        guard let net else {
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS] no champion network — build one first"
            )
            return
        }

        let probes = LichessProbeData.largeSet
        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] === begin n=\(probes.count) net=champion model=\(modelLabel) ==="
        )

        var resultsByCategory: [ProbeCategory: [ProbeResult]] = [:]
        var verdictCounts: [ProbeVerdict: Int] = [:]
        var allResults: [ProbeResult] = []
        allResults.reserveCapacity(probes.count)
        for probe in probes {
            let r = await TacticalProbeRunner.run(probe, against: net)
            resultsByCategory[probe.category, default: []].append(r)
            verdictCounts[r.verdict, default: 0] += 1
            allResults.append(r)
        }

        // Mirror the periodic watcher: a manual run also refreshes the
        // shared `lichessProbeHistory` so any open Monitor / Detail
        // window updates immediately, and the JSON exporter can dump
        // the manual-run snapshot. Capture the four progress fields the
        // export schema carries (training step + derived positions
        // trained + cumulative active training time + arena/promotion
        // counts) at tick time so the snapshot is consistent with the
        // probed weights regardless of when the export is invoked.
        let trainingStep = trainer?.completedTrainSteps
        let positionsTrained = trainingStep.map {
            $0 * TrainingParameters.shared.trainingBatchSize
        }
        let activeTrainingSec = checkpoint?.cumulativeActiveTrainingSec
        let arenaCount = tournamentHistory.count
        let promotionCount = tournamentHistory.lazy.filter { $0.promoted }.count

        let aggregates = LichessProbeHistory.aggregates(from: allResults)
        lichessProbeHistory.record(
            aggregates,
            allResults: allResults,
            modelLabel: modelLabel,
            trainingStep: trainingStep,
            positionsTrained: positionsTrained,
            activeTrainingSec: activeTrainingSec,
            arenaCount: arenaCount,
            promotionCount: promotionCount
        )

        // Per-theme breakdown lines, stable order by raw value.
        var totalArgmaxCorrect = 0
        var totalProbes = 0
        for cat in resultsByCategory.keys.sorted(by: { $0.rawValue < $1.rawValue }) {
            guard let results = resultsByCategory[cat] else { continue }
            var argmax = 0
            var top5 = 0
            var errs = 0
            var sumProb: Float = 0
            var sumRank = 0
            var countRank = 0
            for r in results {
                sumProb += r.expectedProb
                if let rank = r.expectedRank {
                    sumRank += rank
                    countRank += 1
                }
                switch r.verdict {
                case .correctAndConfident, .correctButFlat:
                    argmax += 1
                    top5 += 1
                case .correctInTop5:
                    top5 += 1
                case .wrong:
                    break
                case .error:
                    errs += 1
                }
            }
            totalArgmaxCorrect += argmax
            totalProbes += results.count
            let pct = results.count > 0
                ? String(format: "%.1f", 100.0 * Double(argmax) / Double(results.count))
                : "0.0"
            let avgProb = results.count > 0
                ? String(format: "%.3f", sumProb / Float(results.count))
                : "—"
            let avgRank = countRank > 0
                ? String(format: "%.2f", Float(sumRank) / Float(countRank))
                : "—"
            SessionLogger.shared.log(
                "[TACTICAL-LICHESS]   \(cat.rawValue): argmax=\(argmax)/\(results.count) "
                + "(\(pct)%) top5=\(top5)/\(results.count) "
                + "avgProb=\(avgProb) avgRank=\(avgRank) errors=\(errs)"
            )
        }

        let totalPct = totalProbes > 0
            ? String(format: "%.1f", 100.0 * Double(totalArgmaxCorrect) / Double(totalProbes))
            : "0.0"
        let summary = ProbeVerdict.allOrderedForReport
            .map { v in "\(v.rawValue)=\(verdictCounts[v] ?? 0)" }
            .joined(separator: " ")
        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] === summary: argmax=\(totalArgmaxCorrect)/\(totalProbes) "
            + "(\(totalPct)%)  verdicts: \(summary) ==="
        )
    }
}
