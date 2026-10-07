import Foundation

/// Everything the Record card shows, computed in one call from the games
/// index (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.4). Pure and `Sendable`:
/// the controller computes it on its statistics queue and the views only
/// format it.
///
/// Every filter (all / rated / casual) and every period is computed up
/// front, so changing either in the panel is a lookup, not a recompute, and
/// the controller holds no filter state. Each aggregation is one pass over
/// the rows with dictionaries for grouping; nothing is quadratic in games.
struct LichessBotRecordStatistics: Sendable, Equatable {

    /// The statistics of one filter.
    struct FilterStatistics: Sendable, Equatable {
        let periodRows: LichessBotPeriodValues<LichessBotPeriodStatistics>
        let byPeriod: LichessBotPeriodValues<LichessBotPeriodBreakdowns>
        /// Games with no result (aborted, never started), all time.
        let notCounted: Int
        /// Scored rated games without a recorded rating change, all time.
        let ratedWithoutRatingChange: Int
        /// Scored games without per-move facts, all time.
        let rowsWithoutMoveData: Int
    }

    let byFilter: LichessBotFilterValues<FilterStatistics>
    /// Per speed, the starting ratings of its last
    /// `ratingTrendLimit` rated games, oldest first. From rated games, so
    /// independent of the period and the filter.
    let ratingTrends: [String: [LichessBotRatingPoint]]
    /// Every speed any game was played at, for the Time controls rows.
    let recordSpeeds: Set<String>
    let computedAt: Date
    /// When some period's numbers next change without a new game
    /// (`LichessBotStatsPeriods.nextChange`).
    let validUntil: Date
    /// The time zone the periods were computed in.
    let timeZoneIdentifier: String

    static let ratingTrendLimit = 100

    subscript(filter: LichessBotStatsFilter) -> FilterStatistics {
        byFilter[filter]
    }

    static func compute(rows: [LichessBotGameSummary], now: Date, calendar: Calendar) throws -> LichessBotRecordStatistics {
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        let validUntil = try LichessBotStatsPeriods.nextChange(after: now, rows: rows, calendar: calendar)
        // Per-row derivations that don't depend on the filter or period,
        // done once.
        let derived = rows.map(DerivedRow.init)
        let byFilter = try LichessBotFilterValues<FilterStatistics> { filter in
            let included = derived.filter { filter.includes($0.row) }
            return try filterStatistics(included, now: now, calendar: calendar, starts: starts)
        }
        return LichessBotRecordStatistics(
            byFilter: byFilter,
            ratingTrends: ratingTrends(rows),
            recordSpeeds: Set(rows.map(\.speed)),
            computedAt: now,
            validUntil: validUntil,
            timeZoneIdentifier: calendar.timeZone.identifier
        )
    }

    // MARK: - Per filter

    private static func filterStatistics(_ rows: [DerivedRow], now: Date, calendar: Calendar, starts: LichessBotStatsPeriodStarts) throws -> FilterStatistics {
        // The W–D–L splits through `LichessBotRecordSummary.compute`, so the
        // split rules have one definition.
        let records = try LichessBotRecordSummary.compute(rows: rows.map(\.row), now: now, calendar: calendar)
        var accumulators = LichessBotPeriodValues { _ in PeriodAccumulator() }
        for row in rows {
            for period in LichessBotStatsPeriod.allCases where starts.contains(row.row.createdAt, in: period) {
                accumulators[period].add(row)
            }
        }
        let periodRows = LichessBotPeriodValues { period in
            accumulators[period].periodStatistics(record: records[period])
        }
        let allTime = accumulators.allTime
        return FilterStatistics(
            periodRows: periodRows,
            byPeriod: accumulators.map { $0.breakdowns() },
            notCounted: records.allTime.all.unscored,
            ratedWithoutRatingChange: allTime.ratingChange.gamesWithoutChange,
            rowsWithoutMoveData: allTime.scoredWithoutMoveData
        )
    }

    private static func ratingTrends(_ rows: [LichessBotGameSummary]) -> [String: [LichessBotRatingPoint]] {
        var bySpeed: [String: [LichessBotRatingPoint]] = [:]
        for row in rows where row.rated {
            guard let rating = row.ourRatingBefore else { continue }
            bySpeed[row.speed, default: []].append(LichessBotRatingPoint(gameID: row.gameID, createdAt: row.createdAt, rating: rating))
        }
        return bySpeed.mapValues { points in
            Array(points.sorted { lhs, rhs in
                lhs.createdAt != rhs.createdAt ? lhs.createdAt < rhs.createdAt : lhs.gameID < rhs.gameID
            }.suffix(ratingTrendLimit))
        }
    }
}

// MARK: - Per-row derivations

/// What the statistics read from one row beyond its plain fields, derived
/// once per compute.
private struct DerivedRow {
    let row: LichessBotGameSummary
    let ending: LichessBotGameEnding
    let attribution: ModelAttribution

    init(_ row: LichessBotGameSummary) {
        self.row = row
        ending = LichessBotGameEnding.classify(
            status: row.status,
            winner: row.winner,
            ourScore: row.ourScore,
            localDrawCondition: row.facts?.localDrawCondition,
            drawConditionKnown: row.facts != nil
        )
        attribution = ModelAttribution(row.facts?.moves)
    }
}

/// Which model a game belongs to (§3.8, OD-11): the model key whose
/// generations chose the most of DCM's moves; on a tie, the key that came
/// into use later. Two generations with one key (the same file reloaded)
/// are one model.
private struct ModelAttribution {
    struct Model {
        let key: LichessBotModelKey
        let facts: LichessBotGenerationFacts
    }

    /// Nil when no DCM move names a generation.
    let model: Model?
    /// More than one model decided DCM's moves.
    let mixed: Bool
    let decisionsWithoutGeneration: Int

    init(_ moves: LichessBotGameMoveFacts?) {
        guard let moves else {
            model = nil
            mixed = false
            decisionsWithoutGeneration = 0
            return
        }
        decisionsWithoutGeneration = moves.decisionsWithoutGeneration
        var order: [LichessBotModelKey] = []
        var counts: [LichessBotModelKey: Int] = [:]
        var firstFacts: [LichessBotModelKey: LichessBotGenerationFacts] = [:]
        for generation in moves.generations {
            let key = generation.modelKey
            if counts[key] == nil {
                order.append(key)
                firstFacts[key] = generation
            }
            counts[key, default: 0] += generation.ourMoves
        }
        var best: (key: LichessBotModelKey, moves: Int)?
        var used = 0
        for key in order {
            guard let count = counts[key], count > 0 else { continue }
            used += 1
            // `>=`: on a tie, the key that came into use later wins.
            if let current = best, count < current.moves {
                continue
            }
            best = (key, count)
        }
        if let best, let facts = firstFacts[best.key] {
            model = Model(key: best.key, facts: facts)
        } else {
            model = nil
        }
        mixed = used > 1
    }
}

// MARK: - Accumulators

/// Opponent ratings and the score against them, for one ML solve.
private struct PerformanceAccumulator {
    var opponents: [Int] = []
    var score = 0.0

    mutating func add(_ row: LichessBotGameSummary) {
        guard let ourScore = row.ourScore, let rating = row.opponentRating else { return }
        opponents.append(rating)
        score += ourScore
    }

    var estimate: LichessBotRatingEstimate {
        LichessBotEloMath.performanceRating(opponents: opponents, score: score)
    }

    var opponentAverage: Double? {
        opponents.isEmpty ? nil : Double(opponents.reduce(0, +)) / Double(opponents.count)
    }
}

private struct ModelLineAccumulator {
    var tally = LichessBotResultTally()
    var score = 0.0
    var performance = PerformanceAccumulator()
    var mixed = 0
    var first: Date
    var last: Date

    init(createdAt: Date) {
        first = createdAt
        last = createdAt
    }

    mutating func add(_ row: LichessBotGameSummary, ourScore: Double, mixed isMixed: Bool) {
        tally.add(ourScore: ourScore)
        score += ourScore
        performance.add(row)
        if isMixed { mixed += 1 }
        first = min(first, row.createdAt)
        last = max(last, row.createdAt)
    }

    var line: LichessBotModelRecordLine {
        LichessBotModelRecordLine(
            tally: tally,
            interval: LichessBotEloMath.scoreInterval(score: score, games: tally.scored),
            performance: performance.estimate,
            ratedOpponentGames: performance.opponents.count,
            opponentAverage: performance.opponentAverage,
            mixedGames: mixed,
            firstGame: first,
            lastGame: last
        )
    }
}

private struct CalibrationAccumulator {
    var games = 0
    var missing = 0
    var sumPredicted = (win: 0.0, draw: 0.0, loss: 0.0)
    var sumActual = (win: 0.0, draw: 0.0, loss: 0.0)
    var sumBrier = 0.0

    mutating func add(_ checkpoint: LichessBotGameMoveFacts.Checkpoint, ourScore: Double) {
        let predicted = (win: Double(checkpoint.win), draw: Double(checkpoint.draw), loss: Double(checkpoint.loss))
        let actual = (win: ourScore == 1 ? 1.0 : 0, draw: ourScore == 0.5 ? 1.0 : 0, loss: ourScore == 0 ? 1.0 : 0)
        games += 1
        sumPredicted.win += predicted.win
        sumPredicted.draw += predicted.draw
        sumPredicted.loss += predicted.loss
        sumActual.win += actual.win
        sumActual.draw += actual.draw
        sumActual.loss += actual.loss
        let dw = predicted.win - actual.win
        let dd = predicted.draw - actual.draw
        let dl = predicted.loss - actual.loss
        sumBrier += dw * dw + dd * dd + dl * dl
    }

    func row(moveNumber: Int) -> LichessBotCalibrationRow {
        guard games > 0 else {
            return LichessBotCalibrationRow(moveNumber: moveNumber, games: 0, missing: missing, meanPredictedExpected: nil, meanActualScore: nil, meanPredicted: nil, actualFrequencies: nil, brier: nil, skill: nil)
        }
        let n = Double(games)
        let predicted = LichessBotOutcomeTriple(win: sumPredicted.win / n, draw: sumPredicted.draw / n, loss: sumPredicted.loss / n)
        let frequencies = LichessBotOutcomeTriple(win: sumActual.win / n, draw: sumActual.draw / n, loss: sumActual.loss / n)
        let brier = sumBrier / n
        let reference = 1 - (frequencies.win * frequencies.win + frequencies.draw * frequencies.draw + frequencies.loss * frequencies.loss)
        return LichessBotCalibrationRow(
            moveNumber: moveNumber,
            games: games,
            missing: missing,
            meanPredictedExpected: predicted.win + 0.5 * predicted.draw,
            meanActualScore: frequencies.win + 0.5 * frequencies.draw,
            meanPredicted: predicted,
            actualFrequencies: frequencies,
            brier: brier,
            skill: reference > 0 ? 1 - brier / reference : nil
        )
    }
}

private struct DecisiveAccumulator {
    var plies: [Int] = []
    var leads: [Int] = []
    var never = 0
    var noData = 0

    mutating func add(_ decisive: LichessBotDecisivePly, plies totalPlies: Int) {
        switch decisive {
        case .atPly(let ply):
            plies.append(ply)
            leads.append(totalPlies - ply)
        case .never:
            never += 1
        case .noDecisions:
            noData += 1
        case .notApplicable:
            break
        }
    }

    var summary: LichessBotDecisiveSummary {
        LichessBotDecisiveSummary(
            games: plies.count,
            meanPly: Self.mean(plies),
            medianPly: Self.median(plies),
            meanLead: Self.mean(leads),
            never: never,
            noData: noData
        )
    }

    private static func mean(_ values: [Int]) -> Double? {
        values.isEmpty ? nil : Double(values.reduce(0, +)) / Double(values.count)
    }

    private static func median(_ values: [Int]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 1 ? Double(sorted[middle]) : Double(sorted[middle - 1] + sorted[middle]) / 2
    }
}

private struct HeldAccumulator {
    var held = 0
    var turned: [LichessBotHeldGame] = []

    mutating func add(_ row: LichessBotGameSummary, ourScore: Double, startPly: Int?, turnedWhen isTurned: (Double) -> Bool) {
        guard let startPly else { return }
        held += 1
        if isTurned(ourScore) {
            turned.append(LichessBotHeldGame(gameID: row.gameID, createdAt: row.createdAt, opponentName: row.opponentName ?? row.opponentID, ourScore: ourScore, startPly: startPly))
        }
    }

    var summary: LichessBotHeldSummary {
        LichessBotHeldSummary(
            held: held,
            turned: turned.count,
            recentTurned: Array(turned.sorted { lhs, rhs in
                lhs.createdAt != rhs.createdAt ? lhs.createdAt > rhs.createdAt : lhs.gameID < rhs.gameID
            }.prefix(LichessBotHeldSummary.recentLimit))
        )
    }
}

private struct BandAccumulator {
    var games = 0
    var score = 0.0
    var expected = 0.0
}

/// Everything for one filter and one period, filled in one pass.
private struct PeriodAccumulator {
    var performance = PerformanceAccumulator()
    var ratingChange = LichessBotRatingChange()
    var scoredWithoutMoveData = 0

    var speeds: [String: (tally: LichessBotResultTally, performance: PerformanceAccumulator, ratingChange: LichessBotRatingChange)] = [:]

    var groups: [String: ModelLineAccumulator] = [:]
    var checkpoints: [LichessBotModelKey: (facts: LichessBotGenerationFacts, line: ModelLineAccumulator)] = [:]
    var noModelRecorded = LichessBotResultTally()
    var decisionsWithoutGeneration = 0

    var calibration: [Int: CalibrationAccumulator] = [:]
    var buckets: [Int: (positions: Int, sumExpected: Double, sumActual: Double)] = [:]
    var heldWins = HeldAccumulator()
    var heldLosses = HeldAccumulator()
    var decisiveWins = DecisiveAccumulator()
    var decisiveLosses = DecisiveAccumulator()

    var endings: [LichessBotGameEnding: LichessBotEndingRow] = [:]

    var bands: [Int: BandAccumulator] = [:]
    var gaps: [Int] = []
    var gapScore = 0.0
    var gamesWithoutRatings = 0

    mutating func add(_ derived: DerivedRow) {
        let row = derived.row
        var endingRow = endings[derived.ending] ?? LichessBotEndingRow(ending: derived.ending)
        switch row.ourScore {
        case .some(1): endingRow.wins += 1
        case .some(0.5): endingRow.draws += 1
        case .some(0): endingRow.losses += 1
        default: endingRow.unscored += 1
        }
        endings[derived.ending] = endingRow

        var speed = speeds[row.speed] ?? (LichessBotResultTally(), PerformanceAccumulator(), LichessBotRatingChange())
        speed.tally.add(ourScore: row.ourScore)
        speed.performance.add(row)
        speed.ratingChange.add(row)
        speeds[row.speed] = speed

        // Everything below counts scored games only (§3.1).
        guard let ourScore = row.ourScore else { return }
        performance.add(row)
        ratingChange.add(row)
        addModel(derived, ourScore: ourScore)
        addOpponentStrength(row, ourScore: ourScore)
        guard let moves = row.facts?.moves else {
            scoredWithoutMoveData += 1
            return
        }
        addSelfAssessment(row, moves: moves, ourScore: ourScore)
    }

    private mutating func addModel(_ derived: DerivedRow, ourScore: Double) {
        let row = derived.row
        decisionsWithoutGeneration += derived.attribution.decisionsWithoutGeneration
        guard let model = derived.attribution.model else {
            noModelRecorded.add(ourScore: ourScore)
            return
        }
        let mixed = derived.attribution.mixed
        var group = groups[model.facts.modelID] ?? ModelLineAccumulator(createdAt: row.createdAt)
        group.add(row, ourScore: ourScore, mixed: mixed)
        groups[model.facts.modelID] = group
        var checkpoint = checkpoints[model.key] ?? (model.facts, ModelLineAccumulator(createdAt: row.createdAt))
        checkpoint.line.add(row, ourScore: ourScore, mixed: mixed)
        checkpoints[model.key] = checkpoint
    }

    private mutating func addOpponentStrength(_ row: LichessBotGameSummary, ourScore: Double) {
        guard let ours = row.ourRatingBefore, let theirs = row.opponentRating else {
            gamesWithoutRatings += 1
            return
        }
        let gap = theirs - ours
        var band = bands[LichessBotRatingBand.index(gap: gap)] ?? BandAccumulator()
        band.games += 1
        band.score += ourScore
        band.expected += 1 / (1 + pow(10, Double(gap) / 400))
        bands[LichessBotRatingBand.index(gap: gap)] = band
        gaps.append(gap)
        gapScore += ourScore
    }

    private mutating func addSelfAssessment(_ row: LichessBotGameSummary, moves: LichessBotGameMoveFacts, ourScore: Double) {
        for checkpoint in moves.checkpoints {
            calibration[checkpoint.moveNumber, default: CalibrationAccumulator()].add(checkpoint, ourScore: ourScore)
        }
        for moveNumber in moves.checkpointsWithoutDecision {
            calibration[moveNumber, default: CalibrationAccumulator()].missing += 1
        }
        for bucket in moves.expectedScoreBuckets {
            var total = buckets[bucket.index] ?? (0, 0, 0)
            total.positions += bucket.positions
            total.sumExpected += bucket.sumExpected
            total.sumActual += Double(bucket.positions) * ourScore
            buckets[bucket.index] = total
        }
        heldWins.add(row, ourScore: ourScore, startPly: moves.heldWinStartPly) { $0 < 1 }
        heldLosses.add(row, ourScore: ourScore, startPly: moves.heldLossStartPly) { $0 > 0 }
        switch ourScore {
        case 1: decisiveWins.add(moves.decisive, plies: row.plies)
        case 0: decisiveLosses.add(moves.decisive, plies: row.plies)
        default: break
        }
    }

    func periodStatistics(record: LichessBotPeriodRecord) -> LichessBotPeriodStatistics {
        LichessBotPeriodStatistics(
            record: record,
            performance: performance.estimate,
            ratedOpponentGames: performance.opponents.count,
            opponentAverage: performance.opponentAverage,
            ratingChange: ratingChange
        )
    }

    func breakdowns() -> LichessBotPeriodBreakdowns {
        LichessBotPeriodBreakdowns(
            timeControls: speeds.reduce(into: [:]) { result, entry in
                result[entry.key] = LichessBotTimeControlStatistics(
                    speed: entry.key,
                    tally: entry.value.tally,
                    performance: entry.value.performance.estimate,
                    ratedOpponentGames: entry.value.performance.opponents.count,
                    ratingChange: entry.value.ratingChange
                )
            },
            models: modelStatistics(),
            selfAssessment: selfAssessment(),
            endings: endingStatistics(),
            opponentStrength: opponentStrength()
        )
    }

    private func modelStatistics() -> LichessBotModelStatistics {
        var checkpointsByModel: [String: [LichessBotModelCheckpointStatistics]] = [:]
        for (key, entry) in checkpoints {
            checkpointsByModel[entry.facts.modelID, default: []].append(LichessBotModelCheckpointStatistics(
                key: key,
                sourceKind: entry.facts.sourceKind,
                modelID: entry.facts.modelID,
                trainingStep: entry.facts.trainingStep,
                lineageRunID: entry.facts.lineageRunID,
                cumTrainerStep: entry.facts.cumTrainerStep,
                line: entry.line.line
            ))
        }
        // A group and its checkpoints are filled by the same games
        // (`addModel`), so every group has at least one checkpoint here.
        var groupList: [LichessBotModelGroupStatistics] = []
        for (modelID, accumulator) in groups {
            let groupCheckpoints: [LichessBotModelCheckpointStatistics] = checkpointsByModel[modelID, default: []]
            groupList.append(LichessBotModelGroupStatistics(
                modelID: modelID,
                line: accumulator.line,
                checkpoints: groupCheckpoints.sorted(by: Self.checkpointOrder)
            ))
        }
        groupList.sort { (lhs: LichessBotModelGroupStatistics, rhs: LichessBotModelGroupStatistics) -> Bool in
            if lhs.line.lastGame != rhs.line.lastGame {
                return lhs.line.lastGame > rhs.line.lastGame
            }
            return lhs.modelID < rhs.modelID
        }
        return LichessBotModelStatistics(
            groups: groupList,
            noModelRecorded: noModelRecorded,
            decisionsWithoutGeneration: decisionsWithoutGeneration,
            progression: Self.progression(checkpoints.values.map { ($0.facts, $0.line) })
        )
    }

    /// Training step ascending; a checkpoint without one (a champion
    /// snapshot) after those with one, by first game.
    private static func checkpointOrder(_ lhs: LichessBotModelCheckpointStatistics, _ rhs: LichessBotModelCheckpointStatistics) -> Bool {
        switch (lhs.trainingStep, rhs.trainingStep) {
        case let (left?, right?) where left != right: return left < right
        case (.some, .none): return true
        case (.none, .some): return false
        default: return lhs.line.firstGame < rhs.line.firstGame
        }
    }

    /// The progression chart's points (OD-12). A run is a lineage run when
    /// recorded, else a model ID (one CLI process is one segment). Its
    /// checkpoints are placed by cumulative step when recorded, else by
    /// training step, and consecutive ones are merged until a point holds
    /// `minimumGamesPerPoint` scored games; a trailing remainder stays its
    /// own point. Checkpoints with no step at all can't be placed and are
    /// left out of the chart (they stay in the table). Runs with fewer than
    /// two points are left out.
    private static func progression(_ checkpoints: [(facts: LichessBotGenerationFacts, line: ModelLineAccumulator)]) -> [LichessBotProgressionPoint] {
        typealias Entry = (step: Int, cumulative: Bool, line: ModelLineAccumulator)
        var bySeries: [String: [Entry]] = [:]
        for checkpoint in checkpoints {
            let step: Int
            let cumulative: Bool
            if let cum = checkpoint.facts.cumTrainerStep {
                step = cum
                cumulative = true
            } else if let trainingStep = checkpoint.facts.trainingStep {
                step = trainingStep
                cumulative = false
            } else {
                continue
            }
            bySeries[checkpoint.facts.lineageRunID ?? checkpoint.facts.modelID, default: []].append((step, cumulative, checkpoint.line))
        }
        var points: [LichessBotProgressionPoint] = []
        for (series, unordered) in bySeries.sorted(by: { $0.key < $1.key }) {
            let entries = unordered.sorted { $0.step < $1.step }
            var bins: [[Entry]] = []
            var pending: [Entry] = []
            var pendingGames = 0
            for entry in entries {
                pending.append(entry)
                pendingGames += entry.line.tally.scored
                if pendingGames >= LichessBotModelStatistics.minimumGamesPerPoint {
                    bins.append(pending)
                    pending = []
                    pendingGames = 0
                }
            }
            if !pending.isEmpty {
                bins.append(pending)
            }
            guard bins.count >= 2 else { continue }
            for bin in bins {
                guard let first = bin.first, let last = bin.last else { continue }
                var tally = LichessBotResultTally()
                var performance = PerformanceAccumulator()
                for entry in bin {
                    tally.wins += entry.line.tally.wins
                    tally.draws += entry.line.tally.draws
                    tally.losses += entry.line.tally.losses
                    performance.opponents += entry.line.performance.opponents
                    performance.score += entry.line.performance.score
                }
                let score = Double(tally.wins) + 0.5 * Double(tally.draws)
                points.append(LichessBotProgressionPoint(
                    series: series,
                    step: last.step,
                    stepIsCumulative: last.cumulative,
                    firstStep: first.step,
                    checkpointCount: bin.count,
                    games: tally.scored,
                    score: tally.score,
                    interval: LichessBotEloMath.scoreInterval(score: score, games: tally.scored),
                    performance: performance.estimate
                ))
            }
        }
        return points
    }

    private func selfAssessment() -> LichessBotSelfAssessmentStatistics {
        LichessBotSelfAssessmentStatistics(
            calibration: LichessBotSelfAssessmentDefinition.checkpointMoveNumbers.map { moveNumber in
                (calibration[moveNumber] ?? CalibrationAccumulator()).row(moveNumber: moveNumber)
            },
            reliability: buckets.keys.sorted().compactMap { index in
                buckets[index].map { bucket in
                    LichessBotReliabilityBucket(
                        index: index,
                        positions: bucket.positions,
                        meanPredicted: bucket.sumExpected / Double(bucket.positions),
                        meanActual: bucket.sumActual / Double(bucket.positions)
                    )
                }
            },
            heldWins: heldWins.summary,
            heldLosses: heldLosses.summary,
            decisiveWins: decisiveWins.summary,
            decisiveLosses: decisiveLosses.summary,
            gamesWithoutMoveData: scoredWithoutMoveData
        )
    }

    private func endingStatistics() -> LichessBotEndingStatistics {
        let rows = endings.values.sorted { $0.ending < $1.ending }
        return LichessBotEndingStatistics(
            rows: rows,
            wins: rows.reduce(0) { $0 + $1.wins },
            draws: rows.reduce(0) { $0 + $1.draws },
            losses: rows.reduce(0) { $0 + $1.losses }
        )
    }

    private func opponentStrength() -> LichessBotOpponentStrengthStatistics {
        LichessBotOpponentStrengthStatistics(
            bands: bands.keys.sorted().compactMap { index in
                bands[index].map { band in
                    LichessBotRatingBand(
                        index: index,
                        games: band.games,
                        actualScore: band.score / Double(band.games),
                        meanExpected: band.expected / Double(band.games),
                        interval: LichessBotEloMath.scoreInterval(score: band.score, games: band.games)
                    )
                }
            },
            games: gaps.count,
            gamesWithoutRatings: gamesWithoutRatings,
            fiftyPercentPoint: LichessBotEloMath.ratingOffset(gaps: gaps, score: gapScore)
        )
    }
}
