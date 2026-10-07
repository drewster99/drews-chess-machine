import Foundation

/// The sum of Lichess's per-game rating changes over rated games
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.3, OD-8). Never inferred from
/// consecutive starting ratings: the bot plays several games at once, so a
/// later game's starting rating can predate an earlier game's result. A
/// rated game without a recorded change is counted, not read as zero.
struct LichessBotRatingChange: Sendable, Equatable {
    /// Scored rated games. An aborted rated game changes no rating, so it
    /// is not a game "missing" its change.
    var ratedGames = 0
    /// Those with a recorded `ourRatingDiff`.
    var gamesWithChange = 0
    /// Σ `ourRatingDiff` over `gamesWithChange`.
    var total = 0

    var gamesWithoutChange: Int { ratedGames - gamesWithChange }

    mutating func add(_ row: LichessBotGameSummary) {
        guard row.rated, row.ourScore != nil else { return }
        ratedGames += 1
        if let diff = row.ourRatingDiff {
            gamesWithChange += 1
            total += diff
        }
    }
}

/// Item 1: one row of the period table.
struct LichessBotPeriodStatistics: Sendable, Equatable {
    /// W–D–L and the opponent-kind and color splits, from
    /// `LichessBotPeriodRecord.add` (their one definition).
    let record: LichessBotPeriodRecord
    /// Over scored games whose opponent has a rating (§3.4).
    let performance: LichessBotRatingEstimate
    /// How many games `performance` and `opponentAverage` cover: Lichess AI
    /// games have no rating and are left out of those two only.
    let ratedOpponentGames: Int
    let opponentAverage: Double?
    let ratingChange: LichessBotRatingChange
}

/// Item 2: one speed over one period.
struct LichessBotTimeControlStatistics: Sendable, Equatable {
    let speed: String
    let tally: LichessBotResultTally
    let performance: LichessBotRatingEstimate
    let ratedOpponentGames: Int
    let ratingChange: LichessBotRatingChange
}

/// The Time controls pane's rows (§3.5).
enum LichessBotTimeControlOrder {
    /// Lichess's speeds, fastest first. Only these come from the account:
    /// `account.perfs` also holds puzzle and variant pools, which are not
    /// time controls.
    static let knownSpeeds = ["ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence"]

    /// The rows: each known speed the account has a rating in or a game was
    /// played at, in Lichess's order, then any other speed a game was
    /// played at, verbatim and alphabetically.
    static func speeds(recordSpeeds: Set<String>, accountSpeeds: Set<String>) -> [String] {
        let known = knownSpeeds.filter { recordSpeeds.contains($0) || accountSpeeds.contains($0) }
        let other = recordSpeeds.subtracting(knownSpeeds).sorted()
        return known + other
    }
}

/// The Time controls pane's rating sparkline (OD-9): the speed's recorded
/// starting ratings, oldest first, then the account's current rating as the
/// final point when the account is loaded. Every point is a rating Lichess
/// reported; nothing is interpolated.
struct LichessBotRatingSparklineSeries: Sendable, Equatable {
    let ratings: [Int]
    let firstDate: Date?
    let lastDate: Date?
    let endsWithCurrentRating: Bool

    init(points: [LichessBotRatingPoint], currentRating: Int?) {
        ratings = points.map(\.rating) + (currentRating.map { [$0] } ?? [])
        firstDate = points.first?.createdAt
        lastDate = points.last?.createdAt
        endsWithCurrentRating = currentRating != nil
    }

    /// The ratings' range when there is a line to draw (two points or
    /// more); nil otherwise, and the sparkline draws nothing.
    var range: ClosedRange<Int>? {
        guard ratings.count >= 2, let low = ratings.min(), let high = ratings.max() else { return nil }
        return low...high
    }

    /// `range` as a list of zero or one item with a constant identity, so
    /// the sparkline shows its chart with `ForEach` (no `if` in the body)
    /// and keeps it across updates.
    var drawableRanges: [LichessBotRatingSparklineRange] {
        range.map { [LichessBotRatingSparklineRange(range: $0)] } ?? []
    }

    /// "1402–1561 over 87 rated games, 2026-09-28 to 2026-10-06; last point
    /// is the current rating".
    var help: String {
        guard let low = ratings.min(), let high = ratings.max() else {
            return "No rated game with a starting rating"
        }
        let games = ratings.count - (endsWithCurrentRating ? 1 : 0)
        var text = "\(low)–\(high) over \(games) rated game\(games == 1 ? "" : "s")"
        if let firstDate, let lastDate {
            text += ", \(firstDate.formatted(date: .abbreviated, time: .omitted)) to \(lastDate.formatted(date: .abbreviated, time: .omitted))"
        }
        if endsWithCurrentRating {
            text += "; the last point is the current rating"
        }
        return text
    }
}

/// The range a sparkline with a line to draw spans, identified by a
/// constant: there is only ever one.
struct LichessBotRatingSparklineRange: Identifiable, Equatable {
    let range: ClosedRange<Int>
    var id: Int { 0 }
}

/// A starting rating Lichess reported for one rated game, for the
/// sparkline (OD-9). Nothing is interpolated.
struct LichessBotRatingPoint: Sendable, Equatable {
    let gameID: String
    let createdAt: Date
    let rating: Int
}

/// Item 4: one line of the Models table — a model ID's group, or one
/// checkpoint (model key) inside it.
struct LichessBotModelRecordLine: Sendable, Equatable {
    let tally: LichessBotResultTally
    let interval: LichessBotScoreInterval?
    let performance: LichessBotRatingEstimate
    let ratedOpponentGames: Int
    let opponentAverage: Double?
    /// Games played by more than one model (a mid-game switch), attributed
    /// here by majority.
    let mixedGames: Int
    let firstGame: Date
    let lastGame: Date
}

/// Item 4: one checkpoint (model key) and its record.
struct LichessBotModelCheckpointStatistics: Sendable, Equatable, Identifiable {
    let key: LichessBotModelKey
    let sourceKind: LichessBotModelSourceKind
    let modelID: String
    let trainingStep: Int?
    let lineageRunID: String?
    let cumTrainerStep: Int?
    let line: LichessBotModelRecordLine

    var id: LichessBotModelKey { key }

    /// "step 5,000 · cum 120,000 · Model file · ab12cd34": the training
    /// step, the lineage's cumulative step when recorded, the source, and a
    /// file's hash prefix.
    var label: String {
        var parts: [String] = []
        if let trainingStep {
            parts.append("step \(trainingStep.formatted())")
        }
        if let cumTrainerStep {
            parts.append("cum \(cumTrainerStep.formatted())")
        }
        parts.append(sourceKind.displayName)
        if case .file(let sha256) = key {
            parts.append(String(sha256.prefix(8)))
        }
        return parts.joined(separator: " · ")
    }
}

/// Item 4: one model ID (one CLI process, one lineage segment) and its
/// checkpoints, ordered by training step.
struct LichessBotModelGroupStatistics: Sendable, Equatable, Identifiable {
    let modelID: String
    let line: LichessBotModelRecordLine
    let checkpoints: [LichessBotModelCheckpointStatistics]

    var id: String { modelID }
}

/// One visible row of the Models table: a model ID's group, one of its
/// checkpoints (when the group is expanded), or the "No model recorded"
/// row, in one shape so the table draws every row with one view (no
/// `switch` in a view body). The table is a list of these, so expanding a
/// group inserts rows rather than showing hidden ones.
struct LichessBotModelTableRow: Sendable, Equatable, Identifiable {
    enum Kind: Sendable, Equatable {
        /// A model ID's group; `expanded` says whether its checkpoints show.
        case group(expanded: Bool)
        case checkpoint
        case noModelRecorded
    }

    let id: String
    let kind: Kind
    /// The model ID a group row opens and closes; nil for other rows.
    let groupModelID: String?
    let label: String
    let tally: LichessBotResultTally
    let interval: LichessBotScoreInterval?
    let performance: LichessBotRatingEstimate
    let ratedOpponentGames: Int
    let opponentAverage: Double?
    /// Nil for the "No model recorded" row, which has no model to mix with.
    let mixedGames: Int?
    let firstGame: Date?
    let lastGame: Date?

    /// The rows for `models` with the groups in `expanded` opened. The
    /// "No model recorded" row comes last, and only when it has games.
    static func rows(_ models: LichessBotModelStatistics, expanded: Set<String>) -> [LichessBotModelTableRow] {
        var rows: [LichessBotModelTableRow] = []
        for group in models.groups {
            let isExpanded = expanded.contains(group.modelID)
            rows.append(LichessBotModelTableRow(id: "group:\(group.modelID)", kind: .group(expanded: isExpanded), groupModelID: group.modelID, label: group.modelID, line: group.line))
            if isExpanded {
                rows += group.checkpoints.map { checkpoint in
                    LichessBotModelTableRow(id: "checkpoint:\(checkpoint.key)", kind: .checkpoint, groupModelID: nil, label: checkpoint.label, line: checkpoint.line)
                }
            }
        }
        let none = models.noModelRecorded
        if none.games > 0 {
            rows.append(LichessBotModelTableRow(
                id: "noModelRecorded",
                kind: .noModelRecorded,
                groupModelID: nil,
                label: "No model recorded",
                tally: none,
                interval: LichessBotEloMath.scoreInterval(score: Double(none.wins) + 0.5 * Double(none.draws), games: none.scored),
                performance: .none,
                ratedOpponentGames: 0,
                opponentAverage: nil,
                mixedGames: nil,
                firstGame: nil,
                lastGame: nil
            ))
        }
        return rows
    }

    private init(id: String, kind: Kind, groupModelID: String?, label: String, line: LichessBotModelRecordLine) {
        self.init(
            id: id, kind: kind, groupModelID: groupModelID, label: label,
            tally: line.tally, interval: line.interval, performance: line.performance,
            ratedOpponentGames: line.ratedOpponentGames, opponentAverage: line.opponentAverage,
            mixedGames: line.mixedGames, firstGame: line.firstGame, lastGame: line.lastGame
        )
    }

    private init(id: String, kind: Kind, groupModelID: String?, label: String, tally: LichessBotResultTally, interval: LichessBotScoreInterval?, performance: LichessBotRatingEstimate, ratedOpponentGames: Int, opponentAverage: Double?, mixedGames: Int?, firstGame: Date?, lastGame: Date?) {
        self.id = id
        self.kind = kind
        self.groupModelID = groupModelID
        self.label = label
        self.tally = tally
        self.interval = interval
        self.performance = performance
        self.ratedOpponentGames = ratedOpponentGames
        self.opponentAverage = opponentAverage
        self.mixedGames = mixedGames
        self.firstGame = firstGame
        self.lastGame = lastGame
    }
}

/// One point of the progression chart: consecutive checkpoints of one run
/// merged until they hold enough scored games (OD-12).
struct LichessBotProgressionPoint: Sendable, Equatable, Identifiable {
    /// The series: the lineage run ID for cumulative steps; for
    /// segment-local steps the run's segment, or the model ID without a
    /// lineage. Never both kinds of step in one series.
    let series: String
    /// The point's place in its run, from 0. Part of the ID: two bins of a
    /// run can end at the same step (checkpoints that share a step).
    let ordinal: Int
    /// The last merged checkpoint's lineage cumulative step when recorded,
    /// else its training step.
    let step: Int
    let stepIsCumulative: Bool
    let firstStep: Int
    let checkpointCount: Int
    let games: Int
    let score: Double?
    let interval: LichessBotScoreInterval?
    let performance: LichessBotRatingEstimate

    var id: String { "\(series)#\(ordinal)" }

    /// The x axis's label for `points`: each series is one kind of step,
    /// and the label names the kinds on the chart.
    static func stepAxisLabel(for points: [LichessBotProgressionPoint]) -> String {
        let cumulative = points.contains { $0.stepIsCumulative }
        let segmentLocal = points.contains { !$0.stepIsCumulative }
        switch (cumulative, segmentLocal) {
        case (true, true): return "Cumulative trainer step, or segment step for runs without one"
        case (true, false): return "Cumulative trainer step"
        case (false, _): return "Training step"
        }
    }
}

/// What the progression chart plots on its y axis.
enum LichessBotProgressMetric: String, CaseIterable, Sendable, Identifiable {
    case score
    case performance

    var id: String { rawValue }

    var label: String {
        switch self {
        case .score: return "Score"
        case .performance: return "Perf"
        }
    }
}

/// One mark of the progression chart: a point and, for the score, its
/// interval.
struct LichessBotProgressChartMark: Sendable, Equatable, Identifiable {
    let id: String
    let series: String
    let step: Int
    let value: Double
    let lower: Double?
    let upper: Double?

    /// The marks for `metric`. Score is in percent with its Wilson interval.
    /// A performance rating is plotted only where it is an estimate: a
    /// perfect or zero score is a bound with no value to place, and is left
    /// out (the table shows it).
    static func marks(_ points: [LichessBotProgressionPoint], metric: LichessBotProgressMetric) -> [LichessBotProgressChartMark] {
        points.compactMap { point in
            switch metric {
            case .score:
                guard let score = point.score else { return nil }
                return LichessBotProgressChartMark(
                    id: point.id, series: point.series, step: point.step, value: 100 * score,
                    lower: point.interval.map { 100 * $0.lower }, upper: point.interval.map { 100 * $0.upper }
                )
            case .performance:
                guard case .estimate(let rating) = point.performance else { return nil }
                return LichessBotProgressChartMark(id: point.id, series: point.series, step: point.step, value: rating, lower: nil, upper: nil)
            }
        }
    }
}

/// Item 4 over one period.
struct LichessBotModelStatistics: Sendable, Equatable {
    /// Newest last game first.
    let groups: [LichessBotModelGroupStatistics]
    /// Scored games in which no DCM move names a generation (DCM never
    /// moved, or only export-known moves): their own row, never dropped and
    /// never given to a neighbor.
    let noModelRecorded: LichessBotResultTally
    /// DCM decisions whose generation reference names no listed generation.
    let decisionsWithoutGeneration: Int
    /// Runs with at least two points; the chart is left out with none.
    let progression: [LichessBotProgressionPoint]

    /// Scored games a point needs before it stands alone (OD-12).
    static let minimumGamesPerPoint = 30
}

/// Item 6: the value head's prediction against the result at DCM's Nth
/// move (survivors only: games that ended earlier are not in the row).
struct LichessBotCalibrationRow: Sendable, Equatable {
    let moveNumber: Int
    /// Scored games with a decision at the move.
    let games: Int
    /// Scored games that reached the move without a decision there.
    let missing: Int
    let meanPredictedExpected: Double?
    let meanActualScore: Double?
    let meanPredicted: LichessBotOutcomeTriple?
    let actualFrequencies: LichessBotOutcomeTriple?
    /// Mean 3-class Brier score, 0 (perfect) … 2.
    let brier: Double?
    /// `1 − Brier / Brier_ref`, where `Brier_ref = 1 − Σ fₖ²` scores always
    /// predicting the same games' own result frequencies. Nil with no games
    /// or when every game had the same result (`Brier_ref = 0`).
    let skill: Double?
}

/// Win / draw / loss values (probabilities or frequencies).
struct LichessBotOutcomeTriple: Sendable, Equatable {
    let win: Double
    let draw: Double
    let loss: Double
}

/// Item 6: one expected-score bucket over every DCM decision.
struct LichessBotReliabilityBucket: Sendable, Equatable, Identifiable {
    let index: Int
    let positions: Int
    let meanPredicted: Double
    let meanActual: Double

    var id: Int { index }
}

/// A game in which DCM held a win (or a loss) in its own eyes.
struct LichessBotHeldGame: Sendable, Equatable, Identifiable {
    let gameID: String
    let createdAt: Date
    /// Nil for Lichess's AI, which has no account.
    let opponentName: String?
    let ourScore: Double
    /// The ply of the first move of the held run.
    let startPly: Int

    var id: String { gameID }
}

/// Item 6: held wins and how many were not won, or held losses and how
/// many were not lost.
struct LichessBotHeldSummary: Sendable, Equatable {
    let held: Int
    /// Blown wins (held win, drawn or lost) or saves (held loss, drawn or
    /// won).
    let turned: Int
    /// The most recent turned games, newest first.
    let recentTurned: [LichessBotHeldGame]

    static let recentLimit = 20
}

/// Item 6: when won (or lost) games became settled in the network's eyes.
struct LichessBotDecisiveSummary: Sendable, Equatable {
    /// Games with a decisive ply.
    let games: Int
    let meanPly: Double?
    let medianPly: Double?
    /// Mean `plies − decisive ply`.
    let meanLead: Double?
    let never: Int
    /// Won or lost games with no DCM decision at all.
    let noData: Int
}

/// Item 6 over one period.
struct LichessBotSelfAssessmentStatistics: Sendable, Equatable {
    let calibration: [LichessBotCalibrationRow]
    /// Non-empty buckets.
    let reliability: [LichessBotReliabilityBucket]
    let heldWins: LichessBotHeldSummary
    let heldLosses: LichessBotHeldSummary
    let decisiveWins: LichessBotDecisiveSummary
    let decisiveLosses: LichessBotDecisiveSummary
    /// Scored games without per-move facts.
    let gamesWithoutMoveData: Int
}

/// Item 5: one ending and DCM's results in it.
struct LichessBotEndingRow: Sendable, Equatable, Identifiable {
    let ending: LichessBotGameEnding
    var wins = 0
    var draws = 0
    var losses = 0
    /// Only in the "Not counted" row.
    var unscored = 0

    var id: LichessBotGameEnding { ending }
}

/// Item 5 over one period.
struct LichessBotEndingStatistics: Sendable, Equatable {
    /// Endings with at least one game, in display order.
    let rows: [LichessBotEndingRow]
    let wins: Int
    let draws: Int
    let losses: Int
}

/// Item 3: games against opponents in one 100-point band of the rating gap
/// (opponent minus DCM, at the game's start, within its own speed).
struct LichessBotRatingBand: Sendable, Equatable, Identifiable {
    /// `floor(gap / 100)`, clamped to −5…4.
    let index: Int
    let games: Int
    let actualScore: Double
    let meanExpected: Double
    /// Wilson 95% interval of the actual score (`LichessBotEloMath.scoreInterval`).
    let interval: LichessBotScoreInterval?

    var id: Int { index }

    static let lowestIndex = -5
    static let highestIndex = 4

    /// The band of a gap. A floored division: Swift's `/` truncates toward
    /// zero and would put −1…−99 in band 0.
    static func index(gap: Int) -> Int {
        min(highestIndex, max(lowestIndex, Int((Double(gap) / 100).rounded(.down))))
    }

    /// "< −400", "−400…−301", …, "0…99", …, "≥ 400".
    static func label(index: Int) -> String {
        if index <= lowestIndex {
            return "< \(LichessBotStatsFormat.signed(lowestIndex * 100 + 100))"
        }
        if index >= highestIndex {
            return "≥ \(highestIndex * 100)"
        }
        let low = index * 100
        let high = low + 99
        if index < 0 {
            return "\(LichessBotStatsFormat.signed(low))…\(LichessBotStatsFormat.signed(high))"
        }
        return "\(low)…\(high)"
    }
}

/// Item 3 over one period.
struct LichessBotOpponentStrengthStatistics: Sendable, Equatable {
    /// Non-empty bands, lowest gap first.
    let bands: [LichessBotRatingBand]
    /// Scored games with both ratings.
    let games: Int
    /// Scored games left out for a missing rating (ours or theirs).
    let gamesWithoutRatings: Int
    /// The gap at which DCM scores 50% (§3.7).
    let fiftyPercentPoint: LichessBotRatingEstimate
}

/// Everything the panel's panes show for one period and filter.
struct LichessBotPeriodBreakdowns: Sendable, Equatable {
    /// By speed.
    let timeControls: [String: LichessBotTimeControlStatistics]
    let models: LichessBotModelStatistics
    let selfAssessment: LichessBotSelfAssessmentStatistics
    let endings: LichessBotEndingStatistics
    let opponentStrength: LichessBotOpponentStrengthStatistics
    /// The later panes (§11, L1–L6).
    let later: LichessBotLaterBreakdowns
}
