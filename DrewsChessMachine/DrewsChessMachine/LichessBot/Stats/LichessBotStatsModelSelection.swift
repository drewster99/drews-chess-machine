import Foundation

/// Which games the Record card counts by the model that played them: every
/// game, or only those attributed to one model (`LichessBotModelAttribution`:
/// the model that chose most of DCM's moves). Applied to the rows before the
/// statistics are computed, so every pane and the period table follow it.
/// Codable: the selection is remembered in the defaults.
enum LichessBotStatsModelSelection: Sendable, Hashable, Codable {
    case all
    case model(LichessBotModelKey)

    /// For a log line computed under this selection: nothing for every
    /// model, else " model=<id>@step<n>" or " model=file:<hash prefix>".
    var logSuffix: String {
        switch self {
        case .all:
            return ""
        case .model(.file(let sha256)):
            return " model=file:\(sha256.prefix(8))"
        case .model(.snapshot(let sourceKind, let modelID, let trainingStep)):
            return " model=\(modelID)@step\(trainingStep.map(String.init) ?? "-")(\(sourceKind.rawValue))"
        }
    }

    /// Whether `row` counts. A game with no recorded model counts only for
    /// `.all`.
    func includes(_ row: LichessBotGameSummary) -> Bool {
        switch self {
        case .all:
            return true
        case .model(let key):
            return LichessBotModelAttribution(row.facts?.moves).model?.key == key
        }
    }
}

/// One model the Model filter offers: every model some game is attributed
/// to, with what tells it apart.
struct LichessBotStatsModelChoice: Sendable, Equatable, Identifiable {
    let key: LichessBotModelKey
    let modelID: String
    /// The checkpoint label the Models pane uses (step, cumulative step,
    /// source, file hash prefix).
    let checkpointLabel: String
    let games: Int
    let lastGame: Date

    var id: LichessBotModelKey { key }

    /// "20261006-43-a89C · step 5,000 · Model file · ab12cd34 (95 games)".
    var menuLabel: String {
        "\(modelID) · \(checkpointLabel) (\(games) game\(games == 1 ? "" : "s"))"
    }

    /// Every attributed model in `rows`, most recently played first (ties:
    /// by model ID, then label, so the order is stable).
    static func choices(from rows: [LichessBotGameSummary]) -> [LichessBotStatsModelChoice] {
        var byKey: [LichessBotModelKey: (facts: LichessBotGenerationFacts, games: Int, last: Date)] = [:]
        for row in rows {
            guard let model = LichessBotModelAttribution(row.facts?.moves).model else { continue }
            if var entry = byKey[model.key] {
                entry.games += 1
                entry.last = max(entry.last, row.createdAt)
                byKey[model.key] = entry
            } else {
                byKey[model.key] = (model.facts, 1, row.createdAt)
            }
        }
        return byKey.map { key, entry in
            LichessBotStatsModelChoice(
                key: key,
                modelID: entry.facts.modelID,
                checkpointLabel: LichessBotModelCheckpointStatistics.label(
                    key: key, sourceKind: entry.facts.sourceKind,
                    trainingStep: entry.facts.trainingStep, cumTrainerStep: entry.facts.cumTrainerStep),
                games: entry.games,
                lastGame: entry.last)
        }.sorted { lhs, rhs in
            if lhs.lastGame != rhs.lastGame { return lhs.lastGame > rhs.lastGame }
            if lhs.modelID != rhs.modelID { return lhs.modelID < rhs.modelID }
            return lhs.checkpointLabel < rhs.checkpointLabel
        }
    }
}

/// The Model filter's menu for each Games filter and period: the models
/// that played games the panes show, with those games' counts, so the menu
/// matches the Models pane (the selected period and Rated / Casual / All).
/// Built with the statistics, so changing the period or the Games filter is
/// a lookup.
struct LichessBotStatsModelMenu: Sendable {
    private let byFilter: LichessBotFilterValues<LichessBotPeriodValues<[LichessBotStatsModelChoice]>>

    /// Before the first result: no models.
    static let empty = LichessBotStatsModelMenu(byFilter: LichessBotFilterValues { _ in LichessBotPeriodValues { _ in [] } })

    static func make(rows: [LichessBotGameSummary], now: Date, calendar: Calendar) throws -> LichessBotStatsModelMenu {
        let starts = try LichessBotStatsPeriods.starts(now: now, calendar: calendar)
        return LichessBotStatsModelMenu(byFilter: LichessBotFilterValues { filter in
            let inFilter = rows.filter { filter.includes($0) }
            return LichessBotPeriodValues { period in
                LichessBotStatsModelChoice.choices(from: inFilter.filter { starts.contains($0.createdAt, in: period) })
            }
        })
    }

    func choices(filter: LichessBotStatsFilter, period: LichessBotStatsPeriod) -> [LichessBotStatsModelChoice] {
        byFilter[filter][period]
    }

    /// Every model any game is attributed to (all games, all time).
    var everyModel: [LichessBotStatsModelChoice] {
        byFilter[.all][.allTime]
    }
}
