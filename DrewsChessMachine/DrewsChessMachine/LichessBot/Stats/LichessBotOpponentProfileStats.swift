import Foundation

/// One line of a per-opponent split (by time control, or by the model that
/// played): a label, DCM's record over those games, and Lichess's net
/// rating change across their rated games.
struct LichessBotOpponentSplitRow: Sendable, Equatable, Identifiable {
    let id: String
    let label: String
    let tally: LichessBotResultTally
    let ratingChange: LichessBotRatingChange
    /// Most recent game in the group, for ordering.
    let lastPlayedAt: Date
}

/// A point on a per-opponent trend line: a value Lichess reported (our or
/// their starting rating), or a running score, for one game against the
/// opponent. Nothing is interpolated; games with no recorded value are left
/// out of the series they'd belong to.
struct LichessBotOpponentTrendPoint: Sendable, Equatable, Identifiable {
    let gameID: String
    let createdAt: Date
    /// 0-based position in the opponent's game order (oldest = 0), so a flat
    /// series still has an x axis.
    let index: Int
    let value: Double

    var id: String { gameID }
}

/// Everything the per-opponent detail window shows about one opponent,
/// computed once from DCM's own filed games against them (newest first) and
/// the shared head-to-head fold. Pure, so the window never reads a record
/// file and the aggregation is testable.
///
/// "Vs us" is DCM's record; a game's model is the generation that chose most
/// of DCM's moves (`LichessBotModelAttribution`), matching the Model filter.
struct LichessBotOpponentProfileStats: Sendable, Equatable {
    /// Lowercased Lichess id.
    let opponentID: String
    /// The best display name any game recorded, else the id.
    let name: String
    let kind: LichessBotOpponentKind
    let title: String?

    /// Every filed game against them, newest first (what the history table
    /// shows).
    let games: [LichessBotGameSummary]
    /// DCM's overall record and net rating change against them.
    let headToHead: LichessBotHeadToHead

    let firstPlayedAt: Date?
    let lastPlayedAt: Date?

    /// The run of one result the most recent scored games form; nil with
    /// none scored.
    let currentStreak: LichessBotStreak?
    let longestWinStreak: Int
    let longestLossStreak: Int
    /// The highest-rated opponent rating in a game DCM won; nil with no win
    /// that recorded their rating.
    let highestRatedWin: LichessBotNotableWin?

    /// DCM's record per time control, most games first.
    let bySpeed: [LichessBotOpponentSplitRow]
    /// DCM's record per model that played them, most recently played first.
    let byModel: [LichessBotOpponentSplitRow]

    /// Our starting rating over the games, oldest first (games that recorded
    /// one).
    let ourRatingTrend: [LichessBotOpponentTrendPoint]
    /// Their starting rating over the games, oldest first.
    let theirRatingTrend: [LichessBotOpponentTrendPoint]
    /// DCM's running score fraction after each scored game, oldest first.
    let scoreTrend: [LichessBotOpponentTrendPoint]

    /// The opponent's id lowercased; nil games (no opponent id) can't be
    /// grouped, so the caller passes only rows against one id.
    init(opponentID: String, rows: [LichessBotGameSummary]) {
        self.opponentID = opponentID.lowercased()

        // Newest first for display; oldest first for streaks and trends.
        let newestFirst = rows.sorted { $0.createdAt > $1.createdAt }
        let oldestFirst = Array(newestFirst.reversed())
        games = newestFirst

        name = newestFirst.compactMap { $0.opponentName }.first ?? self.opponentID
        kind = newestFirst.first?.opponentKind ?? .human
        title = newestFirst.compactMap { $0.opponentTitle }.first

        var head = LichessBotHeadToHead()
        for row in newestFirst {
            head.add(row)
        }
        headToHead = head

        firstPlayedAt = oldestFirst.first?.createdAt
        lastPlayedAt = newestFirst.first?.createdAt

        currentStreak = Self.currentStreak(newestFirst: newestFirst)
        (longestWinStreak, longestLossStreak) = Self.longestStreaks(oldestFirst: oldestFirst)
        highestRatedWin = Self.highestRatedWin(newestFirst: newestFirst)

        bySpeed = Self.splitBySpeed(newestFirst: newestFirst)
        byModel = Self.splitByModel(rows: newestFirst)

        ourRatingTrend = Self.trend(oldestFirst: oldestFirst) { $0.ourRatingBefore.map(Double.init) }
        theirRatingTrend = Self.trend(oldestFirst: oldestFirst) { $0.opponentRating.map(Double.init) }
        scoreTrend = Self.scoreTrend(oldestFirst: oldestFirst)
    }

    private static func currentStreak(newestFirst: [LichessBotGameSummary]) -> LichessBotStreak? {
        var streakScore: Double?
        var length = 0
        for row in newestFirst {
            guard let score = row.ourScore else { continue }
            if let current = streakScore {
                if score == current {
                    length += 1
                } else {
                    break
                }
            } else {
                streakScore = score
                length = 1
            }
        }
        guard let streakScore else { return nil }
        return LichessBotStreak(ourScore: streakScore, length: length)
    }

    private static func longestStreaks(oldestFirst: [LichessBotGameSummary]) -> (win: Int, loss: Int) {
        var longestWin = 0
        var longestLoss = 0
        var win = 0
        var loss = 0
        for row in oldestFirst {
            switch row.ourScore {
            case .some(1):
                win += 1
                loss = 0
            case .some(0):
                loss += 1
                win = 0
            case .some:
                win = 0
                loss = 0
            case .none:
                continue
            }
            longestWin = max(longestWin, win)
            longestLoss = max(longestLoss, loss)
        }
        return (longestWin, longestLoss)
    }

    private static func highestRatedWin(newestFirst: [LichessBotGameSummary]) -> LichessBotNotableWin? {
        var best: LichessBotNotableWin?
        for row in newestFirst where row.ourScore == 1 {
            guard let rating = row.opponentRating else { continue }
            if best == nil || rating > best!.rating {
                best = LichessBotNotableWin(gameID: row.gameID, createdAt: row.createdAt, opponentName: row.opponentName, rating: rating)
            }
        }
        return best
    }

    private static func splitBySpeed(newestFirst: [LichessBotGameSummary]) -> [LichessBotOpponentSplitRow] {
        var byKey: [String: (tally: LichessBotResultTally, change: LichessBotRatingChange, last: Date)] = [:]
        var order: [String] = []
        for row in newestFirst {
            var entry = byKey[row.speed] ?? (LichessBotResultTally(), LichessBotRatingChange(), row.createdAt)
            if byKey[row.speed] == nil {
                order.append(row.speed)
            }
            entry.tally.add(ourScore: row.ourScore)
            entry.change.add(row)
            entry.last = max(entry.last, row.createdAt)
            byKey[row.speed] = entry
        }
        return order.compactMap { speed -> LichessBotOpponentSplitRow? in
            guard let entry = byKey[speed] else { return nil }
            return LichessBotOpponentSplitRow(id: "speed:\(speed)", label: speed, tally: entry.tally, ratingChange: entry.change, lastPlayedAt: entry.last)
        }
        .sorted { $0.tally.games > $1.tally.games }
    }

    private static func splitByModel(rows: [LichessBotGameSummary]) -> [LichessBotOpponentSplitRow] {
        let choices = LichessBotStatsModelChoice.choices(from: rows)
        let labels = Dictionary(uniqueKeysWithValues: choices.map { ($0.key, "\($0.modelID) · \($0.checkpointLabel)") })
        var byKey: [LichessBotModelKey: (tally: LichessBotResultTally, change: LichessBotRatingChange, last: Date)] = [:]
        var unattributed: (tally: LichessBotResultTally, change: LichessBotRatingChange, last: Date)?
        for row in rows {
            if let key = LichessBotModelAttribution(row.facts?.moves).model?.key {
                var entry = byKey[key] ?? (LichessBotResultTally(), LichessBotRatingChange(), row.createdAt)
                entry.tally.add(ourScore: row.ourScore)
                entry.change.add(row)
                entry.last = max(entry.last, row.createdAt)
                byKey[key] = entry
            } else {
                var entry = unattributed ?? (LichessBotResultTally(), LichessBotRatingChange(), row.createdAt)
                entry.tally.add(ourScore: row.ourScore)
                entry.change.add(row)
                entry.last = max(entry.last, row.createdAt)
                unattributed = entry
            }
        }
        var result = byKey.map { key, entry in
            LichessBotOpponentSplitRow(
                id: "model:\(key)",
                label: labels[key] ?? "unknown model",
                tally: entry.tally, ratingChange: entry.change, lastPlayedAt: entry.last)
        }
        .sorted { $0.lastPlayedAt > $1.lastPlayedAt }
        if let unattributed {
            result.append(LichessBotOpponentSplitRow(id: "model:none", label: "no recorded model", tally: unattributed.tally, ratingChange: unattributed.change, lastPlayedAt: unattributed.last))
        }
        return result
    }

    private static func trend(oldestFirst: [LichessBotGameSummary], value: (LichessBotGameSummary) -> Double?) -> [LichessBotOpponentTrendPoint] {
        var points: [LichessBotOpponentTrendPoint] = []
        for (index, row) in oldestFirst.enumerated() {
            guard let value = value(row) else { continue }
            points.append(LichessBotOpponentTrendPoint(gameID: row.gameID, createdAt: row.createdAt, index: index, value: value))
        }
        return points
    }

    private static func scoreTrend(oldestFirst: [LichessBotGameSummary]) -> [LichessBotOpponentTrendPoint] {
        var points: [LichessBotOpponentTrendPoint] = []
        var scored = 0
        var running = 0.0
        for (index, row) in oldestFirst.enumerated() {
            guard let score = row.ourScore else { continue }
            scored += 1
            running += score
            points.append(LichessBotOpponentTrendPoint(gameID: row.gameID, createdAt: row.createdAt, index: index, value: running / Double(scored)))
        }
        return points
    }
}
