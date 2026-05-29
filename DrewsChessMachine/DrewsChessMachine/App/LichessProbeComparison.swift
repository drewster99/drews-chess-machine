import Foundation

/// Loaded comparison snapshot read from a previously-exported Lichess
/// Probe JSON (schema v2). Lives on the `LichessProbeDetailView` as
/// optional state — `nil` when no comparison is active, populated when
/// the user clicks Compare and picks a file.
///
/// Holds three pieces of state:
///   - the full decoded payload, so the header can surface the
///     comparison's model label / tick timestamp / app build,
///   - a per-puzzle-id index for O(1) row lookup during rendering,
///   - the source URL, for displaying the filename in the header.
///
/// Built once at load time and treated as immutable. `Sendable` so it
/// can pass cleanly across SwiftUI's view updates and any future
/// background-decoded variant.
struct LichessProbeComparison: Sendable {
    let payload: LoadedPayload
    let byPuzzleId: [String: LoadedPuzzleEntry]
    let sourceURL: URL
    /// Per-theme aggregates derived from the loaded puzzles at init time.
    /// Same shape as `LichessProbeHistory.Aggregate` so the Detail view's
    /// section-header rendering can reuse the live-side helpers verbatim.
    /// Computed once on load — JSON data doesn't change after that.
    let aggregatesByCategory: [ProbeCategory: LichessProbeHistory.Aggregate]
    /// Overall summary across every loaded puzzle. Same shape as the
    /// per-theme aggregate, just totalled.
    let overallSummary: LichessProbeOverallSummary

    init(payload: LoadedPayload, sourceURL: URL) {
        self.payload = payload
        self.sourceURL = sourceURL
        var dict: [String: LoadedPuzzleEntry] = [:]
        dict.reserveCapacity(payload.puzzles.count)
        for entry in payload.puzzles {
            dict[entry.puzzle.id] = entry
        }
        self.byPuzzleId = dict

        let aggregates = Self.aggregatesByCategory(from: payload.puzzles)
        self.aggregatesByCategory = aggregates
        self.overallSummary = LichessProbeOverallSummary(folding: Array(aggregates.values))
    }

    /// Bucket loaded puzzles by theme_category → fold into the same
    /// `LichessProbeHistory.Aggregate` shape the live-data path uses.
    /// Mirrors the verdict semantics in `LichessProbeHistory.aggregates(from:)`
    /// (correctAndConfident + correctButFlat → argmax; .correctInTop5 also
    /// counts toward top5; .error → errored; .wrong → no contribution).
    private static func aggregatesByCategory(
        from puzzles: [LoadedPuzzleEntry]
    ) -> [ProbeCategory: LichessProbeHistory.Aggregate] {
        var byCategory: [ProbeCategory: [LoadedPuzzleEntry]] = [:]
        for entry in puzzles {
            guard let cat = ProbeCategory(rawValue: entry.puzzle.themeCategory) else {
                continue
            }
            byCategory[cat, default: []].append(entry)
        }

        var out: [ProbeCategory: LichessProbeHistory.Aggregate] = [:]
        out.reserveCapacity(byCategory.count)
        for (cat, entries) in byCategory {
            var argmaxCorrect = 0
            var top5Correct = 0
            var errored = 0
            var sumProb: Float = 0
            var sumRank = 0
            var countRank = 0
            for entry in entries {
                let pr = entry.probeResult
                sumProb += pr.expectedProb
                if let rank = pr.expectedRank {
                    sumRank += rank
                    countRank += 1
                }
                switch pr.verdict {
                case "correctAndConfident", "correctButFlat":
                    argmaxCorrect += 1
                    top5Correct += 1
                case "correctInTop5":
                    top5Correct += 1
                case "error":
                    errored += 1
                default:
                    break  // "wrong" or unknown
                }
            }
            out[cat] = LichessProbeHistory.Aggregate(
                theme: cat,
                total: entries.count,
                argmaxCorrect: argmaxCorrect,
                top5Correct: top5Correct,
                errored: errored,
                sumExpectedProb: sumProb,
                sumExpectedRank: sumRank,
                countWithRank: countRank
            )
        }
        return out
    }

    // MARK: - Decodable payloads (schema v2)

    struct LoadedPayload: Decodable, Sendable {
        let schemaVersion: Int
        let exportId: String
        let generatedAt: String
        let tickTimestamp: String
        let modelLabel: String?
        let appBuild: LoadedAppBuild?
        let probeCount: Int
        let puzzles: [LoadedPuzzleEntry]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case exportId = "export_id"
            case generatedAt = "generated_at"
            case tickTimestamp = "tick_timestamp"
            case modelLabel = "model_label"
            case appBuild = "app_build"
            case probeCount = "probe_count"
            case puzzles
        }
    }

    /// Loose decoding — only the fields the comparison header surfaces.
    /// Extra fields in the JSON (build_timestamp, git_branch, git_dirty)
    /// are ignored by JSONDecoder by default, so adding fields to the
    /// exporter side won't break loaders.
    struct LoadedAppBuild: Decodable, Sendable {
        let buildNumber: Int?
        let buildDate: String?
        let gitHash: String?
        let summary: String?

        enum CodingKeys: String, CodingKey {
            case buildNumber = "build_number"
            case buildDate = "build_date"
            case gitHash = "git_hash"
            case summary
        }
    }

    struct LoadedPuzzleEntry: Decodable, Sendable {
        let puzzle: LoadedPuzzle
        let probeResult: LoadedProbeResult

        enum CodingKeys: String, CodingKey {
            case puzzle
            case probeResult = "probe_result"
        }
    }

    /// Only fields used by the comparison's per-row matching and the
    /// derived aggregates. `theme_category` is decoded so the loader
    /// can bucket puzzles into the same `ProbeCategory` keys the live
    /// per-theme summary uses; other puzzle fields (theme, fen,
    /// expected move, rating) are intentionally not decoded — the
    /// live row already carries them via `LichessProbeData.metadata`.
    struct LoadedPuzzle: Decodable, Sendable {
        let id: String
        let themeCategory: String

        enum CodingKeys: String, CodingKey {
            case id
            case themeCategory = "theme_category"
        }
    }

    struct LoadedProbeResult: Decodable, Sendable {
        let verdict: String
        let expectedRank: Int?
        let expectedProb: Float
        let valueWdl: LoadedValueWDL?
        let topMoves: [LoadedTopMove]

        enum CodingKeys: String, CodingKey {
            case verdict
            case expectedRank = "expected_rank"
            case expectedProb = "expected_prob"
            case valueWdl = "value_wdl"
            case topMoves = "top_moves"
        }
    }

    struct LoadedValueWDL: Decodable, Sendable {
        let win: Float
        let draw: Float
        let loss: Float
    }

    struct LoadedTopMove: Decodable, Sendable {
        let notation: String
        let prob: Float
    }
}

/// Sum-of-all-themes summary across every puzzle in a snapshot. Shared
/// shape between the live data (`history.latestPerPuzzleResults`) and
/// the comparison (`LichessProbeComparison`) so the Detail view's
/// OVERALL band renders identically for both sides. Built by folding
/// an array of per-theme `LichessProbeHistory.Aggregate`s — same
/// fields, totalled.
struct LichessProbeOverallSummary: Sendable {
    let totalProbes: Int
    let argmaxCorrect: Int
    let top5Correct: Int
    let errored: Int
    let sumExpectedProb: Float
    let sumExpectedRank: Int
    let countWithRank: Int

    init(folding aggregates: [LichessProbeHistory.Aggregate]) {
        var total = 0
        var argmax = 0
        var top5 = 0
        var err = 0
        var sumProb: Float = 0
        var sumRank = 0
        var countRank = 0
        for a in aggregates {
            total += a.total
            argmax += a.argmaxCorrect
            top5 += a.top5Correct
            err += a.errored
            sumProb += a.sumExpectedProb
            sumRank += a.sumExpectedRank
            countRank += a.countWithRank
        }
        self.totalProbes = total
        self.argmaxCorrect = argmax
        self.top5Correct = top5
        self.errored = err
        self.sumExpectedProb = sumProb
        self.sumExpectedRank = sumRank
        self.countWithRank = countRank
    }

    var argmaxFraction: Float {
        totalProbes > 0 ? Float(argmaxCorrect) / Float(totalProbes) : 0
    }
    var top5Fraction: Float {
        totalProbes > 0 ? Float(top5Correct) / Float(totalProbes) : 0
    }
    var avgExpectedProb: Float {
        totalProbes > 0 ? sumExpectedProb / Float(totalProbes) : 0
    }
    var avgExpectedRank: Float? {
        guard countWithRank > 0 else { return nil }
        return Float(sumExpectedRank) / Float(countWithRank)
    }
}
