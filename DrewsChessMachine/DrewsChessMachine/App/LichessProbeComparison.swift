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

    init(payload: LoadedPayload, sourceURL: URL) {
        self.payload = payload
        self.sourceURL = sourceURL
        var dict: [String: LoadedPuzzleEntry] = [:]
        dict.reserveCapacity(payload.puzzles.count)
        for entry in payload.puzzles {
            dict[entry.puzzle.id] = entry
        }
        self.byPuzzleId = dict
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

    /// Only the puzzle id is needed at row-render time (the live row's
    /// other puzzle fields come from `LichessProbeData.metadata`).
    /// Decoding the rest is wasted work that would also tie us to the
    /// exact schema for fields we don't surface.
    struct LoadedPuzzle: Decodable, Sendable {
        let id: String
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
