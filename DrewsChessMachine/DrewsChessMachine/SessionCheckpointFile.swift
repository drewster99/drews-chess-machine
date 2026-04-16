import Foundation

// MARK: - Errors

enum SessionCheckpointError: LocalizedError {
    case missingChampionFile
    case missingTrainerFile
    case missingSessionJSON
    case invalidJSON(Error, detail: String = "")
    case unsupportedVersion(Int)
    case targetDirectoryExists(URL)

    var errorDescription: String? {
        switch self {
        case .missingChampionFile:
            return "Session directory is missing champion.dcmmodel"
        case .missingTrainerFile:
            return "Session directory is missing trainer.dcmmodel"
        case .missingSessionJSON:
            return "Session directory is missing session.json"
        case .invalidJSON(let err, let detail):
            let base = "session.json could not be decoded: \(err)"
            return detail.isEmpty ? base : "\(base)\nFirst 2000 bytes of file:\n\(detail)"
        case .unsupportedVersion(let v):
            return "Unsupported session.json format version \(v)"
        case .targetDirectoryExists(let url):
            return "Refusing to overwrite existing session at \(url.lastPathComponent)"
        }
    }
}

// MARK: - Serializable shapes for non-Codable project types

/// Codable mirror of `SamplingSchedule`. The original struct is
/// project-owned but not Codable (and making it Codable would
/// drag an import into MPSChessPlayer.swift); the checkpoint path
/// builds this wrapper on save and turns it back into the live
/// struct on load.
struct TauConfigCodable: Codable, Equatable {
    let openingPliesPerPlayer: Int
    let openingTau: Float
    let mainTau: Float

    init(_ schedule: SamplingSchedule) {
        self.openingPliesPerPlayer = schedule.openingPliesPerPlayer
        self.openingTau = schedule.openingTau
        self.mainTau = schedule.mainTau
    }

    var asSamplingSchedule: SamplingSchedule {
        SamplingSchedule(
            openingPliesPerPlayer: openingPliesPerPlayer,
            openingTau: openingTau,
            mainTau: mainTau
        )
    }
}

/// Codable mirror of `TournamentRecord` — saved in the session's
/// arena history for context on resume. Only the audit fields
/// (counts, score, promoted flag, duration) are persisted; live UI
/// state like `isCurrent` is not.
struct ArenaHistoryEntryCodable: Codable, Equatable {
    let finishedAtStep: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    let score: Double
    let promoted: Bool
    let promotedID: String?
    let durationSec: Double
}

// MARK: - Session State

/// Serialized form of a paused training session. Stored at
/// `session.json` inside a `.dcmsession` directory next to the
/// champion and trainer `.dcmmodel` files. Loaded at resume time
/// to re-seed counters, hyperparameter display, and network
/// identity.
struct SessionCheckpointState: Codable, Equatable {
    static let currentFormatVersion: Int = 1

    let formatVersion: Int
    let sessionID: String
    let savedAtUnix: Int64
    let sessionStartUnix: Int64
    /// Accumulated elapsed seconds at save time, measured against
    /// the session's `sessionStart` anchor. On resume, the new
    /// session's anchor is placed at `Date() - elapsedTrainingSec`
    /// so "Total session time" picks up where it left off.
    let elapsedTrainingSec: Double

    // Counters
    let trainingSteps: Int
    let selfPlayGames: Int
    let selfPlayMoves: Int
    let trainingPositionsSeen: Int

    // Hyperparameters (as they were in effect at save time)
    let batchSize: Int
    let learningRate: Float
    let promoteThreshold: Double
    let arenaGames: Int
    let selfPlayTau: TauConfigCodable
    let arenaTau: TauConfigCodable
    let selfPlayWorkerCount: Int

    // Replay-ratio controller settings. Optional so sessions saved
    // before the ratio controller existed still decode; callers use
    // `?? 1.0` / `?? true` at the read site. Swift's synthesized
    // Codable only calls `decodeIfPresent` for Optional types — a
    // non-optional `var` with a default still requires the key.
    var replayRatioTarget: Double?
    var replayRatioAutoAdjust: Bool?

    // Network identity — duplicated from the `.dcmmodel` headers so
    // a future "browse saved sessions" UI can read just
    // `session.json` and still show model IDs.
    let championID: String
    let trainerID: String

    // Arena history (audit log — displayed in the UI on resume)
    let arenaHistory: [ArenaHistoryEntryCodable]

    // MARK: JSON serialization

    static func decode(_ data: Data) throws -> SessionCheckpointState {
        do {
            let state = try JSONDecoder().decode(SessionCheckpointState.self, from: data)
            guard state.formatVersion == Self.currentFormatVersion else {
                throw SessionCheckpointError.unsupportedVersion(state.formatVersion)
            }
            return state
        } catch let err as SessionCheckpointError {
            throw err
        } catch {
            throw SessionCheckpointError.invalidJSON(
                error,
                detail: String(data: data.prefix(2000), encoding: .utf8) ?? "(non-utf8)"
            )
        }
    }

    func encode() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            return try encoder.encode(self)
        } catch {
            throw SessionCheckpointError.invalidJSON(error)
        }
    }
}

// MARK: - Directory layout

/// Filenames for the three items inside a `.dcmsession` directory.
/// A session is a plain directory (not a bundle) so the
/// constituent `.dcmmodel` files are immediately usable when
/// copied out in Finder.
enum SessionCheckpointLayout {
    static let championFilename = "champion.dcmmodel"
    static let trainerFilename = "trainer.dcmmodel"
    static let stateFilename = "session.json"

    static func championURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(championFilename)
    }

    static func trainerURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(trainerFilename)
    }

    static func stateURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(stateFilename)
    }

    /// Read the three raw payloads out of a session directory.
    /// Parsing is deferred to the caller so errors from the two
    /// `.dcmmodel` files surface with their original
    /// `ModelCheckpointError` types.
    static func readAll(
        from directoryURL: URL
    ) throws -> (stateData: Data, championData: Data, trainerData: Data) {
        // Normalize the incoming URL to a plain file-path URL.
        // The file importer on macOS can return file-reference URLs
        // or bookmark URLs whose `appendingPathComponent` doesn't
        // resolve to the expected child path. Reconstructing via
        // `URL(fileURLWithPath:isDirectory:)` strips any of that
        // metadata and gives a clean POSIX-path-based URL whose
        // children resolve correctly.
        let normalizedDir = URL(fileURLWithPath: directoryURL.path, isDirectory: true)
        let fm = FileManager.default
        let championURL = championURL(in: normalizedDir)
        let trainerURL = trainerURL(in: normalizedDir)
        let stateURL = stateURL(in: normalizedDir)

        guard fm.fileExists(atPath: championURL.path) else {
            throw SessionCheckpointError.missingChampionFile
        }
        guard fm.fileExists(atPath: trainerURL.path) else {
            throw SessionCheckpointError.missingTrainerFile
        }
        guard fm.fileExists(atPath: stateURL.path) else {
            throw SessionCheckpointError.missingSessionJSON
        }

        let stateData = try Data(contentsOf: stateURL)
        let championData = try Data(contentsOf: championURL)
        let trainerData = try Data(contentsOf: trainerURL)
        return (stateData, championData, trainerData)
    }
}
