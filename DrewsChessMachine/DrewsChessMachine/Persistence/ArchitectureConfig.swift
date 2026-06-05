//
//  ArchitectureConfig.swift
//  DrewsChessMachine
//
//  Load/save a NetworkArchitecture as `architecture.json` — kept separate from
//  `parameters.json` (training knobs) so architecture and training config stay
//  decoupled. When present at the default location, it drives the next Build,
//  letting you construct a non-default topology without rebuilding the app.
//

import Foundation

enum ArchitectureConfig {
    static let defaultFilename = "architecture.json"

    /// `~/Library/Application Support/DrewsChessMachine/architecture.json`.
    static var defaultFileURL: URL {
        CheckpointPaths.rootURL.appendingPathComponent(defaultFilename)
    }

    /// Decode + validate a NetworkArchitecture from a JSON file.
    static func load(from url: URL) throws -> NetworkArchitecture {
        let data = try Data(contentsOf: url)
        let arch = try JSONDecoder().decode(NetworkArchitecture.self, from: data)
        try arch.validate()
        return arch
    }

    /// The architecture from `architecture.json` at the default location if it
    /// exists and is valid; otherwise nil (caller falls back to `.current`).
    /// A malformed/invalid file logs and returns nil rather than throwing, so a
    /// bad edit can't block building entirely.
    static func loadDefaultIfPresent() -> NetworkArchitecture? {
        let url = defaultFileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let arch = try load(from: url)
            SessionLogger.shared.log("[ARCH-CONFIG] Using \(url.lastPathComponent): \(arch.architectureSummary)")
            return arch
        } catch {
            // `String(describing:)` keeps the precise DecodingError context
            // (key path, type) that `localizedDescription` flattens to a vague
            // "data couldn't be read" — essential for debugging a hand-edited file.
            SessionLogger.shared.log("[ARCH-CONFIG] Ignoring \(url.lastPathComponent): \(String(describing: error))")
            return nil
        }
    }

    /// Write a pretty-printed template (defaults to the current architecture).
    @discardableResult
    static func writeTemplate(_ arch: NetworkArchitecture = .current, to url: URL) throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(arch).write(to: url, options: [.atomic])
        return url
    }
}
