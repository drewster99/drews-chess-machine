//
//  ArchitecturePresetStore.swift
//  DrewsChessMachine
//
//  Resolves architecture presets for the Build-New-Model screen and the
//  `--new-model --architecture <value>` CLI flag (plan §10).
//
//  Two sources:
//  - **Built-ins**: compiled-in (`NetworkArchitecture.Preset`), immutable, never
//    written to disk so they can't drift. Their names are *reserved*.
//  - **User-saved**: one `.json` per preset under
//    `~/Library/Application Support/DrewsChessMachine/Presets/`, each a
//    `NamedArchitecture` ({label, architecture}). The filename stem is the
//    preset *name* (the lookup key); `label` is the human display string.
//
//  A `.json` file in the Presets folder and an `--architecture <path>` target
//  share the same on-disk format (`NamedArchitecture`), so a preset file *is* an
//  arch file that just lives in the well-known folder.
//

import Foundation

/// A topology plus its human label. `label` lives here (outside the
/// purely-topological `NetworkArchitecture`) so the config's identity stays the
/// topology alone (plan §5a decision (b)).
struct NamedArchitecture: Codable, Sendable, Hashable {
    var label: String
    var architecture: NetworkArchitecture
}

enum ArchitecturePresetStore {

    enum StoreError: Error, CustomStringConvertible {
        case presetNotFound(String)
        case reservedName(String)
        case invalid(name: String, detail: String)

        var description: String {
            switch self {
            case .presetNotFound(let n):
                return "No architecture preset named '\(n)' (built-in or user-saved)."
            case .reservedName(let n):
                return "'\(n)' is a reserved built-in preset name and cannot be overwritten."
            case .invalid(let n, let d):
                return "Architecture preset '\(n)' is invalid: \(d)"
            }
        }
    }

    /// `~/Library/Application Support/DrewsChessMachine/Presets/`.
    static var presetsDirURL: URL {
        CheckpointPaths.rootURL.appendingPathComponent("Presets", isDirectory: true)
    }

    // MARK: Built-ins (compiled-in, reserved names)

    /// Friendly display label for a built-in preset.
    private static func builtInLabel(_ p: NetworkArchitecture.Preset) -> String {
        switch p {
        case .v3_8block_3x3:  return "v3 · 8-block 3×3 (historical)"
        case .v3_16block_3x3: return "v3 · 16-block 3×3 (historical)"
        case .v4_12block_3x3: return "v4 · 12-block 3×3"
        case .v4_5block_7x7:  return "v4 · 5-block 7×7 (current)"
        case .v4_8block_3x3:  return "v4 · 8-block 3×3"
        case .v4_4block_3x3_fp32: return "v4 · 4-block 3×3 (fp32)"
        case .v4_5block_7x7_fusion: return "v4 · 5-block 7×7 + feature-skip"
        case .v5_5block_7x7_lnout: return "v5 · 5-block 7×7 + LayerNorm out"
        case .nt8y_3x3stem: return "nt8y · 3-block 15×15 @32 + 3×3 stem"
        }
    }

    /// All built-in presets, keyed by name (the enum rawValue).
    static var builtIns: [(name: String, named: NamedArchitecture)] {
        NetworkArchitecture.Preset.allCases.map { p in
            (name: p.rawValue,
             named: NamedArchitecture(label: builtInLabel(p), architecture: .preset(p)))
        }
    }

    private static var builtInNames: Set<String> {
        Set(NetworkArchitecture.Preset.allCases.map(\.rawValue))
    }

    /// The current default preset as a `NamedArchitecture` — the default the
    /// "New Network…" screen opens to ("latest and greatest").
    static var currentNamed: NamedArchitecture {
        let p = NetworkArchitecture.Preset.current
        return NamedArchitecture(label: builtInLabel(p), architecture: .preset(p))
    }

    // MARK: User-saved presets

    /// Decode + validate a `NamedArchitecture` from a JSON file. Throws on a
    /// malformed or structurally-invalid config (so a bad hand-edit is a clear
    /// error, not a silently-wrong build).
    static func loadFile(at url: URL) throws -> NamedArchitecture {
        let data = try Data(contentsOf: url)
        let named = try JSONDecoder().decode(NamedArchitecture.self, from: data)
        do {
            try named.architecture.validate()
        } catch {
            throw StoreError.invalid(name: url.deletingPathExtension().lastPathComponent,
                                     detail: String(describing: error))
        }
        return named
    }

    /// User-saved presets from the Presets folder, keyed by filename stem.
    /// Files that fail to decode/validate, or that shadow a reserved built-in
    /// name, are skipped (with a log line) rather than aborting the listing.
    static func userPresets() -> [(name: String, named: NamedArchitecture)] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: presetsDirURL, includingPropertiesForKeys: nil) else { return [] }
        var result: [(name: String, named: NamedArchitecture)] = []
        for url in entries where url.pathExtension.lowercased() == "json" {
            let name = url.deletingPathExtension().lastPathComponent
            if builtInNames.contains(name) {
                SessionLogger.shared.log("[PRESET] Skipping user file '\(name).json' — name is a reserved built-in.")
                continue
            }
            do {
                result.append((name: name, named: try loadFile(at: url)))
            } catch {
                SessionLogger.shared.log("[PRESET] Ignoring '\(name).json': \(String(describing: error))")
            }
        }
        return result.sorted { $0.name < $1.name }
    }

    /// Built-ins followed by user-saved presets (built-in names reserved).
    static func allPresets() -> [(name: String, named: NamedArchitecture)] {
        builtIns + userPresets()
    }

    /// Resolve a preset by name: built-in first, then user-saved.
    static func resolve(name: String) throws -> NamedArchitecture {
        if let p = NetworkArchitecture.Preset(rawValue: name) {
            return NamedArchitecture(label: builtInLabel(p), architecture: .preset(p))
        }
        let userURL = presetsDirURL.appendingPathComponent("\(name).json")
        guard FileManager.default.fileExists(atPath: userURL.path) else {
            throw StoreError.presetNotFound(name)
        }
        return try loadFile(at: userURL)
    }

    /// Resolve a `--architecture` CLI value into an architecture + a short
    /// source name. Accepts, in order:
    ///  1. a **name** — a built-in preset, or a user-saved preset in the
    ///     Presets folder, with or without a trailing `.json` (so `nt8y` and
    ///     `nt8y.json` both resolve to `Presets/nt8y.json`);
    ///  2. a **path** — any `NamedArchitecture` JSON file (absolute, `~`-, or
    ///     cwd-relative) when the value is not a known name.
    ///
    /// A malformed/invalid named preset surfaces its `.invalid` error (it is
    /// NOT silently retried as a path) — only "no such name" falls through to
    /// the path interpretation.
    static func resolve(nameOrPath value: String) throws -> (named: NamedArchitecture, sourceName: String) {
        // 1. As a name (built-in, or Presets/<base>.json), tolerating a `.json`
        //    suffix on the passed value.
        let base = value.lowercased().hasSuffix(".json") ? String(value.dropLast(5)) : value
        if !base.isEmpty {
            do {
                return (try resolve(name: base), base)
            } catch StoreError.presetNotFound {
                // Not a known name — try it as a filesystem path below.
            }
        }
        // 2. As a filesystem path.
        let url = URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StoreError.presetNotFound(value)
        }
        return (try loadFile(at: url), url.deletingPathExtension().lastPathComponent)
    }

    /// Save a user preset as `<name>.json` (pretty-printed, sorted keys for a
    /// stable, hand-editable file). Refuses reserved built-in names.
    @discardableResult
    static func save(name: String, label: String, architecture: NetworkArchitecture) throws -> URL {
        guard !builtInNames.contains(name) else { throw StoreError.reservedName(name) }
        try architecture.validate()
        let fm = FileManager.default
        try fm.createDirectory(at: presetsDirURL, withIntermediateDirectories: true)
        let url = presetsDirURL.appendingPathComponent("\(name).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(NamedArchitecture(label: label, architecture: architecture))
            .write(to: url, options: [.atomic])
        SessionLogger.shared.log("[PRESET] Saved '\(name).json': \(architecture.architectureSummary)")
        return url
    }
}
