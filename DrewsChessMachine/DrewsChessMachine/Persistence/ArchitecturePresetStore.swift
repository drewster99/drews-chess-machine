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
///
/// On disk (user presets, `--architecture` files) it also carries
/// `format_version` (`ArchitectureFormat.currentVersion`, always written).
/// A file without it predates the marker and decodes as legacy — absent
/// version-gated fields resolve to their pre-existing behavior; a file with
/// it must state every field its version requires (see `ArchitectureFormat`).
/// The marker is a property of the file, not of the value: it is not stored,
/// so two presets with equal label + architecture are equal.
struct NamedArchitecture: Codable, Sendable, Hashable {
    var label: String
    var architecture: NetworkArchitecture

    init(label: String, architecture: NetworkArchitecture) {
        self.label = label
        self.architecture = architecture
    }

    private enum CodingKeys: String, CodingKey {
        case label
        case architecture
        case formatVersion = "format_version"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let base = ArchitectureFormat.DecodeFormat.from(decoder)
        let version: Int
        if let stated = try c.decodeIfPresent(Int.self, forKey: .formatVersion) {
            version = try ArchitectureFormat.requireSupported(stated, source: base.source)
        } else {
            version = ArchitectureFormat.unversionedLegacyVersion
        }
        label = try c.decode(String.self, forKey: .label)
        architecture = try NetworkArchitecture(
            from: try c.superDecoder(forKey: .architecture),
            format: base.withFormatVersion(version))
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(ArchitectureFormat.currentVersion, forKey: .formatVersion)
        try c.encode(label, forKey: .label)
        try c.encode(architecture, forKey: .architecture)
    }
}

enum ArchitecturePresetStore {

    enum StoreError: Error, Equatable, CustomStringConvertible {
        case presetNotFound(String)
        case reservedName(String)
        case invalid(name: String, detail: String)
        /// The name can't be a preset file name (see `validatePresetName`).
        case invalidPresetName(name: String, reason: String)
        /// A user preset with this name is already saved and the caller did
        /// not confirm replacing it.
        case presetAlreadyExists(String)
        /// Something other than a regular file (a folder, a symbolic link…)
        /// sits where the preset file would go; it is never replaced.
        case presetPathNotARegularFile(name: String, kind: String)
        /// The preset file's URL did not resolve to a direct child of the
        /// Presets folder. Unreachable for a name `validatePresetName`
        /// accepts; checked anyway so a gap in the name rules can't write
        /// elsewhere.
        case presetPathOutsidePresetsFolder(name: String, path: String)

        var description: String {
            switch self {
            case .presetNotFound(let n):
                return "No architecture preset named '\(n)' (built-in or user-saved)."
            case .reservedName(let n):
                return "'\(n)' is a reserved built-in preset name and cannot be overwritten."
            case .invalid(let n, let d):
                return "Architecture preset '\(n)' is invalid: \(d)"
            case .invalidPresetName(let n, let reason):
                return "'\(n)' can't be a preset name: \(reason)"
            case .presetAlreadyExists(let n):
                return "A preset named '\(n)' already exists."
            case .presetPathNotARegularFile(let n, let kind):
                return "'\(n).json' in the Presets folder is not a regular file (\(kind)); it will not be replaced."
            case .presetPathOutsidePresetsFolder(let n, let path):
                return "Preset name '\(n)' resolves to \(path), outside the Presets folder; refusing to write it."
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
        case .nt8y_15x15stem: return "nt8y · 3-block 15×15 @32 + 15×15 stem"
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

    // MARK: User-saved presets

    /// Decode + validate a `NamedArchitecture` from a JSON file. Throws on a
    /// malformed or structurally-invalid config (so a bad hand-edit is a clear
    /// error, not a silently-wrong build).
    ///
    /// The version gate reports `url`'s file name in errors, and a legacy
    /// file's resolutions are logged once per call.
    static func loadFile(at url: URL) throws -> NamedArchitecture {
        let data = try Data(contentsOf: url)
        let format = ArchitectureFormat.DecodeFormat(
            formatVersion: ArchitectureFormat.currentVersion, source: url.lastPathComponent)
        let named = try ArchitectureFormat.makeDecoder(format: format).decode(NamedArchitecture.self, from: data)
        // `NamedArchitecture.init(from:)` re-stamps the version from the
        // file's own marker; the resolutions land on this shared log.
        if let line = legacyPresetLogLine(format: format) {
            SessionLogger.shared.log(line)
        }
        do {
            try named.architecture.validate()
        } catch {
            throw StoreError.invalid(name: url.deletingPathExtension().lastPathComponent,
                                     detail: String(describing: error))
        }
        return named
    }

    /// The one `[ARCH]` line for a preset load that made legacy resolutions
    /// (`format`'s version is the decoder's starting value, not the file's,
    /// so the line names the resolutions rather than a version).
    private static func legacyPresetLogLine(format: ArchitectureFormat.DecodeFormat) -> String? {
        let resolutions = format.legacyLog.resolutions
        guard !resolutions.isEmpty else { return nil }
        return "[ARCH] legacy architecture preset \(format.source): " + resolutions.joined(separator: "; ")
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

    // MARK: Saving

    /// The file-name suffix every user preset carries.
    private static let presetFileExtension = "json"

    /// Longest file name a macOS volume accepts, in UTF-8 bytes.
    private static let maxFileNameUTF8Bytes = Int(NAME_MAX)

    /// Characters a preset name may contain besides letters and digits.
    private static let allowedPresetNamePunctuation: Set<Character> = [" ", "_", "-", "."]

    /// Throws `.invalidPresetName` unless `name` can be the stem of a preset
    /// file directly inside the Presets folder. The name is typed by the
    /// user in Build New Model and becomes `<name>.json`, so it must not be
    /// able to name anything else:
    ///
    /// - **Letters, digits, space, `_`, `-`, `.` only.** Excludes `/` (a
    ///   path separator — `../x` or `a/b` would write outside the folder or
    ///   into a subfolder), `:` (Finder shows it as `/`), and control
    ///   characters. Letters and digits are Unicode (`Character.isLetter` /
    ///   `isNumber`), matching what `BuildNewModelModel.defaultSaveName`
    ///   produces from a label, so the suggested default passes these
    ///   character rules.
    /// - **No leading `.`** — that would be a hidden file, and rules out `.`
    ///   and `..` outright.
    /// - **No leading or trailing whitespace** — invisible in the picker and
    ///   in Finder, so two presets would look identical.
    /// - **No `.json` ending** — `resolve(nameOrPath:)` strips one `.json`
    ///   from what it is given, so a preset saved as `x.json.json` could
    ///   never be reached by its name from `--architecture`.
    /// - **Fits in one file name** together with the `.json` suffix — and
    ///   with room for the hidden staging name every save writes first
    ///   (`FileSafety.temporarySiblingNameOverhead` more bytes), since a
    ///   name that fits only as the final file fails at staging with
    ///   `ENAMETOOLONG`.
    /// - **Not a built-in preset's name** (`.reservedName`).
    static func validatePresetName(_ name: String) throws {
        func reject(_ reason: String) -> StoreError {
            StoreError.invalidPresetName(name: name, reason: reason)
        }
        guard !name.isEmpty else { throw reject("the name is empty") }
        guard name.trimmingCharacters(in: .whitespacesAndNewlines) == name else {
            throw reject("it starts or ends with whitespace")
        }
        guard !name.hasPrefix(".") else { throw reject("it starts with '.'") }
        if let bad = name.first(where: { !($0.isLetter || $0.isNumber || allowedPresetNamePunctuation.contains($0)) }) {
            throw reject("'\(bad)' is not allowed — use letters, digits, spaces, '_', '-' or '.'")
        }
        guard !name.lowercased().hasSuffix("." + presetFileExtension) else {
            throw reject("it ends in '.\(presetFileExtension)' (the extension is added automatically)")
        }
        let presetFileNameUTF8Bytes = name.utf8.count + 1 + presetFileExtension.utf8.count
        let longestPresetFileNameUTF8Bytes = maxFileNameUTF8Bytes - FileSafety.temporarySiblingNameOverhead
        guard presetFileNameUTF8Bytes <= longestPresetFileNameUTF8Bytes else {
            throw reject("it is too long: '<name>.\(presetFileExtension)' may be at most "
                + "\(longestPresetFileNameUTF8Bytes) bytes (UTF-8) so that the save's temporary file name "
                + "still fits, and this one is \(presetFileNameUTF8Bytes)")
        }
        guard !builtInNames.contains(name) else { throw StoreError.reservedName(name) }
    }

    /// Save a user preset as `<name>.json` in the Presets folder
    /// (pretty-printed, sorted keys for a stable, hand-editable file).
    ///
    /// - `name` must pass `validatePresetName`.
    /// - An existing preset of the same name is replaced only when
    ///   `replacingExisting` is true — the caller's record that the user
    ///   confirmed it. Otherwise the write is exclusive and an existing file
    ///   throws `.presetAlreadyExists`, leaving it untouched; the UI catches
    ///   that and asks.
    /// - Only a regular file is ever replaced. A folder or symbolic link
    ///   named `<name>.json` throws `.presetPathNotARegularFile` either way.
    /// - Every write is staged in a temporary sibling and renamed into place
    ///   (`FileSafety.publishNewFile` / `replaceRegularFile`), so a crash
    ///   never leaves a half-written preset.
    @discardableResult
    static func save(name: String, label: String, architecture: NetworkArchitecture, replacingExisting: Bool) throws -> URL {
        try save(name: name, label: label, architecture: architecture,
                 replacingExisting: replacingExisting, presetsDirectory: presetsDirURL)
    }

    /// `save(name:label:architecture:replacingExisting:)` against an explicit
    /// Presets folder — the production entry point passes `presetsDirURL`;
    /// tests pass a temporary folder.
    @discardableResult
    static func save(
        name: String,
        label: String,
        architecture: NetworkArchitecture,
        replacingExisting: Bool,
        presetsDirectory: URL
    ) throws -> URL {
        try validatePresetName(name)
        try architecture.validate()
        try FileManager.default.createDirectory(at: presetsDirectory, withIntermediateDirectories: true)
        let url = presetsDirectory.appendingPathComponent("\(name).\(presetFileExtension)", isDirectory: false)
        guard url.deletingLastPathComponent().standardizedFileURL.path == presetsDirectory.standardizedFileURL.path,
              url.lastPathComponent == "\(name).\(presetFileExtension)" else {
            throw StoreError.presetPathOutsidePresetsFolder(name: name, path: url.standardizedFileURL.path)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(NamedArchitecture(label: label, architecture: architecture))

        let existing = try FileSafety.existingItem(at: url)
        if let existing, existing.kind != .regularFile {
            throw StoreError.presetPathNotARegularFile(name: name, kind: existing.kind.description)
        }
        let replacedExisting = replacingExisting && existing != nil
        do {
            if replacingExisting {
                // Replaces only the regular file checked above (by identity);
                // with nothing there it publishes a new file exclusively.
                try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: existing?.identity)
            } else {
                // Exclusive publish: refuses if anything is at the path,
                // including one that appeared after the check above.
                try FileSafety.publishNewFile(data, to: url)
            }
        } catch FileSafetyError.alreadyExists(path: _, kind: .regularFile) {
            throw StoreError.presetAlreadyExists(name)
        } catch FileSafetyError.alreadyExists(path: _, kind: let kind) {
            throw StoreError.presetPathNotARegularFile(name: name, kind: kind.description)
        } catch FileSafetyError.notARegularFile(path: _, kind: let kind) {
            throw StoreError.presetPathNotARegularFile(name: name, kind: kind.description)
        } catch FileSafetyError.fileChangedSinceWritten {
            // A different file took the preset's place after the user
            // confirmed replacing the one that was there; ask again.
            throw StoreError.presetAlreadyExists(name)
        }
        SessionLogger.shared.log("[PRESET] \(replacedExisting ? "Replaced" : "Saved") '\(url.lastPathComponent)': \(architecture.architectureSummary)")
        return url
    }
}
