//
//  ArchitectureFormat.swift
//  DrewsChessMachine
//
//  The file-format version gate for every carrier of a `NetworkArchitecture`
//  (model safetensors — including a `.dcmsession`'s champion and trainer
//  files — user-saved `Presets/*.json`, `--architecture` files, and
//  `architecture.json`).
//
//  The rule (GitHub issues #1, #2, #7): a field added to the architecture may
//  resolve to "what the engine did before the field existed" ONLY for a file
//  written before the field existed. Files written by the version that
//  introduced the field, and by every later version, must carry it — a
//  missing field there is a hard decode error naming the field and the file,
//  never a silent default. Encoding always writes every field, so a new file
//  is self-describing and the "absent means legacy" rule can never apply to
//  it.
//
//  How the version reaches the decoder: each carrier knows its own version
//  (safetensors `__metadata__["dcm_format_version"]`, a preset's top-level
//  `format_version`) and hands the architecture decoder a `DecodeFormat`,
//  either through `JSONDecoder.userInfo` (`makeDecoder`) or explicitly
//  (`NetworkArchitecture.init(from:format:)`). A decode with NO format
//  supplied is treated as the CURRENT version — strict — so forgetting to
//  pass a version can only ever produce a loud error, never a silent legacy
//  resolution.
//
//  Legacy resolutions are collected on the `DecodeFormat`'s log rather than
//  logged from inside `init(from:)`, so the loader that knows the file's name
//  writes exactly one `[ARCH]` line per load (and display-only readers, such
//  as the model catalog, stay quiet).
//

import Foundation

enum ArchitectureFormat {

    /// The version every writer stamps today. Safetensors write it as the
    /// string `dcm_format_version`; presets and `architecture.json` write it
    /// as the integer `format_version`.
    static let currentVersion = 4

    /// First version whose block groups must carry `se_beta_init`
    /// (`BlockGroup.seBetaInit`). Files older than this resolve a missing
    /// field to `.glorot`, the only behavior that existed before it.
    static let seBetaInitRequiredFromVersion = 4

    /// The version reported for a carrier that predates version markers
    /// entirely — a safetensors file with no `dcm_format_version`, or a
    /// preset / `architecture.json` with no `format_version`. Every such file
    /// was written before `seBetaInitRequiredFromVersion`, so it is legacy.
    static let unversionedLegacyVersion = 3

    /// JSON key of the version marker in presets and `architecture.json`.
    static let jsonVersionKey = "format_version"

    /// `JSONDecoder.userInfo` key carrying a `DecodeFormat`.
    static let decodeFormatUserInfoKey: CodingUserInfoKey = {
        guard let key = CodingUserInfoKey(rawValue: "DrewsChessMachine.ArchitectureFormat.DecodeFormat") else {
            preconditionFailure("CodingUserInfoKey rejected a constant non-empty raw value")
        }
        return key
    }()

    // MARK: Errors

    enum FormatError: Error, CustomStringConvertible, LocalizedError, Equatable {
        /// A version-gated field is absent from a file whose version requires it.
        case missingRequiredField(field: String, location: String, formatVersion: Int, source: String)
        /// The file declares a version newer than this build understands.
        case unsupportedFutureVersion(version: Int, newestSupported: Int, source: String)
        /// The version marker is present but is not a positive integer.
        case unparseableVersion(value: String, source: String)

        var description: String {
            switch self {
            case .missingRequiredField(let field, let location, let version, let source):
                return "\(source): architecture field '\(field)' is missing at \(location); "
                    + "it is required in format v\(version) files (a file of format "
                    + "v\(ArchitectureFormat.seBetaInitRequiredFromVersion) or later must state it explicitly)"
            case .unsupportedFutureVersion(let version, let newest, let source):
                return "\(source): format v\(version) is newer than this build supports (newest: v\(newest))"
            case .unparseableVersion(let value, let source):
                return "\(source): format version '\(value)' is not a positive integer"
            }
        }

        var errorDescription: String? { description }
    }

    // MARK: Decode format

    /// Collects the legacy resolutions made while decoding one file, so the
    /// loader can write them as ONE log line. A reference type so every nested
    /// decoder (block groups inside the architecture inside a preset) appends
    /// to the same list.
    final class LegacyResolutionLog: Sendable {
        private let entries = SyncBox<[String]>([])

        init() {}

        func record(_ resolution: String) {
            entries.modify { $0.append(resolution) }
        }

        var resolutions: [String] { entries.value }
    }

    /// Which format version the architecture JSON being decoded belongs to,
    /// the human name of the file it came from (for errors and the log line),
    /// and where legacy resolutions are recorded.
    struct DecodeFormat: Sendable {
        let formatVersion: Int
        let source: String
        let legacyLog: LegacyResolutionLog

        init(formatVersion: Int, source: String, legacyLog: LegacyResolutionLog = LegacyResolutionLog()) {
            self.formatVersion = formatVersion
            self.source = source
            self.legacyLog = legacyLog
        }

        /// True when `se_beta_init` may be absent and resolves to `.glorot`.
        var allowsMissingSEBetaInit: Bool { formatVersion < ArchitectureFormat.seBetaInitRequiredFromVersion }

        /// The same file, re-stamped with the version a nested carrier
        /// declares (a preset's `format_version`), sharing the log.
        func withFormatVersion(_ version: Int) -> DecodeFormat {
            DecodeFormat(formatVersion: version, source: source, legacyLog: legacyLog)
        }

        /// The format used when a decoder carries none: the current version,
        /// strict. Missing version-gated fields throw rather than resolve.
        static func strictCurrent(decoder: Decoder) -> DecodeFormat {
            let path = decoder.codingPath.map(\.stringValue).joined(separator: ".")
            let source = path.isEmpty ? "architecture JSON" : "architecture JSON at \(path)"
            return DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: source)
        }

        /// The format attached to `decoder` by `makeDecoder`, else `strictCurrent`.
        static func from(_ decoder: Decoder) -> DecodeFormat {
            if let format = decoder.userInfo[ArchitectureFormat.decodeFormatUserInfoKey] as? DecodeFormat {
                return format
            }
            return strictCurrent(decoder: decoder)
        }

        /// One `[ARCH]` line describing every legacy resolution made while
        /// decoding this file, or nil when there were none.
        var legacyLogLine: String? {
            let resolutions = legacyLog.resolutions
            guard !resolutions.isEmpty else { return nil }
            return "[ARCH] legacy file (format v\(formatVersion)) \(source): "
                + resolutions.joined(separator: "; ")
        }

        /// Write `legacyLogLine` to the session log, once. Loaders call this
        /// exactly once per file they load.
        func logLegacyResolutions() {
            guard let line = legacyLogLine else { return }
            SessionLogger.shared.log(line)
        }
    }

    /// A `JSONDecoder` that decodes architecture JSON under `format`.
    static func makeDecoder(format: DecodeFormat) -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.userInfo[decodeFormatUserInfoKey] = format
        return decoder
    }

    // MARK: Version parsing

    /// The version recorded in a safetensors `dcm_format_version` value.
    /// Absent means the file predates version markers (legacy). A value that
    /// is not a positive integer, or is newer than this build, throws.
    static func safetensorsFormatVersion(metadataValue: String?, source: String) throws -> Int {
        guard let metadataValue else { return unversionedLegacyVersion }
        guard let version = Int(metadataValue), version > 0 else {
            throw FormatError.unparseableVersion(value: metadataValue, source: source)
        }
        return try requireSupported(version, source: source)
    }

    /// Rejects versions newer than `currentVersion` and non-positive ones.
    static func requireSupported(_ version: Int, source: String) throws -> Int {
        guard version > 0 else {
            throw FormatError.unparseableVersion(value: String(version), source: source)
        }
        guard version <= currentVersion else {
            throw FormatError.unsupportedFutureVersion(version: version, newestSupported: currentVersion, source: source)
        }
        return version
    }

    /// Human-readable location of the value `decoder` is decoding, e.g.
    /// `block_groups[1]`, for error messages.
    static func location(of decoder: Decoder) -> String {
        var rendered = ""
        for key in decoder.codingPath {
            if let index = key.intValue {
                rendered += "[\(index)]"
            } else {
                rendered += rendered.isEmpty ? key.stringValue : ".\(key.stringValue)"
            }
        }
        return rendered.isEmpty ? "the top level" : rendered
    }
}

// MARK: - Versioned architecture file (architecture.json)

/// A bare architecture file stamped with `format_version` next to the
/// architecture's own keys (the architecture decoder ignores the marker key,
/// and the marker reader ignores the architecture keys). Used by
/// `architecture.json`; presets carry the same marker inside
/// `NamedArchitecture`.
struct VersionedArchitectureFile: Codable, Sendable {
    let architecture: NetworkArchitecture

    init(architecture: NetworkArchitecture) {
        self.architecture = architecture
    }

    private enum MarkerKey: String, CodingKey {
        case formatVersion = "format_version"
    }

    init(from decoder: Decoder) throws {
        let marker = try decoder.container(keyedBy: MarkerKey.self)
        let base = ArchitectureFormat.DecodeFormat.from(decoder)
        let version: Int
        if let stated = try marker.decodeIfPresent(Int.self, forKey: .formatVersion) {
            version = try ArchitectureFormat.requireSupported(stated, source: base.source)
        } else {
            version = ArchitectureFormat.unversionedLegacyVersion
        }
        architecture = try NetworkArchitecture(from: decoder, format: base.withFormatVersion(version))
    }

    func encode(to encoder: Encoder) throws {
        try architecture.encode(to: encoder)
        var marker = encoder.container(keyedBy: MarkerKey.self)
        try marker.encode(ArchitectureFormat.currentVersion, forKey: .formatVersion)
    }
}
