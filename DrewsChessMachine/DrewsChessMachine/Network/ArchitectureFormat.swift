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
//  Version history (each version requires every field the earlier ones did):
//  - v3 and unversioned: legacy; no version-gated architecture fields.
//  - v4: block groups must state `se_beta_init` (issue #7). Older files
//    resolve it to `glorot`.
//  - v5: block groups must state `se_activation` (issue #2). Older files
//    resolve it to the group's own `activation_function`, which is the
//    activation the SE FC1 used before the field existed.
//  - v6: block groups must state `rezero_alpha_cap`, the asymptote of the
//    forward ReZero soft bound `C·tanh(α/C)`. Older files resolve it to
//    `rezero_alpha_init × NetworkArchitecture.rezeroTanhCeilingMultiple`,
//    which is exactly the C the engine computed before the field existed.
//  - v7: no new architecture field; every safetensors model file must
//    carry a `dcm_lineage` record (`LineageRecord`). Older files load with
//    their lineage reported as unrecorded.
//  - v8: the init-neutral options (determinism plan B2, decision D-4):
//    block groups must state `se_gamma_bias_init`, `branch_output_init` and
//    `skip_projection_init`; the architecture must state
//    `policy_head_final_init`, `value_head_final_init` and
//    `value_head_draw_prior`. Older files resolve each to its standard value
//    — the init every model was built with before the options existed.
//  - v9: the architecture must state `stem_activation`,
//    `tower_end_activation`, `feature_skip_activation`,
//    `policy_head_activation`, `value_head_conv_activation`,
//    `value_head_fc1_hidden_activation`; each is `does_not_apply` exactly
//    when the topology lacks the site. Older files resolve an existing site
//    to their own top-level `activation_function`, the activation every one
//    of those sites used before the fields existed, and an absent site to
//    `does_not_apply`. A v9+ block-groups architecture must not state the
//    retired top-level `activation_function`.
//

import Foundation

enum ArchitectureFormat {

    /// The version every writer stamps today. Safetensors write it as the
    /// string `dcm_format_version`; presets and `architecture.json` write it
    /// as the integer `format_version`.
    static let currentVersion = 9

    /// First version whose block groups must carry `se_beta_init`
    /// (`BlockGroup.seBetaInit`). Files older than this resolve a missing
    /// field to `.glorot`, the only behavior that existed before it.
    static let seBetaInitRequiredFromVersion = 4

    /// First version whose block groups must carry `se_activation`
    /// (`BlockGroup.seActivation`, GitHub issue #2). Files older than this
    /// resolve a missing field to the group's own `activation_function` —
    /// before the field existed the SE FC1 always used the group's
    /// activation, so that resolution rebuilds exactly the graph the file
    /// was trained with (ReLU for every model saved before SiLU/GELU/leaky
    /// ReLU existed, and the group's activation for any model that used one).
    static let seActivationRequiredFromVersion = 5

    /// First version whose block groups must carry `rezero_alpha_cap`
    /// (`BlockGroup.rezeroAlphaCap`). Files older than this resolve a missing
    /// field to `rezero_alpha_init × NetworkArchitecture.rezeroTanhCeilingMultiple`
    /// (`BlockGroup.legacyRezeroAlphaCap`) — before the field existed the
    /// forward's soft-bound asymptote was always derived from the init that
    /// way, so the resolution rebuilds exactly the graph the file was trained
    /// with, and the architecture compares (and hashes) equal to what the same
    /// file decoded to before the field existed.
    static let rezeroAlphaCapRequiredFromVersion = 6

    /// First version whose safetensors model files must carry a
    /// `dcm_lineage` record (`LineageRecord`). A file older than this has
    /// no lineage, and its lineage is reported as unrecorded — never
    /// reconstructed.
    static let lineageRequiredFromVersion = 7

    /// First version whose architectures must carry the init-neutral options
    /// (`BlockGroup.seGammaBiasInit`, `branchOutputInit`, `skipProjectionInit`;
    /// `NetworkArchitecture.policyHeadFinalInit`, `valueHeadFinalInit`,
    /// `valueHeadDrawPrior`). Files older than this resolve each missing
    /// option to its standard value — every model built before the options
    /// existed was initialized that way, so the resolution describes exactly
    /// the init it had, and the architecture compares equal to what the same
    /// file decoded to before.
    static let initOptionsRequiredFromVersion = 8

    /// First version whose architectures must carry the six
    /// architecture-level site activations (`NetworkArchitecture.stemActivation`,
    /// `towerEndActivation`, `featureSkipActivation`, `policyHeadActivation`,
    /// `valueHeadConvActivation`, `valueHeadFC1HiddenActivation`) and must not
    /// carry the top-level `activation_function` they replace. Files older
    /// than this resolve each missing site from that `activation_function`
    /// (a site the topology has — every one of them used it before the split)
    /// or to `does_not_apply` (a site it lacks), so the architecture builds
    /// the identical graph and compares equal to what the same file decoded
    /// to before.
    static let siteActivationsRequiredFromVersion = 9

    /// The version reported for a carrier that predates version markers
    /// entirely — a safetensors file with no `dcm_format_version`, or a
    /// preset / `architecture.json` with no `format_version`. Every such file
    /// was written before `seBetaInitRequiredFromVersion` (and therefore
    /// before every later gate), so it is legacy.
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
        /// A field a later version replaced, stated in a file of that version
        /// or newer (where it would otherwise be silently ignored).
        case retiredField(field: String, location: String, formatVersion: Int, source: String, replacedBy: [String])
        /// An architecture-level site activation disagrees with whether the
        /// topology has the site (`NetworkArchitecture.activationSiteMismatch`).
        case activationSiteMismatch(ActivationSiteMismatch, location: String, formatVersion: Int, source: String)
        /// `does_not_apply` in a field whose site always exists: a block
        /// group's `activation_function` / `se_activation`, or the uniform
        /// tower's `activation_function`, which those come from.
        case doesNotApplyAtAnAlwaysPresentSite(field: String, location: String, formatVersion: Int, source: String)
        /// A file older than v9 that states neither some site activations nor
        /// the top-level `activation_function` they resolve from.
        case legacyActivationFunctionMissing(unresolvedSites: [String], location: String, formatVersion: Int, source: String)

        var description: String {
            switch self {
            case .missingRequiredField(let field, let location, let version, let source):
                return "\(source): architecture field '\(field)' is missing at \(location); "
                    + "a format v\(version) file must state it explicitly (only files written before "
                    + "the field existed may omit it)"
            case .unsupportedFutureVersion(let version, let newest, let source):
                return "\(source): format v\(version) is newer than this build supports (newest: v\(newest))"
            case .unparseableVersion(let value, let source):
                return "\(source): format version '\(value)' is not a positive integer"
            case .retiredField(let field, let location, let version, let source, let replacedBy):
                return "\(source): architecture field '\(field)' at \(location) was retired in format "
                    + "v\(ArchitectureFormat.siteActivationsRequiredFromVersion), and a format v\(version) file must "
                    + "not state it; it is replaced by \(replacedBy.joined(separator: ", "))"
            case .activationSiteMismatch(let mismatch, let location, let version, let source):
                return "\(source): format v\(version) architecture at \(location): \(mismatch.description)"
            case .doesNotApplyAtAnAlwaysPresentSite(let field, let location, let version, let source):
                return "\(source): format v\(version) architecture field '\(field)' at \(location) is "
                    + "'\(ActivationFunction.doesNotApply.rawValue)', but that site always exists: "
                    + "choose one of \(ActivationFunction.functionList)"
            case .legacyActivationFunctionMissing(let unresolved, let location, let version, let source):
                return "\(source): format v\(version) architecture at \(location) states neither "
                    + "\(unresolved.joined(separator: ", ")) nor the top-level 'activation_function' they "
                    + "resolve from; a file older than v\(ArchitectureFormat.siteActivationsRequiredFromVersion) "
                    + "must state one or the other"
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

        /// True when `se_activation` may be absent and resolves to the
        /// group's own `activation_function`.
        var allowsMissingSEActivation: Bool { formatVersion < ArchitectureFormat.seActivationRequiredFromVersion }

        /// True when `rezero_alpha_cap` may be absent and resolves to the
        /// group's `rezero_alpha_init × rezeroTanhCeilingMultiple`.
        var allowsMissingRezeroAlphaCap: Bool { formatVersion < ArchitectureFormat.rezeroAlphaCapRequiredFromVersion }
        /// True when the init-neutral options may be absent and resolve to
        /// their standard values.
        var allowsMissingInitOptions: Bool { formatVersion < ArchitectureFormat.initOptionsRequiredFromVersion }
        /// True when the six site activations may be absent and resolve from
        /// the file's top-level `activation_function` (and when that key is
        /// not retired).
        var allowsMissingSiteActivations: Bool { formatVersion < ArchitectureFormat.siteActivationsRequiredFromVersion }

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

    // MARK: Init-neutral options

    /// Decodes one init-neutral option: the stated value; else, for a file
    /// older than `initOptionsRequiredFromVersion` — or one whose structure
    /// already proves it older (`legacyByConstruction`: the pre-block-groups
    /// uniform-tower keys, which no writer of a version with these options
    /// emits) — `standard` (recorded on the format's legacy log as
    /// `<location>.<key> := <rendered standard>`); else a
    /// `missingRequiredField` error. The one rule every init option —
    /// block-group and head — decodes through.
    static func decodeInitOption<Value: Decodable, Key: CodingKey>(
        _ type: Value.Type,
        key: Key,
        in container: KeyedDecodingContainer<Key>,
        decoder: Decoder,
        format: DecodeFormat,
        legacyByConstruction: Bool,
        standard: Value,
        rendered: (Value) -> String
    ) throws -> Value {
        if let stated = try container.decodeIfPresent(type, forKey: key) {
            return stated
        }
        guard legacyByConstruction || format.allowsMissingInitOptions else {
            throw FormatError.missingRequiredField(
                field: key.stringValue,
                location: location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
        let prefix = decoder.codingPath.isEmpty ? "" : "\(location(of: decoder))."
        format.legacyLog.record("\(prefix)\(key.stringValue) := \(rendered(standard))")
        return standard
    }

    // MARK: Site activations

    /// One architecture-level site activation as pass one of
    /// `NetworkArchitecture.init(from:format:)` decodes it: the value, and
    /// whether it was resolved from the file's top-level
    /// `activation_function` (so pass two may still turn it into
    /// `does_not_apply` and must log it) rather than stated.
    struct DecodedSiteActivation: Sendable {
        let value: ActivationFunction
        let wasResolvedFromLegacy: Bool
    }

    /// Decodes one site activation: the stated value, at any version (real
    /// pre-v9 files never state it, but the tests' re-stamped current
    /// encodes do, exactly as `decodeInitOption` allows); else, for a file
    /// older than `siteActivationsRequiredFromVersion` or in the uniform-tower
    /// form (`legacyByConstruction`), the file's `legacyTowerActivation`,
    /// marked resolved; else `missingRequiredField`. A file allowed to
    /// resolve that has no `activation_function` either throws
    /// `legacyActivationFunctionMissing`, naming every site key it lacks
    /// (`allSiteKeys` absent from the container), so the error lists all of
    /// them however many there are and makes nothing up.
    static func decodeSiteActivation<Key: CodingKey>(
        key: Key,
        in container: KeyedDecodingContainer<Key>,
        decoder: Decoder,
        format: DecodeFormat,
        legacyByConstruction: Bool,
        legacyTowerActivation: ActivationFunction?,
        allSiteKeys: [Key]
    ) throws -> DecodedSiteActivation {
        if let stated = try container.decodeIfPresent(ActivationFunction.self, forKey: key) {
            return DecodedSiteActivation(value: stated, wasResolvedFromLegacy: false)
        }
        guard legacyByConstruction || format.allowsMissingSiteActivations else {
            throw FormatError.missingRequiredField(
                field: key.stringValue,
                location: location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
        guard let legacyTowerActivation else {
            throw FormatError.legacyActivationFunctionMissing(
                unresolvedSites: allSiteKeys.filter { !container.contains($0) }.map(\.stringValue),
                location: location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
        return DecodedSiteActivation(value: legacyTowerActivation, wasResolvedFromLegacy: true)
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
