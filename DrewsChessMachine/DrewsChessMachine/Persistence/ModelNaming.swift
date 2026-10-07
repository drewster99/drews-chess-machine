//
//  ModelNaming.swift
//  DrewsChessMachine
//
//  What a model was called when it was made, and the architecture preset its
//  topology started from (MODEL_NAMING_PLAN.md).
//
//  Why it lives in the lineage record (`LineageRecord.modelNaming`, schema 4)
//  rather than beside each network's `ModelID`. The owner's rule is that the
//  name rides with the model: through saves, resumes, branches, promotions
//  and trainer forks, and through a derive unless it is renamed. That is
//  exactly how the record's run-level facts already travel — every model file
//  written since format v7 carries one record, built in one place
//  (`LineageTracker`), and each start kind (fresh, branch, resume, untrained
//  copy) hands its facts to the next file by rules that are tested once. A
//  second carrier beside the `ModelID` would have to be copied by hand at
//  every place a weight copy assigns an identifier, and one missed site would
//  silently show the previous model's name.
//
//  `name` is what the person typed (the New Network screen's Name field,
//  `--name`); nil means none was given — never a placeholder such as
//  "Custom", which would read as a name nobody chose. `presetStart` is the
//  preset the topology began from and whether it was edited away from it, so
//  "from v4_5block_7x7 (edited)" says where a hand-tuned architecture came
//  from without claiming it is the preset.
//

import Foundation

struct ModelNaming: Codable, Equatable, Sendable {
    /// The name given when the model was made (or renamed by a derive);
    /// nil when none was given. Validated by `validatedName(_:)`.
    let name: String?
    /// The preset the topology started from: `recorded(PresetStart)` for a
    /// topology begun from a preset, `recorded(nil)` for one begun from no
    /// preset (New Network's "Custom", an architecture file given by path),
    /// unrecorded only for a derive that names a source whose record has no
    /// naming (the name is new; where the topology began is not known).
    let presetStart: LineageRecord.Recorded<PresetStart?>

    struct PresetStart: Codable, Equatable, Sendable {
        /// The preset's key: a built-in `NetworkArchitecture.Preset` raw
        /// value or a user preset's file stem.
        let preset: String
        /// The topology differs from the preset's (an edit on the New
        /// Network screen, or a derive / graft that changed the
        /// architecture).
        let edited: Bool

        enum CodingKeys: String, CodingKey {
            case preset
            case edited
        }

        init(preset: String, edited: Bool) throws {
            guard !preset.isEmpty else { throw NamingError.emptyPresetName }
            self.preset = preset
            self.edited = edited
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(preset: try c.decode(String.self, forKey: .preset),
                          edited: try c.decode(Bool.self, forKey: .edited))
        }
    }

    enum CodingKeys: String, CodingKey {
        case name
        case presetStart = "preset_start"
    }

    enum NamingError: Error, Equatable, CustomStringConvertible, LocalizedError {
        case emptyName
        case nameTooLong(length: Int)
        case nameHasControlCharacter
        case emptyPresetName

        var description: String {
            switch self {
            case .emptyName:
                return "a model name must not be empty or only spaces"
            case .nameTooLong(let length):
                return "a model name is at most \(ModelNaming.maximumNameLength) characters (this one has \(length))"
            case .nameHasControlCharacter:
                return "a model name must not contain line breaks, tabs or other control characters"
            case .emptyPresetName:
                return "a preset start must name its preset"
            }
        }

        var errorDescription: String? { description }
    }

    /// Long enough for a descriptive name, short enough for a title bar.
    static let maximumNameLength = 120

    /// `raw` with surrounding whitespace removed, or an error naming why it
    /// can't be a model name. One rule for every source of a name (the New
    /// Network field, `--new-model --name`, `--derive-model --name`, a
    /// decoded record).
    static func validatedName(_ raw: String) throws -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { throw NamingError.emptyName }
        guard trimmed.count <= maximumNameLength else { throw NamingError.nameTooLong(length: trimmed.count) }
        // General category Cc only (line breaks, tabs, …): `CharacterSet
        // .controlCharacters` also holds format characters such as the
        // zero-width joiner inside emoji sequences.
        guard !trimmed.unicodeScalars.contains(where: { $0.properties.generalCategory == .control }) else {
            throw NamingError.nameHasControlCharacter
        }
        return trimmed
    }

    /// `name` is validated (`validatedName`) and stored trimmed.
    init(name: String?, presetStart: LineageRecord.Recorded<PresetStart?>) throws {
        self.name = try name.map(Self.validatedName)
        self.presetStart = presetStart
    }

    /// No name given and no preset chosen — the New Network screen's
    /// opening state, and the headless `--train` build of it.
    static let unnamedWithoutPreset = ModelNaming(unvalidatedName: nil, presetStart: .recorded(nil))

    /// For a name with nothing to validate (nil).
    private init(unvalidatedName name: String?, presetStart: LineageRecord.Recorded<PresetStart?>) {
        self.name = name
        self.presetStart = presetStart
    }

    /// The naming of a model whose topology began from `preset` with no
    /// name given, unedited — a fresh CLI run from a preset.
    static func unnamed(fromPreset preset: String) throws -> ModelNaming {
        try ModelNaming(name: nil, presetStart: .recorded(try PresetStart(preset: preset, edited: false)))
    }

    /// Strict like the rest of the record: both keys are required (`name`
    /// may be `null`), and a decoded name must pass the same validation as
    /// a typed one, unchanged — a stored name with surrounding spaces is a
    /// writer bug, not something to tidy on read.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let storedName = try c.decode(String?.self, forKey: .name)
        if let storedName {
            guard try Self.validatedName(storedName) == storedName else {
                throw DecodingError.dataCorruptedError(
                    forKey: .name, in: c, debugDescription: "model name '\(storedName)' has surrounding whitespace")
            }
        }
        self.name = storedName
        self.presetStart = try c.decode(LineageRecord.Recorded<PresetStart?>.self, forKey: .presetStart)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(name, forKey: .name)
        try c.encode(presetStart, forKey: .presetStart)
    }

    /// The naming of a `--derive-model` copy of a model whose record states
    /// `source` (a graft is `ofGraft`).
    ///
    /// - The source's name and preset start carry over.
    /// - A copy that changes the architecture is no longer its preset's
    ///   topology, so an unedited recorded preset start becomes edited.
    /// - `newName` (a derive's `--name`) replaces the name and keeps the
    ///   preset start; for a source with no recorded naming the preset
    ///   start is then unrecorded — the name is new, where the topology
    ///   began is still unknown.
    /// - With no `newName`, a source with no recorded naming stays
    ///   unrecorded.
    static func ofDerive(of source: LineageRecord.Recorded<ModelNaming>, architectureChanged: Bool,
                         renamedTo newName: String?) throws -> LineageRecord.Recorded<ModelNaming> {
        switch source {
        case .unrecorded:
            guard let newName else { return .unrecorded }
            return .recorded(try ModelNaming(name: newName, presetStart: .unrecorded))
        case .recorded(let naming):
            var presetStart = naming.presetStart
            if architectureChanged, case .recorded(let start?) = presetStart, !start.edited {
                presetStart = .recorded(try PresetStart(preset: start.preset, edited: true))
            }
            return .recorded(try ModelNaming(name: newName ?? naming.name, presetStart: presetStart))
        }
    }

    /// The naming of a graft of a model whose record states `source` onto
    /// a target architecture: the output's topology is the target's, so
    /// its preset start is the target preset, unedited (`targetPreset`;
    /// nil for a target given as an architecture file, which is no
    /// preset). The name is `newName`, else the source's; a source with no
    /// recorded naming and no `newName` leaves the name unknown, so the
    /// naming stays unrecorded.
    static func ofGraft(of source: LineageRecord.Recorded<ModelNaming>, targetPreset: String?,
                        renamedTo newName: String?) throws -> LineageRecord.Recorded<ModelNaming> {
        let presetStart: LineageRecord.Recorded<PresetStart?> =
            .recorded(try targetPreset.map { try PresetStart(preset: $0, edited: false) })
        if let newName {
            return .recorded(try ModelNaming(name: newName, presetStart: presetStart))
        }
        guard case .recorded(let naming) = source else { return .unrecorded }
        return .recorded(try ModelNaming(name: naming.name, presetStart: presetStart))
    }

    // MARK: Display

    /// `name · preset p (edited)`, each part only when the naming states
    /// it; nil when it states neither (no name given, no preset, or not
    /// recorded). The title bar's leading text.
    static func headerText(_ naming: LineageRecord.Recorded<ModelNaming>) -> String? {
        guard case .recorded(let value) = naming else { return nil }
        var parts: [String] = []
        if let name = value.name { parts.append(name) }
        if case .recorded(let start?) = value.presetStart { parts.append("preset " + start.displayText) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// `headerText`, or what the naming says when it states neither part:
    /// "no name or preset" / "not recorded". For places that always show a
    /// value (the auto-resume sheet).
    static func compactText(_ naming: LineageRecord.Recorded<ModelNaming>) -> String {
        if let header = headerText(naming) { return header }
        switch naming {
        case .unrecorded: return "not recorded"
        case .recorded: return "no name or preset"
        }
    }

    /// The About popover's Name row.
    static func nameRowText(_ naming: LineageRecord.Recorded<ModelNaming>) -> String {
        switch naming {
        case .unrecorded: return "not recorded (made before models carried names)"
        case .recorded(let value): return value.name ?? "none given"
        }
    }

    /// The About popover's Preset row.
    static func presetRowText(_ naming: LineageRecord.Recorded<ModelNaming>) -> String {
        guard case .recorded(let value) = naming else { return "not recorded (made before models carried names)" }
        switch value.presetStart {
        case .unrecorded: return "not recorded (renamed from a model made before names)"
        case .recorded(nil): return "none (custom architecture)"
        case .recorded(let start?): return start.displayText
        }
    }

    /// `name "my-net", preset v4_5block_7x7 (edited)` / `no name, no
    /// preset`: what a model about to be made will record (the New Network
    /// screen).
    static func recordedSummaryText(_ naming: ModelNaming) -> String {
        let nameText = naming.name.map { "name \"\($0)\"" } ?? "no name"
        let presetText: String
        switch naming.presetStart {
        case .unrecorded: presetText = "preset not recorded"
        case .recorded(nil): presetText = "no preset"
        case .recorded(let start?): presetText = "preset " + start.displayText
        }
        return "\(nameText), \(presetText)"
    }

    /// One-line form for log lines: `name="…" preset=p edited=true`,
    /// with `none` / `unrecorded` for absent parts.
    static func logText(_ naming: LineageRecord.Recorded<ModelNaming>) -> String {
        guard case .recorded(let value) = naming else { return "naming=unrecorded" }
        let nameText = value.name.map { "\"\($0)\"" } ?? "none"
        let presetText: String
        switch value.presetStart {
        case .unrecorded: presetText = "unrecorded"
        case .recorded(nil): presetText = "none"
        case .recorded(let start?): presetText = "\(start.preset) edited=\(start.edited)"
        }
        return "name=\(nameText) preset=\(presetText)"
    }
}

extension ModelNaming.PresetStart {
    /// `v4_5block_7x7` or `v4_5block_7x7 (edited)`.
    var displayText: String { edited ? "\(preset) (edited)" : preset }
}
