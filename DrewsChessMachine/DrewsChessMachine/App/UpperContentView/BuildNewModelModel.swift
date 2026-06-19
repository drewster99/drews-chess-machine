//
//  BuildNewModelModel.swift
//  DrewsChessMachine
//
//  @Observable backing model for the Build-New-Model screen (plan §10). Holds an
//  editable copy of every required topology field, exposes live
//  `parameterCount` / `architectureSummary` / validation, and resolves the
//  edited fields into a `NetworkArchitecture` for Build / Save-as-Preset.
//
//  Mirrors the `TrainingSettingsPopoverModel` pattern: per-field bindings +
//  validation, no business logic — the host wires Build/Save closures.
//

import Foundation
import Observation

@MainActor
@Observable
final class BuildNewModelModel {

    // Editable topology fields (defaults from the current preset; replaced by
    // `load(_:)`). The user's optional label override lives here, outside the
    // topology (plan §5a (b)). The effective `label` is computed below, so a
    // config edited away from a preset shows "Custom", not the preset's name.
    var labelOverride: String = ""
    var inputEncoding: InputEncoding
    /// The tower, edited group-by-group (ARCHITECTURE_EXPANSION_PLAN.md
    /// Feature 2 Phase B). Full fidelity: a loaded mixed tower round-trips
    /// through the editor without collapsing.
    var blockGroups: [BlockGroup]
    /// Stable per-group identities for the editor's `ForEach`, kept strictly
    /// parallel to `blockGroups` through every mutation below. Never persisted
    /// and not part of the architecture — `BlockGroup` stays a pure value type
    /// (Codable/Hashable equality must not depend on identity). Using these as
    /// the row id avoids index-as-identity, so reordering/removing a group
    /// re-associates rows correctly instead of by position.
    private(set) var groupIDs: [UUID]
    var stemConvKernelSize: Int
    var activationFunction: ActivationFunction
    var policyHeadStyle: PolicyHeadStyle
    var policyPreConvChannels: Int
    var valueHeadStyle: ValueHeadStyle
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int
    var computeDataType: ComputeDataType
    /// Feature skip (optional long concat skip). `featureSkipSource == .none`
    /// disables the whole feature. The fusion-mode and final-block toggles are
    /// shown but rejected by `validate()` for now (phase 2).
    var featureSkipSource: FeatureSkipSource
    var featureSkipFusion: FeatureSkipFusion
    var featureSkipToPolicyHead: Bool
    var featureSkipToValueHead: Bool
    var featureSkipToFinalBlock: Bool

    /// Name to save the current config under (Save-as-Preset). Defaults from the
    /// label, sanitized to a filename-safe slug.
    var saveAsName: String = ""

    init(_ named: NamedArchitecture = NamedArchitecture(label: "Custom", architecture: .current)) {
        let a = named.architecture
        self.labelOverride = ""
        self.inputEncoding = a.inputEncoding
        self.blockGroups = a.blockGroups
        self.groupIDs = a.blockGroups.map { _ in UUID() }
        self.stemConvKernelSize = a.stemConvKernelSize
        self.activationFunction = a.activationFunction
        self.policyHeadStyle = a.policyHeadStyle
        self.policyPreConvChannels = a.policyPreConvChannels
        self.valueHeadStyle = a.valueHeadStyle
        self.valueHeadConvChannels = a.valueHeadConvChannels
        self.valueHeadHiddenUnits = a.valueHeadHiddenUnits
        self.computeDataType = a.computeDataType
        self.featureSkipSource = a.featureSkipSource
        self.featureSkipFusion = a.featureSkipFusion
        self.featureSkipToPolicyHead = a.featureSkipToPolicyHead
        self.featureSkipToValueHead = a.featureSkipToValueHead
        self.featureSkipToFinalBlock = a.featureSkipToFinalBlock
    }

    /// Populate every field from a preset (the picker selection is derived from
    /// architecture equality, so no separate "selected" flag is needed).
    func load(_ named: NamedArchitecture) {
        let a = named.architecture
        labelOverride = ""
        inputEncoding = a.inputEncoding
        blockGroups = a.blockGroups
        groupIDs = a.blockGroups.map { _ in UUID() }
        stemConvKernelSize = a.stemConvKernelSize
        activationFunction = a.activationFunction
        policyHeadStyle = a.policyHeadStyle
        policyPreConvChannels = a.policyPreConvChannels
        valueHeadStyle = a.valueHeadStyle
        valueHeadConvChannels = a.valueHeadConvChannels
        valueHeadHiddenUnits = a.valueHeadHiddenUnits
        computeDataType = a.computeDataType
        featureSkipSource = a.featureSkipSource
        featureSkipFusion = a.featureSkipFusion
        featureSkipToPolicyHead = a.featureSkipToPolicyHead
        featureSkipToValueHead = a.featureSkipToValueHead
        featureSkipToFinalBlock = a.featureSkipToFinalBlock
    }

    /// The architecture described by the current fields.
    var architecture: NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: inputEncoding,
            blockGroups: blockGroups,
            stemConvKernelSize: stemConvKernelSize,
            activationFunction: activationFunction,
            policyHeadStyle: policyHeadStyle,
            policyPreConvChannels: policyPreConvChannels,
            valueHeadStyle: valueHeadStyle,
            valueHeadConvChannels: valueHeadConvChannels,
            valueHeadHiddenUnits: valueHeadHiddenUnits,
            computeDataType: computeDataType,
            featureSkipSource: featureSkipSource,
            featureSkipFusion: featureSkipFusion,
            featureSkipToPolicyHead: featureSkipToPolicyHead,
            featureSkipToValueHead: featureSkipToValueHead,
            featureSkipToFinalBlock: featureSkipToFinalBlock
        )
    }

    // MARK: Group manipulation (the editor's add/duplicate/remove/reorder)

    /// Total blocks across all groups (clamped ≥1 for ratios mid-edit).
    var totalBlocks: Int { max(1, blockGroups.reduce(0) { $0 + max(0, $1.count) }) }

    func duplicateGroup(at index: Int) {
        guard blockGroups.indices.contains(index) else { return }
        blockGroups.insert(blockGroups[index], at: index + 1)
        groupIDs.insert(UUID(), at: index + 1)
    }

    func removeGroup(at index: Int) {
        guard blockGroups.indices.contains(index), blockGroups.count > 1 else { return }
        blockGroups.remove(at: index)
        groupIDs.remove(at: index)
    }

    /// Move a group one slot toward the input (-1) or the heads (+1).
    func moveGroup(at index: Int, offset: Int) {
        let target = index + offset
        guard blockGroups.indices.contains(index),
              blockGroups.indices.contains(target) else { return }
        blockGroups.swapAt(index, target)
        groupIDs.swapAt(index, target)
    }

    /// The depth-appropriate ReZero α init for the current TOTAL block count
    /// (expanded across all groups): `1/√blocks`, which keeps the
    /// residual-stream variance ~O(1) at init (each of N blocks contributes
    /// ~α², so α = 1/√N → total ~1). Group α fields are seeded from the
    /// loaded preset and do NOT auto-track the block count, so building a
    /// deep net off a shallow preset silently keeps the shallow α — the
    /// mismatch flag + one-click apply below cover that.
    var recommendedRezeroAlphaInit: Float {
        1.0 / Float(totalBlocks).squareRoot()
    }

    /// True when the group at `index` has ReZero enabled and an α init that
    /// meaningfully differs from `recommendedRezeroAlphaInit`. Tolerance
    /// absorbs float round-trip noise (stored values like 0.447214).
    func rezeroAlphaInitMismatch(at index: Int) -> Bool {
        guard blockGroups.indices.contains(index) else { return false }
        let g = blockGroups[index]
        return g.useRezero && abs(g.rezeroAlphaInit - recommendedRezeroAlphaInit) > 1e-4
    }

    /// `nil` when the current fields form a valid architecture; otherwise the
    /// validation error text (Build is disabled while non-nil).
    var validationError: String? {
        do { try architecture.validate(); return nil }
        catch { return String(describing: error) }
    }

    var isValid: Bool { validationError == nil }

    /// Live parameter count (0 when invalid — the readout shows the error then).
    var parameterCount: Int { isValid ? architecture.parameterCount : 0 }

    /// Estimated on-disk weight bytes (Float32). Activation/training memory is
    /// batch-dependent and not estimated here (best-effort, plan §7).
    var estimatedWeightBytes: Int { parameterCount * MemoryLayout<Float>.size }

    /// Live one-line summary, or the validation error when invalid.
    var summary: String { isValid ? architecture.architectureSummary : (validationError ?? "invalid") }

    /// The preset (built-in or user-saved) whose architecture equals the current
    /// fields, if any — `nil` means "Custom".
    var matchedPreset: NamedArchitecture? {
        let a = architecture
        return ArchitecturePresetStore.allPresets().first(where: { $0.named.architecture == a })?.named
    }

    /// Effective display label: the user's override if set; else the matched
    /// preset's label; else "Custom". So editing away from a preset shows
    /// "Custom" rather than lingering on the preset's name.
    var label: String {
        labelOverride.isEmpty ? (matchedPreset?.label ?? "Custom") : labelOverride
    }

    /// A filename-safe slug derived from the effective label (so an edited config
    /// defaults to "custom", never the original preset's name).
    var defaultSaveName: String {
        let lowered = label.lowercased()
        let mapped = lowered.map { ch -> Character in
            (ch.isLetter || ch.isNumber) ? ch : "_"
        }
        let collapsed = String(mapped).split(separator: "_", omittingEmptySubsequences: true).joined(separator: "_")
        return collapsed.isEmpty ? "custom" : collapsed
    }
}
