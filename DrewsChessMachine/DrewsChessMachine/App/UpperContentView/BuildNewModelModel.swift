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
    var channels: Int
    var numBlocks: Int
    var stemConvKernelSize: Int
    var activationFunction: ActivationFunction
    var blockActivationStyle: BlockActivationStyle
    var blockSkipMerge: BlockSkipMerge
    var blockUseRezero: Bool
    var rezeroAlphaInit: Float
    var blockConv1KernelSize: Int
    var blockConv2KernelSize: Int
    var blockSeStyle: SEStyle
    var blockSeReductionRatio: Int
    var policyHeadStyle: PolicyHeadStyle
    var policyPreConvChannels: Int
    var valueHeadStyle: ValueHeadStyle
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int
    var computeDataType: ComputeDataType

    /// Name to save the current config under (Save-as-Preset). Defaults from the
    /// label, sanitized to a filename-safe slug.
    var saveAsName: String = ""

    init(_ named: NamedArchitecture = NamedArchitecture(label: "Custom", architecture: .current)) {
        let a = named.architecture
        self.labelOverride = ""
        self.inputEncoding = a.inputEncoding
        self.channels = a.channels
        self.numBlocks = a.numBlocks
        self.stemConvKernelSize = a.stemConvKernelSize
        self.activationFunction = a.activationFunction
        self.blockActivationStyle = a.blockActivationStyle
        self.blockSkipMerge = a.blockSkipMerge
        self.blockUseRezero = a.blockUseRezero
        self.rezeroAlphaInit = a.rezeroAlphaInit
        self.blockConv1KernelSize = a.blockConv1KernelSize
        self.blockConv2KernelSize = a.blockConv2KernelSize
        self.blockSeStyle = a.blockSeStyle
        self.blockSeReductionRatio = a.blockSeReductionRatio
        self.policyHeadStyle = a.policyHeadStyle
        self.policyPreConvChannels = a.policyPreConvChannels
        self.valueHeadStyle = a.valueHeadStyle
        self.valueHeadConvChannels = a.valueHeadConvChannels
        self.valueHeadHiddenUnits = a.valueHeadHiddenUnits
        self.computeDataType = a.computeDataType
    }

    /// Populate every field from a preset (the picker selection is derived from
    /// architecture equality, so no separate "selected" flag is needed).
    func load(_ named: NamedArchitecture) {
        let a = named.architecture
        labelOverride = ""
        inputEncoding = a.inputEncoding
        channels = a.channels
        numBlocks = a.numBlocks
        stemConvKernelSize = a.stemConvKernelSize
        activationFunction = a.activationFunction
        blockActivationStyle = a.blockActivationStyle
        blockSkipMerge = a.blockSkipMerge
        blockUseRezero = a.blockUseRezero
        rezeroAlphaInit = a.rezeroAlphaInit
        blockConv1KernelSize = a.blockConv1KernelSize
        blockConv2KernelSize = a.blockConv2KernelSize
        blockSeStyle = a.blockSeStyle
        blockSeReductionRatio = a.blockSeReductionRatio
        policyHeadStyle = a.policyHeadStyle
        policyPreConvChannels = a.policyPreConvChannels
        valueHeadStyle = a.valueHeadStyle
        valueHeadConvChannels = a.valueHeadConvChannels
        valueHeadHiddenUnits = a.valueHeadHiddenUnits
        computeDataType = a.computeDataType
    }

    /// The architecture described by the current fields.
    var architecture: NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: inputEncoding,
            channels: channels,
            numBlocks: numBlocks,
            stemConvKernelSize: stemConvKernelSize,
            activationFunction: activationFunction,
            blockActivationStyle: blockActivationStyle,
            blockSkipMerge: blockSkipMerge,
            blockUseRezero: blockUseRezero,
            rezeroAlphaInit: rezeroAlphaInit,
            blockConv1KernelSize: blockConv1KernelSize,
            blockConv2KernelSize: blockConv2KernelSize,
            blockSeStyle: blockSeStyle,
            blockSeReductionRatio: blockSeReductionRatio,
            policyHeadStyle: policyHeadStyle,
            policyPreConvChannels: policyPreConvChannels,
            valueHeadStyle: valueHeadStyle,
            valueHeadConvChannels: valueHeadConvChannels,
            valueHeadHiddenUnits: valueHeadHiddenUnits,
            computeDataType: computeDataType
        )
    }

    /// The depth-appropriate ReZero α init for the current block count:
    /// `1/√blocks`, which keeps the residual-stream variance ~O(1) at init
    /// (each of N blocks contributes ~α², so α = 1/√N → total ~1). The field is
    /// seeded from the loaded preset and does NOT auto-track the block count, so
    /// building a deep net off a shallow preset silently keeps the shallow α
    /// (e.g. a 50-block net left at the 5-block 0.447). `numBlocks` is clamped
    /// to ≥1 to avoid a divide-by-zero while the field is mid-edit.
    var recommendedRezeroAlphaInit: Float {
        1.0 / Float(max(1, numBlocks)).squareRoot()
    }

    /// True when ReZero is enabled and the α init meaningfully differs from the
    /// depth-appropriate `recommendedRezeroAlphaInit`. Drives the mismatch
    /// highlight + one-click "apply" affordance in the Build-New-Model screen.
    /// Tolerance absorbs float round-trip noise (stored values like 0.447214).
    var rezeroAlphaInitMismatch: Bool {
        blockUseRezero && abs(rezeroAlphaInit - recommendedRezeroAlphaInit) > 1e-4
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
