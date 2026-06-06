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
    // `load(_:)`). `label` lives here, outside the topology (plan §5a (b)).
    var label: String
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

    /// The preset name currently selected in the picker (built-in or user), or
    /// nil = "Custom" once any field diverges. Display-only.
    var selectedPresetName: String?

    /// Name to save the current config under (Save-as-Preset). Defaults from the
    /// label, sanitized to a filename-safe slug.
    var saveAsName: String = ""

    init(_ named: NamedArchitecture = NamedArchitecture(label: "Custom", architecture: .current)) {
        let a = named.architecture
        self.label = named.label
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

    /// Populate every field from a preset and mark it selected.
    func load(name: String, _ named: NamedArchitecture) {
        let a = named.architecture
        label = named.label
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
        selectedPresetName = name
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

    /// A filename-safe slug derived from the label, for the Save-as-Preset default.
    var defaultSaveName: String {
        let lowered = label.lowercased()
        let mapped = lowered.map { ch -> Character in
            (ch.isLetter || ch.isNumber) ? ch : "_"
        }
        let collapsed = String(mapped).split(separator: "_", omittingEmptySubsequences: true).joined(separator: "_")
        return collapsed.isEmpty ? "custom" : collapsed
    }
}
