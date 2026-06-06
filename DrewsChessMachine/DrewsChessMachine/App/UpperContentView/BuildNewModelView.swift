//
//  BuildNewModelView.swift
//  DrewsChessMachine
//
//  The Build-New-Model screen (plan §10): a preset picker (built-ins + user-saved)
//  plus every required topology field, with a live parameter-count / summary /
//  validation readout, "Save as Preset", and Build. The host presents this as a
//  sheet and wires `onBuild` to construct `ChessNetwork(arch:)`.
//
//  Fully functional today for the common v4-family / basic30 / WDL / bf16 case.
//  (basic20 input and scalar-tanh training are completed by the deferred Phase C/D
//  passes; the screen still lets you configure + build them.)
//

import SwiftUI

struct BuildNewModelView: View {

    @State private var model: BuildNewModelModel
    private let onBuild: (NetworkArchitecture) -> Void
    private let onCancel: () -> Void

    @State private var saveStatus: String?

    init(
        initial: NamedArchitecture = NamedArchitecture(label: "Custom", architecture: .current),
        onBuild: @escaping (NetworkArchitecture) -> Void,
        onCancel: @escaping () -> Void
    ) {
        _model = State(initialValue: BuildNewModelModel(initial))
        self.onBuild = onBuild
        self.onCancel = onCancel
    }

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            Text("New Network")
                .font(.title2.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding([.horizontal, .top])

            Form {
                Section("Preset") {
                    Picker("Start from", selection: presetSelection) {
                        Text("Custom").tag(String?.none)
                        ForEach(ArchitecturePresetStore.allPresets(), id: \.name) { entry in
                            Text(entry.named.label).tag(String?.some(entry.name))
                        }
                    }
                }

                Section("Input") {
                    enumPicker("Input encoding", $model.inputEncoding, InputEncoding.allCases)
                    Text(model.inputEncoding.planeDescription)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }

                Section("Tower") {
                    intField("Channels", $model.channels)
                    intField("Blocks", $model.numBlocks)
                    intField("Stem kernel size (odd)", $model.stemConvKernelSize)
                    enumPicker("Activation", $model.activationFunction, ActivationFunction.allCases)
                }

                Section("Residual block") {
                    enumPicker("Activation style", $model.blockActivationStyle, BlockActivationStyle.allCases)
                    enumPicker("Skip merge", $model.blockSkipMerge, BlockSkipMerge.allCases)
                    Toggle("Use ReZero", isOn: $model.blockUseRezero)
                    if model.blockUseRezero {
                        floatField("ReZero α init", $model.rezeroAlphaInit)
                    }
                    intField("Conv 1 kernel size (odd)", $model.blockConv1KernelSize)
                    intField("Conv 2 kernel size (odd)", $model.blockConv2KernelSize)
                    enumPicker("SE style", $model.blockSeStyle, SEStyle.allCases)
                    if model.blockSeStyle != .none {
                        intField("SE reduction ratio", $model.blockSeReductionRatio)
                    }
                }

                Section("Policy head") {
                    enumPicker("Policy style", $model.policyHeadStyle, PolicyHeadStyle.allCases)
                    if model.policyHeadStyle != .simpleConv {
                        intField("Policy pre-conv channels (K)", $model.policyPreConvChannels)
                    }
                }

                Section("Value head") {
                    enumPicker("Value style", $model.valueHeadStyle, ValueHeadStyle.allCases)
                    intField("Value conv channels", $model.valueHeadConvChannels)
                    intField("Value hidden units", $model.valueHeadHiddenUnits)
                }

                Section("Precision") {
                    enumPicker("Compute dtype", $model.computeDataType, ComputeDataType.allCases)
                }

                Section("Name") {
                    TextField("Label", text: $model.labelOverride, prompt: Text(model.label))
                }
            }
            .formStyle(.grouped)

            readout
            actionBar
        }
        .frame(minWidth: 560, minHeight: 620)
    }

    // MARK: Live readout

    @ViewBuilder private var readout: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let err = model.validationError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.callout)
            } else {
                HStack {
                    Text("Parameters:")
                    Text(model.parameterCount.formatted(.number))
                        .monospacedDigit().bold()
                    Text("(\(ByteCountFormatter.string(fromByteCount: Int64(model.estimatedWeightBytes), countStyle: .memory)) F32)")
                        .foregroundStyle(.secondary)
                }
                .font(.callout)
                Text(model.summary)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal)
    }

    // MARK: Actions

    @ViewBuilder private var actionBar: some View {
        @Bindable var model = model
        HStack(spacing: 12) {
            TextField("preset name", text: $model.saveAsName, prompt: Text(model.defaultSaveName))
                .frame(width: 160)
            Button("Save as Preset") {
                let name = model.saveAsName.isEmpty ? model.defaultSaveName : model.saveAsName
                do {
                    let url = try ArchitecturePresetStore.save(
                        name: name, label: model.label, architecture: model.architecture)
                    saveStatus = "Saved \(url.lastPathComponent)"
                } catch {
                    saveStatus = "\(error)"
                }
            }
            .disabled(!model.isValid)

            if let status = saveStatus {
                Text(status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }

            Spacer()
            Button("Cancel", role: .cancel) { onCancel() }
            Button("Build") { onBuild(model.architecture) }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.isValid)
        }
        .padding()
    }

    // MARK: Helpers

    /// Picker selection derived from architecture equality: shows the matching
    /// preset's name when the current fields equal a preset, else "Custom". This
    /// avoids the stale-selection bug where loading a preset's fields would
    /// immediately reset the label to "Custom".
    private var presetSelection: Binding<String?> {
        Binding(
            get: {
                let current = model.architecture
                return ArchitecturePresetStore.allPresets().first(where: { $0.named.architecture == current })?.name
            },
            set: { newName in
                guard let name = newName,
                      let entry = ArchitecturePresetStore.allPresets().first(where: { $0.name == name })
                else { return }
                model.load(entry.named)
            }
        )
    }

    @ViewBuilder
    private func enumPicker<E: CaseIterable & Hashable & RawRepresentable>(
        _ title: String, _ binding: Binding<E>, _ cases: [E]
    ) -> some View where E.RawValue == String {
        Picker(title, selection: binding) {
            ForEach(cases, id: \.self) { Text($0.rawValue).tag($0) }
        }
    }

    @ViewBuilder
    private func intField(_ title: String, _ binding: Binding<Int>) -> some View {
        TextField(title, value: binding, format: .number)
    }

    @ViewBuilder
    private func floatField(_ title: String, _ binding: Binding<Float>) -> some View {
        TextField(title, value: binding, format: .number)
    }
}
