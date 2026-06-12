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

            HStack(spacing: 0) {
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
                        intField("Stem kernel size (odd)", $model.stemConvKernelSize)
                        enumPicker("Tower-level activation", $model.activationFunction, ActivationFunction.allCases)
                        LabeledContent("Total blocks") {
                            Text("\(model.blockGroups.reduce(0) { $0 + $1.count })")
                                .monospacedDigit()
                        }
                    }

                    ForEach(model.blockGroups.indices, id: \.self) { i in
                        Section {
                            groupFields(i)
                        } header: {
                            groupHeader(i)
                        }
                    }

                    Section {
                        Button {
                            model.duplicateGroup(at: model.blockGroups.count - 1)
                        } label: {
                            Label("Add group", systemImage: "plus")
                        }
                    }

                    Section("Policy head") {
                        enumPicker("Policy style", $model.policyHeadStyle, PolicyHeadStyle.allCases)
                        Text(model.policyHeadStyle.styleDescription)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
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
                .frame(minWidth: 540)

                Divider()

                diagramPane
                    .frame(width: 420)
            }

            readout
            actionBar
        }
        .frame(minWidth: 1000, minHeight: 700)
    }

    // MARK: Block-group editor

    @ViewBuilder
    private func groupHeader(_ i: Int) -> some View {
        HStack(spacing: 6) {
            Text("Group \(i + 1)")
            Spacer()
            Button {
                model.moveGroup(at: i, offset: -1)
            } label: {
                Image(systemName: "chevron.up")
            }
            .disabled(i == 0)
            .help("Move this group one step toward the input")
            Button {
                model.moveGroup(at: i, offset: 1)
            } label: {
                Image(systemName: "chevron.down")
            }
            .disabled(i == model.blockGroups.count - 1)
            .help("Move this group one step toward the heads")
            Button {
                model.duplicateGroup(at: i)
            } label: {
                Image(systemName: "plus.square.on.square")
            }
            .help("Insert a copy of this group below it")
            Button {
                model.removeGroup(at: i)
            } label: {
                Image(systemName: "trash")
            }
            .disabled(model.blockGroups.count == 1)
            .help("Remove this group")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
    }

    @ViewBuilder
    private func groupFields(_ i: Int) -> some View {
        @Bindable var model = model
        intField("Blocks (count)", $model.blockGroups[i].count)
        intField("Channels", $model.blockGroups[i].channels)
        intField("Conv 1 kernel size (odd)", $model.blockGroups[i].conv1KernelSize)
        intField("Conv 2 kernel size (odd)", $model.blockGroups[i].conv2KernelSize)
        enumPicker("SE style", $model.blockGroups[i].seStyle, SEStyle.allCases)
        if model.blockGroups[i].seStyle != .none {
            intField("SE reduction ratio", $model.blockGroups[i].seReductionRatio)
        }
        enumPicker("Activation", $model.blockGroups[i].activationFunction, ActivationFunction.allCases)
        enumPicker("Activation style", $model.blockGroups[i].activationStyle, BlockActivationStyle.allCases)
        enumPicker("Skip merge", $model.blockGroups[i].skipMerge, BlockSkipMerge.allCases)
        Toggle("Use ReZero", isOn: $model.blockGroups[i].useRezero)
        if model.blockGroups[i].useRezero {
            // The α init is seeded from the loaded preset and does NOT
            // auto-track the TOTAL block count, so a deep net built off a
            // shallow preset silently keeps the shallow α. Flag the mismatch
            // and offer a one-click snap to 1/√(total blocks) rather than
            // silently overwriting a deliberately-set value.
            HStack {
                floatField("ReZero α init", $model.blockGroups[i].rezeroAlphaInit)
                if model.rezeroAlphaInitMismatch(at: i) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Button(String(format: "Use 1/√%d = %.3f", model.totalBlocks, model.recommendedRezeroAlphaInit)) {
                        model.blockGroups[i].rezeroAlphaInit = model.recommendedRezeroAlphaInit
                    }
                    .controlSize(.small)
                    .help("ReZero α init doesn't match the depth-appropriate value (1/√ total blocks); click to apply.")
                }
            }
        }
        floatField("Dropout multiplier", $model.blockGroups[i].dropoutMultiplier)
    }

    // MARK: Diagram pane (live-updating, renders the draft)

    @ViewBuilder
    private var diagramPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Architecture")
                .font(.headline)
                .padding([.horizontal, .top], 12)
                .padding(.bottom, 6)
            if model.isValid {
                ScrollView(.vertical) {
                    ArchitectureDiagramView(architecture: model.architecture)
                        .padding(12)
                        .frame(maxWidth: .infinity)
                }
            } else {
                VStack {
                    Spacer()
                    Label(model.validationError ?? "invalid configuration",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.orange)
                        .padding()
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            }
        }
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
