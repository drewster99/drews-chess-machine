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
                            ForEach(model.availablePresets, id: \.name) { entry in
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

                    // Iterate by stable group identity (not array index), so a
                    // reorder/remove re-associates rows by identity rather than
                    // by position. `i` is recomputed each body build from the
                    // current arrays, so it is never stale for the bindings.
                    ForEach(Array(zip(model.groupIDs, model.blockGroups.indices)), id: \.0) { (_, i) in
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

                    Section("Feature skip") {
                        enumPicker("Source", $model.featureSkipSource, FeatureSkipSource.allCases)
                        if model.featureSkipSource != .none {
                            Toggle("Route to policy head", isOn: $model.featureSkipToPolicyHead)
                            Toggle("Route to value head", isOn: $model.featureSkipToValueHead)
                            Toggle("Route to final block", isOn: $model.featureSkipToFinalBlock)
                            enumPicker("Fusion mode", $model.featureSkipFusion, FeatureSkipFusion.allCases)
                            Text("concat_direct widens each routed consumer's input in place "
                                + "(no extra tensors). compress_conv_bn_relu builds one shared "
                                + "1×1-conv→BN→act node for the heads — head-only, so it can't "
                                + "combine with final-block routing.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
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
        // Bounds guard. When a group is removed (or reordered), SwiftUI can
        // re-evaluate the *disappearing* row's body — and any binding the row's
        // controls still hold — one more time with its now-stale `i`, after
        // `blockGroups` has already shrunk. Every `blockGroups[i]` access below
        // (the `if seStyle`/`if useRezero` reads and the field bindings) would
        // then subscript out of range and trap (this crashed the app on a group
        // delete via the Output-norm Picker's selection binding). Gating the
        // whole field set on `indices.contains(i)` makes the stale re-eval
        // render empty instead of trapping; valid rows are unaffected.
        if model.blockGroups.indices.contains(i) {
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
            // Optional output normalization (Optional in the model so old configs
            // decode as nil); map nil <-> .none for the non-optional picker. The
            // get/set re-check the index — this binding is retained by the Picker
            // and can fire during the disappearing-row update even past the outer
            // guard, so it must be self-defending.
            enumPicker("Output norm", Binding(
                get: { model.blockGroups.indices.contains(i) ? (model.blockGroups[i].outputNorm ?? .none) : .none },
                set: { if model.blockGroups.indices.contains(i) { model.blockGroups[i].outputNorm = $0 } }
            ), BlockOutputNorm.allCases)
            Toggle("Use ReZero", isOn: $model.blockGroups[i].useRezero)
            if model.blockGroups[i].useRezero {
                // The α init is seeded from the loaded preset and does NOT
                // auto-track the TOTAL block count, so a deep net built off a
                // shallow preset silently keeps the shallow α. Flag the mismatch
                // and offer a one-click snap to 1/√(total blocks) rather than
                // silently overwriting a deliberately-set value.
                HStack {
                    floatField("ReZero α init", $model.blockGroups[i].rezeroAlphaInit)
                    // Two depth-appropriate inits are blessed: 1/√N (default,
                    // variance-preserving) and 1/N (DeepNorm-style, gentler — for
                    // deep towers where the stream *mean* accumulates). Offer a
                    // one-click snap to each; the forward soft-bound is α₀·tanh
                    // (asymptote ≈ α₀, mult 1.0) either way. Warn only when the init
                    // matches neither (a stale value carried from a shallower preset).
                    Button(String(format: "1/√%d=%.3f", model.totalBlocks, model.recommendedRezeroAlphaInit)) {
                        model.blockGroups[i].rezeroAlphaInit = model.recommendedRezeroAlphaInit
                    }
                    .controlSize(.small)
                    .help("Default ReZero init: 1/√(total blocks), variance-preserving. Forward tanh soft-bound ≈ α₀.")
                    Button(String(format: "1/%d=%.3f", model.totalBlocks, model.recommendedRezeroAlphaInit1OverN)) {
                        model.blockGroups[i].rezeroAlphaInit = model.recommendedRezeroAlphaInit1OverN
                    }
                    .controlSize(.small)
                    .help("DeepNorm-style init: 1/(total blocks). Gentler; preferable for very deep towers. Forward tanh soft-bound ≈ α₀.")
                    if model.rezeroAlphaInitMismatch(at: i) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .help("ReZero α init matches neither 1/√N nor 1/N for this depth — likely a stale value from a shallower preset.")
                    }
                }
            }
            floatField("Dropout multiplier", $model.blockGroups[i].dropoutMultiplier)
        }
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
                    model.refreshPresets()
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
                return model.availablePresets.first(where: { $0.named.architecture == current })?.name
            },
            set: { newName in
                guard let name = newName,
                      let entry = model.availablePresets.first(where: { $0.name == name })
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
