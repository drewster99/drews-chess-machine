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
    private let onBuild: (BuildNewModelRequest) -> Void
    private let onCancel: () -> Void

    @State private var saveStatus: String?

    /// The functions a block group's "Activation" picker and the "Use for
    /// every activation" menu offer: `ActivationFunction.functions`, the one
    /// list every activation choice draws from, never `allCases`, which
    /// holds `does_not_apply`. Neither control can take that marker:
    /// `BlockGroup.setActivationFunction` traps on it, and the menu treats
    /// its refusal as a defect. A named list (pinned by
    /// `testEveryActivationChoiceListIsTheFunctionsList`) is what lets a test
    /// see what these two controls offer; a literal written at each site
    /// could later become `allCases` with no test failing.
    static let activationChoices: [ActivationFunction] = ActivationFunction.functions

    /// A Save-as-Preset the store refused because a preset of that name
    /// already exists, captured at click time so "Replace" writes exactly
    /// what the user was looking at when they saved — not whatever the
    /// fields hold by the time they answer the alert.
    private struct PendingPresetReplacement {
        let name: String
        let label: String
        let architecture: NetworkArchitecture
    }

    /// Set together with `isReplacePresetAlertPresented` when a save hits an
    /// existing preset; cleared by either alert button.
    @State private var pendingPresetReplacement: PendingPresetReplacement?
    @State private var isReplacePresetAlertPresented = false

    init(
        initial: NamedArchitecture,
        onBuild: @escaping (BuildNewModelRequest) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.init(model: BuildNewModelModel(initial), onBuild: onBuild, onCancel: onCancel)
    }

    /// Edits `model`, which the caller keeps a reference to — how the
    /// group-removal render tests change the tower under the live screen.
    init(
        model: BuildNewModelModel,
        onBuild: @escaping (BuildNewModelRequest) -> Void,
        onCancel: @escaping () -> Void
    ) {
        _model = State(initialValue: model)
        self.onBuild = onBuild
        self.onCancel = onCancel
    }

    var body: some View {
        @Bindable var model = model
        // Each rebuilds the architecture from every field, so they are read
        // once per redraw here and handed to the rows, not asked again by
        // every row.
        let nonStandard = model.nonStandardInitOptions
        let groupsWithSkipProjection = model.groupsWithSkipProjection
        let existingSites = model.existingActivationSites
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
                        ArchitectureSiteActivationPicker(site: .stem, model: model, existingSites: existingSites)
                        ArchitectureSiteActivationPicker(site: .towerEnd, model: model, existingSites: existingSites)
                        // The `--derive-model --set-activation` rule, from the
                        // same function (`setMainActivationEverywhere`), so an
                        // edit made either way gives the same architecture.
                        // `applyMainActivationEverywhere` throws only for
                        // `does_not_apply` (`setMainActivationEverywhere`'s one
                        // refusal), and this menu lists only
                        // `BuildNewModelView.activationChoices`, which never
                        // holds it. A throw here is therefore a defect, not a
                        // user error.
                        Menu("Use for every activation") {
                            ForEach(BuildNewModelView.activationChoices, id: \.self) { function in
                                Button(function.rawValue) {
                                    do {
                                        try model.applyMainActivationEverywhere(function)
                                    } catch {
                                        preconditionFailure("Use for every activation offered '\(function.rawValue)', "
                                            + "which setMainActivationEverywhere refused: \(error)")
                                    }
                                }
                            }
                        }
                        .help("Sets this activation at every architecture-level site the model has and on every block "
                              + "group's main path (the --derive-model --set-activation rule). A group with an SE block "
                              + "keeps its SE activation.")
                        LabeledContent("Total blocks") {
                            Text(model.totalBlocks?.formatted(.number) ?? "invalid")
                                .monospacedDigit()
                                .foregroundStyle(model.totalBlocks == nil ? AnyShapeStyle(.orange) : AnyShapeStyle(.primary))
                        }
                    }

                    // Rows are keyed and bound by draft identity (see
                    // `BlockGroupDraft`); the enumerated position is used only
                    // for the header's display number and its end-of-list
                    // button states.
                    ForEach(Array(model.blockGroupDrafts.enumerated()), id: \.element.id) { entry in
                        Section {
                            BlockGroupFieldsView(
                                model: model,
                                draft: entry.element,
                                nonStandardInitOptions: nonStandard,
                                groupsWithSkipProjection: groupsWithSkipProjection
                            )
                        } header: {
                            BlockGroupHeaderView(model: model, draft: entry.element, position: entry.offset)
                        }
                    }

                    Section {
                        Button {
                            model.appendCopyOfLastGroup()
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
                        ArchitectureSiteActivationPicker(site: .policyHead, model: model, existingSites: existingSites)
                        InitOptionRow(
                            isNonStandard: nonStandard.contains(.policyHeadFinalInit),
                            stepZeroEffect: InitOptionField.policyHeadFinalInit.stepZeroEffect
                        ) {
                            enumPicker("Final layer init", $model.policyHeadFinalInit, HeadFinalInit.allCases)
                        }
                    }

                    Section("Value head") {
                        enumPicker("Value style", $model.valueHeadStyle, ValueHeadStyle.allCases)
                        intField("Value conv channels", $model.valueHeadConvChannels)
                        ArchitectureSiteActivationPicker(site: .valueHeadConv, model: model, existingSites: existingSites)
                        intField("Value hidden units", $model.valueHeadHiddenUnits)
                        ArchitectureSiteActivationPicker(site: .valueHeadFC1Hidden, model: model, existingSites: existingSites)
                        InitOptionRow(
                            isNonStandard: nonStandard.contains(.valueHeadFinalInit),
                            stepZeroEffect: InitOptionField.valueHeadFinalInit.stepZeroEffect
                        ) {
                            enumPicker("Final layer init", $model.valueHeadFinalInit, HeadFinalInit.allCases)
                        }
                        // A scalar head has no draw class; the field stays
                        // reachable there only to fix a value validate() refuses.
                        if model.valueHeadStyle == .wdlSoftmax
                            || nonStandard.contains(.valueHeadDrawPrior) {
                            InitOptionRow(
                                isNonStandard: nonStandard.contains(.valueHeadDrawPrior),
                                stepZeroEffect: InitOptionField.valueHeadDrawPrior.stepZeroEffect
                            ) {
                                floatField("Initial draw probability", $model.valueHeadDrawPrior)
                            }
                        }
                    }

                    Section("Feature skip") {
                        enumPicker("Source", $model.featureSkipSource, FeatureSkipSource.allCases)
                        // Outside the source check, so it is present (and
                        // disabled) with the feature skip off, like every
                        // other site picker.
                        ArchitectureSiteActivationPicker(site: .featureSkip, model: model, existingSites: existingSites)
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
                        PolicyTailPrecisionPicker(model: model)
                    }

                    Section("Name") {
                        TextField("Label", text: $model.labelOverride, prompt: Text(model.label))
                    }

                    Section("Initialization") {
                        BuildInitSeedField(model: model)
                        InitSetButtonsView(model: model)
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
        // Saving never silently replaces a preset: the store's exclusive
        // write refuses an existing name, and only this confirmation passes
        // `replacingExisting: true`.
        .alert(
            "Replace existing preset?",
            isPresented: $isReplacePresetAlertPresented,
            presenting: pendingPresetReplacement,
            actions: { pending in
                Button("Replace", role: .destructive) {
                    pendingPresetReplacement = nil
                    savePreset(name: pending.name, label: pending.label,
                               architecture: pending.architecture, replacingExisting: true)
                }
                Button("Cancel", role: .cancel) {
                    pendingPresetReplacement = nil
                    saveStatus = "Not saved: preset \"\(pending.name)\" already exists"
                }
            },
            message: { pending in
                Text("A preset named \"\(pending.name)\" is already saved. Replacing it overwrites its saved architecture and label.")
            }
        )
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
                    Text("(\(BinaryByteCount.text(model.estimatedWeightBytes)) F32)")
                        .foregroundStyle(.secondary)
                }
                .font(.callout)
                // Guidance, never a limit: every size that fits in memory
                // builds; anything past the recommended size is flagged.
                if let guidance = model.sizeGuidance {
                    Text(guidance.readout)
                        .font(.callout)
                        .foregroundStyle(guidance.verdict == .withinRecommendedSize
                                         ? AnyShapeStyle(.secondary) : AnyShapeStyle(.orange))
                }
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
                // The saved label is what the preset picker shows. With no
                // explicit label, the effective `model.label` is "Custom" (or
                // a matched preset's label), which would make every saved
                // preset indistinguishable in the picker — so fall back to
                // the preset name the user just chose.
                let savedLabel = model.labelOverride.isEmpty ? name : model.labelOverride
                savePreset(name: name, label: savedLabel, architecture: model.architecture, replacingExisting: false)
            }
            .disabled(!model.isValid)

            if let status = saveStatus {
                Text(status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    .help(status)
            }

            Spacer()
            Button("Cancel", role: .cancel) { onCancel() }
            Button("Build") {
                if let request = model.buildRequest { onBuild(request) }
            }
                .keyboardShortcut(.defaultAction)
                .disabled(model.buildRequest == nil)
        }
        .padding()
    }

    // MARK: Helpers

    /// Write the preset. An existing preset of the same name comes back as
    /// `.presetAlreadyExists` (when `replacingExisting` is false) and opens
    /// the replace confirmation instead of a status error; every other
    /// failure — an invalid name, a folder in the way — lands in the status
    /// text.
    private func savePreset(name: String, label: String, architecture: NetworkArchitecture, replacingExisting: Bool) {
        do {
            let url = try ArchitecturePresetStore.save(
                name: name, label: label, architecture: architecture, replacingExisting: replacingExisting)
            model.refreshPresets()
            saveStatus = "\(replacingExisting ? "Replaced" : "Saved") \(url.lastPathComponent)"
        } catch ArchitecturePresetStore.StoreError.presetAlreadyExists(_) {
            saveStatus = nil
            pendingPresetReplacement = PendingPresetReplacement(name: name, label: label, architecture: architecture)
            isReplacePresetAlertPresented = true
        } catch {
            saveStatus = "\(error)"
        }
    }

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

}

// MARK: - Block-group rows

/// The header of one block group's section: its display number plus the
/// move / duplicate / remove buttons, all acting on the group's draft by
/// identity. `position` is the group's place in the tower when this header
/// was built, used only for the number shown and to disable the moves at the
/// ends.
private struct BlockGroupHeaderView: View {
    let model: BuildNewModelModel
    let draft: BlockGroupDraft
    let position: Int

    var body: some View {
        HStack(spacing: 6) {
            Text("Group \(position + 1)")
            Spacer()
            Button {
                model.moveGroup(draft, offset: -1)
            } label: {
                Image(systemName: "chevron.up")
            }
            .disabled(position == 0)
            .help("Move this group one step toward the input")
            Button {
                model.moveGroup(draft, offset: 1)
            } label: {
                Image(systemName: "chevron.down")
            }
            .disabled(position == model.blockGroupDrafts.count - 1)
            .help("Move this group one step toward the heads")
            Button {
                model.duplicateGroup(draft)
            } label: {
                Image(systemName: "plus.square.on.square")
            }
            .help("Insert a copy of this group below it")
            Button {
                model.removeGroup(draft)
            } label: {
                Image(systemName: "trash")
            }
            .disabled(model.blockGroupDrafts.count == 1)
            .help("Remove this group")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
    }
}

/// Every editable field of one block group, bound through the group's draft
/// (see `BlockGroupDraft`), so a binding SwiftUI retains past this row's
/// removal can only reach the detached draft.
private struct BlockGroupFieldsView: View {
    let model: BuildNewModelModel
    @Bindable var draft: BlockGroupDraft
    /// Read once per redraw by the screen and handed to every row (see
    /// `BlockGroupInitOptionsView`).
    let nonStandardInitOptions: Set<InitOptionField>
    let groupsWithSkipProjection: Set<Int>

    var body: some View {
        intField("Blocks (count)", $draft.group.count)
        intField("Channels", $draft.group.channels)
        intField("Conv 1 kernel size (odd)", $draft.group.conv1KernelSize)
        intField("Conv 2 kernel size (odd)", $draft.group.conv2KernelSize)
        enumPicker("SE style", $draft.group.seStyle, SEStyle.allCases)
        if draft.group.seStyle != .none {
            intField("SE reduction ratio", $draft.group.seReductionRatio)
        }
        // Only scale_and_bias has a β half. Hidden for the other styles;
        // a non-glorot value left behind by switching style away is
        // surfaced by validate() rather than silently reset.
        if draft.group.seStyle == .scaleAndBias {
            enumPicker("SE β init", $draft.group.seBetaInit, SEBetaInit.allCases)
        }
        // Through the draft, so the edit applies the rule
        // `--derive-model --set-activation` shares
        // (`BlockGroup.setActivationFunction`).
        enumPicker("Activation", $draft.activationFunction, BuildNewModelView.activationChoices)
        // The SE FC1's own activation (issue #2), a site that exists only
        // with an SE block (OD-13): always present so the row layout never
        // shifts, disabled and `does_not_apply` on an SE-less group, and
        // asking for a choice when SE is switched on (`BlockGroup.seStyle`
        // clears it when SE is switched off).
        ArchitectureSiteActivationPicker(
            title: "SE activation",
            activation: $draft.group.seActivation,
            availability: .seFC1(of: draft.group))
        // Through the draft, so the model's site sync runs inside the edit
        // (the first group's style decides the stem, the last group's the
        // tower end).
        enumPicker("Activation style", $draft.activationStyle, BlockActivationStyle.allCases)
        enumPicker("Skip merge", $draft.group.skipMerge, BlockSkipMerge.allCases)
        enumPicker("Output norm", $draft.outputNorm, BlockOutputNorm.allCases)
        Toggle("Use ReZero", isOn: $draft.group.useRezero)
        if draft.group.useRezero {
            // The α init and cap are seeded from the loaded preset and do NOT
            // auto-track the TOTAL block count, so a deep net built off a
            // shallow preset silently keeps the shallow values. Flag the
            // mismatch and offer a one-click snap rather than silently
            // overwriting a deliberately-set value.
            HStack {
                floatField("ReZero α init", $draft.group.rezeroAlphaInit)
                    .help("Starting value of each block's trainable ReZero α, set independently of the cap. 0 is the ReZero paper's init: every branch starts off and α learns from step 1.")
                // Two depth-appropriate values are blessed: 1/√N (default,
                // variance-preserving) and 1/N (DeepNorm-style, gentler — for
                // deep towers where the stream *mean* accumulates). Offer a
                // one-click snap to each, setting both the init and the cap.
                // Warn only when a depth-scaled value matches neither (see
                // `rezeroDepthScaleMismatch`). While a block count is
                // invalid there is no depth: both buttons stay in place,
                // disabled, with the formula alone as their label.
                let totalBlocks = model.totalBlocks
                Button(totalBlocks.map { "1/√\($0)=\(String(format: "%.3f", model.recommendedRezeroAlphaInit))" } ?? "1/√N") {
                    model.applyRecommendedRezero(model.recommendedRezeroAlphaInit, to: draft)
                }
                .controlSize(.small)
                .disabled(totalBlocks == nil)
                .help("Default ReZero init: 1/√(total blocks), variance-preserving. Sets both the α init and the cap.")
                Button(totalBlocks.map { "1/\($0)=\(String(format: "%.3f", model.recommendedRezeroAlphaInit1OverN))" } ?? "1/N") {
                    model.applyRecommendedRezero(model.recommendedRezeroAlphaInit1OverN, to: draft)
                }
                .controlSize(.small)
                .disabled(totalBlocks == nil)
                .help("DeepNorm-style init: 1/(total blocks). Gentler; preferable for very deep towers. Sets both the α init and the cap.")
                if model.rezeroDepthScaleMismatch(for: draft.group) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .help("The ReZero cap (or a non-zero α init) matches neither 1/√N nor 1/N for this depth — likely a stale value from a shallower preset.")
                }
            }
            floatField("ReZero cap", $draft.group.rezeroAlphaCap)
                .help("Asymptote C of the forward soft bound C·tanh(α/C): the effective branch scale never exceeds C. Must be > 0. Set independently of the α init.")
        }
        floatField("Dropout multiplier", $draft.group.dropoutMultiplier)
        BlockGroupInitOptionsView(
            model: model,
            draft: draft,
            nonStandard: nonStandardInitOptions,
            groupsWithSkipProjection: groupsWithSkipProjection
        )
    }
}

/// The init-neutral options of one block group. Each appears where its layer
/// exists (an SE block, a post-activation branch, a width-transition skip
/// projection) and also wherever it holds a non-standard value, so a value
/// `validate()` refuses after the layer was switched off can be fixed here.
///
/// `nonStandard` and `groupsWithSkipProjection` come from the screen, which
/// computes each once per redraw: each question rebuilds the architecture
/// from every field, so it is not asked again by every row. Where the skip projections
/// are is worked out per group (`NetworkArchitecture.groupsWithSkipProjection`),
/// never by expanding the tower, so a block count the user is still typing —
/// negative, or near `Int.max` — draws without trapping.
private struct BlockGroupInitOptionsView: View {
    let model: BuildNewModelModel
    @Bindable var draft: BlockGroupDraft
    let nonStandard: Set<InitOptionField>
    let groupsWithSkipProjection: Set<Int>

    var body: some View {
        // A removed group's row is drawn once more after its draft has left
        // the model; it has no position then and shows nothing.
        if let groupIndex = model.positionInTower(of: draft) {
            if draft.group.seStyle != .none || nonStandard.contains(.seGammaBiasInit(group: groupIndex)) {
                InitOptionRow(
                    isNonStandard: nonStandard.contains(.seGammaBiasInit(group: groupIndex)),
                    stepZeroEffect: InitOptionField.seGammaBiasInit(group: groupIndex).stepZeroEffect
                ) {
                    floatField("SE γ bias init", $draft.group.seGammaBiasInit)
                }
            }
            if draft.group.activationStyle == .post || nonStandard.contains(.branchOutputInit(group: groupIndex)) {
                InitOptionRow(
                    isNonStandard: nonStandard.contains(.branchOutputInit(group: groupIndex)),
                    stepZeroEffect: InitOptionField.branchOutputInit(group: groupIndex).stepZeroEffect
                ) {
                    enumPicker("Branch output init", $draft.group.branchOutputInit, BranchOutputInit.allCases)
                }
            }
            if groupsWithSkipProjection.contains(groupIndex) || nonStandard.contains(.skipProjectionInit(group: groupIndex)) {
                InitOptionRow(
                    isNonStandard: nonStandard.contains(.skipProjectionInit(group: groupIndex)),
                    stepZeroEffect: InitOptionField.skipProjectionInit(group: groupIndex).stepZeroEffect
                ) {
                    enumPicker("Skip projection init", $draft.group.skipProjectionInit, SkipProjectionInit.allCases)
                }
            }
        }
    }
}

// MARK: - Field helpers (shared by every view in this file)

@MainActor @ViewBuilder
private func enumPicker<E: CaseIterable & Hashable & RawRepresentable>(
    _ title: String, _ binding: Binding<E>, _ cases: [E]
) -> some View where E.RawValue == String {
    Picker(title, selection: binding) {
        ForEach(cases, id: \.self) { Text($0.rawValue).tag($0) }
    }
}

@MainActor @ViewBuilder
private func intField(_ title: String, _ binding: Binding<Int>) -> some View {
    TextField(title, value: binding, format: .number)
        .monospacedDigit()
}

@MainActor @ViewBuilder
private func floatField(_ title: String, _ binding: Binding<Float>) -> some View {
    TextField(title, value: binding, format: .number)
        .monospacedDigit()
}
