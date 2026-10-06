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

/// What the Build button hands the host: the architecture to build and the
/// init seed the user entered, or nil when the seed is to be drawn at the
/// build (and then shown and logged).
struct BuildNewModelRequest: Equatable {
    let architecture: NetworkArchitecture
    let enteredInitSeed: UInt64?
}

/// The Init seed field's reading.
enum BuildInitSeedEntry: Equatable {
    /// Empty: a seed is drawn at the build.
    case drawnAtBuild
    case entered(UInt64)
    /// Not a decimal UInt64; Build is disabled.
    case invalid(String)
}

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
    /// Feature 2 Phase B), one identity-addressed draft per group (see
    /// `BlockGroupDraft` for why rows are never addressed by index). Full
    /// fidelity: a loaded mixed tower round-trips through the editor without
    /// collapsing. Changed only through the group methods below.
    private(set) var blockGroupDrafts: [BlockGroupDraft]
    var stemConvKernelSize: Int
    /// The six architecture-level site activations, one per site
    /// (`ArchitectureActivationSite`). Each holds `does_not_apply` while the
    /// topology lacks its site: a site that disappears drops its choice
    /// inside the edit that removed it (`syncSiteActivationsWithTopology`),
    /// and one that appears holds `does_not_apply` until a function is
    /// chosen — `validationError` names it and Build / Save stay disabled —
    /// so an earlier choice is never restored silently.
    var stemActivation: ActivationFunction
    var towerEndActivation: ActivationFunction
    var featureSkipActivation: ActivationFunction
    var policyHeadActivation: ActivationFunction
    var valueHeadConvActivation: ActivationFunction
    var valueHeadFC1HiddenActivation: ActivationFunction
    // The topology fields below can make a site appear or disappear, so each
    // edit syncs the site activations before it returns. (Assignments in
    // `init` go through the init accessor and run no observer.)
    var policyHeadStyle: PolicyHeadStyle {
        didSet { syncSiteActivationsWithTopology() }
    }
    var policyPreConvChannels: Int
    var valueHeadStyle: ValueHeadStyle
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int
    /// The head init-neutral options (`NetworkArchitecture.policyHeadFinalInit`,
    /// `valueHeadFinalInit`, `valueHeadDrawPrior`).
    var policyHeadFinalInit: HeadFinalInit
    var valueHeadFinalInit: HeadFinalInit
    var valueHeadDrawPrior: Float
    var computeDataType: ComputeDataType
    /// Feature skip (optional long concat skip). `featureSkipSource == .none`
    /// disables the whole feature. All destinations (policy/value heads, final
    /// block) and both fusion modes are supported; `validate()` rejects only
    /// `compressConvBNReLU` fusion combined with the final-block destination.
    var featureSkipSource: FeatureSkipSource {
        didSet { syncSiteActivationsWithTopology() }
    }
    var featureSkipFusion: FeatureSkipFusion {
        didSet { syncSiteActivationsWithTopology() }
    }
    var featureSkipToPolicyHead: Bool {
        didSet { syncSiteActivationsWithTopology() }
    }
    var featureSkipToValueHead: Bool {
        didSet { syncSiteActivationsWithTopology() }
    }
    /// Does not decide whether the compress fusion node is built
    /// (`featureSkipUsesCompressNode`), so it needs no site sync.
    var featureSkipToFinalBlock: Bool

    /// Name to save the current config under (Save-as-Preset). Defaults from the
    /// label, sanitized to a filename-safe slug.
    var saveAsName: String = ""

    /// The optional init seed (decimal UInt64): the same seed and
    /// architecture mint bit-identical trainable tensors here, in
    /// `--new-model --init-seed` and on every machine; the batch-norm running
    /// statistics are calibrated by a GPU forward pass and match only to float
    /// tolerance (see `NetworkInitMode.randomWeights`). Not part of the
    /// architecture or a preset.
    var initSeedText: String = ""

    /// Cached preset list (built-ins + user-saved). Scanned once at init (and
    /// after a Save-as-Preset via `refreshPresets()`) rather than on every
    /// `body`/`matchedPreset` access: `ArchitecturePresetStore.allPresets()`
    /// enumerates and JSON-decodes the user Presets folder, and `body`
    /// re-evaluates on every keystroke into any field, so reading it live meant
    /// a disk scan several times per render pass.
    private(set) var availablePresets: [(name: String, named: NamedArchitecture)] = []

    /// This Mac's physical memory, which the size refusal and the size
    /// guidance scale to (`ModelSizeGuidance`). Fixed for the process.
    private let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory

    init(_ named: NamedArchitecture = NamedArchitecture(label: "Custom", architecture: .current)) {
        let a = named.architecture
        // The drafts capture `self` (their topology-change callback), which
        // is legal only once every stored property is set: start with no
        // drafts, set everything else, then build them.
        self.blockGroupDrafts = []
        self.labelOverride = ""
        self.inputEncoding = a.inputEncoding
        self.stemConvKernelSize = a.stemConvKernelSize
        self.stemActivation = a.stemActivation
        self.towerEndActivation = a.towerEndActivation
        self.featureSkipActivation = a.featureSkipActivation
        self.policyHeadActivation = a.policyHeadActivation
        self.valueHeadConvActivation = a.valueHeadConvActivation
        self.valueHeadFC1HiddenActivation = a.valueHeadFC1HiddenActivation
        self.policyHeadStyle = a.policyHeadStyle
        self.policyPreConvChannels = a.policyPreConvChannels
        self.valueHeadStyle = a.valueHeadStyle
        self.valueHeadConvChannels = a.valueHeadConvChannels
        self.valueHeadHiddenUnits = a.valueHeadHiddenUnits
        self.policyHeadFinalInit = a.policyHeadFinalInit
        self.valueHeadFinalInit = a.valueHeadFinalInit
        self.valueHeadDrawPrior = a.valueHeadDrawPrior
        self.computeDataType = a.computeDataType
        self.featureSkipSource = a.featureSkipSource
        self.featureSkipFusion = a.featureSkipFusion
        self.featureSkipToPolicyHead = a.featureSkipToPolicyHead
        self.featureSkipToValueHead = a.featureSkipToValueHead
        self.featureSkipToFinalBlock = a.featureSkipToFinalBlock
        self.availablePresets = ArchitecturePresetStore.allPresets()
        self.blockGroupDrafts = a.blockGroups.map { makeDraft($0) }
    }

    /// A draft of `group` wired to this model's site sync. The model is the
    /// only creator of drafts.
    ///
    /// `[weak self]`: SwiftUI can keep a row's bindings alive past the update
    /// that removes the row (see `BlockGroupDraft`), and those bindings hold
    /// the draft, not the model, so a style write can reach a draft whose
    /// model is gone. That draft is detached exactly like a removed one —
    /// the write changes only the orphan draft and there is no model state
    /// left to sync — so the skipped call is the correct outcome there, not
    /// a silent fallback. (`unowned` would crash on that write.)
    private func makeDraft(_ group: BlockGroup) -> BlockGroupDraft {
        BlockGroupDraft(group, onTopologyChange: { [weak self] in
            self?.syncSiteActivationsWithTopology()
        })
    }

    /// Re-scan the preset folder. Call after saving a new user preset so it
    /// appears in the picker without re-scanning on every render.
    func refreshPresets() {
        availablePresets = ArchitecturePresetStore.allPresets()
    }

    /// Populate every field from a preset (the picker selection is derived from
    /// architecture equality, so no separate "selected" flag is needed).
    ///
    /// The assignments run one by one, and each topology field's observer
    /// syncs the site activations against a half-replaced topology, which
    /// can rewrite them. So the six site activations are assigned last, in
    /// one block after every topology field: whatever an intermediate sync
    /// did, they end equal to the snapshot's own (consistent) values.
    func load(_ named: NamedArchitecture) {
        let a = named.architecture
        labelOverride = ""
        inputEncoding = a.inputEncoding
        blockGroupDrafts = a.blockGroups.map { makeDraft($0) }
        stemConvKernelSize = a.stemConvKernelSize
        policyHeadStyle = a.policyHeadStyle
        policyPreConvChannels = a.policyPreConvChannels
        valueHeadStyle = a.valueHeadStyle
        valueHeadConvChannels = a.valueHeadConvChannels
        valueHeadHiddenUnits = a.valueHeadHiddenUnits
        policyHeadFinalInit = a.policyHeadFinalInit
        valueHeadFinalInit = a.valueHeadFinalInit
        valueHeadDrawPrior = a.valueHeadDrawPrior
        computeDataType = a.computeDataType
        featureSkipSource = a.featureSkipSource
        featureSkipFusion = a.featureSkipFusion
        featureSkipToPolicyHead = a.featureSkipToPolicyHead
        featureSkipToValueHead = a.featureSkipToValueHead
        featureSkipToFinalBlock = a.featureSkipToFinalBlock
        stemActivation = a.stemActivation
        towerEndActivation = a.towerEndActivation
        featureSkipActivation = a.featureSkipActivation
        policyHeadActivation = a.policyHeadActivation
        valueHeadConvActivation = a.valueHeadConvActivation
        valueHeadFC1HiddenActivation = a.valueHeadFC1HiddenActivation
    }

    /// The architecture described by the current fields, with every site
    /// the topology lacks cleared to `does_not_apply`
    /// (`clearActivationSitesTheTopologyLacks`), so it never holds a
    /// function at an absent site. A site that exists keeps its stored
    /// value, `does_not_apply` included, which `validate()` then names.
    var architecture: NetworkArchitecture {
        var composed = NetworkArchitecture(
            inputEncoding: inputEncoding,
            blockGroups: blockGroups,
            stemConvKernelSize: stemConvKernelSize,
            stemActivation: stemActivation,
            towerEndActivation: towerEndActivation,
            featureSkipActivation: featureSkipActivation,
            policyHeadStyle: policyHeadStyle,
            policyPreConvChannels: policyPreConvChannels,
            policyHeadActivation: policyHeadActivation,
            valueHeadStyle: valueHeadStyle,
            valueHeadConvChannels: valueHeadConvChannels,
            valueHeadHiddenUnits: valueHeadHiddenUnits,
            valueHeadConvActivation: valueHeadConvActivation,
            valueHeadFC1HiddenActivation: valueHeadFC1HiddenActivation,
            policyHeadFinalInit: policyHeadFinalInit,
            valueHeadFinalInit: valueHeadFinalInit,
            valueHeadDrawPrior: valueHeadDrawPrior,
            computeDataType: computeDataType,
            featureSkipSource: featureSkipSource,
            featureSkipFusion: featureSkipFusion,
            featureSkipToPolicyHead: featureSkipToPolicyHead,
            featureSkipToValueHead: featureSkipToValueHead,
            featureSkipToFinalBlock: featureSkipToFinalBlock
        )
        composed.clearActivationSitesTheTopologyLacks()
        return composed
    }

    // MARK: Architecture-level activation sites

    /// Copies the composed architecture's six site activations back into the
    /// stored fields, so a site the topology just lost holds
    /// `does_not_apply` and a later reappearance asks for a new choice
    /// instead of restoring the old one. Runs synchronously inside every
    /// model mutation that can change which sites exist (the topology
    /// fields' observers, the group methods, a draft's style edit), so no
    /// disappear → reappear sequence can skip it, however edits are
    /// coalesced or rendered.
    func syncSiteActivationsWithTopology() {
        let composed = architecture
        if stemActivation != composed.stemActivation { stemActivation = composed.stemActivation }
        if towerEndActivation != composed.towerEndActivation { towerEndActivation = composed.towerEndActivation }
        if featureSkipActivation != composed.featureSkipActivation { featureSkipActivation = composed.featureSkipActivation }
        if policyHeadActivation != composed.policyHeadActivation { policyHeadActivation = composed.policyHeadActivation }
        if valueHeadConvActivation != composed.valueHeadConvActivation {
            valueHeadConvActivation = composed.valueHeadConvActivation
        }
        if valueHeadFC1HiddenActivation != composed.valueHeadFC1HiddenActivation {
            valueHeadFC1HiddenActivation = composed.valueHeadFC1HiddenActivation
        }
    }

    /// The stored property holding `site`'s activation: the one site → field
    /// map the screen's site pickers bind through
    /// (`ArchitectureSiteActivationPicker.init(site:model:existingSites:)`)
    /// and `storedActivation(at:)` reads.
    static func activationKeyPath(
        at site: ArchitectureActivationSite
    ) -> ReferenceWritableKeyPath<BuildNewModelModel, ActivationFunction> {
        switch site {
        case .stem: return \.stemActivation
        case .towerEnd: return \.towerEndActivation
        case .featureSkip: return \.featureSkipActivation
        case .policyHead: return \.policyHeadActivation
        case .valueHeadConv: return \.valueHeadConvActivation
        case .valueHeadFC1Hidden: return \.valueHeadFC1HiddenActivation
        }
    }

    /// The stored activation of `site` (the value its picker binds).
    func storedActivation(at site: ArchitectureActivationSite) -> ActivationFunction {
        self[keyPath: Self.activationKeyPath(at: site)]
    }

    /// Whether the current topology has `site` (its picker is enabled only
    /// then).
    func siteExists(_ site: ArchitectureActivationSite) -> Bool {
        architecture.hasActivationSite(site)
    }

    /// Every site the current topology has, from one composition of the
    /// architecture — what the screen reads once per redraw.
    var existingActivationSites: Set<ArchitectureActivationSite> {
        let composed = architecture
        return Set(ArchitectureActivationSite.allCases.filter { composed.hasActivationSite($0) })
    }

    /// "Use for every activation": `value` at every existing site and on
    /// every group's main path, through
    /// `NetworkArchitecture.setMainActivationEverywhere` — the rule
    /// `--derive-model --set-activation` applies, so the same edit made
    /// either way gives the same architecture. Copied back through each
    /// existing draft's `activationFunction` (the shared setter), so every
    /// row keeps its identity; the rule never changes a group's
    /// `seActivation`, so there is nothing else to copy. Throws (changing
    /// nothing) only for `does_not_apply`, which its callers never pass.
    func applyMainActivationEverywhere(_ value: ActivationFunction) throws {
        var edited = architecture
        try edited.setMainActivationEverywhere(value)
        SessionLogger.shared.log("[BUTTON] Build New Model: Use for every activation \(value.rawValue)")
        precondition(edited.blockGroups.count == blockGroupDrafts.count,
                     "BuildNewModelModel: setting the activation changed the number of block groups")
        for (draft, group) in zip(blockGroupDrafts, edited.blockGroups) {
            draft.activationFunction = group.activationFunction
        }
        stemActivation = edited.stemActivation
        towerEndActivation = edited.towerEndActivation
        featureSkipActivation = edited.featureSkipActivation
        policyHeadActivation = edited.policyHeadActivation
        valueHeadConvActivation = edited.valueHeadConvActivation
        valueHeadFC1HiddenActivation = edited.valueHeadFC1HiddenActivation
    }

    // MARK: Groups (the editor's add/duplicate/remove/reorder)

    /// The tower's groups as values, in order — what `architecture` builds
    /// from. Edits go through the drafts, never through this copy.
    var blockGroups: [BlockGroup] { blockGroupDrafts.map(\.group) }

    /// Why the tower's shape cannot be built (a block count that is not
    /// positive, or a total that overflows `Int`), or nil when it can.
    /// Checked without walking the tower (`validateTowerShape()`), so it is
    /// safe on whatever the user has typed; everything that needs the block
    /// count reads it only while this is nil.
    var towerShapeError: String? {
        do {
            try architecture.validateTowerShape()
            return nil
        } catch {
            return String(describing: error)
        }
    }

    /// Total blocks across all groups, or nil while the tower shape is
    /// invalid (`towerShapeError`) — there is no meaningful depth to show
    /// or to scale ReZero by then.
    var totalBlocks: Int? {
        towerShapeError == nil ? architecture.numBlocks : nil
    }

    /// The total block count for the ReZero recommendations, which are read
    /// only while the tower shape is valid (the editor disables them
    /// otherwise), so an invalid shape here is a defect.
    private var totalBlocksForRezero: Int {
        guard let totalBlocks else {
            preconditionFailure("BuildNewModelModel: ReZero recommendation read while the tower shape is invalid")
        }
        return totalBlocks
    }

    /// Position of `draft` in the tower, or nil once its group has been
    /// removed. SwiftUI evaluates a removed row's views once more after the
    /// draft has left `blockGroupDrafts`, so anything a row reads while
    /// drawing goes through this and shows nothing for a removed draft: that
    /// row is on its way off screen and has no group left to describe.
    func positionInTower(of draft: BlockGroupDraft) -> Int? {
        blockGroupDrafts.firstIndex(where: { $0 === draft })
    }

    /// Position of `draft` for an edit of the tower. The edits come from a
    /// row's buttons, which act only on a row that is on screen, so a draft
    /// that is not in the model is a defect, never a stale row to skip
    /// quietly.
    private func position(of draft: BlockGroupDraft) -> Int {
        guard let index = positionInTower(of: draft) else {
            preconditionFailure("BuildNewModelModel: block-group draft is not in this model")
        }
        return index
    }

    /// Insert a copy of `draft`'s group directly after it.
    func duplicateGroup(_ draft: BlockGroupDraft) {
        let index = position(of: draft)
        blockGroupDrafts.insert(makeDraft(draft.group), at: index + 1)
        syncSiteActivationsWithTopology()
    }

    /// Append a copy of the last group (the editor's "Add group").
    func appendCopyOfLastGroup() {
        guard let last = blockGroupDrafts.last else {
            preconditionFailure("BuildNewModelModel: a tower always has at least one block group")
        }
        blockGroupDrafts.append(makeDraft(last.group))
        syncSiteActivationsWithTopology()
    }

    /// Remove `draft`'s group. The editor disables removing the only group,
    /// so being asked to is a defect.
    func removeGroup(_ draft: BlockGroupDraft) {
        precondition(blockGroupDrafts.count > 1, "BuildNewModelModel: the only block group cannot be removed")
        blockGroupDrafts.remove(at: position(of: draft))
        syncSiteActivationsWithTopology()
    }

    /// Move `draft`'s group one slot toward the input (-1) or the heads (+1).
    /// The editor disables the move at either end, so a move past it is a
    /// defect.
    func moveGroup(_ draft: BlockGroupDraft, offset: Int) {
        let index = position(of: draft)
        let target = index + offset
        precondition(blockGroupDrafts.indices.contains(target),
                     "BuildNewModelModel: block group \(index) cannot move by \(offset)")
        blockGroupDrafts.swapAt(index, target)
        syncSiteActivationsWithTopology()
    }

    // MARK: Init-neutral options

    /// The "Neutral init" button: every option that has a layer to act on
    /// starts that path as a no-op (`NetworkArchitecture.withNeutralInit`, the
    /// same function `--derive-model --set-neutral-init` applies). The draw
    /// prior is left as it is. Safe on a tower whose shape is still invalid:
    /// where a skip projection is comes from `groupsWithSkipProjection`,
    /// which never expands the tower.
    func applyNeutralInit() {
        SessionLogger.shared.log("[BUTTON] Build New Model: Neutral init")
        applyInitOptions(of: architecture.withNeutralInit())
    }

    /// The "Standard init" button: every option, the draw prior included, back
    /// to the value every model was built with before the options existed
    /// (`NetworkArchitecture.withStandardInit`).
    func applyStandardInit() {
        SessionLogger.shared.log("[BUTTON] Build New Model: Standard init")
        applyInitOptions(of: architecture.withStandardInit())
    }

    /// Copies only the init options of `edited` (the current architecture
    /// with one of the sets applied) into the fields, through the existing
    /// drafts so every row keeps its identity.
    private func applyInitOptions(of edited: NetworkArchitecture) {
        precondition(edited.blockGroups.count == blockGroupDrafts.count,
                     "BuildNewModelModel: an init set changed the number of block groups")
        for (draft, group) in zip(blockGroupDrafts, edited.blockGroups) {
            draft.group.seGammaBiasInit = group.seGammaBiasInit
            draft.group.branchOutputInit = group.branchOutputInit
            draft.group.skipProjectionInit = group.skipProjectionInit
        }
        policyHeadFinalInit = edited.policyHeadFinalInit
        valueHeadFinalInit = edited.valueHeadFinalInit
        valueHeadDrawPrior = edited.valueHeadDrawPrior
    }

    /// The options that differ from the standard init, for the editor's and
    /// the diagram's highlight (always compared with the standard value).
    var nonStandardInitOptions: Set<InitOptionField> {
        Set(architecture.nonStandardInitOptions)
    }

    /// The positions of the groups with a skip projection, where a group's
    /// `skipProjectionInit` takes effect (the field is shown only there).
    /// Worked out per group, never by expanding the tower, so the editor can
    /// read it on every redraw whatever the counts hold.
    var groupsWithSkipProjection: Set<Int> {
        architecture.groupsWithSkipProjection
    }

    // MARK: Group ReZero

    /// Apply one of the depth-appropriate recommendations (`1/√N`, `1/N`) to
    /// `draft`'s group: an explicit, labelled action that sets both the α
    /// init and the cap to `value`. The init and the cap are otherwise
    /// independent — editing one never moves the other — so this is the one
    /// place both change together.
    func applyRecommendedRezero(_ value: Float, to draft: BlockGroupDraft) {
        draft.group.rezeroAlphaInit = value
        draft.group.rezeroAlphaCap = value
    }

    /// The depth-appropriate ReZero α init for the current TOTAL block count
    /// (expanded across all groups): `1/√blocks`, which keeps the
    /// residual-stream variance ~O(1) at init (each of N blocks contributes
    /// ~α², so α = 1/√N → total ~1). Group α fields are seeded from the
    /// loaded preset and do NOT auto-track the block count, so building a
    /// deep net off a shallow preset silently keeps the shallow α — the
    /// mismatch flag + one-click apply below cover that. Read only while the
    /// tower shape is valid (`totalBlocks` is non-nil).
    var recommendedRezeroAlphaInit: Float {
        1.0 / Float(totalBlocksForRezero).squareRoot()
    }

    /// DeepNorm-style alternative ReZero init (`1/N`): gentler than `1/√N`,
    /// preferable for very deep towers where the residual stream's *mean* (not
    /// just variance) accumulates down the stack. Offered alongside `1/√N`;
    /// `1/√N` stays the default. Applying either sets the forward tanh
    /// soft-bound's cap to the same value (`applyRecommendedRezero`). See
    /// documentation/rezero-alpha-clamp.md. Read only while the tower shape
    /// is valid.
    var recommendedRezeroAlphaInit1OverN: Float {
        1.0 / Float(totalBlocksForRezero)
    }

    /// True when `g` has ReZero enabled and a depth-scaled
    /// ReZero value that matches NEITHER depth-appropriate value (`1/√N` nor
    /// `1/N`) — i.e. likely a stale value carried from a shallower preset.
    /// Tolerance absorbs float round-trip noise (stored values like
    /// 0.447214). Either canonical value is valid, so only flag when it's
    /// neither.
    ///
    /// Which values are depth-scaled: the cap always — it is where effective
    /// α saturates, so it sets the trained tower's Σα² — and the init only
    /// when it is non-zero. A zero init is the ReZero-paper init, the same at
    /// every depth, so it is never flagged; a non-zero init is the starting
    /// branch scale and is held to the same recommendation as the cap. For
    /// every group whose cap equals its init (all presets and every model
    /// before the explicit cap) this reduces to a check of the init alone.
    /// While the tower shape is invalid there is no depth to compare with,
    /// so nothing is flagged (the shape error is shown instead).
    func rezeroDepthScaleMismatch(for g: BlockGroup) -> Bool {
        guard g.useRezero, totalBlocks != nil else { return false }
        func matchesNeither(_ value: Float) -> Bool {
            abs(value - recommendedRezeroAlphaInit) > 1e-4
                && abs(value - recommendedRezeroAlphaInit1OverN) > 1e-4
        }
        return matchesNeither(g.rezeroAlphaCap)
            || (g.rezeroAlphaInit != 0 && matchesNeither(g.rezeroAlphaInit))
    }

    /// `nil` when the current fields form an architecture this Mac can
    /// build and train; otherwise why not (Build is disabled while non-nil):
    /// a structural error from `validate()`, or a training state larger than
    /// physical memory (`ModelSizeGuidance`) — the one size refusal. Every
    /// other size is allowed and annotated by `sizeGuidance`.
    var validationError: String? {
        let arch = architecture
        do {
            try arch.validate()
            try ModelSizeGuidance(parameterCount: arch.parameterCount, physicalMemoryBytes: physicalMemoryBytes)
                .requireTrainingStateFitsInPhysicalMemory()
            return nil
        } catch {
            return String(describing: error)
        }
    }

    var isValid: Bool { validationError == nil }

    /// Where the current architecture's parameter count sits on this Mac
    /// (the readout's guidance line), or nil while it is invalid.
    var sizeGuidance: ModelSizeGuidance? {
        guard isValid else { return nil }
        return ModelSizeGuidance(parameterCount: architecture.parameterCount, physicalMemoryBytes: physicalMemoryBytes)
    }

    /// The Init seed field, read.
    var initSeedEntry: BuildInitSeedEntry {
        let text = initSeedText.trimmingCharacters(in: .whitespaces)
        if text.isEmpty { return .drawnAtBuild }
        guard let seed = UInt64(strictDecimal: text) else {
            return .invalid("Init seed must be a whole number from 0 to \(UInt64.max)")
        }
        return .entered(seed)
    }

    /// The Build request, or nil while the architecture or the init seed is
    /// invalid (Build is disabled then).
    var buildRequest: BuildNewModelRequest? {
        guard isValid else { return nil }
        switch initSeedEntry {
        case .drawnAtBuild:
            return BuildNewModelRequest(architecture: architecture, enteredInitSeed: nil)
        case let .entered(seed):
            return BuildNewModelRequest(architecture: architecture, enteredInitSeed: seed)
        case .invalid:
            return nil
        }
    }

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
        return availablePresets.first(where: { $0.named.architecture == a })?.named
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
