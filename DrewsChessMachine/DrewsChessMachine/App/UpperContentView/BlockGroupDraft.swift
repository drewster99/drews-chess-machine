//
//  BlockGroupDraft.swift
//  DrewsChessMachine
//
//  One block group being edited on the Build-New-Model screen.
//

import Observation

/// One editable block group of the Build-New-Model screen, addressed by
/// identity rather than by array position.
///
/// Binding each control to `blockGroups[i]` with the row's integer index
/// captured does not survive a delete. SwiftUI retains a control's binding
/// past the update that removes its row, so a retained binding re-reads
/// `blockGroups[i]` with a stale `i`: out of range for the last group (a trap
/// — it crashed the app through the Output-norm Picker), and silently the
/// *next* group for any other, so a focused field committing during the
/// teardown writes into the neighbouring group. Index-guarded closure
/// bindings only move the trap around and need made-up fallback values for
/// the getter. `ForEach($array)` does not help either: its element bindings
/// are index-addressed closures of exactly the same kind.
///
/// A reference-type draft per row removes the index entirely. Each control
/// binds through the draft object it was built with, so a binding retained
/// past a delete reads and writes a draft that is no longer in the model —
/// it can neither trap nor touch another group.
@MainActor
@Observable
final class BlockGroupDraft: Identifiable {

    /// The group's full recipe, edited in place by the row's controls.
    var group: BlockGroup

    /// Called after every edit of this group that can change which
    /// architecture-level activation sites exist (its activation style: the
    /// first group's decides the stem, the last group's the tower end). The
    /// model that owns the draft passes its site sync, so a site that
    /// disappears drops its choice inside the edit itself. A `let`, so
    /// observation does not track it.
    let onTopologyChange: @MainActor () -> Void

    init(_ group: BlockGroup, onTopologyChange: @escaping @MainActor () -> Void) {
        self.group = group
        self.onTopologyChange = onTopologyChange
    }

    /// The group's main-path activation, set through the rule the Build
    /// screen and `--derive-model --set-activation` share
    /// (`BlockGroup.setActivationFunction`).
    var activationFunction: ActivationFunction {
        get { group.activationFunction }
        set { group.setActivationFunction(newValue) }
    }

    /// The group's activation style, set through the draft so the model's
    /// site sync runs inside the edit (`onTopologyChange`). The style picker
    /// is the screen's only writer of a group's style and binds this, never
    /// `group.activationStyle` directly.
    var activationStyle: BlockActivationStyle {
        get { group.activationStyle }
        set {
            group.activationStyle = newValue
            onTopologyChange()
        }
    }

    /// The group's output normalization as a non-optional value for the
    /// picker: reads through `resolvedOutputNorm` (a legacy `nil` is `.none`)
    /// and writes the explicit value.
    var outputNorm: BlockOutputNorm {
        get { group.resolvedOutputNorm }
        set { group.outputNorm = newValue }
    }
}
