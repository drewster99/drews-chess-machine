//
//  InitSetButtonsView.swift
//  DrewsChessMachine
//
//  The Build-New-Model screen's "Neutral init" / "Standard init" buttons.
//

import SwiftUI

/// Applies one of the two init sets to every block group and head at once,
/// through `BuildNewModelModel.applyNeutralInit` / `applyStandardInit` — the
/// single source of what "neutral" and "standard" mean — and shows how many
/// options currently differ from the standard init.
struct InitSetButtonsView: View {
    let model: BuildNewModelModel

    var body: some View {
        HStack(spacing: 8) {
            Button("Neutral init") {
                model.applyNeutralInit()
            }
            .help("Start every residual branch, SE gate, width-transition projection and head final layer as a no-op: "
                + "near-identity SE gates, zero last BN γ (post-activation groups), identity-like skip projections, "
                + "zero policy and value final layers. The value head's draw prior is left as it is.")
            Button("Standard init") {
                model.applyStandardInit()
            }
            .help("Set every init option, the draw prior included, back to the standard init every model was built with before these options existed.")
            Spacer()
            Text("\(model.nonStandardInitOptions.count) non-standard")
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }
}
