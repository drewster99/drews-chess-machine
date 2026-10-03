//
//  BuildInitSeedField.swift
//  DrewsChessMachine
//
//  The Build-New-Model screen's optional Init seed field.
//

import SwiftUI

/// Optional init seed for the build: empty draws one at the build (shown in
/// the network status and logged), a decimal UInt64 reproduces a mint — the
/// same seed and architecture give bit-identical trainable tensors here, in
/// `--new-model --init-seed` and on every machine, and batch-norm running
/// statistics equal to float tolerance (they are calibrated by a GPU forward
/// pass; see `NetworkInitMode.randomWeights`).
struct BuildInitSeedField: View {
    @Bindable var model: BuildNewModelModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            TextField("Init seed", text: $model.initSeedText, prompt: Text("drawn at build"))
                .font(.body.monospaced())
            Text(caption)
                .font(.caption)
                .foregroundStyle(isInvalid ? .orange : .secondary)
        }
    }

    private var isInvalid: Bool {
        if case .invalid = model.initSeedEntry { return true }
        return false
    }

    private var caption: String {
        switch model.initSeedEntry {
        case .drawnAtBuild:
            return "A seed is drawn at the build and shown with the result. Scheme \(WeightInitScheme.current)."
        case .entered:
            return "Same seed + same architecture = same weights. Scheme \(WeightInitScheme.current)."
        case let .invalid(message):
            return message
        }
    }
}
