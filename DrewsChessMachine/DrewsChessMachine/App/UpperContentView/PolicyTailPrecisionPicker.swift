//
//  PolicyTailPrecisionPicker.swift
//  DrewsChessMachine
//
//  The Build New Model screen's "Policy tail" picker, beside "Compute dtype"
//  in the Precision section: where the policy head leaves the compute dtype
//  for fp32 (`PolicyTailPrecisionSetting`, an architecture field from format
//  v12). On an fp32 model there is no narrow dtype to leave, so the picker
//  stays on screen, disabled, showing "does not apply" — the same rule the
//  activation-site pickers follow for a site the topology lacks, and the same
//  view either way so its identity never changes with the dtype.
//

import SwiftUI

struct PolicyTailPrecisionPicker: View {
    @Bindable var model: BuildNewModelModel

    /// What the picker shows for a compute dtype — kept as a value so a test
    /// can read the rule without drawing the control.
    struct Presentation: Equatable {
        let isEnabled: Bool
        let entries: [PolicyTailPrecisionSetting]
        let help: String
    }

    static func presentation(computeDataType: ComputeDataType) -> Presentation {
        guard computeDataType != .float32 else {
            return Presentation(
                isEnabled: false, entries: [.doesNotApply],
                help: "Does not apply: an fp32 model computes the whole policy head in fp32.")
        }
        return Presentation(
            isEnabled: true, entries: PolicyTailPrecisionSetting.reducedPrecisionCases,
            help: "Where the policy head leaves the compute dtype for fp32. "
                + "\(PolicyTailPrecisionSetting.mixedFinalProjection.rawValue): the pre-block and final projection "
                + "run in the compute dtype and only the logits are widened (faster). "
                + "\(PolicyTailPrecisionSetting.float32FromPreBatchNorm.rawValue): fp32 from the pre-block's "
                + "BatchNorm on (slower; closer to an fp32 network on weights with a large shared policy row).")
    }

    var body: some View {
        let presentation = Self.presentation(computeDataType: model.computeDataType)
        Picker(
            "Policy tail",
            selection: presentation.isEnabled ? $model.reducedPrecisionPolicyTail : .constant(.doesNotApply)
        ) {
            ForEach(presentation.entries, id: \.self) { tail in
                Text(tail == .doesNotApply ? "does not apply" : tail.rawValue).tag(tail)
            }
        }
        .disabled(!presentation.isEnabled)
        .help(presentation.help)
    }
}
