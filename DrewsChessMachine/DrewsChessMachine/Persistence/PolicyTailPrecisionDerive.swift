//
//  PolicyTailPrecisionDerive.swift
//  DrewsChessMachine
//
//  `--derive-model --set-policy-tail-precision <value>`: set the
//  architecture's `policy_tail_precision` (format v12; plan
//  `documentation/plans-active/POLICY_TAIL_ARCHITECTURE_PLAN.md` §5). The tail
//  decides only where the policy head leaves the compute dtype for fp32 — no
//  tensor depends on it — so every tensor is copied bit-exact, and the
//  operation runs on a trained plain-model source (a trainer-state file is
//  still refused by the engine's plain-source guardrail, like every derive).
//  This is how a model gets a tail other than the one it was built or
//  recorded with, now that the process-wide `--policy-tail-precision` launch
//  flag is gone: derive the start model with the tail set.
//
//  Dtype consistency is not checked here: the engine validates the target
//  architecture, whose `validate()` refuses `does_not_apply` on bf16 / fp16
//  and a tail on fp32, so the rule lives in one place.
//

import Foundation

struct SetPolicyTailPrecisionDeriveOperation: DeriveOperation {
    let value: PolicyTailPrecisionSetting

    static let kind = DeriveOperationKind(
        name: "set-policy-tail-precision",
        flag: "--set-policy-tail-precision",
        valueSyntax: PolicyTailPrecisionSetting.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set policy_tail_precision, where the policy head leaves the compute dtype for fp32: "
            + "\(PolicyTailPrecisionSetting.mixedFinalProjection.rawValue) (pre-block and final projection in the "
            + "compute dtype, only the logits widened) or \(PolicyTailPrecisionSetting.float32FromPreBatchNorm.rawValue) "
            + "(fp32 from the pre-block's BatchNorm on) on a bf16 / fp16 model; "
            + "\(PolicyTailPrecisionSetting.doesNotApply.rawValue) is the only value an fp32 model holds. "
            + "No tensor depends on it, so every tensor is copied bit-exact and a trained model is accepted.",
        changedArchitectureFields: [NetworkArchitecture.CodingKeys.policyTailPrecision.rawValue],
        rewrittenTensorsDescription: "none",
        acceptsGroupSelection: false,
        make: { value, _ in
            guard let tail = PolicyTailPrecisionSetting(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-policy-tail-precision",
                    detail: "'\(value)' is not one of \(PolicyTailPrecisionSetting.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetPolicyTailPrecisionDeriveOperation(value: tail)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] { ["value": value.rawValue] }

    /// Refuses a value the model already holds, like every other derive
    /// operation ("nothing to derive"); a value inconsistent with the compute
    /// dtype is refused by the engine's `target.validate()`.
    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard architecture.policyTailPrecision != value else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "policy_tail_precision is already '\(value.rawValue)'; nothing to derive")
        }
        var edited = architecture
        edited.policyTailPrecision = value
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        []
    }
}
