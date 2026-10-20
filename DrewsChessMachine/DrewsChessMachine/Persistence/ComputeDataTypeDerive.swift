//
//  ComputeDataTypeDerive.swift
//  DrewsChessMachine
//
//  `--derive-model --set-compute-dtype <value>`: set the architecture's
//  `compute_data_type` (the dtype the tower builds and runs in). Like
//  `--set-policy-tail-precision`, no tensor depends on it — weights are always
//  stored fp32 and cast to the compute dtype at load (`ChessNetwork`'s load
//  path), so every tensor is copied bit-exact and a trained plain-model source
//  is accepted. The motivating case is a same-weights fp32 / bf16 / fp16 A/B
//  for inference-speed or numerics comparison, which otherwise had no CLI path
//  (Build New Model mints a fresh net; `withComputeDataType` was reachable only
//  from the GUI and the numerics audit).
//
//  Going to fp32 drops the policy tail to `does_not_apply`. Going to bf16 /
//  fp16 needs a tail: this preserves the source's tail when it is already a
//  reduced-precision value, otherwise defaults to `fp32_from_pre_bn` (keeps the
//  policy tail in fp32, so a fp32->bf16->fp16 sweep differs only in the main
//  tower dtype). Dtype/tail consistency is enforced in one place by
//  `NetworkArchitecture.withComputeDataType(_:tail:)`, the one blessed mutator.
//

import Foundation

struct SetComputeDataTypeDeriveOperation: DeriveOperation {
    let value: ComputeDataType

    static let kind = DeriveOperationKind(
        name: "set-compute-dtype",
        flag: "--set-compute-dtype",
        valueSyntax: ComputeDataType.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set compute_data_type, the dtype the tower builds and runs in: "
            + "\(ComputeDataType.float32.rawValue), \(ComputeDataType.bFloat16.rawValue) or "
            + "\(ComputeDataType.float16.rawValue). Weights are stored fp32 and cast at load, so no tensor is "
            + "rewritten and a trained model is accepted. To fp32 the policy tail becomes "
            + "\(PolicyTailPrecisionSetting.doesNotApply.rawValue); to bf16 / fp16 the source's tail is kept when "
            + "already reduced-precision, else \(PolicyTailPrecisionSetting.float32FromPreBatchNorm.rawValue).",
        changedArchitectureFields: [
            NetworkArchitecture.CodingKeys.computeDataType.rawValue,
            NetworkArchitecture.CodingKeys.policyTailPrecision.rawValue,
        ],
        rewrittenTensorsDescription: "none",
        acceptsGroupSelection: false,
        make: { value, _ in
            guard let dtype = ComputeDataType(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-compute-dtype",
                    detail: "'\(value)' is not one of \(ComputeDataType.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetComputeDataTypeDeriveOperation(value: dtype)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] { ["value": value.rawValue] }

    /// Refuses a dtype the model already holds ("nothing to derive"). The tail
    /// consistent with the new dtype is chosen here; `withComputeDataType`
    /// validates the pair, so a bad combination throws there, not here.
    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard architecture.computeDataType != value else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "compute_data_type is already '\(value.rawValue)'; nothing to derive")
        }
        let tail: PolicyTailPrecisionSetting?
        if value == .float32 {
            tail = nil
        } else if PolicyTailPrecisionSetting.reducedPrecisionCases.contains(architecture.policyTailPrecision) {
            tail = architecture.policyTailPrecision
        } else {
            tail = .float32FromPreBatchNorm
        }
        return try architecture.withComputeDataType(value, tail: tail)
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        []
    }
}
