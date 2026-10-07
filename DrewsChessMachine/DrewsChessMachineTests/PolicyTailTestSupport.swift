//
//  PolicyTailTestSupport.swift
//  DrewsChessMachineTests
//
//  Test fixtures change an architecture's compute dtype through
//  `NetworkArchitecture.withComputeDataType(_:tail:)`, the one function that
//  keeps the policy tail consistent with it (format v12; `computeDataType` is
//  `private(set)`). Fixtures that only want "this architecture in another
//  dtype" take the tail every network was built under before the tail became
//  an architecture field — the process default, `mixed_final_projection` — so
//  their expectations are unchanged.
//

@testable import DrewsChessMachine

extension NetworkArchitecture {
    /// This architecture at `computeDataType`: `does_not_apply` on fp32,
    /// `mixed_final_projection` on bf16 / fp16.
    func withComputeDataTypeForTests(_ computeDataType: ComputeDataType) throws -> NetworkArchitecture {
        try withComputeDataType(computeDataType, tail: computeDataType == .float32 ? nil : .mixedFinalProjection)
    }
}
