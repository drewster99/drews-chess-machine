//
//  PolicyTailPrecisionProvenanceTests.swift
//  DrewsChessMachineTests
//
//  The policy-head tail precision changes a bf16 / fp16 model's head
//  arithmetic. It was a process-wide launch flag (`--policy-tail-precision`),
//  recorded on trainer files as the flat `trainer_policy_tail_precision` key
//  and compared on resume as the `policy_tail` gap. From format v12 it is an
//  architecture field (POLICY_TAIL_ARCHITECTURE_PLAN.md PT-D1, PT-D4): it
//  travels with every model file inside the architecture, the flag and the
//  flat key are no longer written, and a resume cannot differ from the
//  file's tail because it builds the trainer from the file's architecture.
//  These pin what replaced the resolver, the process value and the gap.
//

import XCTest
@testable import DrewsChessMachine

final class PolicyTailPrecisionProvenanceTests: XCTestCase {

    // MARK: - Trainer-file metadata

    /// A trainer file states its tail in its architecture and nowhere else:
    /// no flat key, and the lineage configuration copy comes from the same
    /// architecture.
    func testATrainerFileCarriesItsTailInTheArchitectureOnly() async throws {
        var arch = NetworkArchitecture.current
        arch.policyTailPrecision = .float32FromPreBatchNorm
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: arch, initialization: .seeded(initSeed: 1))
        let snapshot = try await trainer.exportResumeSnapshot()
        let metadata = ModelCheckpointMetadata.trainerFile(
            creator: "test",
            trainingStep: snapshot.schedule.completedTrainSteps,
            parentModelID: "",
            notes: "",
            schedule: snapshot.schedule
        )
        let data = try SafetensorsModelIO.encode(
            modelID: "20261002-1-TEST",
            createdAtUnix: 0,
            metadata: metadata,
            weights: snapshot.trainerWeights,
            architecture: trainer.arch,
            includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: metadata.trainerSchedule.map(\.completedTrainSteps), corpus: nil)
        )
        let header = try SafetensorsFile.decode(data)
        XCTAssertNil(header.metadata[SafetensorsModelIO.Key.trainerPolicyTailPrecision])
        let decoded = try SafetensorsModelIO.decode(data)
        XCTAssertEqual(decoded.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertNil(decoded.architectureFormat.legacyLogLine, "a current file states its tail")
    }

    // MARK: - Resume gap

    /// The `policy_tail` gap is gone: a resume builds the trainer from the
    /// file's own architecture. `--accept-inexact policy_tail` is refused as
    /// an unknown token, like any removed gap.
    func testPolicyTailIsNoLongerAResumeGap() {
        XCTAssertNil(ResumeGap(rawValue: "policy_tail"))
        XCTAssertFalse(ResumeGap.allCases.map(\.token).contains("policy_tail"))
        XCTAssertThrowsError(try ResumeGap.parseAcceptList("policy_tail")) { error in
            XCTAssertEqual(error as? ResumeExactnessError, .unknownAcceptToken("policy_tail"))
        }
    }

    // MARK: - Launch flag

    /// The flag is gone from the help text; passing it is an unknown-argument
    /// error in every mode, which the argument scanners decide without any
    /// entry for it.
    func testTheLaunchFlagIsNotDocumented() {
        XCTAssertFalse(CommandLineHelp.usageText.contains("--policy-tail-precision"))
    }
}
