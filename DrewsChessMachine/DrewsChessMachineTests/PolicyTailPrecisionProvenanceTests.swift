//
//  PolicyTailPrecisionProvenanceTests.swift
//  DrewsChessMachineTests
//
//  The policy-head tail precision (`--policy-tail-precision`) changes a
//  bf16 / fp16 model's head arithmetic but used to be recorded nowhere: the
//  GUI, train-vs-UCI, UCI, probes and the bot silently took the build's
//  default, and nothing in a checkpoint, session or results file said which
//  value a lineage trained under. These pin the single resolver, the
//  per-process default every builder uses, the trainer-file metadata key, and
//  the resume decisions.
//

import XCTest
@testable import DrewsChessMachine

final class PolicyTailPrecisionProvenanceTests: XCTestCase {

    typealias Precision = ChessNetwork.PolicyTailPrecision

    // MARK: - Resolver

    func testResolverReadsTheFlagOrTheDefault() throws {
        XCTAssertEqual(try Precision.resolve(arguments: ["--uci"]),
                       Precision.Resolution(value: .default, source: .defaultValue))
        XCTAssertEqual(try Precision.resolve(arguments: ["--replay-corpus", "x", "--policy-tail-precision", "fp32_from_pre_bn"]),
                       Precision.Resolution(value: .float32FromPreBatchNorm, source: .flag))
        XCTAssertEqual(try Precision.resolve(arguments: ["--policy-tail-precision", "mixed_final_projection"]),
                       Precision.Resolution(value: .mixedFinalProjection, source: .flag))
    }

    func testResolverRefusesBadUses() {
        XCTAssertThrowsError(try Precision.resolve(arguments: ["--policy-tail-precision"])) { error in
            XCTAssertEqual(error as? Precision.ResolutionError, .missingValue)
        }
        XCTAssertThrowsError(try Precision.resolve(arguments: ["--policy-tail-precision", "--uci"])) { error in
            XCTAssertEqual(error as? Precision.ResolutionError, .missingValue)
        }
        XCTAssertThrowsError(try Precision.resolve(arguments: ["--policy-tail-precision", "fp16"])) { error in
            XCTAssertEqual(error as? Precision.ResolutionError, .unknownValue("fp16"))
        }
        XCTAssertThrowsError(try Precision.resolve(arguments: [
            "--policy-tail-precision", "fp32_from_pre_bn", "--policy-tail-precision", "fp32_from_pre_bn",
        ])) { error in
            XCTAssertEqual(error as? Precision.ResolutionError, .repeated(count: 2))
        }
    }

    // MARK: - Builders take the process value

    func testTheTestProcessRunsTheDefault() {
        XCTAssertEqual(Precision.process, .default)
        XCTAssertEqual(Precision.processResolution.source, .defaultValue)
    }

    @MainActor
    func testNetworksAndTrainersAreBuiltWithTheProcessValue() throws {
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 1))
        XCTAssertEqual(network.network.policyTailPrecision, Precision.process)
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.policyTailPrecision, Precision.process)
        let viaHyperparameters = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1),
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: .current,
            initialization: .seeded(initSeed: 2)
        )
        XCTAssertEqual(viaHyperparameters.policyTailPrecision, Precision.process)
    }

    // MARK: - Trainer-file metadata

    func testTrainerFileMetadataRoundTripsThePrecision() async throws {
        let arch = NetworkArchitecture.current
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: arch, initialization: .seeded(initSeed: 1), policyTailPrecision: .float32FromPreBatchNorm)
        let snapshot = try await trainer.exportResumeSnapshot()
        let metadata = ModelCheckpointMetadata.trainerFile(
            creator: "test",
            trainingStep: snapshot.schedule.completedTrainSteps,
            parentModelID: "",
            notes: "",
            schedule: snapshot.schedule,
            policyTailPrecision: trainer.policyTailPrecision
        )
        let data = try SafetensorsModelIO.encode(
            modelID: "20261002-1-TEST",
            createdAtUnix: 0,
            metadata: metadata,
            weights: snapshot.trainerWeights,
            architecture: arch,
            includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: metadata.trainerSchedule.map(\.completedTrainSteps), corpus: nil)
        )
        let header = try SafetensorsFile.decode(data)
        XCTAssertEqual(header.metadata[SafetensorsModelIO.Key.trainerPolicyTailPrecision], "fp32_from_pre_bn")
        let decoded = try SafetensorsModelIO.decode(data)
        XCTAssertEqual(decoded.file.metadata.trainerPolicyTailPrecision, .float32FromPreBatchNorm)
    }

    func testAPlainModelFileRecordsNoTrainerPrecision() {
        let metadata = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "")
        XCTAssertNil(metadata.trainerPolicyTailPrecision)
    }

    // MARK: - Resume gap

    /// A recorded mismatch, or a checkpoint that predates recording the
    /// precision, is the `policy_tail` resume gap; a match is none.
    func testADifferentOrUnrecordedPrecisionIsAPolicyTailGap() {
        XCTAssertEqual(PolicyTailPrecisionResume.gaps(saved: .mixedFinalProjection, running: .mixedFinalProjection), [])
        XCTAssertEqual(PolicyTailPrecisionResume.gaps(saved: .float32FromPreBatchNorm, running: .mixedFinalProjection),
                       [.policyTail])
        XCTAssertEqual(PolicyTailPrecisionResume.gaps(saved: nil, running: .mixedFinalProjection), [.policyTail])
        XCTAssertTrue(PolicyTailPrecisionResume.exactResumeLogLine(saved: .float32FromPreBatchNorm, running: .mixedFinalProjection)
            .contains("\(ChessNetwork.PolicyTailPrecision.flag) fp32_from_pre_bn"))
    }
}
