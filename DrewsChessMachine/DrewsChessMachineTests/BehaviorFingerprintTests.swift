//
//  BehaviorFingerprintTests.swift
//  DrewsChessMachineTests
//
//  Determinism plan C1 #33 as refined by the owner: a changed build or OS is
//  a resume gap only when it changes what the run computes, which the
//  behavior fingerprint detects. Same build ⇒ same fingerprint; perturbing a
//  covered behavior ⇒ a different one; a different recipe never compares
//  equal.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class BehaviorFingerprintTests: XCTestCase {

    /// A small architecture, so each computation stays fast.
    private static func architecture(
        encoding: InputEncoding = .basic30, seReductionRatio: Int = 4, compute: ComputeDataType = .float32
    ) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: encoding, channels: 16, numBlocks: 2, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: seReductionRatio,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: compute
        )
    }

    private let fp32 = BehaviorFingerprint.Settings(
        arch: BehaviorFingerprintTests.architecture(), policyTailPrecision: .float32FromPreBatchNorm)

    /// Run streams named under another derivation: every stream's master
    /// seed is shifted, as a changed derivation would shift them.
    private enum ShiftedStreams: FingerprintStreamSource {
        static func streams(masterSeed: UInt64) -> DCMRandomStreams {
            DCMRandomStreams(masterSeed: masterSeed ^ 0x1)
        }
    }

    func testTheSameBuildComputesTheSameFingerprint() async throws {
        let first = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let second = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.recipe, BehaviorFingerprint.recipe)
        XCTAssertEqual(first.sha256.count, 64)
        let cached = try await BehaviorFingerprint.compute(for: fp32)
        XCTAssertEqual(cached, first)
    }

    func testAnotherStreamDerivationChangesTheFingerprint() async throws {
        let base = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let shifted = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: ShiftedStreams.self)
        XCTAssertNotEqual(base.sha256, shifted.sha256)
    }

    func testNumericsSettingsChangeTheFingerprint() async throws {
        let base = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let bf16 = BehaviorFingerprint.Settings(arch: Self.architecture(compute: .bFloat16),
                                                policyTailPrecision: .float32FromPreBatchNorm)
        let bf16Fingerprint = try await BehaviorFingerprint.computeUncached(for: bf16, streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(base.sha256, bf16Fingerprint.sha256)
        let otherTail = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(arch: Self.architecture(compute: .bFloat16),
                                              policyTailPrecision: .mixedFinalProjection),
            streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(bf16Fingerprint.sha256, otherTail.sha256)
        let otherEncoding = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(arch: Self.architecture(encoding: .basic20),
                                              policyTailPrecision: .float32FromPreBatchNorm),
            streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(base.sha256, otherEncoding.sha256)
    }

    /// The fingerprint trains the checkpoint's own architecture, so a change
    /// in one of its blocks — here only the SE reduction ratio — changes it.
    func testAnArchitectureOnlyChangeChangesTheFingerprint() async throws {
        let base = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let otherSE = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(arch: Self.architecture(seReductionRatio: 2),
                                              policyTailPrecision: .float32FromPreBatchNorm),
            streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(base.sha256, otherSE.sha256)
    }

    /// A history encoding trains from stacks the buffer rebuilds, and is as
    /// repeatable as a single-frame one.
    func testAHistoryEncodingFingerprintIsRepeatable() async throws {
        let settings = BehaviorFingerprint.Settings(arch: Self.architecture(encoding: .full10ply200),
                                                    policyTailPrecision: .float32FromPreBatchNorm)
        let first = try await BehaviorFingerprint.computeUncached(for: settings, streamDerivation: DCMRandomStreams.self)
        let second = try await BehaviorFingerprint.computeUncached(for: settings, streamDerivation: DCMRandomStreams.self)
        XCTAssertEqual(first, second)
    }

    /// A recipe-1 fingerprint (the fixed tiny network) never matches one of
    /// this recipe, so a file saved under it resumes with a `build` gap.
    func testARecipeOneFingerprintIsAGapUnderTheCurrentRecipe() throws {
        XCTAssertEqual(BehaviorFingerprint.recipe, 3)
        let running = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let saved = try record(fingerprint: BehaviorFingerprint.Record(recipe: 1, sha256: "ab12"))
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: saved, runningBuild: try laterBuild(than: saved.build),
                                                 runningDevice: saved.device, runningFingerprint: running).gaps, [.build])
    }

    /// A recipe-2 fingerprint was written by a build whose corpus replay and
    /// train-vs-UCI sampled every batch uniformly, so under recipe 3 a file
    /// carrying one resumes with a `build` gap, even with an equal hash.
    func testARecipeTwoFingerprintIsAGapUnderTheCurrentRecipe() throws {
        let running = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let saved = try record(fingerprint: BehaviorFingerprint.Record(recipe: 2, sha256: "ab12"))
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: saved, runningBuild: try laterBuild(than: saved.build),
                                                 runningDevice: saved.device, runningFingerprint: running).gaps, [.build])
    }

    /// The recipe's sampler draws cover the per-game cap, the draw cap and
    /// the length tilt: a change in how any of them shapes a batch changes
    /// the digest. Each variant replaces one recipe constraint with one the
    /// sampler applies differently.
    func testASamplingConstraintChangeChangesTheSamplerDraws() throws {
        let recipe = BehaviorFingerprint.samplerConstraints
        let base = try BehaviorFingerprint.samplerDrawsDigest(constraints: recipe)
        XCTAssertEqual(try BehaviorFingerprint.samplerDrawsDigest(constraints: recipe), base, "the draws repeat")
        let binding = try XCTUnwrap(recipe.last)
        XCTAssertLessThan(binding.maxPerGame, Int.max, "the recipe's last constraint caps games")
        let variants: [(String, ReplayBuffer.SamplingConstraints)] = [
            ("per-game cap", ReplayBuffer.SamplingConstraints(
                maxPerGame: binding.maxPerGame + 1, maxDrawPercent: binding.maxDrawPercent,
                targetMeanGameLengthPlies: binding.targetMeanGameLengthPlies, materialBucketWeights: nil)),
            ("draw cap", ReplayBuffer.SamplingConstraints(
                maxPerGame: binding.maxPerGame, maxDrawPercent: 0,
                targetMeanGameLengthPlies: binding.targetMeanGameLengthPlies, materialBucketWeights: nil)),
            ("length tilt", ReplayBuffer.SamplingConstraints(
                maxPerGame: binding.maxPerGame, maxDrawPercent: binding.maxDrawPercent,
                targetMeanGameLengthPlies: 0, materialBucketWeights: nil)),
        ]
        for (what, variant) in variants {
            let changed = Array(recipe.dropLast()) + [variant]
            XCTAssertNotEqual(try BehaviorFingerprint.samplerDrawsDigest(constraints: changed), base,
                              "a different \(what) must change the recipe's sampler draws")
        }
    }

    // MARK: - The resume decision

    private func record(fingerprint: BehaviorFingerprint.Record?) throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"],
                                         startedAt: start, segmentStartTrainerStep: 0)
        return try tracker.record(at: start, trainerCompletedSteps: 1, segmentLocalStep: 1, segmentGames: 0,
                                  segmentPositions: 0, corpus: nil, parameters: nil,
                                  rng: LineageRecord.RNG(dropoutPhiloxState: nil, streams: nil,
                                                         behaviorFingerprint: fingerprint),
                                  inputs: tracker.testInputs)
    }

    private func laterBuild(than build: LineageRecord.Build) throws -> LineageRecord.Build {
        try LineageRecord.Build(buildNumber: build.buildNumber + 1, gitHash: build.gitHash + "x",
                                gitBranch: build.gitBranch, gitDirty: build.gitDirty,
                                gitDiffSHA256: build.gitDiffSHA256, xcodeBuild: build.xcodeBuild,
                                sdkBuild: build.sdkBuild, configuration: build.configuration)
    }

    func testARebuildWithAMatchingFingerprintIsNotAGap() throws {
        let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let saved = try record(fingerprint: fingerprint)
        let comparison = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: try laterBuild(than: saved.build),
                                                   runningDevice: saved.device, runningFingerprint: fingerprint)
        XCTAssertEqual(comparison.gaps, [])
        XCTAssertEqual(comparison.logLines.count, 1)
        XCTAssertTrue(comparison.logLines[0].contains("behavior fingerprint matches"), comparison.logLines[0])

        let otherOS = LineageRecord.Device(hardwareModel: saved.device.hardwareModel, cpu: saved.device.cpu,
                                           isVirtualMachine: saved.device.isVirtualMachine,
                                           osVersion: saved.device.osVersion + " (later)", gpu: saved.device.gpu)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: saved, runningBuild: saved.build, runningDevice: otherOS,
                                                 runningFingerprint: fingerprint).gaps, [])
    }

    func testADifferentMissingOrOtherRecipeFingerprintIsAGap() throws {
        let running = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let cases: [BehaviorFingerprint.Record?] = [
            BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "cd34"),
            nil,
            BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe + 1, sha256: "ab12"),
        ]
        for savedFingerprint in cases {
            let saved = try record(fingerprint: savedFingerprint)
            let comparison = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: try laterBuild(than: saved.build),
                                                       runningDevice: saved.device, runningFingerprint: running)
            XCTAssertEqual(comparison.gaps, [.build], "\(String(describing: savedFingerprint))")
        }
    }

    /// The saved record's device with every field the comparison covers
    /// changed except the OS version, so only a device change is in play.
    private func otherMac(than device: LineageRecord.Device) -> LineageRecord.Device {
        LineageRecord.Device(hardwareModel: (device.hardwareModel ?? "Mac") + " (other)",
                             cpu: (device.cpu ?? "chip") + " (other)",
                             isVirtualMachine: device.isVirtualMachine.map { !$0 } ?? true,
                             osVersion: device.osVersion,
                             gpu: (device.gpu ?? "gpu") + " (other)")
    }

    /// A resume on another Mac under the same build and OS: the GPU kernels,
    /// and so what a training step computes, can differ, which the behavior
    /// fingerprint detects. A different fingerprint makes it a `device` gap.
    func testADeviceChangeWithADifferentFingerprintIsAGap() throws {
        let running = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let saved = try record(fingerprint: BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "cd34"))
        let comparison = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: saved.build,
                                                   runningDevice: otherMac(than: saved.device),
                                                   runningFingerprint: running)
        XCTAssertEqual(comparison.gaps.map(\.token), ["device"])
        XCTAssertEqual(comparison.logLines.count, 1)
        XCTAssertTrue(comparison.logLines.first?.hasPrefix("[RESUME] device changed (") ?? false,
                      comparison.logLines.first ?? "no line")
        XCTAssertTrue(comparison.logLines.first?.hasSuffix(", behavior fingerprint differs") ?? false,
                      comparison.logLines.first ?? "no line")

        // Each field alone is a change.
        let device = saved.device
        let oneFieldChanges = [
            LineageRecord.Device(hardwareModel: (device.hardwareModel ?? "Mac") + " (other)", cpu: device.cpu,
                                 isVirtualMachine: device.isVirtualMachine, osVersion: device.osVersion, gpu: device.gpu),
            LineageRecord.Device(hardwareModel: device.hardwareModel, cpu: (device.cpu ?? "chip") + " (other)",
                                 isVirtualMachine: device.isVirtualMachine, osVersion: device.osVersion, gpu: device.gpu),
            LineageRecord.Device(hardwareModel: device.hardwareModel, cpu: device.cpu,
                                 isVirtualMachine: device.isVirtualMachine.map { !$0 } ?? true,
                                 osVersion: device.osVersion, gpu: device.gpu),
            LineageRecord.Device(hardwareModel: device.hardwareModel, cpu: device.cpu,
                                 isVirtualMachine: device.isVirtualMachine, osVersion: device.osVersion,
                                 gpu: (device.gpu ?? "gpu") + " (other)"),
        ]
        for changed in oneFieldChanges {
            XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: saved, runningBuild: saved.build, runningDevice: changed,
                                                     runningFingerprint: running).gaps.map(\.token), ["device"],
                           "\(changed)")
        }

        // The same Mac is no change, and logs nothing.
        let same = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: saved.build, runningDevice: saved.device,
                                             runningFingerprint: running)
        XCTAssertEqual(same.gaps, [])
        XCTAssertEqual(same.logLines, [])
    }

    /// Another Mac whose fingerprint matches computes what the saved run
    /// computed: the change is logged, and it is not a gap.
    func testADeviceChangeWithAMatchingFingerprintIsNotAGap() throws {
        let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let saved = try record(fingerprint: fingerprint)
        let comparison = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: saved.build,
                                                   runningDevice: otherMac(than: saved.device),
                                                   runningFingerprint: fingerprint)
        XCTAssertEqual(comparison.gaps, [])
        XCTAssertEqual(comparison.logLines.count, 1)
        XCTAssertTrue(comparison.logLines.first?.hasPrefix("[RESUME] device changed (") ?? false,
                      comparison.logLines.first ?? "no line")
        XCTAssertTrue(comparison.logLines.first?.hasSuffix(", behavior fingerprint matches") ?? false,
                      comparison.logLines.first ?? "no line")
    }

    /// The fingerprint under heavy concurrent GPU work — another trainer
    /// stepping large batches on the same GPU while it is computed — equals
    /// the one computed on an idle GPU. A fingerprint that moved with load
    /// would turn every resume on a busy machine into a `build` / `os` /
    /// `device` gap. Computed for the architecture new models are built
    /// with, not a toy, so the kernels it exercises are production ones.
    func testTheFingerprintRepeatsUnderConcurrentGPULoad() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let arch = NetworkArchitecture.newModelDefault
        let settings = BehaviorFingerprint.Settings(arch: arch, policyTailPrecision: .process)
        let idle = try await BehaviorFingerprint.computeUncached(for: settings, streamDerivation: DCMRandomStreams.self)

        let load = try ChessTrainer(dropoutStream: DCMRandom(seed: 9), arch: arch, initialization: .seeded(initSeed: 9))
        let loadBatchSize = 1024
        let stopLoad = SyncBox(false)
        let loadEnded = SyncBox(false)
        let loadSteps = SyncBox(0)
        let (underLoad, stepsDuringLoadedComputations) = try await withThrowingTaskGroup(
            of: Void.self,
            returning: ([BehaviorFingerprint.Record], Int).self
        ) { group in
            group.addTask {
                defer { loadEnded.value = true }
                while !stopLoad.value {
                    _ = try await load.trainStep(batchSize: loadBatchSize)
                    loadSteps.modify { $0 += 1 }
                }
            }
            // Start measuring only once the load is on the GPU.
            while loadSteps.value == 0 && !loadEnded.value {
                try await Task.sleep(for: .milliseconds(10))
            }
            var fingerprints: [BehaviorFingerprint.Record] = []
            let stepsBefore = loadSteps.value
            if !loadEnded.value {
                for _ in 0..<2 {
                    fingerprints.append(try await BehaviorFingerprint.computeUncached(
                        for: settings, streamDerivation: DCMRandomStreams.self))
                }
            }
            let stepsDuring = loadSteps.value - stepsBefore
            stopLoad.value = true
            // Rethrows the load task's error, if it stopped on one.
            try await group.waitForAll()
            return (fingerprints, stepsDuring)
        }
        XCTAssertEqual(underLoad.count, 2, "the load stopped before the loaded computations ran")
        XCTAssertGreaterThan(stepsDuringLoadedComputations, 0,
                             "the load trainer must step while the loaded fingerprints are computed")
        for (index, loaded) in underLoad.enumerated() {
            XCTAssertEqual(loaded, idle, "fingerprint \(index + 1) under load differs from the idle one")
        }
    }

    func testTheFingerprintRoundTripsInTheRecord() throws {
        let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ef56")
        let saved = try record(fingerprint: fingerprint)
        let text = try saved.jsonText()
        XCTAssertTrue(text.contains("\"behavior_fingerprint\""), text)
        XCTAssertEqual(try LineageRecord.decode(jsonText: text).rng.behaviorFingerprint, fingerprint)
    }
}
