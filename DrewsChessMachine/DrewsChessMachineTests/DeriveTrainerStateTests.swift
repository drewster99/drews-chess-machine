//
//  DeriveTrainerStateTests.swift
//  DrewsChessMachineTests
//
//  `--derive-model` of a trainer-state file (owner decisions 2026-11-12,
//  `documentation/plans-active/DERIVE_TRAINER_STATE_DTYPE_PLAN.md`): only the
//  precision kinds (`--set-compute-dtype`, `--set-policy-tail-precision`)
//  take one, and their output carries the source's trainer state and
//  lineage record so it continues with `--resume-exact`. Pure: files are
//  built and derived in memory, no GPU.
//
//  What each test protects:
//  - the trainer state survives bit for bit (tensors, velocity NaN payloads
//    included; schedule; gradient-norm history; `training_step` = clock) —
//    a lost or rounded piece would make the "exact" resume a different run;
//  - the record is the source's with exactly one more derivation and the
//    derive's naming (no new run: owner decision OD-B) — a new run would
//    break the run's continuity, a changed build or fingerprint would make
//    the resume compare against the derive instead of the training save;
//  - every other kind, and a request mixing one in, is still refused, so no
//    operation that changes what the velocity was accumulated for can reach
//    trainer state;
//  - a source that cannot be resumed (no schedule, no lineage, inconsistent
//    clock) is refused rather than turned into a file that claims it can;
//  - a legacy schedule comes out in the current form only;
//  - chained derives keep pointing at the training save (OD-C).
//

import XCTest
@testable import DrewsChessMachine

final class DeriveTrainerStateTests: XCTestCase {

    /// The trainer clock of every source below.
    private static let clock = 12

    /// The bf16 architecture the sources were "trained" under: the replay
    /// tests' small net (ReZero, SE, both heads) at bf16 with the fp32 tail,
    /// R-elitecyc's precision.
    static func bf16Architecture() throws -> NetworkArchitecture {
        try ResumeEquivalenceTests.architecture.withComputeDataType(.bFloat16, tail: .float32FromPreBatchNorm)
    }

    private static func schedule() -> TrainerScheduleState {
        TrainerScheduleState(completedTrainSteps: clock, lrWarmupSteps: 3, lrMomentumCycle: .disabled)
    }

    private static func gradNormHistory() throws -> GradientNormHistory {
        var history = GradientNormHistory()
        for step in 1...clock {
            try history.append(trainerStep: step, preClipNorm: Float(step) * 0.5, fedCap: 15)
        }
        return history
    }

    /// Distinct values per tensor, so a reordered or swapped tensor shows,
    /// and a NaN with a payload in the first velocity tensor, so a copy that
    /// goes through arithmetic (which may quiet or change it) shows too.
    private static func trainerWeights(_ arch: NetworkArchitecture) -> [[Float]] {
        let base = arch.weightTensorPlan().enumerated().map { index, spec in
            (0..<spec.elementCount).map { Float(index) * 0.01 + Float($0 % 7) * 0.001 + 0.5 }
        }
        var velocity = arch.trainableTensorPlan().enumerated().map { index, spec in
            (0..<spec.elementCount).map { -Float(index) * 0.002 - Float($0 % 5) * 0.0001 }
        }
        velocity[0][0] = Float(bitPattern: 0x7FC0_1234)
        return base + velocity
    }

    /// A lineage record of a corpus-replay segment at `clock` whose
    /// parameters adopt the schedule (as every trainer-file writer's must).
    private static func trainingRecord(cumTrainerStep: Int = clock) throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(
            start: .fresh(initialization: .forTests, naming: .unnamedWithoutPreset), pathKind: .replay,
            argv: ["DrewsChessMachine", "--replay-corpus", "x"], startedAt: start, segmentStartTrainerStep: 0)
        try tracker.noteSegmentStartForTests(trainerStep: 0, policyTailPrecision: .float32FromPreBatchNorm)
        return try tracker.record(
            at: start.addingTimeInterval(600), trainerCompletedSteps: cumTrainerStep, segmentLocalStep: cumTrainerStep,
            segmentGames: 9, segmentPositions: 400, corpus: nil,
            parameters: try .forTests(adopting: schedule()),
            rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: tracker.testInputs)
    }

    /// A trainer-state file as the replay runner writes one.
    static func trainerFile(_ arch: NetworkArchitecture, record: LineageRecord? = nil,
                            gradNormHistory: Bool = true) throws -> Data {
        try SafetensorsModelIO.encode(
            modelID: "20261112-1-TRST", createdAtUnix: 1_790_000_600,
            metadata: gradNormHistory
                ? .trainerFile(creator: ModelCheckpointMetadata.corpusReplayCreator, trainingStep: clock,
                               parentModelID: "20261111-1-PRNT", notes: "trainer-state source",
                               schedule: schedule(), gradNormHistory: try Self.gradNormHistory())
                : .trainerFile(creator: ModelCheckpointMetadata.corpusReplayCreator, trainingStep: clock,
                               parentModelID: "20261111-1-PRNT", notes: "trainer-state source", schedule: schedule()),
            weights: trainerWeights(arch), architecture: arch, includesVelocity: true,
            lineage: try record ?? trainingRecord())
    }

    private func derive(_ data: Data, _ operations: [any DeriveOperation], modelID: String = "20261112-2-DRVD",
                        name: String? = nil) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: data, sourceName: "source.safetensors", operations: operations, newModelID: modelID,
            createdAtUnix: 1_790_001_000, build: "test", invocationArguments: ["DrewsChessMachine", "--derive-model"],
            renamedTo: name)
    }

    private func assertSameTensorsBitExact(_ derived: Data, _ source: Data, file: StaticString = #filePath, line: UInt = #line) throws {
        let (sourceTensors, _) = try SafetensorsFile.decode(source)
        let (derivedTensors, _) = try SafetensorsFile.decode(derived)
        XCTAssertEqual(derivedTensors.map(\.name), sourceTensors.map(\.name), "tensor names and order", file: file, line: line)
        XCTAssertEqual(derivedTensors.map(\.shape), sourceTensors.map(\.shape), "tensor shapes", file: file, line: line)
        for (d, s) in zip(derivedTensors, sourceTensors) {
            XCTAssertEqual(d.data.map(\.bitPattern), s.data.map(\.bitPattern), "\(s.name) bit for bit", file: file, line: line)
        }
        XCTAssertTrue(sourceTensors.contains { $0.name.hasPrefix("opt.") }, "the source carries velocity", file: file, line: line)
    }

    // MARK: - What is carried

    func testComputeDtypeDeriveCarriesTheTrainerStateBitExact() throws {
        let source = try Self.trainerFile(try Self.bf16Architecture())
        let result = try derive(source, [SetComputeDataTypeDeriveOperation(value: .float32)])

        try assertSameTensorsBitExact(result.data, source)
        let decoded = try SafetensorsModelIO.decode(result.data, valueHead: .asStored)
        XCTAssertEqual(decoded.architecture.computeDataType, .float32)
        XCTAssertEqual(decoded.architecture.policyTailPrecision, .doesNotApply)
        XCTAssertEqual(decoded.architecture, try Self.bf16Architecture().withComputeDataType(.float32, tail: nil),
                       "only the precision fields changed")
        XCTAssertTrue(decoded.hasVelocity)
        XCTAssertEqual(decoded.file.metadata.trainerSchedule, Self.schedule())
        XCTAssertEqual(decoded.file.metadata.trainerGradNormHistory, try Self.gradNormHistory())
        XCTAssertEqual(decoded.file.metadata.trainingStep, Self.clock, "training_step is the trainer clock")
        XCTAssertEqual(decoded.file.metadata.creator, ModelDerivation.creator)
        XCTAssertEqual(decoded.file.metadata.parentModelID, "20261112-1-TRST", "parent_model_id is the source, as for any derive")
        XCTAssertEqual(decoded.file.modelID, "20261112-2-DRVD")
        XCTAssertTrue(decoded.file.metadata.notes.contains("trainer state carried"), decoded.file.metadata.notes)
        // A trainer file decodes into exactly what an exact resume reads.
        XCTAssertNoThrow(try TrainerResumeSnapshot(checkpoint: decoded.file, fileName: "derived"))
    }

    func testTheDerivedRecordIsTheSourcesWithOneMoreDerivation() throws {
        let arch = try Self.bf16Architecture()
        let sourceRecord = try Self.trainingRecord()
        let source = try Self.trainerFile(arch, record: sourceRecord)
        let result = try derive(source, [SetComputeDataTypeDeriveOperation(value: .float32)], name: "elitecyc fp32")
        let record = try XCTUnwrap(SafetensorsModelIO.decode(result.data).file.lineageParent.lineage.record)

        XCTAssertEqual(record, result.lineage)
        // No new run: the run, segment, totals, rng, parameters,
        // configuration, build, device, invocation and parent are the
        // training save's.
        XCTAssertEqual(record.run, sourceRecord.run)
        XCTAssertEqual(record.parent, sourceRecord.parent)
        XCTAssertEqual(record.steps, sourceRecord.steps)
        XCTAssertEqual(record.fed, sourceRecord.fed)
        XCTAssertEqual(record.time, sourceRecord.time)
        XCTAssertEqual(record.parameters, sourceRecord.parameters)
        XCTAssertEqual(record.configuration, sourceRecord.configuration)
        XCTAssertEqual(record.runSeeds, sourceRecord.runSeeds)
        XCTAssertEqual(record.build, sourceRecord.build)
        XCTAssertEqual(record.invocation, sourceRecord.invocation)
        XCTAssertEqual(record.device, sourceRecord.device)
        XCTAssertEqual(record.rng, sourceRecord.rng)
        XCTAssertEqual(record.segments, sourceRecord.segments)
        XCTAssertEqual(record.ancestry, sourceRecord.ancestry)
        // What changes: one derivation appended, and the derive's naming.
        XCTAssertEqual(record.derivationHistory.count, sourceRecord.derivationHistory.count + 1)
        let entry = try XCTUnwrap(record.derivationHistory.last)
        XCTAssertEqual(entry, result.record)
        XCTAssertEqual(entry.parentModelID, "20261112-1-TRST")
        XCTAssertEqual(entry.operations.map(\.operation), ["set-compute-dtype"])
        XCTAssertEqual(entry.operations.flatMap(\.rewrittenTensors), [])
        let carried = try XCTUnwrap(entry.trainerState)
        XCTAssertEqual(carried.segmentID, sourceRecord.run.segmentID)
        XCTAssertEqual(carried.trainerCompletedSteps, Self.clock)
        XCTAssertEqual(carried.sourceArchitectureFormatVersion, ArchitectureFormat.currentVersion)
        XCTAssertEqual(try carried.decodedSourceArchitecture(), arch, "the captured architecture, as built")
        XCTAssertEqual(record.modelNaming.value?.name, "elitecyc fp32")
        // The capture rule reads it back as the training save's architecture.
        let capture = try TrainerStateCapture(fileArchitecture: result.targetArchitecture, record: record)
        XCTAssertEqual(capture.architecture, arch)
        XCTAssertEqual(capture.derivations, [entry])
    }

    func testTailDeriveAndBothOperationsInOneCallCarryTheTrainerState() throws {
        let source = try Self.trainerFile(try Self.bf16Architecture())

        let tail = try derive(source, [SetPolicyTailPrecisionDeriveOperation(value: .mixedFinalProjection)])
        try assertSameTensorsBitExact(tail.data, source)
        XCTAssertEqual(tail.targetArchitecture.computeDataType, .bFloat16)
        XCTAssertEqual(tail.targetArchitecture.policyTailPrecision, .mixedFinalProjection)
        XCTAssertNotNil(tail.record.trainerState)
        let tailGaps = ResumeGap.precisionGaps(
            capture: try TrainerStateCapture(fileArchitecture: tail.targetArchitecture, record: tail.lineage),
            runningArchitecture: tail.targetArchitecture)
        XCTAssertEqual(tailGaps.gaps, [.policyTailPrecision])

        let both = try derive(source, [SetComputeDataTypeDeriveOperation(value: .float16),
                                       SetPolicyTailPrecisionDeriveOperation(value: .mixedFinalProjection)])
        try assertSameTensorsBitExact(both.data, source)
        XCTAssertEqual(both.targetArchitecture.computeDataType, .float16)
        XCTAssertEqual(both.targetArchitecture.policyTailPrecision, .mixedFinalProjection)
        XCTAssertEqual(both.record.operations.map(\.operation), ["set-compute-dtype", "set-policy-tail-precision"])
        let bothGaps = ResumeGap.precisionGaps(
            capture: try TrainerStateCapture(fileArchitecture: both.targetArchitecture, record: both.lineage),
            runningArchitecture: both.targetArchitecture)
        XCTAssertEqual(bothGaps.gaps, [.computeDataType, .policyTailPrecision])
    }

    /// A source written before LR keyframes comes out in the current
    /// schedule form only, converted exactly as a resume would convert it.
    func testALegacyScheduleComesOutInTheCurrentFormOnly() throws {
        let current = try Self.trainerFile(try Self.bf16Architecture(), record: try LineageRecord.forTests(
            trainerCompletedSteps: Self.clock, corpus: nil))
        let (tensors, currentMetadata) = try SafetensorsFile.decode(current)
        var metadata = currentMetadata
        metadata[TrainerScheduleState.MetadataKey.lrSchedule] = nil
        metadata[TrainerScheduleState.MetadataKey.legacyLRMomentumCycle] =
            #"{"lrEnabled":true,"lrPeriodSteps":4,"lrCount":0,"lrMin":0.0001,"lrMax":0.01,"lrInvert":false,"#
            + #""momentumEnabled":true,"momentumPeriodSteps":4,"momentumCount":0,"momentumMin":0.8,"momentumMax":0.95,"#
            + #""momentumInvert":true}"#
        metadata[TrainerScheduleState.MetadataKey.legacyLRMomentumCycleEnvelope] =
            #"{"lrPeakEnd":0.001,"lrTroughEnd":1.0e-05,"decayHorizonSteps":10,"momentumFollowsLRCycle":true,"#
            + #""momentumFollowStartLow":0.85,"momentumFollowStartHigh":0.95,"momentumFollowEndLow":0.9,"#
            + #""momentumFollowEndHigh":0.97}"#
        metadata[SafetensorsFile.contentHashKey] = nil
        let legacy = try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
        let converted = try XCTUnwrap(try TrainerScheduleState.decode(fromMetadata: metadata))

        let result = try derive(legacy, [SetComputeDataTypeDeriveOperation(value: .float32)])
        let (_, derivedMetadata) = try SafetensorsFile.decode(result.data)
        for key in TrainerScheduleState.MetadataKey.legacy {
            XCTAssertNil(derivedMetadata[key], "\(key) is a form nothing writes")
        }
        XCTAssertNotNil(derivedMetadata[TrainerScheduleState.MetadataKey.lrSchedule])
        XCTAssertEqual(try TrainerScheduleState.decode(fromMetadata: derivedMetadata), converted)
    }

    // MARK: - Chains (owner decision OD-C)

    func testChainedDerivesKeepPointingAtTheTrainingSave() throws {
        let arch = try Self.bf16Architecture()
        let source = try Self.trainerFile(arch)
        let first = try derive(source, [SetComputeDataTypeDeriveOperation(value: .float32)], modelID: "20261112-2-FP32")
        let second = try derive(first.data, [SetComputeDataTypeDeriveOperation(value: .float16)], modelID: "20261112-3-FP16")
        try assertSameTensorsBitExact(second.data, source)

        let entries = second.lineage.derivationHistory.filter { $0.trainerState != nil }
        XCTAssertEqual(entries.map(\.modelID), ["20261112-2-FP32", "20261112-3-FP16"])
        XCTAssertEqual(Set(entries.compactMap(\.trainerState?.segmentID)).count, 1, "both carried the same segment")
        XCTAssertEqual(second.lineage.run, first.lineage.run)

        let capture = try TrainerStateCapture(fileArchitecture: second.targetArchitecture, record: second.lineage)
        XCTAssertEqual(capture.architecture, arch, "the training save's architecture, not the intermediate fp32 one")
        XCTAssertEqual(capture.derivations.map(\.modelID), ["20261112-2-FP32", "20261112-3-FP16"])
        // fp16 with the fp32 tail (the dtype derive's choice from fp32) vs the
        // captured bf16 with the fp32 tail: only the dtype differs.
        XCTAssertEqual(second.targetArchitecture.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertEqual(ResumeGap.precisionGaps(capture: capture, runningArchitecture: second.targetArchitecture).gaps,
                       [.computeDataType])
    }

    func testAChainBackToTheCapturedPrecisionReportsNoPrecisionGap() throws {
        let arch = try Self.bf16Architecture()
        let source = try Self.trainerFile(arch)
        let fp32 = try derive(source, [SetComputeDataTypeDeriveOperation(value: .float32)], modelID: "20261112-2-FP32")
        let back = try derive(fp32.data, [SetComputeDataTypeDeriveOperation(value: .bFloat16)], modelID: "20261112-3-BACK")
        XCTAssertEqual(back.targetArchitecture, arch)
        let capture = try TrainerStateCapture(fileArchitecture: back.targetArchitecture, record: back.lineage)
        XCTAssertEqual(capture.derivations.count, 2, "the history still records both derives")
        let comparison = ResumeGap.precisionGaps(capture: capture, runningArchitecture: back.targetArchitecture)
        XCTAssertEqual(comparison.gaps, [])
        XCTAssertEqual(comparison.logLines, [])
    }

    // MARK: - Refusals

    /// Every kind that changes what the velocity was accumulated for (or
    /// rewrites a tensor) still refuses a trainer-state file, alone or mixed
    /// with a precision kind.
    func testEveryOtherKindStillRefusesATrainerStateFile() throws {
        let source = try Self.trainerFile(try Self.bf16Architecture())
        let refused: [[any DeriveOperation]] = [
            [SetSEBetaInitDeriveOperation(value: .zero, groupIndices: nil)],
            [SetActivationDeriveOperation(value: .silu)],
            [SetRezeroAlphaCapDeriveOperation(value: 3, groupIndices: nil)],
            [SetComputeDataTypeDeriveOperation(value: .float32), SetActivationDeriveOperation(value: .silu)],
            [SetActivationDeriveOperation(value: .silu), SetComputeDataTypeDeriveOperation(value: .float32)],
        ]
        for operations in refused {
            XCTAssertThrowsError(try derive(source, operations)) { error in
                guard case .sourceHasOptimizerState? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(operations.map { $0.kindName }): expected sourceHasOptimizerState, got \(error)")
                }
            }
        }
        XCTAssertEqual(ModelDerivation.operationKinds.filter(\.acceptsTrainerStateSource).map(\.name),
                       ["set-policy-tail-precision", "set-compute-dtype"])
    }

    /// Velocity without a trainer schedule (a trainer file from before exact
    /// resume) cannot be continued exactly, so it stays refused.
    func testVelocityWithoutAScheduleIsRefused() throws {
        let arch = try Self.bf16Architecture()
        let data = try SafetensorsModelIO.encode(
            modelID: "20261112-1-NOSC", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: Self.clock, parentModelID: "", notes: ""),
            weights: Self.trainerWeights(arch), architecture: arch, includesVelocity: true,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        XCTAssertThrowsError(try derive(data, [SetComputeDataTypeDeriveOperation(value: .float32)])) { error in
            guard case .sourceHasOptimizerState? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected sourceHasOptimizerState, got \(error)")
            }
        }
    }

    /// A trainer-state file without a lineage record has no run, corpus
    /// position or streams to carry. (Built from a current file's decode with
    /// the lineage reported unrecorded, as a pre-v7 file's is.)
    func testATrainerStateSourceWithoutLineageIsRefused() throws {
        let decoded = try SafetensorsModelIO.decode(try Self.trainerFile(try Self.bf16Architecture()), valueHead: .asStored)
        XCTAssertThrowsError(try ModelDerivation.CarriedSourceTrainerState(
            of: decoded, lineage: .unrecorded(formatVersion: 6), sourceName: "old.safetensors")) { error in
            XCTAssertEqual(error as? ModelDerivation.DeriveError, .trainerStateSourceWithoutLineage(source: "old.safetensors"))
        }
    }

    /// A source whose record total is not its trainer clock is refused by
    /// the check every trainer-file writer runs, not written.
    func testAnInconsistentSourceIsRefusedByTheSharedTrainerStateCheck() throws {
        let good = try Self.trainerFile(try Self.bf16Architecture())
        let (tensors, metadata) = try SafetensorsFile.decode(good)
        var edited = metadata
        let skewed = try Self.trainingRecord(cumTrainerStep: Self.clock + 5)
        for (key, value) in try skewed.metadataEntries() { edited[key] = value }
        edited[SafetensorsFile.contentHashKey] = nil
        let inconsistent = try SafetensorsFile.encode(tensors: tensors, metadata: edited)
        XCTAssertThrowsError(try derive(inconsistent, [SetComputeDataTypeDeriveOperation(value: .float32)])) { error in
            guard case .lineageStepDisagreesWithTrainerClock? = error as? SafetensorsModelIO.IOError else {
                return XCTFail("expected lineageStepDisagreesWithTrainerClock, got \(error)")
            }
        }
    }

    /// A graft still refuses a trainer-state file, whatever the derive
    /// operations now accept.
    func testAGraftStillRefusesATrainerStateFile() throws {
        let data = try Self.trainerFile(try Self.bf16Architecture())
        let (tensors, metadata) = try SafetensorsFile.decode(data)
        XCTAssertThrowsError(try ModelDerivation.requirePlainModelSource(
            tensors: tensors, metadata: metadata, sourceName: "trainer.safetensors")) { error in
            guard case .sourceHasOptimizerState? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected sourceHasOptimizerState, got \(error)")
            }
        }
    }

    /// A plain source's derive is unchanged: a new run, no `trainer_state`.
    func testAPlainSourceStillGetsANewDeriveRun() throws {
        let arch = try Self.bf16Architecture()
        let net = arch.weightTensorPlan().map { [Float](repeating: 0.25, count: $0.elementCount) }
        let plain = try SafetensorsModelIO.encode(
            modelID: "20261112-1-PLAN", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: ""),
            weights: net, architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let result = try derive(plain, [SetComputeDataTypeDeriveOperation(value: .float32)])
        XCTAssertNil(result.record.trainerState)
        XCTAssertEqual(result.lineage.run.start, .derive)
        XCTAssertEqual(result.lineage.run.segmentIndex, 0)
    }
}
