import Metal
import MetalPerformanceShadersGraph
import XCTest
@testable import DrewsChessMachine

/// The analyzers' trainer cut (`ChessTrainer.exportAnalysisState`) and the
/// replacement windows that keep an analysis snapshot's identity honest.
final class AnalysisSnapshotCutTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// A small tower, so each network builds quickly.
    private static func architecture(compute: ComputeDataType) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: compute
        )
    }

    /// A trainer holding a seeded network's weights; its load leaves it
    /// awaiting an identity stamp.
    private static func loadedTrainer(_ arch: NetworkArchitecture) async throws -> ChessTrainer {
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: arch, initialization: .overwrittenByLoad)
        try await trainer.loadBaseWeightsResetVelocity(
            try await ChessNetwork(arch: arch, initialization: .seeded(initSeed: 5)).exportWeights())
        return trainer
    }

    private static func bits(_ tensors: [[Float]]?) -> [[UInt32]]? {
        tensors.map { $0.map { $0.map(\.bitPattern) } }
    }

    private static func sameCut(_ a: ChessTrainer.AnalysisState, _ b: ChessTrainer.AnalysisState) -> Bool {
        a.completedSteps == b.completedSteps && bits(a.weights) == bits(b.weights)
            && bits(a.masters) == bits(b.masters) && bits(a.velocity) == bits(b.velocity)
    }

    func testReducedPrecisionCutCarriesMastersVelocityAndStep() async throws {
        try requireMetal()
        let trainer = try await Self.loadedTrainer(Self.architecture(compute: .bFloat16))
        trainer.completedTrainSteps = 7
        let state = try await trainer.exportAnalysisState()
        let network = trainer.network
        XCTAssertEqual(state.names, (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name })
        XCTAssertEqual(state.trainableCount, network.trainableVariables.count)
        XCTAssertEqual(state.completedSteps, 7)
        let exported = try await network.exportWeights()
        XCTAssertEqual(Self.bits(state.weights), Self.bits(exported))
        let masters = try XCTUnwrap(state.masters, "a bf16 trainer keeps fp32 masters")
        let readMasters = try await trainer.readMasterValues()
        XCTAssertEqual(Self.bits(masters), Self.bits(readMasters))
        XCTAssertEqual(state.velocity.count, state.trainableCount)
    }

    func testFloat32CutHasNoMasters() async throws {
        try requireMetal()
        let trainer = try await Self.loadedTrainer(Self.architecture(compute: .float32))
        let state = try await trainer.exportAnalysisState()
        XCTAssertNil(state.masters)
        XCTAssertEqual(state.velocity.count, state.trainableCount)
    }

    func testCutRefusesATrainerAwaitingItsLoad() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1),
                                       arch: Self.architecture(compute: .bFloat16), initialization: .overwrittenByLoad)
        do {
            _ = try await trainer.exportAnalysisState()
            XCTFail("a trainer built for loaded weights must not be analyzed before its load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "exportWeights")
        }
    }

    /// 3c/3d: a cut taken while SGD steps run is wholly before or wholly
    /// after each step — never one step's weights with another's velocity.
    /// fp32, so no macOS-27 bf16 non-finite throw can interfere.
    func testACutTakenDuringTrainingIsOneSideOfAStep() async throws {
        try requireMetal()
        let trainer = try await Self.loadedTrainer(Self.architecture(compute: .float32))
        for _ in 0..<10 {
            let before = try await trainer.exportAnalysisState()
            async let step = trainer.trainStep(batchSize: 8)
            async let during = trainer.exportAnalysisState()
            _ = try await step
            let cut = try await during
            let after = try await trainer.exportAnalysisState()
            XCTAssertFalse(Self.sameCut(before, after), "the step must move the weights and velocity")
            XCTAssertTrue(Self.sameCut(cut, before) || Self.sameCut(cut, after),
                          "a cut must be the state before or after the step, not a mix")
        }
    }

    /// 3b: the sweep's reset opens the trainer's replacement window, and
    /// only an ID stamp closes it — training steps do not.
    func testResetLeavesTheTrainerAwaitingIdentityUntilStamped() async throws {
        try requireMetal()
        let trainer = try await Self.loadedTrainer(Self.architecture(compute: .float32))
        trainer.identifier = ModelID(value: "20261007-1-TEST")
        let stamped = trainer.weightIdentityState
        XCTAssertFalse(stamped.awaitingIdentity)
        try await trainer.resetNetwork(initialization: .seeded(initSeed: 9))
        XCTAssertTrue(trainer.weightIdentityState.awaitingIdentity)
        XCTAssertNotEqual(trainer.weightIdentityState, stamped)
        _ = try await trainer.trainStep(batchSize: 8)
        XCTAssertTrue(trainer.weightIdentityState.awaitingIdentity)
        trainer.identifier = ModelID(value: "20261007-2-TEST")
        XCTAssertFalse(trainer.weightIdentityState.awaitingIdentity)
    }

    /// 3f: a champion load that begins and records its origin inside one
    /// export changes the identity state, even though nothing is open after.
    @MainActor
    func testAChampionReplacementSpanningAnExportIsDetected() {
        var identity = SessionController.ChampionWeightIdentityState()
        let before = identity
        identity.beginReplacement()
        XCTAssertTrue(identity.awaitingOrigin)
        identity.recordOrigin()
        XCTAssertFalse(identity.awaitingOrigin)
        XCTAssertNotEqual(identity, before)
    }

    /// 3f wiring: an open champion replacement refuses the snapshot, and
    /// recording an origin (`championOrigin`'s didSet) closes it.
    @MainActor
    func testAnOpenChampionReplacementRefusesTheSnapshotUntilItsOriginIsRecorded() async throws {
        try requireMetal()
        let controller = SessionController()
        let champion = try ChessMPSNetwork(.randomWeights(initSeed: 21), arch: Self.architecture(compute: .float32))
        champion.identifier = ModelID(value: "20261007-1-CHMP")
        controller.network = champion
        controller.championOrigin = .built(initialization: .forTests)
        controller.noteChampionWeightsReplaced()
        do {
            _ = try await controller.analysisSnapshot(of: .champion, initReferences: AnalysisInitReferenceCache())
            XCTFail("a champion awaiting its origin must not be analyzed")
        } catch AnalysisSnapshotError.weightsBeingReplaced(let role) {
            XCTAssertEqual(role, .champion)
        }
        controller.championOrigin = .built(initialization: .forTests)
        let capture = try await controller.analysisSnapshot(of: .champion, initReferences: AnalysisInitReferenceCache())
        XCTAssertEqual(capture.snapshot.modelID, "20261007-1-CHMP")
        XCTAssertEqual(capture.snapshot.trainingStep, 0)
        guard case .inferenceNetwork = capture.optimizerState else {
            return XCTFail("a champion capture carries no optimizer state")
        }
    }

    /// 3a wiring: a trainer awaiting its identity is refused; once stamped,
    /// its capture carries the ID, the step and the same-cut optimizer state.
    @MainActor
    func testATrainerAwaitingItsIdentityIsRefusedAndAStampedOneCarriesItsCut() async throws {
        try requireMetal()
        let controller = SessionController()
        let trainer = try await Self.loadedTrainer(Self.architecture(compute: .float32))
        controller.trainer = trainer
        do {
            _ = try await controller.analysisSnapshot(of: .trainer, initReferences: AnalysisInitReferenceCache())
            XCTFail("a trainer awaiting its identity must not be analyzed")
        } catch AnalysisSnapshotError.weightsBeingReplaced(let role) {
            XCTAssertEqual(role, .trainer)
        }
        trainer.identifier = ModelID(value: "20261007-2-TRNR")
        trainer.completedTrainSteps = 3
        let capture = try await controller.analysisSnapshot(of: .trainer, initReferences: AnalysisInitReferenceCache())
        XCTAssertEqual(capture.snapshot.modelID, "20261007-2-TRNR")
        XCTAssertEqual(capture.snapshot.trainingStep, 3)
        guard case .trainer(let masters, let velocity) = capture.optimizerState else {
            return XCTFail("a trainer capture carries its optimizer state")
        }
        XCTAssertNil(masters)
        XCTAssertEqual(velocity.count, capture.snapshot.trainableCount)
    }
}
