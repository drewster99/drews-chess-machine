//
//  GuiSaveHarness.swift
//  DrewsChessMachineTests
//
//  A `SessionController` set up as a running Play-and-Train session —
//  tiny champion and trainer, replay buffer, run seed and game serials,
//  stats box and lineage segment — with two fake workers that behave like
//  the real ones at their pause gates: each polls `isRequestedToPause` at
//  its iteration boundary, acknowledges with `markWaiting`, waits until it
//  is released and carries on. The fake self-play worker takes a game
//  serial, appends a short game to the buffer and records it on the stats
//  box every iteration; the fake trainer draws a minibatch, which advances
//  the buffer's sampler. When the trainer acknowledges a pause it records
//  what it saw, so a test can compare a save's record with the state at
//  that instant.
//
//  The workers run on their own threads (not the cooperative pool), as the
//  real workers' blocking work does. Sessions are written only to a
//  temporary folder.
//

import Foundation
import XCTest
@testable import DrewsChessMachine

@MainActor
final class GuiSaveHarness {

    /// What the fake trainer saw when it acknowledged a training pause.
    struct TrainerPauseObservation: Sendable {
        let selfPlayHeld: Bool
        let samplerState: DCMRandom
        let nextGameSerial: Int
        let emittedGames: Int
        let emittedPositions: Int
        let totalPositionsAdded: Int
    }

    static let architecture = ResumeEquivalenceTests.architecture
    nonisolated static let positionsPerGame = 4

    let controller = SessionController()
    let champion: ChessMPSNetwork
    let trainer: ChessTrainer
    let selfPlayGate = WorkerPauseGate()
    let trainingGate = WorkerPauseGate()
    let buffer: ReplayBuffer
    let serials = GameSerialCounter(firstSerial: 0)
    let box: ParallelWorkerStatsBox
    let sessionsDirectory: URL
    let trainerPauseObservations = SyncBox<[TrainerPauseObservation]>([])

    private let stop = SyncBox<Bool>(false)
    private let finishedWorkers = DispatchGroup()

    /// `resumedEmittedGames`: when set, the session is a resumed one — its
    /// stats box seeded with that many emitted games and its lineage
    /// segment an exact resume of a recorded parent, as a GUI resume
    /// starts them.
    init(resumedEmittedGames: Int? = nil) throws {
        let arch = Self.architecture
        sessionsDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GuiSaveHarness-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: sessionsDirectory, withIntermediateDirectories: false)

        champion = try ChessMPSNetwork(.randomWeights(initSeed: 21), arch: arch)
        champion.identifier = ModelID(value: "20261003-1-CHMP")
        trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 22),
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: arch, initialization: .seeded(initSeed: 22))
        trainer.identifier = ModelID(value: "20261003-2-TRNR")
        buffer = try ResumeEquivalenceTests.fixtureBuffer(sampler: DCMRandom(seed: 23))

        let started = Date()
        let tracker: LineageTracker
        if let resumedEmittedGames {
            box = ParallelWorkerStatsBox(
                sessionStart: started, totalGames: resumedEmittedGames, totalMoves: resumedEmittedGames * 4,
                totalGameWallMs: 0, whiteCheckmates: 0, blackCheckmates: 0, stalemates: resumedEmittedGames,
                fiftyMoveDraws: 0, threefoldRepetitionDraws: 0, insufficientMaterialDraws: 0,
                trainingSteps: 0, emittedGames: resumedEmittedGames,
                emittedPositions: resumedEmittedGames * Self.positionsPerGame)
            let parent = LineageTracker.ParentFile(
                modelID: "20261003-2-TRNR", contentSHA256: nil, trainerCompletedSteps: 0,
                lineage: .recorded(try LineageRecord.forTests(trainerCompletedSteps: 0, corpus: nil)),
                derivationHistory: [])
            tracker = try LineageTracker(
                start: .resume(parent: parent, gaps: [], legacyTotals: nil), pathKind: .gui,
                argv: ["DrewsChessMachine"], startedAt: started, segmentStartTrainerStep: trainer.completedTrainSteps)
        } else {
            box = ParallelWorkerStatsBox(sessionStart: started)
            tracker = try LineageTracker(
                start: .fresh(initialization: .forTests), pathKind: .gui, argv: ["DrewsChessMachine"],
                startedAt: started, segmentStartTrainerStep: trainer.completedTrainSteps)
        }

        controller.network = champion
        controller.trainer = trainer
        controller.realTraining = true
        controller.activeSelfPlayGate = selfPlayGate
        controller.activeTrainingGate = trainingGate
        controller.replayBuffer = buffer
        controller.selfPlayGameSerials = serials
        controller.runRandomSeed = RunRandomSeed.resolve(
            mode: .seeded, configuredSeed: 24, commandLineSeed: nil, drawSeed: { 0 })
        controller.runBehaviorFingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab")
        controller.parallelWorkerStatsBox = box
        controller.lineageTracker = tracker
        // The segment counts on this box from its counts now, as
        // `beginLineageSegment` sets it up at a Play-and-Train start.
        let counts = box.snapshot()
        controller.lineageFedCarry.baselineGames = counts.emittedGames
        controller.lineageFedCarry.baselinePositions = counts.emittedPositions
        controller.championOrigin = .built(initialization: .forTests)
        controller.trainingStats = TrainingRunStats()
    }

    /// Start both fake workers.
    func startWorkers() {
        let stop = stop
        let selfPlayGate = selfPlayGate
        let trainingGate = trainingGate
        let buffer = buffer
        let serials = serials
        let box = box
        let observations = trainerPauseObservations
        let floatsPerBoard = BoardEncoder.tensorLength(for: Self.architecture.inputEncoding)
        Self.startWorker(group: finishedWorkers, stop: stop, gate: selfPlayGate, onPause: {}, work: {
            _ = serials.next()
            Self.appendGame(to: buffer, floatsPerBoard: floatsPerBoard)
            box.recordEmittedGame(
                result: .stalemate,
                flushed: FlushedGameStats(positions: Self.positionsPerGame, phaseByPly: .zero, phaseByMaterial: .zero))
        })
        Self.startWorker(group: finishedWorkers, stop: stop, gate: trainingGate, onPause: {
            let counts = box.snapshot()
            observations.modify {
                $0.append(TrainerPauseObservation(
                    selfPlayHeld: selfPlayGate.isRequestedToPause,
                    samplerState: buffer.samplerState(),
                    nextGameSerial: serials.nextSerial,
                    emittedGames: counts.emittedGames,
                    emittedPositions: counts.emittedPositions,
                    totalPositionsAdded: buffer.stateSnapshot().totalPositionsAdded))
            }
        }, work: {
            var boards = [Float](repeating: 0, count: 2 * floatsPerBoard)
            var moves = [Int32](repeating: 0, count: 2)
            var zs = [Float](repeating: 0, count: 2)
            boards.withUnsafeMutableBufferPointer { b in
            moves.withUnsafeMutableBufferPointer { m in
            zs.withUnsafeMutableBufferPointer { z in
                guard let bb = b.baseAddress, let mb = m.baseAddress, let zb = z.baseAddress else {
                    preconditionFailure("minibatch arrays are non-empty, so every base address exists")
                }
                _ = buffer.sample(count: 2, intoBoards: bb, moves: mb, zs: zb)
            }}}
        })
    }

    /// Stop both fake workers and wait for them to exit.
    func stopWorkers() {
        stop.value = true
        finishedWorkers.wait()
    }

    /// Remove the temporary sessions folder.
    func removeSessionsDirectory() throws {
        try FileManager.default.removeItem(at: sessionsDirectory)
    }

    /// Run one `saveSessionInternal` and wait for its outcome.
    func save(trigger: SessionSaveTrigger, includeReplayBuffer: Bool) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            controller.saveSessionInternal(
                champion: champion, trainer: trainer, selfPlayGate: selfPlayGate, trainingGate: trainingGate,
                trigger: trigger, includeReplayBuffer: includeReplayBuffer, sessionsDirectory: sessionsDirectory,
                onComplete: { success in continuation.resume(returning: success) })
        }
    }

    /// The finished session folders in the temporary sessions folder.
    func savedSessions() throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: sessionsDirectory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "dcmsession" }
    }

    /// Wait until a finished session folder appears, up to `timeout`.
    func waitForSavedSession(timeout: TimeInterval) async throws -> URL? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let url = try savedSessions().first { return url }
            try await Task.sleep(for: .milliseconds(50))
        }
        return nil
    }

    private nonisolated static func startWorker(group: DispatchGroup, stop: SyncBox<Bool>, gate: WorkerPauseGate,
                                                onPause: @escaping @Sendable () -> Void,
                                                work: @escaping @Sendable () -> Void) {
        group.enter()
        let thread = Thread {
            defer { group.leave() }
            while !stop.value {
                if gate.isRequestedToPause {
                    onPause()
                    gate.markWaiting()
                    while gate.isRequestedToPause && !stop.value {
                        usleep(500)
                    }
                    gate.markRunning()
                    continue
                }
                work()
                usleep(200)
            }
        }
        // The main thread waits on these workers at teardown; matching its
        // priority avoids a priority inversion there.
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    private nonisolated static func appendGame(to buffer: ReplayBuffer, floatsPerBoard: Int) {
        let count = positionsPerGame
        let boards = [Float](repeating: 0, count: count * floatsPerBoard)
        let moves = [Int32](repeating: 0, count: count)
        let plies = (0..<count).map { UInt16($0) }
        let taus = [Float](repeating: 1, count: count)
        let hashes = (0..<count).map { UInt64($0 + 1) }
        let materials = [UInt8](repeating: 32, count: count)
        let outcomes = [Float](repeating: 0, count: count)
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { pl in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { mc in
        outcomes.withUnsafeBufferPointer { o in
            guard let bb = b.baseAddress, let mb = m.baseAddress, let pb = pl.baseAddress,
                  let tb = t.baseAddress, let hb = h.baseAddress, let mcb = mc.baseAddress,
                  let ob = o.baseAddress else {
                preconditionFailure("game arrays are non-empty, so every base address exists")
            }
            buffer.append(boards: bb, policyIndices: mb, plyIndices: pb, samplingTaus: tb,
                          stateHashes: hb, materialCounts: mcb, gameLength: UInt16(count),
                          workerId: 0, intraWorkerGameIndex: 0, outcomes: ob, count: count)
        }}}}}}}
    }
}
