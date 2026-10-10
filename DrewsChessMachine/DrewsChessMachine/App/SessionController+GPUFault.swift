import Foundation

/// What a GUI run's results record about GPU faults beyond the ledger: the
/// crash dumps written and where training stopped for a fault.
struct GPUFaultRunRecord: Sendable, Equatable {
    var crashDumps: [String] = []
    var stoppedAtTrainerStep: Int?
}

/// Which in-memory weights a GPU fault may have damaged, kept until they are
/// replaced (GPU fault forensics plan, A5). It outlives Stop: a continue after
/// Stop would otherwise resume training on the suspect trainer and save it.
struct GPUFaultTaint: Sendable, Equatable {
    /// The first fault that tainted anything.
    let fault: GPUFaultLedger.Fault
    /// The trainer's weights, optimizer state or RNG state may be wrong.
    /// Cleared by a start that replaces the trainer (from a loaded session,
    /// or a fresh fork of the champion).
    var trainer: Bool
    /// The champion may hold wrong weights (the fault hit a promotion's copy
    /// into it). Cleared by building a network or loading a model / session.
    var champion: Bool
}

/// A stopped Play-and-Train start's view of the fault ledger, kept from Stop
/// until the next start commits or the trainer is dropped. The run's last
/// seconds can still produce a fault after Stop: the training step in flight
/// at the cancel finishes afterwards, and macOS's message for a fault just
/// before Stop reaches the ledger only through the monitor's closing poll.
/// A fault from later is not the stopped trainer's — Play Game, a probe, or
/// another process's GPU reset discarding this one's inference work — so it
/// never taints the trainer.
struct StoppedRunGPUFaultWatch: Sendable {
    let watch: GPUFaultWatch
    /// When Stop ended the run.
    let stoppedAt: Date

    /// How long after Stop a fault still counts as the stopped run's: the
    /// monitor's closing poll runs one poll interval after Stop.
    static let tailSeconds = TimeInterval(GPUFaultMonitor.pollIntervalSeconds)

    /// The first fault since the run began that happened no later than
    /// `tailSeconds` after Stop. In memory; no log read.
    var firstRunFault: GPUFaultLedger.Fault? {
        let tailEnd = stoppedAt.addingTimeInterval(Self.tailSeconds)
        return watch.faultsSinceStart.first { $0.time <= tailEnd }
    }
}

extension SessionController {
    /// The one place a GUI Play-and-Train run acts on a GPU fault (GPU fault
    /// forensics plan, A5), whoever saw it: the training worker (a failed
    /// training-step submission, or a fault in the ledger after a step), the
    /// arena before or after a promotion, a session save's barrier, a
    /// promotion copy, Promote Trainee Now.
    ///
    /// A fault means the trainer's weights, optimizer state or RNG state may
    /// be wrong in ways nothing can locate, so, in this order:
    /// 1. the taint is recorded (refuses a continue, a save and a promotion
    ///    from the suspect weights until they are replaced);
    /// 2. an interactive run suspends training as a divergence at once — the
    ///    existing suspension that skips arenas, Promote Trainee Now and the
    ///    periodic autosave — so nothing acts on the weights while the dump is
    ///    written; a `--train` run takes its termination claim at once, so no
    ///    other ending records a different reason;
    /// 3. the first time in this start, a crash dump is written (Part C);
    /// 4. a `--train` run writes its results (`termination_reason:
    ///    gpu_fault`) and ends through `AutoTrainTermination`, with no save.
    ///
    /// `batchTrainerStep` is the step whose batch is staged: the failing
    /// step for a failure the step reported (`completed + 1`), else the last
    /// completed step (the default). `championAffected` marks a fault in a
    /// promotion's copy into the champion.
    func handleGPUFault(_ fault: GPUFaultLedger.Fault, seenBy context: String,
                        batchTrainerStep: Int? = nil, championAffected: Bool = false) async {
        let trainerStep = trainer?.completedTrainSteps
        SessionLogger.shared.log(
            "[GPU-FAULT] \(context) at trainer step \(trainerStep.map(String.init) ?? "n/a"): \(fault.summary); "
            + "training stops, nothing more is saved from these weights")
        gpuFaultRunRecord.modify { record in
            if record.stoppedAtTrainerStep == nil { record.stoppedAtTrainerStep = trainerStep }
        }
        var taint = gpuFaultTaint ?? GPUFaultTaint(fault: fault, trainer: true, champion: false)
        taint.trainer = true
        taint.champion = taint.champion || championAffected
        gpuFaultTaint = taint

        var claimedStop: TrainingHealthAutoTrainStop?
        if let autoTrainStop = gpuFaultAutoTrainStop {
            guard autoTrainStop.termination.claim() else {
                SessionLogger.shared.log(
                    "[APP] --train: GPU fault stop left to the termination already writing the results")
                return
            }
            claimedStop = autoTrainStop
        } else {
            suspendTrainingOnDivergence(reason: "GPU fault (\(context)): \(fault.summary)")
        }

        if !gpuFaultDumpWritten, let trainer {
            gpuFaultDumpWritten = true
            if let folder = await CrashDumpWriter.dump(
                reason: .gpuFault, detail: "\(context): \(fault.summary)", pathKind: "gui", trainer: trainer,
                batchSize: crashDumpRunBatchSize, batchTrainerStep: batchTrainerStep ?? trainer.completedTrainSteps,
                runProvenance: crashDumpRunProvenance, faults: gpuFaultWatch?.faultsSinceStart ?? [fault]) {
                gpuFaultRunRecord.modify { $0.crashDumps.append(folder.path) }
            }
        }
        if let claimedStop {
            claimedStop.termination.writeResultsAndExit(
                reason: .gpuFault,
                trigger: "GPU fault (\(context)) at trainer step \(trainerStep.map(String.init) ?? "n/a")",
                elapsed: Date().timeIntervalSince(claimedStop.runStart))
        }
    }

    /// A non-finite loss or gradient stopped the GUI trainer. A GPU reset can
    /// show up first as a NaN, before macOS's message reaches the fault
    /// monitor (the 2026-10-09 21:37 reset did), so the barrier runs first: a
    /// fault makes it a GPU fault. Otherwise a crash dump (its batch is the
    /// failing step's), then the existing divergence suspension.
    func handleNonFiniteStep(_ error: Error) async {
        if let fault = await gpuFaultBarrier() {
            await handleGPUFault(fault, seenBy: "training step (non-finite: \(error.localizedDescription))",
                                 batchTrainerStep: trainer.map { $0.completedTrainSteps + 1 })
            return
        }
        suspendTrainingOnDivergence(reason: error.localizedDescription)
        if let trainer {
            if let folder = await CrashDumpWriter.dump(
                reason: .nonFinite, detail: error.localizedDescription, pathKind: "gui", trainer: trainer,
                batchSize: crashDumpRunBatchSize, batchTrainerStep: trainer.completedTrainSteps + 1,
                runProvenance: crashDumpRunProvenance, faults: gpuFaultWatch?.faultsSinceStart ?? []) {
                gpuFaultRunRecord.modify { $0.crashDumps.append(folder.path) }
            }
        }
    }

    /// A near miss in the GUI trainer (a pre-clip gradient norm ≥ 1,000× its
    /// reference): dump the batch and state; training goes on.
    func dumpNearMiss(preClipNorm: Float, decision: GradientCapDecision) async {
        guard let trainer else { return }
        let detail = "pre-clip gradient norm \(preClipNorm) vs reference "
            + (decision.referenceMedian.map { String($0) } ?? "none")
        if let folder = await CrashDumpWriter.dump(
            reason: .nearMiss, detail: detail, pathKind: "gui", trainer: trainer,
            batchSize: crashDumpRunBatchSize, batchTrainerStep: trainer.completedTrainSteps,
            runProvenance: crashDumpRunProvenance, faults: gpuFaultWatch?.faultsSinceStart ?? []) {
            gpuFaultRunRecord.modify { $0.crashDumps.append(folder.path) }
        }
    }

    /// The fault that makes saving or promoting the current weights unsafe:
    /// a tainted trainer (outlives Stop), or a fault since this start found
    /// by a fresh read of macOS's fault messages (≈2 s). Nil when safe.
    /// After Stop, the stopped run's faults are read first: one from its
    /// last seconds may not be recorded until the monitor's closing poll.
    func gpuFaultBarrier() async -> GPUFaultLedger.Fault? {
        if gpuFaultWatch == nil, let stopped = stoppedRunFaultWatch {
            await stopped.watch.monitor.checkNow()
            absorbStoppedRunGPUFault()
        }
        if let taint = gpuFaultTaint, taint.trainer || taint.champion {
            return taint.fault
        }
        guard let watch = gpuFaultWatch else { return nil }
        return await watch.barrier()
    }

    /// A fault recorded after Stop that belongs to the stopped run (from its
    /// last seconds, logged after the monitor's last poll of the run) taints
    /// the trainer, as it would have during the run. In memory; no log read.
    func absorbStoppedRunGPUFault() {
        guard let fault = stoppedRunFaultWatch?.firstRunFault, gpuFaultTaint?.trainer != true else { return }
        var taint = gpuFaultTaint ?? GPUFaultTaint(fault: fault, trainer: true, champion: false)
        taint.trainer = true
        gpuFaultTaint = taint
        SessionLogger.shared.log(
            "[GPU-FAULT] a fault from the stopped run, recorded after Stop, taints its trainer: \(fault.summary)")
    }

    /// Why a Play-and-Train start in `mode` must be refused because of a GPU
    /// fault's taint, or nil to go ahead. A start that keeps the trainer
    /// (continue, new session on the same trainer) needs a clean trainer; one
    /// that forks the trainer from the champion needs a clean champion; one
    /// from a loaded session replaces both.
    func gpuFaultStartRefusal(mode: TrainingStartMode) -> String? {
        guard let taint = gpuFaultTaint else { return nil }
        let keepsTrainer = mode == .continueAfterStop || mode == .newSessionKeepTrainer
        if keepsTrainer && taint.trainer {
            return "the trainer's weights may be wrong after a GPU fault (\(taint.fault.summary)); "
                + "load the last saved session, or start from the champion"
        }
        let forksChampion = mode == .newSessionResetTrainerFromChampion
            || (mode == .freshOrFromLoadedSession && pendingLoadedSession == nil)
        if forksChampion && taint.champion {
            return "the champion's weights may be wrong after a GPU fault during a promotion "
                + "(\(taint.fault.summary)); load the last saved session or model first"
        }
        return nil
    }

    /// A start that passed `gpuFaultStartRefusal` replaces the trainer: the
    /// trainer part of the taint is cleared (the champion part stays until
    /// the champion is replaced).
    func clearTrainerGPUFaultTaint() {
        guard var taint = gpuFaultTaint else { return }
        taint.trainer = false
        gpuFaultTaint = taint.champion ? taint : nil
        SessionLogger.shared.log("[GPU-FAULT] the trainer was replaced; its GPU-fault taint is cleared")
    }

    /// The champion was replaced (a network built, a model or session
    /// loaded): the champion part of the taint is cleared. (A loaded
    /// session's trainer replaces the in-memory one only when Play and Train
    /// starts from it, which clears the trainer part then.)
    func clearChampionGPUFaultTaint() {
        guard var taint = gpuFaultTaint, taint.champion else { return }
        taint.champion = false
        gpuFaultTaint = taint.trainer ? taint : nil
        SessionLogger.shared.log("[GPU-FAULT] the champion was replaced; its GPU-fault taint is cleared")
    }
}
