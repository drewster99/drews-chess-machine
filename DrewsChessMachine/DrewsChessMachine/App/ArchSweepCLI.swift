import Darwin
import Foundation

/// Headless depth (block-count) sweep — invoked from `DrewsChessMachineApp.init`'s
/// pre-flight branch on `--arch-sweep`, before any SwiftUI / Metal GUI setup.
///
/// Investigation tool (not shipped UX) for "why does training appear to hang at
/// large block counts?". For each requested block count it builds a fresh
/// `ChessTrainer` in the regime that hung (basic30, 32ch, 3×3 blocks, SE+/4,
/// WDL 32→FC128, bf16 — depth is the only variable), then times:
///   - `build`: trainer construction = forward graph + autodiff (on the
///     large-stack thread).
///   - `step` ×N: `trainStep(batchSize:)` on random data. Step 1 folds in the
///     first `compile` + first `encode` (Metal pipeline-state compilation);
///     steps 2+ are steady state. (step1 ≫ steady) ⇒ first-encode compilation
///     wall; (steady grows ~linearly) ⇒ ordinary deeper-net compute; (steady
///     explodes) ⇒ a real per-step problem (e.g. CPU fallback).
///
/// Streams one JSON object per line to `--arch-sweep-out` (flushed each write),
/// so a long run's partial progress is readable mid-flight and survives a kill.
enum ArchSweepCLI {

    /// The regime that hung, parameterized only by depth.
    static func benchArch(blocks: Int) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: .basic30,
            channels: 32,
            numBlocks: blocks,
            stemConvKernelSize: 3,
            activationFunction: .relu,
            blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd,
            blockUseRezero: true,
            rezeroAlphaInit: 1.0 / Float(blocks).squareRoot(),
            blockConv1KernelSize: 3,
            blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias,
            blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv,
            policyPreConvChannels: 32,
            valueHeadStyle: .wdlSoftmax,
            valueHeadConvChannels: 32,
            valueHeadHiddenUnits: 128,
            computeDataType: .bFloat16
        )
    }

    static func runAndExit(blocks: [Int], steps: Int, batch: Int, outPath: String) -> Never {
        SessionLogger.shared.start()
        SessionLogger.shared.log(
            "[ARCH-SWEEP-CLI] launched build=\(BuildInfo.buildNumber) blocks=\(blocks) steps=\(steps) batch=\(batch) out=\(outPath)"
        )

        FileManager.default.createFile(atPath: outPath, contents: nil)
        let handle = FileHandle(forWritingAtPath: outPath)

        func emit(_ obj: [String: Any]) {
            print(obj.map { "\($0)=\($1)" }.sorted().joined(separator: " "))
            guard let handle,
                  let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
            try? handle.write(contentsOf: data)
            try? handle.write(contentsOf: Data("\n".utf8))
            try? handle.synchronize()
        }

        emit(["event": "sweep_start", "blocks": blocks, "steps": steps, "batch": batch])

        for n in blocks {
            let arch = benchArch(blocks: n)
            emit(["event": "build_begin", "blocks": n, "params": arch.parameterCount])
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                let trainer = try ChessTrainer(arch: arch)
                let buildMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                emit(["event": "build", "blocks": n, "params": arch.parameterCount, "buildMs": buildMs])

                for s in 1...steps {
                    let timing = try syncWait { try await trainer.trainStep(batchSize: batch) }
                    emit([
                        "event": "step", "blocks": n, "step": s,
                        "totalMs": timing.totalMs, "gpuRunMs": timing.gpuRunMs,
                        "dataPrepMs": timing.dataPrepMs, "readbackMs": timing.readbackMs,
                    ])
                }
                emit(["event": "arch_done", "blocks": n])
            } catch {
                emit(["event": "error", "blocks": n, "error": "\(error)"])
            }
        }
        emit(["event": "sweep_done"])
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    /// Bridge async → sync (mirrors `SweepCLI.syncWait`).
    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = ArchSweepSyncBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("ArchSweepCLI.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}

private final class ArchSweepSyncBox<T>: @unchecked Sendable {
    var success: T?
    var failure: Error?
}
