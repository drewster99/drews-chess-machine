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
/// The destination must be new, or — with `--arch-sweep-out-overwrite` — an
/// existing regular file, which is emptied first; anything else ends the
/// process before the first build (see `runAndExit`).
enum ArchSweepCLI {

    /// `--arch-sweep-out` refuses an existing file unless this flag is also
    /// given; the parser in `DrewsChessMachineApp` passes its presence as
    /// `replaceExistingOut`.
    static let replaceExistingOutFlag = "--arch-sweep-out-overwrite"

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

    /// Run the sweep and exit. `outPath` is opened before the first trainer
    /// is built: it must not exist, or — with `replaceExistingOut` — must be a
    /// regular file, which is emptied. A directory, symbolic link or other
    /// non-regular item there is always refused. Failing to open it, or later
    /// to encode or write a line to it, ends the process with a non-zero exit,
    /// so a sweep never runs on with its JSONL output silently going nowhere.
    static func runAndExit(blocks: [Int], steps: Int, batch: Int, outPath: String, replaceExistingOut: Bool) -> Never {
        let expandedOut = (outPath as NSString).expandingTildeInPath
        SessionLogger.shared.start()
        SessionLogger.shared.log(
            "[ARCH-SWEEP-CLI] launched build=\(BuildInfo.buildNumber) blocks=\(blocks) steps=\(steps) batch=\(batch) out=\(expandedOut) replaceExistingOut=\(replaceExistingOut)"
        )

        let handle: FileHandle
        do {
            handle = try FileSafety.openForWriting(
                at: URL(fileURLWithPath: expandedOut),
                existingRegularFile: replaceExistingOut ? .truncate : .refuse
            )
        } catch FileSafetyError.alreadyExists(path: let path, kind: .regularFile) {
            FileHandle.standardError.write(Data(
                "error: --arch-sweep-out \(path) already exists; pass \(replaceExistingOutFlag) to replace it\n".utf8
            ))
            SessionLogger.shared.shutdown()
            Darwin.exit(55)
        } catch {
            FileHandle.standardError.write(Data(
                "error: --arch-sweep-out: \(error.localizedDescription)\n".utf8
            ))
            SessionLogger.shared.shutdown()
            Darwin.exit(55)
        }

        func emit(_ obj: [String: Any]) {
            print(obj.map { "\($0)=\($1)" }.sorted().joined(separator: " "))
            // A line that cannot be encoded or written ends the sweep: going
            // on would leave the JSONL file silently missing records.
            do {
                try ProbeModelCLI.appendLine(try ProbeModelCLI.encodeLine(obj), to: handle)
            } catch {
                FileHandle.standardError.write(Data(
                    "error: write to --arch-sweep-out failed: \(error.localizedDescription); the file may end in a partial line\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(56)
            }
        }

        SessionLogger.shared.log(ChessNetwork.PolicyTailPrecision.processLogLine)
        emit([
            "event": "sweep_start", "blocks": blocks, "steps": steps, "batch": batch,
            "policy_tail_precision": ChessNetwork.PolicyTailPrecision.process.rawValue,
        ])

        for n in blocks {
            let arch = benchArch(blocks: n)
            emit(["event": "build_begin", "blocks": n, "params": arch.parameterCount])
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                let trainer = try ChessTrainer(
                    dropoutStream: RunMasterSeed.systemDrawn(context: "arch-sweep").generator(.dropout),
                    arch: arch
                )
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
