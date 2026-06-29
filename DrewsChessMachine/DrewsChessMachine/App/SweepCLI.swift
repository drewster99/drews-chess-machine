import Darwin
import Foundation

/// Headless batch-size sweep — invoked from `DrewsChessMachineApp.init`'s
/// pre-flight branch on `--sweep`, before any SwiftUI / `WindowGroup` setup.
/// Builds a fresh training network, runs the same `ChessTrainer.runSweep` the
/// GUI "Sweep Batch Sizes" button uses, prints a throughput table to stdout
/// (and `[SWEEP]` lines to the session log), then exits. No window, no session,
/// no auto-resume — same headless contract as `--uci`.
///
/// Quality is irrelevant here: the sweep trains on random data purely to
/// measure positions/sec at each batch size, so default hyperparameters and a
/// fresh random-weight net are exactly right (the sweep does not reset or load
/// any saved model).
enum SweepCLI {

    /// Build a fresh net, run the sweep, print results, and exit. Never returns.
    static func runAndExit(sizes: [Int]?, secondsPerSize: Double) -> Never {
        SessionLogger.shared.start()
        let dirty = BuildInfo.gitDirty ? "*" : ""
        SessionLogger.shared.log(
            "[SWEEP-CLI] launched build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirty) branch=\(BuildInfo.gitBranch)"
        )

        let sweepSizes = sizes ?? SessionController.sweepSizes

        print("Batch Size Sweep (training-mode BN, fresh random weights)")
        print(String(
            format: "  Target: %.1f s per size   sizes: %@",
            secondsPerSize, sweepSizes.map(String.init).joined(separator: ",")
        ))
        print("")
        print(" Batch   Warmup   Steps    Time   Avg/step    Avg GPU      Pos/sec    Loss")
        print(" -----   ------   -----    ----   --------    -------      -------    ----")

        let rows: [SweepRow]
        do {
            rows = try syncWait {
                // Fresh trainer ⇒ fresh `ChessNetwork` (built in the init), so
                // no resetNetwork / model load is needed.
                let trainer = try ChessTrainer()
                return try await trainer.runSweep(
                    sizes: sweepSizes,
                    targetSecondsPerSize: secondsPerSize,
                    onRowCompleted: { row in
                        // Print live (the sweep can run many seconds) and log
                        // durably as each size finishes.
                        print(Self.tableLine(row))
                        SessionLogger.shared.log(Self.logLine(row))
                    }
                )
            }
        } catch {
            FileHandle.standardError.write(Data("sweep: failed: \(error)\n".utf8))
            SessionLogger.shared.log("[SWEEP-CLI] failed: \(error)")
            SessionLogger.shared.shutdown()
            Darwin.exit(31)
        }

        let completed = rows.compactMap { row -> SweepResult? in
            if case .completed(let r) = row { return r } else { return nil }
        }
        if let best = completed.max(by: { $0.positionsPerSec < $1.positionsPerSec }) {
            print("")
            print(String(
                format: "  Best: batch size %d at %d positions/sec",
                best.batchSize, Int(best.positionsPerSec.rounded())
            ))
            SessionLogger.shared.log(String(
                format: "[SWEEP-CLI] done rows=%d best=batch %d @ %d pos/s",
                rows.count, best.batchSize, Int(best.positionsPerSec.rounded())
            ))
        } else {
            print("\n  (no completed rows)")
        }
        SessionLogger.shared.shutdown()   // flush the [SWEEP] log tail before exit
        Darwin.exit(0)
    }

    /// Fixed-column stdout row (mirrors the GUI table, minus the peak column —
    /// peak RAM isn't tracked on the CLI path).
    private static func tableLine(_ row: SweepRow) -> String {
        switch row {
        case .completed(let r):
            let pos = Int(r.positionsPerSec.rounded())
                .formatted(.number.grouping(.automatic))
                .padding(toLength: 11, withPad: " ", startingAt: 0)
            return String(
                format: "%6d  %5.0fms %6d %6.1fs  %7.2fms  %7.2fms  %@ %+.3f",
                r.batchSize, r.warmupMs, r.steps, r.elapsedSec, r.avgStepMs, r.avgGpuMs, pos, r.lastLoss
            )
        case .skipped(let s):
            return String(
                format: "%6d  skipped — est RAM %.2f GB exceeds device cap",
                s.batchSize, Double(s.estimatedBytes) / 1_073_741_824.0
            )
        }
    }

    private static func logLine(_ row: SweepRow) -> String {
        switch row {
        case .completed(let r):
            return String(
                format: "[SWEEP] batch=%d steps=%d avgStep=%.2fms avgGpu=%.2fms posPerSec=%d loss=%+.4f",
                r.batchSize, r.steps, r.avgStepMs, r.avgGpuMs, Int(r.positionsPerSec.rounded()), r.lastLoss
            )
        case .skipped(let s):
            return String(
                format: "[SWEEP] batch=%d SKIPPED estRAM=%.2fGB",
                s.batchSize, Double(s.estimatedBytes) / 1_073_741_824.0
            )
        }
    }

    /// Bridge async → sync (mirrors `UCIEngine.syncWait`): run `work` on a
    /// detached task, block the pre-flight thread on a semaphore, rethrow.
    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = SweepSyncBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("SweepCLI.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}

/// Result/error holder for `SweepCLI.syncWait`, mutated from the detached task
/// and read after the semaphore — so the unchecked-Sendable conformance is safe.
private final class SweepSyncBox<T>: @unchecked Sendable {
    var success: T?
    var failure: Error?
}
