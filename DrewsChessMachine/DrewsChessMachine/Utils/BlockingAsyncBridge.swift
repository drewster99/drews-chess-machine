import Foundation

/// Runs `work` in a detached `.userInitiated` task and blocks the calling
/// thread until it ends, returning its value or rethrowing its error. For the
/// command-line paths that drive async APIs from a synchronous main-thread
/// flow (`runAndExit`s, the UCI `readLine` loop, `--derive-model`).
///
/// - Never call it from inside a Swift-concurrency task (it traps): parking a
///   cooperative-pool thread can starve the pool `work` needs.
/// - `work` must not need what the blocked thread serves: a main-thread
///   caller blocks the main actor, so awaiting any `@MainActor` member inside
///   `work` deadlocks. Snapshot main-actor state before calling.
///
/// Two semaphore bridges stay outside it on purpose:
/// `ChessMPSNetwork.calibrateBNRunningStats` and `LargeStackBuild`, which are
/// reached from async contexts (a fresh network built inside a task), where
/// this function's precondition would trap.
///
/// The box is written once before the signal and read once after the wait;
/// that semaphore edge is why a non-`Sendable` `T` may leave the task.
func runBlocking<T>(
    file: StaticString = #fileID,
    line: UInt = #line,
    _ work: @Sendable @escaping () async throws -> T
) throws -> T {
    precondition(withUnsafeCurrentTask { $0 == nil },
                 "runBlocking called from inside a Swift-concurrency task", file: file, line: line)
    let box = BlockingResultBox<T>()
    let done = DispatchSemaphore(value: 0)
    Task.detached(priority: .userInitiated) {
        do {
            box.result = .success(try await work())
        } catch {
            box.result = .failure(error)
        }
        done.signal()
    }
    done.wait()
    guard let result = box.result else {
        preconditionFailure("runBlocking: the task ended without a result", file: file, line: line)
    }
    return try result.get()
}

/// `runBlocking`'s hand-off; see its doc for why it may be unchecked.
private final class BlockingResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}
