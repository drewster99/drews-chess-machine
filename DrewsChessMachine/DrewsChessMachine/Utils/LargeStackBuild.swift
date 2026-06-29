import Foundation

/// Stack size for the dedicated MPSGraph build/compile thread.
///
/// MPSGraph's reverse-mode autodiff (`gradientForPrimaryTensor`) and graph
/// `compile` both recurse depth-first over the op DAG — one native stack frame
/// per op in the chain — and do so *inline on the calling thread*. The trainer
/// and the inference network normally build on serial `DispatchQueue`s, whose
/// worker threads carry the platform default stack (~512 KB). A deep tower
/// (~100+ residual blocks) overflows that and the process dies with an
/// uncatchable SIGBUS at the stack guard page. 64 MB is ~128× that default,
/// lifting the depth ceiling into the thousands of blocks.
let graphBuildStackBytes = 64 << 20

/// Error from the large-stack build helper itself, distinct from any error the
/// wrapped work throws.
enum GraphBuildError: LocalizedError {
    /// The dedicated large-stack thread finished without leaving a result.
    /// Unreachable in practice (the semaphore guarantees the worker wrote the
    /// result before signalling); exists so the hand-off never force-unwraps.
    case buildThreadFailed

    var errorDescription: String? {
        switch self {
        case .buildThreadFailed:
            return "Graph-build thread returned no result"
        }
    }
}

/// Cross-thread carrier for ``withLargeBuildStack(_:)``. Holds the work closure
/// (which may capture non-`Sendable` MPSGraph objects) and its result. Marked
/// `@unchecked Sendable` because the helper blocks the calling thread on a
/// semaphore until the worker finishes: the closure runs on exactly one thread,
/// and the result is read only after that happens-before edge, so there is never
/// concurrent access to launder.
private final class GraphBuildTransfer<T>: @unchecked Sendable {
    let work: () throws -> T
    var result: Result<T, Error>?
    init(_ work: @escaping () throws -> T) { self.work = work }
}

/// Run `work` synchronously on a dedicated `Thread` with a large stack
/// (``graphBuildStackBytes``) and return its result, rethrowing any error.
///
/// The caller blocks until the worker completes, so this preserves the serial
/// discipline of whatever queue invoked it — nothing else runs on that queue
/// concurrently with the build. `work` must NOT call back onto the queue it was
/// invoked from: that slot is parked on the semaphore here and would deadlock.
/// (The graph build/compile paths don't.) See ``graphBuildStackBytes`` for why
/// this exists.
func withLargeBuildStack<T>(_ work: @escaping () throws -> T) throws -> T {
    let transfer = GraphBuildTransfer(work)
    let done = DispatchSemaphore(value: 0)
    let thread = Thread {
        // A manually-created Thread does not auto-drain an autorelease pool the
        // way GCD workers do, and the autodiff/compile churns out a mountain of
        // autoreleased tensors; bound the peak explicitly.
        autoreleasepool {
            transfer.result = Result(catching: transfer.work)
        }
        done.signal()
    }
    thread.name = "drewschess.graphbuild"
    thread.stackSize = graphBuildStackBytes
    thread.qualityOfService = .userInitiated
    thread.start()
    done.wait()
    guard let result = transfer.result else {
        throw GraphBuildError.buildThreadFailed
    }
    return try result.get()
}
