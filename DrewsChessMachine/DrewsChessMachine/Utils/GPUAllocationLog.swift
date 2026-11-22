import Foundation
import Metal

/// Device GPU allocation (`MTLDevice.currentAllocatedSize`, the reading `[MEM]`
/// reports) just before and just after one action.
///
/// Exists to attribute GPU memory to the graphs that take it. MPSGraph labels
/// none of its heaps or the buffers it places in them, so a memory census of a
/// running trainer (experiments/20261122-memory-footprint) sees three unnamed
/// ~4 GiB heaps and can only infer which executable owns which. Reading the
/// device total around each compile and each executable's first run names the
/// owner at the moment the memory appears: compiling allocates almost nothing;
/// the first run allocates the executable's heap, which then stays.
///
/// The reading is the device-wide live total, not a per-object count: any
/// other allocation in the process during the action is included, and memory
/// MPSGraph frees during it is netted out (the change can be negative).
struct GPUAllocationChange: Sendable, Equatable {
    let bytesBefore: UInt64
    let bytesAfter: UInt64

    /// `bytesAfter − bytesBefore`, signed.
    var delta: Int64 { signedByteDelta(from: bytesBefore, to: bytesAfter) }

    /// Run `action` and read the device's allocation on each side of it. The
    /// one place an allocation is measured around an action, shared by the
    /// executable caches' `[GPU-ALLOC]` lines and `--train-variant-benchmark`.
    static func measure<Result>(
        on device: MTLDevice, _ action: () throws -> Result
    ) rethrows -> (result: Result, change: GPUAllocationChange) {
        let before = UInt64(device.currentAllocatedSize)
        let result = try action()
        return (result, GPUAllocationChange(bytesBefore: before, bytesAfter: UInt64(device.currentAllocatedSize)))
    }
}

/// The `[GPU-ALLOC]` session-log line: one per compiled graph, written after
/// its first run, with what its compile and its first run added to the
/// device's allocation. Base-2 MB (1 MB = 1024² B).
///
/// `[GPU-ALLOC] <owner> <graph> <key> compile=+X MB firstRun=+Y MB total=Z MB`
/// — owner is the network's command-queue label (the name every GPU
/// submission of that network carries), graph the kind of executable, key what
/// the cache distinguishes it by, total the device allocation after the first
/// run.
enum GPUAllocationLog {
    static func line(
        owner: String, graph: String, key: String,
        compile: GPUAllocationChange, firstRun: GPUAllocationChange
    ) -> String {
        "[GPU-ALLOC] \(owner) \(graph) \(key) compile=\(signedMegabytes(compile.delta)) "
            + "firstRun=\(signedMegabytes(firstRun.delta)) total=\(megabytes(firstRun.bytesAfter))"
    }

    /// `+12.3 MB` / `-4.0 MB` / `+0.0 MB`.
    static func signedMegabytes(_ bytes: Int64) -> String {
        String(format: "%@%.1f MB", bytes < 0 ? "-" : "+", Double(bytes.magnitude) / 1_048_576)
    }

    static func megabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }
}

/// Compiles awaiting their first run, per cache key, so the run can be
/// measured and both halves logged on one `[GPU-ALLOC]` line.
///
/// Lives beside the executable cache it describes and follows that cache's
/// access discipline (each owner touches it only on its serial execution
/// queue). Costs a run one emptiness check once every compiled key has run.
struct GPUAllocationFirstRunLedger<Key: Hashable> {
    private var compilesAwaitingFirstRun: [Key: GPUAllocationChange] = [:]

    /// Record what compiling `key`'s graph changed. A recompile of a key
    /// whose earlier compile never ran replaces it (the older executable is
    /// gone with its cache entry).
    mutating func recordCompile(of key: Key, _ change: GPUAllocationChange) {
        compilesAwaitingFirstRun[key] = change
    }

    /// Start measuring `key`'s run if it is that graph's first: returns the
    /// pending measurement (holding the compile and the reading taken now),
    /// or nil when the key's first run was already logged.
    mutating func beginFirstRun(of key: Key, device: MTLDevice) -> GPUAllocationFirstRun? {
        guard !compilesAwaitingFirstRun.isEmpty,
              let compile = compilesAwaitingFirstRun.removeValue(forKey: key) else { return nil }
        return GPUAllocationFirstRun(compile: compile, device: device,
                                     bytesBeforeRun: UInt64(device.currentAllocatedSize))
    }

    /// Forget every pending compile (the cache was cleared with them).
    mutating func removeAll() {
        compilesAwaitingFirstRun.removeAll()
    }
}

/// One graph's first run being measured (`GPUAllocationFirstRunLedger`).
struct GPUAllocationFirstRun {
    let compile: GPUAllocationChange
    let device: MTLDevice
    let bytesBeforeRun: UInt64

    /// Read the allocation after the run and log the graph's `[GPU-ALLOC]`
    /// line. Call once the run's work has been encoded (or completed).
    func finish(owner: String, graph: String, key: String) {
        let run = GPUAllocationChange(bytesBefore: bytesBeforeRun, bytesAfter: UInt64(device.currentAllocatedSize))
        SessionLogger.shared.log(GPUAllocationLog.line(owner: owner, graph: graph, key: key, compile: compile, firstRun: run))
    }
}
