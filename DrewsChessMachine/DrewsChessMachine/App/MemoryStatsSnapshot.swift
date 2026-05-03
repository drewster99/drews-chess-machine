import Foundation

/// One refresh of the memory stats line shown in the top busy row
/// during Play and Train. Sampled every ~10 s by the snapshot
/// timer (not every frame) so the displayed numbers stay stable.
/// Bytes are stored at the granularity the source APIs return so
/// the formatter can round consistently regardless of when it ran.
struct MemoryStatsSnapshot: Sendable {
    /// Process resident memory from `task_info(TASK_VM_INFO).phys_footprint`.
    let appFootprintBytes: UInt64
    /// `MTLDevice.currentAllocatedSize` — the live GPU working set.
    let gpuAllocatedBytes: UInt64
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the soft cap Metal
    /// asks us to stay under for this device.
    let gpuMaxTargetBytes: UInt64
    /// Total physical memory available to the process, from
    /// `ProcessInfo.processInfo.physicalMemory`. On Apple Silicon
    /// this is the unified-memory total (CPU and GPU draw from the
    /// same pool), so it doubles as "GPU total RAM" for the user-
    /// facing display.
    let gpuTotalBytes: UInt64
}
