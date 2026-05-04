import Darwin
import Foundation

/// Process-wide diagnostic samplers used by the periodic STATS line.
///
/// These exist to chase a long-running slowdown where the trainer's per-
/// step inter-call wall grows monotonically over hours while self-play
/// stays fast. The hypothesis under investigation is that GPU-side
/// allocations (compute pipelines, AGX buffers, IOAccelerator-tagged VM
/// regions) accumulate across arena boundaries and lengthen the cost of
/// every subsequent `commit` because the kernel has to walk a larger
/// residency set / VM region map. Both samplers below answer one part of
/// that question:
///
///  - `currentResidentBytes()` — process resident memory. Already exposed
///    via `ChessTrainer.currentPhysFootprintBytes()` for sweep code; the
///    duplicate-named wrapper here keeps this file self-contained for
///    callers that don't otherwise depend on `ChessTrainer`.
///
///  - `currentVMRegionCount()` — total number of distinct virtual-memory
///    regions in the process address space (what `vmmap` and Activity
///    Monitor would call "VM Regions"). On Apple Silicon every IOSurface,
///    every IOAccelerator-mapped GPU buffer, and every Metal heap shows
///    up here; the count is the strongest available proxy for the
///    "kernel walks N entries on every command-buffer commit" cost.
///    Implemented via repeated `mach_vm_region_recurse` calls, which is
///    the same syscall path `vmmap` uses. Cost is roughly O(regions) —
///    ~1-3 ms per call at typical app sizes (~2K regions). Calling once
///    per `[STATS]` emit (60 s cadence) is well below 1% overhead;
///    callers MUST NOT call it per training step.
enum DiagSampler {
    /// Resident memory in bytes (`phys_footprint`). 0 on failure.
    static func currentResidentBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    intPtr,
                    &count
                )
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// Iterate the process VM region map and return (totalRegions,
    /// ioAcceleratorRegions). The second value is the count of regions
    /// tagged `VM_MEMORY_IOACCELERATOR` (273) — the tag the kernel
    /// applies to AGX-mapped GPU memory. When MPSGraph allocates a new
    /// `MTLBuffer` mid-encoding ("Late MTLBuffer creation" in the Metal
    /// trace), one of these IOAccelerator entries appears. Returns
    /// (0, 0) on failure rather than throwing — the caller is on a
    /// telemetry path and a missed reading is recoverable.
    ///
    /// Cost: one `mach_vm_region_recurse` syscall per region. On a
    /// trained-up DrewsChessMachine session that's ~2000 regions, so
    /// ~1-3 ms per call. Safe at 60 s cadence; do NOT call per step.
    static func currentVMRegionCount() -> (total: UInt32, ioAccelerator: UInt32) {
        var address: vm_address_t = 0
        var total: UInt32 = 0
        var ioAccel: UInt32 = 0
        // VM_MEMORY_IOACCELERATOR is 273. The macro lives in
        // <mach/vm_statistics.h>; reproducing the literal here avoids
        // an import dance for one constant. The kernel's tagging for
        // GPU mappings on Apple Silicon uses this tag.
        let ioAcceleratorTag: UInt32 = 273
        // Cap the loop so a runaway address space doesn't spin
        // forever. ~1 M regions is wildly more than any healthy
        // process; if we ever hit it, something is catastrophically
        // wrong and a partial count is fine.
        let safetyCap: UInt32 = 1_000_000
        // VM_REGION_SUBMAP_INFO_COUNT_64 is defined as the int-count
        // of `vm_region_submap_info_data_64_t` in <mach/vm_region.h>,
        // but the macro doesn't bridge to Swift. Compute it directly.
        let submapInfoIntCount = mach_msg_type_number_t(
            MemoryLayout<vm_region_submap_info_data_64_t>.size / MemoryLayout<Int32>.size
        )
        while total < safetyCap {
            var size: mach_vm_size_t = 0
            var info = vm_region_submap_info_data_64_t()
            var infoCount = submapInfoIntCount
            var depth: natural_t = 0
            let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
                infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) { intPtr in
                    var addr64 = mach_vm_address_t(address)
                    let result = mach_vm_region_recurse(
                        mach_task_self_,
                        &addr64,
                        &size,
                        &depth,
                        intPtr,
                        &infoCount
                    )
                    address = vm_address_t(addr64)
                    return result
                }
            }
            if kr != KERN_SUCCESS { break }
            total += 1
            if info.user_tag == ioAcceleratorTag {
                ioAccel += 1
            }
            address = vm_address_t(mach_vm_address_t(address) + size)
        }
        return (total: total, ioAccelerator: ioAccel)
    }
}
