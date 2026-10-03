import Foundation
import IOKit

/// Facts about this Mac's CPU, GPU and memory, read once at launch for
/// display (the Lichess bot's `!cpu` / `!gpu` / `!ram` / `!about` replies).
///
/// Each fact is read independently and is nil when it can't be read; the
/// reason is kept in `readFailures` so a caller can log it. Nothing is
/// guessed: a consumer shows "unknown" for a missing fact.
struct HardwareInfo: Sendable, Equatable {
    /// One CPU performance level: its sysctl name and physical core count.
    struct CorePerformanceLevel: Sendable, Equatable {
        let name: String
        let physicalCores: Int
    }

    /// `hw.model`: the Mac's model identifier (e.g. `Mac16,8`).
    let hardwareModel: String?
    /// `kern.hv_vmm_present`: whether this macOS runs inside a virtual
    /// machine. Recorded with training runs because a VM's step times are
    /// not comparable to the host's.
    let isVirtualMachine: Bool?
    /// `machdep.cpu.brand_string`: the CPU's marketing name.
    let cpuBrand: String?
    /// `hw.physicalcpu`.
    let cpuPhysicalCores: Int?
    /// `hw.perflevel<i>`, highest performance first.
    let cpuPerformanceLevels: [CorePerformanceLevel]
    /// `hw.memsize`, in bytes.
    let memoryBytes: UInt64?
    /// The IORegistry `AGXAccelerator` service's `model`: the GPU's marketing name.
    let gpuModel: String?
    /// The IORegistry `AGXAccelerator` service's `gpu-core-count`.
    let gpuCoreCount: Int?
    /// Why each missing fact is missing.
    let readFailures: [String]

    /// This Mac's facts, read on first use and kept for the life of the app.
    static let current = HardwareInfo.read()

    /// Read everything now. Cheap (a handful of sysctls and an IORegistry
    /// lookup), but call it once and keep the value.
    static func read() -> HardwareInfo {
        var failures: [String] = []

        let hardwareModel = sysctlString("hw.model", failures: &failures)
        let isVirtualMachine = sysctlInteger("kern.hv_vmm_present", failures: &failures).map { $0 != 0 }
        let brand = sysctlString("machdep.cpu.brand_string", failures: &failures)
        let physicalCores = sysctlInteger("hw.physicalcpu", failures: &failures)
        var levels: [CorePerformanceLevel] = []
        if let levelCount = sysctlInteger("hw.nperflevels", failures: &failures) {
            for level in 0..<levelCount {
                guard let cores = sysctlInteger("hw.perflevel\(level).physicalcpu", failures: &failures),
                      let name = sysctlString("hw.perflevel\(level).name", failures: &failures) else {
                    continue
                }
                levels.append(CorePerformanceLevel(name: name, physicalCores: cores))
            }
        }
        let memory = sysctlInteger("hw.memsize", failures: &failures).map { UInt64($0) }

        var gpuModel: String?
        var gpuCores: Int?
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        if service == IO_OBJECT_NULL {
            failures.append("no AGXAccelerator service in the IORegistry")
        } else {
            defer { IOObjectRelease(service) }
            if let model = IORegistryEntryCreateCFProperty(service, "model" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? String {
                gpuModel = model
            } else {
                failures.append("AGXAccelerator has no string \"model\"")
            }
            if let cores = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? Int {
                gpuCores = cores
            } else {
                failures.append("AGXAccelerator has no integer \"gpu-core-count\"")
            }
        }

        return HardwareInfo(
            hardwareModel: hardwareModel,
            isVirtualMachine: isVirtualMachine,
            cpuBrand: brand,
            cpuPhysicalCores: physicalCores,
            cpuPerformanceLevels: levels,
            memoryBytes: memory,
            gpuModel: gpuModel,
            gpuCoreCount: gpuCores,
            readFailures: failures
        )
    }

    private static func sysctlString(_ name: String, failures: inout [String]) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0 else {
            failures.append("sysctl \(name): \(String(cString: strerror(errno)))")
            return nil
        }
        guard size > 0 else {
            failures.append("sysctl \(name): empty value")
            return nil
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            failures.append("sysctl \(name): \(String(cString: strerror(errno)))")
            return nil
        }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Integer sysctls are `Int32` or `Int64` wide depending on the name; the
    /// size query says which.
    private static func sysctlInteger(_ name: String, failures: inout [String]) -> Int? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0 else {
            failures.append("sysctl \(name): \(String(cString: strerror(errno)))")
            return nil
        }
        switch size {
        case MemoryLayout<Int32>.size:
            var value: Int32 = 0
            guard sysctlbyname(name, &value, &size, nil, 0) == 0 else {
                failures.append("sysctl \(name): \(String(cString: strerror(errno)))")
                return nil
            }
            return Int(value)
        case MemoryLayout<Int64>.size:
            var value: Int64 = 0
            guard sysctlbyname(name, &value, &size, nil, 0) == 0 else {
                failures.append("sysctl \(name): \(String(cString: strerror(errno)))")
                return nil
            }
            return Int(value)
        default:
            failures.append("sysctl \(name): unexpected size \(size)")
            return nil
        }
    }
}
