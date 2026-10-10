import Darwin
import Foundation
import Metal
import os

/// Puts memory pressure, swap, GPU working set and thermal state into the
/// session log as `[MEM]` lines — on every change of memory pressure or
/// thermal state, and every `periodicIntervalSeconds` (GPU fault forensics
/// plan, A6).
///
/// Why: on 2026-10-09 three GPU resets hit the training runs while swap stood
/// at 25.3 of 26.0 GB, and nothing recorded memory at the time, so whether
/// memory played a part could not be judged. The `[MEM]` line also carries
/// the GPU fault monitor's own poll cost, so that monitor's price in a
/// long-lived process is visible.
final class MemoryStatusMonitor: @unchecked Sendable {
    static let shared = MemoryStatusMonitor()

    /// The periodic `[MEM]` line's interval.
    static let periodicIntervalSeconds = 600

    /// The memory-pressure levels the kernel reports.
    enum Pressure: String, Sendable {
        case normal, warning, critical
    }

    /// One reading, as the `[MEM]` line shows it. Pure data, so the line's
    /// format is tested without the kernel.
    struct Reading: Sendable, Equatable {
        /// Nil when `task_info` couldn't be read (the reason is logged).
        let footprintBytes: UInt64?
        /// Nil when there is no Metal device.
        let gpuAllocatedBytes: UInt64?
        let gpuRecommendedMaxBytes: UInt64?
        /// Nil when `vm.swapusage` couldn't be read (the reason is logged).
        let swapUsedBytes: UInt64?
        let swapTotalBytes: UInt64?
        let pressure: Pressure
        let thermal: ProcessInfo.ThermalState
        /// The GPU fault monitor's polls since the previous line.
        let faultMonitorPollsMs: [Double]

        /// `[MEM] <trigger> footprint=… gpu=…/… swap=…/… pressure=… thermal=… gpuFaultPolls=…`,
        /// sizes in base-2 GB.
        func line(trigger: String) -> String {
            func gb(_ bytes: UInt64?) -> String {
                bytes.map { String(format: "%.2fGB", Double($0) / 1_073_741_824) } ?? "n/a"
            }
            let polls: String
            if faultMonitorPollsMs.isEmpty {
                polls = "none"
            } else {
                let sorted = faultMonitorPollsMs.sorted()
                polls = String(format: "n=%d p50=%.0fms max=%.0fms",
                               sorted.count, sorted[sorted.count / 2], sorted[sorted.count - 1])
            }
            return "[MEM] \(trigger) footprint=\(gb(footprintBytes))"
                + " gpu=\(gb(gpuAllocatedBytes))/\(gb(gpuRecommendedMaxBytes))"
                + " swap=\(gb(swapUsedBytes))/\(gb(swapTotalBytes))"
                + " pressure=\(pressure.rawValue) thermal=\(Self.name(of: thermal))"
                + " gpuFaultPolls=(\(polls))"
        }

        static func name(of state: ProcessInfo.ThermalState) -> String {
            switch state {
            case .nominal: return "nominal"
            case .fair: return "fair"
            case .serious: return "serious"
            case .critical: return "critical"
            @unknown default: return "state\(state.rawValue)"
            }
        }
    }

    private let queue = DispatchQueue(label: "drewschessmachine.memory-status", qos: .utility)
    // Queue-confined.
    private var started = false
    private var pressure: Pressure = .normal
    private var pressureSource: DispatchSourceMemoryPressure?
    private var timer: DispatchSourceTimer?
    private var thermalObserver: NSObjectProtocol?
    private var device: MTLDevice?
    private var swapReadFailureLogged = false
    private var footprintReadFailureLogged = false

    private init() {}

    /// Starts the pressure source, the thermal observer and the periodic
    /// line, and writes the first line; later calls do nothing. Returns at
    /// once (the work runs on the monitor's queue).
    func start() {
        queue.async { [weak self] in
            self?.startOnQueue()
        }
    }

    // MARK: - Private (on `queue`)

    /// `start`'s work, on `queue`.
    private func startOnQueue() {
        guard !started else { return }
        started = true
        device = MTLCreateSystemDefaultDevice()
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: queue)
        source.setEventHandler { [weak self] in
            guard let self, let source = self.pressureSource else { return }
            let event = source.data
            if event.contains(.critical) {
                self.pressure = .critical
            } else if event.contains(.warning) {
                self.pressure = .warning
            } else {
                self.pressure = .normal
            }
            self.writeLine(trigger: "pressure-change")
        }
        source.resume()
        pressureSource = source
        let interval = DispatchTimeInterval.seconds(Self.periodicIntervalSeconds)
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + interval, repeating: interval, leeway: .seconds(5))
        timer.setEventHandler { [weak self] in
            self?.writeLine(trigger: "periodic")
        }
        timer.resume()
        self.timer = timer
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            self.queue.async { self.writeLine(trigger: "thermal-change") }
        }
        writeLine(trigger: "start")
    }

    private func writeLine(trigger: String) {
        let swap = readSwap()
        let reading = Reading(
            footprintBytes: readFootprint(),
            gpuAllocatedBytes: device.map { UInt64($0.currentAllocatedSize) },
            gpuRecommendedMaxBytes: device.map { $0.recommendedMaxWorkingSetSize },
            swapUsedBytes: swap?.used,
            swapTotalBytes: swap?.total,
            pressure: pressure,
            thermal: ProcessInfo.processInfo.thermalState,
            faultMonitorPollsMs: GPUFaultMonitor.shared.takePollDurations()
        )
        SessionLogger.shared.log(reading.line(trigger: trigger))
    }

    /// `phys_footprint`; nil (logged once) when `task_info` fails, so the
    /// line shows `n/a` rather than a footprint of zero.
    private func readFootprint() -> UInt64? {
        switch ChessTrainer.readPhysFootprintBytes() {
        case .success(let bytes):
            return bytes
        case .failure(let error):
            if !footprintReadFailureLogged {
                footprintReadFailureLogged = true
                SessionLogger.shared.log(
                    "[MEM] could not read task_info(TASK_VM_INFO) (kern_return \(error.kernReturn)); footprint shows n/a")
            }
            return nil
        }
    }

    /// `vm.swapusage`; nil (logged once) when the read fails.
    private func readSwap() -> (used: UInt64, total: UInt64)? {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else {
            if !swapReadFailureLogged {
                swapReadFailureLogged = true
                SessionLogger.shared.log("[MEM] could not read vm.swapusage (errno \(errno)); swap shows n/a")
            }
            return nil
        }
        return (usage.xsu_used, usage.xsu_total)
    }
}
