import Foundation

let RECORD_COUNT = 250_000
let RECORD_SIZE = 1160
let ROUNDS = 200
let TOTAL_FLOATS = RECORD_COUNT * RECORD_SIZE
let BYTES_PER_BUFFER = TOTAL_FLOATS * MemoryLayout<Float>.stride

func seconds(_ block: () -> Void) -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    block()
    let end = DispatchTime.now().uptimeNanoseconds
    return Double(end - start) / 1_000_000_000.0
}

func fmt(_ s: Double) -> String {
    String(format: "%10.6f s  (%9.3f ms)", s, s * 1000.0)
}

func gib(_ bytes: Int) -> String {
    String(format: "%.2f GiB", Double(bytes) / (1024.0 * 1024.0 * 1024.0))
}

print("Records:        \(RECORD_COUNT)")
print("Record size:    \(RECORD_SIZE) floats (\(RECORD_SIZE * 4) bytes)")
print("Per buffer:     \(gib(BYTES_PER_BUFFER))")
print("Rounds:         \(ROUNDS)")
print("")

let source = UnsafeMutablePointer<Float>.allocate(capacity: TOTAL_FLOATS)
let dest = UnsafeMutablePointer<Float>.allocate(capacity: TOTAL_FLOATS)
defer {
    source.deallocate()
    dest.deallocate()
}

let fillTime = seconds {
    var rng = SystemRandomNumberGenerator()
    var i = 0
    while i < TOTAL_FLOATS {
        let bits = rng.next()
        source[i] = Float(bitPattern: UInt32(truncatingIfNeeded: bits))
        dest[i] = Float(bitPattern: UInt32(truncatingIfNeeded: bits >> 32))
        i += 1
    }
}
print("Fill both buffers with random: \(fmt(fillTime))")

let indices = UnsafeMutablePointer<Int>.allocate(capacity: RECORD_COUNT)
defer { indices.deallocate() }

let bytesPerRecord = RECORD_SIZE * MemoryLayout<Float>.stride

var totalTime: Double = 0
var minTime: Double = .infinity
var maxTime: Double = 0
var sink: Float = 0

for round in 0..<ROUNDS {
    for i in 0..<RECORD_COUNT {
        indices[i] = Int.random(in: 0..<RECORD_COUNT)
    }

    let t = seconds {
        for destIdx in 0..<RECORD_COUNT {
            let srcIdx = indices[destIdx]
            let srcPtr = source.advanced(by: srcIdx * RECORD_SIZE)
            let dstPtr = dest.advanced(by: destIdx * RECORD_SIZE)
            dstPtr.update(from: srcPtr, count: RECORD_SIZE)
        }
    }

    sink += dest[Int.random(in: 0..<TOTAL_FLOATS)]
    totalTime += t
    if t < minTime { minTime = t }
    if t > maxTime { maxTime = t }

    if round == 0 || (round + 1) % 20 == 0 {
        print("  round \(String(format: "%3d", round + 1)): \(fmt(t))")
    }
}

let avg = totalTime / Double(ROUNDS)
let bytesPerRound = RECORD_COUNT * bytesPerRecord
let gbPerSec = Double(bytesPerRound) / avg / 1_000_000_000.0

print("")
print("Rounds:        \(ROUNDS)")
print("Total:         \(fmt(totalTime))")
print("Average:       \(fmt(avg))")
print("Min:           \(fmt(minTime))")
print("Max:           \(fmt(maxTime))")
print("Per round:     \(gib(bytesPerRound)) copied")
print("Throughput:    \(String(format: "%.2f GB/s", gbPerSec))")
print("(sink: \(sink))")
