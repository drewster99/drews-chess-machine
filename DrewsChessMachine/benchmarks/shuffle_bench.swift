import Foundation

let N = 250_000
let ITERATIONS = 500

func seconds(_ block: () -> Void) -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    block()
    let end = DispatchTime.now().uptimeNanoseconds
    return Double(end - start) / 1_000_000_000.0
}

func fmt(_ s: Double) -> String {
    String(format: "%10.6f s  (%9.3f ms)", s, s * 1000.0)
}

var array = [Int]()
let buildTime = seconds {
    array = [Int]()
    array.reserveCapacity(N)
    for i in 0..<N {
        array.append(i)
    }
}
print("Build (\(N) ints):         \(fmt(buildTime))")

precondition(array.count == N)
precondition(array[0] == 0)
precondition(array[111] == 111)
precondition(array[N - 1] == N - 1)

let singleShuffledTime = seconds {
    _ = array.shuffled()
}
print("Single shuffled() copy:    \(fmt(singleShuffledTime))")

var shuffledTotal: Double = 0
var shuffledSink = 0
for _ in 0..<ITERATIONS {
    let t = seconds {
        let out = array.shuffled()
        shuffledSink &+= out[0]
    }
    shuffledTotal += t
}
let shuffledAvg = shuffledTotal / Double(ITERATIONS)
print("shuffled() x\(ITERATIONS) total:    \(fmt(shuffledTotal))")
print("shuffled() avg:            \(fmt(shuffledAvg))")

var inPlace = array
var inPlaceTotal: Double = 0
var inPlaceSink = 0
for _ in 0..<ITERATIONS {
    let t = seconds {
        inPlace.shuffle()
        inPlaceSink &+= inPlace[0]
    }
    inPlaceTotal += t
}
let inPlaceAvg = inPlaceTotal / Double(ITERATIONS)
print("shuffle()  x\(ITERATIONS) total:    \(fmt(inPlaceTotal))")
print("shuffle()  avg:            \(fmt(inPlaceAvg))")

print("")
print("Ratio (shuffled / shuffle): \(String(format: "%.3fx", shuffledAvg / inPlaceAvg))")
print("(sinks: \(shuffledSink) \(inPlaceSink))")
