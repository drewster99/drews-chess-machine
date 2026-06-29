import XCTest
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Characterizes whether an `MPSGraphExecutable` compiled from a graph that
/// contains stateful `graph.variable` ops observes weight mutations made
/// **externally** — via `graph.run(targetOperations: [assign])` on the original
/// graph. That external-assign path is exactly the mechanism
/// `ChessNetwork.loadWeights` uses (it feeds `weightLoadPlaceholders` and runs
/// `weightLoadAssignOps`, which are `g.assign(variable, placeholder)` ops), and
/// the same mechanism arena promotion and checkpoint loading go through.
///
/// This is the load-bearing question for "compile the training step to an
/// `MPSGraphExecutable`" (`GPU_UTILIZATION_PLAN.md`, Phase 2). If the executable
/// keeps its OWN copy of the variable storage (captured at compile time), then
/// promotion / checkpoint load would silently NOT reach the training
/// executable's weights — it would keep training a forked snapshot, and saved
/// checkpoints would read the wrong copy. If the storage is shared with the
/// graph, the executable is much closer to a drop-in.
///
/// The test asserts the sanity invariants unconditionally (initial executable
/// read; the graph-side assign actually took effect on the graph's own state).
/// It then **logs** the visibility outcome rather than asserting it, because the
/// answer is unknown until this runs on-device — promote the observed result to
/// an `XCTAssert` once we've seen it, to lock the behavior as a guard.
///
/// NOTE: this exercises Metal/MPSGraph, so run it only when no training session
/// is live (per the project's "don't run the app/tests during training" rule).
final class MPSGraphExecutableVariableSemanticsTests: XCTestCase {

    func testExecutableObservesExternalGraphAssignToVariable() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let n = 3
        let shape: [NSNumber] = [NSNumber(value: n)]

        func f32(_ xs: [Float]) -> Data { xs.withUnsafeBytes { Data($0) } }
        func readBack(_ td: MPSGraphTensorData) -> [Float] {
            var out = [Float](repeating: .nan, count: n)
            out.withUnsafeMutableBytes { buf in
                if let base = buf.baseAddress { td.mpsndarray().readBytes(base, strideBytes: nil) }
            }
            return out
        }

        let graph = MPSGraph()
        let v = graph.variable(with: f32([1, 1, 1]), shape: shape, dataType: .float32, name: "v")
        let zero = graph.constant(0.0, shape: shape, dataType: .float32)
        // Identity read of the variable (avoids targeting a variable op directly).
        let readV = graph.addition(v, zero, name: "readV")
        let ph = graph.placeholder(shape: shape, dataType: .float32, name: "ph")
        let assign = graph.assign(v, tensor: ph, name: "assign")

        // Compile an executable that reads the variable. `readV` depends only on
        // the variable, so no feeds are required.
        let executable = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [:],
            targetTensors: [readV],
            targetOperations: nil,
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )

        // 1. Executable read before any external mutation: the init value.
        let before = executable.run(with: queue, inputs: [], results: nil, executionDescriptor: nil)
        let beforeVals = readBack(try XCTUnwrap(before.first))
        XCTAssertEqual(beforeVals, [1, 1, 1], "executable initial variable read")

        // 2. Mutate `v` externally via graph.run + assign — the loadWeights path.
        //    MPSGraph does NOT order a `targetOperations` assign before a
        //    `targetTensors` read of the same variable WITHIN one run (no
        //    read-after-write dependency), so reading `readV` in the same run as
        //    the assign would observe the *pre-assign* value. This mirrors the
        //    real app: `loadWeights` runs the assigns in their own run and reads
        //    happen in later runs. So run the assign alone here (its result is
        //    ignored), then confirm persistence with a separate read run below.
        let phND = MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape))
        var nine: [Float] = [9, 9, 9]
        nine.withUnsafeMutableBytes { if let b = $0.baseAddress { phND.writeBytes(b, strideBytes: nil) } }
        _ = graph.run(
            with: queue,
            feeds: [ph: MPSGraphTensorData(phND)],
            targetTensors: [readV],          // ≥1 target required; value ignored
            targetOperations: [assign]
        )

        // Sanity: a SUBSEQUENT read run must reflect the assigned value, proving
        // the assign persisted into the graph's own variable storage.
        let confirm = graph.run(
            with: queue,
            feeds: [:],
            targetTensors: [readV],
            targetOperations: nil
        )
        XCTAssertEqual(readBack(try XCTUnwrap(confirm[readV])), [9, 9, 9],
                       "a read run after the assign should reflect v == 9 on the graph")

        // 3. Re-run the executable. Does it observe the external assign?
        let after = executable.run(with: queue, inputs: [], results: nil, executionDescriptor: nil)
        let afterVals = readBack(try XCTUnwrap(after.first))
        let executableSeesExternalAssign = (afterVals == [9, 9, 9])
        print("[MPSGraphExecutable variable visibility] after external graph.run assign, "
            + "executable reads \(afterVals) — sees external assign: \(executableSeesExternalAssign)")

        // LOCKED 2026-06-02 (on-device, M5 Max): SHARED storage. A compiled
        // executable observes a weight assign made externally via graph.run.
        // So loadWeights / promotion / checkpoint loads (which assign via
        // graph.run) ARE visible to a compiled training executable. The reverse
        // direction — the executable's own assigns being visible to graph.run
        // readers — is what `testGraphRunObservesExecutableAssignToVariable`
        // covers; Phase 2 needs both.
        XCTAssertEqual(afterVals, [9, 9, 9],
            "shared storage: executable must observe the external graph.run assign to v")
    }

    /// Reverse direction of `testExecutableObservesExternalGraphAssignToVariable`:
    /// does an assign performed **by an executable** become visible to a later
    /// `graph.run` read of the same variable? Phase 2 depends on this — the
    /// training executable's SGD assigns must be seen by the `graph.run`-based
    /// readers (arena candidate snapshot, `evaluate`, checkpoint `makeWeightData`),
    /// or we'd snapshot/save stale weights. Also checks persistence across two
    /// executable runs.
    ///
    /// Run only with no training session live.
    func testGraphRunObservesExecutableAssignToVariable() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let n = 3
        let shape: [NSNumber] = [NSNumber(value: n)]

        func f32(_ xs: [Float]) -> Data { xs.withUnsafeBytes { Data($0) } }
        func readBack(_ td: MPSGraphTensorData) -> [Float] {
            var out = [Float](repeating: .nan, count: n)
            out.withUnsafeMutableBytes { buf in
                if let base = buf.baseAddress { td.mpsndarray().readBytes(base, strideBytes: nil) }
            }
            return out
        }
        func feedND(_ xs: [Float]) -> MPSGraphTensorData {
            let nd = MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape))
            var local = xs
            local.withUnsafeMutableBytes { if let b = $0.baseAddress { nd.writeBytes(b, strideBytes: nil) } }
            return MPSGraphTensorData(nd)
        }

        let graph = MPSGraph()
        let v = graph.variable(with: f32([1, 1, 1]), shape: shape, dataType: .float32, name: "v")
        let zero = graph.constant(0.0, shape: shape, dataType: .float32)
        let readV = graph.addition(v, zero, name: "readV")
        let ph = graph.placeholder(shape: shape, dataType: .float32, name: "ph")
        let assign = graph.assign(v, tensor: ph, name: "assign")

        // An executable that PERFORMS the assign (writes v from its `ph` input).
        let assignExecutable = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: [ph: MPSGraphShapedType(shape: shape, dataType: .float32)],
            targetTensors: [readV],
            targetOperations: [assign],
            compilationDescriptor: MPSGraphCompilationDescriptor()
        )

        // 1. Executable assigns v = 7.
        _ = assignExecutable.run(with: queue, inputs: [feedND([7, 7, 7])], results: nil, executionDescriptor: nil)
        // graph.run reads v in a fresh run — does it see the executable's write?
        let read1 = graph.run(with: queue, feeds: [:], targetTensors: [readV], targetOperations: nil)
        let read1Vals = readBack(try XCTUnwrap(read1[readV]))
        print("[MPSGraphExecutable reverse visibility] after executable assign v=7, graph.run reads \(read1Vals)")
        XCTAssertEqual(read1Vals, [7, 7, 7],
            "graph.run read should observe the executable's assign to v")

        // 2. Persistence: a second executable assign is also seen by graph.run.
        _ = assignExecutable.run(with: queue, inputs: [feedND([3, 3, 3])], results: nil, executionDescriptor: nil)
        let read2 = graph.run(with: queue, feeds: [:], targetTensors: [readV], targetOperations: nil)
        XCTAssertEqual(readBack(try XCTUnwrap(read2[readV])), [3, 3, 3],
            "graph.run read should observe the second executable assign to v")
    }
}
