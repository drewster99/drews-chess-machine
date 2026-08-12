import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// `NetworkArchitecture.weightTensorPlan()` is the sole authority for what each
/// weight tensor is CALLED and what shape it has, but nothing verifies that the
/// thing sitting at position `i` of the graph — or of a `.safetensors` file —
/// is actually what the plan says position `i` is.
///
/// Both the in-memory path (`exportWeights()` → `loadWeights()`) and the disk
/// path pair values to meaning purely by INDEX. `SafetensorsModelIO.encode`
/// attaches `names[i]` to `weights[i]`; `decode` walks the plan in order and
/// hands the result to `loadWeights`, which assigns positionally. The plan's own
/// comment says its order "mirrors `ChessNetwork.residualBlock`'s trainables
/// append order EXACTLY" — a contract documented in prose and checked nowhere.
///
/// These tests pin the two halves of that contract.
final class WeightPlanContractTests: XCTestCase {

    // MARK: - Disk side: file shapes must agree with the plan

    /// `decode` must reject a tensor whose stored shape disagrees with the
    /// plan, not merely one whose element COUNT disagrees.
    ///
    /// The two are not equivalent, and the gap is reachable: a `.linear` stored
    /// as `[in, out]` instead of `[out, in]` has an identical element count, so
    /// a count-only guard passes — and then `fromTorchLayout` runs
    /// `transpose2D(torchData, rows: outDim, cols: inDim)` with the dimensions
    /// swapped, silently scrambling the weights. An external writer, a
    /// hand-edited file, or a future layout change all land here.
    func testDecodeRejectsLinearTensorStoredWithTransposedShape() throws {
        let arch = NetworkArchitecture.current
        let plan = arch.weightTensorPlan()

        // A NON-SQUARE .linear: a square one would be shape-identical either
        // way round, so it could not distinguish the two layouts.
        guard let victim = plan.enumerated().first(where: { _, spec in
            spec.kind == .linear && spec.shape.count == 2 && spec.shape[0] != spec.shape[1]
        }) else {
            throw XCTSkip("current preset has no non-square linear tensor to transpose")
        }
        let victimName = victim.element.name

        let weights = plan.map { spec in
            (0..<spec.elementCount).map { Float($0 % 17) * 0.25 }
        }
        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: nil,
                                           parentModelID: "", notes: "plan contract test")
        let good = try SafetensorsModelIO.encode(
            modelID: "20260812-1-PLAN", createdAtUnix: 1_780_000_000,
            metadata: meta, weights: weights, architecture: arch, includesVelocity: false
        )
        // Sanity: the unmodified file round-trips, so a throw below is
        // attributable to the shape edit and nothing else.
        XCTAssertNoThrow(try SafetensorsModelIO.decode(good))

        // Re-emit with ONLY the victim's shape reversed — same name, same bytes,
        // same element count. Only the declared dimensions change.
        let (tensors, metadata) = try SafetensorsFile.decode(good)
        let mutated = tensors.map { t -> SafetensorsTensor in
            guard t.name == victimName else { return t }
            return SafetensorsTensor(name: t.name, shape: t.shape.reversed(), data: t.data)
        }
        let bad = try SafetensorsFile.encode(tensors: mutated, metadata: metadata)

        XCTAssertThrowsError(
            try SafetensorsModelIO.decode(bad),
            "decode accepted '\(victimName)' with its dimensions transposed; the element "
            + "count still matches, so a count-only guard cannot see this, and "
            + "fromTorchLayout will transpose with swapped dims and scramble the weights"
        )
    }

    // MARK: - Graph side: variable shapes must agree with the plan

    /// Every graph variable, in `exportWeights()` order, must match the plan
    /// entry at that position — in BOTH bn modes.
    ///
    /// This is the assumption the `.safetensors` naming rests on: `encode`
    /// labels `weights[i]` with `plan[i].name` and applies `plan[i].kind`'s
    /// layout transform. If the builder's append order ever drifts from the
    /// plan, every saved model is mislabeled at write time and every load
    /// mis-assigned — with no error anywhere.
    ///
    /// Compared SQUEEZED. The plan records logical shapes (a BN gamma is
    /// `[C]`, torch convention) while the builder declares the same tensor
    /// broadcast-ready (`[1, C, 1, 1]`) so it applies across NCHW without a
    /// reshape — 132 positions differ that way in the current preset, all with
    /// identical element counts. Raw equality would therefore fail on a
    /// perfectly correct network; element count alone would be too weak to see
    /// `[in, out]` vs `[out, in]`.
    ///
    /// Unlike the decode test above this is an invariant test, not a
    /// regression test: it holds today, which is exactly why it is worth
    /// pinning before a refactor breaks it silently.
    func testGraphVariableShapesMatchTheArchitecturePlan() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        for mode in [BNMode.training, .inference] {
            let arch = NetworkArchitecture.current
            // Construction itself now enforces this contract, so a divergence
            // surfaces here as a throw rather than as the comparisons below.
            let net = try ChessNetwork(arch: arch, bnMode: mode)
            let plan = arch.weightTensorPlan()
            let allVars = net.trainableVariables + net.bnRunningStatsVariables

            XCTAssertEqual(
                allVars.count, plan.count,
                "\(mode): graph has \(allVars.count) weight variables but the plan describes \(plan.count)"
            )
            for (i, spec) in plan.enumerated() where i < allVars.count {
                let raw = (allVars[i].shape ?? []).map(\.intValue)
                XCTAssertEqual(
                    WeightTensorSpec.squeeze(raw), spec.squeezedShape,
                    "\(mode): position \(i) — plan calls this '\(spec.name)' with shape \(spec.shape), "
                    + "but the graph variable '\(allVars[i].operation.name)' has shape \(raw)"
                )
            }
        }
    }
}
