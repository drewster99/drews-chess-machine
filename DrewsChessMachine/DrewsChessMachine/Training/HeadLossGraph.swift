//
//  HeadLossGraph.swift
//  DrewsChessMachine
//
//  The pieces of the training loss that decide what the heads' shared
//  logit offset feels: the cross-entropy targets and the logit centering.
//  Split out of `ChessTrainer.buildTrainingOps` so their invariants (every
//  target sums to exactly 1; a centered loss gives the shared direction zero
//  gradient) can be tested on a small standalone graph.
//

import MetalPerformanceShadersGraph

/// Graph builders for the head losses' targets and logit centering, all in
/// fp32 (`dataType`, the heads' tail dtype).
///
/// Why these matter together: both heads' logits can grow a shared
/// per-position offset — a constant added to every logit of a position —
/// that softmax cannot see. A target that does not sum to exactly 1 pushes
/// that offset every step (the cross-entropy gradient's sum over a row is
/// `Σy − 1`), and nothing pulls it back. Building the targets in fp32 and
/// renormalizing them removes the push; centering the logits before the loss
/// removes the offset's gradient altogether, whatever the target.
enum HeadLossGraph {

    /// The loss path's dtype: the heads' fp32 tails.
    static let dataType: MPSDataType = ChessNetwork.headTailDataType

    /// Subtract each row's mean over `axis 1` from `logits`.
    ///
    /// The gradient reaching the uncentered logits is then `(I − 11ᵀ/N)g`,
    /// whose component along the all-ones direction is exactly zero, so the
    /// shared offset gets no gradient from anything computed on the centered
    /// tensor. The mean, not the max: `reductionMaximum` has no gradient rule
    /// in MPSGraph, and this tensor is on the loss path.
    ///
    /// Returns the centered logits and the per-row mean (`[batch, 1]`), the
    /// offset itself, for monitoring.
    static func centerLogits(
        _ logits: MPSGraphTensor,
        graph: MPSGraph,
        name: String
    ) -> (centered: MPSGraphTensor, meanPerRow: MPSGraphTensor) {
        let meanPerRow = graph.mean(of: logits, axes: [1], name: "\(name)_mean_per_pos")
        let centered = graph.subtraction(logits, meanPerRow, name: "\(name)_centered")
        return (centered: centered, meanPerRow: meanPerRow)
    }

    /// Renormalize each row of a target so it sums to exactly 1 (to fp32
    /// rounding): `y / Σy`. A target that does not sum to 1 pushes every
    /// logit of the row by the same amount — the direction softmax cannot
    /// see — and the rounded `1/|legal|`, `1/3` and mixing products each
    /// leave such a residue.
    ///
    /// The denominator is floored at a tiny positive value. The floor only
    /// ever binds on a row the loss gives zero weight (the complement target
    /// of a single-legal-move position, see `policyTargets`), where it turns
    /// an all-zero row into an all-zero target instead of NaN — and a NaN
    /// target times a zero weight would still be NaN.
    static func renormalizedTarget(_ raw: MPSGraphTensor, graph: MPSGraph, name: String) -> MPSGraphTensor {
        let floor = graph.constant(1e-12, dataType: dataType)
        let rowSum = graph.reductionSum(with: raw, axis: 1, name: "\(name)_row_sum")
        let rowSumSafe = graph.maximum(rowSum, floor, name: "\(name)_row_sum_safe")
        return graph.division(raw, rowSumSafe, name: name)
    }

    /// The policy loss's targets, `[batch, policySize]` each.
    struct PolicyTargets {
        /// One-hot at the played move. Also read by the played-move
        /// probability diagnostic.
        let oneHot: MPSGraphTensor
        /// The positive (non-negative-advantage) target, renormalized.
        /// Fixed total: `(1 − ε)·oneHot + ε·uniform(legal)`. Per move:
        /// `(1 − total)·oneHot + (total/(n − 1))·otherLegal`,
        /// `total = min(δ·(n − 1), cap)`.
        let smoothed: MPSGraphTensor
        /// The complement (negative-advantage) target, renormalized. Fixed
        /// total: `(1 − ε)·uniform(other legal) + ε·uniform(legal)`. Per
        /// move: `(1 − f)·uniform(other legal) + f·oneHot`, `f = min(δ, cap)`.
        let complement: MPSGraphTensor
        /// `[batch, 1]`: 1 where the position has more than one legal move,
        /// else 0. The weight the complement cross-entropy must carry.
        let complementValid: MPSGraphTensor
    }

    /// The fed scalars that choose and parameterize the policy targets'
    /// label smoothing, each a `[1]` fp32 tensor fed unrounded. Fed rather
    /// than baked into the graph so every one of them — the mode included —
    /// is live-tunable without a graph rebuild.
    struct PolicyLabelSmoothingInputs {
        /// 1 selects the per-move targets, 0 the fixed-total targets
        /// (`PolicyLabelSmoothingMode.graphSelectorValue`).
        let perMoveSelector: MPSGraphTensor
        /// Fixed-total ε.
        let epsilon: MPSGraphTensor
        /// Per-move mass δ.
        let perMove: MPSGraphTensor
        /// Cap on the per-move total.
        let perMoveCap: MPSGraphTensor
    }

    /// The pieces both smoothing forms are built from. Built once per graph
    /// and shared, so the two forms agree exactly on |legal|, on the other
    /// legal moves, and on which positions carry a complement weight.
    private struct PolicyTargetBasis {
        let oneHot: MPSGraphTensor
        let one: MPSGraphTensor
        /// `[batch, 1]`: `max(|legal|, 1)`.
        let legalCountSafe: MPSGraphTensor
        /// `[batch, 1]`: `max(|legal|, 1) − 1`, the number of other legal
        /// moves, ≥ 0.
        let otherLegalCount: MPSGraphTensor
        /// `[batch, 1]`: `max(|legal| − 1, 1)`, a safe divisor for the other
        /// legal moves.
        let otherLegalCountSafe: MPSGraphTensor
        /// `legalMask − oneHot`: 1 at every legal move except the played one.
        let otherLegalMask: MPSGraphTensor
        /// `otherLegalMask / otherLegalCountSafe`: uniform over the other
        /// legal moves, the main mass of both complement targets.
        let uniformOverOtherLegal: MPSGraphTensor
        let complementValid: MPSGraphTensor
    }

    /// |legal| is clamped at 1 against the impossible zero-legal row
    /// (terminal positions never reach the buffer, but a divide-by-zero would
    /// NaN the whole batch).
    private static func policyTargetBasis(
        graph: MPSGraph,
        movePlayed: MPSGraphTensor,
        legalMask: MPSGraphTensor,
        policySize: Int
    ) -> PolicyTargetBasis {
        let oneHot = graph.oneHot(
            withIndicesTensor: movePlayed,
            depth: policySize,
            axis: 1,
            dataType: dataType,
            onValue: 1.0,
            offValue: 0.0,
            name: "move_onehot"
        )
        let one = graph.constant(1.0, dataType: dataType)
        let legalCount = graph.reductionSum(with: legalMask, axis: 1, name: "legal_count_per_pos")
        let legalCountSafe = graph.maximum(legalCount, one, name: "legal_count_safe")
        let otherLegalCount = graph.subtraction(legalCountSafe, one, name: "legal_count_minus_one")
        let otherLegalCountSafe = graph.maximum(otherLegalCount, one, name: "other_legal_count_safe")
        let otherLegalMask = graph.subtraction(legalMask, oneHot, name: "other_legal_mask")
        // uniform(other legal) = (legalMask − oneHot) / max(|legal| − 1, 1).
        let uniformOverOtherLegal = graph.division(otherLegalMask, otherLegalCountSafe, name: "uniform_over_other_legal")
        let complementValid = graph.cast(
            graph.greaterThan(legalCount, one, name: "complement_target_valid_bool"),
            to: dataType,
            name: "complement_target_valid"
        )
        return PolicyTargetBasis(
            oneHot: oneHot,
            one: one,
            legalCountSafe: legalCountSafe,
            otherLegalCount: otherLegalCount,
            otherLegalCountSafe: otherLegalCountSafe,
            otherLegalMask: otherLegalMask,
            uniformOverOtherLegal: uniformOverOtherLegal,
            complementValid: complementValid
        )
    }

    /// Build the fixed-total label-smoothed positive and complement policy
    /// targets — the form every run used before the per-move form existed.
    ///
    /// - `movePlayed`: `[batch]` int32 policy indices.
    /// - `legalMask`: `[batch, policySize]` fp32, 1 at legal cells.
    /// - `epsilon`: `[1]` fp32 label-smoothing ε, fed unrounded.
    ///
    /// **Positive target.** `(1 − ε)·oneHot(played) + ε·uniform(legal)`. At
    /// ε = 0 it is the hard one-hot. Its fixed point
    /// `p(played) = 1 − ε + ε/|legal|` is reachable, unlike the hard
    /// one-hot's `p = 1`, which needs an infinite logit.
    ///
    /// **Complement target** (the negative-advantage branch): the `(1 − ε)`
    /// main mass spread over the OTHER legal moves, the ε share over all
    /// legal moves — "the played move was bad here; mass should sit on any
    /// other legal move". When |legal| = 1 there are no other legal moves:
    /// the raw target is `ε·oneHot(played)`, which sums to ε, not 1 — a
    /// malformed target. There is nothing to learn from a forced move's
    /// "badness", so such a position is flagged by `complementValid = 0` and
    /// must get zero complement weight, rather than a target.
    ///
    /// Training selects between this and the per-move form through
    /// `policyTargets(graph:movePlayed:legalMask:labelSmoothing:policySize:)`;
    /// both build the fixed-total form with the same ops, which is what keeps
    /// fixed-total mode's targets unchanged by the per-move form's arrival.
    static func policyTargets(
        graph: MPSGraph,
        movePlayed: MPSGraphTensor,
        legalMask: MPSGraphTensor,
        epsilon: MPSGraphTensor,
        policySize: Int
    ) -> PolicyTargets {
        let basis = policyTargetBasis(
            graph: graph, movePlayed: movePlayed, legalMask: legalMask, policySize: policySize)
        let fixedTotal = fixedTotalTargets(graph: graph, basis: basis, legalMask: legalMask, epsilon: epsilon)
        return PolicyTargets(
            oneHot: basis.oneHot,
            smoothed: fixedTotal.smoothed,
            complement: fixedTotal.complement,
            complementValid: basis.complementValid
        )
    }

    /// Build the positive and complement policy targets for whichever
    /// smoothing mode `labelSmoothing.perMoveSelector` picks this step.
    ///
    /// Both forms are built and `select` passes the chosen one through
    /// unchanged; a build-time branch would make the mode the one smoothing
    /// knob that needs a graph rebuild to change. Each form is renormalized
    /// on its own, so both sum to exactly 1 whichever is chosen.
    ///
    /// **Per-move positive target.** Every non-played legal move gets
    /// `per = min(δ, cap/(n − 1))` and the played move the rest,
    /// `1 − per·(n − 1)`: exactly δ per alternative while the total
    /// `δ·(n − 1)` is under the cap, and the cap shared equally above it.
    /// Writing the per-alternative mass as a `min` rather than dividing the
    /// capped total back by `n − 1` keeps the raw per-alternative mass
    /// exactly δ below the cap instead of δ rounded through a multiply and a
    /// divide; renormalization then moves it only by the row sum's fp32
    /// rounding. With one legal move
    /// there are no alternatives (`otherLegalMask` is all zero) and the
    /// played move gets exactly 1 — a one-hot, the right target for a forced
    /// move. The equilibrium gap played-vs-each-alternative is
    /// `ln((1 − δ(n − 1))/δ)`, nearly independent of n below the cap.
    ///
    /// **Per-move complement target.** The mirror of the positive one, with
    /// the roles swapped: the target set is now the other legal moves and the
    /// one legal move outside it is the played move, which gets the per-move
    /// floor `f = min(δ, cap)` (δ for one alternative, capped like any total);
    /// the other legal moves share `1 − f` equally. The fixed-total
    /// complement likewise keeps a floor (`ε/|legal|`) on the played move,
    /// and that floor is what makes the negative branch's equilibrium
    /// reachable: without it the complement CE would ask for
    /// `p(played) → 0`, an infinitely negative logit — the unbounded drive
    /// label smoothing exists to remove. (Reusing the positive target's
    /// smoothing part literally, as the fixed-total form does, would put the
    /// per-move mass only on the other legal moves, leave the played move at
    /// exactly 0, and make the complement plain `uniform(other legal)`
    /// whatever δ is.) A single-legal-move position's complement is a
    /// one-hot on the played move (or all-zero at δ = 0) and carries zero
    /// weight through `complementValid`, exactly as in fixed-total mode.
    ///
    /// Illegal cells are exactly 0 in every target: `oneHot`,
    /// `otherLegalMask` and `uniform(legal)` are all 0 there.
    static func policyTargets(
        graph: MPSGraph,
        movePlayed: MPSGraphTensor,
        legalMask: MPSGraphTensor,
        labelSmoothing: PolicyLabelSmoothingInputs,
        policySize: Int
    ) -> PolicyTargets {
        let basis = policyTargetBasis(
            graph: graph, movePlayed: movePlayed, legalMask: legalMask, policySize: policySize)
        let fixedTotal = fixedTotalTargets(
            graph: graph, basis: basis, legalMask: legalMask, epsilon: labelSmoothing.epsilon)
        let perMove = perMoveTargets(
            graph: graph, basis: basis, perMove: labelSmoothing.perMove, perMoveCap: labelSmoothing.perMoveCap)
        let usePerMove = graph.greaterThan(
            labelSmoothing.perMoveSelector,
            graph.constant(0.5, dataType: dataType),
            name: "policy_label_smoothing_use_per_move"
        )
        return PolicyTargets(
            oneHot: basis.oneHot,
            smoothed: graph.select(
                predicate: usePerMove,
                trueTensor: perMove.smoothed,
                falseTensor: fixedTotal.smoothed,
                name: "policy_smoothed_target_selected"
            ),
            complement: graph.select(
                predicate: usePerMove,
                trueTensor: perMove.complement,
                falseTensor: fixedTotal.complement,
                name: "policy_complement_target_selected"
            ),
            complementValid: basis.complementValid
        )
    }

    /// The fixed-total form's two targets; see
    /// `policyTargets(graph:movePlayed:legalMask:epsilon:policySize:)`.
    private static func fixedTotalTargets(
        graph: MPSGraph,
        basis: PolicyTargetBasis,
        legalMask: MPSGraphTensor,
        epsilon: MPSGraphTensor
    ) -> (smoothed: MPSGraphTensor, complement: MPSGraphTensor) {
        // uniform(legal) = legalMask / max(|legal|, 1), per position.
        let uniformOverLegal = graph.division(legalMask, basis.legalCountSafe, name: "uniform_over_legal")

        let oneMinusEpsilon = graph.subtraction(basis.one, epsilon, name: "label_smoothing_one_minus_eps")
        let smoothingPart = graph.multiplication(uniformOverLegal, epsilon, name: "smoothed_target_uniform_part")
        let smoothed = renormalizedTarget(
            graph.addition(
                graph.multiplication(basis.oneHot, oneMinusEpsilon, name: "smoothed_target_onehot_part"),
                smoothingPart,
                name: "policy_smoothed_target_raw"
            ),
            graph: graph,
            name: "policy_smoothed_target"
        )

        let complement = renormalizedTarget(
            graph.addition(
                graph.multiplication(basis.uniformOverOtherLegal, oneMinusEpsilon, name: "complement_target_main_part"),
                smoothingPart,
                name: "policy_complement_target_raw"
            ),
            graph: graph,
            name: "policy_complement_target"
        )
        return (smoothed: smoothed, complement: complement)
    }

    /// The per-move form's two targets; see
    /// `policyTargets(graph:movePlayed:legalMask:labelSmoothing:policySize:)`.
    private static func perMoveTargets(
        graph: MPSGraph,
        basis: PolicyTargetBasis,
        perMove: MPSGraphTensor,
        perMoveCap: MPSGraphTensor
    ) -> (smoothed: MPSGraphTensor, complement: MPSGraphTensor) {
        // per = min(δ, cap / max(n − 1, 1)), `[batch, 1]`.
        let perAlternative = graph.minimum(
            perMove,
            graph.division(perMoveCap, basis.otherLegalCountSafe, name: "per_move_cap_share"),
            name: "per_move_mass_per_alternative"
        )
        // total = per·(n − 1); 0 when there are no other legal moves.
        let total = graph.multiplication(perAlternative, basis.otherLegalCount, name: "per_move_total_mass")
        let smoothed = renormalizedTarget(
            graph.addition(
                graph.multiplication(
                    basis.oneHot,
                    graph.subtraction(basis.one, total, name: "per_move_played_mass"),
                    name: "per_move_target_onehot_part"
                ),
                graph.multiplication(basis.otherLegalMask, perAlternative, name: "per_move_target_alternatives_part"),
                name: "policy_per_move_target_raw"
            ),
            graph: graph,
            name: "policy_per_move_target"
        )

        // f = min(δ, cap): the played move's floor in the mirrored target.
        let playedFloor = graph.minimum(perMove, perMoveCap, name: "per_move_complement_played_floor")
        let complement = renormalizedTarget(
            graph.addition(
                graph.multiplication(
                    basis.uniformOverOtherLegal,
                    graph.subtraction(basis.one, playedFloor, name: "per_move_complement_main_mass"),
                    name: "per_move_complement_main_part"
                ),
                graph.multiplication(basis.oneHot, playedFloor, name: "per_move_complement_played_part"),
                name: "policy_per_move_complement_target_raw"
            ),
            graph: graph,
            name: "policy_per_move_complement_target"
        )
        return (smoothed: smoothed, complement: complement)
    }

    /// Build the W/D/L value target, `[batch, classes]`:
    /// `(1 − ε)·oneHot(1 − z) + ε·(1/classes)`, renormalized. At ε = 0 it is
    /// the hard one-hot.
    ///
    /// Slot index `1 − z` maps z ∈ {+1, 0, −1} to {0 win, 1 draw, 2 loss};
    /// exact in fp32, and the int32 cast truncates toward zero, which is the
    /// identity on those values. A `drawPenalty` rewrite of a draw's z to
    /// `−drawPenalty ∈ (0, 1)` truncates back to slot 1 (draw), so the value
    /// target is unchanged by it — the contempt effect lives in the policy
    /// gradient only; the full `drawPenalty = 1` lands on slot 2 (loss). The
    /// index is clamped to the valid range anyway: an out-of-range one-hot
    /// index would silently produce an all-zero, gradient-free target row.
    static func valueTarget(
        graph: MPSGraph,
        z: MPSGraphTensor,
        epsilon: MPSGraphTensor,
        classes: Int
    ) -> MPSGraphTensor {
        let one = graph.constant(1.0, dataType: dataType)
        let slotIndexFloat = graph.subtraction(one, z, name: "value_slot_index_float")
        let slotIndexClamped = graph.minimum(
            graph.maximum(slotIndexFloat, graph.constant(0.0, dataType: dataType), name: "value_slot_index_lo"),
            graph.constant(Double(classes - 1), dataType: dataType),
            name: "value_slot_index_clamped"
        )
        let slotIndex = graph.reshape(
            graph.cast(slotIndexClamped, to: .int32, name: "value_slot_index"),
            shape: [-1],
            name: "value_slot_index_flat"
        )
        let valueOneHot = graph.oneHot(
            withIndicesTensor: slotIndex,
            depth: classes,
            axis: 1,
            dataType: dataType,
            onValue: 1.0,
            offValue: 0.0,
            name: "value_onehot"
        )
        let oneMinusEpsilon = graph.subtraction(one, epsilon, name: "value_label_smoothing_one_minus_eps")
        let uniform = graph.constant(1.0 / Double(classes), shape: [1, NSNumber(value: classes)], dataType: dataType)
        return renormalizedTarget(
            graph.addition(
                graph.multiplication(valueOneHot, oneMinusEpsilon, name: "value_smoothed_target_onehot_part"),
                graph.multiplication(uniform, epsilon, name: "value_smoothed_target_uniform_part"),
                name: "value_smoothed_target_raw"
            ),
            graph: graph,
            name: "value_smoothed_target"
        )
    }
}
