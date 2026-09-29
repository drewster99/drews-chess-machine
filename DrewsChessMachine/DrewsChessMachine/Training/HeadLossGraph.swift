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
        /// `(1 − ε)·oneHot + ε·uniform(legal)`, renormalized.
        let smoothed: MPSGraphTensor
        /// `(1 − ε)·uniform(other legal) + ε·uniform(legal)`, renormalized.
        let complement: MPSGraphTensor
        /// `[batch, 1]`: 1 where the position has more than one legal move,
        /// else 0. The weight the complement cross-entropy must carry.
        let complementValid: MPSGraphTensor
    }

    /// Build the label-smoothed positive and complement policy targets.
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
    /// |legal| is clamped at 1 against the impossible zero-legal row
    /// (terminal positions never reach the buffer, but a divide-by-zero would
    /// NaN the whole batch).
    static func policyTargets(
        graph: MPSGraph,
        movePlayed: MPSGraphTensor,
        legalMask: MPSGraphTensor,
        epsilon: MPSGraphTensor,
        policySize: Int
    ) -> PolicyTargets {
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

        // uniform(legal) = legalMask / max(|legal|, 1), per position.
        let legalCount = graph.reductionSum(with: legalMask, axis: 1, name: "legal_count_per_pos")
        let legalCountSafe = graph.maximum(legalCount, one, name: "legal_count_safe")
        let uniformOverLegal = graph.division(legalMask, legalCountSafe, name: "uniform_over_legal")

        let oneMinusEpsilon = graph.subtraction(one, epsilon, name: "label_smoothing_one_minus_eps")
        let smoothingPart = graph.multiplication(uniformOverLegal, epsilon, name: "smoothed_target_uniform_part")
        let smoothed = renormalizedTarget(
            graph.addition(
                graph.multiplication(oneHot, oneMinusEpsilon, name: "smoothed_target_onehot_part"),
                smoothingPart,
                name: "policy_smoothed_target_raw"
            ),
            graph: graph,
            name: "policy_smoothed_target"
        )

        // uniform(other legal) = (legalMask − oneHot) / max(|legal| − 1, 1).
        let otherLegalMask = graph.subtraction(legalMask, oneHot, name: "other_legal_mask")
        let otherLegalCountSafe = graph.maximum(
            graph.subtraction(legalCountSafe, one, name: "legal_count_minus_one"),
            one,
            name: "other_legal_count_safe"
        )
        let uniformOverOtherLegal = graph.division(otherLegalMask, otherLegalCountSafe, name: "uniform_over_other_legal")
        let complement = renormalizedTarget(
            graph.addition(
                graph.multiplication(uniformOverOtherLegal, oneMinusEpsilon, name: "complement_target_main_part"),
                smoothingPart,
                name: "policy_complement_target_raw"
            ),
            graph: graph,
            name: "policy_complement_target"
        )
        let complementValid = graph.cast(
            graph.greaterThan(legalCount, one, name: "complement_target_valid_bool"),
            to: dataType,
            name: "complement_target_valid"
        )
        return PolicyTargets(
            oneHot: oneHot,
            smoothed: smoothed,
            complement: complement,
            complementValid: complementValid
        )
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
