import Foundation
import os

/// Per-layer optimizer dynamics, accumulated over every SGD step and logged
/// as one `[DYNAMICS]` line per step line (owner request 2026-11-08).
///
/// Why it exists: the step line's `gNorm=` is one global number, dominated by
/// whichever few large tensors are noisiest. It cannot say *where* a change
/// happened — whether a rise is every layer growing together, one layer
/// running away, or a shift from one set of layers to another. R-elitecyc's
/// gNorm rose 0.40 → 0.92 and only an offline read of 152 checkpoints'
/// momentum buffers could place it (`experiments/20261108-elitecyc-heldout`
/// §4). This makes the same reading live, with the gradient itself (which a
/// checkpoint does not hold) and the update actually applied.
///
/// What the training graph reports per trainable tensor, every step
/// (`ChessTrainer`'s optimizer loop, one fp32 vector of
/// `TrainingDynamicsSums.componentCount` sums per tensor):
///
/// - Σg² — the raw (pre-clip) gradient, the same gradient `gNorm` sums;
/// - Σg·v_old — gradient against the momentum buffer the step starts from;
/// - Σv_old², Σv_new² — the momentum buffer before and after the step;
/// - Σ(Δw)² — the update actually subtracted (lr·v_new plus decoupled decay);
/// - Σw² — the weights the update applies to, before it.
///
/// Per layer (a tensor's name without its trailing `.weight` / `.bias`, so a
/// BN's γ and β or an FC's weight and bias are one layer) each step yields:
///
/// - `g` = ‖g‖ and `v` = ‖v_new‖;
/// - `dww` = ‖Δw‖ / ‖w‖, the relative update — the layer's real learning
///   speed. For a conv followed by BN the raw gradient scales with 1/‖w‖, so
///   ‖g‖ alone misreads a shrinking layer as a more active one; the ratio
///   does not;
/// - `cos` = cos(g, v_old): near 1 when successive gradients keep pushing
///   the same way (a streak, drift-chasing), near 0 when the gradient is
///   noise against the momentum, negative when it reverses it (oscillation).
///   Undefined (left out) on a step whose gradient or incoming momentum is 0.
///
/// Why accumulated over the interval rather than a snapshot at the line:
/// a snapshot is one batch's gradient, noisy and blind to everything between
/// lines. Each line covers every step since the previous one: mean and max
/// of `g` and `dww`, mean of `v` and `cos`, and the step count — so any
/// longer window (per 1,000 steps, per LR cycle) is the steps-weighted mean
/// and the max of maxes of the lines inside it, exactly. What it gives up:
/// the timing of a burst inside the interval (the max shows that one
/// happened, not when). Snapshots at saves stay available from the model
/// files (weights + momentum buffer) as before.
struct TrainingDynamicsLayout: Sendable, Equatable {
    /// One per logged layer, in first-appearance order of the tensor names.
    let layerNames: [String]
    /// For each trainable tensor (graph order), the index of its layer.
    let layerIndexOfTensor: [Int]

    /// Groups tensor names into layers by dropping a trailing `.weight` /
    /// `.bias` component; any other name is its own layer.
    init(tensorNames: [String]) {
        var names: [String] = []
        var indexOfName: [String: Int] = [:]
        var layerIndexOfTensor: [Int] = []
        layerIndexOfTensor.reserveCapacity(tensorNames.count)
        for tensorName in tensorNames {
            let layer = Self.layerName(ofTensor: tensorName)
            if let existing = indexOfName[layer] {
                layerIndexOfTensor.append(existing)
            } else {
                indexOfName[layer] = names.count
                layerIndexOfTensor.append(names.count)
                names.append(layer)
            }
        }
        self.layerNames = names
        self.layerIndexOfTensor = layerIndexOfTensor
    }

    static func layerName(ofTensor tensorName: String) -> String {
        for suffix in [".weight", ".bias"] where tensorName.hasSuffix(suffix) && tensorName.count > suffix.count {
            return String(tensorName.dropLast(suffix.count))
        }
        return tensorName
    }
}

/// The per-tensor sums one step reports, in the order the graph writes them.
enum TrainingDynamicsSums {
    static let gradientSquared = 0
    static let gradientDotOldVelocity = 1
    static let oldVelocitySquared = 2
    static let newVelocitySquared = 3
    static let updateSquared = 4
    static let weightSquared = 5
    static let componentCount = 6
}

/// The accumulator: pure, no locking (`TrainingDynamicsLog` owns one under a
/// lock). Every rule is pinned by `TrainingDynamicsTests`.
struct TrainingDynamicsWindow: Sendable {
    enum RecordError: Error, Equatable {
        /// The step's vector is not `tensorCount × componentCount` long.
        case wrongLength(expected: Int, got: Int)
    }

    /// One layer's running totals over the window.
    struct LayerTotals: Sendable, Equatable {
        var steps = 0
        var gradientNormSum = 0.0
        var gradientNormMax = 0.0
        var velocityNormSum = 0.0
        /// Steps whose ‖w‖ was > 0 (the relative update is defined).
        var relativeUpdateSteps = 0
        var relativeUpdateSum = 0.0
        var relativeUpdateMax = 0.0
        /// Steps whose ‖g‖ and ‖v_old‖ were both > 0 (the cosine is defined).
        var cosineSteps = 0
        var cosineSum = 0.0
    }

    let layout: TrainingDynamicsLayout
    private(set) var steps = 0
    /// Steps rejected because some sum was not finite. They add nothing to
    /// the totals; the line reports how many.
    private(set) var nonFiniteSteps = 0
    private(set) var layers: [LayerTotals]
    /// The whole network as one layer (all tensors' sums added).
    private(set) var network = LayerTotals()

    init(layout: TrainingDynamicsLayout) {
        self.layout = layout
        self.layers = Array(repeating: LayerTotals(), count: layout.layerNames.count)
    }

    /// Adds one step's per-tensor sums (`TrainingDynamicsSums` order).
    mutating func record(stepSums sums: [Float]) throws {
        let tensorCount = layout.layerIndexOfTensor.count
        let expected = tensorCount * TrainingDynamicsSums.componentCount
        guard sums.count == expected else {
            throw RecordError.wrongLength(expected: expected, got: sums.count)
        }
        guard sums.allSatisfy(\.isFinite) else {
            nonFiniteSteps += 1
            return
        }
        // Per-layer sums for this step, in Double.
        var perLayer = Array(repeating: [Double](repeating: 0, count: TrainingDynamicsSums.componentCount),
                             count: layers.count)
        var whole = [Double](repeating: 0, count: TrainingDynamicsSums.componentCount)
        for tensor in 0..<tensorCount {
            let layer = layout.layerIndexOfTensor[tensor]
            let base = tensor * TrainingDynamicsSums.componentCount
            for component in 0..<TrainingDynamicsSums.componentCount {
                let value = Double(sums[base + component])
                perLayer[layer][component] += value
                whole[component] += value
            }
        }
        for layer in 0..<layers.count {
            Self.accumulate(perLayer[layer], into: &layers[layer])
        }
        Self.accumulate(whole, into: &network)
        steps += 1
    }

    private static func accumulate(_ s: [Double], into totals: inout LayerTotals) {
        let gradientNorm = s[TrainingDynamicsSums.gradientSquared].squareRoot()
        let oldVelocityNorm = s[TrainingDynamicsSums.oldVelocitySquared].squareRoot()
        let weightNorm = s[TrainingDynamicsSums.weightSquared].squareRoot()
        totals.steps += 1
        totals.gradientNormSum += gradientNorm
        totals.gradientNormMax = max(totals.gradientNormMax, gradientNorm)
        totals.velocityNormSum += s[TrainingDynamicsSums.newVelocitySquared].squareRoot()
        if weightNorm > 0 {
            let relative = s[TrainingDynamicsSums.updateSquared].squareRoot() / weightNorm
            totals.relativeUpdateSteps += 1
            totals.relativeUpdateSum += relative
            totals.relativeUpdateMax = max(totals.relativeUpdateMax, relative)
        }
        if gradientNorm > 0 && oldVelocityNorm > 0 {
            // Rounding can push |cos| a hair past 1; clamp so a mean can't.
            let cosine = s[TrainingDynamicsSums.gradientDotOldVelocity] / (gradientNorm * oldVelocityNorm)
            totals.cosineSteps += 1
            totals.cosineSum += min(1, max(-1, cosine))
        }
    }

    /// The `[DYNAMICS]` line for the window, or nil when it holds no step.
    ///
    /// `trainerStep=<N> steps=<K> nonFinite=<k>` then, for the whole network
    /// (`all`) and each layer in graph order,
    /// `<layer>=g:<mean>/<max>,v:<mean>,dww:<mean>/<max>,cos:<mean>` — `-`
    /// for a value no step of the window defined.
    func line(trainerStep: Int) -> String? {
        guard steps > 0 || nonFiniteSteps > 0 else { return nil }
        var parts = ["[DYNAMICS] trainerStep=\(trainerStep) steps=\(steps) nonFinite=\(nonFiniteSteps)"]
        parts.append("all=" + Self.fields(network))
        for (index, name) in layout.layerNames.enumerated() {
            parts.append("\(name)=" + Self.fields(layers[index]))
        }
        return parts.joined(separator: " ")
    }

    static func fields(_ t: LayerTotals) -> String {
        func number(_ value: Double) -> String { String(format: "%.4g", value) }
        func mean(_ sum: Double, _ count: Int) -> String { count > 0 ? number(sum / Double(count)) : "-" }
        let gradient = t.steps > 0 ? "\(mean(t.gradientNormSum, t.steps))/\(number(t.gradientNormMax))" : "-"
        let relative = t.relativeUpdateSteps > 0
            ? "\(mean(t.relativeUpdateSum, t.relativeUpdateSteps))/\(number(t.relativeUpdateMax))" : "-"
        return "g:\(gradient),v:\(mean(t.velocityNormSum, t.steps)),dww:\(relative),cos:\(mean(t.cosineSum, t.cosineSteps))"
    }
}

/// The trainer's window under a lock: the trainer records on its execution
/// queue after every step; a path's step-line code drains it from wherever it
/// runs. One per trainer, so GUI, corpus replay and train-vs-UCI read the
/// same accumulation.
final class TrainingDynamicsLog: @unchecked Sendable {
    private let lock: OSAllocatedUnfairLock<TrainingDynamicsWindow>
    private let layout: TrainingDynamicsLayout

    init(layout: TrainingDynamicsLayout) {
        self.layout = layout
        self.lock = OSAllocatedUnfairLock(initialState: TrainingDynamicsWindow(layout: layout))
    }

    func record(stepSums sums: [Float]) throws {
        try lock.withLock { try $0.record(stepSums: sums) }
    }

    /// The line for every step since the previous drain (nil if none), and a
    /// fresh window.
    func drainLine(trainerStep: Int) -> String? {
        let layout = self.layout
        return lock.withLock { window in
            let line = window.line(trainerStep: trainerStep)
            window = TrainingDynamicsWindow(layout: layout)
            return line
        }
    }
}
