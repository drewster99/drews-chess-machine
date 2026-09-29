import MetalPerformanceShadersGraph

/// Collects named intermediate tensors while a `ChessNetwork` graph is being
/// built, for the numerics audit (`NumericsAudit`). Only audit networks create
/// one; production networks pass nil to every builder, so recording costs
/// nothing and adds no graph nodes there.
///
/// Used only on the thread building the graph, during `ChessNetwork.init`,
/// and then read once; it is never shared.
final class AnalysisTapRecorder {
    private(set) var taps: [(name: String, tensor: MPSGraphTensor)] = []

    func record(_ name: String, _ tensor: MPSGraphTensor) {
        taps.append((name: name, tensor: tensor))
    }
}
