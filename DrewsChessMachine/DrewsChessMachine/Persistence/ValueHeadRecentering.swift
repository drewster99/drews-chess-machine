//
//  ValueHeadRecentering.swift
//  DrewsChessMachine
//
//  One-shot removal of the W/D/L value head's shared logit offset, applied
//  when a checkpoint is decoded.
//

import Foundation

/// What checkpoint decode did about the value head's shared logit offset.
///
/// The W/D/L head's three logits can all grow by the same amount — a shared
/// offset that softmax cannot see, so nothing in the loss pulls it back. On
/// long reduced-precision lines it reached magnitudes where the compute
/// dtype's rounding step erased the real differences between the classes.
/// The offset is exactly the per-hidden-row mean over the classes in the
/// final layer's weights plus the mean of its bias, so subtracting those
/// removes it with the softmax output unchanged in exact arithmetic.
enum ValueHeadCentering: Sendable, Equatable {
    /// The file carries the `value_head_centered` marker: every save since
    /// the fix writes it, so its weights are used exactly as stored. This is
    /// what keeps new files bit-exact across save and load.
    case alreadyCentered
    /// No marker, but the head has no shared offset to remove — the
    /// single-logit scalar head, where centering would pin the logit at 0.
    case notApplicable
    /// No marker, and the offset was removed at decode (see the report).
    case recentered(ValueHeadRecenteringReport)
}

/// What decode removed from an unmarked W/D/L head. Logged once per load
/// under `[NUMERICS]`, since the weights in memory no longer match the
/// file's bytes.
struct ValueHeadRecenteringReport: Sendable, Equatable {
    /// L2 norm over hidden units of each unit's mean weight across the
    /// classes — the weight part of the removed offset.
    let meanRowNorm: Double
    /// Mean of the class biases — the bias part of the removed offset.
    let biasMean: Double
    /// True when the file was a trainer file and its optimizer velocity for
    /// the same two tensors was recentered too. Momentum would otherwise put
    /// the offset straight back.
    let velocityRecentered: Bool
}

enum ValueHeadRecentering {

    /// `__metadata__` key every save writes, and its one accepted value.
    static let metadataKey = "value_head_centered"
    static let metadataValue = "1"

    enum RecenteringError: Error, CustomStringConvertible {
        case badMarker(String)
        case missingTensor(String)
        case tensorSizeMismatch(name: String, expected: Int, got: Int)
        case weightCountMismatch(base: Int, withVelocity: Int, got: Int)

        var description: String {
            switch self {
            case .badMarker(let value):
                return "checkpoint __metadata__ '\(ValueHeadRecentering.metadataKey)' is '\(value)', "
                    + "expected '\(ValueHeadRecentering.metadataValue)'"
            case .missingTensor(let name):
                return "value-head recentering: tensor '\(name)' is not in the architecture's plan"
            case .tensorSizeMismatch(let name, let expected, let got):
                return "value-head recentering: tensor '\(name)' has \(got) elements, expected \(expected)"
            case .weightCountMismatch(let base, let withVelocity, let got):
                return "value-head recentering: file has \(got) tensors, expected \(base) (model) "
                    + "or \(withVelocity) (trainer with velocity)"
            }
        }
    }

    /// Whether a file's `__metadata__` marks its value head as already
    /// centered. Absent means the file predates the marker; any value other
    /// than `metadataValue` is a malformed file and throws.
    static func isMarkedCentered(_ markerValue: String?) throws -> Bool {
        guard let markerValue else { return false }
        guard markerValue == metadataValue else { throw RecenteringError.badMarker(markerValue) }
        return true
    }

    /// Apply decode-time recentering to `weights`, ordered as
    /// `SafetensorsModelIO.tensorNames(for: architecture, includesVelocity:)`
    /// (the base plan, then one velocity per trainable when a trainer file).
    ///
    /// Only an unmarked W/D/L head is touched, and only its final layer:
    /// in the native `[hidden, classes]` weight each hidden row loses its
    /// mean over the classes, and the bias loses the mean of its values.
    /// The same subtraction is applied to those two tensors' velocities when
    /// present. Means are computed in `Double`, so the result depends only on
    /// the stored values.
    static func apply(
        to weights: inout [[Float]],
        architecture: NetworkArchitecture,
        includesVelocity: Bool,
        markedCentered: Bool
    ) throws -> ValueHeadCentering {
        if markedCentered { return .alreadyCentered }
        guard architecture.valueHeadStyle == .wdlSoftmax else { return .notApplicable }

        let names = SafetensorsModelIO.tensorNames(for: architecture, includesVelocity: includesVelocity)
        guard names.count == weights.count else {
            let base = SafetensorsModelIO.tensorNames(for: architecture, includesVelocity: false).count
            let withVelocity = SafetensorsModelIO.tensorNames(for: architecture, includesVelocity: true).count
            throw RecenteringError.weightCountMismatch(base: base, withVelocity: withVelocity, got: weights.count)
        }
        let classes = architecture.valueHeadClasses
        let hidden = architecture.valueHeadHiddenUnits
        let weightName = "value.wdl_fc2.weight"
        let biasName = "value.wdl_fc2.bias"

        func index(of name: String) throws -> Int {
            guard let i = names.firstIndex(of: name) else { throw RecenteringError.missingTensor(name) }
            return i
        }
        func requireCount(_ i: Int, _ name: String, _ expected: Int) throws {
            guard weights[i].count == expected else {
                throw RecenteringError.tensorSizeMismatch(name: name, expected: expected, got: weights[i].count)
            }
        }

        let weightIndex = try index(of: weightName)
        let biasIndex = try index(of: biasName)
        try requireCount(weightIndex, weightName, hidden * classes)
        try requireCount(biasIndex, biasName, classes)

        let rowMeans = centerRows(&weights[weightIndex], rows: hidden, columns: classes)
        let biasMean = centerRows(&weights[biasIndex], rows: 1, columns: classes)[0]
        var meanRowSquares: Double = 0
        for m in rowMeans { meanRowSquares += m * m }

        if includesVelocity {
            let weightVelocityName = "opt.\(weightName).velocity"
            let biasVelocityName = "opt.\(biasName).velocity"
            let weightVelocityIndex = try index(of: weightVelocityName)
            let biasVelocityIndex = try index(of: biasVelocityName)
            try requireCount(weightVelocityIndex, weightVelocityName, hidden * classes)
            try requireCount(biasVelocityIndex, biasVelocityName, classes)
            _ = centerRows(&weights[weightVelocityIndex], rows: hidden, columns: classes)
            _ = centerRows(&weights[biasVelocityIndex], rows: 1, columns: classes)
        }

        return .recentered(ValueHeadRecenteringReport(
            meanRowNorm: meanRowSquares.squareRoot(),
            biasMean: biasMean,
            velocityRecentered: includesVelocity
        ))
    }

    /// Subtract each row's mean from a row-major `rows × columns` buffer in
    /// place; returns the removed means.
    private static func centerRows(_ values: inout [Float], rows: Int, columns: Int) -> [Double] {
        var means = [Double](repeating: 0, count: rows)
        for r in 0..<rows {
            let base = r * columns
            var sum: Double = 0
            for c in 0..<columns { sum += Double(values[base + c]) }
            let mean = sum / Double(columns)
            means[r] = mean
            for c in 0..<columns {
                values[base + c] = Float(Double(values[base + c]) - mean)
            }
        }
        return means
    }

    /// The one `[NUMERICS]` line a loader writes for a recentered file, or
    /// nil when decode changed nothing.
    static func logLine(for centering: ValueHeadCentering, source: String) -> String? {
        guard case .recentered(let report) = centering else { return nil }
        return "[NUMERICS] value head recentered on load (\(source)): "
            + String(format: "meanRowNorm=%.6g biasMean=%+.6g", report.meanRowNorm, report.biasMean)
            + " velocity=\(report.velocityRecentered ? "recentered" : "none")"
            + " — weights in memory differ from the file's bytes; softmax output unchanged"
    }
}
