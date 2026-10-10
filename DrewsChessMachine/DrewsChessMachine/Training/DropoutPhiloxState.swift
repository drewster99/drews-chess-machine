//
//  DropoutPhiloxState.swift
//  DrewsChessMachine
//
//  The training graph's dropout RNG state as data (determinism plan, A4 / P4).
//

import Foundation
import Metal
import MetalPerformanceShadersGraph

/// The seven 32-bit words of the Philox state the training graph's dropout
/// draws thread block to block and advance once per step.
///
/// The words' *meaning* belongs to MPSGraph — key and counter words in its own
/// layout — so this type treats them as an opaque blob: read back from the GPU,
/// written back unchanged, never interpreted. An OS update could in principle
/// change the layout or the mapping from state to uniforms; the pinned
/// derived-state test is the canary for that, and a failure there means dropout
/// streams are not comparable across that OS boundary.
struct DropoutPhiloxState: Sendable, Hashable, Codable {
    /// MPSGraph's Philox state tensor shape: `[7]` Int32.
    static let wordCount = 7

    let words: [Int32]

    init(words: [Int32]) throws {
        guard words.count == Self.wordCount else {
            throw DropoutPhiloxStateError.wrongWordCount(found: words.count)
        }
        self.words = words
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(words: container.decode([Int32].self))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(words)
    }

    /// The state MPSGraph derives from `seed`, built in a one-off graph and read
    /// back. Lets the training graph take its initial state as a fed value
    /// instead of a constant baked in at build time, so the same executable
    /// serves any seed and a saved state can be written back.
    static func derived(
        fromSeed seed: Int, device: MTLDevice, commandQueue: MTLCommandQueue
    ) throws -> DropoutPhiloxState {
        let graph = MPSGraph()
        let state = graph.randomPhiloxStateTensor(withSeed: seed, name: "dropout_rng_seed")
        let results = try GPUSubmission.runGraph(
            graph, on: commandQueue, feeds: [:], targetTensors: [state], targetOperations: nil,
            stage: .dropoutStateDerive)
        guard let data = results[state] else {
            throw DropoutPhiloxStateError.missingResult
        }
        return try DropoutPhiloxState(reading: data)
    }

    /// Read the seven words out of an Int32 `[7]` tensor result.
    init(reading data: MPSGraphTensorData) throws {
        guard data.dataType == .int32 else {
            throw DropoutPhiloxStateError.wrongDataType(data.dataType.rawValue)
        }
        let elementCount = data.shape.reduce(1) { $0 * $1.intValue }
        guard elementCount == Self.wordCount else {
            throw DropoutPhiloxStateError.wrongWordCount(found: elementCount)
        }
        var words = [Int32](repeating: 0, count: Self.wordCount)
        words.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else {
                preconditionFailure("DropoutPhiloxState: a non-empty word buffer has a base address")
            }
            data.mpsndarray().readBytes(base, strideBytes: nil)
        }
        try self.init(words: words)
    }

    /// The words as an Int32 `[7]` tensor value, for the training graph's
    /// state-assign placeholder.
    func tensorData(device: MTLDevice) -> MPSGraphTensorData {
        let bytes = words.withUnsafeBytes { Data($0) }
        return MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device),
            data: bytes,
            shape: [NSNumber(value: Self.wordCount)],
            dataType: .int32
        )
    }
}

enum DropoutPhiloxStateError: Error, CustomStringConvertible, LocalizedError {
    case wrongWordCount(found: Int)
    case wrongDataType(UInt32)
    case missingResult
    /// The trainer's network was built without dropout scaffolding, so it has
    /// no Philox state to read or write. Training graphs always carry it; this
    /// is reached only through a network built in inference mode.
    case noDropoutScaffolding

    var description: String {
        switch self {
        case .wrongWordCount(let found):
            return "dropout Philox state must be exactly \(DropoutPhiloxState.wordCount) Int32 words, found \(found)"
        case .wrongDataType(let raw):
            return "dropout Philox state must be Int32 data, found MPSDataType raw value \(raw)"
        case .missingResult:
            return "MPSGraph returned no value for the dropout Philox state"
        case .noDropoutScaffolding:
            return "this network has no dropout RNG state (it was not built as a training graph)"
        }
    }

    var errorDescription: String? { description }
}
