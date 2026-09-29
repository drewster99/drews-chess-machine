//
//  ValueHeadRecenteringTests.swift
//  DrewsChessMachineTests
//
//  Pins the decode-time value-head recentering (`ValueHeadRecentering`):
//
//  - An unmarked W/D/L head loses each hidden row's class mean from the
//    final layer's weights and the mean of its biases, with the softmax
//    output unchanged, and the report names what was removed.
//  - A trainer file's velocity for those two tensors is recentered too, and
//    nothing else in the file moves.
//  - A marked file, and a scalar-tanh head, are left bit-for-bit alone.
//  - Every save writes the marker, so a new file — and a recentered file
//    saved again — round-trips bit-exactly (safetensors and legacy).
//

import XCTest
@testable import DrewsChessMachine

final class ValueHeadRecenteringTests: XCTestCase {

    // MARK: - Fixtures

    /// A small built-in W/D/L preset that the legacy writer can also encode.
    /// Every property here is independent of the tower, so the smallest
    /// historical preset keeps the fixtures fast.
    private static let architecture = NetworkArchitecture.preset(.v3_8block_3x3)

    /// Deterministic SplitMix64 stream, so a failure reproduces.
    private struct SeededGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        /// Uniform in [-1, 1).
        mutating func nextSigned() -> Float {
            Float(Double(next() >> 11) / Double(1 << 53) * 2 - 1)
        }
    }

    /// Fill every tensor named by `tensorNames(for:includesVelocity:)` with
    /// small random values, then give the value head's final layer (and its
    /// velocity, when present) a large shared offset of the kind training
    /// grows: a per-hidden-row constant added across the classes, and a
    /// constant added to every class bias.
    private func makeWeights(
        architecture: NetworkArchitecture,
        includesVelocity: Bool,
        seed: UInt64
    ) throws -> [[Float]] {
        var rng = SeededGenerator(state: seed)
        let names = SafetensorsModelIO.tensorNames(for: architecture, includesVelocity: includesVelocity)
        let plan = architecture.weightTensorPlan()
        let trainables = plan.filter { $0.kind != .bnRunningStat }
        var counts = plan.map(\.elementCount)
        if includesVelocity { counts.append(contentsOf: trainables.map(\.elementCount)) }
        XCTAssertEqual(counts.count, names.count)

        var weights: [[Float]] = counts.map { count in
            (0..<count).map { _ in rng.nextSigned() * 0.1 }
        }
        guard architecture.valueHeadStyle == .wdlSoftmax else { return weights }

        let classes = architecture.valueHeadClasses
        let hidden = architecture.valueHeadHiddenUnits
        func offsetRows(_ name: String, rows: Int, rowOffset: (Int) -> Float) throws {
            let i = try XCTUnwrap(names.firstIndex(of: name), "\(name) missing from the tensor names")
            for r in 0..<rows {
                for c in 0..<classes { weights[i][r * classes + c] += rowOffset(r) }
            }
        }
        try offsetRows("value.wdl_fc2.weight", rows: hidden) { r in Float(r % 7) * 0.75 - 2 }
        try offsetRows("value.wdl_fc2.bias", rows: 1) { _ in 509.5 }
        if includesVelocity {
            try offsetRows("opt.value.wdl_fc2.weight.velocity", rows: hidden) { r in Float(r % 5) * 0.01 }
            try offsetRows("opt.value.wdl_fc2.bias.velocity", rows: 1) { _ in 0.25 }
        }
        return weights
    }

    private func index(of name: String, architecture: NetworkArchitecture, includesVelocity: Bool) throws -> Int {
        try XCTUnwrap(
            SafetensorsModelIO.tensorNames(for: architecture, includesVelocity: includesVelocity).firstIndex(of: name),
            "\(name) missing from the tensor names"
        )
    }

    /// Softmax over the final layer for hidden activation `h`, in Double.
    private func valueSoftmax(
        hidden h: [Double], weights w: [Float], bias b: [Float], classes: Int
    ) -> [Double] {
        var logits = [Double](repeating: 0, count: classes)
        for c in 0..<classes {
            var sum = Double(b[c])
            for r in 0..<h.count { sum += h[r] * Double(w[r * classes + c]) }
            logits[c] = sum
        }
        let maxLogit = logits.reduce(-Double.infinity, Swift.max)
        let exps = logits.map { Foundation.exp($0 - maxLogit) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }

    private func assertBitEqual(_ a: [[Float]], _ b: [[Float]], _ message: String,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "\(message): tensor count", file: file, line: line)
        for (i, (x, y)) in zip(a, b).enumerated() {
            XCTAssertEqual(x.count, y.count, "\(message): tensor \(i) size", file: file, line: line)
            for j in 0..<min(x.count, y.count) where x[j].bitPattern != y[j].bitPattern {
                XCTFail("\(message): tensor \(i) element \(j) differs (\(x[j]) vs \(y[j]))", file: file, line: line)
                return
            }
        }
    }

    private struct NotRecentered: Error, CustomStringConvertible {
        let centering: ValueHeadCentering
        var description: String { "expected .recentered, got \(centering)" }
    }

    private func report(_ centering: ValueHeadCentering) throws -> ValueHeadRecenteringReport {
        guard case .recentered(let report) = centering else { throw NotRecentered(centering: centering) }
        return report
    }

    // MARK: - apply

    func testRecenteringRemovesClassMeansAndKeepsSoftmax() throws {
        let arch = Self.architecture
        XCTAssertEqual(arch.valueHeadStyle, .wdlSoftmax, "the fixture preset is the W/D/L head")
        let original = try makeWeights(architecture: arch, includesVelocity: false, seed: 1)
        var weights = original
        let centering = try ValueHeadRecentering.apply(
            to: &weights, architecture: arch, includesVelocity: false, markedCentered: false)
        let rep = try report(centering)

        let classes = arch.valueHeadClasses
        let hidden = arch.valueHeadHiddenUnits
        let wi = try index(of: "value.wdl_fc2.weight", architecture: arch, includesVelocity: false)
        let bi = try index(of: "value.wdl_fc2.bias", architecture: arch, includesVelocity: false)

        // Every row of the weight, and the bias, now has zero class mean.
        for r in 0..<hidden {
            let mean = (0..<classes).reduce(0.0) { $0 + Double(weights[wi][r * classes + $1]) } / Double(classes)
            XCTAssertEqual(mean, 0, accuracy: 1e-5, "fc2 row \(r) class mean")
        }
        let biasMeanAfter = weights[bi].reduce(0.0) { $0 + Double($1) } / Double(classes)
        XCTAssertEqual(biasMeanAfter, 0, accuracy: 1e-4, "fc2 bias class mean")

        // The report names what was removed.
        let originalBiasMean = original[bi].reduce(0.0) { $0 + Double($1) } / Double(classes)
        XCTAssertEqual(rep.biasMean, originalBiasMean, accuracy: 1e-9)
        var meanRowSquares = 0.0
        for r in 0..<hidden {
            let m = (0..<classes).reduce(0.0) { $0 + Double(original[wi][r * classes + $1]) } / Double(classes)
            meanRowSquares += m * m
        }
        XCTAssertEqual(rep.meanRowNorm, meanRowSquares.squareRoot(), accuracy: 1e-9)
        XCTAssertFalse(rep.velocityRecentered)

        // Softmax is unchanged for arbitrary hidden activations.
        var rng = SeededGenerator(state: 99)
        for trial in 0..<16 {
            let h = (0..<hidden).map { _ in Double(max(0, rng.nextSigned())) }
            let before = valueSoftmax(hidden: h, weights: original[wi], bias: original[bi], classes: classes)
            let after = valueSoftmax(hidden: h, weights: weights[wi], bias: weights[bi], classes: classes)
            for c in 0..<classes {
                XCTAssertEqual(after[c], before[c], accuracy: 1e-5, "trial \(trial) class \(c)")
            }
        }

        // Nothing else moved.
        for i in 0..<weights.count where i != wi && i != bi {
            XCTAssertEqual(weights[i], original[i], "tensor \(i) must be untouched")
        }
    }

    func testTrainerFileVelocityIsRecenteredToo() throws {
        let arch = Self.architecture
        let original = try makeWeights(architecture: arch, includesVelocity: true, seed: 2)
        var weights = original
        let rep = try report(ValueHeadRecentering.apply(
            to: &weights, architecture: arch, includesVelocity: true, markedCentered: false))
        XCTAssertTrue(rep.velocityRecentered)

        let classes = arch.valueHeadClasses
        let hidden = arch.valueHeadHiddenUnits
        let touched = try [
            "value.wdl_fc2.weight", "value.wdl_fc2.bias",
            "opt.value.wdl_fc2.weight.velocity", "opt.value.wdl_fc2.bias.velocity",
        ].map { try index(of: $0, architecture: arch, includesVelocity: true) }
        let wv = touched[2]
        let bv = touched[3]
        for r in 0..<hidden {
            let mean = (0..<classes).reduce(0.0) { $0 + Double(weights[wv][r * classes + $1]) } / Double(classes)
            XCTAssertEqual(mean, 0, accuracy: 1e-6, "fc2 weight velocity row \(r) class mean")
        }
        let biasVelocityMean = weights[bv].reduce(0.0) { $0 + Double($1) } / Double(classes)
        XCTAssertEqual(biasVelocityMean, 0, accuracy: 1e-6, "fc2 bias velocity class mean")
        for i in 0..<weights.count where !touched.contains(i) {
            XCTAssertEqual(weights[i], original[i], "tensor \(i) must be untouched")
        }
    }

    func testMarkedFileIsLeftBitExact() throws {
        let arch = Self.architecture
        let original = try makeWeights(architecture: arch, includesVelocity: true, seed: 3)
        var weights = original
        let centering = try ValueHeadRecentering.apply(
            to: &weights, architecture: arch, includesVelocity: true, markedCentered: true)
        XCTAssertEqual(centering, .alreadyCentered)
        assertBitEqual(weights, original, "marked file")
    }

    func testScalarTanhHeadIsNotRecentered() throws {
        var arch = Self.architecture
        arch.valueHeadStyle = .scalarTanh
        let original = try makeWeights(architecture: arch, includesVelocity: false, seed: 4)
        var weights = original
        let centering = try ValueHeadRecentering.apply(
            to: &weights, architecture: arch, includesVelocity: false, markedCentered: false)
        XCTAssertEqual(centering, .notApplicable)
        assertBitEqual(weights, original, "scalar-tanh head")
    }

    func testMarkerParsing() throws {
        XCTAssertFalse(try ValueHeadRecentering.isMarkedCentered(nil))
        XCTAssertTrue(try ValueHeadRecentering.isMarkedCentered(ValueHeadRecentering.metadataValue))
        XCTAssertThrowsError(try ValueHeadRecentering.isMarkedCentered("0"))
    }

    // MARK: - Safetensors decode / encode

    private func encodeSafetensors(
        _ weights: [[Float]], architecture: NetworkArchitecture, includesVelocity: Bool
    ) throws -> Data {
        try SafetensorsModelIO.encode(
            modelID: "20260928-1-TEST",
            createdAtUnix: 1_800_000_000,
            metadata: ModelCheckpointMetadata(
                creator: "test", trainingStep: 42, parentModelID: "", notes: "value-head recentering"),
            weights: weights,
            architecture: architecture,
            includesVelocity: includesVelocity
        )
    }

    /// The same file with the centering marker removed — what every file
    /// written before the marker existed looks like.
    private func strippingMarker(_ data: Data) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        XCTAssertEqual(metadata[ValueHeadRecentering.metadataKey], ValueHeadRecentering.metadataValue,
                       "every save must write the marker")
        metadata.removeValue(forKey: ValueHeadRecentering.metadataKey)
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    func testNewSafetensorsFileWritesMarkerAndRoundTripsBitExactly() throws {
        let arch = Self.architecture
        for includesVelocity in [false, true] {
            let weights = try makeWeights(architecture: arch, includesVelocity: includesVelocity, seed: 5)
            let data = try encodeSafetensors(weights, architecture: arch, includesVelocity: includesVelocity)
            let (_, metadata) = try SafetensorsFile.decode(data)
            XCTAssertEqual(metadata[ValueHeadRecentering.metadataKey], ValueHeadRecentering.metadataValue)

            let decoded = try SafetensorsModelIO.decode(data)
            XCTAssertEqual(decoded.valueHeadCentering, .alreadyCentered)
            XCTAssertEqual(decoded.file.valueHeadCentering, .alreadyCentered)
            assertBitEqual(decoded.file.weights, weights, "safetensors round trip (velocity=\(includesVelocity))")
        }
    }

    func testUnmarkedSafetensorsFileIsRecenteredOnceThenRoundTripsBitExactly() throws {
        let arch = Self.architecture
        let weights = try makeWeights(architecture: arch, includesVelocity: true, seed: 6)
        let unmarked = try strippingMarker(try encodeSafetensors(weights, architecture: arch, includesVelocity: true))

        // Decode recenters, the same way `apply` does on the raw values.
        let decoded = try SafetensorsModelIO.decode(unmarked)
        let rep = try report(decoded.valueHeadCentering)
        XCTAssertTrue(rep.velocityRecentered)
        XCTAssertEqual(decoded.file.valueHeadCentering, decoded.valueHeadCentering)
        var expected = weights
        _ = try ValueHeadRecentering.apply(
            to: &expected, architecture: arch, includesVelocity: true, markedCentered: false)
        assertBitEqual(decoded.file.weights, expected, "decode-time recentering")

        // Saving the recentered weights writes the marker; loading that file
        // changes nothing more.
        let resaved = try encodeSafetensors(decoded.file.weights, architecture: arch, includesVelocity: true)
        let reloaded = try SafetensorsModelIO.decode(resaved)
        XCTAssertEqual(reloaded.valueHeadCentering, .alreadyCentered)
        assertBitEqual(reloaded.file.weights, decoded.file.weights, "recentered file saved again")
    }

    func testCheckpointManagerLoadRecentersUnmarkedModelFile() throws {
        let arch = Self.architecture
        let weights = try makeWeights(architecture: arch, includesVelocity: false, seed: 7)
        let unmarked = try strippingMarker(try encodeSafetensors(weights, architecture: arch, includesVelocity: false))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("value_head_recenter_\(UUID().uuidString).safetensors")
        try unmarked.write(to: url)
        defer {
            do { try FileManager.default.removeItem(at: url) } catch {
                XCTFail("could not remove \(url.path): \(error)")
            }
        }
        let loaded = try CheckpointManager.loadModelFile(at: url)
        let loadedCentering = try XCTUnwrap(loaded.valueHeadCentering)
        _ = try report(loadedCentering)
        let wi = try index(of: "value.wdl_fc2.weight", architecture: arch, includesVelocity: false)
        let classes = arch.valueHeadClasses
        for r in 0..<arch.valueHeadHiddenUnits {
            let mean = (0..<classes).reduce(0.0) { $0 + Double(loaded.weights[wi][r * classes + $1]) } / Double(classes)
            XCTAssertEqual(mean, 0, accuracy: 1e-5, "loaded fc2 row \(r) class mean")
        }
    }

    // MARK: - Legacy .dcmmodel

    func testLegacyEncodeWritesMarkerAndRoundTripsBitExactly() throws {
        let arch = Self.architecture
        let weights = try makeWeights(architecture: arch, includesVelocity: false, seed: 8)
        let file = ModelCheckpointFile(
            modelID: "20260928-2-TEST",
            createdAtUnix: 1_800_000_001,
            metadata: ModelCheckpointMetadata(
                creator: "test", trainingStep: nil, parentModelID: "", notes: "legacy marker"),
            weights: weights,
            architecture: arch
        )
        let decoded = try ModelCheckpointFile.decode(try file.encode())
        XCTAssertEqual(decoded.valueHeadCentering, .alreadyCentered)
        XCTAssertEqual(decoded.metadata, file.metadata, "the marker must not disturb the metadata")
        assertBitEqual(decoded.weights, weights, "legacy round trip")
    }
}
