//
//  SafetensorsFileTests.swift
//  DrewsChessMachineTests
//
//  Phase 2 of the safetensors work: the from-scratch codec. These tests pin
//  round-trip correctness and integrity. `testEmitsInteropSampleFile` writes a
//  real .safetensors to /tmp so the canonical Python `safetensors` library can
//  load it (run externally) — proving the file is genuinely interoperable, not
//  just self-consistent.
//

import XCTest
@testable import DrewsChessMachine

final class SafetensorsFileTests: XCTestCase {

    func testRoundTripSingleTensor() throws {
        let t = SafetensorsTensor(name: "a.weight", shape: [2, 3], data: [0, 1, 2, 3, 4, 5])
        let bytes = try SafetensorsFile.encode(tensors: [t], metadata: [:])
        let (out, _) = try SafetensorsFile.decode(bytes)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0], t)
    }

    func testRoundTripMultipleTensorsPreservesOrderAndValues() throws {
        let tensors = [
            SafetensorsTensor(name: "stem.conv.weight", shape: [4, 1, 1, 1], data: [1, 2, 3, 4]),
            SafetensorsTensor(name: "blocks.0.rezero_alpha", shape: [1], data: [0.5]),
            SafetensorsTensor(name: "value.wdl_fc2.bias", shape: [1, 3], data: [-1, 0, 1]),
        ]
        let bytes = try SafetensorsFile.encode(tensors: tensors, metadata: ["k": "v"])
        let (out, meta) = try SafetensorsFile.decode(bytes)
        XCTAssertEqual(out, tensors)            // order + values preserved
        XCTAssertEqual(meta["k"], "v")
    }

    func testMetadataRoundTripIncludingContentHash() throws {
        let t = SafetensorsTensor(name: "x", shape: [3], data: [1.5, 2.5, 3.5])
        let bytes = try SafetensorsFile.encode(
            tensors: [t],
            metadata: ["dcm_format_version": "3", "model_id": "20260601-11-bzw3-25"]
        )
        let (_, meta) = try SafetensorsFile.decode(bytes)
        XCTAssertEqual(meta["dcm_format_version"], "3")
        XCTAssertEqual(meta["model_id"], "20260601-11-bzw3-25")
        XCTAssertNotNil(meta[SafetensorsFile.contentHashKey])
        XCTAssertEqual(meta[SafetensorsFile.contentHashKey]?.count, 64) // hex sha256
    }

    func testTamperedDataRegionFailsHash() throws {
        let t = SafetensorsTensor(name: "x", shape: [4], data: [1, 2, 3, 4])
        var bytes = try SafetensorsFile.encode(tensors: [t], metadata: [:])
        // Flip the very last byte (in the data region) — must be caught.
        bytes[bytes.count - 1] ^= 0xFF
        XCTAssertThrowsError(try SafetensorsFile.decode(bytes)) { error in
            guard case SafetensorsError.contentHashMismatch = error else {
                return XCTFail("expected contentHashMismatch, got \(error)")
            }
        }
    }

    func testEmptyAndNegativeAndFractionalValues() throws {
        let t = SafetensorsTensor(name: "v", shape: [5], data: [0, -0.0, 3.14159, -2.71828, .greatestFiniteMagnitude])
        let bytes = try SafetensorsFile.encode(tensors: [t], metadata: [:])
        let (out, _) = try SafetensorsFile.decode(bytes)
        // Exact bit round-trip (memcpy path).
        XCTAssertEqual(out[0].data.map(\.bitPattern), t.data.map(\.bitPattern))
    }

    func testTruncatedFileThrows() {
        XCTAssertThrowsError(try SafetensorsFile.decode(Data([0, 1, 2])))
    }

    // MARK: - Malformed-input hardening (decoder sees external/hand-edited files)

    /// Frame a custom header dict + data region into safetensors bytes,
    /// bypassing encode()'s validation so malformed inputs can be constructed.
    private func frame(_ header: [String: Any], dataRegion: Data = Data()) throws -> Data {
        let hdr = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var out = Data()
        var len = UInt64(hdr.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(hdr)
        out.append(dataRegion)
        return out
    }

    func testHugeHeaderLengthThrowsNotTraps() {
        // 8-byte length prefix = UInt64.max, then a few bytes. Must throw, not
        // trap on Int(headerLen) / the addition.
        var bytes = Data()
        var len = UInt64.max.littleEndian
        withUnsafeBytes(of: &len) { bytes.append(contentsOf: $0) }
        bytes.append(contentsOf: [0, 0, 0, 0])
        XCTAssertThrowsError(try SafetensorsFile.decode(bytes)) { error in
            guard case SafetensorsError.truncated = error else {
                return XCTFail("expected .truncated, got \(error)")
            }
        }
    }

    func testOverflowingShapeProductThrowsNotTraps() throws {
        let header: [String: Any] = [
            "t": ["dtype": "F32", "shape": [NSNumber(value: Int.max), 2],
                  "data_offsets": [0, 0]] as [String: Any]
        ]
        let bytes = try frame(header)
        XCTAssertThrowsError(try SafetensorsFile.decode(bytes)) { error in
            guard case SafetensorsError.shapeProductOverflow = error else {
                return XCTFail("expected .shapeProductOverflow, got \(error)")
            }
        }
    }

    func testNegativeDimensionThrows() throws {
        let header: [String: Any] = [
            "t": ["dtype": "F32", "shape": [-1, 2], "data_offsets": [0, 0]] as [String: Any]
        ]
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header))) { error in
            guard case SafetensorsError.negativeDimension = error else {
                return XCTFail("expected .negativeDimension, got \(error)")
            }
        }
    }

    func testMetadataNotStringMapThrows() throws {
        let header: [String: Any] = ["__metadata__": ["k": 123]]
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header))) { error in
            guard case SafetensorsError.metadataNotStringMap = error else {
                return XCTFail("expected .metadataNotStringMap, got \(error)")
            }
        }
    }

    func testUnsupportedDTypeThrows() throws {
        let header: [String: Any] = [
            "t": ["dtype": "BF16", "shape": [1], "data_offsets": [0, 2]] as [String: Any]
        ]
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header, dataRegion: Data([0, 0])))) { error in
            guard case SafetensorsError.unsupportedDType = error else {
                return XCTFail("expected .unsupportedDType, got \(error)")
            }
        }
    }

    func testOffsetsOutOfRangeThrows() throws {
        let header: [String: Any] = [
            "t": ["dtype": "F32", "shape": [1], "data_offsets": [0, 999]] as [String: Any]
        ]
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header, dataRegion: Data([0, 0, 0, 0])))) { error in
            guard case SafetensorsError.offsetsOutOfRange = error else {
                return XCTFail("expected .offsetsOutOfRange, got \(error)")
            }
        }
    }

    func testDataRegionNotCoveredThrows() throws {
        // One tensor covering 4 of 8 bytes leaves a gap.
        let header: [String: Any] = [
            "t": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]] as [String: Any]
        ]
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header, dataRegion: Data(count: 8)))) { error in
            guard case SafetensorsError.dataRegionNotCovered = error else {
                return XCTFail("expected .dataRegionNotCovered, got \(error)")
            }
        }
    }

    func testShapeCountMismatchThrows() throws {
        // shape [2,2] = 4 floats = 16 bytes, but offsets span 8 bytes.
        let header: [String: Any] = [
            "t": ["dtype": "F32", "shape": [2, 2], "data_offsets": [0, 8]] as [String: Any]
        ]
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header, dataRegion: Data(count: 8)))) { error in
            guard case SafetensorsError.shapeCountMismatch = error else {
                return XCTFail("expected .shapeCountMismatch, got \(error)")
            }
        }
    }

    func testHeaderNotObjectThrows() throws {
        let bytes = try frame([:] as [String: Any]) // empty object decodes fine; build an array instead
        _ = bytes
        let hdr = try JSONSerialization.data(withJSONObject: [1, 2, 3])
        var out = Data()
        var len = UInt64(hdr.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(hdr)
        XCTAssertThrowsError(try SafetensorsFile.decode(out)) { error in
            guard case SafetensorsError.headerNotObject = error else {
                return XCTFail("expected .headerNotObject, got \(error)")
            }
        }
    }

    func testHeaderParseFailedThrows() {
        var out = Data()
        var len = UInt64(3).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(contentsOf: Array("abc".utf8)) // not JSON
        XCTAssertThrowsError(try SafetensorsFile.decode(out)) { error in
            guard case SafetensorsError.headerParseFailed = error else {
                return XCTFail("expected .headerParseFailed, got \(error)")
            }
        }
    }

    func testBadTensorEntryThrows() throws {
        let header: [String: Any] = ["t": 5] // tensor value not a dict
        XCTAssertThrowsError(try SafetensorsFile.decode(try frame(header))) { error in
            guard case SafetensorsError.badTensorEntry = error else {
                return XCTFail("expected .badTensorEntry, got \(error)")
            }
        }
    }

    func testEncodeRejectsShapeDataMismatch() {
        // shape [2,2] = 4 elements but data has 3.
        let t = SafetensorsTensor(name: "x", shape: [2, 2], data: [1, 2, 3])
        XCTAssertThrowsError(try SafetensorsFile.encode(tensors: [t], metadata: [:])) { error in
            guard case SafetensorsError.shapeDataMismatch = error else {
                return XCTFail("expected .shapeDataMismatch, got \(error)")
            }
        }
    }

    func testScalarAndEmptyTensorRoundTrip() throws {
        let scalar = SafetensorsTensor(name: "scalar", shape: [], data: [42.0]) // shape [] = 1 element
        let empty = SafetensorsTensor(name: "empty", shape: [0], data: [])      // 0 elements
        let bytes = try SafetensorsFile.encode(tensors: [scalar, empty], metadata: [:])
        let (out, _) = try SafetensorsFile.decode(bytes)
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(Set(out.map(\.name)), ["scalar", "empty"])
        XCTAssertEqual(out.first { $0.name == "scalar" }?.data, [42.0])
        XCTAssertEqual(out.first { $0.name == "empty" }?.data, [])
    }

    /// Writes a real .safetensors to /tmp for external Python validation:
    ///   python3 -c "from safetensors.numpy import load_file;
    ///               print(load_file('/tmp/dcm_st_interop.safetensors'))"
    /// Values follow value[i] = i so a Python check can verify by formula.
    func testEmitsInteropSampleFile() throws {
        let a = SafetensorsTensor(name: "a.weight", shape: [2, 3], data: [0, 1, 2, 3, 4, 5])
        let b = SafetensorsTensor(name: "b.bias", shape: [4], data: [10, 11, 12, 13])
        let bytes = try SafetensorsFile.encode(
            tensors: [a, b],
            metadata: ["dcm_format_version": "3", "note": "interop sample"]
        )
        let url = URL(fileURLWithPath: "/tmp/dcm_st_interop.safetensors")
        try bytes.write(to: url)
        // Self-check the file we just wrote also re-reads in Swift.
        let (out, meta) = try SafetensorsFile.decode(try Data(contentsOf: url))
        XCTAssertEqual(out.map(\.name).sorted(), ["a.weight", "b.bias"])
        XCTAssertEqual(meta["note"], "interop sample")
    }
}
