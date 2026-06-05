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
