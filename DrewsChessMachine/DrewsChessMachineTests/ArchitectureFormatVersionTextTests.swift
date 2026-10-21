import XCTest
@testable import DrewsChessMachine

/// `ArchitectureFormat.safetensorsFormatVersion`: a file's `dcm_format_version`
/// text is a version exactly when Swift's `Int(String)` parses it to a value
/// from 1 to `currentVersion`.
final class ArchitectureFormatVersionTextTests: XCTestCase {
    private let source = "v.safetensors"

    /// Protects: what counts as a version — an optional sign and ASCII digits,
    /// nothing else, and only a positive value.
    /// Fails if: the parser trims whitespace, accepts underscores, decimals or
    /// non-ASCII digits, or accepts zero or a negative version.
    func testOnlyAPositiveSwiftIntegerIsAVersion() throws {
        XCTAssertEqual(try ArchitectureFormat.safetensorsFormatVersion(metadataValue: "+9", source: source), 9)
        XCTAssertEqual(try ArchitectureFormat.safetensorsFormatVersion(metadataValue: "09", source: source), 9)
        for text in ["0", "-1", "-0", "1_0", " 9", "9 ", "9\n", "9.0", "", "+", "nine", "\u{0669}"] {
            XCTAssertThrowsError(try ArchitectureFormat.safetensorsFormatVersion(metadataValue: text, source: source),
                                 text.debugDescription) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .unparseableVersion(value: text, source: source), text.debugDescription)
            }
        }
    }

    /// Protects: an unversioned file is the legacy format. (A newer version is
    /// refused: SEBetaInitTests.testFutureAndMalformedVersionsAreRejected.)
    /// Fails if: a missing version stops meaning the legacy version.
    func testNoVersionIsTheLegacyFormat() throws {
        XCTAssertEqual(try ArchitectureFormat.safetensorsFormatVersion(metadataValue: nil, source: source),
                       ArchitectureFormat.unversionedLegacyVersion)
    }
}
