import XCTest
@testable import DrewsChessMachine

/// Pins the invariants `generate-build-info.sh` promises about the
/// generated `BuildInfo` of the build under test: the diff hash exists
/// exactly when the compiled project was dirty, and the toolchain fields
/// are never blank (the script refuses to write a blank one, so a blank
/// here would mean the generated file was edited by hand or by another
/// generator).
final class BuildInfoConsistencyTests: XCTestCase {
    func testDiffHashIsNilExactlyWhenClean() throws {
        if BuildInfo.gitDirty {
            let hash = try XCTUnwrap(BuildInfo.gitDiffSHA256, "a dirty build names its diff")
            XCTAssertEqual(hash.count, 64, "a SHA-256 is 64 hex digits")
            XCTAssertTrue(hash.allSatisfy { "0123456789abcdef".contains($0) }, "lowercase hex: \(hash)")
        } else {
            XCTAssertNil(BuildInfo.gitDiffSHA256, "a clean build has no diff")
        }
    }

    func testToolchainFieldsAreNeverBlank() {
        XCTAssertFalse(BuildInfo.xcodeBuild.isEmpty, "Xcode build version")
        XCTAssertFalse(BuildInfo.sdkBuild.isEmpty, "SDK build version")
        XCTAssertFalse(BuildInfo.configuration.isEmpty, "build configuration")
    }
}
