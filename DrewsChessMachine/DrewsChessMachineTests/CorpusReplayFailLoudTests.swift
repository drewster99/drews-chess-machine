import XCTest
@testable import DrewsChessMachine

/// Guards the "fail loud on disk-full" checkpoint-save policy in
/// `CorpusReplayRunner`. A disk-full (ENOSPC) write failure must be classified as
/// out-of-space and escalate to a thrown `CorpusReplayError.diskFullDuringSave`
/// (which halts the run); every other write failure must stay a non-fatal warning
/// (no throw). This is the regression guard for the 2026-07-06 nt8y hole, where
/// training continued through a full disk and lost hours of checkpoints + logs.
final class CorpusReplayFailLoudTests: XCTestCase {

    // MARK: isOutOfSpace classification

    func testCocoaFileWriteOutOfSpaceIsDiskFull() {
        let err = NSError(domain: NSCocoaErrorDomain,
                          code: CocoaError.Code.fileWriteOutOfSpace.rawValue)
        XCTAssertTrue(CorpusReplayRunner.isOutOfSpace(err))
    }

    func testPosixENOSPCIsDiskFull() {
        let err = NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))
        XCTAssertTrue(CorpusReplayRunner.isOutOfSpace(err))
    }

    func testUnderlyingPosixENOSPCIsDiskFull() {
        // Data.write often wraps the POSIX errno as an underlying error inside a
        // Cocoa write error — that nesting must still be recognized.
        let underlying = NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))
        let err = NSError(domain: NSCocoaErrorDomain,
                          code: CocoaError.Code.fileWriteUnknown.rawValue,
                          userInfo: [NSUnderlyingErrorKey: underlying])
        XCTAssertTrue(CorpusReplayRunner.isOutOfSpace(err))
    }

    func testReadOnlyVolumeIsNotDiskFull() {
        // A read-only / no-permission failure is the case that must STAY non-fatal.
        let err = NSError(domain: NSCocoaErrorDomain,
                          code: CocoaError.Code.fileWriteNoPermission.rawValue)
        XCTAssertFalse(CorpusReplayRunner.isOutOfSpace(err))
    }

    func testUnrelatedErrorIsNotDiskFull() {
        let err = NSError(domain: "com.example.whatever", code: 1)
        XCTAssertFalse(CorpusReplayRunner.isOutOfSpace(err))
        // A POSIX error that is not ENOSPC (e.g. EACCES) must not trip it either.
        let eacces = NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES))
        XCTAssertFalse(CorpusReplayRunner.isOutOfSpace(eacces))
    }

    // MARK: reportSaveFailure escalation policy

    func testDiskFullFailureThrowsAndHalts() {
        let diskFull = NSError(domain: NSCocoaErrorDomain,
                               code: CocoaError.Code.fileWriteOutOfSpace.rawValue)
        XCTAssertThrowsError(
            try CorpusReplayRunner.reportSaveFailure(diskFull, step: 42, what: "enumerated checkpoint")
        ) { error in
            guard case CorpusReplayError.diskFullDuringSave(let step, let what) = error else {
                return XCTFail("expected diskFullDuringSave, got \(error)")
            }
            XCTAssertEqual(step, 42)
            XCTAssertEqual(what, "enumerated checkpoint")
        }
    }

    func testNonDiskFullFailureDoesNotThrow() {
        // A read-only-volume failure is logged as a WARNING and does NOT throw —
        // the run keeps going, matching the convenience-autosave contract.
        let readOnly = NSError(domain: NSCocoaErrorDomain,
                               code: CocoaError.Code.fileWriteNoPermission.rawValue)
        XCTAssertNoThrow(
            try CorpusReplayRunner.reportSaveFailure(readOnly, step: 7, what: "trainer-model save (autosave)")
        )
    }
}
