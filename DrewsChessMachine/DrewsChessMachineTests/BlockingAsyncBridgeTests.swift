import XCTest
@testable import DrewsChessMachine

/// `runBlocking`, the one async→sync bridge of the command-line paths: it
/// returns the work's value, rethrows its error, and carries a non-Sendable
/// value across. Synchronous test methods, so the caller is not in a task.
final class BlockingAsyncBridgeTests: XCTestCase {

    private struct Failure: Error, Equatable {
        let code: Int
    }

    private final class NotSendable {
        var value = 0
    }

    func testItReturnsTheValue() throws {
        let value = try runBlocking { () async throws -> Int in
            await Task.yield()
            return 42
        }
        XCTAssertEqual(value, 42)
    }

    func testItRethrowsTheError() {
        XCTAssertThrowsError(try runBlocking { () async throws -> Int in throw Failure(code: 7) }) { error in
            XCTAssertEqual(error as? Failure, Failure(code: 7))
        }
    }

    func testANonSendableValueCrosses() throws {
        let box = try runBlocking { () async throws -> NotSendable in
            let made = NotSendable()
            made.value = 5
            return made
        }
        XCTAssertEqual(box.value, 5)
    }

    /// The precondition's signal: inside a task there is a current task.
    func testTheTaskCheckSeesATask() async {
        XCTAssertTrue(withUnsafeCurrentTask { $0 != nil })
    }

    func testOutsideATaskThereIsNone() {
        XCTAssertTrue(withUnsafeCurrentTask { $0 == nil })
    }
}
