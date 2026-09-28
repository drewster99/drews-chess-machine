import Security
import XCTest
@testable import DrewsChessMachine

/// `LichessBotTokenStore` against the real data-protection keychain (plan
/// §12.2). This needs the app's Keychain Sharing entitlement, so it is also
/// the check that the entitlement is in place. Items use a per-test account
/// name, never a real Lichess id, and are deleted afterwards.
final class LichessBotTokenStoreTests: XCTestCase {

    private let store = LichessBotTokenStore()
    private var account: String!

    override func setUpWithError() throws {
        account = "dcm-keychain-test-\(UUID().uuidString)"
    }

    override func tearDownWithError() throws {
        try store.delete(account: account)
    }

    func testSaveReadReplaceDelete() throws {
        XCTAssertNil(try store.read(account: account))
        try store.save(token: "lip_first", account: account)
        XCTAssertEqual(try store.read(account: account), "lip_first")
        try store.save(token: "lip_second", account: account)
        XCTAssertEqual(try store.read(account: account), "lip_second")
        try store.delete(account: account)
        XCTAssertNil(try store.read(account: account))
        XCTAssertNoThrow(try store.delete(account: account), "deleting a missing item is not an error")
    }

    /// E36: readable after first unlock, so an overnight reconnect with the
    /// screen locked can still authenticate.
    func testItemIsAccessibleAfterFirstUnlock() throws {
        try store.save(token: "lip_attributes", account: account)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: LichessBotTokenStore.service,
            kSecAttrAccount as String: account as String,
            kSecUseDataProtectionKeychain as String: true,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        XCTAssertEqual(status, errSecSuccess)
        let attributes = try XCTUnwrap(result as? [String: Any])
        XCTAssertEqual(attributes[kSecAttrAccessible as String] as? String, kSecAttrAccessibleAfterFirstUnlock as String)
    }
}
