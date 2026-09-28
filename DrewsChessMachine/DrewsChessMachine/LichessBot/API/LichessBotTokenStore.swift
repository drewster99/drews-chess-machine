import Foundation
import Security

enum LichessBotTokenStoreError: LocalizedError, Equatable {
    case keychain(operation: String, status: OSStatus)
    case unreadableItem

    var errorDescription: String? {
        switch self {
        case .keychain(let operation, let status):
            let detail = SecCopyErrorMessageString(status, nil).map { $0 as String } ?? "OSStatus \(status)"
            if status == errSecMissingEntitlement {
                return "Keychain \(operation) failed: the app is missing the Keychain Sharing entitlement the data-protection keychain requires (\(detail))"
            }
            return "Keychain \(operation) failed: \(detail)"
        case .unreadableItem:
            return "The stored Lichess token could not be read as text"
        }
    }
}

/// Stores the Lichess bot's API token in the macOS Keychain (plan §12.2).
///
/// - The **data-protection keychain** (`kSecUseDataProtectionKeychain`): the
///   item is private to DCM, access doesn't prompt on rebuilds, and the
///   accessibility class is honored. It needs the Keychain Sharing
///   entitlement on the app target; without it every call fails with
///   `errSecMissingEntitlement`, which is reported as such.
/// - **`kSecAttrAccessibleAfterFirstUnlock`**: readable once the Mac has been
///   unlocked since boot, including while the screen is locked, so an
///   overnight reconnect can still authenticate (plan E36). The default
///   (`WhenUnlocked`) would fail there.
/// - The token is never logged or written anywhere else. Callers read it
///   once when going online and hold it in memory.
struct LichessBotTokenStore: Sendable {
    static let service = "com.drewben.DrewsChessMachine.lichess-bot"

    /// Store `token` for `account` (the Lichess user id), replacing any
    /// existing item.
    func save(token: String, account: String) throws {
        let data = Data(token.utf8)
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]
        let updateStatus = SecItemUpdate(baseQuery(account: account) as CFDictionary, attributes as CFDictionary)
        switch updateStatus {
        case errSecSuccess:
            return
        case errSecItemNotFound:
            var addQuery = baseQuery(account: account)
            addQuery.merge(attributes) { _, new in new }
            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw LichessBotTokenStoreError.keychain(operation: "add", status: addStatus)
            }
        default:
            throw LichessBotTokenStoreError.keychain(operation: "update", status: updateStatus)
        }
    }

    /// The token stored for `account`, or nil if there is none.
    func read(account: String) throws -> String? {
        var query = baseQuery(account: account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            guard let data = result as? Data, let token = String(data: data, encoding: .utf8) else {
                throw LichessBotTokenStoreError.unreadableItem
            }
            return token
        case errSecItemNotFound:
            return nil
        default:
            throw LichessBotTokenStoreError.keychain(operation: "read", status: status)
        }
    }

    /// Remove the token for `account`. Removing a token that isn't there is
    /// not an error.
    func delete(account: String) throws {
        let status = SecItemDelete(baseQuery(account: account) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw LichessBotTokenStoreError.keychain(operation: "delete", status: status)
        }
    }

    private func baseQuery(account: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: account,
            kSecUseDataProtectionKeychain as String: true,
        ]
    }
}
