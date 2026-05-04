//
//  SyncBox.swift
//  DrewsChessMachine
//
//  Created by Andrew Benson on 5/3/26.
//

import Foundation
import os

final class SyncBox<T: Sendable>: @unchecked Sendable {
    private let lock: OSAllocatedUnfairLock<T>
    public var value: T {
        get {
            lock.withLock { lockedValue in
                return lockedValue
            }
        }
        set {
            lock.withLock { lockedValue in
                lockedValue = newValue
            }
        }
    }
    public func modify(_ modifyValue: @Sendable (inout T) -> Void) {
        lock.withLock { lockedValue in
            modifyValue(&lockedValue)
        }
    }

    public init(_ value: T) {
        lock = OSAllocatedUnfairLock(initialState: value)
    }
}
