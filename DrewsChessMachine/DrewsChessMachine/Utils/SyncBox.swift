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

    /// Mutate the protected value and return a result from the same locked
    /// critical section — the building block for atomic read-modify-return
    /// operations like compare-and-set.
    public func mutate<R: Sendable>(_ body: @Sendable (inout T) -> R) -> R {
        lock.withLock { lockedValue in
            body(&lockedValue)
        }
    }

    /// Read part of the protected value inside the locked critical section.
    /// Unlike `value`, which returns a copy of the whole `T`, this never
    /// copies the value out: when `T` holds an array, a copy alive outside
    /// the lock shares the array's buffer, and a writer that appends to it
    /// meanwhile must then reallocate the whole buffer (copy-on-write).
    public func read<R: Sendable>(_ body: @Sendable (T) -> R) -> R {
        lock.withLock { lockedValue in
            body(lockedValue)
        }
    }

    public init(_ value: T) {
        lock = OSAllocatedUnfairLock(initialState: value)
    }
}
