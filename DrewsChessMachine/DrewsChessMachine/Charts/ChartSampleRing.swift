import Foundation

/// Pre-allocated, chunked storage for chart sample series. Designed
/// to keep long Play-and-Train sessions cheap on the main thread.
///
/// The earlier design held chart samples in a `@State [Sample]` array
/// that grew unbounded at 1 Hz: every appen forced SwiftUI to walk a
/// thousands-long array on every chart re-render, and `Array`'s
/// geometric growth periodically copied the entire backing storage
/// during the heartbeat. This buffer replaces that pattern.
///
/// Storage is partitioned into fixed-size blocks of `blockSize`
/// elements each (24 hours of 1 Hz samples per block). The first
/// block is reserved up-front in `init`. When a block fills, a fresh
/// block of the same size is reserved and appended to the block
/// list. Existing blocks are never reallocated, so stored elements'
/// addresses stay stable for the life of the ring.
///
/// `reset()` discards all but the first block and clears the first
/// block's contents while keeping its reserved capacity, so the next
/// session reuses the existing allocation rather than paying the
/// block reservation cost again.
///
/// The ring is `@MainActor`-isolated. All chart sample appends and
/// reads happen on the main actor (the heartbeat path). Parent
/// SwiftUI views hold a single ring instance across body
/// re-evaluations and mutate it via `append`; they trigger
/// downstream re-renders by updating a separate `@State` (typically
/// a decimated bucket frame) recomputed from the ring after each
/// append. The ring itself is a reference type, so SwiftUI's view
/// diffing never walks its contents.
@MainActor
final class ChartSampleRing<Element> {
    /// Number of slots reserved per block. 24 hours of 1 Hz samples
    /// = 86 400. Sized so a typical training session (single-digit
    /// hours) finishes inside the first block; longer sessions
    /// extend by adding more blocks on the same cadence.
    nonisolated static var blockSize: Int { 86_400 }

    /// Total number of elements appended since construction or the
    /// last `reset()`. Monotonic between resets.
    private(set) var count: Int = 0

    /// Backing storage. `blocks[i]` holds the elements at linear
    /// indices `[i * blockSize, (i+1) * blockSize)`. Empty after
    /// `init` — the first block is reserved lazily on the first
    /// `append`, so a ring that's never written to (e.g. the user
    /// has chart collection turned off) holds zero element storage.
    /// `blocks[0]`'s reserved capacity is retained across `reset()`
    /// calls once it exists.
    private var blocks: [[Element]] = []

    init() {}

    /// Append a single element. O(1) amortized. Never reallocates an
    /// existing block; allocates a fresh block (with the full
    /// `blockSize` reserved capacity) when the current block fills,
    /// or on the very first append when no block exists yet.
    func append(_ element: Element) {
        if blocks.isEmpty {
            var first: [Element] = []
            first.reserveCapacity(Self.blockSize)
            blocks.append(first)
        }
        var lastIdx = blocks.count - 1
        if blocks[lastIdx].count >= Self.blockSize {
            var fresh: [Element] = []
            fresh.reserveCapacity(Self.blockSize)
            blocks.append(fresh)
            lastIdx = blocks.count - 1
        }
        blocks[lastIdx].append(element)
        count += 1
    }

    /// Random access by linear element index. The precondition guards
    /// an internal data-structure invariant; no caller-supplied input
    /// reaches this code path, so a precondition failure indicates a
    /// programming error.
    subscript(_ index: Int) -> Element {
        precondition(
            index >= 0 && index < count,
            "ChartSampleRing index \(index) out of bounds (count=\(count))"
        )
        let blockIdx = index / Self.blockSize
        let offset = index % Self.blockSize
        return blocks[blockIdx][offset]
    }

    /// Most-recently-appended element, or `nil` if the ring is
    /// empty.
    var last: Element? {
        guard count > 0 else { return nil }
        return self[count - 1]
    }

    /// `true` when no elements have been appended (or all elements
    /// have been cleared via `reset`).
    var isEmpty: Bool { count == 0 }

    /// Bulk-append a sequence of elements. Equivalent to calling
    /// `append` in a loop. Used by session-resume code to restore
    /// a saved chart trajectory in one shot. Reserves block storage
    /// up-front for the incoming count so a multi-block restore
    /// pays each fresh-block allocation exactly once. The ring has
    /// no observer hooks, so this only differs from a naive loop
    /// in the up-front capacity reservation.
    func bulkRestore(_ samples: [Element]) {
        guard !samples.isEmpty else { return }
        // Reserve enough block headroom for `samples.count` more
        // appends so the loop below never re-grows `blocks` in the
        // middle. After the call: count goes up by samples.count,
        // and the trailing block holds at most blockSize elements.
        let newCount = count + samples.count
        let blocksNeeded = (newCount + Self.blockSize - 1) / Self.blockSize
        while blocks.count < blocksNeeded {
            var fresh: [Element] = []
            fresh.reserveCapacity(Self.blockSize)
            blocks.append(fresh)
        }
        for sample in samples {
            append(sample)
        }
    }

    /// Drop every appended element. Retains the first block's
    /// reserved capacity so the next session reuses the existing
    /// allocation; releases all subsequent blocks. No-op (and no
    /// allocation) when the ring was never written to.
    func reset() {
        count = 0
        if blocks.isEmpty { return }
        if blocks.count > 1 {
            blocks.removeLast(blocks.count - 1)
        }
        blocks[0].removeAll(keepingCapacity: true)
    }

    /// Binary-search the smallest index `i` whose
    /// `projection(self[i]) >= target`. Returns `count` if every
    /// projection is strictly less than `target`. Stable across
    /// block boundaries because the projection reads through the
    /// random-access subscript.
    ///
    /// Used by chart consumers to clamp a decimation pass to a
    /// visible time window without iterating the full ring.
    func firstIndex(
        elapsedSecAtLeast target: Double,
        projection: (Element) -> Double
    ) -> Int {
        var lo = 0
        var hi = count
        while lo < hi {
            let mid = (lo + hi) / 2
            if projection(self[mid]) < target {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        return lo
    }
}
