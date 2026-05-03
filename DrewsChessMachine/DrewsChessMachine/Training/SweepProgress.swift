import Foundation

/// Live progress reported from inside a sweep task. The trainer fires the
/// progress callback before each step; we publish that to the UI so the
/// user can see "currently sweeping batch=X, step Y, elapsed Z".
struct SweepProgress: Sendable {
    let batchSize: Int
    let stepsSoFar: Int
    let elapsedSec: Double
}
