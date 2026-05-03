import Foundation

/// Which network the Candidate test probe evaluates. `.candidate` is
/// the default — the trainer's current in-flight weights get synced
/// into the candidate inference network and probed. `.champion`
/// bypasses the sync and probes the actual champion network directly,
/// so the user can compare the two at the same position. The champion
/// is frozen between promotions, so its output should be stable over
/// the session; diffing its value-head output against the candidate's
/// at a fixed position is the cheapest way to tell whether training
/// is actually moving the value head (or whether it's saturated at
/// the same spot the random init put it).
enum ProbeNetworkTarget: Sendable, Hashable {
    case candidate
    case champion
}
