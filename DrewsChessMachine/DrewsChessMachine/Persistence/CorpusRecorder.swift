import Foundation
import os

/// Bridges the concurrent self-play flush path to the single-writer
/// `GameCorpus`. Completed (post-filter) games are handed in from the
/// self-play driver's task; this recorder serializes the actual disk appends
/// onto a private serial queue so recording never blocks the self-play hot
/// path and never races the corpus's single-writer contract.
///
/// Lifecycle: created on the main actor at run start (when
/// `recordSelfPlayGames` is on), `record(...)`'d from the driver task per kept
/// game, and `finishAndSeal()`'d once when the run tears down. Recording is
/// best-effort — an append failure is logged and counted but never propagates
/// into training.
final class CorpusRecorder: @unchecked Sendable {
    private let corpus: GameCorpus
    private let queue = DispatchQueue(label: "com.drewschessmachine.corpus-recorder")

    private struct State {
        var finished = false
        var recorded = 0
        var appendErrors = 0
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// The recording corpus's stable id (for `session.json` provenance).
    var corpusID: String { corpus.corpusID }

    private init(corpus: GameCorpus) {
        self.corpus = corpus
    }

    /// Create a fresh recording corpus and begin a self-play ingestion source.
    /// File I/O (directory + `corpus.json` + first shard header) happens on the
    /// caller's thread, before any `record(...)` is dispatched — so the
    /// subsequent queue-only access honors the corpus's single-writer rule.
    static func create(name: String?,
                       comment: String?,
                       shardSoftLimitBytes: Int = GameCorpus.defaultShardSoftLimitBytes) throws -> CorpusRecorder {
        let corpus = try GameCorpus.create(name: name,
                                           comment: comment,
                                           shardSoftLimitBytes: shardSoftLimitBytes)
        try corpus.beginSource(kind: "selfPlay")
        return CorpusRecorder(corpus: corpus)
    }

    /// Record one kept game. Non-blocking: builds the (cheap) `GameRecord`
    /// inline and dispatches the append to the serial queue.
    func record(moves: [ChessMove], result: GameResult) {
        let game = GameRecord(moves: moves, result: result)
        queue.async { [weak self] in
            guard let self else { return }
            if self.state.withLock({ $0.finished }) { return }
            do {
                try self.corpus.append(game)
                self.state.withLock { $0.recorded += 1 }
            } catch {
                self.state.withLock { $0.appendErrors += 1 }
                SessionLogger.shared.log("[CORPUS] append failed: \(error.localizedDescription)")
            }
        }
    }

    /// Flush all pending appends, seal the open shard, and mark the source
    /// complete. Blocks until the queue drains, so the corpus is consistent
    /// before the run's teardown clears its references. Idempotent.
    func finishAndSeal() {
        queue.sync {
            let alreadyFinished: Bool = self.state.withLock { st in
                if st.finished { return true }
                st.finished = true
                return false
            }
            if alreadyFinished { return }
            let recorded = self.state.withLock { $0.recorded }
            let errors = self.state.withLock { $0.appendErrors }
            do {
                try self.corpus.finishSource()
                SessionLogger.shared.log(
                    "[CORPUS] finished recording: \(recorded) games (\(errors) append errors) → corpus \(self.corpus.corpusID)"
                )
            } catch {
                SessionLogger.shared.log("[CORPUS] finishSource failed: \(error.localizedDescription)")
            }
        }
    }
}
