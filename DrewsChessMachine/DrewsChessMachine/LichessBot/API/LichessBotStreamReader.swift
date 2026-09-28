import Foundation

/// One item from a Lichess NDJSON stream, after splitting.
enum LichessBotStreamItem: Sendable, Equatable {
    /// A complete JSON line.
    case line(Data)
    /// A keep-alive (empty) line.
    case keepAlive
    /// A line over the length cap, discarded; the stream continues.
    case oversizeLineDiscarded(byteCount: Int)
    /// The stream ended partway through a line. Recorded as an anomaly; the
    /// partial bytes are not decoded.
    case truncatedAtEnd(byteCount: Int)
}

enum LichessBotStreamError: LocalizedError, Equatable {
    /// No bytes at all — not even a keep-alive — for longer than the
    /// allowed silence. The connection is presumed dead (ISP IP rotation,
    /// sleep/wake, a half-open socket), so the consumer reconnects (plan §6).
    case stalled(silence: Duration)

    var errorDescription: String? {
        switch self {
        case .stalled(let silence):
            return "Lichess stream went silent for \(silence); treating the connection as dead"
        }
    }
}

/// Turns a raw chunk stream into NDJSON items and watches it for stalls.
enum LichessBotStreamReader {

    /// Items from `chunks`, with a watchdog.
    ///
    /// - Parameters:
    ///   - stallTimeout: the silence currently allowed before the stream is
    ///     declared dead, read afresh at every check. Nil means "no watchdog
    ///     right now" — e.g. a game stream while it is the opponent's turn in
    ///     a slow game, if Lichess sends no keep-alives on game streams.
    ///   - checkInterval: how often the watchdog looks.
    ///
    /// Each chunk the transport delivers resets the silence clock. The real
    /// transport delivers at every line end (so a keep-alive counts) and in
    /// bounded pieces of a long line. The stream ends with
    /// `LichessBotStreamError.stalled` when the watchdog fires, with the
    /// transport's error if the connection fails, or normally when the server
    /// closes it. Stopping iteration cancels the underlying connection.
    static func items(
        from chunks: LichessBotChunkStream,
        time: any LichessBotTimeSource,
        stallTimeout: @escaping @Sendable () -> Duration?,
        checkInterval: Duration
    ) -> AsyncThrowingStream<LichessBotStreamItem, Error> {
        AsyncThrowingStream { continuation in
            let lastActivity = SyncBox<Duration>(time.now())

            // The pump and the watchdog are children of one task: whichever
            // ends the stream returns, and the group then cancels the other.
            // Neither relies on `onTermination` for that, because a handler
            // assigned after the stream has already finished is never called.
            // `onTermination` is only for the consumer stopping.
            let reader = Task {
                await withTaskGroup(of: Void.self) { group in
                    group.addTask {
                        var splitter = LichessBotNDJSONSplitter()
                        do {
                            for try await chunk in chunks {
                                lastActivity.value = time.now()
                                for item in splitter.append(chunk) {
                                    switch item {
                                    case .line(let data):
                                        continuation.yield(.line(data))
                                    case .keepAlive:
                                        continuation.yield(.keepAlive)
                                    case .oversizeLineDiscarded(let byteCount):
                                        continuation.yield(.oversizeLineDiscarded(byteCount: byteCount))
                                    }
                                }
                            }
                            if splitter.pendingByteCount > 0 {
                                continuation.yield(.truncatedAtEnd(byteCount: splitter.pendingByteCount))
                            }
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    group.addTask {
                        while !Task.isCancelled {
                            do {
                                try await time.sleep(for: checkInterval)
                            } catch is CancellationError {
                                return
                            } catch {
                                continuation.finish(throwing: error)
                                return
                            }
                            guard let limit = stallTimeout() else { continue }
                            let silence = time.now() - lastActivity.value
                            if silence > limit {
                                continuation.finish(throwing: LichessBotStreamError.stalled(silence: silence))
                                return
                            }
                        }
                    }
                    // Whichever side ends first has ended the stream; the
                    // other has nothing left to do.
                    _ = await group.next()
                    group.cancelAll()
                }
            }

            continuation.onTermination = { _ in
                reader.cancel()
            }
        }
    }
}
