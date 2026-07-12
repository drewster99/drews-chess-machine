import Foundation
import Darwin

/// Writing to a UCI engine that has already exited would raise
/// `SIGPIPE`, killing the whole process before the failed `write`
/// could return. Ignoring it once makes such writes surface as
/// ordinary throwable `EPIPE` errors instead (caught in `send`). This
/// is process-global and idempotent; it is evaluated once, lazily, on
/// the first `UCIArbiter.launch()`.
private let uciArbiterIgnoresSIGPIPE: Void = {
    _ = signal(SIGPIPE, SIG_IGN)
}()

/// The *arbiter* (a.k.a. GUI / controller) side of the UCI protocol:
/// DrewsChessMachine driving an external UCI engine as an opponent for
/// train-vs-UCI game generation.
///
/// This is the mirror image of `UCIEngine` (which is DCM-*as*-engine,
/// driven by cutechess). Here DCM launches an external engine process
/// (Stockfish, Sloppy, …), speaks UCI *to* it over its stdin/stdout,
/// and reads back the engine's moves:
///
/// ```
/// DCM (arbiter) ──uci/setoption/isready/ucinewgame/position/go──▶ engine
/// DCM (arbiter) ◀──────────uciok/readyok/bestmove───────────────  engine
/// ```
///
/// **One process = one concurrent game.** UCI is serial per process —
/// a single engine handles exactly one `position`/`go` at a time — so
/// each `UCIArbiter` backs exactly one game slot in the batched driver.
/// The process is launched **once** (`launch()` + `handshake()`) and
/// reused across many games via `startNewGame()`; it is never
/// respawned per game.
///
/// **Serial-use contract.** A single arbiter instance must have at
/// most one outstanding request at a time (`handshake`,
/// `startNewGame`, or `bestMove`). That matches how the driver uses it
/// — one slot, one in-flight opponent move — and how UCI itself works.
///
/// The chess-agnostic split is deliberate: this actor speaks *strings*
/// (`position startpos moves e2e4 …`, `bestmove e7e8q`). Turning a
/// game's `[ChessMove]` into LAN tokens and resolving a returned LAN
/// token back into a legal `ChessMove` is the caller's job (via
/// `ChessMove.uci` / `ChessMove.parseUCI(_:legal:)`), keeping
/// `MoveGenerator` the single source of truth for legality.
actor UCIArbiter {

    /// A `setoption name <name> value <value>` pair applied during the
    /// handshake — e.g. `UCI_LimitStrength=true`, `UCI_Elo=1400`.
    struct Option: Sendable, Equatable {
        let name: String
        let value: String
    }

    /// Everything needed to launch and drive one external engine.
    struct Configuration: Sendable {
        /// Path to the engine executable.
        let command: URL
        /// Extra process arguments (usually empty for UCI engines).
        let arguments: [String]
        /// Options set once, after `uciok`, before the first game.
        let options: [Option]
        /// The `go` limit string, e.g. `"nodes 1"`, `"depth 6"`,
        /// `"movetime 50"`. Sent verbatim after `go ` on every move.
        let goLimit: String
        /// Human label for logs / stats correlation, e.g. `"stockfish#1"`.
        let label: String
        /// Max wait for `uciok` / `readyok` during handshake and
        /// new-game sync.
        let handshakeTimeout: Duration
        /// Max wait for a `bestmove` after `go`. A slower limit than
        /// this indicates a wedged/misbehaving engine.
        let moveTimeout: Duration

        init(
            command: URL,
            arguments: [String] = [],
            options: [Option] = [],
            goLimit: String,
            label: String,
            handshakeTimeout: Duration = .seconds(10),
            moveTimeout: Duration = .seconds(30)
        ) {
            self.command = command
            self.arguments = arguments
            self.options = options
            self.goLimit = goLimit
            self.label = label
            self.handshakeTimeout = handshakeTimeout
            self.moveTimeout = moveTimeout
        }
    }

    private let config: Configuration
    private var process: Process?
    private var stdinHandle: FileHandle?

    // Line plumbing. A single reader task (`readerTask`) owns the
    // stdout `AsyncStream` iterator entirely within its own task — so
    // the non-`Sendable` iterator never crosses the actor's isolation
    // (which Swift 6 region isolation forbids). It hands each finished
    // line to the actor via `enqueue(_:)`. `receiveLine()` consumes
    // from `lineBuffer`, or parks a single `waiter` continuation until
    // a line arrives, the stream ends, or the wait is cancelled. Per
    // the serial-use contract there is at most one waiter at a time.
    private var readerTask: Task<Void, Never>?
    private var lineBuffer: [String] = []
    private var waiter: CheckedContinuation<String, any Error>?
    private var readerDone = false

    init(configuration: Configuration) {
        self.config = configuration
    }

    // MARK: - Lifecycle

    /// Spawn the engine process and start reading its stdout. Does not
    /// perform the UCI handshake — call `handshake()` next.
    func launch() throws {
        guard process == nil else { return }
        _ = uciArbiterIgnoresSIGPIPE

        // Reset line-reader state so a *relaunch* (after `shutdown()`)
        // starts clean. The previous process's reader set `readerDone =
        // true` at EOF; without this reset the first `receiveLine()` of
        // the new session would throw `.engineTerminated` and the
        // handshake would fail. `shutdown()` awaited the old reader task's
        // completion, so nothing is racing to flip these back, and any
        // parked `waiter` was already resumed there (so it is nil here).
        readerDone = false
        lineBuffer.removeAll(keepingCapacity: true)

        let proc = Process()
        proc.executableURL = config.command
        proc.arguments = config.arguments

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        proc.standardInput = stdinPipe
        proc.standardOutput = stdoutPipe
        // Discard the engine's stderr — banners / search noise we don't
        // consume. (Capturing it is a future diagnostics option.)
        proc.standardError = FileHandle.nullDevice

        let stream = Self.makeLineStream(from: stdoutPipe.fileHandleForReading)
        stdinHandle = stdinPipe.fileHandleForWriting

        do {
            try proc.run()
        } catch {
            stdinHandle = nil
            throw UCIArbiterError.launchFailed(error.localizedDescription)
        }
        process = proc
        // The iterator lives entirely inside this task; only Sendable
        // `String`s cross back to the actor via `enqueue`.
        readerTask = Task { [weak self] in
            for await line in stream {
                await self?.enqueue(line)
            }
            await self?.finishReader()
        }
        SessionLogger.shared.log("[ARBITER] launched \(config.label): \(config.command.path) \(config.arguments.joined(separator: " "))")
    }

    /// Run the `uci` handshake, apply configured options, and confirm
    /// readiness. Skips over `id` / `option` / `info` lines until
    /// `uciok`, then applies each option, then `isready`/`readyok`.
    func handshake() async throws {
        try send("uci")
        _ = try await awaitLine(where: { $0 == "uciok" }, timeout: config.handshakeTimeout)
        for option in config.options {
            try send(UCIProtocol.setOptionCommand(name: option.name, value: option.value))
        }
        try await syncReady()
        SessionLogger.shared.log("[ARBITER] \(config.label) ready (options: \(config.options.map { "\($0.name)=\($0.value)" }.joined(separator: ", ")))")
    }

    /// Reset the engine for a new game (`ucinewgame` + `isready`
    /// barrier). Cheap; no process respawn.
    func startNewGame() async throws {
        try send("ucinewgame")
        try await syncReady()
    }

    /// Ask the engine for its move from the given position.
    ///
    /// - Parameters:
    ///   - startFEN: the game's starting position as a FEN, or `nil`
    ///     for the standard start (`startpos`).
    ///   - moves: the moves played from that start, as UCI LAN tokens
    ///     (`ChessMove.uci`), in order.
    /// - Returns: the engine's `bestmove` — either a LAN token to be
    ///   resolved against the legal-move list, or `.null` when the
    ///   engine reports no move (`bestmove (none)` / `0000`).
    func bestMove(startFEN: String?, moves: [String]) async throws -> UCIBestMove {
        try send(UCIProtocol.positionCommand(startFEN: startFEN, moves: moves))
        try send(UCIProtocol.goCommand(config.goLimit))
        let line = try await awaitLine(
            where: { UCIProtocol.parseBestMove($0) != nil },
            timeout: config.moveTimeout
        )
        // awaitLine's predicate already guaranteed a parse; re-parse to
        // extract the value. A non-parse here would be a logic error.
        guard let best = UCIProtocol.parseBestMove(line) else {
            throw UCIArbiterError.protocolViolation("bestmove line failed to re-parse: \(line)")
        }
        return best
    }

    /// Politely ask the engine to `quit`, then terminate the process and
    /// wait for the stdout reader to finish. Best-effort and non-throwing
    /// — used on teardown and before a relaunch after a wedged engine.
    ///
    /// Awaiting the reader task matters for relaunch: it guarantees the
    /// old reader's final `finishReader()` has run before this returns, so
    /// a subsequent `launch()` starts from clean reader state with no
    /// stale reader task racing to set `readerDone`.
    func shutdown() async {
        // Only send `quit` if the engine is still alive; writing to an
        // exited engine's stdin would fail (now as a caught EPIPE, but
        // there's no reason to attempt it).
        if process?.isRunning == true {
            do {
                try send("quit")
            } catch {
                // Best-effort: the engine may have exited between the
                // isRunning check and the write. Nothing to do on teardown.
            }
        }
        process?.terminate()
        let oldReader = readerTask
        readerTask = nil
        oldReader?.cancel()
        stdinHandle = nil
        process = nil
        // Let the cancelled/EOF'd reader task run its final `finishReader`
        // and complete. The actor is free during this await, so the reader
        // task can make progress.
        await oldReader?.value
    }

    // MARK: - Internals

    /// Send one command line to the engine (a trailing newline is
    /// added). Throws `.engineTerminated` if the pipe is gone.
    private func send(_ command: String) throws {
        guard let handle = stdinHandle else { throw UCIArbiterError.notLaunched }
        do {
            try handle.write(contentsOf: Data((command + "\n").utf8))
        } catch {
            throw UCIArbiterError.engineTerminated
        }
    }

    /// `isready`/`readyok` barrier.
    private func syncReady() async throws {
        try send("isready")
        _ = try await awaitLine(where: { $0 == "readyok" }, timeout: config.handshakeTimeout)
    }

    /// Consume engine output lines until one satisfies `predicate`,
    /// racing the whole wait against `timeout`. Throws `.timeout` if
    /// the engine is silent too long, `.engineTerminated` on EOF.
    private func awaitLine(
        where predicate: @escaping @Sendable (String) -> Bool,
        timeout: Duration
    ) async throws -> String {
        try await withUCITimeout(timeout) { [self] in
            while true {
                let line = try await self.receiveLine()
                if predicate(line) { return line }
            }
        }
    }

    /// A line arrived from the reader task: hand it to a parked waiter,
    /// else buffer it.
    private func enqueue(_ line: String) {
        if let waiter {
            self.waiter = nil
            waiter.resume(returning: line)
        } else {
            lineBuffer.append(line)
        }
    }

    /// The engine closed its stdout (EOF). Fail any parked waiter and
    /// mark the reader done so future reads fail fast.
    private func finishReader() {
        readerDone = true
        if let waiter {
            self.waiter = nil
            waiter.resume(throwing: UCIArbiterError.engineTerminated)
        }
    }

    /// Cancellation reached a parked waiter (e.g. a move timeout fired):
    /// resume it so the awaiting task unwinds instead of hanging.
    private func cancelWaiter() {
        if let waiter {
            self.waiter = nil
            waiter.resume(throwing: CancellationError())
        }
    }

    /// Pull the next line from the engine's stdout: a buffered line if
    /// present, otherwise park until one arrives. Throws
    /// `.engineTerminated` at EOF and `CancellationError` if the
    /// enclosing task (e.g. a `withUCITimeout` race) is cancelled.
    private func receiveLine() async throws -> String {
        if !lineBuffer.isEmpty { return lineBuffer.removeFirst() }
        if readerDone { throw UCIArbiterError.engineTerminated }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                // Re-check under isolation: a line, EOF, or cancellation
                // may have raced in before we parked.
                if !lineBuffer.isEmpty {
                    continuation.resume(returning: lineBuffer.removeFirst())
                } else if readerDone {
                    continuation.resume(throwing: UCIArbiterError.engineTerminated)
                } else if waiter != nil {
                    continuation.resume(throwing: UCIArbiterError.protocolViolation(
                        "concurrent receiveLine — arbiter must be used serially"))
                } else {
                    waiter = continuation
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter() }
        }
    }

    /// Build a newline-delimited `AsyncStream<String>` from a pipe's
    /// read handle via its `readabilityHandler`. Partial lines are
    /// buffered across read chunks (in a serially-accessed box), a
    /// trailing `\r` is stripped, any unterminated tail is flushed at
    /// EOF, and the stream finishes when the engine closes its stdout.
    private static func makeLineStream(from handle: FileHandle) -> AsyncStream<String> {
        AsyncStream { continuation in
            let box = LineBox()
            handle.readabilityHandler = { fileHandle in
                let chunk = fileHandle.availableData
                if chunk.isEmpty {
                    if let tail = box.flush() { continuation.yield(tail) }
                    fileHandle.readabilityHandler = nil
                    continuation.finish()
                    return
                }
                for line in box.appendAndExtractLines(chunk) {
                    continuation.yield(line)
                }
            }
            continuation.onTermination = { _ in
                handle.readabilityHandler = nil
            }
        }
    }
}

/// Partial-line accumulator for `UCIArbiter.makeLineStream`. Marked
/// `@unchecked Sendable` because it is touched only from a single
/// `FileHandle.readabilityHandler`, whose invocations the system
/// serializes per handle — so the mutable `data` needs no lock.
private final class LineBox: @unchecked Sendable {
    /// Initial reserved capacity for the line accumulator, sized to hold a full
    /// pipe-flush burst of `info`/`bestmove` lines — including long deep-search
    /// `info … pv` lines at long time controls — without a resize. `Data` still
    /// grows past it if a single read exceeds it; the compaction paths in
    /// `appendAndExtractLines` retain whatever capacity is reached, so this is
    /// also the steady-state allocation. Tunable in one place.
    private static let initialCapacity = 16_384

    /// Partial-line accumulator, pre-reserved to `initialCapacity` and reused
    /// across reads (never rebuilt per line — see `appendAndExtractLines`).
    private var data = Data(capacity: LineBox.initialCapacity)

    /// Append a chunk and return every complete line it now contains
    /// (newline-terminated), leaving any partial tail buffered.
    ///
    /// Extraction walks a `lineStart` cursor over the accumulated buffer,
    /// decoding each line straight from a slice, and compacts the buffer
    /// **once** at the end — either clearing it while retaining capacity (the
    /// common case, when the read ended on a newline so everything was
    /// consumed) or dropping just the consumed prefix and keeping the partial
    /// tail. The previous version rebuilt `data = Data(data[after(newline)...])`
    /// on *every* line, which is O(remaining) per line and a fresh allocation
    /// each iteration; under a chatty engine (several `info` lines per read,
    /// times N instances) that was the dominant `_platform_memmove` cost. The
    /// cursor scans and copies each line exactly once and reallocates at most
    /// once per chunk — usually zero, reusing the retained capacity.
    func appendAndExtractLines(_ chunk: Data) -> [String] {
        data.append(chunk)
        var lines: [String] = []
        lines.reserveCapacity(4)   // a chunk is usually a few lines; avoid regrowth
        var lineStart = data.startIndex
        while let newline = data[lineStart...].firstIndex(of: 0x0A) {
            var line = String(decoding: data[lineStart..<newline], as: UTF8.self)
            if line.hasSuffix("\r") { line.removeLast() }
            lines.append(line)
            lineStart = data.index(after: newline)
        }
        if lineStart >= data.endIndex {
            data.removeAll(keepingCapacity: true)          // fully consumed
        } else if lineStart > data.startIndex {
            data.removeSubrange(data.startIndex..<lineStart)  // keep partial tail
        }
        return lines
    }

    /// Return any buffered unterminated tail (and clear it), or `nil`.
    func flush() -> String? {
        guard !data.isEmpty else { return nil }
        let tail = String(decoding: data, as: UTF8.self)
        data.removeAll(keepingCapacity: true)
        return tail
    }
}

/// The engine's answer to `go`.
enum UCIBestMove: Equatable, Sendable {
    /// A move in UCI LAN notation (e.g. `"e2e4"`, `"e7e8q"`), to be
    /// resolved against the current legal-move list.
    case move(String)
    /// The engine reported no move (`bestmove (none)` or `bestmove
    /// 0000`) — e.g. a terminal position.
    case null
}

/// Errors surfaced by `UCIArbiter`.
enum UCIArbiterError: LocalizedError, Equatable {
    /// A method was called before `launch()`, or after `shutdown()`.
    case notLaunched
    /// The engine process could not be started.
    case launchFailed(String)
    /// The engine closed its stdout (crashed / exited) mid-conversation.
    case engineTerminated
    /// The engine did not answer within the configured timeout.
    case timeout
    /// The engine emitted something that violates our expectations.
    case protocolViolation(String)

    var errorDescription: String? {
        switch self {
        case .notLaunched:
            return "UCI engine has not been launched (or was shut down)"
        case .launchFailed(let why):
            return "Failed to launch UCI engine: \(why)"
        case .engineTerminated:
            return "UCI engine terminated unexpectedly"
        case .timeout:
            return "UCI engine did not respond within the timeout"
        case .protocolViolation(let what):
            return "UCI protocol violation: \(what)"
        }
    }
}

/// Pure, side-effect-free UCI string construction and parsing. Split
/// out from the actor so the protocol grammar is unit-testable without
/// a subprocess.
enum UCIProtocol {

    /// Build a `position` command from a start position and move list.
    /// `startFEN == nil` uses `startpos`. An empty `moves` omits the
    /// `moves` clause entirely.
    static func positionCommand(startFEN: String?, moves: [String]) -> String {
        var parts: [String] = ["position"]
        if let fen = startFEN {
            parts.append("fen")
            parts.append(fen)
        } else {
            parts.append("startpos")
        }
        if !moves.isEmpty {
            parts.append("moves")
            parts.append(contentsOf: moves)
        }
        return parts.joined(separator: " ")
    }

    /// Build a `go` command from a limit string (`"nodes 1"`,
    /// `"depth 6"`, …). An empty/blank limit yields a bare `go`.
    static func goCommand(_ limit: String) -> String {
        let trimmed = limit.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? "go" : "go \(trimmed)"
    }

    /// Build a `setoption name <name> value <value>` command.
    static func setOptionCommand(name: String, value: String) -> String {
        "setoption name \(name) value \(value)"
    }

    /// Parse a `bestmove` line. Returns `nil` for any line that is not
    /// a `bestmove` line (so callers can skip `info`/`id`/etc.). A
    /// `bestmove` line with no move token, or with `(none)` / `0000`,
    /// parses to `.null`. A trailing `ponder …` clause is ignored.
    static func parseBestMove(_ line: String) -> UCIBestMove? {
        // Called on *every* line the engine emits (the move loop filters
        // with `parseBestMove(...) != nil`), so the common path is rejecting
        // an `info …` line. Split into `Substring`s and compare those against
        // the literals — no per-token `String` is materialised on a rejected
        // line, and exactly one `String` is allocated (the move token) only
        // when the line is a real `bestmove`.
        let tokens = line.split(separator: " ", omittingEmptySubsequences: true)
        guard tokens.first == "bestmove" else { return nil }
        guard tokens.count >= 2 else { return .null }
        let moveToken = tokens[1]
        if moveToken == "(none)" || moveToken == "0000" {
            return .null
        }
        return .move(String(moveToken))
    }
}

/// Run `operation`, failing with `UCIArbiterError.timeout` if it does
/// not finish within `duration`. On timeout (or any throw) the losing
/// child task is cancelled — for `receiveLine()` that unblocks the
/// `AsyncStream` read cooperatively.
private func withUCITimeout<T: Sendable>(
    _ duration: Duration,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(for: duration)
            throw UCIArbiterError.timeout
        }
        do {
            guard let result = try await group.next() else {
                throw UCIArbiterError.timeout
            }
            group.cancelAll()
            return result
        } catch {
            group.cancelAll()
            throw error
        }
    }
}
