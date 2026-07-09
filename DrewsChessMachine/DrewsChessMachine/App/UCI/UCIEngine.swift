import Darwin
import Foundation

/// Bridges DrewsChessMachine to UCI-speaking GUIs / tournament runners
/// (cutechess, c-chess-cli, ChessGUI, etc.).
///
/// Invoked from `DrewsChessMachineApp.init`'s pre-flight branch
/// before any SwiftUI / `WindowGroup` setup runs, so the process
/// behaves as a pure stdin/stdout engine — no window, no menu bar,
/// no `AutoResumeController` countdown sheet.
///
/// Strength comes from a single forward pass per `go`: encode the
/// current position, run `DirectMoveEvaluationSource.evaluate`,
/// `MoveSampler.sampleMove` against the legal-move list,
/// `bestmove <lan>`. There is no search, no iterative deepening, no
/// time-control handling — `go`'s time fields are deliberately
/// ignored. A single eval takes single-digit ms, well below any
/// practical cutechess time control.
///
/// The default sampling schedule is `.arena` (`startTau 2.0 → 0.2`
/// over 45 game-total plies). UCI `Temperature` option lets the
/// caller override with a flat tau for the entire game; values are
/// passed as integers × 100 (UCI `spin` doesn't allow floats), so
/// `Temperature=100` means tau=1.0, `Temperature=10` means tau=0.1
/// (near-argmax), etc. Sentinel value `0` re-enables the default
/// `.arena` schedule.
enum UCIEngine {

    /// UCI temperature option sentinel meaning "use the default
    /// `.arena` schedule" (no flat-tau override).
    private static let temperatureUseSchedule: Int = 0
    /// UCI Temperature default — `0` = schedule on. Surfaced in `uci`
    /// handshake so a GUI's "reset to default" button restores the
    /// schedule rather than forcing a specific tau.
    private static let temperatureDefault: Int = temperatureUseSchedule
    /// Minimum / maximum for the Temperature spin option. `1` = tau
    /// 0.01 (effectively argmax), `1000` = tau 10.0 (very flat).
    private static let temperatureMin: Int = 0
    private static let temperatureMax: Int = 1000

    /// Pre-flight entry point. Loads weights, runs the protocol loop,
    /// never returns. Process exits via `Darwin.exit(0)` on `quit`
    /// or stdin EOF.
    static func runAndExit(modelPath: String?) -> Never {
        // SessionLogger is the file-based log already used by every
        // other launch — repurposed here to capture model load, per-
        // move evaluation, and protocol traffic without polluting
        // stdout (which belongs entirely to the UCI protocol).
        SessionLogger.shared.start()
        let dirtyMarker = BuildInfo.gitDirty ? "*" : ""
        SessionLogger.shared.log(
            "[UCI] launched build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirtyMarker) branch=\(BuildInfo.gitBranch)"
        )
        if let path = SessionLogger.shared.activeLogPath {
            SessionLogger.shared.log("[UCI] session log: \(path)")
        }

        var session = Session()
        if let path = modelPath {
            // Explicit `--model`: load eagerly. An explicit path that fails to
            // resolve is a hard error — the user named a specific file.
            do {
                let loaded = try syncWait { try await UCIModelLoader.resolveAndLoad(explicitPath: path) }
                session.apply(loaded)
                SessionLogger.shared.log(
                    "[UCI] model loaded (--model): id=\(loaded.modelID) source=\(loaded.sourceLabel) params=\(loaded.parameterCount) arch=\(loaded.archSummary)"
                )
            } catch {
                let line = "DrewsChessMachine UCI: failed to load --model: \(error)\n"
                FileHandle.standardError.write(Data(line.utf8))
                SessionLogger.shared.log("[UCI] --model load failed: \(error)")
                Darwin.exit(20)
            }
        } else {
            // Deferred load: launched with no `--model` (the GUI case — cutechess
            // passes no CLI args). Come up with no model so the `uci` handshake
            // returns immediately; the model is loaded when a `setoption name
            // Model` arrives, or lazily on the first `go` (latest session). This
            // avoids dying at startup before the GUI can supply the model.
            SessionLogger.shared.log("[UCI] no --model; model load deferred to 'setoption name Model' or first 'go'")
        }
        runLoop(session: &session)
        Darwin.exit(0)
    }

    // MARK: - Session state

    /// Per-process UCI state: the current engine (resets on
    /// `ucinewgame` / `position`), the user-tunable `Temperature`
    /// option, and a label used in the `id name` line.
    private struct Session {
        /// The live move source. `nil` until a model is loaded — deferred so a
        /// GUI-launched engine (no CLI args) comes up immediately and loads on
        /// `setoption name Model` or lazily on the first `go`. Mutable so the
        /// model can also be hot-swapped mid-session.
        var source: MoveEvaluationSource? = nil
        var modelLabel: String = "(none loaded)"
        /// Trainable parameter count + one-line arch description of the loaded
        /// model, surfaced to the GUI as `info string` lines.
        var paramCount: Int = 0
        var archSummary: String = ""
        /// Absolute path the current model resolved to (`""` = none). Lets
        /// `setoption Model` skip a redundant reload when a GUI re-sends the same
        /// value each game.
        var modelPath: String = ""
        /// Modification time of `modelPath` at load. The idempotent-reload skip
        /// compares BOTH path and mtime so a live `-replay-latest` file the
        /// trainer overwrote in place (same path, new bytes) is picked up on the
        /// next `setoption Model` rather than silently kept stale. `nil` = unknown
        /// (stat failed) → never treated as a match, so we reload to be safe.
        var modelMtime: Date? = nil
        /// Tracks the position the most recent `position` command
        /// established, plus all moves applied on top. Refreshed
        /// from scratch on every `position` command (UCI senders pass
        /// the full move list every time).
        var engine: ChessGameEngine = ChessGameEngine(state: .starting)
        /// Current Temperature option value (0 = use default
        /// `.arena` schedule; otherwise flat tau = value / 100).
        var temperatureSpin: Int = temperatureDefault

        /// Install a freshly-loaded model, replacing any current one.
        mutating func apply(_ loaded: UCIModelLoader.Loaded) {
            source = DirectMoveEvaluationSource(network: loaded.network)
            modelLabel = loaded.modelID
            paramCount = loaded.parameterCount
            archSummary = loaded.archSummary
            modelPath = loaded.resolvedPath
            modelMtime = UCIEngine.fileMtime(loaded.resolvedPath)
        }

        /// Resolve the current `Temperature` option into a sampling
        /// schedule. `0` = the default `.arena` schedule; any other
        /// value clamps to the valid spin range and produces a
        /// flat-tau schedule.
        var schedule: SamplingSchedule {
            if temperatureSpin == temperatureUseSchedule {
                return .arena
            }
            let clamped = max(temperatureMin + 1, min(temperatureMax, temperatureSpin))
            let tau = Float(clamped) / 100.0
            return SamplingSchedule(startTau: tau, decayPerPly: 0, floorTau: tau)
        }
    }

    // MARK: - Protocol loop

    private static func runLoop(session: inout Session) {
        while let raw = readLine(strippingNewline: true) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            SessionLogger.shared.log("[UCI<-GUI] \(line)")
            let tokens = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard let command = tokens.first else { continue }
            switch command {
            case "uci":
                handleUci(session: session)
            case "isready":
                respond("readyok")
            case "ucinewgame":
                session.engine = ChessGameEngine(state: .starting)
            case "position":
                handlePosition(tokens: Array(tokens.dropFirst()), session: &session)
            case "go":
                handleGo(session: &session)
            case "stop":
                // We never start a long-running search, so there is
                // nothing to stop — the next `go` will produce a
                // move on demand.
                break
            case "ponderhit":
                // No pondering. UCI permits engines to ignore this.
                break
            case "setoption":
                handleSetOption(line: line, tokens: Array(tokens.dropFirst()), session: &session)
            case "quit":
                SessionLogger.shared.log("[UCI] quit received; exiting")
                return
            default:
                // Unknown command — UCI spec says ignore (don't reply).
                SessionLogger.shared.log("[UCI] ignoring unknown command: \(command)")
            }
        }
        SessionLogger.shared.log("[UCI] stdin EOF; exiting")
    }

    // MARK: - Handlers

    private static func handleUci(session: Session) {
        let dirty = BuildInfo.gitDirty ? "*" : ""
        respond("id name DrewsChessMachine \(BuildInfo.buildNumber) (\(BuildInfo.gitHash)\(dirty)) model=\(session.modelLabel)")
        respond("id author Andrew Benson")
        // Model: a filesystem path (or bare name/id UCIModelLoader can resolve) to
        // the .safetensors/.dcmmodel to play. This is how a GUI (cutechess) selects
        // the model, since it passes no CLI args — set it in the engine config.
        respond("option name Model type string default \(session.modelPath)")
        respond("option name Temperature type spin default \(temperatureDefault) min \(temperatureMin) max \(temperatureMax)")
        respond("uciok")
        // Informational lines (cutechess logs `info string`): engine build + the
        // model currently loaded, with its parameter count and architecture.
        respond("info string engine=DrewsChessMachine build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(dirty) branch=\(BuildInfo.gitBranch)")
        if session.source != nil {
            respond("info string model=\(session.modelLabel) params=\(session.paramCount) arch=\(session.archSummary)")
        } else {
            respond("info string no model loaded yet — set 'setoption name Model value <path>', or the first 'go' loads the latest session")
        }
    }

    private static func handlePosition(tokens: [String], session: inout Session) {
        // Acceptable shapes:
        //   position startpos
        //   position startpos moves <m1> <m2> ...
        //   position fen <f1> <f2> <f3> <f4> <f5> <f6>
        //   position fen <...> moves <m1> <m2> ...
        guard let first = tokens.first else { return }
        var idx = 0
        let baseState: GameState
        switch first {
        case "startpos":
            baseState = .starting
            idx = 1
        case "fen":
            // FEN has exactly 6 space-separated fields.
            guard tokens.count >= 7 else {
                SessionLogger.shared.log("[UCI] position fen: not enough tokens for a 6-field FEN")
                return
            }
            let fenFields = tokens[1...6].joined(separator: " ")
            do {
                baseState = try FENParser.parse(fenFields)
            } catch {
                SessionLogger.shared.log("[UCI] position fen parse failed: \(error)")
                return
            }
            idx = 7
        default:
            SessionLogger.shared.log("[UCI] position: unexpected token '\(first)'")
            return
        }

        let engine = ChessGameEngine(state: baseState)
        session.engine = engine

        guard idx < tokens.count else { return }
        guard tokens[idx] == "moves" else {
            SessionLogger.shared.log("[UCI] position: expected 'moves', got '\(tokens[idx])'")
            return
        }
        for moveToken in tokens[(idx + 1)...] {
            guard let move = ChessMove.parseUCI(moveToken, legal: session.engine.currentLegalMoves) else {
                SessionLogger.shared.log("[UCI] position moves: illegal or unparseable move '\(moveToken)' — aborting position update")
                return
            }
            do {
                try session.engine.applyMoveAndAdvance(move)
            } catch {
                SessionLogger.shared.log("[UCI] position moves: applyMoveAndAdvance failed for '\(moveToken)': \(error)")
                return
            }
        }
    }

    /// Encode the position `engine` is about to move from, threading the
    /// engine's real ply history so a history encoding (full10ply200) sees the
    /// temporal/repetition planes it was trained on rather than empty frames —
    /// the train/infer skew that otherwise weakens play from an external GUI.
    ///
    /// `internal` (not `private`) purely so a unit test can assert the history
    /// is threaded; the UCI command loop itself is stdin/stdout-driven and so
    /// not directly testable.
    static func encodedBoardForGo(engine: ChessGameEngine, encoding: InputEncoding) -> [Float] {
        var buffer = [Float](repeating: 0, count: BoardEncoder.tensorLength(for: encoding))
        buffer.withUnsafeMutableBufferPointer { buf in
            BoardEncoder.encode(engine.state, history: engine.recentStates, into: buf, encoding: encoding)
        }
        return buffer
    }

    private static func handleGo(session: inout Session) {
        let engine = session.engine
        // Game already over — emit a UCI null move so the GUI gets a
        // deterministic reply instead of hanging on an empty stdout.
        let legal = engine.currentLegalMoves
        guard !legal.isEmpty else {
            respond("bestmove 0000")
            SessionLogger.shared.log("[UCI] go: no legal moves (result=\(String(describing: engine.result))) — sent bestmove 0000")
            return
        }

        // Deferred/lazy load: no model was set (no `--model`, no `setoption name
        // Model`). Load the latest session's weights now — first `go` only;
        // `apply` persists it so every later `go` (and later game) reuses it. This
        // is what keeps a GUI-launched engine playable with no CLI args.
        if session.source == nil {
            do {
                let loaded = try syncWait { try await UCIModelLoader.resolveAndLoad(explicitPath: nil) }
                session.apply(loaded)
                SessionLogger.shared.log("[UCI] go: lazy-loaded default model \(loaded.modelID) params=\(loaded.parameterCount)")
                respond("info string loaded model=\(loaded.modelID) params=\(loaded.parameterCount) arch=\(loaded.archSummary)")
            } catch {
                SessionLogger.shared.log("[UCI] go: no model set and default load failed: \(error)")
                respond("info string no model loaded — set 'setoption name Model value <path>' (default load failed: \(error))")
                respond("bestmove 0000")
                return
            }
        }
        guard let source = session.source else {
            respond("bestmove 0000")
            return
        }

        let state = engine.state
        let encoding = source.inputEncoding
        // Encode via the shared seam, which threads `engine.recentStates` as
        // history. Returns a `let`, so the @Sendable closure passed to syncWait
        // below captures an immutable value (Swift 6 strict concurrency rejects
        // captures of `var`s in concurrently-executing closures).
        let encoded = encodedBoardForGo(engine: engine, encoding: encoding)

        let policySize = ChessNetwork.policySize
        let policyPtr = UnsafeMutablePointer<Float>.allocate(capacity: policySize)
        policyPtr.initialize(repeating: 0, count: policySize)
        defer {
            policyPtr.deinitialize(count: policySize)
            policyPtr.deallocate()
        }

        // `source` (unwrapped above) is a Sendable class; it's the only piece of
        // `session` that crosses the @Sendable boundary below (`Session` itself
        // holds a non-Sendable `ChessGameEngine`, so we never capture it).
        let value: Float
        do {
            let dest = PolicyDestination(UnsafeMutableBufferPointer(start: policyPtr, count: policySize))
            value = try syncWait { try await source.evaluate(encodedBoard: encoded, intoPolicy: dest) }
        } catch {
            // Inference failed — Metal crash, OOM, whatever. UCI has
            // no error response shape, so surface the failure as the
            // null move `0000` (the UCI sentinel for "no legal move")
            // alongside an `info string` so cutechess's engine log
            // records what happened. Returning silently would hang the
            // GUI waiting on stdout; emitting a random legal move
            // would silently hide the bug.
            SessionLogger.shared.log("[UCI] go: forward pass failed: \(error)")
            let stderrLine = "info string forward pass failed: \(error)\n"
            FileHandle.standardError.write(Data(stderrLine.utf8))
            respond("info string forward pass failed: \(error)")
            respond("bestmove 0000")
            return
        }

        let result = sampleResult(
            policyPtr: policyPtr,
            policySize: policySize,
            legal: legal,
            currentPlayer: state.currentPlayer,
            ply: engine.moveHistory.count,
            schedule: session.schedule
        )

        // UCI `info` line gives cutechess engine logs a per-ply
        // snapshot of what the engine "thought" — value scalar
        // (p_win − p_loss), the sampled move's probability, and the
        // tau actually used. score cp is the value × 100 (UCI's
        // centipawn convention) so the bar in cutechess moves.
        let tau = session.schedule.tau(forPly: engine.moveHistory.count)
        let scoreCp = Int((value * 100).rounded())
        respond("info depth 1 score cp \(scoreCp) string tau=\(String(format: "%.3f", tau)) p=\(String(format: "%.3f", result.chosenProbability)) value=\(String(format: "%+.3f", value))")
        respond("bestmove \(result.move.uci)")
        SessionLogger.shared.log("[UCI] go: chose \(result.move.uci) p=\(result.chosenProbability) value=\(value) tau=\(tau) ply=\(engine.moveHistory.count)")
    }

    private static func sampleResult(
        policyPtr: UnsafeMutablePointer<Float>,
        policySize: Int,
        legal: [ChessMove],
        currentPlayer: PieceColor,
        ply: Int,
        schedule: SamplingSchedule
    ) -> MoveSampler.Result {
        let logits = UnsafeBufferPointer(start: policyPtr, count: policySize)
        var probsScratch = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        var etaScratch = [Float](repeating: 0, count: MoveSampler.scratchCapacity)
        return probsScratch.withUnsafeMutableBufferPointer { probs in
            etaScratch.withUnsafeMutableBufferPointer { eta in
                MoveSampler.sampleMove(
                    logits: logits,
                    legalMoves: legal,
                    currentPlayer: currentPlayer,
                    ply: ply,
                    schedule: schedule,
                    probsScratch: probs,
                    etaScratch: eta
                )
            }
        }
    }

    /// The verbatim value of a `setoption … value <x>` line: everything after
    /// the first ` value ` delimiter, with all internal spacing preserved. Per
    /// the UCI protocol the value is free-form text that may contain runs of
    /// spaces, so it is taken as the remainder of the raw line rather than
    /// reassembled from space-split tokens (which would collapse consecutive
    /// spaces). Returns "" when the line ends in `… value` with no value text.
    /// Internal (not private) so it is reachable from the test target.
    static func setOptionValue(fromLine line: String) -> String {
        guard let valueRange = line.range(of: " value ") else { return "" }
        return String(line[valueRange.upperBound...])
    }

    private static func handleSetOption(line: String, tokens: [String], session: inout Session) {
        // UCI shape: `setoption name <Name with possibly spaces> value <value>`.
        // The value is free-form text and, per the protocol, is the remainder of
        // the line after the `value` keyword taken verbatim — it may contain runs
        // of spaces. It must therefore NOT be reassembled from the space-split
        // tokens, which collapses consecutive spaces (e.g. a path like
        // `/a/b  c`). Only the name is derived from tokens.
        guard tokens.first == "name" else {
            SessionLogger.shared.log("[UCI] setoption: missing 'name'")
            return
        }
        guard let valueIdx = tokens.firstIndex(of: "value") else {
            SessionLogger.shared.log("[UCI] setoption: missing 'value'")
            return
        }
        let name = tokens[1..<valueIdx].joined(separator: " ")
        let valueString = Self.setOptionValue(fromLine: line)
        switch name.lowercased() {
        case "temperature":
            guard let v = Int(valueString) else {
                SessionLogger.shared.log("[UCI] setoption Temperature: non-integer value '\(valueString)'")
                return
            }
            let clamped = max(temperatureMin, min(temperatureMax, v))
            session.temperatureSpin = clamped
            SessionLogger.shared.log("[UCI] setoption Temperature=\(clamped)\(clamped == temperatureUseSchedule ? " (use schedule)" : "")")
        case "model":
            handleSetModel(valueString: valueString, session: &session)
        default:
            SessionLogger.shared.log("[UCI] setoption: unknown option '\(name)'")
        }
    }

    /// Load and hot-swap the model named by `setoption name Model value <…>`.
    ///
    /// Idempotent: the value is resolved to an absolute path WITHOUT loading, and
    /// if it matches the model already in the session the (multi-second) reload is
    /// skipped. That makes it safe for a GUI to re-send the option on every game —
    /// cutechess may or may not re-run the `uci`/`setoption` handshake per game,
    /// and we don't want to pay a needless network rebuild + weight load each time.
    /// A resolution or load failure keeps the current model and reports via
    /// `info string` rather than tearing down the session.
    private static func handleSetModel(valueString: String, session: inout Session) {
        let trimmed = valueString.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else {
            respond("info string setoption Model: empty value ignored")
            return
        }
        let resolvedPath: String
        do {
            resolvedPath = try UCIModelLoader.resolveModelURL(explicitPath: trimmed).url.path
        } catch {
            SessionLogger.shared.log("[UCI] setoption Model: cannot resolve '\(trimmed)': \(error)")
            respond("info string model load FAILED: \(error) — keeping \(session.modelLabel)")
            return
        }
        // Skip the reload only when the SAME file is BYTE-identical to what we
        // already hold — same path AND same mtime. A live `-replay-latest`
        // checkpoint keeps its path while the trainer rewrites it in place, so a
        // path-only check would serve stale weights across games; the mtime guard
        // reloads when the bytes changed. A failed stat (mtime nil) reloads.
        let currentMtime = fileMtime(resolvedPath)
        if resolvedPath == session.modelPath,
           let loadedMtime = session.modelMtime, let nowMtime = currentMtime,
           loadedMtime == nowMtime {
            SessionLogger.shared.log("[UCI] setoption Model: '\(trimmed)' already loaded (unchanged) — no reload")
            return
        }
        do {
            let loaded = try syncWait { try await UCIModelLoader.resolveAndLoad(explicitPath: trimmed) }
            session.apply(loaded)
            SessionLogger.shared.log("[UCI] setoption Model -> \(loaded.modelID) (\(loaded.resolvedPath))")
            respond("info string loaded model=\(loaded.modelID) params=\(loaded.parameterCount) arch=\(loaded.archSummary)")
        } catch {
            SessionLogger.shared.log("[UCI] setoption Model FAILED for '\(trimmed)': \(error)")
            respond("info string model load FAILED: \(error) — keeping \(session.modelLabel)")
        }
    }

    // MARK: - I/O helpers

    /// Modification time of the file at `path`, or `nil` if it cannot be stat'd.
    /// Used to detect an in-place overwrite of a live checkpoint (same path, new
    /// bytes). A `nil` return (stat failure) is treated as "changed" by callers so
    /// a redundant reload is preferred over serving possibly-stale weights.
    static func fileMtime(_ path: String) -> Date? {
        do {
            let attrs = try FileManager.default.attributesOfItem(atPath: path)
            return attrs[.modificationDate] as? Date
        } catch {
            return nil
        }
    }

    /// Write one UCI response line to stdout and flush. UCI senders
    /// are line-buffered readers; an un-flushed line can stall the
    /// GUI indefinitely waiting for our `uciok` / `bestmove`.
    private static func respond(_ line: String) {
        let data = Data("\(line)\n".utf8)
        FileHandle.standardOutput.write(data)
        SessionLogger.shared.log("[UCI->GUI] \(line)")
    }

    /// Bridge an async throwing call to the synchronous protocol
    /// loop. Same shape as `ChessMPSNetwork.calibrateBNRunningStats`
    /// uses for its sync init wrapper: a detached Task at user-
    /// initiated priority writes the result into a holder and signals
    /// a semaphore; the calling thread waits, then reads the holder.
    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = SyncWaitResultBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do {
                box.success = try await work()
            } catch {
                box.failure = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure {
            throw error
        }
        guard let success = box.success else {
            preconditionFailure("UCIEngine.syncWait: SyncWaitResultBox carried neither success nor failure")
        }
        return success
    }
}

/// Result/error holder for `UCIEngine.syncWait`. Lives at file scope
/// because generic classes can't be nested inside generic functions
/// in Swift. `@unchecked Sendable` because the Task writes the box
/// exactly once before the semaphore signal and the calling thread
/// reads it exactly once after the wait — the explicit happens-
/// before edge of the semaphore covers what the type system can't
/// see.
private final class SyncWaitResultBox<T>: @unchecked Sendable {
    var success: T?
    var failure: Error?
}
