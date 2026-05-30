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

        let loaded: UCIModelLoader.Loaded
        do {
            // Bridge async → sync via a semaphore so the rest of the
            // loop can run as plain synchronous stdin-driven code.
            // Model load runs exactly once at startup.
            loaded = try syncWait { try await UCIModelLoader.resolveAndLoad(explicitPath: modelPath) }
        } catch {
            let line = "DrewsChessMachine UCI: failed to load model: \(error)\n"
            FileHandle.standardError.write(Data(line.utf8))
            SessionLogger.shared.log("[UCI] model load failed: \(error)")
            Darwin.exit(20)
        }
        SessionLogger.shared.log(
            "[UCI] model loaded: id=\(loaded.modelID) source=\(loaded.sourceLabel)"
        )

        let source = DirectMoveEvaluationSource(network: loaded.network)
        var session = Session(
            source: source,
            modelLabel: loaded.modelID
        )
        runLoop(session: &session)
        Darwin.exit(0)
    }

    // MARK: - Session state

    /// Per-process UCI state: the current engine (resets on
    /// `ucinewgame` / `position`), the user-tunable `Temperature`
    /// option, and a label used in the `id name` line.
    private struct Session {
        let source: MoveEvaluationSource
        let modelLabel: String
        /// Tracks the position the most recent `position` command
        /// established, plus all moves applied on top. Refreshed
        /// from scratch on every `position` command (UCI senders pass
        /// the full move list every time).
        var engine: ChessGameEngine = ChessGameEngine(state: .starting)
        /// Current Temperature option value (0 = use default
        /// `.arena` schedule; otherwise flat tau = value / 100).
        var temperatureSpin: Int = temperatureDefault

        init(source: MoveEvaluationSource, modelLabel: String) {
            self.source = source
            self.modelLabel = modelLabel
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
                handleGo(session: session)
            case "stop":
                // We never start a long-running search, so there is
                // nothing to stop — the next `go` will produce a
                // move on demand.
                break
            case "ponderhit":
                // No pondering. UCI permits engines to ignore this.
                break
            case "setoption":
                handleSetOption(tokens: Array(tokens.dropFirst()), session: &session)
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
        respond("option name Temperature type spin default \(temperatureDefault) min \(temperatureMin) max \(temperatureMax)")
        respond("uciok")
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

    private static func handleGo(session: Session) {
        let engine = session.engine
        // Game already over — emit a UCI null move so the GUI gets a
        // deterministic reply instead of hanging on an empty stdout.
        let legal = engine.currentLegalMoves
        guard !legal.isEmpty else {
            respond("bestmove 0000")
            SessionLogger.shared.log("[UCI] go: no legal moves (result=\(String(describing: engine.result))) — sent bestmove 0000")
            return
        }

        let state = engine.state
        var encodedBuffer = [Float](repeating: 0, count: BoardEncoder.tensorLength)
        encodedBuffer.withUnsafeMutableBufferPointer { buf in
            BoardEncoder.encode(state, into: buf)
        }
        // Rebind to a `let` so the @Sendable closure passed to
        // syncWait below captures an immutable value rather than the
        // mutable `var` (Swift 6 strict concurrency rejects captures
        // of vars in concurrently-executing closures).
        let encoded = encodedBuffer

        let policySize = ChessNetwork.policySize
        let policyPtr = UnsafeMutablePointer<Float>.allocate(capacity: policySize)
        policyPtr.initialize(repeating: 0, count: policySize)
        defer {
            policyPtr.deinitialize(count: policySize)
            policyPtr.deallocate()
        }

        // Pull the Sendable pieces out of `session` before crossing
        // the @Sendable boundary: `MoveEvaluationSource` is a class
        // (Sendable) but `Session` itself contains a `ChessGameEngine`
        // which is not.
        let source = session.source
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

    private static func handleSetOption(tokens: [String], session: inout Session) {
        // UCI shape: `setoption name <Name with possibly spaces> value <value tokens...>`
        guard tokens.first == "name" else {
            SessionLogger.shared.log("[UCI] setoption: missing 'name'")
            return
        }
        guard let valueIdx = tokens.firstIndex(of: "value") else {
            SessionLogger.shared.log("[UCI] setoption: missing 'value'")
            return
        }
        let nameTokens = tokens[1..<valueIdx]
        let valueTokens = tokens[(valueIdx + 1)...]
        let name = nameTokens.joined(separator: " ")
        let valueString = valueTokens.joined(separator: " ")
        switch name.lowercased() {
        case "temperature":
            guard let v = Int(valueString) else {
                SessionLogger.shared.log("[UCI] setoption Temperature: non-integer value '\(valueString)'")
                return
            }
            let clamped = max(temperatureMin, min(temperatureMax, v))
            session.temperatureSpin = clamped
            SessionLogger.shared.log("[UCI] setoption Temperature=\(clamped)\(clamped == temperatureUseSchedule ? " (use schedule)" : "")")
        default:
            SessionLogger.shared.log("[UCI] setoption: unknown option '\(name)'")
        }
    }

    // MARK: - I/O helpers

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
