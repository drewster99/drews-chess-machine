import Foundation
import Dispatch
import os

struct PGNImportConfig: Sendable {
    var inputPath: String
    var corpusName: String?
    var minRating: Int?
    var maxGames: Int?
    var minPlies: Int
    var timeControlClasses: [String]?   // e.g. ["blitz","rapid"]; nil = all
    var shardSoftLimitMB: Int = 64
    /// Stop once the corpus body reaches this many bytes (whole-game boundary).
    var maxStorageBytes: Int? = nil
    /// Worker threads for parsing/replay. nil ⇒ activeProcessorCount − 2.
    var importThreads: Int? = nil
    /// Abort the whole import on the first game that fails to parse/replay
    /// (default). When false, such games are counted and skipped.
    var failOnError: Bool = true
    /// Where the corpus directory is created. nil ⇒ the shared `Corpora/` store.
    var outputParentDirectory: URL? = nil
}

enum PGNImportError: LocalizedError {
    case openFailed(String)
    case gameFailed(String)
    var errorDescription: String? {
        switch self {
        case .openFailed(let s): return "PGN import open failed: \(s)"
        case .gameFailed(let s): return "PGN import failed on \(s)"
        }
    }
}

/// Streaming PGN → game-corpus importer. Decompresses `.pgn.zst` via the
/// `zstd` CLI (or reads a plain `.pgn` directly), parses games one at a time,
/// converts each SAN move list to `ChessMove`s by matching against the
/// engine's legal moves, applies rating / time-control / length filters, and
/// appends the survivors to a fresh corpus. Standard-start games only (FEN
/// setups, variants are skipped; an unparseable move hard-fails by default).
///
/// The CPU-bound per-game replay runs across a worker pool while a single
/// serial writer appends results **in original file order**, so the output is
/// deterministic and independent of thread count.
enum PGNImporter {

    // MARK: - CLI entry

    static func runImportAndExit(config: PGNImportConfig) -> Never {
        SessionLogger.shared.start()
        SessionLogger.shared.log("[PGN] importing \(config.inputPath)")
        do {
            let summary = try runImport(config: config)
            let line = "[PGN] done: imported=\(summary.imported) skipped=\(summary.skipped) parseErrors=\(summary.parseErrors) → corpus \(summary.corpusID)"
            print(line)
            SessionLogger.shared.log(line)
        } catch {
            FileHandle.standardError.write(Data("pgn-import: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[PGN] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(34)
        }
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    struct Summary {
        var imported = 0
        var skipped = 0
        var parseErrors = 0
        var corpusID = ""
    }

    /// One game's worker output: a ready-to-write framed buffer, a filtered
    /// skip, or a hard failure with a diagnostic reason.
    enum ImportSlot: Sendable {
        case imported(frame: Data, plyCount: Int)
        case skipped
        case failed(reason: String)
    }

    // MARK: - Import

    static func runImport(config: PGNImportConfig) throws -> Summary {
        let expanded = (config.inputPath as NSString).expandingTildeInPath
        let inputURL = URL(fileURLWithPath: expanded)

        let corpus = try GameCorpus.create(
            name: config.corpusName ?? inputURL.lastPathComponent,
            comment: "Imported from \(inputURL.lastPathComponent) by build \(BuildInfo.buildNumber) (\(BuildInfo.gitHash))",
            shardSoftLimitBytes: max(1, config.shardSoftLimitMB) * 1024 * 1024,
            parentDirectory: config.outputParentDirectory
        )
        try corpus.beginSource(
            kind: "pgnImport",
            inputFilename: inputURL.lastPathComponent,
            inputURL: inputURL.absoluteString,
            minRating: config.minRating,
            timeControls: config.timeControlClasses,
            maxGames: config.maxGames
        )

        let workerCount = max(1, config.importThreads ?? (ProcessInfo.processInfo.activeProcessorCount - 2))
        let workers = OperationQueue()
        workers.maxConcurrentOperationCount = workerCount
        workers.qualityOfService = .userInitiated
        let writerQueue = DispatchQueue(label: "pgn.import.writer")
        // Bound in-flight items (and therefore the reorder buffer) so the reader
        // can't outrun the workers; generous slack absorbs head-of-line stalls.
        let inFlight = DispatchSemaphore(value: workerCount * 4)
        let group = DispatchGroup()
        let stop = OSAllocatedUnfairLock(initialState: false)
        let box = ImportWriterBox(corpus: corpus, config: config,
                                  inFlight: inFlight, group: group, stop: stop)

        SessionLogger.shared.log(
            "[PGN] workers=\(workerCount) failOnError=\(config.failOnError) " +
            "maxGames=\(config.maxGames.map(String.init) ?? "∞") " +
            "maxStorage=\(config.maxStorageBytes.map(String.init) ?? "∞")")

        var seq = 0
        try streamGames(at: expanded,
                        shouldContinue: { !stop.withLock { $0 } }) { tags, movetext in
            inFlight.wait()
            if stop.withLock({ $0 }) {
                inFlight.signal()
                return
            }
            let mySeq = seq
            seq += 1
            group.enter()
            workers.addOperation {
                let slot = processOneGame(tags: tags, movetext: movetext, config: config)
                writerQueue.async {
                    box.submit(seq: mySeq, slot: slot)
                }
            }
        }
        group.wait()

        // Seal the (partial, on error) corpus so what's written stays valid.
        try corpus.finishSource()

        if let err = box.firstError {
            let msg = "game #\(err.seq + 1): \(err.reason)"
            SessionLogger.shared.log("[PGN] hard-fail — \(msg)")
            throw PGNImportError.gameFailed(msg)
        }
        return box.summary
    }

    // MARK: - Per-game worker (pure, runs off the writer thread)

    private enum BuildResult {
        case imported(GameRecord)
        case skipped
        case failed(String)
    }

    /// Build a game and encode its framed buffer — the parallelizable hot path.
    static func processOneGame(tags: [String: String], movetext: String, config: PGNImportConfig) -> ImportSlot {
        switch buildGame(tags: tags, movetext: movetext, config: config) {
        case .imported(let game):
            return .imported(frame: GameCorpusShardFormat.encodeFramedRecord(game),
                             plyCount: game.moves.count)
        case .skipped:
            return .skipped
        case .failed(let reason):
            return .failed(reason: reason)
        }
    }

    private static func buildGame(tags: [String: String], movetext: String, config: PGNImportConfig) -> BuildResult {
        // Standard start only.
        if tags["FEN"] != nil || tags["SetUp"] == "1" { return .skipped }
        if let variant = tags["Variant"], variant.lowercased() != "standard" { return .skipped }

        guard let result = tags["Result"], let outcome = outcome(forResult: result) else { return .skipped }

        if let minRating = config.minRating {
            let w = tags["WhiteElo"].flatMap { Int($0) }
            let b = tags["BlackElo"].flatMap { Int($0) }
            guard let wr = w, let br = b, wr >= minRating, br >= minRating else { return .skipped }
        }

        if let classes = config.timeControlClasses {
            let cls = timeControlClass(tags["TimeControl"] ?? "")
            guard classes.contains(cls) else { return .skipped }
        }

        let tokens = sanTokens(from: movetext)
        guard !tokens.isEmpty else { return .skipped }

        // Replay by legality only. The recorded game is ground truth, so a
        // claimable-but-unclaimed draw (threefold repetition, fifty-move rule)
        // must NOT stop replay the way ChessGameEngine's self-play
        // auto-termination would — humans legally play on past both. The
        // outcome comes from the PGN Result tag, not from end-state detection.
        // (Repetition bookkeeping isn't needed here: the corpus stores raw
        // moves and is re-encoded at training time.)
        //
        // Each SAN token is resolved against the *pseudo-legal* moves and only
        // the matched candidate is legality-checked — far cheaper than
        // generating and legality-filtering the entire legal move list per ply.
        var state = ChessGameEngine().state
        var moves: [ChessMove] = []
        moves.reserveCapacity(tokens.count)
        for token in tokens {
            guard let move = resolveLegalSANMove(token, state: state) else {
                return .failed("unresolved SAN '\(token)' at ply \(moves.count + 1)")
            }
            moves.append(move)
            state = MoveGenerator.applyMove(move, to: state)
        }

        guard moves.count >= config.minPlies else { return .skipped }
        return .imported(GameRecord(moves: moves, outcome: outcome, terminationReason: nil))
    }

    // MARK: - Streaming reader

    /// Stream the file, invoking `onGame(tags, movetext)` for each complete
    /// game. A plain `.pgn` is read **directly** from its own file handle; a
    /// `.zst` is decompressed by the `zstd` CLI and read from its stdout pipe.
    /// Stops early (and reaps the decompressor) as soon as `shouldContinue()`
    /// turns false.
    ///
    /// Plain files are read directly instead of piped through `cat`: shelling
    /// out just to copy an uncompressed file into a pipe is pure overhead, and
    /// reading the file with `read(upToCount:)` reports EOF only at genuine
    /// end-of-file and throws real read errors instead of raising the way
    /// `availableData` does. (`.zst` still needs the `zstd` CLI to decompress.)
    private static func streamGames(at path: String,
                                    shouldContinue: () -> Bool,
                                    onGame: (_ tags: [String: String], _ movetext: String) -> Void) throws {
        let usesZstd = path.hasSuffix(".zst")
        let process: Process?
        let handle: FileHandle
        if usesZstd {
            let zstd = Process()
            zstd.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            zstd.arguments = ["zstd", "-dc", path]
            let pipe = Pipe()
            zstd.standardOutput = pipe
            do {
                try zstd.run()
            } catch {
                throw PGNImportError.openFailed(error.localizedDescription)
            }
            handle = pipe.fileHandleForReading
            process = zstd
        } else {
            guard let fileHandle = FileHandle(forReadingAtPath: path) else {
                throw PGNImportError.openFailed("cannot open \(path)")
            }
            handle = fileHandle
            process = nil
        }

        // Per-game accumulation state.
        var tags: [String: String] = [:]
        var movetext = ""
        var inMoves = false

        func finalizeGame() {
            if !movetext.isEmpty || !tags.isEmpty {
                if !movetext.isEmpty { onGame(tags, movetext) }
                tags = [:]
                movetext = ""
            }
            inMoves = false
        }

        func processLine(_ raw: String) {
            // Trim newlines too, not just spaces/tabs. The reader splits on \n,
            // so on a CRLF (\r\n) file every line keeps a trailing \r. Under
            // `.whitespaces` (which excludes \r) that stray \r left tag lines
            // ending in "\r" — so `hasSuffix("]")` failed — and blank lines were
            // "\r" so never `isEmpty`; every tag/blank was then misread as
            // movetext and whole games collapsed together. The Lichess Elite
            // export mixes LF and CRLF months, which silently merged or dropped
            // most games in the CRLF months until this used
            // `.whitespacesAndNewlines`.
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.hasPrefix("[") && line.hasSuffix("]") {
                // A tag line that arrives after movetext begins a new game.
                if inMoves { finalizeGame() }
                if let (k, v) = parseTag(line) { tags[k] = v }
            } else if line.isEmpty {
                if inMoves { finalizeGame() }
                // blank between tags and movetext: ignore (movetext begins on
                // the next non-tag line).
            } else {
                inMoves = true
                movetext += " " + line
            }
        }

        // Scan complete lines out of `carry` per chunk, compacting the leftover
        // tail just once per chunk (cheap integer cursor, no per-line memmove).
        var carry = [UInt8]()
        let newline: UInt8 = 0x0A
        let chunkSize = 1 << 20   // fixed-size read buffer
        var endedByStop = false
        var readError: Error? = nil
        while true {
            if !shouldContinue() { endedByStop = true; break }
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: chunkSize)
            } catch {
                readError = error
                break
            }
            // `read(upToCount:)` returns nil only at genuine EOF (and blocks for
            // the pipe until bytes or EOF), so an empty result is unambiguously
            // end-of-input — not the mid-stream false-empty `availableData` gave.
            guard let chunk, !chunk.isEmpty else { break }
            carry.append(contentsOf: chunk)
            var lineStart = 0
            var i = 0
            while i < carry.count {
                if carry[i] == newline {
                    processLine(String(decoding: carry[lineStart..<i], as: UTF8.self))
                    lineStart = i + 1
                }
                i += 1
            }
            if lineStart > 0 { carry.removeFirst(lineStart) }
        }
        // On a clean read to EOF, flush the trailing partial line and emit the
        // final game; on an early stop or a read error, skip it — those games
        // would only be dropped (and any error is surfaced just below).
        if !endedByStop && readError == nil {
            if !carry.isEmpty {
                processLine(String(decoding: carry, as: UTF8.self))
            }
            finalizeGame()
        }
        // Reap the decompressor (zstd path only); the direct file handle closes
        // itself when it deallocates at scope exit.
        if let process {
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
        }
        // Surface a mid-stream read failure loudly rather than sealing a
        // truncated corpus.
        if let readError {
            throw PGNImportError.openFailed("read failed on \(path): \(readError.localizedDescription)")
        }
        // A nonzero `zstd` exit on a clean read (not our own early stop) means
        // decompression failed — missing `zstd` or a corrupt `.zst` — which
        // would otherwise yield a silently truncated corpus. Fail loudly.
        if let process, !endedByStop, process.terminationStatus != 0 {
            throw PGNImportError.openFailed("zstd exited with status \(process.terminationStatus) reading \(path)")
        }
    }

    private static func parseTag(_ line: String) -> (String, String)? {
        // [Key "Value"]
        var s = line
        s.removeFirst()       // [
        s.removeLast()        // ]
        guard let spaceIdx = s.firstIndex(of: " ") else { return nil }
        let key = String(s[s.startIndex..<spaceIdx])
        var value = String(s[s.index(after: spaceIdx)...]).trimmingCharacters(in: .whitespaces)
        if value.hasPrefix("\"") { value.removeFirst() }
        if value.hasSuffix("\"") { value.removeLast() }
        return (key, value)
    }

    // MARK: - Movetext → SAN tokens

    static func sanTokens(from movetext: String) -> [String] {
        let noComments = removeBraced(movetext, open: "{", close: "}")
        let noVariations = removeBraced(noComments, open: "(", close: ")")
        var result: [String] = []
        for raw in noVariations.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" }) {
            if raw.first == "$" { continue }                       // NAG
            let token = stripMoveNumberPrefix(String(raw))
            if token.isEmpty { continue }
            if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" { continue }
            result.append(token)
        }
        return result
    }

    private static func removeBraced(_ s: String, open: Character, close: Character) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        var depth = 0
        for ch in s {
            if ch == open {
                depth += 1
            } else if ch == close {
                if depth > 0 { depth -= 1 }
            } else if depth == 0 {
                out.append(ch)
            }
        }
        return out
    }

    private static func stripMoveNumberPrefix(_ s: String) -> String {
        let chars = Array(s)
        var i = 0
        while i < chars.count && chars[i].isNumber { i += 1 }
        guard i > 0 else { return s }
        // Bare "1-0"/"0-1" start with a digit but are results, not move numbers
        // — only treat as a move number if a dot follows the digits.
        guard i < chars.count && chars[i] == "." else { return s }
        while i < chars.count && chars[i] == "." { i += 1 }
        return String(chars[i...])
    }

    // MARK: - SAN → ChessMove

    /// Match a SAN token against a supplied move list (assumed legal). Returns
    /// the first matching `ChessMove`, or nil. Kept for callers/tests that
    /// already hold the legal moves; replay uses `resolveLegalSANMove`.
    static func matchSAN(_ token: String, state: GameState, legal: [ChessMove]) -> ChessMove? {
        sanCandidates(token, state: state, from: legal).first
    }

    /// Resolve a SAN token to the unique LEGAL move in `state`, generating only
    /// pseudo-legal candidates and legality-checking just the SAN matches. This
    /// avoids generating and legality-filtering the entire legal move list (the
    /// importer's dominant cost) when only one move needs resolving. A pinned
    /// piece that pseudo-matches the token is correctly rejected here. Returns
    /// nil if there is no legal match, or if more than one legal move matches
    /// (an ambiguous token — well-formed SAN always disambiguates), so the
    /// importer fails loudly rather than guessing.
    static func resolveLegalSANMove(_ token: String, state: GameState) -> ChessMove? {
        let color = state.currentPlayer
        let legal = sanCandidates(token, state: state, from: MoveGenerator.pseudoLegalMoves(for: state))
            .filter { candidate in
                !MoveGenerator.isInCheck(MoveGenerator.applyMove(candidate, to: state), color: color)
            }
        return legal.count == 1 ? legal[0] : nil
    }

    /// Every move in `moves` that matches SAN `token` for `state` — by piece
    /// type, destination, disambiguation, promotion, and castling. Pass the
    /// full legal list to get only legal matches, or pseudo-legal moves to
    /// legality-check the matches yourself. Shared by `matchSAN` (first match)
    /// and `resolveLegalSANMove` (first legal match).
    private static func sanCandidates(_ token: String, state: GameState, from moves: [ChessMove]) -> [ChessMove] {
        var san = token
        // Strip check/mate/annotation suffixes.
        while let last = san.last, "+#!?".contains(last) { san.removeLast() }
        if san.isEmpty { return [] }

        // Castling.
        if san == "O-O" || san == "0-0" {
            return moves.filter { m in isKing(m, state) && m.toCol == m.fromCol + 2 }
        }
        if san == "O-O-O" || san == "0-0-0" {
            return moves.filter { m in isKing(m, state) && m.toCol == m.fromCol - 2 }
        }

        var chars = Array(san)

        // Promotion: "=Q" or a trailing piece letter ("e8Q").
        var promotion: PieceType? = nil
        if let eq = chars.firstIndex(of: "=") {
            if eq + 1 < chars.count { promotion = sanPiece(chars[eq + 1]) }
            chars = Array(chars[0..<eq])
        } else if let last = chars.last, "NBRQ".contains(last), chars.count >= 3 {
            promotion = sanPiece(last)
            chars.removeLast()
        }

        // Leading piece letter (absent ⇒ pawn).
        var pieceType: PieceType = .pawn
        var idx = 0
        if let first = chars.first, "NBRQK".contains(first), let p = sanPiece(first) {
            pieceType = p
            idx = 1
        }

        // Strip capture markers; what's left is [disambig]? dest.
        let rest = Array(chars[idx...]).filter { $0 != "x" }
        guard rest.count >= 2 else { return [] }
        guard let dest = square(file: rest[rest.count - 2], rank: rest[rest.count - 1]) else { return [] }

        var disFile: Int? = nil
        var disRow: Int? = nil
        for ch in rest[0..<(rest.count - 2)] {
            if let c = col(forFile: ch) {
                disFile = c
            } else if let n = ch.wholeNumberValue, (1...8).contains(n) {
                disRow = 8 - n
            }
        }

        return moves.filter { m in
            guard m.toRow == dest.row, m.toCol == dest.col else { return false }
            guard m.promotion == promotion else { return false }
            guard let piece = state.board[m.fromRow * 8 + m.fromCol], piece.type == pieceType else { return false }
            if let df = disFile, m.fromCol != df { return false }
            if let dr = disRow, m.fromRow != dr { return false }
            return true
        }
    }

    private static func isKing(_ m: ChessMove, _ state: GameState) -> Bool {
        state.board[m.fromRow * 8 + m.fromCol]?.type == .king
    }

    private static func sanPiece(_ ch: Character) -> PieceType? {
        switch ch {
        case "N": return .knight
        case "B": return .bishop
        case "R": return .rook
        case "Q": return .queen
        case "K": return .king
        default:  return nil
        }
    }

    private static let files = Array("abcdefgh")

    private static func col(forFile ch: Character) -> Int? {
        files.firstIndex(of: ch)
    }

    /// Algebraic (file,rank) → (row,col) in the engine's convention: row 0 is
    /// rank 8, col 0 is file a (matches `FENParser`/`BoardEncoder`).
    private static func square(file: Character, rank: Character) -> (row: Int, col: Int)? {
        guard let col = col(forFile: file),
              let rankNum = rank.wholeNumberValue, (1...8).contains(rankNum) else { return nil }
        return (row: 8 - rankNum, col: col)
    }

    // MARK: - Result / time-control classification

    private static func outcome(forResult r: String) -> GameOutcome? {
        switch r {
        case "1-0":     return .whiteWin
        case "0-1":     return .blackWin
        case "1/2-1/2": return .draw
        default:        return nil
        }
    }

    /// Lichess-style time-control class from a `TimeControl` tag like "600+0".
    static func timeControlClass(_ tc: String) -> String {
        let parts = tc.split(separator: "+")
        let base = Int(parts.first.map(String.init) ?? "") ?? 0
        let inc = parts.count > 1 ? (Int(parts[1]) ?? 0) : 0
        let estimated = base + 40 * inc
        switch estimated {
        case ..<29:   return "ultrabullet"
        case ..<179:  return "bullet"
        case ..<479:  return "blitz"
        case ..<1499: return "rapid"
        default:      return "classical"
        }
    }
}

/// Serial-writer-confined state for the parallel importer. Every property is
/// touched only on the import's writer `DispatchQueue`, so the reference can be
/// shared across the worker closures without further locking — hence
/// `@unchecked Sendable`. The reorder buffer (`pending`) re-imposes original
/// file order on out-of-order worker completions.
private final class ImportWriterBox: @unchecked Sendable {
    private let corpus: GameCorpus
    private let config: PGNImportConfig
    private let inFlight: DispatchSemaphore
    private let group: DispatchGroup
    private let stop: OSAllocatedUnfairLock<Bool>

    var summary: PGNImporter.Summary
    private var pending: [Int: PGNImporter.ImportSlot] = [:]
    private var nextSeq = 0
    private var importedBytes = 0
    private(set) var firstError: (seq: Int, reason: String)? = nil

    init(corpus: GameCorpus,
         config: PGNImportConfig,
         inFlight: DispatchSemaphore,
         group: DispatchGroup,
         stop: OSAllocatedUnfairLock<Bool>) {
        self.corpus = corpus
        self.config = config
        self.inFlight = inFlight
        self.group = group
        self.stop = stop
        var s = PGNImporter.Summary()
        s.corpusID = corpus.corpusID
        self.summary = s
    }

    private func requestStop() { stop.withLock { $0 = true } }
    private func isStopped() -> Bool { stop.withLock { $0 } }

    /// Store one completed slot and drain every now-contiguous slot in order.
    /// Must run on the serial writer queue.
    func submit(seq: Int, slot: PGNImporter.ImportSlot) {
        pending[seq] = slot
        while let s = pending.removeValue(forKey: nextSeq) {
            if firstError == nil { consume(s, seq: nextSeq) }
            nextSeq += 1
            inFlight.signal()
            group.leave()
        }
    }

    private func consume(_ slot: PGNImporter.ImportSlot, seq: Int) {
        if isStopped() { return }   // already past a cap — drop in-flight remainder
        switch slot {
        case .imported(let frame, let ply):
            if let cap = config.maxStorageBytes, importedBytes + frame.count > cap {
                requestStop()
                return
            }
            do {
                try corpus.append(framed: frame, plyCount: ply)
            } catch {
                firstError = (seq, "append failed: \(error.localizedDescription)")
                requestStop()
                return
            }
            summary.imported += 1
            importedBytes += frame.count
            if let mx = config.maxGames, summary.imported >= mx { requestStop() }
            if summary.imported % 10_000 == 0 {
                SessionLogger.shared.log("[PGN] imported \(summary.imported) games…")
            }
        case .skipped:
            summary.skipped += 1
        case .failed(let reason):
            if config.failOnError {
                firstError = (seq, reason)
                requestStop()
            } else {
                summary.parseErrors += 1
            }
        }
    }
}
