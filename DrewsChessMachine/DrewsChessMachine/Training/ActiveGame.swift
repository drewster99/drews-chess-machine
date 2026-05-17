import Foundation

/// One concurrent self-play game in the upcoming tick-driver topology.
///
/// Owns the chess engine plus all per-game completed-ply staging
/// buffers needed for the bulk replay-buffer flush at game end. Lives
/// for the lifetime of the slot (driver-allocated once, reused across
/// many games via `resetForNewGame`), not the lifetime of one game —
/// so per-game reset does not free/realloc unless the user has
/// increased `maxPliesPerGame` past the current capacity.
///
/// **Layout choice (Option A from the design):** each side has its own
/// contiguous staging run (white plies 0,1,2,... then 1:1 mirror for
/// black). Mirrors today's two-`MPSChessPlayer` model bit-for-bit, so
/// the bulk `ReplayBuffer.append` calls at flush time stay one
/// memcpy per side per field. Total per-game memory at `capPlies=150`
/// (the typical value): ~1.2 MB — vs ~7.5 MB per slot under the
/// pre-rework `MPSChessPlayer`-based self-play path (two players ×
/// 3.75 MB pre-allocated 512-ply board scratch).
///
/// **Per-side cap math.** A game of up to `capPlies` total plies has at
/// most `capPlies / 2 + capPlies % 2` plies from white and the rest
/// from black. We allocate `perSideCap = (capPlies + 1) / 2` for both
/// sides — covers the odd-ply case where the game ends on white's
/// move with one more white ply than black.
///
/// **Thread safety.** Driver-owned; mutated only from the tick driver's
/// task. Marked `@unchecked Sendable` to match the project convention
/// for driver-private state (no internal lock — single-task access).
final class ActiveGame: @unchecked Sendable {

    // MARK: - Constants

    /// Per-position encoded-board size in floats. Cached here so init
    /// math doesn't need to reach into `BoardEncoder` for every slot
    /// allocation.
    private static let boardFloats = BoardEncoder.tensorLength

    // MARK: - Identity (set once at init or per-game reset)

    /// Slot-stable worker stamp, attached to each appended position so
    /// `[BATCH-STATS]` can detect per-worker over-representation. Set
    /// once at init by the driver from its monotonic `nextWorkerId`
    /// counter; survives `resetForNewGame`.
    let workerId: UInt16

    /// Bumped on every `resetForNewGame`. The first game on this slot
    /// starts at `1` (init sets `0`; reset bumps before recording any
    /// ply). Stamped into each appended position.
    private(set) var intraWorkerGameIndex: UInt32 = 0

    /// White-side network reference. For self-play this is the
    /// champion; for arena it's whichever network plays white in this
    /// particular game. The driver uses
    /// `currentNetwork` (which reads white/black off this and
    /// `state.currentPlayer`) to partition K games into per-network
    /// sub-batches per tick.
    let whiteNetwork: ChessMPSNetwork

    /// Black-side network reference. See `whiteNetwork`. In self-play
    /// this points at the same instance as `whiteNetwork` — both
    /// sides play the same champion.
    let blackNetwork: ChessMPSNetwork

    // MARK: - Per-game live state

    /// Engine for the currently-in-flight game. Replaced on
    /// `resetForNewGame`; the previous instance is dropped (its move
    /// history and repetition state are no longer needed once the
    /// game has been flushed).
    private(set) var engine: ChessGameEngine

    /// Sampling schedule active for this game. Captured at
    /// `resetForNewGame` time so a mid-game schedule edit doesn't
    /// retroactively change tau values for plies already played
    /// (matches today's behavior — `MPSChessPlayer` reads `schedule`
    /// each ply but the driver only assigns `schedule` between games).
    private(set) var schedule: SamplingSchedule

    /// Maximum total plies (white + black combined) before the game
    /// is dropped without flush. Latched at `resetForNewGame` so
    /// in-flight games keep their original cap (matches today).
    private(set) var maxPliesCap: Int

    /// Wall time the current game started. Used by the driver to
    /// publish `gameDurationMs` into `ParallelWorkerStatsBox`.
    private(set) var gameStartedAt: CFAbsoluteTime

    /// Count of "random-ish" moves this game — plies where the
    /// post-temperature, pre-Dirichlet softmax over legal moves was
    /// essentially uniform. The driver increments this from
    /// `MoveSampler.Result.randomish`. Surfaced through the existing
    /// per-game statistics paths.
    var randomishCount: Int = 0

    // MARK: - Per-game completed-ply staging (per-side)

    /// Current per-side allocated capacity (in plies). Equal to
    /// `(currentCapPlies + 1) / 2`. Tracked separately from
    /// `maxPliesCap` so a reset can decide whether to reuse or
    /// reallocate without re-computing.
    private(set) var perSideCap: Int

    /// Per-side encoded-board staging. Each row is `boardFloats`
    /// floats; row `i` corresponds to the i'th ply that THIS side
    /// played. Allocated via raw pointer so layout matches what
    /// `ReplayBuffer.append(boards: UnsafePointer<Float>, ...)`
    /// expects and the bulk flush is a memcpy-style read.
    private var whiteBoardScratch: UnsafeMutablePointer<Float>
    private var blackBoardScratch: UnsafeMutablePointer<Float>

    /// Per-ply policy index (0..<ChessNetwork.policySize) — the index
    /// of the move actually played, in the policy-vector frame for
    /// the side that played it. Source: `MoveSampler.Result.policyIndex`.
    private var whitePolicyIndices: UnsafeMutablePointer<Int32>
    private var blackPolicyIndices: UnsafeMutablePointer<Int32>

    /// Per-recorded-ply **game-total** ply index — the position's
    /// 0-indexed half-move count from the start of the game. White's
    /// recorded positions land on even plies (0, 2, 4, …), black's
    /// on odd plies (1, 3, 5, …). Saturates at UInt16.max — chess
    /// games can't reach that under any realistic cap. Phase-histogram
    /// bucketing in `PhaseHistogram.plyBucket` consumes this value
    /// directly (cutoffs ≤15 / 16–35 / 36–75 / 76–125 / 126+).
    private var whitePlyIndices: UnsafeMutablePointer<UInt16>
    private var blackPlyIndices: UnsafeMutablePointer<UInt16>

    /// Per-ply sampling temperature used at this ply, from
    /// `schedule.tau(forPly:)`. Stored so post-hoc analysis can
    /// reconstruct what temperature each training example was drawn at.
    private var whiteSamplingTaus: UnsafeMutablePointer<Float>
    private var blackSamplingTaus: UnsafeMutablePointer<Float>

    /// Per-ply 64-bit hash of the encoded-board bytes. Used by the
    /// replay buffer's per-hash WLD counters and diagnostic dedup.
    private var whiteStateHashes: UnsafeMutablePointer<UInt64>
    private var blackStateHashes: UnsafeMutablePointer<UInt64>

    /// Per-ply non-pawn piece count, for material-bucket phase
    /// classification.
    private var whiteMaterialCounts: UnsafeMutablePointer<UInt8>
    private var blackMaterialCounts: UnsafeMutablePointer<UInt8>

    /// Number of plies the white side has recorded this game.
    private(set) var whitePliesRecorded: Int = 0
    /// Number of plies the black side has recorded this game.
    private(set) var blackPliesRecorded: Int = 0

    // MARK: - Convenience

    /// Total plies played in the current game (both sides). Compared
    /// against `maxPliesCap` by the driver's game-end pass to detect
    /// max-plies drops.
    var totalPliesPlayed: Int { whitePliesRecorded + blackPliesRecorded }

    /// Network the current side-to-move uses. Self-play: both sides
    /// resolve to the same champion. Arena: alternates between
    /// candidate and champion across the K active games.
    var currentNetwork: ChessMPSNetwork {
        engine.state.currentPlayer == .white ? whiteNetwork : blackNetwork
    }

    // MARK: - Init / deinit

    /// Allocates all per-side scratch and creates a fresh engine. The
    /// per-side cap is derived from `capPlies` (the total-game cap)
    /// as `(capPlies + 1) / 2`.
    init(
        workerId: UInt16,
        whiteNetwork: ChessMPSNetwork,
        blackNetwork: ChessMPSNetwork,
        capPlies: Int,
        schedule: SamplingSchedule
    ) {
        precondition(capPlies >= 1, "ActiveGame.init: capPlies must be >= 1")
        self.workerId = workerId
        self.whiteNetwork = whiteNetwork
        self.blackNetwork = blackNetwork
        self.engine = ChessGameEngine()
        self.schedule = schedule
        self.maxPliesCap = capPlies
        self.gameStartedAt = CFAbsoluteTimeGetCurrent()

        let sideCap = (capPlies + 1) / 2
        self.perSideCap = sideCap

        self.whiteBoardScratch = Self.allocBoardScratch(sideCap)
        self.blackBoardScratch = Self.allocBoardScratch(sideCap)
        self.whitePolicyIndices = Self.allocInt32(sideCap)
        self.blackPolicyIndices = Self.allocInt32(sideCap)
        self.whitePlyIndices = Self.allocUInt16(sideCap)
        self.blackPlyIndices = Self.allocUInt16(sideCap)
        self.whiteSamplingTaus = Self.allocFloat(sideCap)
        self.blackSamplingTaus = Self.allocFloat(sideCap)
        self.whiteStateHashes = Self.allocUInt64(sideCap)
        self.blackStateHashes = Self.allocUInt64(sideCap)
        self.whiteMaterialCounts = Self.allocUInt8(sideCap)
        self.blackMaterialCounts = Self.allocUInt8(sideCap)
    }

    deinit {
        let bf = Self.boardFloats
        whiteBoardScratch.deinitialize(count: perSideCap * bf)
        whiteBoardScratch.deallocate()
        blackBoardScratch.deinitialize(count: perSideCap * bf)
        blackBoardScratch.deallocate()
        whitePolicyIndices.deinitialize(count: perSideCap); whitePolicyIndices.deallocate()
        blackPolicyIndices.deinitialize(count: perSideCap); blackPolicyIndices.deallocate()
        whitePlyIndices.deinitialize(count: perSideCap);    whitePlyIndices.deallocate()
        blackPlyIndices.deinitialize(count: perSideCap);    blackPlyIndices.deallocate()
        whiteSamplingTaus.deinitialize(count: perSideCap);  whiteSamplingTaus.deallocate()
        blackSamplingTaus.deinitialize(count: perSideCap);  blackSamplingTaus.deallocate()
        whiteStateHashes.deinitialize(count: perSideCap);   whiteStateHashes.deallocate()
        blackStateHashes.deinitialize(count: perSideCap);   blackStateHashes.deallocate()
        whiteMaterialCounts.deinitialize(count: perSideCap); whiteMaterialCounts.deallocate()
        blackMaterialCounts.deinitialize(count: perSideCap); blackMaterialCounts.deallocate()
    }

    // MARK: - Lifecycle

    /// Reset for a new game. Bumps `intraWorkerGameIndex`, refreshes
    /// the engine + schedule + cap, and zeros the per-side fill
    /// counters. Staging contents are NOT zeroed — each ply's
    /// `recordPly` overwrites its slot before any read (matching
    /// today's `MPSChessPlayer.onNewGame` convention).
    ///
    /// If `maxPliesCap > perSideCap * 2` (the user grew the cap
    /// since this slot was allocated), all per-side scratch is
    /// reallocated to the new size. Cap shrinks are ignored — never
    /// shrinks (avoids thrash when the user toggles back and forth).
    func resetForNewGame(maxPliesCap newCap: Int, schedule newSchedule: SamplingSchedule) {
        precondition(newCap >= 1, "ActiveGame.resetForNewGame: newCap must be >= 1")
        intraWorkerGameIndex &+= 1
        engine = ChessGameEngine()
        schedule = newSchedule
        maxPliesCap = newCap
        gameStartedAt = CFAbsoluteTimeGetCurrent()
        randomishCount = 0
        whitePliesRecorded = 0
        blackPliesRecorded = 0

        let neededSideCap = (newCap + 1) / 2
        if neededSideCap > perSideCap {
            growSideScratches(to: neededSideCap)
        }
    }

    // MARK: - Per-ply recording

    /// Record one ply into the side-appropriate staging slot. The
    /// caller (the tick driver) has already encoded the board into
    /// its tick scratch at `encodedBoardSrc`; this method copies that
    /// 1920-float run into the next free slot of the side's
    /// `*BoardScratch` and writes the per-ply metadata fields.
    ///
    /// `side` is which color just played this ply — i.e. the side
    /// whose turn it WAS at the time of the move, NOT
    /// `engine.state.currentPlayer` after the move was applied.
    /// (Driver flow: read currentPlayer → encode → sample → apply
    /// → recordPly with the pre-apply currentPlayer.)
    ///
    /// The chosen move's `policyIndex` and the sampling `tau` come
    /// from `MoveSampler.Result`; the caller passes them through to
    /// avoid recomputation.
    func recordPly(
        side: PieceColor,
        encodedBoardSrc: UnsafePointer<Float>,
        policyIndex: Int,
        samplingTau: Float,
        materialCount: UInt8
    ) {
        let bf = Self.boardFloats
        switch side {
        case .white:
            precondition(
                whitePliesRecorded < perSideCap,
                "ActiveGame.recordPly(.white): per-side cap \(perSideCap) exhausted at ply \(whitePliesRecorded)"
            )
            let dst = whiteBoardScratch + whitePliesRecorded * bf
            dst.update(from: encodedBoardSrc, count: bf)
            let stateHash = ReplayBuffer.hashBoard(dst, count: bf)
            whitePolicyIndices[whitePliesRecorded] = Int32(policyIndex)
            // Game-total ply for white's i-th recorded position is
            // 2*i: white moves on even half-moves (0, 2, 4, …).
            whitePlyIndices[whitePliesRecorded] = UInt16(min(2 * whitePliesRecorded, Int(UInt16.max)))
            whiteSamplingTaus[whitePliesRecorded] = samplingTau
            whiteStateHashes[whitePliesRecorded] = stateHash
            whiteMaterialCounts[whitePliesRecorded] = materialCount
            whitePliesRecorded += 1
        case .black:
            precondition(
                blackPliesRecorded < perSideCap,
                "ActiveGame.recordPly(.black): per-side cap \(perSideCap) exhausted at ply \(blackPliesRecorded)"
            )
            let dst = blackBoardScratch + blackPliesRecorded * bf
            dst.update(from: encodedBoardSrc, count: bf)
            let stateHash = ReplayBuffer.hashBoard(dst, count: bf)
            blackPolicyIndices[blackPliesRecorded] = Int32(policyIndex)
            // Game-total ply for black's i-th recorded position is
            // 2*i + 1: black moves on odd half-moves (1, 3, 5, …).
            blackPlyIndices[blackPliesRecorded] = UInt16(min(2 * blackPliesRecorded + 1, Int(UInt16.max)))
            blackSamplingTaus[blackPliesRecorded] = samplingTau
            blackStateHashes[blackPliesRecorded] = stateHash
            blackMaterialCounts[blackPliesRecorded] = materialCount
            blackPliesRecorded += 1
        }
    }

    // MARK: - Flush

    /// Bulk-push every recorded ply from the just-finished game into
    /// `buffer`, two `append` calls (white side + black side) with the
    /// per-side outcome broadcast across every row. Mirrors the
    /// two-`MPSChessPlayer` flushes today, including the per-side
    /// outcome sign-flip — white's outcome is +1 if white won, -1 if
    /// black won, 0 for draws; black gets the negation.
    ///
    /// Caller is responsible for the draw-keep filter (i.e.
    /// `selfPlayDrawKeepFraction` for drawn games): this method
    /// unconditionally flushes. The caller decides whether to call it.
    ///
    /// Returns the combined `FlushedGameStats` (sum of white and
    /// black flushes), or `nil` if there's nothing to flush.
    @discardableResult
    func flush(buffer: ReplayBuffer, result: GameResult) -> FlushedGameStats? {
        guard whitePliesRecorded > 0 || blackPliesRecorded > 0 else {
            return nil
        }

        let whiteOutcome: Float
        switch result {
        case .checkmate(let winner):
            whiteOutcome = (winner == .white) ? 1.0 : -1.0
        case .stalemate,
             .drawByFiftyMoveRule,
             .drawByInsufficientMaterial,
             .drawByThreefoldRepetition:
            whiteOutcome = 0.0
        }
        let blackOutcome: Float = -whiteOutcome

        let gameLengthClamped = UInt16(min(totalPliesPlayed, Int(UInt16.max)))

        let whiteStats = flushSide(
            buffer: buffer,
            boards: whiteBoardScratch,
            policyIndices: whitePolicyIndices,
            plyIndices: whitePlyIndices,
            samplingTaus: whiteSamplingTaus,
            stateHashes: whiteStateHashes,
            materialCounts: whiteMaterialCounts,
            count: whitePliesRecorded,
            outcome: whiteOutcome,
            gameLength: gameLengthClamped
        )
        let blackStats = flushSide(
            buffer: buffer,
            boards: blackBoardScratch,
            policyIndices: blackPolicyIndices,
            plyIndices: blackPlyIndices,
            samplingTaus: blackSamplingTaus,
            stateHashes: blackStateHashes,
            materialCounts: blackMaterialCounts,
            count: blackPliesRecorded,
            outcome: blackOutcome,
            gameLength: gameLengthClamped
        )
        return whiteStats + blackStats
    }

    private func flushSide(
        buffer: ReplayBuffer,
        boards: UnsafePointer<Float>,
        policyIndices: UnsafePointer<Int32>,
        plyIndices: UnsafePointer<UInt16>,
        samplingTaus: UnsafePointer<Float>,
        stateHashes: UnsafePointer<UInt64>,
        materialCounts: UnsafePointer<UInt8>,
        count: Int,
        outcome: Float,
        gameLength: UInt16
    ) -> FlushedGameStats {
        if count == 0 { return .empty }

        var phaseByPly = PhaseHistogram.zero
        var phaseByMaterial = PhaseHistogram.zero
        for i in 0..<count {
            let plyBucket = PhaseHistogram.plyBucket(ply: Int(plyIndices[i]))
            let matBucket = PhaseHistogram.materialBucket(materialCount: Int(materialCounts[i]))
            switch plyBucket {
            case 0: phaseByPly.open += 1
            case 1: phaseByPly.early += 1
            case 2: phaseByPly.mid += 1
            case 3: phaseByPly.late += 1
            default: phaseByPly.end += 1
            }
            switch matBucket {
            case 0: phaseByMaterial.open += 1
            case 1: phaseByMaterial.early += 1
            case 2: phaseByMaterial.mid += 1
            case 3: phaseByMaterial.late += 1
            default: phaseByMaterial.end += 1
            }
        }

        buffer.append(
            boards: boards,
            policyIndices: policyIndices,
            plyIndices: plyIndices,
            samplingTaus: samplingTaus,
            stateHashes: stateHashes,
            materialCounts: materialCounts,
            gameLength: gameLength,
            workerId: workerId,
            intraWorkerGameIndex: intraWorkerGameIndex,
            outcome: outcome,
            count: count
        )

        return FlushedGameStats(
            positions: count,
            phaseByPly: phaseByPly,
            phaseByMaterial: phaseByMaterial
        )
    }

    // MARK: - Scratch growth

    /// Grow every per-side scratch to a new (strictly larger) per-side
    /// capacity. Only called from `resetForNewGame` when the user has
    /// increased `maxPliesPerGame` past the slot's allocated cap, so
    /// it's safe to discard buffer contents — we're between games.
    private func growSideScratches(to newCap: Int) {
        precondition(newCap > perSideCap, "ActiveGame.growSideScratches: must strictly increase")
        let bf = Self.boardFloats

        whiteBoardScratch.deinitialize(count: perSideCap * bf); whiteBoardScratch.deallocate()
        blackBoardScratch.deinitialize(count: perSideCap * bf); blackBoardScratch.deallocate()
        whitePolicyIndices.deinitialize(count: perSideCap); whitePolicyIndices.deallocate()
        blackPolicyIndices.deinitialize(count: perSideCap); blackPolicyIndices.deallocate()
        whitePlyIndices.deinitialize(count: perSideCap);    whitePlyIndices.deallocate()
        blackPlyIndices.deinitialize(count: perSideCap);    blackPlyIndices.deallocate()
        whiteSamplingTaus.deinitialize(count: perSideCap);  whiteSamplingTaus.deallocate()
        blackSamplingTaus.deinitialize(count: perSideCap);  blackSamplingTaus.deallocate()
        whiteStateHashes.deinitialize(count: perSideCap);   whiteStateHashes.deallocate()
        blackStateHashes.deinitialize(count: perSideCap);   blackStateHashes.deallocate()
        whiteMaterialCounts.deinitialize(count: perSideCap); whiteMaterialCounts.deallocate()
        blackMaterialCounts.deinitialize(count: perSideCap); blackMaterialCounts.deallocate()

        whiteBoardScratch = Self.allocBoardScratch(newCap)
        blackBoardScratch = Self.allocBoardScratch(newCap)
        whitePolicyIndices = Self.allocInt32(newCap)
        blackPolicyIndices = Self.allocInt32(newCap)
        whitePlyIndices = Self.allocUInt16(newCap)
        blackPlyIndices = Self.allocUInt16(newCap)
        whiteSamplingTaus = Self.allocFloat(newCap)
        blackSamplingTaus = Self.allocFloat(newCap)
        whiteStateHashes = Self.allocUInt64(newCap)
        blackStateHashes = Self.allocUInt64(newCap)
        whiteMaterialCounts = Self.allocUInt8(newCap)
        blackMaterialCounts = Self.allocUInt8(newCap)

        perSideCap = newCap
    }

    // MARK: - Allocation helpers

    private static func allocBoardScratch(_ sideCap: Int) -> UnsafeMutablePointer<Float> {
        let count = sideCap * boardFloats
        let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }
    private static func allocInt32(_ count: Int) -> UnsafeMutablePointer<Int32> {
        let p = UnsafeMutablePointer<Int32>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }
    private static func allocUInt16(_ count: Int) -> UnsafeMutablePointer<UInt16> {
        let p = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }
    private static func allocFloat(_ count: Int) -> UnsafeMutablePointer<Float> {
        let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }
    private static func allocUInt64(_ count: Int) -> UnsafeMutablePointer<UInt64> {
        let p = UnsafeMutablePointer<UInt64>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }
    private static func allocUInt8(_ count: Int) -> UnsafeMutablePointer<UInt8> {
        let p = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        p.initialize(repeating: 0, count: count)
        return p
    }
}
