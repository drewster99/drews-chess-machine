import Accelerate
import Foundation

// MARK: - Dirichlet Exploration Noise

/// Mixes a sample from the symmetric Dirichlet distribution into the
/// per-move sampling probabilities to keep self-play opening coverage
/// from collapsing onto the network's current trunk. Without it, the
/// only exploration mechanism is the temperature schedule, which
/// scales the network's existing opinion rather than overriding it —
/// a move the policy gives 0.001 to is sampled ~0.1 % of the time
/// regardless of tau, so off-trunk lines drift out of the replay
/// buffer entirely.
///
/// The mix is `p'(a) = (1 - ε) · p(a) + ε · η(a)` where `η ~ Dir(α)`
/// over the legal moves. With `α < 1` the noise mass concentrates on
/// a small random subset of moves rather than smearing uniformly,
/// giving each game a slightly different "preferred" off-trunk line
/// instead of dumping ε of the probability into chaos.
///
/// AlphaZero applies Dirichlet noise at the MCTS root prior, where
/// ~800 simulations average out the per-game randomness; in this
/// engine the noise affects the sampled move directly, so `epsilon`
/// values comparable to AlphaZero's 0.25 are at the aggressive end of
/// the useful range. `plyLimit` bounds the noise to the opening so
/// middle/endgame play reflects the network's actual judgment.
struct DirichletNoiseConfig: Sendable {
    /// Concentration parameter α. Must be > 0. AlphaZero uses 0.3 for
    /// chess. Smaller values produce spikier noise (mass on fewer
    /// random moves); larger values smear toward uniform.
    let alpha: Float

    /// Mixing weight ε in `p' = (1-ε) · p + ε · η`. Must be in
    /// [0, 1]. AlphaZero baseline is 0.25; for a no-search engine,
    /// 0.10–0.25 is a reasonable starting range.
    let epsilon: Float

    /// Maximum per-player ply (0-indexed) at which noise is mixed in.
    /// At ply >= `plyLimit` the network's policy is sampled
    /// unmodified. Limits noise to the opening, which is where
    /// replay-buffer coverage matters most and where the network's
    /// own argmax tends to repeat.
    let plyLimit: Int

    /// AlphaZero baseline for chess (α=0.3, ε=0.25), gated to the
    /// first 30 plies per player. Apply to self-play only — arena
    /// games should sample without exploration noise.
    static let alphaZero = DirichletNoiseConfig(
        alpha: 0.3,
        epsilon: 0.25,
        plyLimit: 30
    )
}

// MARK: - Sampling Schedule

/// Linear-decay temperature schedule applied by `MPSChessPlayer.sampleMove`.
///
/// Temperature starts at `startTau` on ply 0 and decreases by
/// `decayPerPly` each ply (per-player), bottoming out at `floorTau`.
/// The formula is: `tau(ply) = max(floorTau, startTau - decayPerPly * ply)`.
///
/// Temperature scales the legal-move logits as `logit / tau` before
/// the softmax, so:
///
/// - `tau = 1.0` leaves logits unchanged (raw softmax).
/// - `tau < 1.0` concentrates the distribution on the largest logit,
///   approaching argmax as `tau → 0`.
/// - `tau > 1.0` flattens the distribution toward uniform.
///
/// All tau values must be strictly positive; `tau = 0` would divide
/// by zero and is not a valid argmax shortcut. See
/// `sampling-parameters.md` for the design rationale behind each preset.
struct SamplingSchedule: Sendable {
    /// Temperature on the player's first move (ply 0). Must be > 0.
    let startTau: Float
    /// Temperature reduction per ply (per-player). Must be >= 0.
    let decayPerPly: Float
    /// Minimum temperature — the decay floor. Must be > 0.
    let floorTau: Float
    /// Optional Dirichlet exploration noise mixed into the sampled
    /// move distribution after temperature softmax. `nil` means no
    /// noise (the prior behavior). See `DirichletNoiseConfig`.
    let dirichletNoise: DirichletNoiseConfig?

    /// Custom init so the new `dirichletNoise` field can default to
    /// `nil` without forcing every call site (e.g.
    /// `SessionCheckpointFile`'s positional reconstructor) to know
    /// about it.
    init(
        startTau: Float,
        decayPerPly: Float,
        floorTau: Float,
        dirichletNoise: DirichletNoiseConfig? = nil
    ) {
        self.startTau = startTau
        self.decayPerPly = decayPerPly
        self.floorTau = floorTau
        self.dirichletNoise = dirichletNoise
    }

    /// Self-play data-generation schedule: starts at tau=2.0, decays
    /// by 0.03 per ply, flooring at 0.4 (reached at ply ~53 per
    /// player). Higher starting temperature than the prior 1.0 to
    /// flatten the policy more aggressively in the opening, broaden
    /// replay-buffer coverage of off-trunk lines, and pull more
    /// decisive games out of an early-bootstrap policy that
    /// otherwise concentrates on shuffle moves. Combined with the
    /// AlphaZero-default Dirichlet noise mixed into the opening 30
    /// plies per player.
    static let selfPlay = SamplingSchedule(
        startTau: 2.0,
        decayPerPly: 0.03,
        floorTau: 0.4,
        dirichletNoise: .alphaZero
    )

    /// Arena-evaluation schedule: starts at tau=2.0, decays by 0.04
    /// per ply, flooring at 0.2 (reached at ply 45). Higher starting
    /// temperature than the prior 0.7 to expose more candidate-vs-
    /// champion divergence in the opening (helps surface the small
    /// signal that survives the dominant-draw regime). The faster
    /// decay and lower floor still ensure the middlegame onward
    /// reflects decisive play rather than sampling noise. No
    /// Dirichlet noise — arena games measure actual strength.
    static let arena = SamplingSchedule(
        startTau: 2.0,
        decayPerPly: 0.04,
        floorTau: 0.2
    )

    /// Flat tau=1.0 sampling for every move. Used by Play Game and
    /// other code paths outside the self-play → train → arena loop;
    /// exactly reproduces the pre-schedule behavior of `sampleMove`.
    static let uniform = SamplingSchedule(
        startTau: 1.0,
        decayPerPly: 0.0,
        floorTau: 1.0
    )

    /// The ply (per-player) at which tau reaches the floor. Returns
    /// `Int.max` when `decayPerPly` is zero (no decay).
    var pliesUntilFloor: Int {
        guard decayPerPly > 0 else { return Int.max }
        return Int(ceilf((startTau - floorTau) / decayPerPly))
    }

    /// Temperature to apply on the `ply`-th move (0-indexed) this
    /// player makes in the current game.
    @inline(__always)
    func tau(forPly ply: Int) -> Float {
        max(floorTau, startTau - decayPerPly * Float(ply))
    }
}

// MARK: - MPS Chess Player

/// A chess player that uses a neural network (ChessMPSNetwork) to choose moves.
///
/// Encodes the board from the current player's perspective, runs inference,
/// masks illegal moves, renormalizes, and samples from the policy distribution.
/// Records every played position into a per-game flat scratch buffer so the
/// self-play hot path allocates no per-move `[Float](tensorLength)` tensors; at
/// game end the whole buffer is pushed into the shared `ReplayBuffer` in
/// one bulk copy.
///
/// A player instance is reused across many games within a self-play slot
/// (or lives one game in arena / Play Game). `onNewGame` resets the
/// per-game fill counts so the scratches behave as if fresh each game,
/// without reallocating. Access is single-threaded within the game task;
/// no locking is needed on the player's internal state.
final class MPSChessPlayer: ChessPlayer {
    /// Number of floats in one encoded board position. Kept in sync
    /// with `BoardEncoder.tensorLength`.
    private static let boardFloats = BoardEncoder.tensorLength

    /// Initial ply capacity of the per-game board scratch. 512 plies
    /// covers essentially every realistic chess game; longer games
    /// grow the scratch via `growGameBoardScratch(toPlyCapacity:)` so
    /// the engine still works on extreme positions.
    private static let startingPlyCapacity = 512

    /// Upper bound on the number of legal moves in any chess position.
    /// The mathematical maximum is around 218, so 256 leaves a safety
    /// margin without wasting memory. Used to size `sampleScratch`.
    private static let sampleScratchCapacity = 256

    let identifier: String
    let name: String
    /// Source of (policy, value) predictions. Goes through
    /// `DirectMoveEvaluationSource` (sync single-board inference) for
    /// arena / Play Game / Play Continuous, or through
    /// `BatchedMoveEvaluationSource` (barrier batcher) for self-play,
    /// so N slot tasks coalesce their per-ply forward passes into one
    /// batched `graph.run`.
    private let source: MoveEvaluationSource
    /// Optional replay buffer. When non-nil, this player CAN push each
    /// finished game's labeled positions into the buffer, but the push is
    /// NOT automatic on `onGameEnded` — the caller must explicitly invoke
    /// `flushRecordedGameToReplayBuffer(result:)` after the game ends.
    /// Self-play uses this two-step contract so the slot driver can
    /// decide per-game whether to keep or drop the game (e.g. honour the
    /// `selfPlayDrawKeepFraction` draw filter). The default-nil path is
    /// what Play Game / Play Continuous / arena use — they pass nil and
    /// never call flush, identical to pre-buffer behaviour.
    private let replayBuffer: ReplayBuffer?
    /// Temperature schedule applied per-ply in `sampleMove`. Defaults to
    /// `.uniform` (flat tau=1.0) so non-training callers keep their
    /// pre-schedule behavior; self-play and arena pass their own presets.
    ///
    /// Mutable so the driver can hand a reused player a new schedule at
    /// each game boundary (UI can tune tau without destroying the
    /// slot's allocated scratches). Writes must happen between games —
    /// `sampleMove` reads this once per ply and the player is
    /// single-threaded within a game, so an in-game swap would be a
    /// use-after-free-style race; callers MUST only set this outside
    /// of an active `beginNewGame`.
    var schedule: SamplingSchedule
    private var isWhite = true

    // MARK: - Per-game recorded positions

    /// Raw storage for every encoded position played this game, laid
    /// out as `[ply, plane * 64 + row * 8 + col]` — i.e. ply `p`
    /// occupies `gameBoardScratchPtr + p * boardFloats`. Grown via
    /// `growGameBoardScratch` when a long game outruns the initial
    /// capacity (rare). Owned via `UnsafeMutablePointer` so the
    /// encoder and the network can share the same memory without
    /// Swift array CoW.
    private var gameBoardScratchPtr: UnsafeMutablePointer<Float>
    /// Current allocated capacity of `gameBoardScratchPtr`, in plies
    /// (not floats). `gameBoardScratchPtr` holds
    /// `gameBoardScratchCapacity * boardFloats` floats.
    private var gameBoardScratchCapacity: Int

    /// Reusable destination scratch for the per-ply policy readback.
    /// Manually allocated once at init and freed at deinit, mirroring
    /// the `gameBoardScratchPtr` pattern. Sized to
    /// `ChessNetwork.policySize` floats and passed (as
    /// `UnsafeMutableBufferPointer`) to `source.evaluate` every ply,
    /// satisfying the `MoveEvaluationSource` destination-buffer
    /// contract. One ply runs at a time per player (white then black,
    /// serialized by `ChessMachine.runGameLoop`), so this scratch is
    /// never aliased by concurrent calls. Owned by raw pointer so the
    /// network can write into it via `update(from:count:)` with no
    /// Swift `Array` allocation per call.
    private let policyScratchPtr: UnsafeMutablePointer<Float>
    /// Element count of `policyScratchPtr` — captured once so the
    /// `init` allocation and the `deinit` `deinitialize`/`deallocate`
    /// pair stay in sync, and the per-ply call site doesn't have to
    /// reach into `ChessNetwork.policySize` (which would tie the call
    /// site to the constant changing).
    private static let policyScratchCount = ChessNetwork.policySize

    /// Policy-target indices for each recorded ply, computed via
    /// `PolicyEncoding.policyIndex` in the network's encoder-frame
    /// coordinate system (0..<policySize, currently 0..<4864). Pre-
    /// reserved so growth is rare; grown automatically if the game
    /// exceeds the initial capacity.
    private var gamePolicyIndices: [Int32]

    /// Per-recorded-ply observability metadata — flushed alongside
    /// `gamePolicyIndices` into the replay buffer at game end. Sized in
    /// lock-step with the move-index array.
    /// - `gamePlyIndices[i]`: the player-local ply index for record `i`.
    ///   Note: in self-play with both players writing into the same
    ///   buffer, each player's first record is ply 0, second is ply 1,
    ///   etc. The replay buffer's `plyIndex` field thus measures
    ///   "ply within this player's perspective" rather than the
    ///   absolute game ply. Game length (`gamePlyCountAtFlush`) is the
    ///   total recorded plies for this player at flush time.
    private var gamePlyIndices: [UInt16]

    /// Per-recorded-ply sampling temperature (tau) actually applied
    /// when picking the move. Computed via
    /// `schedule.tau(forPly: gamePliesRecorded)` at the moment of
    /// move selection.
    private var gameSamplingTaus: [Float]

    /// Per-recorded-ply 64-bit hash of the encoded board tensor at
    /// the time of move selection. Used by `ReplayBuffer` to track
    /// duplicate-position counts across the buffer.
    private var gameStateHashes: [UInt64]

    /// Per-recorded-ply non-pawn piece count (0–30). Drives the
    /// "phase by material" bucket of replay-buffer batch stats —
    /// independent of, and complementary to, the ply-based phase
    /// bucket. UInt8 is plenty (chess starts with 16 non-pawn).
    private var gameMaterialCounts: [UInt8]

    /// Stable identity for this player's owning self-play slot. Goes
    /// into the replay buffer's per-position metadata so per-batch
    /// stats can detect over-representation of any one slot. Not used
    /// for training. Default 0 — non-self-play callers (Play Game,
    /// arena) leave it at zero. UInt16 (rather than UInt8) so the
    /// monotonically-increasing slot counter in
    /// `BatchedSelfPlayDriver` doesn't pin everything to a single
    /// id after a few arena-respawn cycles.
    var workerId: UInt16 = 0

    /// Intra-worker monotonically increasing game counter.
    /// Incremented at the top of each `onNewGame` call. The (worker_id,
    /// intraWorkerGameIndex) pair is broadcast across every position
    /// of a single appended game.
    private var intraWorkerGameIndex: UInt32 = 0

    /// Number of plies this player has recorded in the current game.
    /// Indexes both `gameBoardScratchPtr` (by `ply * boardFloats`) and
    /// `gamePolicyIndices` (by `ply`).
    private var gamePliesRecorded: Int = 0

    /// Number of plies recorded so far in the current game — read by
    /// the Play and Train driver to report positions/sec stats.
    var recordedPliesCount: Int { gamePliesRecorded }

    /// Count of plies in the current game where the policy's legal-
    /// move softmax was effectively UNIFORM — i.e., the network had
    /// no meaningful opinion about which legal move to play, so the
    /// sampler picked essentially at random. A "random-ish" ply is
    /// defined as: max legal-move softmax probability is less than
    /// 1.5× the uniform probability `1/n_legal`. Flags:
    ///   - Collapsed policies (all mass on illegal cells → legal
    ///     logits all ~0 → softmax = uniform over legal)
    ///   - Freshly random-init networks (all legal logits similar
    ///     magnitude → softmax is close to uniform)
    /// A useful signal for "is the model actually choosing, or is
    /// every move effectively random?"
    private var gameRandomishMoveCount: Int = 0

    /// Read-only: number of essentially-uniform ("random-ish") plies
    /// this player has played in the current game so far. Drivers can
    /// query this after the game ends to report the random-ish rate.
    var recordedRandomishMoves: Int { gameRandomishMoveCount }

    /// Reusable softmax scratch used by `sampleMove`. Pre-sized to the
    /// max chess-position legal-move count so every per-ply sample runs
    /// without allocating a temporary `[Float]` for the gathered logits.
    private var sampleScratch: [Float]

    /// Reusable scratch for the per-move Dirichlet noise sample. Same
    /// capacity as `sampleScratch`, used only when
    /// `schedule.dirichletNoise != nil` and the current ply is below
    /// the configured ply limit. Held even when noise is disabled —
    /// the storage is trivial (1 KB at 256 floats) and avoids a
    /// branch on every player init.
    private var dirichletScratch: [Float]

    /// Create a player backed by a `MoveEvaluationSource`.
    ///
    /// For self-play, all slot players share one
    /// `BatchedMoveEvaluationSource` so N slot tasks' per-ply forward
    /// passes are coalesced into one batched `graph.run`. For arena,
    /// Play Game, and Play Continuous, pass a
    /// `DirectMoveEvaluationSource` wrapping a `ChessMPSNetwork` — see
    /// the `network:` convenience initializer below.
    ///
    /// Pass a `replayBuffer` to have this player contribute its labeled
    /// positions to a shared training pool at game end; leave it nil for
    /// normal (non-training) play. Pass a `schedule` other than `.uniform`
    /// to apply a two-phase sampling temperature — see `SamplingSchedule`
    /// and `sampling-parameters.md` for the rationale.
    init(
        name: String,
        source: MoveEvaluationSource,
        replayBuffer: ReplayBuffer? = nil,
        schedule: SamplingSchedule = .uniform
    ) {
        precondition(
            schedule.startTau > 0 && schedule.floorTau > 0 && schedule.decayPerPly >= 0,
            "SamplingSchedule: startTau and floorTau must be > 0, decayPerPly must be >= 0"
        )
        if let noise = schedule.dirichletNoise {
            precondition(
                noise.alpha > 0 && noise.epsilon >= 0 && noise.epsilon <= 1 && noise.plyLimit >= 0,
                "DirichletNoiseConfig: alpha must be > 0, epsilon in [0, 1], plyLimit >= 0"
            )
        }
        self.identifier = UUID().uuidString
        self.name = name
        self.source = source
        self.replayBuffer = replayBuffer
        self.schedule = schedule
        self.sampleScratch = [Float](repeating: 0, count: Self.sampleScratchCapacity)
        self.dirichletScratch = [Float](repeating: 0, count: Self.sampleScratchCapacity)

        let initialCapacity = Self.startingPlyCapacity
        let initialFloats = initialCapacity * Self.boardFloats
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: initialFloats)
        ptr.initialize(repeating: 0, count: initialFloats)
        self.gameBoardScratchPtr = ptr
        self.gameBoardScratchCapacity = initialCapacity

        let pPtr = UnsafeMutablePointer<Float>.allocate(capacity: Self.policyScratchCount)
        pPtr.initialize(repeating: 0, count: Self.policyScratchCount)
        self.policyScratchPtr = pPtr

        var indices: [Int32] = []
        indices.reserveCapacity(initialCapacity)
        self.gamePolicyIndices = indices

        var plies: [UInt16] = []
        plies.reserveCapacity(initialCapacity)
        self.gamePlyIndices = plies

        var taus: [Float] = []
        taus.reserveCapacity(initialCapacity)
        self.gameSamplingTaus = taus

        var hashes: [UInt64] = []
        hashes.reserveCapacity(initialCapacity)
        self.gameStateHashes = hashes

        var mats: [UInt8] = []
        mats.reserveCapacity(initialCapacity)
        self.gameMaterialCounts = mats
    }

    deinit {
        gameBoardScratchPtr.deinitialize(count: gameBoardScratchCapacity * Self.boardFloats)
        gameBoardScratchPtr.deallocate()
        policyScratchPtr.deinitialize(count: Self.policyScratchCount)
        policyScratchPtr.deallocate()
    }

    func onNewGame(_ isWhite: Bool) {
        self.isWhite = isWhite
        // Keep all backing storage — only reset the fill counts. The
        // per-ply write loop overwrites slot contents before reading
        // them, so there's no need to zero the scratch between games.
        gamePolicyIndices.removeAll(keepingCapacity: true)
        gamePlyIndices.removeAll(keepingCapacity: true)
        gameSamplingTaus.removeAll(keepingCapacity: true)
        gameStateHashes.removeAll(keepingCapacity: true)
        gameMaterialCounts.removeAll(keepingCapacity: true)
        gamePliesRecorded = 0
        gameRandomishMoveCount = 0
        intraWorkerGameIndex &+= 1
    }

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        guard !legalMoves.isEmpty else {
            throw ChessPlayerError.noLegalMoves
        }

        // Grow the per-game scratch if this ply would overflow the
        // currently-allocated ring. Amortized constant: doubling keeps
        // the total allocations at log2(plies/starting) across a game.
        if gamePliesRecorded >= gameBoardScratchCapacity {
            growGameBoardScratch(toPlyCapacity: gameBoardScratchCapacity * 2)
        }

        let boardFloats = Self.boardFloats
        let rowBase: UnsafeMutablePointer<Float> = gameBoardScratchPtr + gamePliesRecorded * boardFloats

        // Encode the current position directly into this ply's slot
        // in the scratch — zero allocation, no intermediate `[Float]`.
        let rowMutable = UnsafeMutableBufferPointer<Float>(start: rowBase, count: boardFloats)
        BoardEncoder.encode(gameState, into: rowMutable)

        // Feed the encoded bytes through the evaluation source. We
        // materialize `[Float]` from the per-ply scratch row because
        // the evaluation source is potentially actor-isolated and
        // raw `UnsafeBufferPointer` isn't `Sendable`. That copy is
        // tensorLength floats per move; the batcher used to make an
        // equivalent copy internally, so this just moves the copy
        // one actor hop earlier — net allocations are unchanged.
        //
        // The source writes `policySize` raw policy logits directly
        // into our per-player `policyScratchPtr` (handed in as
        // `intoPolicy`); it also returns the scalar `v(position)` from
        // the value head, but self-play no longer records that — the
        // trainer recomputes the policy-gradient baseline from a fresh
        // forward pass each step (the W/D/L value-head rewrite made the
        // play-time-frozen baseline dead), so the return value is
        // discarded here.
        let rowConst = UnsafeBufferPointer<Float>(rowMutable)
        let encoded = Array(rowConst)

        // Hash the encoded position BEFORE evaluation so the hash key
        // matches exactly what the network sees. Cheap (~5 µs at
        // `BoardEncoder.tensorLength` floats with stdlib SipHash).
        let stateHash = ReplayBuffer.hashBoard(rowBase, count: boardFloats)
        let plyTau = schedule.tau(forPly: gamePliesRecorded)
        // Non-pawn piece count for the by-material phase bucket.
        // Iterates the 64-element [Piece?] array and counts squares
        // occupied by anything other than a pawn. Explicit `if let`
        // (rather than for-where + force-unwrap) so the predicate
        // is unambiguous and there's zero force-unwrap risk if a
        // future GameState refactor changes the optionality model.
        var matCount: Int = 0
        for sq in gameState.board {
            if let piece = sq, // only squares containing a piece
                piece.type != .pawn {
                matCount += 1
            }
        }
        let materialCount = UInt8(min(matCount, Int(UInt8.max)))

        let policyDest = PolicyDestination(UnsafeMutableBufferPointer(
            start: policyScratchPtr,
            count: Self.policyScratchCount
        ))
        _ = try await source.evaluate(encodedBoard: encoded, intoPolicy: policyDest)
        let policyView = UnsafeBufferPointer(
            start: policyScratchPtr,
            count: Self.policyScratchCount
        )
        let move = sampleMove(
            from: policyView,
            legalMoves: legalMoves,
            currentPlayer: gameState.currentPlayer
        )

        gamePolicyIndices.append(Int32(PolicyEncoding.policyIndex(move, currentPlayer: gameState.currentPlayer)))
        // UInt16 caps at 65535 — chess games are at most ~6000 plies in
        // pathological cases (50-move/3-fold limits), so the cap is
        // fine; saturate just in case.
        gamePlyIndices.append(UInt16(min(gamePliesRecorded, Int(UInt16.max))))
        gameSamplingTaus.append(plyTau)
        gameStateHashes.append(stateHash)
        gameMaterialCounts.append(materialCount)
        gamePliesRecorded += 1

        return move
    }

    /// Protocol entry point. No-op in MPSChessPlayer — the replay-buffer
    /// flush is deferred to `flushRecordedGameToReplayBuffer(result:)`
    /// so the self-play driver can apply the per-game keep/drop
    /// decision (e.g. `selfPlayDrawKeepFraction`) AFTER it knows the
    /// game's result but BEFORE the bulk append lands. The recorded
    /// per-ply scratch survives this call; `onNewGame` clears it at
    /// the start of the next game whether or not anyone flushed.
    func onGameEnded(_ result: GameResult, finalState: GameState) {
        // Deliberately empty. See `flushRecordedGameToReplayBuffer`.
    }

    /// Bulk-flush every recorded ply from the just-finished game into
    /// the shared replay buffer with the (player-relative) outcome
    /// broadcast across every row. One lock acquisition on the buffer,
    /// one `memcpy`-style copy per field — no per-position round-trips.
    ///
    /// Returns the number of plies pushed (0 if `replayBuffer` is nil
    /// or this player recorded no plies this game — neither is a bug,
    /// just nothing to do).
    ///
    /// Safe to call only between `onGameEnded` and the next
    /// `onNewGame` — the per-game scratch is preserved across the gap,
    /// and `onNewGame` zeroes `gamePliesRecorded`. Calling twice in a
    /// row would double-push.
    @discardableResult
    func flushRecordedGameToReplayBuffer(result: GameResult) -> FlushedGameStats {
        guard let replayBuffer, gamePliesRecorded > 0 else { return .empty }

        let myOutcome: Float
        switch result {
        case .checkmate(let winner):
            myOutcome = (winner == .white) == isWhite ? 1.0 : -1.0
        case .stalemate,
             .drawByFiftyMoveRule,
             .drawByInsufficientMaterial,
             .drawByThreefoldRepetition:
            myOutcome = 0.0
        }

        let pliesPushed = gamePliesRecorded
        let gameLength = UInt16(min(pliesPushed, Int(UInt16.max)))

        // Per-game phase histograms (5 buckets each: open/early/mid/
        // late/end) computed by walking the recorded plies once.
        // Bucket cutoffs match `ReplayBuffer.computeBatchStats` so
        // the View > Emit Window aggregates use the same phase
        // semantics as the per-batch stats lines. Cost is one tight
        // loop over `pliesPushed` integers — negligible relative to
        // the buffer-copy below.
        var phaseByPly = PhaseHistogram.zero
        var phaseByMaterial = PhaseHistogram.zero
        for i in 0..<pliesPushed {
            let plyBucket = PhaseHistogram.plyBucket(ply: Int(gamePlyIndices[i]))
            let matBucket = PhaseHistogram.materialBucket(materialCount: Int(gameMaterialCounts[i]))
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
        gamePolicyIndices.withUnsafeBufferPointer { movesBuf in
            gamePlyIndices.withUnsafeBufferPointer { pliesBuf in
                gameSamplingTaus.withUnsafeBufferPointer { tausBuf in
                    gameStateHashes.withUnsafeBufferPointer { hashesBuf in
                        gameMaterialCounts.withUnsafeBufferPointer { matsBuf in
                            guard
                                let movesBase = movesBuf.baseAddress,
                                let pliesBase = pliesBuf.baseAddress,
                                let tausBase = tausBuf.baseAddress,
                                let hashesBase = hashesBuf.baseAddress,
                                let matsBase = matsBuf.baseAddress
                            else { return }
                            replayBuffer.append(
                                boards: UnsafePointer(gameBoardScratchPtr),
                                policyIndices: movesBase,
                                plyIndices: pliesBase,
                                samplingTaus: tausBase,
                                stateHashes: hashesBase,
                                materialCounts: matsBase,
                                gameLength: gameLength,
                                workerId: workerId,
                                intraWorkerGameIndex: intraWorkerGameIndex,
                                outcome: myOutcome,
                                count: pliesPushed
                            )
                        }
                    }
                }
            }
        }
        return FlushedGameStats(
            positions: pliesPushed,
            phaseByPly: phaseByPly,
            phaseByMaterial: phaseByMaterial
        )
    }

    /// Grow `gameBoardScratchPtr` to hold at least `newCapacity` plies,
    /// preserving the bytes already recorded for the current game. Only
    /// called when a game exceeds the previously-allocated capacity —
    /// does not run during normal-length games.
    private func growGameBoardScratch(toPlyCapacity newCapacity: Int) {
        precondition(newCapacity > gameBoardScratchCapacity,
            "growGameBoardScratch must strictly increase capacity")

        let oldFloats = gameBoardScratchCapacity * Self.boardFloats
        let newFloats = newCapacity * Self.boardFloats

        let newPtr = UnsafeMutablePointer<Float>.allocate(capacity: newFloats)
        newPtr.initialize(repeating: 0, count: newFloats)
        if gamePliesRecorded > 0 {
            newPtr.update(
                from: gameBoardScratchPtr,
                count: gamePliesRecorded * Self.boardFloats
            )
        }
        gameBoardScratchPtr.deinitialize(count: oldFloats)
        gameBoardScratchPtr.deallocate()
        gameBoardScratchPtr = newPtr
        gameBoardScratchCapacity = newCapacity
    }

    // MARK: - Move Sampling

    /// Sample a move from the policy distribution over legal moves.
    ///
    /// The network emits raw logits, not a softmax distribution — softmax is
    /// fused with the legal-move mask here on the CPU. We exponentiate only
    /// the ~30 legal-move logits (with max-subtract for numerical stability)
    /// rather than running softmax over all `policySize` slots and then masking.
    ///
    /// Per-move logit lookup goes through `PolicyEncoding.policyIndex`,
    /// which converts each legal `ChessMove` to its `(channel, row, col)`
    /// encoder-frame address and flat index in one call. The encoder-frame
    /// flip for black-to-move is internal to `PolicyEncoding` and matches
    /// what `BoardEncoder.encode` did to produce the inputs.
    ///
    /// Performance-critical: runs once per ply. The gathered logits, the
    /// in-place softmax, and the sampling all run through `sampleScratch`
    /// so the path is allocation-free. `legalMoves` is guaranteed
    /// non-empty by the caller (game-end is detected before this call).
    ///
    /// `logits` is a non-owning view over this player's own
    /// `policyScratchPtr`, which the evaluation source wrote into via
    /// the `intoPolicy` destination on the immediately-preceding
    /// `source.evaluate` call. The view is valid only for the
    /// duration of this call (until the next `onChooseNextMove`
    /// overwrites the same scratch).
    private func sampleMove(
        from logits: UnsafeBufferPointer<Float>,
        legalMoves: [ChessMove],
        currentPlayer: PieceColor
    ) -> ChessMove {
        let n = legalMoves.count
        // Captured here for the random-ish check below. Done early so
        // the closure can reference the value without re-reading `n`
        // after the scratch is overwritten.
        let uniformProb = 1 / Float(n)
        let randomishCutoff = 1.5 * uniformProb
        precondition(
            n <= sampleScratch.count,
            "MPSChessPlayer.sampleMove: legalMoves.count (\(n)) exceeds scratch capacity \(sampleScratch.count)"
        )

        // Temperature for this ply from the two-phase schedule. Applied
        // as `logit * (1 / tau)` during the gather below — identical to
        // `logit / tau` but cheaper, and the reciprocal is computed once.
        // Multiplying by 1 is exact in IEEE 754, so `.uniform` (tau=1.0)
        // with no Dirichlet noise reproduces the prior sampling behavior
        // bit-for-bit.
        let invTau = 1 / schedule.tau(forPly: gamePliesRecorded)

        // Decide once whether Dirichlet noise applies on this ply so
        // the inner loops avoid re-checking. Single-legal-move
        // positions skip the mix because there's nothing to redistribute.
        let activeNoise: DirichletNoiseConfig?
        if let cfg = schedule.dirichletNoise,
           gamePliesRecorded < cfg.plyLimit,
           n > 1 {
            activeNoise = cfg
        } else {
            activeNoise = nil
        }

        return sampleScratch.withUnsafeMutableBufferPointer { scratch in
            guard let base = scratch.baseAddress else {
                preconditionFailure("MPSChessPlayer.sampleMove: sampleScratch baseAddress is nil")
            }

            // Gather temperature-scaled logits for legal moves only.
            for i in 0..<n {
                scratch[i] = logits[PolicyEncoding.policyIndex(legalMoves[i], currentPlayer: currentPlayer)] * invTau
            }

            // Numerically stable softmax via Accelerate: subtract max,
            // exp, normalize into proper probabilities (sum to 1) so any
            // subsequent Dirichlet mix preserves normalization. Each
            // stage is a single vectorized pass over scratch[0..<n].
            let length = vDSP_Length(n)
            var maxLogit: Float = 0
            vDSP_maxv(base, 1, &maxLogit, length)
            var negMax = -maxLogit
            vDSP_vsadd(base, 1, &negMax, base, 1, length)
            var expCount = Int32(n)
            vvexpf(base, base, &expCount)
            var sum: Float = 0
            vDSP_sve(base, 1, &sum, length)
            // exp() is strictly positive, so sum is strictly positive
            // whenever legalMoves is non-empty.
            var invSum = 1 / sum
            vDSP_vsmul(base, 1, &invSum, base, 1, length)
            var maxProb: Float = 0
            vDSP_maxv(base, 1, &maxProb, length)

            // Record whether the post-temperature softmax over legal
            // moves is essentially uniform — i.e. the sampler was
            // picking at random, not acting on a network opinion. We
            // measure *before* Dirichlet noise because noise is a
            // deliberate exploration mix, not a policy-collapse signal.
            // With n == 1 the max prob is 1.0 and this always fails,
            // which is what we want: a forced move isn't random.
            if n > 1 && maxProb < randomishCutoff {
                gameRandomishMoveCount += 1
            }

            if let noise = activeNoise {
                Self.mixDirichletNoise(
                    into: scratch,
                    legalCount: n,
                    config: noise,
                    etaScratch: &dirichletScratch
                )
            }

            // Inverse-CDF sampling from probabilities in scratch[0..<n].
            let r = Float.random(in: 0..<1)
            var cumulative: Float = 0
            for i in 0..<n {
                cumulative += scratch[i]
                if r < cumulative {
                    return legalMoves[i]
                }
            }

            // Floating-point rounding can leave the cumulative just shy
            // of 1.0; the last legal move catches that.
            return legalMoves[n - 1]
        }
    }

    // MARK: - Dirichlet Noise

    /// Sample `η ~ Dir(α)` over `legalCount` entries (symmetric
    /// Dirichlet, all components share `config.alpha`) and mix into
    /// `probs` in place: `probs[i] = (1-ε) · probs[i] + ε · η[i]`.
    ///
    /// `probs` must already be a normalized probability vector — the
    /// mixture preserves normalization only when both inputs sum to 1.
    /// `etaScratch` is the caller's `dirichletScratch` storage; passed
    /// as `inout` so this static helper can use it without the player's
    /// instance pointer leaking into the closure.
    private static func mixDirichletNoise(
        into probs: UnsafeMutableBufferPointer<Float>,
        legalCount n: Int,
        config: DirichletNoiseConfig,
        etaScratch: inout [Float]
    ) {
        precondition(n <= etaScratch.count,
            "Dirichlet scratch (\(etaScratch.count)) too small for \(n) moves")
        etaScratch.withUnsafeMutableBufferPointer { eta in
            // Symmetric Dir(α) is `n` iid Gamma(α, 1) samples normalized
            // by their sum. With α < 1 the gamma samples are heavily
            // right-skewed and most of the noise mass concentrates on a
            // small random subset of moves — exactly the "spiky" noise
            // shape AlphaZero relies on.
            var gammaSum: Float = 0
            for i in 0..<n {
                let g = sampleGamma(alpha: config.alpha)
                eta[i] = g
                gammaSum += g
            }
            // Each gamma draw is strictly positive, so the sum is too;
            // no zero-sum guard needed. Normalize and mix.
            let invSum = 1 / gammaSum
            let oneMinusEps = 1 - config.epsilon
            let eps = config.epsilon
            for i in 0..<n {
                probs[i] = oneMinusEps * probs[i] + eps * (eta[i] * invSum)
            }
        }
    }

    /// Marsaglia–Tsang Gamma(α, 1) sampler. Supports any `α > 0`. For
    /// `α >= 1` runs the direct algorithm; for `α < 1` uses the boost
    /// trick: draw `G ~ Gamma(α+1, 1)` and return `G * U^(1/α)` where
    /// `U ~ Uniform(0, 1)`.
    @inline(__always)
    private static func sampleGamma(alpha: Float) -> Float {
        if alpha < 1 {
            let g = sampleGammaAtLeastOne(alpha: alpha + 1)
            // U must be strictly positive so U^(1/α) is finite.
            let u = max(Float.random(in: 0..<1), .leastNormalMagnitude)
            return g * powf(u, 1 / alpha)
        }
        return sampleGammaAtLeastOne(alpha: alpha)
    }

    /// Direct Marsaglia–Tsang (2000) algorithm for `α >= 1`. Average
    /// rejection rate is well below 5 % across α in [1, 10], so the
    /// inner `while true` typically exits on its first iteration.
    private static func sampleGammaAtLeastOne(alpha: Float) -> Float {
        let d: Float = alpha - 1.0 / 3.0
        let c: Float = 1 / sqrtf(9 * d)
        while true {
            // Inner reject loop guarantees `v > 0` so `v^3` and
            // `log(v)` are finite below.
            var x: Float = 0
            var v: Float = 0
            repeat {
                x = sampleStandardNormal()
                v = 1 + c * x
            } while v <= 0
            v = v * v * v
            let u = Float.random(in: 0..<1)
            // Squeeze test (cheap acceptance): handles ~98 % of draws
            // without touching log.
            let xx = x * x
            if u < 1 - 0.0331 * xx * xx {
                return d * v
            }
            // Full acceptance test. `u` here is uniform in (0, 1); on
            // the rare `u == 0` draw `log(0) = -inf` would force a
            // reject, which is the correct behavior — bias-free.
            if logf(u) < 0.5 * xx + d * (1 - v + logf(v)) {
                return d * v
            }
        }
    }

    /// Standard normal sample via Box–Muller. Discards one of the two
    /// iid draws per call; we run this ~30 times per ply per player so
    /// the wasted draw is negligible. `u1` is clamped away from zero so
    /// `log(u1)` is finite.
    @inline(__always)
    private static func sampleStandardNormal() -> Float {
        let u1 = max(Float.random(in: 0..<1), .leastNormalMagnitude)
        let u2 = Float.random(in: 0..<1)
        return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2)
    }
}
