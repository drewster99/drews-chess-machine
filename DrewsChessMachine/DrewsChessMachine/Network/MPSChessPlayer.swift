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

    /// Maximum **game-total** ply (0-indexed half-move count from the
    /// start of the game) at which noise is mixed in. At a game-total
    /// ply >= `plyLimit` the network's policy is sampled unmodified.
    /// Limits noise to the opening, which is where replay-buffer
    /// coverage matters most and where the network's own argmax tends
    /// to repeat. (Both sides count plies in the same single
    /// game-total sequence — there's no per-side notion of "ply" in
    /// this engine; "ply" always means a half-move within the game.)
    let plyLimit: Int

    /// AlphaZero baseline for chess (α=0.3, ε=0.25), gated to the
    /// first 30 game-total plies (i.e. the opening 15 white + 15
    /// black half-moves). Apply to self-play only — arena games
    /// should sample without exploration noise.
    static let alphaZero = DirichletNoiseConfig(
        alpha: 0.3,
        epsilon: 0.25,
        plyLimit: 30
    )
}

// MARK: - Sampling Schedule

/// Linear-decay temperature schedule applied by `MPSChessPlayer.sampleMove`.
///
/// Temperature starts at `startTau` on **game-total** ply 0 (the
/// starting position) and decreases by `decayPerPly` each game-total
/// ply (half-move from either side), bottoming out at `floorTau`.
/// The formula is: `tau(ply) = max(floorTau, startTau - decayPerPly * ply)`,
/// where `ply` is the position's game-total half-move count (0 for
/// the starting position, 1 after white's first move, 2 after black's
/// reply, etc.).
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
    /// Temperature on the starting position (game-total ply 0). Must
    /// be > 0.
    let startTau: Float
    /// Temperature reduction per game-total ply (i.e. each half-move
    /// advances the schedule). Must be >= 0.
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
    /// by 0.03 per game-total ply, flooring at 0.4 (reached at
    /// game-total ply ~54, i.e. ~27 half-moves into the game from
    /// either side). Higher starting temperature than the prior 1.0
    /// to flatten the policy more aggressively in the opening,
    /// broaden replay-buffer coverage of off-trunk lines, and pull
    /// more decisive games out of an early-bootstrap policy that
    /// otherwise concentrates on shuffle moves. Combined with the
    /// AlphaZero-default Dirichlet noise mixed into the opening 30
    /// game-total plies (the first 15 white + 15 black half-moves).
    static let selfPlay = SamplingSchedule(
        startTau: 2.0,
        decayPerPly: 0.03,
        floorTau: 0.4,
        dirichletNoise: .alphaZero
    )

    /// Arena-evaluation schedule: starts at tau=2.0, decays by 0.04
    /// per game-total ply, flooring at 0.2 (reached at game-total
    /// ply 45). Higher starting temperature than the prior 0.7 to
    /// expose more candidate-vs-champion divergence in the opening
    /// (helps surface the small signal that survives the dominant-
    /// draw regime). The faster decay and lower floor still ensure
    /// the middlegame onward reflects decisive play rather than
    /// sampling noise. No Dirichlet noise — arena games measure
    /// actual strength.
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

    /// The game-total ply at which tau reaches the floor. Returns
    /// `Int.max` when `decayPerPly` is zero (no decay).
    var pliesUntilFloor: Int {
        guard decayPerPly > 0 else { return Int.max }
        return Int(ceilf((startTau - floorTau) / decayPerPly))
    }

    /// Temperature to apply at the given game-total ply (0-indexed
    /// half-move count from the start of the game — 0 for the
    /// starting position, 1 after white's first move, 2 after
    /// black's reply, etc.).
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
///
/// Used by arena (two instances per game, sourced from per-network
/// `BatchedMoveEvaluationSource` actors) and by Play Game / Human-vs-
/// Network (one AI player against a `HumanChessPlayer`, sourced from
/// `DirectMoveEvaluationSource`). Self-play no longer uses this class
/// — the `BatchedSelfPlayDriver` tick model drives sampling directly
/// via `MoveSampler` without going through a per-game player object.
///
/// A player instance can be reused across many games (arena reuses
/// across an arena's run; Play Game allocates one per game). Access
/// is single-threaded within the game task; no locking is needed on
/// the player's internal state.
final class MPSChessPlayer: ChessPlayer {
    /// Number of floats in one encoded board position. Kept in sync
    /// with `BoardEncoder.tensorLength`.
    private static let boardFloats = BoardEncoder.tensorLength

    /// Upper bound on the number of legal moves in any chess position.
    /// The mathematical maximum is around 218, so 256 leaves a safety
    /// margin without wasting memory. Used to size `sampleScratch`.
    private static let sampleScratchCapacity = 256

    let identifier: String
    let name: String
    /// Source of (policy, value) predictions. Goes through
    /// `DirectMoveEvaluationSource` (sync single-board inference) for
    /// Play Game / Play Continuous / Human-vs-Network, or through
    /// `BatchedMoveEvaluationSource` (barrier batcher, per-network)
    /// for arena tournaments.
    private let source: MoveEvaluationSource
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

    // MARK: - Per-game state

    /// Reusable destination for the per-ply policy readback. Manually
    /// allocated once at init and freed at deinit; the
    /// `MoveEvaluationSource` writes `ChessNetwork.policySize` floats
    /// into it via `update(from:count:)` on every `source.evaluate`
    /// call (no Swift `[Float]` allocation per call). One ply runs at
    /// a time per player (white then black, serialized by
    /// `ChessMachine.runGameLoop`), so this scratch is never aliased
    /// by concurrent calls.
    private let policyScratchPtr: UnsafeMutablePointer<Float>
    /// Element count of `policyScratchPtr`. Captured once so the
    /// `init` allocation and the `deinit` deinitialize/deallocate
    /// pair stay in sync, and the per-ply call site doesn't have to
    /// reach into `ChessNetwork.policySize` directly.
    private static let policyScratchCount = ChessNetwork.policySize

    /// Number of plies this player has chosen in the current game.
    /// Used as the per-side ply count from which the game-total ply
    /// index is derived for `MoveSampler`'s tau schedule + Dirichlet
    /// gate (`2 * gamePliesRecorded + (isWhite ? 0 : 1)`).
    private var gamePliesRecorded: Int = 0

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
    /// For arena, pass a `BatchedMoveEvaluationSource` (per-network)
    /// so multiple concurrent arena games coalesce their per-ply
    /// forward passes into batched `graph.run` calls. For Play Game /
    /// Human-vs-Network, pass a `DirectMoveEvaluationSource` wrapping
    /// a `ChessMPSNetwork`. Pass a `schedule` other than `.uniform`
    /// to apply a two-phase sampling temperature — see
    /// `SamplingSchedule` and `sampling-parameters.md`.
    init(
        name: String,
        source: MoveEvaluationSource,
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
        self.schedule = schedule
        self.sampleScratch = [Float](repeating: 0, count: Self.sampleScratchCapacity)
        self.dirichletScratch = [Float](repeating: 0, count: Self.sampleScratchCapacity)

        let pPtr = UnsafeMutablePointer<Float>.allocate(capacity: Self.policyScratchCount)
        pPtr.initialize(repeating: 0, count: Self.policyScratchCount)
        self.policyScratchPtr = pPtr
    }

    deinit {
        policyScratchPtr.deinitialize(count: Self.policyScratchCount)
        policyScratchPtr.deallocate()
    }

    func onNewGame(_ isWhite: Bool) {
        self.isWhite = isWhite
        gamePliesRecorded = 0
    }

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        guard !legalMoves.isEmpty else {
            throw ChessPlayerError.noLegalMoves
        }

        // Encode the current position. One `[Float]` allocation per
        // ply — the MoveEvaluationSource API takes `[Float]` (because
        // crossing the batcher actor needs a Sendable value), so we
        // can't hand it a raw pointer. At arena's ~40 plies/sec
        // across K games this is microseconds of allocator pressure;
        // not worth a per-instance scratch buffer.
        var encoded = [Float](repeating: 0, count: Self.boardFloats)
        encoded.withUnsafeMutableBufferPointer { buf in
            BoardEncoder.encode(gameState, into: buf)
        }

        // Run inference. The source writes `policySize` raw policy
        // logits into `policyScratchPtr` via the `intoPolicy`
        // destination; it also returns the value-head scalar, but
        // arena / Play Game don't consume it (the trainer recomputes
        // the policy-gradient baseline from a fresh forward pass each
        // step — see the W/D/L value-head rewrite for context), so
        // the return is discarded.
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
        gamePliesRecorded += 1
        return move
    }

    /// Protocol entry point. No-op in MPSChessPlayer — Play Game /
    /// Human play / arena don't ingest into the replay buffer (only
    /// the tick driver does, and it doesn't use `MPSChessPlayer`).
    func onGameEnded(_ result: GameResult, finalState: GameState) {
        // Deliberately empty.
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
        let result = sampleScratch.withUnsafeMutableBufferPointer { probs in
            dirichletScratch.withUnsafeMutableBufferPointer { eta in
                MoveSampler.sampleMove(
                    logits: logits,
                    legalMoves: legalMoves,
                    currentPlayer: currentPlayer,
                    // Game-total ply count for the position currently
                    // under consideration (0-indexed half-move count
                    // from game start). `gamePliesRecorded` is this
                    // player's own per-side move count; white plays
                    // even half-moves (0, 2, 4, …) and black plays
                    // odd (1, 3, 5, …), so the position about to be
                    // sampled is at game-total ply
                    // `2 * gamePliesRecorded + (isWhite ? 0 : 1)`.
                    // Drives both `schedule.tau(forPly:)` and the
                    // Dirichlet ply-limit gate, both expressed in
                    // game-total ply terms.
                    ply: 2 * gamePliesRecorded + (isWhite ? 0 : 1),
                    schedule: schedule,
                    probsScratch: probs,
                    etaScratch: eta
                )
            }
        }
        return result.move
    }
}
