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
    /// Optional replay buffer. When non-nil, this player pushes each finished
    /// game's labeled positions into the buffer from `onGameEnded`. The
    /// default-nil path is what Play Game and Play Continuous use — they
    /// have no use for training data, and passing nil keeps their behavior
    /// identical to before the buffer existed.
    private let replayBuffer: ReplayBuffer?
    /// Temperature schedule applied per-ply in `sampleMove`. Defaults to
    /// `.uniform` (flat tau=1.0) so non-training callers keep their
    /// pre-schedule behavior; self-play and arena pass their own presets.
    private let schedule: SamplingSchedule
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

    /// Policy-target indices for each recorded ply, computed via
    /// `PolicyEncoding.policyIndex` in the network's encoder-frame
    /// coordinate system (0..<policySize, currently 0..<4864). Pre-
    /// reserved so growth is rare; grown automatically if the game
    /// exceeds the initial capacity.
    private var gamePolicyIndices: [Int32]

    /// Inference-time value estimate `v(position)` for each recorded
    /// ply, captured from the same forward pass that already runs to
    /// pick the move. Bulk-flushed into `ReplayBuffer` at game end as
    /// the advantage-baseline column. Same `ply` indexing as
    /// `gamePolicyIndices`.
    private var gameValueScalars: [Float]

    /// Number of plies this player has recorded in the current game.
    /// Indexes both `gameBoardScratchPtr` (by `ply * boardFloats`) and
    /// `gamePolicyIndices` (by `ply`).
    private var gamePliesRecorded: Int = 0

    /// Number of plies recorded so far in the current game — read by
    /// the Play and Train driver to report positions/sec stats.
    var recordedPliesCount: Int { gamePliesRecorded }

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

        var indices: [Int32] = []
        indices.reserveCapacity(initialCapacity)
        self.gamePolicyIndices = indices

        var values: [Float] = []
        values.reserveCapacity(initialCapacity)
        self.gameValueScalars = values
    }

    deinit {
        gameBoardScratchPtr.deinitialize(count: gameBoardScratchCapacity * Self.boardFloats)
        gameBoardScratchPtr.deallocate()
    }

    func onNewGame(_ isWhite: Bool) {
        self.isWhite = isWhite
        // Keep all backing storage — only reset the fill counts. The
        // per-ply write loop overwrites slot contents before reading
        // them, so there's no need to zero the scratch between games.
        gamePolicyIndices.removeAll(keepingCapacity: true)
        gameValueScalars.removeAll(keepingCapacity: true)
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
        // The returned `policy` is a fresh, caller-owned `[Float]`
        // of `policySize` raw logits — batchers can reuse their readback
        // scratch on the next batch without invalidating this
        // array. `value` is the scalar `v(position)` from the value
        // head; we stash it as the advantage baseline so the
        // training policy loss can compute
        // `(z − vBaseline) * −log p(a*)` without paying for a
        // second forward pass.
        let rowConst = UnsafeBufferPointer<Float>(rowMutable)
        let encoded = Array(rowConst)
        let (policy, value) = try await source.evaluate(encodedBoard: encoded)
        let move = policy.withUnsafeBufferPointer { policyBuf in
            sampleMove(from: policyBuf, legalMoves: legalMoves, currentPlayer: gameState.currentPlayer)
        }

        gamePolicyIndices.append(Int32(PolicyEncoding.policyIndex(move, currentPlayer: gameState.currentPlayer)))
        gameValueScalars.append(value)
        gamePliesRecorded += 1

        return move
    }

    func onGameEnded(_ result: GameResult, finalState: GameState) {
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

        guard let replayBuffer, gamePliesRecorded > 0 else { return }

        // Bulk-flush every recorded ply into the shared replay buffer
        // with the now-known outcome broadcast across every row. One
        // lock acquisition, one `memcpy`-style copy per field — no
        // per-position round-trips.
        gamePolicyIndices.withUnsafeBufferPointer { movesBuf in
            gameValueScalars.withUnsafeBufferPointer { valuesBuf in
                guard
                    let movesBase = movesBuf.baseAddress,
                    let valuesBase = valuesBuf.baseAddress
                else { return }
                replayBuffer.append(
                    boards: UnsafePointer(gameBoardScratchPtr),
                    policyIndices: movesBase,
                    vBaselines: valuesBase,
                    outcome: myOutcome,
                    count: gamePliesRecorded
                )
            }
        }
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
    /// `logits` is a non-owning view over the network's policy readback
    /// scratch; it is valid only for the duration of this call.
    private func sampleMove(
        from logits: UnsafeBufferPointer<Float>,
        legalMoves: [ChessMove],
        currentPlayer: PieceColor
    ) -> ChessMove {
        let n = legalMoves.count
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
            // Gather temperature-scaled logits for legal moves only.
            for i in 0..<n {
                scratch[i] = logits[PolicyEncoding.policyIndex(legalMoves[i], currentPlayer: currentPlayer)] * invTau
            }

            // Numerically stable softmax: subtract max, exp, normalize
            // into proper probabilities (sum to 1) so any subsequent
            // Dirichlet mix preserves normalization.
            var maxLogit = scratch[0]
            for i in 1..<n where scratch[i] > maxLogit {
                maxLogit = scratch[i]
            }
            var sum: Float = 0
            for i in 0..<n {
                let e = expf(scratch[i] - maxLogit)
                scratch[i] = e
                sum += e
            }
            // exp() is strictly positive, so sum is strictly positive
            // whenever legalMoves is non-empty.
            let invSum = 1 / sum
            for i in 0..<n {
                scratch[i] *= invSum
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
