import Foundation

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

    /// Self-play data-generation schedule: starts at tau=1.0, decays
    /// by 0.03 per ply, flooring at 0.4 (reached at ply 20). The
    /// smooth ramp keeps early-game diversity for replay-buffer
    /// coverage while gradually tightening toward decisive play.
    static let selfPlay = SamplingSchedule(
        startTau: 1.0,
        decayPerPly: 0.03,
        floorTau: 0.4
    )

    /// Arena-evaluation schedule: starts at tau=1.0, decays by 0.04
    /// per ply, flooring at 0.2 (reached at ply 20). Faster decay
    /// and lower floor than self-play so most of the game reflects
    /// the networks' actual preferences rather than sampling noise.
    static let arena = SamplingSchedule(
        startTau: 1.0,
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
/// self-play hot path allocates no per-move `[Float](1152)` tensors; at
/// game end the whole buffer is pushed into the shared `ReplayBuffer` in
/// one bulk copy.
///
/// Each `ChessMachine` game creates fresh `MPSChessPlayer` instances, so
/// the scratches live for one game. Access is single-threaded within the
/// game task; no locking is needed on the player's internal state.
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
    private let network: ChessMPSNetwork
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

    /// Policy-target indices (0–4095) for each recorded ply, in the
    /// network's flipped coordinate system. Pre-reserved so growth is
    /// rare. Grown automatically if the game exceeds the initial
    /// capacity.
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

    /// Create a player backed by a neural network.
    /// For self-play, two players can share the same network — they take turns,
    /// and BoardEncoder.encode handles the perspective flip automatically.
    /// Pass a `replayBuffer` to have this player contribute its labeled
    /// positions to a shared training pool at game end; leave it nil for
    /// normal (non-training) play. Pass a `schedule` other than `.uniform`
    /// to apply a two-phase sampling temperature — see `SamplingSchedule`
    /// and `sampling-parameters.md` for the rationale.
    init(
        name: String,
        network: ChessMPSNetwork,
        replayBuffer: ReplayBuffer? = nil,
        schedule: SamplingSchedule = .uniform
    ) {
        precondition(
            schedule.startTau > 0 && schedule.floorTau > 0 && schedule.decayPerPly >= 0,
            "SamplingSchedule: startTau and floorTau must be > 0, decayPerPly must be >= 0"
        )
        self.identifier = UUID().uuidString
        self.name = name
        self.network = network
        self.replayBuffer = replayBuffer
        self.schedule = schedule
        self.sampleScratch = [Float](repeating: 0, count: Self.sampleScratchCapacity)

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

        let flip = gameState.currentPlayer == .black

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

        // Feed the same bytes to the network. `policy` is a non-owning
        // view over the network's readback scratch — we consume it in
        // `sampleMove` before returning, and never touch it again.
        // `value` is the scalar v(position) from the value head; we
        // stash it as the advantage baseline so the training policy
        // loss can compute `(z − vBaseline) * −log p(a*)` without
        // paying for a second forward pass.
        let rowConst = UnsafeBufferPointer<Float>(rowMutable)
        let (policy, value) = try network.evaluate(board: rowConst)
        let move = sampleMove(from: policy, legalMoves: legalMoves, flip: flip)

        gamePolicyIndices.append(Int32(Self.networkPolicyIndex(for: move, flip: flip)))
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

    // MARK: - Coordinate Mapping

    /// Convert a move's absolute coordinates to the network's policy index.
    ///
    /// The network always sees the board from the current player's perspective.
    /// When black plays, BoardEncoder flips rows (row → 7-row). The policy output
    /// uses this same flipped coordinate system, so move indices must be flipped
    /// to match. Files (columns) stay unchanged — only ranks flip.
    private static func networkPolicyIndex(for move: ChessMove, flip: Bool) -> Int {
        let fromRow = flip ? (7 - move.fromRow) : move.fromRow
        let toRow = flip ? (7 - move.toRow) : move.toRow
        let fromSquare = fromRow * 8 + move.fromCol
        let toSquare = toRow * 8 + move.toCol
        return fromSquare * 64 + toSquare
    }

    // MARK: - Move Sampling

    /// Sample a move from the policy distribution over legal moves.
    ///
    /// The network emits raw logits, not a softmax distribution — softmax is
    /// fused with the legal-move mask here on the CPU. We exponentiate only
    /// the ~30 legal-move logits (with max-subtract for numerical stability)
    /// rather than running softmax over all 4096 slots and then masking.
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
        flip: Bool
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
        // reproduces the prior sampling behavior bit-for-bit.
        let invTau = 1 / schedule.tau(forPly: gamePliesRecorded)

        return sampleScratch.withUnsafeMutableBufferPointer { scratch in
            // Gather logits for legal moves only.
            for i in 0..<n {
                scratch[i] = logits[Self.networkPolicyIndex(for: legalMoves[i], flip: flip)] * invTau
            }

            // Numerically stable softmax: subtract max, exp, normalize.
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
            // exp() is strictly positive, so sum is strictly positive whenever
            // legalMoves is non-empty.
            let invSum = 1 / sum

            let r = Float.random(in: 0..<1)
            var cumulative: Float = 0
            for i in 0..<n {
                cumulative += scratch[i] * invSum
                if r < cumulative {
                    return legalMoves[i]
                }
            }

            // Floating-point rounding can leave the cumulative just shy of 1.0;
            // the last legal move catches that.
            return legalMoves[n - 1]
        }
    }
}
