import Accelerate
import Foundation

/// Pure, allocation-free move sampling from a policy logit slice.
///
/// Lifts the per-ply sampling math out of `MPSChessPlayer` so the
/// upcoming tick-based self-play driver can drive sampling for K
/// games per tick across a `withTaskGroup` of CPU workers without
/// also instantiating K `MPSChessPlayer`s (each of which would
/// allocate per-instance scratch). `MPSChessPlayer.sampleMove` now
/// delegates here, passing its own pre-allocated scratch buffers
/// — so arena and Play Game / Human-vs-Network keep exactly the
/// same sampling behavior.
///
/// Caller contract:
///
/// - `logits` is a non-owning view over a single position's raw policy
///   slice (`ChessNetwork.policySize` floats). Typically a slice of a
///   batched-eval output buffer (tick driver) or a per-player
///   policyScratch buffer (legacy path).
/// - `legalMoves` is the current position's legal moves; must be
///   non-empty (game-end is detected before this call).
/// - `currentPlayer` matches the player whose turn it is — used to
///   pick the right encoder-frame flip inside
///   `PolicyEncoding.policyIndex`.
/// - `ply` is the side-relative ply index (0 for this player's first
///   move, 1 for their second, etc.) — drives both the tau schedule
///   and the Dirichlet ply-limit check.
/// - `schedule` is the active sampling schedule (tau curve, optional
///   Dirichlet config).
/// - `probsScratch` and `etaScratch` are both at least `scratchCapacity`
///   floats long. The function uses `probsScratch[0..<n]` for the
///   gathered legal-only softmax probabilities and `etaScratch[0..<n]`
///   for Dirichlet noise samples (only when `schedule.dirichletNoise`
///   is active for this ply). Contents are not preserved across calls
///   — caller can reuse the same scratches for the next call.
///
/// Returns the sampled move, its raw policy index (so the caller can
/// record it without re-calling `PolicyEncoding.policyIndex`), and a
/// `randomish` flag (true when the post-temperature legal-only
/// softmax was essentially uniform — i.e. the sampler was picking
/// at random, not acting on a network opinion). The flag is computed
/// pre-Dirichlet so Dirichlet noise doesn't mask a flat-policy signal.
enum MoveSampler {

    /// Upper bound on the number of legal moves in any chess position.
    /// The mathematical maximum is around 218, so 256 leaves a safety
    /// margin. Used to size `probsScratch` and `etaScratch`.
    static let scratchCapacity = 256

    struct Result {
        let move: ChessMove
        let policyIndex: Int
        let randomish: Bool
    }

    static func sampleMove(
        logits: UnsafeBufferPointer<Float>,
        legalMoves: [ChessMove],
        currentPlayer: PieceColor,
        ply: Int,
        schedule: SamplingSchedule,
        probsScratch: UnsafeMutableBufferPointer<Float>,
        etaScratch: UnsafeMutableBufferPointer<Float>
    ) -> Result {
        let n = legalMoves.count
        precondition(n > 0, "MoveSampler.sampleMove: legalMoves must be non-empty")
        precondition(
            n <= probsScratch.count,
            "MoveSampler.sampleMove: legalMoves.count (\(n)) exceeds probsScratch capacity \(probsScratch.count)"
        )
        precondition(
            n <= etaScratch.count,
            "MoveSampler.sampleMove: legalMoves.count (\(n)) exceeds etaScratch capacity \(etaScratch.count)"
        )

        let uniformProb = 1 / Float(n)
        let randomishCutoff = 1.5 * uniformProb

        // `logit * (1 / tau)` is identical to `logit / tau` but cheaper;
        // reciprocal computed once. Multiplying by 1 is exact in IEEE 754
        // so `.uniform` (tau=1.0) with no Dirichlet noise reproduces
        // the prior sampling behavior bit-for-bit.
        let invTau = 1 / schedule.tau(forPly: ply)

        // Decide once whether Dirichlet noise applies on this ply so
        // the inner loops avoid re-checking. Single-legal-move
        // positions skip the mix because there's nothing to redistribute.
        let activeNoise: DirichletNoiseConfig?
        if let cfg = schedule.dirichletNoise,
           ply < cfg.plyLimit,
           n > 1 {
            activeNoise = cfg
        } else {
            activeNoise = nil
        }

        guard let probsBase = probsScratch.baseAddress else {
            preconditionFailure("MoveSampler.sampleMove: probsScratch baseAddress is nil")
        }

        // Cache each legal move's policy index — both for the gather
        // loop below AND for the Result we return (so the caller can
        // record the chosen move's index without re-calling
        // PolicyEncoding.policyIndex). 256-entry stack array dodges
        // any heap allocation.
        var policyIndices = [Int](repeating: 0, count: n)
        for i in 0..<n {
            policyIndices[i] = PolicyEncoding.policyIndex(legalMoves[i], currentPlayer: currentPlayer)
        }

        // Gather temperature-scaled logits for legal moves only.
        for i in 0..<n {
            probsBase[i] = logits[policyIndices[i]] * invTau
        }

        // Numerically stable softmax via Accelerate: subtract max,
        // exp, normalize into proper probabilities (sum to 1) so any
        // subsequent Dirichlet mix preserves normalization. Each
        // stage is a single vectorized pass over probsScratch[0..<n].
        let length = vDSP_Length(n)
        var maxLogit: Float = 0
        vDSP_maxv(probsBase, 1, &maxLogit, length)
        var negMax = -maxLogit
        vDSP_vsadd(probsBase, 1, &negMax, probsBase, 1, length)
        var expCount = Int32(n)
        vvexpf(probsBase, probsBase, &expCount)
        var sum: Float = 0
        vDSP_sve(probsBase, 1, &sum, length)
        // exp() is strictly positive, so sum is strictly positive
        // whenever legalMoves is non-empty.
        var invSum = 1 / sum
        vDSP_vsmul(probsBase, 1, &invSum, probsBase, 1, length)
        var maxProb: Float = 0
        vDSP_maxv(probsBase, 1, &maxProb, length)

        // Random-ish detection: post-temperature softmax over legal
        // moves is essentially uniform — i.e. the sampler is picking
        // at random, not acting on a network opinion. Measured
        // *before* Dirichlet noise because noise is a deliberate
        // exploration mix, not a policy-collapse signal. With n == 1
        // the max prob is 1.0 and this always fails, which is what
        // we want: a forced move isn't random.
        let randomish = n > 1 && maxProb < randomishCutoff

        if let noise = activeNoise {
            mixDirichletNoise(
                into: probsScratch,
                legalCount: n,
                config: noise,
                etaScratch: etaScratch
            )
        }

        // Inverse-CDF sampling from probabilities in probsScratch[0..<n].
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<n {
            cumulative += probsBase[i]
            if r < cumulative {
                return Result(
                    move: legalMoves[i],
                    policyIndex: policyIndices[i],
                    randomish: randomish
                )
            }
        }

        // Floating-point rounding can leave the cumulative just shy
        // of 1.0; the last legal move catches that.
        return Result(
            move: legalMoves[n - 1],
            policyIndex: policyIndices[n - 1],
            randomish: randomish
        )
    }

    // MARK: - Dirichlet Noise

    /// Sample `η ~ Dir(α)` over `legalCount` entries (symmetric
    /// Dirichlet, all components share `config.alpha`) and mix into
    /// `probs` in place: `probs[i] = (1-ε) · probs[i] + ε · η[i]`.
    ///
    /// `probs` must already be a normalized probability vector — the
    /// mixture preserves normalization only when both inputs sum to 1.
    /// `etaScratch` is the caller's pre-allocated η storage; the
    /// caller can reuse it across calls without zeroing first.
    private static func mixDirichletNoise(
        into probs: UnsafeMutableBufferPointer<Float>,
        legalCount n: Int,
        config: DirichletNoiseConfig,
        etaScratch: UnsafeMutableBufferPointer<Float>
    ) {
        guard let probsBase = probs.baseAddress, let etaBase = etaScratch.baseAddress else {
            preconditionFailure("MoveSampler.mixDirichletNoise: baseAddress is nil")
        }

        // Symmetric Dir(α) is `n` iid Gamma(α, 1) samples normalized
        // by their sum. With α < 1 the gamma samples are heavily
        // right-skewed and most of the noise mass concentrates on a
        // small random subset of moves — exactly the "spiky" noise
        // shape AlphaZero relies on.
        var gammaSum: Float = 0
        for i in 0..<n {
            let g = sampleGamma(alpha: config.alpha)
            etaBase[i] = g
            gammaSum += g
        }
        // Each gamma draw is strictly positive, so the sum is too;
        // no zero-sum guard needed. Normalize and mix.
        let invSum = 1 / gammaSum
        let oneMinusEps = 1 - config.epsilon
        let eps = config.epsilon
        for i in 0..<n {
            probsBase[i] = oneMinusEps * probsBase[i] + eps * (etaBase[i] * invSum)
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
