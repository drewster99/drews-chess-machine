import Foundation

// MARK: - Replay Buffer Analyzer
//
// Hypothesis-testing tool for the "why is the network stuck on flat
// endgame policies" question. Given a `ReplayBuffer` snapshot, walks
// every slot once and produces six histograms / cross-tabs designed to
// distinguish three competing stories about what the buffer actually
// teaches:
//
//   1. Channel histogram (76 buckets): does the network ever play
//      long-distance sliding moves at all? If queen-distance-6/7 are
//      essentially absent from the buffer, the on-policy bias never
//      gets a chance to be undone.
//
//   2. Channel × material-bucket: strips out the "long slides are
//      mostly illegal in middle-game" confound. If long-slide channels
//      stay flat even on sparse-material positions, the bias is genuine,
//      not just a legality artifact.
//
//   3. Final-move channel given outcome: of games that ended in
//      checkmate, what move-type delivered the mate? If long-slide
//      channels show meaningful mass here, the bootstrap *is* closing
//      (just slowly); if absent, it has never closed for those motifs.
//
//   4. End-of-game material × outcome: of games that reached a sparse-
//      material final position, what fraction were max-plies-truncated
//      vs. an actual terminal? Cleanest test of "the network reaches
//      winning endgames and fails to convert them."
//
//   5. Game-length × outcome: quantifies how much draw mass comes from
//      the 300/500-ply cap vs. real chess-rules draws.
//
//   6. Material-signature classes: top-N (mover-pieces, opponent-pieces)
//      multisets among games whose final position is in the buffer.
//      Surfaces "KQ_vs_K" etc. as their own buckets with outcome splits,
//      directly answering "how many K+Q-vs-K games does the trainer
//      even see, and how many of them end in mate?".
//
// All analyses operate over a single `ReplayBuffer.withSlotData` pass.
// The analyzer is pure — no UI, no Metal, no logging, no I/O — so both
// the CLI (--analyze-replay-buffer) and the Debug menu item can call it
// without sharing any infrastructure beyond the buffer itself.

enum ReplayBufferAnalyzer {

    // MARK: - Bucket definitions

    /// Inclusive `[low, high]` non-pawn-piece-count ranges for the
    /// material-bucket cross-tab. Cuts chosen so K+1-piece-vs-K
    /// endgames (bucket 0) separate cleanly from "real" endgames
    /// (bucket 1) from middlegames (the upper three buckets).
    static let materialBuckets: [(low: Int, high: Int, label: String)] = [
        (0,  4,  "0-4"),
        (5,  8,  "5-8"),
        (9,  14, "9-14"),
        (15, 22, "15-22"),
        (23, 30, "23-30")
    ]

    /// Game-length histogram bucket size (in plies). 11 buckets cover
    /// 0..549 + a trailing "550+" tail; 500 is the project's hard cap on
    /// self-play game length, so the tail bucket should always be empty
    /// in practice but exists as a safety valve against future cap raises.
    static let gameLengthBucketSize = 50
    static let gameLengthBucketCount = 11

    /// Labels matching the three slots of any outcome-indexed array.
    /// Index 0 = win for the mover at this position, 1 = draw, 2 = loss.
    static let outcomeLabels = ["win", "draw", "loss"]

    /// Maximum number of distinct material-signature classes surfaced in
    /// `Result.topMaterialSignatures`. Keeps the result struct bounded
    /// even on buffers with thousands of distinct signatures.
    static let topMaterialSignatureLimit = 40

    /// Percentiles (in 0..100) reported per bucket in the policy-entropy
    /// probe's `entropyPercentilesNats` array. JSON consumers and the
    /// text-summary formatter both rely on `entropyPercentilesNats[i]`
    /// corresponding to `entropyPercentileLabels[i]`.
    static let entropyPercentileLabels: [Int] = [10, 50, 90]

    /// `K` values used by the policy-entropy probe's
    /// `meanTopKLegalMass` array. `meanTopKLegalMass[i]` is the mean,
    /// over sampled positions in this bucket, of the renormalized
    /// legal-only probability mass on the top-`topKLabels[i]` moves.
    /// Bounded by the number of legal moves at the position — a probe
    /// of "top-5 mass" in a position with 3 legal moves reads 1.0.
    static let topKLabels: [Int] = [1, 3, 5]

    /// Tau used for the "self-play projection" metrics
    /// (`meanEntropyNatsAtSelfPlayTau`, `meanTopKLegalMassAtSelfPlayTau`).
    /// Matches `SamplingSchedule.selfPlay.floorTau` — the steady-state
    /// value the self-play sampler decays to once the opening tau decay
    /// has bottomed out. Reported into the JSON as `selfPlayTauUsed`
    /// per bucket so downstream readers know which tau was applied.
    static let selfPlayTauForProjection: Double = 0.4

    /// Number of equal-size slices the entropy-conditioned channel
    /// histogram splits sampled positions into. Slice 0 is the sharpest
    /// (lowest entropy), slice `entropySliceLabels.count - 1` is the
    /// flattest. Labels and slice count must stay in sync.
    static let entropySliceLabels: [String] = ["sharp", "mid", "flat"]

    /// Percentile boundaries computed from `entropySliceLabels.count`.
    /// For 3 slices: cut at p33 and p67. For N slices: cuts at p(100·i/N)
    /// for i in 1..<N. Used internally by the entropy-slice splitter
    /// to map sorted positions to slice indices.
    static var entropySliceBoundaryPercentiles: [Double] {
        let n = entropySliceLabels.count
        guard n >= 2 else { return [] }
        return (1..<n).map { 100.0 * Double($0) / Double(n) }
    }

    // MARK: - Result struct

    struct Result: Codable, Sendable {

        struct MaterialClassEntry: Codable, Sendable {
            let signature: String
            let gameCount: Int
            let winCount: Int
            let drawCount: Int
            let lossCount: Int
        }

        /// Stratified policy-entropy probe result for one material
        /// bucket. Produced by `runWithPolicyEntropy(...)`, which
        /// forward-passes a random sample of positions per bucket
        /// through a live network. `nil`-entropy fields are not
        /// reported because the bucket had no positions to sample.
        /// 76-channel histogram + position count for one entropy slice
        /// of one material bucket. Produced for analysis (8) "channel
        /// histogram by entropy slice" — surfaces whether the network's
        /// argmax move family differs between its sharp-policy positions
        /// and its flat-policy ones.
        struct ChannelHistogramSlice: Codable, Sendable {
            /// Slice label (matches `entropySliceLabels[sliceIndex]`).
            let sliceLabel: String
            /// Number of positions falling into this entropy slice.
            let positionCount: Int
            /// 76-channel histogram of `arg max p_renorm(legal)` —
            /// the network's most-favored legal move per position.
            let topMoveChannelCounts: [Int]
        }

        struct PolicyEntropyBucketStat: Codable, Sendable {
            /// Material-bucket label (matches `materialBuckets[i].label`),
            /// duplicated into the entry so this list is self-describing
            /// when read in isolation from the JSON.
            let bucketLabel: String
            /// Number of positions actually forward-passed for this
            /// bucket. May be less than `perBucketTarget` when the
            /// bucket contains fewer positions than the target.
            let sampleCount: Int
            /// Mean Shannon entropy (in nats) of the legal-masked
            /// renormalized policy softmax, averaged over `sampleCount`
            /// positions in this bucket. Compare to
            /// `meanUniformEntropyNats` for "how close to uniform."
            /// The value here is the *underlying* policy at tau=1.0;
            /// see `meanEntropyNatsAtSelfPlayTau` for the post-tau
            /// self-play sampling projection.
            let meanEntropyNats: Double
            /// Mean number of legal moves across the sampled positions.
            let meanLegalMoves: Double
            /// Mean of `ln(legalMoveCount)` across the sampled positions
            /// — the entropy a uniformly-flat policy would have. The
            /// ratio `meanEntropyNats / meanUniformEntropyNats` is the
            /// "% of uniform" reading the tactical-probe panel reports.
            let meanUniformEntropyNats: Double
            /// Per-position entropy distribution within the bucket,
            /// reported as values at the percentiles named in
            /// `ReplayBufferAnalyzer.entropyPercentileLabels`. Used to
            /// tell "uniformly flat across positions" apart from "a few
            /// sharp positions and many flat ones hiding behind a
            /// flat-looking mean."
            let entropyPercentilesNats: [Double]
            /// Mean over sampled positions of the renormalized legal-only
            /// probability mass on the top-`K` moves, for the `K` values
            /// in `ReplayBufferAnalyzer.topKLabels`. A low value at the
            /// largest `K` (e.g. `meanTopKLegalMass.last ≈ 0.5` when
            /// `topKLabels.last` is `5`) means the network spreads its
            /// belief broadly across the legal moves instead of
            /// committing to a handful of top candidates. The values
            /// here are at tau=1.0; see `meanTopKLegalMassAtSelfPlayTau`
            /// for the post-tau self-play sampling projection.
            let meanTopKLegalMass: [Double]
            /// Mean over sampled positions of the softmax mass placed on
            /// illegal cells (the `1 - legalSum` before renormalization).
            /// A non-trivial value (> 0.1, say) signals the network has
            /// not fully learned legality even after training — an
            /// independent capacity / training-rate signal from the
            /// flatness numbers.
            let meanIllegalMass: Double

            /// Tau used to derive the `*AtSelfPlayTau` projections.
            /// Echoed into every per-bucket entry so JSON consumers
            /// don't need to look elsewhere for the value.
            let selfPlayTauUsed: Double
            /// Mean entropy (nats) of the legal-renormalized policy
            /// after applying `selfPlayTauUsed` temperature — i.e.,
            /// what self-play actually samples from. Compare directly
            /// to `meanEntropyNats` to see how much temperature
            /// concentrates the policy.
            let meanEntropyNatsAtSelfPlayTau: Double
            /// Mean top-K legal mass after applying `selfPlayTauUsed`.
            /// Compare to `meanTopKLegalMass` to see the same
            /// concentration shift in the top-K view.
            let meanTopKLegalMassAtSelfPlayTau: [Double]

            /// Mean value-head scalar `v = p_win − p_loss ∈ [−1, +1]`
            /// across sampled positions in this bucket. For sparse
            /// material-advantage buckets the well-trained-network
            /// expectation is `≈ +1.0` on the K+Q side and `≈ −1.0`
            /// on the K side; values near zero indicate the value
            /// head hasn't learned to read material imbalance.
            let meanValueScalar: Double
            /// Per-position value-scalar distribution at the percentiles
            /// named in `entropyPercentileLabels` (re-used for layout
            /// consistency). Wide p10↔p90 spread is healthy — it means
            /// the value head distinguishes "I'm winning" positions
            /// from "I'm losing" positions; narrow spread around zero
            /// is the collapsed-value-head signature.
            let valueScalarPercentiles: [Double]

            /// Channel histogram of the network's argmax legal move,
            /// stratified by per-position entropy into
            /// `entropySliceLabels.count` slices (sharp → flat).
            /// Surfaces whether the network's confident moves differ
            /// in move family from its uncertain ones (e.g., are
            /// sharp positions dominated by distance-1 captures while
            /// flat positions spread across long-range slides?).
            let topMoveChannelByEntropySlice: [ChannelHistogramSlice]
        }

        // Header
        let producedAtISO8601: String
        let modelLabel: String

        /// Cross-cutting training-progress context (step count, elapsed
        /// time, build/git provenance). Stamped on by `SessionController`
        /// at export time — the analyzer leaves it `nil` and a `nil`
        /// optional omits its key, so analyzer-only callers and tests
        /// produce JSON unchanged from before this field existed.
        var exportMetadata: AnalysisExportMetadata? = nil
        let bufferCapacity: Int
        let bufferStoredCount: Int
        let distinctGameCount: Int
        let gamesWithCompleteFinalPosition: Int

        // 1) Channel histogram (length = PolicyEncoding.channelCount = 76).
        let channelCounts: [Int]

        // 2) Channel × material-bucket. Outer index follows `materialBuckets`,
        //    inner index is the channel (length 76).
        let channelByMaterialBucket: [[Int]]
        /// Per-material-bucket total position count. Sum equals
        /// `bufferStoredCount`.
        let materialBucketPositionCounts: [Int]

        // 3) Final-move channel given outcome. Keys: "win" / "draw" / "loss"
        //    (mover's outcome at the final position). Values: 76-length array.
        let finalMoveChannelByOutcome: [String: [Int]]

        // 4) End-of-game material × outcome. Outer index follows
        //    `materialBuckets`, inner index follows `outcomeLabels`.
        let finalPositionMaterialByOutcome: [[Int]]

        // 5) Game-length × outcome. Keys: outcome labels. Values: length-bucket
        //    counts, matching `gameLengthBucketLabels`.
        let gameLengthByOutcome: [String: [Int]]
        let gameLengthBucketLabels: [String]

        // 6) Top-N material-signature classes (across games with a complete
        //    final position in the buffer), ranked by game count desc.
        let topMaterialSignatures: [MaterialClassEntry]

        /// Optional 7th analysis — only populated by the
        /// `runWithPolicyEntropy(...)` entry point (the live-network
        /// menu path). The CLI flag's pure-buffer pass leaves this
        /// `nil`. Per-bucket means over a stratified random sample of
        /// positions forward-passed through the network.
        let policyEntropyByMaterialBucket: [PolicyEntropyBucketStat]?

        /// Label of the network used to compute
        /// `policyEntropyByMaterialBucket` — recorded separately from
        /// `modelLabel` (which always identifies the buffer's source,
        /// i.e. the champion that played the games) because the probe
        /// is most meaningful when run against the trainer, whose
        /// policy is actually evolving. Nil iff
        /// `policyEntropyByMaterialBucket` is nil.
        let policyEntropyModelLabel: String?
    }

    // MARK: - Public entry point

    /// Walk every slot of `buffer` under its lock and produce a single
    /// `Result`. `modelLabel` is treated as opaque telemetry — the analyzer
    /// never inspects it, just round-trips it into the result header so
    /// the JSON file records which model produced the buffer being
    /// analyzed.
    ///
    /// All work happens inside a single `withSlotData` closure (and thus
    /// under the buffer's lock) so the lock is acquired exactly once.
    /// The closure must be `@Sendable` returning a `Sendable` value to
    /// satisfy `OSAllocatedUnfairLock.withLock`; that's why we build the
    /// full `Result` in one shot rather than threading mutable
    /// accumulators across multiple lock passes.
    static func run(buffer: ReplayBuffer, modelLabel: String) -> Result {
        // Precompute header inputs outside the closure so the closure
        // captures only Sendable values.
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        let producedAt = iso.string(from: Date())

        // Game-length bucket labels are static given the bucket
        // configuration; build once and capture by value into the closure.
        var lengthLabels: [String] = []
        for b in 0..<gameLengthBucketCount {
            let lo = b * gameLengthBucketSize
            let hi = lo + gameLengthBucketSize - 1
            lengthLabels.append("\(lo)-\(hi)")
        }
        if gameLengthBucketCount > 0 {
            let lo = (gameLengthBucketCount - 1) * gameLengthBucketSize
            lengthLabels[gameLengthBucketCount - 1] = "\(lo)+"
        }
        let gameLengthBucketLabels = lengthLabels
        let topLimit = topMaterialSignatureLimit
        let channelCount = PolicyEncoding.channelCount
        let materialBucketCount = materialBuckets.count
        let outcomeCount = outcomeLabels.count

        return buffer.withSlotData { slots -> Result in
            var channelCounts = Array(repeating: 0, count: channelCount)
            var channelByMaterialBucket = Array(
                repeating: Array(repeating: 0, count: channelCount),
                count: materialBucketCount
            )
            var materialBucketPositionCounts = Array(repeating: 0, count: materialBucketCount)

            // Per-game accumulator. `maxPlySlotIndex` tracks the slot
            // whose `plyIndex` is the largest seen so far for this game —
            // that's the candidate "final position" if
            // `maxPly == length - 1`.
            struct GameAccum {
                var maxPly: Int
                var length: Int
                var outcome: Int
                var maxPlySlotIndex: Int
            }
            var perGame: [UInt32: GameAccum] = [:]
            perGame.reserveCapacity(20_000)

            // Pass 1: per-slot histograms + per-game scratch.
            for i in 0..<slots.count {
                let move = slots.moves[i]
                let chan = channelIndex(moveIndex: move)
                guard chan >= 0 && chan < channelCount else { continue }
                channelCounts[chan] += 1

                let material = Int(slots.materialCount[i])
                let mb = materialBucketIndex(for: material)
                channelByMaterialBucket[mb][chan] += 1
                materialBucketPositionCounts[mb] += 1

                let gameId = slots.workerGameId[i]
                let ply = Int(slots.plyIndex[i])
                let length = Int(slots.gameLength[i])
                let outcome = outcomeBucket(slots.outcomes[i])

                if let existing = perGame[gameId] {
                    if ply > existing.maxPly {
                        perGame[gameId] = GameAccum(
                            maxPly: ply,
                            length: length,
                            outcome: outcome,
                            maxPlySlotIndex: i
                        )
                    }
                } else {
                    perGame[gameId] = GameAccum(
                        maxPly: ply,
                        length: length,
                        outcome: outcome,
                        maxPlySlotIndex: i
                    )
                }
            }

            // Pass 2: per-game final-position analyses (3/4/6).
            //
            // Accumulate into outcome-indexed 2D arrays (not
            // [String: [Int]] dicts) so each mutation is a plain Array
            // subscript chain — `arr[i][j] += 1` — instead of relying
            // on the dict-subscript-then-force-unwrap chain
            // `dict[k]![j] += 1`, whose modify-accessor propagation is
            // subtle. We convert to the dict-keyed public form below
            // before returning.
            var finalMoveByOutcome = Array(
                repeating: Array(repeating: 0, count: channelCount),
                count: outcomeCount
            )
            var finalPositionMaterialByOutcome = Array(
                repeating: Array(repeating: 0, count: outcomeCount),
                count: materialBucketCount
            )
            struct SigAccum {
                var win: Int = 0
                var draw: Int = 0
                var loss: Int = 0
            }
            var sigAccum: [String: SigAccum] = [:]
            sigAccum.reserveCapacity(2_000)
            var gamesWithCompleteFinalPosition = 0

            for (_, g) in perGame {
                let isComplete = (g.length > 0) && (g.maxPly == g.length - 1)
                guard isComplete else { continue }
                gamesWithCompleteFinalPosition += 1

                let slot = g.maxPlySlotIndex
                let move = slots.moves[slot]
                let chan = channelIndex(moveIndex: move)
                guard chan >= 0 && chan < channelCount else { continue }
                finalMoveByOutcome[g.outcome][chan] += 1

                let material = Int(slots.materialCount[slot])
                let mb = materialBucketIndex(for: material)
                finalPositionMaterialByOutcome[mb][g.outcome] += 1

                let planesBase = slots.boards.advanced(
                    by: slot * ReplayBuffer.floatsPerBoard
                )
                let signature = computeMaterialSignature(planesBase: planesBase)
                var entry = sigAccum[signature] ?? SigAccum()
                switch g.outcome {
                case 0: entry.win += 1
                case 1: entry.draw += 1
                case 2: entry.loss += 1
                default: break
                }
                sigAccum[signature] = entry
            }

            // Game-length × outcome (#5). Driven only by `perGame`;
            // no slot access needed but kept inside the closure so the
            // entire reduction happens in one lock pass. Same array-
            // indexed-then-converted-to-dict pattern as above.
            var gameLengthByOutcomeArr = Array(
                repeating: Array(repeating: 0, count: gameLengthBucketCount),
                count: outcomeCount
            )
            for (_, g) in perGame {
                let lb = gameLengthBucketIndex(for: g.length)
                gameLengthByOutcomeArr[g.outcome][lb] += 1
            }

            // Convert outcome-indexed arrays to the dict-keyed public form.
            var finalMoveChannelByOutcome: [String: [Int]] = [:]
            var gameLengthByOutcome: [String: [Int]] = [:]
            for (i, label) in outcomeLabels.enumerated() {
                finalMoveChannelByOutcome[label] = finalMoveByOutcome[i]
                gameLengthByOutcome[label] = gameLengthByOutcomeArr[i]
            }

            // Top-N material signatures.
            let sortedSigs = sigAccum
                .map { (sig, acc) -> Result.MaterialClassEntry in
                    Result.MaterialClassEntry(
                        signature: sig,
                        gameCount: acc.win + acc.draw + acc.loss,
                        winCount: acc.win,
                        drawCount: acc.draw,
                        lossCount: acc.loss
                    )
                }
                .sorted { $0.gameCount > $1.gameCount }
            let topSigs = Array(sortedSigs.prefix(topLimit))

            return Result(
                producedAtISO8601: producedAt,
                modelLabel: modelLabel,
                bufferCapacity: slots.capacity,
                bufferStoredCount: slots.count,
                distinctGameCount: perGame.count,
                gamesWithCompleteFinalPosition: gamesWithCompleteFinalPosition,
                channelCounts: channelCounts,
                channelByMaterialBucket: channelByMaterialBucket,
                materialBucketPositionCounts: materialBucketPositionCounts,
                finalMoveChannelByOutcome: finalMoveChannelByOutcome,
                finalPositionMaterialByOutcome: finalPositionMaterialByOutcome,
                gameLengthByOutcome: gameLengthByOutcome,
                gameLengthBucketLabels: gameLengthBucketLabels,
                topMaterialSignatures: topSigs,
                policyEntropyByMaterialBucket: nil,
                policyEntropyModelLabel: nil
            )
        }
    }

    // MARK: - Live-network entropy sampler

    /// Default per-bucket target for `runWithPolicyEntropy(...)`. Sized
    /// so the cumulative forward-pass cost (target × number of non-empty
    /// buckets) stays in the few-second range on Apple Silicon — enough
    /// to get stable per-bucket means without turning a sub-second
    /// analyzer into a multi-minute job. Buckets with fewer positions
    /// than the target are sampled exhaustively (the sampler clamps).
    static let defaultPolicyEntropyPerBucketTarget = 500

    /// Run the standard buffer analyzer AND a stratified policy-entropy
    /// probe against a live `network`. Use this entry point when the
    /// caller has a network available (i.e. the in-app Debug menu
    /// path). The CLI flag, which never sets up a network, sticks with
    /// `run(buffer:modelLabel:)`.
    ///
    /// `modelLabel` always identifies the buffer's source (the
    /// champion that played the games). `entropyModelLabel` records
    /// which network ran the entropy probe — typically the trainer,
    /// since the champion's policy is frozen between promotions and
    /// the meaningful "is illegal mass falling?" signal lives on the
    /// trainer. When `nil`, `modelLabel` is reused (the historical
    /// behavior, kept for callers that probe with the same network
    /// that filled the buffer).
    ///
    /// The entropy probe samples up to `perBucketTarget` positions
    /// per material bucket, copies board tensors out from under the
    /// buffer's lock (no pointer escape), and forward-passes each
    /// outside the lock so live training isn't paused for the duration
    /// of the inference loop.
    static func runWithPolicyEntropy(
        buffer: ReplayBuffer,
        network: ChessMPSNetwork,
        modelLabel: String,
        entropyModelLabel: String? = nil,
        perBucketTarget: Int = defaultPolicyEntropyPerBucketTarget
    ) async throws -> Result {
        let base = run(buffer: buffer, modelLabel: modelLabel)
        let entropyStats = try await samplePolicyEntropyByMaterialBucket(
            buffer: buffer,
            network: network,
            perBucketTarget: perBucketTarget
        )
        // Produce a new Result with the optional field replaced. Result
        // fields are `let` so we can't mutate in-place; rebuild instead.
        return Result(
            producedAtISO8601: base.producedAtISO8601,
            modelLabel: base.modelLabel,
            bufferCapacity: base.bufferCapacity,
            bufferStoredCount: base.bufferStoredCount,
            distinctGameCount: base.distinctGameCount,
            gamesWithCompleteFinalPosition: base.gamesWithCompleteFinalPosition,
            channelCounts: base.channelCounts,
            channelByMaterialBucket: base.channelByMaterialBucket,
            materialBucketPositionCounts: base.materialBucketPositionCounts,
            finalMoveChannelByOutcome: base.finalMoveChannelByOutcome,
            finalPositionMaterialByOutcome: base.finalPositionMaterialByOutcome,
            gameLengthByOutcome: base.gameLengthByOutcome,
            gameLengthBucketLabels: base.gameLengthBucketLabels,
            topMaterialSignatures: base.topMaterialSignatures,
            policyEntropyByMaterialBucket: entropyStats,
            policyEntropyModelLabel: entropyModelLabel ?? modelLabel
        )
    }

    /// Stratified random sample of `perBucketTarget` positions per
    /// material bucket. Returns one entry per non-empty bucket; empty
    /// buckets are dropped from the result list.
    ///
    /// Two-phase: under the buffer lock, copy out per-bucket board
    /// tensors as Sendable `[Float]` arrays (no escaped pointer);
    /// outside the lock, forward-pass each through `network` and
    /// accumulate the legal-masked policy entropy per bucket.
    static func samplePolicyEntropyByMaterialBucket(
        buffer: ReplayBuffer,
        network: ChessMPSNetwork,
        perBucketTarget: Int
    ) async throws -> [Result.PolicyEntropyBucketStat] {
        let materialBucketCount = materialBuckets.count

        // Phase 1: under lock, bucket every slot index, then random-
        // sample per bucket, copying out board tensors as Sendable
        // value arrays. The lock is released before any forward pass.
        struct BucketSamples: Sendable {
            let bucketIndex: Int
            let boards: [[Float]]
        }
        let allSamples = buffer.withSlotData { slots -> [BucketSamples] in
            var perBucketIndices: [[Int]] = Array(
                repeating: [], count: materialBucketCount
            )
            for i in 0..<slots.count {
                let m = Int(slots.materialCount[i])
                let mb = materialBucketIndex(for: m)
                perBucketIndices[mb].append(i)
            }

            var out: [BucketSamples] = []
            out.reserveCapacity(materialBucketCount)
            for (bucketIdx, indices) in perBucketIndices.enumerated() {
                let target = min(perBucketTarget, indices.count)
                guard target > 0 else {
                    out.append(BucketSamples(bucketIndex: bucketIdx, boards: []))
                    continue
                }
                // Random sample without replacement. `shuffled()` is
                // O(n) on the bucket's index list; cheap relative to
                // the upstream walk.
                let shuffled = indices.shuffled()
                var boards: [[Float]] = []
                boards.reserveCapacity(target)
                for slotIdx in shuffled.prefix(target) {
                    let base = slots.boards.advanced(
                        by: slotIdx * ReplayBuffer.floatsPerBoard
                    )
                    let buf = UnsafeBufferPointer(
                        start: base,
                        count: ReplayBuffer.floatsPerBoard
                    )
                    boards.append(Array(buf))
                }
                out.append(BucketSamples(bucketIndex: bucketIdx, boards: boards))
            }
            return out
        }

        // Phase 2: forward-pass each sample, compute per-position
        // policy metrics (entropy, top-K legal-mass, illegal mass,
        // post-tau projections, argmax channel, value scalar),
        // accumulate per bucket. Forward passes serialize on the
        // network's execution queue.
        let topKCount = topKLabels.count
        let channelCount = PolicyEncoding.channelCount
        let sliceCount = entropySliceLabels.count
        var stats: [Result.PolicyEntropyBucketStat] = []
        for bucket in allSamples {
            guard !bucket.boards.isEmpty else { continue }
            // Per-position values needed for percentile or per-slice
            // analysis are collected into arrays. Scalar means just
            // accumulate as sums to save memory.
            //
            // entropyAndChannel: (entropy, topMoveChannel) for the
            // entropy-slice channel histograms (analysis 8). Sorting
            // by .0 then iterating gives us the sharp→flat ordering.
            var entropyAndChannel: [(entropy: Double, channel: Int)] = []
            entropyAndChannel.reserveCapacity(bucket.boards.count)
            var valueScalars: [Double] = []
            valueScalars.reserveCapacity(bucket.boards.count)
            var legalSum: Double = 0
            var uniformSum: Double = 0
            var topKSums = [Double](repeating: 0, count: topKCount)
            var illegalSum: Double = 0
            var entropyAtTauSum: Double = 0
            var topKAtTauSums = [Double](repeating: 0, count: topKCount)
            var valueScalarSum: Double = 0
            var counted = 0
            for board in bucket.boards {
                let pass = try await forwardPass(network: network, board: board)
                let perPos = legalMaskedPolicyMetrics(
                    rawLogits: pass.logits,
                    boardTensor: board
                )
                // A position with zero legal moves shouldn't normally
                // land in the buffer (terminal positions aren't stored),
                // but defend against it by skipping rather than dividing
                // by zero in `ln(0)`.
                guard perPos.legalCount > 0 else { continue }
                entropyAndChannel.append((perPos.entropy, perPos.topMoveChannel))
                valueScalars.append(Double(pass.valueScalar))
                legalSum += Double(perPos.legalCount)
                uniformSum += log(Double(perPos.legalCount))
                for i in 0..<topKCount {
                    topKSums[i] += perPos.topKLegalMass[i]
                    topKAtTauSums[i] += perPos.topKLegalMassAtSelfPlayTau[i]
                }
                illegalSum += perPos.illegalMass
                entropyAtTauSum += perPos.entropyAtSelfPlayTau
                valueScalarSum += Double(pass.valueScalar)
                counted += 1
            }
            guard counted > 0 else { continue }

            let denom = Double(counted)
            let meanEntropy = entropyAndChannel
                .reduce(0.0) { $0 + $1.entropy } / denom

            // Entropy percentiles.
            let sortedEntropy = entropyAndChannel.map(\.entropy).sorted()
            var percentiles: [Double] = []
            percentiles.reserveCapacity(entropyPercentileLabels.count)
            for p in entropyPercentileLabels {
                percentiles.append(percentile(p: Double(p), sortedValues: sortedEntropy))
            }

            // Value-scalar percentiles (re-using entropyPercentileLabels
            // for layout consistency — same indices, different metric).
            let sortedValue = valueScalars.sorted()
            var valuePercentiles: [Double] = []
            valuePercentiles.reserveCapacity(entropyPercentileLabels.count)
            for p in entropyPercentileLabels {
                valuePercentiles.append(percentile(p: Double(p), sortedValues: sortedValue))
            }

            // Entropy-slice channel histograms. Sort `entropyAndChannel`
            // ascending by entropy, then split into equal-size slices
            // and tally channels within each slice. Slice boundaries
            // are computed via integer arithmetic so slice sizes differ
            // by at most 1 when `counted` doesn't divide evenly.
            let sortedByEntropy = entropyAndChannel.sorted { $0.entropy < $1.entropy }
            var slices: [Result.ChannelHistogramSlice] = []
            slices.reserveCapacity(sliceCount)
            for s in 0..<sliceCount {
                let lo = (s * counted) / sliceCount
                let hi = ((s + 1) * counted) / sliceCount
                var histogram = [Int](repeating: 0, count: channelCount)
                var positionsInSlice = 0
                for i in lo..<hi {
                    let chan = sortedByEntropy[i].channel
                    if chan >= 0 && chan < channelCount {
                        histogram[chan] += 1
                        positionsInSlice += 1
                    }
                }
                slices.append(Result.ChannelHistogramSlice(
                    sliceLabel: entropySliceLabels[s],
                    positionCount: positionsInSlice,
                    topMoveChannelCounts: histogram
                ))
            }

            let bucketLabel = materialBuckets[bucket.bucketIndex].label
            stats.append(Result.PolicyEntropyBucketStat(
                bucketLabel: bucketLabel,
                sampleCount: counted,
                meanEntropyNats: meanEntropy,
                meanLegalMoves: legalSum / denom,
                meanUniformEntropyNats: uniformSum / denom,
                entropyPercentilesNats: percentiles,
                meanTopKLegalMass: topKSums.map { $0 / denom },
                meanIllegalMass: illegalSum / denom,
                selfPlayTauUsed: selfPlayTauForProjection,
                meanEntropyNatsAtSelfPlayTau: entropyAtTauSum / denom,
                meanTopKLegalMassAtSelfPlayTau: topKAtTauSums.map { $0 / denom },
                meanValueScalar: valueScalarSum / denom,
                valueScalarPercentiles: valuePercentiles,
                topMoveChannelByEntropySlice: slices
            ))
        }
        return stats
    }

    /// Linear-interpolation percentile for `p` in `0..100` over the
    /// already-sorted-ascending `sortedValues`. Returns 0 for an empty
    /// input. Matches NumPy's default `interpolation="linear"` mode so
    /// downstream notebooks see the same numbers if they recompute
    /// from `entropyPercentilesNats` source data.
    static func percentile(p: Double, sortedValues: [Double]) -> Double {
        guard !sortedValues.isEmpty else { return 0 }
        if sortedValues.count == 1 { return sortedValues[0] }
        let pos = (p / 100.0) * Double(sortedValues.count - 1)
        let lo = Int(floor(pos))
        let hi = Int(ceil(pos))
        if lo == hi { return sortedValues[lo] }
        let weight = pos - Double(lo)
        return sortedValues[lo] * (1.0 - weight) + sortedValues[hi] * weight
    }

    /// Single forward pass through `network`, returning the raw policy
    /// logits as a `[Float]` and the value-head scalar
    /// `v = p_win − p_loss ∈ [−1, +1]`. Wraps `evaluate(board:consume:)`
    /// in the same `LogitsBox` pattern as the tactical-probe runner,
    /// plus a sibling box for the value scalar that the consume
    /// closure also receives "for free."
    private static func forwardPass(
        network: ChessMPSNetwork,
        board: [Float]
    ) async throws -> (logits: [Float], valueScalar: Float) {
        let logitsBox = LogitsBox()
        let valueBox = ValueBox()
        try await network.evaluate(board: board) { logitsBuf, value in
            logitsBox.set(Array(logitsBuf))
            valueBox.set(value)
        }
        return (logitsBox.take(), valueBox.take())
    }

    /// Per-position policy metrics derived from one forward pass.
    /// All fields are computed under the same softmax + legal-mask
    /// pipeline; the caller bundles them so we only sort/walk the
    /// legal-move list once.
    struct PerPositionPolicyMetrics: Sendable {
        let legalCount: Int
        /// Shannon entropy in nats of the legal-renormalized policy
        /// at tau=1.0 (the network's underlying belief).
        let entropy: Double
        /// Renormalized legal mass on the top-`K` moves for each `K`
        /// in `ReplayBufferAnalyzer.topKLabels`, at tau=1.0. A `K`
        /// larger than `legalCount` reads 1.0 (the full legal mass).
        let topKLegalMass: [Double]
        /// `1 - Σ_{legal} softmax` — fraction of softmax mass placed
        /// on illegal cells before renormalization.
        let illegalMass: Double
        /// Entropy after applying `selfPlayTauForProjection` to the
        /// renormalized legal distribution and renormalizing again.
        /// Equivalent to the entropy of `p_i^{1/tau} / Σ p_j^{1/tau}`.
        let entropyAtSelfPlayTau: Double
        /// Top-K legal mass at `selfPlayTauForProjection`. Same shape
        /// as `topKLegalMass`.
        let topKLegalMassAtSelfPlayTau: [Double]
        /// Policy-encoder channel (0..<`PolicyEncoding.channelCount`)
        /// of the network's argmax legal move at tau=1.0. `-1` for
        /// terminal positions (no legal moves).
        let topMoveChannel: Int
    }

    /// Compute the per-position legal-masked policy metrics. Steps:
    ///   1. Softmax the raw policy logit vector.
    ///   2. Decode the board tensor back to a `GameState` via
    ///      `BoardEncoder.decodeSynthetic`; enumerate legal moves
    ///      with `MoveGenerator.legalMoves`.
    ///   3. Pull each legal move's softmax mass via
    ///      `PolicyEncoding.policyIndex`; record the pre-renormalization
    ///      illegal mass; renormalize so legal-only mass sums to 1.
    ///   4. Compute Shannon entropy on the renormalized distribution.
    ///   5. Sort legal probs descending once; sum the top-`K` prefix
    ///      for each `K` in `topKLabels`.
    ///
    /// Returns a metrics struct with `legalCount = 0` for terminal
    /// positions; the caller skips those from per-bucket means.
    private static func legalMaskedPolicyMetrics(
        rawLogits: [Float],
        boardTensor: [Float]
    ) -> PerPositionPolicyMetrics {
        let softmax = ChessRunner.softmax(rawLogits)
        let state = boardTensor.withUnsafeBufferPointer { buf -> GameState in
            guard let base = buf.baseAddress else {
                preconditionFailure(
                    "ReplayBufferAnalyzer.legalMaskedPolicyMetrics: empty boardTensor"
                )
            }
            return BoardEncoder.decodeSynthetic(from: base)
        }
        let legals = MoveGenerator.legalMoves(for: state)
        let kCount = topKLabels.count
        guard !legals.isEmpty else {
            return PerPositionPolicyMetrics(
                legalCount: 0,
                entropy: 0,
                topKLegalMass: Array(repeating: 0, count: kCount),
                illegalMass: 0,
                entropyAtSelfPlayTau: 0,
                topKLegalMassAtSelfPlayTau: Array(repeating: 0, count: kCount),
                topMoveChannel: -1
            )
        }

        // Per-legal-move policy probability at tau=1.0 plus the
        // associated channel — we'll need the channel to record the
        // argmax for the entropy-slice histogram. Encode once per
        // move (PolicyEncoding.encode already handles the
        // mover-perspective row flip) and reuse its (channel, row, col)
        // for both the flat softmax index and the channel histogram.
        var legalProbs: [Double] = []
        legalProbs.reserveCapacity(legals.count)
        var legalChannels: [Int] = []
        legalChannels.reserveCapacity(legals.count)
        var legalSum: Double = 0
        for move in legals {
            let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
            let flatIdx = chan * 64 + r * 8 + c
            let p = Double(softmax[flatIdx])
            legalProbs.append(p)
            legalChannels.append(chan)
            legalSum += p
        }

        let illegalMass = max(0.0, 1.0 - legalSum)

        // Renormalize legal-only mass to sum to 1, falling back to
        // uniform when the network put essentially zero mass on the
        // legal set (extreme collapse case; matches the tactical-probe
        // runner's defensive fallback).
        var renorm = [Double](repeating: 0, count: legals.count)
        if legalSum > 1e-12 {
            for i in 0..<legals.count {
                renorm[i] = legalProbs[i] / legalSum
            }
        } else {
            let u = 1.0 / Double(legals.count)
            for i in 0..<legals.count { renorm[i] = u }
        }

        // Tau=1.0 entropy + top-K mass.
        var ent = 0.0
        for p in renorm where p > 0 {
            ent -= p * log(p)
        }
        let sortedDesc = renorm.sorted(by: >)
        var topKMass = [Double](repeating: 0, count: kCount)
        for (i, k) in topKLabels.enumerated() {
            let take = min(k, sortedDesc.count)
            var sum: Double = 0
            for j in 0..<take { sum += sortedDesc[j] }
            topKMass[i] = sum
        }

        // Post-tau projection. Raise each renormalized prob to the
        // power `1/tau` and renormalize. Identical (up to numerical
        // precision) to applying tau to the raw logits, masking, and
        // renormalizing — see `runWithPolicyEntropy`'s comment for the
        // algebraic equivalence.
        let invTau = 1.0 / selfPlayTauForProjection
        var pow_renorm = [Double](repeating: 0, count: renorm.count)
        var powSum: Double = 0
        for i in 0..<renorm.count {
            let v = pow(renorm[i], invTau)
            pow_renorm[i] = v
            powSum += v
        }
        var renormAtTau = [Double](repeating: 0, count: renorm.count)
        if powSum > 1e-12 {
            for i in 0..<renorm.count {
                renormAtTau[i] = pow_renorm[i] / powSum
            }
        } else {
            // Pathological — renorm was all zero, fell back to uniform,
            // and 0^invTau = 0. Restore uniform.
            let u = 1.0 / Double(renorm.count)
            for i in 0..<renorm.count { renormAtTau[i] = u }
        }
        var entAtTau = 0.0
        for p in renormAtTau where p > 0 {
            entAtTau -= p * log(p)
        }
        let sortedDescAtTau = renormAtTau.sorted(by: >)
        var topKMassAtTau = [Double](repeating: 0, count: kCount)
        for (i, k) in topKLabels.enumerated() {
            let take = min(k, sortedDescAtTau.count)
            var sum: Double = 0
            for j in 0..<take { sum += sortedDescAtTau[j] }
            topKMassAtTau[i] = sum
        }

        // Argmax channel: find the legal move with the highest tau=1.0
        // renormalized probability. Ties break arbitrarily (first wins),
        // which is fine for histogram aggregation.
        var argmaxIdx = 0
        var argmaxProb = -1.0
        for i in 0..<renorm.count where renorm[i] > argmaxProb {
            argmaxProb = renorm[i]
            argmaxIdx = i
        }
        let topMoveChannel = legalChannels[argmaxIdx]

        return PerPositionPolicyMetrics(
            legalCount: legals.count,
            entropy: ent,
            topKLegalMass: topKMass,
            illegalMass: illegalMass,
            entropyAtSelfPlayTau: entAtTau,
            topKLegalMassAtSelfPlayTau: topKMassAtTau,
            topMoveChannel: topMoveChannel
        )
    }

    // MARK: - Bucket / channel helpers

    /// 0-based material-bucket index given a non-pawn-piece count. Defends
    /// against a corrupted `materialCount` byte by clamping anything
    /// outside [0, 30] into the nearest bucket.
    static func materialBucketIndex(for count: Int) -> Int {
        for (i, bucket) in materialBuckets.enumerated()
        where count >= bucket.low && count <= bucket.high {
            return i
        }
        return count < materialBuckets[0].low ? 0 : materialBuckets.count - 1
    }

    /// 0-based game-length bucket index. Lengths past the configured tail
    /// bucket are clamped into it.
    static func gameLengthBucketIndex(for length: Int) -> Int {
        let raw = length / gameLengthBucketSize
        return min(max(raw, 0), gameLengthBucketCount - 1)
    }

    /// Map a stored outcome float (+1 / 0 / -1) to a 0/1/2 outcome index.
    /// The training pipeline writes exact ±1.0 / 0.0 values, but we
    /// compare against ±0.5 thresholds to be robust to any future
    /// float-quantization in the storage layer.
    static func outcomeBucket(_ v: Float) -> Int {
        if v >  0.5 { return 0 }   // win
        if v < -0.5 { return 2 }   // loss
        return 1                   // draw
    }

    /// Extract the policy *channel* (0..<76) from a flat policy index
    /// (`channel * 64 + row * 8 + col`). Returns the raw channel value
    /// even when out of range so the caller can decide whether to skip
    /// or surface the bad slot.
    static func channelIndex(moveIndex: Int32) -> Int {
        Int(moveIndex) / 64
    }

    // MARK: - Material signature

    /// Walk planes 0–5 (mover's pieces, in PieceType raw-value order:
    /// pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5) and planes
    /// 6–11 (opponent's pieces, same order) and produce a compact
    /// canonical signature string of the form `"MOVER_vs_OPP"` where
    /// each side lists its pieces in descending material order
    /// (K, Q, R, B, N, P) with multiplicity. Example: white K+Q vs.
    /// black K with the white side to move yields `"KQ_vs_K"`.
    ///
    /// `planesBase` must point at the first float of an encoded board
    /// (i.e. `slots.boards + slot * floatsPerBoard`).
    static func computeMaterialSignature(
        planesBase: UnsafePointer<Float>
    ) -> String {
        // (planeIndex, letter) in DESC material order so the output
        // string reads like a chess-engine material listing.
        // PieceType raw values: pawn=0, knight=1, bishop=2, rook=3,
        // queen=4, king=5. Boundary check at function head defends
        // against future PieceType refactors that change the count.
        let outputOrder: [(planeIndex: Int, letter: Character)] = [
            (5, "K"), (4, "Q"), (3, "R"), (2, "B"), (1, "N"), (0, "P")
        ]

        func sideString(planeBaseOffset: Int) -> String {
            var out = ""
            for (planeIndex, letter) in outputOrder {
                let planeOffset = (planeBaseOffset + planeIndex) * 64
                var n = 0
                for sq in 0..<64
                where planesBase[planeOffset + sq] > 0.5 {
                    n += 1
                }
                for _ in 0..<n { out.append(letter) }
            }
            return out
        }

        let mover = sideString(planeBaseOffset: 0)
        let opp = sideString(planeBaseOffset: 6)
        return "\(mover)_vs_\(opp)"
    }

    // MARK: - Channel label (human-readable)

    /// Render a 0..<76 channel as a short readable label so the text
    /// summary can say "queen N dist 5" instead of "channel 4". Used
    /// by both the CLI stderr summary and the in-app `[ANALYSIS]` log
    /// block.
    static func channelLabel(_ chan: Int) -> String {
        guard chan >= 0 && chan < PolicyEncoding.channelCount else {
            return "chan \(chan) (invalid)"
        }
        if chan < 56 {
            let dirNames = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            let dir = dirNames[chan / 7]
            let dist = (chan % 7) + 1
            return "queen \(dir) dist \(dist)"
        } else if chan < 64 {
            let knightNames = [
                "up-right", "right-up", "right-down", "down-right",
                "down-left", "left-down", "left-up", "up-left"
            ]
            return "knight \(knightNames[chan - 56])"
        } else if chan < 73 {
            let offset = chan - 64
            let pieceNames = ["knight", "rook", "bishop"]
            let dirNames = ["forward", "cap-left", "cap-right"]
            return "underpromo \(pieceNames[offset / 3]) \(dirNames[offset % 3])"
        } else {
            let dirNames = ["forward", "cap-left", "cap-right"]
            return "queen-promo \(dirNames[chan - 73])"
        }
    }
}

// MARK: - Text summary rendering

extension ReplayBufferAnalyzer.Result {

    /// Multi-line human-readable digest of the result. Used both by the
    /// CLI's stderr path and by the in-app debug-menu's `[ANALYSIS]`
    /// session-log block. Output is wide-aligned for terminal viewing;
    /// the session-log carries it verbatim (each line goes through
    /// `SessionLogger.log`, which prepends its own timestamp).
    func textSummary() -> String {
        var out = ""

        let fmt: (Int) -> String = { Self.intFormatter.string(from: NSNumber(value: $0)) ?? "\($0)" }
        let pct: (Int, Int) -> String = { num, den in
            guard den > 0 else { return "  -  " }
            let p = Double(num) * 100.0 / Double(den)
            return String(format: "%5.2f%%", p)
        }

        out += "Replay buffer analysis (model: \(modelLabel))\n"
        out += "  produced:  \(producedAtISO8601)\n"
        out += "  positions: \(fmt(bufferStoredCount)) / \(fmt(bufferCapacity)) capacity\n"
        out += "  games:     \(fmt(distinctGameCount)) distinct\n"
        out += "  games with complete final position: \(fmt(gamesWithCompleteFinalPosition))\n"
        out += "\n"

        // 1) Channel histogram — top 12.
        out += "(1) Channel histogram — top 12 by count:\n"
        let chanRanked = channelCounts.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(12)
        for (chan, count) in chanRanked {
            out += String(
                format: "    chan %2d  %@   %@   %@\n",
                chan,
                ReplayBufferAnalyzer.channelLabel(chan).padding(toLength: 24, withPad: " ", startingAt: 0),
                fmt(count).leftPadded(toLength: 10),
                pct(count, bufferStoredCount)
            )
        }
        out += "\n"

        // 1b) Channel histogram — tail for sliding-piece distances 5/6/7.
        //     These are the ones most relevant to the "endgame technique"
        //     question, so surface them explicitly even when they don't
        //     make the top-12 cut.
        out += "(1b) Queen-style sliding distances 5/6/7 (all 8 directions):\n"
        for dist in [5, 6, 7] {
            var sum = 0
            for dir in 0..<8 { sum += channelCounts[dir * 7 + (dist - 1)] }
            out += String(
                format: "    distance %d total:  %@   %@\n",
                dist,
                fmt(sum).leftPadded(toLength: 10),
                pct(sum, bufferStoredCount)
            )
        }
        out += "\n"

        // 2) Channel × material bucket: sliding-distance share per bucket.
        //    Reports the share of *that bucket's positions* spent on
        //    each queen-distance band, so a "long slides flat across all
        //    bucket sparsities" reading jumps out.
        out += "(2) Queen-style distance share by material bucket:\n"
        out += "    bucket        positions   d1     d2     d3     d4     d5     d6     d7\n"
        for (i, bucket) in ReplayBufferAnalyzer.materialBuckets.enumerated() {
            let total = materialBucketPositionCounts[i]
            var distSums = Array(repeating: 0, count: 7)
            for dir in 0..<8 {
                for dist in 1...7 {
                    distSums[dist - 1] += channelByMaterialBucket[i][dir * 7 + (dist - 1)]
                }
            }
            var line = String(
                format: "    %@  %@  ",
                bucket.label.padding(toLength: 10, withPad: " ", startingAt: 0),
                fmt(total).leftPadded(toLength: 10)
            )
            for sum in distSums {
                line += String(format: "%5.1f%% ", total > 0 ? Double(sum) * 100.0 / Double(total) : 0)
            }
            line += "\n"
            out += line
        }
        out += "\n"

        // 3) Final-move channel by outcome — top 6 per outcome.
        out += "(3) Final-move channel by outcome (top 6 per outcome bucket):\n"
        for label in ReplayBufferAnalyzer.outcomeLabels {
            let arr = finalMoveChannelByOutcome[label] ?? []
            let total = arr.reduce(0, +)
            out += "    [\(label)] total final-move count = \(fmt(total))\n"
            let ranked = arr.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(6)
            for (chan, count) in ranked {
                out += String(
                    format: "      chan %2d  %@   %@   %@\n",
                    chan,
                    ReplayBufferAnalyzer.channelLabel(chan).padding(toLength: 24, withPad: " ", startingAt: 0),
                    fmt(count).leftPadded(toLength: 8),
                    pct(count, total)
                )
            }
        }
        out += "\n"

        // 4) End-of-game material × outcome.
        out += "(4) End-of-game material × outcome (rows = material bucket, cols = outcome):\n"
        out += "    bucket           win        draw       loss      total\n"
        for (i, bucket) in ReplayBufferAnalyzer.materialBuckets.enumerated() {
            let row = finalPositionMaterialByOutcome[i]
            let total = row.reduce(0, +)
            out += String(
                format: "    %@  %@  %@  %@  %@\n",
                bucket.label.padding(toLength: 10, withPad: " ", startingAt: 0),
                fmt(row[0]).leftPadded(toLength: 8),
                fmt(row[1]).leftPadded(toLength: 10),
                fmt(row[2]).leftPadded(toLength: 8),
                fmt(total).leftPadded(toLength: 8)
            )
        }
        out += "\n"

        // 5) Game-length × outcome.
        out += "(5) Game-length × outcome (rows = length bucket, cols = outcome):\n"
        out += "    bucket          win        draw       loss      total\n"
        for (b, bucketLabel) in gameLengthBucketLabels.enumerated() {
            let win = gameLengthByOutcome["win"]?[b] ?? 0
            let draw = gameLengthByOutcome["draw"]?[b] ?? 0
            let loss = gameLengthByOutcome["loss"]?[b] ?? 0
            let total = win + draw + loss
            out += String(
                format: "    %@  %@  %@  %@  %@\n",
                bucketLabel.padding(toLength: 10, withPad: " ", startingAt: 0),
                fmt(win).leftPadded(toLength: 8),
                fmt(draw).leftPadded(toLength: 10),
                fmt(loss).leftPadded(toLength: 8),
                fmt(total).leftPadded(toLength: 8)
            )
        }
        out += "\n"

        // 6) Top material-signature classes.
        out += "(6) Top \(ReplayBufferAnalyzer.topMaterialSignatureLimit) material-signature classes (games with complete final position):\n"
        out += "    signature            games        W      D      L\n"
        for entry in topMaterialSignatures {
            out += String(
                format: "    %@  %@   %@  %@  %@\n",
                entry.signature.padding(toLength: 18, withPad: " ", startingAt: 0),
                fmt(entry.gameCount).leftPadded(toLength: 8),
                fmt(entry.winCount).leftPadded(toLength: 6),
                fmt(entry.drawCount).leftPadded(toLength: 6),
                fmt(entry.lossCount).leftPadded(toLength: 6)
            )
        }
        out += "\n"

        // 7) Live-network policy entropy by material bucket. Only
        //    present when produced by `runWithPolicyEntropy(...)`. For
        //    each bucket: mean legal-masked entropy (nats), the matching
        //    perplexity (`exp(meanEntropy)` — "effective number of
        //    moves the sampler picks from"), and the % of uniform
        //    (`meanEntropy / ln(meanLegalMoves)`) for comparison to
        //    the tactical-probe panel's entropy column.
        if let entropyStats = policyEntropyByMaterialBucket {
            let probeLabel = policyEntropyModelLabel ?? modelLabel
            out += "(7) Policy entropy by material bucket"
                + " (live forward-pass sample, network: \(probeLabel)):\n"
            out += "    bucket      samples   meanEnt(nats)  perplexity   legal   uniformEnt   % of uniform\n"
            for stat in entropyStats {
                let perplexity = exp(stat.meanEntropyNats)
                let pctOfUniform = stat.meanUniformEntropyNats > 0
                    ? (stat.meanEntropyNats / stat.meanUniformEntropyNats) * 100.0
                    : 0
                out += String(
                    format: "    %@  %@   %@   %@   %@   %@   %@\n",
                    stat.bucketLabel.padding(toLength: 8, withPad: " ", startingAt: 0),
                    fmt(stat.sampleCount).leftPadded(toLength: 6),
                    String(format: "%6.3f", stat.meanEntropyNats).leftPadded(toLength: 12),
                    String(format: "%6.2f", perplexity).leftPadded(toLength: 9),
                    String(format: "%5.1f", stat.meanLegalMoves).leftPadded(toLength: 6),
                    String(format: "%5.3f", stat.meanUniformEntropyNats).leftPadded(toLength: 10),
                    String(format: "%5.1f%%", pctOfUniform).leftPadded(toLength: 11)
                )
                // Distributional detail: entropy percentiles, top-K
                // legal mass, illegal mass. Indented under the bucket
                // summary line so each bucket reads as one logical
                // record across four lines.
                let entLabels = ReplayBufferAnalyzer.entropyPercentileLabels
                if stat.entropyPercentilesNats.count == entLabels.count {
                    let labelStr = entLabels.map { "p\($0)" }.joined(separator: "/")
                    let valStr = stat.entropyPercentilesNats
                        .map { String(format: "%5.3f", $0) }
                        .joined(separator: " / ")
                    out += "      entropy \(labelStr) (nats):  \(valStr)\n"
                }
                let kLabels = ReplayBufferAnalyzer.topKLabels
                if stat.meanTopKLegalMass.count == kLabels.count {
                    let labelStr = kLabels.map { "top-\($0)" }.joined(separator: "/")
                    let valStr = stat.meanTopKLegalMass
                        .map { String(format: "%5.3f", $0) }
                        .joined(separator: " / ")
                    out += "      mean \(labelStr) legal mass:  \(valStr)\n"
                }
                out += String(
                    format: "      mean illegal mass:         %5.3f\n",
                    stat.meanIllegalMass
                )

                // Post-tau (self-play projection) row.
                if stat.meanTopKLegalMassAtSelfPlayTau.count == kLabels.count {
                    let topKValStr = stat.meanTopKLegalMassAtSelfPlayTau
                        .map { String(format: "%5.3f", $0) }
                        .joined(separator: " / ")
                    out += String(
                        format: "      tau=%.2f projection: meanEnt=%5.3f  perplex=%6.2f  top-1/3/5 mass: %@\n",
                        stat.selfPlayTauUsed,
                        stat.meanEntropyNatsAtSelfPlayTau,
                        exp(stat.meanEntropyNatsAtSelfPlayTau),
                        topKValStr
                    )
                }

                // Value-head row.
                let entLabelStr = entLabels.map { "p\($0)" }.joined(separator: "/")
                if stat.valueScalarPercentiles.count == entLabels.count {
                    let valStr = stat.valueScalarPercentiles
                        .map { String(format: "%+5.3f", $0) }
                        .joined(separator: " / ")
                    out += String(
                        format: "      value scalar:  mean %+5.3f   \(entLabelStr): %@\n",
                        stat.meanValueScalar,
                        valStr
                    )
                }

                // Top-move-channel-by-entropy-slice rows. For each slice,
                // summarize the channel distribution as queen-distance-band
                // shares (d1..d7) + knight + promo shares, so the table
                // stays narrow enough to read in a terminal.
                if !stat.topMoveChannelByEntropySlice.isEmpty {
                    out += "      top-move family by entropy slice:\n"
                    out += "        slice    n      d1      d2      d3      d4      d5      d6      d7    knight   promo\n"
                    for slice in stat.topMoveChannelByEntropySlice {
                        let counts = slice.topMoveChannelCounts
                        let n = slice.positionCount
                        // Queen-distance band shares (sum over 8 directions).
                        var distSums = [Int](repeating: 0, count: 7)
                        for dir in 0..<8 {
                            for dist in 1...7 {
                                distSums[dist - 1] += counts[dir * 7 + (dist - 1)]
                            }
                        }
                        // Knight (channels 56..63), underpromotion (64..72),
                        // queen-promotion (73..75). Group all promotions as
                        // a single "promo" column for compactness.
                        var knightSum = 0
                        for c in 56..<64 { knightSum += counts[c] }
                        var promoSum = 0
                        for c in 64..<76 { promoSum += counts[c] }

                        func sharePct(_ k: Int) -> String {
                            guard n > 0 else { return "  - " }
                            return String(format: "%5.1f%%", Double(k) * 100.0 / Double(n))
                        }
                        var line = String(
                            format: "        %@ %@",
                            slice.sliceLabel.padding(toLength: 6, withPad: " ", startingAt: 0),
                            fmt(n).leftPadded(toLength: 5)
                        )
                        for sum in distSums { line += "  \(sharePct(sum))" }
                        line += "   \(sharePct(knightSum))  \(sharePct(promoSum))"
                        out += "\(line)\n"
                    }
                }
            }
        }

        return out
    }

    private static let intFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        return f
    }()
}

// MARK: - Logits handoff across the evaluate(consume:) boundary

/// Sendable channel for handing the raw policy logits out of the
/// `evaluate(board:consume:)` callback into the awaiting caller after
/// the underlying `MPSGraph.run` completes. Same shape and rationale
/// as `SessionController+TacticalProbe.swift`'s `LogitsBox`; duplicated
/// (rather than shared) because that one is `private` to its file and
/// the value is one stored property of two methods.
private final class LogitsBox: @unchecked Sendable {
    nonisolated(unsafe) private var value: [Float] = []
    func set(_ v: [Float]) { value = v }
    func take() -> [Float] { value }
}

/// Sendable channel for capturing the value-head scalar
/// (`p_win − p_loss`) from the same `evaluate(board:consume:)` callback
/// that `LogitsBox` captures policy logits from. Same lifetime
/// contract: written inside the consume closure, read once after the
/// `await` resumes.
private final class ValueBox: @unchecked Sendable {
    nonisolated(unsafe) private var value: Float = 0
    func set(_ v: Float) { value = v }
    func take() -> Float { value }
}

// MARK: - Small string padding helper

private extension String {
    /// Right-align this string into a `length`-wide field by prepending
    /// spaces. Used by the analyzer's text-summary formatter to keep
    /// numeric columns vertically aligned. Returns `self` unchanged when
    /// the string is already at least `length` characters wide.
    func leftPadded(toLength length: Int) -> String {
        let pad = length - self.count
        return pad > 0 ? String(repeating: " ", count: pad) + self : self
    }
}
