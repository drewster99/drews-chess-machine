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

    // MARK: - Result struct

    struct Result: Codable, Sendable {

        struct MaterialClassEntry: Codable, Sendable {
            let signature: String
            let gameCount: Int
            let winCount: Int
            let drawCount: Int
            let lossCount: Int
        }

        // Header
        let producedAtISO8601: String
        let modelLabel: String
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
                topMaterialSignatures: topSigs
            )
        }
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

        return out
    }

    private static let intFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        return f
    }()
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
