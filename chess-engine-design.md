# Chess Engine Design Notes

A reference document covering the full design of a self-learning chess engine built with Swift and Metal Performance Shaders on Apple Silicon. Written conversationally to be readable after time away.

---

## The Big Picture

The goal is a chess engine that learns entirely from self-play — no human game data required. It starts knowing nothing, plays itself, gets slightly better, plays a stronger version of itself, and repeats. Given enough iterations it discovers chess strategy from scratch.

The system has three major components that work together:

- **A neural network** that looks at a board position and outputs (a) which moves look promising, and (b) how good the position is
- **MCTS (Monte Carlo Tree Search)** that uses the network to intelligently explore possible futures
- **A self-play training loop** that generates games, trains the network, and decides whether the new network is stronger

---

## The Input Tensor

The board is represented as a 3D array — 19 planes, each an 8×8 grid of floats. Think of it as 19 spreadsheets stacked on top of each other, each answering one specific question about the board.

**Always encoded from the current player's perspective.** Whoever's turn it is, their pieces are "my pieces." The board is flipped vertically if needed so the current player is always at the bottom. This means the network never needs to learn separate strategies for white and black — chess knowledge is symmetric.

### The 19 Planes

```
planes 0–5:    my pieces          — one plane each for P, N, B, R, Q, K
                                    1.0 where piece exists, 0.0 elsewhere
planes 6–11:   opponent's pieces  — same structure for their pieces
plane 12:      my kingside castling right        — all 1.0s or all 0.0s
plane 13:      my queenside castling right       — all 1.0s or all 0.0s
plane 14:      opponent kingside castling right  — all 1.0s or all 0.0s
plane 15:      opponent queenside castling right — all 1.0s or all 0.0s
plane 16:      en passant square I can capture   — single 1.0 on target square
plane 17:      en passant square opponent can capture me — single 1.0
plane 18:      halfmove clock     — all same value, normalized 0.0–1.0
```

### Design Decisions Worth Remembering

**Why castling gets whole planes of 1s/0s:** Castling availability cannot be inferred from the current board. Two positions with identical piece placement can have different castling rights depending on move history. The plane carries that lost history forward. The all-1s/all-0s approach is spatially redundant but unambiguous and consistent with everything else.

**Why two en passant planes instead of one:** "I can capture en passant" and "my opponent can capture me en passant" are strategically opposite things — opportunity vs threat. One plane would conflate them.

**Why halfmove clock instead of total move count:** The halfmove clock tracks consecutive moves since the last pawn move or capture. It directly encodes fifty-move rule proximity. Total move count is a less useful phase indicator — the network can infer phase from piece configuration anyway.

**Why current-player-relative encoding:** Eliminates spurious color asymmetry. The network learns chess concepts, not "what white should do" separately from "what black should do." Training signal is symmetric — every position contributes equally regardless of which color is moving. Value head outputs +1 for current player winning, -1 for losing — always relative, never absolute.

---

## The Network Architecture

### Overview

```
input (19 × 8 × 8)
    ↓
stem — expands to 128 channels
    ↓
tower — 8 residual blocks, shape never changes
    ↓
128 × 8 × 8 feature map
    ↙              ↘
policy head      value head
    ↓                ↓
4096 move        single float
probabilities    [-1.0, +1.0]
```

### The Stem

One conv layer that takes 19 input planes to 128 output planes:

```
3×3 conv, 128 filters, padding=1
batch norm
ReLU
→ 128 × 8 × 8
```

Its only job is to translate from "19 planes of chess facts" into "128 planes of learned features" that the tower can work with. The padding=1 ensures the 8×8 spatial dimensions are preserved — board edges matter (rooks on the a-file, kings in corners) so you never want to lose them.

### The Tower

8 identical residual blocks stacked. The shape never changes — 128 × 8 × 8 in, 128 × 8 × 8 out, every block.

**One residual block:**

```
input: 128 × 8 × 8
    ↓
conv 3×3, 128 filters     — mixes across neighbors and all 128 channels
batch norm
ReLU
    ↓
conv 3×3, 128 filters     — refines further
batch norm
    ↓
+ original input           ← skip connection
ReLU
    ↓
output: 128 × 8 × 8
```

**Why two convs per block:** One conv alone doesn't have enough capacity. The second conv operates on features the first already processed — it's one level of abstraction higher.

**Why residual (skip connection):** Without it, gradients during training have to travel backwards through all 8 blocks to reach early layers. Each block they pass through, the signal weakens. By block 1 it's nearly gone — early layers stop learning. The skip connection creates a direct gradient highway: the error signal can jump straight to any layer regardless of depth.

The block learns a *correction* to its input rather than a full transformation:
```
output = input + what_the_block_learned
```

Early in training blocks can output near-zero corrections (acting as identity), only gradually learning to contribute. This is why residual networks train so reliably.

**Why 8 blocks:** Each 3×3 conv expands the receptive field — the area of the board one output value can "see" — by 2 squares in each direction. After ~5 blocks every square can influence every other square. The remaining blocks deepen the abstraction rather than expanding coverage.

**What the tower is learning:** There's a rough hierarchy that emerges from training (nobody programs this):
- Early blocks: raw patterns — is this square occupied, is a piece under attack
- Middle blocks: tactics — pins, forks, open files, pawn chains  
- Late blocks: strategy — king safety, piece coordination, endgame structure

### The Convolution Operation

A 3×3 conv with 128 input and 128 output channels means:
- 128 filters, each of shape 3×3×128
- At each square, each filter looks at that square + 8 neighbors across ALL 128 input channels simultaneously
- 3 × 3 × 128 = 1,152 multiplications per square per filter, summed to one value
- 128 filters → 128 output planes

A 1×1 conv is the same but with no spatial neighborhood — it only mixes across channels at each individual square. Used in the heads to cheaply compress 128 channels to 2 or 1 before flattening.

### Batch Normalization

After each conv layer, values can end up in wildly different ranges — some channels near 0.001, others near 847. This causes:
- Training instability (large values → large gradients → overshooting weight updates)
- Layers constantly re-adapting as upstream distributions shift during training

Batch norm fixes this by normalizing each channel across the training batch to mean=0, std=1. Then two learnable parameters (gamma, beta) let the network find the optimal scale and offset per channel:

```
output = gamma × normalized_value + beta
```

At inference time (single positions during self-play), batch norm uses running averages computed during training rather than batch statistics.

### ReLU

Rectified Linear Unit. Does one thing:

```
f(x) = max(0, x)
```

Negative inputs → 0. Positive inputs → unchanged.

**Why you need it:** Without nonlinearity, stacking conv layers collapses to a single linear transformation no matter how deep you go. Linear functions can only draw straight lines through data. Chess positions don't separate along straight lines. ReLU introduces the bends that let the network learn complex patterns.

**Why ReLU specifically:** Sigmoid and tanh saturate — for large inputs their gradients approach zero, causing vanishing gradients. ReLU has no ceiling on the positive side, so gradients flow freely. It's also trivially fast to compute.

After batch norm, values are centered near zero, so roughly half get zeroed by ReLU and half pass through. This is the intended operating range.

### The Policy Head

Takes the 128 × 8 × 8 trunk output and produces 4096 move probabilities.

```
1×1 conv, 2 filters        → 2 × 8 × 8
batch norm, ReLU
flatten                    → 128 numbers
fully connected (128→4096) → 4096 numbers
softmax                    → 4096 probabilities summing to 1.0
```

The 1×1 conv compresses 128 channels to 2 before flattening. Without it, flattening 128 × 8 × 8 = 8,192 inputs into a FC layer targeting 4096 outputs would require 33 million weights in that one layer alone.

**Move encoding:** Each of the 4096 outputs maps to one (from_square, to_square) pair:
```
index = from_square × 64 + to_square
```
Squares numbered 0–63, row by row from rank 8. Most outputs will be near zero (illegal or obviously bad moves). A few legal reasonable moves get most of the probability mass.

**Illegal move masking:** After getting policy output, compute all legal moves for the position, zero out illegal move indices, renormalize so remaining probabilities sum to 1.0.

**Pawn promotion:** This encoding doesn't distinguish promotion piece. For simplicity, assume queen promotion always — correct 99.9% of the time. Underpromotion to knight is the only realistic exception (avoiding stalemate in rare endgame positions).

**Softmax function:** Takes unbounded floats, exponentiates each, divides by sum. Result is all-positive, sums to 1.0. The exponentiation amplifies differences — the network's most confident move gets disproportionately more probability, weak moves get squeezed toward zero.

### The Value Head

Takes the 128 × 8 × 8 trunk output and produces one float in [-1, 1].

```
1×1 conv, 1 filter         → 1 × 8 × 8
batch norm, ReLU
flatten                    → 64 numbers
fully connected (64→64)    → 64 numbers
ReLU
fully connected (64→1)     → 1 number
tanh                       → float in [-1.0, +1.0]
```

The intermediate 64→64 FC layer gives the head capacity to combine spatial features before collapsing to a scalar. Going straight to 1 output is too aggressive.

**Tanh:** Squashes unbounded FC output to [-1, 1]. The FC layer could output -47 or +831 — tanh maps the entire real line to this fixed range to match the game outcome encoding.

**What the value means:**
```
-1.0 = I am losing
 0.0 = roughly equal  
+1.0 = I am winning
```
Always from the current player's perspective.

### Fully Connected Layers

A fully connected layer is a matrix multiplication. Every input connects to every output through a learned weight.

64 → 1 (value head final layer):
```
output = input[0]×w[0] + input[1]×w[1] + ... + input[63]×w[63]
```
One dot product, one output.

128 → 4096 (policy head):
```
output[0]    = input[0]×w[0,0]    + ... + input[127]×w[127,0]
output[1]    = input[0]×w[0,1]    + ... + input[127]×w[127,1]
...
output[4095] = input[0]×w[0,4095] + ... + input[127]×w[127,4095]
```
128 × 4096 = 524,288 weights in this one layer.

### Parameter Count (Small Config)

```
stem:              19 × 128 × 3 × 3 =     21,888
per residual block: 2 × (128 × 128 × 3 × 3) = 294,912
8 blocks:                              = 2,359,296
policy head:                           ~  135,000
value head:                            ~   75,000
total:                                 ~  2.6M parameters
```

---

## MCTS — Monte Carlo Tree Search

### What It Is

A way to decide which move to make by selectively exploring a tree of possible futures. Rather than searching exhaustively (impossible — chess has ~10^120 positions), it builds a running estimate of move quality by focusing compute on the most promising branches.

### The Four Phases (repeat thousands of times)

**1. Select:** Starting from the root (current position), follow the tree downward, at each node choosing the child with the best UCB score:

```
UCB = wins/visits  +  C × √(ln(parent_visits) / visits)
         ↑                        ↑
    exploitation              exploration
```

C ≈ 1.4 controls the exploration/exploitation balance. Nodes with few visits get a large exploration bonus. Once visited enough times, their actual win rate takes over.

**2. Expand:** At a node that hasn't been fully explored, add a new child node for an untried legal move. Seed its prior probability from the network's policy output.

**3. Evaluate:** Call the neural network on the new node's position. Get back:
- Policy output → prior probabilities for this node's children (replaces random rollout)
- Value output → position evaluation, used as the simulation result

**4. Backpropagate:** The value travels back up the tree, updating visit counts and win estimates for every node on the path.

After thousands of iterations, pick the move with the most visits from the root.

### Why MCTS + Network Together

The network makes MCTS smarter at both ends:
- Policy output biases exploration toward promising moves (don't waste simulations on obviously bad moves)
- Value output replaces slow noisy random rollouts with fast accurate evaluation

MCTS makes the network's training data better:
- The visit count distribution (not just the chosen move) becomes the policy training target — much richer signal than a one-hot move label
- Games played with MCTS contain genuine combination thinking the network couldn't find alone

They improve each other: better network → MCTS explores better → stronger games → better training signal → better network.

### MCTS and Convergence

MCTS probably isn't strictly required — pure self-play without search would eventually learn chess. But without it:
- Games are weaker and noisier (greedy play misses combinations)
- Training needs far more games to reach the same level
- On constrained hardware (one Mac) you can't compensate with volume

MCTS is essentially a way to get more learning per game. With limited compute it tips from "dramatically helpful" toward "practically necessary."

---

## The Self-Play Training Loop

```
initialize: random network weights

loop forever:
    1. generate N self-play games using current network + MCTS
    2. add all positions to replay buffer
    3. sample mini-batches from buffer, train candidate network
    4. play G evaluation games (G/2 each color)
    5. if candidate wins >55% of decisive games → promote
       else → discard, keep current
    6. go to 1
```

### Training Data Generation

Each game of M moves generates approximately M × 2 training positions (each position encoded from the current player's perspective — white's first move from the starting position is the only one that doesn't need flipping).

Each training position stores:
- The board tensor (19 × 8 × 8)
- The MCTS visit distribution (4096 floats) — policy target
- The game outcome from that player's perspective (+1, 0, -1) — value target

### The Replay Buffer

A fixed-size pool of recent positions (e.g. last 500,000). Training samples random mini-batches from this pool rather than processing whole games sequentially. Benefits:
- Decouples game generation from training
- Each position gets used multiple times across different mini-batches
- Prevents overfitting to the most recent games

### Training the Candidate Network

Always train a *candidate* — never update the current network in place while it's generating games. Mixing training updates into game generation creates unstable feedback loops.

**Two loss functions trained jointly:**

```swift
// policy: cross entropy vs MCTS visit distribution
let policyLoss = crossEntropy(predicted: policyOutput, target: mctsVisits)

// value: mean squared error vs actual game outcome
let valueLoss = MSE(predicted: valueOutput, target: gameOutcome)

let totalLoss = policyLoss + valueLoss
```

The shared trunk learns representations useful for both tasks simultaneously. Policy and value heads constrain each other toward a consistent understanding of the position.

### Evaluation and Promotion

Pit candidate vs current network over G games (even number, each plays white exactly G/2 times — eliminates first-move advantage bias).

Promote if candidate wins >55% of decisive games (not 50% — guards against statistical noise from small sample of decisive games when many end in draws).

### Color Balance

White moves first — a real structural advantage. White wins ~37% of games at grandmaster level, black ~27%, draws ~36%.

During self-play this self-corrects: the same network plays both sides, so it's simultaneously giving and receiving the first-move advantage. Any bias toward one color gets immediately exploited by its other-color self.

During evaluation: enforce G/2 games each color so a network that's merely better at white can't appear stronger than it is.

---

## Network Sizing for Apple Silicon

| Config | Filters | Blocks | Params | Notes |
|--------|---------|--------|--------|-------|
| Tiny   | 64      | 5      | ~1M    | First pass, verify training loop |
| Small  | 128     | 8      | ~2.6M  | Sweet spot for self-play on Mac |
| Medium | 256     | 10     | ~25M   | Too slow for self-play volume |

**The small config (128 filters, 8 blocks) is the target.** At ~2ms inference time, 800 MCTS simulations per move = ~1.6 seconds per move. Slow but workable for training. Medium config hits ~8ms → 6.4 seconds per move → self-play crawls.

Inference speed × simulations per move × moves per game × games per hour = training data rate. Faster inference = more games = faster learning.

---

## Implementation Stack

```
python-chess (Python)     → chess rules, move generation for data pipeline
PyTorch + MPS backend     → train on Apple Silicon GPU
coremltools               → export .mlmodel
Swift + CoreML            → run inference
Swift MCTS engine         → tree search wrapping inference
```

Or, for maximum learning: implement the network directly in Metal Performance Shaders (MPSGraph) in Swift, keeping the entire system in one language.

---

## Key Concepts Summary

| Term | What it means |
|------|---------------|
| Tensor | Multi-dimensional array. 19 × 8 × 8 = 19 stacked 8×8 grids |
| Channel / plane | One 8×8 layer in the tensor |
| Conv 3×3 | Filter that looks at each square + 8 neighbors across ALL channels |
| 1×1 conv | Filter that mixes channels at each square with no spatial neighborhood |
| Receptive field | How much of the board one output value can "see" — grows with depth |
| Batch norm | Rescales channel values to stable range after each conv |
| ReLU | Zeros out negatives — introduces nonlinearity so network can learn complex patterns |
| Residual / skip | Adds block input back to output — enables gradient flow through deep networks |
| Policy head | Outputs 4096 move probabilities — seeds MCTS priors |
| Value head | Outputs single float [-1,1] — evaluates position for MCTS and training |
| Softmax | Converts unbounded floats to probabilities summing to 1.0 |
| Tanh | Squashes unbounded float to [-1, 1] |
| Fully connected | Dense matrix multiplication — every input connects to every output |
| MCTS | Tree search that balances exploring new moves vs exploiting known good ones |
| UCB | Formula balancing exploitation (win rate) and exploration (visit count) |
| Self-play | Network plays both sides — opponent difficulty automatically matches current strength |
| Replay buffer | Pool of recent positions sampled for training — decouples generation from training |
| Candidate network | Network being trained — never replaces current until it proves stronger |
| Halfmove clock | Moves since last pawn move or capture — tracks fifty-move rule proximity |
| En passant | Pawn capture rule where you capture a pawn that just moved two squares, as if it moved one |
| Underpromotion | Promoting pawn to knight instead of queen — rare, avoids stalemate in specific endgames |
