# Chess Engine Design Notes

A reference document covering the full design of a self-learning chess engine built with Swift and Metal Performance Shaders on Apple Silicon. Written conversationally for someone who’s pretty new to this (*by* someone who’s pretty new to this).

## My Goal

As Googling and chats with Claude have informed me, there are several ways to improve this that should enable the network to learn at a significantly faster rate. Those things might be part of a future project, but my immediate goal is to try to better understand the basic building blocks of neural networks and machine learning.

With that in mind, I’m hoping to build the most basic convolutional neural network setup and have it learn some chess, playing only itself, starting from completely random weights.

This document is really my own documentation of my learning process as I walk through designing the neural network, before moving on to coding and testing.

---

## The Big Picture

The goal is a chess engine that learns entirely from self-play — no human game data required. It starts knowing nothing, plays itself, gets slightly better, plays a stronger version of itself, and repeats. Given enough iterations it discovers chess strategy from scratch.

The system has two major components that work together:

- **A neural network** that looks at a board position and outputs (a) which moves look promising, and (b) how good the player’s current position is
- **A self-play training loop** that generates games, trains the network, and decides whether the new network is stronger

---

## The Input Tensor

> [!NOTE] What’s a `Tensor`?
> A tensor is just an <u>N-dimensional array</u> of a <u>specific shape</u>.
>
> **Notation:** *The shape is described as the size of each dimension, in square brackets, like this:*
>
> **[]** - Just a single value / scalar — this is a zero-dimensional tensor
> 
> **[2048]** - A Vector, or a 1-dimensional array *(i.e., just a plain old “array”)*, 2048 elements long
> 
> **[8, 8]** - a 2-dimensional array, where both dimensions have a size of 8. This is like a square grid for Chess or Checkers (8x8)
> 
> **[16, 4, 2]** - a 3-dimensional array, with sizes 16, 4, and 2. You can think of this as a 16x4 rectangular grid that is two layers deep. Or you could think of it as a 4x2 grid that is 16 layers deep.
> 

The board is represented as a 3D array — 18 planes/layers, each an 8×8 grid of floats. Think of it as 18 spreadsheets stacked on top of each other, each answering one specific question about the board.

**Always encoded from the <u>current</u> player's perspective.** Whoever's turn it is, their pieces are "my pieces." The board is flipped vertically if needed so the current player is always at the bottom. This means the network never needs to learn separate strategies for white and black — chess knowledge is symmetric.

### The 18 Planes

| Plane | Content | Encoding |
|-------|---------|----------|
| 0 | My pawns | 1.0 where piece exists, 0.0 elsewhere |
| 1 | My knights | 1.0 where piece exists, 0.0 elsewhere |
| 2 | My bishops | 1.0 where piece exists, 0.0 elsewhere |
| 3 | My rooks | 1.0 where piece exists, 0.0 elsewhere |
| 4 | My queens | 1.0 where piece exists, 0.0 elsewhere |
| 5 | My king | 1.0 where piece exists, 0.0 elsewhere |
| 6 | Opponent pawns | 1.0 where piece exists, 0.0 elsewhere |
| 7 | Opponent knights | 1.0 where piece exists, 0.0 elsewhere |
| 8 | Opponent bishops | 1.0 where piece exists, 0.0 elsewhere |
| 9 | Opponent rooks | 1.0 where piece exists, 0.0 elsewhere |
| 10 | Opponent queens | 1.0 where piece exists, 0.0 elsewhere |
| 11 | Opponent king | 1.0 where piece exists, 0.0 elsewhere |
| 12 | My kingside castling right | All 1.0s or all 0.0s |
| 13 | My queenside castling right | All 1.0s or all 0.0s |
| 14 | Opponent kingside castling right | All 1.0s or all 0.0s |
| 15 | Opponent queenside castling right | All 1.0s or all 0.0s |
| 16 | En passant target square | Single 1.0 on capturable square, 0.0 elsewhere |
| 17 | Halfmove clock | All same value, normalized 0.0–1.0 |

### Design Decisions Worth Remembering

**Why castling gets whole planes of 1s/0s:** Castling availability cannot be inferred from the current board. Two positions with identical piece placement can have different castling rights depending on move history. The plane carries that lost history forward. The all-1s/all-0s approach is spatially redundant but unambiguous and consistent with everything else.

**Why only one en passant plane:** En passant must be exercised immediately on the turn following a double pawn push, or the right expires. Since the board is always encoded from the current player's perspective, the only valid en passant state is one the current player can execute. Any en passant opportunity the opponent had was on their turn and has already expired.

**Why halfmove clock instead of total move count:** The halfmove clock tracks consecutive moves since the last pawn move or capture. It directly encodes fifty-move rule proximity. Total move count is a less useful phase indicator — the network can infer phase from piece configuration anyway.

**Why current-player-relative encoding:** Eliminates spurious color asymmetry. The network learns chess concepts, not "what white should do" separately from "what black should do." Training signal is symmetric — every position contributes equally regardless of which color is moving. Value head outputs +1 for current player winning, -1 for losing — always relative, never absolute.

---

## The Network Architecture

### Overview

```
input (18 × 8 × 8)
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

One conv layer that takes 18 input planes to 128 output channels:

```
3×3 conv, 128 filters, padding=1
batch norm
ReLU
→ 128 × 8 × 8
```

Its only job is to translate from "18 planes of chess facts" into "128 channels of learned features" that the tower can work with.

**Terminology note — planes vs channels:** These mean the same thing (one 8×8 grid in the tensor), but convention shifts here. The input uses "planes" because each layer has a human-interpretable meaning you designed — "my pawns," "castling rights." After the stem, the 128 layers are abstract features the network discovered during training with no human-interpretable meaning, so they're called "channels" (the standard neural network term).

**How padding=1 preserves the 8×8 board:** The 3×3 filter needs to read a center square plus its 8 neighbors. Without padding, the filter can only center on squares that have a full 3×3 neighborhood of real data — that excludes the entire border of the board, shrinking the output to 6×6. After a few conv layers the board would disappear entirely.

Padding=1 adds a 1-wide border of zeros around each input plane, expanding each 8×8 plane to 10×10 before the filter slides across it:

```
0 0 0 0 0 0 0 0 0 0
0 . . . . . . . . 0      . = real board data
0 . . . . . . . . 0      0 = padding (zeros)
0 . . . . . . . . 0
0 . . . . . . . . 0
0 . . . . . . . . 0
0 . . . . . . . . 0
0 . . . . . . . . 0
0 . . . . . . . . 0
0 0 0 0 0 0 0 0 0 0
```

Now the filter can center on every original board position, including edges and corners. For a center square like e4, all 9 values in the 3×3 window are real data. For an edge square like a4, three values come from the zero padding:

```
0  a5  b5
0  a4  b4        ← filter centered on a4
0  a3  b3
```

For a corner like a8, five values are zeros:

```
0   0   0
0  a8  b8        ← filter centered on a8
0  a7  b7
```

The zeros don't add information — they just let the math run on edge and corner squares so the output stays 8×8:

```
output_size = input_size - kernel_size + 2×padding + 1
            = 8 - 3 + 2(1) + 1
            = 8
```

This padding happens independently on each of the 18 input planes. The filter reads across all 18 padded planes simultaneously at each position (see [The Convolution Operation](#the-convolution-operation) for the full breakdown of this math). 128 filters produce 128 output channels. Board edges matter (rooks on the a-file, kings in corners) so you never want to lose them.

Skip the math and move on to [The Tower](#the-tower)

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
    ↓                       (no ReLU here — conv2 needs to produce negative
                             values so the skip addition can subtract, not just add)
+ original input           ← skip connection
ReLU                       ← ReLU goes here, after the skip addition
    ↓
output: 128 × 8 × 8
```

**Why two convolutions per block:** Each block has two convolutions with completely separate weights — conv1 has its own 128×128×3×3 weight block, conv2 has a different one. They learn different things. Conv1 detects features from the input. Conv2 combines those features into something higher-level. Together they compute the "correction" that gets added back to the input via the skip connection.

One conv alone doesn't have enough capacity — it can detect features but can't compose them in the same step. Three convs per block was tested in early ResNet research, but the gains are marginal while adding ~147K parameters per block, slowing inference by ~50% per block, and making the gradient path through the block longer (partially undermining the skip connection). AlphaZero and Leela Chess Zero both use 2 — it's the established sweet spot for chess engines.

**How the skip connection works:** The input arrives at the block and goes two places simultaneously:

```
input ──→ conv1 → bn → ReLU → conv2 → bn ──→ (+) → ReLU → output
  │                                              ↑
  └──────────────────────────────────────────────┘
                    skip (the original input, unchanged)
```

One path goes through both convolutions, which transform it. The other path goes straight to the addition at the end, completely unchanged — it bypasses the convolutions entirely. Conv1 sees the block's input. Conv2 sees conv1's processed output. Neither convolution sees the skip. Then at the `+`, the two paths meet and get added element-wise (each of the 128 × 8 × 8 values added to its counterpart).

The block's output is:
```
output = original_input + what_the_convolutions_produced
```

Not just the convolution output — the original input is always preserved. Think of it like editing a document rather than rewriting it from scratch. Each block makes edits to the input. If the edits are bad, the original document is still there.

**Why skip connections matter for training:** Early in training, the convolution weights are random, so their output is essentially noise. Without the skip connection, that noise is all that passes to the next block — useful information from earlier blocks is destroyed. With the skip, even if the convolutions produce garbage, the original input passes through untouched. The convolutions just need to learn a small *improvement* on top of what's already there.

The skip also solves the vanishing gradient problem. During backpropagation, the error signal needs to travel from the output all the way back to the early blocks to update their weights. Through 8 blocks of convolutions, that signal gets weaker at every step. The skip connections provide a direct path — the gradient can flow straight through the additions without passing through any convolutions at all. Every block gets a strong training signal regardless of depth.

Early in training blocks can output near-zero corrections (acting as identity), only gradually learning to contribute. This is why residual networks train so reliably.

**Why 128 channels:** The channel count is the "width" of each block — how many dimensions the network has to represent different patterns at each square. With 128 channels, each block can track 128 different aspects of every board position simultaneously: pawn structure, piece pressure, king proximity, and dozens of other patterns the network discovers during training. These aren't discrete features like "is this square attacked: yes/no" — each channel is a continuous float, and the 128 channels interact (each convolution mixes all channels together), so they function as 128 dimensions of a shared representation rather than 128 independent detectors.

**Why 8 blocks (depth vs width tradeoff):** Each block adds one more level of abstraction — composing the previous block's patterns into increasingly complex ones:
- Early blocks see raw piece positions and detect simple spatial patterns (adjacency, piece types nearby)
- Middle blocks compose those into tactical patterns (pins, forks, open files — "the knight on f6 is attacking the square the bishop on e4 defends")
- Late blocks compose tactics into strategy (king safety, piece coordination — "the king is exposed on the queenside while the opponent has a rook on an open file pointing that direction")

Nobody programs these levels — they emerge because composing simple patterns repeatedly produces increasingly complex ones.

Each 3×3 conv expands the receptive field — the area of the board one output value can "see" — by 2 squares in each direction. After ~5 blocks every square can influence every other square. The remaining blocks deepen the abstraction rather than expanding coverage. More blocks beyond 8 add subtlety but with diminishing returns, and each block adds ~295K parameters and slows inference.

The tradeoff is width (channels) vs depth (blocks). Both add capacity, but they add different kinds:
- More channels → richer representations at each level (more patterns per block)
- More blocks → more levels of composition (deeper abstraction)

AlphaZero and Leela Chess Zero development found that wider networks tend to outperform deeper ones at equivalent parameter counts. The reason: a wider block (more channels) adds capacity without adding sequential inference steps, while more blocks add steps that can't be parallelized. GPUs are good at parallelism, so width is "cheaper" than depth in wall-clock time. This is why we target 128 filters × 8 blocks rather than 64 filters × 16 blocks.

### The Policy Head - What Move Should I Make?

Takes the 128 × 8 × 8 trunk output and produces 4096 move probabilities.

```
1×1 conv, 2 filters        → 2 × 8 × 8
batch norm, ReLU
flatten                    → 128 numbers
fully connected (128→4096) → 4096 numbers
softmax                    → 4096 probabilities summing to 1.0
```

The 1×1 conv compresses 128 channels to 2 before flattening. Without it, flattening 128 × 8 × 8 = 8,192 inputs into a FC layer targeting 4096 outputs would require 33 million weights in that one layer alone. The fully connected layer is a matrix multiplication — every input value connects to every output value through a learned weight (see [Fully Connected Layers](#fully-connected-layers) for the math).

**Move encoding:** Each of the 4096 outputs maps to one (from_square, to_square) pair:
```
index = from_square × 64 + to_square
```
Squares numbered 0–63, row by row from rank 8. Most outputs will be near zero (illegal or obviously bad moves). A few legal reasonable moves get most of the probability mass.

**Illegal move masking:** After getting policy output, compute all legal moves for the position, zero out illegal move indices, renormalize so remaining probabilities sum to 1.0.

**Pawn promotion:** This encoding doesn't distinguish promotion piece. For simplicity, assume queen promotion always — correct 99.9% of the time. Underpromotion to knight is the only realistic exception (avoiding stalemate in rare endgame positions).

**Softmax function:** Takes unbounded floats, exponentiates each, divides by sum. Result is all-positive, sums to 1.0. The exponentiation amplifies differences — the network's most confident move gets disproportionately more probability, weak moves get squeezed toward zero.

### The Value Head - Am I Winning?

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

The intermediate 64→64 [fully connected layer](#fully-connected-layers) gives the head capacity to combine spatial features before collapsing to a scalar. Going straight to 1 output is too aggressive.

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

It's the same idea as convolution — paired multiplications summed together — but with no spatial structure. Every input connects to every output. The convolution only looks at 3×3 neighbors; the FC layer looks at everything at once.

---

## How the Operations Work (Math Detail)

### The Convolution Operation

A convolution produces a completely new output tensor — it does not modify the input or the weights. The input is read, the weights are applied, and the result is a fresh tensor.

```
input (read-only)          weights (fixed during inference)
        \                  /
    multiply paired values and sum everything
                |
                v
    new output tensor (separate from input)
```

**How one filter computes one output value at one position:**

Each filter is a 3D block of weights — for the stem that's 18×3×3 (18 channels × 3 rows × 3 columns = 162 weights). At a given board position (row, col), the filter reads the 3×3 spatial window across every input channel. Each weight is paired with exactly one input value at the same channel and spatial offset — it's not every weight against every input:

```
For one channel, at position (row, col):

weight[0][0] × input[row-1][col-1]     ← top-left
weight[0][1] × input[row-1][col  ]     ← top-center
weight[0][2] × input[row-1][col+1]     ← top-right
weight[1][0] × input[row  ][col-1]     ← middle-left
weight[1][1] × input[row  ][col  ]     ← center
weight[1][2] × input[row  ][col+1]     ← middle-right
weight[2][0] × input[row+1][col-1]     ← bottom-left
weight[2][1] × input[row+1][col  ]     ← bottom-center
weight[2][2] × input[row+1][col+1]     ← bottom-right
```

That's 9 paired multiplications for one channel. Each multiplication pairs one weight with the input value at the same spatial position — not every weight against every input.

This repeats for all input channels (18 in the stem, 128 in the tower). Every channel uses its own 3×3 weights, and the multiplications stay within their own channel. The cross-channel mixing happens at the end: all the products across all channels and all 9 spatial positions get summed together into one number:

```
output[row][col] = sum of all (weight × input) pairs across all channels and all 3×3 positions
```

For the stem: 9 positions × 18 channels = 162 multiply-and-adds → one output value.
For the tower: 9 positions × 128 channels = 1,152 multiply-and-adds → one output value.

**From one value to a full output tensor:**

One filter sliding across all 64 board positions *(with `padding=1` as described in the `Stem` section)* produces one 8×8 output channel. 128 different filters — each with their own 18×3×3 (or 128×3×3) block of weights — produce 128 output channels. The result is a 128 × 8 × 8 tensor.

**1×1 convolution:** The same operation but with kernel size 1×1. No spatial neighborhood — just one position, across all channels. For one filter: 1 × 1 × 128 = 128 multiply-and-adds per position. Used in the heads to cheaply compress 128 channels to 2 or 1 before flattening.

### Batch Normalization

After each conv layer, values can end up in wildly different ranges — some channels near 0.001, others near 847 (not a meaningful number — just illustrating that scales can vary wildly). This causes:
- Training instability (large values → large gradients → overshooting weight updates)
- Layers constantly re-adapting as upstream distributions shift during training

Batch norm fixes this by normalizing each channel independently. For one channel, compute a single μ and σ across all spatial positions (64 squares) and all boards in the training batch. With batch size 256, that's 256 × 64 = 16,384 values going into one μ and σ calculation. Each of the 128 channels gets its own independent μ and σ.

Then every individual value in the channel goes through the same pipeline:

```
// x = one value coming out of the convolution layer (before normalization)

// 1. compute mean and variance for this channel (once, shared across all values)
μ = mean of all values in this channel
σ² = variance of all values in this channel

// 2. normalize to mean=0, std=1
x̂ = (x - μ) / √(σ² + ε)          // ε = 0.00001, prevents division by zero

// 3. scale and shift with learnable parameters (one pair per channel)
y = γ × x̂ + β

// y = the final output value, passed to ReLU
```

γ (gamma) and β (beta) start at 1.0 and 0.0 respectively — an identity transform, so batch norm does nothing at initialization and gradually learns the optimal scale and offset per channel during training.

At inference time (single positions during self-play), there's no batch to compute statistics from. Instead, batch norm uses running averages of μ and σ² accumulated during training — these are frozen and applied as fixed constants.

#### What are those little symbols above?

| Symbol | Name / Pronunciation | What it means here |
|--------|---------------------|--------------------|
| **μ**  | *mu* ("myoo" or "mew") | The average value across all positions in this `channel` — what we subtract to center the data at zero |
| **σ**  | *sigma* ("SIG-muh") | Standard deviation — how spread out the values are. σ² (sigma squared) is the variance. We divide by it to normalize the spread to 1 |
| **x̂** | *x-hat* | The normalized value after subtracting the mean and dividing by the standard deviation — centered at 0, spread of 1 |
| **ε**  | *epsilon* ("EP-si-lon") | A tiny constant (0.00001) added to prevent dividing by zero when the variance happens to be exactly 0 |
| **γ**  | *gamma* ("GAM-uh") | Learned `scale` parameter — one per channel. Lets the network adjust how spread out the normalized values should actually be |
| **β**  | *beta* ("BAY-tuh" or "BEE-tuh") | Learned `shift` parameter — one per channel. Lets the network adjust the center point away from zero if that's better |

#### Calculating standard deviation (σ)

```
Given N values: x₁, x₂, ..., xₙ

1. Compute mean:           μ = (x₁ + x₂ + ... + xₙ) / N
2. Subtract mean:          each (xᵢ - μ)
3. Square each difference: each (xᵢ - μ)²
4. Average them:           σ² = sum of all (xᵢ - μ)² / N
5. Take square root:       σ = √(σ²)
```

In batch normalization, `N` is the number of values in one channel across the batch (e.g. 16,384 for batch size 256 × 64 squares).

### ReLU - All negative values become zero

Rectified Linear Unit. Does one thing:

```
f(x) = max(0, x)
```

Negative inputs → 0. Positive inputs → unchanged.

**Why you need it:** Without nonlinearity, stacking conv layers collapses to a single linear transformation no matter how deep you go. Linear functions can only draw straight lines through data. Chess positions don't separate along straight lines. ReLU introduces the bends that let the network learn complex patterns.

**Why ReLU specifically:** Sigmoid and tanh saturate — for large inputs their gradients approach zero, causing vanishing gradients. ReLU has no ceiling on the positive side, so gradients flow freely. It's also trivially fast to compute.

After batch norm, values are centered near zero, so roughly half get zeroed by ReLU and half pass through. This is the intended operating range.

---

### Parameter Count (Small Config)

**Stem:**

| Component | Shape | Parameters |
|-----------|-------|------------|
| Conv weights | [128, 18, 3, 3] | 20,736 |
| BN gamma | [128] | 128 |
| BN beta | [128] | 128 |
| **Stem total** | | **20,992** |

**Per Residual Block (×8):**

| Component | Shape | Parameters |
|-----------|-------|------------|
| Conv1 weights | [128, 128, 3, 3] | 147,456 |
| BN1 gamma | [128] | 128 |
| BN1 beta | [128] | 128 |
| Conv2 weights | [128, 128, 3, 3] | 147,456 |
| BN2 gamma | [128] | 128 |
| BN2 beta | [128] | 128 |
| **Per block** | | **295,424** |
| **×8 blocks** | | **2,363,392** |

**Policy Head (What Move Should I Make?):**

| Component | Shape | Parameters |
|-----------|-------|------------|
| 1×1 conv weights | [2, 128, 1, 1] | 256 |
| BN gamma | [2] | 2 |
| BN beta | [2] | 2 |
| FC weights | [128, 4096] | 524,288 |
| FC bias | [4096] | 4,096 |
| **Policy total** | | **528,644** |

**Value Head (Am I Winning?):**

| Component | Shape | Parameters |
|-----------|-------|------------|
| 1×1 conv weights | [1, 128, 1, 1] | 128 |
| BN gamma | [1] | 1 |
| BN beta | [1] | 1 |
| FC1 weights | [64, 64] | 4,096 |
| FC1 bias | [64] | 64 |
| FC2 weights | [64, 1] | 64 |
| FC2 bias | [1] | 1 |
| **Value total** | | **4,355** |

**Total:**

| Section | Parameters |
|---------|------------|
| Stem | 20,992 |
| 8 residual blocks | 2,363,392 |
| Policy head | 528,644 |
| Value head | 4,355 |
| **Total** | **2,917,383 (~2.9M)** |

---

## The Self-Play Training Loop

```
initialize: random network weights

loop forever:
    1. generate N self-play games using current network
    2. add all positions to replay buffer
    3. sample mini-batches from buffer, train candidate network
    4. play G evaluation games (G/2 each color)
    5. if candidate wins >55% of decisive games → promote
       else → discard, keep current
    6. go to 1
```

### Move Selection During Self-Play

Each move, the network evaluates the current position and outputs a probability distribution over all moves. Select the move to play by sampling from the legal move probabilities (after masking illegal moves and renormalizing). Sampling rather than always picking the highest-probability move introduces variety into games — necessary so the network encounters diverse positions during training. Early in training when the network knows nothing, this is effectively random play.

### Training Data Generation

Each game of M moves generates M training positions — one per move, each encoded from the current player's perspective.

Each training position stores:
- The board tensor (18 × 8 × 8)
- The move played, as a one-hot vector (4096 floats) — policy target
- The game outcome from that player's perspective (+1, 0, -1) — value target

### The Replay Buffer

A fixed-size pool of recent positions (e.g. last 500,000). Training samples random mini-batches from this pool rather than processing whole games sequentially. Benefits:
- Decouples game generation from training
- Each position gets used multiple times across different mini-batches
- Prevents overfitting to the most recent games

### Training the Candidate Network

Always train a *candidate* — never update the current network in place while it's generating games. Mixing training updates into game generation creates unstable feedback loops.

**Two loss functions trained jointly, plus weight decay:**

```swift
// policy: outcome-weighted cross entropy vs the move actually played
let policyLoss = -z * log(policyOutput[movePlayed])

// value: mean squared error vs actual game outcome
let valueLoss = (z - valueOutput) * (z - valueOutput)

// L2 weight decay on network parameters
let regLoss = c * sumOfSquares(networkWeights)

let totalLoss = policyLoss + valueLoss + regLoss
```

#### Policy loss: `L = −z · log p(a*)`

- **p** — the policy head's predicted probability distribution over all 4096 moves (after softmax, masked to legal moves — see below).
- **a\*** — the move actually played from this position during self-play.
- **z** — the game outcome from the side-to-move's perspective at this position: `+1` won, `−1` lost, `0` drew.

This is the general cross-entropy loss `−Σ π(a) log p(a)` specialized to a one-hot target on `a*` and scaled by the game result. Behavior by outcome:

- `z = +1` → `L = −log p(a*)` → normal cross-entropy, pushes `p(a*)` *up*. "This move was on a winning path — do more of it."
- `z = −1` → `L = +log p(a*)` → sign-flipped, pushes `p(a*)` *down*. "This move was on a losing path — do less of it."
- `z =  0` → `L = 0` → draws contribute nothing to policy training.

Without the `z` weighting, the network would just train to imitate its own moves — a fixed point with no learning signal. The game outcome is what turns self-play into actual improvement: winning-side moves become positive examples, losing-side moves become negative examples.

**Why not just train on winning-side positions only?** It works, but it's wasteful — you throw away half the data, and draws contribute nothing either. Outcome-weighting is strictly better because losing positions still produce useful gradient (pushing bad moves *down*), whereas filtering discards them entirely.

**Caveat — unbounded loss when `z = −1`:** as `p(a*) → 0`, `+log p(a*) → −∞`, so the policy loss is unbounded below on losing positions. In practice this can destabilize training if the learning rate is too high. Mitigations if it becomes a problem: clip the loss, clip gradients, lower the learning rate, or fall back to winning-side-only filtering for the first few training iterations.

**Illegal-move masking.** The policy head's 4096 outputs cover every possible (from, to) square pair, including plenty of moves that are illegal in any given position (blocked, off-board, leaves king in check, wrong piece, etc.). Before the softmax, set the logits of all illegal moves to a large negative number (e.g. `-1e9`). After softmax those slots become exactly `0.0` and the legal moves renormalize to sum to `1.0` among themselves. The network structurally *cannot* emit an illegal move — no wasted capacity learning legality, no contradictory gradients across positions where the same move index is legal in one and illegal in another. Legality is a hard constraint, not a soft preference learned from data.

#### Value loss: `L = (z − v)²`

- **v** — the value head's predicted outcome, a scalar in `[−1, +1]` (via `tanh`) from the side-to-move's perspective.
- **z** — the same outcome target as above: `+1` won, `−1` lost, `0` drew.

Unlike the policy side, **every position contributes** to the value loss — winners, losers, *and* draws. A draw isn't a zero signal here; `z = 0` is an informative target that says "this position was balanced, predict zero." Over many positions the value head converges to the *expected* outcome: "what fraction of a win is this position worth, averaged across all positions like it."

**Why MSE instead of cross-entropy?** Value is a regression problem — predict a continuous scalar, minimize squared distance from the true number. Cross-entropy requires a probability distribution over discrete classes, which doesn't naturally fit a single `[−1, +1]` output. Some implementations discretize the outcome into {win, draw, loss} and use cross-entropy on that three-class distribution, but MSE on a scalar is simpler and works fine for our purposes.

#### Combined loss

```
total = (z - v)²  +  (-z · log p(a*))  +  c · ‖θ‖²
        └─value─┘    └─── policy ───┘    └─L2 reg─┘
```

The `c · ‖θ‖²` term is standard L2 weight decay — a small penalty on the sum of squared network parameters to keep weights from blowing up. Not chess-specific, just healthy regularization. Typical `c` values are around `1e-4`.

The shared trunk learns representations useful for both tasks simultaneously. Gradients from both the policy and value losses flow back through the same convolutional body, forcing it to encode features that are simultaneously good at "which move?" and "who's winning?" Policy and value heads constrain each other toward a consistent understanding of the position.

#### Future: MCTS changes the policy target

Once MCTS is added (later phase — not part of the bootstrap), the policy target changes. Instead of a one-hot on the played move weighted by the game outcome, the target `π` becomes the MCTS visit-count distribution at that position: `π(a) ∝ N(s,a)^(1/τ)`. The loss reverts to the full `−Σ π(a) log p(a)` form (no more `z` scaling), because MCTS is a stronger player than the raw network and training `p` to imitate `π` is itself a policy improvement step — the outcome weighting is no longer needed as the learning signal.

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
| Small  | 128     | 8      | ~2.9M  | Sweet spot for self-play on Mac |
| Medium | 256     | 10     | ~25M   | Too slow for self-play volume |

**The small config (128 filters, 8 blocks) is the target.** At ~2ms inference time, each move is one network evaluation — fast enough to generate a high volume of self-play games. Medium config hits ~8ms per evaluation, reducing training data throughput.

Inference speed × moves per game × games per hour = training data rate. Faster inference = more games = faster learning.

---

## Implementation Stack

```
Swift + MPSGraph          → network definition, training, and inference on Apple Silicon GPU
Swift                     → chess rules, move generation, self-play loop, training orchestration
```

The entire system in one language, with the network implemented directly in Metal Performance Shaders (MPSGraph).

---

## Key Concepts Summary

| Term | What it means |
|------|---------------|
| Tensor | Multi-dimensional array. 18 × 8 × 8 = 18 stacked 8×8 grids |
| Channel / plane | One 8×8 layer in the tensor |
| Conv 3×3 | Filter that looks at each square + 8 neighbors across ALL channels |
| 1×1 conv | Filter that mixes channels at each square with no spatial neighborhood |
| Receptive field | How much of the board one output value can "see" — grows with depth |
| Batch norm | Rescales channel values to stable range after each conv |
| ReLU | Zeros out negatives — introduces nonlinearity so network can learn complex patterns |
| Residual / skip | Adds block input back to output — enables gradient flow through deep networks |
| Policy head | Outputs 4096 move probabilities — guides move selection during self-play |
| Value head | Outputs single float [-1,1] — evaluates how good the position is for the current player |
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

---

## Future Improvements

Things intentionally left out of the initial implementation to keep complexity manageable. Each builds on a working foundation.

### Add MCTS

The initial version uses the network's policy output directly — sample from the legal move probabilities. No tree search. This means the engine plays at the raw strength of the network with no lookahead. Games will be weaker and noisier, and training will need more games to converge, but it eliminates an entire subsystem's worth of debugging while learning how the network and training loop work.

Once the network trains and improves from pure self-play, add MCTS (Monte Carlo Tree Search) to get lookahead. The value head is already there — MCTS just needs it to evaluate leaf nodes. The training loop changes: policy targets become MCTS visit count distributions instead of one-hot move vectors.

**What MCTS is:** A way to decide which move to make by selectively exploring a tree of possible futures. Rather than searching exhaustively (impossible — chess has ~10^120 positions), it builds a running estimate of move quality by focusing compute on the most promising branches.

**The four phases (repeat thousands of times):**

1. **Select:** Starting from the root (current position), follow the tree downward, at each node choosing the child with the best UCB score:

```
UCB = wins/visits  +  C × √(ln(parent_visits) / visits)
         ↑                        ↑
    exploitation              exploration
```

C ≈ 1.4 controls the exploration/exploitation balance. Nodes with few visits get a large exploration bonus. Once visited enough times, their actual win rate takes over.

2. **Expand:** At a node that hasn't been fully explored, add a new child node for an untried legal move.

3. **Evaluate:** Call the neural network on the new node's position. The value output becomes the simulation result.

4. **Backpropagate:** The value travels back up the tree, updating visit counts and win estimates for every node on the path.

After thousands of iterations, pick the move with the most visits from the root.

**Why it helps:** MCTS makes training data better — games contain genuine combination thinking the network couldn't find alone, and the visit count distribution is a much richer policy target than a one-hot move label. It's essentially a way to get more learning per game. With limited compute (one Mac) it tips from "dramatically helpful" toward "practically necessary."

### Upgrade UCB1 to PUCT

With MCTS in place and the network producing policy priors, replace the UCB1 selection formula with PUCT:

```
PUCT = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))
```

This lets the network's policy output guide which branches MCTS explores — moves the network thinks are promising get searched more deeply, moves it thinks are bad get fewer simulations. On a limited simulation budget (800 sims/move on one Mac), this is how you get strong play without exhaustive search.
