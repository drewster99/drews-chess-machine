# MPSGraph Primitives for Chess Engine

A reference document covering the Metal Performance Shaders Graph (MPSGraph) primitives used to implement the chess engine network in Swift. Written conversationally to be readable after time away.

---

## The Core Idea: Declarative Graph Building

MPSGraph is **declarative**. You don't execute operations immediately as you write them. Instead you build a graph describing what to compute — the entire forward pass, loss functions, gradients, and weight updates — and then execute the whole thing at once on the GPU.

Think of it like writing a recipe before cooking rather than cooking one step at a time.

```swift
// BUILD PHASE — done once at startup
let graph = MPSGraph()
// ... define everything here — layers, losses, gradients, optimizer

// EXECUTE PHASE — done every training step
let results = graph.run(
    feeds: [
        input: boardTensorData,
        policyTarget: mctsVisitData,
        valueTarget: gameOutcomeData
    ],
    targetTensors: [policyOutput, valueOutput, totalLoss],
    targetOperations: [updateOps]
)
```

One `graph.run()` call does the entire forward pass, loss computation, backward pass, and weight update — all on the GPU.

---

## MPSGraph

The graph itself. Everything lives inside one of these.

```swift
let graph = MPSGraph()
```

This is your network. You build the entire computation as a graph of operations inside this object.

---

## MPSGraphTensor

Every value flowing through the network is an `MPSGraphTensor`. This is not actual data — it's a **placeholder in the graph** describing the shape and type of data that will flow through that point when the graph executes.

```swift
// the input tensor — batch dimension first, then 19 × 8 × 8
let input = graph.placeholder(
    shape: [batchSize, 19, 8, 8],
    dataType: .float32,
    name: "board_input"
)
```

The batch dimension means you process multiple positions simultaneously — important for training efficiency. During self-play you might use batchSize=1 for single position inference. During training you might use batchSize=256 or more.

Every operation in the graph takes `MPSGraphTensor` inputs and produces `MPSGraphTensor` outputs. The tensors are the edges of the graph — operations are the nodes.

---

## MPSGraphVariable (weights)

The learned parameters — conv filter weights, batch norm gamma/beta, FC layer weights — are `MPSGraphVariable` tensors. Unlike placeholders which receive new data each run, variables **persist between runs** and get updated during training.

```swift
// stem conv weights: 128 output filters, 19 input channels, 3×3 kernel
// shape is [outputChannels, inputChannels, kernelHeight, kernelWidth]
let stemWeights = graph.variable(
    with: randomData,           // initial values — usually small random floats
    shape: [128, 19, 3, 3],
    dataType: .float32,
    name: "stem_conv_weights"
)

// a residual block's first conv weights
// 128 output filters, 128 input channels, 3×3 kernel
let block1Conv1Weights = graph.variable(
    with: randomData,
    shape: [128, 128, 3, 3],
    dataType: .float32,
    name: "block1_conv1_weights"
)
```

These are what backpropagation updates. Every weight matrix, every bias, every batch norm parameter in your network is one of these.

**Weight initialization matters.** Starting all weights at zero means every neuron computes the same thing and learns the same thing — they never differentiate. The standard approach for conv layers is **He initialization** — random values scaled by `sqrt(2 / fan_in)` where fan_in is the number of input connections per output. For a 3×3×128 filter that's `sqrt(2 / (3×3×128)) ≈ 0.047`.

```swift
func heInitData(shape: [Int]) -> Data {
    let fanIn = shape.dropFirst().reduce(1, *)
    let std = sqrt(2.0 / Float(fanIn))
    let count = shape.reduce(1, *)
    var values = [Float](repeating: 0, count: count)
    for i in 0..<count {
        // Box-Muller transform for normal distribution
        let u1 = Float.random(in: Float.ulpOfOne...1)
        let u2 = Float.random(in: 0...1)
        values[i] = std * sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
    return values.withUnsafeBytes { Data($0) }
}
```

---

## Convolution

The core operation of the network. MPSGraph has a dedicated conv2D operation.

```swift
let convDesc = MPSGraphConvolution2DOpDescriptor(
    strideInX: 1,           // move filter one square at a time
    strideInY: 1,
    dilationRateInX: 1,
    dilationRateInY: 1,
    groups: 1,
    paddingStyle: .explicit,
    dataLayout: .NCHW,      // batch, channels, height, width — standard layout
    weightsLayout: .OIHW    // output channels, input channels, height, width
)

// padding = 1 on all sides — keeps 8×8 spatial dimensions intact
convDesc.explicitPaddingWithPaddingLeft(
    1, paddingRight: 1,
    paddingTop: 1, paddingBottom: 1
)

let convOutput = graph.convolution2D(
    input,
    weights: stemWeights,
    descriptor: convDesc,
    name: "stem_conv"
)
// output shape: [batch, 128, 8, 8]
```

### Why padding=1

Without padding, a 3×3 conv shrinks the spatial dimensions — an 8×8 board becomes 6×6 after one conv, then 4×4, and your board position information disappears. Padding=1 adds a border of zeros around the input so edge squares have full 8-neighbor neighborhoods and output stays 8×8. The rule is:

```
padding = (kernel_size - 1) / 2
```

So 3×3 kernel → padding=1. If you ever use 5×5 → padding=2.

### What the conv actually computes

Each of the 128 filters has shape 3×3×128 — it looks at a 3×3 neighborhood across ALL 128 input channels simultaneously. At each square position: 3 × 3 × 128 = 1,152 multiplications, summed to one output value. 128 filters → 128 output planes. MPSGraph handles all of this automatically.

### 1×1 Convolution (used in heads)

Same operation but kernel size 1×1. No spatial neighborhood — just mixes across channels at each individual square. Used in the policy and value heads to cheaply compress 128 channels to 2 or 1 before flattening:

```swift
let headConvDesc = MPSGraphConvolution2DOpDescriptor(
    strideInX: 1, strideInY: 1,
    dilationRateInX: 1, dilationRateInY: 1,
    groups: 1,
    paddingStyle: .explicit,
    dataLayout: .NCHW,
    weightsLayout: .OIHW
)
// no padding needed for 1×1 conv — no spatial neighborhood to preserve
headConvDesc.explicitPaddingWithPaddingLeft(
    0, paddingRight: 0,
    paddingTop: 0, paddingBottom: 0
)

// policy head: compress 128 channels → 2
let policyConvWeights = graph.variable(
    with: heInitData(shape: [2, 128, 1, 1]),
    shape: [2, 128, 1, 1],
    dataType: .float32,
    name: "policy_conv_weights"
)

let policyConvOutput = graph.convolution2D(
    trunkOutput,
    weights: policyConvWeights,
    descriptor: headConvDesc,
    name: "policy_1x1_conv"
)
// output shape: [batch, 2, 8, 8]

// value head: compress 128 channels → 1
let valueConvWeights = graph.variable(
    with: heInitData(shape: [1, 128, 1, 1]),
    shape: [1, 128, 1, 1],
    dataType: .float32,
    name: "value_conv_weights"
)

let valueConvOutput = graph.convolution2D(
    trunkOutput,
    weights: valueConvWeights,
    descriptor: headConvDesc,
    name: "value_1x1_conv"
)
// output shape: [batch, 1, 8, 8]
```

---

## Batch Normalization

After a conv layer multiplies and sums 1,152 values, outputs can end up in wildly different ranges. Batch norm normalizes each channel across the training batch to mean=0, std=1, then applies learnable scale (gamma) and shift (beta).

```swift
// learnable parameters — one per channel
let bnGamma = graph.variable(
    with: onesData(count: 128),     // initialize gamma to 1
    shape: [1, 128, 1, 1],          // broadcast across batch, height, width
    dataType: .float32,
    name: "block1_bn1_gamma"
)

let bnBeta = graph.variable(
    with: zerosData(count: 128),    // initialize beta to 0
    shape: [1, 128, 1, 1],
    dataType: .float32,
    name: "block1_bn1_beta"
)

// running statistics — maintained separately, not learned via gradients
// these are updated during training and used fixed at inference time
var bnMean = MPSGraphTensor(...)    // running mean per channel
var bnVariance = MPSGraphTensor(...)  // running variance per channel

let bnOutput = graph.normalize(
    convOutput,
    mean: bnMean,
    variance: bnVariance,
    gamma: bnGamma,
    beta: bnBeta,
    epsilon: 1e-5,          // small value to avoid division by zero
    name: "block1_bn1"
)
```

`bnGamma` and `bnBeta` are `MPSGraphVariable` tensors — they're learned parameters updated during training just like conv weights. The running mean and variance are updated separately using an exponential moving average during training, then frozen at inference time.

---

## ReLU

One line. Zeros out all negative values, passes positive values through unchanged.

```swift
let reluOutput = graph.reLU(with: bnOutput, name: "block1_relu1")
```

Maps to `f(x) = max(0, x)` applied elementwise across the entire tensor. Without ReLU (or some other nonlinearity), stacking conv layers collapses to a single linear transformation no matter how deep you go — linear operations on linear operations are still linear. ReLU introduces the bends that let the network learn complex nonlinear patterns.

After batch norm, values are centered near zero, so roughly half get zeroed by ReLU and half pass through. This is the intended operating range.

---

## Addition (skip connection)

```swift
let skipOutput = graph.addition(blockInput, convOutput, name: "block1_skip_add")
```

This is the residual connection — adding the block's original input back to its processed output. MPSGraph handles the elementwise addition across the full 128 × 8 × 8 tensor.

The block computes a *correction* to its input rather than a full transformation:
```
output = input + what_the_block_learned
```

### How gradients flow through skip connections automatically

MPSGraph doesn't need to know about skip connections specially — it just knows about the graph. When you wrote `graph.addition(blockInput, convOutput)`, MPSGraph recorded that `skipOutput` depends on both `blockInput` and `convOutput`. Those in turn depend on earlier operations, all the way back to the weight variables.

When `graph.gradients()` is called, it walks the graph backwards using the chain rule. The derivative of addition is 1 for both inputs — gradient flows backward equally down both paths simultaneously:

```
forward:   skipOutput = blockInput + convOutput
backward:  gradient flows to blockInput at full strength  (the highway)
           gradient flows to convOutput at full strength  (through the block)
```

So early layers receive a strong gradient signal via the skip highway regardless of how many blocks they're behind. This is why residual networks train reliably where plain deep networks fail — it emerges automatically from the graph structure, not from any special gradient code.

---

## A Complete Residual Block

Putting the above together into one full block:

```swift
func residualBlock(
    graph: MPSGraph,
    input: MPSGraphTensor,
    blockIndex: Int
) -> MPSGraphTensor {

    let channels = 128

    // conv 1 weights: [outputChannels, inputChannels, kH, kW]
    let conv1Weights = graph.variable(
        with: heInitData(shape: [channels, channels, 3, 3]),
        shape: [channels, channels, 3, 3],
        dataType: .float32,
        name: "block\(blockIndex)_conv1_weights"
    )

    let conv2Weights = graph.variable(
        with: heInitData(shape: [channels, channels, 3, 3]),
        shape: [channels, channels, 3, 3],
        dataType: .float32,
        name: "block\(blockIndex)_conv2_weights"
    )

    let bn1Gamma = graph.variable(with: onesData(count: channels),
        shape: [1, channels, 1, 1], dataType: .float32,
        name: "block\(blockIndex)_bn1_gamma")
    let bn1Beta = graph.variable(with: zerosData(count: channels),
        shape: [1, channels, 1, 1], dataType: .float32,
        name: "block\(blockIndex)_bn1_beta")
    let bn2Gamma = graph.variable(with: onesData(count: channels),
        shape: [1, channels, 1, 1], dataType: .float32,
        name: "block\(blockIndex)_bn2_gamma")
    let bn2Beta = graph.variable(with: zerosData(count: channels),
        shape: [1, channels, 1, 1], dataType: .float32,
        name: "block\(blockIndex)_bn2_beta")

    // first conv → bn → relu
    var x = graph.convolution2D(input, weights: conv1Weights,
        descriptor: convDesc, name: "block\(blockIndex)_conv1")
    x = graph.normalize(x, mean: bn1Mean, variance: bn1Var,
        gamma: bn1Gamma, beta: bn1Beta, epsilon: 1e-5,
        name: "block\(blockIndex)_bn1")
    x = graph.reLU(with: x, name: "block\(blockIndex)_relu1")

    // second conv → bn (no relu yet — relu after skip add)
    x = graph.convolution2D(x, weights: conv2Weights,
        descriptor: convDesc, name: "block\(blockIndex)_conv2")
    x = graph.normalize(x, mean: bn2Mean, variance: bn2Var,
        gamma: bn2Gamma, beta: bn2Beta, epsilon: 1e-5,
        name: "block\(blockIndex)_bn2")

    // skip connection — add original input back
    x = graph.addition(input, x, name: "block\(blockIndex)_skip")

    // relu after skip add
    x = graph.reLU(with: x, name: "block\(blockIndex)_relu2")

    return x
}
```

---

## Reshape (flatten)

Unrolls a multi-dimensional tensor into a flat list of numbers. No math, no weights — just reshaping.

```swift
// policy head: flatten 2 × 8 × 8 → 128
let policyFlattened = graph.reshape(
    policyBnOutput,
    shape: [batchSize, 128],    // 2 × 8 × 8 = 128
    name: "policy_flatten"
)

// value head: flatten 1 × 8 × 8 → 64
let valueFlattened = graph.reshape(
    valueBnOutput,
    shape: [batchSize, 64],     // 1 × 8 × 8 = 64
    name: "value_flatten"
)
```

After flatten, spatial structure is abandoned. Up until this point the network always knew which square was which — the 8×8 layout was preserved. After flatten it's just a list of numbers and the FC layers combine them freely. That's fine because all the spatial reasoning has already been done by the conv layers.

---

## Fully Connected Layer (matrix multiplication)

MPSGraph doesn't have a dedicated FC layer — you use matrix multiplication directly. A fully connected layer is just a matrix multiplication where every input connects to every output through a learned weight.

```swift
// policy head: 128 → 4096
let policyFCWeights = graph.variable(
    with: heInitData(shape: [128, 4096]),
    shape: [128, 4096],
    dataType: .float32,
    name: "policy_fc_weights"
)

let policyLogits = graph.matrixMultiplication(
    primary: policyFlattened,       // [batch, 128]
    secondary: policyFCWeights,     // [128, 4096]
    name: "policy_fc"
)
// output shape: [batch, 4096]
// 128 × 4096 = 524,288 weights in this one layer

// value head: 64 → 64
let valueFCWeights1 = graph.variable(
    with: heInitData(shape: [64, 64]),
    shape: [64, 64],
    dataType: .float32,
    name: "value_fc1_weights"
)

let valueFCOutput1 = graph.matrixMultiplication(
    primary: valueFlattened,
    secondary: valueFCWeights1,
    name: "value_fc1"
)
// apply relu between FC layers in value head
let valueFCRelu = graph.reLU(with: valueFCOutput1, name: "value_fc1_relu")

// value head: 64 → 1
let valueFCWeights2 = graph.variable(
    with: heInitData(shape: [64, 1]),
    shape: [64, 1],
    dataType: .float32,
    name: "value_fc2_weights"
)

let valueFCOutput2 = graph.matrixMultiplication(
    primary: valueFCRelu,
    secondary: valueFCWeights2,
    name: "value_fc2"
)
// output shape: [batch, 1] — unbounded float, will be squashed by tanh
```

---

## Softmax and Tanh

```swift
// policy head: convert 4096 logits to probabilities summing to 1.0
let policyOutput = graph.softMax(
    with: policyLogits,
    axis: 1,                // apply across the move dimension
    name: "policy_softmax"
)

// value head: squash unbounded float to [-1.0, +1.0]
let valueOutput = graph.tanh(
    with: valueFCOutput2,
    name: "value_tanh"
)
```

**Softmax:** Exponentiates each value then divides by the sum. Result is all-positive, sums to 1.0. The exponentiation amplifies differences — the network's most confident move gets disproportionately more probability, weak moves get squeezed toward zero. Most of the 4096 outputs will be near zero (illegal or obviously bad moves).

**Tanh:** Squashes the entire real line to [-1, 1]. The FC layer could output -47 or +831 — tanh maps everything to the fixed range matching the game outcome encoding (-1 loss, 0 draw, +1 win).

---

## Illegal Move Masking

After getting policy output, zero out illegal moves and renormalize:

```swift
func maskIllegalMoves(
    graph: MPSGraph,
    policyOutput: MPSGraphTensor,
    legalMoves: [Int]   // list of legal move indices (0-4095)
) -> MPSGraphTensor {

    // build mask: 1.0 for legal moves, 0.0 for illegal
    var maskValues = [Float](repeating: 0, count: 4096)
    for moveIndex in legalMoves {
        maskValues[moveIndex] = 1.0
    }

    let mask = graph.constant(
        maskValues,
        shape: [1, 4096],
        dataType: .float32
    )

    // zero out illegal moves
    let masked = graph.multiplication(policyOutput, mask, name: "mask_illegal")

    // renormalize so remaining probabilities sum to 1.0
    let sum = graph.reductionSum(with: masked, axis: 1, name: "policy_sum")
    let normalized = graph.division(masked, sum, name: "policy_normalized")

    return normalized
}
```

Move index encoding:
```swift
func moveIndex(from: Int, to: Int) -> Int {
    return from * 64 + to
}
// e.g. e2(52) → e4(36): index = 52 * 64 + 36 = 3364
```

---

## Loss Functions

```swift
// value loss: mean squared error between prediction and actual outcome
// valueTarget is +1.0 (win), 0.0 (draw), or -1.0 (loss) for current player
let valueLoss = graph.meanSquaredError(
    labels: valueTarget,        // placeholder fed actual game outcomes
    predictions: valueOutput,
    name: "value_loss"
)

// policy loss: cross entropy between prediction and MCTS visit distribution
// (or one-hot move vector when training without MCTS)
// NOTE: pass logits (before softmax) to crossEntropyLoss — it applies
// softmax internally for numerical stability
let policyLoss = graph.crossEntropyLoss(
    labels: policyTarget,       // placeholder fed MCTS visit counts or one-hot
    logits: policyLogits,       // the raw FC output, before softmax
    reductionType: .mean,
    labelSmoothing: 0,
    axis: 1,
    name: "policy_loss"
)

// total loss — both heads trained jointly
let totalLoss = graph.addition(policyLoss, valueLoss, name: "total_loss")
```

When training without MCTS (initial phase), `policyTarget` is a one-hot vector — all zeros except a 1.0 at the index of the move that was actually played. When training with MCTS, it's the full visit count distribution — a much richer signal.

---

## Gradient Computation (automatic differentiation)

You don't implement backpropagation. You ask MPSGraph to compute gradients automatically:

```swift
// collect all learnable variables
let allVariables: [MPSGraphVariable] = [
    stemConvWeights,
    block1Conv1Weights, block1Conv2Weights,
    block1BN1Gamma, block1BN1Beta,
    block1BN2Gamma, block1BN2Beta,
    // ... all 8 blocks ...
    policyConvWeights, policyFCWeights,
    valueConvWeights, valueFCWeights1, valueFCWeights2,
    // ... all bn parameters ...
]

// MPSGraph differentiates the entire computation graph automatically
let gradients = graph.gradients(
    of: totalLoss,
    with: allVariables,
    name: "gradients"
)
// gradients is a dictionary: [MPSGraphVariable: MPSGraphTensor]
// each entry is the gradient of totalLoss with respect to that variable
```

MPSGraph knows the derivative of every operation it provides (conv, relu, softmax, addition, matmul, etc.) and chains them together via the chain rule automatically. The skip connections are not a special case — the graph recorded the addition operation, so gradients flow backward through both the skip path and the block path simultaneously, at full strength through the skip path.

This is automatic differentiation — the framework derives the backward pass from your forward pass description. You never write a single line of calculus.

---

## Weight Updates (Adam optimizer)

Adam (Adaptive Moment Estimation) is the standard optimizer for training neural networks. It adapts the learning rate for each parameter individually based on the history of gradients — parameters that have been getting consistent gradients get larger updates, noisy ones get smaller updates.

```swift
let optimizer = MPSGraphAdamOptimizer(
    graph: graph,
    learningRate: 0.001,    // how big each update step is
    beta1: 0.9,             // decay rate for first moment (gradient mean)
    beta2: 0.999,           // decay rate for second moment (gradient variance)
    epsilon: 1e-8           // small value for numerical stability
)

let updateOps = optimizer.applyUpdate(
    variables: allVariables,
    gradients: gradients
)
```

Including `updateOps` in `targetOperations` when running the graph triggers the weight updates. Without it, the graph computes gradients but doesn't apply them.

---

## Executing the Graph

### Training step

```swift
func trainingStep(
    boardPositions: [Float],    // batch of board tensors flattened
    policyTargets: [Float],     // MCTS visit distributions or one-hot moves
    valueTargets: [Float]       // game outcomes: +1, 0, -1
) {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!

    // wrap data in MPSGraphTensorData
    let boardData = MPSGraphTensorData(
        boardPositions,
        shape: [batchSize, 19, 8, 8],
        dataType: .float32
    )
    let policyData = MPSGraphTensorData(
        policyTargets,
        shape: [batchSize, 4096],
        dataType: .float32
    )
    let valueData = MPSGraphTensorData(
        valueTargets,
        shape: [batchSize, 1],
        dataType: .float32
    )

    // run forward pass + backward pass + weight update
    let results = graph.run(
        with: commandQueue,
        feeds: [
            inputPlaceholder: boardData,
            policyTargetPlaceholder: policyData,
            valueTargetPlaceholder: valueData
        ],
        targetTensors: [totalLoss, policyLoss, valueLoss],
        targetOperations: [updateOps]   // triggers weight updates
    )

    // read loss values for logging
    let loss = results[totalLoss]!.mpsndarray().readBytes(...)
}
```

### Inference (during self-play)

```swift
func evaluatePosition(board: [Float]) -> (policy: [Float], value: Float) {
    let boardData = MPSGraphTensorData(
        board,
        shape: [1, 19, 8, 8],   // batch size 1 for single position
        dataType: .float32
    )

    let results = graph.run(
        with: commandQueue,
        feeds: [inputPlaceholder: boardData],
        targetTensors: [policyOutput, valueOutput],
        targetOperations: []    // no weight updates during inference
    )

    let policy = // extract [Float] from results[policyOutput]
    let value = // extract Float from results[valueOutput]

    return (policy, value)
}
```

---

## The Full Variable Inventory

Every learnable parameter in the network:

```
stem:
    stemConvWeights         [128, 19, 3, 3]
    stemBNGamma             [1, 128, 1, 1]
    stemBNBeta              [1, 128, 1, 1]

per residual block (× 8):
    blockN_conv1Weights     [128, 128, 3, 3]
    blockN_bn1Gamma         [1, 128, 1, 1]
    blockN_bn1Beta          [1, 128, 1, 1]
    blockN_conv2Weights     [128, 128, 3, 3]
    blockN_bn2Gamma         [1, 128, 1, 1]
    blockN_bn2Beta          [1, 128, 1, 1]

policy head:
    policyConvWeights       [2, 128, 1, 1]
    policyBNGamma           [1, 2, 1, 1]
    policyBNBeta            [1, 2, 1, 1]
    policyFCWeights         [128, 4096]

value head:
    valueConvWeights        [1, 128, 1, 1]
    valueBNGamma            [1, 1, 1, 1]
    valueBNBeta             [1, 1, 1, 1]
    valueFCWeights1         [64, 64]
    valueFCWeights2         [64, 1]
```

Total: ~2.6M parameters for the small config (128 filters, 8 blocks).

---

## Tensor Shape Flow

The shape of the data at every point in the network (batch dimension omitted for clarity):

```
input:                      19 × 8 × 8

stem conv:                 128 × 8 × 8
stem bn + relu:            128 × 8 × 8

block 1 conv 1:            128 × 8 × 8
block 1 bn + relu:         128 × 8 × 8
block 1 conv 2:            128 × 8 × 8
block 1 bn:                128 × 8 × 8
block 1 skip add + relu:   128 × 8 × 8

... identical through blocks 2–8 ...

trunk output:              128 × 8 × 8

policy 1×1 conv:             2 × 8 × 8
policy bn + relu:             2 × 8 × 8
policy flatten:                     128
policy FC:                        4,096
policy softmax:                   4,096  ← move probabilities

value 1×1 conv:              1 × 8 × 8
value bn + relu:              1 × 8 × 8
value flatten:                       64
value FC1 + relu:                    64
value FC2:                            1
value tanh:                           1  ← position value [-1, +1]
```

---

## Key Concepts Summary

| Term | What it means |
|------|---------------|
| Declarative graph | Build the recipe first, cook (execute) later |
| MPSGraphTensor | A placeholder describing shape/type of data — not actual data |
| MPSGraphVariable | A persistent tensor holding learned weights — updated during training |
| Placeholder | Input to the graph — receives new data each run (board positions, targets) |
| convolution2D | 3D filter operation — looks at spatial neighborhood across all channels |
| normalize | Batch normalization — stabilizes value ranges between layers |
| reLU | Zeros out negatives — introduces nonlinearity |
| addition | Elementwise add — implements skip/residual connections |
| reshape | Flatten multi-dim tensor to 1D list — no math, just reshaping |
| matrixMultiplication | Fully connected layer — every input to every output |
| softMax | Converts logits to probabilities summing to 1.0 |
| tanh | Squashes unbounded float to [-1, 1] |
| crossEntropyLoss | Policy loss — penalizes difference from target distribution |
| meanSquaredError | Value loss — penalizes distance from actual game outcome |
| gradients(of:with:) | Automatic differentiation — computes ∂loss/∂weight for every weight |
| AdamOptimizer | Applies gradients to weights with adaptive learning rates |
| targetOperations | Operations to run (e.g. weight updates) when executing graph |
| He initialization | Weight init scaled by sqrt(2/fan_in) — prevents vanishing/exploding values |
| Automatic differentiation | Framework derives backward pass from forward pass — no manual backprop |
