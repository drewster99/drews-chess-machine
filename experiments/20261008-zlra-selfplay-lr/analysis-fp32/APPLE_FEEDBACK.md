# Feedback Assistant report (draft)

**Product / area:** Metal Performance Shaders Graph (MPSGraph) — automatic differentiation of `convolution2D`

**Title:** MPSGraph returns a wrong fp32 convolution weight gradient when N·H·W·C_in·kH·kW exceeds 2³¹ and the
convolution's output has another consumer

**Environment:** macOS 27.2 (26B123), Apple M5 Max, Xcode 27 (27B52), SDK 26B118. Not tested on other OS versions or chips.

## Description

For a float32 `convolution2D` (NCHW / OIHW, stride 1, explicit same padding) whose output tensor is used by more than one
consumer (for example a training-mode batch normalization that reads it for mean, variance and normalize, or simply when
the output is also requested as a target), the gradient with respect to the **weights** returned by
`gradients(of:with:name:)` is wrong once N × H × W × C_in × kH × kW exceeds 2³¹ − 1 elements.

- The forward output and the gradient with respect to the input are exact.
- bfloat16 and float16 are correct at the same shapes and at twice the size.
- The same shape is exact when the output has a single consumer.
- `graph.run` and a compiled `MPSGraphExecutable` at `.level1` give identical wrong results.
- At N = 8192 the float32 case aborts:
  `MPSCommandBufferImageCache.mm:1420: failed assertion 'Failed to allocate private MTLBuffer for size 43486543872'`.
  43,486,543,872 is exactly 8 × N·H·W·C_in·kH·kW. This suggests the float32 weight-gradient path materializes an
  intermediate of that many elements and indexes it with 32 bits.

## Measurements

Weight gradient vs. an exact float64 CPU correlation of the input and the output gradient. H = W = 8,
C_in = C_out = 128, 9×9 kernel unless noted. "Relative error" is ‖dW_MPSGraph − dW_exact‖ / ‖dW_exact‖.

| N | kernel | N·H·W·C_in·k² | float32 relative error |
|---:|---:|---:|---:|
| 3236 | 9×9 | 2.147·10⁹ (< 2³¹) | 1.1·10⁻⁶ |
| 3264 | 9×9 | 2.166·10⁹ (> 2³¹) | 0.169 |
| 4096 | 9×9 | 2.72·10⁹ | 0.747 |
| 5312 | 7×7 | 2.13·10⁹ (< 2³¹) | 1.3·10⁻⁶ |
| 5376 | 7×7 | 2.16·10⁹ (> 2³¹) | 0.094 |
| 8192 | 9×9 | 5.44·10⁹ | assertion failure (allocation of 40.5 GiB) |

bfloat16 at N = 4096 and 8192: 1.6·10⁻³ and 1.7·10⁻³ (rounding level). float16: 1.8·10⁻⁴ and 1.9·10⁻⁴.
float32 at N = 4096 with the output used only by the loss: 4.9·10⁻⁶.

## Steps to reproduce

Attached `conv_weight_gradient_overflow.swift` (no dependencies beyond Metal / MPSGraph):

```sh
swiftc -O conv_weight_gradient_overflow.swift -o convgrad
./convgrad 4096 9 128 none none   # relative error 0.747
./convgrad 3236 9 128 none none   # relative error 1.1e-6
```

The program builds Y = conv2d(X, W), loss = Σ R·Y, requests ∂loss/∂W together with Y, and compares ∂loss/∂W with
the float64 correlation of X and ∂loss/∂Y read from the same graph.

## Expected

The float32 weight gradient is correct (relative error ~10⁻⁶) at every size, or the graph fails with an error instead of
returning a wrong result.

## Impact

Silent: a float32 training run on a 9×9 / 128-channel convolutional network at batch 4096 trained on this gradient for
28,814 steps; the network degraded steadily with no error or non-finite value.

## Workaround

Splitting the convolution over its input channels (sum of two convolutions over channel halves) gives the correct
gradient at all sizes tested (N up to 32,768).

## Attachments

- `conv_weight_gradient_overflow.swift` — minimal repro with float64 reference
- `conv_weight_gradient_workarounds.swift` — dtype and workaround variants
- `conv_forward_and_input_gradient.swift` — forward output and input gradient are exact
