# Higher precision, worse model: a silent fp32 gradient bug in MPSGraph (draft)

*Draft for the owner to edit. Numbers are from `README.md` in this folder.*

## The experiment

Drew's Chess Machine trains a no-search chess network (8.3M parameters, three residual blocks of 9×9 convolutions) on
Apple Silicon with MPSGraph, normally in bfloat16. We wanted to know whether full fp32 would end somewhere better, so
we took a saved training state at step 384,964, converted it to fp32 bit for bit, and resumed it on exactly the same
batches, schedule and seed as the bf16 run that continued from the same state.

Same weights, same data, more precision. It should have been at least as good.

## What happened

Within 400 steps the fp32 run's policy loss was 0.1 nats above the bf16 run's on identical batches. By step
390,000 it was 330 puzzle-Elo behind on a 90,000-puzzle set (1437 vs 1765), and it kept falling. No NaNs, no
alarms beyond one batch-norm variance warning that cleared itself.

## Finding it

1. **Not the conversion.** Every tensor of the fp32 start file was bit-identical to the bf16 source, and step 1's loss
   matched to four digits.
2. **One step, two precisions.** We ran a single training step in each precision from the same state and recovered the
   gradient from the saved momentum buffers. Every tensor agreed to cosine ≥ 0.993 — except the first convolution of each
   block (cosine 0.42–0.57). The second convolution, the same shape, agreed at 0.999.
3. **A law it was breaking.** That first convolution feeds a batch norm, which makes the loss blind to the convolution's
   weight scale, so its true gradient must be perpendicular to its weights. In bf16 it was. In fp32 it pointed along the
   weights, and over 25,000 steps those weights grew from norm 13 to 43.
4. **Out of the app.** A 60-line Swift program with no app code and a float64 CPU reference reproduced it: one fp32
   convolution, 9×9, 128 channels, batch 4096, whose output is also read by something else — weight gradient 75% wrong.
5. **The boundary.** Batch 3236: exact. Batch 3264: 17% wrong. 4096 × 64 squares × 128 channels × 81 taps is 2.7 billion;
   the error starts exactly where that product passes 2³¹. A 7×7 kernel flips at its own 2³¹ point. At batch 8192 MPS
   aborts trying to allocate a buffer of exactly 8 bytes per element of that product (40.5 GiB).

## Why bf16 was fine

bf16 and fp16 stay correct even past 2³². They don't build the giant intermediate, so they must take a different kernel.
The fp32 path is chosen only when the convolution's output has more than one reader, which is exactly the
conv → batch-norm pattern in every modern residual block.

## The fix

Split each such convolution over its input channels and sum the halves. It's the same function with the same gradient,
but MPSGraph picks the correct kernel. The app now does this automatically in fp32 training graphs, logs it when the
model is built or converted, and keeps a test that fails the day MPSGraph changes.

## Takeaways

- **Higher precision is not a correctness oracle.** The fp32 run was the wrong one.
- **Check gradients against invariants, not just against each other.** Batch-norm scale invariance turned "these
  differ" into "this one is wrong".
- **Compare against the reference at the first matched checkpoint.** We had planned to check at 400k and didn't until
  413k; the numbers had been telling us since step 385,330.
- **Size-dependent kernel bugs hide in small tests.** Every isolated test at batch 512 passed.

*Reported to Apple as FB-[number].*
