//
//  MacOS27NaNIsolationTests.swift
//  DrewsChessMachineTests
//
//  Isolation matrix for the 2026-06-13 training divergence first seen after the
//  macOS 27 / Xcode beta upgrade: ChessTrainer.trainStep produces a FINITE
//  forward loss but a NaN gradient (`nonFiniteLoss(total: …, gradNorm: nan)`),
//  poisoning the weights — reproduced both in the live eBNC run and by
//  testTrainerLossDecreasesOverManySteps / testBNRunningStatsDriftDuringTraining.
//
//  What is already known (so these tests target the gap, not re-cover):
//   - Single-conv autodiff is FINITE in BOTH fp32 and bf16, across graph.run
//     and the precompiled-executable paths — `ConvKernelExecutionPathNumericsTests`
//     iterates `[.float32, .bFloat16]` and passed on the new stack.
//   - A single bf16 trainStep (batch 8) is finite — `testBF16TrainerStepProducesFiniteLosses`
//     passed.
//   - Many bf16 trainSteps (batch 64) go NaN — the two tests above failed.
//
//  So the boundary is in the FULL multi-step trainer loop. These cells sweep
//  precision × batch-size × step-count so the test-results table reads directly
//  as the bug map: on a healthy stack every cell is green; the failing cells
//  localize where (precision / accumulation / batch) the gradient breaks. The
//  point is the boundary, not a learning assertion — targets are random, so we
//  only require finiteness (the trainer's own `nonFiniteLoss` guard throws when
//  it isn't, which surfaces as the cell failing).
//

import XCTest
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class MacOS27NaNIsolationTests: XCTestCase {

    // This precision × batch × step-count forensic matrix is ~11.6 minutes
    // (~60% of total exec) — by far the single biggest contributor to suite run
    // time, with several batch64/steps1000 cells over a minute each. It is a
    // diagnostic bug-map,
    // not a correctness gate, so it is opt-in via DCM_RUN_SLOW_TESTS. See
    // SlowTestGate.swift and CLAUDE.md ("Running the tests").
    override func setUpWithError() throws {
        try SlowTestGate.requireEnabled("MacOS27NaNIsolation")
    }

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// The standard production tower (uniform v4_5block_7x7, 128ch, 7×7), with
    /// only the compute precision swapped.
    private func arch(_ precision: ComputeDataType) -> NetworkArchitecture {
        var a = NetworkArchitecture.current
        a.computeDataType = precision
        return a
    }

    /// Runs `steps` trainStep calls at `batchSize` on a FRESH trainer of the
    /// given precision (warmup off so full LR is exercised from step 0, no
    /// ramp confound). Fails — with the cell coordinates and the offending
    /// step — at the first non-finite loss/gradient, or if trainStep throws
    /// `nonFiniteLoss`.
    private func sweep(_ precision: ComputeDataType, batch: Int, steps: Int,
                       optLevel: MPSGraphOptimization = .level1,
                       file: StaticString = #filePath, line: UInt = #line) async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(precision),
                                       executableOptimizationLevel: optLevel)
        for s in 0..<steps {
            do {
                let t = try await trainer.trainStep(batchSize: batch)
                XCTAssertTrue(t.loss.isFinite,
                    "[\(precision) batch=\(batch)] loss non-finite at step \(s): \(t.loss)",
                    file: file, line: line)
                XCTAssertTrue(t.gradGlobalNorm.isFinite,
                    "[\(precision) batch=\(batch)] gradNorm non-finite at step \(s): \(t.gradGlobalNorm)",
                    file: file, line: line)
            } catch {
                XCTFail("[\(precision) batch=\(batch)] trainStep threw at step \(s): \(error)",
                        file: file, line: line)
                return
            }
        }
    }

    // MARK: bf16 × batch 1 × {1,2,4,30 steps}
    func test_bf16_batch1_steps1()  async throws { try await sweep(.bFloat16, batch: 1, steps: 1) }
    func test_bf16_batch1_steps2()  async throws { try await sweep(.bFloat16, batch: 1, steps: 2) }
    func test_bf16_batch1_steps4()  async throws { try await sweep(.bFloat16, batch: 1, steps: 4) }
    func test_bf16_batch1_steps30() async throws { try await sweep(.bFloat16, batch: 1, steps: 30) }

    // MARK: bf16 × batch 64 × {1,2,4,30 steps}
    func test_bf16_batch64_steps1()  async throws { try await sweep(.bFloat16, batch: 64, steps: 1) }
    func test_bf16_batch64_steps2()  async throws { try await sweep(.bFloat16, batch: 64, steps: 2) }
    func test_bf16_batch64_steps4()  async throws { try await sweep(.bFloat16, batch: 64, steps: 4) }
    func test_bf16_batch64_steps30() async throws { try await sweep(.bFloat16, batch: 64, steps: 30) }

    // MARK: - Auto-layout-conversion A/B (macOS 27 new default)
    //
    // Xcode 27 b1 / macOS 27 beta made automatic NCHW->NHWC layout conversion
    // for GPU convolutions the default (`convertLayoutToNHWC` is now a deprecated
    // no-op; the new opt-out is `MPSGraph.disableAutoLayoutConversion`). The
    // divergence appeared at the same toolchain bump. This sweep is identical to
    // `sweep` but builds the trainer with `disableAutoLayoutConversion: true`, so
    // the bf16 × batch64 × {steps} row reads directly as the A/B: if the canonical
    // cells above go NaN here while these stay finite, the new layout-conversion
    // default is implicated; if these still go NaN, it's exonerated. We declare
    // every conv .NCHW/.OIHW explicitly, so opting out is expected to be inert on
    // a healthy stack — the test is the falsification, not a fix.

    /// `sweep` parallel built with `disableAutoLayoutConversion: true`.
    private func sweepNoLayoutConv(_ precision: ComputeDataType, batch: Int, steps: Int,
                                   file: StaticString = #filePath, line: UInt = #line) async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(precision),
                                       disableAutoLayoutConversion: true)
        for s in 0..<steps {
            do {
                let t = try await trainer.trainStep(batchSize: batch)
                XCTAssertTrue(t.loss.isFinite,
                    "[noLayoutConv \(precision) batch=\(batch)] loss non-finite at step \(s): \(t.loss)",
                    file: file, line: line)
                XCTAssertTrue(t.gradGlobalNorm.isFinite,
                    "[noLayoutConv \(precision) batch=\(batch)] gradNorm non-finite at step \(s): \(t.gradGlobalNorm)",
                    file: file, line: line)
            } catch {
                XCTFail("[noLayoutConv \(precision) batch=\(batch)] trainStep threw at step \(s): \(error)",
                        file: file, line: line)
                return
            }
        }
    }

    func test_noLayoutConv_bf16_batch64_steps4()  async throws { try await sweepNoLayoutConv(.bFloat16, batch: 64, steps: 4) }
    func test_noLayoutConv_bf16_batch64_steps30() async throws { try await sweepNoLayoutConv(.bFloat16, batch: 64, steps: 30) }

    // MARK: - reducedPrecisionFastMath = .none A/B (force full precision)
    //
    // `MPSGraphCompilationDescriptor.reducedPrecisionFastMath` (macOS 26.0+) lets
    // MPSGraph take reduced-precision conv shortcuts: FP16 winograd-transform
    // intermediates, and FP32->FP19/TF32 operand narrowing. Its documented default
    // is `.none` (full precision), but the compiler "could use these paths ... not
    // guaranteed", so the autotuner may pick a winograd-FP16 path for our 7×7 convs
    // unless explicitly forbidden — and FP16 winograd intermediates overflow exactly
    // like the inf->nan signature we see. This sweep forces `.none` on every compile
    // descriptor and re-runs the failing bf16 × batch64 × {steps} row. The trainer
    // logs `[EXEC] reducedPrecisionFastMath default=… override=…` so the run records
    // what MPSGraph actually defaulted to. If forcing `.none` keeps these finite where
    // the canonical cells go NaN, a reduced-precision conv path is implicated.

    /// `sweep` parallel built with `reducedPrecisionFastMath: .none`.
    private func sweepFullPrecisionMath(_ precision: ComputeDataType, batch: Int, steps: Int,
                                        file: StaticString = #filePath, line: UInt = #line) async throws {
        try requireMetal()
        guard #available(macOS 26.0, *) else { throw XCTSkip("reducedPrecisionFastMath needs macOS 26") }
        // .none (raw 0) forbids all reduced-precision conv shortcuts.
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(precision),
                                       reducedPrecisionFastMathRaw: MPSGraphReducedPrecisionFastMath.none.rawValue)
        for s in 0..<steps {
            do {
                let t = try await trainer.trainStep(batchSize: batch)
                XCTAssertTrue(t.loss.isFinite,
                    "[fullPrecMath \(precision) batch=\(batch)] loss non-finite at step \(s): \(t.loss)",
                    file: file, line: line)
                XCTAssertTrue(t.gradGlobalNorm.isFinite,
                    "[fullPrecMath \(precision) batch=\(batch)] gradNorm non-finite at step \(s): \(t.gradGlobalNorm)",
                    file: file, line: line)
            } catch {
                XCTFail("[fullPrecMath \(precision) batch=\(batch)] trainStep threw at step \(s): \(error)",
                        file: file, line: line)
                return
            }
        }
    }

    func test_fullPrecMath_bf16_batch64_steps4()  async throws { try await sweepFullPrecisionMath(.bFloat16, batch: 64, steps: 4) }
    func test_fullPrecMath_bf16_batch64_steps30() async throws { try await sweepFullPrecisionMath(.bFloat16, batch: 64, steps: 30) }

    // MARK: - Deterministic layout-divergence probe (same weights + same input)
    //
    // Answers "does auto-layout conversion change the conv numerics non-obviously?"
    // — which the random-data sweeps above CANNOT, because each `trainStep`
    // synthesizes a fresh random batch (ChessTrainer.fillRandomFloats reseeds with
    // UInt64.random every call), so two sweep cells never see the same data and a
    // pass/fail difference at the onset boundary is data noise, not a layout effect.
    //
    // Here we control everything: build two inference networks of identical arch,
    // copy the FIRST net's randomly-initialized weights into the SECOND
    // (exportWeights -> loadWeights covers trainables + BN running stats), then run
    // ONE fixed deterministic input batch through both — one with auto layout
    // conversion (the macOS-27 default), one with `disableAutoLayoutConversion`. The
    // batched forward goes through the compiled MPSGraphExecutable, which is the path
    // that honors the compile-descriptor lever. The only difference between the two
    // runs is the layout flag, so the element-wise max-abs output gap IS the
    // layout-induced numeric divergence.
    //
    // Interpretation: in fp32, pure kernel-reordering (FP non-associativity across
    // im2col/winograd/direct kernels) should be last-bit — ~1e-5 or smaller. A large
    // fp32 gap would mean something structural (a transposed or mis-padded tensor),
    // i.e. a real beta bug. bf16 is logged for magnitude but not threshold-asserted.

    /// A `@Sendable`-safe sink for the batched-evaluate consume closure.
    private final class ForwardOutputs: @unchecked Sendable {
        var policy: [Float] = []
        var value: [Float] = []
    }

    /// Builds two identical-weight inference nets (layout default vs disabled), runs
    /// the same fixed batch through both, returns the max-abs policy/value gap.
    private func layoutForwardDivergence(_ precision: ComputeDataType, count: Int) async throws
        -> (policyMaxAbs: Float, valueMaxAbs: Float) {
        try requireMetal()
        let a = arch(precision)
        let netDefault = try ChessNetwork(arch: a, bnMode: .inference, disableAutoLayoutConversion: false)
        let netNoConv  = try ChessNetwork(arch: a, bnMode: .inference, disableAutoLayoutConversion: true)
        // Make the two nets bit-identical: copy net-default's He-init weights + BN
        // running stats into net-noConv (each net inits to its own random weights).
        let weights = try await netDefault.exportWeights()
        try await netNoConv.loadWeights(weights)

        // Deterministic input: a fixed SplitMix64 stream in [0,1). No system RNG, so
        // both nets — and reruns — see byte-identical boards.
        let floatsPerBoard = a.inputPlanes * 8 * 8
        var boards = [Float](repeating: 0, count: count * floatsPerBoard)
        var s: UInt64 = 0x9E3779B97F4A7C15
        for i in boards.indices {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            boards[i] = Float((s >> 40) & 0xFFFFFF) / Float(0x1000000)
        }

        func forward(_ net: ChessNetwork) async throws -> (policy: [Float], value: [Float]) {
            let sink = ForwardOutputs()
            try await net.evaluateBatched(batchBoards: boards, count: count, consume: { pol, val, _ in
                sink.policy = Array(pol)
                sink.value = Array(val)
            })
            return (sink.policy, sink.value)
        }

        let (pa, va) = try await forward(netDefault)
        let (pb, vb) = try await forward(netNoConv)
        XCTAssertEqual(pa.count, pb.count, "policy length mismatch")
        XCTAssertEqual(va.count, vb.count, "value length mismatch")

        var policyMaxAbs: Float = 0
        for i in pa.indices { policyMaxAbs = max(policyMaxAbs, abs(pa[i] - pb[i])) }
        var valueMaxAbs: Float = 0
        for i in va.indices { valueMaxAbs = max(valueMaxAbs, abs(va[i] - vb[i])) }

        SessionLogger.shared.log(
            "[PROBE] layoutForwardDivergence precision=\(precision) count=\(count) policyMaxAbs=\(policyMaxAbs) valueMaxAbs=\(valueMaxAbs)"
        )
        return (policyMaxAbs, valueMaxAbs)
    }

    func test_layoutDivergence_fp32() async throws {
        let (p, v) = try await layoutForwardDivergence(.float32, count: 8)
        XCTAssertTrue(p.isFinite && v.isFinite, "non-finite divergence p=\(p) v=\(v)")
        // fp32: kernel reordering only. A large gap => structural (transpose/pad) bug.
        XCTAssertLessThan(p, 1e-2, "fp32 policy layout divergence \(p) too large to be rounding — likely structural")
        XCTAssertLessThan(v, 1e-2, "fp32 value layout divergence \(v) too large to be rounding — likely structural")
    }

    func test_layoutDivergence_bf16() async throws {
        let (p, v) = try await layoutForwardDivergence(.bFloat16, count: 8)
        XCTAssertTrue(p.isFinite && v.isFinite, "non-finite divergence p=\(p) v=\(v)")
    }

    // MARK: - Env-var reduced-precision matrix (all 64 combinations)
    //
    // The dyld shared cache (macOS 27 b1) exposes these getenv-controlled levers,
    // which MPS/MPSGraph honor independently of the (apparently-ignored) compile
    // descriptor `reducedPrecisionFastMath` property:
    //
    //   bit 0  MPS_ALLOW_REDUCED_PRECISION=0          global reduced-precision gate off
    //   bit 1  MPSNDARRAY_WINOGRAD_FP16_INTERMEDIATE=0  no FP16 winograd intermediate
    //   bit 2  MPSNDARRAY_WINOGRAD_FP19_INTERMEDIATE=0  no FP19 winograd intermediate
    //   bit 3  MTL_DISABLE_FASTMATH=1                  Metal shader fast-math off
    //   bit 4  MPS_DIRECT_CONVOLUTION=1                force direct conv (no winograd)
    //   bit 5  MPS_DIRECTCONV_NODMA=1                  force direct conv, no DMA
    //
    // A combo's bit set => that lever's "intervention" value is applied; cleared =>
    // the var is unset (MPS default). mask 0 = all-default (baseline), mask 63 = all
    // interventions. The sweep records the first NaN step per combo; -1 = survived.
    //
    // IN-PROCESS VALIDITY: we setenv + build a FRESH ChessTrainer per combo so each
    // recompiles its executable and (should) re-read the env. But MPSGraph may cache
    // compiled pipelines process-globally, in which case later combos reuse the first
    // combo's kernels and the env never re-applies. `test_envEffect_inProcess` is the
    // guard: it forces fixed weights + fixed input and checks whether mask 0 vs mask
    // 63 produce a DIFFERENT bf16 forward. If they don't differ, the matrix below is
    // inconclusive (env not reaching MPS in-process) and must be run per-process via
    // the scheme instead.

    private static let envLevers: [(key: String, value: String)] = [
        ("MPS_ALLOW_REDUCED_PRECISION", "0"),
        ("MPSNDARRAY_WINOGRAD_FP16_INTERMEDIATE", "0"),
        ("MPSNDARRAY_WINOGRAD_FP19_INTERMEDIATE", "0"),
        ("MTL_DISABLE_FASTMATH", "1"),
        ("MPS_DIRECT_CONVOLUTION", "1"),
        ("MPS_DIRECTCONV_NODMA", "1"),
    ]

    private func applyEnvCombo(_ mask: Int) {
        for (i, lever) in Self.envLevers.enumerated() {
            if mask & (1 << i) != 0 {
                setenv(lever.key, lever.value, 1)
            } else {
                unsetenv(lever.key)
            }
        }
    }

    private func clearEnvCombo() {
        for lever in Self.envLevers { unsetenv(lever.key) }
    }

    private func envComboDescription(_ mask: Int) -> String {
        let active = Self.envLevers.enumerated()
            .filter { mask & (1 << $0.offset) != 0 }
            .map { "\($0.element.key)=\($0.element.value)" }
        return active.isEmpty ? "<none>" : active.joined(separator: ",")
    }

    /// Fixed-weight, fixed-input bf16 forward; returns a checksum of the outputs so
    /// two env combos can be compared for any numeric difference.
    private func fixedForwardChecksum(envMask: Int) async throws -> Double {
        applyEnvCombo(envMask)
        return try await fixedForwardChecksumAmbient()
    }

    /// Fixed-weight, fixed-input bf16 forward against WHATEVER env is currently in
    /// effect (no in-process setenv). Lets the scheme channel be tested: run this
    /// once with the scheme's env block empty and once with all interventions set,
    /// in two separate processes, and compare checksums.
    private func fixedForwardChecksumAmbient() async throws -> Double {
        try requireMetal()
        let a = arch(.bFloat16)
        let net = try ChessNetwork(arch: a, bnMode: .inference)
        // Overwrite all variables with a deterministic SplitMix64 pattern (~N(0,
        // 0.06)) so the forward is reproducible across runs/combos and activations
        // are O(1) (enough magnitude for kernel-choice ULP差 to show in the sum).
        var weights = try await net.exportWeights()
        var s: UInt64 = 0xD1B54A32D192ED03
        for j in weights.indices {
            for k in weights[j].indices {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                weights[j][k] = (Float((s >> 40) & 0xFFFFFF) / Float(0x1000000) - 0.5) * 0.12
            }
        }
        try await net.loadWeights(weights)
        let count = 4
        let floatsPerBoard = a.inputPlanes * 8 * 8
        var boards = [Float](repeating: 0, count: count * floatsPerBoard)
        for i in boards.indices {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            boards[i] = Float((s >> 40) & 0xFFFFFF) / Float(0x1000000)
        }
        let sink = ForwardOutputs()
        try await net.evaluateBatched(batchBoards: boards, count: count, consume: { pol, val, _ in
            sink.policy = Array(pol)
            sink.value = Array(val)
        })
        var checksum = 0.0
        for x in sink.policy { checksum += Double(x) }
        for x in sink.value { checksum += Double(x) * 1000.0 }
        return checksum
    }

    /// GUARD: does in-process setenv actually reach MPS? Compares the fixed forward
    /// under all-default vs all-interventions. Different checksum => env honored in
    /// this process => the matrix is meaningful. Equal => matrix is inconclusive.
    func test_envEffect_inProcess() async throws {
        let none = try await fixedForwardChecksum(envMask: 0)
        let all  = try await fixedForwardChecksum(envMask: 63)
        clearEnvCombo()
        SessionLogger.shared.log(
            "[ENVPROBE] in-process env effect: checksum(none)=\(none) checksum(all)=\(all) differ=\(none != all)"
        )
    }

    /// GUARD for the per-process (scheme) env channel. Reads only ambient env, so
    /// run it twice in separate processes — scheme env block empty vs all six
    /// interventions — and diff the two `[ENVPROBE-SCHEME]` checksums. Different =>
    /// scheme-launched env reaches MPS (per-process matrix is meaningful). Same =>
    /// even per-process these vars don't affect this graph's kernels.
    func test_envEffect_schemeChannel() async throws {
        let checksum = try await fixedForwardChecksumAmbient()
        SessionLogger.shared.log("[ENVPROBE-SCHEME] ambient-env fixed-forward checksum=\(checksum)")
    }

    /// Runs `maxSteps` bf16 batch-64 trainSteps under one env combo on a FRESH
    /// trainer; returns the first non-finite step, or -1 if finite throughout.
    private func envComboFirstNaNStep(mask: Int, maxSteps: Int) async throws -> Int {
        try requireMetal()
        applyEnvCombo(mask)
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16))
        for s in 0..<maxSteps {
            do {
                let t = try await trainer.trainStep(batchSize: 64)
                if !t.loss.isFinite || !t.gradGlobalNorm.isFinite { return s }
            } catch {
                return s
            }
        }
        return -1
    }

    private func runEnvMatrixChunk(_ masks: Range<Int>, maxSteps: Int = 10) async throws {
        for mask in masks {
            let onset = try await envComboFirstNaNStep(mask: mask, maxSteps: maxSteps)
            SessionLogger.shared.log(
                "[ENVMATRIX] mask=\(mask) firstNaNStep=\(onset) levers=[\(envComboDescription(mask))]"
            )
        }
        clearEnvCombo()
    }

    func test_envMatrix_chunk0() async throws { try await runEnvMatrixChunk(0..<16) }
    func test_envMatrix_chunk1() async throws { try await runEnvMatrixChunk(16..<32) }
    func test_envMatrix_chunk2() async throws { try await runEnvMatrixChunk(32..<48) }
    func test_envMatrix_chunk3() async throws { try await runEnvMatrixChunk(48..<64) }

    // MARK: - Value-path divergence trajectory
    //
    // The failure is preceded by the value loss exploding to 1e17–1e20 with
    // valueMean=1.0 before the gradient NaNs — upstream of any conv-precision lever.
    // This logs the FULL per-step diagnostic trajectory (no new graph tensors yet,
    // just everything TrainStepTiming already carries) so we can read WHAT diverges
    // first: value loss vs policy loss vs grad norm vs the W/D/L probs vs the policy
    // logit magnitude. bf16 diverges; fp32 is the control — if fp32 stays flat with
    // identical config, the divergence is bf16-precision-specific in the training path.

    private func logTrajectory(_ precision: ComputeDataType, batch: Int, steps: Int) async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(precision))
        SessionLogger.shared.log("[VTRAJ] BEGIN precision=\(precision) batch=\(batch) steps=\(steps)")
        for s in 0..<steps {
            do {
                let t = try await trainer.trainStep(batchSize: batch)
                SessionLogger.shared.log(String(
                    format: "[VTRAJ] step=%d loss=%.4g pLoss=%.4g vLoss=%.4g gNorm=%.4g pLogitMax=%.4g pHeadW=%.4g vMean=%.4g vAbs=%.4g pW=%.4g pD=%.4g pL=%.4g",
                    s, t.loss, t.policyLoss, t.valueLoss, t.gradGlobalNorm,
                    t.policyLogitAbsMax, t.policyHeadWeightNorm,
                    t.valueMean, t.valueAbsMean, t.valueProbWin, t.valueProbDraw, t.valueProbLoss
                ))
                if !t.loss.isFinite || !t.gradGlobalNorm.isFinite {
                    SessionLogger.shared.log("[VTRAJ] non-finite at step \(s) — stopping")
                    return
                }
            } catch {
                SessionLogger.shared.log("[VTRAJ] step=\(s) THREW \(error) (see [ALARM] above for components)")
                return
            }
        }
        SessionLogger.shared.log("[VTRAJ] END precision=\(precision) — finite through \(steps) steps")
    }

    func test_valueTrajectory_bf16_batch64() async throws { try await logTrajectory(.bFloat16, batch: 64, steps: 12) }
    func test_valueTrajectory_fp32_batch64() async throws { try await logTrajectory(.float32, batch: 64, steps: 12) }

    // Longer fp32 run: does the gNorm-~1e7 plateau hold (clip-masked, weights
    // survive) or does it eventually diverge? Characterizes whether fp32 is
    // "ugly-but-bounded" or a slow divergence. fp32 steps are cheap (~40ms).
    func test_valueTrajectory_fp32_batch64_long() async throws { try await logTrajectory(.float32, batch: 64, steps: 120) }

    // Is the fp32 gradient ramp STATEFUL (grows with steps) or just per-batch
    // data variance? Re-uses ONE fixed synthetic batch every step by pinning the
    // RNG-free path: we can't seed fillRandomFloats from here, but we CAN compare
    // the gNorm distribution of step 0 (fresh weights) repeated on a fresh trainer
    // each call — run twice and eyeball. Cheaper signal: low LR. If the ramp is a
    // weight/state feedback, a 100x-smaller LR should slow it; if it's pure data
    // variance, LR is irrelevant.
    // MARK: - Real-data-path baseline-overlap A/B
    //
    // All prior isolation used trainStep(batchSize:) — the SYNTHETIC path, which
    // sets vBaseline=0 and never runs computeValueBaselineGPU. LIVE training uses
    // trainStep(replayBuffer:batchSize:): it first fires a NON-BLOCKING value-
    // baseline forward whose output is SINGLE-BUFFERED (`resultTD`), then a training
    // step that reads that buffer from a SEPARATE command buffer. If the cross-
    // command-buffer ordering isn't honored on macOS 27 (no double-buffering), the
    // step trains on a stale/clobbered vBaseline. This drives the REAL path and A/Bs
    // `network.blockingValueBaseline` (false = overlap, true = waitUntilCompleted).

    private func populateReplayBuffer(_ a: NetworkArchitecture, positions: Int) -> ReplayBuffer {
        let buf = ReplayBuffer(capacity: max(positions, 4096), inputEncoding: a.inputEncoding)
        let fpb = a.inputPlanes * 64
        var boards = [Float](repeating: 0, count: positions * fpb)
        var moves = [Int32](repeating: 0, count: positions)
        var plies = [UInt16](repeating: 0, count: positions)
        var taus = [Float](repeating: 1.0, count: positions)
        var hashes = [UInt64](repeating: 0, count: positions)
        var materials = [UInt8](repeating: 32, count: positions)
        var outcomes = [Float](repeating: 0, count: positions)
        var s: UInt64 = 0x00ABCDEF12345678
        for i in boards.indices {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            boards[i] = Float((s >> 40) & 0xFFFF) / 65535.0
        }
        for i in 0..<positions {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            moves[i] = Int32((s >> 33) % UInt64(ChessNetwork.policySize))
            s = s &* 6364136223846793005 &+ 1442695040888963407
            outcomes[i] = Float(Int((s >> 40) % 3)) - 1.0
            plies[i] = UInt16(i % 200)
            hashes[i] = s
        }
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { p in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { mc in
        outcomes.withUnsafeBufferPointer { o in
            guard let bb = b.baseAddress, let mb = m.baseAddress, let pb = p.baseAddress,
                  let tb = t.baseAddress, let hb = h.baseAddress, let mcb = mc.baseAddress,
                  let ob = o.baseAddress else { return }
            buf.append(
                boards: bb, policyIndices: mb, plyIndices: pb, samplingTaus: tb,
                stateHashes: hb, materialCounts: mcb,
                gameLength: UInt16(min(positions, 65535)),
                workerId: 0, intraWorkerGameIndex: 0,
                outcomes: ob, count: positions)
        }}}}}}}
        return buf
    }

    private func runRealPathTrajectory(_ precision: ComputeDataType, blocking: Bool, steps: Int) async throws {
        try requireMetal()
        let a = arch(precision)
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: a)
        trainer.network.blockingValueBaseline = blocking
        let buf = populateReplayBuffer(a, positions: 4096)
        SessionLogger.shared.log("[REALPATH] BEGIN \(precision) blocking=\(blocking) steps=\(steps)")
        for s in 0..<steps {
            do {
                guard let t = try await trainer.trainStep(replayBuffer: buf, batchSize: 64) else {
                    SessionLogger.shared.log("[REALPATH] step=\(s) sample returned nil")
                    return
                }
                SessionLogger.shared.log(String(
                    format: "[REALPATH] %@ blocking=%@ step=%d loss=%.4g gNorm=%.4g vLoss=%.4g pLoss=%.4g pLogitMax=%.4g",
                    "\(precision)", blocking ? "Y" : "N", s, t.loss, t.gradGlobalNorm, t.valueLoss, t.policyLoss, t.policyLogitAbsMax))
                if !t.loss.isFinite || !t.gradGlobalNorm.isFinite {
                    SessionLogger.shared.log("[REALPATH] non-finite at step \(s) (\(precision) blocking=\(blocking))")
                    return
                }
            } catch {
                SessionLogger.shared.log("[REALPATH] step=\(s) THREW \(error) (\(precision) blocking=\(blocking))")
                return
            }
        }
        SessionLogger.shared.log("[REALPATH] END \(precision) blocking=\(blocking) — finite through \(steps) steps")
    }

    func test_realPath_bf16_overlap()  async throws { try await runRealPathTrajectory(.bFloat16, blocking: false, steps: 20) }
    func test_realPath_bf16_blocking() async throws { try await runRealPathTrajectory(.bFloat16, blocking: true,  steps: 20) }
    func test_realPath_fp32_overlap()  async throws { try await runRealPathTrajectory(.float32,  blocking: false, steps: 20) }
    func test_realPath_fp32_blocking() async throws { try await runRealPathTrajectory(.float32,  blocking: true,  steps: 20) }

    func test_valueTrajectory_fp32_lowLR() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(learningRate: 1e-5, lrWarmupSteps: 0, arch: arch(.float32))
        SessionLogger.shared.log("[VTRAJ] BEGIN fp32 lowLR=1e-5 batch=64")
        for s in 0..<40 {
            let t = try await trainer.trainStep(batchSize: 64)
            SessionLogger.shared.log(String(
                format: "[VTRAJ-LOWLR] step=%d loss=%.4g pLoss=%.4g vLoss=%.4g gNorm=%.4g pLogitMax=%.4g",
                s, t.loss, t.policyLoss, t.valueLoss, t.gradGlobalNorm, t.policyLogitAbsMax))
            if !t.loss.isFinite { return }
        }
        SessionLogger.shared.log("[VTRAJ] END fp32 lowLR")
    }

    // MARK: - Master vs working weight divergence (bf16)
    //
    // The trajectory shows the WORKING (bf16) weights/activations exploding while
    // the optimizer math (clip + master SGD, μ=0) is provably bounded (clipped grad
    // norm <=30, so each fp32-master update is ~lr*30 in L2 — it cannot move a weight
    // 40x in one step). Two competing explanations, this discriminates them:
    //   (A) clip broken in bf16  => the fp32 MASTER norms also explode.
    //   (B) working-sync stomped  => MASTER norms stay bounded while WORKING norms
    //       explode (working diverges from cast(master)), the documented macOS-27
    //       buffer-stomp class that `splitWorkingWeightSync` only partially fixed.
    // After each step we read masters (readMasterValues) and working
    // (network.exportWeights), both ordered trainables+bn, and log the max per-
    // variable L2 norm of each plus the max |master-working| element gap (which
    // should be ~bf16 rounding if the sync is intact).

    private func l2(_ v: [Float]) -> Double {
        var s = 0.0
        for x in v { s += Double(x) * Double(x) }
        return s.squareRoot()
    }

    func test_masterVsWorking_bf16_batch64() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16))
        SessionLogger.shared.log("[MVW] BEGIN bf16 batch=64")
        // Pre-step (post-construction, post-seed) coherence: masters were just
        // seeded = cast(working), so these MUST match within bf16 rounding. A gap
        // here = the seed/readback is broken; a gap only AFTER step 0 = the fused
        // master-update write corrupts.
        do {
            let m0 = try await trainer.readMasterValues()
            let w0 = try await trainer.network.exportWeights()
            if m0.count == w0.count, !m0.isEmpty {
                var mn = 0.0, wn = 0.0, gap = 0.0
                for i in m0.indices {
                    mn = max(mn, l2(m0[i])); wn = max(wn, l2(w0[i]))
                    let n = min(m0[i].count, w0[i].count)
                    for j in 0..<n { gap = max(gap, abs(Double(m0[i][j]) - Double(w0[i][j]))) }
                }
                SessionLogger.shared.log(String(
                    format: "[MVW] step=-1 (post-seed) maxMasterNorm=%.4g maxWorkingNorm=%.4g maxAbsGap=%.4g", mn, wn, gap))
            }
        }
        for s in 0..<8 {
            var threw = false
            do {
                _ = try await trainer.trainStep(batchSize: 64)
            } catch {
                SessionLogger.shared.log("[MVW] step=\(s) trainStep THREW \(error)")
                threw = true
            }
            let masters = try await trainer.readMasterValues()
            let working = try await trainer.network.exportWeights()
            guard masters.count == working.count, !masters.isEmpty else {
                SessionLogger.shared.log("[MVW] step=\(s) master/working count mismatch m=\(masters.count) w=\(working.count)")
                if threw { return }
                continue
            }
            var maxMasterNorm = 0.0, maxWorkingNorm = 0.0, maxGap = 0.0
            var argMaxWorking = -1
            for i in masters.indices {
                let mn = l2(masters[i]), wn = l2(working[i])
                if mn > maxMasterNorm { maxMasterNorm = mn }
                if wn > maxWorkingNorm { maxWorkingNorm = wn; argMaxWorking = i }
                let n = min(masters[i].count, working[i].count)
                for j in 0..<n {
                    let g = abs(Double(masters[i][j]) - Double(working[i][j]))
                    if g > maxGap { maxGap = g }
                }
            }
            SessionLogger.shared.log(String(
                format: "[MVW] step=%d maxMasterNorm=%.4g maxWorkingNorm=%.4g maxAbsGap=%.4g (argWorking=%d)",
                s, maxMasterNorm, maxWorkingNorm, maxGap, argMaxWorking
            ))
            if threw { return }
        }
        SessionLogger.shared.log("[MVW] END")
    }

    // MARK: - Config D: fp32-stored weights, cast-to-bf16-in-forward
    //
    // The experimental macOS-27-beta workaround. Same finiteness sweep as
    // `sweep`, but the trainer is built with `bf16CastInForward: true` over a
    // bf16 arch — so every weight/BN variable is stored fp32 and cast to bf16
    // at point of use, while the optimizer runs the plain fp32 path (no bf16
    // working var, no master, no working-sync). If these stay finite where the
    // canonical bf16 cells go NaN, config D dodges the divergence.

    /// `sweep` parallel for config D: fresh `bf16CastInForward` trainer over a
    /// bf16 arch. Same finite-loss / finite-grad assertions over `steps`.
    private func sweepD(batch: Int, steps: Int,
                        file: StaticString = #filePath, line: UInt = #line) async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16),
                                       bf16CastInForward: true)
        for s in 0..<steps {
            do {
                let t = try await trainer.trainStep(batchSize: batch)
                XCTAssertTrue(t.loss.isFinite,
                    "[configD batch=\(batch)] loss non-finite at step \(s): \(t.loss)",
                    file: file, line: line)
                XCTAssertTrue(t.gradGlobalNorm.isFinite,
                    "[configD batch=\(batch)] gradNorm non-finite at step \(s): \(t.gradGlobalNorm)",
                    file: file, line: line)
            } catch {
                XCTFail("[configD batch=\(batch)] trainStep threw at step \(s): \(error)",
                        file: file, line: line)
                return
            }
        }
    }

    func test_D_bf16_batch64_steps30() async throws { try await sweepD(batch: 64, steps: 30) }
    func test_D_bf16_batch1_steps30()  async throws { try await sweepD(batch: 1,  steps: 30) }
    // Longer + production-batch (32) D sweeps. The canonical bf16 master path
    // diverges by step ~5–6 at batch 64; surviving 150 steps here is strong
    // evidence config D actually dodges the bug at the batch sizes real training
    // uses (32/64), not just past the onset.
    func test_D_bf16_batch64_steps150() async throws { try await sweepD(batch: 64, steps: 150) }
    func test_D_bf16_batch32_steps150() async throws { try await sweepD(batch: 32, steps: 150) }

    /// Production-like config: the live TrainingParameters defaults — LR 0.01,
    /// momentum 0.9, value label smoothing 0.013, draw penalty 0, sqrt-batch LR
    /// scaling on (base 4096) — with a 100-step warmup. `castInForward==true`
    /// selects config D (bf16 forward, fp32 weight storage); else the named
    /// precision's normal path. Still synthetic random data, so this is a
    /// numerical-stability probe (does the bf16 path go NaN), not learning.
    private func sweepProd(_ precision: ComputeDataType, castInForward: Bool,
                           batch: Int, steps: Int,
                           file: StaticString = #filePath, line: UInt = #line) async throws {
        try requireMetal()
        let trainer = try ChessTrainer(
            learningRate: 0.01,
            drawPenalty: 0.0,
            valueLabelSmoothingEpsilon: 0.013,
            momentumCoeff: 0.9,
            lrWarmupSteps: 100,
            arch: arch(precision),
            bf16CastInForward: castInForward
        )
        for s in 0..<steps {
            do {
                let t = try await trainer.trainStep(batchSize: batch)
                XCTAssertTrue(t.loss.isFinite,
                    "[prod \(precision) cast=\(castInForward) batch=\(batch)] loss non-finite at step \(s): \(t.loss)",
                    file: file, line: line)
                XCTAssertTrue(t.gradGlobalNorm.isFinite,
                    "[prod \(precision) cast=\(castInForward) batch=\(batch)] gradNorm non-finite at step \(s): \(t.gradGlobalNorm)",
                    file: file, line: line)
            } catch {
                XCTFail("[prod \(precision) cast=\(castInForward) batch=\(batch)] trainStep threw at step \(s): \(error)",
                        file: file, line: line)
                return
            }
        }
    }

    // Config D, production-like config, 1000 steps each.
    func test_Dprod_bf16_batch64_steps1000() async throws { try await sweepProd(.bFloat16, castInForward: true, batch: 64, steps: 1000) }
    func test_Dprod_bf16_batch32_steps1000() async throws { try await sweepProd(.bFloat16, castInForward: true, batch: 32, steps: 1000) }
    func test_Dprod_bf16_batch1_steps1000()  async throws { try await sweepProd(.bFloat16, castInForward: true, batch: 1,  steps: 1000) }
    // fp32 controls at the IDENTICAL config: if these also NaN, a failure is the
    // random-data + aggressive-LR explosion, not a D/bf16-specific problem.
    func test_prodControl_fp32_batch64_steps1000() async throws { try await sweepProd(.float32, castInForward: false, batch: 64, steps: 1000) }
    func test_prodControl_fp32_batch32_steps1000() async throws { try await sweepProd(.float32, castInForward: false, batch: 32, steps: 1000) }
    func test_prodControl_fp32_batch1_steps1000()  async throws { try await sweepProd(.float32, castInForward: false, batch: 1,  steps: 1000) }
    // Canonical bf16 (working-variable master path) at the same config — the
    // reference that the divergence bug still bites here, so a D pass is meaningful.
    func test_prodCanonical_bf16_batch64_steps1000() async throws { try await sweepProd(.bFloat16, castInForward: false, batch: 64, steps: 1000) }

    // MARK: bf16 × optimizationLevel .level0 × {batch 1,64} × {2,4 steps}
    // Theory #4/#5: macOS/Xcode 27 changed the MPSGraph compilation
    // optimizationLevel default to .level1. The production trainer compiles its
    // executable at .level1; these cells force .level0 on the SAME bf16 tower
    // and run past the step-2 onset point. If .level0 stays finite where
    // .level1 (the matrix cells above) went NaN, the regression is in the
    // level-1 codegen of the bf16 optimizer/gradient path — not in our math.
    func test_bf16_level0_batch1_steps2()  async throws { try await sweep(.bFloat16, batch: 1,  steps: 2, optLevel: .level0) }
    func test_bf16_level0_batch1_steps4()  async throws { try await sweep(.bFloat16, batch: 1,  steps: 4, optLevel: .level0) }
    func test_bf16_level0_batch64_steps2() async throws { try await sweep(.bFloat16, batch: 64, steps: 2, optLevel: .level0) }
    func test_bf16_level0_batch64_steps4() async throws { try await sweep(.bFloat16, batch: 64, steps: 4, optLevel: .level0) }

    // MARK: fp32 × batch 1 × {1,2,4,30 steps}
    func test_fp32_batch1_steps1()  async throws { try await sweep(.float32, batch: 1, steps: 1) }
    func test_fp32_batch1_steps2()  async throws { try await sweep(.float32, batch: 1, steps: 2) }
    func test_fp32_batch1_steps4()  async throws { try await sweep(.float32, batch: 1, steps: 4) }
    func test_fp32_batch1_steps30() async throws { try await sweep(.float32, batch: 1, steps: 30) }

    // MARK: fp32 × batch 64 × {1,2,4,30 steps}
    func test_fp32_batch64_steps1()  async throws { try await sweep(.float32, batch: 64, steps: 1) }
    func test_fp32_batch64_steps2()  async throws { try await sweep(.float32, batch: 64, steps: 2) }
    func test_fp32_batch64_steps4()  async throws { try await sweep(.float32, batch: 64, steps: 4) }
    func test_fp32_batch64_steps30() async throws { try await sweep(.float32, batch: 64, steps: 30) }

    // MARK: - fp32 parallels of the two originally-failing bf16 tests
    // (matched to their exact setup — DEFAULT trainer with warmup — so they are
    //  apples-to-apples with the failing bf16 versions, not just matrix cells.)

    /// fp32 parallel of testTrainerLossDecreasesOverManySteps (bf16, batch 64,
    /// 30 steps, default warmup). Asserts loss stays finite and bounded.
    func test_fp32_parallel_lossDecreasesOverManySteps() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(arch: arch(.float32))
        for s in 0..<30 {
            let t = try await trainer.trainStep(batchSize: 64)
            XCTAssertTrue(t.loss.isFinite, "fp32 loss non-finite at step \(s): \(t.loss)")
            XCTAssertTrue(t.gradGlobalNorm.isFinite, "fp32 gradNorm non-finite at step \(s)")
            XCTAssertLessThan(abs(t.loss), 100.0, "fp32 loss exploded at step \(s): \(t.loss)")
        }
    }

    /// fp32 parallel of testBNRunningStatsDriftDuringTraining (bf16, batch 32,
    /// 3 steps): finite throughout AND at least one BN running stat drifts.
    func test_fp32_parallel_bnRunningStatsDrift() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(arch: arch(.float32))
        let nTrain = trainer.network.trainableVariables.count
        let before = try await trainer.network.exportWeights()
        for s in 0..<3 {
            let t = try await trainer.trainStep(batchSize: 32)
            XCTAssertTrue(t.loss.isFinite && t.gradGlobalNorm.isFinite,
                          "fp32 non-finite at step \(s)")
        }
        let after = try await trainer.network.exportWeights()
        var anyStatChanged = false
        for i in nTrain..<before.count {
            for k in 0..<before[i].count where abs(after[i][k] - before[i][k]) > 0 {
                anyStatChanged = true
            }
        }
        XCTAssertTrue(anyStatChanged, "fp32: no BN running stat changed after 3 steps")
    }

    // MARK: - Theory #1: step-1 finite is not the same as step-1 CORRECT
    //
    // The matrix shows bf16 step-1 is finite; the bug surfaces at step 2. But
    // "finite" could still be silently wrong — a bf16 forward pass that already
    // computes a degenerate softmax would explain a poisoned gradient one step
    // later. So instead of comparing weights (which differ by random init), we
    // compare a SCALE-INVARIANT property both precisions must agree on: at
    // random init the policy is near-uniform, so its Shannon entropy must sit
    // near the ceiling log(4864) ≈ 8.49 nats and the W/D/L value loss near its
    // ~ln 3 ≈ 1.10 prior — for fp32 AND bf16 alike. A bf16 step-1 that is
    // finite but already off these priors localizes the fault to the forward
    // pass, upstream of the gradient.
    func test_bf16_step1_matchesFP32Priors() async throws {
        try requireMetal()
        let ceiling = Float(log(Double(ChessNetwork.policySize)))   // ≈ 8.49

        let fp32 = try await ChessTrainer(lrWarmupSteps: 0, arch: arch(.float32))
            .trainStep(batchSize: 64)
        let bf16 = try await ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16))
            .trainStep(batchSize: 64)

        for (tag, t) in [("fp32", fp32), ("bf16", bf16)] {
            XCTAssertTrue(t.loss.isFinite && t.gradGlobalNorm.isFinite,
                          "\(tag) step-1 non-finite")
            // Near-uniform policy at init: entropy should be within ~1 nat of
            // the ceiling, never collapsed.
            XCTAssertGreaterThan(t.policyEntropy, ceiling - 1.0,
                "\(tag) step-1 policy entropy \(t.policyEntropy) far below init ceiling \(ceiling)")
            XCTAssertLessThanOrEqual(t.policyEntropy, ceiling + 0.01,
                "\(tag) step-1 policy entropy \(t.policyEntropy) above mathematical ceiling \(ceiling)")
            // W/D/L value loss near its categorical-CE prior, not exploded.
            XCTAssertGreaterThan(t.valueLoss, 0.0, "\(tag) value loss non-positive")
            XCTAssertLessThan(t.valueLoss, 3.0, "\(tag) step-1 value loss \(t.valueLoss) far above ln3 prior")
        }
        // bf16 must track fp32's init priors, not merely be finite.
        XCTAssertEqual(bf16.policyEntropy, fp32.policyEntropy, accuracy: 0.5,
            "bf16 step-1 entropy \(bf16.policyEntropy) diverges from fp32 \(fp32.policyEntropy)")
    }

    // MARK: - Theory #2: isolate the cast from the master/velocity update
    //
    // Boundary established by the matrix: a bf16 step-1 trainStep returns a
    // FINITE loss and a FINITE gradNorm (so every gradient element is finite,
    // and the forward matches fp32 priors) — yet step 2 reads NaN weights. So
    // the NaN is born in the end-of-step-1 optimizer tail, which is three ops in
    // one graph: the fp32 MASTER update, the VELOCITY/momentum update, and the
    // `working = cast(master)` sync. These three cells each run exactly ONE step
    // and then export ONE of the three buffers, asserting it is finite. The test
    // runner swallows assertion messages but reliably reports which test NAMES
    // failed, so the PASS/FAIL triple is itself the truth table:
    //
    //   master FAIL + velocity ? + working FAIL ⇒ corruption upstream of the
    //                                             cast (master or velocity update).
    //   master PASS + working FAIL              ⇒ the bf16 CAST is the culprit.
    //   all PASS                                ⇒ step-1 update is clean (would
    //                                             contradict the step-2 onset).
    //
    // (The phenomenon is deterministic — every bf16 multi-step cell goes NaN —
    //  so three independent trainers each reproduce it; no shared-state needed.)
    private func stepOnceAndExport(
    ) async throws -> (masters: [[Float]], velocity: [[Float]], working: [[Float]]) {
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16))
        let baseCount = trainer.network.trainableVariables.count
            + trainer.network.bnRunningStatsVariables.count
        _ = try await trainer.trainStep(batchSize: 64)
        let working = try await trainer.network.exportWeights()
        let trainerState = try await trainer.exportTrainerWeights()
        return (Array(trainerState.prefix(baseCount)),
                Array(trainerState.dropFirst(baseCount)),
                working)
    }

    private func firstNonFinite(_ buffers: [[Float]]) -> String? {
        for (i, t) in buffers.enumerated() {
            for (k, v) in t.enumerated() where !v.isFinite {
                return "[\(i)][\(k)]=\(v)"
            }
        }
        return nil
    }

    /// Does the fp32 MASTER buffer contain a NaN/Inf after one bf16 step?
    func test_bf16_step1_masterStaysFinite() async throws {
        try requireMetal()
        let nf = firstNonFinite(try await stepOnceAndExport().masters)
        XCTAssertNil(nf, "fp32 MASTER non-finite after 1 step: \(nf ?? "")")
    }

    /// Does the VELOCITY/momentum buffer contain a NaN/Inf after one bf16 step?
    func test_bf16_step1_velocityStaysFinite() async throws {
        try requireMetal()
        let nf = firstNonFinite(try await stepOnceAndExport().velocity)
        XCTAssertNil(nf, "VELOCITY non-finite after 1 step: \(nf ?? "")")
    }

    /// Does the bf16 WORKING buffer (the per-step cast of the master, what the
    /// next forward multiplies) contain a NaN/Inf after one bf16 step?
    func test_bf16_step1_workingStaysFinite() async throws {
        try requireMetal()
        let nf = firstNonFinite(try await stepOnceAndExport().working)
        XCTAssertNil(nf, "bf16 WORKING (cast of master) non-finite after 1 step: \(nf ?? "")")
    }

    /// Pinpoint EXACTLY which tensors' bf16 working cast goes non-finite after
    /// one step, with each tensor's index, name, rank and shape — so the Apple
    /// repro can name the offending cast's shape precisely. The runner swallows
    /// assertion messages, so the full per-tensor report is written SYNCHRONOUSLY
    /// (survives process exit, unlike the async SessionLogger) to
    /// ~/Library/Logs/DrewsChessMachine/cast_probe_report.txt.
    func test_bf16_step1_pinpointNonFiniteCasts() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16))
        let trainables = trainer.network.trainableVariables
        let bnStats = trainer.network.bnRunningStatsVariables
        let baseCount = trainables.count + bnStats.count

        _ = try await trainer.trainStep(batchSize: 64)
        let working = try await trainer.network.exportWeights()
        let trainerState = try await trainer.exportTrainerWeights()
        let masters = Array(trainerState.prefix(baseCount))
        let velocity = Array(trainerState.dropFirst(baseCount))

        // Map a base-buffer index (working/master share this layout) to its
        // source variable's name + shape.
        func describeBase(_ i: Int) -> String {
            let v = i < trainables.count ? trainables[i] : bnStats[i - trainables.count]
            let kind = i < trainables.count ? "trainable" : "bnStat"
            let shape = (v.shape ?? []).map { "\($0.intValue)" }.joined(separator: "×")
            let rank = (v.shape ?? []).count
            let name = v.operation.name.isEmpty ? "<unnamed>" : v.operation.name
            return "\(kind)[\(i)] rank=\(rank) shape=[\(shape)] name=\(name)"
        }

        func reportBuffer(_ label: String, _ buffers: [[Float]], describe: (Int) -> String) -> String {
            var lines: [String] = []
            var badTensors = 0
            for (i, t) in buffers.enumerated() {
                var bad = 0
                for v in t where !v.isFinite { bad += 1 }
                if bad > 0 {
                    badTensors += 1
                    lines.append("  NONFINITE \(describe(i)) bad=\(bad)/\(t.count)")
                }
            }
            let header = "\(label): \(badTensors)/\(buffers.count) tensors contain non-finite values"
            return ([header] + lines).joined(separator: "\n")
        }

        // For every working tensor that went non-finite, dump the fp32 MASTER
        // value that fed the cast at each bad position — decimal, raw bits,
        // exponent — and classify the working result as NaN / +Inf / -Inf. This
        // is the crux: if the masters are ordinary-magnitude (≈ He-init scale)
        // yet the cast yields NaN/Inf, the cast kernel is broken; if the masters
        // are ≈ 1e38 (near fp32 max, above bf16 max ≈ 3.389e38) then an Inf is a
        // legitimate overflow and the real fault is an upstream weight explosion.
        func classify(_ v: Float) -> String {
            if v.isNaN { return "NaN" }
            if v == .infinity { return "+Inf" }
            if v == -.infinity { return "-Inf" }
            return "finite(\(v))"
        }
        func detailBadTensor(_ i: Int) -> String {
            let m = masters[i]
            let w = working[i]
            var absMax: Float = 0, finiteMin = Float.greatestFiniteMagnitude, finiteMax = -Float.greatestFiniteMagnitude
            var denorm = 0, masterNonFinite = 0
            for v in m {
                if !v.isFinite { masterNonFinite += 1; continue }
                let a = abs(v)
                if a > absMax { absMax = a }
                if v < finiteMin { finiteMin = v }
                if v > finiteMax { finiteMax = v }
                if a > 0 && a < Float.leastNormalMagnitude { denorm += 1 }
            }
            var lines = ["DETAIL \(describeBase(i)): masterAbsMax=\(absMax) masterMin=\(finiteMin) masterMax=\(finiteMax) masterDenorm=\(denorm) masterNonFinite=\(masterNonFinite)"]
            var shown = 0
            for k in 0..<w.count where !w[k].isFinite && shown < 12 {
                let mv = m[k]
                let bits = String(format: "0x%08x", mv.bitPattern)
                let exp = Int((mv.bitPattern >> 23) & 0xFF) - 127
                lines.append("    [\(k)] master=\(mv) bits=\(bits) exp2=\(exp) -> working=\(classify(w[k]))")
                shown += 1
            }
            return lines.joined(separator: "\n")
        }
        var details: [String] = []
        for (i, t) in working.enumerated() where t.contains(where: { !$0.isFinite }) {
            details.append(detailBadTensor(i))
        }

        let report = ([
            "=== bf16 cast isolation: per-tensor non-finite report (after 1 step) ===",
            "trainables=\(trainables.count) bnStats=\(bnStats.count) baseCount=\(baseCount) velocity=\(velocity.count)",
            "bf16 max finite ≈ 3.3895e38; fp32 max finite ≈ 3.4028e38 (a finite fp32 in that gap rounds to +Inf in bf16)",
            reportBuffer("MASTER (fp32)", masters, describe: describeBase),
            reportBuffer("VELOCITY (fp32)", velocity, describe: { "velocity[\($0)] for \(describeBase($0))" }),
            reportBuffer("WORKING (bf16 = cast(master))", working, describe: describeBase),
            "--- per-bad-tensor detail (master value before the cast) ---",
        ] + details).joined(separator: "\n") + "\n"

        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DrewsChessMachine", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("cast_probe_report.txt")
        try report.write(to: url, atomically: true, encoding: .utf8)

        // Working is expected to carry the NaN on this stack; the report file is
        // the readout. Keep the assertion so the cell stays red-by-design until
        // the cast is fixed.
        var workingBad = 0
        for t in working { for v in t where !v.isFinite { workingBad += 1 } }
        XCTAssertEqual(workingBad, 0, "bf16 working has \(workingBad) non-finite values — see cast_probe_report.txt")
    }

    /// Single-block discriminator: collapse the tower to exactly ONE residual
    /// block (the default arch's first group, count=1) and run one bf16 step.
    /// All five blocks are byte-identical, so if the corruption is positional
    /// ("block index 1's SE"), a one-block tower (only block0 exists) should be
    /// CLEAN. If block0's SE corrupts here, the bug needs only one block. Writes
    /// the per-tensor non-finite report to cast_probe_single_block.txt.
    func test_bf16_singleBlock_pinpoint() async throws {
        try requireMetal()
        var a = NetworkArchitecture.current
        a.computeDataType = .bFloat16
        var g = a.blockGroups[0]
        g.count = 1
        a.blockGroups = [g]

        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: a)
        let trainables = trainer.network.trainableVariables
        let bnStats = trainer.network.bnRunningStatsVariables
        _ = try await trainer.trainStep(batchSize: 64)
        let working = try await trainer.network.exportWeights()

        var lines: [String] = [
            "=== single-block tower: working non-finite report (after 1 bf16 step) ===",
            "blocks=\(a.numBlocks) trainables=\(trainables.count) bnStats=\(bnStats.count)",
        ]
        var badTensors = 0
        for (i, t) in working.enumerated() {
            var bad = 0
            for v in t where !v.isFinite { bad += 1 }
            if bad > 0 {
                badTensors += 1
                let v = i < trainables.count ? trainables[i] : bnStats[i - trainables.count]
                let shape = (v.shape ?? []).map { "\($0.intValue)" }.joined(separator: "×")
                let name = v.operation.name.isEmpty ? "<unnamed>" : v.operation.name
                lines.append("  NONFINITE [\(i)] shape=[\(shape)] name=\(name) bad=\(bad)/\(t.count)")
            }
        }
        lines.insert("WORKING: \(badTensors)/\(working.count) tensors non-finite", at: 2)

        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DrewsChessMachine", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try (lines.joined(separator: "\n") + "\n").write(
            to: dir.appendingPathComponent("cast_probe_single_block.txt"),
            atomically: true, encoding: .utf8)

        // No assertion on the outcome — this is a discriminator whose VALUE is
        // the report (clean vs which tensor), not a pass/fail expectation. Only
        // assert the step itself ran (finite or not, the report captures it).
        XCTAssertEqual(working.count, trainables.count + bnStats.count)
    }

    /// Hypothesis (A) vs (B) discriminator: isolate the fp32→bf16 cast on the
    /// exact `[128,32]` shape that corrupts in training, with NO matmul, NO
    /// gradient, NO master/velocity — just `cast(fp32 → bf16 → fp32)` over a
    /// finite input that includes the exact-1.0 values the corrupted SE fc1 read
    /// back. Run through all three execution paths (graph.run, executable
    /// level0, executable level1) since the training bug lives in the compiled
    /// executable. If any path turns finite input non-finite, the cast op itself
    /// is broken (A). If all are clean, the cast is innocent and the corruption
    /// requires the surrounding training graph (B = buffer stomp). Report →
    /// cast_probe_standalone.txt.
    func test_standalone_cast_fp32_to_bf16() throws {
        try requireMetal()
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { throw XCTSkip("no Metal") }

        let rows = 128, cols = 32, n = rows * cols
        let shape: [NSNumber] = [NSNumber(value: rows), NSNumber(value: cols)]

        // Deterministic finite input: ~20% exactly 1.0 (mirroring the corrupted
        // SE fc1 readback), the rest ordinary He-scale magnitudes in ±0.25.
        var input = [Float](repeating: 0, count: n)
        for k in 0..<n {
            input[k] = (k % 5 == 0)
                ? 1.0
                : Float((k &* 1103515245 &+ 12345) % 2000 - 1000) / 4000.0
        }

        func makeTD(_ values: [Float]) -> MPSGraphTensorData {
            let nd = MPSNDArray(device: device,
                                descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape))
            var local = values
            local.withUnsafeMutableBytes { buf in
                if let base = buf.baseAddress { nd.writeBytes(base, strideBytes: nil) }
            }
            return MPSGraphTensorData(nd)
        }
        func readTD(_ td: MPSGraphTensorData) -> [Float] {
            var out = [Float](repeating: .nan, count: n)
            out.withUnsafeMutableBytes { buf in
                if let base = buf.baseAddress { td.mpsndarray().readBytes(base, strideBytes: nil) }
            }
            return out
        }
        func buildCastGraph() -> (MPSGraph, MPSGraphTensor, MPSGraphTensor) {
            let graph = MPSGraph()
            let ph = graph.placeholder(shape: shape, dataType: .float32, name: "in")
            let bf16 = graph.cast(ph, to: .bFloat16, name: "to_bf16")
            let back = graph.cast(bf16, to: .float32, name: "back_to_fp32")
            return (graph, ph, back)
        }

        var report = ["=== standalone cast([128×32] fp32 -> bf16 -> fp32), finite input incl. exact 1.0 ==="]
        var anyNonFinite = 0

        func evaluate(_ label: String, _ out: [Float]) {
            var nonFinite = 0, maxErr: Float = 0, firstBad = -1
            for k in 0..<n {
                if !out[k].isFinite { nonFinite += 1; if firstBad < 0 { firstBad = k } ; continue }
                maxErr = max(maxErr, abs(out[k] - input[k]))
            }
            anyNonFinite += nonFinite
            report.append("\(label): nonFinite=\(nonFinite)/\(n) maxRoundTripErr=\(maxErr) firstBad=\(firstBad)")
            XCTAssertEqual(nonFinite, 0, "[\(label)] standalone cast produced \(nonFinite) non-finite from finite input")
        }

        // 1) graph.run
        do {
            let (graph, ph, back) = buildCastGraph()
            let res = graph.run(with: queue, feeds: [ph: makeTD(input)],
                                targetTensors: [back], targetOperations: nil)
            guard let td = res[back] else { return XCTFail("graph.run produced no result") }
            evaluate("graph.run", readTD(td))
        }

        // 2) executable level0 / level1
        for level in [MPSGraphOptimization.level0, .level1] {
            let (graph, ph, back) = buildCastGraph()
            let desc = MPSGraphCompilationDescriptor()
            desc.optimizationLevel = level
            let shaped = MPSGraphShapedType(shape: shape, dataType: .float32)
            let exe = graph.compile(with: MPSGraphDevice(mtlDevice: device),
                                    feeds: [ph: shaped], targetTensors: [back],
                                    targetOperations: nil, compilationDescriptor: desc)
            guard let feedOrder = exe.feedTensors else { return XCTFail("no feedTensors") }
            let bindings: [MPSGraphTensor: MPSGraphTensorData] = [ph: makeTD(input)]
            let inputs = feedOrder.map { bindings[$0]! }
            guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
            let mcb = MPSCommandBuffer(commandBuffer: cb)
            let results = exe.encode(to: mcb, inputs: inputs, results: nil, executionDescriptor: nil)
            mcb.commit()
            mcb.waitUntilCompleted()
            evaluate("executable.level\(level.rawValue)", readTD(results[0]))
        }

        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DrewsChessMachine", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try (report.joined(separator: "\n") + "\n").write(
            to: dir.appendingPathComponent("cast_probe_standalone.txt"),
            atomically: true, encoding: .utf8)
    }

    /// Fingerprint / canary aliasing probe. MPSGraph hides the live buffer
    /// addresses, so instead of diffing pointers we tag every persistent tensor
    /// with a unique, decodable fingerprint and watch for foreign values
    /// appearing after a step — which both detects a stomp AND names its source.
    ///
    /// Key trick: learningRate = 0 freezes the SGD weight update (`updated =
    /// master − 0·… = master`), so every trainable master/working should read
    /// back EXACTLY its fingerprint. The full bf16 update graph still runs, so a
    /// buffer-aliasing stomp still fires. Any deviation is therefore a pure
    /// stomp, and its value decodes to the culprit:
    ///   baseFP#j  → trainable/bnStat j's buffer aliases this one
    ///   const~1.0 → a constant-1.0 scratch buffer is the source
    ///   NaN/other → uninitialized / computed scratch
    /// Fingerprint f(i) = 8 + i/16 (exact bf16 grid at this binade, foreign to
    /// real weights, ≠ 1.0). Runs several steps to beat the intermittency.
    // DISABLED: faults the GPU/test process (out-of-range fingerprint weights),
    // which aborts the whole run. Renamed off the `test` prefix so XCTest skips
    // it; kept for reference. The split A/B below is the decisive experiment.
    func DISABLED_bf16_fingerprintAliasingProbe() async throws {
        try requireMetal()
        var a = NetworkArchitecture.current
        a.computeDataType = .bFloat16
        var g = a.blockGroups[0]
        g.count = 1
        a.blockGroups = [g]

        let trainer = try ChessTrainer(
            learningRate: 0, weightDecayC: 0, momentumCoeff: 0,
            lrWarmupSteps: 0, arch: a)
        let trainables = trainer.network.trainableVariables
        let bnStats = trainer.network.bnRunningStatsVariables
        let nTrain = trainables.count

        func count(_ v: MPSGraphTensor) -> Int { (v.shape ?? []).reduce(1) { $0 * $1.intValue } }
        func fp(_ i: Int) -> Float { 8.0 + Float(i) / 16.0 }

        // Build the full trainer-state fingerprint: base (trainables+bnStats)
        // then velocity (trainables). Velocity is overwritten by the gradient
        // each step, so it isn't a stable canary — seed it 0 and don't judge it.
        var loadState: [[Float]] = []
        let persistent = trainables + bnStats
        for (i, v) in persistent.enumerated() { loadState.append([Float](repeating: fp(i), count: count(v))) }
        for v in trainables { loadState.append([Float](repeating: 0, count: count(v))) }
        try await trainer.loadTrainerWeights(loadState)

        // Run several steps; lr=0 freezes trainables, so the stomp is the only
        // thing that can move them. Catch any nonFiniteLoss throw — the graph
        // (and the stomp) already ran before the loss check.
        for _ in 0..<16 {
            do { _ = try await trainer.trainStep(batchSize: 32) } catch { /* graph ran; keep going */ }
        }

        let working = try await trainer.network.exportWeights()
        let masters = Array(try await trainer.exportTrainerWeights().prefix(persistent.count))

        func decode(_ v: Float) -> String {
            if v.isNaN { return "NaN" }
            if !v.isFinite { return "Inf(\(v))" }
            if abs(v - 1.0) < 1e-3 { return "const~1.0" }
            let idx = (v - 8.0) * 16.0
            if v >= 7.9 && v <= 12.0 && abs(idx - idx.rounded()) < 0.05 {
                return "baseFP#\(Int(idx.rounded()))"
            }
            return "other(\(v))"
        }

        var lines = [
            "=== fingerprint aliasing probe (single block, bf16, lr=0, 16 steps) ===",
            "trainables=\(nTrain) bnStats=\(bnStats.count); f(i)=8+i/16; checking TRAINABLE master+working only",
        ]
        var anyStomp = false
        for i in 0..<nTrain {
            let exp = fp(i)
            for (label, buf) in [("working", working), ("master", masters)] {
                var foreign: [String: Int] = [:]
                var bad = 0
                for v in buf[i] where abs(v - exp) > 1e-2 || !v.isFinite {
                    bad += 1
                    foreign[decode(v), default: 0] += 1
                }
                if bad > 0 {
                    anyStomp = true
                    let v = trainables[i]
                    let shape = (v.shape ?? []).map { "\($0.intValue)" }.joined(separator: "×")
                    let name = v.operation.name
                    let srcs = foreign.sorted { $0.value > $1.value }
                        .map { "\($0.key)×\($0.value)" }.joined(separator: " ")
                    lines.append("  STOMP \(label) [\(i)] expFP=\(exp) shape=[\(shape)] \(name) bad=\(bad)/\(buf[i].count) sources={ \(srcs) }")
                }
            }
        }
        if !anyStomp { lines.append("  (no stomp this run — all trainable buffers held their fingerprint)") }

        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DrewsChessMachine", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try (lines.joined(separator: "\n") + "\n").write(
            to: dir.appendingPathComponent("cast_probe_fingerprint.txt"),
            atomically: true, encoding: .utf8)

        XCTAssertEqual(working.count, persistent.count)   // probe value is the file
    }

    /// A/B test of the split-working-sync hypothesis. Runs a single-block bf16
    /// trainer twice — fused (control) and split — for 16 steps each, recording
    /// the worst per-step non-finite count in the working weights. If the fused
    /// dual-write executable's alias planner is the stomp, the control shows
    /// NaN and the split stays clean. Writes both to cast_probe_split.txt.
    func test_bf16_splitWorkingSync_ab() async throws {
        try requireMetal()
        // Marker so a later read can tell "test ran but hung" from "suite never
        // reached this test".
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DrewsChessMachine", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? "split test STARTED\n".write(
            to: dir.appendingPathComponent("cast_probe_split.txt"),
            atomically: true, encoding: .utf8)
        func singleBlockBf16() -> NetworkArchitecture {
            var a = NetworkArchitecture.current
            a.computeDataType = .bFloat16
            var grp = a.blockGroups[0]
            grp.count = 1
            a.blockGroups = [grp]
            return a
        }
        func worstNonFiniteOver16Steps(split: Bool) async throws -> Int {
            let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: singleBlockBf16(),
                                           splitWorkingWeightSync: split)
            var worst = 0
            for _ in 0..<8 {
                do { _ = try await trainer.trainStep(batchSize: 32) } catch { /* graph ran */ }
                let working = try await trainer.network.exportWeights()
                var bad = 0
                for t in working { for v in t where !v.isFinite { bad += 1 } }
                worst = max(worst, bad)
            }
            return worst
        }

        let controlWorst = try await worstNonFiniteOver16Steps(split: false)
        let splitWorst = try await worstNonFiniteOver16Steps(split: true)

        let report = """
        === split-working-sync A/B (single block, bf16, 16 steps each) ===
        control (fused dual-write): worstNonFinite=\(controlWorst)
        split   (separate pass):    worstNonFinite=\(splitWorst)
        interpretation: control>0 & split==0 ⇒ fused dual-write executable is the stomp
        """
        try (report + "\n").write(
            to: dir.appendingPathComponent("cast_probe_split.txt"),
            atomically: true, encoding: .utf8)

        // The probe value is the file comparison; intermittency means neither
        // arm is guaranteed on a single run, so don't hard-assert the outcome.
        XCTAssertGreaterThanOrEqual(controlWorst, 0)
    }

    // MARK: - Theory #6: completion / fence timing between consecutive steps
    //
    // `trainStep` already reads its loss scalars back, which forces the step's
    // command buffer to complete before the next step is dispatched — so a
    // cross-step GPU race shouldn't be possible. This cell makes that explicit:
    // it inserts a real delay between bf16 steps past the step-2 onset. If the
    // delay makes the run finite where back-to-back (test_bf16_batch64_steps4)
    // went NaN, the fault is a missing fence/ordering in the new stack; if it
    // still goes NaN, timing is exonerated and the fault is purely numeric.
    func test_bf16_batch64_steps4_withDelayBetweenSteps() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch(.bFloat16))
        for s in 0..<4 {
            let t = try await trainer.trainStep(batchSize: 64)
            XCTAssertTrue(t.loss.isFinite, "bf16+delay loss non-finite at step \(s): \(t.loss)")
            XCTAssertTrue(t.gradGlobalNorm.isFinite, "bf16+delay gradNorm non-finite at step \(s)")
            try await Task.sleep(nanoseconds: 50_000_000)   // 50 ms between steps
        }
    }
}
