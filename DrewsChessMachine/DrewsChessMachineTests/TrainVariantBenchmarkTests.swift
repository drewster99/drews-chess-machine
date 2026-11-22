import XCTest
@testable import DrewsChessMachine

/// Pure logic behind `--train-variant-benchmark`: the step types it times,
/// the percentile summary its table reports, the signed allocation delta,
/// and the argument parser.
final class TrainVariantBenchmarkTests: XCTestCase {

    // MARK: - Step types

    /// Protects: the three step types are the three distinct kinds of step
    /// the one training executable serves: no readback, diagnostics readback,
    /// and diagnostics readback plus the KL probe (a production probe step is
    /// also a diagnostics step).
    /// Fails if: two types map to the same flags (two rows would time the same
    /// step and one kind would go unmeasured), or the probe type stops reading
    /// diagnostics back.
    func testStepTypesCoverPlainDiagnosticsAndProbeOnce() {
        let combinations = TrainingStepType.allCases.map { [$0.includeDiagnostics, $0.runsKLProbe] }
        XCTAssertEqual(combinations, [[false, false], [true, false], [true, true]])
    }

    /// Protects: the KL-probe interval the benchmark sets makes every step of
    /// the probe type run the probe (1) and no step of the others (0),
    /// through the production schedule `ChessTrainer.isKLProbeStep`.
    /// Fails if: the interval for the probe type stops probing some steps, or
    /// another type probes.
    func testKLProbeIntervalSelectsTheProbeOnEveryStepOrNone() {
        for stepType in TrainingStepType.allCases {
            for stepIndex in [0, 1, 2, 7, 1000, 1001] {
                XCTAssertEqual(
                    ChessTrainer.isKLProbeStep(stepIndex: stepIndex, interval: stepType.klProbeInterval),
                    stepType.runsKLProbe,
                    "\(stepType.label) step \(stepIndex)")
            }
        }
    }

    // MARK: - Timing distribution

    /// Protects: a window with no steps has no distribution.
    /// Fails if: an empty sample list yields zeros (an invented measurement).
    func testEmptySamplesHaveNoDistribution() {
        XCTAssertNil(TimingDistribution(samples: []))
    }

    /// Protects: one sample is every statistic.
    /// Fails if: interpolation reads past the end with a single sample.
    func testOneSampleIsEveryStatistic() throws {
        let d = try XCTUnwrap(TimingDistribution(samples: [42.5]))
        XCTAssertEqual(d.count, 1)
        for value in [d.mean, d.p10, d.median, d.p90, d.min, d.max] {
            XCTAssertEqual(value, 42.5)
        }
    }

    /// Protects: type-7 (linear interpolation) percentiles of unsorted input,
    /// numpy's default. Reference values from numpy.percentile.
    /// Fails if: the input isn't sorted first, the rank uses n instead of
    /// n − 1, or nearest-rank replaces interpolation.
    func testPercentilesInterpolateLinearlyOverSortedSamples() throws {
        // np.percentile([1..10], [10, 50, 90]) = 1.9, 5.5, 9.1
        let d = try XCTUnwrap(TimingDistribution(samples: [7, 3, 10, 1, 5, 9, 2, 8, 6, 4]))
        XCTAssertEqual(d.p10, 1.9, accuracy: 1e-12)
        XCTAssertEqual(d.median, 5.5, accuracy: 1e-12)
        XCTAssertEqual(d.p90, 9.1, accuracy: 1e-12)
        XCTAssertEqual(d.mean, 5.5, accuracy: 1e-12)
        XCTAssertEqual(d.min, 1)
        XCTAssertEqual(d.max, 10)
        XCTAssertEqual(d.count, 10)
    }

    /// Protects: two samples interpolate between them.
    /// Fails if: the upper rank index is clamped wrongly at n = 2.
    func testTwoSamplesInterpolateBetweenThem() throws {
        let d = try XCTUnwrap(TimingDistribution(samples: [20, 10]))
        XCTAssertEqual(d.p10, 11, accuracy: 1e-12)
        XCTAssertEqual(d.median, 15, accuracy: 1e-12)
        XCTAssertEqual(d.p90, 19, accuracy: 1e-12)
    }

    /// Protects: the quantile's endpoints are the extremes and an odd count's
    /// median is the middle sample.
    /// Fails if: p = 1 reads one past the end, or the odd median interpolates.
    func testQuantileEndpointsAndOddMedian() {
        let sorted: [Double] = [1, 4, 9]
        XCTAssertEqual(TimingDistribution.quantile(sorted, 0), 1)
        XCTAssertEqual(TimingDistribution.quantile(sorted, 1), 9)
        XCTAssertEqual(TimingDistribution.quantile(sorted, 0.5), 4)
    }

    // MARK: - Allocation delta

    /// Protects: allocation deltas are signed (allocations can shrink).
    /// Fails if: a shrink traps on unsigned underflow or reports positive.
    func testSignedByteDeltaGrowShrinkAndEqual() {
        XCTAssertEqual(signedByteDelta(from: 100, to: 250), 150)
        XCTAssertEqual(signedByteDelta(from: 250, to: 100), -150)
        XCTAssertEqual(signedByteDelta(from: 7, to: 7), 0)
        XCTAssertEqual(signedByteDelta(from: 0, to: 1 << 40), 1 << 40)
    }

    // MARK: - Arguments

    /// Protects: the defaults the help text states (current preset, batch
    /// 4096, 120 s per variant, 5 warm-up steps, no JSON).
    /// Fails if: a default drifts from the documented one.
    func testDefaults() throws {
        let options = try TrainVariantBenchmarkCLI.parseOptions(["--train-variant-benchmark"])
        XCTAssertEqual(options.source, .currentPreset)
        XCTAssertEqual(options.batchSize, 4096)
        XCTAssertEqual(options.secondsPerVariant, 120)
        XCTAssertEqual(options.warmupSteps, 5)
        XCTAssertNil(options.jsonOutPath)
    }

    /// Protects: every value flag reaches its option, in any order.
    /// Fails if: a flag is parsed into the wrong field or ignored.
    func testEveryFlagParses() throws {
        let options = try TrainVariantBenchmarkCLI.parseOptions([
            "--json-out", "/x/out.json", "--train-variant-benchmark", "--batch", "1024",
            "--seconds-per-variant", "30.5", "--warmup", "1", "--start-model", "m.safetensors",
        ])
        XCTAssertEqual(options.source, .startModel("m.safetensors"))
        XCTAssertEqual(options.batchSize, 1024)
        XCTAssertEqual(options.secondsPerVariant, 30.5)
        XCTAssertEqual(options.warmupSteps, 1)
        XCTAssertEqual(options.jsonOutPath, "/x/out.json")
        let byArchitecture = try TrainVariantBenchmarkCLI.parseOptions(
            ["--train-variant-benchmark", "--architecture", "v4_5block_7x7"])
        XCTAssertEqual(byArchitecture.source, .architecture("v4_5block_7x7"))
    }

    /// Protects: every malformed command line is refused before any GPU work.
    /// Fails if: one of these edge cases is accepted.
    func testRefusesMalformedArguments() {
        let cases: [(String, [String])] = [
            ("architecture and start model together", ["--architecture", "a", "--start-model", "b"]),
            ("unknown flag", ["--sweep"]),
            ("stray positional", ["model.safetensors"]),
            ("flag given twice", ["--batch", "64", "--batch", "128"]),
            ("missing value at end", ["--batch"]),
            ("value is another flag", ["--json-out", "--batch", "64"]),
            ("batch zero", ["--batch", "0"]),
            ("batch negative", ["--batch", "-5"]),
            ("batch not a number", ["--batch", "lots"]),
            ("seconds zero", ["--seconds-per-variant", "0"]),
            ("seconds infinite", ["--seconds-per-variant", "inf"]),
            ("seconds NaN", ["--seconds-per-variant", "nan"]),
            ("warmup zero", ["--warmup", "0"]),
        ]
        for (name, arguments) in cases {
            XCTAssertThrowsError(
                try TrainVariantBenchmarkCLI.parseOptions(["--train-variant-benchmark"] + arguments), name)
        }
    }
}
