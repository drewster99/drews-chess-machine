import XCTest
@testable import DrewsChessMachine

/// The activation-aware parked classification (`BatchNormPassThrough`) and
/// its `LayerHealth` counts: parity with the Python mirror
/// (`experiments/20261005-lr-schedule-ab/bn_liveness.py`) on reference values
/// it produced, the exact relu / leaky_relu equivalence with the dead /
/// mostly-off bands, and arm B-silu's real batch-norm parameters at trainer
/// steps 20,000 / 21,000 / 22,000 (read once, read-only, from its
/// checkpoints into a test resource; nothing under ~/Library is read here).
final class BatchNormPassThroughTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private struct Reference: Decodable {
        struct Case: Decodable {
            let activation: String
            let gamma: Double
            let beta: Double
            let excessPassThrough: Double
            let passThrough: Double
            let band: String

            enum CodingKeys: String, CodingKey {
                case activation, gamma, beta, band
                case excessPassThrough = "excess_pass_through"
                case passThrough = "pass_through"
            }
        }

        let parkedBelow: Double
        let mostlyOffBelow: Double
        let siluDerivativeRoot: Double
        let geluDerivativeRoot: Double
        let cases: [Case]

        enum CodingKeys: String, CodingKey {
            case cases
            case parkedBelow = "parked_below"
            case mostlyOffBelow = "mostly_off_below"
            case siluDerivativeRoot = "silu_derivative_root"
            case geluDerivativeRoot = "gelu_derivative_root"
        }
    }

    private struct BSilu: Decodable {
        struct Checkpoint: Decodable {
            let file: String
            let contentSHA256: String
            let trainingStep: Int
            let cumTrainerStep: Int
            let sites: [Site]

            enum CodingKeys: String, CodingKey {
                case file, sites
                case contentSHA256 = "content_sha256"
                case trainingStep = "training_step"
                case cumTrainerStep = "cum_trainer_step"
            }
        }

        struct Site: Decodable {
            let site: String
            let activation: String
            let gamma: [Double]
            let beta: [Double]
            let expected: Expected
        }

        struct Expected: Decodable {
            let parked: Int
            let mostlyOff: Int
            let minPassThrough: Double
            let medianPassThrough: Double
            let nonFinite: Int

            enum CodingKeys: String, CodingKey {
                case parked
                case mostlyOff = "mostly_off"
                case minPassThrough = "min_pass_through"
                case medianPassThrough = "median_pass_through"
                case nonFinite = "non_finite"
            }
        }

        let checkpoints: [Checkpoint]
    }

    private func activation(_ name: String) throws -> ActivationFunction {
        guard let function = ActivationFunction(rawValue: name) else {
            throw S.ResourceError.missing("activation \(name)")
        }
        return function
    }

    // MARK: Parity with the Python mirror

    func testConstantsMatchThePythonMirror() throws {
        let reference = try JSONDecoder().decode(
            Reference.self, from: try S.resourceData("BatchNormPassThroughReference", extension: "json"))
        XCTAssertEqual(BatchNormPassThrough.parkedBelow, reference.parkedBelow, accuracy: 1e-15)
        XCTAssertEqual(BatchNormPassThrough.mostlyOffBelow, reference.mostlyOffBelow, accuracy: 1e-15)
        XCTAssertEqual(BatchNormPassThrough.siluDerivativeRoot, reference.siluDerivativeRoot, accuracy: 1e-12)
        XCTAssertEqual(BatchNormPassThrough.geluDerivativeRoot, reference.geluDerivativeRoot, accuracy: 1e-12)
        XCTAssertLessThan(abs(BatchNormPassThrough.siluDerivative(BatchNormPassThrough.siluDerivativeRoot)), 1e-12)
        XCTAssertLessThan(abs(BatchNormPassThrough.geluDerivative(BatchNormPassThrough.geluDerivativeRoot)), 1e-12)
    }

    func testPassThroughMatchesThePythonMirror() throws {
        let reference = try JSONDecoder().decode(
            Reference.self, from: try S.resourceData("BatchNormPassThroughReference", extension: "json"))
        XCTAssertEqual(reference.cases.count, 120)
        for item in reference.cases {
            let function = try activation(item.activation)
            let label = "\(item.activation) γ=\(item.gamma) β=\(item.beta)"
            let x = BatchNormPassThrough.excessPassThrough(activation: function, gamma: item.gamma, beta: item.beta)
            let p = BatchNormPassThrough.passThrough(activation: function, gamma: item.gamma, beta: item.beta)
            XCTAssertEqual(x, item.excessPassThrough, accuracy: max(1e-13, 1e-10 * abs(item.excessPassThrough)), label)
            XCTAssertEqual(p, item.passThrough, accuracy: max(1e-13, 1e-10 * abs(item.passThrough)), label)
            XCTAssertEqual(
                BatchNormPassThrough.band(activation: function, gamma: item.gamma, beta: item.beta).rawValue, item.band, label)
        }
    }

    func testGaussLegendreRuleIsExactForPolynomials() {
        let (nodes, weights) = BatchNormPassThrough.gaussLegendre
        XCTAssertEqual(nodes.count, 16)
        XCTAssertEqual(weights.reduce(0, +), 2, accuracy: 1e-14)
        // Exact for degree ≤ 31: ∫ x^30 dx over [−1, 1] = 2/31.
        let integral = zip(nodes, weights).reduce(0) { $0 + $1.1 * pow($1.0, 30) }
        XCTAssertEqual(integral, 2.0 / 31.0, accuracy: 1e-13)
        XCTAssertEqual(nodes, nodes.sorted())
    }

    // MARK: Exact equivalence for relu / leaky_relu

    func testReluAndLeakyParkedEqualsDeadExactly() {
        var generator = SplitMix64(seed: 20_261_006)
        var gamma: [Float] = [0, 0, 0, -0.0]
        var beta: [Float] = [1, 0, -1, -0.0]
        for _ in 0..<20_000 {
            gamma.append(Float(generator.nextNormal() * 2))
            beta.append(Float(generator.nextNormal() * 3 - 2))
        }
        // Exactly on the bands' edges too.
        gamma.append(contentsOf: [1, 1, 2, 2])
        beta.append(contentsOf: [-3, -2, -6, -4])
        let variance = [Float](repeating: 1, count: gamma.count)
        for function in [ActivationFunction.relu, .leakyRelu] {
            let health = LayerHealth.batchNormSiteHealth(
                site: .init(name: "s", channels: gamma.count, activation: function),
                gamma: gamma, beta: beta, runningVariance: variance)
            // Independent of `band` (which reuses the β/|γ| comparisons): the
            // pass-through model's X = Φ(β/|γ|) against its own Φ(−3) / Φ(−2)
            // lines must land every channel in the same band.
            var parkedByExcessPassThrough = 0
            var mostlyOffByExcessPassThrough = 0
            for (g, b) in zip(gamma, beta) {
                let x = BatchNormPassThrough.excessPassThrough(activation: function, gamma: Double(g), beta: Double(b))
                if x < BatchNormPassThrough.parkedBelow {
                    parkedByExcessPassThrough += 1
                } else if x < BatchNormPassThrough.mostlyOffBelow {
                    mostlyOffByExcessPassThrough += 1
                }
            }
            XCTAssertEqual(health.deadChannelCount, parkedByExcessPassThrough, function.rawValue)
            XCTAssertEqual(health.parkedChannelCount, parkedByExcessPassThrough, function.rawValue)
            XCTAssertEqual(health.mostlyOffChannelCount, mostlyOffByExcessPassThrough, function.rawValue)
            XCTAssertEqual(health.parkedMostlyOffChannelCount, mostlyOffByExcessPassThrough, function.rawValue)
            XCTAssertNotNil(health.parkedChannelCount)
        }
    }

    func testUnactivatedSiteHasNoParkedCount() {
        let health = LayerHealth.batchNormSiteHealth(
            site: .init(name: "stem.bn", channels: 2, activation: nil), gamma: [1, 1], beta: [-10, 10],
            runningVariance: [1, 1])
        XCTAssertNil(health.parkedChannelCount)
        XCTAssertNil(health.minPassThrough)
    }

    func testSmoothSitesAreClassifiedByPassThrough() {
        // SiLU at |γ| = 1: parked below β ≈ −9.06; GELU's line is nearer.
        let gamma: [Float] = [1, 1, 1, 1, .nan]
        let beta: [Float] = [0, -6, -8, -9.5, 0]
        let silu = LayerHealth.batchNormSiteHealth(
            site: .init(name: "s", channels: 5, activation: .silu), gamma: gamma, beta: beta,
            runningVariance: [1, 1, 1, 1, 1])
        XCTAssertNil(silu.deadChannelCount, "the relu classification still says n/a")
        XCTAssertEqual(silu.parkedChannelCount, 1)
        XCTAssertEqual(silu.parkedMostlyOffChannelCount, 2, "β = −6 and −8 (X 0.016 and 0.0033)")
        XCTAssertEqual(silu.nonFiniteChannelCount, 1)
        let gelu = LayerHealth.batchNormSiteHealth(
            site: .init(name: "g", channels: 5, activation: .gelu), gamma: gamma, beta: beta,
            runningVariance: [1, 1, 1, 1, 1])
        XCTAssertEqual(gelu.parkedChannelCount, 3)
    }

    // MARK: Arm B-silu's real parameters

    func testBSiluCheckpointsMatchBnLiveness() throws {
        let data = try JSONDecoder().decode(
            BSilu.self, from: try S.resourceData("TrainingHealthBSiluBatchNorm", extension: "json"))
        XCTAssertEqual(data.checkpoints.map(\.cumTrainerStep), [20_000, 21_000, 22_000])
        var report: [String] = []
        for checkpoint in data.checkpoints {
            XCTAssertFalse(checkpoint.contentSHA256.isEmpty)
            for site in checkpoint.sites {
                let function = try activation(site.activation)
                let health = LayerHealth.batchNormSiteHealth(
                    site: .init(name: site.site, channels: site.gamma.count, activation: function),
                    gamma: site.gamma.map { Float($0) }, beta: site.beta.map { Float($0) },
                    runningVariance: [Float](repeating: 1, count: site.gamma.count))
                let label = "step \(checkpoint.cumTrainerStep) \(site.site) (\(site.activation))"
                XCTAssertEqual(health.parkedChannelCount, site.expected.parked, label)
                XCTAssertEqual(health.parkedMostlyOffChannelCount, site.expected.mostlyOff, label)
                XCTAssertEqual(try XCTUnwrap(health.minPassThrough), site.expected.minPassThrough, accuracy: 1e-9, label)
                XCTAssertEqual(try XCTUnwrap(health.medianPassThrough), site.expected.medianPassThrough, accuracy: 1e-9, label)
                report.append("\(label): parked \(health.parkedChannelCount ?? -1) mostly off \(health.parkedMostlyOffChannelCount ?? -1) min P \(String(format: "%.6f", health.minPassThrough ?? .nan)) median P \(String(format: "%.4f", health.medianPassThrough ?? .nan))")
            }
        }
        // The owner's checks: policy.pre_bn (leaky_relu) 0 parked at 20k,
        // 20 parked / 7 mostly off at 21k, 21 / 8 at 22k.
        func site(_ step: Int, _ name: String) throws -> BSilu.Site {
            let checkpoint = try XCTUnwrap(data.checkpoints.first { $0.cumTrainerStep == step })
            return try XCTUnwrap(checkpoint.sites.first { $0.site == name })
        }
        XCTAssertEqual(try site(20_000, "policy.pre_bn").expected.parked, 0)
        XCTAssertEqual(try site(21_000, "policy.pre_bn").expected.parked, 20)
        XCTAssertEqual(try site(21_000, "policy.pre_bn").expected.mostlyOff, 7)
        XCTAssertEqual(try site(22_000, "policy.pre_bn").expected.parked, 21)
        XCTAssertEqual(try site(22_000, "policy.pre_bn").expected.mostlyOff, 8)
        print("[B-silu parked counts]\n" + report.joined(separator: "\n"))
    }

    func testCompactLineCarriesParkedFieldsWithoutChangingTheReluFields() throws {
        let data = try JSONDecoder().decode(
            BSilu.self, from: try S.resourceData("TrainingHealthBSiluBatchNorm", extension: "json"))
        let checkpoint = try XCTUnwrap(data.checkpoints.first { $0.cumTrainerStep == 21_000 })
        let sites = try checkpoint.sites.map { site in
            LayerHealth.batchNormSiteHealth(
                site: .init(name: site.site, channels: site.gamma.count, activation: try activation(site.activation)),
                gamma: site.gamma.map { Float($0) }, beta: site.beta.map { Float($0) },
                runningVariance: [Float](repeating: 1, count: site.gamma.count))
        }
        let summary = LayerHealthSummary(
            scope: .batchNormStateOnly, batchNormSites: sites, squeezeExcitationFC1: nil, valueFC1: nil,
            velocityNotIncludedReason: "live", reZero: [], examinedTensorCount: 0, examinedValueCount: 0,
            nonFiniteValueCount: 0, nonFiniteTensorCount: 0, nonFiniteTensorNames: [], largestMagnitudeTensors: nil)
        let line = summary.compactLine()
        XCTAssertTrue(line.contains("reluSites=2/9 ch=144 dead=21 "), line)
        XCTAssertTrue(line.contains(" parkedSites=9/9 parkedCh=1040 parked=21 parkedOff=11 parkedBy=policy.pre_bn:20/128,value.bn:1/16"), line)
        let digest = LayerHealthDigest(summary: summary)
        XCTAssertEqual(digest.deadChannels?.modeledChannelCount, 1040)
        XCTAssertEqual(digest.deadChannels?.parkedChannelCount, 21)
        XCTAssertEqual(digest.deadChannels?.modeledSiteCount, 9)
        // The table gains a parked section and keeps its batch-norm rows.
        let detail = summary.detailedLines()
        XCTAssertTrue(detail.contains { $0.hasPrefix("parked channels — every activation") })
        XCTAssertTrue(detail.contains { $0.hasPrefix("  blocks.0.bn1    silu          128    n/a") })
    }
}

/// A small deterministic generator for the equivalence test's inputs.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextUniform() -> Double { Double(next() >> 11) / Double(1 << 53) }
    mutating func nextNormal() -> Double {
        let u1 = max(nextUniform(), .leastNonzeroMagnitude)
        let u2 = nextUniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}
