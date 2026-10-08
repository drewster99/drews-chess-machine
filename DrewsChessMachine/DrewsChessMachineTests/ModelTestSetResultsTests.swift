import XCTest
@testable import DrewsChessMachine

/// Model-file test-set results (test-set results plan D1, D4): the JSON form
/// and its reading, and the evaluator's numbers against a direct fold of the
/// per-position results.
final class ModelTestSetResultsTests: XCTestCase {

    private static func setResult(id: String = "lichess-200", positions: Int = 200, pElo: ModelTestSetResults.PuzzleElo = .estimate(1630.4)) -> ModelTestSetResults.SetResult {
        ModelTestSetResults.SetResult(
            id: id, title: "Lichess puzzles, 200", description: "200 puzzles.",
            fingerprintSHA256: String(repeating: "a", count: 64), positions: positions,
            top1Correct: 99, top5Correct: 170, avgCorrectProbability: 0.1923, avgCorrectRank: 3.667,
            nll: 2.071, pElo: pElo,
            themes: [.init(id: "hangingPiece", title: "Hanging piece", correct: 20, total: 25)]
        )
    }

    private static func results(_ sets: [ModelTestSetResults.SetResult]) -> ModelTestSetResults {
        ModelTestSetResults(evaluatedAtUnix: 1_791_414_697, build: 2427, policyTailPrecision: "mixed_final_projection", sets: sets)
    }

    private static func read(_ field: ModelTestSetResultsField) throws -> ModelTestSetResultsInFile {
        ModelTestSetResultsField.reading(fromMetadata: [ModelTestSetResultsField.metadataKey: try field.metadataValue()])
    }

    // MARK: - JSON

    func testEveryCaseRoundTrips() throws {
        let cases: [ModelTestSetResultsField] = [
            .evaluated(Self.results([Self.setResult(), Self.setResult(id: "lichess-wide", positions: 4435)])),
            .evaluated(Self.results([Self.setResult(pElo: .allCorrect)])),
            .evaluated(Self.results([Self.setResult(pElo: .allWrong)])),
            .failed(reason: "the forward pass failed for 3 of 200 positions in lichess-200"),
        ]
        for field in cases {
            XCTAssertEqual(try Self.read(field), .recorded(field))
        }
        XCTAssertEqual(ModelTestSetResultsField.reading(fromMetadata: [:]), .notRecorded)
    }

    /// The exact text: sorted keys, snake_case, pElo bounds as null + bound.
    func testTheJSONText() throws {
        let failed = try ModelTestSetResultsField.failed(reason: "r").metadataValue()
        XCTAssertEqual(failed, #"{"reason":"r","schema":1,"status":"failed"}"#)

        let bounded = try ModelTestSetResultsField.evaluated(Self.results([Self.setResult(pElo: .allCorrect)])).metadataValue()
        XCTAssertTrue(bounded.hasPrefix(#"{"build":2427,"evaluated_at_unix":1791414697,"policy_tail_precision":"mixed_final_projection","schema":1,"sets":[{"avg_correct_probability":0.1923,"avg_correct_rank":3.667,"description":"200 puzzles.","fingerprint_sha256":""#), bounded)
        XCTAssertTrue(bounded.contains(#""pelo":null,"pelo_bound":"all_correct","positions":200,"#), bounded)
        XCTAssertTrue(bounded.contains(#""themes":[{"correct":20,"id":"hangingPiece","title":"Hanging piece","total":25}],"#), bounded)
        XCTAssertTrue(bounded.hasSuffix(#""status":"evaluated"}"#), bounded)

        let estimated = try ModelTestSetResultsField.evaluated(Self.results([Self.setResult()])).metadataValue()
        XCTAssertTrue(estimated.contains(#""pelo":1630.4,"pelo_bound":null,"#), estimated)
    }

    func testANonFiniteNumberIsNeverWritten() {
        let field = ModelTestSetResultsField.evaluated(Self.results([Self.setResult(pElo: .estimate(.infinity))]))
        XCTAssertThrowsError(try field.metadataValue())
    }

    func testUnreadableValuesAreReportedNotThrown() throws {
        let unreadable: [String] = [
            "not json",
            #"{"schema":2,"status":"failed","reason":"r"}"#,
            #"{"schema":1,"status":"maybe"}"#,
            #"{"schema":1,"status":"failed"}"#,
        ]
        for raw in unreadable {
            guard case .unreadable = ModelTestSetResultsField.reading(fromMetadata: [ModelTestSetResultsField.metadataKey: raw]) else {
                return XCTFail("\(raw) should read as unreadable")
            }
        }
        // Both the estimate and a bound, and neither, are refused.
        let good = try ModelTestSetResultsField.evaluated(Self.results([Self.setResult()])).metadataValue()
        for broken in [good.replacingOccurrences(of: #""pelo_bound":null"#, with: #""pelo_bound":"all_wrong""#),
                       good.replacingOccurrences(of: #""pelo":1630.4"#, with: #""pelo":null"#)] {
            guard case .unreadable = ModelTestSetResultsField.reading(fromMetadata: [ModelTestSetResultsField.metadataKey: broken]) else {
                return XCTFail("\(broken) should read as unreadable")
            }
        }
    }

    func testTheLargestSetIsSummarized() {
        let results = Self.results([Self.setResult(id: "small", positions: 200), Self.setResult(id: "big", positions: 4435), Self.setResult(id: "tie", positions: 4435)])
        XCTAssertEqual(results.largestSet?.id, "big", "ties keep the first")
        XCTAssertNil(Self.results([]).largestSet)
    }

    // MARK: - Evaluator

    /// A seeded network's results equal a direct fold of the same
    /// per-position results (independent of `ProbeBatterySummary`), set by
    /// set, theme by theme.
    func testTheEvaluatorMatchesADirectFold() async throws {
        let small200 = Self.subset(LichessProbeData.set200, count: 40)
        let smallWide = Self.subset(LichessProbeData.wide, count: 60)
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 3))
        let weights = try await network.exportWeights()

        let field = await ModelTestSetEvaluator(testSets: [small200, smallWide]).evaluate(weights: weights, architecture: network.network.arch)
        guard case .evaluated(let results) = field else { return XCTFail("not evaluated: \(field)") }
        XCTAssertEqual(results.sets.map(\.id), [small200.id, smallWide.id])
        XCTAssertEqual(results.build, BuildInfo.buildNumber)

        let probes = small200.probes + smallWide.probes
        var input: [Float] = []
        for probe in probes {
            input.append(contentsOf: BoardEncoder.encode(probe.state, encoding: network.inputEncoding))
        }
        let batch = await TacticalProbeRunner.runBatch(probes, encodedInput: input, against: network)
        let slices = [Array(batch.results[0..<40]), Array(batch.results[40..<100])]
        for (set, direct) in zip(results.sets, slices) {
            let top1 = direct.filter { $0.verdict == .correctAndConfident || $0.verdict == .correctButFlat }.count
            let top5 = top1 + direct.filter { $0.verdict == .correctInTop5 }.count
            XCTAssertEqual(set.positions, direct.count)
            XCTAssertEqual(set.top1Correct, top1)
            XCTAssertEqual(set.top5Correct, top5)
            XCTAssertEqual(set.avgCorrectProbability, Double(direct.map(\.expectedProb).reduce(0, +)) / Double(direct.count), accuracy: 1e-5)
            XCTAssertEqual(set.avgCorrectRank, Double(direct.compactMap(\.expectedRank).reduce(0, +)) / Double(direct.count), accuracy: 1e-5)
            XCTAssertEqual(set.nll, direct.map { ProbeBookmoveNLL.nats(expectedProb: $0.expectedProb) }.reduce(0, +) / Double(direct.count), accuracy: 1e-9)
            for theme in set.themes {
                let inTheme = direct.filter { $0.probe.category.lichessThemeID == theme.id }
                XCTAssertEqual(theme.total, inTheme.count, theme.id)
                XCTAssertEqual(theme.correct, inTheme.filter { $0.verdict == .correctAndConfident || $0.verdict == .correctButFlat }.count, theme.id)
                XCTAssertEqual(theme.title, ProbeCategory(lichessThemeID: theme.id)?.title)
            }
            XCTAssertEqual(set.themes.map(\.total).reduce(0, +), direct.count)
        }
    }

    func testAnErroredPositionFailsTheSetInsteadOfCountingAsWrong() {
        let set = Self.subset(LichessProbeData.set200, count: 3)
        let errored = set.probes.map { TacticalProbeRunner.errorResult(for: $0) }
        XCTAssertThrowsError(try ModelTestSetEvaluator.setResult(set, results: errored)) { error in
            XCTAssertTrue("\(error)".contains("the forward pass failed for 3 of 3 positions"), "\(error)")
        }
    }

    func testTooFewTensorsIsAFailedResultNotAThrow() async {
        let field = await ModelTestSetEvaluator(testSets: [Self.subset(LichessProbeData.set200, count: 2)])
            .evaluate(weights: [[0]], architecture: .current)
        guard case .failed(let reason) = field else { return XCTFail("expected failure, got \(field)") }
        XCTAssertTrue(reason.contains("tensors"), reason)
    }

    private static func subset(_ set: ProbeTestSet, count: Int) -> ProbeTestSet {
        ProbeTestSet(id: set.id, title: set.title, description: set.description,
                     fingerprintSHA256: set.fingerprintSHA256, probes: Array(set.probes.prefix(count)))
    }
}
