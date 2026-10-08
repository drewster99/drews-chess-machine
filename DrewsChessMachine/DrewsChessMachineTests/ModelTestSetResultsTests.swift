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
        ModelTestSetResults(evaluatedAtUnix: 1_791_414_697, build: 2427, policyTailPrecision: .mixedFinalProjection, sets: sets)
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
        let input = TacticalProbeRunner.encodedBoards(probes, encoding: network.inputEncoding)
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

    /// The per-theme fold comes back in `ProbeCategory` order every time, so
    /// the overall sums (NLL above all) are added in one order and a
    /// result is bit-reproducible. It came back in `Dictionary` order, which
    /// changes between dictionaries: two evaluations of the same weights
    /// recorded NLLs differing in the last bit.
    func testTheThemeFoldIsInCategoryOrderEveryTime() {
        let results = LichessProbeData.wide.probes.map { TacticalProbeRunner.errorResult(for: $0) }
        let expected = ProbeCategory.allCases.filter { category in results.contains { $0.probe.category == category } }
        XCTAssertEqual(expected.count, 13)
        for _ in 0..<20 {
            XCTAssertEqual(LichessProbeHistory.aggregates(from: results).map(\.theme), expected)
        }
    }

    /// A network whose forward pass is non-finite fails the evaluation, and
    /// every position is an error, instead of the uniform-fallback numbers a
    /// NaN softmax used to produce.
    func testNonFiniteOutputsFailTheEvaluation() async throws {
        let set = Self.subset(LichessProbeData.set200, count: 20)
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 3))
        let poisoned = try await network.exportWeights().map { [Float](repeating: .nan, count: $0.count) }

        let field = await ModelTestSetEvaluator(testSets: [set]).evaluate(weights: poisoned, architecture: network.network.arch)
        guard case .failed(let reason) = field else { return XCTFail("expected failure, got \(field)") }
        XCTAssertTrue(reason.contains("20 of 20 positions have non-finite policy logits or value outputs"), reason)

        try await network.network.loadWeights(poisoned)
        let input = TacticalProbeRunner.encodedBoards(set.probes, encoding: network.inputEncoding)
        let batch = await TacticalProbeRunner.runBatch(set.probes, encodedInput: input, against: network)
        XCTAssertEqual(batch.nonFinitePositions, 20)
        XCTAssertTrue(batch.results.allSatisfy { $0.verdict == .error })
    }

    func testTheSharedEncoderIsTheBoardsInOrder() {
        let probes = Array(LichessProbeData.set200.probes.prefix(5))
        let expected = probes.flatMap { BoardEncoder.encode($0.state, encoding: .basic24) }
        XCTAssertEqual(TacticalProbeRunner.encodedBoards(probes, encoding: .basic24), expected)
    }

    // MARK: - Reuse of an identical evaluation

    /// Identical base weights under an equal architecture are evaluated once:
    /// a trainer file (velocity attached) after its champion reuses the
    /// champion's evaluation; one changed float, another architecture or a
    /// failed evaluation does not reuse.
    func testIdenticalBaseWeightsReuseTheEvaluation() async throws {
        let evaluator = ModelTestSetEvaluator(testSets: [Self.subset(LichessProbeData.set200, count: 8)])
        let arch = NetworkArchitecture.current
        let base = try await ChessMPSNetwork(.randomWeights(initSeed: 21)).exportWeights()
        let velocity = arch.trainableTensorPlan().map { [Float](repeating: 0.25, count: $0.elementCount) }

        let first = await evaluator.resultsForSave(weights: base, architecture: arch, file: "champion.safetensors")
        XCTAssertEqual(first.source, .evaluatedForThisFile)
        let second = await evaluator.resultsForSave(weights: base + velocity, architecture: arch, file: "trainer.safetensors")
        XCTAssertEqual(second.source, .reusedFrom(file: "champion.safetensors"))
        XCTAssertEqual(second.field, first.field)

        var changed = base
        changed[0][0] = changed[0][0].nextUp
        let third = await evaluator.resultsForSave(weights: changed, architecture: arch, file: "changed.safetensors")
        XCTAssertEqual(third.source, .evaluatedForThisFile)

        var otherArch = arch
        otherArch.blockGroups[0].count += 1
        let tooFew = await evaluator.resultsForSave(weights: base, architecture: otherArch, file: "other.safetensors")
        XCTAssertEqual(tooFew.source, .evaluatedForThisFile)
        guard case .failed = tooFew.field else { return XCTFail("the other architecture's tensors are missing") }
        let again = await evaluator.resultsForSave(weights: base, architecture: otherArch, file: "other.safetensors")
        XCTAssertEqual(again.source, .evaluatedForThisFile, "a failure is never reused")

        let remembered = await evaluator.resultsForSave(weights: base, architecture: arch, file: "x")
        XCTAssertEqual(remembered.source, .reusedFrom(file: "champion.safetensors"), "still remembered")
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
