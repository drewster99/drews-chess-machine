import XCTest
@testable import DrewsChessMachine

final class SessionManifestTests: XCTestCase {

    // MARK: Folder-name parsing

    func testFolderNameParsing_allKnownTriggers() {
        for trigger in ["manual", "periodic", "promote", "sigusr2"] {
            let name = "20260612-191329-20260601-12-5K7Z-\(trigger).dcmsession"
            let parsed = SessionManifest.parseFolderName(name)
            XCTAssertEqual(parsed.trigger, trigger, "trigger for \(name)")
            XCTAssertEqual(parsed.lineageTag, "5K7Z", "lineage for \(name)")
            XCTAssertNotNil(parsed.savedAt, "date for \(name)")
        }
        // The timestamp prefix is UTC by convention.
        let parsed = SessionManifest.parseFolderName(
            "20260612-191329-20260601-12-5K7Z-manual.dcmsession"
        )
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let comps = cal.dateComponents([.year, .month, .day, .hour, .minute, .second],
                                       from: parsed.savedAt!)
        XCTAssertEqual(comps.year, 2026)
        XCTAssertEqual(comps.month, 6)
        XCTAssertEqual(comps.day, 12)
        XCTAssertEqual(comps.hour, 19)
        XCTAssertEqual(comps.minute, 13)
        XCTAssertEqual(comps.second, 29)
    }

    func testFolderNameParsing_nonconformingNames() {
        XCTAssertNil(SessionManifest.parseFolderName("not-a-session").trigger)
        XCTAssertNil(SessionManifest.parseFolderName("short.dcmsession").trigger)
        let weird = SessionManifest.parseFolderName("a-b-c-d-e-f.dcmsession")
        // Conforms structurally (6 components) — fields come back even if
        // the date is unparseable.
        XCTAssertEqual(weird.trigger, "f")
        XCTAssertEqual(weird.lineageTag, "e")
        XCTAssertNil(weird.savedAt)
    }

    // MARK: Extraction — legacy camelCase architecture

    private func legacyDict() -> [String: Any] {
        [
            "formatVersion": 1,
            "championID": "20260601-11-bzw3-32",
            "trainerID": "20260601-11-bzw3-33",
            "trainingSteps": 533_000,
            "elapsedTrainingSec": 360_000.5,
            "emittedGames": 1_200_000,
            "emittedPositions": 216_000_000,
            "replayBufferStoredCount": 1_000_000,
            "replayBufferCapacity": 1_000_000,
            "buildNumber": 1818,
            "buildGitHash": "6e4e233",
            "buildGitDirty": true,
            "buildGitBranch": "safetensors-storage",
            "savedAtUnix": 1_781_290_409.0,
            "learningRate": 0.01,
            "batchSize": 4096,
            "weightDecayCoeff": 0.0005,
            "dropoutRate": 0.0,
            "momentumCoeff": 0.9,
            "promoteThreshold": 0.53,
            "selfPlayWorkerCount": 800,
            "whiteCheckmates": 350_000,
            "blackCheckmates": 348_000,
            "stalemates": 121_000,
            "fiftyMoveDraws": 59_000,
            "threefoldRepetitionDraws": 130_000,
            "insufficientMaterialDraws": 196_000,
            "architecture": [
                "architectureVersion": 4,
                "channels": 128,
                "numBlocks": 5,
                "inputPlanes": 30,
                "parameterCount": 8_445_748,
                "policySize": 4864,
                "seReductionRatio": 4,
                "valueHeadClasses": 3
            ],
            "arenaHistory": [
                ["promoted": false], ["promoted": true], ["promoted": false]
            ],
            "lichessProbeHistory": [
                "latestPromotionCount": 32,
                "latestArenaCount": 380,
                "overall": [["puzzleElo": 970.0], ["puzzleElo": 985.5]]
            ],
            "lichessProbeWideHistory": [
                "overall": [["puzzleElo": 880.0], ["puzzleElo": 891.25]]
            ]
        ]
    }

    func testExtract_legacyDict() {
        let m = SessionManifest.extract(
            jsonDict: legacyDict(),
            folderName: "20260612-191329-20260601-12-5K7Z-sigusr2.dcmsession",
            disk: 7_900_000_000, srcBytes: 123, srcMTime: 456
        )
        XCTAssertNil(m.loadError)
        XCTAssertEqual(m.lineageTag, "5K7Z")
        XCTAssertEqual(m.trigger, "sigusr2")
        XCTAssertEqual(m.championID, "20260601-11-bzw3-32")
        XCTAssertEqual(m.trainerID, "20260601-11-bzw3-33")
        XCTAssertEqual(m.trainingSteps, 533_000)
        XCTAssertEqual(m.channels, 128)
        XCTAssertEqual(m.numBlocks, 5)
        XCTAssertEqual(m.inputPlanes, 30)
        XCTAssertEqual(m.parameterCount, 8_445_748)
        XCTAssertNotNil(m.architectureSummary)
        XCTAssertTrue(m.architectureSummary!.contains("5x[128ch]"),
                      "legacy summary should carry blocks×channels: \(m.architectureSummary!)")
        // Probe-history promotion counter is preferred over the arena scan.
        XCTAssertEqual(m.promotionCount, 32)
        XCTAssertEqual(m.arenaCount, 3, "explicit arenaHistory count wins")
        XCTAssertEqual(m.latestPElo200, 985.5)
        XCTAssertEqual(m.latestPEloWide, 891.25)
        XCTAssertEqual(m.dropoutRate, 0.0)
        XCTAssertEqual(m.weightDecay, 0.0005)
        // savedAtUnix wins over the folder-name timestamp.
        XCTAssertEqual(m.savedAt?.timeIntervalSince1970, 1_781_290_409.0)
        XCTAssertEqual(m.drawCount, 121_000 + 59_000 + 130_000 + 196_000)
    }

    func testExtract_promotionFallbackToArenaScan() {
        var dict = legacyDict()
        dict["lichessProbeHistory"] = nil
        dict["lichessProbeWideHistory"] = nil
        let m = SessionManifest.extract(
            jsonDict: dict,
            folderName: "20260612-191329-20260601-12-5K7Z-manual.dcmsession",
            disk: nil, srcBytes: nil, srcMTime: nil
        )
        XCTAssertEqual(m.promotionCount, 1, "falls back to counting promoted arena entries")
        XCTAssertNil(m.latestPElo200)
    }

    private func extractDefault(_ dict: [String: Any]) -> SessionManifest {
        SessionManifest.extract(
            jsonDict: dict,
            folderName: "20260612-191329-20260601-12-5K7Z-manual.dcmsession",
            disk: nil, srcBytes: nil, srcMTime: nil)
    }

    /// #5: an empty `arenaHistory` array (count 0) must not mask the probe
    /// history's arena count.
    func testExtract_emptyArenaHistoryFallsBackToProbeCount() {
        var dict = legacyDict()
        dict["arenaHistory"] = [Any]()
        // legacyDict's lichessProbeHistory carries latestArenaCount = 380.
        XCTAssertEqual(extractDefault(dict).arenaCount, 380)
    }

    /// #6: a non-finite FINAL probe tick must not discard earlier finite
    /// history — the manifest reports the most recent finite pElo.
    func testExtract_pEloWalksBackPastNonFiniteFinalTick() {
        var dict = legacyDict()
        dict["lichessProbeWideHistory"] = [
            "overall": [["puzzleElo": 880.0], ["puzzleElo": 891.25],
                        ["puzzleElo": Double.nan]]
        ]
        XCTAssertEqual(extractDefault(dict).latestPEloWide, 891.25)
    }

    /// #4: a PARTIAL draw-counter block (some keys missing) reports nil, not
    /// a misleadingly-low total. Full block still sums (covered by
    /// testExtract_legacyDict).
    func testExtract_partialDrawCountersReportNil() {
        var dict = legacyDict()
        dict["insufficientMaterialDraws"] = nil  // drop one of the four
        XCTAssertNil(extractDefault(dict).drawCount)
    }

    // MARK: Extraction — runtime-config architecture

    func testExtract_runtimeConfigArchUsesCanonicalSummary() throws {
        let arch = NetworkArchitecture.current
        let archData = try JSONEncoder().encode(arch)
        let archDict = try JSONSerialization.jsonObject(with: archData) as! [String: Any]
        let dict: [String: Any] = ["architecture": archDict, "trainingSteps": 42]
        let m = SessionManifest.extract(
            jsonDict: dict,
            folderName: "20260612-000000-20260612-1-AAAA-manual.dcmsession",
            disk: nil, srcBytes: nil, srcMTime: nil
        )
        XCTAssertEqual(m.architectureSummary, arch.architectureSummary)
        XCTAssertEqual(m.parameterCount, arch.parameterCount)
        XCTAssertEqual(m.channels, arch.towerOutputChannels)
        XCTAssertEqual(m.numBlocks, arch.numBlocks)
    }

    // MARK: makeManifestData round-trip

    func testMakeManifestData_roundTrip() throws {
        let json = try JSONSerialization.data(withJSONObject: legacyDict())
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let jsonURL = tmp.appendingPathComponent("session.json")
        try json.write(to: jsonURL)

        let data = try SessionManifest.makeManifestData(
            sessionJSON: json,
            folderName: "20260612-191329-20260601-12-5K7Z-periodic.dcmsession",
            sessionDirURL: tmp,
            sessionJSONURL: jsonURL
        )
        let decoded = try JSONDecoder().decode(SessionManifest.self, from: data)
        XCTAssertEqual(decoded.trigger, "periodic")
        XCTAssertEqual(decoded.trainingSteps, 533_000)
        XCTAssertEqual(decoded.sourceJSONBytes, Int64(json.count))
        XCTAssertNotNil(decoded.diskBytes)
        XCTAssertNil(decoded.loadError)
    }

    // MARK: Unreadable sessions stay visible

    func testExtract_missingFolderYieldsErrorRow() {
        let bogus = URL(fileURLWithPath: "/nonexistent/20260101-000000-20260101-1-ZZZZ-manual.dcmsession")
        let m = SessionManifest.extract(fromSessionFolder: bogus)
        XCTAssertNotNil(m.loadError)
        XCTAssertEqual(m.lineageTag, "ZZZZ", "identity still parsed from the name")
        XCTAssertEqual(m.trigger, "manual")
    }
}
