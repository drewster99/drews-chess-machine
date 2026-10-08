//
//  ModelNamingTests.swift
//  DrewsChessMachineTests
//
//  The model naming plan (MODEL_NAMING_PLAN.md): a model's name and starting
//  preset live in its lineage record (schema 4) and ride with it — through
//  fresh starts, branches, resumes, untrained copies and champion files — and
//  older records read the naming back as unrecorded, never filled in.
//

import XCTest
@testable import DrewsChessMachine

final class ModelNamingTests: XCTestCase {

    private let start = Date(timeIntervalSince1970: 1_790_000_000)

    private func naming(_ name: String?, preset: String?, edited: Bool = false) throws -> ModelNaming {
        try ModelNaming(name: name,
                        presetStart: .recorded(try preset.map { try ModelNaming.PresetStart(preset: $0, edited: edited) }))
    }

    private func record(_ tracker: LineageTracker) throws -> LineageRecord {
        try tracker.record(at: start.addingTimeInterval(60), trainerCompletedSteps: 10, segmentLocalStep: 10,
                           segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: nil,
                           rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: tracker.testInputs)
    }

    private func freshRecord(_ modelNaming: ModelNaming) throws -> LineageRecord {
        try record(try LineageTracker(start: .fresh(initialization: .forTests, naming: modelNaming), pathKind: .replay,
                                      argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 0))
    }

    private func parent(_ lineage: LineageRecord.Presence) -> LineageTracker.ParentFile {
        LineageTracker.ParentFile(modelID: "20261007-1-NAME", contentSHA256: nil, trainerCompletedSteps: 10,
                                  lineage: lineage, derivationHistory: [])
    }

    private func jsonObject(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    private func text(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
    }

    // MARK: - Name validation

    func testNamesAreTrimmedAndValidated() throws {
        XCTAssertEqual(try ModelNaming.validatedName("  my-net  "), "my-net")
        XCTAssertEqual(try ModelNaming.validatedName(String(repeating: "x", count: ModelNaming.maximumNameLength)),
                       String(repeating: "x", count: ModelNaming.maximumNameLength))
        // A zero-width joiner inside an emoji sequence is a format character,
        // not a control character.
        XCTAssertEqual(try ModelNaming.validatedName("net 👨‍👩‍👧"), "net 👨‍👩‍👧")
        XCTAssertThrowsError(try ModelNaming.validatedName("")) {
            XCTAssertEqual($0 as? ModelNaming.NamingError, .emptyName)
        }
        XCTAssertThrowsError(try ModelNaming.validatedName("   ")) {
            XCTAssertEqual($0 as? ModelNaming.NamingError, .emptyName)
        }
        XCTAssertThrowsError(try ModelNaming.validatedName("two\nlines")) {
            XCTAssertEqual($0 as? ModelNaming.NamingError, .nameHasControlCharacter)
        }
        XCTAssertThrowsError(try ModelNaming.validatedName("tab\there")) {
            XCTAssertEqual($0 as? ModelNaming.NamingError, .nameHasControlCharacter)
        }
        XCTAssertThrowsError(try ModelNaming.validatedName(String(repeating: "x", count: ModelNaming.maximumNameLength + 1))) {
            XCTAssertEqual($0 as? ModelNaming.NamingError, .nameTooLong(length: ModelNaming.maximumNameLength + 1))
        }
        XCTAssertThrowsError(try ModelNaming.PresetStart(preset: "", edited: false)) {
            XCTAssertEqual($0 as? ModelNaming.NamingError, .emptyPresetName)
        }
    }

    // MARK: - Schema 4 coding

    func testASchemaFourRecordRoundTripsItsNaming() throws {
        let modelNaming = try naming("my-net", preset: "v4_5block_7x7", edited: true)
        let recorded = try freshRecord(modelNaming)
        XCTAssertEqual(recorded.schema, 4)
        XCTAssertEqual(recorded.modelNaming, .recorded(modelNaming))
        let jsonText = try recorded.jsonText()
        XCTAssertEqual(try LineageRecord.decode(jsonText: jsonText), recorded)
        let stored = try XCTUnwrap(try jsonObject(jsonText)["model_naming"] as? [String: Any])
        XCTAssertEqual(stored["recorded"] as? Bool, true)
        let value = try XCTUnwrap(stored["value"] as? [String: Any])
        XCTAssertEqual(value["name"] as? String, "my-net")
        let presetStart = try XCTUnwrap(value["preset_start"] as? [String: Any])
        XCTAssertEqual(presetStart["recorded"] as? Bool, true)
        let presetValue = try XCTUnwrap(presetStart["value"] as? [String: Any])
        XCTAssertEqual(Set(presetValue.keys), ["preset", "edited"])
        XCTAssertEqual(presetValue["preset"] as? String, "v4_5block_7x7")
        XCTAssertEqual(presetValue["edited"] as? Bool, true)
    }

    func testANoNameNoPresetNamingRoundTripsWithExplicitNulls() throws {
        let recorded = try freshRecord(.unnamedWithoutPreset)
        let jsonText = try recorded.jsonText()
        XCTAssertEqual(try LineageRecord.decode(jsonText: jsonText).modelNaming, .recorded(.unnamedWithoutPreset))
        let value = try XCTUnwrap((try jsonObject(jsonText)["model_naming"] as? [String: Any])?["value"] as? [String: Any])
        XCTAssertTrue(value["name"] is NSNull, "an absent name is an explicit null")
        let presetStart = try XCTUnwrap(value["preset_start"] as? [String: Any])
        XCTAssertTrue(presetStart["value"] is NSNull, "no preset is a recorded null")
    }

    func testASchemaFourRecordWithoutItsNamingIsRefused() throws {
        var object = try jsonObject(try freshRecord(.unnamedWithoutPreset).jsonText())
        object.removeValue(forKey: "model_naming")
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: try text(object)))
    }

    func testAStoredNameWithSurroundingWhitespaceIsRefused() throws {
        var object = try jsonObject(try freshRecord(try naming("ok", preset: nil)).jsonText())
        var wrapper = try XCTUnwrap(object["model_naming"] as? [String: Any])
        var value = try XCTUnwrap(wrapper["value"] as? [String: Any])
        value["name"] = " ok"
        wrapper["value"] = value
        object["model_naming"] = wrapper
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: try text(object)))
    }

    // MARK: - Older records stay readable

    func testARealSchemaThreeRecordReadsItsNamingAsUnrecorded() throws {
        let decoded = try LineageRecord.decode(jsonText: LineageSchemaThreeFixtures.guiPromotedChampion)
        XCTAssertEqual(decoded.schema, 3)
        XCTAssertEqual(decoded.modelNaming, .unrecorded)
        // Never written back as schema 3: the conversion is schema 4 and
        // keeps the naming unrecorded.
        XCTAssertThrowsError(try decoded.jsonText())
        let converted = try decoded.withoutTrainerState()
        XCTAssertEqual(converted.schema, LineageRecord.currentSchema)
        XCTAssertEqual(converted.modelNaming, .unrecorded)
        XCTAssertEqual(try LineageRecord.decode(jsonText: try converted.jsonText()), converted)
    }

    func testARealSchemaTwoRecordReadsItsNamingAsUnrecorded() throws {
        let decoded = try LineageRecord.decode(jsonText: LineageSchemaTwoFixtures.fatconv98ContStep6093)
        XCTAssertEqual(decoded.schema, 2)
        XCTAssertEqual(decoded.modelNaming, .unrecorded)
    }

    func testASchemaThreeRecordCarryingModelNamingIsRefused() throws {
        var object = try jsonObject(LineageSchemaThreeFixtures.guiPromotedChampion)
        object["model_naming"] = ["recorded": false]
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: try text(object)))
    }

    // MARK: - Inheritance

    func testBranchesAndResumesCarryTheParentsNaming() throws {
        let modelNaming = try naming("carried", preset: "nt8y")
        let parentRecord = try freshRecord(modelNaming)
        for start in [LineageTracker.Start.branch(parent: parent(.recorded(parentRecord))),
                      .resume(parent: parent(.recorded(parentRecord)), gaps: [], legacyTotals: nil)] {
            let tracker = try LineageTracker(start: start, pathKind: .replay, argv: ["dcm"],
                                             startedAt: self.start, segmentStartTrainerStep: 10)
            XCTAssertEqual(tracker.modelNaming, .recorded(modelNaming))
            XCTAssertEqual(try record(tracker).modelNaming, .recorded(modelNaming))
        }
    }

    func testABranchFromAFileWithoutARecordIsUnrecorded() throws {
        let tracker = try LineageTracker(start: .branch(parent: parent(.unrecorded(formatVersion: 6))), pathKind: .replay,
                                         argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 0)
        XCTAssertEqual(try record(tracker).modelNaming, .unrecorded)
    }

    @MainActor
    func testAChampionFileOfAPromotionKeepsTheRunsNaming() throws {
        let modelNaming = try naming("promoted", preset: "v4_5block_7x7")
        let runRecord = try freshRecord(modelNaming)
        let champion = try runRecord.withoutTrainerState()
        XCTAssertEqual(champion.modelNaming, .recorded(modelNaming))
        let origin = SessionController.ChampionOrigin.file(parent(.recorded(champion)), startWeights: .notLoaded)
        XCTAssertEqual(try SessionController.championFileLineageRecord(origin: origin, at: start).modelNaming,
                       .recorded(modelNaming))
    }

    @MainActor
    func testABuiltChampionsFileRecordsItsBuildNaming() throws {
        let modelNaming = try naming("built here", preset: nil)
        let record = try SessionController.championFileLineageRecord(
            origin: .built(initialization: .forTests, naming: modelNaming), at: start)
        XCTAssertEqual(record.modelNaming, .recorded(modelNaming))
    }

    // MARK: - Untrained copies (derive, graft, a GUI save of a pre-lineage champion)

    func testACopyThatChangesTheArchitectureMarksAnUnchangedPresetEdited() throws {
        let source: LineageRecord.Recorded<ModelNaming> = .recorded(try naming("src", preset: "nt8y"))
        XCTAssertEqual(try ModelNaming.ofDerive(of: source, architectureChanged: true, renamedTo: nil),
                       .recorded(try naming("src", preset: "nt8y", edited: true)))
        XCTAssertEqual(try ModelNaming.ofDerive(of: source, architectureChanged: false, renamedTo: nil), source)
        // Already edited stays edited; no preset stays no preset.
        let edited: LineageRecord.Recorded<ModelNaming> = .recorded(try naming(nil, preset: "nt8y", edited: true))
        XCTAssertEqual(try ModelNaming.ofDerive(of: edited, architectureChanged: true, renamedTo: nil), edited)
        let custom: LineageRecord.Recorded<ModelNaming> = .recorded(try naming("c", preset: nil))
        XCTAssertEqual(try ModelNaming.ofDerive(of: custom, architectureChanged: true, renamedTo: nil), custom)
    }

    func testARenameReplacesTheNameAndKeepsThePresetStart() throws {
        let source: LineageRecord.Recorded<ModelNaming> = .recorded(try naming("old", preset: "nt8y"))
        XCTAssertEqual(try ModelNaming.ofDerive(of: source, architectureChanged: false, renamedTo: "new"),
                       .recorded(try naming("new", preset: "nt8y")))
    }

    func testACopyOfAnUnrecordedNamingStaysUnrecordedUnlessRenamed() throws {
        XCTAssertEqual(try ModelNaming.ofDerive(of: .unrecorded, architectureChanged: true, renamedTo: nil),
                       .unrecorded)
        XCTAssertEqual(try ModelNaming.ofDerive(of: .unrecorded, architectureChanged: false, renamedTo: "named"),
                       .recorded(try ModelNaming(name: "named", presetStart: .unrecorded)))
    }

    func testAGraftRecordsTheTargetPresetUnedited() throws {
        let source: LineageRecord.Recorded<ModelNaming> = .recorded(try naming("src", preset: "v4_5block_7x7", edited: true))
        XCTAssertEqual(try ModelNaming.ofGraft(of: source, targetPreset: "nt8y", renamedTo: nil),
                       .recorded(try naming("src", preset: "nt8y")))
        XCTAssertEqual(try ModelNaming.ofGraft(of: source, targetPreset: nil, renamedTo: "g"),
                       .recorded(try naming("g", preset: nil)), "a target given by path is no preset")
        XCTAssertEqual(try ModelNaming.ofGraft(of: .unrecorded, targetPreset: "nt8y", renamedTo: nil), .unrecorded,
                       "the name of an unrecorded source is unknown")
        XCTAssertEqual(try ModelNaming.ofGraft(of: .unrecorded, targetPreset: "nt8y", renamedTo: "g"),
                       .recorded(try naming("g", preset: "nt8y")))
    }

    func testUntrainedCopyRecordWritesTheNamingItIsGiven() throws {
        let sourceRecord = try freshRecord(try naming("old", preset: "nt8y"))
        let renamed = try ModelNaming.ofDerive(of: sourceRecord.modelNaming, architectureChanged: false, renamedTo: "renamed")
        let copy = try LineageTracker.untrainedCopyRecord(
            source: parent(.recorded(sourceRecord)), derivation: nil, sourceArchitecture: nil, naming: renamed,
            pathKind: .derive, argv: ["dcm"], at: start)
        XCTAssertEqual(copy.modelNaming, .recorded(try naming("renamed", preset: "nt8y")))
        XCTAssertEqual(try LineageRecord.decode(jsonText: try copy.jsonText()), copy)
    }

    func testResultsJSONCarriesTheModelNaming() throws {
        let recorder = CliTrainingRecorder()
        recorder.setFinalLineage(try freshRecord(try naming("cli", preset: "nt8y")), checkpointSHA256: nil)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: try recorder.encodedJSONData(totalTrainingSeconds: 1)) as? [String: Any])
        let lineage = try XCTUnwrap(object["lineage"] as? [String: Any])
        let stored = try XCTUnwrap(lineage["model_naming"] as? [String: Any])
        XCTAssertEqual((stored["value"] as? [String: Any])?["name"] as? String, "cli")
    }

    // MARK: - Display

    func testNameplateText() throws {
        let both = ModelNameplate(naming: .recorded(try naming("my-net", preset: "v4_5block_7x7", edited: true)),
                                  weightsSource: .thisProcess)
        XCTAssertEqual(both.headerText,
                       "my-net · preset v4_5block_7x7 (edited) · format v\(ArchitectureFormat.currentVersion)")
        let nameOnly = ModelNameplate(naming: .recorded(try naming("solo", preset: nil)),
                                      weightsSource: .file(.safetensors(architectureFormat: 9)))
        XCTAssertEqual(nameOnly.headerText, "solo · format v9")
        let presetOnly = ModelNameplate(naming: .recorded(try naming(nil, preset: "nt8y")),
                                        weightsSource: .file(.safetensors(architectureFormat: 12)))
        XCTAssertEqual(presetOnly.headerText, "preset nt8y · format v12")
        let neither = ModelNameplate(naming: .recorded(.unnamedWithoutPreset), weightsSource: .thisProcess)
        XCTAssertEqual(neither.headerText, "format v\(ArchitectureFormat.currentVersion)")
        let legacy = ModelNameplate(naming: .unrecorded, weightsSource: .file(.legacyDCMModel(formatVersion: 2)))
        XCTAssertEqual(legacy.headerText, "legacy .dcmmodel v2")
    }

    func testAboutRowsSayWhatIsAbsent() throws {
        XCTAssertEqual(ModelNaming.nameRowText(.recorded(.unnamedWithoutPreset)), "none given")
        XCTAssertEqual(ModelNaming.presetRowText(.recorded(.unnamedWithoutPreset)), "none (custom architecture)")
        XCTAssertEqual(ModelNaming.nameRowText(.unrecorded), "not recorded (made before models carried names)")
        XCTAssertEqual(ModelNaming.presetRowText(.recorded(try ModelNaming(name: "n", presetStart: .unrecorded))),
                       "not recorded (renamed from a model made before names)")
        XCTAssertEqual(ModelNaming.compactText(.unrecorded), "not recorded")
        XCTAssertEqual(ModelNaming.compactText(.recorded(.unnamedWithoutPreset)), "no name or preset")
    }

    func testTheChampionNameplateFollowsItsOrigin() throws {
        let modelNaming = try naming("n", preset: nil)
        XCTAssertEqual(ModelNameplate(championOrigin: .built(initialization: .forTests, naming: modelNaming)),
                       ModelNameplate(naming: .recorded(modelNaming), weightsSource: .thisProcess))
        let loaded = SessionController.ChampionOrigin.file(
            parent(.recorded(try freshRecord(modelNaming))),
            startWeights: .loaded(.alreadyCentered, fileFormat: .safetensors(architectureFormat: 11)))
        XCTAssertEqual(ModelNameplate(championOrigin: loaded),
                       ModelNameplate(naming: .recorded(modelNaming),
                                      weightsSource: .file(.safetensors(architectureFormat: 11))))
        let preLineage = SessionController.ChampionOrigin.file(
            parent(.unrecorded(formatVersion: 6)),
            startWeights: .loaded(.alreadyCentered, fileFormat: .safetensors(architectureFormat: 6)))
        XCTAssertEqual(ModelNameplate(championOrigin: preLineage).naming, .unrecorded)
    }

    @MainActor
    func testTheTitleBarPutsTheNameplateBeforeTheTopology() throws {
        let arch = NetworkArchitecture.preset(.v4_5block_7x7)
        XCTAssertEqual(TitleBarView.architectureText(arch: arch, nameplate: nil), arch.shortLabel)
        XCTAssertFalse(arch.shortLabel.hasPrefix("v"), "the retired v3/v4/v5 label is gone")
        let nameplate = ModelNameplate(naming: .recorded(try naming("t", preset: nil)), weightsSource: .thisProcess)
        XCTAssertEqual(TitleBarView.architectureText(arch: arch, nameplate: nameplate),
                       "t · format v\(ArchitectureFormat.currentVersion) · " + arch.shortLabel)
    }

    // MARK: - Session picker

    private func manifest(_ dict: [String: Any]) -> SessionManifest {
        SessionManifest.extract(jsonDict: dict, folderName: "20261007-120000-20261007-1-NAME-manual.dcmsession",
                                disk: nil, srcBytes: nil, srcMTime: nil)
    }

    @MainActor
    func testTheSessionPickerReadsTheNamingFromSessionJSON() throws {
        let modelNaming = try naming("picker", preset: "nt8y")
        let lineage = try jsonObject(try freshRecord(modelNaming).jsonText())
        let named = manifest(["lineage": lineage])
        XCTAssertEqual(named.modelNaming, .recorded(modelNaming))
        XCTAssertEqual(SessionPickerModel.makeGroups(from: [named]).first?.modelNameText, "picker · preset nt8y")

        XCTAssertEqual(manifest([:]).modelNaming, .unrecorded, "a session without a lineage")
        XCTAssertEqual(manifest(["lineage": try jsonObject(LineageSchemaThreeFixtures.guiPromotedChampion)]).modelNaming,
                       .unrecorded, "a schema-3 record")
        XCTAssertNil(manifest(["lineage": ["model_naming": ["recorded": "yes"]]]).modelNaming, "malformed: not guessed")
    }

    @MainActor
    func testAManifestWrittenBeforeNamingDecodesWithoutIt() throws {
        var object = try jsonObject(String(decoding: try JSONEncoder().encode(manifest([:])), as: UTF8.self))
        object.removeValue(forKey: "modelNaming")
        let decoded = try JSONDecoder().decode(SessionManifest.self, from: try JSONSerialization.data(withJSONObject: object))
        XCTAssertNil(decoded.modelNaming)
        XCTAssertNil(SessionPickerModel.makeGroups(from: [decoded]).first?.modelNameText)
    }

    // MARK: - New Network screen

    @MainActor
    func testTheBuildScreenRecordsTheChosenPresetTheEditAndTheName() throws {
        let model = BuildNewModelModel(NamedArchitecture(label: "Custom", architecture: .newModelDefault))
        XCTAssertEqual(try model.makeModelNaming(), .unnamedWithoutPreset, "the screen opens on no preset")

        let preset = NetworkArchitecture.Preset.v4_5block_7x7
        model.loadPreset(named: preset.rawValue, NamedArchitecture(label: "p", architecture: .preset(preset)))
        XCTAssertEqual(try model.makeModelNaming(), try naming(nil, preset: preset.rawValue))

        model.stemConvKernelSize = 5
        XCTAssertEqual(try model.makeModelNaming(), try naming(nil, preset: preset.rawValue, edited: true))

        model.labelOverride = "  my net  "
        XCTAssertEqual(model.buildRequest?.naming, try naming("my net", preset: preset.rawValue, edited: true))
        XCTAssertNil(model.nameError)

        model.labelOverride = "bad\nname"
        XCTAssertNotNil(model.nameError)
        XCTAssertNil(model.buildRequest, "Build is disabled for a name that can't be recorded")

        model.labelOverride = ""
        model.noteSavedAsPreset(named: "saved_one", architecture: model.architecture)
        XCTAssertEqual(try model.makeModelNaming(), try naming(nil, preset: "saved_one"))
    }
}
