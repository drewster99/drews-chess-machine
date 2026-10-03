//
//  BuildNewModelDraftTests.swift
//  DrewsChessMachineTests
//
//  The Build-New-Model editor addresses block groups by draft identity, not
//  by array index (see `BlockGroupDraft`). SwiftUI keeps a control's binding
//  alive past the update that removes its row, so a binding to a removed
//  group must not be able to reach any group still in the tower. These tests
//  pin that a removed draft is detached, and that duplicate / move / remove
//  act on the group they were given.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class BuildNewModelDraftTests: XCTestCase {

    /// Three single-block groups with distinct widths, so each is
    /// recognizable in the resulting architecture.
    private func threeGroupModel() -> BuildNewModelModel {
        var arch = NetworkArchitecture.current
        let base = arch.blockGroups[0]
        arch.blockGroups = [16, 32, 64].map { width in
            var group = base
            group.count = 1
            group.channels = width
            return group
        }
        return BuildNewModelModel(NamedArchitecture(label: "test", architecture: arch))
    }

    func testARemovedDraftIsDetached() {
        let model = threeGroupModel()
        let middle = model.blockGroupDrafts[1]
        model.removeGroup(middle)
        let afterRemoval = model.architecture
        XCTAssertEqual(afterRemoval.blockGroups.map(\.channels), [16, 64])
        // A binding retained past the removal writes into the detached draft
        // only: neither the removed position's new occupant nor any other
        // group changes.
        middle.group.channels = 999
        middle.activationFunction = .gelu
        XCTAssertEqual(model.architecture, afterRemoval)
    }

    func testDuplicateMoveAndRemoveActOnTheGivenGroup() {
        let model = threeGroupModel()
        let first = model.blockGroupDrafts[0]
        let last = model.blockGroupDrafts[2]
        model.duplicateGroup(first)
        XCTAssertEqual(model.blockGroups.map(\.channels), [16, 16, 32, 64])
        XCTAssertFalse(model.blockGroupDrafts[1] === first, "a duplicate is a new draft")
        model.moveGroup(last, offset: -1)
        XCTAssertEqual(model.blockGroups.map(\.channels), [16, 16, 64, 32])
        XCTAssertTrue(model.blockGroupDrafts[2] === last)
        model.removeGroup(first)
        XCTAssertEqual(model.blockGroups.map(\.channels), [16, 64, 32])
        model.appendCopyOfLastGroup()
        XCTAssertEqual(model.blockGroups.map(\.channels), [16, 64, 32, 32])
    }

    /// The output-norm picker reads a legacy `nil` as `.none` and writes the
    /// explicit value.
    func testOutputNormReadsLegacyNilAsNone() {
        var arch = NetworkArchitecture.current
        arch.blockGroups[0].outputNorm = nil
        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: arch))
        let draft = model.blockGroupDrafts[0]
        XCTAssertEqual(draft.outputNorm, BlockOutputNorm.none)
        draft.outputNorm = .layerNorm
        XCTAssertEqual(draft.group.outputNorm, .layerNorm)
    }
}
